# Triton Attention Helpers

This note maps helper functions from `triton_attention_helpers.py`.

## Purpose

The helper file is:

- `vllm/v1/attention/ops/triton_attention_helpers.py`

These helpers are `@triton.jit` functions used by `kernel_unified_attention(...)` and `reduce_segments(...)`.

They keep sequence lookup, mask construction, ALiBi, softcap, and online-softmax bookkeeping shared and readable.

## cdiv_fn

`cdiv_fn(x, y)` returns ceiling division.

It is used for tile counts and segment counts.

## apply_softcap

`apply_softcap(S, x)` computes:

- `x * tanh(S / x)`

It is implemented with exponentials instead of direct `tanh`.

This bounds attention scores before softmax when a model uses logits softcap.

## find_seq_idx

`find_seq_idx(...)` in `vllm/v1/attention/ops/triton_attention_helpers.py` maps an index from a flattened kernel grid back to the sequence that owns it. `query_start_len_ptr` points to `cu_seqlens_q`, the cumulative query boundaries with shape `[num_seqs + 1]`.

It has two modes:

- q-block mode for attention kernels.
- raw query-token mode for `reduce_segments(...)`.

In both modes, the binary search returns the largest sequence index `i` whose search boundary is less than or equal to `target_idx`. What changes between the modes is the unit of `target_idx` and therefore the boundary expression being searched.

### Raw Query-Token Mode

Raw mode is selected with `use_q_block_mode=False`. Here, `target_idx` is a packed query-token index, and the search boundary for sequence `i` is simply:

- `cu_seqlens_q[i]`

For example, suppose the query lengths are `[5, 2, 6]`. The cumulative boundaries are:

- `cu_seqlens_q = [0, 5, 7, 13]`

The packed query tensor is partitioned as:

- sequence 0 owns token indices `[0, 5)`.
- sequence 1 owns token indices `[5, 7)`.
- sequence 2 owns token indices `[7, 13)`.

If `target_idx = 6`, the largest boundary not exceeding `6` is `5`, so the search returns `seq_idx = 1`. If `target_idx = 10`, the largest boundary not exceeding `10` is `7`, so it returns `seq_idx = 2`. `reduce_segments(...)` uses this mode because its first grid dimension directly enumerates packed query tokens rather than query blocks.

### Q-Block Mode

Q-block mode is selected with `use_q_block_mode=True`. Here, `target_idx` is a global query-block program index, so searching token boundaries directly would use the wrong units. The helper transforms each sequence boundary into query-block-grid units:

- `q_block_boundary[i] = cu_seqlens_q[i] // BLOCK_Q + i`

The integer division converts cumulative token positions to whole `BLOCK_Q` units, while `+ i` reserves one additional possible partial-block position for each preceding sequence. This creates the same monotonic upper-bound layout used by the attention kernel launch without requiring the CPU to calculate and sum every sequence's exact `ceil(query_len / BLOCK_Q)` count.

Using the same query lengths `[5, 2, 6]` with `BLOCK_Q = 4`, the transformed boundaries are:

- sequence 0 start: `0 // 4 + 0 = 0`.
- sequence 1 start: `5 // 4 + 1 = 2`.
- sequence 2 start: `7 // 4 + 2 = 3`.
- conceptual end: `13 // 4 + 3 = 6`.

The resulting global query-block ownership is:

| Global q-block | Sequence | Local q-block | First local query position | Valid? |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 0 | yes |
| 1 | 0 | 1 | 4 | yes |
| 2 | 1 | 0 | 0 | yes |
| 3 | 2 | 0 | 0 | yes |
| 4 | 2 | 1 | 4 | yes |
| 5 | 2 | 2 | 8 | no |

For `target_idx = 4`, the binary search finds that transformed boundary `3` for sequence 2 is the largest boundary not exceeding `4`, so it returns `seq_idx = 2`. `resolve_seq_and_query_len(...)` then computes `q_block_local_idx = 4 - 3 = 1`, meaning this program starts at local query position `1 * BLOCK_Q = 4` within sequence 2.

Global q-block 5 also maps to sequence 2, with local q-block 2, but that block would start at local position `2 * 4 = 8` while the sequence has only 6 query tokens. This is an intentionally overlaunched block from the upper-bound grid, and `kernel_unified_attention(...)` immediately returns when it detects `8 >= 6`.

The important distinction is therefore: raw mode finds the owner of an actual packed query token, while q-block mode finds the owner of a possibly overallocated `BLOCK_Q`-sized kernel tile. Only q-block mode needs transformed boundaries and the later validity check.

## resolve_seq_and_query_len

`resolve_seq_and_query_len(...)` uses `find_seq_idx(...)` to map a flattened query block to request-local metadata.

It returns:

- `seq_idx`
- `q_block_local_idx`
- `cur_batch_in_all_start_index`
- `cur_batch_query_len`
- `seq_len`

This is what lets `kernel_unified_attention(...)` use an upper-bound launch grid without CPU materializing every query block.

## init_softmax_M

`init_softmax_M(...)` initializes the online softmax row max.

Without sinks:

- `M = -inf`

With sinks:

- `M` starts from the per-head sink value.

In 3D mode, only segment 0 loads sinks.

## compute_tile_loop_bounds

`compute_tile_loop_bounds(...)` computes:

- `loop_lo`: the first logical KV tile index that this program must process; tile `j` begins at sequence token position `j * TILE_SIZE`.
- `loop_hi`: the exclusive logical KV tile index where iteration stops, so the kernel executes `for j in range(loop_lo, loop_hi)`.
- `max_seq_prefix_len`: the exclusive token-level upper bound on valid KV positions for this query block, used by `seq_offset < max_seq_prefix_len` to mask unused entries in the final tile.

It combines:

- causal prefix length.
- sliding-window tile pruning.
- chunked-attention tile pruning.
- MM prefix behavior.
- 3D segment scoping.

For sliding window, it narrows the tile loop to only tiles that can contain valid keys.

For 3D mode, it further intersects the loop with the current segment's tile range.

## compute_kv_seq_mask

`compute_kv_seq_mask(...)` builds the token-level KV mask.

Base behavior is causal:

- `seq_offset <= query_abs_pos`

Then it can apply:

- chunked lookback mask.
- sliding-window mask.
- MM prefix bidirectional ranges.

The order is important:

- `(causal AND sliding_or_chunked_window) OR mm_prefix`

This lets MM prefix ranges override sliding-window restriction inside those ranges.

## apply_alibi_to_score

`apply_alibi_to_score(...)` adds ALiBi bias to the score matrix.

It supports:

- linear ALiBi.
- sqrt ALiBi when `USE_ALIBI_SQRT` is true.

`TRITON_ATTN` advertises `supports_alibi_sqrt() -> True`.

## load_qq_bias_tile

`load_qq_bias_tile(...)` loads query-query bias for keys that correspond to query rows.

This is optional and controlled by `USE_QQ_BIAS`.

## softmax_step

`softmax_step(S, M, L)` performs one online softmax update.

It returns:

- new row max
- new exp sum
- probabilities for the current tile
- accumulator rescale factor `alpha`

The caller multiplies its accumulator by `alpha` and adds the current tile contribution.

## store_segm_reduce_scalars

`store_segm_reduce_scalars(...)` stores per-segment:

- row max `M`
- exp sum `L`

`reduce_segments(...)` later reads these values to combine 3D decode segments.

## Key Files

- `vllm/v1/attention/ops/triton_attention_helpers.py`
- `vllm/v1/attention/ops/triton_unified_attention.py`
