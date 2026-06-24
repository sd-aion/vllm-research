# kernel_unified_attention

This note maps the main Triton paged attention kernel used by standard `TRITON_ATTN`.

## Location

The kernel lives in:

- `vllm/v1/attention/ops/triton_unified_attention.py`

The function is:

- `kernel_unified_attention(...)`

This is the standard decoder paged-attention kernel for `TRITON_ATTN`.

## Program IDs

The kernel uses:

- `tl.program_id(0)`: flattened query block index.
- `tl.program_id(1)`: KV head index.
- `tl.program_id(2)`: segment index when `IS_3D` is true.

In 2D mode, segment index is treated as 0.

## Sequence Resolution

The kernel calls:

- `resolve_seq_and_query_len(...)`

`resolve_seq_and_query_len(...)` is defined in `vllm/v1/attention/ops/triton_attention_helpers.py`. The launch grid places the query blocks for all variable-length sequences on one flattened axis, so a global program ID does not directly identify either its request or its position within that request. The helper performs a binary search over the cumulative query boundaries in `cu_seqlens_q`, finds the owning sequence, and converts the global block index into sequence-local coordinates needed for Q addressing, causal masking, and paged-KV traversal.

- `seq_idx`: the index of the sequence in the current attention batch that owns this query-block program.
- `q_block_local_idx`: identifies which tile of the current sequence's query tokens this kernel program handles; `0` handles tokens starting at local position `0`, `1` handles tokens starting at `BLOCK_Q`, and in general the tile starts at `q_block_local_idx * BLOCK_Q`.
- `cur_batch_in_all_start_index`: the sequence's starting token offset in the packed Q tensor, equal to `cu_seqlens_q[seq_idx]`.
- `cur_batch_query_len`: the number of new query tokens for this sequence in the current scheduler step, computed from adjacent `cu_seqlens_q` boundaries.
- `seq_len`: the full valid key/value length for this sequence, including cached context and the current query tokens, loaded from `seqused_k`.

The first query position represented by the program is `q_block_local_idx * BLOCK_Q`. If that position is greater than or equal to `cur_batch_query_len`, the upper-bound launch grid assigned a query block that does not actually exist for this sequence, so the kernel returns before loading Q/K/V or computing attention. Valid programs use `cur_batch_in_all_start_index` to locate their rows in packed Q and use `seq_len - cur_batch_query_len` as the amount of previously cached context.

## Query Coordinates

The kernel builds:

- `offs_m`: flattened query rows inside the program.
- `offs_d`: head-dimension offsets.
- `offs_t`: KV tile offsets.
- `query_pos`: query token position inside the current request.
- `query_offset_0`: packed token index.
- `query_offset_1`: query head index.

`query_offset_1` spans the query heads that map to the current KV head.

This is how the kernel supports MQA/GQA.

## Query Load

The query tile shape is:

- `(BLOCK_M, HEAD_SIZE_PADDED)`

If `USE_TD_QO` is true, the kernel uses:

- `_load_q_td(...)`

Otherwise it uses pointer arithmetic and `tl.load(...)`.

The load is masked by:

- dimension mask
- valid query token mask
- valid query head mask

## Block Table Lookup

The kernel computes:

- `block_table_offset = seq_idx * block_table_stride`

For each KV tile, it loads physical block IDs from:

- `block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE`

The block table maps logical token blocks in the request to physical cache blocks.

This is the read-side counterpart to `slot_mapping`, which was used earlier to write K/V into cache.

## KV Tile Load

The kernel loads K and V tiles from paged cache.

For pointer mode:

- K is loaded as `(HEAD_SIZE_PADDED, TILE_SIZE)`.
- V is loaded as `(TILE_SIZE, HEAD_SIZE_PADDED)`.

For tensor descriptor mode:

- `_load_kv_tile_td(...)` loads the KV tile with `tl.make_tensor_descriptor`.

The tile is masked by:

- dimension mask
- tile validity mask from `seq_len`

## KV Quantization

The helper `_cast_kv_tile(...)` handles KV tile casting and dequantization.

Modes are:

- `KV_QUANT_MODE == 0`: no quantization.
- `KV_QUANT_MODE == 1`: FP8 per-tensor.
- `KV_QUANT_MODE == 2`: INT8 per-token-head.
- `KV_QUANT_MODE == 3`: FP8 per-token-head.

For FP8 per-tensor, the tile is dequantized with tensor-wide scale unless Q is also FP8 and scale folding is used.

For per-token-head modes, the kernel later loads per-token-head scales from scale-cache tensors.

## Per-Token-Head Scales

When `KV_QUANT_MODE >= 2`, the kernel loads:

- `k_token_head_scales`
- `v_token_head_scales`

The scale cache index uses:

- physical block index
- offset inside block
- KV head index

The K scale is fused into the attention score multiplication.

The V scale is applied to probabilities before multiplying by V.

## Mask Construction

The kernel calls:

- `compute_kv_seq_mask(...)`

The default mask is causal:

- key position must be `<= query_abs_pos`

Then it can apply:

- chunked lookback mask.
- sliding-window mask.
- MM prefix bidirectional ranges.

The final score mask also includes:

- valid query head mask.
- valid query token mask.
- valid KV tile mask.

Invalid positions are set to `-inf`.

## Score Computation

The score tile is:

- `S = score_scale * dot(Q, K)`

For per-token-head quantization:

- `S = dot(Q, K) * (score_scale * k_token_head_scales)`

Then optional transforms are applied:

- softcap through `apply_softcap(...)`.
- ALiBi through `apply_alibi_to_score(...)`.
- QQ bias through `load_qq_bias_tile(...)`.

## Sinks

Sinks are handled through `init_softmax_M(...)`.

Without sinks, the running row max `M` starts at `-inf`.

With sinks, `M` starts from the sink value for each query head.

In 3D mode, only segment 0 includes sinks so `reduce_segments(...)` does not count them multiple times.

## Online Softmax

The kernel uses online softmax.

For each KV tile:

- `softmax_step(S, M, L)` computes new row max, new exp sum, probabilities, and accumulator rescale factor.
- accumulator is multiplied by `alpha`.
- `dot(P, V)` is added to the accumulator.
- `M` and `L` are updated.

This avoids materializing the full attention matrix.

## Sliding Window Value Mask

There is an extra V mask when `SLIDING_WINDOW` is active.

It zeros V rows outside the window before accumulation.

This guards against masked positions contributing through numerical edge cases.

## 2D Epilogue

In 2D mode:

- `acc = acc / L[:, None]`
- optional FP8 query descale adjusts value scale.
- optional FP8 output scaling and clamping are applied.
- output is stored either with tensor descriptor or pointer arithmetic.

This writes final output directly to `output_ptr`.

## 3D Epilogue

In 3D mode:

- the kernel does not divide by global `L`.
- it stores per-segment partial `acc` into `segm_output_ptr`.
- it stores per-segment `M` and `L` into `segm_max_ptr` and `segm_expsum_ptr`.

Then `reduce_segments(...)` combines those partials.

## Key Files

- `vllm/v1/attention/ops/triton_unified_attention.py`
- `vllm/v1/attention/ops/triton_attention_helpers.py`
