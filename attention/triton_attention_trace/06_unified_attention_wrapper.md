# Unified Attention Wrapper

This note covers `unified_attention(...)`, the Python wrapper that launches the main Triton attention kernel.

## Location

The wrapper lives in:

- `vllm/v1/attention/ops/triton_unified_attention.py`

It launches:

- `kernel_unified_attention`

It may also launch:

- `reduce_segments`

## Public Arguments

The wrapper receives:

- `q`: packed query tensor.
- `k`: key cache tensor.
- `v`: value cache tensor.
- `out`: output tensor.
- `cu_seqlens_q`: cumulative query starts.
- `max_seqlen_q`: maximum query length in the batch.
- `seqused_k`: valid K length per sequence.
- `max_seqlen_k`: maximum K length in the batch.
- `softmax_scale`: attention scale.
- `causal`: must be true.
- `window_size`: sliding-window tuple.
- `block_table`: paged cache block table.
- `softcap`: logits softcap value.
- `q_descale`, `k_descale`, `v_descale`: quantization descales.
- segment scratch tensors.
- `alibi_slopes`: supplies one slope per query head for adding an ALiBi relative-position bias to attention scores before softmax.
- `output_scale`: enables FP8 output quantization by scaling the final attention result by `1 / output_scale`, clamping it to the FP8 range, and storing it in the output tensor.
- `qq_bias`: supplies an additive query-to-query bias matrix whose applicable tile is added to scores for keys that belong to the current query-token range.
- `sinks`: supplies one attention-sink logit per query head and initializes the online softmax with that extra virtual attention target.
- `mm_prefix_range`: stores the multimodal token ranges for each sequence so the causal mask can allow full bidirectional attention within those ranges.
- `use_alibi_sqrt`: selects the square-root relative-distance variant of ALiBi instead of the normal linear-distance bias.
- `kv_quant_mode`: tells the kernel whether KV is unquantized, FP8 per-tensor, INT8 per-token-head, or FP8 per-token-head so it can choose the corresponding scale and dequantization path.
- per-token-head scale caches.
- `chunk_lookback`: enables chunk-aligned causal attention and specifies how many preceding chunks, in addition to the current chunk, each query may attend; `-1` disables this mode.
- `use_td`: enables Triton tensor descriptors for hardware-assisted 2D block loads of paged K/V and, when layout constraints permit, Q loads and output stores, primarily targeting Intel Xe2/Xe3 GPUs.

## Basic Derived Values

The wrapper derives:

- `block_size = v.shape[1]`
- `num_seqs = len(seqused_k)`
- `num_query_heads = q.shape[1]`
- `num_kv_heads = k.shape[2]`
- `num_queries_per_kv = num_query_heads // num_kv_heads`
- `head_size = q.shape[2]`
- `head_size_padded = triton.next_power_of_2(head_size)`

It also computes whether optional features are active:

- `use_per_token_head_scales`
- `use_mm_prefix`
- `use_alibi_slopes`
- `use_qq_bias`

## BLOCK_M And BLOCK_Q

The wrapper computes:

- `BLOCK_M = 16 if num_queries_per_kv <= 16 else next_power_of_2(num_queries_per_kv)`
- `BLOCK_Q = BLOCK_M // num_queries_per_kv`

`BLOCK_M` is the flattened query-row tile dimension over `(query positions * query heads per KV head)`.

`BLOCK_Q` is how many query token positions fit in one program for one KV-head group.

For common GQA/MQA shapes, one program handles a small block of query positions and all query heads that map to one KV head.

## total_num_q_blocks

`query_len[i]` is the number of new query tokens being processed for sequence `i` in the current scheduler step. It is not the sequence's total KV length: during prefill it may be a prompt chunk containing many tokens, while during ordinary decode it is usually `1`.

Conceptually, the batch has a `query_lens` vector with shape `[num_seqs]`, but the wrapper receives its prefix-sum representation `cu_seqlens_q` with shape `[num_seqs + 1]`. The length of sequence `i` is recovered as `query_len[i] = cu_seqlens_q[i + 1] - cu_seqlens_q[i]`.

The actual query tensor packs all sequences together with shape `[total_query_tokens, num_query_heads, head_size]`, where `total_query_tokens = sum_i query_len[i]` and therefore `q.shape[0]` is the total rather than any individual sequence's query length.

Ideally the kernel would launch:

- `sum_i ceil(query_len[i] / BLOCK_Q)`

Each sequence needs `ceil(query_len[i] / BLOCK_Q)` programs because one program handles at most `BLOCK_Q` query positions for one KV-head group. Summing that expression would produce exactly the number of useful query-block indices and no programs would be wasted.

The problem is that `q` packs all query tokens from the batch into one tensor, so `q.shape[0]` exposes only `sum_i query_len[i]`. The individual lengths are represented by `cu_seqlens_q`, which is normally device-resident; reading those values on the CPU to calculate the exact launch size can introduce a device-to-host transfer and synchronization on every attention call.

So the wrapper uses an upper bound:

- `total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs`

The bound follows from `ceil(x / B) <= floor(x / B) + 1`: summing one extra possible block per sequence gives `sum_i floor(query_len[i] / BLOCK_Q) + num_seqs`, and that is no greater than `floor(sum_i query_len[i] / BLOCK_Q) + num_seqs`. The final expression needs only the packed tensor shape and sequence count, both already available as host-side metadata.

For example, suppose query lengths are `[3, 3]` and `BLOCK_Q = 4`. The exact launch needs `ceil(3 / 4) + ceil(3 / 4) = 2` query blocks, while the upper bound gives `6 // 4 + 2 = 3`. The third query-block program is harmless surplus work.

Inside `kernel_unified_attention(...)`, `resolve_seq_and_query_len(...)` from `vllm/v1/attention/ops/triton_attention_helpers.py` uses a binary search over `cu_seqlens_q` to map each global query-block program ID to a sequence and a local query-block index. The kernel checks `q_block_local_idx * BLOCK_Q >= cur_batch_query_len` and immediately returns when the upper bound assigned a nonexistent block, as the third program does in the example.

This trades a small amount of possible GPU overlaunch for avoiding a CPU synchronization in the latency-sensitive launch path. The second grid dimension still launches one program per KV head for every query-block index, so a surplus query block creates surplus programs across all KV heads, but each exits before loading Q/K/V or computing attention.

## Tile Size Selection

The wrapper computes two tile sizes:

- `TILE_SIZE_PREFILL`
- `TILE_SIZE_DECODE`

Here, `element_size` is `q.element_size()`: the number of bytes used by one scalar element of the query tensor. It is `1` for FP8 or INT8, `2` for FP16 or BF16, and `4` for FP32; it is unrelated to `head_size`, sequence length, or KV-cache block size.

The query element width matters because it changes how many bytes a tile loads and processes. In this implementation the decode kernel uses a KV tile size of at least `32` for one-byte FP8 queries, while query dtypes with elements of two or more bytes use the default decode tile size of `16`.

The helper `_get_tile_size(...)` returns:

- 32 for prefill by default.
- 16 for decode when element size is at least 2.
- 32 for decode when element size is 1.
- 32 for some Gemma3 decode cases.

When tensor descriptor mode is enabled, tile size is clamped to `block_size` because `BLOCK_SIZE % TILE_SIZE == 0` is required.

## Sliding Window And Chunked Attention

The wrapper converts:

- `sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0`

If `chunk_lookback > -1`, it derives:

- `chunk_size = sliding_window_val // (chunk_lookback + 1)`

Chunked attention uses chunk-aligned masking and takes precedence over ordinary sliding-window masking inside helpers.

## Tensor Descriptor Gating

`use_td` can enable tensor descriptors.

The wrapper gates Q/output tensor descriptors separately through `use_td_qo`.

Q/output tensor descriptors require:

- `num_queries_per_kv` is a power of 2.
- `head_size == head_size_padded`.
- query heads are contiguous with `q.stride(1) == head_size`.
- output heads are contiguous with `out.stride(1) == head_size`.

KV tile descriptors can still be used when Q/output descriptors are disabled.

## 2D Versus 3D Dispatch

The wrapper selects 3D mode only when all of these are true:

- segment scratch tensors exist.
- `max_seqlen_q <= 1`. (decode)
- `num_seqs <= seq_threshold_3D`.
- batch invariance is disabled.

So 3D mode is a small-decode optimization.

Otherwise it uses 2D mode.

2D grid:

- `(total_num_q_blocks, num_kv_heads)`

3D grid:

- `(total_num_q_blocks, num_kv_heads, num_par_softmax_segments)`

## Kernel Launch

The wrapper passes a large signature into `kernel_unified_attention[...]`.

The arguments fall into groups:

- output and optional segment scratch pointers.
- Q pointer and KV cache pointers.
- sinks, block table, sequence lengths, ALiBi, QQ bias, MM prefix range.
- scalar scales and softcap.
- head counts and layout strides.
- block size, tile size, head size, padded head size.
- feature toggles as `tl.constexpr`.
- scale-cache pointers and strides.
- quant mode constants.
- chunked attention constants.
- tensor descriptor toggles.

## reduce_segments Launch

If `use_3d` is true, the wrapper launches:

- `reduce_segments[(q.shape[0], num_query_heads)](...)`

This reduces per-segment partial attention outputs into final output.

## Key Files

- `vllm/v1/attention/ops/triton_unified_attention.py`
- `vllm/v1/attention/ops/triton_attention_helpers.py`
- `vllm/v1/attention/backends/triton_attn.py`
