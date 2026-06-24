# Adjacent Triton Paged And Decode Kernels

This note covers Triton attention kernels that exist near `TRITON_ATTN` but are not the standard `TritonAttentionBackend` decoder path.

## Why This File Exists

There are real Triton decode and paged attention kernels in `vllm/v1/attention/ops`.

However, standard `TritonAttentionBackend.forward(...)` calls:

- `unified_attention(...)`

It does not call:

- `decode_attention_fwd(...)`
- `chunked_prefill_paged_decode(...)`
- `kernel_paged_attention_2d`

So these kernels should be documented, but separated from the standard `TRITON_ATTN` call path.

## triton_decode_attention.py

The file is:

- `vllm/v1/attention/ops/triton_decode_attention.py`

It implements memory-efficient paged decode attention.

The public wrapper is:

- `decode_attention_fwd(...)`

It dispatches to:

- `decode_attention_fwd_normal(...)`
- `decode_attention_fwd_grouped(...)`

The important kernels are:

- `_fwd_kernel_stage1`
- `_fwd_grouped_kernel_stage1`
- `_fwd_kernel_stage2`

## triton_decode_attention Stage 1

`_fwd_kernel_stage1` handles normal decode.

Program IDs are:

- batch index
- query head
- KV split id

It splits the KV sequence into `NUM_KV_SPLITS`.

For each split, it:

- loads one query vector.
- walks a slice of paged K/V.
- applies FP8 dequant scales when needed.
- computes scores.
- applies optional logit cap.
- runs online softmax over the split.
- stores partial output and `e_max + log(e_sum)` into an intermediate tensor.

This is a split-KV decode design.

## Grouped Stage 1

`_fwd_grouped_kernel_stage1` is the grouped-query variant.

It is used when the grouped layout can process multiple query heads per program more efficiently.

The exact wrapper selection happens in `decode_attention_fwd(...)`.

## Stage 2

`_fwd_kernel_stage2` reduces split-KV partials.

It combines intermediate outputs using their stored softmax normalization state.

This is conceptually similar to `reduce_segments(...)`, but for the older/separate decode kernel family.

## Users Of triton_decode_attention

Search shows this file is used by:

- `vllm/v1/attention/backends/mla/triton_mla.py`
- `vllm/v1/attention/ops/triton_turboquant_decode.py`
- kernel tests in `tests/kernels/attention/test_triton_decode_attention.py`

It is not called by `vllm/v1/attention/backends/triton_attn.py`.

## chunked_prefill_paged_decode.py

The file is:

- `vllm/v1/attention/ops/chunked_prefill_paged_decode.py`

The relevant Triton kernel is:

- `kernel_paged_attention_2d`

The public wrapper is:

- `chunked_prefill_paged_decode(...)`

This path combines context/prefill and paged decode behavior for chunked prefill style backends.

It is used by ROCm attention paths, not by standard `TritonAttentionBackend`.

## kernel_paged_attention_2d

`kernel_paged_attention_2d` is a paged attention kernel with grid:

- sequence index
- KV head index

It handles:

- query-to-KV-head grouping.
- block table lookup.
- paged key/value cache reads.
- sinks.
- ALiBi.
- sliding window.
- FP8 output.

It is useful to study as another paged attention design, but the standard `TRITON_ATTN` path uses `kernel_unified_attention(...)` instead.

## prefix_prefill.py

The file is:

- `vllm/v1/attention/ops/prefix_prefill.py`

It defines:

- `_fwd_kernel`
- `_fwd_kernel_alibi`
- `context_attention_fwd(...)`

These are prefix prefill kernels used by adjacent chunked/prefix paths.

Do not confuse this file with `triton_prefill_attention.py`, which also defines a `context_attention_fwd(...)` wrapper.

## paged_attn.py

The file is:

- `vllm/v1/attention/ops/paged_attn.py`

It defines:

- `PagedAttention`

This class wraps platform/custom ops for cache splitting and cache writes.

It is used by ROCm attention paths and older/custom paged attention paths.

It is not the standard vLLM-owned Triton unified kernel.

## Key Takeaway

There are multiple Triton attention kernels in vLLM.

For standard `TRITON_ATTN`, the main kernels are:

- cache update kernels from `triton_reshape_and_cache_flash.py`
- `kernel_unified_attention`
- `reduce_segments`
- encoder `context_attention_fwd(...)`

The separate paged decode kernels exist, but they serve other paths.

## Key Files

- `vllm/v1/attention/ops/triton_decode_attention.py`
- `vllm/v1/attention/ops/chunked_prefill_paged_decode.py`
- `vllm/v1/attention/ops/prefix_prefill.py`
- `vllm/v1/attention/ops/paged_attn.py`
- `vllm/v1/attention/backends/mla/triton_mla.py`
- `vllm/v1/attention/ops/triton_turboquant_decode.py`
- `vllm/v1/attention/backends/rocm_attn.py`

