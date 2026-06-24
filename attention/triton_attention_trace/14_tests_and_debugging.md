# Triton Attention Tests And Debugging

This note lists useful tests and debugging entry points for `TRITON_ATTN` and adjacent Triton attention kernels.

## Standard Unified Attention Tests

Start with:

- `tests/kernels/attention/test_triton_unified_attention.py`

This test file imports:

- `unified_attention`

It compares against reference paged attention implementations.

It covers cases such as:

- paged KV cache reads.
- different query and KV head counts.
- sliding window.
- tensor descriptor mode.
- MM prefix style masking.
- quantized KV modes.

## Encoder / Context Attention Tests

Read:

- `tests/kernels/attention/test_triton_prefill_attention.py`

This covers:

- `context_attention_fwd(...)`
- direct Q/K/V attention.
- causal and non-causal behavior.
- sliding window behavior.

This is useful for understanding the encoder/direct-context kernel.

## Cache Update Tests

Read:

- `tests/kernels/attention/test_cache.py`
- `tests/quantization/test_per_token_kv_cache.py`

These cover:

- `triton_reshape_and_cache_flash(...)`
- `triton_reshape_and_cache_flash_per_token_head_quant(...)`
- slot mapping behavior.
- per-token-head scale writes.
- INT8/FP8 cache quantization behavior.

## Separate Decode Kernel Tests

Read:

- `tests/kernels/attention/test_triton_decode_attention.py`

This covers:

- `decode_attention_fwd(...)`
- stage 1 split-KV decode kernels.
- stage 2 reduction kernel.

Remember that this is adjacent decode infrastructure, not the standard `TritonAttentionBackend` decoder path.

## Backend Selection Tests

Read:

- `tests/v1/attention/test_attention_backends.py`
- `tests/v1/attention/test_rocm_attention_backends_selection.py`

These are useful for backend eligibility and platform selection behavior.

## Compile And Custom Op Tests

Read compile-related tests involving:

- `vllm::unified_attention_with_output`

Useful files include:

- `tests/compile/test_config.py`
- `tests/compile/passes/test_fusion_attn.py`
- `tests/compile/passes/distributed/test_sequence_parallelism.py`

These explain why attention is often hidden behind an opaque custom-op boundary for compilation.

## Debugging The Call Path

For standard decoder `TRITON_ATTN`, follow this path:

1. `vllm/model_executor/layers/attention/attention.py`
2. `TritonAttentionImpl.do_kv_cache_update(...)`
3. `triton_reshape_and_cache_flash(...)` or per-token-head variant
4. `TritonAttentionImpl.forward(...)`
5. `unified_attention(...)`
6. `kernel_unified_attention(...)`
7. optionally `reduce_segments(...)`

For encoder attention, follow:

1. `TritonAttentionImpl.forward(...)`
2. `_forward_encoder_attention(...)`
3. `context_attention_fwd(...)`
4. `_fwd_kernel`

For adjacent decode infrastructure, follow:

1. caller such as `TRITON_MLA` or TurboQuant
2. `decode_attention_fwd(...)`
3. `_fwd_kernel_stage1` or `_fwd_grouped_kernel_stage1`
4. `_fwd_kernel_stage2`

## Environment Flags

Useful environment/config flags include:

- `VLLM_TRITON_ATTN_USE_TD`: controls tensor descriptor mode.
- `VLLM_BATCH_INVARIANT`: disables 3D segmented path and affects deterministic behavior.
- `--attention-backend TRITON_ATTN`: forces backend selection, subject to validation.

## Key Files

- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/ops/triton_unified_attention.py`
- `vllm/v1/attention/ops/triton_attention_helpers.py`
- `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`
- `vllm/v1/attention/ops/triton_prefill_attention.py`
- `vllm/v1/attention/ops/triton_decode_attention.py`

