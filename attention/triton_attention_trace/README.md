# Triton Attention Trace

This folder maps the standard vLLM `TRITON_ATTN` backend end to end.

The focus is standard MHA/MQA/GQA attention and the vLLM-owned Triton kernels behind it.

The short mental model is:

- `TritonAttentionBackend` declares broad capabilities and uses a paged KV-cache layout compatible with FlashAttention-style cache storage.
- `TritonAttentionImpl.do_kv_cache_update(...)` writes current K/V into the paged KV cache with Triton cache-update kernels.
- `TritonAttentionImpl.forward(...)` calls `unified_attention(...)` for standard decoder attention.
- `unified_attention(...)` launches `kernel_unified_attention(...)` for decoder prefill, decoder decode, and mixed batches.
- For small decode batches, `kernel_unified_attention(...)` can run in segmented 3D mode and then call `reduce_segments(...)`.
- Encoder and encoder-only attention use `context_attention_fwd(...)`, which operates on direct packed Q/K/V tensors and does not use paged KV cache.

Read in this order:

1. `01_selection_and_table_support.md`
2. `02_backend_contract.md`
3. `03_metadata_builder.md`
4. `04_standard_decoder_call_path.md`
5. `05_kv_cache_update_kernels.md`
6. `06_unified_attention_wrapper.md`
7. `07_kernel_unified_attention.md`
8. `08_decode_mode_inside_unified_attention.md`
9. `09_reduce_segments_3d_decode.md`
10. `10_context_prefill_encoder_kernel.md`
11. `11_attention_helpers.md`
12. `12_feature_variants_and_limits.md`
13. `13_adjacent_paged_decode_kernels.md`
14. `14_tests_and_debugging.md`

The most important source files are:

- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/ops/triton_unified_attention.py`
- `vllm/v1/attention/ops/triton_attention_helpers.py`
- `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`
- `vllm/v1/attention/ops/triton_prefill_attention.py`
- `vllm/v1/attention/ops/triton_decode_attention.py`
- `vllm/v1/attention/ops/chunked_prefill_paged_decode.py`
- `vllm/v1/attention/ops/prefix_prefill.py`
- `vllm/v1/attention/ops/paged_attn.py`

The key naming warning is that `triton_prefill_attention.py` is not the normal decoder-prefill path for standard `TRITON_ATTN`. Standard decoder prefill writes K/V to paged cache and then uses `kernel_unified_attention(...)`; `triton_prefill_attention.py` is used by this backend for encoder/direct Q/K/V attention.

