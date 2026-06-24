# TurboQuant Attention Trace

This folder maps the vLLM `TURBOQUANT` attention backend from selection through KV-cache storage, prefill, continuation prefill, and split-KV decode.

TurboQuant is primarily a compressed KV-cache backend. It does not replace every attention computation with a single TurboQuant kernel: first-chunk prefill uses uncompressed Q/K/V with FlashAttention or PyTorch SDPA, while decode reads the compressed paged cache with dedicated Triton kernels.

The shortest useful mental model is:

- A `turboquant_*` KV-cache dtype causes vLLM to select `TurboQuantAttentionBackend` for eligible decoder layers.
- Current K/V vectors are compressed by `triton_turboquant_store(...)` and written into a combined uint8 cache slot selected by `slot_mapping`.
- First-chunk prefill computes attention from the original uncompressed K/V because all required tokens are already present in the current batch.
- Decode rotates the query when MSE-quantized keys are used, reads compressed K/V through the request's block table, performs split-KV online softmax, and merges split results.
- Continuation prefill uses synthetic decode for chunks of at most 128 tokens and bulk-dequantizes the old cache for larger chunks.

Read the documents in this order:

1. `01_selection_and_table_support.md`
2. `02_turboquant_algorithm_and_presets.md`
3. `03_kv_cache_shape_and_layout.md`
4. `04_backend_and_impl_contract.md`
5. `05_metadata_and_cuda_graphs.md`
6. `06_runtime_call_path.md`
7. `07_store_launcher_and_cache_addressing.md`
8. `08_value_quantization_and_packing.md`
9. `09_fp8_store_kernel.md`
10. `10_mse_store_kernel.md`
11. `11_prefill_paths.md`
12. `12_continuation_prefill.md`
13. `13_decode_launcher.md`
14. `14_decode_stage1_kernel.md`
15. `15_decode_stage2_reduction.md`
16. `16_full_dequant_kernel.md`
17. `17_distributed_features_and_limits.md`
18. `18_tests_and_debugging.md`

The main source files are:

- `vllm/v1/attention/backends/turboquant_attn.py`
- `vllm/v1/attention/ops/triton_turboquant_store.py`
- `vllm/v1/attention/ops/triton_turboquant_decode.py`
- `vllm/model_executor/layers/quantization/turboquant/config.py`
- `vllm/model_executor/layers/quantization/turboquant/centroids.py`
- `vllm/v1/kv_cache_interface.py`
- `vllm/model_executor/layers/attention/attention.py`
- `tests/quantization/test_turboquant.py`

All paths in this trace are relative to `/home/ubuntu/vllm`.
