# FlashAttention Trace

This folder maps the standard vLLM `FLASH_ATTN` backend end to end.

The focus is standard attention for MHA/MQA/GQA, not MLA decode backends, ROCm AITER FlashAttention, or FlashInfer.

The short mental model is:

- vLLM selects `FlashAttentionBackend` through the normal attention selector.
- `FlashAttentionBackend` declares capabilities, KV-cache shape, metadata builder, and runtime impl class.
- `FlashAttentionMetadataBuilder` turns common batch metadata into FlashAttention-specific launch metadata.
- `FlashAttentionImpl` performs attention by calling FlashAttention library entry points and vLLM-side helper ops.
- KV-cache writes are handled separately from the attention forward because `FLASH_ATTN` sets `forward_includes_kv_cache_update = False`.
- The core attention kernels are behind `flash_attn_varlen_func(...)`, while vLLM owns cache update, distributed combine, and output merge helpers.

Read in this order:

1. `01_selection_and_versions.md`
2. `02_backend_contract_and_feature_table.md`
3. `03_metadata_builder.md`
4. `04_kv_cache_and_slot_mapping.md`
5. `05_forward_prefill_decode.md`
6. `06_kernels_and_ops.md`
7. `07_distributed_flash_attention.md`
8. `08_attention_variants.md`
9. `09_cascade_attention.md`

The most important source files are:

- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/fa_utils.py`
- `vllm/vllm_flash_attn/flash_attn_interface.py`
- `vllm/model_executor/layers/attention/attention.py`
- `vllm/v1/attention/backend.py`
- `vllm/v1/attention/selector.py`
- `vllm/platforms/cuda.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/block_table.py`

