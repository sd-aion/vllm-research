# FlashAttention Selection And Versions

This note maps how vLLM chooses `FLASH_ATTN` and how that backend chooses FA2, FA3, or FA4 underneath.

## Launch Inputs

The user-facing controls are the normal attention controls:

- `--attention-backend FLASH_ATTN`
- `--attention-config.backend FLASH_ATTN`
- `-ac.backend FLASH_ATTN`
- Python `AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN)`
- Python `LLM(..., attention_backend="FLASH_ATTN")`

These fields normalize into `AttentionConfig` from `vllm/config/attention.py`.

Important `AttentionConfig` fields for FlashAttention are:

- `backend`: selects the backend enum or leaves it as `None` for auto-selection.
- `flash_attn_version`: optionally forces FA2, FA3, or FA4 for the `FLASH_ATTN` backend.
- `flash_attn_max_num_splits_for_cuda_graph`: controls the upper bound used by FA3 scheduler metadata under full CUDA graph capture.
- `use_non_causal`: asks for non-causal decoder attention, which `FLASH_ATTN` supports.

`flash_attn_max_num_splits_for_cuda_graph` only matters for the FA3 ahead-of-time scheduler path when full CUDA graphs are enabled. FA3 can split long attention work into multiple pieces, and those splits require intermediate buffers whose shape depends on the maximum split count. During CUDA graph capture, vLLM needs those buffers and scheduler metadata addresses/shapes to be stable across replay, so `FlashAttentionMetadataBuilder` preallocates scheduler metadata and passes a bounded `num_splits` value instead of letting FA3 use an unbounded heuristic. The default value comes from `AttentionConfig` in `vllm/config/attention.py`; the builder reads it in `FlashAttentionMetadataBuilder.__init__(...)` in `vllm/v1/attention/backends/flash_attn.py`, stores it as `self.max_num_splits`, and later passes it into `get_scheduler_metadata(...)` when the current batch is small enough to use the captured full CUDA graph. Outside that full-CUDA-graph FA3 path, `max_num_splits` is usually `0`, which means FA3 can choose its split count heuristically instead of being constrained for graph replay.

## Registry Entry

The enum entry lives in `vllm/v1/attention/backends/registry.py`:

- `AttentionBackendEnum.FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"`

That enum value resolves to the actual backend class through `AttentionBackendEnum.get_class()`.

This matters for plugin work because vLLM selects a backend class first; runtime impl construction happens later inside the `Attention` layer.

## Explicit Selection

When the user explicitly selects `FLASH_ATTN`, CUDA platform code validates that exact backend before using it.

The relevant path is:

1. `AttentionConfig.backend` stores `AttentionBackendEnum.FLASH_ATTN`.
2. `Attention` in `vllm/model_executor/layers/attention/attention.py` calls `get_attn_backend(...)`.
3. `get_attn_backend(...)` in `vllm/v1/attention/selector.py` builds an `AttentionSelectorConfig`.
4. `current_platform.get_attn_backend_cls(...)` in `vllm/platforms/cuda.py` calls `FlashAttentionBackend.validate_configuration(...)`.
5. If validation returns no invalid reasons, CUDA returns the class path for `FlashAttentionBackend`.

Explicit selection is therefore still subject to capability validation.

## Auto Selection

When no backend is specified, CUDA uses a priority list and picks the first valid backend.

For standard MHA/MQA/GQA attention, `vllm/platforms/cuda.py` currently orders CUDA backends like this:

- Blackwell / SM 10.x: `FLASHINFER`, then `FLASH_ATTN`, then `TRITON_ATTN`, then `FLEX_ATTENTION`, then `TURBOQUANT`.
- Ampere/Hopper / SM 8.x-9.x: `FLASH_ATTN`, then `FLASHINFER`, then `TRITON_ATTN`, then `FLEX_ATTENTION`, then `TURBOQUANT`.

This is why `FLASH_ATTN` is typically the default standard attention backend on Ampere and Hopper, but not necessarily the first choice on Blackwell.

## Selector Inputs

The selector does not just ask for "FlashAttention yes/no".

The `AttentionSelectorConfig` in `vllm/v1/attention/selector.py` carries:

- `head_size`
- `dtype`
- `kv_cache_dtype`
- `block_size`
- `use_mla`
- `has_sink`
- `use_sparse`
- `use_mm_prefix`
- `use_per_head_quant_scales`
- `attn_type`
- `use_non_causal`
- `use_batch_invariant`
- `use_kv_connector`

`FlashAttentionBackend.validate_configuration(...)` inherits the common validation logic from `AttentionBackend` in `vllm/v1/attention/backend.py` and uses the FlashAttention-specific capability methods in `vllm/v1/attention/backends/flash_attn.py`.

## Version Selection

`FLASH_ATTN` is one vLLM backend name, but it can call FA2, FA3, or FA4 underneath.

The version logic lives in `get_flash_attn_version(...)` in `vllm/v1/attention/backends/fa_utils.py`.

The default CUDA behavior is:

- SM90 / Hopper prefers FA3 when FA3 is available.
- SM100+ / Blackwell prefers FA4 when FA4 is available.
- Other CUDA devices fall back to FA2.
- `AttentionConfig.flash_attn_version` can override the default when the requested version is supported.

There are important fallback rules:

- FA3 and FA4 do not support ALiBi in this path, so ALiBi forces FA2.
- FA4 can fall back to FA2 for unsupported Blackwell head sizes due to TMEM limits.
- Batch invariance currently forces FA4 back to FA2.
- Some SM90 cases with FA3 limitations can upgrade to FA4 if FA4 is supported.

## Why The Table Has Three FLASH_ATTN Rows

`docs/design/attention_backends.md` expands the single `FLASH_ATTN` backend into FA2, FA3, and FA4 rows.

That expansion is generated by `tools/pre_commit/generate_attention_backend_docs.py`.

The table rows are version-specific feature summaries over the same backend class, not three separate backend classes for standard attention.

## Key Files

- `vllm/config/attention.py`
- `vllm/v1/attention/backends/registry.py`
- `vllm/v1/attention/selector.py`
- `vllm/platforms/cuda.py`
- `vllm/v1/attention/backends/fa_utils.py`
- `vllm/v1/attention/backends/flash_attn.py`
- `tools/pre_commit/generate_attention_backend_docs.py`
