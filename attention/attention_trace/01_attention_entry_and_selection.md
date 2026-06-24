# Attention Entry And Selection

This note maps the path from launch-time inputs to the final backend class that gets attached to an attention layer.

## The Minimal Path

At a high level, the current v1 selection flow is:

1. User passes `--attention-backend` or `attention_config`
2. vLLM normalizes that into `AttentionConfig`
3. Each `Attention` layer gathers its local requirements
4. `get_attn_backend(...)` builds an `AttentionSelectorConfig`
5. The current platform chooses a valid backend class
6. The layer instantiates that backend's impl

The key files are:

- `vllm/engine/arg_utils.py`
- `vllm/config/attention.py`
- `vllm/v1/attention/selector.py`
- `vllm/platforms/cuda.py`
- `vllm/model_executor/layers/attention/attention.py`

## Launch-Side Entry Points

There are two main user-facing ways to influence the backend:

- `--attention-backend` Added in the `AttentionConfig` argument group in `vllm/engine/arg_utils.py`
- `attention_config` Passed as an object or dict in Python APIs such as `vllm.entrypoints.llm.LLM`

`LLM(...)` converts `dict | None | instance` into a concrete `AttentionConfig` instance in `vllm/entrypoints/llm.py`.

`EngineArgs` does an additional normalization pass in `vllm/engine/arg_utils.py`:

- If `attention_backend` is set, it is treated as an override
- `attention_backend` and `attention_config.backend` are mutually exclusive
- The override reuses `AttentionConfig.validate_backend_before(...)`

## `AttentionConfig`

The core config object is `vllm/config/attention.py`.

Important fields for generic attention are:

- `backend`
- `flash_attn_version`
- `use_prefill_decode_attention`
- `use_trtllm_attention`
- `disable_flashinfer_q_quantization`
- `use_prefill_query_quantization`
- `use_non_causal`
- `flex_attn_block_m`
- `flex_attn_block_n`
- `flex_attn_q_block_size`
- `flex_attn_kv_block_size`

Important normalization rule:

- `"auto"` becomes `None`

That matters because `None` is what triggers platform auto-selection.

## What The Layer Contributes

The backend is not selected from config alone. The layer contributes its own requirements in `vllm/model_executor/layers/attention/attention.py`.

Important per-layer inputs are:

- `head_size`
- default compute dtype
- `kv_cache_dtype`
- `has_sink`
- `use_mm_prefix`
- `attn_type`
- whether per-head quant scales are required

`sliding_window` is also computed at layer construction time, but note the important nuance:

- `sliding_window` affects runtime behavior and KV-cache spec construction
- it is not a direct field in `AttentionSelectorConfig`
- backend eligibility is therefore expressed indirectly, through supported attention types, block sizes, and later runtime compatibility

## `AttentionSelectorConfig`

`vllm/v1/attention/selector.py` packages the layer and engine state into an `AttentionSelectorConfig`.

The selection inputs include:

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

`block_size` only becomes part of selection if the user explicitly fixed the KV manager block size.

## Explicit Backend Selection

If `AttentionConfig.backend` is not `None`, platform selection first validates that exact backend in `current_platform.get_attn_backend_cls(...)`.

On CUDA, `vllm/platforms/cuda.py` does:

1. import the selected backend class
2. call `backend_class.validate_configuration(...)`
3. error out if any invalid reasons are returned
4. otherwise use that backend directly

So "explicit backend selection" still means "explicit backend request subject to capability validation."

## Auto-Selection When Nothing Is Specified

If no backend is selected, the platform chooses one.

On CUDA the logic is:

1. build a backend priority list
2. validate each candidate against the current selector config
3. keep only valid candidates
4. choose the highest-priority valid backend

This logic lives in:

- `_get_backend_priorities(...)`
- `get_valid_backends(...)`
- `get_attn_backend_cls(...)`

The priority list is platform- and hardware-specific. For non-MLA CUDA paths it is typically some ordering of:

- `FLASH_ATTN`
- `FLASHINFER`
- `TRITON_ATTN`
- `FLEX_ATTENTION`
- `TURBOQUANT`

The exact order depends on device capability.

## What Makes A Backend Eligible

Eligibility is determined by `AttentionBackend.validate_configuration(...)` in `vllm/v1/attention/backend.py`.

That function checks:

- head size
- compute dtype
- KV-cache dtype
- block size
- `mm_prefix`
- sink support
- sparse vs non-sparse
- per-head quant scales
- compute capability
- attention type
- non-causal support
- batch invariance
- KV connector support
- backend-specific combination logic

So "default backend selection" is not one simple if/else. It is a platform priority list filtered by this capability contract.

## KV Cache Layout Side Effect

After the backend class is chosen, `selector.py` asks:

- `backend.get_required_kv_cache_layout()`

If the backend requires a specific KV layout, selector code sets that layout globally before returning the backend class.

This is an important detail for plugin work: backend selection can have layout side effects before runtime execution starts.

## Best Files To Read Next

- `vllm/config/attention.py`
- `vllm/engine/arg_utils.py`
- `vllm/v1/attention/selector.py`
- `vllm/platforms/cuda.py`
- `vllm/model_executor/layers/attention/attention.py`
