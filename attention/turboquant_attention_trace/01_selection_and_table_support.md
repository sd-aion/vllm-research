# TurboQuant Selection And Table Support

This document maps backend selection and every `TURBOQUANT` entry in `docs/design/attention_backends.md` to code.

## Registry

`AttentionBackendEnum.TURBOQUANT` is registered in `vllm/v1/attention/backends/registry.py` with the class path `vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend`.

The enum resolves that string lazily when selection needs the backend class.

## How A User Selects TurboQuant

The normal entry point is a TurboQuant cache dtype rather than merely `--attention-backend TURBOQUANT`.

Supported launch values are:

- `--kv-cache-dtype turboquant_k8v4`
- `--kv-cache-dtype turboquant_4bit_nc`
- `--kv-cache-dtype turboquant_k3v4_nc`
- `--kv-cache-dtype turboquant_3bit_nc`

These strings are part of `CacheDType` in `vllm/config/cache.py` and map to `torch.uint8` storage in `vllm/utils/torch_utils.py`.

`get_attn_backend(...)` in `vllm/v1/attention/selector.py` carries the requested cache dtype into `AttentionSelectorConfig` and asks the active platform to choose a compatible backend.

On CUDA, `TURBOQUANT` appears last in the normal standard-attention priority list in `vllm/platforms/cuda.py`. Earlier backends reject a `turboquant_*` cache dtype, while `TurboQuantAttentionBackend.supports_kv_cache_dtype(...)` accepts it, so TurboQuant becomes the compatible candidate.

ROCm includes TurboQuant after its native and Triton candidates in `vllm/platforms/rocm.py`. XPU has an explicit fast path in `vllm/platforms/xpu.py` that returns TurboQuant whenever the cache dtype starts with `turboquant_`.

An explicit `--attention-backend TURBOQUANT` still goes through capability validation. It is not sufficient with an ordinary `auto`, fp16, or bf16 cache dtype because this backend intentionally accepts only `turboquant_*` cache dtypes.

## Boundary Layer Protection

`EngineArgs.create_engine_config(...)` in `vllm/engine/arg_utils.py` calls `TurboQuantConfig.get_boundary_skip_layers(...)` when the resolved cache dtype is TurboQuant.

For a dense model, the default skips the first two and last two layers by adding their indices to `cache_config.kv_cache_dtype_skip_layers`. In `Attention.__init__(...)` from `vllm/model_executor/layers/attention/attention.py`, skipped layers replace the TurboQuant cache dtype with `auto`, so those layers select a normal attention backend and receive a normal KV-cache layout.

For hybrid models, boundary skipping is disabled because they may contain only a small number of full-attention layers. `_get_full_attention_layer_indices(...)` recognizes the `layer_types`, `layers_block_type`, and `attn_type_list` model-config conventions.

This creates a valid mixed-backend model: most layers can use packed TurboQuant pages while protected layers use native K/V pages. KV-cache grouping keeps incompatible specs separate.

## FlashAttention Version Override

`EngineArgs.create_engine_config(...)` in `vllm/engine/arg_utils.py` forces `attention_config.flash_attn_version = 2` for TurboQuant when the user leaves the version unset or requests FA3/FA4.

The reason given in code is compatibility between TurboQuant layers and the native boundary layers: the current boundary-layer integration expects the FA2 implementation and is not yet compatible with FlashAttention 3 or newer.

This does not mean TurboQuant decode uses FA2. The override applies to FlashAttention-based paths such as boundary layers and TurboQuant prefill fallbacks; TurboQuant decode remains a dedicated Triton path.

## Table Row

The standard-attention table in `docs/design/attention_backends.md` reports:

- Dtypes: fp16 and bf16.
- KV dtypes: the four `turboquant_*` presets.
- Block sizes: 16, 32, 64, and 128.
- Head sizes: Any.
- Sink: unsupported.
- Non-causal: unsupported.
- MM Prefix: unsupported.
- DCP: unsupported.
- Attention types: Decoder.
- Compute capability: Any.

## Capability Mapping

Activation dtype support comes from `TurboQuantAttentionBackend.supported_dtypes = [torch.float16, torch.bfloat16]` in `vllm/v1/attention/backends/turboquant_attn.py`.

KV dtype support comes from `supported_kv_cache_dtypes` and the broader prefix check in `supports_kv_cache_dtype(...)`.

Block-size support comes from `get_supported_kernel_block_sizes() -> [16, 32, 64, 128]`. The base `supports_block_size(...)` implementation accepts a framework block size divisible by one of these values, so virtual block splitting can make a larger compatible framework page valid.

Head size is shown as `Any` because `supports_head_size(...)` accepts every positive effective head size. The selector may pass an effective size derived from the packed slot rather than the model's original head dimension, so this method deliberately avoids a conventional fixed head-size list.

Sink, non-causal, and MM-prefix support are false because TurboQuant does not override the corresponding false defaults in `AttentionBackend`, and its kernels contain no implementation of those features.

DCP is false because `TurboQuantAttentionImpl` leaves `can_return_lse_for_decode = False`. Its internal stage-two kernel writes an LSE scratch result, but the implementation does not expose the `(output, lse)` runtime contract required by DCP.

Decoder-only support is explicit in `supports_attn_type(...)`.

Compute capability is `Any` because the backend inherits `supports_compute_capability(...) -> True`; individual platform and Triton capabilities still determine whether execution actually succeeds.

## Important Selection Limits

The selector does not have a separate sliding-window capability method. `TurboQuantAttentionImpl.__init__(...)` accepts `sliding_window`, but the value is not retained or used by the TurboQuant kernels, so the current implementation must not be interpreted as implementing sliding-window masking.

`supports_per_head_quant_scales()` returns false because TurboQuant stores its own per-vector packed metadata rather than consuming vLLM's generic per-head KV scale contract.

The backend inherits `supports_kv_connector() -> True`, but connector interoperability must preserve the backend-specific packed bytes and cache spec; it must not reinterpret a TurboQuant page as a standard K/V tensor.
