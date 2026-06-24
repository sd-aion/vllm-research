# FlashAttention Backend Contract And Feature Table

This note maps the `FLASH_ATTN` rows in `docs/design/attention_backends.md` to concrete backend methods in `vllm/v1/attention/backends/flash_attn.py`.

## Core Backend Class

The backend class is:

- `FlashAttentionBackend` in `vllm/v1/attention/backends/flash_attn.py`

It implements the common `AttentionBackend` contract from `vllm/v1/attention/backend.py`.

The most important class methods are:

- `get_name()`: returns `"FLASH_ATTN"`.
- `get_impl_cls()`: returns `FlashAttentionImpl`.
- `get_builder_cls()`: returns `FlashAttentionMetadataBuilder`.
- `get_kv_cache_shape(...)`: defines the logical KV-cache tensor shape.
- `get_kv_cache_stride_order(...)`: defines physical layout order for the KV cache.
- `supports_head_size(...)`: implements the flexible head-size support behind the table's `Any`.
- `supports_kv_cache_dtype(...)`: controls `auto`, fp16/bf16, and FP8 eligibility.
- `supports_sink()`: depends on whether the selected FlashAttention version supports sinks.
- `supports_non_causal()`: returns `True`.
- `supports_attn_type(...)`: supports decoder, encoder, encoder-only, and encoder-decoder.
- `supports_compute_capability(...)`: requires CUDA compute capability at least 8.0.
- `supports_combination(...)`: adds combination-level rejection, especially sink support below SM90.

`get_kv_cache_shape(...)` and `get_kv_cache_stride_order(...)` are covered in more detail in `04_kv_cache_and_slot_mapping.md` because they are the two methods that define how FlashAttention's paged KV cache is shaped logically and laid out physically in memory.

## Dtypes

The table says:

- FA2: fp16, bf16
- FA3: fp16, bf16
- FA4: fp16, bf16

This maps to:

- `FlashAttentionBackend.supported_dtypes = [torch.float16, torch.bfloat16]`

There is no fp32 model dtype support for standard `FLASH_ATTN`.

## KV Dtypes

The base class field says:

- `supported_kv_cache_dtypes = ["auto", "float16", "bfloat16"]`

`supports_kv_cache_dtype(...)` adds special FP8 handling in the backend source:

- `None` is accepted.
- `auto`, `float16`, and `bfloat16` are accepted normally.
- `fp8` and `fp8_e4m3` are accepted on XPU.
- On CUDA, `fp8` and `fp8_e4m3` require FA3 on SM90.

The generated table shows FA3 with `fp8`, `fp8_e4m3`, and `fp8_e5m2` because `tools/pre_commit/generate_attention_backend_docs.py` expands the FA3 row with version-specific FP8 metadata. The direct source branch to read is still `supports_kv_cache_dtype(...)`, and the generated table is the user-facing summary of intended feature support.

## Block Sizes

The table says `%16`.

This maps to:

- `get_supported_kernel_block_sizes()`: usually returns `[MultipleOf(16)]`.
- `get_kv_cache_shape(...)`: raises if `block_size % 16 != 0`.
- `supports_block_size(...)`: inherited from `AttentionBackend`, where a framework block size is valid if it is a multiple of a kernel-supported granularity.

There is one important hybrid-model special case:

- If the model is hybrid and Mamba cache dtype is float32, `get_supported_kernel_block_sizes()` returns `[16, 32, 64]` to avoid a NaN propagation issue.

## Head Sizes

The table says `Any`, but that does not mean literally every integer.

`supports_head_size(...)` says:

- head size must be divisible by 8.
- head size up to 256 is supported.
- head size up to 512 is supported when FA4 is supported.
- otherwise larger head sizes are rejected.

So the table's `Any` means "not a short fixed enumerated list like 64/128/256", not "all values without constraints".

## Sink

The table says:

- FA2: no sink support
- FA3: sink support
- FA4: sink support

This maps to:

- `supports_sink()` checks `is_flash_attn_varlen_func_available()` and `flash_attn_supports_sinks()`.
- `flash_attn_supports_sinks()` in `fa_utils.py` returns true for FA3 and FA4.
- `supports_combination(...)` rejects sinks on compute capability below 9.0.
- `FlashAttentionImpl.__init__(...)` asserts that sinks match the number of query heads and are supported by the selected FlashAttention version.

In the kernel call, sinks are passed as `s_aux` to `flash_attn_varlen_func(...)`.

## Non-Causal

The table says non-causal is supported.

This maps to:

- `supports_non_causal()` returns `True`.
- `FlashAttentionMetadata.causal` carries the runtime causal flag.
- `FlashAttentionImpl.forward(...)` passes `causal=attn_metadata.causal` to `flash_attn_varlen_func(...)`.
- Encoder attention hard-codes `causal=False`.

Non-causal means the attention mask is bidirectional instead of lower-triangular, so token `i` can attend to tokens after `i`.

## MM Prefix

The table says MM prefix is not supported.

`FlashAttentionBackend` does not override `supports_mm_prefix()`, so it inherits the base default:

- `supports_mm_prefix() -> False`

That means `use_mm_prefix=True` causes `validate_configuration(...)` to reject `FLASH_ATTN` with "partial multimodal token full attention not supported".

This is distinct from multimodal encoder attention. `FLASH_ATTN` can be used for some encoder attention paths, but the decoder-side `mm_prefix` feature in the standard paged-KV-cache path is not supported by this backend.

## DCP

The table says DCP is supported.

This maps to:

- `FlashAttentionImpl.can_return_lse_for_decode = True`.
- `AttentionImplBase.__new__(...)` sets `need_to_return_lse_for_decode = (dcp_world_size > 1 and can_return_lse_for_decode)`.
- `check_attention_cp_compatibility(...)` in `vllm/v1/worker/cp_utils.py` requires `need_to_return_lse_for_decode` when DCP is enabled.
- `_forward_with_dcp(...)` calls `flash_attn_varlen_func(..., return_softmax_lse=True)` and combines partial outputs using LSE-aware distributed helpers.

DCP support is an impl-level runtime property, not just a backend-class table flag.

## Attention Types

The table says `All`.

This maps to `supports_attn_type(...)`, which accepts:

- `AttentionType.DECODER`
- `AttentionType.ENCODER`
- `AttentionType.ENCODER_ONLY`
- `AttentionType.ENCODER_DECODER`

Decoder and cross-attention use the KV-cache path.

Encoder and encoder-only attention use `_forward_encoder_attention(...)`, which calls FlashAttention directly on Q/K/V tensors without paged KV-cache update.

## Compute Capability

The table says:

- FA2: `>=8.0`
- FA3: `9.x`
- FA4: `>=10.0`

The backend class-level check is:

- `supports_compute_capability(...) -> capability >= DeviceCapability(8, 0)`

The version-specific restrictions come from `get_flash_attn_version(...)` and the generated docs metadata.

In practice:

- Ampere can use FA2.
- Hopper normally uses FA3.
- Blackwell normally uses FA4, with fallbacks for unsupported combinations.

## Batch Invariance

The feature table does not show batch invariance, but it matters for selection.

`FlashAttentionBackend.supports_batch_invariance()` returns `True`.

At runtime, `VLLM_BATCH_INVARIANT` affects scheduler metadata and `num_splits`, and `get_flash_attn_version(...)` avoids FA4 when batch invariance is enabled.

## Per-Head Quant Scales

The feature table does not show per-head quant scales directly.

`supports_per_head_quant_scales()` returns true when the selected FlashAttention version is at least FA3.

This matters when the layer requests per-head quant scales through `use_per_head_quant_scales`.

## Key Files

- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/fa_utils.py`
- `vllm/v1/attention/backend.py`
- `vllm/v1/worker/cp_utils.py`
- `docs/design/attention_backends.md`
- `tools/pre_commit/generate_attention_backend_docs.py`
