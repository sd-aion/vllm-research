# Triton Attention Backend Contract

This note maps `TritonAttentionBackend` to the common vLLM attention backend contract.

## Core Backend Class

The backend class is:

- `TritonAttentionBackend` in `vllm/v1/attention/backends/triton_attn.py`

It implements the common `AttentionBackend` contract from `vllm/v1/attention/backend.py`.

The important backend methods are:

- `get_name()`: returns `"TRITON_ATTN"`.
- `get_impl_cls()`: returns `TritonAttentionImpl`.
- `get_builder_cls()`: returns `TritonAttentionMetadataBuilder`.
- `get_kv_cache_shape(...)`: defines the logical KV-cache tensor shape.
- `get_kv_cache_stride_order(...)`: defines physical KV-cache layout.
- `get_supported_kernel_block_sizes()`: declares `%16` block-size support.
- `supports_block_size(...)`: enforces multiples of 16.
- `supports_head_size(...)`: accepts head sizes at least 32.
- `supports_mm_prefix()`: returns true.
- `supports_sink()`: returns true.
- `supports_attn_type(...)`: accepts all attention types.
- `supports_alibi_sqrt()`: returns true.
- `supports_compute_capability(...)`: returns true.

## KV Cache Shape

For normal KV cache dtypes, `get_kv_cache_shape(...)` returns:

- `(num_blocks, 2, block_size, num_kv_heads, head_size)`

For per-token-head quantized KV cache dtypes, it returns:

- `(num_blocks, 2, block_size, num_kv_heads, head_size + scale_pad)`

The extra `scale_pad` is not extra K/V hidden dimension data.

It is reserved storage at the end of each `(block, K-or-V, token-slot, KV-head)` row so Triton can store one `float32` scale next to that row's quantized K or V values.

This is used by `int8_per_token_head` and `fp8_per_token_head`.

The reason `scale_pad` is calculated as `get_dtype_size(torch.float32) // get_dtype_size(cache_dtype)` is that the last dimension of the KV-cache shape is counted in units of the cache dtype, not bytes.

A `float32` scale is 4 bytes.

For `int8_per_token_head` and `fp8_per_token_head`, the cache dtype is 1 byte per element, so one `float32` scale needs `4 / 1 = 4` cache-dtype slots.

So if `head_size = 128` and the cache dtype is int8/FP8, the stored row length becomes `132`: the first `128` entries are quantized K or V values, and the final `4` one-byte slots are reinterpreted together as one `float32` scale.

If a future cache dtype used 2-byte elements, the same formula would reserve `4 / 2 = 2` cache-dtype slots for one `float32` scale.

This is why the formula is based on dtype sizes: it reserves exactly enough inline tail storage, in cache-element units, to hold one 4-byte scale per token/head row.

## KV Cache Stride Order

`get_kv_cache_stride_order(...)` follows `get_kv_cache_layout()` from `vllm/v1/attention/backends/utils.py`.

It supports:

- `NHD`
- `HND`

With `include_num_layers_dimension=True`, the layer dimension is included in the returned permutation.

The important point is that Triton attention can use the same logical cache shape while changing physical stride order based on layout.

## Separate KV Cache Update

`TritonAttentionBackend.forward_includes_kv_cache_update = False`.

That means attention forward and KV-cache update are separate operations.

For decoder attention:

- `do_kv_cache_update(...)` writes K/V into paged cache.
- `forward(...)` later calls `unified_attention(...)` to read from that paged cache.

For encoder and encoder-only attention:

- `do_kv_cache_update(...)` returns without writing cache.
- `_forward_encoder_attention(...)` uses direct Q/K/V tensors.

## Metadata Builder

The builder class is:

- `TritonAttentionMetadataBuilder`

It builds:

- `TritonAttentionMetadata`

This metadata contains normal paged-attention fields plus Triton-specific fields for the 3D segmented decode path and MM prefix.

## Runtime Impl

The impl class is:

- `TritonAttentionImpl`

It handles:

- quantization scale-cache setup
- encoder direct Q/K/V attention
- decoder paged attention through `unified_attention(...)`
- KV-cache update through Triton cache-update kernels
- optional fused RoPE plus KV-cache update on ROCm AITER paths

## Runtime Impl Flags

`TritonAttentionImpl` does not set:

- `can_return_lse_for_decode`
- `supports_pcp`

So it inherits false defaults from `AttentionImplBase`.

This is why standard `TRITON_ATTN` is not DCP-capable and not PCP-capable in the generic compatibility check.

## Quantization Contract

`TritonAttentionBackend` supports several KV quant modes:

- no quantization
- FP8 per-tensor
- INT8 per-token-head
- FP8 per-token-head

`TritonAttentionImpl.__init__(...)` computes:

- `self._kv_quant_mode = get_kv_quant_mode(kv_cache_dtype)`
- `self._is_per_token_head_quant = self._kv_quant_mode.is_per_token_head`

The quant mode changes both cache shape and runtime kernel arguments.

## Tensor Descriptor Mode

`TritonAttentionImpl.__init__(...)` reads:

- `VLLM_TRITON_ATTN_USE_TD`

Tensor descriptor mode controls whether the unified kernel uses `tl.make_tensor_descriptor` for Q/K/V loads and output stores.

The default is platform-sensitive:

- enabled automatically on XPU.
- disabled elsewhere unless forced.

## Key Files

- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/backend.py`
- `vllm/v1/attention/backends/utils.py`
- `vllm/v1/kv_cache_interface.py`
- `vllm/envs.py`
