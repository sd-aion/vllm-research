# Triton KV Cache Update Kernels

This note covers the Triton kernels that write K/V into the paged KV cache.

## Backend Entry Point

The backend method is:

- `TritonAttentionImpl.do_kv_cache_update(...)`

For encoder and encoder-only attention, this returns immediately.

For decoder and cross-attention, it writes K/V into the paged cache.

## Normal Cache Update Wrapper

The normal wrapper is:

- `triton_reshape_and_cache_flash(...)` in `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`

It launches:

- `reshape_and_cache_kernel_flash`

Inputs are:

- `key`: `[num_tokens, num_heads, head_size]`
- `value`: `[num_tokens, num_heads, head_size]`
- `key_cache`
- `value_cache`
- `slot_mapping`: `[num_tokens]`
- `kv_cache_dtype`
- `k_scale`
- `v_scale`

## reshape_and_cache_kernel_flash Grid

The launch grid is:

- axis 0: token index
- axis 1: tile over flattened `(num_heads * head_size)`

The grid is effectively:

- `(slot_mapping.shape[0], ceil(num_heads * head_size / TILE_SIZE))`

Each program handles one token and one vector tile.

## Slot Mapping In The Kernel

The kernel loads:

- `slot_idx = slot_mapping[token_idx]`

If `slot_idx < 0`, the token is padding and the program returns.

Then:

- `block_idx = slot_idx // block_size`
- `block_offset = slot_idx % block_size`

That converts the flat slot mapping into a physical cache block and offset inside the block.

## Normal Layout Branch

For the normal layout, the target offset is:

- `block_idx * block_stride + block_offset * page_stride + cur_head * head_stride + cur_dim`

The same target index is used for key and value caches.

The flattened tile position is decomposed as:

- `cur_head = tile_pos // head_size`
- `cur_dim = tile_pos % head_size`

## Head-Major Layout Branch

If `key_cache.ndim == 5`, the wrapper treats the cache as head-major.

The value addressing is conceptually:

- `[block, head, dim, slot]`

The key addressing is conceptually:

- `[block, head, dim // x, slot, x]`

The `x` value comes from the inner cache dimension.

This branch exists for layouts where key cache is vectorized into small chunks.

## FP8 Cache Update

`FP8_KV_CACHE` describes the destination cache, while `key_load.dtype` and `value_load.dtype` describe the incoming K/V tensors. These are separate facts: an FP8 cache commonly receives FP16 or BF16 K/V produced by the model, but the kernel also accepts K/V that a caller has already materialized as FP8.

For the usual FP16/BF16 input path, the kernel converts a real K/V value into the cache's scaled FP8 representation by dividing by the configured scale before the store:

- key stores `key_load / k_scale`.
- value stores `value_load / v_scale`.

The destination pointer has an FP8 element type, so `tl.store` then casts the scaled result to that concrete FP8 format. During attention, the cached number is interpreted using the corresponding scale, conceptually reconstructing `cached_key * k_scale` or `cached_value * v_scale`.

If `key_load.dtype.is_fp8()` or `value_load.dtype.is_fp8()` is true, the input tensor itself already contains FP8 cache-domain values. This can happen when an upstream or fused quantized path produces pre-quantized K/V, or when another caller invokes this reusable cache-update operation with FP8 tensors. In that case the kernel copies the FP8 value directly because dividing by the scale again would quantize an already quantized value a second time. Merely loading FP8 model weights does not trigger this branch; the actual runtime dtype of the K/V tensor passed to the operation must be FP8.

For example, suppose the real key value is `4.0` and `k_scale` is `0.5`. An FP16 input must be transformed to the FP8 cache-domain value `4.0 / 0.5 = 8.0`, which will later reconstruct as `8.0 * 0.5 = 4.0`. If the incoming key tensor is already FP8 and contains `8.0` under that same scale convention, storing `8.0` directly is correct; dividing it again would store `16.0` and later reconstruct the incorrect value `8.0`.

In the normal Triton backend flow, K/V usually arrive from the attention projection in the model dtype, so the division-and-cast branch is the common one. The existing-FP8 branch makes the kernel safe for pre-quantized K/V producers without forcing them to dequantize and immediately requantize their tensors.

## Per-Token-Head Quant Update

The per-token-head wrapper is:

- `triton_reshape_and_cache_flash_per_token_head_quant(...)`

It launches:

- `_reshape_cache_per_token_head`

The grid is:

- `(num_tokens, num_kv_heads)`

Each program handles one `(token, head)` pair.

## _reshape_cache_per_token_head

This kernel does:

1. load one key head vector and one value head vector.
2. compute `absmax` for the key head.
3. compute `k_scale = max(abs(key_head)) / QUANT_MAX`, clamped to at least `1e-6`.
4. store `k_scale` into `k_scale_cache`.
5. quantize and store key data.
6. repeat the same process for value.

The scale cache shapes are:

- `k_scale_cache`: `[num_blocks, block_size, num_kv_heads]`
- `v_scale_cache`: `[num_blocks, block_size, num_kv_heads]`

The quantization range depends on cache dtype:

- int8 uses `127.0` and `-128.0`.
- FP8 uses the platform FP8 min/max.

## Inline Scale Cache Views

For per-token-head quantization, `TritonAttentionBackend.get_kv_cache_shape(...)` pads the head dimension so one float32 scale can be stored inline.

The logical cache shape is `(num_blocks, 2, block_size, num_kv_heads, head_size + scale_pad)`, where dimension `1` selects K or V and `scale_pad = sizeof(float32) / sizeof(cache_dtype)`. For one-byte INT8 or FP8 cache elements, `scale_pad` is `4`, so a head size of `128` produces rows of `132` bytes.

Each `(block, K-or-V, slot, KV-head)` row is laid out as `[head_size quantized data elements][one float32 scale stored in the scale_pad tail bytes]`. The K row therefore contains a quantized key vector followed by its K scale, while the corresponding V row contains a quantized value vector followed by its independently computed V scale.

This shape is defined by `TritonAttentionBackend.get_kv_cache_shape(...)` in `vllm/v1/attention/backends/triton_attn.py`.

`TritonAttentionImpl._ensure_scale_caches(...)` builds strided float32 views over the raw KV cache storage.

Those views become:

- `self._k_scale_cache`
- `self._v_scale_cache`

The cache-update kernel writes scales through these views.

The attention kernel later reads those scales through `k_scale_cache_ptr` and `v_scale_cache_ptr`.

The view construction is implemented by `TritonAttentionImpl._ensure_scale_caches(...)` in `vllm/v1/attention/backends/triton_attn.py`; it does not allocate separate scale buffers, but reinterprets the final `scale_pad` bytes of every inline cache row as one float32 value.

## Diff-KV Cache Kernel

The same file also defines:

- `reshape_and_cache_kernel_flash_diffkv`
- `triton_reshape_and_cache_flash_diffkv(...)`

This is for paths where key and value head dimensions differ and K/V are stored in a combined cache layout.

It is not the normal standard `TRITON_ATTN` path, but it is relevant to related FlashAttention diff-KV users.

## Key Files

- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`
- `vllm/v1/kv_cache_interface.py`
