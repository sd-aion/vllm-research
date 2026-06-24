# FlashAttention KV Cache And Slot Mapping

This note explains the KV-cache contract for `FLASH_ATTN` and how it connects to slot mapping, block tables, and cache updates.

## KV Cache Shape
 
`FlashAttentionBackend.get_kv_cache_shape(...)` in `vllm/v1/attention/backends/flash_attn.py` defines the logical tensor shape that the FlashAttention backend expects for one layer's paged KV cache.

The method returns:

- `(num_blocks, 2, block_size, num_kv_heads, head_size)`

The dimensions mean:

- `num_blocks`: physical KV-cache blocks allocated for the layer.
- `2`: key and value tensors.
- `block_size`: token slots per physical block.
- `num_kv_heads`: local KV heads on this rank.
- `head_size`: per-head dimension.

The `2` dimension is important because `FlashAttentionImpl.do_kv_cache_update(...)` later does `key_cache, value_cache = kv_cache.unbind(1)`, so dimension 1 is treated as the K/V selector.

This shape is logical, meaning it describes how vLLM and the backend index the cache semantically: choose a block, choose key or value, choose a token slot inside the block, choose a KV head, then choose a channel inside the head.

`get_kv_cache_shape(...)` rejects block sizes that are not multiples of 16 because FlashAttention's paged KV path requires page/block granularity compatible with 16-token tiles.

The `cache_dtype_str` argument is present because the backend contract allows dtype-specific shapes, but FlashAttention's standard path does not change this shape based on the cache dtype.

One practical consequence is that the block table and slot mapping both assume this logical shape: `block_table` points to entries in the `num_blocks` dimension, while `slot_mapping` can be decomposed into `block = slot // block_size` and `offset = slot % block_size` for the `block_size` dimension.

## KV Cache Stride Order

`FlashAttentionBackend.get_kv_cache_stride_order(...)` in `vllm/v1/attention/backends/flash_attn.py` defines how the logical dimensions from `get_kv_cache_shape(...)` should be physically laid out in memory.

This separation exists because the kernel-facing logical shape and the allocation-friendly memory order are not always the same thing.

The layout comes from:

- `get_kv_cache_layout()` in `vllm/v1/attention/backends/utils.py`

`get_kv_cache_layout()` reads `VLLM_KV_CACHE_LAYOUT` if the user set it, otherwise it falls back to the KV connector's default layout.

For the normal 5D FlashAttention logical shape `(num_blocks, 2, block_size, num_kv_heads, head_size)`, `NHD` returns:

- `(0, 1, 2, 3, 4)`

That means the physical allocation order matches the logical order.

For the same 5D shape, `HND` returns:

- `(0, 1, 3, 2, 4)`

That means vLLM allocates/views the cache with the KV-head dimension before the token-in-block dimension, so the physical order becomes `(num_blocks, 2, num_kv_heads, block_size, head_size)`.

When `include_num_layers_dimension=True`, the returned stride order includes the layer dimension as well.

For `NHD` with a layer dimension prepended, the code comments describe the physical order as `(num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)` and returns `(1, 0, 2, 3, 4, 5)`.

For `HND` with a layer dimension prepended, the code comments describe the physical order as `(num_blocks, num_kv_heads, num_layers, 2, block_size, head_size)` and returns `(1, 4, 0, 2, 3, 5)`.

The worker allocation path uses this stride order by first computing the logical KV-cache shape, then permuting the shape with `kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)`, and then building an inverse permutation so the attention layer can still see the original logical shape.

That means `get_kv_cache_stride_order(...)` is not just documentation; it directly controls the tensor view/stride that the worker constructs before the backend's cache update and attention kernels interpret `kv_cache`.

If this method returns the wrong order for a backend, the cache can still have the right total number of elements but the kernel will interpret the wrong dimension as token, head, or K/V, which is a layout bug rather than a math bug.

## Block Size Contract

The kernel block-size support is usually:

- `[MultipleOf(16)]`

The practical meaning is:

- the framework block size must be divisible by 16.
- the FlashAttention paged kernel can operate with that page granularity.
- vLLM can still use larger framework block sizes when they are compatible with the kernel's minimum granularity.

The hybrid Mamba float32-cache exception returns `[16, 32, 64]` for safer block-size choices.

## forward_includes_kv_cache_update

`FlashAttentionBackend.forward_includes_kv_cache_update = False`.

That means the attention forward call does not write new K/V into the KV cache itself.

Instead, vLLM performs the cache update as a separate step through the backend impl method:

- `FlashAttentionImpl.do_kv_cache_update(...)`

This split matters at the custom-op boundary because vLLM can update KV cache before calling the actual attention kernel.

## do_kv_cache_update

`do_kv_cache_update(...)` receives:

- `key`
- `value`
- `kv_cache`
- `slot_mapping`
- scale tensors from the layer

For decoder and cross-attention, it unbinds the KV cache into:

- `key_cache`
- `value_cache`

Then it calls:

- `reshape_and_cache_flash(...)`

For encoder and encoder-only attention, it returns immediately because encoder attention uses direct Q/K/V tensors and does not use paged KV cache.

## reshape_and_cache_flash

`reshape_and_cache_flash(...)` is imported through `vllm/v1/attention/backends/fa_utils.py`.

On CUDA it comes from:

- `vllm._custom_ops.reshape_and_cache_flash`

That Python wrapper calls:

- `torch.ops._C_cache_ops.reshape_and_cache_flash`

The operation reshapes the packed key/value tensors and scatters them into the paged KV cache according to `slot_mapping`.

It also receives:

- `kv_cache_dtype`
- key scale
- value scale

That is how quantized KV-cache storage is handled for FlashAttention cache writes.

## Slot Mapping

`slot_mapping` maps each actual token in the current batch to a physical KV-cache slot.

For a token index `i`, the value `slot_mapping[i]` can be interpreted as:

- `block = slot_mapping[i] // block_size`
- `offset = slot_mapping[i] % block_size`

The block index identifies the physical KV-cache block.

The offset identifies the token slot inside that block.

`reshape_and_cache_flash(...)` uses this mapping to scatter each token's K/V into the correct physical location.

## Block Table

The block table maps each request's logical sequence blocks to physical KV-cache blocks.

The attention kernel uses the block table to read cached K/V for each request.

The distinction is:

- `slot_mapping` is for writing the current batch's new K/V tokens into physical cache slots.
- `block_table` is for reading a request's historical cached K/V blocks during attention.

FlashAttention passes `block_table` to `flash_attn_varlen_func(...)` as the paged cache table.

## How Final Partial Blocks Work

The block table does not need to store "how many tokens are in the last block" directly.

The valid length comes from `seq_lens` / `seqused_k`.

`seq_lens` is the per-request total sequence length that vLLM puts into `CommonAttentionMetadata` and then into `FlashAttentionMetadata`.

For normal FlashAttention paged attention, `FlashAttentionImpl.forward(...)` passes `attn_metadata.seq_lens` to `flash_attn_varlen_func(...)` as the argument named `seqused_k`.

`seqused_k` means "sequence length actually used for K/V" for each request.

So the block table tells the kernel where the physical KV blocks are, while `seqused_k` tells the kernel how many token positions inside those blocks are valid.

Example: if `block_size = 16` and a request has `seq_lens[req] = 35`, the block table may point to three physical blocks for that request, but only token positions `0..34` are valid; the third block contributes only offsets `0..2`, and offsets `3..15` must be ignored.

This is why the final block does not need a separate "used slots" field: `seq_lens % block_size` tells the kernel how much of the final block is real, and `seq_lens` itself tells the kernel the total K/V range.

This is why `FlashAttentionMetadata` carries both `block_table` and `seq_lens`.

## DCP And CP-Aware Slot Mapping

When context parallelism is active, the block table path becomes CP-aware.

`BlockTable.compute_slot_mapping(...)` in `vllm/v1/worker/block_table.py` computes:

- `total_cp_world_size = pcp_world_size * dcp_world_size`
- `total_cp_rank = pcp_rank * dcp_world_size + dcp_rank`

Then it passes those values into the Triton slot-mapping kernel.

For FlashAttention DCP, this matters because each DCP rank owns only part of the cached context, and the slot mapping must point to the rank-local physical cache layout.

## Key Files

- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/fa_utils.py`
- `vllm/_custom_ops.py`
- `vllm/v1/worker/block_table.py`
- `vllm/v1/attention/backends/utils.py`
