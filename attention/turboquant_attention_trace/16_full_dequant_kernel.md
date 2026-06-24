# Full Dequantization Kernel

`_tq_full_dequant_kv(...)` in `vllm/v1/attention/ops/triton_turboquant_decode.py` reconstructs packed cache rows into dense fp16 K/V for large continuation prefill.

## Inputs

- `KV_cache_ptr` points to packed uint8 pages.
- `Block_table_ptr: [B, max_num_blocks]` maps logical pages to physical cache blocks.
- `Centroids_ptr` supplies MSE reconstruction values.
- `K_out_ptr: [B, Hk, max_seq, D]` receives fp16 keys.
- `V_out_ptr: [B, Hk, max_seq, D]` receives fp16 values.
- `stride_ko_b`, `stride_ko_h`, and `stride_ko_s` locate key-output batch, head, and sequence rows; the final dimension is contiguous.
- `stride_vo_b`, `stride_vo_h`, and `stride_vo_s` locate value-output batch, head, and sequence rows; the final dimension is contiguous.
- `stride_cache_block`, `stride_cache_pos`, and `stride_cache_head` locate packed cache blocks, token positions, and KV-head slots in bytes.
- `stride_bt_b` is the element distance between block-table request rows.
- Layout constants describe head dimension, page size, head count, key/value packing, norm correction, and FP8 format.

## Program Mapping

The first program axis is logical sequence position `pos`.

The second axis flattens batch and KV head:

```text
bid = bh // NUM_KV_HEADS
hid = bh % NUM_KV_HEADS
```

The current backend calls this kernel for one request with grid `(alloc_len, Hk)`.

## Paged Addressing

For each logical position:

```text
page_idx = pos // BLOCK_SIZE
page_off = pos % BLOCK_SIZE
block_num = block_table[bid, page_idx]
slot_base = block_num * stride_cache_block
          + page_off * stride_cache_pos
          + hid * stride_cache_head
```

`alloc_len` is rounded up to a page boundary. The caller consumes only positions below `cached_len`; padded reconstructed rows are ignored.

## FP8 Key Reconstruction

The kernel loads one key byte per valid dimension, bitcasts to E4B15 or E4NV, converts to float32, and stores fp16 in K output.

These keys are already in the model's original key coordinate space.

## MSE Key Reconstruction

The kernel extracts each 3-bit or 4-bit centroid index from adjacent cache bytes, gathers centroid values, and optionally renormalizes the centroid vector to unit L2 norm.

It loads the fp16 original key norm at offset `MSE_BYTES` and computes:

```text
k_rotated[d] = key_norm * centroid[index[d]]
```

The output remains in rotated Hadamard space. `_continuation_prefill(...)` later applies `k_rotated @ Pi` in one fp16 matrix multiplication to return all cached keys to original space.

## Value Reconstruction

The kernel starts value addressing at `slot_base + KPS`.

It unpacks 4-bit nibbles or arbitrary 3-bit fields, loads fp16 scale and minimum, and computes:

```text
v[d] = index[d] * scale + minimum
```

The result is stored as fp16 in V output.

## Why The Kernel Is Separate From Decode

Normal decode reconstructs only a small K/V tile and immediately consumes it for score and value accumulation. Writing a complete dense cache would add unnecessary global-memory traffic.

Large continuation prefill is different: many query rows reuse the same historical K/V. Materializing the historical cache once allows FlashAttention or SDPA to reuse it efficiently across all current query rows.

## Workspace And Lifetime

The caller obtains K/V output buffers from `WorkspaceManager.get_simultaneous(...)` rather than creating persistent buffers on every attention layer.

The kernel overwrites every position that the caller later reads, so the workspace is not zeroed before launch.
