# TurboQuant Decode Stage-One Kernel

`_tq_decode_stage1(...)` in `vllm/v1/attention/ops/triton_turboquant_decode.py` reads packed paged K/V, computes one split's attention, and stores a partial output plus LSE.

## Pointer Arguments

- `Q_rot_ptr: [B, Hq, D]` is rotated float32 Q for MSE keys or contiguous raw Q for FP8 keys.
- `KV_cache_ptr: [num_blocks, block_size, Hk, padded_slot]` is packed uint8 K/V.
- `Block_table_ptr: [B, max_num_blocks]` maps request pages to physical blocks.
- `Seq_lens_ptr: [B]` gives valid KV lengths.
- `Centroids_ptr: [2^MSE_BITS]` provides MSE key reconstruction values.
- `Mid_o_ptr: [B, Hq, NUM_KV_SPLITS, D + 1]` receives partial output and LSE.

The runtime stride arguments are:

- `stride_qb`: element distance between Q batch rows.
- `stride_qh`: element distance between query heads in one batch row.
- `stride_cache_block`: byte distance between physical cache blocks.
- `stride_cache_pos`: byte distance between token positions in one cache block.
- `stride_cache_head`: byte distance between KV-head slots at one token position.
- `stride_bt_b`: element distance between block-table request rows.
- `stride_mid_b`: element distance between intermediate batch rows.
- `stride_mid_h`: element distance between intermediate query heads.
- `stride_mid_s`: element distance between intermediate KV splits.

## Compile-Time Arguments

- `NUM_KV_HEADS` is local `Hk`.
- `HEAD_DIM` is `D`.
- `BLOCK_SIZE` is the paged-cache block size.
- `NUM_KV_SPLITS` is the fixed split count.
- `KV_GROUP_SIZE = Hq // Hk` maps GQA query heads to KV heads.
- `MSE_BITS`, `MSE_BYTES`, and `KPS` describe packed keys.
- `VQB` and `VAL_DATA_BYTES` describe packed values.
- `ATTN_SCALE` is the QK score scale.
- `BLOCK_D` is the power-of-two vector width.
- `BLOCK_KV` is the number of logical KV tokens processed per loop iteration and is currently four.
- `KEY_FP8`, `NORM_CORRECTION`, and `FP8_E4B15` specialize key reconstruction.

## Program IDs

```text
bid = program_id(0)
hid = program_id(1)
sid = program_id(2)
```

One program handles one request row, one query head, and one KV split.

The corresponding KV head is:

```text
kv_head = hid // KV_GROUP_SIZE
```

For `Hq = 16` and `Hk = 4`, `KV_GROUP_SIZE = 4`; query heads 0-3 share KV head 0, query heads 4-7 share KV head 1, and so on.

## Split Range

For sequence length `L`:

```text
split_len = ceil(L / NUM_KV_SPLITS)
split_start = split_len * sid
split_end = min(split_start + split_len, L)
```

The program returns if `split_start >= split_end`.

The split count remains fixed, but the amount of useful work adapts through these boundaries.

## Query Load

The program loads one `[BLOCK_D]` query vector as float32 and masks dimensions `d >= HEAD_DIM`.

For MSE keys, it also precomputes each dimension's packed bit offset, source byte, bit shift, and bit mask.

For 3-bit values, it precomputes the corresponding value byte and shift vectors.

## Paged Cache Traversal

The kernel iterates from `split_start` to `split_end` in `BLOCK_KV`-token tiles.

For every logical position `kv_off`:

```text
page_idx = kv_off // BLOCK_SIZE
page_off = kv_off % BLOCK_SIZE
block_num = block_table[bid, page_idx]
slot_base = block_num * stride_cache_block
          + page_off * stride_cache_pos
          + kv_head * stride_cache_head
```

The final tile masks positions at or beyond `split_end`.

## FP8 Key Scores

For FP8 keys, each slot's first `D` bytes are loaded and bitcast to the platform's FP8 type, then converted to float32.

Scores are:

```text
score[t] = sum_d(q[d] * k_fp8[t, d]) * ATTN_SCALE
```

No key scale or norm metadata is loaded.

## MSE Key Scores

For each key coordinate, the kernel loads two adjacent bytes, combines them into a 16-bit window, shifts to the coordinate's starting bit, and masks three or four bits.

The unpacked index gathers a float32 centroid value.

When norm correction is enabled, each tile row computes:

```text
centroid_inv_norm[t] = 1 / sqrt(sum_d(c[t,d]^2) + 1e-16)
c[t,d] = c[t,d] * centroid_inv_norm[t]
```

The kernel loads the key's fp16 norm from bytes at offset `MSE_BYTES` and computes:

```text
score[t] = key_norm[t] * sum_d(q_rot[d] * c[t,d]) * ATTN_SCALE
```

This evaluates the dot product in rotated space without materializing a full key vector in original space.

## Value Reconstruction

Value bytes begin at `slot_base + KPS`.

For 4-bit values, the kernel selects the low or high nibble from byte `d // 2`.

For 3-bit values, it combines two adjacent bytes and extracts three bits starting at `3*d`.

It loads fp16 scale and minimum from the four bytes following `VAL_DATA_BYTES`, then reconstructs:

```text
value[t,d] = value_index[t,d] * value_scale[t] + value_minimum[t]
```

## Online Softmax

The program maintains one scalar row maximum `m_prev`, one exponential sum `l_prev`, and one float32 value accumulator `[BLOCK_D]`.

For each tile:

```text
m_new = max(m_prev, max(scores))
old_rescale = exp(m_prev - m_new)
p[t] = exp(score[t] - m_new)
acc = acc * old_rescale + sum_t(p[t] * value[t,:])
l_prev = l_prev * old_rescale + sum_t(p[t])
m_prev = m_new
```

This is numerically stable and never materializes the complete score vector.

## Partial Result

At the end of a valid split:

```text
partial_output = acc / l_prev
partial_lse = m_prev + log(l_prev)
```

The kernel stores the output in `Mid_o[..., :D]` and LSE in `Mid_o[..., D]`.

Stage two needs both values: averaging partial outputs directly would be wrong because splits can carry very different total softmax mass.
