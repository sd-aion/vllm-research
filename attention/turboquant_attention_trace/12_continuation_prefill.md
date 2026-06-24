# Continuation Prefill

A continuation prefill processes a multi-token query chunk for a request that already has cached context.

For one request:

```text
q_len = number of current chunk tokens
seq_len = cached context plus current chunk
cached_len = seq_len - q_len
```

The implementation chooses between two paths in `_prefill_attention(...)` from `vllm/v1/attention/backends/turboquant_attn.py`.

## Short Continuations

`_CONTINUATION_DECODE_THRESHOLD = 128`.

When `q_len <= 128`, the implementation treats every token in the current chunk as an independent decode row and calls `triton_turboquant_decode_attention(...)` once with a synthetic batch of size `q_len`.

The K/V update has already stored all current chunk tokens in the cache, so every synthetic row can read its allowed prefix from the packed cache.

## Synthetic Sequence Lengths

For current token position `p` inside the chunk, causal attention should expose:

```text
cached_len + p + 1
```

The implementation slices a cached device arange:

```text
synth_seq_lens = arange[cached_len + 1 : seq_len + 1]
```

This produces shape `[q_len]` with one valid KV length per synthetic decode row.

Example with `cached_len = 5` and `q_len = 3`:

```text
synth_seq_lens = [6, 7, 8]
```

The first new token sees five cached tokens plus itself, the second sees those six plus itself, and the third sees all eight positions.

## Expanded Block Table

The original request has one block-table row `[1, max_num_blocks]`.

The implementation creates:

```text
synth_bt = block_table[i:i+1].expand(q_len, -1)
```

This is a view that presents the same logical-to-physical page mapping for every synthetic row without copying page IDs.

Each synthetic row differs only in `seq_lens`; all rows belong to the same request and therefore share the block table.

## Why This Avoids A Long-Context Collapse

Bulk-dequantizing all cached K/V for every small continuation chunk costs work proportional to `cached_len` before attention begins.

Repeated over many chunks, that reconstruction can approach `O(total_length^2 / chunk_size)` extra data movement. The decode kernel instead dequantizes cache tiles only while computing the required attention rows.

## Large Continuations

When `q_len > 128`, the implementation calls `_continuation_prefill(...)`.

Large chunks provide enough query parallelism for dense FlashAttention or SDPA to be attractive, so the backend reconstructs the old K/V once and then runs a dense attention operation.

## Bulk Dequantization

The method rounds `cached_len` up to a multiple of the cache block size and obtains two workspace tensors shaped:

```text
[1, Hk, alloc_len, D]
```

`_tq_full_dequant_kv` reconstructs cached K and V into fp16. Only `:cached_len` is consumed, so padded positions need not be initialized.

The workspace manager allows layers that execute sequentially to reuse one allocation rather than retaining large per-layer dequant buffers. Stable workspace allocation is also compatible with CUDA graph capture.

## Inverse Rotation

MSE keys reconstructed by `_tq_full_dequant_kv` are still in rotated centroid space, scaled by their original norms.

The implementation reshapes all cached key vectors to `[-1, D]` and multiplies by the fp16 Hadamard matrix `_tq_Pi_half`:

```text
k_original_space = k_rotated @ Pi
```

FP8 keys were never rotated, so that multiplication is skipped.

## Combining Cached And Current K/V

The implementation allocates dense tensors:

```text
k_full: [seq_len, Hk, D]
v_full: [seq_len, Hk, D]
```

The first `cached_len` rows receive reconstructed cache values, and the final `q_len` rows receive the current raw `key_chunk` and `val_chunk`.

Using raw current K/V avoids immediately dequantizing bytes that were just quantized for future cache use.

## Dense Attention

With FlashAttention, Q has length `q_len` and K/V have length `seq_len`. Causal varlen attention right-aligns the shorter query sequence with the end of K/V, so query row `p` can attend through absolute position `cached_len + p`.

Without FlashAttention, the SDPA fallback constructs an explicit boolean mask:

```text
q_absolute_position[p] = cached_len + p
mask[p, j] = j <= q_absolute_position[p]
```

It then runs SDPA with `enable_gqa` when needed and returns `[q_len, Hq, D]`.
