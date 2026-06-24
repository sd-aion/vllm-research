# TurboQuant Prefill Paths

Prefill logic is implemented by `_prefill_attention(...)` in `vllm/v1/attention/backends/turboquant_attn.py`.

## Cache Update Happens First

Before prefill attention runs, `do_kv_cache_update(...)` has already compressed all current K/V and written them to cache.

First-chunk prefill does not read those compressed bytes. It uses raw K/V to avoid unnecessary quantization error and to use high-throughput dense attention kernels.

The early cache write is still required because the stored K/V must remain available for later decode or continuation-prefill steps.

## Full-Batch First-Chunk Fast Path

The implementation tests:

```text
_HAS_FLASH_ATTN and max_query_len == max_seq_len
```

For a pure prefill batch, equality means no request has earlier cached context: each request's total sequence consists entirely of its current query tokens.

The implementation calls `_flash_attn_varlen(...)` once with:

- `q`, `k`, and `v` as packed tensors for all requests.
- `cu_seqlens_q = query_start_loc`.
- `cu_seqlens_k = query_start_loc` because Q and K lengths are equal.
- `max_seqlen_q = max_query_len`.
- `max_seqlen_k = max_query_len`.
- `causal = True` and `softmax_scale = self.scale`.

`_flash_attn_varlen(...)` passes an explicit FA version when `get_flash_attn_version(...)` returned one, and omits the keyword on platforms where that argument is inappropriate.

## Why Prefill Tokens Do Not All Attend Bidirectionally

The dense prefill kernel computes many query rows together, but causal masking still applies separately to every row.

For local prompt positions `i` and `j`, the score is valid only when `j <= i`:

```text
score(i, j) = (q_i dot k_j) * scale       if j <= i
score(i, j) = -infinity                   if j > i
```

Computing the full triangular prompt in parallel does not allow an earlier token to see a later token.

## Per-Request Path

If the full-batch fast-path condition fails, `_prefill_attention(...)` iterates over requests using `query_start_loc` boundaries.

The loop prefers `query_start_loc_cpu` and `seq_lens_cpu` to avoid calling `.tolist()` on GPU tensors, which would synchronize the host with the device.

For each request:

```text
q_start = query_start_loc[i]
q_end = query_start_loc[i + 1]
q_len = q_end - q_start
seq_len = seq_lens[i]
```

`q_len == seq_len` identifies a first chunk for that request.

## Per-Request FlashAttention

When FlashAttention is available, the implementation reuses a two-element int32 device tensor `_cu_2 = [0, q_len]` and calls varlen attention for the single request.

Reusing the tensor avoids allocating and copying a new cumulative-length tensor on every request iteration.

## PyTorch SDPA Fallback

Without FlashAttention, Q/K/V are transposed to head-major tensors and passed to `torch.nn.functional.scaled_dot_product_attention(...)` with:

- `is_causal=True`.
- `scale=self.scale`.
- `enable_gqa=(num_kv_heads < num_query_heads)`.

The output is transposed back to `[q_len, Hq, D]`.

## MQA And GQA

Raw prefill delegates MQA/GQA expansion to FlashAttention or SDPA.

The model can have fewer KV heads than query heads as long as `Hq` is divisible by `Hk`. Query heads in the same group attend to the same K/V head but maintain separate Q vectors and output vectors.

## Mixed Batch Nuance

In a mixed batch, long decode requests may have a much larger `seq_len` than new prefill requests.

`forward(...)` computes a prefill-only maximum sequence length from the CPU metadata suffix before calling `_prefill_attention(...)`. Otherwise a decode request could make `max_seq_len` exceed `max_query_len`, incorrectly hiding that every prefill request is a first chunk and disabling the full-batch FlashAttention path.
