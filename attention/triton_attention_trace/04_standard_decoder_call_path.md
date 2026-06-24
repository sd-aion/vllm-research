# Standard Triton Decoder Call Path

This note explains the normal decoder path for standard `TRITON_ATTN`.

## The Core Split

For decoder attention, `TRITON_ATTN` has two main stages:

1. update KV cache
2. compute attention from paged KV cache

The KV update is handled by:

- `TritonAttentionImpl.do_kv_cache_update(...)`

The attention compute is handled by:

- `TritonAttentionImpl.forward(...)`
- `unified_attention(...)`
- `kernel_unified_attention(...)`

## Decoder Prefill

Decoder prefill processes prompt tokens.

For each attention layer, prefill must do both:

- write the prompt K/V into the layer's KV cache.
- compute attention outputs for prompt Q tokens so the hidden states can continue to the next layer.

Writing K/V to cache is not enough.

For prompt token `i`, causal decoder attention computes attention over tokens `0..i`.

In standard `TRITON_ATTN`, decoder prefill does:

1. model creates Q/K/V for prompt tokens.
2. `do_kv_cache_update(...)` writes K/V into paged KV cache using `slot_mapping`.
3. `forward(...)` calls `unified_attention(...)`.
4. `kernel_unified_attention(...)` reads K/V from paged cache through `block_table` and `seq_lens`.
5. output hidden states continue through the transformer layer stack.

## Decoder Decode

Decoder decode processes newly generated tokens.

It uses the same structural path:

1. model creates Q/K/V for the new token or tokens.
2. `do_kv_cache_update(...)` writes new K/V into paged KV cache.
3. `forward(...)` calls `unified_attention(...)`.
4. `kernel_unified_attention(...)` reads the cached context plus the new token's K/V.

The metadata differs:

- decode usually has `max_query_len == 1`.
- prefill usually has `max_query_len > 1`.
- `seq_lens` is much larger than query length during decode because it includes historical cached context.

## Mixed Batches

Continuous batching can mix prefill and decode requests.

`TRITON_ATTN` uses one metadata shape for both.

The key fields are:

- `query_start_loc`: where each request's current query tokens start.
- `seq_lens`: how many K/V tokens each request can attend over.
- `block_table`: where each request's logical KV blocks live physically.
- `max_query_len`: whether the batch includes multi-token query chunks.

The unified kernel maps flattened query blocks back to request-local positions through `resolve_seq_and_query_len(...)`.

## Why context_attention_fwd Is Not The Normal Decoder Prefill Path

`context_attention_fwd(...)` from `triton_prefill_attention.py` operates on direct packed Q/K/V tensors.

`TritonAttentionImpl` uses it in `_forward_encoder_attention(...)`.

The standard decoder prefill path uses paged KV cache and `unified_attention(...)`.

This means the filename `triton_prefill_attention.py` should not be read as "the standard decoder prefill kernel for `TRITON_ATTN`".

## Forward Arguments To unified_attention

`TritonAttentionImpl.forward(...)` passes:

- `q=query[:num_actual_tokens]`
- `k=key_cache`
- `v=value_cache`
- `out=output[:num_actual_tokens]`
- `cu_seqlens_q=query_start_loc`
- `max_seqlen_q=max_query_len`
- `seqused_k=seq_lens`
- `max_seqlen_k=max_seq_len`
- `softmax_scale=self.scale`
- `causal=True`
- `window_size=self.sliding_window`
- `block_table=block_table`
- `softcap=self.logits_soft_cap`
- `q_descale`, `k_descale`, `v_descale`
- segment scratch tensors
- `sinks=self.sinks`
- `output_scale=output_scale`
- `mm_prefix_range=mm_prefix_range_tensor`
- `kv_quant_mode=self._kv_quant_mode`
- per-token-head scale caches when active
- `chunk_lookback=self.chunk_lookback`
- `use_td=self.use_td`

## Key Files

- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/ops/triton_unified_attention.py`
- `vllm/v1/attention/ops/triton_attention_helpers.py`
- `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`

