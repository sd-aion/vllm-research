# FlashAttention Forward, Prefill, And Decode

This note traces `FlashAttentionImpl.forward(...)` and explains how prefill and decode use the same backend path with different metadata.

## Runtime Impl Class

The runtime implementation is:

- `FlashAttentionImpl` in `vllm/v1/attention/backends/flash_attn.py`

The layer constructs it through:

- `impl_cls = self.attn_backend.get_impl_cls()`
- `self.impl = impl_cls(...)`

The constructor receives local rank-specific dimensions:

- `num_heads`
- `head_size`
- `scale`
- `num_kv_heads`
- `alibi_slopes`
- `sliding_window`
- `kv_cache_dtype`
- `logits_soft_cap`
- `attn_type`
- `kv_sharing_target_layer_name`
- `sinks`

These are local dimensions after tensor parallel sharding, not necessarily global model dimensions.

## Constructor State

`FlashAttentionImpl.__init__(...)` sets:

- `self.num_heads`
- `self.head_size`
- `self.scale`
- `self.num_kv_heads`
- `self.alibi_slopes`
- `self.sliding_window`
- `self.kv_cache_dtype`
- `self.logits_soft_cap`
- `self.num_queries_per_kv`
- `self.attn_type`
- `self.vllm_flash_attn_version`
- `self.sinks`
- `self.supports_quant_query_input`
- `self.dcp_combine`

The selected FlashAttention version is logged once.

The DCP combine function is chosen based on `dcp_comm_backend`:

- `dcp_a2a_lse_reduce` for all-to-all DCP.
- `cp_lse_ag_out_rs` for the default all-gather/reduce-scatter style path.

## Forward Inputs

`forward(...)` receives:

- `query`: `[num_tokens, num_heads, head_size]`
- `key`: `[num_tokens, num_kv_heads, head_size]`
- `value`: `[num_tokens, num_kv_heads, head_size]`
- `kv_cache`: `[num_blocks, 2, block_size, num_kv_heads, head_size]`
- `attn_metadata`: `FlashAttentionMetadata`
- `output`: output buffer

For quantized output arguments, `FlashAttentionImpl` currently raises `NotImplementedError`.

If `attn_metadata is None`, the method is in a profiling path and fills output with zeros.

## Encoder Attention Path

If `attn_type` is `ENCODER_ONLY` or `ENCODER`, `forward(...)` calls:

- `_forward_encoder_attention(...)`

This path does not use paged KV cache.

It passes direct Q/K/V tensors to `flash_attn_varlen_func(...)` with:

- `cu_seqlens_q = query_start_loc`
- `cu_seqlens_k = query_start_loc`
- `max_seqlen_q = max_query_len`
- `max_seqlen_k = max_query_len`
- `causal=False`

Encoder attention is bidirectional by construction.

## Decoder And Cross-Attention Path

For decoder and cross-attention, `forward(...)` unbinds the KV cache:

- `key_cache, value_cache = kv_cache.unbind(1)`

It then canonicalizes degenerate singleton strides because FA3/FA4 TMA paths require alignment-friendly strides.

If KV cache is quantized, it views the cache tensors as the platform FP8 dtype.

The stride canonicalization sentence means vLLM fixes only the tensor layout metadata of `key_cache` and `value_cache` before calling the FA3/FA4 kernel.

A tensor has data, shape, and strides; the data is the actual GPU memory, while the strides describe how to walk that memory along each dimension.

If a dimension has size `1`, PyTorch can tolerate odd strides because moving along that dimension never reaches a second element.

FA3/FA4 on H100 uses TMA, and TMA is stricter about stride alignment, so a stride that is harmless to PyTorch can still be invalid or inefficient for the kernel.

`canonicalize_singleton_dim_strides(...)` makes those singleton-dimension strides look normal/aligned for the kernel without moving data and without changing any K/V values.

In simple terms, `canonicalize_singleton_dim_strides(t)` walks the tensor dimensions from right to left and repairs only dimensions whose size is `1`.

For normal contiguous layout, the stride of a dimension should equal the product of the sizes of all dimensions to its right.

The variable `prev_stride` tracks exactly that expected stride while walking right-to-left.

If the current dimension has size `1` and its stride is not the expected `prev_stride`, the function replaces that stride with `prev_stride`.

It only changes size-1 dimensions because changing their stride is safe: there is only one element along that dimension, so no actual indexing behavior changes.

If no stride needs fixing, the original tensor is returned unchanged.

If a stride is fixed, `t.as_strided(t.shape, strides)` returns a new tensor view over the same memory with the corrected stride metadata.

That is why the function is zero-copy: it does not allocate a new KV cache and does not rewrite the cache contents.

Example: suppose a cache view has shape `(8, 16, 1, 128)`.

In a normal contiguous layout, the strides would be `(2048, 128, 128, 1)` because moving along the `1`-sized third dimension would normally skip `128` elements.

PyTorch might still give that size-1 dimension a stride like `1`, producing strides such as `(2048, 128, 1, 1)`, because there is only one element on that dimension so PyTorch indexing still works.

For FA3/FA4 TMA, that stride `1` can be a problem because it is not aligned enough.

The function changes the stride metadata back to `(2048, 128, 128, 1)`.

This is safe because the third dimension has size `1`, so no code can index a second head along that dimension and observe a different memory location.

The quantized-cache sentence means that if the KV cache is stored in FP8, vLLM changes the tensor view so FlashAttention interprets the same cache bytes as the platform FP8 dtype.

That `.view(current_platform.fp8_dtype())` is not dequantization; it does not convert the cache into fp16 or bf16.

It only tells FlashAttention to load the K/V cache as FP8 data, and then the kernel uses the corresponding scale/descale tensors such as `k_descale` and `v_descale` during attention.

Then the implementation chooses between:

- normal paged attention
- DCP attention
- cascade attention

## Normal Paged Attention

The normal path calls:

- `flash_attn_varlen_func(...)`

Important arguments are:

- `q=query[:num_actual_tokens]`
- `k=key_cache`
- `v=value_cache`
- `out=output[:num_actual_tokens]`
- `cu_seqlens_q=query_start_loc`
- `max_seqlen_q=max_query_len`
- `seqused_k=seq_lens`
- `max_seqlen_k=max_seq_len`
- `softmax_scale=self.scale`
- `causal=attn_metadata.causal`
- `alibi_slopes=self.alibi_slopes`
- `window_size=self.sliding_window`
- `block_table=block_table`
- `softcap=self.logits_soft_cap`
- `scheduler_metadata=scheduler_metadata`
- `fa_version=self.vllm_flash_attn_version`
- `q_descale`, `k_descale`, `v_descale`
- `num_splits=attn_metadata.max_num_splits`
- `s_aux=self.sinks`

The key point is that the same call handles prefill and decode because `query_start_loc`, `max_query_len`, `seq_lens`, and `block_table` describe the current batch shape.

## Prefill

Prefill processes prompt tokens, usually many tokens per request.

In prefill:

- `max_query_len` is often greater than 1.
- `query_start_loc` packs variable-length prompt chunks.
- `seq_lens` usually equals the current prompt length for each request.
- `causal=True` for decoder models, so each prompt token attends only to previous prompt tokens and itself.
- New K/V tokens are first written into the paged KV cache through `do_kv_cache_update(...)`.
- The FlashAttention call reads from the paged KV cache through `block_table` and `seqused_k`.

This means vLLM's FlashAttention prefill still uses the paged KV-cache interface in the decoder path.

A prompt token does not attend to all other prompt tokens in causal decoder prefill.

Instead, token `i` attends to tokens `<= i`; non-causal prefill is the case where tokens can attend bidirectionally.

## Decode

Decode processes new generated tokens against cached KV.

In ordinary decode:

- `max_query_len` is usually 1.
- `query_start_loc` has one query token per active request.
- `seq_lens` is large because it includes the cached context plus the new token.
- `block_table` points to existing physical cache blocks for each request.
- `seqused_k=seq_lens` tells the kernel how much of each request's cache is valid.

FlashAttention uses the same `flash_attn_varlen_func(...)` entry point.

The library may internally choose a FlashDecoding-style execution strategy for some shapes, but vLLM's backend-level public call remains `flash_attn_varlen_func(...)`.

## Mixed Prefill And Decode

Continuous batching can produce mixed batches where some requests are prefilling and others are decoding.

The metadata supports that because each request has its own query range and sequence length.

The backend does not expose separate public `prefill_forward(...)` and `decode_forward(...)` methods.

Runtime specialization happens through metadata and through the FlashAttention library's kernel dispatch.

## Quantization Scales

The forward path creates descale tensors with shape:

- `(num_sequences, num_kv_heads)`

It passes:

- `q_descale` when quantized query input is supported.
- `k_descale`
- `v_descale`

For FA2, scheduler metadata and descale arguments are not supported in the wrapper path.

For FA3 and newer paths, descale arguments can be passed into the FlashAttention call.

## Key Files

- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/model_executor/layers/attention/attention.py`
- `vllm/vllm_flash_attn/flash_attn_interface.py`
