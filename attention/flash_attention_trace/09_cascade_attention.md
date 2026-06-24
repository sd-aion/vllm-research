# FlashAttention Cascade Attention

This note explains the cascade attention path inside the `FLASH_ATTN` backend.

## What Cascade Attention Is

Cascade attention is an optimization for batches where many requests share a long common prefix.

Instead of each request independently reading and attending over the same shared prefix, vLLM can split attention into:

- attention over the common prefix
- attention over each request's suffix

Then it merges the two partial attention states using softmax LSE.

The goal is to reduce repeated memory bandwidth and work for shared-prefix batches.

## Entry Points

The relevant functions are in `vllm/v1/attention/backends/flash_attn.py`:

- `use_cascade_attention(...)`
- `cascade_attention(...)`

The metadata builder sets `use_cascade = common_prefix_len > 0`.

Runtime forward chooses cascade only when `attn_metadata.use_cascade` is true.

## Heuristic

`use_cascade_attention(...)` first checks support constraints:

- common prefix must be at least 256 tokens.
- ALiBi is not supported.
- sliding window is not supported.
- local attention is not supported.
- number of requests must be at least 8.
- DCP must not be active.

Then it uses a rough performance model to compare cascade attention against normal FlashDecoding-style behavior.

The heuristic considers:

- number of query heads
- number of KV heads
- query lengths
- number of common-prefix tiles
- number of SMs

If the model estimates cascade is faster, it enables cascade.

## Metadata For Cascade

`FlashAttentionMetadataBuilder.build(...)` creates extra metadata:

- `cu_prefix_query_lens = [0, num_actual_tokens]`
- `prefix_kv_lens = [common_prefix_len]`
- `suffix_kv_lens = seq_lens[:num_reqs] - common_prefix_len`
- `prefix_scheduler_metadata`
- `scheduler_metadata` for suffix attention

The common prefix is treated as one shared KV region.

The suffix remains per request.

## Prefix Attention Call

`cascade_attention(...)` first processes the shared prefix.

It calls `flash_attn_varlen_func(...)` with:

- `q=query`
- `k=key_cache`
- `v=value_cache`
- `cu_seqlens_q=cu_prefix_query_lens`
- `seqused_k=prefix_kv_lens`
- `max_seqlen_q=num_tokens`
- `max_seqlen_k=common_prefix_len`
- `causal=False`
- `block_table=block_table[:1]`
- `return_softmax_lse=True`
- `scheduler_metadata=prefix_scheduler_metadata`
- `s_aux=s_aux`

The prefix call is non-causal because it is only computing contribution from an already-computed shared prefix to the current query tokens.

It returns:

- `prefix_output`
- `prefix_lse`

## Suffix Attention Call

Then `cascade_attention(...)` processes request-specific suffix KV.

It calls `flash_attn_varlen_func(...)` with:

- `q=query`
- `k=key_cache`
- `v=value_cache`
- `cu_seqlens_q=cu_query_lens`
- `seqused_k=suffix_kv_lens`
- `max_seqlen_q=max_query_len`
- `max_seqlen_k=max_kv_len - common_prefix_len`
- `causal=True`
- `block_table=block_table[:, num_common_kv_blocks:]`
- `return_softmax_lse=True`
- `scheduler_metadata=suffix_scheduler_metadata`

It returns:

- `suffix_output`
- `suffix_lse`

## LSE Merge

The two partial outputs cannot be added directly.

Each partial output was normalized by its own softmax denominator.

So vLLM calls:

- `merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)`

This performs an exact LSE-weighted merge equivalent to computing attention over prefix and suffix together.

This same conceptual merge appears in the DCP path too, where partial attention states come from different context shards.

## Why merge_attn_states Is Needed

`prefix_output` and `suffix_output` are not raw score sums.

Each one is already the result of a separate softmax attention call.

That means `prefix_output` was normalized by the prefix softmax denominator, and `suffix_output` was normalized by the suffix softmax denominator.

If vLLM simply added the two tensors, the result would overweight both sides because each side would act as if its own local denominator were the full denominator.

The LSE tensors solve this.

`prefix_lse` and `suffix_lse` are the log-sum-exp values for the prefix and suffix score ranges.

The merge kernel reconstructs the relative softmax weights from those LSEs, computes the combined denominator, and produces the same output as if FlashAttention had attended over prefix and suffix together in one attention problem.

Conceptually, for each token and head:

- `p_scale = exp(prefix_lse) / (exp(prefix_lse) + exp(suffix_lse))`
- `s_scale = exp(suffix_lse) / (exp(prefix_lse) + exp(suffix_lse))`
- `output = prefix_output * p_scale + suffix_output * s_scale`

The real kernel uses a numerically stable version by subtracting `max(prefix_lse, suffix_lse)` before exponentiating.

## merge_attn_states Arguments

`merge_attn_states(...)` is defined in `vllm/v1/attention/ops/merge_attn_states.py`.

Its arguments are:

- `output`: destination tensor for the final merged attention output, shaped `[num_tokens, num_heads, head_size]`.
- `prefix_output`: partial attention output from the prefix-side attention call, shaped `[num_tokens, num_heads, head_size]`.
- `prefix_lse`: per-head/per-token log-sum-exp values from the prefix-side attention call, shaped `[num_heads, num_tokens]`.
- `suffix_output`: partial attention output from the suffix-side attention call, shaped `[num_tokens, num_heads, head_size]`.
- `suffix_lse`: per-head/per-token log-sum-exp values from the suffix-side attention call, shaped `[num_heads, num_tokens]`.
- `output_lse`: optional destination for the merged log-sum-exp values, shaped `[num_heads, num_tokens]`; this is useful when another later stage still needs the combined LSE.
- `prefill_tokens_with_context`: optional count of tokens that should actually merge prefix and suffix states; tokens at indices greater than or equal to this value have no prefix context and are copied directly from `suffix_output`.
- `output_scale`: optional scalar scale used when writing FP8 output; if the output tensor is FP8, the merge path scales and clamps the merged result before storing it.

In FlashAttention cascade, the call is currently `merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)`, so `output_lse`, `prefill_tokens_with_context`, and `output_scale` use their defaults.

## Current Limitations

Cascade attention currently rejects:

- ALiBi
- sliding window
- local attention
- short common prefixes
- small request counts
- DCP

These restrictions are implementation and heuristic constraints, not mathematical limitations of attention itself.

## Why This Matters For Plugin Work

If a new backend wants to support cascade-style optimization, it needs more than a normal paged attention kernel.

It needs:

- a way to compute prefix attention and return LSE.
- a way to compute suffix attention and return LSE.
- a merge path compatible with vLLM's `merge_attn_states(...)`.
- metadata builder support for common-prefix split metadata.
- restrictions or validation for unsupported combinations.

## Key Files

- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/ops/merge_attn_states.py`
- `vllm/vllm_flash_attn/flash_attn_interface.py`
