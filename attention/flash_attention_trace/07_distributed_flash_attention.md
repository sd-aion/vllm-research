# Distributed FlashAttention

This note explains how `FLASH_ATTN` behaves under TP, DCP, CP interleave, and PCP-related validation.

## Tensor Parallelism

Tensor parallelism changes the local head counts passed into `FlashAttentionImpl`.

The impl receives:

- `num_heads`: local query heads on this TP rank.
- `num_kv_heads`: local KV heads on this TP rank.
- `head_size`: unchanged per-head dimension.

FlashAttention runs on local rank tensors.

It does not see the global attention-head set as one tensor.

For MQA/GQA, `flash_attn_varlen_func(...)` supports fewer KV heads than query heads as long as query heads are divisible by KV heads.

## DCP Capability

`FlashAttentionImpl.can_return_lse_for_decode = True`.

This is the key capability that lets DCP work.

When DCP is active, `AttentionImplBase.__new__(...)` sets:

- `need_to_return_lse_for_decode = True`

Then `check_attention_cp_compatibility(...)` in `vllm/v1/worker/cp_utils.py` accepts the impl for DCP.

## Why DCP Needs LSE

DCP shards cached decode context across ranks.

Each rank computes attention over only its local context shard.

Softmax normalization is global across all context shards, so each rank must return:

- partial attention output
- softmax LSE for that partial attention

The LSE values let vLLM combine partial outputs exactly, instead of averaging or summing incorrect softmax-normalized outputs.

## DCP Metadata

`FlashAttentionMetadataBuilder.build(...)` computes:

- `dcp_context_kv_lens`
- `max_dcp_context_kv_len`

It derives local context lengths with:

- `get_dcp_local_seq_lens(...)`

The local max length accounts for:

- `dcp_world_size`
- `cp_kv_cache_interleave_size`

This avoids over-allocating workspace while still covering the maximum local shard length.

## _forward_with_dcp

When `self.dcp_world_size > 1`, normal forward dispatches to:

- `FlashAttentionImpl._forward_with_dcp(...)`

That path has two attention computations.

First, it computes attention against the local DCP context shard:

- all-gather query across DCP heads/ranks with `get_dcp_group().all_gather(query, dim=1)`.
- call `flash_attn_varlen_func(...)` with `k=key_cache`, `v=value_cache`, `seqused_k=dcp_context_kv_lens`, `max_seqlen_k=max_dcp_context_kv_len`, `causal=False`, and `return_softmax_lse=True`.
- combine partial context outputs across DCP ranks using `self.dcp_combine(...)`.

Second, it computes attention over the current query/new-token K/V region:

- call `flash_attn_varlen_func(...)` with direct `k=key`, `v=value`, `cu_seqlens_k=cu_seqlens_q`, `max_seqlen_k=max_query_len`, `causal=attn_metadata.causal`, and `return_softmax_lse=True`.

Finally, it merges context attention and query attention:

- `merge_attn_states(output, context_attn_out_cor, context_lse_cor, query_attn_out, query_lse)`

This split is why DCP support is more than just passing `block_table` into the normal kernel.

## DCP Communication Backend

`FlashAttentionImpl.__init__(...)` chooses:

- `dcp_a2a_lse_reduce` when `parallel_config.dcp_comm_backend == "a2a"`.
- `cp_lse_ag_out_rs` otherwise.

`cp_lse_ag_out_rs(...)` combines LSE-weighted outputs and reduce-scatters the result by head.

`dcp_a2a_lse_reduce(...)` uses all-to-all communication to exchange packed output/LSE state and then combines it.

Both paths are designed to produce the same global softmax result.

## CP Interleave

`cp_kv_cache_interleave_size` affects how context-parallel KV slots are striped across ranks.

FlashAttention metadata uses it when computing DCP-local context lengths.

The block table and slot mapping side also use total CP rank coordinates, so the kernel sees a rank-local cache layout that already reflects CP ownership.

For backend authors, this means the kernel path must not assume logical token positions map linearly into one rank-local contiguous cache without considering block table and CP-aware sequence lengths.

## PCP

`FlashAttentionImpl` does not set `supports_pcp = True` in the current code.

Because `AttentionImplBase.supports_pcp` defaults to `False`, `check_attention_cp_compatibility(...)` rejects this impl if `prefill_context_parallel_size > 1`.

So even though FlashAttention has strong prefill kernels, the standard `FLASH_ATTN` impl is not currently advertising vLLM PCP support in this path.

For a future PCP-capable FlashAttention-style backend, the kernel/runtime path would need to support sharded prefill query ranges, distributed or gathered K/V context, CP-aware slot mapping, and correct communication for partial-query/full-KV or partial-query/partial-KV strategies.

## Pipeline And Data Parallelism

Pipeline parallelism partitions layers across stages.

It does not usually change the FlashAttention backend contract for a single layer.

Data parallelism replicates execution across DP ranks.

It does not usually change FlashAttention kernel inputs inside one rank.

## Key Files

- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/utils.py`
- `vllm/v1/worker/cp_utils.py`
- `vllm/v1/attention/ops/common.py`
- `vllm/v1/attention/ops/dcp_alltoall.py`
- `vllm/v1/attention/ops/merge_attn_states.py`
- `vllm/v1/worker/block_table.py`

