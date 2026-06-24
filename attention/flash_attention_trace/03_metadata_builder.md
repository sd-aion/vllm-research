# FlashAttention Metadata Builder

This note explains how vLLM turns common runtime batch metadata into the object consumed by `FlashAttentionImpl.forward(...)`.

## Metadata Classes

The FlashAttention-specific metadata class is:

- `FlashAttentionMetadata` in `vllm/v1/attention/backends/flash_attn.py`

It is built by:

- `FlashAttentionMetadataBuilder` in `vllm/v1/attention/backends/flash_attn.py`

The builder subclasses `AttentionMetadataBuilder[FlashAttentionMetadata]` from `vllm/v1/attention/backend.py`.

## Inputs From Common Metadata

`FlashAttentionMetadataBuilder.build(...)` receives `CommonAttentionMetadata`.

Important common fields used by FlashAttention are:

- `num_actual_tokens`: number of real tokens excluding padding.
- `max_query_len`: largest query length in the batch.
- `query_start_loc`: cumulative starts for each request's query tokens.
- `max_seq_len`: largest total sequence length in the batch.
- `seq_lens`: current total sequence length for each request.
- `block_table_tensor`: per-request mapping from logical KV blocks to physical KV blocks.
- `slot_mapping`: per-token physical KV slot locations for KV-cache update.
- `causal`: runtime causal/non-causal flag.

The builder copies these into `FlashAttentionMetadata` with backend-specific additions.

## FlashAttentionMetadata Fields

The primary fields are:

- `num_actual_tokens`: used to slice away padding before launching kernels.
- `max_query_len`: passed as `max_seqlen_q` to `flash_attn_varlen_func(...)`.
- `query_start_loc`: passed as `cu_seqlens_q`.
- `max_seq_len`: passed as `max_seqlen_k` in normal paged attention.
- `seq_lens`: passed as `seqused_k` when using the paged KV cache.
- `block_table`: passed as the page table for paged KV attention.
- `slot_mapping`: used by the separate KV-cache update path.
- `causal`: forwarded to the FlashAttention kernel call.

These fields are enough for the standard non-DCP, non-cascade path.

## Why query_start_loc And seq_lens Are Both Needed

`query_start_loc` describes where each request's new query tokens live in the packed query tensor.

`seq_lens` describes how many K/V tokens the query is allowed to attend over for each request.

During prefill, query lengths can be large because many prompt tokens are processed together.

During decode, query length is usually one token per active request, while `seq_lens` includes the full cached context plus the new token.

This is why the same metadata object can represent both prefill and decode.

## AOT Scheduler Metadata

FA3 can use ahead-of-time scheduler metadata.

The relevant pieces are:

- `self.aot_schedule = get_flash_attn_version() == 3`
- `get_scheduler_metadata(...)` from `vllm/v1/attention/backends/fa_utils.py`
- `scheduler_metadata` stored in `FlashAttentionMetadata`
- `prefix_scheduler_metadata` for cascade prefix attention

The lower-level wrapper in `vllm/vllm_flash_attn/flash_attn_interface.py` calls:

- `torch.ops._vllm_fa3_C.get_scheduler_metadata`

Scheduler metadata precomputes launch scheduling decisions for FA3 based on batch size, max Q/K lengths, head counts, dtype, page size, causal mode, sliding window, and split count.

## CUDA Graph Support

`FlashAttentionMetadataBuilder._cudagraph_support` is:

- `AttentionCGSupport.ALWAYS` for FA3 or XPU.
- `AttentionCGSupport.UNIFORM_BATCH` otherwise.

These enum values come from `AttentionCGSupport` in `vllm/v1/attention/backend.py`.

`AttentionCGSupport.ALWAYS` means the backend metadata path is considered safe for CUDA graph capture across all normal attention batch shapes, including mixed prefill/decode batches where different requests can have different query lengths.

`AttentionCGSupport.UNIFORM_BATCH` means CUDA graph capture is only considered safe when the batch has uniform query shape, such as decode/spec-decode batches where each request has the same query length.

For FlashAttention, the code comment in `vllm/v1/attention/backends/flash_attn.py` says FA3 supports full CUDA graphs for all cases, while FA2 has a special `max_query_len=1` packed-GQA path that can make graphs captured for single-token decode unsafe for mixed prefill/decode.

The builder also handles `flash_attn_max_num_splits_for_cuda_graph`.

When full CUDA graphs are enabled and scheduler metadata exists, the builder copies the computed scheduler metadata into a preallocated tensor and zeroes the remaining slots.

That keeps captured graph shapes stable while avoiding stale scheduler entries.

## DCP Metadata

When DCP is active, the builder computes local context lengths for the current DCP rank.

The relevant code path is:

- `get_dcp_local_seq_lens(...)` from `vllm/v1/attention/backends/utils.py`
- `self._dcp_context_kv_lens`
- `FlashAttentionMetadata.dcp_context_kv_lens`
- `FlashAttentionMetadata.max_dcp_context_kv_len`

`get_dcp_local_seq_lens(...)` takes the per-request context length and computes how many cached context tokens belong to the current DCP rank after decode-context sharding and `cp_kv_cache_interleave_size` are applied.

`self._dcp_context_kv_lens` is a persistent GPU buffer allocated by `FlashAttentionMetadataBuilder.__init__(...)`; the builder writes the current step's local context lengths into this buffer instead of allocating a fresh tensor every build.

`FlashAttentionMetadata.dcp_context_kv_lens` is the slice of that buffer for the active requests in the current batch, and `_forward_with_dcp(...)` passes it to `flash_attn_varlen_func(...)` as `seqused_k`.

`FlashAttentionMetadata.max_dcp_context_kv_len` is the maximum local context length bound used for the FA call's `max_seqlen_k`; the builder computes it as `ceil(max_seq_len / (dcp_world_size * interleave_size)) * interleave_size` to avoid a GPU-to-CPU sync while still allocating enough workspace.

The builder computes:

- `query_lens = query_start_loc[1:] - query_start_loc[:-1]`
- `context_kv_lens = seq_lens - query_lens`
- local context lengths after DCP sharding and CP interleave

This lets `_forward_with_dcp(...)` run attention over only the local rank's shard of cached context.

## Cascade Metadata

Cascade attention is enabled when `common_prefix_len > 0` and the heuristic accepts it.

The builder creates:

- `cu_prefix_query_lens = [0, num_actual_tokens]`
- `prefix_kv_lens = [common_prefix_len]`
- `suffix_kv_lens = seq_lens[:num_reqs] - common_prefix_len`
- `prefix_scheduler_metadata`
- `scheduler_metadata` for the suffix part

The runtime then computes attention against the shared prefix and request-specific suffix separately, and merges them with LSE weighting.

## Sliding Window Metadata

The builder does not carry a standalone `sliding_window` field in `FlashAttentionMetadata`.

Sliding-window settings live on `FlashAttentionImpl.sliding_window`.

The builder only needs sliding-window information for FA3 AOT scheduling through `self.aot_sliding_window`.

It discovers whether all FlashAttention layers share one sliding-window config via `_get_sliding_window_configs(...)`.

If multiple sliding-window configs exist, AOT scheduling is disabled because the scheduler metadata expects a stable window value.

## update_block_table

`FlashAttentionMetadataBuilder.supports_update_block_table = True`.

`update_block_table(...)` shallow-copies existing metadata and replaces:

- `block_table`
- `slot_mapping`

This is useful when runtime code needs to reuse most metadata while updating cache mapping tensors, especially around CUDA graph and worker-side execution paths.

## Key Files

- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backend.py`
- `vllm/v1/attention/backends/fa_utils.py`
- `vllm/v1/attention/backends/utils.py`
- `vllm/vllm_flash_attn/flash_attn_interface.py`
