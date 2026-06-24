# Distributed, Hybrid, And Special Cases

KV-cache management is local to each worker's tensor shard but coordinated through globally consistent request block IDs and cache-group structure.

This chapter covers parallelism and cache lifecycles that differ from ordinary full-attention decode.

## Tensor Parallelism

Tensor parallelism shards attention heads and projection weights across ranks.

`ModelConfig.get_num_attention_heads(parallel_config)` returns query heads local to the TP rank.

`get_num_kv_heads(...)` returns local KV heads, using at least one where MQA/GQA has fewer total KV heads than TP ranks.

Each TP rank therefore allocates cache pages sized for its local KV heads:

```text
local page bytes = block_size * local_num_kv_heads * per-head K/V storage
```

The scheduler block ID sequence is logically the same across TP ranks, but block `b` points to a different head shard on each worker.

A complete transferred/offloaded prefix is valid only when every TP rank has moved its local shard.

Connector worker metadata aggregation enforces this by counting job completion across workers.

## MQA KV Replication

When total KV heads are fewer than TP ranks, vLLM can replicate KV heads so every query-head shard has local K/V access.

Replicated ranks store equivalent K/V values in separate local cache tensors.

The attention backend receives already resolved local `num_kv_heads`; cache management does not perform head communication during paged reads.

Connector TP mappings must preserve which rank owns or replicates each shard.

## Pipeline Parallelism

Pipeline ranks host different model layers.

The global cache grouping plan is projected to each worker's resident layers by `_project_kv_cache_groups_to_worker(...)` in `vllm/v1/core/kv_cache_utils.py`.

Each PP rank allocates only its local layer tensors while maintaining group/block semantics compatible with scheduler output.

Disaggregated connectors that span PP need handshake metadata keyed by `(pp_rank, tp_rank)`.

The default base handshake implementation rejects nonzero PP ranks unless a connector overrides PP-aware handling.

## Data Parallelism

Data-parallel replicas schedule different request subsets and maintain independent block pools and physical cache tensors.

KV is not automatically synchronized between DP replicas.

External connectors or request routing can make prefixes available across replicas, but local block IDs remain engine-local destinations.

## Context Parallelism Overview

PCP and DCP shard token context rather than heads.

The total context-parallel size is:

```text
total_cp_world_size = pcp_world_size * dcp_world_size
total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
```

Each rank stores only its assigned token chunks.

Full-attention memory sizing divides maximum sequence token capacity across total CP size before converting to pages.

## CP Virtual Blocks

A local cache page of `block_size` rows represents a global virtual span of:

```text
block_size * total_cp_world_size
```

Every rank uses the same request block-table index for that virtual span, but writes only positions assigned to its CP rank.

The local block offset compacts the rank's interleaved token chunks.

## cp_kv_cache_interleave_size

Interleave size controls how many consecutive global token positions belong to one rank before ownership rotates.

For interleave `I` and CP size `C`:

```text
owner(position_in_virtual_block) = floor(offset / I) mod C
```

`I = 1` round-robins individual tokens.

`I = block_size` assigns block-sized contiguous chunks before rotating.

The value must satisfy block-size validation and distributed attention implementation requirements.

## PCP

Prefill Context Parallelism partitions prompt tokens across ranks.

An attention implementation must set `supports_pcp = True` and understand local query/KV metadata, collective combination, and rank ownership.

The cache manager allocates local capacity based on PCP sharding, and slot mapping suppresses writes for nonlocal prompt tokens.

A backend that assumes every prompt K/V row is locally materialized cannot use PCP merely because its cache shape is valid.

## DCP

Decode Context Parallelism partitions historical decode context across ranks.

Each rank computes attention against local K/V and returns local output plus softmax LSE.

Distributed reduction merges attention states using LSE-weighted formulas.

`check_attention_cp_compatibility(...)` in `vllm/v1/worker/cp_utils.py` requires `need_to_return_lse_for_decode` for DCP.

The cache manager and block-table slot logic handle token ownership, but the attention implementation must expose compatible LSE and collective behavior.

## Speculative Decoding

Speculative decoding reserves `num_lookahead_tokens` beyond finalized request tokens.

Lookahead pages ensure draft tokens have physical cache slots during proposal/verification.

Rejected draft tokens must not be committed to prefix cache.

`KVCacheManager.allocate_slots(...)` caps committed token count at `request.num_tokens`, separating allocated speculative capacity from finalized hashable content.

## EAGLE And MTP Groups

EAGLE/MTP draft layers may have their own cache groups marked `is_eagle_group`.

Hybrid prefix lookup can request an extra block and then drop it to match the draft model's shifted token/cache relationship.

Group generation annotates DeepSeek and other model-specific draft/main cache relationships.

Worker block tables and drafter metadata receive per-group mappings.

## Asynchronous Spec Decode

Async speculative execution can make GPU sequence-length tensors authoritative while CPU copies lag.

Metadata construction can omit CPU fields and use persistent GPU buffers.

Slot mappings and block tables must remain valid across deferred draft proposal/verification steps and any microbatch slicing.

## Sliding Window

Sliding-window cache management can release historical pages that future queries cannot attend to.

The active requirement includes:

- Initial positions still needed by model-specific semantics, if any.
- Last `sliding_window - 1` computed tokens.
- Current scheduled tokens.
- One extra page for unaligned window start.

Released logical positions are represented by null blocks where table shape must remain stable.

Attention kernels still apply the exact token-level window mask; block reclamation is a coarser memory optimization.

DCP is currently rejected for `SlidingWindowSpec.max_memory_usage_bytes(...)` in this checkout.

## Chunked Local Attention

Chunked-local attention partitions the sequence into aligned fixed-size chunks.

A query attends within its chunk and configured lookback rather than a continuously rolling window.

The cache manager can reclaim completed chunks that are no longer reachable.

Prefix hit length must align with both cache pages and attention chunk semantics.

## Attention Sinks

Sink attention permanently retains initial sink tokens while managing a bounded recent context.

`SinkFullAttentionSpec` carries `sink_len`, and `SinkFullAttentionManager` preserves both sink and recent regions.

The attention backend must separately implement sink logits/softmax semantics; retaining sink K/V alone is insufficient.

## MLA

MLA stores compressed latent K/V, often with one logical KV head and backend-specific layout.

`storage_block_size` may differ from scheduler `block_size` through `compress_ratio`.

Page bytes can include fixed-format latent vectors, RoPE components, scales, and alignment padding.

Worker reshaping must use storage block size while scheduler block tables still represent logical token spans.

Connectors should transfer opaque page bytes unless they explicitly understand MLA format conversion.

## DeepSeek V4 Grouping

DeepSeek V4 uses specialized cache grouping and page-size handling for main/draft layers and custom FP8 layouts.

`_get_kv_cache_config_deepseek_v4(...)` and related group annotations in `vllm/v1/core/kv_cache_utils.py` avoid forcing incompatible tuples into generic grouping assumptions.

Unused reserved tensor slots can appear in `KVCacheTensor`; offload canonicalization filters entries with no actual layer references.

## TurboQuant

TurboQuant uses `TQFullAttentionSpec` and a combined uint8 K/V slot.

The scheduler lifecycle remains full-attention block management, but page-size bytes and physical tensor shape are backend-specific.

Connector transfers must preserve packed bytes and preset identity.

Generic per-token-head scale logic does not describe TurboQuant's inline key norms and value metadata.

## Per-Token-Head Quantized Cache

The cache spec budgets float32 K and V scales for every token/head.

Some backends carve those scale arrays from page tails and expose strided float32 views.

Virtual block splitting, connector page copies, and CP ownership must move scale bytes together with quantized data.

A connector that copies only the apparent K/V tensor view and omits inline scale tail produces structurally valid but numerically corrupted cache.

## NVFP4

NVFP4 uses packed data and block scales whose physical dimension is not equal to logical head size.

Cache page formulas, backend shapes, and transfer bytes must all use `nvfp4_kv_cache_full_dim(...)`.

Kernel requirements can be stricter than generic scheduler block-size requirements because quantization scale groups need alignment.

## Mamba Hybrid Models

Hybrid models can combine attention pages and Mamba state pages in one block pool by padding to compatible page sizes.

The same block ID then refers to one attention page in attention tensors and one state page in Mamba tensors.

Managers decide different reachability/commit behavior for each group while the hybrid coordinator finds a common schedulable prefix and capacity.

Mamba pages may require zeroing when newly allocated because stale recurrent state is immediately semantically meaningful.

## Mamba Cache Modes

`none` minimizes persistent state and does not support ordinary block-boundary prefix reuse.

`align` stores selected boundary states and tracks scheduler-step alignment.

`all` stores every block-boundary state and can participate in prefix caching similarly to attention, though one state summarizes the preceding block.

Speculative Mamba execution reserves additional state blocks.

## Cross Attention

Cross-attention cache holds encoder states for decoder queries.

Capacity follows maximum encoder input length rather than decoder model length.

The decoder request lifetime owns those pages, and encoder cache manager release occurs when request completes or is aborted.

General KV connector support currently excludes encoder-decoder models in scheduler construction.

## Encoder-Only Attention

Encoder-only attention computes direct Q/K/V without persistent autoregressive cache.

`EncoderOnlyAttentionSpec.max_memory_usage_bytes(...)` returns zero.

Worker metadata supplies dummy block tables/slot mappings where common interfaces require tensors, but no persistent cache writes should occur.

## Cross-Layer KV Sharing

Some architectures generate K/V in one layer and reuse it in later layers.

Shared layers point to the same physical cache tensor and cache group.

Only the producer updates K/V.

Consumers must use compatible head shape, dtype, positional semantics, and backend layout.

Connectors should transfer the shared storage once rather than duplicating each alias.

## Fast Prefill Sharing

`kv_sharing_fast_prefill` identifies suffix layers that may skip prefill token work because K/V is produced elsewhere.

The model runner adjusts metadata for eligible layers, but the feature remains backend/model specific.

The optimization must not skip tokens whose shared cache rows have not been produced.

## KV Cache Zeroing

`KVCacheConfig.needs_kv_cache_zeroing` is true when Mamba layers are present.

The model runner tracks newly allocated block IDs and zeros relevant physical pages before model use.

Ordinary attention pages do not need zeroing because sequence lengths and slot writes mask stale rows.

## Offloading Under Parallelism

Configured `kv_offloading_size` is divided across workers according to world size in native CPU spec accounting.

One logical offload key is globally complete only when all worker shards finish.

Uniform cross-layer layouts reduce transfer count per worker but do not combine TP shards into one process's memory.

RDMA connectors exchange memory-registration and TP/PP mapping metadata so remote producers and consumers match shards correctly.

## Connector Layout Compatibility

A connector can require a cache layout through `get_required_kvcache_layout(...)`.

Backend selection validates whether an attention backend supports connector use and the required physical arrangement.

For HMA, connectors must understand group-specific page references or canonical opaque pages rather than assuming every group has standard `[2, blocks, ...]` storage.

## Distributed Integration Checklist

A new cache/backend path must answer:

- Are query and KV heads local, replicated, or communicated under TP?
- Does page sizing use local or global head count?
- Which token positions are local under PCP/DCP?
- Does slot mapping correctly suppress nonlocal writes?
- Can decode return LSE for DCP state merging?
- Does prefill support PCP partitioned context?
- Can connectors map TP and PP shards consistently?
- Are quantization scales transferred with data?
- Are speculative and hybrid groups represented separately where needed?
