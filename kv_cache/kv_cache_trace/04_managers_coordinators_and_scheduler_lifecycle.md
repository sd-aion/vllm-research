# Managers, Coordinators, And Scheduler Lifecycle

This chapter explains how cache specs become request-specific block sequences and how scheduling decisions allocate, retain, skip, or release those blocks.

## KVCacheBlocks

`KVCacheBlocks` in `vllm/v1/core/kv_cache_manager.py` is the allocation interface returned to the scheduler.

Its `blocks` field is:

```text
tuple[group_id -> sequence[KVCacheBlock]]
```

The outer dimension is cache group because future groups may use different block sizes and therefore hold different block counts for the same request length.

`get_block_ids(...)` converts metadata objects into worker-facing integer lists.

`get_unhashed_block_ids_all_groups()` identifies newly allocated/uncommitted blocks while skipping padding null blocks.

## KVCacheManager Facade

`KVCacheManager` is the scheduler-facing facade.

It owns a coordinator, exposes the coordinator's shared `BlockPool`, records cache event metadata, and provides an immutable empty allocation object to reduce Python garbage.

Important methods are:

- `get_computed_blocks(request)` for local prefix hits.
- `allocate_slots(...)` for hit attachment, external load slots, current computation, and lookahead.
- `free(request)` for request release.
- `remove_skipped_blocks(...)` for manager-specific history reclamation.
- `evict_blocks(...)` for connector-reported invalid pages.
- `reset_prefix_cache()` for global hash invalidation.
- `cache_blocks(...)` for delayed commit after external transfer.
- `get_blocks(...)` and block-ID helpers for scheduler output.
- `take_events()` and prefix stats for observability.

## Coordinator Selection

`get_kv_cache_coordinator(...)` in `vllm/v1/core/kv_cache_coordinator.py` chooses:

- `KVCacheCoordinatorNoPrefixCache` when prefix caching is disabled.
- `UnitaryKVCacheCoordinator` when there is exactly one cache group.
- `HybridKVCacheCoordinator` when multiple cache groups have coordinated lifecycles.

All coordinators use the shared block pool but differ in request block structures and cross-group hit calculation.

## KVCacheCoordinator Contract

The base coordinator defines operations needed by `KVCacheManager`:

- Count blocks required for a candidate request state.
- Allocate cached blocks and newly needed blocks.
- Allocate slots corresponding to external connector hits.
- Commit finalized blocks.
- Find the longest cache hit.
- Remove history skipped by the cache policy.
- Free a request in eviction-priority order.
- Compute common prefix blocks across running requests.
- Start a new scheduling step for managers with step-local state.

The coordinator hides whether a request has one list of blocks or several manager-specific lists.

## SingleTypeKVCacheManager

`SingleTypeKVCacheManager` in `vllm/v1/core/single_type_kv_cache_manager.py` is the per-lifecycle manager abstraction used inside coordinators.

It stores request-to-block sequences for one or more groups with compatible specs.

Its class/static methods let a coordinator calculate prefix hits without constructing a temporary manager instance for every query.

Manager selection is driven by cache-spec registration and `get_manager_for_kv_cache_spec(...)`.

## FullAttentionManager

Full attention retains a contiguous block prefix from token zero through the current sequence tail.

Allocation count is approximately:

```text
ceil(required_tokens / block_size) - currently_attached_blocks
```

Prefix lookup is downward-closed: if the first `n` blocks hit, every shorter aligned prefix also hits.

Common-prefix calculation compares shared leading block identities across running requests.

Freeing reverses logical block order so deeper prefix hashes are evicted before broadly reusable shallow prefixes.

## SlidingWindowManager

Sliding-window attention does not need every historical page once all future queries are beyond it.

The manager computes which logical block positions remain reachable by any current/future query in the active window.

Unreachable positions can be replaced with the null block and their physical references released before new allocation.

Prefix hits are more complex than full attention because useful cached blocks can include a beginning prefix and a reachable tail window while middle blocks may be unnecessary.

The manager enforces contiguous/reachable semantics required by worker block tables and window masks.

## ChunkedLocalAttentionManager

Chunked local attention uses aligned attention chunks rather than a rolling trailing window.

The manager retains blocks belonging to the current relevant chunk region and aligns lookup/admission to `attention_chunk_size`.

Blocks from completed chunks can be skipped once no later query can attend to them.

## MambaManager

Mamba cache pages represent recurrent states at selected token boundaries.

Behavior depends on `mamba_cache_mode`:

- `none` maintains only current state and speculative requirements.
- `align` retains state at selected aligned/scheduler boundaries.
- `all` caches every block-boundary state for prefix reuse.

Mamba manager allocation, skip, and commit logic differs because one state page summarizes preceding tokens rather than storing one K/V row per token.

`new_step_starts()` resets step-local alignment information.

## CrossAttentionManager

Cross-attention pages cache encoder-side states and are allocated according to encoder token count.

They do not grow one page per decoder token.

The manager coordinates their lifetime with the decoder request and encoder cache manager.

KV connectors currently reject encoder-decoder models in the scheduler path because the connector lifecycle does not generally cover cross-attention state.

## SinkFullAttentionManager

Sink attention combines full/prefix behavior with permanent retention of initial sink tokens and a bounded recent context.

The manager derives reachability from `sink_len` and window semantics while preserving hash-compatible sink blocks.

## HybridKVCacheCoordinator

The hybrid coordinator groups manager classes/specs and iteratively finds a hit length valid for every group.

One group may report a shorter hit because its window, chunk alignment, EAGLE behavior, or block size differs.

The coordinator reduces candidate hit length and reruns dependent lookups until it stabilizes.

Full-attention hits can be looked up once and truncated because they are downward-closed.

For EAGLE groups, lookup can temporarily request one extra block and then drop the final block to match draft-model semantics.

The returned token hit length is therefore the longest prefix that every required group can represent consistently.

## Scheduler Construction

`Scheduler.__init__(...)` in `vllm/v1/core/sched/scheduler.py` creates `KVCacheManager` from resolved `KVCacheConfig`, scheduler/hash block sizes, model length, scheduler token budget, caching flags, and CP sizes.

If a KV connector is configured, the scheduler creates a scheduler-role connector and binds the manager's `BlockPool` after manager construction.

This binding lets offload connectors inspect/touch GPU prefix blocks and coordinate asynchronous ownership.

## Beginning A Scheduler Step

`KVCacheManager.new_step_starts()` forwards to managers that maintain step-local state.

The scheduler then considers running and waiting requests, available token budget, encoder budget, preemption pressure, and connector state.

## Local Prefix Lookup

For a new/waiting request, the scheduler calls `get_computed_blocks(...)`.

The result contains local GPU prefix blocks and an aligned count of locally computed tokens.

Lookup is skipped for requests requiring behavior incompatible with prefix reuse, such as some prompt-logprob or pooling modes.

## External Prefix Lookup

If a connector exists, `connector.get_num_new_matched_tokens(request, num_computed_tokens)` reports how many additional tokens exist outside local GPU cache.

The method can be called multiple times and should be side-effect free.

The connector returns both a token count and a flag indicating whether loading can be handled asynchronously.

Local and external prefix hit statistics are kept separately.

## allocate_slots Inputs

`KVCacheManager.allocate_slots(...)` accepts:

- `request`: current request state.
- `num_new_tokens`: tokens to compute this step.
- `num_new_computed_tokens`: newly discovered local prefix-hit tokens.
- `new_computed_blocks`: local cached blocks corresponding to those hits.
- `num_lookahead_tokens`: speculative slots needed beyond scheduled computation.
- `num_external_computed_tokens`: connector-resident tokens needing local slots/load.
- `delay_cache_blocks`: postpone prefix commitment while transfer is incomplete.
- `num_encoder_tokens`: cross-attention allocation requirement.
- `full_sequence_must_fit`: admission gate for complete prompt/sequence capacity.
- `reserved_blocks`: capacity protected for already in-flight work.

## Allocation Regions

The logical request is divided into:

```text
[already computed][new local cache hit][external cache hit][new compute][lookahead]
```

The first regions may already have physical blocks or may need blocks allocated as transfer destinations.

New compute and lookahead always need reachable writable slots unless a specialized manager skips storage.

## Allocation Stage One: Reclaim And Admission

The coordinator first removes blocks no longer reachable under sliding/chunked/Mamba policy.

It then asks each manager how many additional pool blocks are required.

If `full_sequence_must_fit` is true, capacity is checked against the entire request rather than only the current chunk. This prevents chunked prefill from admitting more requests than can later progress.

`reserved_blocks` protects capacity needed by in-flight sequences while an asynchronous connector load attempts admission.

If required blocks exceed available blocks, `allocate_slots(...)` returns `None` without partially attaching new allocations.

## Allocation Stage Two: Attach Hits

Local computed blocks are touched and appended to request state.

External computed tokens receive local destination blocks because their remote bytes must be copied into GPU pages before attention can read them.

Coordinator state records the distinction so delayed cache commitment and connector metadata remain correct.

## Allocation Stage Three: New Compute And Lookahead

The coordinator allocates pages through the block pool until every manager can represent `computed + new + lookahead` token requirements.

Speculative lookahead blocks reserve write locations for draft proposals but are not necessarily committed to prefix cache.

## Cache Commitment

When prefix caching is enabled and commitment is not delayed, the manager commits finalized full blocks through `BlockPool.cache_full_blocks(...)`.

The token limit is capped at `request.num_tokens` so unverified speculative draft tokens do not enter prefix cache.

When connector transfer is pending, `delay_cache_blocks` prevents local hash publication before bytes are valid.

Later scheduler completion handling calls cache methods once the worker reports successful receive.

## Scheduling Output

The scheduler includes new/updated block IDs per request and cache group in `SchedulerOutput`.

It also asks the connector to build step metadata after allocation, so worker transfers reference the exact GPU destination/source blocks selected by the manager.

## Decode Growth

During decode, most requests add one finalized token per step, though speculative decode can schedule several candidates.

The current tail page remains attached and unhashed until full.

When a token crosses a page boundary, the next scheduler step allocates a new block and appends its ID to the request block table.

Once the old page becomes full and tokens are accepted, it can be hashed and shared.

## Preemption

When blocks are insufficient, the scheduler can preempt a lower-priority request.

Local request references are released, while full hashed blocks may remain as free cached entries.

Unhashed partial state cannot be recovered from local prefix cache and normally must be recomputed unless a connector/offload path saved it.

Connector metadata receives preemption information before worker execution so an offload worker can preserve selected pages before overwrite.

## Request Finish

At finish, the scheduler asks the connector whether block freeing must be delayed for asynchronous send/save.

The connector can assume temporary responsibility and later report `finished_sending`.

If no delay is required, `KVCacheManager.free(request)` immediately releases request references.

Encoder cache is released through its separate manager.

## Connector Load Completion

Worker `KVConnectorOutput.finished_recving` tells the scheduler that remote K/V is now valid in allocated GPU blocks.

The scheduler updates request state, commits eligible loaded blocks, and transitions the request from waiting-for-KV to runnable computation.

Invalid block IDs from worker load errors are evicted from prefix hash lookup before recompute or failure policy proceeds.

## Reset And Shutdown

Scheduler prefix reset first requires local blocks to be unreferenced.

Optional connector reset clears external cache state if supported.

At shutdown, scheduler-role connectors and cache-related background services are terminated after pending lifecycle handling.
