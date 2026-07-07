# Managers, Coordinators, And Scheduler Lifecycle

This chapter explains how cache specs become request-specific block sequences and how scheduling decisions allocate, retain, skip, or release those blocks.

## KVCacheBlocks

`KVCacheBlocks` in `vllm/v1/core/kv_cache_manager.py` is the allocation interface returned to the scheduler.

Its `blocks` field is:

```text
tuple[group_id -> sequence[KVCacheBlock]]
```

`blocks[i][j]` refers to the i-th kv_cache_group and the j-th block of tokens.

The outer dimension is cache group because future groups may use different block sizes and therefore hold different block counts for the same request length.

`get_block_ids(...)` converts metadata objects into worker-facing integer lists.

`get_unhashed_block_ids_all_groups()` identifies newly allocated/uncommitted blocks while skipping padding null blocks.

## KVCacheManager Facade

`KVCacheManager` is the scheduler-facing facade.

It owns a coordinator, exposes the coordinator's shared `BlockPool`, records cache event metadata, and provides an immutable empty allocation object to reduce Python garbage.

Important methods are:

- `get_computed_blocks(request)` for local prefix hits. You get the computed (cached) blocks for the request. It returns at tup;e that has a list of blocks that are computed for the request and the number of computed tokens.
- `allocate_slots(...)` decides which physical KV blocks this request will use for the current scheduler step: it attaches local prefix-hit blocks, allocates destination blocks for connector-loaded KV, allocates writable blocks for newly computed tokens, reserves lookahead blocks for speculative decoding, and returns `None` if there is not enough block capacity.
- `free(request)` for request release. This releases the request's ownership of KV-cache block IDs; it does not necessarily erase GPU memory.
- `remove_skipped_blocks(...)` releases blocks that a lifecycle-specific policy no longer needs, such as old sliding-window pages, completed chunk-local regions, or obsolete Mamba state pages.
- `evict_blocks(...)` removes prefix-cache hash metadata for specific physical block IDs, mainly when a connector reports that loaded or transferred pages are invalid and must not be reused as prefix hits.
- `reset_prefix_cache()` clears all prefix-cache hash metadata globally, but only when normal blocks are not actively referenced.
- `cache_blocks(...)` commits finalized full blocks into prefix cache by assigning block hashes; it can be delayed when external connector transfer has not yet made the bytes valid.
- `get_blocks(...)` and block-ID helpers expose the request's currently attached block IDs so the scheduler can put them into `SchedulerOutput` for workers.
- `take_events()` and prefix stats expose block-store/remove events and cache-hit statistics for metrics, logging, and connector/event consumers.

## Coordinator Selection

The coordinator is the object inside `KVCacheManager` that orchestrates one or more per-lifecycle managers. The scheduler talks to `KVCacheManager`; `KVCacheManager` delegates group-specific work to the coordinator; the coordinator delegates lifecycle-specific details to `SingleTypeKVCacheManager` implementations such as full attention, sliding window, chunked local attention, or Mamba.

The distinction is:

- `KVCacheManager`: scheduler-facing facade for the whole cache system.
- `KVCacheCoordinator`: decides how one request's operation spans one or more cache groups.
- `SingleTypeKVCacheManager`: implements the block policy for one cache lifecycle, such as full attention or sliding-window attention.

A cache lifecycle means the rule for when cached state is allocated, retained, skipped, reusable, and freed. Full attention retains a growing prefix from token zero; sliding-window attention eventually drops old pages; chunked-local attention keeps chunk-aligned regions; Mamba stores recurrent state pages rather than ordinary per-token K/V rows. These different lifecycles need different manager policies even though they all use `KVCacheBlock` IDs.

`get_kv_cache_coordinator(...)` in `vllm/v1/core/kv_cache_coordinator.py` chooses:

- `KVCacheCoordinatorNoPrefixCache` when prefix caching is disabled.
- `UnitaryKVCacheCoordinator` when there is exactly one cache group.
- `HybridKVCacheCoordinator` when multiple cache groups have coordinated lifecycles.

`KVCacheCoordinatorNoPrefixCache` skips prefix-cache lookup and only coordinates allocation/freeing. `UnitaryKVCacheCoordinator` handles the simple case where each request has one block sequence because there is only one cache group. `HybridKVCacheCoordinator` handles multiple groups, where the same request may need separate block sequences and a prefix hit is only valid if all required groups can represent the same computed-token prefix.

All coordinators use the shared block pool, but they differ in request block structures, cache-hit calculation, and how they combine manager results across groups.

## KVCacheCoordinator Contract

The base coordinator defines operations needed by `KVCacheManager`:

- Count how many additional blocks a request would need. This is used before allocation so the scheduler can reject or preempt instead of partially allocating a request that cannot fit.
- Attach cached blocks and allocate new blocks. Cached blocks come from local prefix-cache hits; new blocks are writable GPU destinations for tokens that will be computed now.
- Allocate slots for external connector hits. If a connector says some prefix tokens exist outside GPU memory, the coordinator reserves local block IDs where those remote KV bytes will be loaded.
- Commit finalized blocks. Once tokens are accepted and a block is full, the coordinator asks managers to publish that block into prefix cache.
- Find the longest cache hit. For a new request, the coordinator checks how much of the prefix is already available locally, across the relevant cache groups.
- Remove skipped history. Some lifecycles no longer need old blocks, such as sliding-window pages outside the window, so the coordinator lets managers release those references.
- Free a request. When a request finishes or is preempted, the coordinator releases its block references in an order chosen to preserve more useful prefix blocks longer.
- Compute common prefix blocks across running requests. This is used by scheduler/runtime metadata paths that can exploit a shared prefix among active requests.
- Start a new scheduling step for managers with step-local state. Some managers track temporary per-step information, so the coordinator gives them a reset hook at the beginning of each scheduler iteration.

The coordinator hides whether a request has one list of blocks or several manager-specific lists.

## SingleTypeKVCacheManager

`SingleTypeKVCacheManager` in `vllm/v1/core/single_type_kv_cache_manager.py` is the per-lifecycle manager abstraction used inside coordinators.

It owns the request-local block state for one cache lifecycle. The main structure is `req_to_blocks`, a mapping from request ID to that request's ordered `KVCacheBlock` list for this manager. This is the scheduler-side source of truth for which physical block IDs a request currently owns in that lifecycle.

The manager also stores the lifecycle's effective `block_size`, the `kv_cache_spec`, the shared `BlockPool`, the `kv_cache_group_id`, the shared null block, and `num_cached_block`, which tracks how many of a running request's blocks have already been committed into prefix cache.

Its instance methods answer practical scheduler questions: how many more blocks are needed, which cached blocks can be attached, how to allocate new blocks, which old blocks can be skipped, when blocks should be committed to prefix cache, and how to free a request's block list.

Its class/static methods let a coordinator calculate prefix hits without constructing a temporary manager instance for every query. This matters for hybrid coordination, where the coordinator may ask multiple manager classes whether the same candidate prefix length is representable.

Manager selection is driven by cache-spec registration and `get_manager_for_kv_cache_spec(...)`.

Subclasses override the lifecycle-specific parts. Full attention keeps a contiguous prefix; sliding window can skip old pages; chunked local attention skips completed chunks; Mamba tracks recurrent state pages instead of ordinary per-token K/V blocks.

## FullAttentionManager

`FullAttentionManager` is the simplest lifecycle: every future token can attend to every earlier token, so the request must retain all KV blocks from token zero through the current sequence tail. There are no skipped middle regions and no window-based reclamation while the request is alive.

For `block_size = 16`, a request with 40 tokens needs three logical blocks:

```text
block 0: tokens 0-15
block 1: tokens 16-31
block 2: tokens 32-39  partial tail
```

The manager's request state is therefore just one ordered block list for the contiguous prefix.

Allocation count is approximately:

```text
ceil(required_tokens / block_size) - currently_attached_blocks
```

For example, if the request needs 40 tokens and already has two blocks attached, it needs `ceil(40 / 16) - 2 = 1` more block. Prefix-hit blocks count as already attached once the manager touches them and appends them to the request state.

Prefix lookup is downward-closed: if the first `n` blocks hit, every shorter aligned prefix also hits. That property holds because full attention needs an unbroken prefix from the beginning of the sequence. If blocks `[0, 1, 2]` are cached, then `[0]` and `[0, 1]` are also valid cached prefixes. If block `1` is missing, block `2` cannot be used by itself because attention for later tokens needs the complete preceding context.

Common-prefix calculation compares shared leading block identities across running requests. If two active requests start with the same physical cached blocks, the scheduler/metadata path can treat that shared leading region as a common prefix.

Freeing reverses logical block order so deeper tail blocks are returned before shallow prefix blocks. For `[block0, block1, block2]`, the manager frees `[block2, block1, block0]`. This makes later, less broadly reusable blocks become eviction candidates before early prefix blocks when all else is equal.

## SlidingWindowManager

`SlidingWindowManager` handles layers where each query can only attend to a bounded recent window instead of the entire prefix. Once all future queries are too far away to attend to old tokens, those old KV pages are no longer useful for this layer.

Example with `sliding_window = 4` and the next token about to be computed after token `6`:

```text
tokens:        0  1  2  3  4  5  6  7
computed:      |-------- 0..6 --------|
next token:                         7
visible window for token 7:          4  5  6  7
skipped for this layer: 0  1  2  3
```

The source method `get_num_skipped_tokens(...)` in `vllm/v1/core/single_type_kv_cache_manager.py` computes this as:

```text
max(0, num_computed_tokens - sliding_window + 1)
```

The manager uses that skipped-token count to decide which logical block positions are no longer reachable. Those positions can be replaced with the null block, and their real physical block references can be released back to the pool before new allocation. This is different from full attention, where old prefix blocks remain needed until the request finishes.

Prefix hits are more complex than full attention. Full attention needs an unbroken prefix from block zero. Sliding-window attention may only need a contiguous run of cached blocks around the current window boundary. The implementation searches for a usable contiguous run and fills irrelevant earlier positions with null blocks so worker block tables still have the expected logical shape.

The `reachable_block_mask(...)` method controls which newly full blocks should be inserted into prefix cache. Blocks that can never serve a valid future sliding-window hit are masked out, so vLLM does not spend prefix-cache map entries on unreachable pages.

Freeing also differs from full attention. `SlidingWindowManager.free(...)` separates cached and uncached blocks. Cached blocks are appended normally to the free queue block in block pool so they can remain prefix-cache candidates; uncached blocks are prepended so they are reused sooner, because they do not provide prefix-cache value.

Common-prefix calculation returns `0` for sliding-window layers today because old prefix positions are often null blocks rather than meaningful shared full-prefix blocks. This avoids treating skipped sliding-window history like a normal full-attention common prefix.

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

Sink attention is used for attention-sink style models where the first few tokens are kept visible as stable anchor tokens, while the rest of the sequence may use a bounded recent context. It is "full-like" for the sink prefix and "window-like" for the later rolling context.

`SinkFullAttentionSpec` in `vllm/v1/kv_cache_interface.py` carries `sink_len`, the number of initial tokens that should be permanently retained. `SinkFullAttentionManager` in `vllm/v1/core/single_type_kv_cache_manager.py` requires `sink_len` to be positive and divisible by `block_size`, then reserves:

```text
num_sink_blocks = sink_len / block_size
```

from the block pool during manager initialization.

The manager derives reachability from both `sink_len` and window semantics. The initial sink blocks are preserved because later tokens may still attend to them, while non-sink history outside the active recent window can be treated more like sliding-window history. Prefix-cache behavior must preserve hash-compatible sink blocks because those initial blocks are shared/reusable in the same way ordinary full-attention prefix blocks are.

## HybridKVCacheCoordinator

`HybridKVCacheCoordinator` in `vllm/v1/core/kv_cache_coordinator.py` is used when a model has more than one KV-cache group, such as full attention plus sliding-window attention, or attention plus another lifecycle. In that case, one request may need multiple block lists, one per group, and a prefix-cache hit is only usable if every required group can represent a consistent computed prefix.

The coordinator first groups KV-cache groups that have the same spec and manager class. That lets it ask one manager class about several equivalent group IDs at once. It also sorts full-attention groups first because full attention gives a strong initial upper bound for the prefix hit length.

The core lookup method is `HybridKVCacheCoordinator.find_longest_cache_hit(...)`. It starts with:

```text
hit_length = max_cache_hit_length
```

Then it checks each spec group against that candidate length. Each group can either accept the candidate or return a shorter hit. If any group returns a shorter hit, the coordinator lowers `hit_length` and repeats the checks. This is a fixed-point loop: the candidate length only moves downward, so the process eventually stabilizes.

Why this is needed:

```text
full-attention group says:      128 tokens cached
sliding-window group says:       96 tokens usable
chunked-local group says:        64 tokens aligned/usable

final hybrid hit length:         64 tokens
```

The final answer must be the length that all groups can support together. Returning 128 would be wrong because some non-full group cannot provide the required cache state for that much prefix.

Full-attention hits are special because they are downward-closed. If full attention found cached blocks for 128 tokens, then 64 tokens is automatically valid too. So after a later group shrinks the candidate, the coordinator can truncate the already-found full-attention block list instead of redoing full-attention lookup.

Groups with a larger physical block size may need `BlockHashListWithBlockSize` to view the request hash chain at their own block granularity. Groups with sliding-window or chunked-local semantics may return shorter results because they require reachable/window-aligned contiguous runs rather than arbitrary cached blocks.

For EAGLE/MTP draft groups, lookup can temporarily ask for one extra block and then drop the final block. This matches draft-model semantics where the group may need to peek one block past an aligned boundary but should not report that lookahead block as accepted computed context.

`cache_blocks(...)` also aligns committed token counts to `scheduler_block_size` in this coordinator. That keeps prefix-cache publication consistent with the hit lengths that hybrid lookup can later return.

The returned token hit length is therefore the longest prefix that every required group can represent consistently, along with one block list per original KV-cache group.

## Scheduler Construction

`Scheduler.__init__(...)` in `vllm/v1/core/sched/scheduler.py` creates `KVCacheManager` from resolved `KVCacheConfig`, scheduler/hash block sizes, model length, scheduler token budget, caching flags, and CP sizes.

If a KV connector is configured, `Scheduler.__init__(...)` creates a scheduler-role connector with `KVConnectorFactory.create_connector(..., role=KVConnectorRole.SCHEDULER, ...)`. Each worker has its own worker-role connector; the scheduler-role connector is the control-plane side that participates in admission, prefix lookup, request-finish decisions, and metadata construction.

The scheduler binds the manager's `BlockPool` only after `KVCacheManager` is constructed:

```text
self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)
```

That ordering matters because the `BlockPool` is created inside the cache manager/coordinator, not by the connector. Once bound, connector implementations can inspect local GPU prefix-cache state, coordinate which block IDs correspond to remote/offloaded KV, and decide whether finishing a request should immediately free blocks or delay freeing while an asynchronous save/send still owns them.

This binding is especially important for offload/disaggregated connectors. They need to reason about the same scheduler block IDs as the cache manager, so connector metadata and block-pool ownership stay consistent when loading, saving, evicting invalid pages, or taking over temporary responsibility for blocks after request finish.

## Beginning A Scheduler Step

`KVCacheManager.new_step_starts()` in `vllm/v1/core/kv_cache_manager.py` is called at the beginning of each scheduler iteration. It forwards through the coordinator to every `SingleTypeKVCacheManager`.

Most managers do nothing here. The method is a reset hook for managers that keep temporary bookkeeping valid only within one scheduler step. For example, one Mamba path tracks `cached_blocks_this_step` so it can avoid treating blocks cached earlier in the same scheduling iteration as ordinary reusable prefix hits. `new_step_starts()` clears that temporary set before the next scheduling pass.

So "step-local state" means manager metadata that should not survive across scheduler iterations. It is not clearing KV cache tensors or request block ownership.

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
- `full_sequence_must_fit`: require the KV cache to have enough free capacity for the request's full current sequence, not just the tokens scheduled in this step. In the scheduler, this is passed from `self.scheduler_reserve_full_isl` in `vllm/v1/core/sched/scheduler.py`.
- `reserved_blocks`: number of currently free KV blocks that this allocation is not allowed to consume. In `Scheduler.schedule(...)` in `vllm/v1/core/sched/scheduler.py`, this is nonzero for async KV-connector loads and is computed by `_inflight_prefill_reserved_blocks(...)`.

## Allocation Regions

The logical request is divided into:

```text
[already computed][new local cache hit][external cache hit][new compute][lookahead]
```

The regions mean:

- `already computed`: token positions `[0, request.num_computed_tokens)` for this request at the start of this scheduler step. In `KVCacheManager.allocate_slots(...)` in `vllm/v1/core/kv_cache_manager.py`, this is the `comp` region and is literally `request.num_computed_tokens`.
- `new local cache hit`: additional prefix tokens found in the local GPU prefix cache during this step. These tokens were computed earlier by some previous request or previous execution, committed into prefix cache, and are now being attached to the current request without recomputing them.
- `external cache hit`: prefix tokens reported by a KV connector as available remotely/offloaded and needing local destination slots.
- `new compute`: tokens the model will actually execute in this step.
- `lookahead`: extra speculative/draft-token slots reserved beyond the current accepted computation.

Example: suppose Request B has a 100-token prompt and chunked prefill scheduled only tokens `0..31` in an earlier scheduler step. Before the next scheduler step, `request.num_computed_tokens = 32`, so tokens `0..31` are the `already computed` region for Request B. If prefix lookup now finds cached blocks for tokens `32..63`, those are `new local cache hit`. If no cache hit exists for tokens `64..95`, those become `new compute` when the scheduler chooses to run them.

The first regions may already have physical blocks or may need blocks allocated as transfer destinations.

New compute and lookahead always need reachable writable slots unless a specialized manager skips storage.

## Allocation Stage One: Reclaim And Admission

The coordinator first removes blocks no longer reachable under sliding/chunked/Mamba policy.

It then asks each manager how many additional pool blocks are required.

If `full_sequence_must_fit` is true, `KVCacheManager.allocate_slots(...)` in `vllm/v1/core/kv_cache_manager.py` checks whether the full request sequence can fit before allocating anything. It computes `full_num_tokens = min(request.num_tokens, max_model_len)` and asks the coordinator how many blocks would be needed for that whole sequence after accounting for prefix-cache hits and lifecycle rules such as sliding window.

Without this flag, chunked prefill can admit a request because the next small chunk fits even though the remaining prompt would not fit later. With this flag, a 4,000-token prompt scheduled in 512-token chunks is admitted only if the KV cache can hold the relevant blocks for the 4,000-token request, not merely the first 512-token chunk. This is stricter, but it avoids over-admitting requests that will stall or force predictable preemption in later chunks.

`reserved_blocks` protects requests that were already admitted and still need more KV blocks to finish their prefill. The scheduler tracks those requests in `self._inflight_prefills` in `vllm/v1/core/sched/scheduler.py`. For each one, `_request_remaining_blocks(...)` estimates how many more blocks it still needs for its full sequence, and `_inflight_prefill_reserved_blocks(...)` sums those estimates.

This matters for async KV connector loads because an async load can allocate destination GPU blocks and then wait for remote KV bytes without making model-forward progress. While it waits, those blocks are held and are not easy to preempt safely. So `KVCacheManager.allocate_slots(...)` in `vllm/v1/core/kv_cache_manager.py` checks capacity using:

```text
available_blocks = block_pool.get_num_free_blocks() - reserved_blocks
```

Example: if the pool has 100 free blocks but already-admitted prefills still need 30 blocks to finish, an async KV load is admitted as if only 70 blocks are available. This prevents the async load from consuming capacity that those older in-flight prefills are relying on.

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

If no delay is required, `KVCacheManager.free(request)` in `vllm/v1/core/kv_cache_manager.py` immediately releases request references. "Free" here means the request no longer owns its KV-cache block IDs. The manager forwards to the coordinator/per-group managers, decrements block reference counts, and returns blocks with `ref_cnt == 0` to the free queue. The GPU bytes are not necessarily cleared; full hashed blocks may remain reusable as prefix-cache entries until they are evicted or overwritten.

Encoder cache is released through its separate manager.

## Connector Load Completion

Worker `KVConnectorOutput.finished_recving` tells the scheduler that remote K/V is now valid in allocated GPU blocks.

The scheduler updates request state, commits eligible loaded blocks, and transitions the request from waiting-for-KV to runnable computation.

Invalid block IDs from worker load errors are evicted from prefix hash lookup before recompute or failure policy proceeds.

## Reset And Shutdown

Scheduler prefix reset first requires local blocks to be unreferenced.

Optional connector reset clears external cache state if supported.

At shutdown, scheduler-role connectors and cache-related background services are terminated after pending lifecycle handling.
