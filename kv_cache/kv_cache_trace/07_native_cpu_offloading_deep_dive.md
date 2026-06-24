# Native CPU KV Offloading Deep Dive

Native KV offloading extends effective cache capacity by retaining reusable cache pages in CPU memory and loading them back into scheduler-allocated GPU pages on demand.

The primary implementation is `OffloadingConnector`; `SimpleCPUOffloadConnector` is an alternate implementation selected by an environment flag.

## Configuration Resolution

`VllmConfig._post_init_kv_transfer_config(...)` in `vllm/config/vllm.py` translates top-level cache offload settings into connector configuration.

Offloading activates only when `cache_config.kv_offloading_size` is non-null.

For `kv_offloading_backend = native`:

- Default connector is `OffloadingConnector`.
- `VLLM_USE_SIMPLE_KV_OFFLOAD` selects `SimpleCPUOffloadConnector`.
- `cpu_bytes_to_use = kv_offloading_size * 2^30` is placed in connector extra config.
- Connector role becomes `kv_both`.

For `kv_offloading_backend = lmcache`, vLLM selects `LMCacheMPConnector`; capacity is managed by the external LMCache service rather than passed as native CPU bytes.

The configured offload size is server-wide across TP/world ranks, not independently allocated in full on every rank.

## Main Component Stack

The primary native path is layered as:

```text
OffloadingConnector
  scheduler role -> OffloadingConnectorScheduler -> OffloadingManager
  worker role    -> OffloadingConnectorWorker -> OffloadingWorker -> OffloadingHandler
  CPU medium     -> CPUOffloadingSpec -> CPUOffloadingManager + CpuGpuOffloadingHandlers
```

This separation allows another medium, such as filesystem or object storage tiering, to implement the same scheduler/worker transfer abstractions.

## OffloadingSpecFactory

`OffloadingSpecFactory` in `vllm/v1/kv_offload/factory.py` selects an `OffloadingSpec` from configuration.

An offloading spec owns:

- Resolved `VllmConfig` and `KVCacheConfig`.
- Block-size relationship between GPU scheduler blocks and offload blocks.
- Scheduler-side `OffloadingManager` construction.
- Worker-side transfer-handler construction.
- Supported source/destination `LoadStoreSpec` type pairs.

`CPUOffloadingSpec` lives in `vllm/v1/kv_offload/cpu/spec.py`.

## Offload Keys

`OffloadKey` helpers are defined in `vllm/v1/kv_offload/base.py`.

An offload key combines:

```text
(prefix block hash, cache group index)
```

The same token prefix needs separate stored data per cache group.

Offload keys intentionally follow prefix hashes rather than transient GPU block IDs because GPU pages are recycled.

## Offload Block Size Factor

The offload subsystem can combine or split scheduler blocks using a `block_size_factor` derived by the offloading spec.

Transfer specs therefore describe offload-level blocks and the corresponding GPU block ID sequences.

The factor must preserve complete cache-group data and hash alignment.

## CPU Capacity Calculation

`CPUOffloadingSpec` derives total bytes consumed by one offloaded block across workers.

Conceptually:

```text
gpu_bytes_per_block_per_worker = sum(worker KVCacheTensor sizes) / gpu_num_blocks
global_bytes_per_block = gpu_bytes_per_block_per_worker * world_size
offload_bytes_per_block = global_bytes_per_block * block_size_factor
num_cpu_blocks = cpu_bytes_to_use / aligned_offload_bytes_per_block
```

The per-worker CPU page is `offload_bytes_per_block / world_size`.

Alignment can add padding to the global offload record.

The global conceptual layout is:

```text
[worker0 block data][worker1 block data]...[workerN block data][optional padding]
```

Each worker physically handles its local slice.

## Required Layout

`OffloadingConnector.get_required_kvcache_layout(...)` returns `HND`.

This layout requirement ensures the CPU/GPU transfer handlers can interpret page bytes in the expected order.

The connector also sets `prefer_cross_layer_blocks = True`, allowing one transfer to cover all compatible layers for a block when backend stride order supports it.

## Canonical Cache Representation

`OffloadingConnectorWorker.register_kv_caches(...)` converts backend-specific tensors into a canonical opaque-byte representation.

`CanonicalKVCacheTensor` contains a tensor viewed as:

```text
[num_blocks, page_size_bytes]
```

`CanonicalKVCacheRef` identifies a canonical tensor index and the unpadded bytes belonging to a layer/group.

`CanonicalKVCaches` contains canonical tensors plus group-to-data-reference lists.

This representation lets transfer handlers copy pages without understanding K/V head dimensions or quantization.

## Canonicalizing Attention Tensors

For an `AttentionSpec`, the worker obtains the layer tensor's untyped storage and creates an int8 view `[num_blocks, page_size_bytes]`.

It records both padded page bytes and `real_page_size_bytes` so transfers can distinguish reserved stride from meaningful data.

The layer tensor must start at storage offset zero for this canonicalization path.

## Canonicalizing Mamba State

Mamba layers expose a list of strided state tensors sharing one raw allocation.

The worker reconstructs the raw page view from the first state tensor's untyped storage.

The canonical page transfers all state components and alignment padding together.

## Shared Raw Tensors

`KVCacheTensor.shared_by` can make several layer names refer to one raw tensor.

Registration verifies that shared layers have identical tensor counts, data pointers, and strides before selecting one canonical representative.

Group references then list each logical layer's meaningful page segment.

## Cross-Layer Cache Registration

For uniform cross-layer allocation, `register_cross_layers_kv_cache(...)` verifies that physical block dimension is first in the layer-aware backend stride order.

It then exposes one canonical page with size:

```text
group_page_size * num_layers
```

One transfer job can copy all layer data for a block contiguously.

## Scheduler Offload Configuration

`SchedulerOffloadConfig.from_spec(...)` in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` converts cache groups into transfer planning parameters.

Per-group config includes hash/block relationships, data references, and manager behavior needed to map request prefix hashes to GPU/offload blocks.

## RequestOffloadState

The offload scheduler creates request state on `on_new_request(...)`.

State includes:

- Request and per-group prefix/offload keys.
- Number of externally hit blocks.
- GPU block ID groups allocated by the scheduler.
- Stored progress and pending transfer jobs.
- Per-medium request context from `OffloadingManager`.

`update_offload_keys()` follows request hash growth as new full token blocks become available.

`update_block_id_groups(...)` binds keys to actual scheduler allocations.

## External Lookup

`OffloadingConnectorScheduler.get_num_new_matched_tokens(...)` searches each required offload group for the longest available prefix beyond local GPU hits.

For full attention, `_maximal_prefix_lookup(...)` stops at the first missing or in-flight key.

Sliding-window lookup follows manager-specific reachability rather than assuming every middle page is required.

`OffloadingManager.lookup(...)` returns:

- `True` when the key exists and is ready.
- `False` when absent.
- `None` when a matching store is still in flight and lookup should retry later.

## CPUOffloadingManager

`CPUOffloadingManager` is defined in `vllm/v1/kv_offload/cpu/manager.py`.

It owns:

- A finite CPU block ID space.
- Lazy allocation/reuse free list.
- LRU or ARC cache policy.
- Optional block-access frequency tracker.
- Optional offloading events.

`BlockStatus` stores CPU block ID, readiness, and load reference count.

## Store Threshold

`store_threshold` can require an offload key to appear in lookup a minimum number of times before consuming CPU capacity.

Values below two effectively disable filtering.

The tracker uses a bounded LRU table controlled by `max_tracker_size`.

This avoids spending bandwidth and CPU memory on prefixes unlikely to be reused.

## LRU Policy

`LRUCachePolicy` in `vllm/v1/kv_offload/cpu/policies/lru.py` keeps keys in recency order.

Lookup retrieves block status, `touch(...)` moves keys toward most-recent position, and eviction removes oldest unprotected, unreferenced entries.

## ARC Policy

`ARCCachePolicy` in `vllm/v1/kv_offload/cpu/policies/arc.py` adapts between recency and frequency using ARC-style resident and ghost lists.

It can retain frequently reused prefixes that pure LRU would evict during scans.

Both policies must avoid evicting blocks with active load references or keys protected by the current store request.

## Preparing A Load

`CPUOffloadingManager.prepare_load(keys, req_context)`:

1. Resolves every ready key to a CPU `BlockStatus`.
2. Increments each block's load reference count so it cannot be evicted.
3. Returns `CPULoadStoreSpec` containing CPU block IDs.

After worker transfer completion, `complete_load(...)` decrements those references.

## Preparing A Store

`prepare_store(...)`:

1. Applies store-frequency filtering.
2. Removes keys already present.
3. Computes additional CPU blocks required.
4. Evicts unprotected policy entries if needed.
5. Allocates fresh or recycled CPU block IDs.
6. Inserts not-ready `BlockStatus` entries.
7. Returns keys, CPU block IDs, and evicted keys.

Returning `None` means capacity cannot be obtained without evicting protected/in-use entries.

## Store Completion

On success, `complete_store(...)` marks pending blocks ready by transitioning their status to an unreferenced resident state.

On failure, it removes pending policy entries and returns their CPU IDs to the free list.

Events are emitted only for successfully stored or removed keys.

## Transfer Jobs

`TransferJob` in `offloading/common.py` combines request ID with a `TransferSpec`.

`OffloadingConnectorMetadata` contains scheduler-assigned `load_jobs`, `store_jobs`, and optional `jobs_to_flush`.

Job IDs provide stable correlation between scheduler planning and completion reports from every worker.

## Worker Completion Aggregation

Each worker reports completed IDs through `OffloadingWorkerMetadata.completed_jobs` with count one.

`aggregate(...)` sums counts across workers.

The scheduler processes a transfer as globally complete only after the count reaches expected worker count.

This prevents one TP/PP rank from exposing a prefix before all shards are transferred.

## OffloadingConnectorWorker

The worker wrapper owns a generic `OffloadingWorker`, registered handlers, active load jobs, deferred store jobs, completion metadata, and transfer statistics.

`start_kv_transfers(...)` submits load jobs immediately.

`prepare_store_kv(...)` records store jobs but defers submission until the next step.

Deferral starts offload after token-sampling-related transfers, reducing impact on generation latency.

## Preemption Fences

`handle_preemptions(...)` submits deferred stores and waits for `jobs_to_flush` when GPU block reuse could race with an unfinished transfer.

The scheduler therefore cannot overwrite a source page while a worker still reads it for CPU storage.

## Generic OffloadingWorker

`OffloadingWorker` in `vllm/v1/kv_offload/worker/worker.py` dispatches transfers by `(source spec type, destination spec type)`.

Handlers implement:

- `transfer_async(job_id, spec)`.
- `get_finished()`.
- `wait(job_ids)`.
- `shutdown()`.

This decouples scheduler job planning from CPU, filesystem, object-store, or future transfer engines.

## CPU/GPU Handlers

`CpuGpuOffloadingHandlers` and `SingleDirectionOffloadingHandler` live in `vllm/v1/kv_offload/cpu/gpu_worker.py`.

They create one handler for GPU-to-CPU stores and another for CPU-to-GPU loads.

CPU storage is backed by a shared/pinned offload region sized from page bytes and CPU block count.

Pinned memory enables asynchronous device DMA where platform support is available.

## Transfer Descriptors

A transfer contains source block IDs, destination block IDs, canonical cache tensors/data refs, and direction.

`compute_sub_block_ptrs(...)` derives addresses for cases where one offload block contains multiple scheduler/cache sub-blocks.

Descriptor buffers are reused to avoid rebuilding and copying large pointer lists for every layer/block transfer.

## Swap Kernels

`vllm/v1/kv_offload/cpu/swap_blocks_triton.py` implements Triton-assisted block copying for supported layouts/platforms.

The kernel copies opaque page bytes according to source and destination block IDs rather than decoding K/V values.

Fallback selection in `_select_swap_blocks_fn(...)` chooses platform-appropriate transfer operations.

Correctness requires canonical page bytes and block-size factor to match physical storage exactly.

## Streams And Asynchrony

Handlers launch transfers on dedicated streams and associate completion events with job IDs.

`get_finished()` polls completed events without synchronizing all outstanding work.

`wait(job_ids)` fences only jobs whose source/destination lifetime requires completion.

Transfer results report job ID, success, byte count, elapsed time, and direction for stats.

## Store Completion Semantics

Native stores do not use `finished_sending` request IDs for scheduler release.

The scheduler tracks store jobs through aggregated worker metadata and fences GPU block reuse through `jobs_to_flush`.

Loads still return `finished_recving` so waiting requests resume through the common connector scheduler path.

## Reset Behavior

Offload reset clears policy mappings and CPU block allocation state.

The scheduler uses a stale-job threshold and flushes in-flight loads before new stores can reuse CPU block IDs.

This prevents pre-reset completion callbacks from corrupting post-reset block status.

## Events And Metrics

CPU manager emits medium `CPU` store/remove events with offload keys.

`OffloadingConnectorStats` records transfer bytes, time, and direction.

`OffloadPromMetrics` exposes connector metrics through the common Prometheus integration.

## SimpleCPUOffloadConnector

The simple implementation lives in `vllm/v1/simple_kv_offload/` and is wrapped by `simple_cpu_offload_connector.py`.

It divides configured CPU bytes by world size unless `cpu_bytes_to_use_per_rank` explicitly overrides capacity.

It requires prefix caching; without prefix hashes it disables itself.

Scheduler behavior supports eager and `lazy_offload` modes.

Worker behavior uses a `DmaCopyBackend` background copy loop and returns transfer completions through the same connector output contract.

Its reset path is currently unimplemented because pending transfers must be synchronized before GPU prefix state can be safely cleared.

## Eager Versus Lazy Simple Offload

Eager mode prepares stores as blocks become available and aims to preserve reusable prefixes promptly.

Lazy mode estimates target blocks and delays/selects stores to reduce transfer volume and contention.

Both modes maintain scheduler states for loads, stores, pending CPU hits, request cleanup, and cache events.

## Tiering Framework

`vllm/v1/kv_offload/tiering/` extends the generic offload interfaces to multiple storage tiers.

Implementations include filesystem and object-store managers plus an example tier.

Tiering preserves the same key principle: scheduler tracks content by prefix/group key, while worker handlers move opaque canonical page bytes between concrete media.

## LMCache Difference

`kv_offloading_backend = lmcache` routes through `LMCacheMPConnector`, not `CPUOffloadingManager`.

LMCache owns storage capacity, lookup, eviction, and transfer service outside the local native CPU block pool.

vLLM still performs local GPU destination allocation, connector metadata exchange, completion handling, invalidation, and attention runtime exactly through the connector contract.

## End-To-End Offload Example

Assume request A finishes with two full hashed GPU blocks and CPU policy has capacity.

The scheduler builds store keys `(h0, group0)` and `(h1, group0)`, reserves CPU block IDs 4 and 5, and sends a store job mapping GPU blocks to those CPU blocks.

The worker defers the store until the next step, then asynchronously copies canonical page bytes GPU-to-CPU.

After all workers report completion, CPU block statuses become ready and GPU request blocks may be released/reused.

Request B later shares prefix hash `h0`. Local GPU lookup misses because the original page was evicted, but offload lookup finds ready CPU block 4.

The scheduler allocates a new GPU block, builds a load job from CPU block 4 to that GPU block, and marks B waiting for external KV.

Workers copy every rank's/layer's canonical bytes, report completion, and the scheduler commits the destination block hash and resumes B from the external prefix length.
