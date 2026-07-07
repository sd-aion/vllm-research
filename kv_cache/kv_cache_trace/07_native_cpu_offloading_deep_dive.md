# Native CPU KV Offloading Deep Dive

Native KV offloading extends effective KV-cache capacity by copying reusable KV-cache pages from GPU memory to CPU memory and loading them back into scheduler-allocated GPU pages when another request hits the same prefix. The important mental model is: the scheduler tracks content by prefix hash and cache group, while workers copy opaque page bytes between concrete GPU block IDs and concrete CPU block IDs.

The main implementation is `OffloadingConnector` in `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`. The alternate simpler implementation is `SimpleCPUOffloadConnector` in `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py`, with most logic under `vllm/v1/simple_kv_offload/`.

## Configuration Resolution

`VllmConfig._post_init_kv_transfer_config(...)` in `vllm/config/vllm.py` translates cache-offload CLI/config fields into a `KVTransferConfig`.

Native CPU offloading activates when `cache_config.kv_offloading_size` is set. The size is configured in GiB and converted to bytes as `cpu_bytes_to_use = kv_offloading_size * 2^30`.

For `kv_offloading_backend = native`, vLLM selects `OffloadingConnector` by default. If `VLLM_USE_SIMPLE_KV_OFFLOAD` is enabled, vLLM selects `SimpleCPUOffloadConnector` instead. In both native cases, `kv_role` becomes `kv_both` because the same engine can load old KV from CPU and store newly computed KV to CPU.

For `kv_offloading_backend = lmcache`, vLLM selects an LMCache connector rather than the native CPU manager. LMCache owns the storage capacity, lookup, and eviction policy outside vLLM's native CPU block pool.

The configured offload size is server-wide across worker ranks. `CPUOffloadingSpec` divides the effective storage layout across `world_size` when calculating how many global offloaded blocks fit.

## Main Component Stack

The native offloading stack has a connector layer, a scheduler-planning layer, a worker-transfer layer, and a storage-medium layer.

```text
OffloadingConnector
  scheduler role -> OffloadingConnectorScheduler -> OffloadingManager
  worker role    -> OffloadingConnectorWorker -> OffloadingWorker -> OffloadingHandler
  CPU medium     -> CPUOffloadingSpec -> CPUOffloadingManager + CpuGpuOffloadingHandlers
```

`OffloadingConnector` in `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` is the public KV connector class registered with the connector factory. It implements the common connector methods and delegates to scheduler-side or worker-side objects depending on `KVConnectorRole`.

`OffloadingConnectorScheduler` in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` is the scheduler-side planner. It performs prefix lookup in the offload medium, decides which GPU blocks should load from CPU, decides which newly computed GPU blocks should store to CPU, creates transfer jobs, and processes worker completion reports.

`OffloadingConnectorWorker` in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py` is the worker-side executor. It registers KV-cache tensors, converts them into canonical page views, submits load/store jobs to `OffloadingWorker`, and reports completed job IDs back to the scheduler.

`OffloadingWorker` in `vllm/v1/kv_offload/worker/worker.py` is the generic transfer dispatcher. It does not know CPU policy or prefix hashes; it receives a source spec and destination spec, finds the registered handler for that pair of media, and starts the transfer.

`CPUOffloadingManager` in `vllm/v1/kv_offload/cpu/manager.py` is the CPU storage manager. It owns CPU block IDs, cache policy state, load/store readiness, and eviction.

`CpuGpuOffloadingHandlers` in `vllm/v1/kv_offload/cpu/gpu_worker.py` are worker-side copy handlers for GPU-to-CPU stores and CPU-to-GPU loads.

## OffloadingConnector

`OffloadingConnector` is defined in `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`.

Its constructor calls `OffloadingSpecFactory.create_spec(...)` from `vllm/v1/kv_offload/factory.py`. The spec determines which offload medium is used, how large offloaded blocks are, which scheduler manager is used, and which worker transfer handlers are registered.

When constructed with `KVConnectorRole.SCHEDULER`, `OffloadingConnector` creates an `OffloadingConnectorScheduler`. When constructed with `KVConnectorRole.WORKER`, it creates an `OffloadingConnectorWorker`.

Important connector-level behavior:

- `prefer_cross_layer_blocks` returns `True`, meaning this connector prefers a uniform KV layout where all layers for a physical block can be transferred as one contiguous region when the attention backend supports that layout.
- `get_required_kvcache_layout(...)` returns `"HND"`, forcing a KV layout expected by the native transfer handlers.
- `request_finished(...)` and `request_finished_all_groups(...)` both delegate to `OffloadingConnectorScheduler.request_finished(...)`. HMA is supported because grouped block-ID tuples are handled by the scheduler path.
- `wait_for_layer_load(...)`, `save_kv_layer(...)`, and `wait_for_save()` are no-ops in this connector because the native offloading path is block-job oriented rather than layer-hook oriented. Loads are submitted in `start_load_kv(...)`; stores are deferred through `get_finished(...)` and submitted at the start of a later step.

## OffloadingSpecFactory And OffloadingSpec

`OffloadingSpecFactory` in `vllm/v1/kv_offload/factory.py` maps a spec name to an `OffloadingSpec` class. Built-in registrations include `CPUOffloadingSpec` from `vllm/v1/kv_offload/cpu/spec.py` and `TieringOffloadingSpec` from `vllm/v1/kv_offload/tiering/spec.py`.

`OffloadingSpec` in `vllm/v1/kv_offload/base.py` is the abstract description of an offload medium. It is shared by scheduler and worker code.

An `OffloadingSpec` defines:

- `gpu_block_size`: per-cache-group token block sizes after context-parallel adjustment.
- `hash_block_size`: the scheduler prefix-hash block size used by `Request.block_hashes`.
- `block_size_factor`: how many GPU scheduler blocks are grouped into one offloaded block.
- `offload_prompt_only`: whether only prompt/prefill blocks should be offloaded.
- `get_manager()`: returns the scheduler-side `OffloadingManager`.
- `get_handlers(kv_caches)`: yields worker-side transfer handlers keyed by source and destination medium spec types.

`block_size_factor` matters because offload blocks can be larger than GPU blocks. If GPU block size is 16 tokens and offloaded block size is 64 tokens, then `block_size_factor = 4`, and one offload key maps to four GPU block IDs per group.

## Keys And Content Identity

`OffloadKey` is defined in `vllm/v1/kv_offload/base.py`.

An `OffloadKey` is a packed key:

```text
OffloadKey = prefix_block_hash + group_idx
```

`make_offload_key(block_hash, group_idx)` creates the packed key. `get_offload_block_hash(...)` extracts the prefix hash. `get_offload_group_idx(...)` extracts the cache group index.

The key uses a prefix hash instead of a GPU block ID because GPU block IDs are temporary allocation slots. A GPU block ID can be reused for different requests over time. The prefix hash identifies the actual KV content, and the group index distinguishes KV data for different cache groups.

Example: request A computes prefix block hash `h0` for cache group 0 and stores it to CPU. Later request B has the same prefix hash `h0`. Even if A's original GPU block was freed, B can find CPU data using `OffloadKey(h0, 0)`.

## LoadStoreSpec And TransferSpec

`LoadStoreSpec` in `vllm/v1/kv_offload/base.py` is abstract metadata describing where a transfer source or destination lives.

`GPULoadStoreSpec` in `vllm/v1/kv_offload/base.py` describes GPU block IDs. It stores a flat `block_ids` array plus `group_sizes` and `block_indices` so the worker can interpret grouped cache layouts and offloaded-block alignment.

`CPULoadStoreSpec` in `vllm/v1/kv_offload/cpu/common.py` describes CPU block IDs. It identifies CPU slots managed by `CPUOffloadingManager`.

`TransferSpec` in `vllm/v1/kv_offload/worker/worker.py` is the pair `(src_spec, dst_spec)`. For a load, the source is usually `CPULoadStoreSpec` and the destination is `GPULoadStoreSpec`. For a store, the source is usually `GPULoadStoreSpec` and the destination is `CPULoadStoreSpec`.

`TransferJob` in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py` adds a request ID around a `TransferSpec`. Transfer jobs are keyed by scheduler-assigned job IDs so worker completions can be correlated back to scheduler state.

## CPUOffloadingSpec

`CPUOffloadingSpec` is defined in `vllm/v1/kv_offload/cpu/spec.py`.

It reads `cpu_bytes_to_use` from `kv_connector_extra_config`. Native offload configuration places this there during `VllmConfig._post_init_kv_transfer_config(...)`.

It computes CPU capacity like this:

```text
total_gpu_kv_bytes = sum(kv_cache_tensor.size for kv_cache_tensor in kv_cache_config.kv_cache_tensors)
kv_bytes_per_block_per_worker = total_gpu_kv_bytes / gpu_num_blocks
global_bytes_per_gpu_block = kv_bytes_per_block_per_worker * world_size
global_bytes_per_offloaded_block = global_bytes_per_gpu_block * block_size_factor
num_cpu_blocks = cpu_bytes_to_use / aligned_global_bytes_per_offloaded_block
cpu_page_size_per_worker = global_bytes_per_offloaded_block / world_size
```

The global offload record conceptually contains each worker's shard:

```text
[worker0 block bytes][worker1 block bytes]...[workerN block bytes][optional padding]
```

Each worker physically handles only its local slice. The scheduler still treats the offloaded block as one logical global block because all worker shards must complete before the prefix is safe to reuse.

`CPUOffloadingSpec.get_manager()` builds `CPUOffloadingManager`. It passes the cache policy name, event setting, `store_threshold`, and `max_tracker_size`.

`CPUOffloadingSpec.get_handlers(...)` yields two registered handler pairs:

- `GPULoadStoreSpec -> CPULoadStoreSpec` for GPU-to-CPU stores.
- `CPULoadStoreSpec -> GPULoadStoreSpec` for CPU-to-GPU loads.

## SchedulerOffloadConfig

`SchedulerOffloadConfig` is defined in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`.

It is a scheduler-side normalized view of the offload spec. It contains one `GroupOffloadConfig` per KV-cache group.

`GroupOffloadConfig` stores:

- `group_idx`: cache group index.
- `gpu_block_size`: token count per GPU scheduler block for this group.
- `offloaded_block_size`: token count per offloaded block for this group.
- `hash_block_size_factor`: number of request hash blocks per offloaded block.
- `sliding_window_size_in_blocks`: window size expressed in offloaded blocks, or `None` for full attention.
- `alignment_block_count`: optional optimization for hybrid models where sliding-window groups have smaller blocks than full-attention alignment.

`get_sliding_window_size_in_blocks(...)` in the same file returns `None` for full attention, a positive window length for sliding-window attention, and `1` for Mamba-like state groups.

This config exists because scheduler lookup and store planning must handle full attention, sliding window attention, and Mamba differently.

## RequestOffloadState

`RequestOffloadState` is defined in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`.

It is the scheduler-side offload state for one request. It is created by `OffloadingConnectorScheduler.on_new_request(...)`.

It stores:

- `req`: the `Request` object.
- `req_context`: a `ReqContext` from `vllm/v1/kv_offload/base.py`, containing request ID and optional transfer params.
- `offloading_context`: a `RequestOffloadingContext` returned by the manager, including the offload policy.
- `group_states`: one `RequestGroupState` per cache group.
- `num_locally_computed_tokens`: local GPU cache hit length before external lookup.
- `transfer_jobs`: in-flight job IDs for this request.
- `max_offload_tokens`: optional request-provided cap from `request.kv_transfer_params`.

`RequestGroupState` stores group-local state:

- `offload_keys`: content keys for offloaded blocks in this group.
- `block_ids`: GPU block IDs allocated to the request in this group.
- `next_stored_block_idx`: next offloaded-block index that has not yet been considered for store.
- `num_hit_blocks`: number of offloaded blocks hit when the request started.

`update_offload_keys()` converts `Request.block_hashes` into `OffloadKey` values at offloaded-block granularity. If one offloaded block covers four hash blocks, it takes every fourth hash that represents the completed offload block.

`update_block_id_groups(...)` appends newly allocated scheduler GPU block IDs into the request's group states. This is how store planning later knows which GPU blocks contain the content for a given offload key.

## OffloadingManager Contract

`OffloadingManager` is defined in `vllm/v1/kv_offload/base.py`.

It is the scheduler-side interface for an offload medium. It does not copy bytes. It answers whether content exists, reserves offload blocks for loads/stores, tracks eviction, and marks operations complete.

Important methods:

- `lookup(key, req_context)`: returns `True` if the key is stored and ready, `False` if absent, or `None` if the backend needs the scheduler to retry later.
- `prepare_load(keys, req_context)`: reserves the given stored keys so they cannot be evicted while a worker load is in flight, and returns a source `LoadStoreSpec`.
- `touch(keys, req_context)`: updates recency/frequency metadata for cache policy without necessarily loading bytes.
- `complete_load(keys, req_context)`: releases load protection after workers finish loading.
- `prepare_store(keys, req_context)`: reserves storage for new keys and returns where workers should store them.
- `complete_store(keys, req_context, success)`: marks pending stores ready on success or removes failed pending entries.
- `on_new_request(req_context)`: returns per-request offloading policy/context.
- `reset_cache()`: clears manager state and evicts tracked blocks.

## CPUOffloadingManager

`CPUOffloadingManager` is defined in `vllm/v1/kv_offload/cpu/manager.py`.

It implements `OffloadingManager` for a finite CPU block ID space. CPU block IDs are local scheduler-side identifiers for offload storage slots, analogous to GPU block IDs but for CPU storage.

It owns:

- `_num_blocks`: total CPU offload block capacity.
- `_num_allocated_blocks`: number of CPU block IDs ever allocated so far.
- `_free_list`: recycled CPU block IDs from evicted or failed entries.
- `_policy`: cache policy object, either `LRUCachePolicy` or `ARCCachePolicy`.
- `counts`: optional bounded lookup-frequency tracker for `store_threshold`.
- `events`: optional offloading store/remove events.

`BlockStatus` from `vllm/v1/kv_offload/cpu/policies/base.py` stores CPU block state. It includes a CPU `block_id`, readiness state, and `ref_cnt`. `ref_cnt` protects a CPU block from eviction while a load uses it.

`lookup(...)` updates the optional frequency tracker, checks policy residency, returns `False` for miss, returns `None` for a matching block whose store is still in flight, and returns `True` only when the block is resident and ready.

`prepare_load(...)` looks up every key, asserts each key is ready, increments each `BlockStatus.ref_cnt`, and returns a `CPULoadStoreSpec` containing CPU block IDs.

`complete_load(...)` decrements those `ref_cnt` values after workers report the load job complete.

`prepare_store(...)` filters by `store_threshold`, skips keys already present, evicts unprotected entries if capacity is needed, allocates fresh or recycled CPU block IDs, inserts pending not-ready `BlockStatus` objects, and returns a `PrepareStoreOutput`.

`complete_store(..., success=True)` marks pending blocks ready by setting them resident with `ref_cnt = 0`. On failure, it removes not-ready entries and returns CPU block IDs to the free list.

## Store Threshold And Cache Policies

`store_threshold` is read by `CPUOffloadingSpec.get_manager(...)` from connector extra config and passed to `CPUOffloadingManager`.

When `store_threshold >= 2`, the manager keeps a bounded `OrderedDict` of lookup counts. A key must be observed at least `store_threshold` times before `prepare_store(...)` will consume CPU capacity for it. This avoids storing prefixes that are unlikely to be reused.

`max_tracker_size` bounds that lookup-count table. When full, the oldest tracked key is removed.

`LRUCachePolicy` in `vllm/v1/kv_offload/cpu/policies/lru.py` keeps resident keys in recency order and evicts least-recent unprotected blocks.

`ARCCachePolicy` in `vllm/v1/kv_offload/cpu/policies/arc.py` uses ARC-style resident and ghost lists to adapt between recency and frequency. It can preserve frequently reused prefixes better than plain LRU during scan-like workloads.

Both policies must avoid evicting blocks with active `ref_cnt` and blocks protected by the current store request.

## External Lookup

`OffloadingConnectorScheduler.get_num_new_matched_tokens(...)` is defined in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`.

The scheduler calls this after local GPU prefix-cache lookup. Its input `num_computed_tokens` means tokens already available locally on GPU. The method returns how many more tokens can be loaded from offload beyond that local prefix.

The method does this:

1. Clears previous group block IDs for this lookup attempt.
2. Calls `RequestOffloadState.update_offload_keys()` so current request hashes are available as offload keys.
3. Stores `num_locally_computed_tokens`.
4. Calls `_lookup(...)` to find a consistent hit length across all groups.
5. Updates per-group hit block counts.
6. Touches relevant keys in the manager so cache policy sees them as recently/frequently used.
7. Returns `(num_hit_tokens, bool(num_hit_tokens))`, where the boolean tells the scheduler the load can be async.

`_maximal_prefix_lookup(...)` handles full-attention groups. It scans keys from the start and stops at the first miss. If any lookup returns `None`, it returns `None` so the scheduler retries later.

`_sliding_window_lookup(...)` handles sliding-window groups. It scans from the end and looks for the last run of enough consecutive hit blocks to satisfy the active window. This is different from full attention because old middle blocks may no longer be reachable.

`_lookup(...)` combines full-attention and sliding-window results. A full-attention group may report a longer hit than a sliding-window group can represent, so `_lookup(...)` iterates until the token hit length is valid for every required group.

If `_blocks_being_loaded` contains any key needed for this hit, `_lookup(...)` returns `None`. That prevents duplicate loads of the same offloaded key while another request is already loading it.

## Allocating Destination GPU Blocks

`OffloadingConnectorScheduler.update_state_after_alloc(...)` is defined in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`.

This method runs after `KVCacheManager.allocate_slots(...)` chooses actual GPU block IDs.

Its job is to turn "load N external tokens" into a concrete load job:

1. Determine which GPU blocks are already local and which GPU blocks still need external data.
2. Build `keys_to_load`, the offload keys that should be read from CPU.
3. Build `dst_block_ids`, the physical GPU block IDs where those bytes should be copied.
4. Build a source `LoadStoreSpec` by calling `manager.prepare_load(keys_to_load, req_context)`.
5. Build a destination `GPULoadStoreSpec(dst_block_ids, group_sizes, block_indices)`.
6. Create a scheduler job ID and store a `TransferJob(req_id, (src_spec, dst_spec))`.
7. Track the job in `RequestOffloadState.transfer_jobs` and `_jobs`.
8. Add keys to `_blocks_being_loaded` when prefix caching is enabled.

`group_sizes` tells the worker how many GPU block IDs belong to each cache group in the flat destination list.

`block_indices` tells the worker the logical starting block index for each group. This matters when an offload block is larger than a GPU block and the first GPU block in a transfer is not aligned to an offload-block boundary.

## Store Job Planning

`OffloadingConnectorScheduler._build_store_jobs(...)` is defined in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`.

It decides which newly computed GPU KV blocks should be copied to CPU.

Important filters:

- If `offload_prompt_only` is true, it clamps `num_offloadable_tokens` to `request.num_prompt_tokens`, so decode-generated blocks are not offloaded.
- If `max_offload_tokens` is set in request transfer params, stores are capped to that token count.
- Sliding-window groups skip block IDs equal to `0`, because zero means null/skipped/stale placeholder rather than useful KV data.
- Hybrid sliding-window groups can skip blocks that can never serve a future load hit under full-attention alignment.

For each group, `_build_store_jobs(...)` maps offload keys to GPU block IDs. If `block_size_factor > 1`, it selects all GPU sub-block IDs needed for that offloaded block and records correct `group_sizes` and `block_indices`.

Then it calls `manager.prepare_store(new_offload_keys, req_context)`. That reserves CPU block IDs and returns a destination `CPULoadStoreSpec`.

The scheduler creates a store `TransferJob` whose source is `GPULoadStoreSpec` and destination is the manager-provided CPU spec.

Sliding-window source block IDs are immediately tracked in `_block_id_to_pending_jobs` because sliding-window blocks can be released before the request finishes. Non-sliding-window source blocks are tracked when the request finishes because they remain protected by normal request ownership while the request is alive.

## Connector Metadata

`OffloadingConnectorMetadata` is defined in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py`.

It contains:

- `load_jobs`: scheduler job ID to `TransferJob` for CPU-to-GPU loads.
- `store_jobs`: scheduler job ID to `TransferJob` for GPU-to-CPU stores.
- `jobs_to_flush`: optional job IDs that workers must wait for before GPU block reuse can proceed safely.

`OffloadingConnectorScheduler.build_connector_meta(...)` creates this metadata. It first updates request state from scheduler output, asks the manager to end the schedule step, decides which old jobs must be flushed, builds store jobs, then clears current-batch temporary sets.

Flush jobs are added when:

- a request is preempted and its pending stores must be safe before block reuse.
- a GPU block ID involved in a pending store appears in newly allocated block IDs.
- all tracked requests are finished and there may not be a future step to naturally complete jobs.

## Worker Canonical KV Cache Registration

`OffloadingConnectorWorker.register_kv_caches(...)` is defined in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py`.

It receives the final worker KV-cache tensors after allocation and reshaping. Its job is to expose them as opaque byte pages so transfer handlers do not need to understand attention head dimensions, K/V dimensions, quantization, or Mamba state layout.

The canonical classes are defined in `vllm/v1/kv_offload/base.py`:

- `CanonicalKVCacheTensor`: one tensor viewed with block dimension first and byte-like page storage.
- `CanonicalKVCacheRef`: reference to a canonical tensor plus the meaningful unpadded bytes for one layer/group entry.
- `CanonicalKVCaches`: list of unique canonical tensors plus per-cache-group references.

For attention layers, `register_kv_caches(...)` takes the layer KV tensor's untyped storage and creates an int8 view shaped:

```text
[num_blocks, page_size_bytes]
```

It also records `real_page_size_bytes` separately from padded `page_size_bytes`. The padded page stride may reserve extra bytes for allocator/group compatibility, but only real page bytes are meaningful layer data.

For Mamba layers, the worker reconstructs a raw page view from the first state tensor's untyped storage. Mamba state may be represented as multiple tensors sharing one raw allocation, so canonicalization treats the raw state page as one transferable byte region.

For `KVCacheTensor.shared_by`, the worker verifies that shared logical layers point to the same tensor storage, data pointer, and stride. It then chooses one canonical representative and records refs for every logical layer.

After canonicalization, `_register_handlers(...)` calls `spec.get_handlers(canonical_kv_caches)` and registers each handler with `OffloadingWorker`.

## Uniform Cross-Layer Registration

`OffloadingConnectorWorker.register_cross_layers_kv_cache(...)` is defined in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py`.

This path is used when vLLM allocated a uniform cross-layer KV tensor because `prefer_cross_layer_blocks = True` and the attention backend supports a layer dimension in its stride order.

The worker verifies that the physical block dimension is first in the cross-layer tensor layout. Then it views the raw storage as:

```text
[num_blocks, group_page_size * num_layers]
```

That means one transfer can copy all layer bytes for physical block `b` as a single contiguous region.

## OffloadingConnectorWorker Runtime

`OffloadingConnectorWorker` in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py` owns:

- `worker`: the generic `OffloadingWorker`.
- `_load_jobs`: active load job ID to request ID mapping.
- `_unsubmitted_store_jobs`: store jobs deferred until the next step.
- `_connector_worker_meta`: completed job IDs to report to the scheduler.
- `kv_connector_stats`: transfer byte/time counters.

`start_kv_transfers(metadata)` submits any deferred store jobs first, clears the deferred list, then submits every load job in `metadata.load_jobs`.

`prepare_store_kv(metadata)` does not immediately run stores. It appends store jobs to `_unsubmitted_store_jobs`. This intentionally defers GPU-to-CPU stores to the next step so they do not block token generation latency in the step that produced the KV.

`handle_preemptions(metadata)` submits deferred stores and waits for `metadata.jobs_to_flush` when the scheduler says GPU block reuse could race with pending transfer reads.

`get_finished(finished_req_ids)` polls the generic worker for completed transfer jobs. It records transfer stats, marks completed job IDs in `_connector_worker_meta`, and returns request IDs for completed loads as `finished_recving`. Native stores do not use `finished_sending`; store completion is reported through `OffloadingWorkerMetadata.completed_jobs`.

`build_connector_worker_meta()` returns completed job IDs and resets the local completion metadata object.

## Generic OffloadingWorker And Handlers

`OffloadingWorker` is defined in `vllm/v1/kv_offload/worker/worker.py`.

It is a dispatcher from `(source LoadStoreSpec type, destination LoadStoreSpec type)` to an `OffloadingHandler`.

An `OffloadingHandler` implements:

- `transfer_async(job_id, spec)`: starts a transfer and returns whether it was submitted.
- `get_finished()`: returns completed transfer results.
- `wait(job_ids)`: blocks until specific job IDs finish.
- `shutdown()`: releases handler resources.

For native CPU, `CPUOffloadingSpec.get_handlers(...)` registers two directions:

- GPU pages to CPU pages for stores.
- CPU pages to GPU pages for loads.

The transfer worker therefore does not care whether the medium is CPU, filesystem, object store, or future tier. It only routes specs to handlers.

## CPU/GPU Copy Handlers

`CpuGpuOffloadingHandlers` and `SingleDirectionOffloadingHandler` live in `vllm/v1/kv_offload/cpu/gpu_worker.py`.

They allocate CPU storage for the offload block pool and implement the actual page copies. CPU storage is backed by a shared or pinned offload region sized by CPU page bytes and CPU block count.

Pinned CPU memory allows asynchronous DMA-style GPU/CPU copies when the platform supports it. If pinned memory is unavailable, transfers may still work but can be slower or less asynchronous.

The handler copies opaque canonical pages. It does not interpret key heads, value heads, quantization scales, or tensor semantics. Correctness depends on canonical page views matching the worker's actual KV-cache storage layout.

## Transfer Descriptors And Sub-Blocks

Transfer descriptors contain source block IDs, destination block IDs, canonical cache tensors, canonical data refs, and direction.

`compute_sub_block_ptrs(...)` in the CPU/GPU worker path derives addresses when one offloaded block contains multiple GPU blocks. This is the worker-side counterpart of scheduler `block_size_factor`.

Example: if one CPU offload block stores four GPU blocks, a load job may need only GPU sub-blocks 2 and 3 of that offload block. `block_indices` and descriptor pointer computation let the worker skip the unused leading sub-blocks.

Descriptor buffers are reused to avoid rebuilding large pointer lists for every job.

## Swap Kernels And Copy Path

`vllm/v1/kv_offload/cpu/swap_blocks_triton.py` contains Triton-assisted block-copy kernels for supported platforms/layouts.

The swap path copies page bytes according to source and destination block IDs. It does not decode K/V values.

Fallback selection in the CPU/GPU handler chooses platform-appropriate copy functions when Triton kernels are not suitable.

The key correctness requirement is that the canonical byte page size, block-size factor, source block IDs, destination block IDs, and actual KV-cache storage layout all agree.

## Completion Aggregation

`OffloadingWorkerMetadata` is defined in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py`.

Each worker reports completed transfer job IDs as:

```text
completed_jobs[job_id] = 1
```

`OffloadingWorkerMetadata.aggregate(...)` sums counts across workers. `OffloadingConnectorScheduler.update_connector_output(...)` decrements `TransferJobStatus.pending_count` by those counts.

A job is globally complete when `pending_count == 0`. Only then does the scheduler call `manager.complete_store(...)` or `manager.complete_load(...)`.

This matters under tensor parallelism and multi-worker execution because one rank finishing its local CPU copy is not enough. The offloaded prefix is reusable only after every required worker shard finishes.

## Store Completion And Block Reuse

Native stores do not return request IDs through `finished_sending`. Instead, stores complete through worker metadata job counts.

The scheduler tracks `TransferJobStatus` objects in `OffloadingConnectorScheduler._jobs`. A store job status includes the offload keys and the source GPU block IDs that may need fencing.

`_block_id_to_pending_jobs` maps GPU block IDs to pending store job IDs. If the scheduler later allocates one of those GPU block IDs for another request, `build_connector_meta(...)` adds the corresponding job IDs to `jobs_to_flush`.

The worker sees `jobs_to_flush` in metadata and `OffloadingConnectorWorker.handle_preemptions(...)` calls `worker.wait(...)` for those jobs. This prevents a GPU page from being overwritten while a GPU-to-CPU store is still reading it.

## Reset Behavior

`OffloadingConnectorScheduler.reset_cache(...)` and `CPUOffloadingManager.reset_cache(...)` clear offload scheduler and CPU manager state.

`CPUOffloadingManager.reset_cache(...)` clears policy entries, the CPU free list, and the allocated-block counter.

The scheduler uses `_stale_job_threshold` to ignore completions from jobs created before reset. It also flushes in-flight loads before new stores can reuse CPU block IDs. This prevents a late completion callback from corrupting the post-reset state for a reused CPU block ID.

## Events And Metrics

`CPUOffloadingManager.take_events(...)` emits `OffloadingEvent` records when keys are stored or removed from CPU.

`OffloadingConnectorScheduler.take_events(...)` converts those into KV-cache events such as `BlockStored` and `BlockRemoved` from `vllm/distributed/kv_events.py`.

`OffloadingConnectorStats` in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py` records transfer byte counts, elapsed time, and direction.

`OffloadPromMetrics` in the same file exposes native offload metrics through vLLM's common connector Prometheus integration.

## SimpleCPUOffloadConnector

`SimpleCPUOffloadConnector` is defined in `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py`.

Most scheduler logic lives in `SimpleCPUOffloadScheduler` in `vllm/v1/simple_kv_offload/manager.py`.

Most worker logic lives in `SimpleCPUOffloadWorker` in `vllm/v1/simple_kv_offload/worker.py`.

Metadata classes live in `vllm/v1/simple_kv_offload/metadata.py`.

The simple implementation mirrors GPU KV-cache configuration into a CPU-side KV-cache configuration, uses a CPU `BlockPool`, and copies between GPU and CPU block IDs with a `DmaCopyBackend` from `vllm/v1/simple_kv_offload/copy_backend.py`.

It requires prefix caching because CPU offload lookup is keyed by prefix hashes. Without prefix hashes, it cannot safely identify reusable KV content.

It supports eager and lazy modes. Eager mode schedules CPU stores promptly as reusable blocks appear. Lazy mode estimates target free blocks and delays/selects stores to reduce transfer volume and contention.

Simple CPU offload is useful as a smaller path to understand, but the primary native framework is the generic `OffloadingConnector` plus `OffloadingSpec` stack.

## Tiering Framework

`vllm/v1/kv_offload/tiering/` extends the same offload interfaces to multiple storage tiers.

The key principle is unchanged: scheduler state tracks content by `OffloadKey`, worker handlers copy canonical page bytes between concrete media, and completion metadata tells the scheduler when keys become ready or safe to evict.

Tiering can add filesystem or object-store managers without changing the attention runtime or scheduler's GPU block allocation model.

## End-To-End Example

Assume request A computes a prompt with two full GPU KV blocks in cache group 0. The scheduler has prefix hashes `h0` and `h1`.

When A's blocks become offloadable, `RequestOffloadState.update_offload_keys()` creates:

```text
OffloadKey(h0, group0)
OffloadKey(h1, group0)
```

`_build_store_jobs(...)` asks `CPUOffloadingManager.prepare_store(...)` for CPU locations. Suppose the manager allocates CPU block IDs `4` and `5`.

The scheduler builds a store job:

```text
source:      GPULoadStoreSpec([gpu_block_for_h0, gpu_block_for_h1])
destination: CPULoadStoreSpec([4, 5])
```

The worker later copies canonical page bytes from the GPU blocks to CPU blocks. Every worker reports the job ID in `OffloadingWorkerMetadata.completed_jobs`. When all workers report completion, the scheduler calls `CPUOffloadingManager.complete_store(...)`, and keys `(h0, group0)` and `(h1, group0)` become ready CPU-resident prefix entries.

Later request B has the same prefix hash `h0`. Local GPU prefix lookup misses because A's GPU page was evicted. `get_num_new_matched_tokens(...)` looks up `OffloadKey(h0, group0)` in the CPU manager and gets a ready hit.

The scheduler allocates a new GPU destination block for B and calls `update_state_after_alloc(...)`. That creates a load job:

```text
source:      CPULoadStoreSpec([4])
destination: GPULoadStoreSpec([new_gpu_block_for_B])
```

The worker copies CPU block `4` into B's newly allocated GPU block. After all workers complete, the scheduler calls `CPUOffloadingManager.complete_load(...)`, commits the loaded GPU block into prefix-cache state, and resumes B with that external prefix already computed.
