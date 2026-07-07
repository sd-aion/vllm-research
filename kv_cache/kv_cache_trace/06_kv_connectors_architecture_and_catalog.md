# KV Connector Architecture And Catalog

KV connectors connect scheduler cache decisions to worker-side transfer mechanisms.

They support local CPU offload, disaggregated prefill/decode, peer GPU transfer, remote cache services, shared filesystems, and hidden-state transfer through one scheduler/worker contract.

## Configuration

`KVTransferConfig` is defined in `vllm/config/kv_transfer.py`.

Important fields are:

- `kv_connector`: registered connector class name.
- `engine_id`: unique engine identity used by transfer protocols.
- `kv_role`: `kv_producer`, `kv_consumer`, or `kv_both`.
- `kv_rank` and `kv_parallel_size`: topology values for connectors that use explicit ranks.
- `kv_ip` and `kv_port`: basic endpoint configuration.
- `kv_buffer_device` and `kv_buffer_size`: connector staging location and capacity.
- `kv_connector_extra_config`: connector-specific settings.
- `kv_connector_module_path`: external module containing a custom connector class.
- `enable_permute_local_kv`: experimental local layout permutation.
- `kv_load_failure_policy`: fail the request or recompute after failed load.

Setting a connector requires an explicit KV role.

The KV role describes this engine's data-flow direction in the transfer topology:

- `kv_producer`: this engine produces K/V that another engine can consume. In disaggregated prefill/decode, this is typically the prefill engine: it computes prompt K/V, stores or sends those pages through the connector, and generally does not load remote K/V for the same request path.
- `kv_consumer`: this engine consumes K/V produced elsewhere. In disaggregated prefill/decode, this is typically the decode engine: it asks the connector which prefix K/V is available, allocates local destination blocks, receives/loads the remote pages, and skips recomputing those prefix tokens. Consumer-only paths should not save newly produced K/V back through connector store APIs unless the connector explicitly adds another policy.
- `kv_both`: this engine can both load and save K/V. This is common for single-instance offload or shared-cache setups, where the same vLLM instance may load cached prefix pages from CPU/remote storage and later store newly computed full blocks for future reuse.

In code, `KVTransferConfig.is_kv_producer` in `vllm/config/kv_transfer.py` is true for `kv_producer` and `kv_both`, while `KVTransferConfig.is_kv_consumer` is true for `kv_consumer` and `kv_both`.

Offloading configuration can synthesize `KVTransferConfig` from `CacheConfig`, so users of `--kv-offloading-size` do not need to manually name `OffloadingConnector`.

## Factory And Registration

`KVConnectorFactory` lives in `vllm/distributed/kv_transfer/kv_connector/factory.py`.

The factory registry stores lazy module/class loaders so unused connector dependencies are not imported.

`create_connector(...)` receives `VllmConfig`, a scheduler or worker role, and resolved `KVCacheConfig`. The scheduler/worker role is `KVConnectorRole` from `vllm/distributed/kv_transfer/kv_connector/v1/base.py`, not the same thing as `kv_role`.

An external `kv_connector_module_path` takes priority over the built-in registry.

External connectors must accept the current three-argument constructor including `kv_cache_config`.

## Scheduler And Worker Separation

vLLM creates distinct instances of the same connector class:

- Scheduler-role connector runs in the scheduler process and manages token/block availability, request transfer state, and metadata planning.
- Worker-role connector runs beside model execution and moves actual K/V tensors.

Scheduler code must not directly access GPU tensors.

Worker code must not independently change scheduler block ownership.

The two sides communicate through `KVConnectorMetadata`, `KVConnectorWorkerMetadata`, and `KVConnectorOutput`.

## KVConnectorRole

`KVConnectorRole` in `vllm/distributed/kv_transfer/kv_connector/v1/base.py` identifies whether the constructed object is scheduler-side or worker-side.

This is separate from `kv_role` producer/consumer topology.

- `KVConnectorRole.SCHEDULER`: creates the control-plane connector used by `Scheduler` in `vllm/v1/core/sched/scheduler.py`. This object decides what should happen: how many remote prefix tokens match, which local blocks were allocated for a load, whether a finished request's blocks can be freed immediately, and what transfer metadata should be sent to workers. It should reason in request IDs, token counts, block IDs, and metadata, not directly move layer tensors.
- `KVConnectorRole.WORKER`: creates the data-plane connector used beside model execution. This object receives scheduler-produced metadata and actually loads/saves K/V tensors layer by layer, waits for async transfers, reports finished sends/receives, and builds worker-to-scheduler result metadata.

So `KVConnectorRole` answers "where does this connector object run inside vLLM?" while `kv_role` answers "is this engine producing K/V, consuming K/V, or both?" A scheduler-role connector can still be configured as `kv_producer`, `kv_consumer`, or `kv_both`; same for the worker-role connector. They are two independent axes.

## Metadata Types

The abstract metadata base classes are defined in `vllm/distributed/kv_transfer/kv_connector/v1/base.py`.

`KVConnectorMetadata` in `vllm/distributed/kv_transfer/kv_connector/v1/base.py` is the per-scheduler-step instruction packet from the scheduler connector to the worker connector. The scheduler has already decided which requests are scheduled, which block IDs belong to them, which blocks need remote load, which blocks should be saved, and which old blocks may be overwritten. This metadata tells workers what transfer actions to perform for that step.

Examples: `OffloadingConnectorMetadata` in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py` has `load_jobs`, `store_jobs`, and `jobs_to_flush`, so the worker knows which CPU/GPU block copies to run. `NixlConnectorMetadata` in `vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py` has `reqs_to_recv`, `reqs_to_save`, and `reqs_to_send`, so the worker knows which request pages are incoming, which local pages should be saved, and which completed request data should be sent. `LMCacheConnectorMetadata` in `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py` stores per-request metadata used to look up, load, and store KV through LMCache.

`KVConnectorWorkerMetadata` in `vllm/distributed/kv_transfer/kv_connector/v1/base.py` is the worker's report back to the scheduler connector after it has acted on the step metadata. It usually does not carry the KV tensors themselves. Instead, it reports transfer status, such as which async load/store jobs completed, which receives finished, or which pages failed and should be invalidated or recomputed.

`KVConnectorWorkerMetadata.aggregate(...)` combines reports from multiple workers or TP ranks into one scheduler-visible report. For example, `OffloadingWorkerMetadata` in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py` records completed job IDs per worker; aggregation sums completions so the scheduler can treat a transfer as complete only after all required workers have reported it.

`KVConnectorHandshakeMetadata` in `vllm/distributed/kv_transfer/kv_connector/v1/base.py` is not ordinary per-step scheduling metadata. It is setup information exchanged between connector participants so later transfers know how to communicate. Depending on the connector, this can include engine identity, endpoint information, memory-registration handles, buffer descriptions, or rank/peer information needed before normal load/save jobs can run.

Concrete metadata classes are connector-specific. Examples include `LMCacheConnectorMetadata` in `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py`, `NixlConnectorMetadata` in `vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py`, `OffloadingConnectorMetadata` in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py`, and `P2pNcclConnectorMetadata` in `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py`.

## SupportsHMA

`SupportsHMA` means the connector understands hybrid memory allocation: one request may have multiple KV-cache groups, and each group can have its own block-ID sequence.

Connectors that implement `SupportsHMA` can receive all cache groups through `request_finished_all_groups(...)`.

Without HMA support, a connector generally assumes a single block sequence and cannot safely interpret hybrid models.

`KVConnectorFactory.create_connector(...)` rejects a non-HMA connector when hybrid cache management is enabled.

`MultiConnector` supports HMA only when every configured child supports it.

## Base Worker Methods

`KVConnectorBase_V1` defines these worker-side methods.

### register_kv_caches

`register_kv_caches(kv_caches)` receives final layer-name-to-cache tensors after allocation and reshaping.

Connectors use it to discover device pointers, strides, layer order, page sizes, and canonical per-block views.

### register_cross_layers_kv_cache

`register_cross_layers_kv_cache(kv_cache, attn_backend)` registers a uniform tensor containing multiple layers when cross-layer block layout is enabled.

The connector validates that physical block dimension placement supports contiguous per-block transfer.

### set_host_xfer_buffer_ops

`set_host_xfer_buffer_ops(copy_operation)` installs a generic host-transfer block-copy operation for connectors that use vLLM staging buffers.

### bind_connector_metadata

`bind_connector_metadata(metadata)` attaches scheduler-produced metadata for the current execution step. Here, "bind" means the worker connector stores that metadata object on itself, usually in an internal field, so later connector methods can read the same step instructions without passing the metadata through every method call.

Layer hooks and transfer methods access this bound object until `clear_connector_metadata()` is called.

### handle_preemptions

`handle_preemptions(metadata)` runs before new model execution and gives the worker a chance to save/fence blocks that the scheduler may reuse.

Preemption means the scheduler is stopping or evicting a request from active execution so its KV blocks may be returned to the pool and reused by another request. A fence is a safety barrier: the connector records or waits on enough transfer state to ensure old KV bytes are not overwritten before any required offload/send has captured them.

### start_load_kv

`start_load_kv(forward_context)` starts incoming transfers, preferably asynchronously.

The connector can use forward context to align requests, layers, or metadata with model execution.

Incoming transfer means remote/offloaded KV pages are being copied into the local GPU block IDs that the scheduler already allocated as destinations. `forward_context` is the per-forward shared runtime context that contains the current batch's attention metadata, layer metadata, and connector metadata; the connector uses it to match transfer work to the same request/layer ordering the model is about to execute.

### wait_for_layer_load

`wait_for_layer_load(layer_name)` blocks only when a model layer is about to consume K/V whose transfer has not completed.

Layer-wise connectors can overlap later-layer loads with earlier-layer computation.

Block-oriented connectors that finish all loads before model use can implement this as a no-op.

Layer-wise means the connector treats each model layer's KV tensor separately. For example, while layer 0 is computing, the connector may still be loading KV for layer 10. `wait_for_layer_load("layer.10")` is the point where the worker must stop if layer 10's needed KV has not arrived yet.

### save_kv_layer

`save_kv_layer(layer_name, kv_layer, attn_metadata, ...)` is called when one model layer has finished writing its K/V into the GPU KV cache and the connector may need to copy that layer's K/V somewhere else.

It can enqueue outgoing transfers without waiting for all layers.

Here, `kv_layer` is the actual KV-cache tensor or view for that layer, and `attn_metadata` tells the connector which request slots in that tensor matter. Enqueue means the connector starts or records the copy/send work and lets model execution continue instead of waiting immediately.

### wait_for_save

`wait_for_save()` provides a completion fence when scheduler/request lifetime requires all sends to be safe.

Some offload connectors defer stores and intentionally make this method a no-op.

A save is an outgoing transfer from local GPU KV cache to another place, such as CPU memory, remote storage, or a decode engine. The fence matters when the scheduler wants to free or reuse the source GPU blocks: if the transfer has not safely captured the bytes, reusing the blocks could corrupt the saved KV.

### get_finished

`get_finished(finished_req_ids)` returns request IDs whose sends and receives have completed.

The scheduler uses receives to resume blocked consumers and sends to release delayed producer blocks.

`finished_req_ids` is the set of requests the scheduler believes are done at the request level. The connector may still be doing async transfer cleanup for those requests, so it returns only the IDs whose connector-owned send/receive work is actually complete.

### get_block_ids_with_load_errors

This method returns GPU block IDs whose contents cannot be trusted after load failure.

The scheduler evicts their prefix hashes before applying fail or recompute policy.

A load error means the connector tried to fill local GPU KV blocks from an external source, but some blocks did not arrive correctly. Those physical block IDs may contain missing, partial, or stale KV bytes, so the scheduler must remove them from prefix-cache lookup before it retries, recomputes, or fails the request.

### build_connector_worker_meta

This method returns connector-specific completion metadata for aggregation across workers.

Worker metadata is the worker's report back to the scheduler connector for this step. Aggregation is needed because TP or multi-worker execution can produce one report per worker; the scheduler needs a combined view before deciding whether a transfer is complete globally.

### Statistics, Events, And Shutdown

`get_kv_connector_stats()`, `get_kv_connector_kv_cache_events()`, and `shutdown()` expose transfer observations, cache events, and resource cleanup.

Stats are counters/timings useful for monitoring transfer behavior. KV-cache events describe cache-store/cache-remove style actions for observers. `shutdown()` tears down background threads, network connections, memory registrations, or offload resources owned by the connector.

## Base Scheduler Methods

### bind_gpu_block_pool

`bind_gpu_block_pool(block_pool)` gives scheduler-side connectors controlled access to local block metadata.

Native offload uses it to coordinate GPU prefix blocks and offload state.

The block pool is scheduler-side metadata for physical KV block IDs: which blocks are free, referenced, hashed for prefix cache, or reserved as the null block. Binding it lets a connector make decisions using the same block IDs that the scheduler and KV-cache manager use.

### on_new_request

`on_new_request(request)` initializes connector-specific request context before lookup.

This is called when the scheduler first sees a request on the connector path. A connector can create per-request bookkeeping here, such as lookup keys, remote-cache state, transfer counters, or routing information.

### get_num_new_matched_tokens

This side-effect-free method reports the additional prefix length available externally beyond local GPU hits.

It returns `(count_or_none, load_can_be_async)`.

`None` can represent an unresolved lookup whose result is not ready yet.

External prefix length means tokens whose K/V is not already in this worker's local GPU prefix cache but may exist in a connector backend, such as CPU offload, LMCache, NIXL, Mooncake, or another engine. Side-effect-free means the scheduler may call this more than once during admission decisions, so the method should answer lookup state without consuming it or starting irreversible work.

`load_can_be_async` tells the scheduler whether the request can wait in a remote-KV loading state while the connector fills destination GPU blocks, instead of blocking the scheduling loop until the load completes.

### update_state_after_alloc

After `KVCacheManager` allocates destination/source blocks, `update_state_after_alloc(request, blocks, num_external_tokens)` binds connector state to exact GPU block IDs.

Destination blocks are local GPU block IDs allocated to receive externally loaded K/V. Source blocks are local GPU block IDs whose contents may need to be saved or sent elsewhere. This method matters because lookup happens in token space first, but actual transfer work needs concrete physical block IDs.

### build_connector_meta

`build_connector_meta(scheduler_output)` packages current load/store/preemption work for workers.

It runs after block allocation so transfer specs never refer to hypothetical destinations.

The result is the `KVConnectorMetadata` object placed into `SchedulerOutput` for this engine step. It tells worker-side connectors which requests and block IDs need action during the upcoming model execution step.

### update_connector_output

This method consumes worker completions, stats, events, and worker metadata returned with model-runner output.

The model runner returns connector output after workers have attempted loads, saves, sends, receives, or preemption handling. The scheduler-side connector uses this method to update request transfer state, mark async jobs complete, record failures, and prepare future scheduling decisions.

### request_finished

`request_finished(request, block_ids)` tells the connector that normal execution is complete.

It returns whether local block freeing should be delayed and optional transfer parameters.

HMA connectors implement `request_finished_all_groups(...)` for grouped block-ID tuples.

Delayed freeing means the connector temporarily takes responsibility for the request's KV blocks after the scheduler is otherwise done with the request. This is needed when an async save/send still needs to read those GPU blocks; the scheduler must not return them to the free pool until the connector later reports completion through `get_finished(...)`.

Optional transfer parameters can be returned with request output so another engine or client can locate the produced KV, depending on connector protocol.

### take_events And reset_cache

`take_events()` drains connector-produced cache events.

`reset_cache()` clears external connector cache when implemented.

Cache events are observable records such as "KV stored", "KV removed", or connector-specific cache changes. `reset_cache()` is a coarse invalidation hook for connector-managed storage outside the normal GPU block pool.

## Connector Class-Level Methods

`get_required_kvcache_layout(...)` can force HND, NHD, or another supported backend layout.

`requires_piecewise_for_cudagraph(...)` tells compilation that connector hooks require piecewise graph boundaries.

`get_finished_count()` can override output aggregation count when it differs from worker world size.

`build_kv_connector_stats(...)` and `build_prom_metrics(...)` construct connector-specific metric types.

Handshake setters distribute peer metadata by TP rank or `(PP rank, TP rank)`.

## Scheduler Lifecycle

The scheduler connector lifecycle is mainly inside `Scheduler.schedule(...)` and `Scheduler.update_from_output(...)` in `vllm/v1/core/sched/scheduler.py`.

1. `on_new_request` initializes connector-side state when a request first enters the scheduler path. This is where a connector can create lookup keys, remote-cache handles, or per-request transfer bookkeeping before any cache lookup happens.
2. The scheduler asks `KVCacheManager.get_computed_blocks(...)` for local GPU prefix-cache hits. These are blocks already present in the local vLLM block pool, so they do not require connector transfer.
3. If a connector exists, `get_num_new_matched_tokens(request, num_new_local_computed_tokens)` asks how many additional prefix tokens exist outside local GPU cache. The result is "external hits after local hits," not the total prompt length.
4. `KVCacheManager.allocate_slots(...)` allocates local GPU block IDs. For external hits, these are destination blocks where remote/offloaded K/V will be loaded. For newly computed tokens, these are writable blocks where model execution will store new K/V.
5. `update_state_after_alloc(request, blocks, num_external_tokens)` tells the scheduler-side connector the exact block IDs chosen by the cache manager. This converts an abstract lookup result, such as "load 128 external tokens," into concrete physical destinations.
6. `build_connector_meta(scheduler_output)` creates the `KVConnectorMetadata` attached to `SchedulerOutput`. This is the worker instruction packet for the next engine step: which requests need loads, which blocks should be saved, which preempted blocks need handling, and any connector-specific transfer parameters.
7. Workers execute the step and return `KVConnectorOutput` with completed sends/receives, invalid block IDs, stats, events, and optional worker metadata. This returns through model-runner output, not through the normal token sampler path.
8. `update_connector_output(...)` consumes that worker report and advances scheduler-side connector state. For example, it can mark async loads complete, notice failed block loads, update cache events, or aggregate worker metadata.
9. When a request finishes, `request_finished(request, block_ids)` decides whether the scheduler can free the request's GPU blocks immediately. If an async save/send still needs those blocks, the connector returns a delay signal and later releases them after `get_finished(...)` reports completion.

## Worker Lifecycle

`KVConnectorModelRunnerMixin` in `vllm/v1/worker/kv_connector_model_runner_mixin.py` integrates connectors with model execution. Layer-level hooks are installed through `maybe_transfer_kv_layer(...)` in `vllm/model_executor/layers/attention/kv_transfer_utils.py`.

The worker lifecycle is:

1. The worker receives `scheduler_output.kv_connector_metadata`. `KVConnectorModelRunnerMixin._get_kv_connector_output(...)` binds that metadata onto the worker connector with `bind_connector_metadata(...)`, so all connector methods in this step read the same instruction packet.
2. Before model execution, the runner can call connector preemption handling. In `GPUModelRunner`, `handle_preemptions(...)` is used when scheduler metadata says some old blocks may be overwritten and the connector needs to save or fence them first.
3. `_get_kv_connector_output(...)` calls `start_load_kv(get_forward_context())`. This starts incoming transfers into already allocated GPU destination blocks. Many connectors start this asynchronously so transfer can overlap with later setup or model execution.
4. During attention execution, `maybe_transfer_kv_layer(...)` wraps the attention custom-op boundary. On entry to a layer, it calls `wait_for_layer_load(layer_name)` so the layer does not read K/V before its transfer is complete.
5. The attention layer runs normally: current K/V is written into the KV cache, and attention reads local cached K/V through the backend. Connector integration should not change the attention API; it only ensures the needed cache bytes are present.
6. On exit from the attention layer, `maybe_transfer_kv_layer(...)` calls `save_kv_layer(layer_name, kv_cache, attn_metadata)`. This lets the connector save or send that layer's newly produced K/V if the current request/step requires it.
7. When the model forward exits, `_get_kv_connector_output(...)` calls `wait_for_save()` unless finalization is deferred. This is the point where connectors either wait for required outgoing transfers or intentionally defer them according to connector semantics.
8. The mixin collects connector results into `KVConnectorOutput`: `finished_sending`, `finished_recving`, `invalid_block_ids`, stats, KV-cache events, and `kv_connector_worker_meta`.
9. Finally, the worker clears bound metadata with `clear_connector_metadata()` unless finalization was deferred. Clearing matters because the next scheduler step will bind a new metadata object with different request/block instructions.

## No-Forward Steps

A scheduler step can contain connector transfer work but no model tokens.

`kv_connector_no_forward(...)` installs a forward context, executes connector load/save lifecycle, and returns a `ModelRunnerOutput` containing only connector output.

This prevents asynchronous transfers from stalling merely because no compute batch is launched.

## Uniform Cross-Layer Layout

Connectors can set `prefer_cross_layer_blocks = True`.

When backend stride order supports an inserted layer dimension and the model has one compatible group, the worker allocates one cross-layer tensor.

This makes all layer bytes for physical block `b` addressable as one transfer region instead of issuing one transfer per layer.

The connector must still honor the attention backend's logical views and strides.

## Memory Registration Safety

RDMA and similar connectors may pin/register GPU virtual addresses.

Registering memory means the connector gives a communication library, such as an RDMA/NIXL/Mooncake path, a GPU virtual address range and asks it to create a transfer handle for that memory. After registration, the connector assumes that the same virtual address still refers to the same physical GPU pages when a later remote read/write happens.

PyTorch `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` uses CUDA virtual memory management. That allocator can keep the same virtual address range while remapping it to different physical pages over the engine lifetime. Normal PyTorch kernels are fine with that because they follow the current mapping, but an external registered-memory handle may still point at the old physical pages.

That creates silent-corruption or transfer-failure risk for KV connectors. A connector may register the KV-cache memory, PyTorch may later remap the backing pages, and then the connector's remote transfer can read/write stale or invalid physical memory. In practice, vLLM comments call out failures such as `IBV_WC_REM_ACCESS_ERR` or `NIXL_ERR_REMOTE_DISCONNECT` for inter-node transfers.

`VllmConfig._verify_kv_transfer_compat(...)` in `vllm/config/vllm.py` rejects this combination whenever any KV connector is configured and `expandable_segments:True` is present. The check is conservative because vLLM cannot reliably know whether every in-tree or external connector will pin/register memory.

The exception is the CuMem allocator path. When `model_config.enable_cumem_allocator` is enabled, vLLM routes KV allocations through `CuMemAllocator`'s pool, where expandable segments are disabled around that pool and the KV-cache pages have stable mappings. That is why the compatibility check allows connectors with `expandable_segments:True` only when the CuMem allocator is enabled.

## Failure Handling

`kv_load_failure_policy = fail` terminates a request when required cache cannot be loaded.

`recompute` invalidates failed destination blocks and reschedules tokens for local computation.

Workers must report every suspect block ID; leaving a failed page hash-visible could cause later requests to consume incomplete K/V.

## Registered Connector Catalog

The registry is in `vllm/distributed/kv_transfer/kv_connector/factory.py`.

### OffloadingConnector

`OffloadingConnector` is the primary native offload framework and supports HMA.

It uses scheduler-side offload managers and worker-side handlers selected by an `OffloadingSpec`.

Native CPU offload forces HND layout and prefers cross-layer blocks.

### SimpleCPUOffloadConnector

This is a smaller native CPU offload implementation selected when `VLLM_USE_SIMPLE_KV_OFFLOAD` is enabled.

It requires prefix caching and supports HMA, eager/lazy stores, asynchronous DMA copy, and block-pool-aware scheduler state.

### LMCacheConnectorV1

This adapter integrates vLLM with LMCache storage and event handling in the current process architecture.

It maps vLLM block/token metadata into LMCache lookup, retrieve, and store operations.

### LMCacheMPConnector

This variant uses a standalone LMCache server process and a multi-process protocol.

`--kv-offloading-backend lmcache` selects this connector with `kv_both` role.

### NixlConnector

NIXL supports high-performance peer/storage transfer including RDMA-like transports, memory registration, handshake metadata, and TP mapping.

It is split into connector, scheduler, worker, metadata, stats, and topology helpers and is marked HMA-capable.

### MooncakeConnector

Mooncake uses its transfer engine for distributed KV movement and has scheduler/worker components, transfer metadata, and metrics.

It supports HMA.

### MooncakeStoreConnector

This connector uses Mooncake's distributed store abstraction rather than only point-to-point transfer.

It includes coordinator, scheduler, worker, protocol, data, event, and metrics layers and supports HMA.

### P2pNcclConnector

This connector transfers K/V directly between paired GPU engines with NCCL and a tensor memory pool.

It targets explicit producer/consumer disaggregated-prefill topologies.

### HF3FSKVConnector

HF3FS stores and retrieves KV data through a shared filesystem service.

Its implementation includes client, metadata server, gather/scatter helpers, native utility code, events, and metrics.

### MoRIIOConnector

MoRIIO provides another remote I/O engine with scheduler/worker separation and connector-specific metadata.

It does not advertise `SupportsHMA` in its connector class in this checkout.

### FlexKVConnectorV1

FlexKV integrates an external flexible KV storage/service implementation through the v1 connector contract.

It does not advertise HMA through the marker in this checkout.

### MultiConnector

`MultiConnector` composes several child connectors.

It aggregates metadata, completions, stats, and metrics, and only qualifies for HMA when all children do.

### DecodeBenchConnector

This HMA-capable connector provides controlled transfer behavior for decode benchmarking and connector lifecycle measurement.

### ExampleConnector

This is the reference implementation for basic distributed K/V transfer and custom connector development.

### ExampleHiddenStatesConnector

This HMA-capable example transfers hidden states and alters execution flow, demonstrating that connector metadata can carry more than ordinary K/V pages.

Chunked prefill is explicitly rejected for this connector.

## Choosing A Connector Boundary

A connector should use block IDs and cache specs instead of assuming a fixed tensor shape.

Attention backends remain responsible for logical K/V encoding, while connectors should transfer opaque page bytes unless they explicitly implement layout conversion.

For a new attention backend, connector compatibility requires correct page bytes, stable block-major addressing, declared required layout, and tests that transferred bytes reproduce local attention output.
