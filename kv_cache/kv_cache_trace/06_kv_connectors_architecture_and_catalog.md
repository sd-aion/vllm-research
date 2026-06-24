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

Offloading configuration can synthesize `KVTransferConfig` from `CacheConfig`, so users of `--kv-offloading-size` do not need to manually name `OffloadingConnector`.

## Factory And Registration

`KVConnectorFactory` lives in `vllm/distributed/kv_transfer/kv_connector/factory.py`.

The factory registry stores lazy module/class loaders so unused connector dependencies are not imported.

`create_connector(...)` receives `VllmConfig`, a scheduler or worker role, and resolved `KVCacheConfig`.

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

A scheduler-side object can plan either production or consumption depending on engine configuration; a worker-side object executes the corresponding transfers.

## Metadata Types

`KVConnectorMetadata` is scheduler-to-worker step metadata.

It can contain request IDs, source/destination block IDs, transfer parameters, load jobs, store jobs, and preemption fences.

`KVConnectorWorkerMetadata` is worker-to-scheduler metadata and defines `aggregate(...)` for multi-worker/TP result combination.

`KVConnectorHandshakeMetadata` carries persistent connection and memory-registration information exchanged between engines or ranks.

## SupportsHMA

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

`bind_connector_metadata(metadata)` attaches scheduler-produced metadata for the current execution step.

Layer hooks and transfer methods access this bound object until `clear_connector_metadata()` is called.

### handle_preemptions

`handle_preemptions(metadata)` runs before new model execution and gives the worker a chance to save/fence blocks that the scheduler may reuse.

### start_load_kv

`start_load_kv(forward_context)` starts incoming transfers, preferably asynchronously.

The connector can use forward context to align requests, layers, or metadata with model execution.

### wait_for_layer_load

`wait_for_layer_load(layer_name)` blocks only when a model layer is about to consume K/V whose transfer has not completed.

Layer-wise connectors can overlap later-layer loads with earlier-layer computation.

Block-oriented connectors that finish all loads before model use can implement this as a no-op.

### save_kv_layer

`save_kv_layer(layer_name, kv_layer, attn_metadata, ...)` is called from layer-wise transfer hooks after a layer has produced K/V.

It can enqueue outgoing transfers without waiting for all layers.

### wait_for_save

`wait_for_save()` provides a completion fence when scheduler/request lifetime requires all sends to be safe.

Some offload connectors defer stores and intentionally make this method a no-op.

### get_finished

`get_finished(finished_req_ids)` returns request IDs whose sends and receives have completed.

The scheduler uses receives to resume blocked consumers and sends to release delayed producer blocks.

### get_block_ids_with_load_errors

This method returns GPU block IDs whose contents cannot be trusted after load failure.

The scheduler evicts their prefix hashes before applying fail or recompute policy.

### build_connector_worker_meta

This method returns connector-specific completion metadata for aggregation across workers.

### Statistics, Events, And Shutdown

`get_kv_connector_stats()`, `get_kv_connector_kv_cache_events()`, and `shutdown()` expose transfer observations, cache events, and resource cleanup.

## Base Scheduler Methods

### bind_gpu_block_pool

`bind_gpu_block_pool(block_pool)` gives scheduler-side connectors controlled access to local block metadata.

Native offload uses it to coordinate GPU prefix blocks and offload state.

### on_new_request

`on_new_request(request)` initializes connector-specific request context before lookup.

### get_num_new_matched_tokens

This side-effect-free method reports the additional prefix length available externally beyond local GPU hits.

It returns `(count_or_none, load_can_be_async)`.

`None` can represent an unresolved lookup whose result is not ready yet.

### update_state_after_alloc

After `KVCacheManager` allocates destination/source blocks, `update_state_after_alloc(request, blocks, num_external_tokens)` binds connector state to exact GPU block IDs.

### build_connector_meta

`build_connector_meta(scheduler_output)` packages current load/store/preemption work for workers.

It runs after block allocation so transfer specs never refer to hypothetical destinations.

### update_connector_output

This method consumes worker completions, stats, events, and worker metadata returned with model-runner output.

### request_finished

`request_finished(request, block_ids)` tells the connector that normal execution is complete.

It returns whether local block freeing should be delayed and optional transfer parameters.

HMA connectors implement `request_finished_all_groups(...)` for grouped block-ID tuples.

### take_events And reset_cache

`take_events()` drains connector-produced cache events.

`reset_cache()` clears external connector cache when implemented.

## Connector Class-Level Methods

`get_required_kvcache_layout(...)` can force HND, NHD, or another supported backend layout.

`requires_piecewise_for_cudagraph(...)` tells compilation that connector hooks require piecewise graph boundaries.

`get_finished_count()` can override output aggregation count when it differs from worker world size.

`build_kv_connector_stats(...)` and `build_prom_metrics(...)` construct connector-specific metric types.

Handshake setters distribute peer metadata by TP rank or `(PP rank, TP rank)`.

## Scheduler Lifecycle

The scheduler connector lifecycle is:

1. `on_new_request` initializes state.
2. Local prefix cache is queried.
3. `get_num_new_matched_tokens` queries external availability.
4. `KVCacheManager.allocate_slots` allocates local destinations.
5. `update_state_after_alloc` records block IDs.
6. `build_connector_meta` creates worker transfer instructions.
7. Worker output returns completions/errors.
8. `update_connector_output` advances scheduler-side jobs.
9. `request_finished` decides whether block release is immediate or delayed.

## Worker Lifecycle

`KVConnectorModelRunnerMixin` and `vllm/v1/worker/gpu/kv_connector.py` integrate connectors with execution.

The worker lifecycle is:

1. Bind step metadata.
2. Handle preemption fences.
3. Start background loads.
4. Wait at layer boundaries if required.
5. Execute cache updates and attention.
6. Save layer/block data if required.
7. Wait or defer saves according to connector semantics.
8. Collect finished sends/receives, invalid blocks, events, stats, and worker metadata.
9. Clear bound metadata.

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

PyTorch expandable segments can remap those virtual addresses to different physical pages, invalidating registrations.

`VllmConfig._verify_kv_transfer_compat(...)` rejects connectors with `expandable_segments:True` unless the CuMem allocator provides stable pool mappings.

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
