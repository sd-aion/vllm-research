# Profiling, Grouping, And Physical Allocation

This follows layer cache specs through memory profiling, grouping, final `KVCacheConfig` construction, worker allocation, reshaping, and binding to model layers.

## High-Level Startup Flow

The startup flow is:

1. Model attention/state layers report per-layer `KVCacheSpec` objects.
2. Cache utilities group compatible specs and resolve scheduler block sizes.
3. Workers profile non-cache memory usage and report available cache memory.
4. The engine computes a block count and concrete `KVCacheConfig` for each worker.
5. Workers allocate raw byte tensors according to `KVCacheTensor` entries.
6. Attention backends reshape those bytes into kernel-specific tensors.
7. The model runner binds tensors to static forward-context layer entries.
8. Connector workers register the final cache tensors if transfer is enabled.

## Gathering Layer Specs

`GPUModelRunner.get_kv_cache_spec(...)` in `vllm/v1/worker/gpu_model_runner.py` asks cache-bearing layers for their specs.

For standard attention, `Attention.get_kv_cache_spec(...)` in `vllm/model_executor/layers/attention/attention.py` selects full, sliding-window, TurboQuant, or other attention specs from layer properties and cache dtype.

Mamba and related state-space layers report `MambaSpec` from `vllm/v1/kv_cache_interface.py` through their own layer interfaces.

Encoder-only layers can be represented by `EncoderOnlyAttentionSpec` from `vllm/v1/kv_cache_interface.py` even though they allocate no persistent K/V pages.

## Cache Spec Kind

`get_kv_cache_spec_kind(...)` in `vllm/v1/kv_cache_interface.py` classifies specs for events, metrics, and manager selection.

Specialized subclasses are checked before base classes so sliding-window MLA, sink attention, and ordinary full attention remain distinguishable.

Custom cache specs should register with `KVCacheSpecRegistry` in `vllm/v1/kv_cache_spec_registry.py`; otherwise grouping and manager selection cannot safely infer behavior.

## Resolving Block Sizes

`resolve_kv_cache_block_sizes(...)` in `vllm/v1/core/kv_cache_utils.py` determines scheduler and hash granularity after all groups are known.

The chosen hash block size must divide every scheduler group block size.

When no explicit hash size is supplied, vLLM generally chooses the smallest useful common granularity from resolved groups.

Backend kernel block size is resolved later on the worker because it depends on the set of attention backends attached to each cache group.

## Grouping Goals

KV-cache grouping has two simultaneous goals:

- Layers in one group share one scheduler block table because they require token slots with the same lifecycle.
- Physical cache tensors preserve each layer/backend's page bytes and layout.

A `KVCacheGroupSpec` contains:

- `layer_names`: model layers represented by the group.
- `kv_cache_spec`: merged group-level storage/lifecycle spec.
- `is_eagle_group`: whether the group contains EAGLE/MTP draft attention layers.

`KVCacheGroupSpec` is defined in `vllm/v1/kv_cache_interface.py`.

The scheduler manages one block sequence per request per group.

## Uniform Specs

If all layer specs are the same cache type and block size, `UniformTypeKVCacheSpecs.from_specs(...)` in `vllm/v1/kv_cache_interface.py` can combine them.

Its page size is the sum of member layer page sizes, while its maximum number of pages is the maximum required by a member.

This lets the scheduler treat a set of same-lifecycle layers as one manager layer without losing physical byte accounting.

## Hybrid KV Cache Manager

HMA refers to hybrid KV-cache allocation for models containing different cache lifecycles, such as full attention plus sliding-window attention or attention plus Mamba.

`group_and_unify_kv_cache_specs(...)` and `get_kv_cache_groups(...)` in `vllm/v1/core/kv_cache_utils.py` create groups whose pages can share a common block pool while preserving manager-specific request block sequences.

The grouping code can:

- Unify page sizes by padding smaller specs.
- Bucket layers by page size.
- Approximate a common grouping size when page sizes differ.
- Preserve special DeepSeek V4 grouping requirements.
- Annotate speculative EAGLE groups.
- Fall back to uniform groups when hybrid management is disabled.

## Why Page Sizes Are Unified

The global `BlockPool` from `vllm/v1/core/block_pool.py` hands out integer block IDs such as `17`, `42`, and `99`. In the hybrid manager, multiple KV-cache groups may be backed by one global block pool, so the allocator needs one block ID to correspond to the same amount of reserved memory across those groups.

The problem is that different cache types can naturally have different bytes per page. For example, a full-attention group might need a 1 MB page while a sliding-window group might only need a 512 KB page. If both groups use the same block-ID pool without adjustment, then "allocate block ID 17" is ambiguous for memory accounting: one group treats it as 1 MB of capacity, while another treats it as 512 KB.

vLLM therefore tries to make group page sizes compatible. When possible, it pads the smaller page so both groups reserve the same `page_size_bytes`. In the example above, the sliding-window group may logically use only 512 KB of KV data, but reserve a 1 MB page stride so the allocator can treat every block ID as the same capacity unit.

This padding does not mean the attention kernel sees extra tokens, heads, or head dimensions. The logical KV-cache spec remains the same. `page_size_padded` records the larger reserved page stride used for allocation/accounting, so block ID `b` can consistently mean the `b`th allocation unit in every group using the shared pool.

## KVCacheTensor And KVCacheConfig

`KVCacheTensor` in `vllm/v1/kv_cache_interface.py` describes one raw worker allocation:

- `size`: bytes to allocate.
- `shared_by`: layer names that view the same raw tensor.

`KVCacheConfig` in `vllm/v1/kv_cache_interface.py` contains:

- `num_blocks`: scheduler-visible blocks in the pool.
- `kv_cache_tensors`: raw worker allocation instructions - list of kv cache tensors
- `kv_cache_groups`: scheduler group specifications.

`KVCacheConfig.needs_kv_cache_zeroing` in `vllm/v1/kv_cache_interface.py` is true for Mamba-containing configurations because recurrent state may require defined zero initialization.

## Memory Profiling

Memory profiling starts during v1 engine KV-cache initialization in `LLMEngineCore._initialize_kv_caches(...)` in `vllm/v1/engine/core.py`. That method first asks the executor for each worker's layer-level KV-cache specs through `model_executor.get_kv_cache_specs()`, whose abstract executor entry point is in `vllm/v1/executor/abstract.py`. If the model has KV cache, it then calls `model_executor.determine_available_memory()` to get the per-worker byte budget available for KV-cache allocation.

The executor method is defined generically in `vllm/v1/executor/abstract.py` and dispatches `determine_available_memory` to workers with `collective_rpc(...)`, whose abstract and concrete executor implementations live in `vllm/v1/executor/abstract.py`, `vllm/v1/executor/uniproc_executor.py`, and `vllm/v1/executor/multiproc_executor.py`. In the single-process executor, `UniprocExecutor.determine_available_memory(...)` in `vllm/v1/executor/uniproc_executor.py` returns the one local worker's result in list form so the engine can treat single-worker and multi-worker cases uniformly.

For GPU workers, the real profiling logic is `Worker.determine_available_memory(...)` in `vllm/v1/worker/gpu_worker.py`. If `cache_config.kv_cache_memory_bytes` is set, this method still calls `self.model_runner.profile_run()`, usually `GPUModelRunner.profile_run(...)` in `vllm/v1/worker/gpu_model_runner.py`, so the model is compiled/warmed for the configured batch shape, but it returns the explicit `kv_cache_memory_bytes` value instead of using utilization-derived capacity. In that branch, `gpu_memory_utilization` is intentionally ignored for KV-cache sizing.

If `kv_cache_memory_bytes` is not set, the worker runs a dummy forward pass under `memory_profiling(...)` from `vllm/utils/mem_utils.py` and calls `self.model_runner.profile_run()`. The model runner implementation is `GPUModelRunner.profile_run(...)` in `vllm/v1/worker/gpu_model_runner.py`; this exercises representative model execution without allocating the final KV cache. The worker also records the torch peak allocation increase and may estimate CUDA graph memory with `GPUModelRunner.profile_cudagraph_memory(...)` in `vllm/v1/worker/gpu_model_runner.py` when CUDA graphs are enabled.

The requested memory budget is computed from `gpu_memory_utilization` in `request_memory(...)` in `vllm/v1/worker/utils.py`:

```text
requested_memory = ceil(total_device_memory * gpu_memory_utilization)
```

After profiling, `Worker.determine_available_memory(...)` in `vllm/v1/worker/gpu_worker.py` computes non-KV memory as weights plus profiled torch activation peak plus non-torch memory growth. The KV-cache byte budget is then:

```text
available_kv_cache_memory_bytes =
    requested_memory
    - non_kv_cache_memory
    - applied_cuda_graph_memory_estimate
```

Those per-worker byte budgets flow back to `LLMEngineCore._initialize_kv_caches(...)`, which calls `get_kv_cache_configs(...)` in `vllm/v1/core/kv_cache_utils.py`. That function converts "these workers have N bytes available for KV cache" into concrete worker `KVCacheConfig` objects: it merges KV-cache specs across workers, forms global KV-cache groups, applies hybrid grouping/page-size unification, handles `num_gpu_blocks_override`, optionally auto-fits `max_model_len`, checks memory sufficiency, computes `num_blocks`, and emits the raw `KVCacheTensor` allocation plan plus the scheduler-visible `KVCacheGroupSpec`s.

## Block Count

For uniform page sizes, the basic capacity relationship is:

```text
num_blocks = available_cache_bytes / bytes_consumed_by_one_block_id_across_all_groups
```

`_pool_bytes_per_block(...)` in `vllm/v1/core/kv_cache_utils.py` sums the bytes that one scheduler block ID consumes across cache groups.

`get_num_blocks(...)` in `vllm/v1/core/kv_cache_utils.py` applies available memory, page accounting, and overrides.

`may_override_num_blocks(...)` in `vllm/v1/core/kv_cache_utils.py` applies `num_gpu_blocks_override` when configured.

`num_gpu_blocks_override` is a debug/testing config that forces vLLM to use a fixed number of KV-cache blocks instead of the block count computed from available memory profiling.

The null block is block ID `0` reserved by `BlockPool` in `vllm/v1/core/block_pool.py` as a placeholder, not a normal request-owned KV page. vLLM uses it when a block-table position must exist but should not point at useful cache data, such as skipped prefix blocks in sliding-window or chunked-local cache management. Because block ID `0` is removed from the free list and marked `is_null`, usable request capacity is one block less than the raw `num_blocks` count.

## Memory Sufficiency And Model Length

`check_enough_kv_cache_memory(...)` and `_check_enough_kv_cache_memory(...)` in `vllm/v1/core/kv_cache_utils.py` validate that at least one request of the configured maximum model length can be admitted under the cache specs.

`estimate_max_model_len(...)` and group-aware variants in `vllm/v1/core/kv_cache_utils.py` estimate the largest model length that fits if configured length is too high.

Auto-fit logic accounts for manager-specific maximum memory rather than assuming every layer retains every token.

Sliding-window and chunked-local specs can therefore admit longer sequences than a naive full-attention formula with the same physical bytes.

## Concurrency Estimate

`get_max_concurrency_for_kv_cache_config(...)` in `vllm/v1/core/kv_cache_utils.py` estimates how many maximum-length sequences can coexist from resolved block capacity and per-request page requirements.

This is a capacity estimate, not a scheduler guarantee, because real requests have varying lengths, shared prefixes, windows, connector loads, and speculative lookahead.

## Scheduler Projection Versus Worker Projection

`generate_scheduler_kv_cache_config(...)` in `vllm/v1/core/kv_cache_utils.py` creates the scheduler's group/block view.

`_project_kv_cache_groups_to_worker(...)` in `vllm/v1/core/kv_cache_utils.py` maps global grouping decisions back to each worker's local layers and byte allocations.

TP ranks normally receive equivalent group structures but page sizes reflect local KV-head counts and local cache dtypes.

Pipeline ranks receive only the layer tensors resident on that stage.

## Raw Worker Allocation

`GPUModelRunner._allocate_kv_cache_tensors(...)` in `vllm/v1/worker/gpu_model_runner.py` allocates each `KVCacheTensor` as a flat `torch.int8` tensor of the requested byte count.

It uses zero initialization, which is required for some state caches and provides deterministic initial storage.

Every layer in `shared_by` is initially mapped to the same raw allocation object.

At this stage the tensor has correct bytes but no K/V semantic shape.

## Reshaping Attention Caches

`GPUModelRunner._reshape_kv_cache_tensors(...)` in `vllm/v1/worker/gpu_model_runner.py` iterates attention groups and asks each backend for its cache shape.

The flow is:

1. Compute scheduler `num_blocks` from raw bytes and `page_size_bytes`.
2. Convert scheduler blocks into kernel blocks using scheduler/kernel block-size ratio.
3. Choose `storage_block_size` for compressed MLA or kernel block size otherwise.
4. Call `attn_backend.get_kv_cache_shape(...)`, whose backend contract is declared in `vllm/v1/attention/backend.py`.
5. Obtain backend stride order or use logical identity order.
6. View the raw bytes as the cache spec dtype.
7. Construct contiguous or padded strided storage.
8. Permute back to the backend's logical dimension order while preserving physical strides.

## Shape And Byte Invariant

The worker first allocates KV cache as flat byte storage, then turns that storage into a backend-specific tensor view using the shape from `attn_backend.get_kv_cache_shape(...)`. Without padding, the flat byte count must exactly equal the logical tensor size:

Without page padding:

```text
raw_tensor.numel() bytes == product(kv_cache_shape) * dtype_size
```

With page padding, each block/page reserves more bytes than the logical K/V elements need. For example, a page may contain `16,256` bytes of real K/V data but reserve a `16,384` byte stride for allocator compatibility. In that case, a normal reshape would be wrong because it would pack pages tightly. Instead, `torch.as_strided(...)` makes the block dimension jump by `page_size_bytes / dtype_size`, so the logical tensor skips the padding bytes between pages. The attention backend still sees the same logical K/V dimensions; the padding is reserved storage, not extra tokens, heads, or head dimensions.

## Backend Stride Order

`get_kv_cache_stride_order(...)` is declared on `AttentionBackend` in `vllm/v1/attention/backend.py` and returns a permutation describing desired physical dimension order.

The worker first permutes the target shape into physical order, creates a view, then applies the inverse permutation so kernels see the documented logical shape with backend-required strides.

This lets a backend request block-major, head-major, K/V-major, or transfer-friendly placement without changing its logical API.

## Hybrid Attention And Mamba Layout

When attention and Mamba coexist, `GPUModelRunner._update_hybrid_attention_mamba_layout(...)` in `vllm/v1/worker/gpu_model_runner.py` can alter attention cache strides from a logical K/V-major view toward block-major physical placement.

It uses `get_kv_cache_block_dim(...)`, declared in `vllm/v1/attention/backend.py`, to determine whether the backend's block dimension is zero or one.

This allows one shared block ID to correspond to physically aligned attention and Mamba pages while preserving the attention tensor's logical shape.

## Mamba Tensor Views

For `MambaSpec` in `vllm/v1/kv_cache_interface.py`, one raw page may contain several state tensors with different dtypes and shapes.

The worker creates an `as_strided` view for each state component using the padded page stride and a byte-derived storage offset.

The result associated with a Mamba layer is a list of state tensors rather than one conventional K/V tensor.

## Uniform Cross-Layer Allocation For Connectors

`KVConnectorModelRunnerMixin.use_uniform_kv_cache(...)` in `vllm/v1/worker/kv_connector_model_runner_mixin.py` can select one cross-layer tensor when:

- A connector is active.
- The connector prefers cross-layer blocks.
- There is exactly one compatible attention group.
- The backend stride-order API supports an inserted layer dimension.

A uniform layout means all layers KV caches will share the same underlying tensor, where for a given block number, the respective KV data for all layers will be contiguous. Note that this doesn't mean all the layers share the same KV cache, but the tensor has a layer dimension.

`KVConnectorModelRunnerMixin.allocate_uniform_kv_caches(...)` in `vllm/v1/worker/kv_connector_model_runner_mixin.py` prepends `num_layers` to the backend shape and uses the backend's layer-aware stride order.

The goal is to make all layers' data for physical block `b` contiguous or efficiently addressable for one connector transfer.

Layer-specific cache tensors become views into this cross-layer allocation.

## Cross-Layer KV Sharing

Models such as YOCO can declare that one layer reuses another layer's cache.

`GPUModelRunner.maybe_add_kv_sharing_layers_to_kv_cache_groups(...)` in `vllm/v1/worker/gpu_model_runner.py` adds sharing layers to the target group for metadata purposes.

`GPUModelRunner.initialize_kv_cache_tensors(...)` in `vllm/v1/worker/gpu_model_runner.py` then assigns the exact target tensor object to the sharing layer instead of allocating separate bytes.

The common attention layer skips duplicate cache writes when `kv_sharing_target_layer_name` is set.

## Binding To Layers

`bind_kv_cache(...)` in `vllm/v1/worker/utils.py` is the handoff point where the reshaped per-layer cache tensors become visible to model execution. Its input is `kv_caches`, a dictionary from layer name to the finalized tensor view for that layer.

`bind_kv_cache(...)` 

- Fills the ModelRunner's kv cache list (`runner_kv_caches`) with kv_caches.
- Associates each attention layer in the `forward_context` with its corresponding KV cache in kv_caches.

The function binds the same cache tensors in two places. First, it fills the model runner's `runner_kv_caches` list in layer-index order. This keeps the runner's internal cache collection aligned with model layer order. Second, it writes each tensor onto the corresponding `Attention` object stored in `compilation_config.static_forward_context` by setting `forward_context[layer_name].kv_cache = kv_cache`.

That second binding is the important runtime path. The model's `Attention.forward(...)` call does not receive the KV-cache tensor as a normal Python argument from the model definition. Instead, attention custom ops call `get_attention_context(...)` in `vllm/model_executor/layers/attention/attention.py`. That helper reads the active `ForwardContext`, looks up the layer's attention metadata, retrieves the static attention layer object from `forward_context.no_compile_layers[layer_name]`, then gets the actual cache tensor from `attn_layer.kv_cache`.

`get_attention_context(...)` also retrieves the layer-specific slot mapping from `forward_context.slot_mapping`. So the runtime attention path gets four pieces from shared context rather than the model method signature: attention metadata, the attention layer object, the bound KV-cache tensor, and slot mapping.

This indirection matters for compiled execution. The model graph can keep a stable layer/module interface, while the runner swaps in batch-specific metadata and slot mappings through the forward context and keeps long-lived KV-cache tensors attached to the static attention layer entries.

## Connector Registration

A KV connector is the vLLM component responsible for moving KV cache between this worker and some external or remote storage/transport path. Examples include disaggregated prefill/decode transfer, CPU/offload connectors, LMCache-style connectors, and NIXL/Mooncake-style transport connectors. The base worker/scheduler contract is defined by `KVConnectorBase_V1` in `vllm/distributed/kv_transfer/kv_connector/v1/base.py`.

After cache initialization, an active connector registers the final layer-name-to-tensor mapping. In the normal per-layer layout, the relevant API is `register_kv_caches(kv_caches)`, where `kv_caches` maps each layer name to its finalized KV-cache tensor view. Connector implementations such as NIXL expose this through `register_kv_caches(...)` in `vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py` and then delegate to worker-side registration logic in files such as `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`.

Registration must happen after reshaping because the connector needs the real tensor views that attention will use, not just raw byte allocations. At that point, the tensor has the backend-specific shape, stride order, page padding, layer-sharing aliases, and Mamba/attention layout adjustments already applied. Registering earlier would risk registering the wrong shape or a raw buffer that is not the tensor the runtime attention path actually reads and writes.

There is a separate optimized path for uniform cross-layer layout. If `KVConnectorModelRunnerMixin.allocate_uniform_kv_caches(...)` in `vllm/v1/worker/kv_connector_model_runner_mixin.py` creates one cross-layer tensor, connectors that support it can register that tensor through `register_cross_layers_kv_cache(...)` instead of registering many independent per-layer tensors. That is useful when a connector wants block `b` for all layers to be physically grouped for transfer.

Connectors that pin or register GPU memory require stable physical addresses. If a connector records a pointer or registers a GPU memory range with a transport library, that address must not later move. `VllmConfig._verify_kv_transfer_compat(...)` in `vllm/config/vllm.py` rejects unsafe expandable-segment allocator combinations unless the CuMem allocator provides stable pages.

## Initialization Order For A New Backend

A new attention backend affects startup in this order:

1. The layer creates a cache spec with correct page bytes.
2. Grouping decides which layers share block-table lifecycle.
3. Profiling converts total bytes into `num_blocks`.
4. Backend block-size capabilities determine kernel splitting.
5. Backend shape and stride methods construct the physical tensor view.
6. Metadata builders receive group spec and kernel block size.
7. Cache tensors bind to forward context.
8. Connectors validate and register the final layout.

An error in page bytes can survive until reshape or, worse, produce a view that aliases neighboring pages incorrectly. Page accounting and backend shape must therefore be tested together.
