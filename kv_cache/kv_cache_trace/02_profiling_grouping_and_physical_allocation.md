# Profiling, Grouping, And Physical Allocation

This chapter follows layer cache specs through memory profiling, grouping, final `KVCacheConfig` construction, worker allocation, reshaping, and binding to model layers.

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

Mamba and related state-space layers report `MambaSpec` through their own layer interfaces.

Encoder-only layers can be represented by `EncoderOnlyAttentionSpec` even though they allocate no persistent K/V pages.

## Cache Spec Kind

`get_kv_cache_spec_kind(...)` in `vllm/v1/kv_cache_interface.py` classifies specs for events, metrics, and manager selection.

Specialized subclasses are checked before base classes so sliding-window MLA, sink attention, and ordinary full attention remain distinguishable.

Custom cache specs should register with `KVCacheSpecRegistry`; otherwise grouping and manager selection cannot safely infer behavior.

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

The scheduler manages one block sequence per request per group.

## Uniform Specs

If all layer specs are the same cache type and block size, `UniformTypeKVCacheSpecs.from_specs(...)` can combine them.

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

The global `BlockPool` identifies physical capacity in block IDs shared across groups.

For one block ID to address corresponding storage in each group, group pages need compatible allocation accounting.

Padding a smaller logical page to a shared `page_size_bytes` allows block ID `b` to represent the `b`th page in every physical tensor participating in that pool.

`page_size_padded` records that reserved stride without changing logical key/value dimensions.

## KVCacheTensor And KVCacheConfig

`KVCacheTensor` describes one raw worker allocation:

- `size`: bytes to allocate.
- `shared_by`: layer names that view the same raw tensor.

`KVCacheConfig` contains:

- `num_blocks`: scheduler-visible blocks in the pool.
- `kv_cache_tensors`: raw worker allocation instructions.
- `kv_cache_groups`: scheduler group specifications.

`KVCacheConfig.needs_kv_cache_zeroing` is true for Mamba-containing configurations because recurrent state may require defined zero initialization.

## Memory Profiling

Workers profile model execution without the full KV cache to determine memory remaining after weights, activations, compilation, and runtime overhead.

The worker/executor path calls model-runner profiling methods and then `get_kv_cache_configs(...)` in `vllm/v1/core/kv_cache_utils.py` to turn available bytes into worker-specific configurations.

If `kv_cache_memory_bytes` is supplied, that explicit budget replaces utilization-derived cache capacity.

Otherwise usable memory is derived from total device memory, `gpu_memory_utilization`, and measured non-cache usage.

## Block Count

For uniform page sizes, the basic capacity relationship is:

```text
num_blocks = available_cache_bytes / bytes_consumed_by_one_block_id_across_all_groups
```

`_pool_bytes_per_block(...)` sums the bytes that one scheduler block ID consumes across cache groups.

`get_num_blocks(...)` applies available memory, page accounting, and overrides.

`may_override_num_blocks(...)` applies `num_gpu_blocks_override` when configured.

The null block consumes one pool entry, so usable request capacity is slightly less than the raw block count.

## Memory Sufficiency And Model Length

`check_enough_kv_cache_memory(...)` validates that at least one request of the configured maximum model length can be admitted under the cache specs.

`estimate_max_model_len(...)` and group-aware variants estimate the largest model length that fits if configured length is too high.

Auto-fit logic accounts for manager-specific maximum memory rather than assuming every layer retains every token.

Sliding-window and chunked-local specs can therefore admit longer sequences than a naive full-attention formula with the same physical bytes.

## Concurrency Estimate

`get_max_concurrency_for_kv_cache_config(...)` estimates how many maximum-length sequences can coexist from resolved block capacity and per-request page requirements.

This is a capacity estimate, not a scheduler guarantee, because real requests have varying lengths, shared prefixes, windows, connector loads, and speculative lookahead.

## Scheduler Projection Versus Worker Projection

`generate_scheduler_kv_cache_config(...)` creates the scheduler's group/block view.

`_project_kv_cache_groups_to_worker(...)` maps global grouping decisions back to each worker's local layers and byte allocations.

TP ranks normally receive equivalent group structures but page sizes reflect local KV-head counts and local cache dtypes.

Pipeline ranks receive only the layer tensors resident on that stage.

## Raw Worker Allocation

`GPUModelRunner._allocate_kv_cache_tensors(...)` allocates each `KVCacheTensor` as a flat `torch.int8` tensor of the requested byte count.

It uses zero initialization, which is required for some state caches and provides deterministic initial storage.

Every layer in `shared_by` is initially mapped to the same raw allocation object.

At this stage the tensor has correct bytes but no K/V semantic shape.

## Reshaping Attention Caches

`GPUModelRunner._reshape_kv_cache_tensors(...)` iterates attention groups and asks each backend for its cache shape.

The flow is:

1. Compute scheduler `num_blocks` from raw bytes and `page_size_bytes`.
2. Convert scheduler blocks into kernel blocks using scheduler/kernel block-size ratio.
3. Choose `storage_block_size` for compressed MLA or kernel block size otherwise.
4. Call `attn_backend.get_kv_cache_shape(...)`.
5. Obtain backend stride order or use logical identity order.
6. View the raw bytes as the cache spec dtype.
7. Construct contiguous or padded strided storage.
8. Permute back to the backend's logical dimension order while preserving physical strides.

## Shape And Byte Invariant

Without page padding:

```text
raw_tensor.numel() bytes == product(kv_cache_shape) * dtype_size
```

With page padding, logical elements occupy less than the page stride, so `torch.as_strided(...)` advances block rows by `page_size_bytes / dtype_size`.

Padding bytes remain reserved but are not part of logical K/V dimensions.

## Backend Stride Order

`get_kv_cache_stride_order(...)` returns a permutation describing desired physical dimension order.

The worker first permutes the target shape into physical order, creates a view, then applies the inverse permutation so kernels see the documented logical shape with backend-required strides.

This lets a backend request block-major, head-major, K/V-major, or transfer-friendly placement without changing its logical API.

## Hybrid Attention And Mamba Layout

When attention and Mamba coexist, `_update_hybrid_attention_mamba_layout(...)` can alter attention cache strides from a logical K/V-major view toward block-major physical placement.

It uses `get_kv_cache_block_dim(...)` to determine whether the backend's block dimension is zero or one.

This allows one shared block ID to correspond to physically aligned attention and Mamba pages while preserving the attention tensor's logical shape.

## Mamba Tensor Views

For `MambaSpec`, one raw page may contain several state tensors with different dtypes and shapes.

The worker creates an `as_strided` view for each state component using the padded page stride and a byte-derived storage offset.

The result associated with a Mamba layer is a list of state tensors rather than one conventional K/V tensor.

## Uniform Cross-Layer Allocation For Connectors

`KVConnectorModelRunnerMixin.use_uniform_kv_cache(...)` can select one cross-layer tensor when:

- A connector is active.
- The connector prefers cross-layer blocks.
- There is exactly one compatible attention group.
- The backend stride-order API supports an inserted layer dimension.

`allocate_uniform_kv_caches(...)` prepends `num_layers` to the backend shape and uses the backend's layer-aware stride order.

The goal is to make all layers' data for physical block `b` contiguous or efficiently addressable for one connector transfer.

Layer-specific cache tensors become views into this cross-layer allocation.

## Cross-Layer KV Sharing

Models such as YOCO can declare that one layer reuses another layer's cache.

`maybe_add_kv_sharing_layers_to_kv_cache_groups(...)` adds sharing layers to the target group for metadata purposes.

`initialize_kv_cache_tensors(...)` then assigns the exact target tensor object to the sharing layer instead of allocating separate bytes.

The common attention layer skips duplicate cache writes when `kv_sharing_target_layer_name` is set.

## Binding To Layers

`bind_kv_cache(...)` associates finalized cache tensors with attention-layer entries in `compilation_config.static_forward_context` and model-runner cache collections.

At runtime, `get_attention_context(...)` retrieves the correct layer tensor from forward context rather than receiving it as an explicit model-forward argument.

## Connector Registration

After cache initialization, an active connector registers the final layer-name-to-tensor mapping.

Registration must happen after reshaping because connectors need real strides, layouts, layer sharing, and potentially the uniform cross-layer tensor.

Connectors that pin or register GPU memory require stable physical addresses; `VllmConfig._verify_kv_transfer_compat(...)` rejects unsafe expandable-segment allocator combinations unless the CuMem allocator provides stable pages.

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
