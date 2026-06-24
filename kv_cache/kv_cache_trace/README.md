# vLLM KV Cache Management Trace

This folder maps the vLLM v1 KV-cache system from launch configuration to physical cache bytes, scheduler ownership, attention execution, prefix reuse, connector transfer, offloading, and final release.

The trace is written for someone who understands continuous batching but wants to add or modify an attention backend without violating cache-management assumptions.

## Short Mental Model

The scheduler and worker view the cache differently:

- The scheduler owns logical `KVCacheBlock` objects, request-to-block assignments, reference counts, prefix hashes, and eviction policy.
- The worker owns physical device tensors whose byte sizes and views are derived from layer cache specs and attention-backend layout methods.
- A request's block table maps its logical token pages to scheduler-selected physical block IDs.
- A slot mapping gives the physical write slot for each token computed in the current scheduler step.
- Attention kernels use block tables and sequence lengths to read historical K/V, while cache-update kernels use slot mappings to write current K/V.
- Prefix caching keeps the hash and physical contents of full blocks after their request releases its reference.
- KV connectors can report externally available prefix tokens and move cache pages between GPU memory and another process, device, host allocation, or storage service.
- Native CPU offloading is implemented as a KV connector with scheduler-side lookup/planning and worker-side asynchronous transfers.

## Important Distinctions

`hash_block_size` is the token granularity used to hash request prefixes.

`scheduler_block_size` is the token granularity managed by a cache group in the scheduler.

`kernel_block_size` is the page granularity accepted by an attention kernel. One scheduler block can be exposed as multiple kernel blocks when virtual block splitting is valid.

`KVCacheSpec` describes logical storage requirements and page-byte accounting.

`AttentionBackend.get_kv_cache_shape(...)` and `get_kv_cache_stride_order(...)` describe the physical tensor view expected by a kernel.

`KVCacheConfig` is the resolved allocation plan shared between scheduler and workers after memory profiling and cache grouping.

`KVCacheBlocks` is a scheduler-facing allocation result containing one block sequence per cache group.

## Reading Order

1. `01_configuration_specs_quantization.md`
2. `02_profiling_grouping_and_physical_allocation.md`
3. `03_blocks_prefix_caching_and_eviction.md`
4. `04_managers_coordinators_and_scheduler_lifecycle.md`
5. `05_block_tables_slot_mapping_and_attention_runtime.md`
6. `06_kv_connectors_architecture_and_catalog.md`
7. `07_native_cpu_offloading_deep_dive.md`
8. `08_distributed_hybrid_and_special_cases.md`
9. `09_new_attention_backend_integration_and_debugging.md`

## Primary Source Areas

- `vllm/config/cache.py`
- `vllm/config/kv_transfer.py`
- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/kv_cache_coordinator.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/`
- `vllm/v1/kv_offload/`

All source paths are relative to `/home/ubuntu/vllm`.
