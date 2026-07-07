# Configuration, Cache Specs, And Quantization

This chapter explains the configuration and type system that determine how many cache bytes vLLM allocates and how an attention backend must interpret those bytes.

## CacheConfig

`CacheConfig` is defined in `vllm/config/cache.py` and is constructed from engine arguments in `vllm/engine/arg_utils.py`.

Important fields are:

- `block_size`: default physical cache block size in tokens; the default is 16 before platform/backend adjustment.
- `hash_block_size`: token block size used for request block hashes and prefix-cache lookup.
- `gpu_memory_utilization`: fraction of device memory available to the model executor when an explicit cache byte budget is absent.
- `kv_cache_memory_bytes`: explicit per-GPU KV-cache byte budget; when set, it takes precedence over utilization-derived capacity.
- `cache_dtype`: logical cache dtype or quantized cache format.
- `num_gpu_blocks_override`: testing/debug override for the profiled block count.
- `enable_prefix_caching`: enables block hashing, lookup, and retention after request completion.
- `prefix_caching_hash_algo`: chooses serialization and hash function.
- `calculate_kv_scales`: deprecated dynamic FP8 per-tensor scale calculation switch.
- `kv_cache_dtype_skip_layers`: layer indices or attention-type labels that retain native cache dtype.
- `sliding_window`: model-level window copied into cache configuration.
- `mamba_block_size`, `mamba_cache_dtype`, `mamba_ssm_cache_dtype`, and `mamba_cache_mode`: state-cache controls for hybrid models.
- `kv_sharing_fast_prefill`: enables metadata changes for eligible cache-sharing models. (WIP acc to v 0.23.0)
- `kv_offloading_size`: total offload capacity in GiB, summed across TP ranks.
- `kv_offloading_backend`: native vLLM offload or LMCache.
- `num_gpu_blocks` and `num_cpu_blocks`: resolved fields populated after profiling.

`CacheConfig._apply_block_size_default(...)` records whether block sizes were user supplied. Platform and backend resolution can still adjust defaults before `VllmConfig.validate_block_size(...)` enforces DCP, interleave, and Mamba constraints.

## Three Block Sizes

The source uses several block-size concepts that should not be conflated.

### Hash Block Size

`hash_block_size` controls how request token hashes are generated.

It can be smaller than physical cache blocks if every physical group block size is divisible by it. Smaller hash granularity allows groups with different physical block sizes to share a common prefix-key stream.

If physical block size is 32 and hash block size is 8, one physical block corresponds to four chained request hashes.

### Scheduler Block Size

The scheduler block size belongs to a `KVCacheSpec` and determines how many logical request tokens one scheduler-managed `KVCacheBlock` represents for that group.

Different groups can theoretically use different scheduler block sizes, which is why `KVCacheBlocks.blocks[group][block]` is group-major rather than token-block-major.

### Kernel Block Size

The kernel block size is negotiated from all attention backends in a cache group by `prepare_kernel_block_sizes(...)` in `vllm/v1/worker/utils.py`.

If a scheduler page contains 256 tokens but a kernel accepts 64-token pages, the worker exposes four kernel pages for each scheduler block by increasing the kernel-visible block count.

This is virtual block splitting: the allocation and scheduler block ID remain coarse, while the physical tensor view presents a finer page dimension.

The relationship must be integral:

```text
num_kernel_blocks_per_scheduler_block = scheduler_block_size / kernel_block_size
kernel_num_blocks = scheduler_num_blocks * num_kernel_blocks_per_scheduler_block
```

## Cache Dtypes

`CacheDType` in `vllm/config/cache.py` includes:

- `auto`, `float16`, and `bfloat16` for native storage.
- `fp8`, `fp8_e4m3`, `fp8_e5m2`, and `fp8_inc` for FP8 variants.
- `fp8_ds_mla` for DeepSeek MLA-specific packed layouts.
- `int8_per_token_head` and `fp8_per_token_head` for dynamic scale-per-token-head storage.
- `nvfp4` for packed FP4 data with block scales.
- Four `turboquant_*` formats with backend-specific packed K/V slots.

`resolve_kv_cache_dtype_string(...)` and `kv_cache_dtype_str_to_dtype(...)` in `vllm/utils/torch_utils.py` convert logical names into physical torch dtypes where a generic dtype exists.

Special packed layouts may use an element dtype such as uint8 while defining byte meaning through their cache spec and backend.

## KVQuantMode

`KVQuantMode` is defined in `vllm/v1/kv_cache_interface.py` so kernels can dispatch by an enum instead of repeatedly matching strings.

The modes are:

- `NONE = 0` for native or backend-specific formats not represented by the generic enum.
- `FP8_PER_TENSOR = 1` for FP8 K/V using layer/tensor scales.
- `INT8_PER_TOKEN_HEAD = 2` for one dynamic K scale and V scale per token and KV head.
- `FP8_PER_TOKEN_HEAD = 3` for FP8 data with the same per-token-head scale granularity.
- `NVFP4 = 4` for packed four-bit data and FP8 block scales.

`get_kv_quant_mode(...)` maps cache dtype strings to this enum.

TurboQuant is not mapped to a generic `KVQuantMode`; its backend owns the entire byte layout and quantization contract.

## KVCacheSpec Base Contract

`KVCacheSpec` in `vllm/v1/kv_cache_interface.py` describes one layer or merged layer set.

The essential contract is:

- `block_size`: logical tokens represented by one scheduler page.
- `page_size_bytes`: total bytes reserved per page, including scales and padding.
- `storage_block_size`: number of physically stored token rows, normally equal to `block_size` but different for compressed MLA.
- `max_memory_usage_bytes(vllm_config)`: upper bound needed by this spec.
- `copy_with_new_block_size(...)`: recreate the spec at another block size.
- `merge(...)`: combine compatible layer specs into one group-level spec.
- `is_uniform_with_collection(...)`: determine whether specs can use uniform-type grouping.

`KVCacheSpecRegistry` in `vllm/v1/kv_cache_spec_registry.py` lets custom subclasses register their uniform-type base and scheduler manager.

## AttentionSpec

`AttentionSpec` adds:

- `num_kv_heads`: KV heads local to the worker rank.
- `head_size`: K head dimension.
- `dtype`: physical tensor element dtype.
- `kv_quant_mode`: generic quantization mode.
- `page_size_padded`: optional alignment-expanded page size.

For ordinary equal-sized K and V, unquantized real page bytes are:

```text
2 * block_size * num_kv_heads * head_size * dtype_size
```

The factor two accounts for K and V.

Per-token-head quantization adds two float32 scale arrays:

```text
scale_bytes_per_page = 2 * block_size * num_kv_heads * sizeof(float32)
```

The scale storage is carved from the raw allocation even when a backend exposes it through separate strided tensor views.

## FullAttentionSpec

`FullAttentionSpec` adds `head_size_v`, optional `sliding_window`, and optional `attention_chunk_size`.

Its generic real page formula is:

```text
block_size * num_kv_heads * (head_size + head_size_v) * dtype_size
```

Its maximum memory normally covers `ceil(max_model_len / block_size)` pages, divided across total context-parallel ranks when PCP or DCP shards token context.

When hybrid allocation is disabled, a sliding-window layer can still use `FullAttentionSpec`; the worker computes windowed attention, but the scheduler allocates cache for the full sequence.

## SlidingWindowSpec

`SlidingWindowSpec` retains only pages that can contribute to future attention.

Its admission bound includes the active window, newly scheduled tokens, and one extra page for an unaligned window start:

```text
ceil((sliding_window - 1 + max_num_batched_tokens) / block_size) + 1
```

The corresponding `SlidingWindowManager` removes unreachable historical blocks and may replace their positions with the shared null block.

## ChunkedLocalAttentionSpec

`ChunkedLocalAttentionSpec` models aligned attention chunks rather than a rolling window.

Its maximum retained region is bounded by one attention chunk plus the current scheduler token budget.

`ChunkedLocalAttentionManager` aligns prefix-hit and skipped-block logic to chunk boundaries.

## MLAAttentionSpec

`MLAAttentionSpec` stores a latent K/V representation rather than separate full MHA K and V.

It supports:

- `cache_dtype_str` for backend-specific MLA formats.
- `alignment` and `page_size_padded` for transfer/kernel alignment.
- `compress_ratio`, which makes `storage_block_size = block_size / compress_ratio`.
- `model_version` for layouts such as DeepSeek V4.

Some MLA formats override byte accounting completely, such as the fixed per-token byte layouts used by `fp8_ds_mla`.

`SlidingWindowMLASpec` combines compressed MLA storage with scheduler window retention.

## TQFullAttentionSpec

`TQFullAttentionSpec` carries `tq_slot_size` and overrides real page bytes with:

```text
block_size * num_kv_heads * tq_slot_size
```

The TurboQuant backend stores one combined packed K/V byte slot per token-head instead of generic full-size K and V vectors.

All merged TurboQuant specs must use the same slot size.

## NVFP4

NVFP4 stores two four-bit values per byte and includes FP8 block scales.

`nvfp4_kv_cache_full_dim(...)` in `vllm/utils/torch_utils.py` computes the storage dimension needed for packed data plus scales.

The spec uses that physical dimension rather than pretending each logical head coordinate occupies one dtype element.

Both K and V layouts must be included, and unequal `head_size_v` is handled by summing their packed dimensions.

## Per-Tensor FP8

Generic per-tensor FP8 stores K/V in FP8 cache tensors and uses layer scale tensors such as `_k_scale` and `_v_scale` from the concrete `Attention` layer.

`maybe_calc_kv_scales(...)` in `vllm/model_executor/layers/attention/attention.py` can update these scales when dynamic calculation is enabled, though the controlling configuration is deprecated.

Backends decide whether scales are applied during cache update, attention score computation, or folded into query/output scaling.

## Per-Token-Head Quantization

For `int8_per_token_head` and `fp8_per_token_head`, each token-head row computes independent K and V scales.

This reduces the range shared by unrelated tokens and heads but consumes scale bytes and adds cache-update/decode work.

The logical spec budgets float32 scales, while backends such as Triton can expose scale caches as strided views over an inline padded tail in the same raw allocation.

The backend must keep its `get_kv_cache_shape(...)`, page-size accounting, scale-view construction, update kernel, and attention kernel consistent.

## MambaSpec

`MambaSpec` represents recurrent state rather than conventional K/V vectors.

It contains one or more state shapes and dtypes, optional page padding, Mamba backend type, cache mode, and speculative blocks.

Its page size is the sum of all state-tensor byte sizes, optionally raised to a padded page size shared with attention groups.

Modes are:

- `none`: only current state plus speculative state requirements.
- `align`: cache selected scheduler-step or aligned states.
- `all`: cache state at every block boundary for prefix reuse.

## Other Specs

`CrossAttentionSpec` reserves pages based on maximum encoder length rather than decoder model length.

`EncoderOnlyAttentionSpec` reports zero persistent KV-cache memory.

`SinkFullAttentionSpec` carries sink retention information and otherwise follows full-attention page accounting.

`HiddenStateCacheSpec` marks MLA-shaped hidden-state cache used by extraction/speculative workflows.

`UniformTypeKVCacheSpecs` combines multiple same-type layer specs into one page whose byte size is the sum of member page sizes.

## Backend Responsibilities

A cache spec budgets bytes, but the attention backend defines how those bytes become a tensor.

The backend must provide:

- `get_kv_cache_shape(...)` with dimensions whose product matches the unpadded allocation interpretation.
- `get_kv_cache_block_dim(...)` or a shape from which the block dimension can be discovered.
- `get_kv_cache_stride_order(...)` when physical ordering differs from logical shape.
- `get_required_kv_cache_layout(...)` when connector/platform layout selection is constrained.
- block-size support compatible with scheduler-to-kernel splitting.
- cache-update and attention kernels that interpret dtype, scales, padding, and strides identically.

The most important invariant is:

```text
bytes budgeted by KVCacheSpec == bytes consumed by the backend tensor layout and auxiliary scale views
```
