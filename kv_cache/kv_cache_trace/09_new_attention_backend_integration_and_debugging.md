# New Attention Backend Integration And Debugging

This chapter is a decision-oriented checklist for adding an attention module that participates correctly in vLLM cache management.

The backend must align five independent contracts:

1. Logical cache lifetime through `KVCacheSpec` and scheduler managers.
2. Byte accounting through page-size formulas and grouping.
3. Physical tensor shape/strides through `AttentionBackend`.
4. Runtime write/read semantics through slot mappings, block tables, and metadata.
5. Transfer/distributed semantics through connectors, HMA, TP, PCP, and DCP.

## Start With The Semantic Cache Unit

Before writing a kernel, define what one token contributes to persistent cache.

Questions include:

- Are K and V separate vectors or one compressed latent vector?
- Are K and V head sizes equal?
- How many KV heads exist per TP rank?
- Does one stored row correspond to one logical token?
- Are scales per tensor, layer, block, token, head, or token-head?
- Are scales stored inline, in a separate tensor, or in layer buffers?
- Does the cache retain every token, a sliding window, aligned chunks, sink tokens, or summarized recurrent state?
- Can layers share the exact same cache bytes?

Do not derive page bytes until these semantics are explicit.

## Choose Or Create A KVCacheSpec

Use `FullAttentionSpec` when the scheduler must retain a continuous prefix and page bytes follow K/V storage.

Use `SlidingWindowSpec` or `ChunkedLocalAttentionSpec` only when the scheduler can safely reclaim unreachable blocks under those exact attention masks.

Use `MLAAttentionSpec` for latent/compressed MLA lifecycles with custom storage block size.

Use a specialized subclass such as `TQFullAttentionSpec` when lifecycle is full attention but byte accounting is custom.

Create a new `KVCacheSpec` only when existing manager and page formulas cannot represent the cache.

## Registering A Custom Spec

A custom spec must be registered through `KVCacheSpecRegistry` in `vllm/v1/kv_cache_spec_registry.py`.

Registration identifies:

- The uniform-type base used for grouping.
- The `SingleTypeKVCacheManager` implementation responsible for lifecycle.
- Spec-kind classification when events and metrics need a stable label.

If lifecycle matches full attention, prefer subclassing/merging with full-attention behavior rather than creating a new manager.

## Define Page Bytes

For each scheduler page, list every byte:

```text
K data
K metadata/scales
V data
V metadata/scales
alignment padding
```

Then define:

```text
real_page_size_bytes = meaningful bytes per scheduler block
page_size_bytes = max(real_page_size_bytes, padded_page_size)
```

For a generic MHA cache:

```text
block_size * num_kv_heads * (head_size_k + head_size_v) * dtype_size
```

For packed formats, use actual packed dimensions rather than multiplying logical head size by a nominal dtype.

## Page-Shape Invariant

For an unpadded cache:

```text
product(backend kernel-visible shape) * dtype_size
== scheduler_num_blocks * page_size_bytes
```

When scheduler blocks are split into kernel blocks, account for the larger kernel-visible block count and smaller kernel block size; total bytes must remain unchanged.

For padded pages, logical shape consumes `real_page_size_bytes`, while block stride consumes `page_size_bytes`.

## Implement Attention.get_kv_cache_spec

The model's `Attention` layer or custom cache-bearing layer must return the correct spec from `get_kv_cache_spec(vllm_config)`.

Inputs should come from resolved layer state:

- Final framework block size.
- Local KV-head count.
- K/V head dimensions.
- Physical cache dtype.
- Quantization mode.
- Window/chunk/sink behavior.
- Backend-specific packed byte values.

The result must not depend on transient batch state.

## Backend Capability Methods

The backend class should accurately define:

- `supported_dtypes`.
- `supports_kv_cache_dtype(...)`.
- `supports_head_size(...)`.
- `get_supported_kernel_block_sizes()`.
- `supports_block_size(...)` through inherited or custom logic.
- `get_preferred_block_size(...)` where a default is inappropriate.
- `supports_attn_type(...)`.
- `supports_kv_connector()`.
- `get_required_kv_cache_layout()` when transfer layout is constrained.
- distributed capabilities such as sink, non-causal, MM prefix, and compute capability.

Do not report a broad capability because construction succeeds if runtime kernels silently ignore the feature.

## get_kv_cache_shape

`get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)` defines the logical tensor view expected by kernels.

It must use the kernel block size passed by the worker, not assume scheduler block size.

Common layouts include:

```text
[2, num_blocks, block_size, num_kv_heads, head_size]
[num_blocks, 2, block_size, num_kv_heads, head_size]
[num_blocks, block_size, num_kv_heads, packed_slot_bytes]
```

The shape should expose all data bytes or be paired with deliberate scale/padding views over the raw allocation.

## Block Dimension

The inherited `get_kv_cache_block_dim(...)` inserts a sentinel `num_blocks` and finds its dimension in the shape.

Override it only when shape values can be ambiguous or block indexing is not discoverable that way.

Hybrid layout and connector registration rely on this dimension.

## Stride Order

Implement `get_kv_cache_stride_order(...)` when desired physical order differs from logical shape.

Return a permutation of logical dimensions in physical-major order.

Support `include_num_layers_dimension=True` if uniform cross-layer connector allocation should be available.

For efficient per-block transfer, physical block dimension should be early enough that all layer/page bytes can be addressed as contiguous regions.

Test actual `tensor.shape`, `tensor.stride()`, `storage_offset()`, and untyped storage size after worker initialization.

## Page Padding

If hardware or transfer requires alignment, put it in `page_size_padded` rather than pretending logical head dimension contains semantic elements.

The worker constructs a block-strided view whose block stride skips padding.

Cache-update and attention kernels should never read padding as K/V unless the format explicitly uses it for metadata.

Connectors can transfer padded opaque pages or only meaningful bytes according to canonical references, but source and destination strides must agree.

## Block-Size Negotiation

List kernel-supported block sizes honestly.

If a scheduler block is divisible by a supported kernel block, virtual splitting can preserve allocation while exposing smaller pages.

Test mapping:

```text
scheduler_block_id -> consecutive kernel block IDs
```

Verify both cache shape and block tables apply the same split factor.

Reject combinations where quantization group alignment, kernel indexing, or packed metadata cannot be split safely.

## Cache Update Contract

Decide whether `forward_includes_kv_cache_update` is true.

For a separate update path, implement `AttentionImpl.do_kv_cache_update(...)` and set the backend flag false.

For a fused path, ensure forward writes every current token before any operation that reads it.

The update path must:

- Slice or mask to actual tokens.
- Skip negative/padding slots.
- Derive physical block and offset from kernel block size.
- Write all local KV heads.
- Apply RoPE before storage when the model/backend contract requires post-RoPE keys.
- Quantize data and write every required scale/metadata field.
- Preserve shared-prefix blocks by writing only scheduler-provided new slots.

## Slot Mapping Tests

Test slots at:

- First offset in physical block zero.
- Last offset in a block.
- First token after a block boundary.
- Noncontiguous physical block rows.
- Virtual split boundaries.
- Negative padded slots.
- Multiple cache groups.
- CP local and nonlocal positions.

For each slot, verify exact cache byte location changed and neighboring pages did not.

## Paged Read Contract

Attention reads history through `block_table[request, logical_page]` plus offset.

The kernel must use `seq_len` to mask invalid final-page rows.

It must support shared/noncontiguous physical blocks without assuming request K/V is contiguous in memory.

Quantized reads must load scales from the same page/token/head used by data.

Window/chunk/MM/sink masks are semantic masks in addition to page validity.

## Metadata Class

Define a backend metadata dataclass containing only runtime tensors/scalars needed by kernels.

Common useful values are:

- Block table.
- Slot mapping.
- Sequence lengths.
- Query cumulative lengths.
- Maximum query and KV lengths.
- Decode/prefill split.
- CP-local sequence lengths.
- MM prefix ranges.
- Window/chunk settings.
- CUDA graph scheduler metadata.

Document shape, dtype, producer, and consumer for every field.

## Metadata Builder

Implement `AttentionMetadataBuilder.build(...)` from `CommonAttentionMetadata`.

Decide explicitly:

- Whether batch reordering is required.
- Whether speculative multi-token requests use decode kernels.
- Whether metadata supports fast block-table substitution.
- CUDA graph support level.
- Drafting metadata behavior.
- Cascade attention behavior.

If `supports_update_block_table = True`, implement safe replacement of both block table and slot mapping.

## Forward Context And Custom Ops

Use the common attention layer/custom-op boundary unless the backend has a justified alternate integration.

Forward context supplies layer-specific metadata and cache tensors.

Opaque custom ops must declare mutation/order semantics so compilation does not move cache reads before writes.

Output-buffer support should match backend declaration.

## Prefill Correctness

Test first-chunk causal prefill where raw current K/V may be used while cache is populated for future steps.

Test continuation prefill where queries attend to both cached history and current chunk.

Test mixed batches with decode rows and prefill chunks.

Verify query/KV causal alignment when query length is shorter than total sequence length.

## Decode Correctness

Test one-token decode across one and multiple pages.

Test MHA, GQA, and MQA head mapping.

Test page IDs in a nonmonotonic order to catch accidental contiguous-cache assumptions.

Test partial final pages and maximum sequence length.

Compare output and optional LSE against a dense reference after applying cache quantization/dequantization.

## Quantization Bring-Up

Bring up quantized cache in stages:

1. Validate page-size and physical views without running attention.
2. Run cache update and inspect packed data/scales.
3. Dequantize cache into a reference tensor.
4. Compare reference reconstructed K/V against expected quantized values.
5. Run single-token attention where expected softmax is trivial.
6. Run multi-token and multi-page attention.
7. Add graph, connector, and distributed tests.

Quantization error tolerance should be format-specific and measured separately from kernel arithmetic error.

## Connector Compatibility

Set `supports_kv_connector()` false until page transfer is verified if layout is unusual.

For compatibility, verify:

- `page_size_bytes` includes all data and metadata.
- Layer tensor starts at an expected storage offset or connector canonicalization supports offsets.
- Block pages can be exposed as opaque byte rows.
- Required HND/NHD layout is declared and selectable.
- Cross-layer stride order is either supported or safely disabled.
- Loaded pages reproduce local output bitwise or within expected quantization tolerance.
- Invalid/incomplete loads are reported before hash publication.

## HMA Compatibility

A connector-capable backend may participate in hybrid groups with different specs.

Check that group page padding, block IDs, and connector group references transfer only the intended layer bytes.

Do not assume group zero's block table or slot mapping applies to every layer.

Custom connectors need `SupportsHMA` only if they correctly handle all groups.

## Native Offload Compatibility

Test canonicalization into `[num_blocks, page_size_bytes]`.

Test GPU-to-CPU and CPU-to-GPU copies for multiple noncontiguous block IDs.

Verify padding, inline scales, and packed bytes survive round-trip.

Test eviction while loads are referenced and source GPU block reuse while stores are in flight.

## TP Requirements

Page size and tensor shape use local KV-head count.

Every rank must produce equivalent completion metadata for one logical connector job.

Test replicated KV heads where total KV heads are less than TP size.

No kernel should index global head IDs into a local cache tensor.

## PCP Requirements

Set `supports_pcp = True` only if prefill handles rank-local token partitions and required collectives.

Slot mappings already suppress nonlocal K/V writes, but attention metadata and kernels must understand global positions and local cache compaction.

Test interleave sizes one and larger aligned chunks.

## DCP Requirements

Set `can_return_lse_for_decode = True` only if decode returns numerically compatible softmax LSE.

Local output and LSE must represent attention over exactly the rank-local KV subset.

Distributed merge must match dense attention over the union.

Test uneven local sequence lengths and empty partitions.

## CUDA Graph Requirements

Declare one of `ALWAYS`, `UNIFORM_BATCH`, `UNIFORM_SINGLE_TOKEN_DECODE`, or `NEVER` through the metadata builder.

Capture must use persistent block-table, slot-mapping, metadata, output, and workspace addresses.

Padded requests use null blocks and padded tokens use negative slots.

Any runtime branch depending on sequence length must remain compatible with capture/replay sizes.

Fixed split counts can stabilize grids at the cost of empty work.

## Cache Sharing Requirements

If layers can share cache, verify identical physical encoding and positional semantics.

Only the producer layer should update the shared tensor.

Consumers still need metadata associated with the target cache group.

Connector registration should deduplicate aliases.

## Existing Test Areas

Relevant tests include:

- `tests/v1/core/test_kv_cache_utils.py` for grouping, sizing, and configuration.
- `tests/v1/core/test_prefix_caching.py` for hash lookup and block reuse.
- `tests/v1/core/test_single_type_kv_cache_manager.py` for manager lifecycles.
- `tests/v1/core/test_kv_cache_metrics.py` for residency metrics.
- `tests/v1/core/test_reset_prefix_cache_e2e.py` for reset behavior.
- `tests/models/quantization/test_per_token_kv_cache.py` and `tests/quantization/test_per_token_kv_cache.py` for generic quantized cache.
- `tests/kernels/test_compressor_kv_cache.py` for compressed cache operations.
- `tests/v1/kv_connector/unit/` for connector lifecycle, failure, HMA, layout, and implementations.
- `tests/v1/kv_offload/` for native manager, worker, shared region, transfer kernels, and tiering.
- `tests/v1/simple_kv_offload/` for the simple alternate path.

## Minimum New-Backend Test Matrix

The minimum matrix should cover:

- Every supported activation dtype.
- Every supported cache dtype/quantization mode.
- Minimum, typical, and maximum head sizes.
- Every declared kernel block size.
- Scheduler block size equal to and larger than kernel block size.
- One and multiple KV heads, including GQA.
- One token, partial page, exact page, and multi-page sequences.
- Pure prefill, pure decode, continuation prefill, and mixed batch.
- Noncontiguous physical block IDs.
- Prefix hit and shared-prefix concurrent requests.
- Preemption and block reuse.
- CUDA graph capture/replay at supported level.
- Connector/offload round-trip when advertised.
- TP and any advertised PCP/DCP modes.

## Debugging: Allocation Size Mismatch

Symptoms include reshape failure, `numel() % page_size_bytes != 0`, out-of-bounds view construction, or unexpectedly low block count.

Check:

- Spec `real_page_size_bytes` and padding.
- Physical dtype size.
- Backend shape product.
- Auxiliary scale bytes.
- Scheduler/kernel split factor.
- Local versus global KV-head count.

## Debugging: Correct Prefill, Wrong Decode

This usually means first-chunk prefill uses raw K/V successfully while paged cache write/read is wrong.

Check:

- Slot mapping and negative-slot handling.
- K/V axis order.
- Block dimension and stride order.
- Page-table logical-to-physical lookup.
- Quantization scales.
- RoPE placement before storage.

## Debugging: Correct First Page, Wrong Later Pages

Check block-table stride, block-size constant, virtual block splitting, and physical block ID expansion.

Use deliberately nonsequential block IDs so accidental `base + logical_page` addressing fails clearly.

## Debugging: Wrong Final Tokens Only

Check sequence lengths, final-page masks, query/current chunk alignment, and whether current K/V update completed before attention.

Do not infer valid length from number of block-table entries.

## Debugging: Prefix Cache Corruption

Check that only full finalized blocks receive hashes.

Verify extra hash keys include every input that changes K/V.

Confirm shared blocks are never written after another request touches them.

On connector failure, ensure destination block hashes are evicted before recomputation.

## Debugging: Offload Round-Trip Failure

Compare raw page bytes before store and after load.

Check canonical page size, padding, shared storage offsets, cross-layer stride order, and worker shard mapping.

If raw bytes match but output differs, inspect metadata/sequence lengths rather than transfer.

## Debugging: Intermittent RDMA Failure

Check expandable-segment allocator settings and whether registered GPU addresses were remapped.

Verify handshake metadata corresponds to current wake-up/reallocation generation.

Re-register connector memory after cache allocation changes.

## Debugging: DCP Numerical Mismatch

Check local KV ownership, local sequence lengths, local output normalization, LSE base/shape, and distributed merge formula.

A locally normalized output cannot be averaged equally across ranks; it must be weighted by rank softmax mass.

## Useful Inspection Commands

```bash
rg -n "class .*KVCache|class .*Manager|class BlockPool" vllm/v1
rg -n "slot_mapping|block_table" vllm/v1/worker vllm/v1/attention
rg -n "KVConnectorBase_V1|register_connector" vllm/distributed/kv_transfer
rg -n "OffloadingConnector|CPUOffloading" vllm/v1 vllm/distributed/kv_transfer
rg -n "KVQuantMode|get_kv_quant_mode" vllm
```

## Bring-Up Order

Use this implementation order:

1. Define cache semantics and byte layout.
2. Implement/cache-spec page accounting.
3. Implement backend shape and stride order.
4. Verify worker allocation views without kernels.
5. Implement slot-mapped cache update.
6. Implement paged read attention.
7. Implement metadata builder and runtime dispatch.
8. Add prefix sharing and preemption tests.
9. Add quantized formats one at a time.
10. Add CUDA graph support.
11. Add connector/offload compatibility.
12. Add TP and advertised context-parallel modes.

This order isolates page-layout errors before distributed, graph, or connector state makes failures harder to localize.
