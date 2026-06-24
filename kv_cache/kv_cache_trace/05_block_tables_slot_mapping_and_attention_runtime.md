# Block Tables, Slot Mapping, And Attention Runtime

This chapter connects scheduler-owned block IDs to worker-side tensors and explains how attention writes current K/V and reads historical K/V.

## Two Addressing Problems

KV-cache execution needs two different mappings:

- A block table answers: for request `r` and logical token page `p`, which physical cache block contains that page?
- A slot mapping answers: for current packed token `x`, which flat physical cache slot should its K/V update write?

The block table persists for a request across steps.

The slot mapping is rebuilt for tokens scheduled in the current step.

## Scheduler Block IDs

`KVCacheManager` returns `KVCacheBlocks` containing integer physical block IDs per cache group.

`SchedulerOutput` reports newly allocated or complete request block ID sequences to workers.

The scheduler does not send cache bytes; all workers independently maintain tensors indexed by the same scheduler block IDs.

## Worker BlockTable

`BlockTable` and `MultiGroupBlockTable` are defined in `vllm/v1/worker/block_table.py`.

One `BlockTable` owns:

- A CPU/GPU `block_table` buffer shaped `[max_num_reqs, max_num_kernel_blocks_per_req]` with int32 IDs.
- `num_blocks_per_row` on CPU for valid row lengths.
- A CPU/GPU `slot_mapping` buffer shaped `[max_num_batched_tokens]` with int64 flat slots.
- Scheduler and kernel block-size conversion state.
- PCP/DCP rank and KV interleave parameters.

`MultiGroupBlockTable` contains one such table per cache group and forwards row operations to all groups.

## Request Rows

The model runner maintains a dense batch index for active cached request state.

`add_row(block_ids, row_idx)` replaces a request row.

`append_row(...)` adds newly allocated decode/lookahead blocks to an existing row.

`clear_row(...)` removes stale IDs when a request leaves the worker batch.

`move_row(...)` and `swap_row(...)` support batch compaction/reordering without rebuilding every table.

`commit_block_table(num_reqs)` copies active CPU rows to the persistent GPU buffer before metadata construction and model execution.

## Scheduler Blocks To Kernel Blocks

When scheduler and kernel block sizes match, scheduler IDs are written directly.

When one scheduler block is virtually split, `map_to_kernel_blocks(...)` expands each scheduler ID.

For scheduler block size 32 and kernel block size 16:

```text
scheduler block 0 -> kernel blocks [0, 1]
scheduler block 1 -> kernel blocks [2, 3]
scheduler block 2 -> kernel blocks [4, 5]
```

The formula for scheduler block `b` and sub-block `k` is:

```text
kernel_block_id = b * blocks_per_kv_block + k
```

This matches worker cache reshaping, where physical scheduler pages are viewed as additional smaller pages along the block dimension.

## Block Table Meaning

For a non-CP request and kernel block size `B`, logical token position `t` uses:

```text
logical_block_index = t // B
offset_in_block = t % B
physical_block = block_table[request_row, logical_block_index]
flat_slot = physical_block * B + offset_in_block
```

The attention backend expands `flat_slot` into K/V tensor dimensions according to its cache layout.

## Final Block Occupancy

The block table does not record how many positions in its final block are valid.

Validity comes from request sequence lengths in attention metadata.

For block size 16 and sequence length 35, the table has three block IDs, but only offsets zero through two are valid in the third block.

Kernels mask positions `>= seq_len`, not merely positions outside the allocated table length.

## Slot Mapping Kernel

`BlockTable.compute_slot_mapping(...)` launches `_compute_slot_mapping_kernel(...)` from `vllm/v1/worker/block_table.py`.

Inputs are:

- `query_start_loc[num_reqs + 1]`: packed current-token boundaries per request.
- `positions[num_tokens]`: absolute logical positions of current tokens.
- GPU block-table rows.
- Kernel block size.
- CP world/rank and interleave settings.
- Persistent output slot-mapping buffer.

The kernel launches one program per request plus one padding program.

## Basic Slot Formula

Without context parallelism:

```text
block_index = position // block_size
block_offset = position % block_size
block_number = block_table[request, block_index]
slot_mapping[token] = block_number * block_size + block_offset
```

The value is flat across physical blocks and token rows; KV head and K/V dimensions are handled by the update kernel.

## Slot Mapping Example

Let block size be four and a request block-table row be `[5, 2, 9]`.

Positions zero through three map to physical block 5, positions four through seven map to physical block 2, and positions eight through eleven map to physical block 9.

Position six maps to:

```text
block_index = 6 // 4 = 1
block_offset = 6 % 4 = 2
physical_block = block_table[1] = 2
slot = 2 * 4 + 2 = 10
```

An update kernel receiving slot 10 derives physical block 2 and offset 2 with division and modulo by four.

## Padding Slot ID

Unused padded token positions receive `PAD_SLOT_ID`, currently represented as a negative slot such as `-1`.

Cache-update kernels must skip negative slots.

The slot-mapping kernel includes a final padding program that overwrites the entire unused tail of the persistent buffer, preventing stale valid slots from a previous larger batch from causing writes during CUDA graph replay.

## Context Parallel Slot Mapping

With total CP world size `C`, one scheduler block-table entry represents a virtual token span of:

```text
virtual_block_size = local_block_size * C
```

For absolute position `p`:

```text
virtual_block_index = p // virtual_block_size
virtual_offset = p % virtual_block_size
owner_rank = (virtual_offset // interleave_size) % C
```

Only the owner rank receives a real slot.

Other ranks receive `PAD_SLOT_ID` and do not store that token's K/V locally.

The owner rank compacts interleaved chunks into a local offset:

```text
round = virtual_offset // (C * interleave_size)
remainder = virtual_offset % interleave_size
local_offset = round * interleave_size + remainder
```

The final local slot is `physical_block * local_block_size + local_offset`.

## InputBatch Updates

`GPUModelRunner._update_states(...)` consumes scheduler output and updates cached request rows.

New requests call `add_row(...)` with their complete group block IDs.

Continuing requests call `append_row(...)` for newly allocated blocks.

Batch compaction keeps block-table rows synchronized with request-state indices.

Before input preparation completes, block tables are committed to GPU and slot mappings are computed for the packed token positions.

## Multiple Cache Groups

Every cache group receives its own block table and slot mapping because block sizes and physical block sequences can differ.

`_get_slot_mappings(...)` in `GPUModelRunner` creates:

- `slot_mappings_by_gid`: group ID to tensor, used while building common attention metadata.
- `slot_mappings_by_layer`: layer name to the tensor of its group, installed in forward context.

Layers in the same group share a slot mapping because they share block-table lifecycle and block size.

## CommonAttentionMetadata

`GPUModelRunner._build_attention_metadata(...)` creates `CommonAttentionMetadata` from packed batch state.

Important cache-related fields include:

- `query_start_loc`: cumulative current query-token boundaries.
- `seq_lens` and CPU upper bounds: total valid KV lengths.
- `block_table_tensor`: current group's device block-table rows.
- `slot_mapping`: current group's write destinations.
- `num_actual_tokens`: excludes CUDA graph padding.
- `max_query_len` and `max_seq_len`: kernel selection and launch bounds.
- CP-local sequence lengths when context parallelism is enabled.
- encoder sequence lengths for cross-attention groups.

The first group builds a base common object. Later groups replace block table and slot mapping while retaining common packed-query state.

## Backend Metadata Builders

Each `AttentionGroup` has an `AttentionMetadataBuilder` created from its merged cache spec, layer names, `VllmConfig`, and device.

The builder transforms common metadata into backend-specific objects, such as FlashAttention cumulative lengths or Triton multimodal ranges.

If `supports_update_block_table` is true, the model runner can reuse metadata structure and substitute another group's block table and slot mapping more cheaply.

Otherwise it invokes the builder for each group.

## Forward Context

The model runner installs per-layer attention metadata, layer objects, cache tensors, and slot mappings in forward context.

`Attention.forward(...)` in `vllm/model_executor/layers/attention/attention.py` does not receive scheduler structures directly in the model signature.

`get_attention_context(layer_name)` retrieves:

- Backend-specific metadata.
- The concrete attention layer.
- The layer's physical KV-cache tensor.
- The layer/group slot mapping.

This keeps model code independent of continuous-batching cache plumbing.

## Cache Update Boundary

Backends declare `forward_includes_kv_cache_update`.

When true, the attention implementation writes current K/V as part of its forward path.

When false, the common layer invokes `unified_kv_cache_update(...)` before `unified_attention_with_output(...)`.

A dummy tensor dependency ensures `torch.compile` preserves update-before-read ordering even though attention does not consume a meaningful update result.

## Writing Current K/V

A generic cache-update kernel receives:

- Current key tensor `[num_tokens, num_kv_heads, head_size]`.
- Current value tensor.
- Physical cache tensor/views.
- `slot_mapping[num_tokens]`.
- Optional scales and quantization mode.

For each valid token slot it computes block and offset, then writes all local KV heads.

Quantized paths compute/copy scales and packed values according to the backend layout.

Negative slots are skipped, which handles padding and nonlocal CP tokens.

## Reading Historical K/V

A paged attention kernel receives block table and sequence lengths rather than slot mappings.

For each logical key position it derives logical page and page offset, loads physical block ID from the request row, and addresses the backend cache tensor.

This indirection lets requests own noncontiguous physical pages and share prefix blocks without copying them into a contiguous sequence tensor.

## Prefill And Decode

Prefill can process many current query tokens and may use direct current K/V for first-chunk dense attention, but K/V still needs to be written for later steps.

Continuation prefill reads previous pages plus current K/V under causal masking.

Decode usually writes one new token and reads every reachable historical page.

The same block table supports all phases; metadata query lengths and sequence lengths tell the backend which behavior is active.

## Cache Sharing Between Layers

When one layer reuses another layer's cache, both layer names resolve to the same physical tensor and cache group.

The sharing layer skips duplicate cache updates.

Its attention metadata must still point to the shared group's block table and valid sequence lengths.

Fast-prefill sharing can wrap or override metadata for eligible suffix layers, but correctness still depends on the target layer having produced every cache row that consumers read.

## CUDA Graph Stability

Block-table and slot-mapping buffers are persistent so graph capture and replay use stable device addresses.

Padded request rows are filled with null block ID zero.

Padded token slots are filled with negative IDs.

Capture metadata may use short placeholder sequence lengths, with replay updating persistent input buffers before launch.

## CuMem Wake-Up

Sleep/wake or CuMem reallocation can change data pointers.

Block-table implementations that cache pointer tensors must rebuild them after wake-up.

Physical K/V views and connector registrations must similarly be restored against the new allocation addresses.

## End-To-End Example

Assume block size four and two active requests.

Request A has sequence length six with block row `[5, 2]` and schedules one decode token at absolute position six.

Request B has sequence length three with block row `[8]` and schedules prompt positions zero through two.

Packed Q contains four rows: A position six, then B positions zero, one, and two. `query_start_loc = [0, 1, 4]`.

Slot mapping becomes:

```text
A pos 6: block_table_A[1]=2, offset=2, slot=10
B pos 0: block_table_B[0]=8, offset=0, slot=32
B pos 1: block_table_B[0]=8, offset=1, slot=33
B pos 2: block_table_B[0]=8, offset=2, slot=34
```

Cache update writes K/V for slots `[10, 32, 33, 34]`.

Decode attention for A reads logical positions zero through six through blocks 5 and 2.

Prefill attention for B applies a causal triangular mask over positions zero through two and leaves offset three in physical block 8 invalid because `seq_len_B = 3`.
