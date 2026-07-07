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

`append_row(block_ids, row_idx)` in `vllm/v1/worker/block_table.py` extends an existing request row instead of replacing it. The row already contains the block IDs for the request's earlier tokens; `append_row(...)` writes the newly allocated block IDs after the current valid row length and increments `num_blocks_per_row[row_idx]`. This is the worker-side update for a continuing request whose KV-cache block table grows during decode, chunked prefill, or speculative lookahead.

`clear_row(...)` removes stale IDs when a request leaves the worker batch.

`move_row(src, tgt)` in `vllm/v1/worker/block_table.py` copies the valid block IDs from one request row to another row and copies the valid row length. It is used by `InputBatch.condense(...)` in `vllm/v1/worker/gpu_input_batch.py` after requests finish or are removed. If row `2` becomes empty and the last active request is at row `7`, condense can move row `7` into row `2` so active requests stay packed at low indices.

`swap_row(src, tgt)` in `vllm/v1/worker/block_table.py` exchanges two request rows and their row lengths. It is used when the input batch explicitly swaps request indices, such as `InputBatch.swap_states(...)` in `vllm/v1/worker/gpu_input_batch.py`.

Both methods update worker-side metadata only: the row-to-block-ID table changes, but the actual KV cache tensors are not copied. The physical KV pages stay where they are; only the request batch row that points to those page IDs changes.

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

The slot-mapping buffer is `BlockTable.slot_mapping` in `vllm/v1/worker/block_table.py`. It is a reusable CPU/GPU buffer sized to `max_num_batched_tokens`; each scheduler step overwrites the prefix corresponding to the currently scheduled packed tokens, and the extra padding program fills unused CUDA-graph padding slots with `PAD_SLOT_ID`.

With multiple KV-cache groups, `MultiGroupBlockTable` owns one `BlockTable` per group, so each group has its own slot-mapping buffer. That matters when groups have different block sizes or block-table rows.

The kernel launches one program per request plus one padding program.

## Basic Slot Formula

Without context parallelism:

```text
block_index = position // block_size
block_offset = position % block_size
block_number = block_table[request, block_index]
slot_mapping[token] = block_number * block_size + block_offset
```

Variable meanings:

- `position`: absolute logical token position within the request sequence.
- `block_size`: number of token slots in one kernel KV-cache block.
- `block_index`: logical block index inside this request, computed from `position`.
- `block_offset`: token offset inside that logical block.
- `request`: worker batch row index for this request.
- `block_table[request, block_index]`: physical KV-cache block ID assigned to that logical request block.
- `block_number`: physical KV-cache block ID read from the block table.
- `slot_mapping[token]`: flat physical KV-cache slot where this scheduled token's K/V should be written.
- `token`: index of the current scheduled token inside the packed batch.

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

`GPUModelRunner._update_states(...)` in `vllm/v1/worker/gpu_model_runner.py` is the worker-side method that applies one `SchedulerOutput` to the model runner's persistent batch state. The scheduler tells the worker which requests are scheduled this step, how many tokens each request has computed, and what new block IDs were allocated. `_update_states(...)` turns that scheduler message into concrete updates to the worker's request table and block table.

There are two worker-side request structures involved:

- `GPUModelRunner.requests`: a dict from `req_id` to `CachedRequestState`, holding the worker's cached copy of request state such as token IDs, output IDs, block IDs, and `num_computed_tokens`.
- `InputBatch`: a dense active-batch table where each active request has a row index `req_index`. `InputBatch.req_id_to_index` maps `req_id` to that row.

The method first removes requests that are not scheduled in this step from the persistent `InputBatch` rows. Their `CachedRequestState` can remain in `GPUModelRunner.requests`, because the request may be scheduled again later.

New requests get a fresh `CachedRequestState` from scheduler-provided request data. When they are inserted into `InputBatch`, `InputBatch.add_request(...)` in `vllm/v1/worker/gpu_input_batch.py` assigns a row index, records `req_id_to_index[req_id] = req_index`, copies token/count metadata, and calls `block_table.add_row(request.block_ids, req_index)` with the complete group block IDs.

Continuing requests call `append_row(...)` for newly allocated blocks. The call site is `GPUModelRunner._update_states(...)` in `vllm/v1/worker/gpu_model_runner.py`: when `new_block_ids` arrives for a request that is already present in the persistent batch, the runner appends those IDs to that request's existing block-table row. This keeps the worker row aligned with the scheduler's request block sequence without rebuilding the whole row every step.

For a continuing request already in `InputBatch`, `_update_states(...)` finds the row with `self.input_batch.req_id_to_index.get(req_id)`, updates `input_batch.num_computed_tokens_cpu[req_index]`, appends new block IDs to both `req_state.block_ids` and `input_batch.block_table`, and updates token buffers if new output tokens were produced.

If a request was preempted and later resumed, it may not currently have an `InputBatch` row. In that case `_update_states(...)` replaces the cached block-ID sequence on `CachedRequestState` with the newly assigned scheduler block IDs and then re-adds the request to `InputBatch` as a resumed request.

After additions/removals, `InputBatch.condense(...)` in `vllm/v1/worker/gpu_input_batch.py` fills holes in the dense batch rows. When a row moves, it also calls block-table row movement methods so `req_id_to_index`, token buffers, count buffers, and block-table rows all keep the same row meaning.

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

`unified_kv_cache_update(...)` in `vllm/model_executor/layers/attention/attention.py` mutates the KV-cache tensor through `attn_layer.impl.do_kv_cache_update(...)`, but its useful effect is a side effect, not a meaningful returned value. It therefore returns an empty dummy tensor.

`Attention.forward(...)` passes that dummy tensor into `unified_attention_with_output(...)` as `kv_cache_dummy_dep`. `unified_attention_with_output(...)` immediately deletes the argument, but accepting it creates a visible data dependency for `torch.compile`: the attention op appears to depend on the output of the cache-update op.

That dependency prevents the compiler from legally reordering attention before the KV-cache write. Without it, the compiler may see two opaque custom ops where the first has no consumed result and could move the attention read ahead of the update, which would make decode read stale or missing K/V for the current tokens.

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
