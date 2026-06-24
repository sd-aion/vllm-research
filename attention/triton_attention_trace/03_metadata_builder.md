# Triton Attention Metadata Builder

This note explains `TritonAttentionMetadataBuilder` and `TritonAttentionMetadata`.

## Metadata Class

The metadata class is:

- `TritonAttentionMetadata` in `vllm/v1/attention/backends/triton_attn.py`

It contains the normal attention metadata fields:

- `num_actual_tokens`
- `max_query_len`
- `query_start_loc`
- `max_seq_len`
- `seq_lens`
- `block_table`
- `slot_mapping`

It also contains Triton-specific fields:

- `seq_threshold_3D`: maximum number of sequences for which Triton can switch from the normal 2D launch to the segmented 3D decode launch.
- `num_par_softmax_segments`: number of parallel KV-range segments used per sequence in the 3D segmented decode path.
- `softmax_segm_output`: scratch buffer that stores each segment's partial attention output before `reduce_segments(...)` combines segments.
- `softmax_segm_max`: scratch buffer that stores each segment's running softmax max value for numerically stable segmented reduction.
- `softmax_segm_expsum`: scratch buffer that stores each segment's running softmax exp-sum value for the final segmented softmax merge.
- `mm_prefix_range`: Python-side mapping from sequence index to multimodal prefix ranges that should get bidirectional/full attention.
- `mm_prefix_range_tensor`: padded GPU tensor version of `mm_prefix_range` passed into the Triton kernel for mask construction.

## Common Metadata Mapping

`TritonAttentionMetadataBuilder.build(...)` receives `CommonAttentionMetadata`.

The direct mapping is:

- `num_actual_tokens` controls slicing away padded tokens.
- `max_query_len` controls prefill/decode mode decisions inside `unified_attention(...)`.
- `query_start_loc` is passed as cumulative query starts.
- `seq_lens` is passed as `seqused_k`.
- `max_seq_len` gives the maximum K length.
- `block_table_tensor` maps logical KV blocks to physical KV blocks.
- `slot_mapping` maps current tokens to physical cache slots for KV update.

## 3D Decode Threshold

The builder computes:

- `self.seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv`

`MIN_LAUNCH_GRID_SIZE_2D` is 128.

The idea is that the 2D unified kernel grid is `(num_q_blocks, num_kv_heads)`.

For small decode batches, this grid can be too small to occupy the GPU well.

So the backend can use a 3D segmented path with an extra segment dimension.

`seq_threshold_3D` is the batch-size cutoff for that decision.

In `unified_attention(...)`, the 3D path is used only when the request is decode-like (`max_seqlen_q <= 1`), the segmented scratch buffers exist, batch invariance is off, and `num_seqs <= seq_threshold_3D`.

If the batch has more sequences than this threshold, the normal 2D launch already has enough `(query block, KV head)` programs, so Triton stays on the 2D path.

If the batch has fewer sequences than this threshold, the 2D grid may launch too few programs, so the 3D path adds a third grid dimension: `(num_q_blocks, num_kv_heads, num_par_softmax_segments)`.

That third dimension splits the KV range for each decode query into multiple parallel softmax segments.

Each segment writes partial output and softmax state into `softmax_segm_output`, `softmax_segm_max`, and `softmax_segm_expsum`, and `reduce_segments(...)` merges those partial segment results afterward.

Example: if `num_heads_kv = 8`, then `seq_threshold_3D = 128 // 8 = 16`.

That means decode batches with at most 16 sequences can use 3D segmented decode, while larger decode batches usually use the normal 2D kernel.

NOTE: Dive deeper into this for kernels

## CUDA Graph Threshold Adjustment

If decode CUDA graphs are enabled, the builder adjusts `seq_threshold_3D` to the nearest CUDA graph capture size.

This keeps the captured graph aligned with the same execution branch that will be used at runtime.

The relevant config is:

- `vllm_config.compilation_config.cudagraph_mode`
- `vllm_config.compilation_config.cudagraph_capture_sizes`

The reason is that CUDA graphs capture a concrete sequence of kernel launches for a specific padded batch size.

For Triton attention, the branch choice itself changes the launch structure: the 2D path launches `kernel_unified_attention` with grid `(num_q_blocks, num_kv_heads)`, while the 3D path launches it with grid `(num_q_blocks, num_kv_heads, num_par_softmax_segments)` and then launches `reduce_segments(...)`.

So a graph captured for the 2D path cannot safely be replayed for a runtime batch that wants the 3D path, and a graph captured for the 3D path cannot represent the simpler 2D path.

`cudagraph_capture_sizes` is the list of padded token/request sizes that vLLM plans to capture.

When decode CUDA graphs are enabled, `TritonAttentionMetadataBuilder.__init__(...)` picks the capture size closest to the natural `seq_threshold_3D` and uses that as the actual threshold.

That makes the 2D-vs-3D decision snap to one of the graph sizes vLLM will actually capture.

Example: if the natural threshold is `16`, but the configured decode graph capture sizes are `[1, 2, 4, 8, 16, 32]`, the threshold stays `16`.

If the capture sizes were `[1, 2, 4, 8, 32]`, the nearest capture size would be selected so graph capture and runtime replay still agree on which branch owns the nearby batch sizes.

## Segment Scratch Buffers

The builder allocates scratch tensors for the 3D segmented path:

- `softmax_segm_output`
- `softmax_segm_max`
- `softmax_segm_expsum`

The shapes are based on:

- `seq_threshold_3D`
- `num_heads_q`
- `NUM_PAR_SOFTMAX_SEGMENTS`
- `next_power_of_2(headdim)`

These buffers hold per-segment partial attention outputs and softmax state before `reduce_segments(...)` combines them.

## build_for_cudagraph_capture

`build_for_cudagraph_capture(...)` calls normal `build(...)` and then fills `seq_lens` with 1.

The reason is practical:

- setting `seq_lens` to `max_model_len` during full graph capture can make graph capture extremely slow.
- using length 1 keeps capture lightweight while preserving expected tensor shapes.

## MM Prefix Tensor

`TritonAttentionMetadata.compute_mm_prefix_range_tensor(...)` converts a Python dict into a padded GPU tensor.

Input shape conceptually is:

- `dict[seq_index, list[(start, end)]]`

Output shape is:

- `(num_seqs, max_ranges, 2)`

Empty ranges are padded with `(0, 0)`.

If all ranges are trivial, it returns `None`.

The kernel uses this tensor in `compute_kv_seq_mask(...)` to add bidirectional attention ranges for multimodal prefix regions.

## Cascade Fields

`TritonAttentionMetadata` includes cascade fields:

- `use_cascade`
- `common_prefix_len`
- `cu_prefix_query_lens`
- `prefix_kv_lens`
- `suffix_kv_lens`

However, `TritonAttentionBackend.use_cascade_attention(...)` returns false and `TritonAttentionImpl.forward(...)` asserts `attn_metadata.use_cascade is False`.

So these fields exist due to shared metadata structure patterns, but standard `TRITON_ATTN` does not use cascade attention.

## Key Files

- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/backend.py`
- `vllm/v1/attention/ops/triton_unified_attention.py`
