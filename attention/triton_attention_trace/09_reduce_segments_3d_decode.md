# reduce_segments For 3D Decode

This note explains the second-stage reduction used by the segmented 3D decode path.

## Location

The reduction kernel is:

- `reduce_segments(...)` in `vllm/v1/attention/ops/triton_unified_attention.py`

It is launched only when `unified_attention(...)` selected `use_3d=True`.

## Inputs

Important inputs are:

- `output_ptr`: final output tensor.
- `segm_output_ptr`: per-segment partial outputs.
- `segm_max_ptr`: per-segment row maxima.
- `segm_expsum_ptr`: per-segment exp sums.
- `seq_lens_ptr`: sequence lengths.
- `query_start_len_ptr`: cumulative query starts.
- `num_query_heads`
- `output_stride_0`
- `output_stride_1`
- `TILE_SIZE`
- `HEAD_SIZE`
- `HEAD_SIZE_PADDED`
- `BLOCK_Q`
- `NUM_SEGMENTS_PER_SEQ`
- `USE_FP8`

## Grid

The launch grid is:

- `(q.shape[0], num_query_heads)`

Each program handles one query token and one query head.

## Sequence Lookup

The kernel calls:

- `find_seq_idx(...)`

Here it uses raw query-token mode, not q-block mode.

It finds which request owns the current query token.

Then it loads that request's `seq_len`.

## Active Segment Count

The kernel computes:

- `tiles_per_segment = ceil(seq_len / (NUM_SEGMENTS_PER_SEQ * TILE_SIZE))`
- `act_num_segments = ceil(seq_len / (tiles_per_segment * TILE_SIZE))`

This lets it ignore unused segment slots for shorter sequences.

## LSE-Style Recombination

Each segment has:

- `segm_max`
- `segm_expsum`
- `segm_output`

The kernel computes:

- `overall_max = max(segm_max)`
- rescaled `segm_expsum = segm_expsum * exp(segm_max - overall_max)`
- `overall_expsum = sum(rescaled segm_expsum)`
- rescaled partial outputs with the same `exp(segm_max - overall_max)` factor
- final `acc = sum(rescaled segm_output) / overall_expsum`

This is the same mathematical reason DCP and cascade attention need LSE-aware merging: partial softmax states must be combined with their normalization factors.

## FP8 Output

If `USE_FP8` is true:

- `acc = acc * out_scale_inv`
- `acc` is clamped to FP8 min/max

Then it stores to `output_ptr`.

## Why It Matters

Without `reduce_segments(...)`, 3D decode would produce one partial output per segment.

The reduction kernel turns those partials into the exact full-sequence attention result.

This is why the 3D decode path is a pair of kernels, not just one launch.

## Key Files

- `vllm/v1/attention/ops/triton_unified_attention.py`
- `vllm/v1/attention/ops/triton_attention_helpers.py`

