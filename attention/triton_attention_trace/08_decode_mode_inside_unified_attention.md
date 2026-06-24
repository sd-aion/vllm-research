# Decode Mode Inside Unified Attention

This note explains how decoder decode is handled by `unified_attention(...)`.

## No Separate Standard TRITON_ATTN Decode Call

Standard `TritonAttentionBackend` does not call `triton_decode_attention.py`.

For standard decoder attention, decode uses:

- `unified_attention(...)`
- `kernel_unified_attention(...)`
- optionally `reduce_segments(...)`

So the standard paged decode path is a mode inside the unified kernel.

## Decode Shape

Decode usually has:

- `max_seqlen_q == 1`
- one query token per active request
- large `seq_lens`
- block table entries pointing to existing cached context

The query tensor contains the current decode token Q values.

The K/V cache already contains previous context and is updated with the new token before attention runs.

## 2D Decode

The 2D grid is:

- `(total_num_q_blocks, num_kv_heads)`

Each program handles one query block for one KV head.

The program loops over KV tiles for that request and computes online softmax over the full valid sequence.

2D mode is used when:

- the batch is large enough.
- prefill is present.
- segment scratch buffers are absent.
- batch invariance is enabled.
- `max_seqlen_q > 1`.

## 3D Decode

3D mode is used for small decode batches when all required segment scratch tensors exist.

The 3D grid is:

- `(total_num_q_blocks, num_kv_heads, num_par_softmax_segments)`

Each segment processes a slice of the KV tiles.

This increases parallelism when the 2D grid would be too small.

## Segment Count

The default segment count is:

- `NUM_PAR_SOFTMAX_SEGMENTS = 16`

This value comes from `triton_attn.py`.

The metadata builder allocates scratch buffers sized for this segment count.

## Why 3D Needs A Reduction Kernel

Each segment computes a partial softmax over only part of the KV sequence.

A partial softmax output is normalized over its segment, not over the full sequence.

So the segments cannot be summed directly.

The kernel stores each segment's:

- partial output accumulator
- row max `M`
- exp sum `L`

Then `reduce_segments(...)` reconstructs the globally normalized output.

## Decode With Sliding Window

Decode can use sliding window through `SLIDING_WINDOW`.

The loop bounds helper prunes KV tiles so the kernel skips tiles outside the allowed window.

The mask helper also applies token-level sliding-window filtering.

## Decode With MM Prefix

MM prefix can extend the causal/sliding mask with bidirectional ranges.

When `mm_prefix_range` is present:

- `USE_MM_PREFIX=True`
- `compute_kv_seq_mask(...)` ORs valid bidirectional ranges into the mask.

This is a real feature of the unified Triton kernel.

## Decode With Sinks

Sinks initialize the online softmax state through `init_softmax_M(...)`.

In 3D mode only segment 0 includes sink contribution.

This prevents sink values from being counted once per segment.

## Key Files

- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/ops/triton_unified_attention.py`
- `vllm/v1/attention/ops/triton_attention_helpers.py`

