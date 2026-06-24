# TurboQuant Decode Launcher

`triton_turboquant_decode_attention(...)` in `vllm/v1/attention/ops/triton_turboquant_decode.py` launches a two-stage split-KV decode.

## Public Arguments

- `query: [B, Hq, D]` contains one decode query per batch row, or synthetic continuation queries.
- `kv_cache: [num_blocks, block_size, Hk, padded_slot]` is the packed uint8 cache.
- `block_table: [B, max_num_blocks]` maps logical pages to physical blocks.
- `seq_lens: [B]` gives each row's valid logical KV length.
- `Pi: [D, D]` is the Hadamard transform; the current launcher only needs it to derive `PiT` when that tensor was not supplied.
- `centroids: [2^mse_bits]` reconstructs MSE key coordinates.
- `scale` multiplies QK scores, normally `1 / sqrt(D)`.
- `mse_bits` selects 3-bit or 4-bit key index unpacking and is zero for FP8 keys.
- `key_packed_size` is the byte offset from slot start to packed value data.
- `value_quant_bits` selects 3-bit or 4-bit value unpacking.
- `key_fp8` selects direct FP8 key loads instead of centroid reconstruction.
- `norm_correction` requests unit renormalization of reconstructed centroid vectors.
- `PiT` optionally supplies a precomputed contiguous query transform.
- `mid_o_buf`, `output_buf`, and `lse_buf` optionally supply workspace-manager storage.
- `buf_holder` receives fallback allocations as dynamic attributes when shared workspace is unavailable.
- `max_num_kv_splits` fixes the stage-one split count and defaults to 32.

## Derived Dimensions

The launcher derives:

```text
B, Hq, D = query.shape
Hk = kv_cache.shape[2]
block_size = kv_cache.shape[1]
kv_group_size = Hq // Hk
```

`_get_layout(...)` caches `mse_bytes`, `val_data_bytes`, centroid count, and `BLOCK_D = next_power_of_2(D)` by `(D, mse_bits, value_quant_bits, key_packed_size)`.

## Query Preparation

For FP8 keys, the cache stores keys in the original post-RoPE coordinate space, so the launcher uses a contiguous raw query. The stage-one argument remains named `Q_rot_ptr`, but no rotation has occurred in this mode.

For MSE keys, cached key coordinates are in Hadamard space. The launcher computes:

```text
q_float = query.float()
q_rot = q_float @ PiT
```

`q_rot` is float32 and contiguous. Rotating Q once is cheaper than inverse-rotating every cached key before every score calculation.

## Fixed Split-KV Count

`NUM_KV_SPLITS` is assigned directly from `max_num_kv_splits`; it is not selected dynamically from the current sequence lengths.

The fixed count provides:

- A stable stage-one grid `(B, Hq, NUM_KV_SPLITS)`.
- A stable intermediate shape `[B, Hq, NUM_KV_SPLITS, D + 1]`.
- Predictable CUDA graph capture and replay.
- Potentially empty splits for short sequences, which stage one returns from and stage two skips.

## Intermediate Buffer

`mid_o` has float32 shape `[B, Hq, NUM_KV_SPLITS, D + 1]`.

For each valid split, the first `D` entries hold that split's normalized attention output and the final entry holds its softmax LSE.

The launcher uses a supplied workspace view when it is large enough; otherwise it allocates a tensor and optionally attaches it to `buf_holder` to preserve its lifetime.

## Stage-One Launch

The launcher uses:

```text
BLOCK_KV = 4
grid = (B, Hq, NUM_KV_SPLITS)
num_warps = 1
num_stages = 1
```

The source argument comment says `BLOCK_KV` is a token tile, and the actual launcher value is four. Each program owns one batch row, one query head, and one sequence split.

## Stage-Two Buffers

Output has shape `[B, Hq, D]` and the same dtype as Q. LSE scratch has float32 shape `[B, Hq]`.

Workspace views are used when available and compatible; otherwise the launcher allocates fallback tensors.

## Stage-Two Launch

The reduction grid is `(B, Hq)` with one program per final output row.

It invokes `_fwd_kernel_stage2(...)` from `vllm/v1/attention/ops/triton_decode_attention.py` with four warps and two stages.

The launcher returns only `output`. The populated `lse` tensor is internal scratch and is not returned to `TurboQuantAttentionImpl`.

## Workspace Manager Integration

`TurboQuantAttentionImpl._decode_attention(...)` requests three simultaneous views from `WorkspaceManager` in `vllm/v1/worker/workspace.py`:

```text
mid_o: [B, Hq, S, D + 1] float32
output: [B, Hq, D] query dtype
lse: [B, Hq] float32
```

The manager packs aligned views into one reusable byte allocation. Layers execute sequentially, so one workspace can serve many layers without multiplying scratch memory by layer count.
