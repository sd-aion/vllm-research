# FP8 Store Kernel

`_tq_fused_store_fp8(...)` in `vllm/v1/attention/ops/triton_turboquant_store.py` handles the `turboquant_k8v4` preset.

## Kernel Arguments

- `Key_ptr: [NH, D]` points to contiguous raw fp16/bf16 keys.
- `Value_ptr: [NH, D]` points to contiguous raw fp16/bf16 values.
- `KV_cache_ptr: [total_bytes]` is a flattened uint8 cache view.
- `Slot_mapping_ptr: [N]` selects one destination slot per token.
- `stride_cache_block` is the byte distance between physical cache blocks.
- `stride_cache_pos` is the byte distance between token positions in one block.
- `stride_cache_head` is the byte distance between KV-head slots at one token position.
- `D`, `H`, and `BLOCK_SIZE` define tensor and paging dimensions.
- `BLOCK_D` is the power-of-two vector width used by the program.
- `KPS` is the byte offset from slot start to value start; in this mode it equals `D`.
- `VQB`, `VAL_DATA_BYTES`, `BLOCK_VAL`, and `BLOCK_GRP` configure value packing.
- `FP8_E4B15` chooses the Triton FP8 interpretation.

`BLOCK_VAL` is carried as part of the shared value-helper interface, while the current packing branches form their address vectors from `BLOCK_D`, `VAL_DATA_BYTES`, and `BLOCK_GRP`.

## Program Mapping

The grid is `(N * H,)`, and program `pid` owns one token-head pair.

The program resolves `token_idx`, `head_idx`, and `slot_base` exactly as described in `07_store_launcher_and_cache_addressing.md`.

## FP8 Key Conversion

The program loads up to `BLOCK_D` key coordinates, masking lanes where `d >= D`.

It converts each valid coordinate to either `tl.float8e4b15` or `tl.float8e4nv`, then bitcasts that FP8 value to `tl.uint8` and stores one byte per coordinate.

The bitcast is important: the cache is typed as uint8 for generic allocation and byte packing, but those bytes represent FP8 values and must be bitcast back to the matching FP8 type during decode.

No explicit scale is stored for FP8 keys. Quantization is entirely determined by the selected FP8 format's representable values.

## Why The Format Depends On Hardware

`_use_fp8_e4b15(device)` in `triton_turboquant_decode.py` returns one for CUDA-like capabilities below 8.9 and zero otherwise.

Store and decode both use this result. A byte written as E4B15 must be read as E4B15; interpreting the same bit pattern as E4NV could produce a different value.

## Fused Value Store

After writing key bytes, the same program calls `_store_quantized_value(...)` with the already computed source and destination coordinates.

For `turboquant_k8v4`, values use 4-bit affine quantization, so two value coordinates are packed per byte and fp16 scale/minimum metadata follows.

## Complete D=128 Slot

For `turboquant_k8v4` and `D = 128`:

```text
bytes 0..127: FP8 key coordinates
bytes 128..191: 64 packed 4-bit value bytes
bytes 192..193: fp16 value scale
bytes 194..195: fp16 value minimum
```

The complete slot is 196 bytes.

## Launch Configuration

The launcher uses four warps and one stage. There is no autotuning table in this implementation.

The kernel's parallelism comes from the number of token-head pairs; each program performs vector conversion, reductions for value min/max, packing, and scattered cache stores.
