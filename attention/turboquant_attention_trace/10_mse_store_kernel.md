# MSE Store Kernel

`_tq_fused_store_mse(...)` in `vllm/v1/attention/ops/triton_turboquant_store.py` handles 3-bit and 4-bit Lloyd-Max keys.

## Work Before The Kernel

`triton_turboquant_store(...)` converts keys to float32, computes one L2 norm per token-head vector, normalizes each row, and applies the Hadamard transform:

```text
norms: [NH, 1]
y: [NH, D] = normalized_keys @ PiT
```

Values are also converted to a float32 `[NH, D]` view.

The kernel therefore receives already rotated keys and does not perform a matrix multiplication.

## Kernel Arguments

- `Y_ptr: [NH, D]` contains normalized, rotated keys in float32.
- `Norms_ptr: [NH]` contains original key L2 norms in float32.
- `Value_ptr: [NH, D]` contains raw values in float32.
- `Midpoints_ptr: [N_CENTROIDS - 1]` contains sorted bucket boundaries.
- `KV_cache_ptr` and `Slot_mapping_ptr` control scattered packed-cache writes.
- `stride_cache_block` is the byte distance between physical cache blocks.
- `stride_cache_pos` is the byte distance between token positions in a block.
- `stride_cache_head` is the byte distance between KV-head slots at one position.
- `BLOCK_SIZE` converts each flat slot mapping into a physical block and an offset within that block.
- `D`, `H`, and `BLOCK_D` define source rows and vector width.
- `MSE_BYTES` is the number of bytes occupied by packed key indices.
- `KPS = MSE_BYTES + 2` includes the fp16 norm.
- Value constants configure the shared value helper.
- `MSE_BITS` is three or four, and `N_CENTROIDS = 2^MSE_BITS`.
- `BLOCK_GRP` is the padded number of eight-coordinate groups used by 3-bit packing.

## Binary-Search Bucketization

Each valid coordinate starts with a search interval over the midpoint array.

The kernel executes exactly `MSE_BITS` search iterations. At each iteration it loads the current midpoint and chooses the lower or upper half based on `y >= midpoint`.

For 3-bit keys, three comparisons choose one of eight centroids. For 4-bit keys, four comparisons choose one of sixteen centroids.

The resulting integer index is the quantized coordinate. Actual centroid values are not stored per token because every layer can reconstruct them from its shared centroid table.

## Four-Bit Key Packing

Two centroid indices are packed into each byte using the same low/high-nibble convention as 4-bit values.

For `D = 128`, the key-index region occupies 64 bytes.

## Three-Bit Key Packing

Eight centroid indices are packed into three bytes using a 24-bit register value.

For `D = 128`, the key-index region occupies 48 bytes.

## Norm Storage

The norm is converted to fp16, bitcast to uint16, and written as two uint8 bytes at offsets `MSE_BYTES` and `MSE_BYTES + 1`.

Decode multiplies the centroid dot product by this norm, restoring the magnitude discarded by normalization.

## Value Storage

The kernel calls `_store_quantized_value(...)` with `KPS` as the value offset, so value bytes begin after both key indices and norm metadata.

This combines key bucketization, index packing, norm storage, value reduction, value packing, and cache scattering in one Triton program after the external rotation GEMM.

## D=128 Four-Bit Example

For `turboquant_4bit_nc`:

```text
bytes 0..63: 128 four-bit centroid indices
bytes 64..65: fp16 key norm
bytes 66..129: 128 four-bit value indices
bytes 130..131: fp16 value scale
bytes 132..133: fp16 value minimum
```

The slot is 134 bytes.

## D=128 Three-Bit-Key Example

For `turboquant_k3v4_nc`:

```text
bytes 0..47: 128 three-bit centroid indices
bytes 48..49: fp16 key norm
bytes 50..113: 128 four-bit value indices
bytes 114..115: fp16 value scale
bytes 116..117: fp16 value minimum
```

The slot is 118 bytes.

## Norm Correction Is A Read-Time Choice

The store representation is the same whether norm correction is enabled or disabled. The preset's `norm_correction` flag controls how decode normalizes the reconstructed centroid vector before score computation.
