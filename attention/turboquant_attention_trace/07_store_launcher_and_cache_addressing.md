# Store Launcher And Cache Addressing

`triton_turboquant_store(...)` in `vllm/v1/attention/ops/triton_turboquant_store.py` selects the key-storage kernel and passes both paths to the shared value-quantization helper.

## Inputs

- `key: [N, H, D]` contains current post-RoPE keys in fp16 or bf16.
- `value: [N, H, D]` contains current values in fp16 or bf16.
- `kv_cache: [num_blocks, block_size, H, padded_slot]` is uint8 packed storage.
- `slot_mapping: [N]` contains one flat physical cache slot per token.
- `PiT: [D, D]` is the float32 Hadamard transform used by MSE keys.
- `midpoints: [2^mse_bits - 1]` contains sorted Lloyd-Max decision boundaries.
- `mse_bits` is zero for FP8 keys and three or four for MSE keys.
- `key_packed_size` is the byte offset where the packed value starts.
- `value_quant_bits` is three or four.
- `key_fp8` selects `_tq_fused_store_fp8` instead of `_tq_fused_store_mse`.

The launcher reads `N`, `H`, and `D` from K and computes `NH = N * H`.

## One Program Per Token-Head Pair

Both store kernels launch a one-dimensional grid `(NH,)`.

For program ID `pid`:

```text
token_idx = pid // H
head_idx = pid % H
```

This means one Triton program owns a complete key vector and value vector for one token and one KV head.

## Slot Mapping

The kernel loads `slot = slot_mapping[token_idx]`.

If `slot < 0`, the program returns without writing. Negative mappings are used for tokens that participate in computation but should not occupy a KV-cache location.

Otherwise:

```text
blk = slot // BLOCK_SIZE
off = slot % BLOCK_SIZE
slot_base = blk * stride_cache_block
          + off * stride_cache_pos
          + head_idx * stride_cache_head
```

`slot_base` is a byte offset because the flattened cache pointer is uint8 and each stride unit is one byte.

## Example

Suppose `BLOCK_SIZE = 16`, `slot_mapping[token_idx] = 37`, and the program handles KV head 2.

```text
blk = 37 // 16 = 2
off = 37 % 16 = 5
slot_base = 2 * stride_cache_block + 5 * stride_cache_pos + 2 * stride_cache_head
```

The key begins at `slot_base`, and the value begins at `slot_base + key_packed_size`.

## Compile-Time Sizes

The launcher computes:

- `BLOCK_D = next_power_of_2(D)` for vector loads and reductions.
- `mse_bytes = ceil(D * mse_bits / 8)`.
- `val_data_bytes = ceil(D * value_quant_bits / 8)`.
- `BLOCK_VAL = next_power_of_2(val_data_bytes)`.
- `block_grp = next_power_of_2(D // 8)` for groups of eight 3-bit values.

Masked lanes beyond `D` allow non-power-of-two head dimensions to use power-of-two Triton vectors.

## FP8 Key Dispatch

When `key_fp8` is true, the launcher makes contiguous `[NH, D]` views of K and V, selects the platform FP8 format with `_use_fp8_e4b15(...)`, and launches `_tq_fused_store_fp8`.

The key conversion and cache scatter happen inside Triton. No external key-rotation GEMM is needed.

## MSE Key Dispatch

When `key_fp8` is false, the launcher performs these PyTorch operations before Triton:

```text
k_flat = key.float().reshape(NH, D)
norms = ||k_flat||_2 per row
x_hat = k_flat / (norms + 1e-8)
y = x_hat @ PiT
v_flat = value.float().reshape(NH, D)
```

It then launches `_tq_fused_store_mse` to bucketize and pack `y`, store the norms, quantize values, and scatter all bytes into cache.

The matrix multiplication is intentionally outside Triton because the implementation expects cuBLAS or the active PyTorch GEMM backend to perform the dense Hadamard multiplication efficiently.

## Relationship To Attention

The store launcher only mutates cache. It does not compute attention output.

The shared custom-op dependency ensures this launcher completes before prefill or decode reads the newly written locations.
