# Value Quantization And Packing

`_store_quantized_value(...)` in `vllm/v1/attention/ops/triton_turboquant_store.py` is an inlined Triton helper called by both key-store kernels.

## Inputs And Addressing

The helper receives the raw value pointer, flattened source-row base, packed cache pointer, destination `slot_base`, vector offsets, masks, and compile-time layout constants.

The packed value begins at:

```text
val_cache_offset = KPS
```

`KPS` is the complete packed-key size, including the fp16 norm for an MSE key.

## Per-Vector Affine Quantization

The helper loads one complete value vector as float32 and computes the minimum and maximum only over valid dimensions.

For `b` bits:

```text
QMAX = 2^b - 1
v_scale = max((v_max - v_min) / QMAX, 1e-8)
q[d] = clamp(int((v[d] - v_min) / v_scale + 0.5), 0, QMAX)
```

The minimum scale avoids division by zero for constant or nearly constant vectors.

Decode reconstructs a coordinate with:

```text
v_hat[d] = q[d] * v_scale + v_min
```

The scale and minimum are converted to fp16 and stored as raw little-endian bytes after the packed integer data.

## Four-Bit Packing

For 4-bit values, two adjacent quantized coordinates fit in one byte.

For a pair `(q0, q1)`:

```text
packed_byte = (q0 & 0xF) | ((q1 & 0xF) << 4)
```

The lower nibble holds the even coordinate and the upper nibble holds the odd coordinate.

For `D = 128`, the data region uses `128 / 2 = 64` bytes.

## Three-Bit Packing

For 3-bit values, groups of eight coordinates use exactly 24 bits, or three bytes.

For `q0` through `q7`:

```text
packed24 = q0 | q1<<3 | q2<<6 | q3<<9 | q4<<12 | q5<<15 | q6<<18 | q7<<21
b0 = packed24 & 0xFF
b1 = (packed24 >> 8) & 0xFF
b2 = (packed24 >> 16) & 0xFF
```

Values may cross byte boundaries. The decode kernel therefore loads two neighboring bytes and right-shifts the combined 16-bit window for an arbitrary coordinate.

For `D = 128`, there are 16 groups and the data region uses `16 * 3 = 48` bytes.

## Metadata Tail

For both bit widths:

```text
scale_offset = KPS + VAL_DATA_BYTES
```

The following four bytes are:

```text
[scale low byte | scale high byte | minimum low byte | minimum high byte]
```

Scale and minimum are fp16 bit patterns, not integer quantization parameters.

## Small Four-Bit Example

For a two-element vector `[2.0, 6.0]`:

```text
v_min = 2.0
v_max = 6.0
scale = 4 / 15
q = [0, 15]
packed byte = 0xF0
```

Decode produces approximately `[0 * scale + 2, 15 * scale + 2] = [2, 6]`.

## Numerical Consequences

Every token-head value vector has its own scale and minimum, so one outlier affects only that vector rather than an entire layer or cache tensor.

The fp16 metadata introduces rounding in addition to integer quantization. The attention kernel reconstructs values as float32 before multiplying by softmax probabilities and accumulates the weighted result in float32.
