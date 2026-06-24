# TurboQuant KV Cache Shape And Layout

TurboQuant uses a backend-specific packed cache rather than the standard logical shape with a separate K/V axis.

## Logical Tensor Shape

`TurboQuantAttentionBackend.get_kv_cache_shape(...)` in `vllm/v1/attention/backends/turboquant_attn.py` returns:

```text
(num_blocks, block_size, num_kv_heads, slot_size_aligned)
```

The tensor dtype is `torch.uint8`, so the final dimension is measured directly in bytes.

A standard backend commonly uses a shape such as `(2, num_blocks, block_size, num_kv_heads, head_size)`, where the leading dimension selects K or V. TurboQuant instead stores K bytes immediately followed by V bytes in one token-head slot:

```text
[packed key | key metadata | packed value | value scale | value minimum | optional padding]
```

There is no leading K/V dimension.

## Slot Size

`TurboQuantConfig.key_packed_size`, `value_packed_size`, `slot_size`, and `slot_size_aligned` in `vllm/model_executor/layers/quantization/turboquant/config.py` define the final dimension.

`slot_size_aligned` rounds an odd total up to an even number. The even size allows generic vLLM code to represent an effective half-slot head size as an integer where required by cache allocation plumbing.

For an MSE-key preset, one slot is:

```text
byte 0                                      byte MSE_BYTES - 1: packed centroid indices
byte MSE_BYTES                              byte MSE_BYTES + 1: fp16 key norm
byte KPS                                    byte KPS + VAL_DATA_BYTES - 1: packed values
byte KPS + VAL_DATA_BYTES                   next 1 byte: fp16 value scale
byte KPS + VAL_DATA_BYTES + 2               next 1 byte: fp16 value minimum
remaining byte, if any: alignment padding
```

For FP8 keys, the first `D` bytes are FP8 key values and there is no key norm.

## Concrete Example

For `D = 128` and `turboquant_k3v4_nc`:

```text
MSE index bytes = ceil(128 * 3 / 8) = 48
key bytes = 48 + 2 norm bytes = 50
value data bytes = ceil(128 * 4 / 8) = 64
value bytes = 64 + 2 scale bytes + 2 minimum bytes = 68
slot bytes = 50 + 68 = 118
```

The cache shape is therefore `(num_blocks, block_size, num_kv_heads, 118)`.

## Cache Spec And Page Accounting

`Attention.get_kv_cache_spec(...)` in `vllm/model_executor/layers/attention/attention.py` creates `TQFullAttentionSpec` when the layer's cache dtype starts with `turboquant_`.

`TQFullAttentionSpec` is defined in `vllm/v1/kv_cache_interface.py`. It overrides `real_page_size_bytes` with:

```text
block_size * num_kv_heads * tq_slot_size
```

This override is necessary because the standard full-attention formula assumes separate full-precision K and V vectors and would substantially overallocate packed TurboQuant pages.

`TQFullAttentionSpec.merge(...)` requires every spec in the merged group to have the same `tq_slot_size`, preventing incompatible presets or dimensions from sharing one allocation interpretation.

The hybrid-model page-size calculation in `vllm/platforms/interface.py` also handles `TQFullAttentionSpec`. If boundary layers use native cache, it takes the maximum of the TurboQuant and native page sizes where shared hybrid-cache padding requires one safe page size.

## Slot Mapping On Writes

The store kernels receive `slot_mapping` with shape `[N]`, one flat cache slot for every current token.

For a mapping value `slot` and framework `BLOCK_SIZE`:

```text
physical_block = slot // BLOCK_SIZE
offset_in_block = slot % BLOCK_SIZE
```

The token-head byte base is then:

```text
slot_base = physical_block * stride_cache_block
          + offset_in_block * stride_cache_pos
          + kv_head * stride_cache_head
```

This is a direct physical write address. A negative slot means the token must not be cached, and the store program returns before writing.

## Block Tables On Reads

Decode does not use the write-time slot mapping. It receives `block_table[B, max_num_blocks]`, where each row maps a request's logical page number to a physical cache block.

For logical token position `t`:

```text
logical_page = t // BLOCK_SIZE
page_offset = t % BLOCK_SIZE
physical_block = block_table[request, logical_page]
```

The kernel combines that physical block with `page_offset` and the KV-head index to recover the same slot base used by the store kernel.

`seq_lens[B]` tells decode how many logical positions in each block-table row are valid, including how much of the final page is occupied. The block table itself does not encode final-page occupancy.

## Strides

TurboQuant does not override `get_kv_cache_stride_order(...)`, so physical dimension order follows the logical tensor shape unless allocation infrastructure applies an external layout policy.

Because each element is one uint8 byte, `kv_cache.stride(i)` is numerically both an element stride and a byte stride. The launchers pass strides directly into Triton without multiplying by an element size.

The inherited `get_kv_cache_block_dim(...)` probes `get_kv_cache_shape(...)` and discovers dimension zero as the physical-block dimension.

The backend does not override `get_required_kv_cache_layout(...)`, so it returns `None`; the packed tensor shape, rather than a generic NHD/HND layout enum, is the authoritative contract.

`get_preferred_block_size(...)` is inherited from `AttentionBackend`. It retains a compatible framework default or chooses the smallest declared kernel block size when the default is incompatible.

## Mixed TurboQuant And Native Layers

Layers in `kv_cache_dtype_skip_layers` set their local cache dtype to `auto` during `Attention` construction. They therefore create a normal `FullAttentionSpec`, select a compatible native backend, and never pass their cache tensor to TurboQuant kernels.

This separation is mandatory: a standard backend cannot interpret the packed final dimension, and TurboQuant cannot interpret a standard K/V cache tensor.
