# TurboQuant Runtime Call Path

This document follows one layer invocation from the model's `Attention` module to the TurboQuant store and attention paths.

## Construction

`Attention.__init__(...)` in `vllm/model_executor/layers/attention/attention.py` resolves `TurboQuantAttentionBackend`, obtains `TurboQuantAttentionImpl`, and stores both on the layer.

The layer also creates a `TQFullAttentionSpec`, and the model runner later attaches the allocated packed KV-cache tensor through forward context.

## Forward Context

The model definition calls `Attention.forward(...)` without explicitly passing block tables, sequence lengths, slot mappings, or KV-cache tensors.

The model runner installs those runtime values in the forward context from `vllm/forward_context.py`. `get_attention_context(...)` in `attention.py` retrieves the layer-specific metadata, layer object, cache tensor, and slot mapping when a registered custom op executes.

This indirection keeps model forward signatures tensor-oriented while allowing continuous batching metadata to change every scheduler step.

## Shape Preparation

The shared `Attention.forward(...)` reshapes tensors before entering custom ops:

```text
query: [-1, num_query_heads, head_size]
key:   [-1, num_kv_heads, head_size]
value: [-1, num_kv_heads, head_size_v]
output:[-1, num_query_heads, head_size_v]
```

TurboQuant currently assumes key and value head dimensions equal `head_size`.

## Separate KV Cache Update

Because `TurboQuantAttentionBackend.forward_includes_kv_cache_update = False`, the shared layer invokes `unified_kv_cache_update(...)` before invoking attention.

The custom op resolves forward context and calls:

```text
self.impl.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
```

`TurboQuantAttentionImpl.do_kv_cache_update(...)` slices K/V to `num_actual_tokens`, reshapes them to `[N, Hk, D]`, initializes quantization state, and calls `_store_kv(...)`.

`_store_kv(...)` delegates compression and cache writes to `triton_turboquant_store(...)` in `vllm/v1/attention/ops/triton_turboquant_store.py`.

## Ordering Dependency

The cache update mutates storage that the attention op reads, but the output of the update is otherwise unused.

`Attention.forward(...)` passes a dummy tensor returned by the update into `unified_attention_with_output(...)` as `kv_cache_dummy_dep`. The attention op deletes the value, but the tensor dependency prevents `torch.compile` from reordering attention before the cache mutation.

This ordering is especially important for continuation prefill and decode because the current token's K/V must already be visible in the paged cache.

## Direct And Opaque Custom-Op Calls

When `use_direct_call` is true, Python calls `unified_kv_cache_update(...)` and `unified_attention_with_output(...)` directly.

Otherwise it calls the registered `torch.ops.vllm` custom ops. The opaque route gives compilation and graph systems a stable operator boundary, while the direct route avoids dispatcher overhead in execution modes where that boundary is unnecessary.

Both routes invoke the same implementation methods and preserve the same dummy dependency.

## Attention Forward

`unified_attention_with_output(...)` calls `TurboQuantAttentionImpl.forward(...)` with Q/K/V, the packed cache, TurboQuant metadata, and the preallocated output.

`forward(...)` first handles padding and empty cases, reshapes Q to `[N, Hq, D]`, and ensures Hadamard and centroid tensors exist on the device.

It then selects one of three batch-level cases:

- Pure decode calls `_decode_attention(...)` for the complete batch.
- Pure prefill calls `_prefill_attention(...)` with the current raw K/V.
- Mixed execution constructs decode and prefill metadata views, runs each path on its contiguous slice, and writes both results into one temporary output.

Finally, it writes `[N, Hq, D]` into the supplied 2D or 3D output buffer and returns that same buffer.

## Why Prefill Still Receives Key And Value

The cache update has already compressed K/V, but first-chunk prefill deliberately computes from the original K/V tensors to avoid quantization error and decompression overhead during the high-throughput prompt computation.

Short continuation prefill reads the compressed cache through the decode kernel. Large continuation prefill dequantizes previously cached K/V but still uses raw K/V for the new chunk.

## Cache Sharing

The shared attention layer skips the update if `kv_sharing_target_layer_name` points to an earlier layer whose cache is reused.

TurboQuant itself does not implement a separate sharing protocol in `forward(...)`; it relies on the common layer and forward-context machinery to supply the correct cache tensor and ordering.
