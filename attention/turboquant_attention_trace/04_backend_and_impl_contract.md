# TurboQuant Backend And Implementation Contract

The backend class declares allocation and selection capabilities, while the implementation class owns runtime computation.

## Backend Class

`TurboQuantAttentionBackend` is defined in `vllm/v1/attention/backends/turboquant_attn.py`.

- `get_name()` returns `TURBOQUANT`, which must match the registry enum name.
- `get_impl_cls()` returns `TurboQuantAttentionImpl`.
- `get_builder_cls()` returns `TurboQuantMetadataBuilder`.
- `get_kv_cache_shape(...)` computes the packed uint8 shape from `TurboQuantConfig` by fetching aligned slot size.
- `get_supported_kernel_block_sizes()` returns 16, 32, 64, and 128.
- `supports_kv_cache_dtype(...)` accepts non-null strings beginning with `turboquant_`.
- `supports_head_size(...)` accepts any positive effective head size.
- `supports_attn_type(...)` accepts only `AttentionType.DECODER`.
- `supports_per_head_quant_scales()` returns false because this backend uses its own packed quantization metadata.

`accept_output_buffer = True` says the implementation can write into the output tensor allocated by the shared `Attention` layer.

`forward_includes_kv_cache_update = False` says cache mutation must occur through `do_kv_cache_update(...)` before `forward(...)`.

## Inherited Capabilities

The backend inherits false for sinks, MM prefix, non-causal decoder attention, sparse attention, and batch invariance.

It inherits true for generic compute capability and KV connector support, although packed-layout compatibility remains an integration requirement.

It does not declare a required generic `KVCacheLayoutType`; its concrete shape and `TQFullAttentionSpec` define the interpretation.

## Implementation Construction

`Attention.__init__(...)` in `vllm/model_executor/layers/attention/attention.py` obtains the implementation class with:

```text
impl_cls = self.attn_backend.get_impl_cls()
self.impl = impl_cls(...)
```

The arguments received by `TurboQuantAttentionImpl.__init__(...)` are:

- `num_heads`: query heads local to this tensor-parallel rank.
- `head_size`: model head dimension `D`.
- `scale`: attention score multiplier, normally `1 / sqrt(D)`.
- `num_kv_heads`: KV heads local to this rank.
- `alibi_slopes`: accepted by the common interface but not used.
- `sliding_window`: accepted by the common interface but not used by TurboQuant masking.
- `kv_cache_dtype`: selects one of the four TurboQuant configurations.
- `logits_soft_cap`: accepted but not used by the TurboQuant score kernel.
- `attn_type`: validated as decoder attention.
- `kv_sharing_target_layer_name`: accepted by the interface; cache-update dispatch handles sharing outside the implementation.

The implementation computes `num_kv_groups = num_heads // num_kv_heads`. During decode, query head `h` maps to KV head `h // num_kv_groups`, which implements MQA and GQA.

## Precomputed Layout Constants

The constructor builds `TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)` and precomputes in `vllm/vllm/v1/attention/backends/turboquant_attn.py`:

- `_mse_bytes = ceil(D * key_mse_bits / 8)` for MSE keys, or `D` for FP8 keys.
- `_val_data_bytes = ceil(D * value_bits / 8)`.
- `_n_centroids`, although the runtime kernels primarily derive the required count from the bit width.
- `fa_version` from `get_flash_attn_version(...)` for prefill delegation.
- `max_num_kv_splits` from `attention_config.tq_max_kv_splits_for_cuda_graph`.

The fixed split count makes the decode launch grid and scratch-buffer shapes stable under CUDA graph capture.

## Layer-Resident Quantization State

`_ensure_on_device(layer, device)` initializes these dynamic layer attributes once:

- `_tq_PiT`: float32 normalized Hadamard matrix used for key/query rotation.
- `_tq_Pi`: the same matrix used as the inverse because it is symmetric.
- `_tq_Pi_half`: fp16 copy used by large continuation-prefill inverse rotation.
- `_tq_centroids`: float32 Lloyd-Max reconstruction values.
- `_tq_midpoints`: float32 decision boundaries between sorted centroids.
- `_tq_cached`: initialization marker.

The Hadamard matrix is also cached across layers by `_build_hadamard_cached(D, device_str)`.

## Runtime Impl Capabilities

`supports_quant_query_input = False` means the shared attention layer must not pre-quantize Q using its generic query-quantization path.

The implementation inherits `can_return_lse_for_decode = False`. Although the internal reduction allocates an LSE tensor, `forward(...)` returns only attention output and does not expose LSE to distributed decode combination.

The implementation inherits `supports_pcp = False`, so prefill context parallelism is rejected by runtime compatibility validation.

`do_kv_cache_update(...)` and `forward(...)` are the two essential runtime methods. The former compresses current K/V into cache, and the latter computes attention after that mutation has been ordered.

## AttentionLayer Protocol

`AttentionLayer` is defined as a protocol in `vllm/v1/attention/backend.py` and is the type accepted by `TurboQuantAttentionImpl.forward(...)`.

The protocol exposes the common attention layer's Q/K/V scale tensors and its forward signature. TurboQuant does not consume the generic scale fields because its packed cache carries TurboQuant-specific key norms and value affine metadata.

At runtime, the concrete object is the shared `Attention` module from `vllm/model_executor/layers/attention/attention.py`. TurboQuant additionally attaches `_tq_*` tensors dynamically after treating the protocol-typed layer as `Any`.
