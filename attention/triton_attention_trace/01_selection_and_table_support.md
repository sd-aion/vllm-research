# Triton Attention Selection And Table Support

This note maps how vLLM selects `TRITON_ATTN` and how the row in `docs/design/attention_backends.md` maps to code.

## Registry Entry

The backend enum entry lives in `vllm/v1/attention/backends/registry.py`:

- `AttentionBackendEnum.TRITON_ATTN = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"`

That resolves to `TritonAttentionBackend` through `AttentionBackendEnum.get_class()`.

## Launch Inputs

The user can request Triton attention through the normal attention backend controls:

- `--attention-backend TRITON_ATTN`
- `--attention-config.backend TRITON_ATTN`
- `-ac.backend TRITON_ATTN`
- Python `AttentionConfig(backend=AttentionBackendEnum.TRITON_ATTN)`
- Python `LLM(..., attention_backend="TRITON_ATTN")`

These inputs normalize into `AttentionConfig` from `vllm/config/attention.py`.

## Explicit Selection

When selected explicitly, the platform validates `TritonAttentionBackend` against the layer's selector config.

The relevant path is:

1. `Attention` in `vllm/model_executor/layers/attention/attention.py` calls `get_attn_backend(...)`.
2. `get_attn_backend(...)` in `vllm/v1/attention/selector.py` builds `AttentionSelectorConfig`.
3. `current_platform.get_attn_backend_cls(...)` validates the selected backend.
4. `TritonAttentionBackend.validate_configuration(...)` checks capability methods inherited from `AttentionBackend`.
5. If valid, the backend class path is returned.

Explicit selection is therefore still subject to validation.

## Auto Selection

On CUDA, `TRITON_ATTN` is usually not the first candidate, but it is in the standard attention priority list.

For standard MHA/MQA/GQA:

- Blackwell / SM 10.x: `FLASHINFER`, `FLASH_ATTN`, `TRITON_ATTN`, `FLEX_ATTENTION`, `TURBOQUANT`.
- Ampere/Hopper / SM 8.x-9.x: `FLASH_ATTN`, `FLASHINFER`, `TRITON_ATTN`, `FLEX_ATTENTION`, `TURBOQUANT`.

On ROCm and XPU, platform code can prefer or fall back to `TRITON_ATTN` more often because FlashAttention and FlashInfer support differs by platform.

Some model config paths can force `TRITON_ATTN` when the model has head dimensions or attention requirements not covered by other backends.

## Feature Table Row

The `docs/design/attention_backends.md` row says:

- Backend: `TRITON_ATTN`
- Dtypes: fp16, bf16, fp32
- KV Dtypes: `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, `int8_per_token_head`, `fp8_per_token_head`
- Block Sizes: `%16`
- Head Sizes: `Any`
- Sink: supported
- Non-Causal: not supported
- MM Prefix: supported
- DCP: not supported
- Attention Types: all
- Compute Capability: any

## Mapping To Code

Dtypes map to:

- `TritonAttentionBackend.supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]`

KV dtypes map to:

- `TritonAttentionBackend.supported_kv_cache_dtypes`

Block sizes map to:

- `get_supported_kernel_block_sizes() -> [MultipleOf(16)]`
- `supports_block_size(...) -> block_size % 16 == 0`
- `get_kv_cache_shape(...)` also rejects block sizes not divisible by 16.

Head sizes are shown as `Any`, but the method says:

- `supports_head_size(head_size) -> head_size >= 32`

Sink support maps to:

- `supports_sink() -> True`
- runtime `sinks` are passed into `unified_attention(...)`.

Non-causal is false because:

- `TritonAttentionBackend` does not override `supports_non_causal()`, so decoder non-causal support remains false.
- `unified_attention(...)` asserts `causal`.

MM prefix is true because:

- `supports_mm_prefix() -> True`
- `TritonAttentionMetadata.compute_mm_prefix_range_tensor(...)` creates kernel input.
- `compute_kv_seq_mask(...)` ORs bidirectional mm-prefix ranges into the normal mask.

DCP is false because:

- `TritonAttentionImpl` does not set `can_return_lse_for_decode = True`.
- DCP runtime validation requires decode softmax LSE support.

Attention types are all because:

- `supports_attn_type(...)` accepts decoder, encoder, encoder-only, and encoder-decoder.

Compute capability is any because:

- `supports_compute_capability(...) -> True`

## Key Files

- `vllm/v1/attention/backends/registry.py`
- `vllm/config/attention.py`
- `vllm/v1/attention/selector.py`
- `vllm/platforms/cuda.py`
- `vllm/platforms/rocm.py`
- `vllm/platforms/xpu.py`
- `vllm/v1/attention/backends/triton_attn.py`
- `docs/design/attention_backends.md`

