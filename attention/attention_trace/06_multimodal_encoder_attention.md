# Multimodal Encoder Attention

This note covers the multimodal encoder attention branch separately from the generic decoder attention path.

The reason to split it out is simple:

- generic decoder attention is paged-KV-cache attention
- multimodal encoder attention is usually ViT-style attention without paged KV

They share some backend naming and platform ideas, but the execution model is different.

## Main Entry Point

The key file is:

- `vllm/model_executor/layers/attention/mm_encoder_attention.py`

The main operator is:

- `MMEncoderAttention`

This is registered as a custom op named:

- `mm_encoder_attn`

## How Backend Selection Differs

Generic decoder attention chooses a backend class via:

- `get_attn_backend(...)`
- `current_platform.get_attn_backend_cls(...)`

Multimodal encoder attention instead chooses a ViT-specific backend enum via:

- `get_vit_attn_backend(...)` in `vllm/model_executor/models/vision.py`
- `current_platform.get_vit_attn_backend(...)`

So this branch does not reuse the generic paged decoder selector directly.

## What The MM Encoder Path Optimizes For

The multimodal encoder path is built for:

- no paged KV cache
- packed / variable-length encoder sequences
- ViT-friendly backend choices

That is why the supported backends are a separate platform method:

- `get_supported_vit_attn_backends()`

On CUDA this usually prioritizes a set including:

- `FLASH_ATTN`
- `TRITON_ATTN`
- `TORCH_SDPA`
- `FLASHINFER`

## Runtime Shape Differences

`MMEncoderAttention` computes encoder-style sequence metadata such as:

- cumulative sequence lengths
- max sequence length
- optional padded sequence lengths for backend-specific paths

There is no paged decoder KV-cache ownership story here.

That means this branch bypasses most of the generic decoder concepts discussed in the other notes:

- no paged KV block table
- no slot-mapping-driven cache update
- no decoder-style metadata builders

## Multimodal Config Hooks

The multimodal encoder path can also be influenced by multimodal config values.

Important ones include:

- `mm_encoder_attn_backend`
- `mm_encoder_attn_dtype`

`vision.py` reads the multimodal config and forwards the backend override into `get_vit_attn_backend(...)`.

## FP8-Specific Path

`MMEncoderAttention` also has a dedicated FP8 path.

Important points:

- FP8 enablement is checked against multimodal config
- the current implementation ties FP8 support to FlashInfer cuDNN support and native FP8-capable hardware
- encoder attention scales may be loaded or auto-saved separately from decoder attention KV-cache quantization concerns

This is another reason not to mix this branch into the generic decoder note:

- the quantization and backend constraints are different

## Relationship To The Generic Attention Architecture

The two branches share these ideas:

- backend choice is platform-mediated
- custom ops are used at the execution boundary
- backend families include FlashAttention, Triton, and FlashInfer

But they diverge in the most important runtime assumptions:

- decoder path: paged KV cache plus per-step metadata
- MM encoder path: direct encoder attention over packed sequences

If you are adding a decoder attention backend plugin, this file is useful for comparison, but it is not the main integration path.

## Best Files To Read Next

- `vllm/model_executor/layers/attention/mm_encoder_attention.py`
- `vllm/model_executor/models/vision.py`
- `vllm/platforms/cuda.py`
