# Sliding Window And MM Prefix

This note covers the two mask/layout variants that most visibly alter the generic decoder-attention path:

- sliding-window attention
- `mm_prefix`

Both stay within the normal decoder attention architecture, but they change what metadata is needed and what a backend must support.

## Sliding-Window Attention

Sliding-window attention means a decoder token only attends to a bounded local history instead of the full causal history.

Conceptually:

- full causal attention: token `i` can see all `j <= i`
- sliding-window causal attention: token `i` can only see a bounded subset of recent `j <= i`

## Where Sliding Window Enters The Path

At layer construction time in `vllm/model_executor/layers/attention/attention.py`, the layer determines `sliding_window` from:

- `per_layer_sliding_window`, if provided
- otherwise `cache_config.sliding_window`
- otherwise `None`

At model-config level, `ModelConfig.get_sliding_window()` reads it from the HF text config.

Related model-side config nuance:

- `ModelConfig.disable_sliding_window` can force the HF sliding-window setting off even if the checkpoint exposes one

So sliding-window behavior is model/config driven before it becomes backend behavior.

## KV-Cache Spec Side

Sliding window also affects the KV-cache spec layer in `vllm/v1/kv_cache_interface.py`.

Important points:

- `FullAttentionSpec` may still carry a `sliding_window` field in hybrid allocation cases
- `SlidingWindowSpec` is the dedicated spec when the cache manager models it as sliding-window KV
- per-request admission and memory usage are different from full attention

The practical point is:

- sliding window is not only a kernel mask
- it also changes KV retention and memory-accounting assumptions

## Runtime / Kernel Side

Backends typically translate the integer sliding-window size into backend-local kernel parameters.

Examples:

- `FlashAttentionImpl` converts it into a window tuple
- `TritonAttentionImpl` does the same
- `FlexAttention` builds an explicit sliding-window mask modifier

So the same high-level feature is expressed differently depending on backend:

- native kernel window parameters
- explicit mask logic

## Current Restriction With DCP

The current KV-cache interface explicitly rejects DCP with `SlidingWindowSpec`:

- `assert decode_context_parallel_size == 1`
- error text: `DCP not support sliding window.`

That means sliding-window attention is currently an important architectural boundary for distributed decode support.

## `mm_prefix`

`mm_prefix` is a decoder-only mask override used for multimodal prefix-LM style behavior.

The idea is:

- decoder attention is still causal by default
- selected multimodal token ranges get bidirectional/full attention within that range

This is not full non-causal attention everywhere. It is a targeted range-based override.

## Where `mm_prefix` Enters Selection

At layer construction time:

- `Attention` sets `self.use_mm_prefix` from `model_config.is_mm_prefix_lm`

That flag is passed into `get_attn_backend(...)`, which places it in `AttentionSelectorConfig.use_mm_prefix`.

Selection then rejects backends that do not support it via:

- `supports_mm_prefix()`
- `validate_configuration(...)`

So `mm_prefix` is a real selector capability, not only a runtime feature.

## Where `mm_prefix` Enters Runtime

In `vllm/v1/worker/gpu_model_runner.py`, the runner constructs per-request multimodal ranges when `is_mm_prefix_lm` is true.

The flow is:

1. inspect request multimodal features
2. extract embed ranges
3. skip audio ranges
4. build `req_doc_ranges`
5. attach the ranges to attention metadata

The metadata attachment happens through `_set_mm_prefix_range_for_metadata(...)`.

Representative metadata fields:

- `mm_prefix_range`
- `mm_prefix_range_tensor`

Those appear explicitly in Triton attention metadata.

## Practical Sliding-Window Interaction

There is an important runtime filter in `gpu_model_runner.py`:

- if a multimodal range exceeds the sliding-window budget, the runner skips that range

Why:

- a large bidirectional multimodal range can punch through the locality that sliding window is trying to preserve

So the interaction is not:

- sliding window and `mm_prefix` always combine blindly

It is:

- the runner tries to keep `mm_prefix` only when the range remains compatible with the intended local-attention regime

## Backend Support Today

Representative generic backends with `mm_prefix` support include:

- Triton attention
- Flex attention

Backends that do not advertise `supports_mm_prefix()` are filtered out during selection for `is_mm_prefix_lm` models.

## Why These Two Features Belong Together

`sliding_window` and `mm_prefix` are different features, but they belong in the same architecture note because both change:

- mask semantics
- metadata requirements
- KV/block access patterns
- distributed support boundaries

They are the main "decoder attention is not just plain full causal attention" variants in the generic v1 path.

## Best Files To Read Next

- `vllm/model_executor/layers/attention/attention.py`
- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/backends/flex_attention.py`
