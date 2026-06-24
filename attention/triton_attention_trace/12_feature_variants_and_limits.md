# Triton Attention Feature Variants And Limits

This note covers optional features and limitations in standard `TRITON_ATTN`.

## Sinks

`TritonAttentionBackend.supports_sink()` returns true.

Runtime sinks are passed to `unified_attention(...)` as:

- `sinks`

Inside `kernel_unified_attention(...)`, sinks initialize the online softmax row max through `init_softmax_M(...)`.

In 3D mode, only segment 0 includes sinks to avoid counting them multiple times.

## MM Prefix

`TritonAttentionBackend.supports_mm_prefix()` returns true.

The metadata builder converts ranges into:

- `mm_prefix_range_tensor`

The unified kernel receives:

- `USE_MM_PREFIX=True`
- `MAX_MM_RANGES`
- `mm_prefix_range_ptr`

`compute_kv_seq_mask(...)` ORs valid bidirectional ranges into the ordinary causal/sliding mask.

This is why `TRITON_ATTN` supports the table's `MM Prefix` column.

## Decoder Non-Causal

The table says non-causal is not supported.

That refers to non-causal decoder attention in the standard paged-KV-cache path.

`unified_attention(...)` asserts:

- `assert causal, "Only causal attention is supported"`

Encoder attention can still be bidirectional because it uses `context_attention_fwd(..., is_causal=False)`.

## Sliding Window

`TritonAttentionImpl.__init__(...)` converts sliding window into:

- `(sliding_window - 1, 0)` for decoder-like attention.
- `(sliding_window - 1, sliding_window - 1)` for encoder/encoder-only attention.
- `(-1, -1)` when disabled.

For decoder unified attention, the wrapper passes `SLIDING_WINDOW = 1 + window_size[0]`.

The helper functions prune tile loops and apply token-level masks.

## Chunked / Block-Local Attention

`chunk_lookback` enables chunked masking.

When active, the wrapper derives:

- `chunk_size = sliding_window_val // (chunk_lookback + 1)`

`compute_kv_seq_mask(...)` uses chunk index differences instead of ordinary sliding-window distance.

This is relevant to block-local layers such as Gemma-style local attention.

## ALiBi And ALiBi Sqrt

`TritonAttentionImpl` accepts `alibi_slopes`.

`TritonAttentionBackend.supports_alibi_sqrt()` returns true.

The unified kernel applies ALiBi with:

- `apply_alibi_to_score(...)`

The sqrt variant is controlled by:

- `USE_ALIBI_SQRT`

## Softcap

`logits_soft_cap` is stored as `self.logits_soft_cap`.

If absent, it is set to 0.

The unified kernel applies softcap when:

- `USE_SOFTCAP = softcap > 0`

The helper computes a tanh-style bound on scores.

## FP8 Output Quantization

`TritonAttentionImpl.fused_output_quant_supported(...)` returns true for:

- `kFp8StaticTensorSym`

If `output_scale` is provided, the unified kernel can write FP8 output.

It scales and clamps the accumulator before storing.

Block-scale output quantization is not supported in `TritonAttentionImpl.forward(...)`.

## KV Quant Modes

Supported KV modes include:

- none
- FP8 per-tensor
- INT8 per-token-head
- FP8 per-token-head

Per-token-head quantization changes both:

- cache shape
- attention kernel scale-cache arguments

## Tensor Descriptors

`VLLM_TRITON_ATTN_USE_TD` controls tensor descriptor mode.

When enabled and supported by layout constraints, the kernel uses:

- `_load_q_td(...)`
- `_load_kv_tile_td(...)`
- `_store_output_td(...)`

This is mainly for platforms that benefit from hardware 2D block reads.

## Batch Invariance

`TritonAttentionBackend.supports_batch_invariance()` returns true.

When `VLLM_BATCH_INVARIANT` is enabled, `unified_attention(...)` avoids 3D segmented mode.

This helps keep outputs stable across batch composition.

## DCP

Standard `TRITON_ATTN` does not support DCP.

It does not advertise decode LSE return support.

`check_attention_cp_compatibility(...)` rejects it when DCP requires `need_to_return_lse_for_decode`.

## PCP

Standard `TRITON_ATTN` does not advertise PCP support.

It inherits `supports_pcp = False` from `AttentionImplBase`.

So `prefill_context_parallel_size > 1` is not supported by this impl.

## Key Files

- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/ops/triton_unified_attention.py`
- `vllm/v1/attention/ops/triton_attention_helpers.py`
- `vllm/v1/worker/cp_utils.py`

