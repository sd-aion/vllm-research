# FlashAttention Attention Variants

This note covers the important variants and optional features around `FLASH_ATTN`.

## Decoder Attention

Decoder attention is the standard paged-KV-cache path.

It uses:

- `do_kv_cache_update(...)` to write current K/V into cache.
- `flash_attn_varlen_func(...)` with `block_table` and `seqused_k` to read paged cache.
- `causal=True` by default for autoregressive models.

This is the path most standard LLM serving traffic uses.

## Encoder Attention

`FLASH_ATTN` supports encoder attention through:

- `supports_attn_type(...)`
- `_forward_encoder_attention(...)`

Encoder attention does not update or read paged KV cache.

It calls `flash_attn_varlen_func(...)` directly on Q/K/V tensors with `causal=False`.

This is relevant for encoder-only and multimodal encoder attention paths that select FlashAttention as the underlying attention implementation.

## Encoder-Decoder And Cross-Attention

`supports_attn_type(...)` includes `AttentionType.ENCODER_DECODER`.

The standard forward path treats decoder and cross-attention as cache-using paths.

The exact layer wrapper decides which attention type is being constructed, but `FLASH_ATTN` advertises support for the full set of attention types.

## Non-Causal Attention

`FlashAttentionBackend.supports_non_causal()` returns `True`.

Non-causal decoder attention means the runtime can pass `causal=False` for a decoder-model attention layer.

Mathematically, causal attention masks future positions:

- `out_i = softmax((q_i K_{\le i}^T) / sqrt(d)) V_{\le i}`

Non-causal attention allows bidirectional attention:

- `out_i = softmax((q_i K_{1:L}^T) / sqrt(d)) V_{1:L}`

In FlashAttention, this is ultimately controlled by the `causal` argument passed to `flash_attn_varlen_func(...)`.

## Sliding Window

`FlashAttentionImpl.__init__(...)` converts the model's `sliding_window` integer into a FlashAttention window tuple.

For decoder attention:

- `self.sliding_window = (sliding_window - 1, 0)`

For encoder-only attention:

- `self.sliding_window = (sliding_window - 1, sliding_window - 1)`

For no sliding window:

- `self.sliding_window = (-1, -1)`

The tuple is passed to `flash_attn_varlen_func(...)` as `window_size`.

For decoder sliding window, each query attends to a bounded left context and no future tokens.

For encoder-only sliding window, each token can attend to a bounded window on both sides.

## Sinks

Attention sinks are optional learned/static positions that remain globally attendable even when the rest of attention is windowed or streaming-oriented.

`FLASH_ATTN` supports sinks only when `flash_attn_supports_sinks()` is true.

In practice:

- FA2 does not support sinks.
- FA3 and FA4 support sinks.
- compute capability below 9.0 is rejected for sink use.

Runtime sinks are passed to `flash_attn_varlen_func(...)` as:

- `s_aux=self.sinks`

## ALiBi

`FlashAttentionImpl` accepts `alibi_slopes`.

When ALiBi is present, `get_flash_attn_version(...)` avoids FA3 and FA4 and uses FA2.

The reason is that FA3 and FA4 assert that ALiBi is unsupported in this wrapper path.

When supported, `alibi_slopes` are passed to `flash_attn_varlen_func(...)`.

## Softcap

`logits_soft_cap` is stored as `self.logits_soft_cap`.

If no softcap is configured, vLLM passes `0`, which FlashAttention interprets as disabled.

The value is passed to `flash_attn_varlen_func(...)` as:

- `softcap=self.logits_soft_cap`

Softcap modifies attention logits before softmax and is model-dependent.

## Batch Invariance

`FlashAttentionBackend.supports_batch_invariance()` returns `True`.

At runtime, `VLLM_BATCH_INVARIANT` affects:

- `get_flash_attn_version(...)`, which avoids FA4.
- metadata scheduling, where `max_num_splits` can be forced to `1`.
- encoder attention, where `num_splits` is set to `1` when batch invariance is enabled.

The purpose is to reduce output sensitivity to batch composition.

## Per-Head Quant Scales

`FlashAttentionBackend.supports_per_head_quant_scales()` returns true for FlashAttention version 3 or newer.

Forward passes use scale tensors from the attention layer:

- `layer._q_scale`
- `layer._k_scale`
- `layer._v_scale`

These can be expanded and passed as `q_descale`, `k_descale`, and `v_descale`.

## MM Prefix

`FLASH_ATTN` does not support the decoder-side `mm_prefix` feature.

It inherits `supports_mm_prefix() -> False` from `AttentionBackend`.

If `use_mm_prefix=True`, backend validation rejects `FLASH_ATTN`.

Do not confuse this with encoder attention for multimodal models.

`FLASH_ATTN` can be selected for some multimodal encoder attention paths, but the feature-table `MM Prefix` column refers to partial multimodal token full attention in the decoder paged-KV-cache path.

## Sparse

`FLASH_ATTN` standard attention is not a sparse MLA backend.

The sparse column in the feature table applies to MLA-specific sparse backends.

For standard `FLASH_ATTN`, sparse attention is not the relevant feature surface.

## Key Files

- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/fa_utils.py`
- `vllm/vllm_flash_attn/flash_attn_interface.py`
- `vllm/model_executor/layers/attention/mm_encoder_attention.py`

