# Context Attention Kernel

This note explains the direct Q/K/V Triton context attention kernel used by `TRITON_ATTN` for encoder attention.

## Location

The wrapper is:

- `context_attention_fwd(...)` in `vllm/v1/attention/ops/triton_prefill_attention.py`

It launches:

- `_fwd_kernel`

## Important Naming Nuance

The file is named `triton_prefill_attention.py`.

For standard `TritonAttentionBackend`, this is not the normal decoder prefill path.

`TritonAttentionImpl._forward_encoder_attention(...)` calls `context_attention_fwd(...)` for encoder and encoder-only attention.

Standard decoder prefill uses `unified_attention(...)` over paged KV cache.

## Direct Q/K/V Inputs

`context_attention_fwd(...)` receives:

- `q`: `[b * s, head, head_dim]`
- `k`: `[b * s, kv_head, head_dim]`
- `v`: `[b * s, kv_head, head_dim]`
- `o`: output tensor
- `b_start_loc`: request start locations
- `b_seq_len`: request sequence lengths
- `max_input_len`: maximum sequence length
- `is_causal`
- `softmax_scale`
- `sliding_window_q`
- `sliding_window_k`

There is no block table.

There is no slot mapping.

There is no paged KV cache read.

## Launch Grid

The wrapper launches `_fwd_kernel` with:

- `grid = (batch, head, ceil(max_input_len / BLOCK))`

Program IDs are:

- `program_id(0)`: batch/request.
- `program_id(1)`: query head.
- `program_id(2)`: query block inside the sequence.

## BLOCK Selection

`get_block_size(dtype)` returns:

- 32 for fp32.
- 128 for CUDA-like platforms with compute capability at least 8.0.
- 64 otherwise.

`BLOCK_M` and `BLOCK_N` both use this block size.

`BLOCK_DMODEL` is `next_power_of_2(head_dim)`.

## Kernel Flow

`_fwd_kernel` does:

1. resolve current batch, head, and query block.
2. map query head to KV head through `kv_group_num`.
3. load Q tile.
4. loop over K/V tiles.
5. apply causal mask if `IS_CAUSAL`.
6. apply bidirectional sliding-window masks.
7. compute `qk = dot(q, k)`.
8. run online softmax with running max and sum.
9. accumulate `P @ V`.
10. divide by final softmax sum and store output.

## Causal And Non-Causal

For encoder attention, `TritonAttentionImpl._forward_encoder_attention(...)` passes:

- `is_causal=False`

This is why `TRITON_ATTN` can support encoder bidirectional attention even though decoder non-causal support is false in the feature table.

The feature table's `Non-Causal` column refers to non-causal attention in the decoder paged-KV-cache path.

## Sliding Window

The kernel supports two sliding-window constants:

- `SLIDING_WINDOW_Q`
- `SLIDING_WINDOW_K`

These allow a bidirectional local attention window for encoder-like attention.

The mask checks:

- `pos_q - pos_k <= SLIDING_WINDOW_Q`
- `pos_k - pos_q <= SLIDING_WINDOW_K`

## Key Files

- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/ops/triton_prefill_attention.py`
- `vllm/v1/attention/ops/vit_attn_wrappers.py`

