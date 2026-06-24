# FlashAttention Kernels And Ops

This note lists the actual kernels and ops used by the `FLASH_ATTN` backend.

## Important Boundary

The core attention kernel is not implemented directly in `FlashAttentionImpl`.

`FlashAttentionImpl` calls `flash_attn_varlen_func(...)`, and that wrapper dispatches to FA2, FA3, or FA4 library ops.

vLLM still owns several surrounding ops:

- KV-cache update through `reshape_and_cache_flash`.
- DCP output/LSE combine through context-parallel helper ops.
- Cascade and DCP partial-output merge through `merge_attn_states`.

## flash_attn_varlen_func

The main FlashAttention entry point is:

- `flash_attn_varlen_func(...)`

It is imported in `vllm/v1/attention/backends/fa_utils.py`.

On CUDA, it comes from:

- `vllm.vllm_flash_attn.flash_attn_varlen_func`

The wrapper implementation is in:

- `vllm/vllm_flash_attn/flash_attn_interface.py`

It supports:

- variable-length packed Q batches.
- MQA/GQA through fewer KV heads than Q heads.
- causal and non-causal masks.
- sliding window through `window_size`.
- paged KV cache through `block_table` plus `seqused_k`.
- direct non-paged Q/K/V through `cu_seqlens_k`.
- optional return of softmax LSE.
- optional sink support through `s_aux` for supported versions.
- optional scheduler metadata for FA3.

## FA2 Kernel Path

When `fa_version == 2`, the wrapper calls:

- `torch.ops._vllm_fa2_C.varlen_fwd`

This is the FA2 variable-length forward op.

Important FA2 constraints in the wrapper:

- It does not support scheduler metadata.
- It does not support `q_descale`, `k_descale`, or `v_descale` in this wrapper path.
- It does not support `s_aux`.
- It does not support `num_splits > 1`.

FA2 is the broad fallback path, especially for Ampere and for combinations where FA3/FA4 are not usable.

## FA3 Kernel Path

When `fa_version == 3`, the wrapper calls:

- `torch.ops._vllm_fa3_C.fwd`

FA3 also exposes scheduler metadata through:

- `torch.ops._vllm_fa3_C.get_scheduler_metadata`

FA3 supports the richer runtime options used by vLLM:

- scheduler metadata
- `q_descale`, `k_descale`, `v_descale`
- `num_splits`
- sinks through `s_aux`
- CP-related arguments in the wrapper signature

The backend typically selects FA3 on Hopper / SM90.

## FA4 Kernel Path

When `fa_version == 4`, the wrapper imports:

- `_flash_attn_fwd` from `vllm.vllm_flash_attn.cute.interface`

Then it calls:

- `_flash_attn_fwd(...)`

FA4 is the Blackwell-oriented path.

The wrapper passes:

- `page_table=block_table`
- `seqused_k`
- `cu_seqlens_q`
- `cu_seqlens_k`
- `softmax_scale`
- `causal`
- `softcap`
- left and right sliding-window values
- `num_splits`
- `return_lse`
- `learnable_sink=s_aux`

FA4 can be rejected or downgraded by `get_flash_attn_version(...)` for some head sizes, ALiBi, or batch-invariance constraints.

## get_scheduler_metadata

The scheduler metadata path is FA3-specific.

`FlashAttentionMetadataBuilder` calls `get_scheduler_metadata(...)` from `fa_utils.py`.

The CUDA wrapper calls:

- `torch.ops._vllm_fa3_C.get_scheduler_metadata`

The inputs include:

- batch size
- max Q length
- max K length
- query head count
- KV head count
- head dimension
- cache sequence lengths
- Q cumulative lengths
- page size
- causal flag
- sliding window
- number of splits

The output is passed back into `flash_attn_varlen_func(...)` as `scheduler_metadata`.

## reshape_and_cache_flash

The KV-cache update op is:

- `reshape_and_cache_flash(...)`

On CUDA it is imported from:

- `vllm._custom_ops.reshape_and_cache_flash`

The Python wrapper calls:

- `torch.ops._C_cache_ops.reshape_and_cache_flash`

This op is not the FlashAttention attention kernel.

It is a vLLM cache-write op that reshapes packed key/value tensors and scatters them into paged cache storage according to `slot_mapping`.

Profiling may show the underlying CUDA kernel name as:

- `reshape_and_cache_flash_kernel`

## merge_attn_states

The partial-output merge helper is:

- `merge_attn_states(...)` in `vllm/v1/attention/ops/merge_attn_states.py`

It merges two attention outputs using their softmax LSE values.

FlashAttention uses it in:

- DCP, to merge context attention and query attention.
- Cascade attention, to merge common-prefix attention and suffix attention.

On CUDA, when dtype and head size are supported, it calls:

- `vllm._custom_ops.merge_attn_states`

That wrapper calls the custom C++/CUDA op.

Otherwise it falls back to:

- `vllm/v1/attention/ops/triton_merge_attn_states.py`

## DCP Combine Ops

For DCP, `FlashAttentionImpl` chooses one of:

- `cp_lse_ag_out_rs(...)` from `vllm/v1/attention/ops/common.py`
- `dcp_a2a_lse_reduce(...)` from `vllm/v1/attention/ops/dcp_alltoall.py`

Both combine partial attention outputs across DCP ranks using LSE-aware weighting.

`cp_lse_ag_out_rs(...)` uses an all-gather style LSE correction followed by reduce-scatter over output heads.

`dcp_a2a_lse_reduce(...)` packs output and LSE for all-to-all exchange, then unpacks and combines with exact LSE weighting.

## Kernel Names You Should Expect

At the vLLM Python level, the important calls are:

- `flash_attn_varlen_func(...)`
- `reshape_and_cache_flash(...)`
- `get_scheduler_metadata(...)`
- `merge_attn_states(...)`
- `cp_lse_ag_out_rs(...)`
- `dcp_a2a_lse_reduce(...)`

At the lower op level, the important calls are:

- `torch.ops._vllm_fa2_C.varlen_fwd`
- `torch.ops._vllm_fa3_C.fwd`
- `torch.ops._vllm_fa3_C.get_scheduler_metadata`
- `vllm.vllm_flash_attn.cute.interface._flash_attn_fwd`
- `torch.ops._C_cache_ops.reshape_and_cache_flash`
- `torch.ops._C.merge_attn_states`

## Key Files

- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/fa_utils.py`
- `vllm/vllm_flash_attn/flash_attn_interface.py`
- `vllm/vllm_flash_attn/cute/interface.py`
- `vllm/_custom_ops.py`
- `vllm/v1/attention/ops/merge_attn_states.py`
- `vllm/v1/attention/ops/common.py`
- `vllm/v1/attention/ops/dcp_alltoall.py`

