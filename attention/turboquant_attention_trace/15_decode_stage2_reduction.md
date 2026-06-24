# Decode Stage-Two Reduction

TurboQuant reuses `_fwd_kernel_stage2(...)` from `vllm/v1/attention/ops/triton_decode_attention.py`.

## Inputs

- `Mid_O: [B, Hq, NUM_KV_SPLITS, D + 1]` contains each split's normalized output and LSE.
- `o: [B, Hq, D]` receives the final output.
- `lse: [B, Hq]` receives final softmax LSE as internal scratch.
- `B_Seqlen: [B]` allows the kernel to identify which fixed splits are nonempty.
- `stride_mid_ob` is the element distance between intermediate batch rows.
- `stride_mid_oh` is the element distance between intermediate query heads.
- `stride_mid_os` is the element distance between intermediate KV splits.
- `stride_obs` is the element distance between final-output batch rows.
- `stride_oh` is the element distance between final-output query heads.
- `stride_lse_bs` is the element distance between final-LSE batch rows.
- `NUM_KV_SPLITS`, `BLOCK_DV`, and `Lv=D` specialize the loop and vector width.
- `OUTPUT_FP16` requests an explicit fp16 cast when Q and output are fp16; bf16 output relies on pointer-store conversion.

## Program Mapping

The grid is `(B, Hq)`, so one program merges every split for one request and one query head.

The program initializes:

```text
global_max = -infinity
global_sum = 0
global_acc = zeros[D]
```

## Which Splits Are Valid

Stage two recomputes the same split boundaries as stage one from `seq_len` and `NUM_KV_SPLITS`.

It only loads a split when `split_end > split_start`, preventing uninitialized intermediate rows from empty fixed-grid splits from contributing.

## Why LSE Is Needed

For split `s`, stage one stored:

```text
o_s = sum_j(exp(score_j) * v_j) / Z_s
lse_s = log(Z_s)
```

The global result is not the arithmetic mean of `o_s`. It must weight each partial output by its softmax partition mass:

```text
o = sum_s(exp(lse_s) * o_s) / sum_s(exp(lse_s))
```

## Stable Merge

The kernel performs that merge without exponentiating large absolute LSE values.

For each valid split:

```text
new_max = max(global_max, lse_s)
old_scale = exp(global_max - new_max)
split_scale = exp(lse_s - new_max)
global_acc = global_acc * old_scale + split_scale * o_s
global_sum = global_sum * old_scale + split_scale
global_max = new_max
```

After all splits:

```text
output = global_acc / global_sum
final_lse = global_max + log(global_sum)
```

This is the same log-sum-exp identity used to merge independently normalized attention states.

## Output And LSE

The final output is stored in query dtype. The final LSE is stored as float32 in the supplied scratch tensor.

`triton_turboquant_decode_attention(...)` returns only output, and `TurboQuantAttentionImpl.forward(...)` follows the ordinary tensor-only return contract.

## Why This Does Not Provide DCP

DCP assigns different context partitions to different ranks and must merge their outputs using each rank's LSE.

TurboQuant's LSE is only visible inside the local launcher and is not returned through the attention implementation. `TurboQuantAttentionImpl` also leaves `can_return_lse_for_decode = False`, so `check_attention_cp_compatibility(...)` in `vllm/v1/worker/cp_utils.py` rejects DCP before execution.

Supporting DCP would require an implementation-level output/LSE interface and cross-rank merge semantics, not merely the existence of this local scratch tensor.
