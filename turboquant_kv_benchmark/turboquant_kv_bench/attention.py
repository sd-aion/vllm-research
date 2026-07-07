"""Dense reference attention for reconstructed KV tensors."""

from __future__ import annotations

import math

import torch


def expand_kv_for_gqa(kv: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """Repeat KV heads so Hk matches Hq for GQA/MQA comparisons."""
    num_kv_heads = kv.shape[2]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads={num_query_heads} must be divisible by "
            f"num_kv_heads={num_kv_heads}"
        )
    repeat = num_query_heads // num_kv_heads
    return kv.repeat_interleave(repeat, dim=2)


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scenario: str,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return output, logits, and probabilities for dense attention.

    q shape is [B,Tq,Hq,D]. k/v shapes are [B,S,Hk,D].
    Decode attends every query to the full cache. Prefill applies a standard
    causal mask with Tq=S.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    k_exp = expand_kv_for_gqa(k, q.shape[2])
    v_exp = expand_kv_for_gqa(v, q.shape[2])

    qf = q.float().transpose(1, 2)
    kf = k_exp.float().transpose(1, 2)
    vf = v_exp.float().transpose(1, 2)

    logits = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    if scenario == "prefill":
        query_len = q.shape[1]
        seq_len = k.shape[1]
        if query_len != seq_len:
            raise ValueError("Prefill reference expects query_len == seq_len")
        mask = torch.ones(query_len, seq_len, device=q.device, dtype=torch.bool).tril()
        logits = logits.masked_fill(~mask.view(1, 1, query_len, seq_len), -torch.inf)
    elif scenario != "decode":
        raise ValueError(f"Unknown scenario {scenario!r}")

    probs = torch.softmax(logits, dim=-1)
    out = torch.matmul(probs, vf).transpose(1, 2).to(q.dtype)
    return out, logits, probs


def attention_from_logits(
    q: torch.Tensor,
    v: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return output/probabilities from precomputed attention logits."""
    v_exp = expand_kv_for_gqa(v, q.shape[2])
    vf = v_exp.float().transpose(1, 2)
    probs = torch.softmax(logits, dim=-1)
    out = torch.matmul(probs, vf).transpose(1, 2).to(q.dtype)
    return out, probs
