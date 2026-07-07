"""Metric helpers for tensor and attention comparisons."""

from __future__ import annotations

import torch


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().reshape(-1)


def tensor_metrics(
    ref: torch.Tensor,
    got: torch.Tensor,
    *,
    prefix: str,
    atol: float,
    rtol: float,
) -> dict[str, float | bool]:
    """Compute standard reconstruction metrics."""
    ref_f_all = _flatten(ref)
    got_f_all = _flatten(got)
    finite_mask = torch.isfinite(ref_f_all) & torch.isfinite(got_f_all)
    if finite_mask.any():
        ref_f = ref_f_all[finite_mask]
        got_f = got_f_all[finite_mask]
    else:
        ref_f = ref_f_all
        got_f = got_f_all
    diff = got_f - ref_f
    ref_norm = torch.linalg.vector_norm(ref_f).clamp_min(1e-12)
    got_norm = torch.linalg.vector_norm(got_f).clamp_min(1e-12)
    cosine = torch.dot(ref_f, got_f) / (ref_norm * got_norm)
    metrics: dict[str, float | bool] = {
        f"{prefix}_max_abs": float(diff.abs().max().item()),
        f"{prefix}_mean_abs": float(diff.abs().mean().item()),
        f"{prefix}_rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
        f"{prefix}_rel_l2": float((torch.linalg.vector_norm(diff) / ref_norm).item()),
        f"{prefix}_cosine": float(cosine.item()),
        f"{prefix}_allclose": bool(torch.allclose(got, ref, atol=atol, rtol=rtol)),
        f"{prefix}_finite_fraction": float(finite_mask.float().mean().item()),
    }
    if ref.ndim >= 1:
        ref_rows = ref.detach().float().reshape(-1, ref.shape[-1])
        got_rows = got.detach().float().reshape(-1, got.shape[-1])
        row_cos = torch.nn.functional.cosine_similarity(ref_rows, got_rows, dim=-1)
        ref_row_norm = ref_rows.norm(dim=-1).clamp_min(1e-12)
        got_row_norm = got_rows.norm(dim=-1)
        norm_ratio = got_row_norm / ref_row_norm
        metrics.update(
            {
                f"{prefix}_row_cosine_mean": float(row_cos.mean().item()),
                f"{prefix}_row_cosine_min": float(row_cos.min().item()),
                f"{prefix}_norm_ratio_mean": float(norm_ratio.mean().item()),
                f"{prefix}_norm_ratio_min": float(norm_ratio.min().item()),
                f"{prefix}_norm_ratio_max": float(norm_ratio.max().item()),
            }
        )
    return metrics


def probability_metrics(
    ref_probs: torch.Tensor,
    got_probs: torch.Tensor,
    *,
    prefix: str,
    topk: int,
) -> dict[str, float]:
    eps = 1e-12
    ref = ref_probs.detach().float().clamp_min(eps)
    got = got_probs.detach().float().clamp_min(eps)
    kl = torch.sum(ref * (torch.log(ref) - torch.log(got)), dim=-1).clamp_min(0.0)
    ref_top1 = ref.argmax(dim=-1)
    got_top1 = got.argmax(dim=-1)

    k = min(topk, ref.shape[-1])
    ref_topk = ref.topk(k, dim=-1).indices
    got_topk = got.topk(k, dim=-1).indices
    overlap = (ref_topk.unsqueeze(-1) == got_topk.unsqueeze(-2)).any(dim=-1).float()

    return {
        f"{prefix}_kl_mean": float(kl.mean().item()),
        f"{prefix}_kl_max": float(kl.max().item()),
        f"{prefix}_max_abs": float((got_probs - ref_probs).abs().max().item()),
        f"{prefix}_mean_abs": float((got_probs - ref_probs).abs().mean().item()),
        f"{prefix}_top1_agreement": float((ref_top1 == got_top1).float().mean().item()),
        f"{prefix}_top{k}_overlap": float(overlap.mean().item()),
    }


def combined_metrics(
    *,
    ref_k: torch.Tensor,
    got_k: torch.Tensor,
    ref_v: torch.Tensor,
    got_v: torch.Tensor,
    ref_logits: torch.Tensor,
    got_logits: torch.Tensor,
    ref_probs: torch.Tensor,
    got_probs: torch.Tensor,
    ref_out: torch.Tensor,
    got_out: torch.Tensor,
    atol: float,
    rtol: float,
    topk: int,
) -> dict[str, float | bool]:
    metrics: dict[str, float | bool] = {}
    metrics.update(tensor_metrics(ref_k, got_k, prefix="k", atol=atol, rtol=rtol))
    metrics.update(tensor_metrics(ref_v, got_v, prefix="v", atol=atol, rtol=rtol))
    metrics.update(
        tensor_metrics(ref_logits, got_logits, prefix="logits", atol=atol, rtol=rtol)
    )
    metrics.update(probability_metrics(ref_probs, got_probs, prefix="probs", topk=topk))
    metrics.update(tensor_metrics(ref_out, got_out, prefix="out", atol=atol, rtol=rtol))
    ref_lse = torch.logsumexp(ref_logits.float(), dim=-1)
    got_lse = torch.logsumexp(got_logits.float(), dim=-1)
    metrics.update(tensor_metrics(ref_lse, got_lse, prefix="lse", atol=atol, rtol=rtol))
    return metrics
