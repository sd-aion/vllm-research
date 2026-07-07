"""Pure-PyTorch quantize/dequantize paths for benchmark comparisons."""

from __future__ import annotations

import math
from functools import lru_cache

import torch


def _gaussian_pdf(x: float, sigma2: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi * sigma2)) * math.exp(-(x * x) / (2.0 * sigma2))


def _trapz(f, a: float, b: float, n: int = 200) -> float:
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


def solve_lloyd_max(
    dim: int,
    bits: int,
    *,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve Lloyd-Max centroids for N(0, 1/dim)."""
    if bits <= 0:
        raise ValueError(f"bits must be positive, got {bits}")
    n_levels = 2**bits
    sigma2 = 1.0 / dim
    sigma = math.sqrt(sigma2)

    def pdf(x: float) -> float:
        return _gaussian_pdf(x, sigma2)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]
    for _ in range(max_iter):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        edges = [lo * 3.0] + boundaries + [hi * 3.0]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num = _trapz(lambda x: x * pdf(x), a, b)
            den = _trapz(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])
        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


@lru_cache(maxsize=64)
def _cached_lloyd_max(dim: int, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    return solve_lloyd_max(dim, bits)


def get_centroids_and_boundaries(
    dim: int,
    bits: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    centroids, boundaries = _cached_lloyd_max(dim, bits)
    return centroids.to(device=device), boundaries.to(device=device)


def mse_quant_dequant(
    x: torch.Tensor,
    *,
    bits: int,
    rotation: torch.Tensor,
    norm_correction: bool,
    metadata_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """TurboQuant MSE-style vector reconstruction.

    The input is treated as row vectors with shape [..., D]. It stores a norm
    and Lloyd-Max centroid indices for each rotated unit vector, then returns
    the reconstructed tensor.
    """
    dim = x.shape[-1]
    centroids, boundaries = get_centroids_and_boundaries(dim, bits, device=x.device)
    xf = x.float()
    norms = xf.norm(dim=-1, keepdim=True)
    unit = xf / (norms + 1e-8)
    rotated = unit @ rotation.float()
    indices = torch.bucketize(rotated.contiguous(), boundaries.contiguous())
    reconstructed_rot = centroids[indices]
    if norm_correction:
        reconstructed_rot = reconstructed_rot / (
            reconstructed_rot.norm(dim=-1, keepdim=True) + 1e-8
        )
    stored_norms = norms.to(metadata_dtype).float()
    reconstructed = (stored_norms * reconstructed_rot) @ rotation.float().T
    return reconstructed.to(x.dtype), {
        "indices": indices,
        "norms": stored_norms.squeeze(-1),
    }


def affine_minmax_quant_dequant(
    x: torch.Tensor,
    *,
    bits: int,
    metadata_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """vLLM TurboQuant value quantization: per-vector affine min/max."""
    xf = x.float()
    qmax = float(2**bits - 1)
    mins = xf.amin(dim=-1, keepdim=True)
    maxs = xf.amax(dim=-1, keepdim=True)
    scales = ((maxs - mins) / qmax).clamp_min(1e-8)
    q = torch.round((xf - mins) / scales).clamp(0, qmax)
    stored_scales = scales.to(metadata_dtype).float()
    stored_mins = mins.to(metadata_dtype).float()
    reconstructed = q * stored_scales + stored_mins
    return reconstructed.to(x.dtype), {
        "q": q.to(torch.int32),
        "scales": stored_scales.squeeze(-1),
        "mins": stored_mins.squeeze(-1),
    }


def build_qjl_matrix(
    *,
    qjl_dim: int,
    head_dim: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the random projection used by the QJL residual sketch."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    matrix = torch.randn(
        qjl_dim,
        head_dim,
        generator=gen,
        device=device,
        dtype=torch.float32,
    )
    return matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def prod_quant_dequant(
    x: torch.Tensor,
    *,
    bits: int,
    rotation: torch.Tensor,
    qjl_matrix: torch.Tensor,
    norm_correction: bool,
    metadata_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Paper-style TurboQuant product reconstruction.

    This combines an MSE reconstruction using b-1 bits with a 1-bit QJL sketch
    of the residual. It is intended as a correctness benchmark model, not as a
    packed-cache implementation.
    """
    if bits < 2:
        raise ValueError("TurboQuant product mode needs at least 2 bits")

    mse_recon, mse_meta = mse_quant_dequant(
        x,
        bits=bits - 1,
        rotation=rotation,
        norm_correction=norm_correction,
        metadata_dtype=metadata_dtype,
    )
    residual = x.float() - mse_recon.float()
    residual_norm = residual.norm(dim=-1, keepdim=True)
    unit_residual = residual / (residual_norm + 1e-8)
    signs = torch.sign(unit_residual @ qjl_matrix.float().T)
    signs = torch.where(signs >= 0, torch.ones_like(signs), -torch.ones_like(signs))
    stored_residual_norm = residual_norm.to(metadata_dtype).float()
    residual_recon = (
        math.sqrt(math.pi / 2.0)
        / float(qjl_matrix.shape[0])
        * stored_residual_norm
        * (signs @ qjl_matrix.float())
    )
    reconstructed = mse_recon.float() + residual_recon
    return reconstructed.to(x.dtype), {
        "mse": mse_meta,
        "qjl_signs": signs.to(torch.int8),
        "residual_norms": stored_residual_norm.squeeze(-1),
    }


def prod_score_k_mse_v(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_bits: int,
    value_bits: int,
    rotation: torch.Tensor,
    qjl_matrix: torch.Tensor,
    norm_correction: bool,
    metadata_dtype: torch.dtype,
    scenario: str,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Estimate attention logits with QJL-corrected K scores and MSE V.

    This mode uses QJL where it naturally fits: as an inner-product estimator
    for the key residual. Values are reconstructed with MSE because attention
    output needs actual value vectors for the weighted sum.
    """
    if key_bits < 2:
        raise ValueError("Product-score key mode needs at least 2 key bits")

    k_mse, k_mse_meta = mse_quant_dequant(
        k,
        bits=key_bits - 1,
        rotation=rotation,
        norm_correction=norm_correction,
        metadata_dtype=metadata_dtype,
    )
    v_hat, _ = mse_quant_dequant(
        v,
        bits=value_bits,
        rotation=rotation,
        norm_correction=norm_correction,
        metadata_dtype=metadata_dtype,
    )

    qf = q.float()
    k_mse_exp = k_mse.float().repeat_interleave(q.shape[2] // k.shape[2], dim=2)
    logits_mse = torch.einsum("bqhd,bshd->bhqs", qf, k_mse_exp) * scale

    residual = k.float() - k_mse.float()
    residual_norm = residual.norm(dim=-1, keepdim=True)
    unit_residual = residual / residual_norm.clamp_min(1e-8)
    signs = torch.where(
        unit_residual @ qjl_matrix.float().T >= 0,
        torch.ones((), device=k.device),
        -torch.ones((), device=k.device),
    )
    stored_residual_norm = residual_norm.to(metadata_dtype).float()

    q_proj = qf @ qjl_matrix.float().T
    signs_exp = signs.repeat_interleave(q.shape[2] // k.shape[2], dim=2)
    residual_norm_exp = stored_residual_norm.repeat_interleave(
        q.shape[2] // k.shape[2],
        dim=2,
    )
    residual_score = (
        math.sqrt(math.pi / 2.0)
        / float(qjl_matrix.shape[0])
        * torch.einsum("bqhm,bshm->bhqs", q_proj, signs_exp)
        * residual_norm_exp.squeeze(-1).transpose(1, 2).unsqueeze(2)
        * scale
    )
    logits = logits_mse + residual_score

    if scenario == "prefill":
        query_len = q.shape[1]
        seq_len = k.shape[1]
        if query_len != seq_len:
            raise ValueError("Prefill product-score mode expects query_len == seq_len")
        mask = torch.ones(query_len, seq_len, device=q.device, dtype=torch.bool).tril()
        logits = logits.masked_fill(~mask.view(1, 1, query_len, seq_len), -torch.inf)

    return k_mse, v_hat, {
        "logits": logits,
        "k_mse": k_mse_meta,
        "qjl_signs": signs.to(torch.int8),
        "residual_norms": stored_residual_norm.squeeze(-1),
    }


def _align_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return int(math.ceil(value / multiple) * multiple)


def estimate_raw_bytes_per_token_head(
    *,
    mode: str,
    head_dim: int,
    key_bits: int,
    value_bits: int,
    qjl_dim: int,
) -> int:
    """Estimate packed bytes per token-head for K+V under each mode."""
    if mode == "no_quant":
        return 2 * head_dim * 2
    if mode == "vllm_style":
        key_bytes = math.ceil(head_dim * key_bits / 8) + 2
        value_bytes = math.ceil(head_dim * value_bits / 8) + 4
        return key_bytes + value_bytes
    if mode == "google_mse_kv":
        key_bytes = math.ceil(head_dim * key_bits / 8) + 2
        value_bytes = math.ceil(head_dim * value_bits / 8) + 2
        return key_bytes + value_bytes
    if mode in {"google_prod_kv", "google_prod_score_k_mse_v"}:
        key_bytes = math.ceil(head_dim * (key_bits - 1) / 8) + 2
        key_bytes += math.ceil(qjl_dim / 8) + 2
        if mode == "google_prod_kv":
            value_bytes = math.ceil(head_dim * (value_bits - 1) / 8) + 2
            value_bytes += math.ceil(qjl_dim / 8) + 2
        else:
            value_bytes = math.ceil(head_dim * value_bits / 8) + 2
        return key_bytes + value_bytes
    raise ValueError(f"Unknown mode {mode!r}")


def estimate_bytes_per_token_head(
    *,
    mode: str,
    head_dim: int,
    key_bits: int,
    value_bits: int,
    qjl_dim: int,
    align_to: int = 16,
) -> int:
    """Estimate aligned bytes per token-head K/V slot."""
    raw_bytes = estimate_raw_bytes_per_token_head(
        mode=mode,
        head_dim=head_dim,
        key_bits=key_bits,
        value_bits=value_bits,
        qjl_dim=qjl_dim,
    )
    return _align_up(raw_bytes, align_to)


def reconstruct_kv(
    *,
    mode: str,
    k: torch.Tensor,
    v: torch.Tensor,
    key_bits: int,
    value_bits: int,
    rotation: torch.Tensor,
    qjl_matrix: torch.Tensor,
    norm_correction: bool,
    metadata_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reconstructed K/V for one benchmark mode."""
    if mode == "no_quant":
        return k, v
    if mode == "vllm_style":
        k_hat, _ = mse_quant_dequant(
            k,
            bits=key_bits,
            rotation=rotation,
            norm_correction=norm_correction,
            metadata_dtype=metadata_dtype,
        )
        v_hat, _ = affine_minmax_quant_dequant(
            v,
            bits=value_bits,
            metadata_dtype=metadata_dtype,
        )
        return k_hat, v_hat
    if mode == "google_mse_kv":
        k_hat, _ = mse_quant_dequant(
            k,
            bits=key_bits,
            rotation=rotation,
            norm_correction=norm_correction,
            metadata_dtype=metadata_dtype,
        )
        v_hat, _ = mse_quant_dequant(
            v,
            bits=value_bits,
            rotation=rotation,
            norm_correction=norm_correction,
            metadata_dtype=metadata_dtype,
        )
        return k_hat, v_hat
    if mode == "google_prod_kv":
        k_hat, _ = prod_quant_dequant(
            k,
            bits=key_bits,
            rotation=rotation,
            qjl_matrix=qjl_matrix,
            norm_correction=norm_correction,
            metadata_dtype=metadata_dtype,
        )
        v_hat, _ = prod_quant_dequant(
            v,
            bits=value_bits,
            rotation=rotation,
            qjl_matrix=qjl_matrix,
            norm_correction=norm_correction,
            metadata_dtype=metadata_dtype,
        )
        return k_hat, v_hat
    raise ValueError(f"Unknown mode {mode!r}")
