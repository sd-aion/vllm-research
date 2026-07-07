"""Synthetic Q/K/V generation for correctness benchmarking."""

from __future__ import annotations

import math

import torch


def _rand(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.rand(shape, generator=generator, device=device, dtype=torch.float32)


def _uniform_signed(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return _rand(shape, generator=generator, device=device) * 2.0 - 1.0


def _laplace_like(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    u = _uniform_signed(shape, generator=generator, device=device).clamp(-0.999, 0.999)
    return -torch.sign(u) * torch.log1p(-u.abs())


def _student_t_like(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    u = _rand(shape, generator=generator, device=device).clamp(1e-4, 1.0 - 1e-4)
    return torch.tan(math.pi * (u - 0.5)).clamp(-25.0, 25.0)


def sample_tensor(
    shape: tuple[int, ...],
    *,
    distribution: str,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Generate non-normal benchmark inputs.

    The default intentionally combines random signs, random per-vector scales,
    and sparse outliers so that the benchmark is not optimized for Gaussian
    inputs.
    """
    if distribution == "uniform":
        return _uniform_signed(shape, generator=generator, device=device)
    if distribution == "laplace_like":
        return _laplace_like(shape, generator=generator, device=device)
    if distribution == "student_t_like":
        return _student_t_like(shape, generator=generator, device=device)
    if distribution == "outlier_mixture":
        base = _uniform_signed(shape, generator=generator, device=device)
        mask = _rand(shape, generator=generator, device=device) < 0.02
        outliers = _uniform_signed(shape, generator=generator, device=device) * 40.0
        return torch.where(mask, outliers, base)
    if distribution == "mixed_non_gaussian":
        base = 0.55 * _uniform_signed(shape, generator=generator, device=device)
        base = base + 0.35 * _laplace_like(shape, generator=generator, device=device)
        base = base + 0.10 * _student_t_like(shape, generator=generator, device=device)

        scale_shape = shape[:-1] + (1,)
        log_scale = _rand(scale_shape, generator=generator, device=device)
        scale = torch.exp(math.log(0.05) + log_scale * (math.log(20.0) - math.log(0.05)))
        mask = _rand(shape, generator=generator, device=device) < 0.01
        outliers = _uniform_signed(shape, generator=generator, device=device) * 30.0
        return torch.where(mask, outliers, base) * scale
    raise ValueError(f"Unknown data distribution {distribution!r}")


def make_qkv(
    *,
    batch_size: int,
    seq_len: int,
    query_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scenario: str,
    distribution: str,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create Q, K, V tensors.

    Shapes are Q=[B,Tq,Hq,D], K=[B,S,Hk,D], and V=[B,S,Hk,D]. Prefill uses
    Tq=S so causal prompt attention can be compared directly. Decode uses the
    CLI query length.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    if scenario == "prefill":
        actual_query_len = seq_len
    elif scenario == "decode":
        actual_query_len = query_len
    else:
        raise ValueError(f"Unknown scenario {scenario!r}")

    q = sample_tensor(
        (batch_size, actual_query_len, num_query_heads, head_dim),
        distribution=distribution,
        generator=gen,
        device=device,
    )
    k = sample_tensor(
        (batch_size, seq_len, num_kv_heads, head_dim),
        distribution=distribution,
        generator=gen,
        device=device,
    )
    v = sample_tensor(
        (batch_size, seq_len, num_kv_heads, head_dim),
        distribution=distribution,
        generator=gen,
        device=device,
    )
    return q.to(dtype), k.to(dtype), v.to(dtype)
