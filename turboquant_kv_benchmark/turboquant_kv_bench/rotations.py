"""Rotation matrix builders used by the benchmark."""

from __future__ import annotations

import math

import torch


def build_hadamard(dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build a normalized Sylvester Hadamard matrix.

    This matches the style used by vLLM TurboQuant. It requires a power-of-two
    head dimension and returns an orthonormal matrix.
    """
    if dim <= 0 or dim & (dim - 1):
        raise ValueError(f"Hadamard rotation requires power-of-two dim, got {dim}")

    h = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while h.shape[0] < dim:
        top = torch.cat([h, h], dim=1)
        bottom = torch.cat([h, -h], dim=1)
        h = torch.cat([top, bottom], dim=0)
    return (h / math.sqrt(dim)).to(dtype=dtype)


def build_qr_rotation(
    dim: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a random orthonormal matrix from a normal matrix via QR.

    The benchmark only uses normal sampling here because the paper-style random
    orthogonal rotation is defined this way. Synthetic Q/K/V inputs are sampled
    from non-normal distributions by default.
    """
    mat = torch.randn(dim, dim, generator=generator, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(mat)
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q.to(dtype=dtype)


def build_rotation(
    name: str,
    dim: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build one supported rotation matrix."""
    if name == "hadamard":
        return build_hadamard(dim, device=device, dtype=dtype)
    if name == "qr":
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return build_qr_rotation(dim, generator=gen, device=device, dtype=dtype)
    raise ValueError(f"Unknown rotation {name!r}")


def check_orthonormal(rotation: torch.Tensor) -> float:
    """Return max absolute error of R @ R.T against identity."""
    dim = rotation.shape[0]
    ident = torch.eye(dim, device=rotation.device, dtype=torch.float32)
    err = rotation.float() @ rotation.float().T - ident
    return float(err.abs().max().item())
