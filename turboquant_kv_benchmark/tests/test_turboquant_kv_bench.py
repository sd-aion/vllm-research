from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from turboquant_kv_bench.attention import attention_reference
from turboquant_kv_bench.data import make_qkv
from turboquant_kv_bench.quantizers import affine_minmax_quant_dequant, mse_quant_dequant
from turboquant_kv_bench.rotations import build_rotation, check_orthonormal


def test_rotations_are_orthonormal() -> None:
    device = torch.device("cpu")
    for name in ["hadamard", "qr"]:
        rotation = build_rotation(name, 16, seed=7, device=device)
        assert check_orthonormal(rotation) < 1e-5


def test_no_quant_attention_matches_itself() -> None:
    q, k, v = make_qkv(
        batch_size=1,
        seq_len=8,
        query_len=1,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=16,
        scenario="decode",
        distribution="mixed_non_gaussian",
        seed=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    out_a, logits_a, probs_a = attention_reference(q, k, v, scenario="decode")
    out_b, logits_b, probs_b = attention_reference(q, k, v, scenario="decode")
    assert torch.equal(out_a, out_b)
    assert torch.equal(logits_a, logits_b)
    assert torch.equal(probs_a, probs_b)


def test_affine_minmax_small_vector() -> None:
    x = torch.tensor([[[[2.0, 6.0]]]])
    recon, meta = affine_minmax_quant_dequant(x, bits=4, metadata_dtype=torch.float32)
    assert torch.allclose(recon, x)
    assert torch.allclose(meta["scales"], torch.tensor([[[4.0 / 15.0]]]))
    assert torch.allclose(meta["mins"], torch.tensor([[[2.0]]]))


def test_mse_quant_dequant_shape_and_finiteness() -> None:
    x = torch.rand(2, 3, 4, 16) * 2 - 1
    rotation = build_rotation("hadamard", 16, seed=1, device=torch.device("cpu"))
    recon, meta = mse_quant_dequant(
        x,
        bits=3,
        rotation=rotation,
        norm_correction=True,
        metadata_dtype=torch.float32,
    )
    assert recon.shape == x.shape
    assert meta["indices"].shape == x.shape
    assert torch.isfinite(recon).all()


def test_cli_smoke(tmp_path: Path) -> None:
    script = ROOT / "bench_turboquant_kv.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--batch-size",
            "1",
            "--seq-len",
            "8",
            "--query-len",
            "1",
            "--num-query-heads",
            "2",
            "--num-kv-heads",
            "1",
            "--head-dim",
            "16",
            "--key-bits",
            "3",
            "--value-bits",
            "3",
            "--qjl-dim",
            "16",
            "--rotation",
            "hadamard",
            "--scenario",
            "decode",
            "--modes",
            "vllm_style,google_mse_kv,google_prod_score_k_mse_v,google_prod_kv",
            "--save-tensors",
            "none",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "output_dir" in result.stdout
    run_dirs = list(tmp_path.glob("run_*"))
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "args.json").exists()
    assert (run_dirs[0] / "metrics.jsonl").exists()
    assert (run_dirs[0] / "summary.csv").exists()
    assert (run_dirs[0] / "summary.md").exists()
