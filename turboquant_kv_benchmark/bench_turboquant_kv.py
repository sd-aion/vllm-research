#!/usr/bin/env python3
"""CLI for pure-PyTorch TurboQuant KV-cache correctness benchmarking."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    tqdm = None

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from turboquant_kv_bench.attention import attention_from_logits, attention_reference
from turboquant_kv_bench.data import make_qkv
from turboquant_kv_bench.metrics import combined_metrics
from turboquant_kv_bench.quantizers import (
    estimate_bytes_per_token_head,
    estimate_raw_bytes_per_token_head,
    prod_score_k_mse_v,
    reconstruct_kv,
)
from turboquant_kv_bench.reports import (
    aggregate_rows,
    append_jsonl,
    write_csv,
    write_json,
    write_markdown,
)
from turboquant_kv_bench.rotations import build_rotation, check_orthonormal


DEFAULT_MODES = [
    "vllm_style",
    "google_mse_kv",
    "google_prod_score_k_mse_v",
    "google_prod_kv",
]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _resolve_qjl_dim(value: str, head_dim: int) -> int:
    if value == "auto":
        return head_dim
    return int(value)


def _parse_csv_strings(value: str) -> list[str]:
    if value == "all":
        return ["mixed_non_gaussian", "uniform", "laplace_like", "student_t_like", "outlier_mixture"]
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_dtype(value: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype {value!r}")
    return mapping[value]


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: value for key, value in vars(args).items()}


def _should_skip_mode(mode: str, key_bits: int, value_bits: int) -> bool:
    return mode.startswith("google_prod") and (key_bits < 2 or value_bits < 2)


def _count_work_items(
    *,
    head_dims: list[int],
    seq_lens: list[int],
    scenarios: list[str],
    distributions: list[str],
    rotations: list[str],
    key_bits_list: list[int],
    value_bits_list: list[int],
    modes: list[str],
    num_trials: int,
) -> int:
    total = 0
    for _head_dim in head_dims:
        for _seq_len in seq_lens:
            for _scenario in scenarios:
                for _distribution in distributions:
                    for _trial_idx in range(num_trials):
                        for _rotation_name in rotations:
                            for key_bits in key_bits_list:
                                for value_bits in value_bits_list:
                                    for mode in modes:
                                        if not _should_skip_mode(
                                            mode, key_bits, value_bits
                                        ):
                                            total += 1
    return total


def _sample_tensors(tensors: dict[str, torch.Tensor], *, all_tensors: bool) -> dict[str, torch.Tensor]:
    if all_tensors:
        return {key: value.detach().cpu() for key, value in tensors.items()}
    samples = {}
    for key, value in tensors.items():
        slices = tuple(slice(0, min(size, 2 if idx != 1 else 8)) for idx, size in enumerate(value.shape))
        samples[key] = value[slices].detach().cpu()
    return samples


def run(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    if args.slot_align < 1:
        raise ValueError("--slot-align must be >= 1")
    dtype = _parse_dtype(args.dtype)
    metadata_dtype = _parse_dtype(args.metadata_dtype)
    key_bits_list = _parse_csv_ints(args.key_bits)
    value_bits_list = _parse_csv_ints(args.value_bits)
    seq_lens = _parse_csv_ints(args.seq_lens or str(args.seq_len))
    head_dims = _parse_csv_ints(args.head_dims or str(args.head_dim))
    scenarios = _parse_csv_strings(args.scenario)
    distributions = _parse_csv_strings(args.data_distribution)
    rotations = ["hadamard", "qr"] if args.rotation == "both" else _parse_csv_strings(args.rotation)
    modes = DEFAULT_MODES if args.modes == "default" else _parse_csv_strings(args.modes)

    if args.include_no_quant:
        modes = ["no_quant"] + modes

    total_work_items = _count_work_items(
        head_dims=head_dims,
        seq_lens=seq_lens,
        scenarios=scenarios,
        distributions=distributions,
        rotations=rotations,
        key_bits_list=key_bits_list,
        value_bits_list=value_bits_list,
        modes=modes,
        num_trials=args.num_trials,
    )
    progress_disabled = args.progress == "never" or (
        args.progress == "auto" and not sys.stderr.isatty()
    )
    progress_context = (
        tqdm(
            total=total_work_items,
            desc="benchmark rows",
            unit="row",
            dynamic_ncols=True,
            leave=True,
            disable=progress_disabled,
        )
        if total_work_items and tqdm is not None
        else nullcontext()
    )

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser().resolve() / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    write_json(output_dir / "args.json", _jsonable_args(args))
    write_json(
        output_dir / "manifest.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "source_notes": {
                "vllm_turboquant_backend": "/home/ubuntu/vllm/vllm/v1/attention/backends/turboquant_attn.py",
                "vllm_turboquant_config": "/home/ubuntu/vllm/vllm/model_executor/layers/quantization/turboquant/config.py",
                "local_research_notes": "/home/ubuntu/sasmit/vllm-research/attention/turboquant_attention_trace/",
                "paper": "https://arxiv.org/abs/2504.19874",
            },
        },
    )

    rows: list[dict[str, Any]] = []
    tensor_samples: dict[str, torch.Tensor] = {}
    with progress_context as progress:
        for head_dim in head_dims:
            fp16_bytes = 2 * head_dim * 2

            for seq_len in seq_lens:
                for scenario in scenarios:
                    for distribution in distributions:
                        for trial_idx in range(args.num_trials):
                            data_seed = (
                                args.seed
                                + 1_000_003 * trial_idx
                                + 100_003 * head_dim
                                + 10_007 * seq_len
                                + 503 * scenarios.index(scenario)
                                + 97 * distributions.index(distribution)
                            )
                            q, k, v = make_qkv(
                                batch_size=args.batch_size,
                                seq_len=seq_len,
                                query_len=args.query_len,
                                num_query_heads=args.num_query_heads,
                                num_kv_heads=args.num_kv_heads,
                                head_dim=head_dim,
                                scenario=scenario,
                                distribution=distribution,
                                seed=data_seed,
                                device=device,
                                dtype=dtype,
                            )
                            ref_out, ref_logits, ref_probs = attention_reference(
                                q,
                                k,
                                v,
                                scenario=scenario,
                            )

                            for rotation_name in rotations:
                                rotation = build_rotation(
                                    rotation_name,
                                    head_dim,
                                    seed=args.seed + 17,
                                    device=device,
                                    dtype=torch.float32,
                                )
                                qjl_dim = _resolve_qjl_dim(args.qjl_dim, head_dim)
                                qjl_matrix = torch.empty(0, device=device)
                                if any(mode.startswith("google_prod") for mode in modes):
                                    from turboquant_kv_bench.quantizers import build_qjl_matrix

                                    qjl_matrix = build_qjl_matrix(
                                        qjl_dim=qjl_dim,
                                        head_dim=head_dim,
                                        seed=args.seed + 29,
                                        device=device,
                                    )

                                for key_bits in key_bits_list:
                                    for value_bits in value_bits_list:
                                        for mode in modes:
                                            if _should_skip_mode(
                                                mode, key_bits, value_bits
                                            ):
                                                continue
                                            if mode == "google_prod_score_k_mse_v":
                                                scale = 1.0 / (head_dim**0.5)
                                                k_hat, v_hat, prod_meta = (
                                                    prod_score_k_mse_v(
                                                        q=q,
                                                        k=k,
                                                        v=v,
                                                        key_bits=key_bits,
                                                        value_bits=value_bits,
                                                        rotation=rotation,
                                                        qjl_matrix=qjl_matrix,
                                                        norm_correction=(
                                                            not args.disable_norm_correction
                                                        ),
                                                        metadata_dtype=metadata_dtype,
                                                        scenario=scenario,
                                                        scale=scale,
                                                    )
                                                )
                                                got_logits = prod_meta["logits"]
                                                got_out, got_probs = attention_from_logits(
                                                    q,
                                                    v_hat,
                                                    got_logits,
                                                )
                                            else:
                                                k_hat, v_hat = reconstruct_kv(
                                                    mode=mode,
                                                    k=k,
                                                    v=v,
                                                    key_bits=key_bits,
                                                    value_bits=value_bits,
                                                    rotation=rotation,
                                                    qjl_matrix=qjl_matrix,
                                                    norm_correction=(
                                                        not args.disable_norm_correction
                                                    ),
                                                    metadata_dtype=metadata_dtype,
                                                )
                                                got_out, got_logits, got_probs = (
                                                    attention_reference(
                                                        q,
                                                        k_hat,
                                                        v_hat,
                                                        scenario=scenario,
                                                    )
                                                )
                                            raw_bytes_per_token_head = (
                                                estimate_raw_bytes_per_token_head(
                                                    mode=mode,
                                                    head_dim=head_dim,
                                                    key_bits=key_bits,
                                                    value_bits=value_bits,
                                                    qjl_dim=qjl_dim,
                                                )
                                            )
                                            bytes_per_token_head = (
                                                estimate_bytes_per_token_head(
                                                    mode=mode,
                                                    head_dim=head_dim,
                                                    key_bits=key_bits,
                                                    value_bits=value_bits,
                                                    qjl_dim=qjl_dim,
                                                    align_to=args.slot_align,
                                                )
                                            )
                                            row: dict[str, Any] = {
                                                "mode": mode,
                                                "scenario": scenario,
                                                "distribution": distribution,
                                                "rotation": rotation_name,
                                                "rotation_orthonormal_max_err": (
                                                    check_orthonormal(rotation)
                                                ),
                                                "batch_size": args.batch_size,
                                                "seq_len": seq_len,
                                                "query_len": q.shape[1],
                                                "trial_idx": trial_idx,
                                                "num_trials": args.num_trials,
                                                "num_query_heads": args.num_query_heads,
                                                "num_kv_heads": args.num_kv_heads,
                                                "head_dim": head_dim,
                                                "key_bits": key_bits,
                                                "value_bits": value_bits,
                                                "qjl_dim": qjl_dim,
                                                "raw_bytes_per_token_head": (
                                                    raw_bytes_per_token_head
                                                ),
                                                "bytes_per_token_head": (
                                                    bytes_per_token_head
                                                ),
                                                "slot_align": args.slot_align,
                                                "compression_ratio_vs_fp16": (
                                                    fp16_bytes / bytes_per_token_head
                                                ),
                                                "norm_correction": (
                                                    not args.disable_norm_correction
                                                ),
                                                "metadata_dtype": args.metadata_dtype,
                                                "data_seed": data_seed,
                                            }
                                            row.update(
                                                combined_metrics(
                                                    ref_k=k,
                                                    got_k=k_hat,
                                                    ref_v=v,
                                                    got_v=v_hat,
                                                    ref_logits=ref_logits,
                                                    got_logits=got_logits,
                                                    ref_probs=ref_probs,
                                                    got_probs=got_probs,
                                                    ref_out=ref_out,
                                                    got_out=got_out,
                                                    atol=args.atol,
                                                    rtol=args.rtol,
                                                    topk=args.topk,
                                                )
                                            )
                                            rows.append(row)

                                            if progress is not None:
                                                progress.update(1)

                                            if (
                                                args.save_tensors != "none"
                                                and not tensor_samples
                                            ):
                                                tensor_samples = _sample_tensors(
                                                    {
                                                        "q": q,
                                                        "k": k,
                                                        "v": v,
                                                        "k_hat": k_hat,
                                                        "v_hat": v_hat,
                                                        "ref_out": ref_out,
                                                        "got_out": got_out,
                                                        "ref_logits": ref_logits,
                                                        "got_logits": got_logits,
                                                    },
                                                    all_tensors=args.save_tensors == "all",
                                                )

    write_json(
        output_dir / "config.json",
        {
            "key_bits": key_bits_list,
            "value_bits": value_bits_list,
            "head_dims": head_dims,
            "seq_lens": seq_lens,
            "num_trials": args.num_trials,
            "scenarios": scenarios,
            "distributions": distributions,
            "rotations": rotations,
            "modes": modes,
        },
    )
    group_keys = [
        "scenario",
        "distribution",
        "rotation",
        "head_dim",
        "seq_len",
        "key_bits",
        "value_bits",
        "mode",
    ]
    summary_rows = aggregate_rows(rows, group_keys=group_keys)
    append_jsonl(output_dir / "metrics.jsonl", rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_markdown(output_dir / "summary.md", summary_rows)
    write_csv(output_dir / "raw_metrics.csv", rows)

    if tensor_samples:
        torch.save(
            {
                "args": _jsonable_args(args),
                "samples": tensor_samples,
            },
            output_dir / "tensors_sample.pt",
        )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_raw_rows": len(rows),
                "num_summary_rows": len(summary_rows),
            },
            indent=2,
        )
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pure-PyTorch TurboQuant KV-cache correctness benchmark"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--seq-lens",
        default=None,
        help="Comma-separated sequence length sweep. Overrides --seq-len.",
    )
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--num-query-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--head-dims",
        default=None,
        help="Comma-separated head-dimension sweep. Overrides --head-dim.",
    )
    parser.add_argument("--key-bits", default="3,4")
    parser.add_argument("--value-bits", default="3,4")
    parser.add_argument("--qjl-dim", default="128", help="Integer or 'auto'.")
    parser.add_argument("--slot-align", type=int, default=16)
    parser.add_argument(
        "--progress",
        choices=["auto", "always", "never"],
        default="auto",
        help="Show a tqdm progress bar. auto shows it only on an interactive terminal.",
    )
    parser.add_argument("--rotation", default="both", help="hadamard, qr, or both")
    parser.add_argument("--scenario", default="decode,prefill")
    parser.add_argument("--data-distribution", default="mixed_non_gaussian")
    parser.add_argument("--modes", default="default")
    parser.add_argument("--include-no-quant", action="store_true")
    parser.add_argument("--disable-norm-correction", action="store_true")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--metadata-dtype", default="float16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--save-tensors",
        choices=["none", "samples", "all"],
        default="samples",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/ubuntu/sasmit/vllm-research/turboquant_kv_benchmark/results",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
