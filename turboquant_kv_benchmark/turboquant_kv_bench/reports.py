"""Report writers for benchmark output bundles."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sort_value(value: Any) -> tuple[int, Any]:
    if isinstance(value, bool):
        return (2, str(value))
    if isinstance(value, int | float):
        return (0, value)
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _percentile(values: list[float], q: float) -> float:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    group_keys: list[str],
) -> list[dict[str, Any]]:
    """Aggregate raw per-trial rows into mean/p50/p95 rows."""
    deterministic_keys = {
        "raw_bytes_per_token_head",
        "bytes_per_token_head",
        "compression_ratio_vs_fp16",
        "slot_align",
    }
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(group_key) for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in grouped.items():
        summary = {group_key: value for group_key, value in zip(group_keys, key)}
        summary["num_trials"] = len(group)
        numeric_keys = sorted(
            {
                row_key
                for row in group
                for row_key, value in row.items()
                if row_key not in group_keys and _is_number(value)
            }
        )
        for row_key in numeric_keys:
            vals = [float(row[row_key]) for row in group if _is_number(row.get(row_key))]
            finite_vals = [val for val in vals if math.isfinite(val)]
            if not finite_vals:
                continue
            if row_key in deterministic_keys:
                summary[row_key] = finite_vals[0]
                continue
            summary[f"{row_key}_mean"] = statistics.mean(finite_vals)
            summary[f"{row_key}_p50"] = _percentile(finite_vals, 0.50)
            summary[f"{row_key}_p95"] = _percentile(finite_vals, 0.95)
        summaries.append(summary)
    return sorted(summaries, key=lambda row: tuple(_sort_value(row.get(k, "")) for k in group_keys))


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("# TurboQuant KV Benchmark\n\nNo rows produced.\n")
        return

    group_keys = [
        "scenario",
        "distribution",
        "rotation",
        "head_dim",
        "seq_len",
        "key_bits",
        "value_bits",
    ]
    columns = [
        "mode",
        "num_trials",
        "out_rel_l2_mean",
        "out_rel_l2_p50",
        "out_rel_l2_p95",
        "out_cosine_mean",
        "out_cosine_p50",
        "out_cosine_p95",
        "out_rmse_mean",
        "out_rmse_p50",
        "out_rmse_p95",
        "probs_kl_mean_mean",
        "probs_kl_mean_p50",
        "probs_kl_mean_p95",
        "logits_rmse_mean",
        "logits_rmse_p50",
        "logits_rmse_p95",
        "k_row_cosine_mean_mean",
        "k_row_cosine_mean_p50",
        "k_row_cosine_mean_p95",
        "v_row_cosine_mean_mean",
        "v_row_cosine_mean_p50",
        "v_row_cosine_mean_p95",
        "raw_bytes_per_token_head",
        "bytes_per_token_head",
        "slot_align",
        "compression_ratio_vs_fp16",
    ]
    available = [col for col in columns if col in rows[0]]
    lines = ["# TurboQuant KV Benchmark", ""]
    sorted_rows = sorted(
        rows,
        key=lambda row: tuple(_sort_value(row.get(k, "")) for k in group_keys + ["mode"]),
    )

    current_group: tuple[Any, ...] | None = None
    for row in sorted_rows:
        group = tuple(row.get(key) for key in group_keys)
        if group != current_group:
            if current_group is not None:
                lines.append("")
            current_group = group
            group_text = ", ".join(f"{key}={row.get(key)}" for key in group_keys)
            lines.append(f"## {group_text}")
            lines.append("")
            lines.append("| " + " | ".join(available) + " |")
            lines.append("| " + " | ".join(["---"] * len(available)) + " |")
        vals = []
        for col in available:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    path.write_text("\n".join(lines))
