# TurboQuant KV Correctness Benchmark

This folder contains a pure-PyTorch benchmark for comparing reconstructed KV-cache behavior, not runtime speed. It compares vLLM's TurboQuant-style cache behavior against paper-style TurboQuant variants using synthetic non-normal Q/K/V tensors.

The important vLLM modeling choice is:

```text
vLLM TurboQuant:
  K = MSE TurboQuant key quantization with stored norm
  V = per-vector min/max affine quantization
```

The value side is intentionally not modeled as QJL or full TurboQuant for vLLM, because the current vLLM path stores values with a min/max affine quantizer.

## Compared Modes

- `vllm_style`: MSE/norm-corrected K plus min/max affine V.
- `google_mse_kv`: MSE TurboQuant reconstruction for both K and V.
- `google_prod_score_k_mse_v`: product-style K score estimator plus MSE V. QJL corrects the K residual contribution directly in attention logits instead of reconstructing a full K vector.
- `google_prod_kv`: paper-style product reconstruction for both K and V. This uses MSE with one fewer bit plus a 1-bit QJL residual sketch.
- `no_quant`: optional exact baseline row.


## Google/Paper Product Algorithm Used Here

For one vector `x` with dimension `D`, the benchmark does:

```text
rho = ||x||_2
u = x / rho
z = u R
```

`R` is either a QR-generated random orthonormal matrix or a vLLM-style normalized Hadamard matrix.

For target bit width `b`, product mode uses:

```text
MSE bits = b - 1
QJL bits = 1
```

The MSE part quantizes each coordinate of `z` to a Lloyd-Max centroid for `N(0, 1 / D)`, reconstructs the rotated direction, optionally normalizes it back to unit length, applies the saved norm, and rotates back.

The residual part computes:

```text
e = x - x_mse
gamma = ||e||_2
w = e / gamma
qjl = sign(S w)
e_hat = sqrt(pi / 2) / m * gamma * S^T qjl
x_hat = x_mse + e_hat
```

`S` is a random projection matrix with shape `[m, D]`, where `m` is `--qjl-dim`.

`google_prod_score_k_mse_v` uses the product sketch differently. It does not use QJL to build a full reconstructed K vector. It computes:

```text
score(q, k) = q · k_mse + qjl_estimate(q, k - k_mse)
```

Then it uses MSE-reconstructed V for the final weighted sum. This better matches the fact that keys are used through dot products, while values must still be available as vectors for `softmax(scores) @ V`.

## Example

```bash
python /home/ubuntu/sasmit/vllm-research/turboquant_kv_benchmark/bench_turboquant_kv.py \
  --batch-size 2 \
  --seq-lens 128,512,2048 \
  --num-trials 20 \
  --query-len 1 \
  --num-query-heads 8 \
  --num-kv-heads 2 \
  --head-dims 64,128,256 \
  --key-bits 3,4 \
  --value-bits 3,4 \
  --qjl-dim auto \
  --slot-align 16 \
  --progress auto \
  --rotation both \
  --scenario decode,prefill \
  --data-distribution mixed_non_gaussian \
  --seed 123
```

## Outputs

Each run creates:

```text
results/run_YYYYMMDD_HHMMSS/
  args.json
  config.json
  manifest.json
  metrics.jsonl
  summary.csv
  summary.md
  raw_metrics.csv
  tensors_sample.pt
```

`args.json` stores the raw CLI arguments. `config.json` stores expanded sweeps. `manifest.json` stores environment and source-reference metadata. `metrics.jsonl` and `raw_metrics.csv` contain one row per trial/case. `summary.csv` and `summary.md` group rows by scenario, distribution, rotation, head dimension, sequence length, and bit widths, then list modes inside each group. Summaries report mean, p50, and p95 for numeric quality metrics. `raw_bytes_per_token_head`, aligned `bytes_per_token_head`, `slot_align`, and `compression_ratio_vs_fp16` are deterministic per group, so they are reported as single values.

Tensor saving is controlled by:

```bash
--save-tensors none|samples|all
```

The default is `samples`, which stores small tensor slices. Use `all` only for small shapes.

Progress display is controlled by:

```bash
--progress auto|always|never
```

The default is `auto`, which shows a `tqdm` progress bar only on an interactive terminal. Use `always` for redirected logs or non-interactive shells, and `never` for silent batch jobs.

## Metrics

The benchmark reports:

- K and V reconstruction error: max absolute error, mean absolute error, RMSE, relative L2, cosine similarity, row-wise cosine, norm-ratio stats, and allclose status.
- Attention logits: same tensor metrics against unquantized `QK^T / sqrt(D)`.
- Attention probabilities: KL divergence, max/mean probability error, top-1 agreement, and top-k overlap.
- Final attention outputs: max absolute error, RMSE, relative L2, cosine similarity, and allclose status.
- Storage estimate: bytes per token-head and compression ratio versus fp16 K/V.

## Tests

```bash
python -m pytest /home/ubuntu/sasmit/vllm-research/turboquant_kv_benchmark/tests
```
