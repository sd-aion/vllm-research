#!/usr/bin/env bash
set -euo pipefail

# Edit these values for the benchmark shape/sweep you want.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/bench_turboquant_kv.py" \
  --batch-size 2 \
  --seq-lens 128,512 \
  --num-trials 1000 \
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
  --scenario decode \
  --data-distribution mixed_non_gaussian \
  --modes default \
  --include-no-quant \
  --dtype float32 \
  --metadata-dtype float16 \
  --device cpu \
  --seed 123 \
  --atol 1e-3 \
  --rtol 1e-3 \
  --topk 5 \
  --save-tensors samples \
  --output-dir "${SCRIPT_DIR}/results"
