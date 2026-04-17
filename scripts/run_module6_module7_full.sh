#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

RESULTS_ROOT="${1:-results/module7_full_run}"
MODULE6_DIR="$RESULTS_ROOT/module6"
MODULE7_DIR="$RESULTS_ROOT/module7"

python run_module6.py \
  --download-data \
  --conditions all13 \
  --reps 10 \
  --rep-start 1 \
  --epochs 300 \
  --batch-size 16 \
  --output-mode overwrite \
  --results-dir "$MODULE6_DIR"

python run_module7.py \
  --results-dir "$MODULE6_DIR" \
  --output-dir "$MODULE7_DIR"

echo "Module 6 outputs: $MODULE6_DIR"
echo "Module 7 outputs: $MODULE7_DIR"
