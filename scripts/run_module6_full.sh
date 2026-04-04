#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="${1:-results/module6_full}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FILE="$RESULTS_DIR/module6_full_run.log"

mkdir -p "$RESULTS_DIR"

echo "[1/3] Python and CUDA environment check"
python - <<'PY'
import torch
print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_name:", torch.cuda.get_device_name(0))
PY

echo "[2/3] Module 6 quick test mode"
python run_module6.py --test --results-dir "$RESULTS_DIR" 2>&1 | tee -a "$LOG_FILE"

echo "[3/3] Module 6 full run: all13 x 10 reps"
python run_module6.py \
  --conditions all13 \
  --reps 10 \
  --rep-start 1 \
  --epochs 300 \
  --batch-size 16 \
  --num-workers "$NUM_WORKERS" \
  --output-mode overwrite \
  --results-dir "$RESULTS_DIR" \
  2>&1 | tee -a "$LOG_FILE"

echo "Run completed."
echo "Main outputs:"
echo "- $RESULTS_DIR/pooled_results.csv"
echo "- $RESULTS_DIR/table7_2_comparison.csv"
echo "- $RESULTS_DIR/table7_2_comparison_report.txt"
echo "- $LOG_FILE"
