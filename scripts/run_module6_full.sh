#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="${1:-results/module6_full}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GDRIVE_FILE_ID="${GDRIVE_FILE_ID:-1LJenZ-VXktBbroTI3btVSkRRq-glSWCb}"
FORCE_DATA_DOWNLOAD="${FORCE_DATA_DOWNLOAD:-0}"

DATA_DIR="$PROJECT_ROOT/data"
FOLD_CSV_PATH="$DATA_DIR/fold-lists.csv"
RT_DIR_PATH="$DATA_DIR/cdma_features/rt"
IT_DIR_PATH="$DATA_DIR/cdma_features/it"

LOG_FILE="$RESULTS_DIR/module6_full_run.log"

mkdir -p "$RESULTS_DIR"

echo "[1/4] Python and CUDA environment check"
"$PYTHON_BIN" - <<'PY'
import torch
print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_name:", torch.cuda.get_device_name(0))
PY

echo "[2/4] Data bootstrap check (download from Google Drive if missing)"
PROJECT_ROOT="$PROJECT_ROOT" \
RESULTS_DIR="$RESULTS_DIR" \
GDRIVE_FILE_ID="$GDRIVE_FILE_ID" \
FORCE_DATA_DOWNLOAD="$FORCE_DATA_DOWNLOAD" \
"$PYTHON_BIN" - <<'PY'
import os
import shutil
import zipfile
from pathlib import Path

try:
  import gdown
except ImportError as import_error:
  raise RuntimeError(
    "gdown is not installed. Install dependencies first (requirements.txt)."
  ) from import_error


def find_first(root: Path, name: str) -> Path:
  matches = [path for path in root.rglob(name)]
  if not matches:
    raise FileNotFoundError(f"Could not find {name} under {root}")
  return matches[0]


def find_dir(root: Path, dir_name: str) -> Path:
  matches = [path for path in root.rglob(dir_name) if path.is_dir()]
  if not matches:
    raise FileNotFoundError(f"Could not find directory {dir_name} under {root}")
  return matches[0]


project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
results_dir = Path(os.environ["RESULTS_DIR"]).resolve()
file_id = os.environ["GDRIVE_FILE_ID"]
force_download = os.environ.get("FORCE_DATA_DOWNLOAD", "0") == "1"

data_dir = project_root / "data"
fold_csv_path = data_dir / "fold-lists.csv"
rt_dir_path = data_dir / "cdma_features" / "rt"
it_dir_path = data_dir / "cdma_features" / "it"

have_existing_data = (
  fold_csv_path.exists()
  and rt_dir_path.exists()
  and it_dir_path.exists()
  and any(rt_dir_path.glob("*_frames.npy"))
  and any(it_dir_path.glob("*_frames.npy"))
)

if have_existing_data and not force_download:
  print("Data already prepared. Skipping download.")
  print("fold-lists.csv:", fold_csv_path)
  print("RT files:", len(list(rt_dir_path.glob("*_frames.npy"))))
  print("IT files:", len(list(it_dir_path.glob("*_frames.npy"))))
else:
  archive_path = results_dir / "androids_features.zip"
  extract_dir = results_dir / "androids_extracted"

  results_dir.mkdir(parents=True, exist_ok=True)
  if force_download and archive_path.exists():
    archive_path.unlink()

  if force_download and extract_dir.exists():
    shutil.rmtree(extract_dir)
  extract_dir.mkdir(parents=True, exist_ok=True)

  print(f"Downloading Androids features from Google Drive (id={file_id})")
  gdown.download(id=file_id, output=str(archive_path), quiet=False)

  print(f"Extracting archive to: {extract_dir}")
  with zipfile.ZipFile(archive_path, "r") as zip_file:
    zip_file.extractall(extract_dir)

  source_fold_csv = find_first(extract_dir, "fold-lists.csv")
  source_cdma_dir = find_dir(extract_dir, "cdma_features")

  target_cdma_dir = data_dir / "cdma_features"
  data_dir.mkdir(parents=True, exist_ok=True)

  if fold_csv_path.exists():
    fold_csv_path.unlink()
  if target_cdma_dir.exists():
    shutil.rmtree(target_cdma_dir)

  shutil.copy2(source_fold_csv, fold_csv_path)
  shutil.copytree(source_cdma_dir, target_cdma_dir)

  print("Data preparation complete.")
  print("fold-lists.csv:", fold_csv_path)
  print("RT files:", len(list(rt_dir_path.glob("*_frames.npy"))))
  print("IT files:", len(list(it_dir_path.glob("*_frames.npy"))))
PY

echo "[3/4] Module 6 quick test mode"
"$PYTHON_BIN" run_module6.py --test --results-dir "$RESULTS_DIR" 2>&1 | tee -a "$LOG_FILE"

echo "[4/4] Module 6 full run: all13 x 10 reps"
"$PYTHON_BIN" run_module6.py \
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
echo "- $RESULTS_DIR/module6_test_mode_report.txt"
echo "- $LOG_FILE"
