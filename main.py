from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIRECTORY = PROJECT_ROOT / "src"

if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from cdma.module6_full_experiment import main as module6_main
from cdma.module7_statistics import run_statistical_analysis

DEFAULT_RESULTS_ROOT = Path("results") / "main_full_run"


def _run_module6(project_root: Path, module6_results_dir: Path, download_data: bool) -> int:
    module6_args = [
        "run_module6.py",
        "--project-root",
        str(project_root),
        "--results-dir",
        str(module6_results_dir),
        "--conditions",
        "all13",
        "--reps",
        "10",
        "--rep-start",
        "1",
        "--epochs",
        "300",
        "--batch-size",
        "16",
        "--output-mode",
        "overwrite",
    ]

    if download_data:
        module6_args.append("--download-data")

    previous_argv = list(sys.argv)
    try:
        sys.argv = module6_args
        return module6_main()
    finally:
        sys.argv = previous_argv


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-command CDMA pipeline runner. "
            "Default behavior: run full Module 6 (all13 x 10 reps) then Module 7 analysis."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root output directory. Module 6 and Module 7 outputs are saved under this path.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not trigger dataset download/bootstrap in Module 6.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip Module 6 and only run Module 7 using existing Module 6 outputs.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Run only Module 6 and skip Module 7 statistical analysis.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    results_root = args.results_root if args.results_root.is_absolute() else PROJECT_ROOT / args.results_root
    module6_results_dir = results_root / "module6"
    module7_results_dir = results_root / "module7"

    print(f"[main] Project root: {PROJECT_ROOT}")
    print(f"[main] Module 6 results: {module6_results_dir}")
    print(f"[main] Module 7 results: {module7_results_dir}")

    if not args.skip_training:
        print("[main] Running Module 6 full experiment...")
        module6_exit_code = _run_module6(
            project_root=PROJECT_ROOT,
            module6_results_dir=module6_results_dir,
            download_data=not args.skip_download,
        )
        if module6_exit_code != 0:
            print(f"[main] Module 6 failed with exit code {module6_exit_code}")
            return int(module6_exit_code)
        print("[main] Module 6 completed.")

    if args.skip_analysis:
        print("[main] Skipping Module 7 analysis (--skip-analysis).")
        return 0

    pooled_csv = module6_results_dir / "pooled_results.csv"
    fold_predictions_csv = module6_results_dir / "fold_predictions.csv"

    if not pooled_csv.exists():
        raise FileNotFoundError(
            f"Module 7 input not found: {pooled_csv}. "
            "Run without --skip-training, or provide existing outputs under --results-root."
        )

    if not fold_predictions_csv.exists():
        raise FileNotFoundError(
            f"Module 7 input not found: {fold_predictions_csv}. "
            "Run without --skip-training, or provide existing outputs under --results-root."
        )

    print("[main] Running Module 7 statistical analysis...")
    run_statistical_analysis(
        pooled_csv=str(pooled_csv),
        fold_predictions_csv=str(fold_predictions_csv),
        output_dir=str(module7_results_dir),
    )
    print("[main] Module 7 completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
