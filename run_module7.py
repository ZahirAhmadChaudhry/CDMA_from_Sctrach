from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIRECTORY = PROJECT_ROOT / "src"

if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from cdma.module7_statistics import run_statistical_analysis


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run Module 7 statistical analysis.")
    parser.add_argument("--results-dir", default="results/module4", help="Directory containing pooled_results.csv and fold_predictions.csv")
    parser.add_argument("--output-dir", default="results/module7", help="Directory where Module 7 outputs are saved")
    args = parser.parse_args()

    run_statistical_analysis(
        pooled_csv=str(Path(args.results_dir) / "pooled_results.csv"),
        fold_predictions_csv=str(Path(args.results_dir) / "fold_predictions.csv"),
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
