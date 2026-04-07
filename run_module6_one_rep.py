from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIRECTORY = PROJECT_ROOT / "src"

if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from cdma.module6_full_experiment import main as module6_main


def main() -> int:
    # Defaults match Module 6 outputs, but force a single full sweep across all 13 conditions.
    default_arguments = [
        "--download-data",
        "--conditions", "all13",
        "--reps", "1",
        "--rep-start", "1",
        "--epochs", "300",
        "--batch-size", "16",
        "--output-mode", "overwrite",
        "--results-dir", "results/module6_one_rep",
    ]

    # User-supplied flags can override defaults by passing the same option later.
    sys.argv = [sys.argv[0], *default_arguments, *sys.argv[1:]]
    return module6_main()


if __name__ == "__main__":
    raise SystemExit(main())
