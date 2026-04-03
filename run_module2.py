from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIRECTORY = PROJECT_ROOT / "src"

if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from cdma.module2_validation import main


if __name__ == "__main__":
    raise SystemExit(main())
