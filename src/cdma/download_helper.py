import logging
import zipfile
from pathlib import Path

from cdma.config import GOOGLE_DRIVE_FEATURE_ARCHIVE_ID

LOGGER = logging.getLogger(__name__)


def download_and_extract_feature_archive(
    target_zip_path: Path,
    extract_root: Path,
    file_id: str = GOOGLE_DRIVE_FEATURE_ARCHIVE_ID,
    force_download: bool = False,
) -> Path:
    """Download the Androids feature archive from Google Drive and extract it."""
    try:
        import gdown
    except ImportError as import_error:
        raise RuntimeError(
            "gdown is required for --download-features. Install dependencies from requirements.txt first."
        ) from import_error

    target_zip_path.parent.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    if target_zip_path.exists() and not force_download:
        LOGGER.info("Feature archive already exists: %s", target_zip_path)
    else:
        LOGGER.info("Downloading feature archive from Google Drive (id=%s)", file_id)
        gdown.download(id=file_id, output=str(target_zip_path), quiet=False)
        if not target_zip_path.exists():
            raise RuntimeError(f"Feature archive download failed: {target_zip_path}")

    LOGGER.info("Extracting archive to: %s", extract_root)
    with zipfile.ZipFile(target_zip_path, mode="r") as zip_file:
        zip_file.extractall(path=extract_root)

    return extract_root
