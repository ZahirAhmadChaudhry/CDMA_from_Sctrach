from dataclasses import dataclass
from pathlib import Path

GOOGLE_DRIVE_FEATURE_ARCHIVE_ID = "1LJenZ-VXktBbroTI3btVSkRRq-glSWCb"


@dataclass(frozen=True)
class Module1Config:
    project_root: Path
    fold_csv_path: Path
    rt_feature_dir: Path
    it_feature_dir: Path
    report_output_path: Path
    expected_sample_pid: str = "01_CF56_1"
    expected_sample_rt_shape: tuple[int, int, int] = (109, 128, 32)
    expected_sample_it_shape: tuple[int, int, int] = (290, 128, 32)
    frame_size: int = 128
    feature_dim: int = 32
    rt_fold_columns: tuple[int, int, int, int, int] = (0, 1, 2, 3, 4)
    it_fold_columns: tuple[int, int, int, int, int] = (7, 8, 9, 10, 11)


def default_module1_config(project_root: Path) -> Module1Config:
    normalized_root = project_root.resolve()
    return Module1Config(
        project_root=normalized_root,
        fold_csv_path=normalized_root / "data" / "fold-lists.csv",
        rt_feature_dir=normalized_root / "data" / "cdma_features" / "rt",
        it_feature_dir=normalized_root / "data" / "cdma_features" / "it",
        report_output_path=normalized_root / "results" / "module1_validation_report.txt",
    )
