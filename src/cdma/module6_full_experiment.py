from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
import statistics
import time
import zipfile
from pathlib import Path

from cdma.config import GOOGLE_DRIVE_FEATURE_ARCHIVE_ID
from cdma.module4_cdma import BATCH_SIZE, CHECKPOINT_EVERY_EPOCHS, EPOCHS, LOG_EVERY_EPOCHS, PREVIEW_PARTICIPANTS
from cdma.module4_cdma import run_sanity_checks
from cdma.module5_experiment_runner import (
    DEFAULT_CONDITIONS,
    Module5RunConfig,
    configure_logging,
    parse_conditions,
    parse_reps,
    run_experiment_suite,
)

LOGGER = logging.getLogger(__name__)

THESIS_F1_BY_CONDITION = {
    "ba1_rt": 84.7,
    "ba1_it": 83.3,
    "itmla_rt": 89.0,
    "itmla_it": 87.4,
    "ba2_rt": 89.9,
    "ba2_it": 87.3,
    "ba3_rt": 90.2,
    "ba3_it": 89.7,
    "ctga_rt": 90.7,
    "ctga_it": 89.8,
    "ba4": 90.1,
    "ba5": 90.7,
    "full_cdma": 92.5,
}

LABEL_BY_CONDITION = {
    "ba1_rt": "BA1 (Read)",
    "ba1_it": "BA1 (Spont.)",
    "itmla_rt": "IT-MLA (Read)",
    "itmla_it": "IT-MLA (Spont.)",
    "ba2_rt": "BA2 (Read)",
    "ba2_it": "BA2 (Interview)",
    "ba3_rt": "BA3 (Read)",
    "ba3_it": "BA3 (Spont.)",
    "ctga_rt": "CT-GA (Read)",
    "ctga_it": "CT-GA (Spont.)",
    "ba4": "BA4",
    "ba5": "BA5",
    "full_cdma": "CDMA",
}

COMPARISON_FIELDS = [
    "condition",
    "label",
    "thesis_f1",
    "observed_reps",
    "observed_f1_mean",
    "observed_f1_std",
    "delta_f1_points",
]


def _pooled_results_csv_path(results_dir: Path) -> Path:
    return results_dir / "pooled_results.csv"


def _comparison_csv_path(results_dir: Path) -> Path:
    return results_dir / "table7_2_comparison.csv"


def _comparison_report_path(results_dir: Path) -> Path:
    return results_dir / "table7_2_comparison_report.txt"


def _test_mode_report_path(results_dir: Path) -> Path:
    return results_dir / "module6_test_mode_report.txt"


def _find_first(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name)]
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    return matches[0]


def _find_dir(root: Path, dir_name: str) -> Path:
    matches = [path for path in root.rglob(dir_name) if path.is_dir()]
    if not matches:
        raise FileNotFoundError(f"Could not find directory {dir_name} under {root}")
    return matches[0]


def _count_feature_files(feature_dir: Path) -> int:
    if not feature_dir.exists():
        return 0
    return len(list(feature_dir.glob("*_frames.npy")))


def _bootstrap_data_from_gdrive(
    project_root: Path,
    results_dir: Path,
    gdrive_file_id: str,
    force_download: bool,
) -> None:
    try:
        import gdown
    except ImportError as import_error:
        raise RuntimeError(
            "gdown is required for --download-data. Install dependencies from requirements.txt first."
        ) from import_error

    archive_path = results_dir / "androids_features.zip"
    extract_dir = results_dir / "androids_extracted"
    data_dir = project_root / "data"
    fold_csv_path = data_dir / "fold-lists.csv"
    cdma_features_dir = data_dir / "cdma_features"

    results_dir.mkdir(parents=True, exist_ok=True)

    if force_download and archive_path.exists():
        archive_path.unlink()

    if archive_path.exists() and not force_download:
        LOGGER.info("Using cached feature archive: %s", archive_path)
    else:
        LOGGER.info("Downloading feature archive from Google Drive (id=%s)", gdrive_file_id)
        gdown.download(id=gdrive_file_id, output=str(archive_path), quiet=False)

    if force_download and extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Extracting feature archive to %s", extract_dir)
    with zipfile.ZipFile(archive_path, mode="r") as zip_file:
        zip_file.extractall(path=extract_dir)

    source_fold_csv = _find_first(extract_dir, "fold-lists.csv")
    source_cdma_features_dir = _find_dir(extract_dir, "cdma_features")

    data_dir.mkdir(parents=True, exist_ok=True)

    if fold_csv_path.exists():
        fold_csv_path.unlink()
    if cdma_features_dir.exists():
        shutil.rmtree(cdma_features_dir)

    shutil.copy2(source_fold_csv, fold_csv_path)
    shutil.copytree(source_cdma_features_dir, cdma_features_dir)

    LOGGER.info("Data prepared under %s", data_dir)


def _confirm_data_ready(project_root: Path) -> None:
    fold_csv_path = project_root / "data" / "fold-lists.csv"
    rt_feature_dir = project_root / "data" / "cdma_features" / "rt"
    it_feature_dir = project_root / "data" / "cdma_features" / "it"

    if not fold_csv_path.exists():
        raise FileNotFoundError(f"Missing fold list file: {fold_csv_path}")
    if not rt_feature_dir.exists():
        raise FileNotFoundError(f"Missing RT feature directory: {rt_feature_dir}")
    if not it_feature_dir.exists():
        raise FileNotFoundError(f"Missing IT feature directory: {it_feature_dir}")

    rt_file_count = _count_feature_files(rt_feature_dir)
    it_file_count = _count_feature_files(it_feature_dir)

    if rt_file_count <= 0 or it_file_count <= 0:
        raise RuntimeError(
            (
                "Feature directories exist but no frame files were found. "
                f"rt_files={rt_file_count}, it_files={it_file_count}"
            )
        )

    LOGGER.info(
        "Data readiness confirmed: fold_csv=%s rt_files=%d it_files=%d",
        fold_csv_path,
        rt_file_count,
        it_file_count,
    )


def _load_pooled_rows(pooled_results_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not pooled_results_path.exists():
        return rows

    with pooled_results_path.open(mode="r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "condition": row["condition"],
                    "rep": int(row["rep"]),
                    "accuracy": float(row["accuracy"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                }
            )

    return rows


def _build_comparison_rows(
    pooled_rows: list[dict[str, object]],
    target_conditions: list[str],
) -> list[dict[str, object]]:
    rows_by_condition: dict[str, list[float]] = {condition: [] for condition in target_conditions}

    for pooled_row in pooled_rows:
        condition = str(pooled_row["condition"])
        if condition not in rows_by_condition:
            continue
        rows_by_condition[condition].append(float(pooled_row["f1"]) * 100.0)

    comparison_rows: list[dict[str, object]] = []

    for condition in target_conditions:
        observed_f1_values = rows_by_condition[condition]
        observed_rep_count = len(observed_f1_values)
        thesis_f1 = THESIS_F1_BY_CONDITION.get(condition, float("nan"))

        if observed_rep_count == 0:
            observed_f1_mean = float("nan")
            observed_f1_std = float("nan")
            delta_f1_points = float("nan")
        else:
            observed_f1_mean = float(statistics.mean(observed_f1_values))
            observed_f1_std = float(statistics.pstdev(observed_f1_values)) if observed_rep_count > 1 else 0.0
            delta_f1_points = observed_f1_mean - thesis_f1

        comparison_rows.append(
            {
                "condition": condition,
                "label": LABEL_BY_CONDITION.get(condition, condition),
                "thesis_f1": thesis_f1,
                "observed_reps": observed_rep_count,
                "observed_f1_mean": observed_f1_mean,
                "observed_f1_std": observed_f1_std,
                "delta_f1_points": delta_f1_points,
            }
        )

    return comparison_rows


def _format_metric(value: float) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value:.2f}"


def _save_comparison_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode="w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "condition": row["condition"],
                    "label": row["label"],
                    "thesis_f1": _format_metric(float(row["thesis_f1"])),
                    "observed_reps": int(row["observed_reps"]),
                    "observed_f1_mean": _format_metric(float(row["observed_f1_mean"])),
                    "observed_f1_std": _format_metric(float(row["observed_f1_std"])),
                    "delta_f1_points": _format_metric(float(row["delta_f1_points"])),
                }
            )


def _save_comparison_report(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "=== Module 6: Table 7.2 Comparison ===",
        "",
        "Columns: thesis_f1, observed_f1_mean, observed_f1_std, delta_f1_points",
        "",
        "condition | thesis_f1 | observed_reps | observed_f1_mean | observed_f1_std | delta_f1_points",
    ]

    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row["condition"]),
                    _format_metric(float(row["thesis_f1"])),
                    str(int(row["observed_reps"])),
                    _format_metric(float(row["observed_f1_mean"])),
                    _format_metric(float(row["observed_f1_std"])),
                    _format_metric(float(row["delta_f1_points"])),
                ]
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_test_mode(results_dir: Path) -> int:
    start_time = time.time()
    sanity_result = run_sanity_checks()
    elapsed_seconds = time.time() - start_time

    lines = [
        "=== Module 6 Test Mode ===",
        f"elapsed_seconds: {elapsed_seconds:.3f}",
        f"all_modes_forward_ok: {sanity_result.all_modes_forward_ok}",
        (
            "output_key_mismatch_modes: "
            + (", ".join(sanity_result.output_key_mismatch_modes) if sanity_result.output_key_mismatch_modes else "none")
        ),
        f"p_hat_shape_ok: {sanity_result.p_hat_shape_ok}",
        f"no_nan_outputs: {sanity_result.no_nan_outputs}",
        f"loss_scaling_ok: {sanity_result.loss_scaling_ok}",
        f"gradient_flow_ok: {sanity_result.gradient_flow_ok}",
    ]

    report_path = _test_mode_report_path(results_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for line in lines:
        LOGGER.info(line)

    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 6 full experiment runner for all 13 conditions.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root path.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "module6",
        help="Output directory for experiment artifacts.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="all13",
        help="Comma-separated condition list, or all13.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=10,
        help="Number of repetitions to run.",
    )
    parser.add_argument(
        "--rep-start",
        type=int,
        default=1,
        help="Starting repetition index (inclusive).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Training epochs per fold.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Participant-level batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--log-every-epochs",
        type=int,
        default=LOG_EVERY_EPOCHS,
        help="Print epoch loss every N epochs (plus first and last).",
    )
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=CHECKPOINT_EVERY_EPOCHS,
        help="Save checkpoints every N epochs (plus final).",
    )
    parser.add_argument(
        "--preview-participants",
        type=int,
        default=PREVIEW_PARTICIPANTS,
        help="How many participant predictions to preview after each fold.",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        default="overwrite",
        choices=["detailed", "overwrite"],
        help="detailed=per-fold files + all-fold files, overwrite=single all-fold files per condition/rep.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick validation mode for all 13 modes (target under 30 seconds).",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Training smoke mode: 1 rep, first 2 conditions, 3 epochs.",
    )
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip Module 4 sanity checks before training suite.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume and force all requested runs.",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download and prepare data from Google Drive before running experiments.",
    )
    parser.add_argument(
        "--force-data-download",
        action="store_true",
        help="Force re-download and re-extract of the dataset archive.",
    )
    parser.add_argument(
        "--gdrive-file-id",
        type=str,
        default=GOOGLE_DRIVE_FEATURE_ARCHIVE_ID,
        help="Google Drive file id for the Androids feature archive.",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    cli_args = parse_arguments()

    project_root = cli_args.project_root.resolve()
    results_dir = cli_args.results_dir if cli_args.results_dir.is_absolute() else project_root / cli_args.results_dir

    if cli_args.test:
        LOGGER.info("Running Module 6 test mode using dummy-data sanity checks across all 13 modes.")
        return _run_test_mode(results_dir=results_dir)

    if cli_args.download_data:
        _bootstrap_data_from_gdrive(
            project_root=project_root,
            results_dir=results_dir,
            gdrive_file_id=cli_args.gdrive_file_id,
            force_download=cli_args.force_data_download,
        )

    try:
        _confirm_data_ready(project_root)
    except Exception as data_error:
        raise RuntimeError(
            (
                "Dataset setup check failed. Ensure data/fold-lists.csv and data/cdma_features/{rt,it} exist, "
                "or re-run with --download-data."
            )
        ) from data_error

    requested_conditions = parse_conditions(cli_args.conditions)
    requested_reps = parse_reps(rep_count=cli_args.reps, rep_start=cli_args.rep_start)
    requested_epochs = cli_args.epochs

    if cli_args.quick_test:
        requested_conditions = requested_conditions[:2]
        requested_reps = requested_reps[:1]
        requested_epochs = 3

    run_config = Module5RunConfig(
        project_root=project_root,
        results_dir=results_dir,
        conditions=requested_conditions,
        reps=requested_reps,
        epochs=requested_epochs,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        log_every_epochs=cli_args.log_every_epochs,
        checkpoint_every_epochs=cli_args.checkpoint_every_epochs,
        preview_participants=cli_args.preview_participants,
        resume=not cli_args.no_resume,
        output_mode=cli_args.output_mode,
        skip_sanity_check=cli_args.skip_sanity_check,
    )

    LOGGER.info(
        "Starting Module 6 with conditions=%s reps=%s epochs=%d output_mode=%s results_dir=%s",
        run_config.conditions,
        run_config.reps,
        run_config.epochs,
        run_config.output_mode,
        run_config.results_dir,
    )

    run_experiment_suite(run_config)

    pooled_rows = _load_pooled_rows(_pooled_results_csv_path(run_config.results_dir))
    comparison_rows = _build_comparison_rows(
        pooled_rows=pooled_rows,
        target_conditions=run_config.conditions,
    )

    comparison_csv_path = _comparison_csv_path(run_config.results_dir)
    comparison_report_path = _comparison_report_path(run_config.results_dir)

    _save_comparison_csv(comparison_csv_path, comparison_rows)
    _save_comparison_report(comparison_report_path, comparison_rows)

    LOGGER.info("Saved Table 7.2 comparison CSV to %s", comparison_csv_path)
    LOGGER.info("Saved Table 7.2 comparison report to %s", comparison_report_path)

    for row in comparison_rows:
        LOGGER.info(
            (
                "condition=%s reps=%d thesis_f1=%s observed_f1_mean=%s observed_f1_std=%s delta_points=%s"
            ),
            row["condition"],
            int(row["observed_reps"]),
            _format_metric(float(row["thesis_f1"])),
            _format_metric(float(row["observed_f1_mean"])),
            _format_metric(float(row["observed_f1_std"])),
            _format_metric(float(row["delta_f1_points"])),
        )

    missing_conditions = [
        condition
        for condition in run_config.conditions
        if condition not in DEFAULT_CONDITIONS
    ]
    if missing_conditions:
        LOGGER.warning("Unexpected condition names in run config: %s", missing_conditions)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
