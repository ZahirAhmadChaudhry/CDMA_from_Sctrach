from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from cdma.config import Module1Config, default_module1_config
from cdma.download_helper import download_and_extract_feature_archive

LOGGER = logging.getLogger(__name__)
EXPECTED_FOLD_SIZES = {"fold1": 23, "fold2": 23, "fold3": 22, "fold4": 22, "fold5": 22}


@dataclass
class FeatureQualityInfo:
    checked_file_count: int = 0
    dtype_violations: list[str] = field(default_factory=list)
    shape_violations: list[str] = field(default_factory=list)
    nan_only_files: list[str] = field(default_factory=list)
    zero_only_files: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return any(
            (
                self.dtype_violations,
                self.shape_violations,
                self.nan_only_files,
                self.zero_only_files,
            )
        )


@dataclass
class Module1ValidationReport:
    rt_file_count: int
    it_file_count: int
    both_stream_file_count: int
    shared_fold_participant_count: int
    usable_participant_count: int
    control_count: int
    depressed_count: int
    rt_fold_sizes: dict[str, int]
    it_fold_sizes: dict[str, int]
    rt_fold_balance: dict[str, dict[str, float | int]]
    it_fold_balance: dict[str, dict[str, float | int]]
    rt_missing_from_files: set[str]
    it_missing_from_files: set[str]
    rt_extra_files: set[str]
    it_extra_files: set[str]
    rt_cross_fold_duplicates: dict[str, list[str]]
    it_cross_fold_duplicates: dict[str, list[str]]
    sample_rt_shape: tuple[int, ...] | None
    sample_it_shape: tuple[int, ...] | None
    rt_quality: FeatureQualityInfo
    it_quality: FeatureQualityInfo
    checklist_items: list[tuple[str, bool, str]]
    warnings: list[str] = field(default_factory=list)

    @property
    def all_checks_passed(self) -> bool:
        return all(check_item[1] for check_item in self.checklist_items)


def clean_participant_id(raw_value: object) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, float) and np.isnan(raw_value):
        return None

    value = str(raw_value).strip()
    if not value or value.lower() == "nan":
        return None

    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        value = value[1:-1].strip()

    return value if value else None


def deduplicate_preserving_order(participant_ids: list[str]) -> list[str]:
    seen_ids: set[str] = set()
    ordered_ids: list[str] = []
    for participant_id in participant_ids:
        if participant_id in seen_ids:
            continue
        seen_ids.add(participant_id)
        ordered_ids.append(participant_id)
    return ordered_ids


def parse_fold_lists(
    fold_csv_path: Path,
    fold_columns: tuple[int, int, int, int, int],
) -> dict[str, list[str]]:
    fold_table = pd.read_csv(fold_csv_path, header=None)

    fold_mapping: dict[str, list[str]] = {}
    for fold_index, column_index in enumerate(fold_columns, start=1):
        fold_name_from_csv = clean_participant_id(fold_table.iat[1, column_index])
        fold_name = fold_name_from_csv or f"fold{fold_index}"

        fold_participants: list[str] = []
        for row_index in range(2, len(fold_table.index)):
            participant_id = clean_participant_id(fold_table.iat[row_index, column_index])
            if participant_id is None:
                continue
            fold_participants.append(participant_id)

        fold_mapping[fold_name] = deduplicate_preserving_order(fold_participants)

    return fold_mapping


def find_cross_fold_duplicates(fold_mapping: dict[str, list[str]]) -> dict[str, list[str]]:
    participant_fold_map: dict[str, list[str]] = defaultdict(list)
    for fold_name, participant_ids in fold_mapping.items():
        for participant_id in participant_ids:
            participant_fold_map[participant_id].append(fold_name)

    duplicate_mapping: dict[str, list[str]] = {}
    for participant_id, assigned_folds in participant_fold_map.items():
        if len(assigned_folds) > 1:
            duplicate_mapping[participant_id] = assigned_folds

    return duplicate_mapping


def index_feature_files(feature_directory: Path) -> dict[str, Path]:
    indexed_paths: dict[str, Path] = {}
    duplicate_ids: list[str] = []

    for feature_path in sorted(feature_directory.glob("*_frames.npy")):
        participant_id = feature_path.name.replace("_frames.npy", "")
        if participant_id in indexed_paths:
            duplicate_ids.append(participant_id)
            continue
        indexed_paths[participant_id] = feature_path

    if duplicate_ids:
        LOGGER.warning(
            "Duplicate frame files detected for IDs (kept first occurrence): %s",
            ", ".join(sorted(set(duplicate_ids))),
        )

    return indexed_paths


def participant_label_from_id(participant_id: str) -> int:
    participant_chunks = participant_id.split("_")
    if len(participant_chunks) < 2:
        raise ValueError(f"Unexpected participant id format: {participant_id}")

    cohort_prefix = participant_chunks[1][:2]
    if cohort_prefix in {"CF", "CM"}:
        return 0
    if cohort_prefix in {"PF", "PM"}:
        return 1

    raise ValueError(f"Unknown cohort label in participant id: {participant_id}")


def compute_fold_balance(
    fold_mapping: dict[str, list[str]], usable_participant_ids: set[str]
) -> dict[str, dict[str, float | int]]:
    fold_balance: dict[str, dict[str, float | int]] = {}

    for fold_name, fold_participants in fold_mapping.items():
        filtered_participants = [
            participant_id
            for participant_id in fold_participants
            if participant_id in usable_participant_ids
        ]
        depressed_count = sum(
            participant_label_from_id(participant_id) for participant_id in filtered_participants
        )
        control_count = len(filtered_participants) - depressed_count
        depressed_ratio = (
            depressed_count / len(filtered_participants)
            if filtered_participants
            else float("nan")
        )

        fold_balance[fold_name] = {
            "count": len(filtered_participants),
            "control": control_count,
            "depressed": depressed_count,
            "depressed_ratio": depressed_ratio,
        }

    return fold_balance


def validate_feature_quality(
    file_index: dict[str, Path], expected_frame_size: int, expected_feature_dim: int
) -> FeatureQualityInfo:
    quality_info = FeatureQualityInfo()

    for participant_id, feature_path in file_index.items():
        feature_array = np.load(feature_path, mmap_mode="r")
        quality_info.checked_file_count += 1

        if feature_array.dtype != np.float32:
            quality_info.dtype_violations.append(f"{participant_id}:{feature_array.dtype}")

        has_expected_shape = (
            feature_array.ndim == 3
            and feature_array.shape[1] == expected_frame_size
            and feature_array.shape[2] == expected_feature_dim
        )
        if not has_expected_shape:
            quality_info.shape_violations.append(
                f"{participant_id}:{tuple(feature_array.shape)}"
            )

        if np.isnan(feature_array).all():
            quality_info.nan_only_files.append(participant_id)

        if np.all(feature_array == 0.0):
            quality_info.zero_only_files.append(participant_id)

    return quality_info


def _format_id_sample(participant_ids: set[str], max_items: int = 8) -> str:
    sorted_ids = sorted(participant_ids)
    if not sorted_ids:
        return "none"

    sample_ids = sorted_ids[:max_items]
    suffix = "" if len(sorted_ids) <= max_items else f" ... (+{len(sorted_ids) - max_items} more)"
    return ", ".join(sample_ids) + suffix


def _fold_sizes(fold_mapping: dict[str, list[str]]) -> dict[str, int]:
    return {fold_name: len(participant_ids) for fold_name, participant_ids in fold_mapping.items()}


def build_module1_validation_report(
    config: Module1Config,
    continue_with_intersection: bool = True,
) -> Module1ValidationReport:
    rt_folds = parse_fold_lists(config.fold_csv_path, config.rt_fold_columns)
    it_folds = parse_fold_lists(config.fold_csv_path, config.it_fold_columns)

    rt_fold_ids = set().union(*rt_folds.values())
    it_fold_ids = set().union(*it_folds.values())

    rt_cross_fold_duplicates = find_cross_fold_duplicates(rt_folds)
    it_cross_fold_duplicates = find_cross_fold_duplicates(it_folds)

    rt_file_index = index_feature_files(config.rt_feature_dir)
    it_file_index = index_feature_files(config.it_feature_dir)

    rt_file_ids = set(rt_file_index.keys())
    it_file_ids = set(it_file_index.keys())

    rt_missing_from_files = rt_fold_ids - rt_file_ids
    it_missing_from_files = it_fold_ids - it_file_ids
    rt_extra_files = rt_file_ids - rt_fold_ids
    it_extra_files = it_file_ids - it_fold_ids

    both_stream_file_ids = rt_file_ids & it_file_ids
    shared_fold_participant_ids = rt_fold_ids & it_fold_ids

    if continue_with_intersection:
        usable_participant_ids = both_stream_file_ids & shared_fold_participant_ids
    else:
        usable_participant_ids = both_stream_file_ids
        if rt_missing_from_files or it_missing_from_files:
            missing_ids = sorted(rt_missing_from_files | it_missing_from_files)
            raise ValueError(
                "Missing frame files for fold participants: " + ", ".join(missing_ids)
            )

    control_count = 0
    depressed_count = 0
    for participant_id in usable_participant_ids:
        participant_label = participant_label_from_id(participant_id)
        if participant_label == 0:
            control_count += 1
        else:
            depressed_count += 1

    sample_rt_shape: tuple[int, ...] | None = None
    sample_it_shape: tuple[int, ...] | None = None
    if config.expected_sample_pid in rt_file_index:
        sample_rt_shape = tuple(np.load(rt_file_index[config.expected_sample_pid], mmap_mode="r").shape)
    if config.expected_sample_pid in it_file_index:
        sample_it_shape = tuple(np.load(it_file_index[config.expected_sample_pid], mmap_mode="r").shape)

    rt_quality = validate_feature_quality(
        file_index=rt_file_index,
        expected_frame_size=config.frame_size,
        expected_feature_dim=config.feature_dim,
    )
    it_quality = validate_feature_quality(
        file_index=it_file_index,
        expected_frame_size=config.frame_size,
        expected_feature_dim=config.feature_dim,
    )

    rt_fold_sizes = _fold_sizes(rt_folds)
    it_fold_sizes = _fold_sizes(it_folds)
    rt_fold_balance = compute_fold_balance(rt_folds, usable_participant_ids)
    it_fold_balance = compute_fold_balance(it_folds, usable_participant_ids)

    checklist_items: list[tuple[str, bool, str]] = [
        (
            "110 RT files found",
            len(rt_file_index) == 110,
            f"found {len(rt_file_index)}",
        ),
        (
            "110 IT files found",
            len(it_file_index) == 110,
            f"found {len(it_file_index)}",
        ),
        (
            "110 participants with both RT and IT",
            len(both_stream_file_ids) == 110,
            f"found {len(both_stream_file_ids)}",
        ),
        (
            "52 control and 58 depressed participants",
            control_count == 52 and depressed_count == 58,
            f"control={control_count}, depressed={depressed_count}",
        ),
        (
            "RT fold sizes are 23,23,22,22,22",
            rt_fold_sizes == EXPECTED_FOLD_SIZES,
            f"sizes={rt_fold_sizes}",
        ),
        (
            "IT fold sizes are 23,23,22,22,22",
            it_fold_sizes == EXPECTED_FOLD_SIZES,
            f"sizes={it_fold_sizes}",
        ),
        (
            "No participant appears in multiple RT folds",
            len(rt_cross_fold_duplicates) == 0,
            f"duplicate_count={len(rt_cross_fold_duplicates)}",
        ),
        (
            "No participant appears in multiple IT folds",
            len(it_cross_fold_duplicates) == 0,
            f"duplicate_count={len(it_cross_fold_duplicates)}",
        ),
        (
            "All RT fold participants exist in RT feature files",
            len(rt_missing_from_files) == 0,
            f"missing_count={len(rt_missing_from_files)}",
        ),
        (
            "All IT fold participants exist in IT feature files",
            len(it_missing_from_files) == 0,
            f"missing_count={len(it_missing_from_files)}",
        ),
        (
            "Sample 01_CF56_1 RT shape matches expected",
            sample_rt_shape == config.expected_sample_rt_shape,
            f"shape={sample_rt_shape}",
        ),
        (
            "Sample 01_CF56_1 IT shape matches expected",
            sample_it_shape == config.expected_sample_it_shape,
            f"shape={sample_it_shape}",
        ),
        (
            "RT feature files are float32 and numerically valid",
            not rt_quality.has_issues,
            (
                "dtype_violations="
                f"{len(rt_quality.dtype_violations)}, shape_violations={len(rt_quality.shape_violations)}, "
                f"all_nan={len(rt_quality.nan_only_files)}, all_zero={len(rt_quality.zero_only_files)}"
            ),
        ),
        (
            "IT feature files are float32 and numerically valid",
            not it_quality.has_issues,
            (
                "dtype_violations="
                f"{len(it_quality.dtype_violations)}, shape_violations={len(it_quality.shape_violations)}, "
                f"all_nan={len(it_quality.nan_only_files)}, all_zero={len(it_quality.zero_only_files)}"
            ),
        ),
    ]

    warnings: list[str] = []
    if rt_missing_from_files:
        warnings.append(
            "RT fold participants missing in files: " + _format_id_sample(rt_missing_from_files)
        )
    if it_missing_from_files:
        warnings.append(
            "IT fold participants missing in files: " + _format_id_sample(it_missing_from_files)
        )
    if rt_extra_files:
        warnings.append("RT files not referenced by folds: " + _format_id_sample(rt_extra_files))
    if it_extra_files:
        warnings.append("IT files not referenced by folds: " + _format_id_sample(it_extra_files))

    return Module1ValidationReport(
        rt_file_count=len(rt_file_index),
        it_file_count=len(it_file_index),
        both_stream_file_count=len(both_stream_file_ids),
        shared_fold_participant_count=len(shared_fold_participant_ids),
        usable_participant_count=len(usable_participant_ids),
        control_count=control_count,
        depressed_count=depressed_count,
        rt_fold_sizes=rt_fold_sizes,
        it_fold_sizes=it_fold_sizes,
        rt_fold_balance=rt_fold_balance,
        it_fold_balance=it_fold_balance,
        rt_missing_from_files=rt_missing_from_files,
        it_missing_from_files=it_missing_from_files,
        rt_extra_files=rt_extra_files,
        it_extra_files=it_extra_files,
        rt_cross_fold_duplicates=rt_cross_fold_duplicates,
        it_cross_fold_duplicates=it_cross_fold_duplicates,
        sample_rt_shape=sample_rt_shape,
        sample_it_shape=sample_it_shape,
        rt_quality=rt_quality,
        it_quality=it_quality,
        checklist_items=checklist_items,
        warnings=warnings,
    )


def _format_fold_balance(
    fold_balance: dict[str, dict[str, float | int]],
) -> list[str]:
    formatted_lines: list[str] = []
    for fold_name in sorted(fold_balance.keys()):
        balance_item = fold_balance[fold_name]
        depressed_ratio = float(balance_item["depressed_ratio"]) * 100
        formatted_lines.append(
            (
                f"  - {fold_name}: count={balance_item['count']}, control={balance_item['control']}, "
                f"depressed={balance_item['depressed']}, depressed_ratio={depressed_ratio:.2f}%"
            )
        )
    return formatted_lines


def format_module1_report(report: Module1ValidationReport) -> str:
    output_lines: list[str] = [
        "=== Module 1: Data Loading and Validation Report ===",
        "",
        "[Counts]",
        f"- RT frame files: {report.rt_file_count}",
        f"- IT frame files: {report.it_file_count}",
        f"- Participants with both streams in files: {report.both_stream_file_count}",
        f"- Participants shared by RT/IT folds: {report.shared_fold_participant_count}",
        f"- Usable participants (intersection policy): {report.usable_participant_count}",
        f"- Label counts on usable participants: control={report.control_count}, depressed={report.depressed_count}",
        "",
        "[Fold Sizes]",
        f"- RT fold sizes: {report.rt_fold_sizes}",
        f"- IT fold sizes: {report.it_fold_sizes}",
        "",
        "[Fold Class Balance: RT folds]",
    ]
    output_lines.extend(_format_fold_balance(report.rt_fold_balance))
    output_lines.extend(["", "[Fold Class Balance: IT folds]"])
    output_lines.extend(_format_fold_balance(report.it_fold_balance))

    output_lines.extend(
        [
            "",
            "[Sample Shape Check]",
            f"- 01_CF56_1 RT shape: {report.sample_rt_shape}",
            f"- 01_CF56_1 IT shape: {report.sample_it_shape}",
            "",
            "[Checklist]",
        ]
    )

    for check_name, check_passed, details in report.checklist_items:
        status = "PASS" if check_passed else "FAIL"
        output_lines.append(f"- [{status}] {check_name} ({details})")

    output_lines.extend(["", "[Warnings]"])
    if report.warnings:
        for warning in report.warnings:
            output_lines.append(f"- {warning}")
    else:
        output_lines.append("- none")

    output_lines.extend(
        [
            "",
            "[Module Scope]",
            "- Module 1 completed. No Dataset/DataLoader or model code executed.",
        ]
    )

    return "\n".join(output_lines)


def save_report(report_text: str, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text + "\n", encoding="utf-8")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Module 1 data loading and validation checks.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root path.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional custom path for the validation report output.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail when fold participants are missing in feature files.",
    )
    parser.add_argument(
        "--download-features",
        action="store_true",
        help="Download and extract feature archive before validation.",
    )
    parser.add_argument(
        "--feature-archive-path",
        type=Path,
        default=Path("data") / "androids_features.zip",
        help="Path for cached feature zip archive when download is requested.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of archive even if zip already exists.",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    cli_args = parse_arguments()

    config = default_module1_config(cli_args.project_root)
    if cli_args.report_path is not None:
        config = replace(config, report_output_path=cli_args.report_path.resolve())

    if cli_args.download_features:
        archive_path = (
            cli_args.feature_archive_path
            if cli_args.feature_archive_path.is_absolute()
            else config.project_root / cli_args.feature_archive_path
        )
        download_and_extract_feature_archive(
            target_zip_path=archive_path,
            extract_root=config.project_root / "data",
            force_download=cli_args.force_download,
        )

    validation_report = build_module1_validation_report(
        config=config,
        continue_with_intersection=not cli_args.strict_missing,
    )
    report_text = format_module1_report(validation_report)

    print(report_text)
    save_report(report_text, config.report_output_path)

    LOGGER.info("Module 1 report saved to %s", config.report_output_path)
    LOGGER.info("Module 1 complete. Waiting for verification before Module 2.")

    if cli_args.strict_missing and not validation_report.all_checks_passed:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
