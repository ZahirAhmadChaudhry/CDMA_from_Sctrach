from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cdma.module2_data import FeatureNormalizer, Module2DataLoaders, collate_fn, get_dataloaders

LOGGER = logging.getLogger(__name__)
EXPECTED_EFFECTIVE_RT_FOLD_SIZES = {
    "fold1": 22,
    "fold2": 23,
    "fold3": 22,
    "fold4": 21,
    "fold5": 22,
}


@dataclass
class Module2ValidationReport:
    train_participant_count: int
    test_participant_count: int
    total_participant_count: int
    train_batch_count: int
    rt_fold_sizes_effective: dict[str, int]
    normalizer_mean_shape: tuple[int, ...]
    normalizer_std_shape: tuple[int, ...]
    normalized_train_mean_max_abs: float
    normalized_train_std_max_abs_error: float
    test_raw_train_mean_l2_gap: float
    test_raw_train_std_l2_gap: float
    checklist_items: list[tuple[str, bool, str]]
    notes: list[str] = field(default_factory=list)

    @property
    def all_checks_passed(self) -> bool:
        return all(check_item[1] for check_item in self.checklist_items)


def _stack_feature_stats(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    feature_sum = np.zeros((32,), dtype=np.float64)
    feature_square_sum = np.zeros((32,), dtype=np.float64)
    total_vector_count = 0

    for feature_array in arrays:
        flattened_array = feature_array.reshape(-1, feature_array.shape[-1]).astype(np.float64, copy=False)
        feature_sum = feature_sum + flattened_array.sum(axis=0)
        feature_square_sum = feature_square_sum + np.square(flattened_array).sum(axis=0)
        total_vector_count = total_vector_count + int(flattened_array.shape[0])

    if total_vector_count == 0:
        raise ValueError("Cannot compute feature statistics for zero vectors.")

    mean_vector = feature_sum / total_vector_count
    variance_vector = feature_square_sum / total_vector_count - np.square(mean_vector)
    variance_vector = np.maximum(variance_vector, 0.0)
    std_vector = np.sqrt(variance_vector)

    return mean_vector.astype(np.float32), std_vector.astype(np.float32), total_vector_count


def compute_raw_stats_for_participants(
    participant_ids: list[str],
    data_loaders: Module2DataLoaders,
) -> tuple[np.ndarray, np.ndarray, int]:
    feature_arrays: list[np.ndarray] = []

    for participant_id in participant_ids:
        rt_array = np.load(data_loaders.split_info.rt_file_index[participant_id]).astype(np.float32, copy=False)
        it_array = np.load(data_loaders.split_info.it_file_index[participant_id]).astype(np.float32, copy=False)
        feature_arrays.append(rt_array)
        feature_arrays.append(it_array)

    return _stack_feature_stats(feature_arrays)


def compute_normalized_dataset_stats(dataset) -> tuple[np.ndarray, np.ndarray, int]:
    feature_arrays: list[np.ndarray] = []

    for sample in dataset.samples:
        feature_arrays.append(sample.rt_frames.numpy())
        feature_arrays.append(sample.it_frames.numpy())

    return _stack_feature_stats(feature_arrays)


def verify_collate_batch(batch: dict[str, object]) -> tuple[bool, str]:
    rt_frames = batch["rt_frames"]
    it_frames = batch["it_frames"]
    rt_mask = batch["rt_mask"]
    it_mask = batch["it_mask"]
    n_rt = batch["n_rt"]
    n_it = batch["n_it"]
    labels = batch["labels"]
    participant_ids = batch["pids"]

    if not isinstance(rt_frames, torch.Tensor) or not isinstance(it_frames, torch.Tensor):
        return False, "Frame tensors are missing or invalid."

    if rt_frames.ndim != 4 or it_frames.ndim != 4:
        return False, f"Unexpected tensor ranks: rt={rt_frames.ndim}, it={it_frames.ndim}"

    if rt_frames.shape[0] != 4 or it_frames.shape[0] != 4:
        return False, f"Batch size mismatch for test batch: rt={rt_frames.shape[0]}, it={it_frames.shape[0]}"

    if rt_frames.shape[2] != 128 or rt_frames.shape[3] != 32:
        return False, f"RT tensor shape tail mismatch: {tuple(rt_frames.shape)}"

    if it_frames.shape[2] != 128 or it_frames.shape[3] != 32:
        return False, f"IT tensor shape tail mismatch: {tuple(it_frames.shape)}"

    if not isinstance(rt_mask, torch.Tensor) or not isinstance(it_mask, torch.Tensor):
        return False, "Mask tensors are missing."

    if not isinstance(n_rt, torch.Tensor) or not isinstance(n_it, torch.Tensor):
        return False, "Length tensors are missing."

    if not isinstance(labels, torch.Tensor):
        return False, "Labels tensor is missing."

    if labels.shape != (4,):
        return False, f"Labels shape mismatch: {tuple(labels.shape)}"

    if not isinstance(participant_ids, list) or len(participant_ids) != 4:
        return False, "Participant id list does not contain 4 items."

    for batch_index in range(4):
        sample_n_rt = int(n_rt[batch_index].item())
        sample_n_it = int(n_it[batch_index].item())

        if sample_n_rt < rt_frames.shape[1]:
            if not torch.all(rt_frames[batch_index, sample_n_rt:] == 0):
                return False, f"RT padded values are not zero for batch index {batch_index}"
        if sample_n_it < it_frames.shape[1]:
            if not torch.all(it_frames[batch_index, sample_n_it:] == 0):
                return False, f"IT padded values are not zero for batch index {batch_index}"

    rt_mask_sums = rt_mask.sum(dim=1).to(dtype=torch.long)
    it_mask_sums = it_mask.sum(dim=1).to(dtype=torch.long)
    if not torch.equal(rt_mask_sums, n_rt):
        return False, "RT mask sums do not match n_rt."
    if not torch.equal(it_mask_sums, n_it):
        return False, "IT mask sums do not match n_it."

    return True, "batch shapes, zero padding, and masks are valid"


def build_module2_validation_report(
    project_root: Path,
    test_fold_name: str,
    batch_size: int,
    num_workers: int,
) -> Module2ValidationReport:
    data_loaders = get_dataloaders(
        project_root=project_root,
        test_fold_name=test_fold_name,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_train=True,
    )

    train_ids = data_loaders.split_info.train_participant_ids
    test_ids = data_loaders.split_info.test_participant_ids

    train_mean_vector, train_std_vector, _ = compute_normalized_dataset_stats(data_loaders.train_dataset)
    test_raw_mean_vector, test_raw_std_vector, _ = compute_raw_stats_for_participants(
        participant_ids=test_ids,
        data_loaders=data_loaders,
    )

    if data_loaders.normalizer.mean_ is None or data_loaders.normalizer.std_ is None:
        raise RuntimeError("Normalizer is not fitted after dataloader construction.")

    train_mean_gap = float(np.max(np.abs(train_mean_vector)))
    train_std_gap = float(np.max(np.abs(train_std_vector - 1.0)))

    raw_train_mean_vector = data_loaders.normalizer.mean_
    raw_train_std_vector = data_loaders.normalizer.std_

    mean_l2_gap = float(np.linalg.norm(test_raw_mean_vector - raw_train_mean_vector))
    std_l2_gap = float(np.linalg.norm(test_raw_std_vector - raw_train_std_vector))

    sample_test_pid = test_ids[0]
    test_sample_index = data_loaders.test_dataset.participant_id_to_index[sample_test_pid]
    normalized_sample = data_loaders.test_dataset.samples[test_sample_index]
    raw_sample_rt = np.load(data_loaders.split_info.rt_file_index[sample_test_pid]).astype(np.float32, copy=False)
    expected_rt_from_train_stats = data_loaders.normalizer.transform(raw_sample_rt)

    uses_training_stats = np.allclose(
        normalized_sample.rt_frames.numpy(),
        expected_rt_from_train_stats,
        atol=1e-5,
    )

    test_only_normalizer = FeatureNormalizer().fit(
        participant_ids=test_ids,
        rt_file_index=data_loaders.split_info.rt_file_index,
        it_file_index=data_loaders.split_info.it_file_index,
    )
    differs_from_test_only_stats = not np.allclose(
        test_only_normalizer.mean_,
        data_loaders.normalizer.mean_,
        atol=1e-5,
    )

    collate_test_loader = DataLoader(
        dataset=data_loaders.train_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False,
    )
    collate_test_batch = next(iter(collate_test_loader))
    collate_ok, collate_detail = verify_collate_batch(collate_test_batch)

    filtered_rt_fold_sizes = {
        fold_name: len(participant_ids)
        for fold_name, participant_ids in data_loaders.split_info.filtered_rt_folds.items()
    }

    train_test_overlap = set(train_ids) & set(test_ids)

    checklist_items: list[tuple[str, bool, str]] = [
        (
            "Normalizer mean has shape (32,) and std has shape (32,)",
            data_loaders.normalizer.mean_.shape == (32,) and data_loaders.normalizer.std_.shape == (32,),
            f"mean_shape={data_loaders.normalizer.mean_.shape}, std_shape={data_loaders.normalizer.std_.shape}",
        ),
        (
            "Normalizer std has no zero values",
            bool(np.all(data_loaders.normalizer.std_ > 0.0)),
            f"min_std={float(np.min(data_loaders.normalizer.std_)):.6f}",
        ),
        (
            "Normalized training features are approximately mean=0",
            train_mean_gap < 1e-3,
            f"max_abs_mean={train_mean_gap:.6e}",
        ),
        (
            "Normalized training features are approximately std=1",
            train_std_gap < 1e-3,
            f"max_abs_std_error={train_std_gap:.6e}",
        ),
        (
            "Test dataset normalization uses training statistics",
            uses_training_stats and differs_from_test_only_stats,
            (
                f"uses_training_stats={uses_training_stats}, "
                f"differs_from_test_only_stats={differs_from_test_only_stats}"
            ),
        ),
        (
            "collate_fn output shapes and masks are valid for batch size 4",
            collate_ok,
            collate_detail,
        ),
        (
            "Training dataloader batch count is 5-6 for 88 participants at batch size 16",
            len(train_ids) == 88 and 5 <= len(data_loaders.train_loader) <= 6,
            f"train_participants={len(train_ids)}, train_batches={len(data_loaders.train_loader)}",
        ),
        (
            "Person independence: no overlap between train and test participants",
            len(train_test_overlap) == 0,
            f"overlap_count={len(train_test_overlap)}",
        ),
        (
            "RT-only effective fold sizes are 22,23,22,21,22",
            filtered_rt_fold_sizes == EXPECTED_EFFECTIVE_RT_FOLD_SIZES,
            f"sizes={filtered_rt_fold_sizes}",
        ),
    ]

    notes = [
        f"raw train-vs-test mean L2 gap={mean_l2_gap:.6f}",
        f"raw train-vs-test std L2 gap={std_l2_gap:.6f}",
        (
            "RT-only participants excluded from folds due to missing IT features: "
            + ", ".join(sorted(data_loaders.split_info.missing_rt_fold_ids))
        ),
    ]

    return Module2ValidationReport(
        train_participant_count=len(train_ids),
        test_participant_count=len(test_ids),
        total_participant_count=len(data_loaders.split_info.all_participant_ids),
        train_batch_count=len(data_loaders.train_loader),
        rt_fold_sizes_effective=filtered_rt_fold_sizes,
        normalizer_mean_shape=tuple(data_loaders.normalizer.mean_.shape),
        normalizer_std_shape=tuple(data_loaders.normalizer.std_.shape),
        normalized_train_mean_max_abs=train_mean_gap,
        normalized_train_std_max_abs_error=train_std_gap,
        test_raw_train_mean_l2_gap=mean_l2_gap,
        test_raw_train_std_l2_gap=std_l2_gap,
        checklist_items=checklist_items,
        notes=notes,
    )


def format_module2_report(report: Module2ValidationReport) -> str:
    output_lines: list[str] = [
        "=== Module 2: Dataset, Normalizer, DataLoader Report ===",
        "",
        "[Split Summary]",
        f"- Train participants: {report.train_participant_count}",
        f"- Test participants: {report.test_participant_count}",
        f"- Total participants: {report.total_participant_count}",
        f"- Effective RT fold sizes: {report.rt_fold_sizes_effective}",
        f"- Train batch count: {report.train_batch_count}",
        "",
        "[Normalizer Stats]",
        f"- mean shape: {report.normalizer_mean_shape}",
        f"- std shape: {report.normalizer_std_shape}",
        f"- train normalized mean max abs: {report.normalized_train_mean_max_abs:.6e}",
        f"- train normalized std max abs error: {report.normalized_train_std_max_abs_error:.6e}",
        f"- test raw vs train mean L2 gap: {report.test_raw_train_mean_l2_gap:.6f}",
        f"- test raw vs train std L2 gap: {report.test_raw_train_std_l2_gap:.6f}",
        "",
        "[Checklist]",
    ]

    for check_name, check_passed, check_detail in report.checklist_items:
        status = "PASS" if check_passed else "FAIL"
        output_lines.append(f"- [{status}] {check_name} ({check_detail})")

    output_lines.extend(["", "[Notes]"])
    for note in report.notes:
        output_lines.append(f"- {note}")

    output_lines.extend(
        [
            "",
            "[Module Scope]",
            "- Module 2 completed. No model training or Module 3 logic executed.",
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
    parser = argparse.ArgumentParser(description="Run Module 2 dataset and dataloader validation checks.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root path.",
    )
    parser.add_argument(
        "--test-fold",
        type=str,
        default="fold1",
        help="RT fold to use as test fold for this validation run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for train and test dataloaders.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("results") / "module2_validation_report.txt",
        help="Output report path.",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    cli_args = parse_arguments()

    project_root = cli_args.project_root.resolve()
    report_path = (
        cli_args.report_path
        if cli_args.report_path.is_absolute()
        else project_root / cli_args.report_path
    )

    report = build_module2_validation_report(
        project_root=project_root,
        test_fold_name=cli_args.test_fold,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
    )
    report_text = format_module2_report(report)

    print(report_text)
    save_report(report_text, report_path)

    LOGGER.info("Module 2 report saved to %s", report_path)
    LOGGER.info("Module 2 complete. Waiting for verification before Module 3.")

    return 0 if report.all_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
