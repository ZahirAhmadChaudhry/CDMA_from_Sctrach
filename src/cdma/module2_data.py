from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cdma.config import Module1Config, default_module1_config
from cdma.module1_validation import (
    deduplicate_preserving_order,
    find_cross_fold_duplicates,
    index_feature_files,
    parse_fold_lists,
    participant_label_from_id,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParticipantSample:
    participant_id: str
    label: int
    rt_frames: torch.Tensor
    it_frames: torch.Tensor
    n_rt: int
    n_it: int


@dataclass(frozen=True)
class Module2SplitInfo:
    filtered_rt_folds: dict[str, list[str]]
    train_participant_ids: list[str]
    test_participant_ids: list[str]
    all_participant_ids: list[str]
    missing_rt_fold_ids: set[str]
    rt_file_index: dict[str, Path]
    it_file_index: dict[str, Path]


@dataclass(frozen=True)
class Module2DataLoaders:
    train_loader: DataLoader
    test_loader: DataLoader
    train_dataset: "AndroidsDataset"
    test_dataset: "AndroidsDataset"
    normalizer: "FeatureNormalizer"
    split_info: Module2SplitInfo


class FeatureNormalizer:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.fitted_vector_count_: int = 0

    def fit(
        self,
        participant_ids: list[str],
        rt_file_index: dict[str, Path],
        it_file_index: dict[str, Path],
    ) -> "FeatureNormalizer":
        if not participant_ids:
            raise ValueError("FeatureNormalizer.fit requires at least one participant.")

        feature_sum = np.zeros((32,), dtype=np.float64)
        feature_square_sum = np.zeros((32,), dtype=np.float64)
        total_vector_count = 0

        for participant_id in participant_ids:
            rt_frames = np.load(rt_file_index[participant_id]).astype(np.float32, copy=False)
            it_frames = np.load(it_file_index[participant_id]).astype(np.float32, copy=False)

            stream_arrays = [rt_frames, it_frames]
            for stream_array in stream_arrays:
                flattened_array = stream_array.reshape(-1, stream_array.shape[-1]).astype(
                    np.float64,
                    copy=False,
                )
                feature_sum = feature_sum + flattened_array.sum(axis=0)
                feature_square_sum = feature_square_sum + np.square(flattened_array).sum(axis=0)
                total_vector_count = total_vector_count + int(flattened_array.shape[0])

        if total_vector_count == 0:
            raise ValueError("Cannot fit normalizer with zero vectors.")

        mean_vector = feature_sum / total_vector_count
        variance_vector = feature_square_sum / total_vector_count - np.square(mean_vector)
        variance_vector = np.maximum(variance_vector, self.epsilon)

        self.mean_ = mean_vector.astype(np.float32)
        self.std_ = np.sqrt(variance_vector).astype(np.float32)
        self.fitted_vector_count_ = total_vector_count

        return self

    def _assert_fitted(self) -> None:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FeatureNormalizer must be fitted before transform.")

    def transform(self, frame_array: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        normalized_array = (frame_array.astype(np.float32, copy=False) - self.mean_) / self.std_
        return normalized_array.astype(np.float32, copy=False)


class AndroidsDataset(Dataset):
    def __init__(
        self,
        participant_ids: list[str],
        rt_file_index: dict[str, Path],
        it_file_index: dict[str, Path],
        label_map: dict[str, int],
        normalizer: FeatureNormalizer,
    ) -> None:
        self.participant_ids = list(participant_ids)
        self.participant_id_to_index: dict[str, int] = {}
        self.samples: list[ParticipantSample] = []

        for sample_index, participant_id in enumerate(self.participant_ids):
            rt_raw_frames = np.load(rt_file_index[participant_id]).astype(np.float32, copy=False)
            it_raw_frames = np.load(it_file_index[participant_id]).astype(np.float32, copy=False)

            rt_normalized_frames = normalizer.transform(rt_raw_frames)
            it_normalized_frames = normalizer.transform(it_raw_frames)

            sample = ParticipantSample(
                participant_id=participant_id,
                label=label_map[participant_id],
                rt_frames=torch.from_numpy(rt_normalized_frames.copy()),
                it_frames=torch.from_numpy(it_normalized_frames.copy()),
                n_rt=int(rt_normalized_frames.shape[0]),
                n_it=int(it_normalized_frames.shape[0]),
            )

            self.participant_id_to_index[participant_id] = sample_index
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        return {
            "pid": sample.participant_id,
            "label": sample.label,
            "rt_frames": sample.rt_frames,
            "it_frames": sample.it_frames,
            "n_rt": sample.n_rt,
            "n_it": sample.n_it,
        }


def collate_fn(batch_samples: list[dict[str, object]]) -> dict[str, object]:
    if not batch_samples:
        raise ValueError("collate_fn received an empty batch.")

    batch_size = len(batch_samples)
    max_n_rt = 0
    max_n_it = 0

    for batch_sample in batch_samples:
        n_rt = int(batch_sample["n_rt"])
        n_it = int(batch_sample["n_it"])
        if n_rt > max_n_rt:
            max_n_rt = n_rt
        if n_it > max_n_it:
            max_n_it = n_it

    sample_rt_tensor = batch_samples[0]["rt_frames"]
    sample_it_tensor = batch_samples[0]["it_frames"]
    if not isinstance(sample_rt_tensor, torch.Tensor) or not isinstance(sample_it_tensor, torch.Tensor):
        raise TypeError("Batch samples must contain torch.Tensor values for frame tensors.")

    frame_size = int(sample_rt_tensor.shape[1])
    feature_dim = int(sample_rt_tensor.shape[2])

    rt_frames = torch.zeros((batch_size, max_n_rt, frame_size, feature_dim), dtype=torch.float32)
    it_frames = torch.zeros((batch_size, max_n_it, frame_size, feature_dim), dtype=torch.float32)
    rt_mask = torch.zeros((batch_size, max_n_rt), dtype=torch.float32)
    it_mask = torch.zeros((batch_size, max_n_it), dtype=torch.float32)

    n_rt_tensor = torch.zeros((batch_size,), dtype=torch.long)
    n_it_tensor = torch.zeros((batch_size,), dtype=torch.long)
    labels_tensor = torch.zeros((batch_size,), dtype=torch.float32)
    participant_ids: list[str] = []

    for batch_index, batch_sample in enumerate(batch_samples):
        participant_ids.append(str(batch_sample["pid"]))

        n_rt = int(batch_sample["n_rt"])
        n_it = int(batch_sample["n_it"])

        rt_sample_tensor = batch_sample["rt_frames"]
        it_sample_tensor = batch_sample["it_frames"]
        if not isinstance(rt_sample_tensor, torch.Tensor) or not isinstance(it_sample_tensor, torch.Tensor):
            raise TypeError("Batch samples must contain torch.Tensor values for frame tensors.")

        rt_frames[batch_index, :n_rt] = rt_sample_tensor
        it_frames[batch_index, :n_it] = it_sample_tensor
        rt_mask[batch_index, :n_rt] = 1.0
        it_mask[batch_index, :n_it] = 1.0

        n_rt_tensor[batch_index] = n_rt
        n_it_tensor[batch_index] = n_it
        labels_tensor[batch_index] = float(batch_sample["label"])

    return {
        "rt_frames": rt_frames,
        "it_frames": it_frames,
        "rt_mask": rt_mask,
        "it_mask": it_mask,
        "n_rt": n_rt_tensor,
        "n_it": n_it_tensor,
        "labels": labels_tensor,
        "pids": participant_ids,
    }


def build_label_map(participant_ids: list[str]) -> dict[str, int]:
    label_map: dict[str, int] = {}
    for participant_id in participant_ids:
        label_map[participant_id] = participant_label_from_id(participant_id)
    return label_map


def build_rt_only_split(config: Module1Config, test_fold_name: str) -> Module2SplitInfo:
    rt_folds = parse_fold_lists(config.fold_csv_path, config.rt_fold_columns)
    rt_duplicates = find_cross_fold_duplicates(rt_folds)
    if rt_duplicates:
        duplicate_ids = sorted(rt_duplicates.keys())
        raise ValueError("RT folds contain duplicate participant assignments: " + ", ".join(duplicate_ids))

    rt_file_index = index_feature_files(config.rt_feature_dir)
    it_file_index = index_feature_files(config.it_feature_dir)

    rt_file_ids = set(rt_file_index.keys())
    it_file_ids = set(it_file_index.keys())
    both_stream_file_ids = rt_file_ids & it_file_ids

    filtered_rt_folds: dict[str, list[str]] = {}
    for fold_name, fold_participant_ids in rt_folds.items():
        filtered_participant_ids: list[str] = []
        for participant_id in fold_participant_ids:
            if participant_id in both_stream_file_ids:
                filtered_participant_ids.append(participant_id)
        filtered_rt_folds[fold_name] = filtered_participant_ids

    if test_fold_name not in filtered_rt_folds:
        raise ValueError(f"Unknown RT fold name: {test_fold_name}")

    missing_rt_fold_ids = set().union(*rt_folds.values()) - both_stream_file_ids

    test_participant_ids = deduplicate_preserving_order(filtered_rt_folds[test_fold_name])

    train_participant_ids_raw: list[str] = []
    for fold_name, fold_participant_ids in filtered_rt_folds.items():
        if fold_name == test_fold_name:
            continue
        for participant_id in fold_participant_ids:
            train_participant_ids_raw.append(participant_id)

    train_participant_ids = deduplicate_preserving_order(train_participant_ids_raw)

    train_set = set(train_participant_ids)
    test_set = set(test_participant_ids)
    overlap_ids = train_set & test_set
    if overlap_ids:
        raise ValueError(
            "Train/test split overlap detected in RT fold split: "
            + ", ".join(sorted(overlap_ids))
        )

    all_participant_ids = deduplicate_preserving_order(train_participant_ids + test_participant_ids)

    LOGGER.info(
        "RT-only split prepared. test_fold=%s train=%d test=%d total=%d",
        test_fold_name,
        len(train_participant_ids),
        len(test_participant_ids),
        len(all_participant_ids),
    )

    return Module2SplitInfo(
        filtered_rt_folds=filtered_rt_folds,
        train_participant_ids=train_participant_ids,
        test_participant_ids=test_participant_ids,
        all_participant_ids=all_participant_ids,
        missing_rt_fold_ids=missing_rt_fold_ids,
        rt_file_index=rt_file_index,
        it_file_index=it_file_index,
    )


def get_dataloaders(
    project_root: Path,
    test_fold_name: str,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Module2DataLoaders:
    config = default_module1_config(project_root.resolve())

    split_info = build_rt_only_split(config=config, test_fold_name=test_fold_name)
    label_map = build_label_map(split_info.all_participant_ids)

    normalizer = FeatureNormalizer()
    normalizer.fit(
        participant_ids=split_info.train_participant_ids,
        rt_file_index=split_info.rt_file_index,
        it_file_index=split_info.it_file_index,
    )

    train_dataset = AndroidsDataset(
        participant_ids=split_info.train_participant_ids,
        rt_file_index=split_info.rt_file_index,
        it_file_index=split_info.it_file_index,
        label_map=label_map,
        normalizer=normalizer,
    )
    test_dataset = AndroidsDataset(
        participant_ids=split_info.test_participant_ids,
        rt_file_index=split_info.rt_file_index,
        it_file_index=split_info.it_file_index,
        label_map=label_map,
        normalizer=normalizer,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return Module2DataLoaders(
        train_loader=train_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        normalizer=normalizer,
        split_info=split_info,
    )
