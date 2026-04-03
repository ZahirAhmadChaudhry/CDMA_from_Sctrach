from __future__ import annotations

import argparse
import csv
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cdma.config import Module1Config, default_module1_config
from cdma.module1_validation import (
    deduplicate_preserving_order,
    index_feature_files,
    parse_fold_lists,
    participant_label_from_id,
)

LOGGER = logging.getLogger(__name__)

FEATURE_DIM = 32
LSTM_HIDDEN = 32
FRAME_SIZE = 128
EPOCHS = 300
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
THRESHOLD = 0.5
FRAME_EVAL_BATCH_SIZE = 256


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    stream: str
    use_itmla: bool


CONDITIONS: dict[str, ConditionSpec] = {
    "ba1_rt": ConditionSpec(name="ba1_rt", stream="rt", use_itmla=False),
    "ba1_it": ConditionSpec(name="ba1_it", stream="it", use_itmla=False),
    "itmla_rt": ConditionSpec(name="itmla_rt", stream="rt", use_itmla=True),
    "itmla_it": ConditionSpec(name="itmla_it", stream="it", use_itmla=True),
}


@dataclass(frozen=True)
class StreamSplitInfo:
    stream: str
    raw_folds: dict[str, list[str]]
    filtered_folds: dict[str, list[str]]
    train_participant_ids: list[str]
    test_participant_ids: list[str]
    all_participant_ids: list[str]
    missing_fold_participants: set[str]
    file_index: dict[str, Path]


@dataclass(frozen=True)
class FoldRunResult:
    condition: str
    stream: str
    rep: int
    seed: int
    fold_name: str
    device: str
    train_participant_count: int
    test_participant_count: int
    train_frame_count: int
    test_frame_count: int
    training_losses: list[float]
    metrics: dict[str, float]
    predictions: list[dict[str, object]]
    elapsed_seconds: float
    filtered_fold_sizes: dict[str, int]
    missing_fold_participants: set[str]


@dataclass(frozen=True)
class ConditionRunResult:
    condition: str
    stream: str
    rep: int
    fold_results: list[FoldRunResult]
    pooled_metrics: dict[str, float]
    pooled_prediction_count: int
    duplicate_prediction_ids: set[str]


@dataclass(frozen=True)
class ArchitectureCheckResult:
    itmla_shape_ok: bool
    itmla_formula_ok: bool
    itmla_max_diff: float
    lstm_context_shape_ok: bool
    lstm_probability_shape_ok: bool


class StreamFeatureNormalizer:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, participant_ids: list[str], file_index: dict[str, Path]) -> "StreamFeatureNormalizer":
        if not participant_ids:
            raise ValueError("StreamFeatureNormalizer.fit requires at least one participant.")

        feature_sum = np.zeros((FEATURE_DIM,), dtype=np.float64)
        feature_square_sum = np.zeros((FEATURE_DIM,), dtype=np.float64)
        total_vector_count = 0

        for participant_id in participant_ids:
            frame_array = np.load(file_index[participant_id]).astype(np.float32, copy=False)
            flattened_vectors = frame_array.reshape(-1, frame_array.shape[-1]).astype(np.float64, copy=False)
            feature_sum = feature_sum + flattened_vectors.sum(axis=0)
            feature_square_sum = feature_square_sum + np.square(flattened_vectors).sum(axis=0)
            total_vector_count = total_vector_count + int(flattened_vectors.shape[0])

        if total_vector_count == 0:
            raise ValueError("Cannot fit stream normalizer with zero vectors.")

        mean_vector = feature_sum / total_vector_count
        variance_vector = feature_square_sum / total_vector_count - np.square(mean_vector)
        variance_vector = np.maximum(variance_vector, self.epsilon)

        self.mean_ = mean_vector.astype(np.float32)
        self.std_ = np.sqrt(variance_vector).astype(np.float32)
        return self

    def transform(self, frame_array: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StreamFeatureNormalizer must be fitted before transform.")
        normalized_array = (frame_array.astype(np.float32, copy=False) - self.mean_) / self.std_
        return normalized_array.astype(np.float32, copy=False)


class FrameTrainingDataset(Dataset):
    def __init__(
        self,
        participant_ids: list[str],
        participant_frames: dict[str, torch.Tensor],
        label_map: dict[str, int],
    ) -> None:
        self.participant_frames = participant_frames
        self.label_map = label_map
        self.frame_index_map: list[tuple[str, int]] = []

        for participant_id in participant_ids:
            frame_count = int(participant_frames[participant_id].shape[0])
            for frame_index in range(frame_count):
                self.frame_index_map.append((participant_id, frame_index))

    def __len__(self) -> int:
        return len(self.frame_index_map)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        participant_id, frame_index = self.frame_index_map[index]
        frame_tensor = self.participant_frames[participant_id][frame_index]
        label_value = float(self.label_map[participant_id])
        label_tensor = torch.tensor(label_value, dtype=torch.float32)
        return frame_tensor, label_tensor


class ITMLALayer(nn.Module):
    def forward(self, frame_batch: torch.Tensor) -> torch.Tensor:
        frame_average = frame_batch.mean(dim=1, keepdim=True)
        cosine_similarities = torch_functional.cosine_similarity(
            frame_batch,
            frame_average.expand_as(frame_batch),
            dim=-1,
        )
        return frame_batch * (1.0 + cosine_similarities.unsqueeze(-1))


class LSTM1Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=FEATURE_DIM,
            hidden_size=LSTM_HIDDEN,
            batch_first=True,
        )
        self.classifier = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, frame_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, _ = self.lstm(frame_batch)
        context_vectors = hidden_states.mean(dim=1)
        probabilities = torch.sigmoid(self.classifier(context_vectors))
        return context_vectors, probabilities


class Chapter6FrameClassifier(nn.Module):
    def __init__(self, use_itmla: bool) -> None:
        super().__init__()
        self.use_itmla = use_itmla
        self.itmla = ITMLALayer() if use_itmla else None
        self.lstm1 = LSTM1Layer()

    def forward(self, frame_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        processed_frames = frame_batch
        if self.itmla is not None:
            processed_frames = self.itmla(processed_frames)
        context_vectors, frame_probabilities = self.lstm1(processed_frames)
        return context_vectors, frame_probabilities


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_label_map(participant_ids: list[str]) -> dict[str, int]:
    label_map: dict[str, int] = {}
    for participant_id in participant_ids:
        label_map[participant_id] = participant_label_from_id(participant_id)
    return label_map


def build_stream_split(
    config: Module1Config,
    stream: str,
    test_fold_name: str,
) -> StreamSplitInfo:
    if stream not in {"rt", "it"}:
        raise ValueError(f"Unsupported stream: {stream}")

    fold_columns = config.rt_fold_columns if stream == "rt" else config.it_fold_columns
    stream_feature_dir = config.rt_feature_dir if stream == "rt" else config.it_feature_dir

    raw_folds = parse_fold_lists(config.fold_csv_path, fold_columns)
    file_index = index_feature_files(stream_feature_dir)
    available_ids = set(file_index.keys())

    filtered_folds: dict[str, list[str]] = {}
    for fold_name, fold_ids in raw_folds.items():
        filtered_ids: list[str] = []
        for participant_id in fold_ids:
            if participant_id in available_ids:
                filtered_ids.append(participant_id)
        filtered_folds[fold_name] = deduplicate_preserving_order(filtered_ids)

    if test_fold_name not in filtered_folds:
        raise ValueError(f"Unknown fold name for stream {stream}: {test_fold_name}")

    raw_ids = set().union(*raw_folds.values())
    missing_fold_participants = raw_ids - available_ids

    test_participant_ids = deduplicate_preserving_order(filtered_folds[test_fold_name])

    train_ids_raw: list[str] = []
    for fold_name, fold_ids in filtered_folds.items():
        if fold_name == test_fold_name:
            continue
        for participant_id in fold_ids:
            train_ids_raw.append(participant_id)

    train_participant_ids = deduplicate_preserving_order(train_ids_raw)

    overlap_ids = set(train_participant_ids) & set(test_participant_ids)
    if overlap_ids:
        raise ValueError(
            "Train/test overlap detected in stream split: " + ", ".join(sorted(overlap_ids))
        )

    all_participant_ids = deduplicate_preserving_order(train_participant_ids + test_participant_ids)

    return StreamSplitInfo(
        stream=stream,
        raw_folds=raw_folds,
        filtered_folds=filtered_folds,
        train_participant_ids=train_participant_ids,
        test_participant_ids=test_participant_ids,
        all_participant_ids=all_participant_ids,
        missing_fold_participants=missing_fold_participants,
        file_index=file_index,
    )


def load_normalized_participant_frames(
    participant_ids: list[str],
    file_index: dict[str, Path],
    normalizer: StreamFeatureNormalizer,
) -> dict[str, torch.Tensor]:
    participant_frames: dict[str, torch.Tensor] = {}
    for participant_id in participant_ids:
        frame_array = np.load(file_index[participant_id]).astype(np.float32, copy=False)
        normalized_frames = normalizer.transform(frame_array)
        participant_frames[participant_id] = torch.from_numpy(normalized_frames.copy())
    return participant_frames


def train_frame_level_model(
    model: Chapter6FrameClassifier,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    condition: str,
    fold_name: str,
    rep: int,
) -> list[float]:
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss(reduction="mean")
    epoch_losses: list[float] = []

    for epoch_index in range(epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for frame_batch, label_batch in train_loader:
            frame_batch_device = frame_batch.to(device)
            label_batch_device = label_batch.unsqueeze(1).to(device)

            _, probability_batch = model(frame_batch_device)
            loss = criterion(probability_batch, label_batch_device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = running_loss + float(loss.item())
            batch_count = batch_count + 1

        mean_epoch_loss = running_loss / max(1, batch_count)
        epoch_losses.append(mean_epoch_loss)
        LOGGER.info(
            "condition=%s rep=%d fold=%s epoch=%d/%d loss=%.6f",
            condition,
            rep,
            fold_name,
            epoch_index + 1,
            epochs,
            mean_epoch_loss,
        )

    return epoch_losses


def predict_participants_majority_vote(
    model: Chapter6FrameClassifier,
    participant_ids: list[str],
    participant_frames: dict[str, torch.Tensor],
    label_map: dict[str, int],
    device: torch.device,
    threshold: float,
    frame_eval_batch_size: int,
) -> list[dict[str, object]]:
    model.eval()
    predictions: list[dict[str, object]] = []

    with torch.no_grad():
        for participant_id in participant_ids:
            frame_tensor = participant_frames[participant_id]
            frame_probabilities: list[torch.Tensor] = []

            frame_count = int(frame_tensor.shape[0])
            for start_index in range(0, frame_count, frame_eval_batch_size):
                end_index = min(frame_count, start_index + frame_eval_batch_size)
                frame_batch = frame_tensor[start_index:end_index].to(device)
                _, probability_batch = model(frame_batch)
                frame_probabilities.append(probability_batch.squeeze(1).cpu())

            all_frame_probabilities = torch.cat(frame_probabilities, dim=0)
            frame_votes = (all_frame_probabilities > threshold).to(dtype=torch.float32)
            majority_probability = float(frame_votes.mean().item())
            mean_probability = float(all_frame_probabilities.mean().item())
            predicted_label = int(majority_probability > threshold)
            true_label = int(label_map[participant_id])

            predictions.append(
                {
                    "participant_id": participant_id,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "majority_probability": majority_probability,
                    "mean_probability": mean_probability,
                    "frame_count": frame_count,
                }
            )

    return predictions


def compute_binary_metrics(predictions: list[dict[str, object]]) -> dict[str, float]:
    true_labels = [int(prediction["true_label"]) for prediction in predictions]
    predicted_labels = [int(prediction["predicted_label"]) for prediction in predictions]

    accuracy = float(accuracy_score(true_labels, predicted_labels))
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score),
    }


def run_single_fold(
    project_root: Path,
    condition: str,
    test_fold_name: str,
    rep: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    frame_eval_batch_size: int,
) -> FoldRunResult:
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition}")

    condition_spec = CONDITIONS[condition]
    seed = rep * 42
    set_seed(seed)

    config = default_module1_config(project_root)
    split_info = build_stream_split(
        config=config,
        stream=condition_spec.stream,
        test_fold_name=test_fold_name,
    )

    label_map = build_label_map(split_info.all_participant_ids)
    normalizer = StreamFeatureNormalizer().fit(
        participant_ids=split_info.train_participant_ids,
        file_index=split_info.file_index,
    )

    participant_frames = load_normalized_participant_frames(
        participant_ids=split_info.all_participant_ids,
        file_index=split_info.file_index,
        normalizer=normalizer,
    )

    train_dataset = FrameTrainingDataset(
        participant_ids=split_info.train_participant_ids,
        participant_frames=participant_frames,
        label_map=label_map,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    device = resolve_device()
    model = Chapter6FrameClassifier(use_itmla=condition_spec.use_itmla).to(device)

    start_time = time.time()
    epoch_losses = train_frame_level_model(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=epochs,
        condition=condition,
        fold_name=test_fold_name,
        rep=rep,
    )

    predictions = predict_participants_majority_vote(
        model=model,
        participant_ids=split_info.test_participant_ids,
        participant_frames=participant_frames,
        label_map=label_map,
        device=device,
        threshold=THRESHOLD,
        frame_eval_batch_size=frame_eval_batch_size,
    )
    metrics = compute_binary_metrics(predictions)

    elapsed_seconds = float(time.time() - start_time)
    filtered_fold_sizes = {
        fold_name: len(participant_ids)
        for fold_name, participant_ids in split_info.filtered_folds.items()
    }

    test_frame_count = 0
    for prediction in predictions:
        test_frame_count = test_frame_count + int(prediction["frame_count"])

    return FoldRunResult(
        condition=condition,
        stream=condition_spec.stream,
        rep=rep,
        seed=seed,
        fold_name=test_fold_name,
        device=str(device),
        train_participant_count=len(split_info.train_participant_ids),
        test_participant_count=len(split_info.test_participant_ids),
        train_frame_count=len(train_dataset),
        test_frame_count=test_frame_count,
        training_losses=epoch_losses,
        metrics=metrics,
        predictions=predictions,
        elapsed_seconds=elapsed_seconds,
        filtered_fold_sizes=filtered_fold_sizes,
        missing_fold_participants=split_info.missing_fold_participants,
    )


def run_condition_across_folds(
    project_root: Path,
    condition: str,
    rep: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    frame_eval_batch_size: int,
) -> ConditionRunResult:
    fold_names = ["fold1", "fold2", "fold3", "fold4", "fold5"]
    fold_results: list[FoldRunResult] = []
    pooled_predictions: list[dict[str, object]] = []

    for fold_name in fold_names:
        fold_result = run_single_fold(
            project_root=project_root,
            condition=condition,
            test_fold_name=fold_name,
            rep=rep,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            frame_eval_batch_size=frame_eval_batch_size,
        )
        fold_results.append(fold_result)

        for prediction in fold_result.predictions:
            pooled_prediction = dict(prediction)
            pooled_prediction["fold_name"] = fold_name
            pooled_predictions.append(pooled_prediction)

    seen_participants: set[str] = set()
    duplicate_ids: set[str] = set()
    for prediction in pooled_predictions:
        participant_id = str(prediction["participant_id"])
        if participant_id in seen_participants:
            duplicate_ids.add(participant_id)
        seen_participants.add(participant_id)

    pooled_metrics = compute_binary_metrics(pooled_predictions)

    return ConditionRunResult(
        condition=condition,
        stream=CONDITIONS[condition].stream,
        rep=rep,
        fold_results=fold_results,
        pooled_metrics=pooled_metrics,
        pooled_prediction_count=len(pooled_predictions),
        duplicate_prediction_ids=duplicate_ids,
    )


def run_architecture_checks() -> ArchitectureCheckResult:
    random_input = torch.randn((4, FRAME_SIZE, FEATURE_DIM), dtype=torch.float32)

    itmla_layer = ITMLALayer()
    itmla_output = itmla_layer(random_input)
    expected_itmla_output = random_input * (
        1.0
        + torch_functional.cosine_similarity(
            random_input,
            random_input.mean(dim=1, keepdim=True).expand_as(random_input),
            dim=-1,
        ).unsqueeze(-1)
    )

    itmla_max_diff = float(torch.max(torch.abs(itmla_output - expected_itmla_output)).item())
    itmla_shape_ok = tuple(itmla_output.shape) == tuple(random_input.shape)
    itmla_formula_ok = itmla_max_diff < 1e-6

    lstm_layer = LSTM1Layer()
    context_vectors, probabilities = lstm_layer(random_input)
    context_shape_ok = tuple(context_vectors.shape) == (4, LSTM_HIDDEN)
    probability_shape_ok = tuple(probabilities.shape) == (4, 1)

    return ArchitectureCheckResult(
        itmla_shape_ok=itmla_shape_ok,
        itmla_formula_ok=itmla_formula_ok,
        itmla_max_diff=itmla_max_diff,
        lstm_context_shape_ok=context_shape_ok,
        lstm_probability_shape_ok=probability_shape_ok,
    )


def save_predictions_csv(output_path: Path, condition: str, rep: int, predictions: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode="w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "condition",
                "rep",
                "fold",
                "participant_id",
                "true_label",
                "predicted_label",
                "majority_probability",
                "mean_probability",
                "frame_count",
            ],
        )
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(
                {
                    "condition": condition,
                    "rep": rep,
                    "fold": prediction.get("fold_name", "single_fold"),
                    "participant_id": prediction["participant_id"],
                    "true_label": prediction["true_label"],
                    "predicted_label": prediction["predicted_label"],
                    "majority_probability": f"{float(prediction['majority_probability']):.6f}",
                    "mean_probability": f"{float(prediction['mean_probability']):.6f}",
                    "frame_count": int(prediction["frame_count"]),
                }
            )


def format_fold_report(
    architecture_checks: ArchitectureCheckResult,
    fold_result: FoldRunResult,
) -> str:
    first_loss = fold_result.training_losses[0] if fold_result.training_losses else float("nan")
    last_loss = fold_result.training_losses[-1] if fold_result.training_losses else float("nan")
    loss_decreased = last_loss < first_loss

    lines: list[str] = [
        "=== Module 3: Chapter 6 Frame-Level Pipeline Report ===",
        "",
        "[Architecture Checks]",
        f"- ITMLA output shape check: {architecture_checks.itmla_shape_ok}",
        f"- ITMLA formula check: {architecture_checks.itmla_formula_ok} (max_diff={architecture_checks.itmla_max_diff:.6e})",
        f"- LSTM1 context shape check: {architecture_checks.lstm_context_shape_ok}",
        f"- LSTM1 probability shape check: {architecture_checks.lstm_probability_shape_ok}",
        "",
        "[Run Configuration]",
        f"- condition: {fold_result.condition}",
        f"- stream: {fold_result.stream}",
        f"- rep: {fold_result.rep}",
        f"- seed: {fold_result.seed}",
        f"- fold: {fold_result.fold_name}",
        f"- device: {fold_result.device}",
        f"- train participants: {fold_result.train_participant_count}",
        f"- test participants: {fold_result.test_participant_count}",
        f"- train frames: {fold_result.train_frame_count}",
        f"- test frames: {fold_result.test_frame_count}",
        f"- elapsed seconds: {fold_result.elapsed_seconds:.2f}",
        f"- filtered fold sizes: {fold_result.filtered_fold_sizes}",
        "",
        "[Metrics]",
        f"- accuracy: {fold_result.metrics['accuracy']:.4f}",
        f"- precision: {fold_result.metrics['precision']:.4f}",
        f"- recall: {fold_result.metrics['recall']:.4f}",
        f"- f1: {fold_result.metrics['f1']:.4f}",
        "",
        "[Training Loss]",
        f"- first epoch loss: {first_loss:.6f}",
        f"- last epoch loss: {last_loss:.6f}",
        f"- loss decreased: {loss_decreased}",
        "",
        "[Notes]",
        "- p_c and p_o are computed by majority vote over frame-level decisions (threshold 0.5).",
        "- Loss is BCE over frame predictions, each frame labeled with participant class.",
        (
            "- Missing fold participants for this stream: "
            + (", ".join(sorted(fold_result.missing_fold_participants)) if fold_result.missing_fold_participants else "none")
        ),
        "",
        "[Module Scope]",
        "- Module 3 implemented Chapter 6 single-stream frame-level training only.",
    ]

    return "\n".join(lines)


def format_all_folds_report(
    architecture_checks: ArchitectureCheckResult,
    condition_result: ConditionRunResult,
) -> str:
    lines: list[str] = [
        "=== Module 3: Chapter 6 All-Folds Report ===",
        "",
        "[Architecture Checks]",
        f"- ITMLA output shape check: {architecture_checks.itmla_shape_ok}",
        f"- ITMLA formula check: {architecture_checks.itmla_formula_ok} (max_diff={architecture_checks.itmla_max_diff:.6e})",
        f"- LSTM1 context shape check: {architecture_checks.lstm_context_shape_ok}",
        f"- LSTM1 probability shape check: {architecture_checks.lstm_probability_shape_ok}",
        "",
        "[Run Configuration]",
        f"- condition: {condition_result.condition}",
        f"- stream: {condition_result.stream}",
        f"- rep: {condition_result.rep}",
        f"- pooled prediction count: {condition_result.pooled_prediction_count}",
        f"- duplicate pooled participant ids: {len(condition_result.duplicate_prediction_ids)}",
        "",
        "[Fold Metrics]",
    ]

    for fold_result in condition_result.fold_results:
        lines.append(
            (
                f"- {fold_result.fold_name}: accuracy={fold_result.metrics['accuracy']:.4f}, "
                f"precision={fold_result.metrics['precision']:.4f}, "
                f"recall={fold_result.metrics['recall']:.4f}, "
                f"f1={fold_result.metrics['f1']:.4f}, "
                f"train_frames={fold_result.train_frame_count}, test_frames={fold_result.test_frame_count}"
            )
        )

    lines.extend(
        [
            "",
            "[Pooled Metrics]",
            f"- accuracy: {condition_result.pooled_metrics['accuracy']:.4f}",
            f"- precision: {condition_result.pooled_metrics['precision']:.4f}",
            f"- recall: {condition_result.pooled_metrics['recall']:.4f}",
            f"- f1: {condition_result.pooled_metrics['f1']:.4f}",
            "",
            "[Module Scope]",
            "- Module 3 implemented Chapter 6 single-stream frame-level training only.",
        ]
    )

    return "\n".join(lines)


def save_report(report_text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text + "\n", encoding="utf-8")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Module 3 Chapter 6 frame-level experiments.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root path.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="ba1_rt",
        choices=sorted(CONDITIONS.keys()),
        help="Condition to run.",
    )
    parser.add_argument(
        "--test-fold",
        type=str,
        default="fold1",
        help="Fold used as test split for single-fold run.",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Run all 5 folds and compute pooled metrics.",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=1,
        help="Repetition index; seed is rep * 42.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Frame-level batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--frame-eval-batch-size",
        type=int,
        default=FRAME_EVAL_BATCH_SIZE,
        help="Frame batch size used during participant-level evaluation.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "module3",
        help="Directory where Module 3 reports and predictions are saved.",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a short smoke test with 3 epochs on a single fold.",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    cli_args = parse_arguments()

    project_root = cli_args.project_root.resolve()
    results_dir = (
        cli_args.results_dir
        if cli_args.results_dir.is_absolute()
        else project_root / cli_args.results_dir
    )

    run_epochs = 3 if cli_args.quick_test else cli_args.epochs
    architecture_checks = run_architecture_checks()

    if cli_args.all_folds:
        condition_result = run_condition_across_folds(
            project_root=project_root,
            condition=cli_args.condition,
            rep=cli_args.rep,
            epochs=run_epochs,
            batch_size=cli_args.batch_size,
            num_workers=cli_args.num_workers,
            frame_eval_batch_size=cli_args.frame_eval_batch_size,
        )

        pooled_predictions: list[dict[str, object]] = []
        for fold_result in condition_result.fold_results:
            for prediction in fold_result.predictions:
                pooled_prediction = dict(prediction)
                pooled_prediction["fold_name"] = fold_result.fold_name
                pooled_predictions.append(pooled_prediction)

        prediction_path = results_dir / f"{cli_args.condition}_rep{cli_args.rep}_all_folds_predictions.csv"
        report_path = results_dir / f"{cli_args.condition}_rep{cli_args.rep}_all_folds_report.txt"

        save_predictions_csv(
            output_path=prediction_path,
            condition=cli_args.condition,
            rep=cli_args.rep,
            predictions=pooled_predictions,
        )
        report_text = format_all_folds_report(
            architecture_checks=architecture_checks,
            condition_result=condition_result,
        )
        print(report_text)
        save_report(report_text, report_path)

        LOGGER.info("Module 3 pooled predictions saved to %s", prediction_path)
        LOGGER.info("Module 3 report saved to %s", report_path)

        has_duplicates = len(condition_result.duplicate_prediction_ids) > 0
        return 1 if has_duplicates else 0

    fold_result = run_single_fold(
        project_root=project_root,
        condition=cli_args.condition,
        test_fold_name=cli_args.test_fold,
        rep=cli_args.rep,
        epochs=run_epochs,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        frame_eval_batch_size=cli_args.frame_eval_batch_size,
    )

    prediction_path = results_dir / (
        f"{cli_args.condition}_rep{cli_args.rep}_{cli_args.test_fold}_predictions.csv"
    )
    report_path = results_dir / f"{cli_args.condition}_rep{cli_args.rep}_{cli_args.test_fold}_report.txt"

    fold_predictions: list[dict[str, object]] = []
    for prediction in fold_result.predictions:
        fold_prediction = dict(prediction)
        fold_prediction["fold_name"] = fold_result.fold_name
        fold_predictions.append(fold_prediction)

    save_predictions_csv(
        output_path=prediction_path,
        condition=cli_args.condition,
        rep=cli_args.rep,
        predictions=fold_predictions,
    )

    report_text = format_fold_report(
        architecture_checks=architecture_checks,
        fold_result=fold_result,
    )
    print(report_text)
    save_report(report_text, report_path)

    LOGGER.info("Module 3 fold predictions saved to %s", prediction_path)
    LOGGER.info("Module 3 report saved to %s", report_path)
    LOGGER.info("Module 3 complete. Waiting for verification before Module 4.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
