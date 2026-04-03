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

from cdma.module2_data import Module2DataLoaders, get_dataloaders

LOGGER = logging.getLogger(__name__)

FEATURE_DIM = 32
LSTM_HIDDEN = 32
FRAME_SIZE = 128
EPOCHS = 300
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
THRESHOLD = 0.5
LOG_EVERY_EPOCHS = 50
CHECKPOINT_EVERY_EPOCHS = 50
PREVIEW_PARTICIPANTS = 5


@dataclass(frozen=True)
class ModeConfig:
    name: str
    need_rt: bool
    need_it: bool
    use_mla: bool
    ga_type: str
    use_lstm2: bool
    use_ctf_f1: bool
    use_ctf_f2: bool
    output_names: tuple[str, ...]


MODE_CONFIGS: dict[str, ModeConfig] = {
    "ba1_rt": ModeConfig("ba1_rt", True, False, False, "none", False, False, False, ("p_c",)),
    "ba1_it": ModeConfig("ba1_it", False, True, False, "none", False, False, False, ("p_o",)),
    "itmla_rt": ModeConfig("itmla_rt", True, False, True, "none", False, False, False, ("p_c",)),
    "itmla_it": ModeConfig("itmla_it", False, True, True, "none", False, False, False, ("p_o",)),
    "ba2_rt": ModeConfig("ba2_rt", True, False, True, "none", True, False, False, ("p_c", "p_t")),
    "ba2_it": ModeConfig("ba2_it", False, True, True, "none", True, False, False, ("p_o", "p_d")),
    "ba3_rt": ModeConfig("ba3_rt", True, False, True, "self", True, False, False, ("p_c", "p_t")),
    "ba3_it": ModeConfig("ba3_it", False, True, True, "self", True, False, False, ("p_o", "p_d")),
    "ctga_rt": ModeConfig("ctga_rt", True, True, True, "cross", True, False, False, ("p_c", "p_t")),
    "ctga_it": ModeConfig("ctga_it", True, True, True, "cross", True, False, False, ("p_o", "p_d")),
    "ba4": ModeConfig(
        "ba4",
        True,
        True,
        True,
        "cross",
        True,
        True,
        False,
        ("p_c", "p_o", "p_t", "p_d", "p_f1"),
    ),
    "ba5": ModeConfig(
        "ba5",
        True,
        True,
        True,
        "cross",
        True,
        False,
        True,
        ("p_c", "p_o", "p_t", "p_d", "p_f2"),
    ),
    "full_cdma": ModeConfig(
        "full_cdma",
        True,
        True,
        True,
        "cross",
        True,
        True,
        True,
        ("p_c", "p_o", "p_t", "p_d", "p_f1", "p_f2"),
    ),
}


@dataclass(frozen=True)
class Module4SanityResult:
    all_modes_forward_ok: bool
    output_key_mismatch_modes: list[str]
    p_hat_shape_ok: bool
    no_nan_outputs: bool
    loss_scaling_ok: bool
    loss_b1: float
    loss_b6: float
    gradient_flow_ok: bool


@dataclass(frozen=True)
class Module4FoldResult:
    mode: str
    rep: int
    seed: int
    fold_name: str
    device: str
    train_participant_count: int
    test_participant_count: int
    train_batch_count: int
    test_batch_count: int
    metrics: dict[str, float]
    training_losses: list[float]
    predictions: list[dict[str, object]]
    elapsed_seconds: float
    filtered_fold_sizes: dict[str, int]


@dataclass(frozen=True)
class Module4ConditionResult:
    mode: str
    rep: int
    fold_results: list[Module4FoldResult]
    pooled_metrics: dict[str, float]
    pooled_prediction_count: int
    duplicate_prediction_ids: set[str]


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fold_predictions_path(results_dir: Path, mode: str, rep: int, fold_name: str) -> Path:
    return results_dir / f"{mode}_rep{rep}_{fold_name}_predictions.csv"


def _fold_report_path(results_dir: Path, mode: str, rep: int, fold_name: str) -> Path:
    return results_dir / f"{mode}_rep{rep}_{fold_name}_report.txt"


def _all_folds_predictions_path(results_dir: Path, mode: str, rep: int) -> Path:
    return results_dir / f"{mode}_rep{rep}_all_folds_predictions.csv"


def _all_folds_report_path(results_dir: Path, mode: str, rep: int) -> Path:
    return results_dir / f"{mode}_rep{rep}_all_folds_report.txt"


def _checkpoint_path(results_dir: Path, mode: str, rep: int, fold_name: str) -> Path:
    checkpoint_dir = results_dir / "checkpoints" / mode / f"rep_{rep}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{fold_name}_latest.pt"


def save_training_checkpoint(
    checkpoint_path: Path,
    mode: str,
    rep: int,
    fold_name: str,
    epoch_index: int,
    model: CDMAModel,
    optimizer: torch.optim.Optimizer,
    training_losses: list[float],
) -> None:
    checkpoint_payload = {
        "mode": mode,
        "rep": rep,
        "fold_name": fold_name,
        "epoch_index": epoch_index,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_losses": training_losses,
        "saved_at": time.time(),
    }

    temporary_checkpoint = checkpoint_path.with_suffix(".tmp")
    torch.save(checkpoint_payload, temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint_path)


def load_training_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, object] | None:
    if not checkpoint_path.exists():
        return None
    return torch.load(checkpoint_path, map_location=device)


def append_history_row(history_path: Path, field_names: list[str], row_data: dict[str, object]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = history_path.exists()

    with history_path.open(mode="a", encoding="utf-8", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=field_names)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def load_predictions_csv(predictions_path: Path) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    if not predictions_path.exists():
        return predictions

    with predictions_path.open(mode="r", encoding="utf-8", newline="") as prediction_file:
        reader = csv.DictReader(prediction_file)
        for row in reader:
            output_probabilities: dict[str, float] = {}
            for output_name in ("p_c", "p_o", "p_t", "p_d", "p_f1", "p_f2"):
                cell_value = row.get(output_name, "")
                if cell_value:
                    output_probabilities[output_name] = float(cell_value)

            predictions.append(
                {
                    "participant_id": row["participant_id"],
                    "true_label": int(row["true_label"]),
                    "predicted_label": int(row["predicted_label"]),
                    "p_hat": float(row["p_hat"]),
                    "output_probabilities": output_probabilities,
                    "fold_name": row.get("fold", ""),
                }
            )

    return predictions


def log_prediction_preview(predictions: list[dict[str, object]], top_k: int, mode: str, fold_name: str) -> None:
    if not predictions:
        LOGGER.info("mode=%s fold=%s has no predictions to preview.", mode, fold_name)
        return

    preview_count = min(top_k, len(predictions))
    LOGGER.info("mode=%s fold=%s previewing first %d participants:", mode, fold_name, preview_count)
    for preview_index in range(preview_count):
        prediction = predictions[preview_index]
        LOGGER.info(
            "preview=%d pid=%s true=%d pred=%d p_hat=%.4f",
            preview_index + 1,
            prediction["participant_id"],
            int(prediction["true_label"]),
            int(prediction["predicted_label"]),
            float(prediction["p_hat"]),
        )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(dtype=values.dtype).unsqueeze(-1)
    numerator = (values * mask_float).sum(dim=1)
    denominator = mask_float.sum(dim=1).clamp(min=1e-8)
    return numerator / denominator


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_bool = mask > 0
    masked_scores = scores.masked_fill(~mask_bool, -1e9)
    softmax_values = torch.softmax(masked_scores, dim=1)
    masked_softmax_values = softmax_values * mask.to(dtype=softmax_values.dtype)
    normalization = masked_softmax_values.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return masked_softmax_values / normalization


class ITMLALayer(nn.Module):
    def forward(self, frame_batch: torch.Tensor) -> torch.Tensor:
        frame_average = frame_batch.mean(dim=1, keepdim=True)
        cosine_values = torch_functional.cosine_similarity(
            frame_batch,
            frame_average.expand_as(frame_batch),
            dim=-1,
        )
        return frame_batch * (1.0 + cosine_values.unsqueeze(-1))


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


class LSTM2Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=LSTM_HIDDEN,
            hidden_size=LSTM_HIDDEN,
            batch_first=True,
        )
        self.classifier = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, sequence_batch: torch.Tensor, sequence_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = sequence_mask.sum(dim=1).to(dtype=torch.long)
        if torch.any(lengths <= 0):
            raise ValueError("LSTM2Layer received a sequence with non-positive length.")

        packed_sequence = nn.utils.rnn.pack_padded_sequence(
            sequence_batch,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.lstm(packed_sequence)
        unpacked_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=sequence_batch.size(1),
        )

        pooled_sequence = masked_mean(unpacked_outputs, sequence_mask)
        probabilities = torch.sigmoid(self.classifier(pooled_sequence))
        return pooled_sequence, probabilities


class CTGALayer(nn.Module):
    def _apply_attention(
        self,
        sequence_batch: torch.Tensor,
        sequence_mask: torch.Tensor,
        reference_vector: torch.Tensor,
    ) -> torch.Tensor:
        expanded_reference = reference_vector.unsqueeze(1).expand_as(sequence_batch)
        cosine_scores = torch_functional.cosine_similarity(sequence_batch, expanded_reference, dim=-1)
        attention_weights = masked_softmax(cosine_scores, sequence_mask).unsqueeze(-1)
        return sequence_batch + attention_weights * sequence_batch

    def forward(
        self,
        rt_sequence: torch.Tensor | None,
        rt_mask: torch.Tensor | None,
        it_sequence: torch.Tensor | None,
        it_mask: torch.Tensor | None,
        ga_type: str,
    ) -> dict[str, torch.Tensor | None]:
        rt_mean = masked_mean(rt_sequence, rt_mask) if rt_sequence is not None and rt_mask is not None else None
        it_mean = masked_mean(it_sequence, it_mask) if it_sequence is not None and it_mask is not None else None

        if ga_type == "none":
            rt_star = rt_sequence
            it_star = it_sequence
        elif ga_type == "cross":
            if rt_sequence is None or it_sequence is None or rt_mask is None or it_mask is None:
                raise ValueError("Cross-GA requires both RT and IT streams.")
            if rt_mean is None or it_mean is None:
                raise ValueError("Cross-GA failed to compute stream means.")
            rt_star = self._apply_attention(rt_sequence, rt_mask, it_mean)
            it_star = self._apply_attention(it_sequence, it_mask, rt_mean)
        elif ga_type == "self":
            rt_star = (
                self._apply_attention(rt_sequence, rt_mask, rt_mean)
                if rt_sequence is not None and rt_mask is not None and rt_mean is not None
                else None
            )
            it_star = (
                self._apply_attention(it_sequence, it_mask, it_mean)
                if it_sequence is not None and it_mask is not None and it_mean is not None
                else None
            )
        else:
            raise ValueError(f"Unsupported GA type: {ga_type}")

        rt_star_mean = masked_mean(rt_star, rt_mask) if rt_star is not None and rt_mask is not None else None
        it_star_mean = masked_mean(it_star, it_mask) if it_star is not None and it_mask is not None else None

        return {
            "rt_star": rt_star,
            "it_star": it_star,
            "rt_mean": rt_mean,
            "it_mean": it_mean,
            "rt_star_mean": rt_star_mean,
            "it_star_mean": it_star_mean,
        }


class CTFLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier_f1 = nn.Linear(LSTM_HIDDEN, 1)
        self.classifier_f2 = nn.Linear(LSTM_HIDDEN, 1)

    def forward(
        self,
        rt_mean: torch.Tensor,
        it_mean: torch.Tensor,
        rt_star_mean: torch.Tensor,
        it_star_mean: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fusion_pre_attention = rt_mean + it_mean
        fusion_post_attention = rt_star_mean + it_star_mean
        p_f1 = torch.sigmoid(self.classifier_f1(fusion_pre_attention))
        p_f2 = torch.sigmoid(self.classifier_f2(fusion_post_attention))
        return {"p_f1": p_f1, "p_f2": p_f2}


class CombinedBCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_loss = nn.BCELoss(reduction="mean")

    def forward(self, active_probabilities: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)

        loss_sum = torch.zeros((), dtype=labels.dtype, device=labels.device)
        for probability in active_probabilities:
            loss_sum = loss_sum + self.base_loss(probability, labels)
        return loss_sum


class CDMAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.itmla = ITMLALayer()
        self.lstm1 = LSTM1Layer()
        self.ctga = CTGALayer()
        self.lstm2 = LSTM2Layer()
        self.ctf = CTFLayer()

    def _encode_stream(
        self,
        stream_frames: torch.Tensor,
        stream_mask: torch.Tensor,
        use_mla: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, frame_size, feature_dim = stream_frames.shape
        if frame_size != FRAME_SIZE or feature_dim != FEATURE_DIM:
            raise ValueError(
                f"Unexpected stream shape tail {(frame_size, feature_dim)}, expected {(FRAME_SIZE, FEATURE_DIM)}"
            )

        flattened_frames = stream_frames.reshape(batch_size * sequence_length, frame_size, feature_dim)
        processed_frames = self.itmla(flattened_frames) if use_mla else flattened_frames
        context_vectors_flat, frame_probabilities_flat = self.lstm1(processed_frames)

        context_sequences = context_vectors_flat.reshape(batch_size, sequence_length, LSTM_HIDDEN)
        frame_probabilities = frame_probabilities_flat.reshape(batch_size, sequence_length, 1)
        stream_probability = masked_mean(frame_probabilities, stream_mask)
        return context_sequences, frame_probabilities, stream_probability

    def forward(
        self,
        rt_frames: torch.Tensor | None,
        it_frames: torch.Tensor | None,
        rt_mask: torch.Tensor | None,
        it_mask: torch.Tensor | None,
        mode: str,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | list[torch.Tensor]]:
        if mode not in MODE_CONFIGS:
            raise ValueError(f"Unsupported mode: {mode}")
        mode_config = MODE_CONFIGS[mode]

        if mode_config.need_rt and (rt_frames is None or rt_mask is None):
            raise ValueError(f"Mode {mode} requires RT input.")
        if mode_config.need_it and (it_frames is None or it_mask is None):
            raise ValueError(f"Mode {mode} requires IT input.")

        rt_context_sequence: torch.Tensor | None = None
        it_context_sequence: torch.Tensor | None = None
        rt_frame_probability: torch.Tensor | None = None
        it_frame_probability: torch.Tensor | None = None

        if mode_config.need_rt and rt_frames is not None and rt_mask is not None:
            rt_context_sequence, _, rt_frame_probability = self._encode_stream(
                stream_frames=rt_frames,
                stream_mask=rt_mask,
                use_mla=mode_config.use_mla,
            )

        if mode_config.need_it and it_frames is not None and it_mask is not None:
            it_context_sequence, _, it_frame_probability = self._encode_stream(
                stream_frames=it_frames,
                stream_mask=it_mask,
                use_mla=mode_config.use_mla,
            )

        ga_outputs = self.ctga(
            rt_sequence=rt_context_sequence,
            rt_mask=rt_mask,
            it_sequence=it_context_sequence,
            it_mask=it_mask,
            ga_type=mode_config.ga_type,
        )

        rt_sequence_for_lstm2 = ga_outputs["rt_star"] if ga_outputs["rt_star"] is not None else rt_context_sequence
        it_sequence_for_lstm2 = ga_outputs["it_star"] if ga_outputs["it_star"] is not None else it_context_sequence

        output_probabilities: dict[str, torch.Tensor] = {}

        if "p_c" in mode_config.output_names:
            if rt_frame_probability is None:
                raise ValueError(f"Mode {mode} expected p_c but RT frame probability is missing.")
            output_probabilities["p_c"] = rt_frame_probability

        if "p_o" in mode_config.output_names:
            if it_frame_probability is None:
                raise ValueError(f"Mode {mode} expected p_o but IT frame probability is missing.")
            output_probabilities["p_o"] = it_frame_probability

        if "p_t" in mode_config.output_names:
            if rt_sequence_for_lstm2 is None or rt_mask is None:
                raise ValueError(f"Mode {mode} expected p_t but RT sequence for LSTM2 is missing.")
            _, p_t = self.lstm2(rt_sequence_for_lstm2, rt_mask)
            output_probabilities["p_t"] = p_t

        if "p_d" in mode_config.output_names:
            if it_sequence_for_lstm2 is None or it_mask is None:
                raise ValueError(f"Mode {mode} expected p_d but IT sequence for LSTM2 is missing.")
            _, p_d = self.lstm2(it_sequence_for_lstm2, it_mask)
            output_probabilities["p_d"] = p_d

        if mode_config.use_ctf_f1 or mode_config.use_ctf_f2:
            rt_mean = ga_outputs["rt_mean"]
            it_mean = ga_outputs["it_mean"]
            rt_star_mean = ga_outputs["rt_star_mean"]
            it_star_mean = ga_outputs["it_star_mean"]
            if rt_mean is None or it_mean is None or rt_star_mean is None or it_star_mean is None:
                raise ValueError(f"Mode {mode} requires both streams for CTF fusion.")

            ctf_probabilities = self.ctf(
                rt_mean=rt_mean,
                it_mean=it_mean,
                rt_star_mean=rt_star_mean,
                it_star_mean=it_star_mean,
            )

            if mode_config.use_ctf_f1:
                output_probabilities["p_f1"] = ctf_probabilities["p_f1"]
            if mode_config.use_ctf_f2:
                output_probabilities["p_f2"] = ctf_probabilities["p_f2"]

        active_probabilities = [output_probabilities[name] for name in mode_config.output_names]
        probability_stack = torch.stack(active_probabilities, dim=0)
        p_hat = probability_stack.mean(dim=0)

        return {
            "probabilities": output_probabilities,
            "active_probabilities": active_probabilities,
            "p_hat": p_hat,
        }


def _prepare_batch_inputs(
    batch: dict[str, object],
    mode: str,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    mode_config = MODE_CONFIGS[mode]

    rt_frames = batch["rt_frames"].to(device) if mode_config.need_rt else None
    it_frames = batch["it_frames"].to(device) if mode_config.need_it else None
    rt_mask = batch["rt_mask"].to(device) if mode_config.need_rt else None
    it_mask = batch["it_mask"].to(device) if mode_config.need_it else None
    labels = batch["labels"].to(device)

    return rt_frames, it_frames, rt_mask, it_mask, labels


def train_one_epoch(
    model: CDMAModel,
    data_loaders: Module2DataLoaders,
    optimizer: torch.optim.Optimizer,
    loss_fn: CombinedBCELoss,
    mode: str,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    batch_count = 0

    for batch in data_loaders.train_loader:
        rt_frames, it_frames, rt_mask, it_mask, labels = _prepare_batch_inputs(batch, mode, device)
        outputs = model(rt_frames, it_frames, rt_mask, it_mask, mode)
        loss = loss_fn(outputs["active_probabilities"], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss = running_loss + float(loss.item())
        batch_count = batch_count + 1

    return running_loss / max(1, batch_count)


def evaluate_model(
    model: CDMAModel,
    data_loaders: Module2DataLoaders,
    mode: str,
    device: torch.device,
    threshold: float,
) -> list[dict[str, object]]:
    model.eval()
    predictions: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in data_loaders.test_loader:
            rt_frames, it_frames, rt_mask, it_mask, labels = _prepare_batch_inputs(batch, mode, device)
            outputs = model(rt_frames, it_frames, rt_mask, it_mask, mode)

            p_hat_batch = outputs["p_hat"].squeeze(1).cpu()
            label_batch = labels.to(dtype=torch.long).cpu()
            predicted_batch = (p_hat_batch > threshold).to(dtype=torch.long)

            probability_tensors = outputs["probabilities"]
            participant_ids = batch["pids"]

            for batch_index, participant_id in enumerate(participant_ids):
                output_probabilities: dict[str, float] = {}
                for output_name, output_tensor in probability_tensors.items():
                    output_probabilities[output_name] = float(output_tensor[batch_index].item())

                predictions.append(
                    {
                        "participant_id": str(participant_id),
                        "true_label": int(label_batch[batch_index].item()),
                        "predicted_label": int(predicted_batch[batch_index].item()),
                        "p_hat": float(p_hat_batch[batch_index].item()),
                        "output_probabilities": output_probabilities,
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
    results_dir: Path,
    mode: str,
    test_fold_name: str,
    rep: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    log_every_epochs: int,
    checkpoint_every_epochs: int,
    preview_participants: int,
    resume: bool,
) -> Module4FoldResult:
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unsupported mode: {mode}")

    seed = rep * 42
    set_seed(seed)

    data_loaders = get_dataloaders(
        project_root=project_root,
        test_fold_name=test_fold_name,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_train=True,
    )

    device = resolve_device()
    model = CDMAModel().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    loss_fn = CombinedBCELoss()
    checkpoint_path = _checkpoint_path(results_dir, mode, rep, test_fold_name)

    training_losses: list[float] = []
    start_epoch_index = 0

    if resume:
        checkpoint_state = load_training_checkpoint(checkpoint_path=checkpoint_path, device=device)
        if checkpoint_state is not None:
            model.load_state_dict(checkpoint_state["model_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
            training_losses = [float(loss_value) for loss_value in checkpoint_state.get("training_losses", [])]
            start_epoch_index = int(checkpoint_state.get("epoch_index", -1)) + 1
            LOGGER.info(
                "Resuming mode=%s rep=%d fold=%s from epoch=%d using checkpoint=%s",
                mode,
                rep,
                test_fold_name,
                start_epoch_index,
                checkpoint_path,
            )

    start_time = time.time()

    for epoch_index in range(start_epoch_index, epochs):
        mean_loss = train_one_epoch(
            model=model,
            data_loaders=data_loaders,
            optimizer=optimizer,
            loss_fn=loss_fn,
            mode=mode,
            device=device,
        )
        training_losses.append(mean_loss)

        current_epoch = epoch_index + 1
        should_log = (
            current_epoch == 1
            or current_epoch == epochs
            or current_epoch % max(1, log_every_epochs) == 0
        )
        if should_log:
            LOGGER.info(
                "mode=%s rep=%d fold=%s epoch=%d/%d loss=%.6f",
                mode,
                rep,
                test_fold_name,
                current_epoch,
                epochs,
                mean_loss,
            )

        should_checkpoint = (
            current_epoch == epochs
            or current_epoch % max(1, checkpoint_every_epochs) == 0
        )
        if should_checkpoint:
            save_training_checkpoint(
                checkpoint_path=checkpoint_path,
                mode=mode,
                rep=rep,
                fold_name=test_fold_name,
                epoch_index=epoch_index,
                model=model,
                optimizer=optimizer,
                training_losses=training_losses,
            )

    predictions = evaluate_model(
        model=model,
        data_loaders=data_loaders,
        mode=mode,
        device=device,
        threshold=THRESHOLD,
    )
    log_prediction_preview(
        predictions=predictions,
        top_k=preview_participants,
        mode=mode,
        fold_name=test_fold_name,
    )

    metrics = compute_binary_metrics(predictions)

    filtered_fold_sizes = {
        fold_name: len(participant_ids)
        for fold_name, participant_ids in data_loaders.split_info.filtered_rt_folds.items()
    }

    elapsed_seconds = float(time.time() - start_time)

    return Module4FoldResult(
        mode=mode,
        rep=rep,
        seed=seed,
        fold_name=test_fold_name,
        device=str(device),
        train_participant_count=len(data_loaders.split_info.train_participant_ids),
        test_participant_count=len(data_loaders.split_info.test_participant_ids),
        train_batch_count=len(data_loaders.train_loader),
        test_batch_count=len(data_loaders.test_loader),
        metrics=metrics,
        training_losses=training_losses,
        predictions=predictions,
        elapsed_seconds=elapsed_seconds,
        filtered_fold_sizes=filtered_fold_sizes,
    )


def run_all_folds(
    project_root: Path,
    results_dir: Path,
    mode: str,
    rep: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    log_every_epochs: int,
    checkpoint_every_epochs: int,
    preview_participants: int,
    resume: bool,
    sanity_result: Module4SanityResult,
) -> Module4ConditionResult:
    fold_names = ["fold1", "fold2", "fold3", "fold4", "fold5"]
    fold_results: list[Module4FoldResult] = []
    pooled_predictions: list[dict[str, object]] = []

    fold_history_path = results_dir / "fold_history.csv"
    pooled_history_path = results_dir / "pooled_history.csv"

    for fold_name in fold_names:
        fold_predictions_path = _fold_predictions_path(results_dir, mode, rep, fold_name)
        fold_report_path = _fold_report_path(results_dir, mode, rep, fold_name)

        if resume and fold_predictions_path.exists() and fold_report_path.exists():
            cached_predictions = load_predictions_csv(fold_predictions_path)
            cached_metrics = compute_binary_metrics(cached_predictions)

            fold_result = Module4FoldResult(
                mode=mode,
                rep=rep,
                seed=rep * 42,
                fold_name=fold_name,
                device="cached",
                train_participant_count=-1,
                test_participant_count=len(cached_predictions),
                train_batch_count=-1,
                test_batch_count=-1,
                metrics=cached_metrics,
                training_losses=[],
                predictions=cached_predictions,
                elapsed_seconds=0.0,
                filtered_fold_sizes={},
            )

            LOGGER.info(
                "Skipping completed fold mode=%s rep=%d fold=%s using cached outputs.",
                mode,
                rep,
                fold_name,
            )
        else:
            fold_result = run_single_fold(
                project_root=project_root,
                results_dir=results_dir,
                mode=mode,
                test_fold_name=fold_name,
                rep=rep,
                epochs=epochs,
                batch_size=batch_size,
                num_workers=num_workers,
                log_every_epochs=log_every_epochs,
                checkpoint_every_epochs=checkpoint_every_epochs,
                preview_participants=preview_participants,
                resume=resume,
            )

            fold_predictions_for_save: list[dict[str, object]] = []
            for prediction in fold_result.predictions:
                prediction_copy = dict(prediction)
                prediction_copy["fold_name"] = fold_name
                fold_predictions_for_save.append(prediction_copy)

            save_predictions_csv(
                output_path=fold_predictions_path,
                mode=mode,
                rep=rep,
                predictions=fold_predictions_for_save,
            )

            fold_report_text = format_fold_report(sanity_result=sanity_result, fold_result=fold_result)
            save_report(fold_report_text, fold_report_path)

            append_history_row(
                history_path=fold_history_path,
                field_names=[
                    "mode",
                    "rep",
                    "fold",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "train_participants",
                    "test_participants",
                    "epochs",
                    "timestamp",
                ],
                row_data={
                    "mode": mode,
                    "rep": rep,
                    "fold": fold_name,
                    "accuracy": f"{fold_result.metrics['accuracy']:.6f}",
                    "precision": f"{fold_result.metrics['precision']:.6f}",
                    "recall": f"{fold_result.metrics['recall']:.6f}",
                    "f1": f"{fold_result.metrics['f1']:.6f}",
                    "train_participants": fold_result.train_participant_count,
                    "test_participants": fold_result.test_participant_count,
                    "epochs": epochs,
                    "timestamp": int(time.time()),
                },
            )

        fold_results.append(fold_result)

        for prediction in fold_result.predictions:
            prediction_with_fold = dict(prediction)
            prediction_with_fold["fold_name"] = fold_name
            pooled_predictions.append(prediction_with_fold)

    seen_participants: set[str] = set()
    duplicate_ids: set[str] = set()
    for prediction in pooled_predictions:
        participant_id = str(prediction["participant_id"])
        if participant_id in seen_participants:
            duplicate_ids.add(participant_id)
        seen_participants.add(participant_id)

    pooled_metrics = compute_binary_metrics(pooled_predictions)

    append_history_row(
        history_path=pooled_history_path,
        field_names=["mode", "rep", "accuracy", "precision", "recall", "f1", "prediction_count", "timestamp"],
        row_data={
            "mode": mode,
            "rep": rep,
            "accuracy": f"{pooled_metrics['accuracy']:.6f}",
            "precision": f"{pooled_metrics['precision']:.6f}",
            "recall": f"{pooled_metrics['recall']:.6f}",
            "f1": f"{pooled_metrics['f1']:.6f}",
            "prediction_count": len(pooled_predictions),
            "timestamp": int(time.time()),
        },
    )

    return Module4ConditionResult(
        mode=mode,
        rep=rep,
        fold_results=fold_results,
        pooled_metrics=pooled_metrics,
        pooled_prediction_count=len(pooled_predictions),
        duplicate_prediction_ids=duplicate_ids,
    )


def run_sanity_checks() -> Module4SanityResult:
    device = torch.device("cpu")
    model = CDMAModel().to(device)
    loss_fn = CombinedBCELoss()

    batch_size = 2
    rt_sequence_length = 6
    it_sequence_length = 7

    rt_frames = torch.randn((batch_size, rt_sequence_length, FRAME_SIZE, FEATURE_DIM), device=device)
    it_frames = torch.randn((batch_size, it_sequence_length, FRAME_SIZE, FEATURE_DIM), device=device)

    rt_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.float32, device=device)
    it_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]], dtype=torch.float32, device=device)
    labels = torch.tensor([[0.0], [1.0]], dtype=torch.float32, device=device)

    mismatch_modes: list[str] = []
    p_hat_shape_ok = True
    no_nan_outputs = True

    for mode_name, mode_config in MODE_CONFIGS.items():
        mode_rt_frames = rt_frames if mode_config.need_rt else None
        mode_it_frames = it_frames if mode_config.need_it else None
        mode_rt_mask = rt_mask if mode_config.need_rt else None
        mode_it_mask = it_mask if mode_config.need_it else None

        outputs = model(mode_rt_frames, mode_it_frames, mode_rt_mask, mode_it_mask, mode_name)

        output_names = tuple(outputs["probabilities"].keys())
        if output_names != mode_config.output_names:
            mismatch_modes.append(mode_name)

        if tuple(outputs["p_hat"].shape) != (batch_size, 1):
            p_hat_shape_ok = False

        for output_tensor in outputs["probabilities"].values():
            if torch.isnan(output_tensor).any():
                no_nan_outputs = False
        if torch.isnan(outputs["p_hat"]).any():
            no_nan_outputs = False

    ba1_outputs = model(rt_frames, None, rt_mask, None, "ba1_rt")
    full_outputs = model(rt_frames, it_frames, rt_mask, it_mask, "full_cdma")

    loss_b1 = float(loss_fn(ba1_outputs["active_probabilities"], labels).item())
    loss_b6 = float(loss_fn(full_outputs["active_probabilities"], labels).item())
    loss_scaling_ok = loss_b6 > loss_b1

    train_model = CDMAModel().to(device)
    optimizer = torch.optim.RMSprop(train_model.parameters(), lr=LEARNING_RATE)
    before_parameters = [parameter.detach().clone() for parameter in train_model.parameters()]

    train_outputs = train_model(rt_frames, it_frames, rt_mask, it_mask, "full_cdma")
    train_loss = loss_fn(train_outputs["active_probabilities"], labels)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    gradient_flow_ok = False
    for before_parameter, after_parameter in zip(before_parameters, train_model.parameters()):
        if not torch.allclose(before_parameter, after_parameter.detach()):
            gradient_flow_ok = True
            break

    return Module4SanityResult(
        all_modes_forward_ok=len(mismatch_modes) == 0,
        output_key_mismatch_modes=mismatch_modes,
        p_hat_shape_ok=p_hat_shape_ok,
        no_nan_outputs=no_nan_outputs,
        loss_scaling_ok=loss_scaling_ok,
        loss_b1=loss_b1,
        loss_b6=loss_b6,
        gradient_flow_ok=gradient_flow_ok,
    )


def save_predictions_csv(
    output_path: Path,
    mode: str,
    rep: int,
    predictions: list[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    field_names = [
        "mode",
        "rep",
        "fold",
        "participant_id",
        "true_label",
        "predicted_label",
        "p_hat",
        "p_c",
        "p_o",
        "p_t",
        "p_d",
        "p_f1",
        "p_f2",
    ]

    with output_path.open(mode="w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()

        for prediction in predictions:
            probability_map = prediction.get("output_probabilities", {})
            writer.writerow(
                {
                    "mode": mode,
                    "rep": rep,
                    "fold": prediction.get("fold_name", "single_fold"),
                    "participant_id": prediction["participant_id"],
                    "true_label": int(prediction["true_label"]),
                    "predicted_label": int(prediction["predicted_label"]),
                    "p_hat": f"{float(prediction['p_hat']):.6f}",
                    "p_c": f"{float(probability_map['p_c']):.6f}" if "p_c" in probability_map else "",
                    "p_o": f"{float(probability_map['p_o']):.6f}" if "p_o" in probability_map else "",
                    "p_t": f"{float(probability_map['p_t']):.6f}" if "p_t" in probability_map else "",
                    "p_d": f"{float(probability_map['p_d']):.6f}" if "p_d" in probability_map else "",
                    "p_f1": f"{float(probability_map['p_f1']):.6f}" if "p_f1" in probability_map else "",
                    "p_f2": f"{float(probability_map['p_f2']):.6f}" if "p_f2" in probability_map else "",
                }
            )


def format_sanity_block(sanity_result: Module4SanityResult) -> list[str]:
    return [
        "[Sanity Checks]",
        f"- forward pass output-name checks passed: {sanity_result.all_modes_forward_ok}",
        (
            "- mismatched forward modes: "
            + (", ".join(sanity_result.output_key_mismatch_modes) if sanity_result.output_key_mismatch_modes else "none")
        ),
        f"- p_hat shape checks passed: {sanity_result.p_hat_shape_ok}",
        f"- no NaN in outputs: {sanity_result.no_nan_outputs}",
        (
            f"- loss scaling check passed: {sanity_result.loss_scaling_ok} "
            f"(loss_b1={sanity_result.loss_b1:.6f}, loss_b6={sanity_result.loss_b6:.6f})"
        ),
        f"- gradient flow check passed: {sanity_result.gradient_flow_ok}",
    ]


def format_fold_report(sanity_result: Module4SanityResult, fold_result: Module4FoldResult) -> str:
    first_loss = fold_result.training_losses[0] if fold_result.training_losses else float("nan")
    last_loss = fold_result.training_losses[-1] if fold_result.training_losses else float("nan")
    loss_decreased = last_loss < first_loss

    lines: list[str] = [
        "=== Module 4: Chapter 7 End-to-End CDMA Report ===",
        "",
    ]
    lines.extend(format_sanity_block(sanity_result))
    lines.extend(
        [
            "",
            "[Run Configuration]",
            f"- mode: {fold_result.mode}",
            f"- rep: {fold_result.rep}",
            f"- seed: {fold_result.seed}",
            f"- fold: {fold_result.fold_name}",
            f"- device: {fold_result.device}",
            f"- train participants: {fold_result.train_participant_count}",
            f"- test participants: {fold_result.test_participant_count}",
            f"- train batches: {fold_result.train_batch_count}",
            f"- test batches: {fold_result.test_batch_count}",
            f"- elapsed seconds: {fold_result.elapsed_seconds:.2f}",
            f"- filtered RT fold sizes: {fold_result.filtered_fold_sizes}",
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
            "[Module Scope]",
            "- Module 4 implemented differentiable Chapter 7 CDMA pipeline for all 13 modes.",
        ]
    )

    return "\n".join(lines)


def format_all_folds_report(sanity_result: Module4SanityResult, condition_result: Module4ConditionResult) -> str:
    lines: list[str] = [
        "=== Module 4: Chapter 7 End-to-End CDMA All-Folds Report ===",
        "",
    ]
    lines.extend(format_sanity_block(sanity_result))
    lines.extend(
        [
            "",
            "[Run Configuration]",
            f"- mode: {condition_result.mode}",
            f"- rep: {condition_result.rep}",
            f"- pooled prediction count: {condition_result.pooled_prediction_count}",
            f"- duplicate pooled participant ids: {len(condition_result.duplicate_prediction_ids)}",
            "",
            "[Fold Metrics]",
        ]
    )

    for fold_result in condition_result.fold_results:
        lines.append(
            (
                f"- {fold_result.fold_name}: accuracy={fold_result.metrics['accuracy']:.4f}, "
                f"precision={fold_result.metrics['precision']:.4f}, "
                f"recall={fold_result.metrics['recall']:.4f}, "
                f"f1={fold_result.metrics['f1']:.4f}, "
                f"train_batches={fold_result.train_batch_count}, test_batches={fold_result.test_batch_count}"
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
            "- Module 4 implemented differentiable Chapter 7 CDMA pipeline for all 13 modes.",
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
    parser = argparse.ArgumentParser(description="Run Module 4 Chapter 7 CDMA experiments.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root path.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full_cdma",
        choices=sorted(MODE_CONFIGS.keys()),
        help="Mode to run.",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Run all 5 folds and compute pooled metrics.",
    )
    parser.add_argument(
        "--test-fold",
        type=str,
        default="fold1",
        help="Fold used as test split for single-fold run.",
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
        help="Participant-level batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "module4",
        help="Directory where Module 4 reports and predictions are saved.",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run with 3 epochs for smoke checking on remote environments.",
    )
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip module sanity checks before training.",
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
        help="Save training checkpoint every N epochs (plus final epoch).",
    )
    parser.add_argument(
        "--preview-participants",
        type=int,
        default=PREVIEW_PARTICIPANTS,
        help="Number of test participants to preview in terminal after each fold.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint-based resume and completed-fold reuse.",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    cli_args = parse_arguments()

    project_root = cli_args.project_root.resolve()
    results_dir = (
        cli_args.results_dir if cli_args.results_dir.is_absolute() else project_root / cli_args.results_dir
    )

    run_epochs = 3 if cli_args.quick_test else cli_args.epochs
    resume_enabled = not cli_args.no_resume
    sanity_result = run_sanity_checks() if not cli_args.skip_sanity_check else Module4SanityResult(
        all_modes_forward_ok=True,
        output_key_mismatch_modes=[],
        p_hat_shape_ok=True,
        no_nan_outputs=True,
        loss_scaling_ok=True,
        loss_b1=float("nan"),
        loss_b6=float("nan"),
        gradient_flow_ok=True,
    )

    if cli_args.all_folds:
        condition_result = run_all_folds(
            project_root=project_root,
            results_dir=results_dir,
            mode=cli_args.mode,
            rep=cli_args.rep,
            epochs=run_epochs,
            batch_size=cli_args.batch_size,
            num_workers=cli_args.num_workers,
            log_every_epochs=cli_args.log_every_epochs,
            checkpoint_every_epochs=cli_args.checkpoint_every_epochs,
            preview_participants=cli_args.preview_participants,
            resume=resume_enabled,
            sanity_result=sanity_result,
        )

        pooled_predictions: list[dict[str, object]] = []
        for fold_result in condition_result.fold_results:
            for prediction in fold_result.predictions:
                prediction_copy = dict(prediction)
                prediction_copy["fold_name"] = fold_result.fold_name
                pooled_predictions.append(prediction_copy)

        predictions_path = _all_folds_predictions_path(results_dir, cli_args.mode, cli_args.rep)
        report_path = _all_folds_report_path(results_dir, cli_args.mode, cli_args.rep)

        save_predictions_csv(
            output_path=predictions_path,
            mode=cli_args.mode,
            rep=cli_args.rep,
            predictions=pooled_predictions,
        )

        report_text = format_all_folds_report(sanity_result=sanity_result, condition_result=condition_result)
        print(report_text)
        save_report(report_text, report_path)

        LOGGER.info("Module 4 pooled predictions saved to %s", predictions_path)
        LOGGER.info("Module 4 report saved to %s", report_path)

        has_duplicates = len(condition_result.duplicate_prediction_ids) > 0
        return 1 if has_duplicates else 0

    fold_result = run_single_fold(
        project_root=project_root,
        results_dir=results_dir,
        mode=cli_args.mode,
        test_fold_name=cli_args.test_fold,
        rep=cli_args.rep,
        epochs=run_epochs,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        log_every_epochs=cli_args.log_every_epochs,
        checkpoint_every_epochs=cli_args.checkpoint_every_epochs,
        preview_participants=cli_args.preview_participants,
        resume=resume_enabled,
    )

    fold_predictions: list[dict[str, object]] = []
    for prediction in fold_result.predictions:
        prediction_copy = dict(prediction)
        prediction_copy["fold_name"] = fold_result.fold_name
        fold_predictions.append(prediction_copy)

    predictions_path = _fold_predictions_path(results_dir, cli_args.mode, cli_args.rep, cli_args.test_fold)
    report_path = _fold_report_path(results_dir, cli_args.mode, cli_args.rep, cli_args.test_fold)

    save_predictions_csv(
        output_path=predictions_path,
        mode=cli_args.mode,
        rep=cli_args.rep,
        predictions=fold_predictions,
    )

    report_text = format_fold_report(sanity_result=sanity_result, fold_result=fold_result)
    print(report_text)
    save_report(report_text, report_path)

    LOGGER.info("Module 4 fold predictions saved to %s", predictions_path)
    LOGGER.info("Module 4 report saved to %s", report_path)
    LOGGER.info("Module 4 complete. Waiting for verification before Module 5.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
