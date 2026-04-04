from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from cdma.module4_cdma import (
    BATCH_SIZE,
    CHECKPOINT_EVERY_EPOCHS,
    EPOCHS,
    LOG_EVERY_EPOCHS,
    MODE_CONFIGS,
    PREVIEW_PARTICIPANTS,
    Module4ConditionResult,
    Module4FoldResult,
    Module4SanityResult,
    compute_binary_metrics,
    format_all_folds_report,
    format_fold_report,
    run_sanity_checks,
    run_single_fold,
    save_predictions_csv,
    save_report,
)

LOGGER = logging.getLogger(__name__)

FOLD_NAMES = ["fold1", "fold2", "fold3", "fold4", "fold5"]
OUTPUT_PROBABILITY_NAMES = ["p_c", "p_o", "p_t", "p_d", "p_f1", "p_f2"]
DEFAULT_CONDITIONS = list(MODE_CONFIGS.keys())

FOLD_PREDICTIONS_FIELDS = [
    "condition",
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
    "timestamp",
]

POOLED_RESULTS_FIELDS = [
    "condition",
    "rep",
    "seed",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "prediction_count",
    "duplicate_prediction_ids",
    "timestamp",
]

COMPLETED_FOLDS_FIELDS = [
    "condition",
    "rep",
    "fold",
    "timestamp",
]


@dataclass(frozen=True)
class Module5RunConfig:
    project_root: Path
    results_dir: Path
    conditions: list[str]
    reps: list[int]
    epochs: int
    batch_size: int
    num_workers: int
    log_every_epochs: int
    checkpoint_every_epochs: int
    preview_participants: int
    resume: bool
    output_mode: str
    skip_sanity_check: bool


@dataclass
class Module5State:
    prediction_keys: set[tuple[str, str, str, str]]
    completed_fold_keys: set[tuple[str, str, str]]
    pooled_result_keys: set[tuple[str, str]]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _all_folds_predictions_path(results_dir: Path, condition: str, rep: int) -> Path:
    return results_dir / f"{condition}_rep{rep}_all_folds_predictions.csv"


def _all_folds_report_path(results_dir: Path, condition: str, rep: int) -> Path:
    return results_dir / f"{condition}_rep{rep}_all_folds_report.txt"


def _fold_predictions_path(results_dir: Path, condition: str, rep: int, fold_name: str) -> Path:
    return results_dir / f"{condition}_rep{rep}_{fold_name}_predictions.csv"


def _fold_report_path(results_dir: Path, condition: str, rep: int, fold_name: str) -> Path:
    return results_dir / f"{condition}_rep{rep}_{fold_name}_report.txt"


def _fold_predictions_table_path(results_dir: Path) -> Path:
    return results_dir / "fold_predictions.csv"


def _pooled_results_table_path(results_dir: Path) -> Path:
    return results_dir / "pooled_results.csv"


def _completed_folds_table_path(results_dir: Path) -> Path:
    return results_dir / "completed_folds.csv"


def _ensure_csv_header(csv_path: Path, field_names: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return

    with csv_path.open(mode="w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()


def _load_prediction_keys(csv_path: Path) -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    if not csv_path.exists():
        return keys

    with csv_path.open(mode="r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            keys.add(
                (
                    row["condition"],
                    row["rep"],
                    row["fold"],
                    row["participant_id"],
                )
            )

    return keys


def _load_completed_fold_keys(csv_path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    if not csv_path.exists():
        return keys

    with csv_path.open(mode="r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            keys.add((row["condition"], row["rep"], row["fold"]))

    return keys


def _load_pooled_result_keys(csv_path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not csv_path.exists():
        return keys

    with csv_path.open(mode="r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            keys.add((row["condition"], row["rep"]))

    return keys


def _load_cached_fold_predictions(
    fold_predictions_csv_path: Path,
    condition: str,
    rep: int,
    fold_name: str,
) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    if not fold_predictions_csv_path.exists():
        return predictions

    rep_str = str(rep)

    with fold_predictions_csv_path.open(mode="r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row["condition"] != condition:
                continue
            if row["rep"] != rep_str:
                continue
            if row["fold"] != fold_name:
                continue

            output_probabilities: dict[str, float] = {}
            for output_name in OUTPUT_PROBABILITY_NAMES:
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
                    "fold_name": fold_name,
                }
            )

    return predictions


def _append_rows(csv_path: Path, field_names: list[str], rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    _ensure_csv_header(csv_path, field_names)

    with csv_path.open(mode="a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        for row in rows:
            writer.writerow(row)


def _build_fold_predictions_rows(
    condition: str,
    rep: int,
    fold_name: str,
    predictions: list[dict[str, object]],
    prediction_keys: set[tuple[str, str, str, str]],
) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    rep_str = str(rep)
    row_timestamp = int(time.time())

    for prediction in predictions:
        participant_id = str(prediction["participant_id"])
        key = (condition, rep_str, fold_name, participant_id)
        if key in prediction_keys:
            continue

        prediction_keys.add(key)
        probability_map = prediction.get("output_probabilities", {})

        output_rows.append(
            {
                "condition": condition,
                "rep": rep,
                "fold": fold_name,
                "participant_id": participant_id,
                "true_label": int(prediction["true_label"]),
                "predicted_label": int(prediction["predicted_label"]),
                "p_hat": f"{float(prediction['p_hat']):.6f}",
                "p_c": f"{float(probability_map['p_c']):.6f}" if "p_c" in probability_map else "",
                "p_o": f"{float(probability_map['p_o']):.6f}" if "p_o" in probability_map else "",
                "p_t": f"{float(probability_map['p_t']):.6f}" if "p_t" in probability_map else "",
                "p_d": f"{float(probability_map['p_d']):.6f}" if "p_d" in probability_map else "",
                "p_f1": f"{float(probability_map['p_f1']):.6f}" if "p_f1" in probability_map else "",
                "p_f2": f"{float(probability_map['p_f2']):.6f}" if "p_f2" in probability_map else "",
                "timestamp": row_timestamp,
            }
        )

    return output_rows


def _build_completed_fold_row(condition: str, rep: int, fold_name: str) -> dict[str, object]:
    return {
        "condition": condition,
        "rep": rep,
        "fold": fold_name,
        "timestamp": int(time.time()),
    }


def _build_pooled_result_row(
    condition: str,
    rep: int,
    metrics: dict[str, float],
    prediction_count: int,
    duplicate_prediction_ids: set[str],
) -> dict[str, object]:
    return {
        "condition": condition,
        "rep": rep,
        "seed": rep * 42,
        "accuracy": f"{metrics['accuracy']:.6f}",
        "precision": f"{metrics['precision']:.6f}",
        "recall": f"{metrics['recall']:.6f}",
        "f1": f"{metrics['f1']:.6f}",
        "prediction_count": prediction_count,
        "duplicate_prediction_ids": len(duplicate_prediction_ids),
        "timestamp": int(time.time()),
    }


def _find_duplicate_participant_ids(predictions: list[dict[str, object]]) -> set[str]:
    seen_participants: set[str] = set()
    duplicate_ids: set[str] = set()

    for prediction in predictions:
        participant_id = str(prediction["participant_id"])
        if participant_id in seen_participants:
            duplicate_ids.add(participant_id)
        else:
            seen_participants.add(participant_id)

    return duplicate_ids


def _load_module5_state(results_dir: Path) -> Module5State:
    fold_predictions_csv_path = _fold_predictions_table_path(results_dir)
    completed_folds_csv_path = _completed_folds_table_path(results_dir)
    pooled_results_csv_path = _pooled_results_table_path(results_dir)

    _ensure_csv_header(fold_predictions_csv_path, FOLD_PREDICTIONS_FIELDS)
    _ensure_csv_header(completed_folds_csv_path, COMPLETED_FOLDS_FIELDS)
    _ensure_csv_header(pooled_results_csv_path, POOLED_RESULTS_FIELDS)

    return Module5State(
        prediction_keys=_load_prediction_keys(fold_predictions_csv_path),
        completed_fold_keys=_load_completed_fold_keys(completed_folds_csv_path),
        pooled_result_keys=_load_pooled_result_keys(pooled_results_csv_path),
    )


def run_condition(
    config: Module5RunConfig,
    state: Module5State,
    sanity_result: Module4SanityResult,
    condition: str,
) -> list[dict[str, object]]:
    if condition not in MODE_CONFIGS:
        raise ValueError(f"Unsupported condition: {condition}")

    fold_predictions_csv_path = _fold_predictions_table_path(config.results_dir)
    completed_folds_csv_path = _completed_folds_table_path(config.results_dir)
    pooled_results_csv_path = _pooled_results_table_path(config.results_dir)

    pooled_rows: list[dict[str, object]] = []

    for rep in config.reps:
        rep_str = str(rep)
        pooled_key = (condition, rep_str)

        if config.resume and pooled_key in state.pooled_result_keys:
            LOGGER.info("Skipping completed condition=%s rep=%d based on pooled_results.csv", condition, rep)
            continue

        fold_results: list[Module4FoldResult] = []
        pooled_predictions: list[dict[str, object]] = []

        for fold_name in FOLD_NAMES:
            fold_key = (condition, rep_str, fold_name)

            use_cached_fold = config.resume and fold_key in state.completed_fold_keys
            if use_cached_fold:
                cached_predictions = _load_cached_fold_predictions(
                    fold_predictions_csv_path=fold_predictions_csv_path,
                    condition=condition,
                    rep=rep,
                    fold_name=fold_name,
                )

                if cached_predictions:
                    cached_metrics = compute_binary_metrics(cached_predictions)
                    fold_result = Module4FoldResult(
                        mode=condition,
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
                        "Skipping completed fold condition=%s rep=%d fold=%s from fold_predictions.csv",
                        condition,
                        rep,
                        fold_name,
                    )
                else:
                    LOGGER.warning(
                        (
                            "Completed fold marker exists but fold predictions are missing for "
                            "condition=%s rep=%d fold=%s. Re-running this fold."
                        ),
                        condition,
                        rep,
                        fold_name,
                    )
                    fold_result = run_single_fold(
                        project_root=config.project_root,
                        results_dir=config.results_dir,
                        mode=condition,
                        test_fold_name=fold_name,
                        rep=rep,
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        log_every_epochs=config.log_every_epochs,
                        checkpoint_every_epochs=config.checkpoint_every_epochs,
                        preview_participants=config.preview_participants,
                        resume=config.resume,
                    )
            else:
                fold_result = run_single_fold(
                    project_root=config.project_root,
                    results_dir=config.results_dir,
                    mode=condition,
                    test_fold_name=fold_name,
                    rep=rep,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    log_every_epochs=config.log_every_epochs,
                    checkpoint_every_epochs=config.checkpoint_every_epochs,
                    preview_participants=config.preview_participants,
                    resume=config.resume,
                )

            fold_results.append(fold_result)

            fold_predictions_with_fold_name: list[dict[str, object]] = []
            for prediction in fold_result.predictions:
                prediction_copy = dict(prediction)
                prediction_copy["fold_name"] = fold_name
                fold_predictions_with_fold_name.append(prediction_copy)
                pooled_predictions.append(prediction_copy)

            fold_rows = _build_fold_predictions_rows(
                condition=condition,
                rep=rep,
                fold_name=fold_name,
                predictions=fold_predictions_with_fold_name,
                prediction_keys=state.prediction_keys,
            )
            _append_rows(fold_predictions_csv_path, FOLD_PREDICTIONS_FIELDS, fold_rows)

            if fold_key not in state.completed_fold_keys:
                _append_rows(
                    completed_folds_csv_path,
                    COMPLETED_FOLDS_FIELDS,
                    [_build_completed_fold_row(condition=condition, rep=rep, fold_name=fold_name)],
                )
                state.completed_fold_keys.add(fold_key)

            if config.output_mode == "detailed":
                fold_predictions_path = _fold_predictions_path(config.results_dir, condition, rep, fold_name)
                fold_report_path = _fold_report_path(config.results_dir, condition, rep, fold_name)

                save_predictions_csv(
                    output_path=fold_predictions_path,
                    mode=condition,
                    rep=rep,
                    predictions=fold_predictions_with_fold_name,
                )
                save_report(
                    report_text=format_fold_report(sanity_result=sanity_result, fold_result=fold_result),
                    output_path=fold_report_path,
                )

            if config.output_mode == "overwrite":
                interim_metrics = compute_binary_metrics(pooled_predictions)
                interim_duplicates = _find_duplicate_participant_ids(pooled_predictions)
                interim_result = Module4ConditionResult(
                    mode=condition,
                    rep=rep,
                    fold_results=fold_results,
                    pooled_metrics=interim_metrics,
                    pooled_prediction_count=len(pooled_predictions),
                    duplicate_prediction_ids=interim_duplicates,
                )

                save_predictions_csv(
                    output_path=_all_folds_predictions_path(config.results_dir, condition, rep),
                    mode=condition,
                    rep=rep,
                    predictions=pooled_predictions,
                )
                save_report(
                    report_text=format_all_folds_report(sanity_result=sanity_result, condition_result=interim_result),
                    output_path=_all_folds_report_path(config.results_dir, condition, rep),
                )

        duplicate_prediction_ids = _find_duplicate_participant_ids(pooled_predictions)
        pooled_metrics = compute_binary_metrics(pooled_predictions)

        condition_result = Module4ConditionResult(
            mode=condition,
            rep=rep,
            fold_results=fold_results,
            pooled_metrics=pooled_metrics,
            pooled_prediction_count=len(pooled_predictions),
            duplicate_prediction_ids=duplicate_prediction_ids,
        )

        save_predictions_csv(
            output_path=_all_folds_predictions_path(config.results_dir, condition, rep),
            mode=condition,
            rep=rep,
            predictions=pooled_predictions,
        )
        save_report(
            report_text=format_all_folds_report(sanity_result=sanity_result, condition_result=condition_result),
            output_path=_all_folds_report_path(config.results_dir, condition, rep),
        )

        pooled_row = _build_pooled_result_row(
            condition=condition,
            rep=rep,
            metrics=pooled_metrics,
            prediction_count=len(pooled_predictions),
            duplicate_prediction_ids=duplicate_prediction_ids,
        )
        _append_rows(pooled_results_csv_path, POOLED_RESULTS_FIELDS, [pooled_row])
        pooled_rows.append(pooled_row)

        state.pooled_result_keys.add(pooled_key)

        LOGGER.info(
            (
                "Completed condition=%s rep=%d pooled metrics: "
                "accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f predictions=%d duplicates=%d"
            ),
            condition,
            rep,
            pooled_metrics["accuracy"],
            pooled_metrics["precision"],
            pooled_metrics["recall"],
            pooled_metrics["f1"],
            len(pooled_predictions),
            len(duplicate_prediction_ids),
        )

    return pooled_rows


def run_experiment_suite(config: Module5RunConfig) -> list[dict[str, object]]:
    if config.output_mode not in {"detailed", "overwrite"}:
        raise ValueError(f"Unsupported output mode: {config.output_mode}")

    state = _load_module5_state(config.results_dir)

    sanity_result = (
        Module4SanityResult(
            all_modes_forward_ok=True,
            output_key_mismatch_modes=[],
            p_hat_shape_ok=True,
            no_nan_outputs=True,
            loss_scaling_ok=True,
            loss_b1=float("nan"),
            loss_b6=float("nan"),
            gradient_flow_ok=True,
        )
        if config.skip_sanity_check
        else run_sanity_checks()
    )

    all_pooled_rows: list[dict[str, object]] = []

    for condition in config.conditions:
        condition_rows = run_condition(
            config=config,
            state=state,
            sanity_result=sanity_result,
            condition=condition,
        )
        all_pooled_rows.extend(condition_rows)

    return all_pooled_rows


def parse_conditions(raw_conditions: str) -> list[str]:
    normalized = raw_conditions.strip().lower()
    if normalized in {"all", "all13", "*"}:
        return list(DEFAULT_CONDITIONS)

    parsed_conditions: list[str] = []
    for condition_token in raw_conditions.split(","):
        condition_name = condition_token.strip()
        if not condition_name:
            continue
        if condition_name not in MODE_CONFIGS:
            raise ValueError(f"Unsupported condition in --conditions: {condition_name}")
        if condition_name not in parsed_conditions:
            parsed_conditions.append(condition_name)

    if not parsed_conditions:
        raise ValueError("No valid conditions were provided.")

    return parsed_conditions


def parse_reps(rep_count: int, rep_start: int) -> list[int]:
    if rep_count <= 0:
        raise ValueError("--reps must be positive.")
    if rep_start <= 0:
        raise ValueError("--rep-start must be positive.")

    return list(range(rep_start, rep_start + rep_count))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 5 runner: pooled k-fold experiments with resume support.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root path.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "module5",
        help="Output directory for fold_predictions.csv and pooled_results.csv.",
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
        "--quick-test",
        action="store_true",
        help="Fast smoke mode: 1 rep, first 2 conditions, 3 epochs.",
    )
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip Module 4 sanity checks before running experiments.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume and force all requested runs.",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    cli_args = parse_arguments()

    project_root = cli_args.project_root.resolve()
    results_dir = cli_args.results_dir if cli_args.results_dir.is_absolute() else project_root / cli_args.results_dir

    conditions = parse_conditions(cli_args.conditions)
    reps = parse_reps(rep_count=cli_args.reps, rep_start=cli_args.rep_start)
    run_epochs = cli_args.epochs

    if cli_args.quick_test:
        conditions = conditions[:2]
        reps = reps[:1]
        run_epochs = 3

    run_config = Module5RunConfig(
        project_root=project_root,
        results_dir=results_dir,
        conditions=conditions,
        reps=reps,
        epochs=run_epochs,
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
        (
            "Starting Module 5 runner with conditions=%s reps=%s epochs=%d results_dir=%s "
            "resume=%s output_mode=%s"
        ),
        run_config.conditions,
        run_config.reps,
        run_config.epochs,
        run_config.results_dir,
        run_config.resume,
        run_config.output_mode,
    )

    pooled_rows = run_experiment_suite(run_config)

    if pooled_rows:
        LOGGER.info("Module 5 run complete. New pooled rows written: %d", len(pooled_rows))
    else:
        LOGGER.info("Module 5 run complete. No new pooled rows were written (likely resume skip).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
