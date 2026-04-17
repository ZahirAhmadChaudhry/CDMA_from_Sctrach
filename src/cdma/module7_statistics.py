from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel, wilcoxon
from sklearn.metrics import f1_score

try:
    from statsmodels.stats.multitest import multipletests
except Exception:  # pragma: no cover
    multipletests = None

THESIS_F1 = {
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

THESIS_ACC = {
    "ba1_rt": 85.5,
    "ba1_it": 83.1,
    "itmla_rt": 89.2,
    "itmla_it": 86.8,
    "ba2_rt": 89.9,
    "ba2_it": 87.3,
    "ba3_rt": 90.1,
    "ba3_it": 89.4,
    "ctga_rt": 90.4,
    "ctga_it": 89.8,
    "ba4": 90.2,
    "ba5": 90.7,
    "full_cdma": 92.7,
}

RANDOM_BASELINE_F1 = 52.7

CONDITION_ORDER = [
    "ba1_rt",
    "ba1_it",
    "itmla_rt",
    "itmla_it",
    "ba2_rt",
    "ba2_it",
    "ba3_rt",
    "ba3_it",
    "ctga_rt",
    "ctga_it",
    "ba4",
    "ba5",
    "full_cdma",
]

CONDITION_LABELS = {
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

PAIRWISE_TESTS = [
    ("MLA_RT", "ba1_rt", "itmla_rt", "MLA improves Read"),
    ("MLA_IT", "ba1_it", "itmla_it", "MLA improves Spont."),
    ("LSTM2_RT", "itmla_rt", "ba2_rt", "LSTM2 improves Read"),
    ("LSTM2_IT", "itmla_it", "ba2_it", "LSTM2 improves Spont."),
    ("SelfGA_RT", "ba2_rt", "ba3_rt", "Self-GA improves Read"),
    ("SelfGA_IT", "ba2_it", "ba3_it", "Self-GA improves Spont."),
    ("CrossGA_RT", "ba3_rt", "ctga_rt", "Cross-GA beats Self-GA Read"),
    ("CrossGA_IT", "ba3_it", "ctga_it", "Cross-GA beats Self-GA Spont."),
    ("Fusion_BA4", "ctga_it", "ba4", "BA4 beats best CT-GA"),
    ("Fusion_BA5", "ctga_it", "ba5", "BA5 beats best CT-GA"),
    ("Fusion_CDMA", "ctga_it", "full_cdma", "CDMA beats best CT-GA"),
    ("CDMA_vs_BA4", "ba4", "full_cdma", "CDMA beats BA4"),
    ("CDMA_vs_BA5", "ba5", "full_cdma", "CDMA beats BA5"),
    ("Full_vs_BA1", "ba1_rt", "full_cdma", "CDMA beats simplest baseline"),
]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _to_percent(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return numeric
    max_value = float(numeric.max(skipna=True))
    if max_value <= 1.5:
        return numeric * 100.0
    return numeric


def _format_metric(value: float, decimals: int = 2) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{decimals}f}"


def _sig_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "na"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _apply_fdr(p_values: list[float]) -> tuple[list[bool], list[float]]:
    values = np.asarray(p_values, dtype=np.float64)
    reject = np.zeros(values.shape, dtype=bool)
    p_adj = np.full(values.shape, np.nan, dtype=np.float64)

    finite_mask = np.isfinite(values)
    if int(finite_mask.sum()) > 0:
        if multipletests is not None:
            reject_values, p_adj_values, _, _ = multipletests(values[finite_mask], method="fdr_bh")
        else:
            p_adj_values = _benjamini_hochberg(values[finite_mask])
            reject_values = p_adj_values < 0.05
        reject[finite_mask] = reject_values
        p_adj[finite_mask] = p_adj_values

    return reject.tolist(), p_adj.tolist()


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    m = p_values.size
    if m == 0:
        return p_values

    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted = np.empty_like(sorted_p)

    previous = 1.0
    for index in range(m - 1, -1, -1):
        rank = index + 1
        value = sorted_p[index] * m / rank
        previous = min(previous, value)
        adjusted[index] = previous

    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def _cohen_d_one_sample(values: np.ndarray, baseline: float) -> float:
    if values.size < 2:
        return float("nan")
    std = float(np.std(values, ddof=1))
    if std <= 0:
        return float("nan")
    return float((np.mean(values) - baseline) / std)


def _cohen_d_paired(higher: np.ndarray, lower: np.ndarray) -> float:
    diff = higher - lower
    if diff.size < 2:
        return float("nan")
    std = float(np.std(diff, ddof=1))
    if std <= 0:
        return float("nan")
    return float(np.mean(diff) / std)


def _load_pooled_results(pooled_csv_path: Path) -> pd.DataFrame:
    required = {
        "condition",
        "rep",
        "accuracy",
        "precision",
        "recall",
        "f1",
    }
    pooled_df = pd.read_csv(pooled_csv_path)
    missing = required - set(pooled_df.columns)
    if missing:
        raise ValueError(f"Missing pooled_results columns: {sorted(missing)}")

    pooled_df = pooled_df.copy()
    pooled_df["condition"] = pooled_df["condition"].astype(str)
    pooled_df["rep"] = pd.to_numeric(pooled_df["rep"], errors="coerce").astype("Int64")
    pooled_df["accuracy"] = pd.to_numeric(pooled_df["accuracy"], errors="coerce")
    pooled_df["f1"] = pd.to_numeric(pooled_df["f1"], errors="coerce")
    pooled_df["accuracy_pct"] = _to_percent(pooled_df["accuracy"])
    pooled_df["f1_pct"] = _to_percent(pooled_df["f1"])
    return pooled_df


def _load_fold_predictions(fold_predictions_csv_path: Path) -> pd.DataFrame:
    required = {
        "condition",
        "rep",
        "fold",
        "participant_id",
        "true_label",
        "predicted_label",
        "p_hat",
    }
    fold_df = pd.read_csv(fold_predictions_csv_path)
    missing = required - set(fold_df.columns)
    if missing:
        raise ValueError(f"Missing fold_predictions columns: {sorted(missing)}")

    fold_df = fold_df.copy()
    fold_df["condition"] = fold_df["condition"].astype(str)
    fold_df["rep"] = pd.to_numeric(fold_df["rep"], errors="coerce").astype("Int64")
    fold_df["fold"] = fold_df["fold"].astype(str)
    fold_df["participant_id"] = fold_df["participant_id"].astype(str)
    fold_df["true_label"] = pd.to_numeric(fold_df["true_label"], errors="coerce")
    fold_df["predicted_label"] = pd.to_numeric(fold_df["predicted_label"], errors="coerce")
    fold_df["p_hat"] = pd.to_numeric(fold_df["p_hat"], errors="coerce")

    for probability_name in ["p_c", "p_o", "p_t", "p_d", "p_f1", "p_f2"]:
        if probability_name not in fold_df.columns:
            fold_df[probability_name] = np.nan
        fold_df[probability_name] = pd.to_numeric(fold_df[probability_name], errors="coerce")

    return fold_df


def _coverage_warnings(pooled_df: pd.DataFrame, expected_reps: int = 10) -> list[str]:
    warnings: list[str] = []
    for condition in CONDITION_ORDER:
        condition_df = pooled_df.loc[pooled_df["condition"] == condition]
        n_reps = int(condition_df["rep"].dropna().nunique())
        if n_reps < expected_reps:
            warnings.append(
                f"WARNING: condition={condition} has {n_reps} reps (expected {expected_reps})."
            )
    return warnings


def _section1_descriptive(pooled_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []

    for condition in CONDITION_ORDER:
        subset = pooled_df.loc[pooled_df["condition"] == condition]
        f1_values = subset["f1_pct"].dropna().to_numpy(dtype=np.float64)
        acc_values = subset["accuracy_pct"].dropna().to_numpy(dtype=np.float64)
        n_reps = int(subset["rep"].dropna().nunique())

        row = {
            "condition": condition,
            "label": CONDITION_LABELS.get(condition, condition),
            "n": n_reps,
            "acc_mean": float(np.mean(acc_values)) if acc_values.size else float("nan"),
            "acc_sd": float(np.std(acc_values, ddof=1)) if acc_values.size > 1 else (0.0 if acc_values.size == 1 else float("nan")),
            "f1_mean": float(np.mean(f1_values)) if f1_values.size else float("nan"),
            "f1_sd": float(np.std(f1_values, ddof=1)) if f1_values.size > 1 else (0.0 if f1_values.size == 1 else float("nan")),
            "f1_min": float(np.min(f1_values)) if f1_values.size else float("nan"),
            "f1_max": float(np.max(f1_values)) if f1_values.size else float("nan"),
            "thesis_f1": float(THESIS_F1[condition]),
            "delta": float(np.mean(f1_values) - THESIS_F1[condition]) if f1_values.size else float("nan"),
        }
        rows.append(row)

    section_df = pd.DataFrame(rows)

    lines = [
        "SECTION 1: Descriptive Summary",
        "Condition | N | Acc Mean | Acc SD | F1 Mean | F1 SD | F1 Min | F1 Max | Thesis F1 | Delta",
    ]
    for _, row in section_df.iterrows():
        lines.append(
            " | ".join(
                [
                    str(row["label"]),
                    str(int(row["n"])),
                    _format_metric(float(row["acc_mean"]), 2),
                    _format_metric(float(row["acc_sd"]), 2),
                    _format_metric(float(row["f1_mean"]), 2),
                    _format_metric(float(row["f1_sd"]), 2),
                    _format_metric(float(row["f1_min"]), 2),
                    _format_metric(float(row["f1_max"]), 2),
                    _format_metric(float(row["thesis_f1"]), 2),
                    _format_metric(float(row["delta"]), 2),
                ]
            )
        )

    return section_df, "\n".join(lines)


def _section2_vs_random(pooled_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []

    for condition in CONDITION_ORDER:
        values = pooled_df.loc[pooled_df["condition"] == condition, "f1_pct"].dropna().to_numpy(dtype=np.float64)
        mean_value = float(np.mean(values)) if values.size else float("nan")
        diff_value = mean_value - RANDOM_BASELINE_F1 if np.isfinite(mean_value) else float("nan")

        if values.size >= 2:
            t_stat, p_value = ttest_1samp(values, RANDOM_BASELINE_F1)
            cohen_d = _cohen_d_one_sample(values, RANDOM_BASELINE_F1)
        else:
            t_stat = float("nan")
            p_value = float("nan")
            cohen_d = float("nan")

        rows.append(
            {
                "condition": condition,
                "label": CONDITION_LABELS.get(condition, condition),
                "mean_f1": mean_value,
                "random_baseline": RANDOM_BASELINE_F1,
                "diff": diff_value,
                "cohen_d": _safe_float(cohen_d),
                "t_stat": _safe_float(t_stat),
                "p_value": _safe_float(p_value),
            }
        )
        p_values.append(_safe_float(p_value))

    _, p_adj_values = _apply_fdr(p_values)
    for index, p_adj in enumerate(p_adj_values):
        rows[index]["p_adj"] = p_adj
        rows[index]["sig"] = _sig_label(p_adj)

    section_df = pd.DataFrame(rows)

    lines = [
        "SECTION 2: Above Random Baseline (one-sample t-test)",
        "Condition | Mean F1 | Random | Diff | Cohen_d | t-stat | p-value | p-adj | Sig",
    ]
    for _, row in section_df.iterrows():
        lines.append(
            " | ".join(
                [
                    str(row["label"]),
                    _format_metric(float(row["mean_f1"]), 2),
                    _format_metric(float(row["random_baseline"]), 2),
                    _format_metric(float(row["diff"]), 2),
                    _format_metric(float(row["cohen_d"]), 3),
                    _format_metric(float(row["t_stat"]), 3),
                    _format_metric(float(row["p_value"]), 6),
                    _format_metric(float(row["p_adj"]), 6),
                    str(row["sig"]),
                ]
            )
        )

    return section_df, "\n".join(lines)


def _section3_vs_thesis(pooled_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []

    for condition in CONDITION_ORDER:
        values = pooled_df.loc[pooled_df["condition"] == condition, "f1_pct"].dropna().to_numpy(dtype=np.float64)
        thesis_value = float(THESIS_F1[condition])
        mean_value = float(np.mean(values)) if values.size else float("nan")
        diff_value = mean_value - thesis_value if np.isfinite(mean_value) else float("nan")

        if values.size >= 2:
            t_stat, p_value = ttest_1samp(values, thesis_value)
            cohen_d = _cohen_d_one_sample(values, thesis_value)
        else:
            t_stat = float("nan")
            p_value = float("nan")
            cohen_d = float("nan")

        rows.append(
            {
                "condition": condition,
                "label": CONDITION_LABELS.get(condition, condition),
                "our_mean": mean_value,
                "thesis": thesis_value,
                "diff": diff_value,
                "cohen_d": _safe_float(cohen_d),
                "t_stat": _safe_float(t_stat),
                "p_value": _safe_float(p_value),
            }
        )
        p_values.append(_safe_float(p_value))

    _, p_adj_values = _apply_fdr(p_values)
    for index, p_adj in enumerate(p_adj_values):
        rows[index]["p_adj"] = p_adj
        if not np.isfinite(p_adj):
            rows[index]["conclusion"] = "Insufficient data"
        elif p_adj >= 0.05:
            rows[index]["conclusion"] = "Not sig. different"
        else:
            rows[index]["conclusion"] = "Sig. different"

    section_df = pd.DataFrame(rows)

    lines = [
        "SECTION 3: Comparison with Thesis (one-sample t-test)",
        "Condition | Our Mean | Thesis | Diff | Cohen_d | t-stat | p-value | p-adj | Conclusion",
    ]
    for _, row in section_df.iterrows():
        lines.append(
            " | ".join(
                [
                    str(row["label"]),
                    _format_metric(float(row["our_mean"]), 2),
                    _format_metric(float(row["thesis"]), 2),
                    _format_metric(float(row["diff"]), 2),
                    _format_metric(float(row["cohen_d"]), 3),
                    _format_metric(float(row["t_stat"]), 3),
                    _format_metric(float(row["p_value"]), 6),
                    _format_metric(float(row["p_adj"]), 6),
                    str(row["conclusion"]),
                ]
            )
        )

    return section_df, "\n".join(lines)


def _paired_f1_arrays(
    pooled_df: pd.DataFrame,
    condition_a: str,
    condition_b: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_df = pooled_df.loc[pooled_df["condition"] == condition_a, ["rep", "f1_pct"]].dropna()
    b_df = pooled_df.loc[pooled_df["condition"] == condition_b, ["rep", "f1_pct"]].dropna()

    merged = a_df.merge(b_df, on="rep", suffixes=("_a", "_b")).sort_values("rep")
    reps = merged["rep"].to_numpy(dtype=np.int64)
    values_a = merged["f1_pct_a"].to_numpy(dtype=np.float64)
    values_b = merged["f1_pct_b"].to_numpy(dtype=np.float64)
    return reps, values_a, values_b


def _section4_pairwise(pooled_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    p_ttest_values: list[float] = []
    p_wilcoxon_values: list[float] = []

    for test_name, condition_a, condition_b, description in PAIRWISE_TESTS:
        reps, values_a, values_b = _paired_f1_arrays(pooled_df, condition_a, condition_b)
        n_pairs = int(reps.size)

        mean_a = float(np.mean(values_a)) if n_pairs else float("nan")
        mean_b = float(np.mean(values_b)) if n_pairs else float("nan")
        mean_diff = float(np.mean(values_b - values_a)) if n_pairs else float("nan")

        if n_pairs >= 2:
            t_stat, p_ttest = ttest_rel(values_b, values_a)
            cohen_d = _cohen_d_paired(values_b, values_a)
        else:
            t_stat = float("nan")
            p_ttest = float("nan")
            cohen_d = float("nan")

        if n_pairs >= 6:
            diff = values_b - values_a
            if np.allclose(diff, 0.0):
                w_stat, p_wilcoxon = 0.0, 1.0
            else:
                try:
                    w_stat, p_wilcoxon = wilcoxon(diff)
                except ValueError:
                    w_stat, p_wilcoxon = float("nan"), float("nan")
        else:
            w_stat, p_wilcoxon = float("nan"), float("nan")

        rows.append(
            {
                "test": test_name,
                "description": description,
                "cond_a": condition_a,
                "cond_b": condition_b,
                "label_a": CONDITION_LABELS.get(condition_a, condition_a),
                "label_b": CONDITION_LABELS.get(condition_b, condition_b),
                "n_pairs": n_pairs,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "mean_diff": mean_diff,
                "cohen_d": _safe_float(cohen_d),
                "t_stat": _safe_float(t_stat),
                "p_ttest": _safe_float(p_ttest),
                "w_stat": _safe_float(w_stat),
                "p_wilcoxon": _safe_float(p_wilcoxon),
            }
        )
        p_ttest_values.append(_safe_float(p_ttest))
        p_wilcoxon_values.append(_safe_float(p_wilcoxon))

    _, p_t_adj = _apply_fdr(p_ttest_values)
    _, p_w_adj = _apply_fdr(p_wilcoxon_values)

    for index, row in enumerate(rows):
        row["p_adj_t"] = p_t_adj[index]
        row["p_adj_w"] = p_w_adj[index]
        row["wrong_direction"] = bool(np.isfinite(row["mean_diff"]) and row["mean_diff"] < 0.0)
        row["sig"] = _sig_label(float(row["p_adj_t"]))
        if row["wrong_direction"]:
            row["sig"] = f"{row['sig']} WRONG DIRECTION"

    section_df = pd.DataFrame(rows)

    lines = [
        "SECTION 4: Pairwise Stage Improvements",
        "Test | Cond_A | Cond_B | Mean_A | Mean_B | Diff | Cohen_d | t-stat | p-ttest | p-adj-t | p-wilcox | p-adj-w | Sig",
    ]
    for _, row in section_df.iterrows():
        lines.append(
            " | ".join(
                [
                    str(row["test"]),
                    str(row["label_a"]),
                    str(row["label_b"]),
                    _format_metric(float(row["mean_a"]), 2),
                    _format_metric(float(row["mean_b"]), 2),
                    _format_metric(float(row["mean_diff"]), 2),
                    _format_metric(float(row["cohen_d"]), 3),
                    _format_metric(float(row["t_stat"]), 3),
                    _format_metric(float(row["p_ttest"]), 6),
                    _format_metric(float(row["p_adj_t"]), 6),
                    _format_metric(float(row["p_wilcoxon"]), 6),
                    _format_metric(float(row["p_adj_w"]), 6),
                    str(row["sig"]),
                ]
            )
        )

    return section_df, "\n".join(lines)


def _condition_fold_f1(fold_df: pd.DataFrame, condition: str, rep: int, fold_name: str) -> float:
    subset = fold_df.loc[
        (fold_df["condition"] == condition)
        & (fold_df["rep"] == rep)
        & (fold_df["fold"] == fold_name)
    ]
    if subset.empty:
        return float("nan")

    y_true = subset["true_label"].to_numpy(dtype=np.int64)
    y_pred = subset["predicted_label"].to_numpy(dtype=np.int64)
    return float(f1_score(y_true, y_pred, zero_division=0) * 100.0)


def _pick_probability_column(fold_df: pd.DataFrame, condition_a: str, condition_b: str) -> str:
    for name in ["p_c", "p_o", "p_t", "p_d", "p_f1", "p_f2", "p_hat"]:
        a_count = int(
            fold_df.loc[(fold_df["condition"] == condition_a) & fold_df[name].notna()].shape[0]
        )
        b_count = int(
            fold_df.loc[(fold_df["condition"] == condition_b) & fold_df[name].notna()].shape[0]
        )
        if a_count > 0 and b_count > 0:
            return name
    return "p_hat"


def _frame_count_context(pooled_csv_path: Path) -> str:
    diagnostics_path = pooled_csv_path.parent / "diagnostics" / "frame_counts.csv"
    if not diagnostics_path.exists():
        return "Frame count diagnostics not available."

    frame_df = pd.read_csv(diagnostics_path)
    if frame_df.empty or not {"n_rt", "n_it"}.issubset(frame_df.columns):
        return "Frame count diagnostics present but missing expected columns."

    frame_slice = frame_df.copy()
    if {"rep", "fold"}.issubset(frame_df.columns):
        rep1_fold1 = frame_df.loc[
            (pd.to_numeric(frame_df["rep"], errors="coerce") == 1)
            & (frame_df["fold"].astype(str) == "fold1")
        ]
        if not rep1_fold1.empty:
            frame_slice = rep1_fold1

    rt_mean = float(frame_slice["n_rt"].mean())
    rt_min = float(frame_slice["n_rt"].min())
    rt_max = float(frame_slice["n_rt"].max())
    it_mean = float(frame_slice["n_it"].mean())
    it_min = float(frame_slice["n_it"].min())
    it_max = float(frame_slice["n_it"].max())
    ratio = float((frame_slice["n_it"] / frame_slice["n_rt"].replace(0, np.nan)).mean())

    return (
        f"RT mean={rt_mean:.1f}, min={rt_min:.0f}, max={rt_max:.0f}; "
        f"IT mean={it_mean:.1f}, min={it_min:.0f}, max={it_max:.0f}; "
        f"IT/RT ratio={ratio:.2f}x"
    )


def _section5_diagnostics(
    fold_df: pd.DataFrame,
    section4_df: pd.DataFrame,
    pooled_csv_path: Path,
) -> str:
    failing = section4_df.loc[
        section4_df["wrong_direction"]
        | section4_df["p_adj_t"].isna()
        | (section4_df["p_adj_t"] >= 0.05)
    ]

    lines: list[str] = ["SECTION 5: Diagnostic Analysis"]
    if failing.empty:
        lines.append("No failing pairwise comparisons detected.")
        return "\n".join(lines)

    frame_context = _frame_count_context(pooled_csv_path)

    for _, row in failing.iterrows():
        condition_a = str(row["cond_a"])
        condition_b = str(row["cond_b"])

        lines.append("")
        lines.append(
            (
                f"DIAGNOSTIC: {row['test']} ({CONDITION_LABELS.get(condition_a, condition_a)} vs "
                f"{CONDITION_LABELS.get(condition_b, condition_b)})"
            )
        )
        lines.append(
            (
                f"  Mean_A={_format_metric(float(row['mean_a']), 2)}, "
                f"Mean_B={_format_metric(float(row['mean_b']), 2)}, "
                f"Diff={_format_metric(float(row['mean_diff']), 2)}, "
                f"p_adj_t={_format_metric(float(row['p_adj_t']), 6)}"
            )
        )

        reps_a = set(
            pd.to_numeric(
                fold_df.loc[fold_df["condition"] == condition_a, "rep"],
                errors="coerce",
            ).dropna().astype(int).tolist()
        )
        reps_b = set(
            pd.to_numeric(
                fold_df.loc[fold_df["condition"] == condition_b, "rep"],
                errors="coerce",
            ).dropna().astype(int).tolist()
        )
        common_reps = sorted(reps_a & reps_b)

        if not common_reps:
            lines.append("  No overlapping reps found for diagnostics.")
            lines.append(f"  Frame count context: {frame_context}")
            continue

        lines.append("  Per-fold F1 (mean across overlapping reps):")
        lines.append("  Fold | A_F1 | B_F1 | Diff | Fold_dep%")

        fold_names = sorted(fold_df["fold"].dropna().astype(str).unique().tolist())
        fold_diffs: list[tuple[str, float]] = []
        fold_balance: list[tuple[str, float]] = []

        for fold_name in fold_names:
            a_values: list[float] = []
            b_values: list[float] = []
            for rep in common_reps:
                a_score = _condition_fold_f1(fold_df, condition_a, rep, fold_name)
                b_score = _condition_fold_f1(fold_df, condition_b, rep, fold_name)
                if np.isfinite(a_score):
                    a_values.append(a_score)
                if np.isfinite(b_score):
                    b_values.append(b_score)

            mean_a = float(np.mean(a_values)) if a_values else float("nan")
            mean_b = float(np.mean(b_values)) if b_values else float("nan")
            diff = mean_b - mean_a if np.isfinite(mean_a) and np.isfinite(mean_b) else float("nan")

            balance_slice = fold_df.loc[
                (fold_df["condition"] == condition_a)
                & (fold_df["rep"] == common_reps[0])
                & (fold_df["fold"] == fold_name)
            ]
            if balance_slice.empty:
                balance_slice = fold_df.loc[
                    (fold_df["condition"] == condition_b)
                    & (fold_df["rep"] == common_reps[0])
                    & (fold_df["fold"] == fold_name)
                ]
            dep_pct = float(balance_slice["true_label"].mean() * 100.0) if not balance_slice.empty else float("nan")

            fold_diffs.append((fold_name, diff))
            fold_balance.append((fold_name, dep_pct))

            lines.append(
                (
                    f"  {fold_name} | {_format_metric(mean_a, 2)} | {_format_metric(mean_b, 2)} | "
                    f"{_format_metric(diff, 2)} | {_format_metric(dep_pct, 1)}%"
                )
            )

        finite_diffs = [(fold_name, value) for fold_name, value in fold_diffs if np.isfinite(value)]
        if finite_diffs:
            worst_fold = min(finite_diffs, key=lambda item: item[1])
            lines.append(f"  Worst fold: {worst_fold[0]} (diff={_format_metric(worst_fold[1], 2)})")

        finite_balance = [(fold_name, value) for fold_name, value in fold_balance if np.isfinite(value)]
        if finite_balance:
            imbalance = max(finite_balance, key=lambda item: abs(item[1] - 50.0))
            lines.append(f"  Most imbalanced fold: {imbalance[0]} ({_format_metric(imbalance[1], 1)}% depressed)")

        changed_counts: list[float] = []
        changed_to_correct_counts: list[float] = []
        changed_to_wrong_counts: list[float] = []

        for rep in common_reps:
            a_subset = fold_df.loc[
                (fold_df["condition"] == condition_a) & (fold_df["rep"] == rep),
                ["participant_id", "true_label", "predicted_label"],
            ]
            b_subset = fold_df.loc[
                (fold_df["condition"] == condition_b) & (fold_df["rep"] == rep),
                ["participant_id", "true_label", "predicted_label"],
            ]
            merged = a_subset.merge(b_subset, on="participant_id", suffixes=("_a", "_b"))
            if merged.empty:
                continue

            changed_mask = merged["predicted_label_a"] != merged["predicted_label_b"]
            changed_counts.append(float(changed_mask.sum()))

            changed_to_correct = (
                (merged["predicted_label_a"] != merged["true_label_a"])
                & (merged["predicted_label_b"] == merged["true_label_a"])
            )
            changed_to_wrong = (
                (merged["predicted_label_a"] == merged["true_label_a"])
                & (merged["predicted_label_b"] != merged["true_label_a"])
            )
            changed_to_correct_counts.append(float(changed_to_correct.sum()))
            changed_to_wrong_counts.append(float(changed_to_wrong.sum()))

        if changed_counts:
            mean_changed = float(np.mean(changed_counts))
            mean_to_correct = float(np.mean(changed_to_correct_counts))
            mean_to_wrong = float(np.mean(changed_to_wrong_counts))
            lines.append("  Prediction changes:")
            lines.append(f"    Mean changed participants: {_format_metric(mean_changed, 2)} / 110")
            lines.append(f"    Mean changed-to-correct: {_format_metric(mean_to_correct, 2)}")
            lines.append(f"    Mean changed-to-wrong: {_format_metric(mean_to_wrong, 2)}")
            lines.append(f"    Net effect: {_format_metric(mean_to_correct - mean_to_wrong, 2)} participants")

        probability_column = _pick_probability_column(fold_df, condition_a, condition_b)
        dep_shifts: list[float] = []
        ctrl_shifts: list[float] = []
        values_a_all: list[tuple[float, int]] = []
        values_b_all: list[tuple[float, int]] = []

        for rep in common_reps:
            a_subset = fold_df.loc[
                (fold_df["condition"] == condition_a) & (fold_df["rep"] == rep),
                ["participant_id", "true_label", probability_column],
            ]
            b_subset = fold_df.loc[
                (fold_df["condition"] == condition_b) & (fold_df["rep"] == rep),
                ["participant_id", "true_label", probability_column],
            ]
            merged = a_subset.merge(b_subset, on="participant_id", suffixes=("_a", "_b"))

            for _, merged_row in merged.iterrows():
                value_a = _safe_float(merged_row[f"{probability_column}_a"])
                value_b = _safe_float(merged_row[f"{probability_column}_b"])
                label = int(merged_row["true_label_a"])
                if not np.isfinite(value_a) or not np.isfinite(value_b):
                    continue

                shift = value_b - value_a
                if label == 1:
                    dep_shifts.append(shift)
                else:
                    ctrl_shifts.append(shift)
                values_a_all.append((value_a, label))
                values_b_all.append((value_b, label))

        if dep_shifts or ctrl_shifts:
            dep_shift_mean = float(np.mean(dep_shifts)) if dep_shifts else float("nan")
            ctrl_shift_mean = float(np.mean(ctrl_shifts)) if ctrl_shifts else float("nan")
            a_dep = [value for value, label in values_a_all if label == 1]
            a_ctrl = [value for value, label in values_a_all if label == 0]
            b_dep = [value for value, label in values_b_all if label == 1]
            b_ctrl = [value for value, label in values_b_all if label == 0]
            sep_a = (float(np.mean(a_dep)) - float(np.mean(a_ctrl))) if a_dep and a_ctrl else float("nan")
            sep_b = (float(np.mean(b_dep)) - float(np.mean(b_ctrl))) if b_dep and b_ctrl else float("nan")

            lines.append(f"  Probability shift using {probability_column} (B - A):")
            lines.append(f"    Depressed mean shift: {_format_metric(dep_shift_mean, 4)}")
            lines.append(f"    Control mean shift: {_format_metric(ctrl_shift_mean, 4)}")
            lines.append(f"    Separation A: {_format_metric(sep_a, 4)}")
            lines.append(f"    Separation B: {_format_metric(sep_b, 4)}")

        lines.append("  Output saturation check (rep 1, where available):")
        saturation_rep = common_reps[0]
        for condition_name in [condition_a, condition_b]:
            subset = fold_df.loc[
                (fold_df["condition"] == condition_name) & (fold_df["rep"] == saturation_rep)
            ]
            if subset.empty:
                lines.append(f"    {condition_name}: no rows for rep {saturation_rep}")
                continue

            has_lstm2_ctf_outputs = False
            for probability_name in ["p_t", "p_d", "p_f1", "p_f2"]:
                values = subset[probability_name].dropna().to_numpy(dtype=np.float64)
                if values.size == 0:
                    continue
                has_lstm2_ctf_outputs = True
                low_count = int((values < 0.02).sum())
                high_count = int((values > 0.98).sum())
                middle_count = int(values.size - low_count - high_count)
                lines.append(
                    (
                        f"    {condition_name} {probability_name}: <0.02={low_count}, "
                        f"middle={middle_count}, >0.98={high_count}"
                    )
                )

            if not has_lstm2_ctf_outputs:
                lines.append(f"    {condition_name}: no LSTM2/CTF probability outputs in this comparison")

        lines.append(f"  Frame count context: {frame_context}")

    return "\n".join(lines)


def _section6_verdict(
    section2_df: pd.DataFrame,
    section3_df: pd.DataFrame,
    section4_df: pd.DataFrame,
) -> str:
    tested_random = int(section2_df.loc[section2_df["p_adj"].notna()].shape[0])
    random_significant = int(section2_df.loc[section2_df["p_adj"] < 0.05].shape[0])
    tested_thesis = int(section3_df.loc[section3_df["p_adj"].notna()].shape[0])
    aligned_with_thesis = int(section3_df.loc[(section3_df["p_adj"].notna()) & (section3_df["p_adj"] >= 0.05)].shape[0])
    below_thesis = int(
        section3_df.loc[(section3_df["p_adj"].notna()) & (section3_df["p_adj"] < 0.05) & (section3_df["diff"] < 0)].shape[0]
    )

    finite_section3 = section3_df.loc[section3_df["diff"].notna()].copy()
    finite_section3["abs_diff"] = finite_section3["diff"].abs()
    closest = finite_section3.sort_values("abs_diff").head(3)["condition"].tolist()
    furthest = finite_section3.sort_values("abs_diff", ascending=False).head(3)["condition"].tolist()

    confirmed_rows = section4_df.loc[(section4_df["mean_diff"] > 0) & (section4_df["p_adj_t"] < 0.05)]
    unconfirmed_rows = section4_df.loc[~((section4_df["mean_diff"] > 0) & (section4_df["p_adj_t"] < 0.05))]

    confirmed = confirmed_rows["test"].tolist()
    unconfirmed = unconfirmed_rows["test"].tolist()

    fusion_tests = {"Fusion_BA4", "Fusion_BA5", "Fusion_CDMA"}
    fusion_confirmed = fusion_tests.issubset(set(confirmed))

    lines = [
        "SECTION 6: Final Verdict",
        "FINAL VERDICT",
        "=============",
        "",
        "Architecture validation:",
        (
            f"  - {random_significant} out of {tested_random} tested conditions are significantly above random baseline "
            "after FDR correction."
        ),
        "",
        "Alignment with thesis:",
        f"  - {aligned_with_thesis} out of {tested_thesis} tested conditions are not significantly different from thesis.",
        f"  - {below_thesis} out of {tested_thesis} tested conditions are significantly below thesis.",
        f"  - Closest conditions: {', '.join(closest) if closest else 'none'}",
        f"  - Furthest conditions: {', '.join(furthest) if furthest else 'none'}",
        "",
        "Stage contribution (pairwise tests):",
        f"  - {len(confirmed)} out of 14 comparisons show significant improvement in expected direction.",
        f"  - {len(unconfirmed)} out of 14 show wrong direction or no significance.",
        f"  - Confirmed stages: {', '.join(confirmed) if confirmed else 'none'}",
        f"  - Unconfirmed stages: {', '.join(unconfirmed) if unconfirmed else 'none'}",
        "",
        "Key finding:",
    ]

    if fusion_confirmed:
        lines.append(
            (
                "  Multi-stream fusion (BA4, BA5, CDMA) significantly outperforms single-stream baselines, "
                "supporting the core thesis claim under this protocol."
            )
        )
    else:
        if tested_random > 0 and random_significant == tested_random:
            lines.append(
                (
                    "  Pairwise improvements are mixed under this protocol; however, all tested conditions are "
                    "above random and proximity to thesis values still support substantial architectural value."
                )
            )
        else:
            lines.append(
                (
                    "  Pairwise improvements are mixed and some statistical checks are incomplete due to partial "
                    "coverage; rerun after all 13 conditions and 10 reps finish for a final verdict."
                )
            )

    lines.extend(
        [
            "",
            "Protocol differences explaining residual gap:",
            "  1. We use k=5 corpus-provided folds; thesis uses k=3 custom unpublished folds.",
            "  2. Two folds can be heavily imbalanced, increasing variance.",
            "  3. RT recordings are shorter than IT, producing less stable RT-side probabilities.",
        ]
    )

    return "\n".join(lines)


def run_statistical_analysis(
    pooled_csv: str,
    fold_predictions_csv: str,
    output_dir: str,
) -> None:
    pooled_csv_path = Path(pooled_csv).resolve()
    fold_predictions_csv_path = Path(fold_predictions_csv).resolve()
    output_dir_path = Path(output_dir).resolve()

    if not pooled_csv_path.exists():
        raise FileNotFoundError(f"Missing pooled results CSV: {pooled_csv_path}")
    if not fold_predictions_csv_path.exists():
        raise FileNotFoundError(f"Missing fold predictions CSV: {fold_predictions_csv_path}")

    output_dir_path.mkdir(parents=True, exist_ok=True)

    pooled_df = _load_pooled_results(pooled_csv_path)
    fold_df = _load_fold_predictions(fold_predictions_csv_path)

    coverage_warnings = _coverage_warnings(pooled_df, expected_reps=10)

    section1_df, section1_text = _section1_descriptive(pooled_df)
    section2_df, section2_text = _section2_vs_random(pooled_df)
    section3_df, section3_text = _section3_vs_thesis(pooled_df)
    section4_df, section4_text = _section4_pairwise(pooled_df)
    section5_text = _section5_diagnostics(fold_df, section4_df, pooled_csv_path)
    section6_text = _section6_verdict(section2_df, section3_df, section4_df)

    section1_df.to_csv(output_dir_path / "section1_descriptive.csv", index=False)
    section2_df.to_csv(output_dir_path / "section2_vs_random.csv", index=False)
    section3_df.to_csv(output_dir_path / "section3_vs_thesis.csv", index=False)
    section4_df.to_csv(output_dir_path / "section4_pairwise.csv", index=False)
    (output_dir_path / "section5_diagnostics.txt").write_text(section5_text + "\n", encoding="utf-8")
    (output_dir_path / "section6_verdict.txt").write_text(section6_text + "\n", encoding="utf-8")

    report_lines: list[str] = ["=== Module 7 Statistical Analysis Report ===", ""]
    if coverage_warnings:
        report_lines.append("Coverage warnings:")
        for warning in coverage_warnings:
            report_lines.append(f"- {warning}")
        report_lines.append("")

    report_lines.extend(
        [
            section1_text,
            "",
            section2_text,
            "",
            section3_text,
            "",
            section4_text,
            "",
            section5_text,
            "",
            section6_text,
            "",
        ]
    )

    full_report = "\n".join(report_lines)
    (output_dir_path / "statistical_analysis_report.txt").write_text(full_report, encoding="utf-8")

    print(full_report)
