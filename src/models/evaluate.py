from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.models.dataset import DISEASE_LABELS, SUPPORTED_LABELS, safe_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs" / "vision"


def to_study_level(pred_df: pd.DataFrame) -> pd.DataFrame:
    prob_cols = [col for col in pred_df.columns if col.startswith("prob_")]
    true_cols = [col for col in pred_df.columns if col.startswith("true_")]

    study_probs = pred_df.groupby("study_id")[prob_cols].mean().reset_index()
    study_true = pred_df.groupby("study_id")[true_cols].max().reset_index()
    return study_true.merge(study_probs, on="study_id")


def compute_metric_table(
    study_df: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    thresholds = thresholds or {label: 0.5 for label in DISEASE_LABELS}
    rows = []

    for label in DISEASE_LABELS:
        name = safe_name(label)
        y_true = study_df[f"true_{name}"].astype(int)
        y_prob = study_df[f"prob_{name}"].astype(float)
        y_pred = (y_prob >= thresholds.get(label, 0.5)).astype(int)

        rows.append(
            {
                "label": label,
                "test_positive_studies": int(y_true.sum()),
                "AUROC": None if y_true.nunique() < 2 else roc_auc_score(y_true, y_prob),
                "AP": average_precision_score(y_true, y_prob),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "threshold": thresholds.get(label, 0.5),
                "note": "data-limited" if label in ["Edema", "Pneumothorax"] else "",
            }
        )

    return pd.DataFrame(rows)


def macro_auroc_supported(metric_table: pd.DataFrame) -> float:
    return metric_table[metric_table["label"].isin(SUPPORTED_LABELS)]["AUROC"].mean()


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, float(best_f1)


def choose_thresholds(val_study_preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in DISEASE_LABELS:
        name = safe_name(label)
        y_true = val_study_preds[f"true_{name}"].astype(int).values
        y_prob = val_study_preds[f"prob_{name}"].astype(float).values
        threshold, val_f1 = best_f1_threshold(y_true, y_prob)
        rows.append({"label": label, "threshold": threshold, "val_F1": val_f1})
    return pd.DataFrame(rows)


def thresholds_to_dict(thresholds_df: pd.DataFrame) -> dict[str, float]:
    return dict(zip(thresholds_df["label"], thresholds_df["threshold"]))
