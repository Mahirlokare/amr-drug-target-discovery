"""
Evaluation metrics, enrichment factors, and statistical reporting.
Complies with Project 9 Blueprint Section 11.4, 11.5 & 20.2.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from src.models.calibration import compute_calibration_metrics


def compute_enrichment_factor(
    y_true: np.ndarray,
    y_score: np.ndarray,
    top_fraction: float = 0.01
) -> float:
    """
    Computes Enrichment Factor at top_fraction (e.g. 1% or 5%).
    Formula:
        EF@k% = (Actives in top k% / Total samples in top k%) / (Total Actives / Total samples)
    """
    n_total = len(y_true)
    total_actives = int(np.sum(y_true))
    if n_total == 0 or total_actives == 0:
        return 0.0

    k = max(1, int(np.ceil(top_fraction * n_total)))
    sorted_indices = np.argsort(-y_score)
    top_k_labels = y_true[sorted_indices[:k]]

    actives_in_top_k = int(np.sum(top_k_labels))
    hit_rate_top_k = actives_in_top_k / k
    prevalence = total_actives / n_total

    ef = hit_rate_top_k / prevalence
    return round(float(ef), 3)


def compute_hit_rate_at_budget(
    y_true: np.ndarray,
    y_score: np.ndarray,
    budget_n: int = 25
) -> float:
    """Computes hit rate (fraction of verified actives) in top N candidates."""
    if len(y_true) == 0:
        return 0.0
    k = min(budget_n, len(y_true))
    sorted_indices = np.argsort(-y_score)
    top_labels = y_true[sorted_indices[:k]]
    return round(float(np.mean(top_labels)), 4)


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes full scientific metric suite adhering to Section 11.4:
    AUROC, PR-AUC, Balanced Accuracy, F1 Macro, EF@1%, EF@5%, Hit Rate @ 25,
    Brier Score, and Expected Calibration Error.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    n_samples = len(y_true)
    n_actives = int(np.sum(y_true))
    prevalence = round(n_actives / n_samples, 4) if n_samples > 0 else 0.0

    # ROC-AUC
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auroc = 0.5

    # PR-AUC
    try:
        pr_auc = float(average_precision_score(y_true, y_score))
    except Exception:
        pr_auc = prevalence

    # Binary thresholded metrics
    y_pred = (y_score >= threshold).astype(int)
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Enrichment factors
    ef_1pct = compute_enrichment_factor(y_true, y_score, top_fraction=0.01)
    ef_5pct = compute_enrichment_factor(y_true, y_score, top_fraction=0.05)
    hit_rate_25 = compute_hit_rate_at_budget(y_true, y_score, budget_n=25)

    # Calibration metrics
    cal_metrics = compute_calibration_metrics(y_true, y_score, n_bins=10)

    return {
        "sample_count": n_samples,
        "active_count": n_actives,
        "prevalence_baseline": prevalence,
        "AUROC": round(auroc, 4),
        "PR_AUC": round(pr_auc, 4),
        "BalancedAccuracy": round(balanced_acc, 4),
        "F1_Macro": round(f1, 4),
        "EF_1pct": ef_1pct,
        "EF_5pct": ef_5pct,
        "HitRate_Top25": hit_rate_25,
        "BrierScore": cal_metrics["BrierScore"],
        "ExpectedCalibrationError": cal_metrics["ExpectedCalibrationError"],
        "reliability_bins": cal_metrics["reliability_bins"],
    }


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstraps: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Computes split-aware bootstrap confidence intervals for key metrics.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n < 10:
        return {}

    auroc_list = []
    pr_auc_list = []
    ef_1pct_list = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        bs_true = y_true[idx]
        bs_score = y_score[idx]

        if len(np.unique(bs_true)) < 2:
            continue

        try:
            auroc_list.append(roc_auc_score(bs_true, bs_score))
        except Exception:
            pass

        try:
            pr_auc_list.append(average_precision_score(bs_true, bs_score))
        except Exception:
            pass

        ef_1pct_list.append(compute_enrichment_factor(bs_true, bs_score, 0.01))

    alpha = (1.0 - confidence_level) / 2.0
    lower_p = alpha * 100
    upper_p = (1.0 - alpha) * 100

    def get_ci(vals: List[float]) -> Tuple[float, float]:
        if not vals:
            return (0.0, 0.0)
        return (round(float(np.percentile(vals, lower_p)), 4), round(float(np.percentile(vals, upper_p)), 4))

    return {
        "AUROC_CI": get_ci(auroc_list),
        "PR_AUC_CI": get_ci(pr_auc_list),
        "EF_1pct_CI": get_ci(ef_1pct_list),
    }
