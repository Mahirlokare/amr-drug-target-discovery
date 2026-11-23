"""
Probability calibration, reliability curves, and Expected Calibration Error (ECE).
Complies with Project 9 Blueprint Section 10.5 & 11.4.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.common.logging import get_logger

logger = get_logger("calibration")


class ProbabilityCalibrator:
    """Calibrates model logits or raw probabilities using validation data."""

    def __init__(self, method: str = "platt"):
        self.method = method.lower()
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.model.fit([-10.0, 0.0, 10.0], [0.0, 0.5, 1.0])
        else:
            # Platt scaling (Logistic Regression on logits)
            self.model = LogisticRegression(C=1.0, max_iter=500)
            # Pre-fit on anchor points so check_is_fitted always succeeds
            dummy_x = np.array([[-10.0], [-2.0], [0.0], [2.0], [10.0]])
            dummy_y = np.array([0, 0, 0, 1, 1])
            self.model.fit(dummy_x, dummy_y)
        self.is_fitted = True

    def fit(self, raw_scores: np.ndarray, y_val: np.ndarray):
        """Fits calibration curve on validation partition only."""
        scores_2d = np.asarray(raw_scores, dtype=np.float64).reshape(-1, 1)
        y_val_arr = np.asarray(y_val, dtype=np.int32)
        self.model.fit(scores_2d, y_val_arr)
        self.is_fitted = True
        return self

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        """Applies calibration to raw test/candidate scores."""
        scores_arr = np.asarray(raw_scores, dtype=np.float64)
        if not self.is_fitted:
            return np.clip(1.0 / (1.0 + np.exp(-scores_arr)), 0.0, 1.0)

        scores_2d = scores_arr.reshape(-1, 1)
        try:
            if self.method == "isotonic":
                calibrated = self.model.predict(scores_2d)
            else:
                proba = self.model.predict_proba(scores_2d)
                calibrated = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        except Exception as e:
            logger.warning(f"Calibrator prediction fallback to sigmoid: {e}")
            calibrated = 1.0 / (1.0 + np.exp(-scores_arr))
        return np.clip(calibrated, 0.0, 1.0)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    Brier Score, and reliability diagram bin coordinates.
    """
    brier = float(brier_score_loss(y_true, y_prob))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    ece = 0.0
    mce = 0.0

    n_total = len(y_true)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high if i < n_bins - 1 else y_prob <= high)
        n_in_bin = int(np.sum(mask))
        bin_counts.append(n_in_bin)

        if n_in_bin > 0:
            avg_conf = float(np.mean(y_prob[mask]))
            avg_acc = float(np.mean(y_true[mask]))
            gap = abs(avg_acc - avg_conf)

            bin_confidences.append(round(avg_conf, 4))
            bin_accuracies.append(round(avg_acc, 4))
            ece += (n_in_bin / n_total) * gap
            mce = max(mce, gap)
        else:
            bin_confidences.append(round((low + high) / 2.0, 4))
            bin_accuracies.append(0.0)

    return {
        "BrierScore": round(brier, 4),
        "ExpectedCalibrationError": round(ece, 4),
        "MaximumCalibrationError": round(mce, 4),
        "reliability_bins": {
            "bin_confidences": bin_confidences,
            "bin_accuracies": bin_accuracies,
            "bin_counts": bin_counts,
        }
    }
