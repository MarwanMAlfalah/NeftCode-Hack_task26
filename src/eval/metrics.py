"""Shared regression metrics for baseline evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PLATFORM_TARGET_SCALES = {
    "target_delta_kinematic_viscosity_pct": 2439.25,
    "target_oxidation_eot_a_per_cm": 160.62,
}


def _as_2d_array(values: np.ndarray | list[list[float]] | list[float]) -> np.ndarray:
    """Convert arbitrary array-like values into a 2D float array."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def compute_target_scales(y_train: np.ndarray, target_names: Iterable[str]) -> dict[str, float]:
    """Compute robust per-target scales for combined-score normalization.

    The combined score uses per-target RMSE normalized by the training-fold IQR.
    If a target's IQR is zero, the function falls back to standard deviation and
    finally to `1.0` to keep the metric numerically stable.
    """

    y_array = _as_2d_array(y_train)
    scales: dict[str, float] = {}
    for index, name in enumerate(target_names):
        column = y_array[:, index]
        q75, q25 = np.nanpercentile(column, [75, 25])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.nanstd(column))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        scales[name] = scale
    return scales


def evaluate_regression_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Iterable[str],
    target_scales: dict[str, float],
) -> dict[str, float]:
    """Compute target-wise metrics plus a scale-balanced combined score."""

    y_true_array = _as_2d_array(y_true)
    y_pred_array = np.nan_to_num(
        _as_2d_array(y_pred),
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )
    y_pred_array = np.clip(y_pred_array, -1e6, 1e6)
    metrics: dict[str, float] = {}
    normalized_rmse: list[float] = []

    for index, target_name in enumerate(target_names):
        true_column = y_true_array[:, index]
        pred_column = y_pred_array[:, index]
        rmse = float(np.sqrt(mean_squared_error(true_column, pred_column)))
        mae = float(mean_absolute_error(true_column, pred_column))
        r2 = float(r2_score(true_column, pred_column))
        nrmse = rmse / target_scales[target_name]

        metrics[f"{target_name}__rmse"] = rmse
        metrics[f"{target_name}__mae"] = mae
        metrics[f"{target_name}__r2"] = r2
        metrics[f"{target_name}__nrmse_iqr"] = nrmse
        normalized_rmse.append(nrmse)

    metrics["combined_score"] = float(np.mean(normalized_rmse))
    metrics["combined_r2_mean"] = float(
        np.mean([metrics[f"{target_name}__r2"] for target_name in target_names])
    )
    return metrics


def evaluate_platform_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Iterable[str],
) -> dict[str, float]:
    """Compute the fixed-constant MAE platform score used by the competition."""

    y_true_array = _as_2d_array(y_true)
    y_pred_array = np.nan_to_num(
        _as_2d_array(y_pred),
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )
    y_pred_array = np.clip(y_pred_array, -1e6, 1e6)

    metrics: dict[str, float] = {}
    normalized_mae: list[float] = []
    for index, target_name in enumerate(target_names):
        if target_name not in PLATFORM_TARGET_SCALES:
            raise KeyError(f"Missing fixed platform scale for target: {target_name}")
        true_column = y_true_array[:, index]
        pred_column = y_pred_array[:, index]
        mae = float(mean_absolute_error(true_column, pred_column))
        nmae = mae / PLATFORM_TARGET_SCALES[target_name]
        metrics[f"{target_name}__platform_mae"] = mae
        metrics[f"{target_name}__platform_nmae"] = nmae
        normalized_mae.append(nmae)

    metrics["platform_score"] = float(np.mean(normalized_mae))
    return metrics


def evaluate_platform_proxy_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Iterable[str],
    target_scales: dict[str, float],
) -> dict[str, float]:
    """Compute an MAE-based platform-style proxy using fold-local target scales.

    The original platform formula is not stored in this repository, so this proxy
    uses the mean of per-target MAE values normalized by the training-fold IQR.
    This preserves the intended emphasis on MAE while keeping targets comparable.
    """

    y_true_array = _as_2d_array(y_true)
    y_pred_array = np.nan_to_num(
        _as_2d_array(y_pred),
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )
    y_pred_array = np.clip(y_pred_array, -1e6, 1e6)

    metrics: dict[str, float] = {}
    normalized_mae: list[float] = []
    for index, target_name in enumerate(target_names):
        true_column = y_true_array[:, index]
        pred_column = y_pred_array[:, index]
        mae = float(mean_absolute_error(true_column, pred_column))
        nmae = mae / target_scales[target_name]
        metrics[f"{target_name}__platform_mae_proxy"] = mae
        metrics[f"{target_name}__platform_nmae_iqr"] = nmae
        normalized_mae.append(nmae)

    metrics["platform_proxy_score"] = float(np.mean(normalized_mae))
    return metrics
