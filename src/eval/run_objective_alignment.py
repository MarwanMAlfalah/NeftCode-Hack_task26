"""Objective-alignment sprint for the frozen hybrid Deep Sets family-only model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import yeojohnson, yeojohnson_normmax

from src.config import (
    CV_OUTPUTS_DIR,
    OBJECTIVE_ALIGNMENT_REPORT_OUTPUT_PATH,
    OBJECTIVE_ALIGNMENT_RESULTS_OUTPUT_PATH,
    REPORTS_DIR,
)
from src.eval.metrics import evaluate_platform_proxy_predictions, evaluate_regression_predictions
from src.models.train_baselines import OXIDATION_TARGET, TARGET_COLUMNS, VISCOSITY_TARGET
from src.models.train_deep_sets import (
    DeepSetsConfig,
    HybridVariant,
    LossConfig,
    evaluate_single_deep_sets_configuration,
    load_deep_sets_data,
)


@dataclass(frozen=True)
class FittedLocalTargetStrategy:
    name: str
    description: str
    transform: callable
    inverse_transform: callable


@dataclass(frozen=True)
class LocalTargetStrategyFactory:
    name: str
    description: str
    builder: callable

    def fit_from_training_targets(self, y_train: np.ndarray) -> FittedLocalTargetStrategy:
        return self.builder(y_train)


@dataclass(frozen=True)
class ObjectiveAlignmentExperiment:
    experiment_name: str
    target_strategy: object
    loss_config: LossConfig
    checkpoint_metric: str


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=7.5e-4)
    return parser.parse_args()


def _identity_strategy() -> FittedLocalTargetStrategy:
    def _identity(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)

    return FittedLocalTargetStrategy(
        name="raw",
        description="Train both targets on the original scale.",
        transform=_identity,
        inverse_transform=_identity,
    )


def _log1p_signed_strategy() -> FittedLocalTargetStrategy:
    def _transform(values: np.ndarray) -> np.ndarray:
        transformed = np.asarray(values, dtype=float).copy()
        transformed[:, 0] = np.sign(transformed[:, 0]) * np.log1p(np.abs(transformed[:, 0]))
        return transformed

    def _inverse(values: np.ndarray) -> np.ndarray:
        restored = np.asarray(values, dtype=float).copy()
        restored[:, 0] = np.sign(restored[:, 0]) * np.expm1(np.abs(restored[:, 0]))
        return restored

    return FittedLocalTargetStrategy(
        name="log1p_signed_viscosity",
        description="Signed log1p transform on viscosity only.",
        transform=_transform,
        inverse_transform=_inverse,
    )


def _yeo_johnson_inverse(transformed: np.ndarray, lam: float) -> np.ndarray:
    transformed = np.asarray(transformed, dtype=float)
    restored = np.empty_like(transformed, dtype=float)
    nonnegative = transformed >= 0

    if abs(lam) < 1e-8:
        restored[nonnegative] = np.expm1(transformed[nonnegative])
    else:
        restored[nonnegative] = np.power(lam * transformed[nonnegative] + 1.0, 1.0 / lam) - 1.0

    if abs(lam - 2.0) < 1e-8:
        restored[~nonnegative] = 1.0 - np.exp(-transformed[~nonnegative])
    else:
        restored[~nonnegative] = 1.0 - np.power(1.0 - (2.0 - lam) * transformed[~nonnegative], 1.0 / (2.0 - lam))
    return restored


def _build_yeo_johnson_strategy(y_train: np.ndarray) -> FittedLocalTargetStrategy:
    viscosity = np.asarray(y_train, dtype=float)[:, 0]
    lam = float(yeojohnson_normmax(viscosity))

    def _transform(values: np.ndarray) -> np.ndarray:
        transformed = np.asarray(values, dtype=float).copy()
        transformed[:, 0] = yeojohnson(transformed[:, 0], lmbda=lam)
        return transformed

    def _inverse(values: np.ndarray) -> np.ndarray:
        restored = np.asarray(values, dtype=float).copy()
        restored[:, 0] = _yeo_johnson_inverse(restored[:, 0], lam=lam)
        return restored

    return FittedLocalTargetStrategy(
        name="yeo_johnson_viscosity",
        description="Fold-local Yeo-Johnson transform on viscosity only.",
        transform=_transform,
        inverse_transform=_inverse,
    )


def build_stage1_experiments() -> list[ObjectiveAlignmentExperiment]:
    aligned_loss = LossConfig(
        name="huber_huber",
        use_robust_viscosity_loss=True,
        use_robust_oxidation_loss=True,
        viscosity_delta=1.0,
        oxidation_delta=0.75,
    )
    return [
        ObjectiveAlignmentExperiment(
            experiment_name="reference__raw__mse__combined_checkpoint",
            target_strategy=_identity_strategy(),
            loss_config=LossConfig(name="mse_mse", use_robust_viscosity_loss=False, use_robust_oxidation_loss=False),
            checkpoint_metric="combined_score",
        ),
        ObjectiveAlignmentExperiment(
            experiment_name="aligned__raw__huber__platform_proxy_checkpoint",
            target_strategy=_identity_strategy(),
            loss_config=aligned_loss,
            checkpoint_metric="platform_proxy_score",
        ),
        ObjectiveAlignmentExperiment(
            experiment_name="aligned__yeo_johnson_viscosity__huber__platform_proxy_checkpoint",
            target_strategy=LocalTargetStrategyFactory(
                name="yeo_johnson_viscosity",
                description="Fold-local Yeo-Johnson transform on viscosity only.",
                builder=_build_yeo_johnson_strategy,
            ),
            loss_config=aligned_loss,
            checkpoint_metric="platform_proxy_score",
        ),
        ObjectiveAlignmentExperiment(
            experiment_name="aligned__log1p_signed_viscosity__huber__platform_proxy_checkpoint",
            target_strategy=_log1p_signed_strategy(),
            loss_config=aligned_loss,
            checkpoint_metric="platform_proxy_score",
        ),
    ]


def _summarize_experiment(
    experiment_name: str,
    fold_metrics: pd.DataFrame,
    oof_predictions: pd.DataFrame,
) -> pd.DataFrame:
    platform_records: list[dict[str, float]] = []
    for fold_index, fold_frame in oof_predictions.groupby("fold_index", dropna=False):
        proxy_metrics = evaluate_platform_proxy_predictions(
            y_true=fold_frame.loc[:, [f"{VISCOSITY_TARGET}__true", f"{OXIDATION_TARGET}__true"]].to_numpy(dtype=float),
            y_pred=fold_frame.loc[:, [f"{VISCOSITY_TARGET}__pred", f"{OXIDATION_TARGET}__pred"]].to_numpy(dtype=float),
            target_names=TARGET_COLUMNS,
            target_scales={
                VISCOSITY_TARGET: float(fold_frame["viscosity_scale"].iloc[0]),
                OXIDATION_TARGET: float(fold_frame["oxidation_scale"].iloc[0]),
            },
        )
        record = {"fold_index": int(fold_index), **proxy_metrics}
        platform_records.append(record)

    platform_frame = pd.DataFrame.from_records(platform_records)
    merged = fold_metrics.merge(platform_frame, on="fold_index", how="left", validate="one_to_one")
    merged["experiment_name"] = experiment_name

    summary_metrics = {
        "combined_score__mean": float(merged["combined_score"].mean()),
        "combined_score__std": float(merged["combined_score"].std(ddof=1)),
        "platform_proxy_score__mean": float(merged["platform_proxy_score"].mean()),
        "platform_proxy_score__std": float(merged["platform_proxy_score"].std(ddof=1)),
        f"{VISCOSITY_TARGET}__rmse__mean": float(merged[f"{VISCOSITY_TARGET}__rmse"].mean()),
        f"{OXIDATION_TARGET}__rmse__mean": float(merged[f"{OXIDATION_TARGET}__rmse"].mean()),
        f"{VISCOSITY_TARGET}__mae__mean": float(merged[f"{VISCOSITY_TARGET}__mae"].mean()),
        f"{OXIDATION_TARGET}__mae__mean": float(merged[f"{OXIDATION_TARGET}__mae"].mean()),
        f"{VISCOSITY_TARGET}__platform_nmae_iqr__mean": float(merged[f"{VISCOSITY_TARGET}__platform_nmae_iqr"].mean()),
        f"{OXIDATION_TARGET}__platform_nmae_iqr__mean": float(merged[f"{OXIDATION_TARGET}__platform_nmae_iqr"].mean()),
        "best_epoch__mean": float(merged["best_epoch"].mean()),
    }
    metadata_columns = ["model_name", "target_strategy", "best_params_json", "best_inner_cv_score"]
    metadata = merged.iloc[0][metadata_columns].to_dict()
    return pd.DataFrame.from_records([{"experiment_name": experiment_name, **metadata, **summary_metrics}])


def build_report(results: pd.DataFrame) -> str:
    reference = results.loc[results["experiment_name"] == "reference__raw__mse__combined_checkpoint"].iloc[0]
    aligned = results.loc[results["experiment_name"] != "reference__raw__mse__combined_checkpoint"].copy()
    best = aligned.sort_values("platform_proxy_score__mean", ascending=True).iloc[0]
    improvement = float(reference["platform_proxy_score__mean"] - best["platform_proxy_score__mean"])
    relative_improvement = improvement / float(reference["platform_proxy_score__mean"]) if reference["platform_proxy_score__mean"] else 0.0
    meaningful_improvement = improvement > 0.01 and relative_improvement > 0.01

    lines = [
        "# Objective Alignment Report",
        "",
        "## Assumption",
        "- The platform proxy formula was not present in the repository, so this sprint used a local MAE-based proxy: the mean of per-target MAE values normalized by the training-fold IQR.",
        "- The original RMSE-based combined score was kept for logging and comparison.",
        "",
        "## Candidate Comparison",
        "",
        "```text",
        results[
            [
                "experiment_name",
                "target_strategy",
                "combined_score__mean",
                "platform_proxy_score__mean",
                f"{VISCOSITY_TARGET}__rmse__mean",
                f"{OXIDATION_TARGET}__rmse__mean",
                f"{VISCOSITY_TARGET}__mae__mean",
                f"{OXIDATION_TARGET}__mae__mean",
                "best_epoch__mean",
            ]
        ].sort_values("platform_proxy_score__mean").to_string(index=False),
        "```",
        "",
        "## Decision",
        f"- Best target strategy: `{best['target_strategy']}`",
        f"- Best experiment: `{best['experiment_name']}`",
        f"- Platform proxy delta vs current reference: `{improvement:+.4f}`",
        f"- Relative platform proxy improvement vs current reference: `{relative_improvement:+.2%}`",
        f"- Improvement meaningful enough to justify Stage 2: `{'yes' if meaningful_improvement else 'no'}`",
        "",
        "## Notes",
        "- Huber-Huber training was used for the aligned experiments.",
        "- Checkpoint selection for aligned experiments used the MAE-based platform proxy rather than the RMSE-style combined score.",
        "- Yeo-Johnson and signed-log1p were applied to viscosity only.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    CV_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    prepared_data = load_deep_sets_data()
    variant = HybridVariant(name="hybrid_deep_sets_v2_family_only", use_component_embedding=False, use_tabular_branch=True)

    all_results: list[pd.DataFrame] = []
    for experiment in build_stage1_experiments():
        config = DeepSetsConfig(
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            checkpoint_metric=experiment.checkpoint_metric,
        )
        artifacts = evaluate_single_deep_sets_configuration(
            prepared_data=prepared_data,
            config=config,
            variant=variant,
            target_strategy=experiment.target_strategy,
            outer_splits=args.outer_splits,
            seed=42,
            loss_config=experiment.loss_config,
            extra_metadata={
                "experiment_name": experiment.experiment_name,
                "checkpoint_metric": experiment.checkpoint_metric,
                "loss_name": experiment.loss_config.name,
            },
        )
        all_results.append(
            _summarize_experiment(
                experiment_name=experiment.experiment_name,
                fold_metrics=artifacts.fold_metrics,
                oof_predictions=artifacts.oof_predictions,
            )
        )

    results = pd.concat(all_results, ignore_index=True).sort_values("platform_proxy_score__mean").reset_index(drop=True)
    _write_csv(results, OBJECTIVE_ALIGNMENT_RESULTS_OUTPUT_PATH)
    _write_text(build_report(results), OBJECTIVE_ALIGNMENT_REPORT_OUTPUT_PATH)
    print(f"objective_alignment_results: {OBJECTIVE_ALIGNMENT_RESULTS_OUTPUT_PATH}")
    print(f"objective_alignment_report: {OBJECTIVE_ALIGNMENT_REPORT_OUTPUT_PATH}")
    print(f"best_target_strategy: {results.iloc[0]['target_strategy']}")


if __name__ == "__main__":
    main()
