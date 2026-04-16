"""Grouped-CV training utilities for the compact Deep Sets v1 model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from src.config import (
    RANDOM_SEED,
    TEST_JOINED_OUTPUT_PATH,
    TRAIN_JOINED_OUTPUT_PATH,
    TRAIN_TARGET_COLUMNS,
)
from src.eval.metrics import compute_target_scales, evaluate_regression_predictions
from src.models.deep_sets import (
    DeepSetsRegressor,
    DeepSetsSchema,
    FeatureNormalizer,
    ScenarioTensorData,
    TargetScaler,
    build_dataloader,
    build_deep_sets_schema,
    build_scenario_tensor_data,
    set_torch_seed,
)
from src.models.train_baselines import (
    OXIDATION_TARGET,
    TARGET_COLUMNS,
    VISCOSITY_TARGET,
    aggregate_cv_results,
    build_target_strategies,
)


@dataclass(frozen=True)
class DeepSetsConfig:
    """Compact, data-efficient Deep Sets training configuration."""

    batch_size: int = 32
    max_epochs: int = 250
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 35
    min_delta: float = 1e-4
    grad_clip_norm: float = 5.0
    property_projection_dim: int = 16
    family_embedding_dim: int = 8
    component_embedding_dim: int = 8
    element_hidden_dim: int = 64
    condition_hidden_dim: int = 32
    fusion_hidden_dim: int = 64
    dropout: float = 0.10


@dataclass(frozen=True)
class PreparedDeepSetsData:
    """Scenario-level padded tensors plus stable vocab schema."""

    train_data: ScenarioTensorData
    test_data: ScenarioTensorData
    schema: DeepSetsSchema


@dataclass(frozen=True)
class FitArtifacts:
    """Best trained model plus fold-local normalization state."""

    model_state_dict: dict[str, torch.Tensor]
    feature_normalizer: FeatureNormalizer
    target_scaler: TargetScaler
    best_epoch: int
    best_val_loss: float
    train_history: list[dict[str, float]]


def load_deep_sets_data(
    train_path: Path = TRAIN_JOINED_OUTPUT_PATH,
    test_path: Path = TEST_JOINED_OUTPUT_PATH,
) -> PreparedDeepSetsData:
    """Load prepared component rows and convert them into padded scenario tensors."""

    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    schema = build_deep_sets_schema(train_frame=train_frame, test_frame=test_frame)
    train_data = build_scenario_tensor_data(train_frame, schema=schema, include_targets=True)
    test_data = build_scenario_tensor_data(test_frame, schema=schema, include_targets=False)
    return PreparedDeepSetsData(train_data=train_data, test_data=test_data, schema=schema)


def build_model(schema: DeepSetsSchema, config: DeepSetsConfig) -> DeepSetsRegressor:
    """Instantiate the compact Deep Sets regressor from config."""

    return DeepSetsRegressor(
        schema=schema,
        property_projection_dim=config.property_projection_dim,
        family_embedding_dim=config.family_embedding_dim,
        component_embedding_dim=config.component_embedding_dim,
        element_hidden_dim=config.element_hidden_dim,
        condition_hidden_dim=config.condition_hidden_dim,
        fusion_hidden_dim=config.fusion_hidden_dim,
        dropout=config.dropout,
    )


def _select_validation_indices(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split training scenarios into train/early-stop subsets using groups only."""

    unique_groups = np.unique(groups)
    if len(unique_groups) < 4:
        split_point = max(1, len(groups) // 5)
        valid_indices = np.arange(split_point, dtype=int)
        train_indices = np.arange(split_point, len(groups), dtype=int)
        return train_indices, valid_indices

    inner_cv = GroupKFold(n_splits=min(4, len(unique_groups)), shuffle=True, random_state=seed)
    train_indices, valid_indices = next(iter(inner_cv.split(np.zeros(len(groups)), groups=groups)))
    return np.asarray(train_indices, dtype=int), np.asarray(valid_indices, dtype=int)


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a dataloader batch to the requested device."""

    return {name: tensor.to(device) for name, tensor in batch.items()}


def _evaluate_loss(model: DeepSetsRegressor, data: ScenarioTensorData, batch_size: int, device: torch.device) -> float:
    """Compute the mean validation loss on scaled targets."""

    dataloader = build_dataloader(data=data, batch_size=batch_size, shuffle=False)
    losses: list[float] = []
    criterion = torch.nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            predictions = model(batch)
            losses.append(float(criterion(predictions, batch["targets"]).detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("inf")


def fit_deep_sets_model(
    train_data: ScenarioTensorData,
    groups: np.ndarray,
    schema: DeepSetsSchema,
    config: DeepSetsConfig,
    seed: int,
    device: torch.device | None = None,
) -> FitArtifacts:
    """Train Deep Sets with an internal grouped early-stopping split."""

    set_torch_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inner_train_indices, inner_valid_indices = _select_validation_indices(groups=groups, seed=seed)

    raw_inner_train = train_data.subset(inner_train_indices)
    raw_inner_valid = train_data.subset(inner_valid_indices)
    feature_normalizer = FeatureNormalizer.fit(raw_inner_train)
    normalized_inner_train = feature_normalizer.transform(raw_inner_train)
    normalized_inner_valid = feature_normalizer.transform(raw_inner_valid)

    target_scaler = TargetScaler.fit(normalized_inner_train.targets)
    normalized_inner_train = normalized_inner_train.with_targets(
        target_scaler.transform(normalized_inner_train.targets)
    )
    normalized_inner_valid = normalized_inner_valid.with_targets(
        target_scaler.transform(normalized_inner_valid.targets)
    )

    train_loader = build_dataloader(
        data=normalized_inner_train,
        batch_size=config.batch_size,
        shuffle=True,
    )
    model = build_model(schema=schema, config=config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = torch.nn.MSELoss()

    best_state_dict = deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch)
            loss = criterion(predictions, batch["targets"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        val_loss = _evaluate_loss(
            model=model,
            data=normalized_inner_valid,
            batch_size=config.batch_size,
            device=device,
        )
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        if val_loss + config.min_delta < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    return FitArtifacts(
        model_state_dict=best_state_dict,
        feature_normalizer=feature_normalizer,
        target_scaler=target_scaler,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_history=history,
    )


def predict_deep_sets(
    raw_data: ScenarioTensorData,
    schema: DeepSetsSchema,
    config: DeepSetsConfig,
    fit_artifacts: FitArtifacts,
    batch_size: int,
    device: torch.device | None = None,
) -> np.ndarray:
    """Run Deep Sets inference and return predictions on the raw target scale."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalized = fit_artifacts.feature_normalizer.transform(raw_data).with_targets(None)
    dataloader = build_dataloader(data=normalized, batch_size=batch_size, shuffle=False)

    model = build_model(schema=schema, config=config).to(device)
    model.load_state_dict(fit_artifacts.model_state_dict)
    model.eval()

    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(batch).detach().cpu().numpy()
            predictions.append(outputs)

    scaled_predictions = np.vstack(predictions).astype(np.float32)
    return fit_artifacts.target_scaler.inverse_transform(scaled_predictions)


def _serialize_training_metadata(config: DeepSetsConfig, fit_artifacts: FitArtifacts) -> str:
    """Serialize stable model metadata for the fold output table."""

    payload = {
        "config": asdict(config),
        "best_epoch": fit_artifacts.best_epoch,
        "best_val_loss": fit_artifacts.best_val_loss,
    }
    return json.dumps(payload, sort_keys=True)


def _build_error_analysis_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize out-of-fold residual severity for the best Deep Sets run."""

    residuals = predictions.copy()
    residuals["viscosity_abs_error"] = (
        residuals[f"{VISCOSITY_TARGET}__true"] - residuals[f"{VISCOSITY_TARGET}__pred"]
    ).abs()
    residuals["oxidation_abs_error"] = (
        residuals[f"{OXIDATION_TARGET}__true"] - residuals[f"{OXIDATION_TARGET}__pred"]
    ).abs()
    residuals["combined_normalized_abs_error"] = (
        residuals["viscosity_abs_error"] / residuals["viscosity_scale"]
        + residuals["oxidation_abs_error"] / residuals["oxidation_scale"]
    ) / 2.0
    return residuals.sort_values("combined_normalized_abs_error", ascending=False)


def _load_best_tabular_reference(
    path: Path = Path("outputs/cv/baseline_ablation_results.csv"),
) -> dict[str, object]:
    """Load the current best tabular comparison row used in the Deep Sets report."""

    if not path.exists():
        return {
            "feature_setting": "conditions_structure_family",
            "combined_score__mean": 1.7163,
            "target_delta_kinematic_viscosity_pct__rmse__mean": np.nan,
            "target_oxidation_eot_a_per_cm__rmse__mean": np.nan,
        }

    frame = pd.read_csv(path)
    best_row = frame.sort_values("combined_score__mean", ascending=True).iloc[0]
    return best_row.to_dict()


def build_deep_sets_report(
    summary_results: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    best_predictions: pd.DataFrame,
    schema: DeepSetsSchema,
    best_tabular_reference: dict[str, object],
) -> str:
    """Create a concise markdown report comparing Deep Sets against the tabular bar."""

    best_row = summary_results.iloc[0]
    tabular_score = float(best_tabular_reference["combined_score__mean"])
    delta_vs_tabular = float(best_row["combined_score__mean"]) - tabular_score

    comparison_columns = [
        "rank_combined_score",
        "model_name",
        "target_strategy",
        "combined_score__mean",
        f"{VISCOSITY_TARGET}__rmse__mean",
        f"{OXIDATION_TARGET}__rmse__mean",
        f"{VISCOSITY_TARGET}__mae__mean",
        f"{OXIDATION_TARGET}__mae__mean",
    ]

    transform_lines: list[str] = []
    if set(summary_results["target_strategy"]) >= {"raw", "viscosity_asinh"}:
        transform_comparison = summary_results.pivot(
            index="model_name",
            columns="target_strategy",
            values=["combined_score__mean", f"{VISCOSITY_TARGET}__rmse__mean"],
        )
        for model_name in sorted(summary_results["model_name"].unique()):
            raw_score = transform_comparison.get(("combined_score__mean", "raw"))
            asinh_score = transform_comparison.get(("combined_score__mean", "viscosity_asinh"))
            raw_rmse = transform_comparison.get((f"{VISCOSITY_TARGET}__rmse__mean", "raw"))
            asinh_rmse = transform_comparison.get((f"{VISCOSITY_TARGET}__rmse__mean", "viscosity_asinh"))
            if raw_score is None or asinh_score is None or raw_rmse is None or asinh_rmse is None:
                continue
            transform_lines.append(
                f"- `{model_name}`: combined score delta `{asinh_score.loc[model_name] - raw_score.loc[model_name]:+.4f}`, "
                f"viscosity RMSE delta `{asinh_rmse.loc[model_name] - raw_rmse.loc[model_name]:+.4f}`"
            )

    hardest_cases = _build_error_analysis_table(best_predictions).head(5)
    hardest_case_lines = [
        (
            f"- `{row['scenario_id']}`: combined normalized abs error `{row['combined_normalized_abs_error']:.3f}`, "
            f"viscosity true/pred `{row[f'{VISCOSITY_TARGET}__true']:.2f}` / `{row[f'{VISCOSITY_TARGET}__pred']:.2f}`, "
            f"oxidation true/pred `{row[f'{OXIDATION_TARGET}__true']:.2f}` / `{row[f'{OXIDATION_TARGET}__pred']:.2f}`"
        )
        for row in hardest_cases.to_dict(orient="records")
    ]

    fold_dispersion = fold_metrics.groupby(["model_name", "target_strategy"])[
        ["combined_score", f"{VISCOSITY_TARGET}__rmse", f"{OXIDATION_TARGET}__rmse"]
    ].std()

    lines = [
        "# Deep Sets CV Report",
        "",
        "## Data Snapshot",
        f"- Train scenarios: `{len(best_predictions['scenario_id'].unique())}`",
        f"- Max components per scenario: `{schema.max_components}`",
        f"- Property columns per component: `{len(schema.property_columns)}`",
        (
            f"- Learned vocab sizes: families `{schema.family_vocab_size}`, "
            f"components `{schema.component_vocab_size}`, catalysts `{schema.catalyst_vocab_size}`"
        ),
        "",
        "## Deep Sets Comparison",
        "",
        "```text",
        summary_results.loc[:, comparison_columns].to_string(index=False),
        "```",
        "",
        "## Best Deep Sets v1",
        (
            f"- Best configuration: `{best_row['model_name']}` with target strategy `{best_row['target_strategy']}` "
            f"and mean combined score `{best_row['combined_score__mean']:.4f}`"
        ),
        (
            f"- Mean RMSEs: viscosity `{best_row[f'{VISCOSITY_TARGET}__rmse__mean']:.4f}`, "
            f"oxidation `{best_row[f'{OXIDATION_TARGET}__rmse__mean']:.4f}`"
        ),
        "",
        "## Comparison To Current Best Tabular Baseline",
        (
            f"- Tabular reference: `{best_tabular_reference['feature_setting']}` with mean combined score "
            f"`{tabular_score:.4f}`"
        ),
        (
            f"- Deep Sets delta vs tabular: `{delta_vs_tabular:+.4f}` "
            f"({'better' if delta_vs_tabular < 0 else 'worse'})"
        ),
        "",
        "## Input Design",
        "- One set element per component row with family embedding, component embedding, standardized mass fraction, compressed property vector, property mask, and coverage/source flags.",
        "- Scenario condition branch uses temperature, time, biofuel mass fraction, and catalyst category embedding.",
        "- Pooling is permutation-invariant mean plus max before fusion into a 2-target regression head.",
        "",
        "## Viscosity Transform Effect",
    ]

    if transform_lines:
        lines.extend(transform_lines)
    else:
        lines.append("- Only one target strategy was available.")

    lines.extend(
        [
            "",
            "## Hardest Out-of-Fold Scenarios",
            *hardest_case_lines,
            "",
            "## Risks",
            "- The dataset is small for a neural model, so fold variance remains meaningful even with the compact architecture.",
            "- Component identity embeddings can overfit rare components; the family path is likely the more stable generalization route.",
            "- Missing numeric properties are handled by masks and flags, but the model still sees zero-filled values behind those masks.",
        ]
    )

    if not fold_dispersion.empty:
        volatile = fold_dispersion.sort_values("combined_score", ascending=False).head(3)
        lines.extend(["", "## Fold Volatility"])
        for (model_name, target_strategy), row in volatile.iterrows():
            lines.append(
                f"- `{model_name}` / `{target_strategy}`: combined score std `{row['combined_score']:.4f}`, "
                f"viscosity RMSE std `{row[f'{VISCOSITY_TARGET}__rmse']:.4f}`, "
                f"oxidation RMSE std `{row[f'{OXIDATION_TARGET}__rmse']:.4f}`"
            )

    return "\n".join(lines) + "\n"


def run_deep_sets_cv(
    prepared_data: PreparedDeepSetsData,
    config: DeepSetsConfig | None = None,
    outer_splits: int = 5,
    seed: int = RANDOM_SEED,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Run grouped CV for Deep Sets across the supported target strategies."""

    config = config or DeepSetsConfig()
    target_strategies = build_target_strategies()
    train_data = prepared_data.train_data
    groups = np.asarray(train_data.scenario_ids, dtype=object)
    outer_cv = GroupKFold(n_splits=outer_splits, shuffle=True, random_state=seed)

    fold_records: list[dict[str, object]] = []
    prediction_records: list[dict[str, object]] = []

    for fold_index, (train_index, valid_index) in enumerate(
        outer_cv.split(np.zeros(len(groups)), groups=groups),
        start=1,
    ):
        raw_train_fold = train_data.subset(train_index)
        raw_valid_fold = train_data.subset(valid_index)
        y_train_raw = raw_train_fold.targets.astype(np.float32)
        y_valid_raw = raw_valid_fold.targets.astype(np.float32)
        target_scales = compute_target_scales(y_train_raw, TARGET_COLUMNS)

        for target_strategy in target_strategies:
            transformed_train_fold = raw_train_fold.with_targets(
                target_strategy.transform(y_train_raw).astype(np.float32)
            )
            transformed_valid_fold = raw_valid_fold.with_targets(
                target_strategy.transform(y_valid_raw).astype(np.float32)
            )

            start_time = time.perf_counter()
            fit_artifacts = fit_deep_sets_model(
                train_data=transformed_train_fold,
                groups=groups[train_index],
                schema=prepared_data.schema,
                config=config,
                seed=seed + fold_index,
                device=device,
            )
            fit_time = time.perf_counter() - start_time

            valid_predictions_transformed = predict_deep_sets(
                raw_data=transformed_valid_fold,
                schema=prepared_data.schema,
                config=config,
                fit_artifacts=fit_artifacts,
                batch_size=config.batch_size,
                device=device,
            )
            valid_predictions_raw = target_strategy.inverse_transform(valid_predictions_transformed)
            metrics = evaluate_regression_predictions(
                y_true=y_valid_raw,
                y_pred=valid_predictions_raw,
                target_names=TARGET_COLUMNS,
                target_scales=target_scales,
            )

            fold_record = {
                "fold_index": fold_index,
                "model_name": "deep_sets_v1",
                "target_strategy": target_strategy.name,
                "n_train": len(train_index),
                "n_valid": len(valid_index),
                "fit_time_seconds": fit_time,
                "best_inner_cv_score": fit_artifacts.best_val_loss,
                "best_params_json": _serialize_training_metadata(config=config, fit_artifacts=fit_artifacts),
                "best_epoch": fit_artifacts.best_epoch,
                "viscosity_scale": target_scales[VISCOSITY_TARGET],
                "oxidation_scale": target_scales[OXIDATION_TARGET],
            }
            fold_record.update(metrics)
            fold_records.append(fold_record)

            for row_offset, scenario_id in enumerate(raw_valid_fold.scenario_ids):
                prediction_records.append(
                    {
                        "fold_index": fold_index,
                        "model_name": "deep_sets_v1",
                        "target_strategy": target_strategy.name,
                        "scenario_id": scenario_id,
                        f"{VISCOSITY_TARGET}__true": y_valid_raw[row_offset, 0],
                        f"{VISCOSITY_TARGET}__pred": valid_predictions_raw[row_offset, 0],
                        f"{OXIDATION_TARGET}__true": y_valid_raw[row_offset, 1],
                        f"{OXIDATION_TARGET}__pred": valid_predictions_raw[row_offset, 1],
                        "viscosity_scale": target_scales[VISCOSITY_TARGET],
                        "oxidation_scale": target_scales[OXIDATION_TARGET],
                    }
                )

    fold_metrics = pd.DataFrame.from_records(fold_records)
    summary_results = aggregate_cv_results(fold_metrics)
    best_row = summary_results.iloc[0]
    oof_predictions = pd.DataFrame.from_records(prediction_records)
    best_predictions = oof_predictions.loc[
        (oof_predictions["model_name"] == best_row["model_name"])
        & (oof_predictions["target_strategy"] == best_row["target_strategy"])
    ].copy()
    report = build_deep_sets_report(
        summary_results=summary_results,
        fold_metrics=fold_metrics,
        best_predictions=best_predictions,
        schema=prepared_data.schema,
        best_tabular_reference=_load_best_tabular_reference(),
    )
    return summary_results, fold_metrics, report
