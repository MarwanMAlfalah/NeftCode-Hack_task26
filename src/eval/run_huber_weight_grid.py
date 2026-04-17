"""Medium-horizon Huber-delta and light-weight tuning for the current meta-stack lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from package_submission import build_bundle_from_predictions, validate_predictions_csv
from src.config import (
    CV_OUTPUTS_DIR,
    HUBER_WEIGHT_GRID_REPORT_OUTPUT_PATH,
    HUBER_WEIGHT_GRID_RESULTS_OUTPUT_PATH,
    OUTPUTS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
)
from src.eval.metrics import evaluate_platform_predictions
from src.eval.run_gp_ensemble_stage2 import (
    _build_stack_predictions,
    _fit_final_stack_models,
    _fit_full_gp_predictions,
    _predict_with_final_stack_models,
    _to_internal_columns,
    _to_submission_columns,
)
from src.models.train_baselines import (
    OXIDATION_TARGET,
    TARGET_COLUMNS,
    VISCOSITY_TARGET,
    load_baseline_training_data,
    select_baseline_feature_columns,
)
from src.models.train_deep_sets import (
    DeepSetsConfig,
    HybridVariant,
    LossConfig,
    evaluate_single_deep_sets_configuration,
    get_target_strategy_by_name,
    load_deep_sets_data,
    train_full_deep_sets_variant_ensemble_and_predict,
)


LIVE_PLATFORM_ANCHOR_SCORE = 0.104084
META_RESULTS_PATH = REPO_ROOT / "outputs" / "cv" / "meta_stack_search_results.csv"
GP_STAGE2_OOF_PATH = REPO_ROOT / "outputs" / "cv" / "gp_stage2_oof_predictions.csv"
WEIGHT_SCHEMES = ["none", "visc_tail_q90", "ox_hard_q75", "joint_light"]
VISCOSITY_DELTAS = [0.75, 1.0, 1.25, 1.5]
OXIDATION_DELTAS = [0.5, 0.75, 1.0]
ENSEMBLE_SEEDS = [0, 1, 2, 3, 4]
PACKAGE_PREFIX = "neftekod_dot_submission_huber_weight_grid"
VALIDATED_FEATURE_SETTING = "conditions_structure_family"
VALIDATED_FEATURE_COUNT = 77


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=7.5e-4)
    return parser.parse_args()


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _build_variant() -> HybridVariant:
    return HybridVariant(
        name="hybrid_deep_sets_v2_family_only",
        use_component_embedding=False,
        use_tabular_branch=True,
    )


def _build_config(args: argparse.Namespace) -> DeepSetsConfig:
    return DeepSetsConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        checkpoint_metric="platform_score",
    )


def _load_live_anchor_row() -> pd.Series:
    results = pd.read_csv(META_RESULTS_PATH)
    if results.empty:
        raise ValueError(f"Missing meta-stack search results: {META_RESULTS_PATH}")
    if "rank_platform_score" in results.columns:
        ranked = results.sort_values(
            ["rank_platform_score", "platform_score__mean"],
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        ranked = results.sort_values("platform_score__mean", kind="mergesort").reset_index(drop=True)
    return ranked.iloc[0].copy()


def _load_validated_feature_columns(prepared_baseline) -> list[str]:
    feature_columns = select_baseline_feature_columns(prepared_baseline, VALIDATED_FEATURE_SETTING)
    if len(feature_columns) != VALIDATED_FEATURE_COUNT:
        raise ValueError(
            "The validated GP regime drifted away from the expected 77-feature setting: "
            f"{VALIDATED_FEATURE_SETTING} produced {len(feature_columns)} columns."
        )
    return feature_columns


def _load_gp_oof_frame() -> pd.DataFrame:
    frame = pd.read_csv(GP_STAGE2_OOF_PATH)
    required_columns = {
        "scenario_id",
        "fold_index",
        f"{VISCOSITY_TARGET}__true",
        f"{OXIDATION_TARGET}__true",
        "viscosity_scale",
        "oxidation_scale",
        "gp_matern_white_viscosity_pred",
        "gp_matern_white_oxidation_pred",
        "gp_matern_white_dot_viscosity_pred",
        "gp_matern_white_dot_oxidation_pred",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required GP OOF columns: {missing}")
    return frame.sort_values(["fold_index", "scenario_id"]).reset_index(drop=True)


def _effective_weights(weights_json: str) -> dict[str, float]:
    weights = json.loads(weights_json)
    return {
        source_name: float(weight)
        for source_name, weight in weights.items()
        if abs(float(weight)) > 1e-6
    }


def _evaluate_candidate(
    frame: pd.DataFrame,
    candidate_name: str,
    viscosity_pred: np.ndarray,
    oxidation_pred: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    fold_records: list[dict[str, object]] = []
    prediction_frame = frame.loc[
        :,
        [
            "scenario_id",
            "fold_index",
            "viscosity_scale",
            "oxidation_scale",
            f"{VISCOSITY_TARGET}__true",
            f"{OXIDATION_TARGET}__true",
        ],
    ].copy()
    prediction_frame[f"{VISCOSITY_TARGET}__pred"] = np.asarray(viscosity_pred, dtype=float)
    prediction_frame[f"{OXIDATION_TARGET}__pred"] = np.asarray(oxidation_pred, dtype=float)

    for fold_index, fold_frame in prediction_frame.groupby("fold_index", dropna=False):
        platform_metrics = evaluate_platform_predictions(
            y_true=fold_frame.loc[:, [f"{VISCOSITY_TARGET}__true", f"{OXIDATION_TARGET}__true"]].to_numpy(dtype=float),
            y_pred=fold_frame.loc[:, [f"{VISCOSITY_TARGET}__pred", f"{OXIDATION_TARGET}__pred"]].to_numpy(dtype=float),
            target_names=TARGET_COLUMNS,
        )
        fold_records.append(
            {
                "candidate_name": candidate_name,
                "fold_index": int(fold_index),
                f"{VISCOSITY_TARGET}__mae": float(platform_metrics[f"{VISCOSITY_TARGET}__platform_mae"]),
                f"{OXIDATION_TARGET}__mae": float(platform_metrics[f"{OXIDATION_TARGET}__platform_mae"]),
                f"{VISCOSITY_TARGET}__nmae": float(platform_metrics[f"{VISCOSITY_TARGET}__platform_nmae"]),
                f"{OXIDATION_TARGET}__nmae": float(platform_metrics[f"{OXIDATION_TARGET}__platform_nmae"]),
                "platform_score": float(platform_metrics["platform_score"]),
            }
        )

    fold_metrics = pd.DataFrame.from_records(fold_records).sort_values("fold_index").reset_index(drop=True)
    summary = {
        "candidate_name": candidate_name,
        "platform_score__mean": float(fold_metrics["platform_score"].mean()),
        "platform_score__std": float(fold_metrics["platform_score"].std(ddof=1)),
        f"{VISCOSITY_TARGET}__mae__mean": float(fold_metrics[f"{VISCOSITY_TARGET}__mae"].mean()),
        f"{OXIDATION_TARGET}__mae__mean": float(fold_metrics[f"{OXIDATION_TARGET}__mae"].mean()),
        f"{VISCOSITY_TARGET}__nmae__mean": float(fold_metrics[f"{VISCOSITY_TARGET}__nmae"].mean()),
        f"{OXIDATION_TARGET}__nmae__mean": float(fold_metrics[f"{OXIDATION_TARGET}__nmae"].mean()),
    }
    return fold_metrics, summary


def _build_tuned_deep_oof(
    args: argparse.Namespace,
    loss_config: LossConfig,
) -> pd.DataFrame:
    prepared_data = load_deep_sets_data()
    artifacts = evaluate_single_deep_sets_configuration(
        prepared_data=prepared_data,
        config=_build_config(args),
        variant=_build_variant(),
        target_strategy=get_target_strategy_by_name("raw"),
        outer_splits=args.outer_splits,
        seed=RANDOM_SEED,
        loss_config=loss_config,
        extra_metadata={
            "checkpoint_metric": "platform_score",
            "loss_name": loss_config.name,
            "sample_weight_scheme": loss_config.sample_weight_scheme,
        },
    )
    return artifacts.oof_predictions.rename(
        columns={
            f"{VISCOSITY_TARGET}__pred": "deep_sets_viscosity_pred",
            f"{OXIDATION_TARGET}__pred": "deep_sets_oxidation_pred",
        }
    ).loc[
        :,
        [
            "scenario_id",
            "fold_index",
            "deep_sets_viscosity_pred",
            "deep_sets_oxidation_pred",
        ],
    ].sort_values(["fold_index", "scenario_id"]).reset_index(drop=True)


def _build_meta_lineage_oof(
    args: argparse.Namespace,
    loss_config: LossConfig,
    anchor_row: pd.Series,
    gp_oof: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    deep_oof = _build_tuned_deep_oof(args=args, loss_config=loss_config)
    merged = gp_oof.merge(
        deep_oof,
        on=["scenario_id", "fold_index"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["fold_index", "scenario_id"]).reset_index(drop=True)
    if len(merged) != len(gp_oof):
        raise ValueError("Deep OOF merge changed row count for the tuning sweep.")

    current_stack_input = merged.rename(
        columns={
            "gp_matern_white_viscosity_pred": "gp_viscosity_pred",
            "gp_matern_white_oxidation_pred": "gp_oxidation_pred",
        }
    )
    current_stack_viscosity, current_stack_oxidation = _build_stack_predictions(
        current_stack_input.loc[
            :,
            [
                "scenario_id",
                "fold_index",
                "deep_sets_viscosity_pred",
                "deep_sets_oxidation_pred",
                "gp_viscosity_pred",
                "gp_oxidation_pred",
                f"{VISCOSITY_TARGET}__true",
                f"{OXIDATION_TARGET}__true",
                "viscosity_scale",
                "oxidation_scale",
            ],
        ]
    )
    stack_dot_input = merged.rename(
        columns={
            "gp_matern_white_dot_viscosity_pred": "gp_viscosity_pred",
            "gp_matern_white_dot_oxidation_pred": "gp_oxidation_pred",
        }
    )
    stack_dot_viscosity, stack_dot_oxidation = _build_stack_predictions(
        stack_dot_input.loc[
            :,
            [
                "scenario_id",
                "fold_index",
                "deep_sets_viscosity_pred",
                "deep_sets_oxidation_pred",
                "gp_viscosity_pred",
                "gp_oxidation_pred",
                f"{VISCOSITY_TARGET}__true",
                f"{OXIDATION_TARGET}__true",
                "viscosity_scale",
                "oxidation_scale",
            ],
        ]
    )

    viscosity_weights = _effective_weights(str(anchor_row["viscosity_weights_json"]))
    oxidation_weights = _effective_weights(str(anchor_row["oxidation_weights_json"]))

    source_predictions = {
        "stage15": {
            VISCOSITY_TARGET: merged["deep_sets_viscosity_pred"].to_numpy(dtype=float),
            OXIDATION_TARGET: merged["deep_sets_oxidation_pred"].to_numpy(dtype=float),
        },
        "gp_matern_white": {
            VISCOSITY_TARGET: merged["gp_matern_white_viscosity_pred"].to_numpy(dtype=float),
            OXIDATION_TARGET: merged["gp_matern_white_oxidation_pred"].to_numpy(dtype=float),
        },
        "gp_matern_white_dot": {
            VISCOSITY_TARGET: merged["gp_matern_white_dot_viscosity_pred"].to_numpy(dtype=float),
            OXIDATION_TARGET: merged["gp_matern_white_dot_oxidation_pred"].to_numpy(dtype=float),
        },
        "current_stack": {
            VISCOSITY_TARGET: current_stack_viscosity,
            OXIDATION_TARGET: current_stack_oxidation,
        },
        "stack_matern_white_dot": {
            VISCOSITY_TARGET: stack_dot_viscosity,
            OXIDATION_TARGET: stack_dot_oxidation,
        },
    }

    tuned_viscosity = np.zeros(len(merged), dtype=float)
    tuned_oxidation = np.zeros(len(merged), dtype=float)
    for source_name, weight in viscosity_weights.items():
        tuned_viscosity += float(weight) * source_predictions[source_name][VISCOSITY_TARGET]
    for source_name, weight in oxidation_weights.items():
        tuned_oxidation += float(weight) * source_predictions[source_name][OXIDATION_TARGET]

    lineage_frame = merged.loc[
        :,
        [
            "scenario_id",
            "fold_index",
            "viscosity_scale",
            "oxidation_scale",
            f"{VISCOSITY_TARGET}__true",
            f"{OXIDATION_TARGET}__true",
        ],
    ].copy()
    lineage_frame["tuned_viscosity_pred"] = tuned_viscosity
    lineage_frame["tuned_oxidation_pred"] = tuned_oxidation
    lineage_frame["current_stack_viscosity_pred"] = current_stack_viscosity
    lineage_frame["current_stack_oxidation_pred"] = current_stack_oxidation
    lineage_frame["stack_dot_viscosity_pred"] = stack_dot_viscosity
    lineage_frame["stack_dot_oxidation_pred"] = stack_dot_oxidation
    return lineage_frame, {
        "deep_oof": deep_oof,
        "current_stack_input": current_stack_input,
        "stack_dot_input": stack_dot_input,
        "viscosity_weights": viscosity_weights,
        "oxidation_weights": oxidation_weights,
    }


def _package_candidate(
    args: argparse.Namespace,
    loss_config: LossConfig,
    anchor_row: pd.Series,
    lineage_artifacts: dict[str, object],
) -> dict[str, str]:
    prepared_baseline = load_baseline_training_data()
    feature_columns = _load_validated_feature_columns(prepared_baseline)
    gp_matern_white_test = _fit_full_gp_predictions(
        prepared_data=prepared_baseline,
        feature_columns=feature_columns,
        kernel_name="matern_white",
    )
    gp_matern_white_dot_test = _fit_full_gp_predictions(
        prepared_data=prepared_baseline,
        feature_columns=feature_columns,
        kernel_name="matern_white_dot",
    )

    prepared_deep = load_deep_sets_data()
    deep_test = _to_internal_columns(
        train_full_deep_sets_variant_ensemble_and_predict(
            prepared_data=prepared_deep,
            variant=_build_variant(),
            target_strategy_name="raw",
            seeds=ENSEMBLE_SEEDS,
            config=_build_config(args),
            loss_config=loss_config,
        )
    )

    current_stack_models = _fit_final_stack_models(
        lineage_artifacts["current_stack_input"].loc[
            :,
            [
                "deep_sets_viscosity_pred",
                "deep_sets_oxidation_pred",
                "gp_viscosity_pred",
                "gp_oxidation_pred",
                "scenario_id",
                f"{VISCOSITY_TARGET}__true",
                f"{OXIDATION_TARGET}__true",
            ],
        ]
    )
    stack_dot_models = _fit_final_stack_models(
        lineage_artifacts["stack_dot_input"].loc[
            :,
            [
                "deep_sets_viscosity_pred",
                "deep_sets_oxidation_pred",
                "gp_viscosity_pred",
                "gp_oxidation_pred",
                "scenario_id",
                f"{VISCOSITY_TARGET}__true",
                f"{OXIDATION_TARGET}__true",
            ],
        ]
    )

    current_stack_test = _predict_with_final_stack_models(
        deep_frame=deep_test.rename(
            columns={
                TARGET_COLUMNS[0]: "deep_sets_viscosity_pred",
                TARGET_COLUMNS[1]: "deep_sets_oxidation_pred",
            }
        ),
        gp_frame=gp_matern_white_test.rename(
            columns={
                TARGET_COLUMNS[0]: "gp_viscosity_pred",
                TARGET_COLUMNS[1]: "gp_oxidation_pred",
            }
        ),
        models=current_stack_models,
    )
    stack_dot_test = _predict_with_final_stack_models(
        deep_frame=deep_test.rename(
            columns={
                TARGET_COLUMNS[0]: "deep_sets_viscosity_pred",
                TARGET_COLUMNS[1]: "deep_sets_oxidation_pred",
            }
        ),
        gp_frame=gp_matern_white_dot_test.rename(
            columns={
                TARGET_COLUMNS[0]: "gp_viscosity_pred",
                TARGET_COLUMNS[1]: "gp_oxidation_pred",
            }
        ),
        models=stack_dot_models,
    )

    source_predictions = {
        "stage15": deep_test,
        "gp_matern_white": gp_matern_white_test,
        "gp_matern_white_dot": gp_matern_white_dot_test,
        "current_stack": current_stack_test,
        "stack_matern_white_dot": stack_dot_test,
    }
    viscosity_weights = lineage_artifacts["viscosity_weights"]
    oxidation_weights = lineage_artifacts["oxidation_weights"]

    scenario_ids = deep_test["scenario_id"].to_numpy()
    final_viscosity = np.zeros(len(scenario_ids), dtype=float)
    final_oxidation = np.zeros(len(scenario_ids), dtype=float)
    for source_name, weight in viscosity_weights.items():
        final_viscosity += float(weight) * source_predictions[source_name][TARGET_COLUMNS[0]].to_numpy(dtype=float)
    for source_name, weight in oxidation_weights.items():
        final_oxidation += float(weight) * source_predictions[source_name][TARGET_COLUMNS[1]].to_numpy(dtype=float)

    final_predictions = pd.DataFrame(
        {
            "scenario_id": scenario_ids,
            TARGET_COLUMNS[0]: final_viscosity,
            TARGET_COLUMNS[1]: final_oxidation,
        }
    ).sort_values("scenario_id").reset_index(drop=True)
    combo_slug = (
        f"vd{int(round(loss_config.viscosity_delta * 100)):03d}"
        f"_od{int(round(loss_config.oxidation_delta * 100)):03d}"
        f"_{loss_config.sample_weight_scheme}"
    )
    candidate_stem = f"{PACKAGE_PREFIX}_{combo_slug}"
    candidate_dir = OUTPUTS_DIR / "submissions" / candidate_stem
    predictions_path = candidate_dir / "predictions.csv"
    zip_name = f"{candidate_stem}.zip"

    candidate_dir.mkdir(parents=True, exist_ok=True)
    submission_frame = _to_submission_columns(final_predictions)
    submission_frame.to_csv(predictions_path, index=False, encoding="utf-8")
    validate_predictions_csv(predictions_path)
    zip_path = build_bundle_from_predictions(predictions_path=predictions_path, zip_name=zip_name)
    return {
        "predictions_path": str(predictions_path),
        "zip_path": str(zip_path),
    }


def _is_strongly_stable(candidate_row: pd.Series, anchor_summary: dict[str, float]) -> bool:
    return (
        float(candidate_row["fold_improvement_rate"]) >= 0.80
        and float(candidate_row["platform_score__std"]) <= float(anchor_summary["platform_score__std"])
        and not bool(candidate_row["oxidation_regression"])
    )


def _build_report(
    results: pd.DataFrame,
    anchor_summary: dict[str, float],
    best_row: pd.Series,
    packaged_paths: dict[str, str] | None,
) -> str:
    lines = [
        "# Huber Weight Grid Report",
        "",
        "## Scope",
        "- Stayed on the current Stage 1.5 / meta-stack family only.",
        "- Kept raw targets, the validated 77-feature GP regime, and the current live meta-lineage blend weights fixed.",
        f"- Current best live platform anchor kept untouched: `{LIVE_PLATFORM_ANCHOR_SCORE:.6f}`",
        f"- Current local meta-family anchor lineage score: `{anchor_summary['platform_score__mean']:.6f}`",
        "",
        "## Candidate Ranking",
        "",
        "```text",
        results.head(12)[
            [
                "rank_platform_score",
                "viscosity_delta",
                "oxidation_delta",
                "sample_weight_scheme",
                "platform_score__mean",
                "local_gain_vs_meta_anchor",
                f"{VISCOSITY_TARGET}__mae__mean",
                f"{OXIDATION_TARGET}__mae__mean",
                "fold_improvement_rate",
                "oxidation_regression",
                "strong_stability_evidence",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Best Setting",
        f"- Best delta/weight combination: viscosity delta `{best_row['viscosity_delta']}`, oxidation delta `{best_row['oxidation_delta']}`, sample weight `{best_row['sample_weight_scheme']}`",
        f"- Local tuned-lineage score: `{best_row['platform_score__mean']:.6f}`",
        f"- Local gain vs current meta-family anchor: `{best_row['local_gain_vs_meta_anchor']:+.6f}`",
        f"- Fold improvement rate vs current meta-family anchor: `{best_row['fold_improvement_rate']:.2%}`",
        f"- Oxidation regression vs current meta-family anchor: `{'yes' if best_row['oxidation_regression'] else 'no'}`",
        f"- Strong stability evidence: `{'yes' if best_row['strong_stability_evidence'] else 'no'}`",
        f"- New platform attempt justified: `{'yes' if best_row['package_recommended'] else 'no'}`",
    ]
    if packaged_paths is not None:
        lines.extend(
            [
                f"- Packaged predictions path: `{packaged_paths['predictions_path']}`",
                f"- Packaged ZIP path: `{packaged_paths['zip_path']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    CV_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    anchor_row = _load_live_anchor_row()
    gp_oof = _load_gp_oof_frame()
    anchor_viscosity = gp_oof["best_meta_viscosity_pred"].to_numpy(dtype=float) if "best_meta_viscosity_pred" in gp_oof.columns else None
    anchor_oxidation = gp_oof["best_meta_oxidation_pred"].to_numpy(dtype=float) if "best_meta_oxidation_pred" in gp_oof.columns else None
    if anchor_viscosity is None or anchor_oxidation is None:
        raise ValueError("The GP Stage 2 OOF artifact is missing the current best meta predictions.")
    anchor_fold_metrics, anchor_summary = _evaluate_candidate(
        frame=gp_oof,
        candidate_name="current_meta_family_anchor",
        viscosity_pred=anchor_viscosity,
        oxidation_pred=anchor_oxidation,
    )

    result_rows: list[dict[str, object]] = []
    best_lineage_artifacts: dict[str, object] | None = None

    for viscosity_delta in VISCOSITY_DELTAS:
        for oxidation_delta in OXIDATION_DELTAS:
            for sample_weight_scheme in WEIGHT_SCHEMES:
                loss_config = LossConfig(
                    name=f"huber_huber__vd{viscosity_delta:.2f}__od{oxidation_delta:.2f}__{sample_weight_scheme}",
                    use_robust_viscosity_loss=True,
                    use_robust_oxidation_loss=True,
                    viscosity_delta=viscosity_delta,
                    oxidation_delta=oxidation_delta,
                    sample_weight_scheme=sample_weight_scheme,
                )
                lineage_frame, lineage_artifacts = _build_meta_lineage_oof(
                    args=args,
                    loss_config=loss_config,
                    anchor_row=anchor_row,
                    gp_oof=gp_oof,
                )
                fold_metrics, summary = _evaluate_candidate(
                    frame=lineage_frame,
                    candidate_name=loss_config.name,
                    viscosity_pred=lineage_frame["tuned_viscosity_pred"].to_numpy(dtype=float),
                    oxidation_pred=lineage_frame["tuned_oxidation_pred"].to_numpy(dtype=float),
                )
                local_gain = float(anchor_summary["platform_score__mean"] - summary["platform_score__mean"])
                fold_improvement_rate = float(
                    np.mean(
                        fold_metrics["platform_score"].to_numpy(dtype=float)
                        < anchor_fold_metrics["platform_score"].to_numpy(dtype=float)
                    )
                )
                oxidation_regression = (
                    float(summary[f"{OXIDATION_TARGET}__mae__mean"])
                    > float(anchor_summary[f"{OXIDATION_TARGET}__mae__mean"])
                )
                candidate_row = {
                    "viscosity_delta": viscosity_delta,
                    "oxidation_delta": oxidation_delta,
                    "sample_weight_scheme": sample_weight_scheme,
                    **summary,
                    "local_gain_vs_meta_anchor": local_gain,
                    "fold_improvement_rate": fold_improvement_rate,
                    "oxidation_regression": oxidation_regression,
                }
                candidate_row["strong_stability_evidence"] = _is_strongly_stable(
                    candidate_row,
                    anchor_summary=anchor_summary,
                )
                candidate_row["package_recommended"] = (
                    local_gain >= 0.0020
                    or (
                        0.0010 <= local_gain < 0.0020
                        and candidate_row["strong_stability_evidence"]
                        and not oxidation_regression
                    )
                )
                result_rows.append(candidate_row)
                if best_lineage_artifacts is None or summary["platform_score__mean"] < best_lineage_artifacts["summary"]["platform_score__mean"]:
                    best_lineage_artifacts = {
                        "loss_config": loss_config,
                        "lineage_artifacts": lineage_artifacts,
                        "summary": summary,
                    }

    results = pd.DataFrame.from_records(result_rows).sort_values(
        [
            "platform_score__mean",
            f"{VISCOSITY_TARGET}__mae__mean",
            f"{OXIDATION_TARGET}__mae__mean",
        ]
    ).reset_index(drop=True)
    results["rank_platform_score"] = np.arange(1, len(results) + 1)
    best_row = results.iloc[0]
    packaged_paths = None
    if bool(best_row["package_recommended"]) and best_lineage_artifacts is not None:
        packaged_paths = _package_candidate(
            args=args,
            loss_config=best_lineage_artifacts["loss_config"],
            anchor_row=anchor_row,
            lineage_artifacts=best_lineage_artifacts["lineage_artifacts"],
        )

    report = _build_report(
        results=results,
        anchor_summary=anchor_summary,
        best_row=best_row,
        packaged_paths=packaged_paths,
    )
    _write_csv(results, HUBER_WEIGHT_GRID_RESULTS_OUTPUT_PATH)
    _write_text(report, HUBER_WEIGHT_GRID_REPORT_OUTPUT_PATH)

    print(f"huber_weight_grid_results: {HUBER_WEIGHT_GRID_RESULTS_OUTPUT_PATH}")
    print(f"huber_weight_grid_report: {HUBER_WEIGHT_GRID_REPORT_OUTPUT_PATH}")
    print(
        "best_huber_weight_combo: "
        f"visc_delta={best_row['viscosity_delta']}, "
        f"ox_delta={best_row['oxidation_delta']}, "
        f"weight={best_row['sample_weight_scheme']}"
    )
    print(f"local_gain_vs_meta_anchor: {best_row['local_gain_vs_meta_anchor']:.6f}")
    print(f"platform_attempt_justified: {'yes' if bool(best_row['package_recommended']) else 'no'}")
    if packaged_paths is not None:
        print(f"huber_weight_grid_predictions: {packaged_paths['predictions_path']}")
        print(f"huber_weight_grid_zip: {packaged_paths['zip_path']}")


if __name__ == "__main__":
    main()
