"""Low-risk external-data augmentation experiment for the current tabular/meta lineage."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    EXTERNAL_AUGMENTED_REPORT_OUTPUT_PATH,
    EXTERNAL_AUGMENTED_RESULTS_OUTPUT_PATH,
    EXTERNAL_DATA_AUDIT_REPORT_OUTPUT_PATH,
    GP_STAGE2_OOF_PREDICTIONS_OUTPUT_PATH,
    META_STACK_SEARCH_RESULTS_OUTPUT_PATH,
    RANDOM_SEED,
    TRAIN_SCENARIO_FEATURES_OUTPUT_PATH,
)
from src.data.prepare_external_data import write_external_data_audit
from src.eval.metrics import evaluate_platform_predictions
from src.models.train_baselines import OXIDATION_TARGET, TARGET_COLUMNS, VISCOSITY_TARGET


LIVE_PLATFORM_ANCHOR_SCORE = 0.104084
CONDITION_FEATURE_COLUMNS = [
    "test_temperature_c",
    "test_duration_h",
    "biofuel_mass_fraction_pct",
    "catalyst_dosage_category",
]
BLEND_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25]
MIN_EXTERNAL_SCENARIOS_FOR_RUN = 3


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _load_local_anchor_row() -> pd.Series:
    results = pd.read_csv(META_STACK_SEARCH_RESULTS_OUTPUT_PATH)
    if results.empty:
        raise ValueError(f"Missing local meta-stack results: {META_STACK_SEARCH_RESULTS_OUTPUT_PATH}")
    ranked = results.sort_values(
        ["rank_platform_score", "platform_score__mean"],
        kind="mergesort",
    ).reset_index(drop=True)
    return ranked.iloc[0].copy()


def _load_anchor_artifact() -> pd.DataFrame:
    frame = pd.read_csv(GP_STAGE2_OOF_PREDICTIONS_OUTPUT_PATH)
    required_columns = [
        "scenario_id",
        "fold_index",
        f"{VISCOSITY_TARGET}__true",
        f"{OXIDATION_TARGET}__true",
        "best_meta_candidate_name",
        "best_meta_viscosity_pred",
        "best_meta_oxidation_pred",
    ]
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required anchor columns: {missing}")
    return frame.loc[:, required_columns].sort_values(["fold_index", "scenario_id"]).reset_index(drop=True)


def _load_internal_condition_frame() -> pd.DataFrame:
    features = pd.read_csv(TRAIN_SCENARIO_FEATURES_OUTPUT_PATH)
    required_columns = ["scenario_id", *CONDITION_FEATURE_COLUMNS]
    missing = sorted(set(required_columns) - set(features.columns))
    if missing:
        raise ValueError(f"Missing internal condition columns: {missing}")
    return features.loc[:, required_columns].sort_values("scenario_id").reset_index(drop=True)


def _build_anchor_frame() -> tuple[pd.DataFrame, pd.Series]:
    anchor_artifact = _load_anchor_artifact()
    condition_frame = _load_internal_condition_frame()
    merged = anchor_artifact.merge(condition_frame, on="scenario_id", how="inner", validate="one_to_one")
    if len(merged) != len(anchor_artifact):
        raise ValueError("Condition features failed to align with the local anchor artifact.")
    local_anchor_row = _load_local_anchor_row()
    return merged.sort_values(["fold_index", "scenario_id"]).reset_index(drop=True), local_anchor_row


def _summarize_candidate(
    frame: pd.DataFrame,
    candidate_name: str,
    viscosity_pred: np.ndarray,
    oxidation_pred: np.ndarray,
    status: str,
    blend_weight: float | None,
    external_scenario_count: int,
) -> dict[str, object]:
    prediction_frame = frame.loc[
        :,
        [
            "scenario_id",
            "fold_index",
            f"{VISCOSITY_TARGET}__true",
            f"{OXIDATION_TARGET}__true",
        ],
    ].copy()
    prediction_frame[f"{VISCOSITY_TARGET}__pred"] = np.asarray(viscosity_pred, dtype=float)
    prediction_frame[f"{OXIDATION_TARGET}__pred"] = np.asarray(oxidation_pred, dtype=float)

    fold_records: list[dict[str, object]] = []
    for fold_index, fold_frame in prediction_frame.groupby("fold_index", dropna=False):
        metrics = evaluate_platform_predictions(
            y_true=fold_frame.loc[:, [f"{VISCOSITY_TARGET}__true", f"{OXIDATION_TARGET}__true"]].to_numpy(dtype=float),
            y_pred=fold_frame.loc[:, [f"{VISCOSITY_TARGET}__pred", f"{OXIDATION_TARGET}__pred"]].to_numpy(dtype=float),
            target_names=TARGET_COLUMNS,
        )
        fold_records.append(
            {
                "fold_index": int(fold_index),
                "platform_score": float(metrics["platform_score"]),
                f"{VISCOSITY_TARGET}__platform_mae": float(metrics[f"{VISCOSITY_TARGET}__platform_mae"]),
                f"{OXIDATION_TARGET}__platform_mae": float(metrics[f"{OXIDATION_TARGET}__platform_mae"]),
            }
        )
    fold_metrics = pd.DataFrame.from_records(fold_records).sort_values("fold_index").reset_index(drop=True)
    return {
        "candidate_name": candidate_name,
        "status": status,
        "blend_weight": blend_weight if blend_weight is not None else np.nan,
        "external_supervised_scenario_count": external_scenario_count,
        "platform_score__mean": float(fold_metrics["platform_score"].mean()),
        "platform_score__std": float(fold_metrics["platform_score"].std(ddof=1)),
        f"{VISCOSITY_TARGET}__platform_mae__mean": float(fold_metrics[f"{VISCOSITY_TARGET}__platform_mae"].mean()),
        f"{OXIDATION_TARGET}__platform_mae__mean": float(fold_metrics[f"{OXIDATION_TARGET}__platform_mae"].mean()),
    }


def _fit_external_condition_predictions(
    anchor_frame: pd.DataFrame,
    external_scenarios: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    viscosity_predictions = np.zeros(len(anchor_frame), dtype=float)
    oxidation_predictions = np.zeros(len(anchor_frame), dtype=float)
    if external_scenarios.empty:
        return viscosity_predictions, oxidation_predictions

    external_X = external_scenarios.loc[:, CONDITION_FEATURE_COLUMNS].to_numpy(dtype=float)
    external_y = external_scenarios.loc[:, [VISCOSITY_TARGET, OXIDATION_TARGET]].to_numpy(dtype=float)
    external_weight = external_scenarios["augmentation_sample_weight"].to_numpy(dtype=float)

    for fold_index in sorted(anchor_frame["fold_index"].unique()):
        train_mask = anchor_frame["fold_index"] != fold_index
        valid_mask = anchor_frame["fold_index"] == fold_index

        internal_train = anchor_frame.loc[train_mask, :]
        X_train = np.vstack(
            [
                internal_train.loc[:, CONDITION_FEATURE_COLUMNS].to_numpy(dtype=float),
                external_X,
            ]
        )
        y_train = np.vstack(
            [
                internal_train.loc[:, [f"{VISCOSITY_TARGET}__true", f"{OXIDATION_TARGET}__true"]].to_numpy(dtype=float),
                external_y,
            ]
        )
        sample_weight = np.concatenate(
            [
                np.ones(len(internal_train), dtype=float),
                external_weight,
            ]
        )
        imputer = SimpleImputer(strategy="median")
        X_train_imputed = imputer.fit_transform(X_train)
        X_valid_imputed = imputer.transform(
            anchor_frame.loc[valid_mask, CONDITION_FEATURE_COLUMNS].to_numpy(dtype=float)
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_valid_scaled = scaler.transform(X_valid_imputed)

        viscosity_model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
        oxidation_model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
        viscosity_model.fit(X_train_scaled, y_train[:, 0], sample_weight=sample_weight)
        oxidation_model.fit(X_train_scaled, y_train[:, 1], sample_weight=sample_weight)

        viscosity_predictions[valid_mask] = viscosity_model.predict(X_valid_scaled)
        oxidation_predictions[valid_mask] = oxidation_model.predict(X_valid_scaled)

    return viscosity_predictions, oxidation_predictions


def _build_report(
    results: pd.DataFrame,
    local_anchor_row: pd.Series,
    external_scenario_count: int,
    best_candidate_row: pd.Series | None,
) -> str:
    lines = [
        "# External Augmented Report",
        "",
        "## Scope",
        "- Live platform anchor 0.104084 stayed untouched.",
        "- Official shipping path stayed untouched.",
        "- This sprint only staged and tested a low-risk external-data sidecar path for the current tabular/meta family.",
        "- Ranking metric: `0.5 * (visc_MAE / 2439.25 + ox_MAE / 160.62)`.",
        "",
        "## Inputs",
        f"- Current local meta-family anchor lineage: `{local_anchor_row['candidate_name']}` at `{local_anchor_row['platform_score__mean']:.6f}`",
        f"- External supervised scenarios available after audit: `{external_scenario_count}`",
        f"- External audit report: `{EXTERNAL_DATA_AUDIT_REPORT_OUTPUT_PATH}`",
        "",
        "## Candidates",
        "",
        "```text",
        results[
            [
                "candidate_name",
                "status",
                "blend_weight",
                "external_supervised_scenario_count",
                "platform_score__mean",
                "local_gain_vs_meta_anchor",
                f"{VISCOSITY_TARGET}__platform_mae__mean",
                f"{OXIDATION_TARGET}__platform_mae__mean",
            ]
        ].to_string(index=False),
        "```",
        "",
    ]
    if best_candidate_row is None:
        lines.extend(
            [
                "## Decision",
                "- No external-data candidate was run because the audited dataset does not yet contain enough supervised scenarios.",
                "- The repo is ready for manual extraction, but there is not enough trustworthy external signal yet to justify a local augmentation test.",
            ]
        )
    else:
        lines.extend(
            [
                "## Decision",
                f"- Best external candidate: `{best_candidate_row['candidate_name']}` at `{best_candidate_row['platform_score__mean']:.6f}`",
                f"- Local gain vs current meta-family anchor: `{best_candidate_row['local_gain_vs_meta_anchor']:+.6f}`",
                f"- Meaningful packaging threshold met: `{'yes' if bool(best_candidate_row['meaningful_gain']) else 'no'}`",
                "- No submission was packaged automatically in this sprint.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    artifacts = write_external_data_audit()
    anchor_frame, local_anchor_row = _build_anchor_frame()
    anchor_summary = _summarize_candidate(
        frame=anchor_frame,
        candidate_name="current_meta_family_anchor",
        viscosity_pred=anchor_frame["best_meta_viscosity_pred"].to_numpy(dtype=float),
        oxidation_pred=anchor_frame["best_meta_oxidation_pred"].to_numpy(dtype=float),
        status="evaluated",
        blend_weight=0.0,
        external_scenario_count=int(artifacts.scenario_rows["usable_for_supervised_augmentation"].sum()),
    )
    anchor_summary["local_gain_vs_meta_anchor"] = 0.0
    anchor_summary["meaningful_gain"] = False

    results_rows: list[dict[str, object]] = [anchor_summary]
    external_scenarios = artifacts.scenario_rows.loc[
        artifacts.scenario_rows["usable_for_supervised_augmentation"],
        [
            "external_scenario_id",
            "temperature_c",
            "duration_h",
            "biofuel_pct",
            "catalyst_category",
            "target_visc_rel_pct",
            "target_ox_acm",
            "augmentation_sample_weight",
        ],
    ].rename(
        columns={
            "temperature_c": "test_temperature_c",
            "duration_h": "test_duration_h",
            "biofuel_pct": "biofuel_mass_fraction_pct",
            "catalyst_category": "catalyst_dosage_category",
            "target_visc_rel_pct": VISCOSITY_TARGET,
            "target_ox_acm": OXIDATION_TARGET,
        }
    ).reset_index(drop=True)

    best_candidate_row: pd.Series | None = None
    if len(external_scenarios) >= MIN_EXTERNAL_SCENARIOS_FOR_RUN:
        external_viscosity_pred, external_oxidation_pred = _fit_external_condition_predictions(
            anchor_frame=anchor_frame,
            external_scenarios=external_scenarios,
        )
        for blend_weight in BLEND_WEIGHTS:
            candidate_viscosity = (
                (1.0 - blend_weight) * anchor_frame["best_meta_viscosity_pred"].to_numpy(dtype=float)
                + blend_weight * external_viscosity_pred
            )
            candidate_oxidation = (
                (1.0 - blend_weight) * anchor_frame["best_meta_oxidation_pred"].to_numpy(dtype=float)
                + blend_weight * external_oxidation_pred
            )
            candidate_summary = _summarize_candidate(
                frame=anchor_frame,
                candidate_name=f"external_condition_blend__w{int(round(blend_weight * 100)):02d}",
                viscosity_pred=candidate_viscosity,
                oxidation_pred=candidate_oxidation,
                status="evaluated",
                blend_weight=blend_weight,
                external_scenario_count=len(external_scenarios),
            )
            candidate_summary["local_gain_vs_meta_anchor"] = (
                float(anchor_summary["platform_score__mean"]) - float(candidate_summary["platform_score__mean"])
            )
            candidate_summary["meaningful_gain"] = candidate_summary["local_gain_vs_meta_anchor"] >= 0.0020
            results_rows.append(candidate_summary)
        results = pd.DataFrame.from_records(results_rows).sort_values(
            ["platform_score__mean", f"{VISCOSITY_TARGET}__platform_mae__mean", f"{OXIDATION_TARGET}__platform_mae__mean"],
            kind="mergesort",
        ).reset_index(drop=True)
        best_external_candidates = results.loc[results["candidate_name"] != "current_meta_family_anchor"].copy()
        if not best_external_candidates.empty:
            best_candidate_row = best_external_candidates.iloc[0]
    else:
        skipped_row = {
            "candidate_name": "external_condition_blend__not_run",
            "status": "skipped_not_enough_supervised_external_scenarios",
            "blend_weight": np.nan,
            "external_supervised_scenario_count": len(external_scenarios),
            "platform_score__mean": np.nan,
            "platform_score__std": np.nan,
            f"{VISCOSITY_TARGET}__platform_mae__mean": np.nan,
            f"{OXIDATION_TARGET}__platform_mae__mean": np.nan,
            "local_gain_vs_meta_anchor": np.nan,
            "meaningful_gain": False,
        }
        results_rows.append(skipped_row)
        results = pd.DataFrame.from_records(results_rows)

    if "local_gain_vs_meta_anchor" not in results.columns:
        results["local_gain_vs_meta_anchor"] = results["platform_score__mean"].apply(
            lambda value: float(anchor_summary["platform_score__mean"]) - float(value) if pd.notna(value) else np.nan
        )
    if "meaningful_gain" not in results.columns:
        results["meaningful_gain"] = results["local_gain_vs_meta_anchor"].fillna(-np.inf) >= 0.0020

    report = _build_report(
        results=results,
        local_anchor_row=local_anchor_row,
        external_scenario_count=len(external_scenarios),
        best_candidate_row=best_candidate_row,
    )
    _write_csv(results, EXTERNAL_AUGMENTED_RESULTS_OUTPUT_PATH)
    _write_text(report, EXTERNAL_AUGMENTED_REPORT_OUTPUT_PATH)

    print(f"external_data_audit: {EXTERNAL_DATA_AUDIT_REPORT_OUTPUT_PATH}")
    print(f"external_augmented_results: {EXTERNAL_AUGMENTED_RESULTS_OUTPUT_PATH}")
    print(f"external_augmented_report: {EXTERNAL_AUGMENTED_REPORT_OUTPUT_PATH}")
    print(f"local_meta_anchor: {local_anchor_row['candidate_name']}")
    print(f"external_supervised_scenarios: {len(external_scenarios)}")
    if best_candidate_row is not None:
        print(f"best_external_candidate: {best_candidate_row['candidate_name']}")
        print(f"local_gain_vs_meta_anchor: {best_candidate_row['local_gain_vs_meta_anchor']:.6f}")
    else:
        print("best_external_candidate: none")


if __name__ == "__main__":
    main()
