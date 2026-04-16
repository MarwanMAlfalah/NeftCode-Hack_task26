"""Grouped-CV ablations for the current best scenario-level baseline."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BASELINE_ABLATION_REPORT_OUTPUT_PATH,
    BASELINE_ABLATION_RESULTS_OUTPUT_PATH,
    BASELINE_CV_RESULTS_OUTPUT_PATH,
    CV_OUTPUTS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TRAIN_SCENARIO_FEATURES_OUTPUT_PATH,
    TRAIN_TARGETS_OUTPUT_PATH,
)
from src.models.train_baselines import (
    GroupedCVArtifacts,
    PreparedBaselineData,
    TARGET_COLUMNS,
    VISCOSITY_TARGET,
    OXIDATION_TARGET,
    _build_error_analysis_table,
    evaluate_single_baseline_configuration,
    get_model_spec_by_name,
    get_target_strategy_by_name,
    load_baseline_training_data,
)


SETTING_DEFINITIONS = OrderedDict(
    [
        ("conditions_only", ["scenario_conditions"]),
        ("conditions_structure", ["scenario_conditions", "structure_and_mass"]),
        (
            "conditions_structure_family",
            ["scenario_conditions", "structure_and_mass", "component_families"],
        ),
        (
            "conditions_structure_family_coverage",
            [
                "scenario_conditions",
                "structure_and_mass",
                "component_families",
                "coverage_and_missingness",
            ],
        ),
        (
            "full_feature_set",
            [
                "scenario_conditions",
                "structure_and_mass",
                "component_families",
                "coverage_and_missingness",
                "weighted_numeric_properties",
            ],
        ),
    ]
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ablation runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    return parser.parse_args()


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def load_best_baseline_configuration(path: Path = BASELINE_CV_RESULTS_OUTPUT_PATH) -> pd.Series:
    """Load the saved baseline summary and return the best current configuration."""

    summary = pd.read_csv(path)
    if summary.empty:
        raise ValueError("Baseline CV results are empty; run the baseline pipeline first.")
    summary = summary.sort_values("rank_combined_score").reset_index(drop=True)
    return summary.iloc[0]


def build_feature_setting_columns(
    feature_manifest: dict[str, object],
    available_columns: list[str],
) -> OrderedDict[str, dict[str, object]]:
    """Translate manifest feature groups into concrete ablation settings."""

    manifest_groups: dict[str, list[str]] = feature_manifest["feature_group_columns"]
    settings: OrderedDict[str, dict[str, object]] = OrderedDict()

    for setting_name, group_names in SETTING_DEFINITIONS.items():
        columns: list[str] = []
        for group_name in group_names:
            columns.extend(manifest_groups[group_name])
        columns = [column for column in columns if column in available_columns]
        settings[setting_name] = {
            "feature_groups": group_names,
            "feature_columns": columns,
            "feature_count": len(columns),
        }
    return settings


def subset_prepared_data(
    prepared_data: PreparedBaselineData,
    feature_columns: list[str],
) -> PreparedBaselineData:
    """Create a feature-subset view of the prepared baseline data."""

    return PreparedBaselineData(
        scenario_ids=prepared_data.scenario_ids.copy(),
        X=prepared_data.X.loc[:, feature_columns].copy(),
        y=prepared_data.y.copy(),
        feature_manifest=prepared_data.feature_manifest,
    )


def aggregate_ablation_results(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold ablation metrics into a setting-level comparison table."""

    metric_columns = [
        "combined_score",
        f"{VISCOSITY_TARGET}__rmse",
        f"{OXIDATION_TARGET}__rmse",
        f"{VISCOSITY_TARGET}__mae",
        f"{OXIDATION_TARGET}__mae",
        f"{VISCOSITY_TARGET}__r2",
        f"{OXIDATION_TARGET}__r2",
        "fit_time_seconds",
        "best_inner_cv_score",
    ]
    summary = (
        fold_metrics.groupby(
            ["feature_setting", "feature_groups", "feature_count", "model_name", "target_strategy"],
            dropna=False,
        )[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "__".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary = summary.sort_values(
        ["combined_score__mean", f"{VISCOSITY_TARGET}__rmse__mean", f"{OXIDATION_TARGET}__rmse__mean"]
    ).reset_index(drop=True)
    summary["rank_combined_score"] = np.arange(1, len(summary) + 1)
    return summary


def _parse_feature_group_string(value: str) -> list[str]:
    """Decode the JSON feature-group list stored in the results table."""

    parsed = json.loads(value)
    return [str(item) for item in parsed]


def _reverse_family_slug_map(feature_manifest: dict[str, object]) -> dict[str, str]:
    """Map family slugs back to the original family label."""

    return {
        slug: family
        for family, slug in feature_manifest.get("component_family_slug_map", {}).items()
    }


def _condition_support_counts(train_features: pd.DataFrame) -> Counter:
    """Count how many scenarios share the same exact condition tuple."""

    condition_columns = [
        "test_temperature_c",
        "test_duration_h",
        "biofuel_mass_fraction_pct",
        "catalyst_dosage_category",
    ]
    tuples = [
        tuple(row)
        for row in train_features.loc[:, condition_columns].itertuples(index=False, name=None)
    ]
    return Counter(tuples)


def _normalize_signature_value(value: object) -> object:
    """Convert values into a signature-safe representation."""

    if pd.isna(value):
        return "__nan__"
    if isinstance(value, float):
        return round(value, 8)
    return value


def _build_composition_signature_counts(train_features: pd.DataFrame) -> Counter:
    """Count duplicate non-condition scenario signatures."""

    excluded_columns = {"scenario_id", *TARGET_COLUMNS, "test_temperature_c", "test_duration_h", "biofuel_mass_fraction_pct", "catalyst_dosage_category"}
    signature_columns = [column for column in train_features.columns if column not in excluded_columns]

    signatures = []
    for row in train_features.loc[:, signature_columns].itertuples(index=False, name=None):
        signatures.append(tuple(_normalize_signature_value(value) for value in row))
    return Counter(signatures)


def _scenario_condition_tuple(row: pd.Series) -> tuple[object, ...]:
    """Return the exact scenario condition tuple used in rarity checks."""

    return (
        row["test_temperature_c"],
        row["test_duration_h"],
        row["biofuel_mass_fraction_pct"],
        row["catalyst_dosage_category"],
    )


def _scenario_signature(row: pd.Series, signature_columns: list[str]) -> tuple[object, ...]:
    """Return a stable non-condition signature for one scenario."""

    return tuple(_normalize_signature_value(row[column]) for column in signature_columns)


def _top_family_descriptors(
    row: pd.Series,
    train_features: pd.DataFrame,
    reverse_family_map: dict[str, str],
    limit: int = 3,
) -> list[str]:
    """Describe the top family mass-share features for a scenario."""

    family_columns = [column for column in train_features.columns if column.startswith("family__") and column.endswith("__mass_share")]
    if not family_columns:
        return []

    top = row[family_columns].sort_values(ascending=False).head(limit)
    descriptors: list[str] = []
    for column, share in top.items():
        if share <= 0:
            continue
        slug = column[len("family__") : -len("__mass_share")]
        family_name = reverse_family_map.get(slug, slug)
        percentile = float((train_features[column] <= share).mean() * 100.0)
        descriptors.append(f"{family_name} {share:.1%} ({percentile:.0f}th pct)")
    return descriptors


def _scenario_driver_lines(
    scenario_id: str,
    scenario_rows: pd.DataFrame,
    error_table: pd.DataFrame,
    reverse_family_map: dict[str, str],
    condition_support_counts: Counter,
    signature_counts: Counter,
) -> list[str]:
    """Generate concise diagnostics for a hard scenario."""

    row = scenario_rows.loc[scenario_rows["scenario_id"] == scenario_id].iloc[0]
    error_row = error_table.loc[error_table["scenario_id"] == scenario_id].iloc[0]

    condition_support = condition_support_counts[_scenario_condition_tuple(row)]
    excluded_columns = {"scenario_id", *TARGET_COLUMNS, "test_temperature_c", "test_duration_h", "biofuel_mass_fraction_pct", "catalyst_dosage_category"}
    signature_columns = [column for column in scenario_rows.columns if column not in excluded_columns]
    peer_count = signature_counts[_scenario_signature(row, signature_columns)]

    viscosity_percentile = float(
        (scenario_rows[VISCOSITY_TARGET] <= row[VISCOSITY_TARGET]).mean() * 100.0
    )
    oxidation_percentile = float(
        (scenario_rows[OXIDATION_TARGET] <= row[OXIDATION_TARGET]).mean() * 100.0
    )

    reasons: list[str] = []
    if condition_support <= 2:
        reasons.append(f"rare condition tuple seen in only {condition_support} scenario(s)")
    if viscosity_percentile >= 95:
        reasons.append(f"extreme viscosity target at the {viscosity_percentile:.1f}th percentile")
    if oxidation_percentile >= 95:
        reasons.append(f"elevated oxidation target at the {oxidation_percentile:.1f}th percentile")
    if row["usable_property_row_ratio"] < scenario_rows["usable_property_row_ratio"].median():
        reasons.append("below-median usable numeric property coverage")
    if row["missing_all_props_row_ratio"] > scenario_rows["missing_all_props_row_ratio"].median():
        reasons.append("above-median missing-property burden")
    if peer_count > 1:
        reasons.append(f"has {peer_count - 1} near-duplicate composition peer(s), so small condition changes may drive large target swings")

    top_families = _top_family_descriptors(row, scenario_rows, reverse_family_map)
    lines = [
        f"- `{scenario_id}`: OOF combined normalized abs error `{error_row['combined_normalized_abs_error']:.3f}`, "
        f"viscosity true/pred `{error_row[f'{VISCOSITY_TARGET}__true']:.2f}` / `{error_row[f'{VISCOSITY_TARGET}__pred']:.2f}`, "
        f"oxidation true/pred `{error_row[f'{OXIDATION_TARGET}__true']:.2f}` / `{error_row[f'{OXIDATION_TARGET}__pred']:.2f}`.",
        f"  Drivers: {'; '.join(reasons) if reasons else 'no single dominant driver detected from scenario-level aggregates.'}",
    ]
    if top_families:
        lines.append(f"  Composition: {', '.join(top_families)}.")
    return lines


def build_ablation_report(
    baseline_best_row: pd.Series,
    ablation_summary: pd.DataFrame,
    best_predictions: pd.DataFrame,
    train_features_with_targets: pd.DataFrame,
    feature_manifest: dict[str, object],
) -> str:
    """Write a concise decision-oriented markdown report for feature-group ablations."""

    best_setting_row = ablation_summary.iloc[0]
    conditions_only_score = ablation_summary.loc[
        ablation_summary["feature_setting"] == "conditions_only", "combined_score__mean"
    ].iloc[0]

    reverse_family_map = _reverse_family_slug_map(feature_manifest)
    error_table = _build_error_analysis_table(best_predictions)
    condition_support_counts = _condition_support_counts(train_features_with_targets)
    signature_counts = _build_composition_signature_counts(train_features_with_targets)

    step_lines: list[str] = []
    ordered_summary = (
        ablation_summary.set_index("feature_setting").loc[list(SETTING_DEFINITIONS.keys())].reset_index()
    )
    previous_score = None
    for row in ordered_summary.to_dict(orient="records"):
        current_score = row["combined_score__mean"]
        if previous_score is None:
            previous_score = current_score
            continue
        delta = current_score - previous_score
        direction = "improved" if delta < 0 else "worsened"
        step_lines.append(
            f"- `{row['feature_setting']}` {direction} the combined score by `{abs(delta):.4f}` versus the previous step."
        )
        previous_score = current_score

    weighted_row = ordered_summary.loc[
        ordered_summary["feature_setting"] == "full_feature_set"
    ].iloc[0]
    coverage_row = ordered_summary.loc[
        ordered_summary["feature_setting"] == "conditions_structure_family_coverage"
    ].iloc[0]
    weighted_delta = float(weighted_row["combined_score__mean"] - coverage_row["combined_score__mean"])
    weighted_verdict = (
        "helpful enough to justify keeping as a compressed property channel"
        if weighted_delta < -0.05
        else "not strong enough to justify all 308 weighted-property features unchanged"
    )

    worst_case_lines: list[str] = []
    for scenario_id in error_table.head(5)["scenario_id"].tolist():
        worst_case_lines.extend(
            _scenario_driver_lines(
                scenario_id=scenario_id,
                scenario_rows=train_features_with_targets,
                error_table=error_table,
                reverse_family_map=reverse_family_map,
                condition_support_counts=condition_support_counts,
                signature_counts=signature_counts,
            )
        )

    focus_lines: list[str] = []
    for scenario_id in ["train_106", "train_107"]:
        focus_lines.extend(
            _scenario_driver_lines(
                scenario_id=scenario_id,
                scenario_rows=train_features_with_targets,
                error_table=error_table,
                reverse_family_map=reverse_family_map,
                condition_support_counts=condition_support_counts,
                signature_counts=signature_counts,
            )
        )

    deep_sets_lines = [
        "- Keep scenario conditions as explicit inputs; the hardest cases suggest strong condition-composition interactions.",
        "- Feed per-component mass fractions and component identity/family embeddings directly rather than relying only on wide aggregated family shares.",
        "- Include per-component numeric property vectors plus explicit coverage flags (`used_exact_batch_props`, `used_typical_fallback`, `missing_all_props`).",
    ]
    if weighted_delta < -0.05:
        deep_sets_lines.append(
            "- Retain the property channel in Deep Sets v1 because the full weighted-property block materially improved the linear baseline."
        )
    else:
        deep_sets_lines.append(
            "- Use a compressed or masked property channel in Deep Sets v1; the full weighted-property block added width faster than it added accuracy."
        )

    lines = [
        "# Baseline Ablation Report",
        "",
        "## Starting Point",
        f"- Current best baseline configuration from `baseline_cv_results.csv`: `{baseline_best_row['model_name']}` / `{baseline_best_row['target_strategy']}`.",
        f"- Baseline combined score: `{baseline_best_row['combined_score__mean']:.4f}`.",
        "",
        "## Feature-Group Comparison",
        "",
        "```text",
        ablation_summary[
            [
                "rank_combined_score",
                "feature_setting",
                "feature_count",
                "combined_score__mean",
                f"{VISCOSITY_TARGET}__rmse__mean",
                f"{OXIDATION_TARGET}__rmse__mean",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Decisions",
        f"- Best feature-group configuration: `{best_setting_row['feature_setting']}` with combined score `{best_setting_row['combined_score__mean']:.4f}`.",
        f"- Improvement versus `conditions_only`: `{conditions_only_score - best_setting_row['combined_score__mean']:+.4f}`.",
        *step_lines,
        f"- Weighted-property verdict: the full block is `{weighted_verdict}` (delta vs previous step `{weighted_delta:+.4f}` with `{int(weighted_row['feature_count'])}` features total).",
        "",
        "## Worst Out-of-Fold Scenarios",
        *worst_case_lines,
        "",
        "## Focus Scenarios: train_106 and train_107",
        *focus_lines,
        "",
        "## Deep Sets v1 Implications",
        *deep_sets_lines,
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run grouped-CV ablations for the best current baseline family."""

    args = parse_args()
    CV_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_best_row = load_best_baseline_configuration()
    model_spec = get_model_spec_by_name(str(baseline_best_row["model_name"]))
    target_strategy = get_target_strategy_by_name(str(baseline_best_row["target_strategy"]))

    prepared_data = load_baseline_training_data()
    feature_settings = build_feature_setting_columns(
        feature_manifest=prepared_data.feature_manifest,
        available_columns=prepared_data.X.columns.tolist(),
    )

    ablation_fold_metrics: list[pd.DataFrame] = []
    ablation_predictions: list[pd.DataFrame] = []
    for setting_name, setting in feature_settings.items():
        subset = subset_prepared_data(prepared_data, setting["feature_columns"])
        artifacts = evaluate_single_baseline_configuration(
            prepared_data=subset,
            model_spec=model_spec,
            target_strategy=target_strategy,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            seed=RANDOM_SEED,
            extra_metadata={
                "feature_setting": setting_name,
                "feature_groups": json.dumps(setting["feature_groups"]),
                "feature_count": setting["feature_count"],
            },
        )
        ablation_fold_metrics.append(artifacts.fold_metrics)
        ablation_predictions.append(artifacts.oof_predictions)

    fold_metrics = pd.concat(ablation_fold_metrics, ignore_index=True)
    summary = aggregate_ablation_results(fold_metrics)

    best_setting = summary.iloc[0]["feature_setting"]
    best_predictions = pd.concat(ablation_predictions, ignore_index=True)
    best_predictions = best_predictions.loc[
        best_predictions["feature_setting"] == best_setting
    ].sort_values("scenario_id").reset_index(drop=True)

    train_features = pd.read_csv(TRAIN_SCENARIO_FEATURES_OUTPUT_PATH)
    train_targets = pd.read_csv(TRAIN_TARGETS_OUTPUT_PATH)
    train_features_with_targets = train_features.merge(
        train_targets.loc[:, ["scenario_id", *TARGET_COLUMNS]],
        on="scenario_id",
        how="inner",
        validate="one_to_one",
    )

    report = build_ablation_report(
        baseline_best_row=baseline_best_row,
        ablation_summary=summary,
        best_predictions=best_predictions,
        train_features_with_targets=train_features_with_targets,
        feature_manifest=prepared_data.feature_manifest,
    )

    _write_csv(summary, BASELINE_ABLATION_RESULTS_OUTPUT_PATH)
    _write_text(report, BASELINE_ABLATION_REPORT_OUTPUT_PATH)

    print(f"baseline_ablation_results: {BASELINE_ABLATION_RESULTS_OUTPUT_PATH}")
    print(f"baseline_ablation_report: {BASELINE_ABLATION_REPORT_OUTPUT_PATH}")
    print(
        "best_feature_setting: "
        f"{summary.iloc[0]['feature_setting']} / {summary.iloc[0]['combined_score__mean']:.4f}"
    )


if __name__ == "__main__":
    main()
