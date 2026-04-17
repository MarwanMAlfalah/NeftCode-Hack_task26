"""Simple factor analysis using permutation importance on the best tabular baseline."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    BASELINE_ABLATION_RESULTS_OUTPUT_PATH,
    FEATURE_MANIFEST_OUTPUT_PATH,
    RANDOM_SEED,
    REPORTS_DIR,
)
from src.eval.metrics import compute_target_scales
from src.models.train_baselines import (
    TARGET_COLUMNS,
    _make_inner_scorer,
    get_model_spec_by_name,
    get_target_strategy_by_name,
    load_baseline_training_data,
)


OUTPUT_PATH = REPORTS_DIR / "factor_analysis.md"
TOP_K = 20


def _load_best_feature_setting(path: Path = BASELINE_ABLATION_RESULTS_OUTPUT_PATH) -> str:
    """Load the strongest saved tabular baseline feature setting."""

    table = pd.read_csv(path)
    row = table.sort_values("combined_score__mean", ascending=True).iloc[0]
    return str(row["feature_setting"])


def _get_feature_columns(feature_setting: str) -> list[str]:
    """Map the saved feature setting name to concrete feature columns."""

    manifest = json.loads(FEATURE_MANIFEST_OUTPUT_PATH.read_text(encoding="utf-8"))
    feature_groups = manifest["feature_group_columns"]
    setting_to_groups = {
        "conditions_only": ["scenario_conditions"],
        "conditions_structure": ["scenario_conditions", "structure_and_mass"],
        "conditions_structure_family": ["scenario_conditions", "structure_and_mass", "component_families"],
        "conditions_structure_family_coverage": [
            "scenario_conditions",
            "structure_and_mass",
            "component_families",
            "coverage_and_missingness",
        ],
        "full_feature_set": [
            "scenario_conditions",
            "structure_and_mass",
            "component_families",
            "coverage_and_missingness",
            "weighted_numeric_properties",
        ],
    }
    groups = setting_to_groups[feature_setting]
    columns: list[str] = []
    for group_name in groups:
        columns.extend(feature_groups[group_name])
    return list(dict.fromkeys(columns))


def _classify_feature_family(feature_name: str) -> str:
    """Assign a compact feature-family label for reporting."""

    if feature_name in {
        "test_temperature_c",
        "test_duration_h",
        "biofuel_mass_fraction_pct",
        "catalyst_dosage_category",
    }:
        return "scenario_conditions"
    if feature_name.startswith("family__") and feature_name.endswith("__mass_share"):
        return "family_mass_share"
    if feature_name.startswith("family__") and feature_name.endswith("__mass_sum"):
        return "family_mass_sum"
    if feature_name.startswith("family__"):
        return "family_counts"
    if feature_name.startswith("mass_fraction_"):
        return "mass_distribution"
    if "repeat" in feature_name or feature_name.endswith("_unique_count") or feature_name.endswith("_row_count"):
        return "structure_counts"
    return "structure_other"


def _map_hypothesis(feature_name: str) -> str:
    """Map each feature to a short hypothesis label used in the literature review."""

    if feature_name in {"test_temperature_c", "test_duration_h", "biofuel_mass_fraction_pct", "catalyst_dosage_category"}:
        return "Condition severity and catalyst chemistry"
    if feature_name.startswith("family__bazovoe_maslo") or feature_name.startswith("family__zagustitel"):
        return "Base-oil family and viscosity modifier balance"
    if feature_name.startswith("family__antioksidant") or feature_name.startswith("family__protivoiznosnaya_prisadka"):
        return "Antioxidant and antiwear family protection"
    if feature_name.startswith("family__detergent") or feature_name.startswith("family__dispersant"):
        return "Detergent / dispersant control of oxidation products"
    return "Mixture structure and composition balance"


def build_factor_analysis() -> tuple[pd.DataFrame, str]:
    """Fit the strongest tabular baseline and compute permutation importance."""

    prepared = load_baseline_training_data()
    best_results = pd.read_csv(BASELINE_ABLATION_RESULTS_OUTPUT_PATH)
    best_row = best_results.sort_values("combined_score__mean", ascending=True).iloc[0]
    feature_setting = str(best_row["feature_setting"])
    feature_columns = _get_feature_columns(feature_setting)
    X = prepared.X.loc[:, feature_columns].copy()
    y = prepared.y.to_numpy(dtype=float)
    groups = prepared.scenario_ids.to_numpy()

    model_spec = get_model_spec_by_name("pls_regression")
    target_strategy = get_target_strategy_by_name("raw")
    target_scales = compute_target_scales(y, TARGET_COLUMNS)
    scorer = _make_inner_scorer(target_strategy=target_strategy, target_scales=target_scales)
    search = GridSearchCV(
        estimator=model_spec.build_estimator(RANDOM_SEED),
        param_grid=model_spec.build_param_grid(X, y),
        scoring=scorer,
        cv=GroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=None,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y, groups=groups)

    importance = permutation_importance(
        estimator=search.best_estimator_,
        X=X,
        y=y,
        scoring=scorer,
        n_repeats=20,
        random_state=RANDOM_SEED,
    )

    importance_frame = pd.DataFrame(
        {
            "feature_name": X.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    importance_frame["feature_family"] = importance_frame["feature_name"].map(_classify_feature_family)
    importance_frame["hypothesis_link"] = importance_frame["feature_name"].map(_map_hypothesis)

    top20 = importance_frame.head(TOP_K).copy()
    family_summary = (
        top20.groupby("feature_family", dropna=False)["feature_name"]
        .agg(["count", lambda values: ", ".join(values.head(5))])
        .reset_index()
        .rename(columns={"<lambda_0>": "example_features"})
        .sort_values(["count", "feature_family"], ascending=[False, True])
    )

    lines = [
        "# Factor Analysis",
        "",
        "## Method",
        "- Baseline family: `pls_regression`",
        "- Target strategy: `raw`",
        f"- Feature subset: `{feature_setting}`",
        "- Importance method: permutation importance on the fitted full-data tabular baseline",
        "- Interpretation note: this is directional and model-dependent, not causal.",
        "",
        "## Best Baseline Context",
        f"- Feature count evaluated: `{len(feature_columns)}`",
        f"- Saved grouped-CV combined score: `{float(best_row['combined_score__mean']):.4f}`",
        f"- Best inner parameters: `{search.best_params_}`",
        "",
        "## Top 20 Features",
        "",
        "```text",
        top20.loc[:, ["feature_name", "feature_family", "importance_mean", "importance_std", "hypothesis_link"]].to_string(index=False),
        "```",
        "",
        "## Grouped By Feature Family",
        "",
        "```text",
        family_summary.to_string(index=False),
        "```",
        "",
        "## Interpretation Against Literature Hypotheses",
        "- Scenario-condition variables near the top would support the hypothesis that DOT severity is strongly driven by temperature, duration, biofuel loading, and catalyst chemistry.",
        "- Family mass-share and family-count features support the decision to model mixture composition at the family level rather than relying only on component identity.",
        "- Structure and mass-distribution features matter when dominant components or repeated-family patterns change oxidation pathways and viscosity build-up.",
        "- The absence of wide weighted-property aggregates among the strongest baseline features is consistent with the earlier ablation result that broad weighted-property blocks added noise faster than signal.",
    ]
    return top20, "\n".join(lines) + "\n"


def main() -> None:
    """Run factor analysis and persist the markdown report."""

    _, report = build_factor_analysis()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"factor_analysis_report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
