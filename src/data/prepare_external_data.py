"""Validate, audit, and stage manually collected external oxidation-data records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    EXTERNAL_COLLECTED_RECORDS_PATH,
    EXTERNAL_DATA_AUDIT_REPORT_OUTPUT_PATH,
    EXTERNAL_DATA_DIR,
    TRAIN_SCENARIO_FEATURES_OUTPUT_PATH,
)


EXTERNAL_REQUIRED_COLUMNS = [
    "external_scenario_id",
    "source_id",
    "source_type",
    "source_url",
    "test_family",
    "temperature_c",
    "duration_h",
    "biofuel_pct",
    "catalyst_category",
    "component_type",
    "mass_fraction_pct",
    "target_visc_rel_pct",
    "target_ox_acm",
    "target_ox_proxy",
    "notes",
    "condition_similarity_score",
    "source_reliability_score",
]
NUMERIC_COLUMNS = [
    "temperature_c",
    "duration_h",
    "biofuel_pct",
    "catalyst_category",
    "mass_fraction_pct",
    "target_visc_rel_pct",
    "target_ox_acm",
    "target_ox_proxy",
    "condition_similarity_score",
    "source_reliability_score",
]
SCENARIO_LEVEL_COLUMNS = [
    "source_id",
    "source_type",
    "source_url",
    "test_family",
    "temperature_c",
    "duration_h",
    "biofuel_pct",
    "catalyst_category",
    "target_visc_rel_pct",
    "target_ox_acm",
    "target_ox_proxy",
]
EXTERNAL_COMPONENT_TAXONOMY = [
    "base_oil_g1",
    "base_oil_g2",
    "base_oil_g3",
    "base_oil_g4_pao",
    "base_oil_g5_ester",
    "base_oil_g5_other",
    "aminic_antioxidant",
    "phenolic_antioxidant",
    "tocopherol_antioxidant",
    "sulfur_phenolic_antioxidant",
    "peroxide_decomposer",
    "mixed_antioxidant_package",
    "ca_detergent",
    "mg_detergent",
    "mixed_metal_detergent",
    "salicylate_detergent",
    "phenate_detergent",
    "sulfonate_detergent",
    "ashless_dispersant",
    "zddp",
    "molybdenum_friction_modifier",
    "boron_additive",
    "vi_improver",
    "antifoam",
    "pour_point_depressant",
    "other_additive",
    "unsaturated_hydrocarbon_additive",
    "biofuel_component",
    "group_v_stability_booster",
]
ALLOWED_SOURCE_TYPES = {
    "peer_reviewed_article": 0.95,
    "technical_report": 0.90,
    "standard_summary": 0.85,
    "conference_paper": 0.80,
    "review_article": 0.70,
    "datasheet": 0.60,
    "other": 0.50,
}
MASS_SUM_TOLERANCE = 5.0
MIN_SCENARIO_COMPLETENESS = 0.85
MIN_CONDITION_SIMILARITY = 0.55
MIN_SOURCE_RELIABILITY = 0.70


@dataclass(frozen=True)
class ExternalDataArtifacts:
    """Validated external-data tables plus audit metadata."""

    component_rows: pd.DataFrame
    scenario_rows: pd.DataFrame
    audit_markdown: str


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _load_reference_conditions() -> pd.DataFrame:
    reference = pd.read_csv(TRAIN_SCENARIO_FEATURES_OUTPUT_PATH)
    return reference.loc[
        :,
        [
            "scenario_id",
            "test_temperature_c",
            "test_duration_h",
            "biofuel_mass_fraction_pct",
            "catalyst_dosage_category",
        ],
    ].drop_duplicates(subset=["scenario_id"]).reset_index(drop=True)


def _empty_component_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EXTERNAL_REQUIRED_COLUMNS)


def load_external_component_rows(path: Path = EXTERNAL_COLLECTED_RECORDS_PATH) -> pd.DataFrame:
    """Load the manually collected external component-row CSV."""

    if not path.exists():
        raise FileNotFoundError(f"Missing external-data file: {path}")
    frame = pd.read_csv(path)
    missing_columns = sorted(set(EXTERNAL_REQUIRED_COLUMNS) - set(frame.columns))
    extra_columns = sorted(set(frame.columns) - set(EXTERNAL_REQUIRED_COLUMNS))
    if missing_columns:
        raise ValueError(f"External-data file is missing required columns: {missing_columns}")
    if extra_columns:
        raise ValueError(f"External-data file contains unexpected columns: {extra_columns}")
    return frame.loc[:, EXTERNAL_REQUIRED_COLUMNS].copy()


def _coerce_external_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in prepared.columns:
        if prepared[column].dtype == object:
            prepared[column] = prepared[column].replace(r"^\s*$", np.nan, regex=True)
    for column in NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared


def _compute_condition_similarity(
    frame: pd.DataFrame,
    reference_conditions: pd.DataFrame,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float, index=frame.index, name="recommended_condition_similarity_score")

    numeric_reference = reference_conditions.loc[
        :,
        ["test_temperature_c", "test_duration_h", "biofuel_mass_fraction_pct"],
    ].to_numpy(dtype=float)
    ranges = reference_conditions.loc[
        :,
        ["test_temperature_c", "test_duration_h", "biofuel_mass_fraction_pct"],
    ].max() - reference_conditions.loc[:, ["test_temperature_c", "test_duration_h", "biofuel_mass_fraction_pct"]].min()
    scale = np.clip(ranges.to_numpy(dtype=float), 1.0, None)
    catalyst_reference = reference_conditions["catalyst_dosage_category"].to_numpy(dtype=float)

    scores: list[float] = []
    for row in frame.itertuples(index=False):
        values = np.asarray([row.temperature_c, row.duration_h, row.biofuel_pct], dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(row.catalyst_category):
            scores.append(np.nan)
            continue
        normalized_distance = np.abs(numeric_reference - values) / scale
        catalyst_penalty = np.where(catalyst_reference == float(row.catalyst_category), 0.0, 0.25)
        nearest_distance = float(np.min(normalized_distance.mean(axis=1) + catalyst_penalty))
        scores.append(float(np.exp(-2.0 * nearest_distance)))
    return pd.Series(scores, index=frame.index, name="recommended_condition_similarity_score")


def _build_component_validation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    reference_conditions = _load_reference_conditions()
    validated = _coerce_external_frame(frame)

    validated["component_type_valid"] = validated["component_type"].isin(EXTERNAL_COMPONENT_TAXONOMY)
    validated["source_type_valid"] = validated["source_type"].isin(ALLOWED_SOURCE_TYPES)
    validated["source_url_present"] = validated["source_url"].notna()
    validated["scenario_id_present"] = validated["external_scenario_id"].notna()
    validated["source_id_present"] = validated["source_id"].notna()
    validated["test_family_present"] = validated["test_family"].notna()
    validated["mass_fraction_nonnegative"] = validated["mass_fraction_pct"].fillna(-1.0) >= 0.0
    validated["recommended_condition_similarity_score"] = _compute_condition_similarity(
        frame=validated,
        reference_conditions=reference_conditions,
    )
    validated["recommended_source_reliability_score"] = validated["source_type"].map(ALLOWED_SOURCE_TYPES)
    validated["effective_condition_similarity_score"] = validated["condition_similarity_score"].where(
        validated["condition_similarity_score"].between(0.0, 1.0, inclusive="both"),
        validated["recommended_condition_similarity_score"],
    )
    validated["effective_source_reliability_score"] = validated["source_reliability_score"].where(
        validated["source_reliability_score"].between(0.0, 1.0, inclusive="both"),
        validated["recommended_source_reliability_score"],
    )
    required_value_columns = [
        "external_scenario_id",
        "source_id",
        "source_type",
        "source_url",
        "test_family",
        "temperature_c",
        "duration_h",
        "biofuel_pct",
        "catalyst_category",
        "component_type",
        "mass_fraction_pct",
    ]
    validated["row_completeness_fraction"] = validated[required_value_columns].notna().mean(axis=1)
    validated["row_completeness_ok"] = validated["row_completeness_fraction"] >= MIN_SCENARIO_COMPLETENESS
    validated["duplicate_scenario_component"] = validated.duplicated(
        subset=["external_scenario_id", "component_type"],
        keep=False,
    )
    validated["duplicate_full_row"] = validated.duplicated(
        subset=EXTERNAL_REQUIRED_COLUMNS,
        keep=False,
    )
    validated["row_schema_valid"] = (
        validated["component_type_valid"]
        & validated["source_type_valid"]
        & validated["source_url_present"]
        & validated["scenario_id_present"]
        & validated["source_id_present"]
        & validated["test_family_present"]
        & validated["mass_fraction_nonnegative"]
        & validated["row_completeness_ok"]
        & ~validated["duplicate_scenario_component"]
    )
    validated["usable_for_supervised_augmentation_row"] = (
        validated["row_schema_valid"]
        & validated["target_visc_rel_pct"].notna()
        & validated["target_ox_acm"].notna()
        & (validated["effective_condition_similarity_score"].fillna(0.0) >= MIN_CONDITION_SIMILARITY)
        & (validated["effective_source_reliability_score"].fillna(0.0) >= MIN_SOURCE_RELIABILITY)
    )
    return validated


def _first_non_null(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def _build_external_scenario_rows(component_rows: pd.DataFrame) -> pd.DataFrame:
    if component_rows.empty:
        base_columns = [
            "external_scenario_id",
            "source_id",
            "source_type",
            "source_url",
            "test_family",
            "temperature_c",
            "duration_h",
            "biofuel_pct",
            "catalyst_category",
            "target_visc_rel_pct",
            "target_ox_acm",
            "target_ox_proxy",
            "component_row_count",
            "component_type_unique_count",
            "mass_fraction_sum",
            "mass_fraction_sum_ok",
            "scenario_conditions_constant",
            "scenario_targets_constant",
            "scenario_schema_valid",
            "scenario_completeness_fraction",
            "effective_condition_similarity_score",
            "effective_source_reliability_score",
            "usable_for_supervised_augmentation",
            "augmentation_sample_weight",
        ]
        component_share_columns = [f"component_type__{component_type}__mass_share" for component_type in EXTERNAL_COMPONENT_TAXONOMY]
        return pd.DataFrame(columns=[*base_columns, *component_share_columns])

    grouped = component_rows.groupby("external_scenario_id", dropna=False, sort=True)
    scenario_rows = grouped.agg(
        source_id=("source_id", _first_non_null),
        source_type=("source_type", _first_non_null),
        source_url=("source_url", _first_non_null),
        test_family=("test_family", _first_non_null),
        temperature_c=("temperature_c", _first_non_null),
        duration_h=("duration_h", _first_non_null),
        biofuel_pct=("biofuel_pct", _first_non_null),
        catalyst_category=("catalyst_category", _first_non_null),
        target_visc_rel_pct=("target_visc_rel_pct", _first_non_null),
        target_ox_acm=("target_ox_acm", _first_non_null),
        target_ox_proxy=("target_ox_proxy", _first_non_null),
        component_row_count=("component_type", "size"),
        component_type_unique_count=("component_type", lambda values: int(pd.Series(values).nunique(dropna=False))),
        mass_fraction_sum=("mass_fraction_pct", "sum"),
        scenario_completeness_fraction=("row_completeness_fraction", "mean"),
        effective_condition_similarity_score=("effective_condition_similarity_score", "mean"),
        effective_source_reliability_score=("effective_source_reliability_score", "mean"),
    ).reset_index()

    condition_constant = grouped.apply(
        lambda subset: all(subset[column].nunique(dropna=False) <= 1 for column in SCENARIO_LEVEL_COLUMNS[:-3])
    )
    target_constant = grouped.apply(
        lambda subset: all(subset[column].nunique(dropna=False) <= 1 for column in ["target_visc_rel_pct", "target_ox_acm", "target_ox_proxy"])
    )
    duplicate_components = grouped["duplicate_scenario_component"].max()
    row_schema_valid = grouped["row_schema_valid"].all()
    usable_rows = grouped["usable_for_supervised_augmentation_row"].all()

    scenario_rows["scenario_conditions_constant"] = scenario_rows["external_scenario_id"].map(condition_constant).fillna(False)
    scenario_rows["scenario_targets_constant"] = scenario_rows["external_scenario_id"].map(target_constant).fillna(False)
    scenario_rows["duplicate_scenario_component"] = scenario_rows["external_scenario_id"].map(duplicate_components).fillna(False)
    scenario_rows["scenario_schema_valid"] = scenario_rows["external_scenario_id"].map(row_schema_valid).fillna(False)
    scenario_rows["mass_fraction_sum_ok"] = scenario_rows["mass_fraction_sum"].between(
        100.0 - MASS_SUM_TOLERANCE,
        100.0 + MASS_SUM_TOLERANCE,
        inclusive="both",
    )
    scenario_rows["usable_for_supervised_augmentation"] = (
        scenario_rows["scenario_schema_valid"]
        & scenario_rows["scenario_conditions_constant"]
        & scenario_rows["scenario_targets_constant"]
        & scenario_rows["mass_fraction_sum_ok"]
        & ~scenario_rows["duplicate_scenario_component"]
        & scenario_rows["target_visc_rel_pct"].notna()
        & scenario_rows["target_ox_acm"].notna()
        & (scenario_rows["scenario_completeness_fraction"] >= MIN_SCENARIO_COMPLETENESS)
        & (scenario_rows["effective_condition_similarity_score"].fillna(0.0) >= MIN_CONDITION_SIMILARITY)
        & (scenario_rows["effective_source_reliability_score"].fillna(0.0) >= MIN_SOURCE_RELIABILITY)
        & scenario_rows["external_scenario_id"].map(usable_rows).fillna(False)
    )
    scenario_rows["augmentation_sample_weight"] = np.where(
        scenario_rows["usable_for_supervised_augmentation"],
        np.clip(
            0.35
            * scenario_rows["effective_condition_similarity_score"].fillna(0.0)
            * scenario_rows["effective_source_reliability_score"].fillna(0.0),
            0.05,
            0.35,
        ),
        0.0,
    )

    share_frame = (
        component_rows.pivot_table(
            index="external_scenario_id",
            columns="component_type",
            values="mass_fraction_pct",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(columns=EXTERNAL_COMPONENT_TAXONOMY, fill_value=0.0)
        .rename(columns=lambda value: f"component_type__{value}__mass_share")
        .reset_index()
    )
    total_mass = scenario_rows["mass_fraction_sum"].replace(0.0, np.nan)
    for column in share_frame.columns:
        if column == "external_scenario_id":
            continue
        share_frame[column] = share_frame[column].astype(float)
    share_frame = share_frame.merge(
        scenario_rows.loc[:, ["external_scenario_id", "mass_fraction_sum"]],
        on="external_scenario_id",
        how="left",
        validate="one_to_one",
    )
    component_share_columns = [column for column in share_frame.columns if column.startswith("component_type__")]
    for column in component_share_columns:
        share_frame[column] = np.divide(
            share_frame[column],
            share_frame["mass_fraction_sum"].replace(0.0, np.nan),
        ).fillna(0.0)
    share_frame = share_frame.drop(columns=["mass_fraction_sum"])

    scenario_rows = scenario_rows.merge(
        share_frame,
        on="external_scenario_id",
        how="left",
        validate="one_to_one",
    )
    scenario_rows[component_share_columns] = scenario_rows[component_share_columns].fillna(0.0)
    return scenario_rows.sort_values("external_scenario_id").reset_index(drop=True)


def _build_audit_markdown(component_rows: pd.DataFrame, scenario_rows: pd.DataFrame) -> str:
    invalid_component_count = int((~component_rows["row_schema_valid"]).sum()) if not component_rows.empty else 0
    usable_scenario_count = int(scenario_rows["usable_for_supervised_augmentation"].sum()) if not scenario_rows.empty else 0
    duplicate_component_count = int(component_rows["duplicate_scenario_component"].sum()) if not component_rows.empty else 0
    average_similarity = (
        float(component_rows["effective_condition_similarity_score"].dropna().mean())
        if not component_rows.empty and component_rows["effective_condition_similarity_score"].notna().any()
        else float("nan")
    )
    average_reliability = (
        float(component_rows["effective_source_reliability_score"].dropna().mean())
        if not component_rows.empty and component_rows["effective_source_reliability_score"].notna().any()
        else float("nan")
    )
    top_component_issues = (
        component_rows.loc[~component_rows["row_schema_valid"], ["external_scenario_id", "source_id", "component_type", "row_completeness_fraction"]]
        .head(10)
        .to_string(index=False)
        if invalid_component_count > 0
        else "No invalid component rows."
    )
    scenario_summary = (
        scenario_rows[
            [
                "external_scenario_id",
                "mass_fraction_sum",
                "mass_fraction_sum_ok",
                "scenario_completeness_fraction",
                "effective_condition_similarity_score",
                "effective_source_reliability_score",
                "usable_for_supervised_augmentation",
            ]
        ].head(12).to_string(index=False)
        if not scenario_rows.empty
        else "No scenario rows loaded yet."
    )
    lines = [
        "# External Data Audit",
        "",
        "## Scope",
        "- External ingestion is staged only for sidecar experimentation.",
        "- The live 0.104084 platform anchor and the official shipping path remain untouched.",
        "- This audit validates schema, taxonomy, condition similarity, source reliability, completeness, and duplicate safety before any modeling use.",
        "",
        "## Summary",
        f"- Component rows loaded: `{len(component_rows)}`",
        f"- External scenarios loaded: `{len(scenario_rows)}`",
        f"- Invalid component rows: `{invalid_component_count}`",
        f"- Duplicate scenario/component rows: `{duplicate_component_count}`",
        f"- Usable supervised augmentation scenarios: `{usable_scenario_count}`",
        f"- Mean effective condition similarity score: `{average_similarity:.3f}`" if np.isfinite(average_similarity) else "- Mean effective condition similarity score: `n/a`",
        f"- Mean effective source reliability score: `{average_reliability:.3f}`" if np.isfinite(average_reliability) else "- Mean effective source reliability score: `n/a`",
        "",
        "## Validation Rules",
        f"- Required columns enforced: `{len(EXTERNAL_REQUIRED_COLUMNS)}`",
        f"- Allowed component taxonomy size: `{len(EXTERNAL_COMPONENT_TAXONOMY)}`",
        f"- Scenario mass-sum tolerance: `100 +/- {MASS_SUM_TOLERANCE:.1f}`",
        f"- Minimum completeness for supervised use: `{MIN_SCENARIO_COMPLETENESS:.2f}`",
        f"- Minimum condition similarity for supervised use: `{MIN_CONDITION_SIMILARITY:.2f}`",
        f"- Minimum source reliability for supervised use: `{MIN_SOURCE_RELIABILITY:.2f}`",
        "",
        "## Scenario Snapshot",
        "",
        "```text",
        scenario_summary,
        "```",
        "",
        "## Top Component-Level Issues",
        "",
        "```text",
        top_component_issues,
        "```",
        "",
        "## Readiness",
        (
            "- The schema is ready for manual extraction; the next step is to add high-quality records from the priority source catalog."
            if len(component_rows) == 0
            else "- The staged dataset can be used only through the external augmentation path; invalid rows should be fixed before broader use."
        ),
    ]
    return "\n".join(lines) + "\n"


def build_external_data_artifacts(path: Path = EXTERNAL_COLLECTED_RECORDS_PATH) -> ExternalDataArtifacts:
    """Validate the external component-row file and build scenario-level artifacts."""

    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_component_rows = load_external_component_rows(path)
    component_rows = _build_component_validation_frame(raw_component_rows)
    scenario_rows = _build_external_scenario_rows(component_rows)
    audit_markdown = _build_audit_markdown(component_rows=component_rows, scenario_rows=scenario_rows)
    return ExternalDataArtifacts(
        component_rows=component_rows,
        scenario_rows=scenario_rows,
        audit_markdown=audit_markdown,
    )


def write_external_data_audit(path: Path = EXTERNAL_DATA_AUDIT_REPORT_OUTPUT_PATH) -> ExternalDataArtifacts:
    """Build artifacts and write the markdown audit report."""

    artifacts = build_external_data_artifacts()
    _write_text(artifacts.audit_markdown, path)
    return artifacts


def main() -> None:
    artifacts = write_external_data_audit()
    print(f"external_component_rows: {len(artifacts.component_rows)}")
    print(f"external_scenarios: {len(artifacts.scenario_rows)}")
    print(f"external_supervised_scenarios: {int(artifacts.scenario_rows['usable_for_supervised_augmentation'].sum())}")
    print(f"external_data_audit_report: {EXTERNAL_DATA_AUDIT_REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
