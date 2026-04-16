"""Validation audit and deterministic scenario-level feature generation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_MANIFEST_OUTPUT_PATH,
    PREPROCESSING_AUDIT_OUTPUT_PATH,
    PROPERTY_LONG_OUTPUT_PATH,
    SCENARIO_CONDITION_COLUMNS,
    TEST_JOINED_OUTPUT_PATH,
    TEST_SCENARIO_FEATURES_OUTPUT_PATH,
    TRAIN_JOINED_OUTPUT_PATH,
    TRAIN_SCENARIO_FEATURES_OUTPUT_PATH,
)
from src.data.prepare_properties import CYRILLIC_TO_LATIN


PROPERTY_PREFIX = "prop__"
TARGET_PREFIX = "target_"
ROW_FLAG_COLUMNS = [
    "batch_id_was_missing",
    "used_exact_batch_props",
    "used_typical_fallback",
    "missing_all_props",
    "has_usable_property_coverage",
]
PROPERTY_JOIN_SOURCE_LEVELS = ["exact_only", "exact_plus_typical", "typical_only", "missing"]


@dataclass(frozen=True)
class FeatureSchema:
    """Shared schema needed to keep train/test feature generation compatible."""

    property_columns: list[str]
    component_families: list[str]
    component_family_slug_map: dict[str, str]


@dataclass(frozen=True)
class FeatureBuildResult:
    """Scenario-level outputs and metadata for downstream modeling."""

    train_features: pd.DataFrame
    test_features: pd.DataFrame
    feature_manifest: dict[str, object]
    audit_report_markdown: str


def _to_ascii_slug(value: str) -> str:
    """Convert text into a deterministic ASCII snake_case slug."""

    lowered = value.lower().strip()
    pieces: list[str] = []
    for char in lowered:
        if char in CYRILLIC_TO_LATIN:
            pieces.append(CYRILLIC_TO_LATIN[char])
        elif ord(char) < 128:
            pieces.append(char)
        else:
            pieces.append(" ")
    text = "".join(pieces)
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def extract_component_family(component_id: object) -> str:
    """Collapse component identifiers into stable family labels."""

    if pd.isna(component_id):
        return "unknown_component_family"
    text = str(component_id).strip()
    return re.sub(r"_\d+$", "", text) or text


def _build_unique_slug_map(values: Iterable[str]) -> dict[str, str]:
    """Assign unique ASCII slugs while preserving deterministic ordering."""

    slug_counts: dict[str, int] = {}
    slug_map: dict[str, str] = {}
    for value in sorted(set(values)):
        base_slug = _to_ascii_slug(value)
        count = slug_counts.get(base_slug, 0)
        slug_counts[base_slug] = count + 1
        slug_map[value] = base_slug if count == 0 else f"{base_slug}_{count + 1}"
    return slug_map


def load_prepared_table(path: Path) -> pd.DataFrame:
    """Load a prepared mixture table and coerce stable dtypes used by the feature builder."""

    frame = pd.read_csv(path)
    for column in ROW_FLAG_COLUMNS:
        if column in frame.columns:
            frame[column] = (
                frame[column]
                .astype("string")
                .str.lower()
                .map({"true": True, "false": False})
                .fillna(False)
                .astype(bool)
            )

    if "catalyst_dosage_category" in frame.columns:
        frame["catalyst_dosage_category"] = pd.to_numeric(
            frame["catalyst_dosage_category"], errors="coerce"
        )

    return frame


def validate_prepared_scenarios(
    frame: pd.DataFrame, scenario_columns: Iterable[str] = SCENARIO_CONDITION_COLUMNS
) -> None:
    """Ensure scenario-level condition columns are constant within each scenario."""

    grouped = frame.groupby("scenario_id", dropna=False)
    invalid = [
        column
        for column in scenario_columns
        if column in frame.columns and (grouped[column].nunique(dropna=False) > 1).any()
    ]
    if invalid:
        raise ValueError(
            "Scenario condition columns vary within scenario_id: " + ", ".join(sorted(invalid))
        )


def infer_feature_schema(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> FeatureSchema:
    """Infer the shared property and component-family schema from prepared train/test rows."""

    combined_columns = set(train_rows.columns) | set(test_rows.columns)
    property_columns = sorted(column for column in combined_columns if column.startswith(PROPERTY_PREFIX))

    component_families: set[str] = set()
    for frame in (train_rows, test_rows):
        if "component_id" in frame.columns:
            component_families.update(frame["component_id"].dropna().map(extract_component_family).tolist())

    family_slug_map = _build_unique_slug_map(component_families)
    return FeatureSchema(
        property_columns=property_columns,
        component_families=sorted(component_families),
        component_family_slug_map=family_slug_map,
    )


def _scenario_index(frame: pd.DataFrame) -> pd.Index:
    """Return a deterministic sorted scenario index."""

    return pd.Index(sorted(frame["scenario_id"].dropna().unique()), name="scenario_id")


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a series to numeric dtype without mutating the caller."""

    return pd.to_numeric(series, errors="coerce")


def build_condition_features(frame: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Build one scenario-level row of condition features."""

    validate_prepared_scenarios(frame)
    condition_features = (
        frame.loc[:, ["scenario_id", *SCENARIO_CONDITION_COLUMNS]]
        .drop_duplicates(subset=["scenario_id"])
        .set_index("scenario_id")
        .reindex(index)
    )
    return condition_features


def build_structure_features(frame: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Build scenario-level component, batch, and mass-distribution statistics."""

    prepared = frame.copy()
    prepared["component_family"] = prepared["component_id"].map(extract_component_family)
    prepared["component_batch_key"] = (
        prepared["component_id"].astype("string") + "||" + prepared["batch_id"].astype("string")
    )

    grouped = prepared.groupby("scenario_id", sort=True, dropna=False)
    features = pd.DataFrame(index=index)
    features["component_row_count"] = grouped.size().reindex(index).fillna(0).astype(int)
    features["component_unique_count"] = grouped["component_id"].nunique(dropna=False).reindex(index).fillna(0).astype(int)
    features["component_family_unique_count"] = grouped["component_family"].nunique(dropna=False).reindex(index).fillna(0).astype(int)
    features["component_batch_unique_count"] = grouped["component_batch_key"].nunique(dropna=False).reindex(index).fillna(0).astype(int)
    features["batch_unique_count"] = grouped["batch_id"].nunique(dropna=False).reindex(index).fillna(0).astype(int)

    component_counts = prepared.groupby(["scenario_id", "component_id"], sort=True, dropna=False).size()
    component_batch_counts = prepared.groupby(["scenario_id", "component_batch_key"], sort=True, dropna=False).size()
    batch_counts = prepared.groupby(["scenario_id", "batch_id"], sort=True, dropna=False).size()

    features["repeated_component_count"] = component_counts.gt(1).groupby(level=0).sum().reindex(index).fillna(0).astype(int)
    features["max_component_repeat_count"] = component_counts.groupby(level=0).max().reindex(index).fillna(0).astype(int)
    features["repeated_component_batch_count"] = component_batch_counts.gt(1).groupby(level=0).sum().reindex(index).fillna(0).astype(int)
    features["max_component_batch_repeat_count"] = component_batch_counts.groupby(level=0).max().reindex(index).fillna(0).astype(int)
    features["repeated_batch_count"] = batch_counts.gt(1).groupby(level=0).sum().reindex(index).fillna(0).astype(int)
    features["max_batch_repeat_count"] = batch_counts.groupby(level=0).max().reindex(index).fillna(0).astype(int)

    features["repeated_component_row_count"] = features["component_row_count"] - features["component_unique_count"]
    features["repeated_component_batch_row_count"] = (
        features["component_row_count"] - features["component_batch_unique_count"]
    )
    features["repeated_batch_row_count"] = features["component_row_count"] - features["batch_unique_count"]

    mass_grouped = grouped["mass_fraction_pct"]
    features["mass_fraction_sum"] = mass_grouped.sum().reindex(index)
    features["mass_fraction_mean"] = mass_grouped.mean().reindex(index)
    features["mass_fraction_std"] = mass_grouped.apply(lambda values: values.std(ddof=0)).reindex(index)
    features["mass_fraction_min"] = mass_grouped.min().reindex(index)
    features["mass_fraction_max"] = mass_grouped.max().reindex(index)
    features["mass_fraction_median"] = mass_grouped.median().reindex(index)
    features["mass_fraction_q25"] = mass_grouped.quantile(0.25).reindex(index)
    features["mass_fraction_q75"] = mass_grouped.quantile(0.75).reindex(index)
    features["mass_fraction_range"] = features["mass_fraction_max"] - features["mass_fraction_min"]
    features["mass_fraction_top1"] = mass_grouped.apply(
        lambda values: values.sort_values(ascending=False).head(1).sum()
    ).reindex(index)
    features["mass_fraction_top2_sum"] = mass_grouped.apply(
        lambda values: values.sort_values(ascending=False).head(2).sum()
    ).reindex(index)
    features["mass_fraction_top3_sum"] = mass_grouped.apply(
        lambda values: values.sort_values(ascending=False).head(3).sum()
    ).reindex(index)

    total_mass = features["mass_fraction_sum"].replace(0, np.nan)
    features["mass_fraction_top1_share"] = (features["mass_fraction_top1"] / total_mass).fillna(0.0)
    features["mass_fraction_top3_share"] = (features["mass_fraction_top3_sum"] / total_mass).fillna(0.0)
    return features


def build_family_features(frame: pd.DataFrame, index: pd.Index, schema: FeatureSchema) -> pd.DataFrame:
    """Aggregate component family counts and mass shares at the scenario level."""

    prepared = frame.copy()
    prepared["component_family"] = prepared["component_id"].map(extract_component_family)
    total_mass = prepared.groupby("scenario_id", sort=True)["mass_fraction_pct"].sum().reindex(index)

    features = pd.DataFrame(index=index)
    for family in schema.component_families:
        slug = schema.component_family_slug_map[family]
        family_rows = prepared.loc[prepared["component_family"] == family]
        grouped = family_rows.groupby("scenario_id", sort=True, dropna=False)

        row_count = grouped.size().reindex(index).fillna(0).astype(int)
        mass_sum = grouped["mass_fraction_pct"].sum().reindex(index).fillna(0.0)
        unique_component_count = grouped["component_id"].nunique(dropna=False).reindex(index).fillna(0).astype(int)
        batch_unique_count = grouped["batch_id"].nunique(dropna=False).reindex(index).fillna(0).astype(int)
        mass_share = (mass_sum / total_mass.replace(0, np.nan)).fillna(0.0)

        features[f"family__{slug}__row_count"] = row_count
        features[f"family__{slug}__unique_component_count"] = unique_component_count
        features[f"family__{slug}__batch_unique_count"] = batch_unique_count
        features[f"family__{slug}__mass_sum"] = mass_sum
        features[f"family__{slug}__mass_share"] = mass_share

    return features


def build_coverage_features(
    frame: pd.DataFrame, index: pd.Index, property_columns: list[str]
) -> pd.DataFrame:
    """Aggregate row-level property coverage and missingness indicators."""

    grouped = frame.groupby("scenario_id", sort=True, dropna=False)
    row_count = grouped.size().reindex(index).fillna(0)
    total_mass = grouped["mass_fraction_pct"].sum().reindex(index).fillna(0.0)

    features = pd.DataFrame(index=index)
    features["exact_property_row_count"] = grouped["used_exact_batch_props"].sum().reindex(index).fillna(0).astype(int)
    features["typical_fallback_row_count"] = grouped["used_typical_fallback"].sum().reindex(index).fillna(0).astype(int)
    features["usable_property_row_count"] = grouped["has_usable_property_coverage"].sum().reindex(index).fillna(0).astype(int)
    features["missing_all_props_row_count"] = grouped["missing_all_props"].sum().reindex(index).fillna(0).astype(int)
    features["missing_batch_id_row_count"] = grouped["batch_id_was_missing"].sum().reindex(index).fillna(0).astype(int)

    row_denominator = row_count.replace(0, np.nan)
    features["exact_property_row_ratio"] = (features["exact_property_row_count"] / row_denominator).fillna(0.0)
    features["typical_fallback_row_ratio"] = (features["typical_fallback_row_count"] / row_denominator).fillna(0.0)
    features["usable_property_row_ratio"] = (features["usable_property_row_count"] / row_denominator).fillna(0.0)
    features["missing_all_props_row_ratio"] = (features["missing_all_props_row_count"] / row_denominator).fillna(0.0)
    features["missing_batch_id_row_ratio"] = (features["missing_batch_id_row_count"] / row_denominator).fillna(0.0)

    for flag_column in ["used_exact_batch_props", "used_typical_fallback", "has_usable_property_coverage", "missing_all_props"]:
        numerator = (
            frame["mass_fraction_pct"].where(frame[flag_column], 0.0).groupby(frame["scenario_id"]).sum().reindex(index).fillna(0.0)
        )
        features[f"{flag_column}__mass_share"] = (numerator / total_mass.replace(0, np.nan)).fillna(0.0)

    join_source_counts = (
        pd.crosstab(frame["scenario_id"], frame["property_join_source"])
        .reindex(index=index, columns=PROPERTY_JOIN_SOURCE_LEVELS, fill_value=0)
    )
    for level in PROPERTY_JOIN_SOURCE_LEVELS:
        features[f"property_join_source__{level}__row_count"] = join_source_counts[level].astype(int)
        features[f"property_join_source__{level}__row_ratio"] = (
            join_source_counts[level] / row_denominator
        ).fillna(0.0)

    if property_columns:
        property_count_matrix = grouped[property_columns].count().reindex(index).fillna(0)
        property_any_matrix = grouped[property_columns].apply(lambda values: values.notna().any(axis=0))
        property_any_matrix = property_any_matrix.reindex(index).fillna(False)
        features["property_column_nonmissing_any_count"] = property_any_matrix.sum(axis=1).astype(int)
        features["property_cell_nonmissing_count_total"] = property_count_matrix.sum(axis=1).astype(int)
        density_denominator = row_count * len(property_columns)
        features["property_cell_nonmissing_density"] = (
            features["property_cell_nonmissing_count_total"] / density_denominator.replace(0, np.nan)
        ).fillna(0.0)
    else:
        features["property_column_nonmissing_any_count"] = 0
        features["property_cell_nonmissing_count_total"] = 0
        features["property_cell_nonmissing_density"] = 0.0

    return features


def build_weighted_property_features(
    frame: pd.DataFrame, index: pd.Index, property_columns: list[str]
) -> pd.DataFrame:
    """Aggregate numeric component properties into scenario-level weighted features."""

    feature_columns: dict[str, pd.Series] = {}
    total_mass = frame.groupby("scenario_id", sort=True)["mass_fraction_pct"].sum().reindex(index).fillna(0.0)
    mass = _coerce_numeric_series(frame["mass_fraction_pct"])

    for property_column in property_columns:
        values = _coerce_numeric_series(frame[property_column])
        valid = values.notna() & mass.notna()

        row_count = valid.groupby(frame["scenario_id"]).sum().reindex(index).fillna(0).astype(int)
        nonmissing_mass = mass.where(valid, 0.0).groupby(frame["scenario_id"]).sum().reindex(index).fillna(0.0)
        weighted_sum = (
            (values.where(valid, 0.0) * mass.where(valid, 0.0))
            .groupby(frame["scenario_id"])
            .sum()
            .reindex(index)
            .fillna(0.0)
        )
        weighted_mean = (weighted_sum / nonmissing_mass.replace(0, np.nan)).reindex(index)
        nonmissing_mass_share = (nonmissing_mass / total_mass.replace(0, np.nan)).reindex(index).fillna(0.0)

        feature_columns[f"{property_column}__nonmissing_row_count"] = row_count
        feature_columns[f"{property_column}__weighted_sum"] = weighted_sum
        feature_columns[f"{property_column}__weighted_mean"] = weighted_mean
        feature_columns[f"{property_column}__nonmissing_mass_share"] = nonmissing_mass_share

    return pd.DataFrame(feature_columns, index=index)


def build_scenario_features(frame: pd.DataFrame, schema: FeatureSchema) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build deterministic scenario-level features from prepared mixture rows."""

    if "scenario_id" not in frame.columns:
        raise KeyError("Prepared mixture rows must contain scenario_id.")

    unexpected_targets = [column for column in frame.columns if column.startswith(TARGET_PREFIX)]
    working = frame.drop(columns=unexpected_targets, errors="ignore").copy()

    index = _scenario_index(working)
    feature_groups = OrderedDict()
    feature_groups["scenario_conditions"] = build_condition_features(working, index)
    feature_groups["structure_and_mass"] = build_structure_features(working, index)
    feature_groups["component_families"] = build_family_features(working, index, schema)
    feature_groups["coverage_and_missingness"] = build_coverage_features(
        working, index, schema.property_columns
    )
    feature_groups["weighted_numeric_properties"] = build_weighted_property_features(
        working, index, schema.property_columns
    )

    scenario_features = pd.concat(feature_groups.values(), axis=1).copy().reset_index()
    if any(column.startswith(TARGET_PREFIX) for column in scenario_features.columns):
        raise ValueError("Scenario features unexpectedly contain target columns.")
    if scenario_features["scenario_id"].duplicated().any():
        raise ValueError("Scenario feature table contains duplicate scenario_id values.")

    manifest_groups = {
        group_name: list(group_frame.columns)
        for group_name, group_frame in feature_groups.items()
    }
    return scenario_features, manifest_groups


def build_preprocessing_audit_report(
    train_rows: pd.DataFrame, test_rows: pd.DataFrame, clean_properties: pd.DataFrame
) -> str:
    """Produce a short markdown audit of the prepared preprocessing layer."""

    def summarize_rows(name: str, frame: pd.DataFrame) -> dict[str, object]:
        row_count = len(frame)
        return {
            "split": name,
            "rows": row_count,
            "scenarios": frame["scenario_id"].nunique(),
            "exact_property_rows": int(frame["used_exact_batch_props"].sum()),
            "exact_property_rows_pct": 100.0 * float(frame["used_exact_batch_props"].mean()),
            "usable_numeric_rows": int(frame["has_usable_property_coverage"].sum()),
            "usable_numeric_rows_pct": 100.0 * float(frame["has_usable_property_coverage"].mean()),
            "typical_fallback_rows": int(frame["used_typical_fallback"].sum()),
            "typical_fallback_rows_pct": 100.0 * float(frame["used_typical_fallback"].mean()),
            "missing_all_props_rows": int(frame["missing_all_props"].sum()),
            "missing_all_props_rows_pct": 100.0 * float(frame["missing_all_props"].mean()),
            "missing_batch_id_rows": int(frame["batch_id_was_missing"].sum()),
            "missing_batch_id_rows_pct": 100.0 * float(frame["batch_id_was_missing"].mean()),
        }

    summary_rows = [summarize_rows("train", train_rows), summarize_rows("test", test_rows)]
    summary_table = pd.DataFrame(summary_rows)

    valid_clean_rows = int(clean_properties["property_name"].notna().all() and clean_properties["property_value"].notna().all())
    multi_unit = (
        clean_properties.assign(property_unit_display=clean_properties["property_unit"].fillna("<missing_unit>"))
        .groupby("property_name", dropna=False)["property_unit_display"]
        .agg(lambda values: sorted(set(values)))
        .reset_index()
    )
    multi_unit = multi_unit.loc[multi_unit["property_unit_display"].map(len) > 1].copy()

    lines = [
        "# Preprocessing Audit",
        "",
        "## Clean Property Table Validation",
        f"- `component_properties_clean.csv` rows: {len(clean_properties)}",
        f"- Blank `property_name` rows: {int(clean_properties['property_name'].isna().sum())}",
        f"- Blank `property_value` rows: {int(clean_properties['property_value'].isna().sum())}",
        f"- Blank/invalid property rows excluded successfully: {'yes' if valid_clean_rows else 'no'}",
        "",
        "## Coverage Summary",
        "",
        "| split | rows | scenarios | exact property rows | usable numeric rows | typical fallback rows | missing_all_props rows | missing batch_id rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary_rows:
        lines.append(
            "| {split} | {rows} | {scenarios} | {exact_property_rows} ({exact_property_rows_pct:.1f}%) | "
            "{usable_numeric_rows} ({usable_numeric_rows_pct:.1f}%) | {typical_fallback_rows} ({typical_fallback_rows_pct:.1f}%) | "
            "{missing_all_props_rows} ({missing_all_props_rows_pct:.1f}%) | {missing_batch_id_rows} ({missing_batch_id_rows_pct:.1f}%) |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "- `used_exact_batch_props` tracks an exact component/batch lookup row, not guaranteed full numeric coverage.",
            "- `has_usable_property_coverage` is the stronger numeric-coverage flag after exact-plus-typical coalescing.",
            "",
            "## Property Names With Multiple Units",
        ]
    )

    if multi_unit.empty:
        lines.append("- None")
    else:
        for row in multi_unit.sort_values("property_name").itertuples(index=False):
            units = ", ".join(f"`{unit}`" for unit in row.property_unit_display)
            lines.append(f"- `{row.property_name}`: {units}")

    return "\n".join(lines) + "\n"


def _write_markdown(text: str, path: Path) -> Path:
    """Write markdown content to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Persist a feature table to disk with deterministic row ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values("scenario_id").to_csv(path, index=False)
    return path


def _write_json(payload: dict[str, object], path: Path) -> Path:
    """Write JSON metadata to disk with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_train_test_scenario_features(
    train_rows: pd.DataFrame, test_rows: pd.DataFrame, clean_properties: pd.DataFrame
) -> FeatureBuildResult:
    """Build the audit report plus train/test scenario features from prepared inputs."""

    schema = infer_feature_schema(train_rows, test_rows)
    train_features, train_groups = build_scenario_features(train_rows, schema)
    test_features, test_groups = build_scenario_features(test_rows, schema)

    if train_features.columns.tolist() != test_features.columns.tolist():
        raise ValueError("Train and test scenario features are not schema-compatible.")

    audit_report_markdown = build_preprocessing_audit_report(train_rows, test_rows, clean_properties)
    manifest = {
        "feature_table_columns": train_features.columns.tolist(),
        "feature_group_columns": train_groups,
        "feature_group_column_counts": {name: len(columns) for name, columns in train_groups.items()},
        "property_columns": schema.property_columns,
        "component_family_slug_map": schema.component_family_slug_map,
        "excluded_target_columns": [
            column for column in train_rows.columns if column.startswith(TARGET_PREFIX)
        ],
        "train_scenarios": int(train_features["scenario_id"].nunique()),
        "test_scenarios": int(test_features["scenario_id"].nunique()),
    }
    manifest["feature_group_columns_test"] = test_groups

    return FeatureBuildResult(
        train_features=train_features,
        test_features=test_features,
        feature_manifest=manifest,
        audit_report_markdown=audit_report_markdown,
    )


def run_feature_pipeline() -> FeatureBuildResult:
    """Run the preprocessing audit and scenario-level feature generation pipeline."""

    train_rows = load_prepared_table(TRAIN_JOINED_OUTPUT_PATH)
    test_rows = load_prepared_table(TEST_JOINED_OUTPUT_PATH)
    clean_properties = pd.read_csv(PROPERTY_LONG_OUTPUT_PATH)

    result = build_train_test_scenario_features(train_rows, test_rows, clean_properties)
    _write_markdown(result.audit_report_markdown, PREPROCESSING_AUDIT_OUTPUT_PATH)
    _write_csv(result.train_features, TRAIN_SCENARIO_FEATURES_OUTPUT_PATH)
    _write_csv(result.test_features, TEST_SCENARIO_FEATURES_OUTPUT_PATH)
    _write_json(result.feature_manifest, FEATURE_MANIFEST_OUTPUT_PATH)
    return result


def main() -> None:
    """CLI entrypoint for the audit and scenario feature build."""

    result = run_feature_pipeline()
    print(f"audit_report: {PREPROCESSING_AUDIT_OUTPUT_PATH}")
    print(f"train_features: {TRAIN_SCENARIO_FEATURES_OUTPUT_PATH} shape={result.train_features.shape}")
    print(f"test_features: {TEST_SCENARIO_FEATURES_OUTPUT_PATH} shape={result.test_features.shape}")
    print(f"feature_manifest: {FEATURE_MANIFEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
