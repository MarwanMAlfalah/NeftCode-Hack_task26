"""Property preparation and deterministic exact-or-typical joins for mixtures."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd

from src.config import (
    INTERIM_DIR,
    MISSING_BATCH_ID_TOKEN,
    PROPERTY_CATALOG_OUTPUT_PATH,
    PROPERTY_EXACT_OUTPUT_PATH,
    PROPERTY_ID_COLUMNS,
    PROPERTY_LONG_OUTPUT_PATH,
    PROPERTY_PIVOT_ALL_OUTPUT_PATH,
    PROPERTY_TYPICAL_OUTPUT_PATH,
    PROCESSED_DIR,
    TEST_JOINED_OUTPUT_PATH,
    TEST_NORMALIZED_OUTPUT_PATH,
    TRAIN_JOINED_OUTPUT_PATH,
    TRAIN_NORMALIZED_OUTPUT_PATH,
    TYPICAL_BATCH_ID,
)
from src.data.load_raw import RawDatasets, load_all_raw_datasets
from src.data.prepare_targets import build_train_scenario_targets, save_train_scenario_targets


CYRILLIC_TO_LATIN = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "i",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


@dataclass(frozen=True)
class PreparedPropertyArtifacts:
    """Prepared property tables ready for deterministic joins."""

    clean_long: pd.DataFrame
    property_catalog: pd.DataFrame
    pivot_all: pd.DataFrame
    exact_lookup: pd.DataFrame
    typical_lookup: pd.DataFrame


@dataclass(frozen=True)
class PreparationOutputs:
    """Paths written by the end-to-end ingestion and preparation flow."""

    written_files: dict[str, Path]


def _normalize_batch_ids(series: pd.Series) -> pd.Series:
    """Apply the project's explicit missing-batch policy.

    Missing batch IDs are preserved as a deterministic token instead of being dropped.
    This prevents accidental null joins while still allowing:
    1. exact matching on explicitly missing batch IDs when both sides are missing
    2. per-property fallback to the component's `typical` batch when exact values are unavailable
    """

    normalized = series.copy()
    return normalized.fillna(MISSING_BATCH_ID_TOKEN)


def _transliterate_to_ascii(value: str) -> str:
    """Transliterate Russian text into an ASCII-friendly representation."""

    output: list[str] = []
    for char in unicodedata.normalize("NFKD", value.lower()):
        if unicodedata.combining(char):
            continue
        if char in CYRILLIC_TO_LATIN:
            output.append(CYRILLIC_TO_LATIN[char])
        elif ord(char) < 128:
            output.append(char)
        else:
            output.append(" ")
    return "".join(output)


def _property_slug(property_name: str, property_unit: str | None) -> str:
    """Build a deterministic ASCII slug for a property descriptor."""

    parts = [property_name]
    if property_unit:
        parts.append(property_unit)
    text = _transliterate_to_ascii(" ".join(parts))
    text = text.replace("%", " pct ")
    text = text.replace("°", " deg ")
    text = text.replace("²", "2")
    text = text.replace("³", "3")
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "property"


def _sorted_output(frame: pd.DataFrame, sort_columns: list[str]) -> pd.DataFrame:
    """Return a copy sorted by the provided columns when they exist."""

    available = [column for column in sort_columns if column in frame.columns]
    if not available:
        return frame.reset_index(drop=True)
    return frame.sort_values(available, na_position="last").reset_index(drop=True)


def prepare_mixtures_for_property_join(mixtures: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixture rows so property joins are explicit and reproducible."""

    prepared = mixtures.copy()
    batch_missing = prepared["batch_id"].isna()
    if "batch_id_was_missing" in prepared.columns:
        batch_missing = batch_missing | prepared["batch_id_was_missing"].fillna(False).astype(bool)
    prepared["batch_id_was_missing"] = batch_missing
    prepared["batch_id"] = _normalize_batch_ids(prepared["batch_id"])
    return _sorted_output(prepared, ["scenario_id", "component_id", "batch_id"])


def clean_component_properties(component_properties: pd.DataFrame) -> pd.DataFrame:
    """Drop blank property rows while preserving original names, units, and raw values."""

    required = {"component_id", "batch_id", "property_name", "property_unit", "property_value", "property_value_numeric"}
    missing = required.difference(component_properties.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise KeyError(f"Component-properties table is missing required columns: {joined}")

    prepared = component_properties.copy()
    prepared["batch_id_was_missing"] = prepared["batch_id"].isna()
    prepared["batch_id"] = _normalize_batch_ids(prepared["batch_id"])

    valid_rows = (
        prepared["component_id"].notna()
        & prepared["property_name"].notna()
        & prepared["property_value"].notna()
    )

    clean = prepared.loc[
        valid_rows,
        [
            "component_id",
            "batch_id",
            "batch_id_was_missing",
            "property_name",
            "property_unit",
            "property_value",
            "property_value_numeric",
        ],
    ].copy()
    clean["is_numeric_property_value"] = clean["property_value_numeric"].notna()
    return _sorted_output(clean, ["component_id", "batch_id", "property_name", "property_unit"])


def build_numeric_property_catalog(clean_properties: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic catalog for numeric property pivot columns."""

    numeric_descriptors = clean_properties.loc[
        clean_properties["is_numeric_property_value"], ["property_name", "property_unit"]
    ].copy()
    numeric_descriptors["property_unit_normalized"] = numeric_descriptors["property_unit"].fillna("")
    numeric_descriptors = (
        numeric_descriptors.drop_duplicates()
        .sort_values(["property_name", "property_unit_normalized"])
        .reset_index(drop=True)
    )

    slug_counts: dict[str, int] = {}
    records: list[dict[str, object]] = []
    for row in numeric_descriptors.itertuples(index=False):
        property_unit = row.property_unit if row.property_unit_normalized else None
        base_slug = f"prop__{_property_slug(row.property_name, property_unit)}"
        count = slug_counts.get(base_slug, 0)
        slug_counts[base_slug] = count + 1
        property_key = base_slug if count == 0 else f"{base_slug}_{count + 1}"
        records.append(
            {
                "property_key": property_key,
                "property_name": row.property_name,
                "property_unit": property_unit or pd.NA,
            }
        )

    return pd.DataFrame.from_records(records, columns=["property_key", "property_name", "property_unit"])


def pivot_numeric_properties(
    clean_properties: pd.DataFrame, property_catalog: pd.DataFrame
) -> pd.DataFrame:
    """Pivot numeric component properties to one row per (component_id, batch_id)."""

    if property_catalog.empty:
        return pd.DataFrame(columns=[*PROPERTY_ID_COLUMNS])

    numeric_properties = clean_properties.copy()
    numeric_properties["property_unit_merge"] = numeric_properties["property_unit"].fillna("")
    catalog = property_catalog.copy()
    catalog["property_unit_merge"] = catalog["property_unit"].fillna("")

    numeric_properties = numeric_properties.merge(
        catalog,
        left_on=["property_name", "property_unit_merge"],
        right_on=["property_name", "property_unit_merge"],
        how="inner",
        validate="many_to_one",
    )

    pivot = (
        numeric_properties.pivot_table(
            index=PROPERTY_ID_COLUMNS,
            columns="property_key",
            values="property_value_numeric",
            aggfunc="first",
        )
        .sort_index(axis=1)
        .reset_index()
    )
    pivot.columns.name = None
    return _sorted_output(pivot, ["component_id", "batch_id"])


def build_property_artifacts(component_properties: pd.DataFrame) -> PreparedPropertyArtifacts:
    """Prepare clean long properties plus exact and typical numeric lookups."""

    clean_long = clean_component_properties(component_properties)
    property_catalog = build_numeric_property_catalog(clean_long)
    pivot_all = pivot_numeric_properties(clean_long, property_catalog)
    exact_lookup = _sorted_output(
        pivot_all.loc[pivot_all["batch_id"] != TYPICAL_BATCH_ID].copy(),
        ["component_id", "batch_id"],
    )
    typical_lookup = _sorted_output(
        pivot_all.loc[pivot_all["batch_id"] == TYPICAL_BATCH_ID].drop(columns=["batch_id"]).copy(),
        ["component_id"],
    )

    return PreparedPropertyArtifacts(
        clean_long=clean_long,
        property_catalog=property_catalog,
        pivot_all=pivot_all,
        exact_lookup=exact_lookup,
        typical_lookup=typical_lookup,
    )


def join_properties_to_mixtures(
    mixtures: pd.DataFrame,
    exact_lookup: pd.DataFrame,
    typical_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Attach numeric properties with exact-first, typical-second coalescing.

    Exact match policy:
    - normalize missing batch IDs to `__missing_batch__`
    - merge on (component_id, batch_id)

    Fallback policy:
    - for each numeric property column, keep the exact batch value when present
    - otherwise use the component's `typical` batch value for that property
    """

    prepared = prepare_mixtures_for_property_join(mixtures)

    exact_property_columns = [column for column in exact_lookup.columns if column not in PROPERTY_ID_COLUMNS]
    typical_property_columns = [column for column in typical_lookup.columns if column != "component_id"]
    property_columns = sorted(set(exact_property_columns) | set(typical_property_columns))

    exact_lookup_renamed = exact_lookup.copy()
    exact_lookup_renamed["exact_property_key_match"] = True
    exact_lookup_renamed = exact_lookup_renamed.rename(
        columns={column: f"{column}__exact" for column in exact_property_columns}
    )

    typical_lookup_renamed = typical_lookup.copy()
    typical_lookup_renamed["typical_property_key_match"] = True
    typical_lookup_renamed = typical_lookup_renamed.rename(
        columns={column: f"{column}__typical" for column in typical_property_columns}
    )

    joined = prepared.merge(
        exact_lookup_renamed,
        on=["component_id", "batch_id"],
        how="left",
        validate="many_to_one",
    )
    joined = joined.merge(
        typical_lookup_renamed,
        on=["component_id"],
        how="left",
        validate="many_to_one",
    )

    fallback_usage_masks: list[pd.Series] = []
    for column in property_columns:
        exact_column = f"{column}__exact"
        typical_column = f"{column}__typical"

        exact_values = joined[exact_column] if exact_column in joined.columns else pd.Series(pd.NA, index=joined.index)
        typical_values = (
            joined[typical_column] if typical_column in joined.columns else pd.Series(pd.NA, index=joined.index)
        )

        joined[column] = exact_values.combine_first(typical_values)
        fallback_usage_masks.append(exact_values.isna() & typical_values.notna())

    joined["used_exact_batch_props"] = joined.get("exact_property_key_match", False).fillna(False).astype(bool)
    if fallback_usage_masks:
        used_typical_fallback = pd.concat(fallback_usage_masks, axis=1).any(axis=1)
        joined["has_usable_property_coverage"] = joined[property_columns].notna().any(axis=1)
    else:
        used_typical_fallback = pd.Series(False, index=joined.index)
        joined["has_usable_property_coverage"] = False

    joined["used_typical_fallback"] = used_typical_fallback.astype(bool)
    joined["missing_all_props"] = ~joined["has_usable_property_coverage"]

    joined["property_join_source"] = "missing"
    joined.loc[joined["used_typical_fallback"] & ~joined["used_exact_batch_props"], "property_join_source"] = "typical_only"
    joined.loc[joined["used_exact_batch_props"] & ~joined["used_typical_fallback"], "property_join_source"] = "exact_only"
    joined.loc[joined["used_exact_batch_props"] & joined["used_typical_fallback"], "property_join_source"] = "exact_plus_typical"

    cleanup_columns = [
        column
        for column in joined.columns
        if column.endswith("__exact")
        or column.endswith("__typical")
        or column in {"exact_property_key_match", "typical_property_key_match"}
    ]
    joined = joined.drop(columns=cleanup_columns)
    return _sorted_output(joined, ["scenario_id", "component_id", "batch_id"])


def _write_csv(frame: pd.DataFrame, path: Path, sort_columns: list[str]) -> Path:
    """Write a CSV after deterministic sorting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    _sorted_output(frame, sort_columns).to_csv(path, index=False)
    return path


def _write_join_policy_metadata(path: Path) -> Path:
    """Persist a compact machine-readable description of the batch join policy."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "missing_batch_id_policy": {
            "normalized_token": MISSING_BATCH_ID_TOKEN,
            "behavior": "missing batch IDs are normalized to an explicit token; exact matches are attempted first, then typical values fill property-level gaps",
        },
        "property_join_order": ["exact_component_batch", "component_typical_fallback"],
        "coverage_flags": [
            "used_exact_batch_props",
            "used_typical_fallback",
            "missing_all_props",
            "has_usable_property_coverage",
            "property_join_source",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_preparation_pipeline(raw_datasets: RawDatasets | None = None) -> PreparationOutputs:
    """Run the ingestion and property-preparation flow end to end."""

    datasets = raw_datasets or load_all_raw_datasets()

    train_mixtures = prepare_mixtures_for_property_join(datasets.train_mixtures)
    test_mixtures = prepare_mixtures_for_property_join(datasets.test_mixtures)
    scenario_targets = build_train_scenario_targets(train_mixtures)
    property_artifacts = build_property_artifacts(datasets.component_properties)

    train_joined = join_properties_to_mixtures(
        train_mixtures,
        property_artifacts.exact_lookup,
        property_artifacts.typical_lookup,
    )
    test_joined = join_properties_to_mixtures(
        test_mixtures,
        property_artifacts.exact_lookup,
        property_artifacts.typical_lookup,
    )

    written_files = {
        "train_mixtures_normalized": _write_csv(train_mixtures, TRAIN_NORMALIZED_OUTPUT_PATH, ["scenario_id", "component_id", "batch_id"]),
        "test_mixtures_normalized": _write_csv(test_mixtures, TEST_NORMALIZED_OUTPUT_PATH, ["scenario_id", "component_id", "batch_id"]),
        "train_scenario_targets": save_train_scenario_targets(scenario_targets),
        "component_properties_clean": _write_csv(property_artifacts.clean_long, PROPERTY_LONG_OUTPUT_PATH, ["component_id", "batch_id", "property_name"]),
        "component_property_catalog": _write_csv(property_artifacts.property_catalog, PROPERTY_CATALOG_OUTPUT_PATH, ["property_key"]),
        "component_properties_pivot_all": _write_csv(property_artifacts.pivot_all, PROPERTY_PIVOT_ALL_OUTPUT_PATH, ["component_id", "batch_id"]),
        "component_properties_lookup_exact": _write_csv(property_artifacts.exact_lookup, PROPERTY_EXACT_OUTPUT_PATH, ["component_id", "batch_id"]),
        "component_properties_lookup_typical": _write_csv(property_artifacts.typical_lookup, PROPERTY_TYPICAL_OUTPUT_PATH, ["component_id"]),
        "train_mixtures_with_properties": _write_csv(train_joined, TRAIN_JOINED_OUTPUT_PATH, ["scenario_id", "component_id", "batch_id"]),
        "test_mixtures_with_properties": _write_csv(test_joined, TEST_JOINED_OUTPUT_PATH, ["scenario_id", "component_id", "batch_id"]),
        "property_join_policy": _write_join_policy_metadata(PROCESSED_DIR / "property_join_policy.json"),
    }
    return PreparationOutputs(written_files=written_files)


def main() -> None:
    """CLI entrypoint for the deterministic preparation flow."""

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outputs = run_preparation_pipeline()
    for name, path in outputs.written_files.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
