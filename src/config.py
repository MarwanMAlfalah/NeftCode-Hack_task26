"""Project-wide configuration for deterministic data preparation."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CV_OUTPUTS_DIR = OUTPUTS_DIR / "cv"
REPORTS_DIR = OUTPUTS_DIR / "reports"
EXTERNAL_DATA_DIR = PROJECT_ROOT / "external_data"

TRAIN_MIXTURES_FILENAME = "daimler_mixtures_train.csv"
TEST_MIXTURES_FILENAME = "daimler_mixtures_test.csv"
COMPONENT_PROPERTIES_FILENAME = "daimler_component_properties.csv"

TRAIN_MIXTURES_PATH = RAW_DIR / TRAIN_MIXTURES_FILENAME
TEST_MIXTURES_PATH = RAW_DIR / TEST_MIXTURES_FILENAME
COMPONENT_PROPERTIES_PATH = RAW_DIR / COMPONENT_PROPERTIES_FILENAME

TYPICAL_BATCH_ID = "typical"
MISSING_BATCH_ID_TOKEN = "__missing_batch__"

MIXTURE_COLUMN_RENAMES = {
    "scenario_id": "scenario_id",
    "Компонент": "component_id",
    "Наименование партии": "batch_id",
    "Массовая доля, %": "mass_fraction_pct",
    "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C": "test_temperature_c",
    "Время испытания | - Daimler Oxidation Test (DOT), ч": "test_duration_h",
    "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс": "biofuel_mass_fraction_pct",
    "Дозировка катализатора, категория": "catalyst_dosage_category",
}

TRAIN_TARGET_COLUMN_RENAMES = {
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %": "target_delta_kinematic_viscosity_pct",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm": "target_oxidation_eot_a_per_cm",
}

PROPERTY_COLUMN_RENAMES = {
    "Компонент": "component_id",
    "Наименование партии": "batch_id",
    "Наименование показателя": "property_name",
    "Единица измерения_по_партиям": "property_unit",
    "Значение показателя": "property_value",
}

MIXTURE_NUMERIC_COLUMNS = [
    "mass_fraction_pct",
    "test_temperature_c",
    "test_duration_h",
    "biofuel_mass_fraction_pct",
]

TRAIN_TARGET_COLUMNS = [
    "target_delta_kinematic_viscosity_pct",
    "target_oxidation_eot_a_per_cm",
]

SCENARIO_CONDITION_COLUMNS = [
    "test_temperature_c",
    "test_duration_h",
    "biofuel_mass_fraction_pct",
    "catalyst_dosage_category",
]

MIXTURE_ID_COLUMNS = [
    "scenario_id",
    "component_id",
    "batch_id",
]

PROPERTY_ID_COLUMNS = [
    "component_id",
    "batch_id",
]

PROPERTY_LONG_OUTPUT_PATH = INTERIM_DIR / "component_properties_clean.csv"
PROPERTY_CATALOG_OUTPUT_PATH = INTERIM_DIR / "component_property_catalog.csv"
PROPERTY_PIVOT_ALL_OUTPUT_PATH = INTERIM_DIR / "component_properties_pivot_all.csv"
PROPERTY_EXACT_OUTPUT_PATH = INTERIM_DIR / "component_properties_lookup_exact.csv"
PROPERTY_TYPICAL_OUTPUT_PATH = INTERIM_DIR / "component_properties_lookup_typical.csv"

TRAIN_NORMALIZED_OUTPUT_PATH = INTERIM_DIR / "mixtures_train_normalized.csv"
TEST_NORMALIZED_OUTPUT_PATH = INTERIM_DIR / "mixtures_test_normalized.csv"
TRAIN_JOINED_OUTPUT_PATH = INTERIM_DIR / "mixtures_train_with_properties.csv"
TEST_JOINED_OUTPUT_PATH = INTERIM_DIR / "mixtures_test_with_properties.csv"

TRAIN_TARGETS_OUTPUT_PATH = PROCESSED_DIR / "train_scenario_targets.csv"
TRAIN_SCENARIO_FEATURES_OUTPUT_PATH = PROCESSED_DIR / "train_scenario_features.csv"
TEST_SCENARIO_FEATURES_OUTPUT_PATH = PROCESSED_DIR / "test_scenario_features.csv"
FEATURE_MANIFEST_OUTPUT_PATH = PROCESSED_DIR / "feature_manifest.json"
PREPROCESSING_AUDIT_OUTPUT_PATH = REPORTS_DIR / "preprocessing_audit.md"
BASELINE_CV_RESULTS_OUTPUT_PATH = CV_OUTPUTS_DIR / "baseline_cv_results.csv"
BASELINE_FOLD_METRICS_OUTPUT_PATH = CV_OUTPUTS_DIR / "baseline_fold_metrics.csv"
BASELINE_REPORT_OUTPUT_PATH = REPORTS_DIR / "baseline_report.md"
BASELINE_ABLATION_RESULTS_OUTPUT_PATH = CV_OUTPUTS_DIR / "baseline_ablation_results.csv"
BASELINE_ABLATION_REPORT_OUTPUT_PATH = REPORTS_DIR / "baseline_ablation_report.md"
DEEP_SETS_CV_RESULTS_OUTPUT_PATH = CV_OUTPUTS_DIR / "deep_sets_cv_results.csv"
DEEP_SETS_FOLD_METRICS_OUTPUT_PATH = CV_OUTPUTS_DIR / "deep_sets_fold_metrics.csv"
DEEP_SETS_REPORT_OUTPUT_PATH = REPORTS_DIR / "deep_sets_report.md"
DEEP_SETS_V2_CV_RESULTS_OUTPUT_PATH = CV_OUTPUTS_DIR / "deep_sets_v2_cv_results.csv"
DEEP_SETS_V2_FOLD_METRICS_OUTPUT_PATH = CV_OUTPUTS_DIR / "deep_sets_v2_fold_metrics.csv"
DEEP_SETS_V2_REPORT_OUTPUT_PATH = REPORTS_DIR / "deep_sets_v2_report.md"
FINAL_MODEL_SELECTION_RESULTS_OUTPUT_PATH = CV_OUTPUTS_DIR / "final_model_selection_results.csv"
FINAL_MODEL_SELECTION_REPORT_OUTPUT_PATH = REPORTS_DIR / "final_model_selection_report.md"
STABILITY_SPRINT_RESULTS_OUTPUT_PATH = CV_OUTPUTS_DIR / "stability_sprint_results.csv"
STABILITY_SPRINT_REPORT_OUTPUT_PATH = REPORTS_DIR / "stability_sprint_report.md"
TARGET_SPECIALIST_RESULTS_OUTPUT_PATH = CV_OUTPUTS_DIR / "target_specialist_results.csv"
TARGET_SPECIALIST_REPORT_OUTPUT_PATH = REPORTS_DIR / "target_specialist_report.md"
CHEMISTRY_ENSEMBLE_RESULTS_OUTPUT_PATH = CV_OUTPUTS_DIR / "chemistry_ensemble_results.csv"
CHEMISTRY_ENSEMBLE_REPORT_OUTPUT_PATH = REPORTS_DIR / "chemistry_ensemble_report.md"
LOCAL_RECALIBRATION_RESULTS_OUTPUT_PATH = CV_OUTPUTS_DIR / "local_recalibration_results.csv"
LOCAL_RECALIBRATION_REPORT_OUTPUT_PATH = REPORTS_DIR / "local_recalibration_report.md"
OBJECTIVE_ALIGNMENT_RESULTS_OUTPUT_PATH = CV_OUTPUTS_DIR / "objective_alignment_results.csv"
OBJECTIVE_ALIGNMENT_REPORT_OUTPUT_PATH = REPORTS_DIR / "objective_alignment_report.md"
OBJECTIVE_ALIGNMENT_FIXED_METRIC_RESULTS_OUTPUT_PATH = CV_OUTPUTS_DIR / "objective_alignment_fixed_metric_results.csv"
OBJECTIVE_ALIGNMENT_FIXED_METRIC_REPORT_OUTPUT_PATH = REPORTS_DIR / "objective_alignment_fixed_metric_report.md"
GP_ENSEMBLE_RESULTS_OUTPUT_PATH = CV_OUTPUTS_DIR / "gp_ensemble_results.csv"
GP_ENSEMBLE_REPORT_OUTPUT_PATH = REPORTS_DIR / "gp_ensemble_report.md"
GP_STAGE2_OOF_PREDICTIONS_OUTPUT_PATH = CV_OUTPUTS_DIR / "gp_stage2_oof_predictions.csv"
META_STACK_SEARCH_RESULTS_OUTPUT_PATH = CV_OUTPUTS_DIR / "meta_stack_search_results.csv"
PAIRED_BOOTSTRAP_CI_OUTPUT_PATH = CV_OUTPUTS_DIR / "paired_bootstrap_ci_stage15_vs_best_meta.csv"
GP_STAGE2_REGIME_AUDIT_REPORT_OUTPUT_PATH = REPORTS_DIR / "gp_stage2_regime_audit.md"
META_STACK_SEARCH_REPORT_OUTPUT_PATH = REPORTS_DIR / "meta_stack_search_report.md"
PAIRED_BOOTSTRAP_CI_REPORT_OUTPUT_PATH = REPORTS_DIR / "paired_bootstrap_ci_stage15_vs_best_meta.md"
HUBER_WEIGHT_GRID_RESULTS_OUTPUT_PATH = CV_OUTPUTS_DIR / "huber_weight_grid_results.csv"
HUBER_WEIGHT_GRID_REPORT_OUTPUT_PATH = REPORTS_DIR / "huber_weight_grid_report.md"
EXTERNAL_COLLECTED_RECORDS_PATH = EXTERNAL_DATA_DIR / "collected_records.csv"
EXTERNAL_SOURCE_CATALOG_PATH = EXTERNAL_DATA_DIR / "source_catalog.md"
EXTERNAL_DATA_AUDIT_REPORT_OUTPUT_PATH = REPORTS_DIR / "external_data_audit.md"
EXTERNAL_AUGMENTED_RESULTS_OUTPUT_PATH = CV_OUTPUTS_DIR / "external_augmented_results.csv"
EXTERNAL_AUGMENTED_REPORT_OUTPUT_PATH = REPORTS_DIR / "external_augmented_report.md"

RANDOM_SEED = 42
