"""Tests for deterministic scenario-level feature generation."""

from __future__ import annotations

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.features.build_scenario_features import build_train_test_scenario_features


def _base_row(**overrides: object) -> dict[str, object]:
    row = {
        "scenario_id": "s1",
        "component_id": "Base_1",
        "batch_id": "b1",
        "mass_fraction_pct": 1.0,
        "test_temperature_c": 160.0,
        "test_duration_h": 168.0,
        "biofuel_mass_fraction_pct": 0.0,
        "catalyst_dosage_category": 1,
        "batch_id_was_missing": False,
        "used_exact_batch_props": False,
        "used_typical_fallback": False,
        "missing_all_props": False,
        "has_usable_property_coverage": True,
        "property_join_source": "exact_only",
        "prop__density": 10.0,
        "prop__viscosity": 1.0,
    }
    row.update(overrides)
    return row


class FeatureBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train_rows = pd.DataFrame(
            [
                _base_row(
                    scenario_id="s1",
                    component_id="Base_1",
                    batch_id="b1",
                    mass_fraction_pct=2.0,
                    used_exact_batch_props=True,
                    prop__density=10.0,
                    prop__viscosity=1.0,
                    target_delta_kinematic_viscosity_pct=25.4,
                    target_oxidation_eot_a_per_cm=98.04,
                ),
                _base_row(
                    scenario_id="s1",
                    component_id="Base_2",
                    batch_id="b2",
                    mass_fraction_pct=1.0,
                    used_typical_fallback=True,
                    used_exact_batch_props=False,
                    property_join_source="typical_only",
                    prop__density=20.0,
                    prop__viscosity=None,
                    target_delta_kinematic_viscosity_pct=25.4,
                    target_oxidation_eot_a_per_cm=98.04,
                ),
                _base_row(
                    scenario_id="s1",
                    component_id="Add_1",
                    batch_id="b3",
                    mass_fraction_pct=3.0,
                    has_usable_property_coverage=False,
                    missing_all_props=True,
                    property_join_source="missing",
                    prop__density=None,
                    prop__viscosity=None,
                    target_delta_kinematic_viscosity_pct=25.4,
                    target_oxidation_eot_a_per_cm=98.04,
                ),
                _base_row(
                    scenario_id="s2",
                    component_id="Add_2",
                    batch_id="b4",
                    mass_fraction_pct=4.0,
                    test_temperature_c=150.0,
                    biofuel_mass_fraction_pct=5.0,
                    catalyst_dosage_category=2,
                    used_exact_batch_props=True,
                    property_join_source="exact_only",
                    prop__density=5.0,
                    prop__viscosity=2.0,
                    target_delta_kinematic_viscosity_pct=11.0,
                    target_oxidation_eot_a_per_cm=77.7,
                ),
            ]
        )

        self.test_rows = pd.DataFrame(
            [
                _base_row(
                    scenario_id="t1",
                    component_id="Base_3",
                    batch_id="b5",
                    mass_fraction_pct=6.0,
                    used_typical_fallback=True,
                    used_exact_batch_props=False,
                    property_join_source="typical_only",
                    prop__density=30.0,
                    prop__viscosity=3.0,
                ),
                _base_row(
                    scenario_id="t1",
                    component_id="Add_3",
                    batch_id="b6",
                    mass_fraction_pct=2.0,
                    has_usable_property_coverage=False,
                    missing_all_props=True,
                    property_join_source="missing",
                    prop__density=None,
                    prop__viscosity=None,
                ),
            ]
        )

        self.clean_properties = pd.DataFrame(
            [
                {"property_name": "Density", "property_unit": "g/cm3", "property_value": "10.0"},
                {"property_name": "Density", "property_unit": "kg/m3", "property_value": "1000.0"},
                {"property_name": "Viscosity", "property_unit": "mm2/s", "property_value": "1.0"},
            ]
        )

    def test_feature_builder_returns_one_row_per_scenario_and_no_targets(self) -> None:
        result = build_train_test_scenario_features(
            self.train_rows, self.test_rows, self.clean_properties
        )

        self.assertEqual(result.train_features["scenario_id"].tolist(), ["s1", "s2"])
        self.assertEqual(result.test_features["scenario_id"].tolist(), ["t1"])
        self.assertTrue(result.train_features["scenario_id"].is_unique)
        self.assertTrue(result.test_features["scenario_id"].is_unique)
        self.assertEqual(
            result.train_features.columns.tolist(),
            result.test_features.columns.tolist(),
        )
        self.assertFalse(
            any(column.startswith("target_") for column in result.train_features.columns)
        )

    def test_weighted_property_and_family_aggregations_are_correct(self) -> None:
        result = build_train_test_scenario_features(
            self.train_rows, self.test_rows, self.clean_properties
        )
        scenario = result.train_features.set_index("scenario_id").loc["s1"]

        self.assertEqual(scenario["component_row_count"], 3)
        self.assertEqual(scenario["component_unique_count"], 3)
        self.assertEqual(scenario["exact_property_row_count"], 1)
        self.assertEqual(scenario["typical_fallback_row_count"], 1)
        self.assertEqual(scenario["missing_all_props_row_count"], 1)
        self.assertAlmostEqual(scenario["mass_fraction_sum"], 6.0)
        self.assertAlmostEqual(scenario["mass_fraction_top1_share"], 0.5)
        self.assertEqual(scenario["family__add__row_count"], 1)
        self.assertAlmostEqual(scenario["family__base__mass_sum"], 3.0)
        self.assertAlmostEqual(scenario["family__base__mass_share"], 0.5)
        self.assertAlmostEqual(scenario["prop__density__weighted_sum"], 40.0)
        self.assertAlmostEqual(scenario["prop__density__weighted_mean"], 40.0 / 3.0)
        self.assertAlmostEqual(scenario["prop__density__nonmissing_mass_share"], 0.5)
        self.assertEqual(scenario["prop__density__nonmissing_row_count"], 2)
        self.assertEqual(scenario["property_join_source__missing__row_count"], 1)

    def test_feature_builder_is_target_agnostic(self) -> None:
        original = build_train_test_scenario_features(
            self.train_rows, self.test_rows, self.clean_properties
        )

        changed_targets = self.train_rows.copy()
        changed_targets["target_delta_kinematic_viscosity_pct"] = [999.0, 999.0, 999.0, -5.0]
        changed_targets["target_oxidation_eot_a_per_cm"] = [0.1, 0.1, 0.1, 123.4]

        changed = build_train_test_scenario_features(
            changed_targets, self.test_rows, self.clean_properties
        )

        assert_frame_equal(original.train_features, changed.train_features)


if __name__ == "__main__":
    unittest.main()
