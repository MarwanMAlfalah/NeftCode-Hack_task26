"""Tests for scenario-level target table preparation."""

from __future__ import annotations

import unittest

import pandas as pd

from src.data.prepare_targets import build_train_scenario_targets


class PrepareTargetsTest(unittest.TestCase):
    def test_build_train_scenario_targets_deduplicates_by_scenario(self) -> None:
        train_rows = pd.DataFrame(
            [
                {
                    "scenario_id": "train_1",
                    "component_id": "a",
                    "batch_id": "1",
                    "mass_fraction_pct": 10.0,
                    "test_temperature_c": 160.0,
                    "test_duration_h": 168.0,
                    "biofuel_mass_fraction_pct": 0.0,
                    "catalyst_dosage_category": "1",
                    "target_delta_kinematic_viscosity_pct": 25.4,
                    "target_oxidation_eot_a_per_cm": 98.04,
                },
                {
                    "scenario_id": "train_1",
                    "component_id": "b",
                    "batch_id": "2",
                    "mass_fraction_pct": 90.0,
                    "test_temperature_c": 160.0,
                    "test_duration_h": 168.0,
                    "biofuel_mass_fraction_pct": 0.0,
                    "catalyst_dosage_category": "1",
                    "target_delta_kinematic_viscosity_pct": 25.4,
                    "target_oxidation_eot_a_per_cm": 98.04,
                },
                {
                    "scenario_id": "train_2",
                    "component_id": "c",
                    "batch_id": "3",
                    "mass_fraction_pct": 100.0,
                    "test_temperature_c": 150.0,
                    "test_duration_h": 168.0,
                    "biofuel_mass_fraction_pct": 5.0,
                    "catalyst_dosage_category": "2",
                    "target_delta_kinematic_viscosity_pct": 11.0,
                    "target_oxidation_eot_a_per_cm": 77.7,
                },
            ]
        )

        scenario_targets = build_train_scenario_targets(train_rows)

        self.assertEqual(scenario_targets["scenario_id"].tolist(), ["train_1", "train_2"])
        self.assertEqual(len(scenario_targets), 2)
        self.assertEqual(
            scenario_targets.columns.tolist(),
            [
                "scenario_id",
                "test_temperature_c",
                "test_duration_h",
                "biofuel_mass_fraction_pct",
                "catalyst_dosage_category",
                "target_delta_kinematic_viscosity_pct",
                "target_oxidation_eot_a_per_cm",
            ],
        )

    def test_prepared_targets_do_not_leak_component_level_columns(self) -> None:
        train_rows = pd.DataFrame(
            [
                {
                    "scenario_id": "train_1",
                    "component_id": "a",
                    "batch_id": "1",
                    "mass_fraction_pct": 10.0,
                    "test_temperature_c": 160.0,
                    "test_duration_h": 168.0,
                    "biofuel_mass_fraction_pct": 0.0,
                    "catalyst_dosage_category": "1",
                    "target_delta_kinematic_viscosity_pct": 25.4,
                    "target_oxidation_eot_a_per_cm": 98.04,
                },
                {
                    "scenario_id": "train_1",
                    "component_id": "b",
                    "batch_id": "2",
                    "mass_fraction_pct": 90.0,
                    "test_temperature_c": 160.0,
                    "test_duration_h": 168.0,
                    "biofuel_mass_fraction_pct": 0.0,
                    "catalyst_dosage_category": "1",
                    "target_delta_kinematic_viscosity_pct": 25.4,
                    "target_oxidation_eot_a_per_cm": 98.04,
                },
            ]
        )

        scenario_targets = build_train_scenario_targets(train_rows)

        self.assertTrue(scenario_targets["scenario_id"].is_unique)
        self.assertNotIn("component_id", scenario_targets.columns)
        self.assertNotIn("batch_id", scenario_targets.columns)
        self.assertNotIn("mass_fraction_pct", scenario_targets.columns)

    def test_inconsistent_targets_raise_an_error(self) -> None:
        train_rows = pd.DataFrame(
            [
                {
                    "scenario_id": "train_1",
                    "test_temperature_c": 160.0,
                    "test_duration_h": 168.0,
                    "biofuel_mass_fraction_pct": 0.0,
                    "catalyst_dosage_category": "1",
                    "target_delta_kinematic_viscosity_pct": 25.4,
                    "target_oxidation_eot_a_per_cm": 98.04,
                },
                {
                    "scenario_id": "train_1",
                    "test_temperature_c": 160.0,
                    "test_duration_h": 168.0,
                    "biofuel_mass_fraction_pct": 0.0,
                    "catalyst_dosage_category": "1",
                    "target_delta_kinematic_viscosity_pct": 26.0,
                    "target_oxidation_eot_a_per_cm": 98.04,
                },
            ]
        )

        with self.assertRaisesRegex(ValueError, "vary within scenario_id"):
            build_train_scenario_targets(train_rows)


if __name__ == "__main__":
    unittest.main()
