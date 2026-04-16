"""Tests for property lookup preparation and exact-vs-typical joins."""

from __future__ import annotations

import unittest

import pandas as pd

from src.data.prepare_properties import build_property_artifacts, join_properties_to_mixtures


class PropertyJoinTest(unittest.TestCase):
    def test_exact_then_typical_property_fallback_is_deterministic(self) -> None:
        component_properties = pd.DataFrame(
            [
                {
                    "component_id": "comp_a",
                    "batch_id": "batch_1",
                    "property_name": "Kinematic Viscosity",
                    "property_unit": "mm2/s",
                    "property_value": "1.1",
                    "property_value_numeric": 1.1,
                },
                {
                    "component_id": "comp_a",
                    "batch_id": "typical",
                    "property_name": "Kinematic Viscosity",
                    "property_unit": "mm2/s",
                    "property_value": "9.9",
                    "property_value_numeric": 9.9,
                },
                {
                    "component_id": "comp_b",
                    "batch_id": "typical",
                    "property_name": "Kinematic Viscosity",
                    "property_unit": "mm2/s",
                    "property_value": "2.2",
                    "property_value_numeric": 2.2,
                },
                {
                    "component_id": "comp_f",
                    "batch_id": "batch_f",
                    "property_name": "Kinematic Viscosity",
                    "property_unit": "mm2/s",
                    "property_value": "3.3",
                    "property_value_numeric": 3.3,
                },
                {
                    "component_id": "comp_f",
                    "batch_id": "typical",
                    "property_name": "Density",
                    "property_unit": "g/cm3",
                    "property_value": "4.4",
                    "property_value_numeric": 4.4,
                },
                {
                    "component_id": "comp_d",
                    "batch_id": None,
                    "property_name": "Kinematic Viscosity",
                    "property_unit": "mm2/s",
                    "property_value": "5.5",
                    "property_value_numeric": 5.5,
                },
                {
                    "component_id": "comp_c",
                    "batch_id": "batch_c",
                    "property_name": None,
                    "property_unit": None,
                    "property_value": None,
                    "property_value_numeric": None,
                },
            ]
        )

        mixtures = pd.DataFrame(
            [
                {"scenario_id": "s1", "component_id": "comp_a", "batch_id": "batch_1"},
                {"scenario_id": "s1", "component_id": "comp_b", "batch_id": "missing_exact"},
                {"scenario_id": "s1", "component_id": "comp_f", "batch_id": "batch_f"},
                {"scenario_id": "s1", "component_id": "comp_d", "batch_id": None},
                {"scenario_id": "s1", "component_id": "comp_e", "batch_id": "batch_e"},
            ]
        )

        artifacts = build_property_artifacts(component_properties)
        joined = join_properties_to_mixtures(mixtures, artifacts.exact_lookup, artifacts.typical_lookup)

        property_keys = artifacts.property_catalog.set_index("property_name")["property_key"].to_dict()
        viscosity_key = property_keys["Kinematic Viscosity"]
        density_key = property_keys["Density"]

        records = joined.set_index("component_id").to_dict(orient="index")

        self.assertEqual(records["comp_a"][viscosity_key], 1.1)
        self.assertTrue(records["comp_a"]["used_exact_batch_props"])
        self.assertFalse(records["comp_a"]["used_typical_fallback"])
        self.assertEqual(records["comp_a"]["property_join_source"], "exact_only")

        self.assertEqual(records["comp_b"][viscosity_key], 2.2)
        self.assertFalse(records["comp_b"]["used_exact_batch_props"])
        self.assertTrue(records["comp_b"]["used_typical_fallback"])
        self.assertEqual(records["comp_b"]["property_join_source"], "typical_only")

        self.assertEqual(records["comp_f"][viscosity_key], 3.3)
        self.assertEqual(records["comp_f"][density_key], 4.4)
        self.assertTrue(records["comp_f"]["used_exact_batch_props"])
        self.assertTrue(records["comp_f"]["used_typical_fallback"])
        self.assertEqual(records["comp_f"]["property_join_source"], "exact_plus_typical")

        self.assertEqual(records["comp_d"][viscosity_key], 5.5)
        self.assertTrue(records["comp_d"]["batch_id_was_missing"])
        self.assertTrue(records["comp_d"]["used_exact_batch_props"])

        self.assertTrue(records["comp_e"]["missing_all_props"])
        self.assertFalse(records["comp_e"]["has_usable_property_coverage"])
        self.assertEqual(records["comp_e"]["property_join_source"], "missing")


if __name__ == "__main__":
    unittest.main()
