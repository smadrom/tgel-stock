import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tgel_stock import recipe

RECIPES = os.path.join(os.path.dirname(__file__), "..", "..", "recipes")


class RecipeTests(unittest.TestCase):
    def test_locomotive_recipe_loads_frozen_envelope(self):
        r = recipe.load(os.path.join(RECIPES, "basic-diesel-locomotive.rollingstock.json"))
        self.assertEqual(r.model_id, "rolling-stock.locomotive.road-switcher-v2")
        self.assertEqual(r.kind, "locomotive")
        self.assertAlmostEqual(r.length_over_couplers, 17.1196)
        self.assertAlmostEqual(r.body_length, 16.0)
        self.assertAlmostEqual(r.width, 3.1242)
        self.assertAlmostEqual(r.height, 4.4196)
        self.assertAlmostEqual(r.wheel_radius, 0.508)
        self.assertAlmostEqual(r.bogie_centre_offset, 4.7244)
        self.assertAlmostEqual(r.bogie_wheelbase, 2.7432)
        self.assertAlmostEqual(r.bogie_pivot_height, 1.10)
        self.assertIn("body", r.livery)

    def test_wagon_recipe_loads_frozen_envelope(self):
        r = recipe.load(os.path.join(RECIPES, "basic-box-wagon.rollingstock.json"))
        self.assertEqual(r.kind, "wagon")
        self.assertAlmostEqual(r.length_over_couplers, 13.5128)
        self.assertAlmostEqual(r.wheel_radius, 0.4191)
        self.assertAlmostEqual(r.bogie_pivot_height, 0.96)

    def test_missing_field_rejected(self):
        import json, tempfile
        with open(os.path.join(RECIPES, "basic-box-wagon.rollingstock.json"), encoding="utf-8") as f:
            data = json.load(f)
        del data["wheelRadius"]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump(data, tmp)
            path = tmp.name
        try:
            with self.assertRaises(ValueError):
                recipe.load(path)
        finally:
            os.unlink(path)

    def test_forbidden_path_rejected(self):
        with self.assertRaises(ValueError):
            recipe.load("C:/exports/RC2/loco.rollingstock.json")


if __name__ == "__main__":
    unittest.main()
