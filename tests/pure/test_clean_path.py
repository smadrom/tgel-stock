import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tgel_stock import assert_clean_path  # noqa: E402


class AssertCleanPathTests(unittest.TestCase):
    def test_rejects_rc2_token_mixed_case_backslashes(self):
        with self.assertRaises(ValueError):
            assert_clean_path("C:\\Exports\\RC2\\x.fbx")

    def test_rejects_legacy_token_mixed_case(self):
        with self.assertRaises(ValueError):
            assert_clean_path("d:/stuff/Legacy/y.json")

    def test_rejects_exportedproject_token(self):
        with self.assertRaises(ValueError):
            assert_clean_path("C:\\Temp\\ExportedProject\\scene.unity")

    def test_rejects_models_glb_token(self):
        with self.assertRaises(ValueError):
            assert_clean_path("assets/models_glb/train.glb")

    def test_accepts_clean_path(self):
        assert_clean_path("tools/rolling-stock-generation/recipes/a.json")


if __name__ == "__main__":
    unittest.main()
