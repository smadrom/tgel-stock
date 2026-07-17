import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tgel_stock import canonical
from tgel_stock import manifest


class ManifestTests(unittest.TestCase):
    POSITIONS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    NORMALS = [(0.0, 0.0, 1.0)] * 3
    UVS = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    TRIANGLES = [(0, 1, 2)]
    BOUNDS_MIN = (0.0, 0.0, 0.0)
    BOUNDS_MAX = (1.0, 1.0, 0.0)
    TEXTURE_BYTES = b"px"
    EXPECTED_META = {
        "atlasResolution": [4096, 4096],
        "blenderVersion": "5.1.2",
        "bodyLength": 16.0,
        "bogieCentreOffset": 4.7244,
        "bogiePivotHeight": 1.1,
        "bogieWheelbase": 2.7432,
        "couplerHeight": 0.86,
        "couplerPivotToFace": 0.6,
        "height": 4.4196,
        "kind": "locomotive",
        "lengthOverCouplers": 17.1196,
        "modelId": "m-id",
        "recipeDigest": "rdigest",
        "schema": "tgel.rollingstock.manifest.v2",
        "scriptDigest": "sdigest",
        "trackGauge": 1.435,
        "wheelBackToBack": 1.348,
        "wheelRadius": 0.508,
        "wheelWidth": 0.135,
        "width": 3.1242,
    }
    META_KWARGS = {
        "kind": "locomotive",
        "atlas_resolution": (4096, 4096),
        "length_over_couplers": 17.1196001,
        "body_length": 16.0000001,
        "width": 3.1242001,
        "height": 4.4196001,
        "track_gauge": 1.4350001,
        "wheel_back_to_back": 1.3480001,
        "wheel_width": 0.1350001,
        "wheel_radius": 0.5080001,
        "coupler_height": 0.8600001,
        "coupler_pivot_to_face": 0.6000001,
        "bogie_centre_offset": 4.7244001,
        "bogie_wheelbase": 2.7432001,
        "bogie_pivot_height": 1.1000001,
    }

    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.texture_path = os.path.join(self._tempdir.name, "body_albedo.png")
        with open(self.texture_path, "wb") as handle:
            handle.write(self.TEXTURE_BYTES)
        self.manifest_path = os.path.join(self._tempdir.name, "manifest.json")

    def _build_and_write(self):
        built = manifest.Manifest()
        built.set_meta(
            "m-id", "rdigest", "sdigest", "5.1.2",
            **self.META_KWARGS)
        built.add_node("B/Child", "B", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        built.add_node("A/Root", None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        built.add_mesh(
            "body", "A/Root", self.POSITIONS, self.NORMALS, self.UVS,
            self.TRIANGLES, self.BOUNDS_MIN, self.BOUNDS_MAX)
        built.add_texture("body_albedo", self.texture_path, (4, 4), "sRGB")
        built.write(self.manifest_path)
        with open(self.manifest_path, "rb") as handle:
            raw = handle.read()
        return raw, json.loads(raw.decode("utf-8"))

    def test_written_file_uses_lf_newlines_and_trailing_newline(self):
        raw, _ = self._build_and_write()
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\r", raw)

    def test_meta_fields_round_trip(self):
        _, parsed = self._build_and_write()
        self.assertEqual(parsed["meta"], self.EXPECTED_META)

    def test_meta_frozen_measurements_are_rounded_before_serialization(self):
        raw, parsed = self._build_and_write()
        self.assertEqual(parsed["meta"], self.EXPECTED_META)
        self.assertNotIn(b"17.1196001", raw)
        self.assertNotIn(b"0.8600001", raw)

    def test_meta_measurements_are_required_keyword_only(self):
        positional_values = tuple(self.META_KWARGS.values())
        with self.assertRaises(TypeError):
            manifest.Manifest().set_meta(
                "m-id", "rdigest", "sdigest", "5.1.2", *positional_values)

        missing = dict(self.META_KWARGS)
        del missing["wheel_radius"]
        with self.assertRaises(TypeError):
            manifest.Manifest().set_meta(
                "m-id", "rdigest", "sdigest", "5.1.2", **missing)

    def test_meta_rejects_every_non_finite_measurement(self):
        measurement_keys = tuple(
            key for key in self.META_KWARGS
            if key not in ("kind", "atlas_resolution"))
        for key in measurement_keys:
            for invalid in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(key=key, invalid=invalid):
                    values = dict(self.META_KWARGS)
                    values[key] = invalid
                    with self.assertRaises(ValueError):
                        manifest.Manifest().set_meta(
                            "m-id", "rdigest", "sdigest", "5.1.2", **values)

    def test_meta_rejects_invalid_atlas_resolution(self):
        for invalid in ((4096,), (0, 4096), (4096.5, 4096), (4096, float("inf"))):
            with self.subTest(invalid=invalid):
                values = dict(self.META_KWARGS)
                values["atlas_resolution"] = invalid
                with self.assertRaises((TypeError, ValueError)):
                    manifest.Manifest().set_meta(
                        "m-id", "rdigest", "sdigest", "5.1.2", **values)

    def test_nodes_sorted_by_path(self):
        _, parsed = self._build_and_write()
        paths = [node["path"] for node in parsed["nodes"]]
        self.assertEqual(paths, ["A/Root", "B/Child"])

    def test_mesh_entry_matches_direct_geometry_hash(self):
        _, parsed = self._build_and_write()
        mesh = parsed["meshes"][0]
        self.assertEqual(mesh["vertexCount"], 3)
        self.assertEqual(mesh["triangleCount"], 1)
        self.assertEqual(len(mesh["semanticHash"]), 64)
        self.assertEqual(
            mesh["semanticHash"],
            canonical.geometry_hash(
                self.POSITIONS, self.NORMALS, self.UVS, self.TRIANGLES))

    def test_texture_entry_hashes_file_and_strips_directory(self):
        _, parsed = self._build_and_write()
        texture = parsed["textures"][0]
        self.assertEqual(
            texture["sha256"], hashlib.sha256(self.TEXTURE_BYTES).hexdigest())
        self.assertEqual(texture["file"], "body_albedo.png")

    def test_node_position_rounded_to_six_decimals(self):
        built = manifest.Manifest()
        built.add_node(
            "A/Root", None, (0.12345678, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        node = built.to_dict()["nodes"][0]
        self.assertEqual(node["localPosition"][0], 0.123457)

    def test_top_level_json_keys_sorted(self):
        _, parsed = self._build_and_write()
        self.assertEqual(list(parsed.keys()), sorted(parsed.keys()))


if __name__ == "__main__":
    unittest.main()
