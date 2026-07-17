import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tgel_stock import canonical


class CanonicalTests(unittest.TestCase):
    TRI = [(0, 1, 2)]
    UV = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    N = [(0.0, 0.0, 1.0)] * 3

    def test_hash_is_stable_reference_vector(self):
        pos = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        first = canonical.geometry_hash(pos, self.N, self.UV, self.TRI)
        second = canonical.geometry_hash(pos, self.N, self.UV, self.TRI)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_sub_tolerance_noise_ignored(self):
        pos_a = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        pos_b = [(0.000001, 0.0, 0.0), (1.000004, 0.0, 0.0), (0.0, 1.0, 0.0)]
        self.assertEqual(
            canonical.geometry_hash(pos_a, self.N, self.UV, self.TRI),
            canonical.geometry_hash(pos_b, self.N, self.UV, self.TRI))

    def test_real_change_changes_hash(self):
        pos_a = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        pos_b = [(0.0, 0.0, 0.0), (1.001, 0.0, 0.0), (0.0, 1.0, 0.0)]
        self.assertNotEqual(
            canonical.geometry_hash(pos_a, self.N, self.UV, self.TRI),
            canonical.geometry_hash(pos_b, self.N, self.UV, self.TRI))

    def test_negative_zero_normalized(self):
        pos_a = [(0.0, 0.0, 0.0)]
        pos_b = [(-0.0, 0.0, 0.0)]
        self.assertEqual(
            canonical.geometry_hash(pos_a, [], [], []),
            canonical.geometry_hash(pos_b, [], [], []))


if __name__ == "__main__":
    unittest.main()
