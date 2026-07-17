import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import bmesh  # noqa: E402

from tgel_stock import scene  # noqa: E402
from tgel_stock.parts import coupler  # noqa: E402

EXPECTED_KEYS = {
    "DraftGear",
    "Shank",
    "Head",
    "KnucklePin",
    "CutLever",
}

MAX_ABS_X = 0.45
MAX_ABS_Y = 0.30
HEAD_MAX_Z_MIN = 0.598
HEAD_MAX_Z_MAX = 0.602
HEAD_MAX_WIDTH = 0.35

failures = []


def assert_manifold(label, obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    non_manifold_edges = sum(1 for e in bm.edges if not e.is_manifold)
    degenerate_faces = sum(1 for f in bm.faces if f.calc_area() < 1e-9)
    bm.free()
    if non_manifold_edges:
        failures.append(f"{label} has {non_manifold_edges} non-manifold edges")
    if degenerate_faces:
        failures.append(f"{label} has {degenerate_faces} degenerate faces (area < 1e-9)")


scene.reset()
meshes = coupler.build_meshes()

keys = set(meshes.keys())
missing_keys = EXPECTED_KEYS - keys
extra_keys = keys - EXPECTED_KEYS
if missing_keys:
    failures.append(f"build_meshes missing keys: {sorted(missing_keys)}")
if extra_keys:
    failures.append(f"build_meshes has extra keys: {sorted(extra_keys)}")

total_triangles = 0

if not missing_keys:
    for key, obj in meshes.items():
        _positions, _normals, _uvs, triangles, bounds_min, bounds_max = scene.collect_mesh_data(obj)
        total_triangles += len(triangles)
        assert_manifold(key, obj)

        max_abs_x = max(abs(bounds_min[0]), abs(bounds_max[0]))
        if max_abs_x > MAX_ABS_X:
            failures.append(f"{key} max|x| {max_abs_x} exceeds {MAX_ABS_X}")

        max_abs_y = max(abs(bounds_min[1]), abs(bounds_max[1]))
        if max_abs_y > MAX_ABS_Y:
            failures.append(f"{key} max|y| {max_abs_y} exceeds {MAX_ABS_Y}")

    _p, _n, _u, _t, head_min, head_max = scene.collect_mesh_data(meshes["Head"])
    if not (HEAD_MAX_Z_MIN <= head_max[2] <= HEAD_MAX_Z_MAX):
        failures.append(
            f"Head max_z {head_max[2]} not within [{HEAD_MAX_Z_MIN}, {HEAD_MAX_Z_MAX}]")

    head_width = head_max[0] - head_min[0]
    if head_width > HEAD_MAX_WIDTH:
        failures.append(f"Head x-width {head_width} exceeds {HEAD_MAX_WIDTH}")

    if not (300 <= total_triangles <= 8000):
        failures.append(f"combined triangles {total_triangles} not in [300, 8000]")

if failures:
    print("BPY TESTS FAIL:", "; ".join(failures))
    sys.exit(1)
print("BPY TESTS OK")
sys.exit(0)
