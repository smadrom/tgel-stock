import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import bmesh  # noqa: E402

from tgel_stock import scene  # noqa: E402
from tgel_stock.parts import wheel  # noqa: E402

TOLERANCE = 1e-4
RADIUS = 0.508
HALF_WIDTH = 0.0675

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

wheel_obj = wheel.build_wheel("W", RADIUS)
positions, _normals, _uvs, triangles, bounds_min, bounds_max = scene.collect_mesh_data(wheel_obj)

max_radius = max(
    max(abs(p[1]), abs(p[2])) for p in positions
) if positions else 0.0
if not (RADIUS - TOLERANCE <= max_radius <= RADIUS + 0.025 + 0.002 + TOLERANCE):
    failures.append(f"wheel radius bound {max_radius} not in [{RADIUS}, {RADIUS + 0.027}]")

width_span = bounds_max[0] - bounds_min[0]
if not (0.135 - TOLERANCE <= width_span <= 0.170 + TOLERANCE):
    failures.append(f"wheel width span {width_span} not in [0.135, 0.170]")

if abs(bounds_max[0] + bounds_min[0]) > 1e-6:
    failures.append(
        f"wheel bounds not symmetric about origin: min_x {bounds_min[0]}, max_x {bounds_max[0]}")

vertex_count = len(positions)
if not (200 <= vertex_count <= 4000):
    failures.append(f"wheel vertex count {vertex_count} not in [200, 4000]")

for tri in triangles:
    for idx in tri:
        if not (0 <= idx < len(positions)):
            failures.append(f"triangle index {idx} out of range for {len(positions)} vertices")
            break

assert_manifold("wheel", wheel_obj)

scene.reset()

meshes = wheel.build_wheelset_meshes("front.leading", RADIUS)
expected_keys = {
    "AxleVisual",
    "WheelLeft.TreadAndFlange",
    "WheelLeft.Hub",
    "WheelRight.TreadAndFlange",
    "WheelRight.Hub",
}
missing_keys = expected_keys - set(meshes.keys())
if missing_keys:
    failures.append(f"build_wheelset_meshes missing keys: {sorted(missing_keys)}")
else:
    _, _, _, _, lt_min, lt_max = scene.collect_mesh_data(meshes["WheelLeft.TreadAndFlange"])
    if abs(lt_max[0] - HALF_WIDTH) > TOLERANCE:
        failures.append(
            f"WheelLeft.TreadAndFlange max_x {lt_max[0]} != +{HALF_WIDTH} (flange side, untranslated)")
    if abs(lt_min[0] + HALF_WIDTH) > TOLERANCE:
        failures.append(
            f"WheelLeft.TreadAndFlange min_x {lt_min[0]} != -{HALF_WIDTH} (untranslated)")
    for key in sorted(expected_keys):
        assert_manifold(key, meshes[key])

if failures:
    print("BPY TESTS FAIL:", "; ".join(failures))
    sys.exit(1)
print("BPY TESTS OK")
sys.exit(0)
