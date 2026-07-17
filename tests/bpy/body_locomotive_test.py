import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import bmesh  # noqa: E402

from tgel_stock import recipe as recipe_module  # noqa: E402
from tgel_stock import scene  # noqa: E402
from tgel_stock.parts import body_locomotive  # noqa: E402

RECIPE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "recipes", "basic-diesel-locomotive.rollingstock.json")

# Exact 21-part interface. The task brief's running prose tally says "20
# parts", but its own literal enumeration (Frame, Walkway, FuelTank,
# LongHood, RadiatorFanFront, RadiatorFanRear, ExhaustStack, Cab, CabGlass,
# CabRoof, ShortHood, HeadlightFront, HeadlightRear, Horn, Bell,
# HandrailLeft, HandrailRight, StepsFrontLeft, StepsFrontRight,
# StepsRearLeft, StepsRearRight) counts to 21 -- the same kind of prose-vs-
# enumeration mismatch documented in body_wagon.py's module docstring
# ("40" vs the actual 38). This module and its test build/assert exactly
# these 21 keys.
EXPECTED_KEYS = {
    "Frame", "Walkway", "FuelTank", "LongHood",
    "RadiatorFanFront", "RadiatorFanRear", "ExhaustStack",
    "Cab", "CabGlass", "CabRoof", "ShortHood",
    "HeadlightFront", "HeadlightRear", "Horn", "Bell",
    "HandrailLeft", "HandrailRight",
    "StepsFrontLeft", "StepsFrontRight", "StepsRearLeft", "StepsRearRight",
}

# CabGlass is authored as 4 single-sided flat quads (glass panes filling the
# Cab's window openings) -- see body_locomotive.py's module docstring. A
# flat, single-sided quad has every edge on the mesh boundary (1 adjacent
# face), which bmesh's edge.is_manifold reports as non-manifold by
# construction; there is no way to make a literal flat pane manifold without
# giving it volume, which would break the "8 triangles (4 quads)" assertion
# below. This is a disclosed, deliberate exception to the manifold check,
# the same way body_wagon.py's BrakeWheel is a disclosed exception to the
# body-length envelope check.
MANIFOLD_EXEMPT = {"CabGlass"}

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
recipe_obj = recipe_module.load(RECIPE_PATH)
meshes = body_locomotive.build_meshes(recipe_obj)

keys = set(meshes.keys())
missing_keys = EXPECTED_KEYS - keys
extra_keys = keys - EXPECTED_KEYS
if missing_keys:
    failures.append(f"build_meshes missing keys: {sorted(missing_keys)}")
if extra_keys:
    failures.append(f"build_meshes has extra keys: {sorted(extra_keys)}")

WIDTH_LIMIT = recipe_obj.width / 2.0 + 0.001
HEIGHT_LIMIT = recipe_obj.height + 0.001
LENGTH_LIMIT = recipe_obj.body_length / 2.0 + 0.001
MIN_Y_FLOOR = 0.4

total_triangles = 0

if not missing_keys:
    for key, obj in meshes.items():
        _positions, normals, _uvs, triangles, bounds_min, bounds_max = scene.collect_mesh_data(obj)
        total_triangles += len(triangles)
        if key not in MANIFOLD_EXEMPT:
            assert_manifold(key, obj)

        if bounds_min[1] < MIN_Y_FLOOR:
            failures.append(f"{key} min_y {bounds_min[1]} below the {MIN_Y_FLOOR} floor")

        for n in normals:
            length_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2]
            if length_sq < 1e-8:
                failures.append(f"{key} has a degenerate (near-zero) normal")
                break

        max_abs_x = max(abs(bounds_min[0]), abs(bounds_max[0]))
        if max_abs_x > WIDTH_LIMIT:
            failures.append(f"{key} max|x| {max_abs_x} exceeds width envelope {WIDTH_LIMIT}")

        if bounds_max[1] > HEIGHT_LIMIT:
            failures.append(f"{key} max_y {bounds_max[1]} exceeds height envelope {HEIGHT_LIMIT}")

        max_abs_z = max(abs(bounds_min[2]), abs(bounds_max[2]))
        if max_abs_z > LENGTH_LIMIT:
            failures.append(f"{key} |z| {max_abs_z} exceeds body-length envelope {LENGTH_LIMIT}")

    # Window-opening proxy: a plain box Cab would triangulate to 12
    # triangles; the real cut-through window openings (picture-frame wall
    # solids) must push the count well past that.
    if "Cab" in meshes:
        _p, _n, _u, cab_tris, _mn, _mx = scene.collect_mesh_data(meshes["Cab"])
        if len(cab_tris) <= 12:
            failures.append(f"Cab has only {len(cab_tris)} triangles; window openings not detected")

    if "CabGlass" in meshes:
        _p, _n, _u, glass_tris, _mn, _mx = scene.collect_mesh_data(meshes["CabGlass"])
        if len(glass_tris) != 8:
            failures.append(f"CabGlass has {len(glass_tris)} triangles, expected 8 (4 quads)")

    # CabRoof top must derive from recipe.height (see body_locomotive.py's
    # module docstring for the roof y-chain derivation), pinned to +/-0.01.
    if "CabRoof" in meshes:
        _p, _n, _u, _t, _roof_min, roof_max = scene.collect_mesh_data(meshes["CabRoof"])
        if abs(roof_max[1] - recipe_obj.height) > 0.01:
            failures.append(
                f"CabRoof top {roof_max[1]} not within 0.01 of recipe height {recipe_obj.height}")

    if not (6000 <= total_triangles <= 120000):
        failures.append(f"total triangles {total_triangles} not in [6000, 120000]")

if failures:
    print("BPY TESTS FAIL:", "; ".join(failures))
    sys.exit(1)
print("BPY TESTS OK")
sys.exit(0)
