import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import bmesh  # noqa: E402

from tgel_stock import recipe as recipe_module  # noqa: E402
from tgel_stock import scene  # noqa: E402
from tgel_stock.parts import body_wagon  # noqa: E402

RECIPE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "recipes", "basic-box-wagon.rollingstock.json")

# Exact 38-part interface (see body_wagon.py module docstring for the
# disclosed discrepancy against the plan's "40 named parts" tally).
EXPECTED_KEYS = {
    "Underframe", "CentreSill", "Floor", "BoxBody",
    *[f"SideRibLeft{i:02d}" for i in range(8)],
    *[f"SideRibRight{i:02d}" for i in range(8)],
    "DoorLeft", "DoorRight",
    "DoorRailLeftTop", "DoorRailLeftBottom", "DoorRailRightTop", "DoorRailRightBottom",
    "GabledRoof", "RoofWalk", "RoofRibs",
    "LadderFrontLeft", "LadderRearRight",
    "BrakeWheel",
    "EndSillFront", "EndSillRear",
    "StirrupStepFrontLeft", "StirrupStepFrontRight",
    "StirrupStepRearLeft", "StirrupStepRearRight",
}

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
meshes = body_wagon.build_meshes(recipe_obj)

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
BRAKE_LENGTH_LIMIT = recipe_obj.length_over_couplers / 2.0 + 0.001

total_triangles = 0

if not missing_keys:
    for key, obj in meshes.items():
        _positions, normals, _uvs, triangles, bounds_min, bounds_max = scene.collect_mesh_data(obj)
        total_triangles += len(triangles)
        assert_manifold(key, obj)

        if bounds_min[1] < -0.01:
            failures.append(f"{key} min_y {bounds_min[1]} below railhead")

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
        if key == "BrakeWheel":
            # BrakeWheel sits on the B-end coupler face (z = -body_length/2 -
            # 0.05); that 0.05 m is coupled-face clearance inside
            # length_over_couplers, not body length, so it is checked
            # against the longer envelope separately (see body_wagon.py
            # _assert_envelope for the matching in-builder guard).
            if max_abs_z > BRAKE_LENGTH_LIMIT:
                failures.append(
                    f"BrakeWheel |z| {max_abs_z} exceeds coupled-face clearance {BRAKE_LENGTH_LIMIT}")
        else:
            if max_abs_z > LENGTH_LIMIT:
                failures.append(f"{key} |z| {max_abs_z} exceeds body-length envelope {LENGTH_LIMIT}")

    # The roof y-chain must be derived from the recipe envelope: the walk
    # top rides a hair (1e-3) under recipe.height, so pin it to +/-0.01.
    _p, _n, _u, _t, _walk_min, walk_max = scene.collect_mesh_data(meshes["RoofWalk"])
    if abs(walk_max[1] - recipe_obj.height) > 0.01:
        failures.append(
            f"RoofWalk top {walk_max[1]} not within 0.01 of recipe height {recipe_obj.height}")

    # Regression for the detached tread defect: every corner step must be a
    # body-connected U-frame, not a small box floating at y ~= 0.25.  Its
    # hangers reach the body/floor seam and its span follows the ladder stiles.
    expected_step_bottom = (
        body_wagon.STIRRUP_STEP_Y
        - body_wagon.STIRRUP_STEP_TREAD_HEIGHT / 2.0)
    expected_step_span = 2.0 * body_wagon.STIRRUP_STEP_HALF_SPAN
    body_half_width = body_wagon.BOX_BODY_WIDTH / 2.0
    expected_profile = {
        (round(expected_step_bottom, 6), round(-body_wagon.STIRRUP_STEP_HALF_SPAN, 6)),
        (round(body_wagon.STIRRUP_STEP_TOP_Y, 6), round(-body_wagon.STIRRUP_STEP_HALF_SPAN, 6)),
        (round(body_wagon.STIRRUP_STEP_TOP_Y, 6), round(
            -body_wagon.STIRRUP_STEP_HALF_SPAN + body_wagon.STIRRUP_STEP_HANGER_WIDTH, 6)),
        (round(body_wagon.STIRRUP_STEP_Y + body_wagon.STIRRUP_STEP_TREAD_HEIGHT / 2.0, 6), round(
            -body_wagon.STIRRUP_STEP_HALF_SPAN + body_wagon.STIRRUP_STEP_HANGER_WIDTH, 6)),
        (round(body_wagon.STIRRUP_STEP_Y + body_wagon.STIRRUP_STEP_TREAD_HEIGHT / 2.0, 6), round(
            body_wagon.STIRRUP_STEP_HALF_SPAN - body_wagon.STIRRUP_STEP_HANGER_WIDTH, 6)),
        (round(body_wagon.STIRRUP_STEP_TOP_Y, 6), round(
            body_wagon.STIRRUP_STEP_HALF_SPAN - body_wagon.STIRRUP_STEP_HANGER_WIDTH, 6)),
        (round(body_wagon.STIRRUP_STEP_TOP_Y, 6), round(body_wagon.STIRRUP_STEP_HALF_SPAN, 6)),
        (round(expected_step_bottom, 6), round(body_wagon.STIRRUP_STEP_HALF_SPAN, 6)),
    }
    for suffix, x_sign, z_sign in (
            ("FrontLeft", -1.0, 1.0), ("FrontRight", 1.0, 1.0),
            ("RearLeft", -1.0, -1.0), ("RearRight", 1.0, -1.0)):
        name = f"StirrupStep{suffix}"
        positions, _n, _u, triangles, step_min, step_max = scene.collect_mesh_data(meshes[name])
        if abs(step_min[1] - expected_step_bottom) > 1e-4:
            failures.append(
                f"{name} bottom {step_min[1]} does not preserve the accepted tread height")
        if abs(step_max[1] - body_wagon.STIRRUP_STEP_TOP_Y) > 1e-4:
            failures.append(
                f"{name} top {step_max[1]} does not meet the body/floor seam "
                f"{body_wagon.STIRRUP_STEP_TOP_Y}")
        if abs((step_max[2] - step_min[2]) - expected_step_span) > 1e-4:
            failures.append(
                f"{name} span {step_max[2] - step_min[2]} does not align with ladder stiles")
        expected_centre_z = z_sign * (
            recipe_obj.body_length / 2.0 - body_wagon.STIRRUP_STEP_Z_INSET)
        actual_centre_z = (step_min[2] + step_max[2]) / 2.0
        if abs(actual_centre_z - expected_centre_z) > 1e-4:
            failures.append(
                f"{name} centre z {actual_centre_z} != corner anchor {expected_centre_z}")
        attached_x = step_max[0] if x_sign < 0.0 else step_min[0]
        if abs(abs(attached_x) - body_half_width) > 1e-4:
            failures.append(
                f"{name} inner face {attached_x} does not meet body side {x_sign * body_half_width}")
        actual_profile = {
            (round(position[1], 6), round(position[2] - actual_centre_z, 6))
            for position in positions
        }
        if actual_profile != expected_profile:
            failures.append(
                f"{name} does not preserve the accepted open U profile: {sorted(actual_profile)}")
        if len(triangles) != 28:
            failures.append(f"{name} triangles {len(triangles)} != open U-frame contract 28")

    if not (3000 <= total_triangles <= 60000):
        failures.append(f"total triangles {total_triangles} not in [3000, 60000]")

if failures:
    print("BPY TESTS FAIL:", "; ".join(failures))
    sys.exit(1)
print("BPY TESTS OK")
sys.exit(0)
