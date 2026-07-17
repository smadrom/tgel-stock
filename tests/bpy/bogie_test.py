import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import bmesh  # noqa: E402

from tgel_stock import scene  # noqa: E402
from tgel_stock.parts import bogie  # noqa: E402

TOLERANCE = 1e-4

EXPECTED_KEYS = {
    "Bolster",
    "SideFrameLeft",
    "SideFrameRight",
    "CrossMemberFront",
    "CrossMemberRear",
    "AxleboxLeftLeading",
    "AxleboxRightLeading",
    "AxleboxLeftTrailing",
    "AxleboxRightTrailing",
    "SpringLeftFront",
    "SpringLeftRear",
    "SpringRightFront",
    "SpringRightRear",
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


def run_case(label, wheelbase, wheel_radius, pivot_height):
    scene.reset()
    meshes = bogie.build_frame_meshes(wheelbase, wheel_radius, pivot_height)

    keys = set(meshes.keys())
    missing_keys = EXPECTED_KEYS - keys
    extra_keys = keys - EXPECTED_KEYS
    if missing_keys:
        failures.append(f"[{label}] build_frame_meshes missing keys: {sorted(missing_keys)}")
    if extra_keys:
        failures.append(f"[{label}] build_frame_meshes has extra keys: {sorted(extra_keys)}")
    if missing_keys:
        return

    total_triangles = 0
    min_y_floor = -pivot_height - 0.01

    for key, obj in meshes.items():
        _positions, _normals, _uvs, triangles, bounds_min, bounds_max = scene.collect_mesh_data(obj)
        total_triangles += len(triangles)
        if bounds_min[1] < min_y_floor:
            failures.append(
                f"[{label}] {key} min_y {bounds_min[1]} below railhead floor {min_y_floor}")
        assert_manifold(f"[{label}] {key}", obj)

    for key in ("SideFrameLeft", "SideFrameRight"):
        _p, _n, _u, _t, bounds_min, bounds_max = scene.collect_mesh_data(meshes[key])
        z_span = bounds_max[2] - bounds_min[2]
        expected_span = wheelbase + 0.42
        if abs(z_span - expected_span) > 0.05:
            failures.append(
                f"[{label}] {key} z span {z_span} not within 0.05 of expected {expected_span}")

    axlebox_expectations = {
        "AxleboxLeftLeading": wheelbase / 2.0,
        "AxleboxRightLeading": wheelbase / 2.0,
        "AxleboxLeftTrailing": -wheelbase / 2.0,
        "AxleboxRightTrailing": -wheelbase / 2.0,
    }
    for key, expected_z in axlebox_expectations.items():
        _p, _n, _u, _t, bounds_min, bounds_max = scene.collect_mesh_data(meshes[key])
        centre_z = (bounds_min[2] + bounds_max[2]) / 2.0
        if abs(centre_z - expected_z) > 0.01:
            failures.append(
                f"[{label}] {key} centre_z {centre_z} != expected {expected_z} (+/-0.01)")

    if not (500 <= total_triangles <= 20000):
        failures.append(f"[{label}] total frame triangles {total_triangles} not in [500, 20000]")


# Locomotive parameters.
run_case("locomotive", wheelbase=2.7432, wheel_radius=0.508, pivot_height=1.10)

# Wagon parameters (must be genuinely parametric).
run_case("wagon", wheelbase=1.6764, wheel_radius=0.4191, pivot_height=0.96)

if failures:
    print("BPY TESTS FAIL:", "; ".join(failures))
    sys.exit(1)
print("BPY TESTS OK")
sys.exit(0)
