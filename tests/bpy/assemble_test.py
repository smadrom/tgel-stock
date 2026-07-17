import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tgel_stock import assemble  # noqa: E402
from tgel_stock import manifest as manifest_module  # noqa: E402
from tgel_stock import recipe as recipe_module  # noqa: E402
from tgel_stock import scene  # noqa: E402

TOLERANCE = 1e-4

WAGON_RECIPE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "recipes", "basic-box-wagon.rollingstock.json")
LOCOMOTIVE_RECIPE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "recipes", "basic-diesel-locomotive.rollingstock.json")

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)
YAW_180_QUAT = (0.0, 1.0, 0.0, 0.0)

EXPECTED_MESH_COUNTS = {
    "wagon": 94,       # 38 body + 2x13 bogie + 2x5 coupler + 4x5 wheelset
    "locomotive": 77,  # 21 body + 2x13 bogie + 2x5 coupler + 4x5 wheelset
}

TRIANGLE_BUDGETS = {
    "wagon": (6000, 100000),
    "locomotive": (10000, 150000),
}

failures = []


def expected_node_paths():
    """Independent re-derivation of the node table's path set from the
    naming-map contract, so this test does not simply mirror assemble.py's
    own internals back at itself.
    """
    paths = {
        "Visuals", "Visuals/Body", "Visuals/RunningGear",
        "Couplers", "Markers", "Interaction",
        "Markers/BogieTrack__front", "Markers/BogieTrack__rear",
    }
    for bogie_id in ("front", "rear"):
        bogie_path = f"Visuals/RunningGear/Bogie__{bogie_id}"
        paths.add(bogie_path)
        paths.add(f"{bogie_path}/Frame")
        for wheelset_id in ("leading", "trailing"):
            wheelset_path = f"{bogie_path}/Wheelset__{bogie_id}_{wheelset_id}"
            paths.add(wheelset_path)
            paths.add(f"{wheelset_path}/WheelLeft")
            paths.add(f"{wheelset_path}/WheelRight")
    for end in ("front", "rear"):
        pivot_path = f"Couplers/CouplerPivot__{end}"
        paths.add(pivot_path)
        paths.add(f"{pivot_path}/CouplerFace")
    return paths


SPOT_CHECK_PATHS = (
    "Visuals/Body",
    "Visuals/RunningGear/Bogie__front",
    "Visuals/RunningGear/Bogie__front/Frame",
    "Visuals/RunningGear/Bogie__front/Wheelset__front_leading",
    "Visuals/RunningGear/Bogie__front/Wheelset__front_leading/WheelRight",
    "Visuals/RunningGear/Bogie__rear/Wheelset__rear_trailing/WheelLeft",
    "Couplers/CouplerPivot__front",
    "Couplers/CouplerPivot__rear/CouplerFace",
    "Markers/BogieTrack__rear",
    "Interaction",
)


def close(a, b, tol=TOLERANCE):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def run_case(label, recipe_path):
    scene.reset()
    recipe_obj = recipe_module.load(recipe_path)
    assembly = assemble.build_vehicle(recipe_obj)

    if assembly.recipe is not recipe_obj:
        failures.append(f"[{label}] assembly.recipe does not reference the input recipe")

    nodes_by_path = {n.path: n for n in assembly.nodes}
    actual_paths = set(nodes_by_path.keys())
    expected_paths = expected_node_paths()

    missing = expected_paths - actual_paths
    extra = actual_paths - expected_paths
    if missing:
        failures.append(f"[{label}] node table missing paths: {sorted(missing)}")
    if extra:
        failures.append(f"[{label}] node table has unexpected extra paths: {sorted(extra)}")

    for path in SPOT_CHECK_PATHS:
        if path not in nodes_by_path:
            failures.append(f"[{label}] spot-check path missing from node table: {path}")

    if missing or any(p not in nodes_by_path for p in SPOT_CHECK_PATHS):
        return

    # Root parent conventions.
    for top in ("Visuals", "Couplers", "Markers", "Interaction"):
        if nodes_by_path[top].parent != "":
            failures.append(f"[{label}] {top}.parent expected '' got {nodes_by_path[top].parent!r}")

    # CouplerPivot__front local position.
    loc = recipe_obj.length_over_couplers
    expected_front = (0.0, 0.860, loc / 2.0 - 0.600)
    actual_front = nodes_by_path["Couplers/CouplerPivot__front"].local_position
    if not close(actual_front, expected_front):
        failures.append(
            f"[{label}] CouplerPivot__front local_position {actual_front} != {expected_front}")

    expected_rear = (0.0, 0.860, -loc / 2.0 + 0.600)
    actual_rear = nodes_by_path["Couplers/CouplerPivot__rear"].local_position
    if not close(actual_rear, expected_rear):
        failures.append(
            f"[{label}] CouplerPivot__rear local_position {actual_rear} != {expected_rear}")

    # CouplerFace local position, both ends.
    for end in ("front", "rear"):
        face = nodes_by_path[f"Couplers/CouplerPivot__{end}/CouplerFace"]
        if not close(face.local_position, (0.0, 0.0, 0.600)):
            failures.append(
                f"[{label}] CouplerFace ({end}) local_position {face.local_position} != (0,0,0.6)")

    # BogieTrack markers.
    front_track = nodes_by_path["Markers/BogieTrack__front"]
    if not close(front_track.local_position, (0.0, 0.0, recipe_obj.bogie_centre_offset)):
        failures.append(
            f"[{label}] BogieTrack__front local_position {front_track.local_position} "
            f"!= (0,0,{recipe_obj.bogie_centre_offset})")
    rear_track = nodes_by_path["Markers/BogieTrack__rear"]
    if not close(rear_track.local_position, (0.0, 0.0, -recipe_obj.bogie_centre_offset)):
        failures.append(
            f"[{label}] BogieTrack__rear local_position {rear_track.local_position} "
            f"!= (0,0,{-recipe_obj.bogie_centre_offset})")

    # Wheelset y position and WheelLeft/WheelRight x + rotation, for both bogies.
    expected_wheelset_y = recipe_obj.wheel_radius - recipe_obj.bogie_pivot_height
    for bogie_id in ("front", "rear"):
        for wheelset_id in ("leading", "trailing"):
            wheelset_path = (
                f"Visuals/RunningGear/Bogie__{bogie_id}/Wheelset__{bogie_id}_{wheelset_id}")
            wheelset_node = nodes_by_path[wheelset_path]
            if abs(wheelset_node.local_position[1] - expected_wheelset_y) > TOLERANCE:
                failures.append(
                    f"[{label}] {wheelset_path} y {wheelset_node.local_position[1]} "
                    f"!= {expected_wheelset_y}")

            left = nodes_by_path[f"{wheelset_path}/WheelLeft"]
            if abs(left.local_position[0] - (-0.7415)) > TOLERANCE:
                failures.append(
                    f"[{label}] {wheelset_path}/WheelLeft x {left.local_position[0]} != -0.7415")
            if not close(left.local_rotation_quat, IDENTITY_QUAT):
                failures.append(
                    f"[{label}] {wheelset_path}/WheelLeft rotation not identity: "
                    f"{left.local_rotation_quat}")

            right = nodes_by_path[f"{wheelset_path}/WheelRight"]
            if abs(right.local_position[0] - 0.7415) > TOLERANCE:
                failures.append(
                    f"[{label}] {wheelset_path}/WheelRight x {right.local_position[0]} != 0.7415")
            if not close(right.local_rotation_quat, YAW_180_QUAT):
                failures.append(
                    f"[{label}] {wheelset_path}/WheelRight rotation {right.local_rotation_quat} "
                    f"!= {YAW_180_QUAT}")

    # Rotation exceptions: identity everywhere except CouplerPivot__rear and every WheelRight.
    identity_spot_checked = False
    for path, node in nodes_by_path.items():
        is_exception = path == "Couplers/CouplerPivot__rear" or path.endswith("/WheelRight")
        expected_quat = YAW_180_QUAT if is_exception else IDENTITY_QUAT
        if not close(node.local_rotation_quat, expected_quat):
            failures.append(
                f"[{label}] {path} rotation {node.local_rotation_quat} != expected {expected_quat}")
        if path == "Visuals" and close(node.local_rotation_quat, IDENTITY_QUAT):
            identity_spot_checked = True
    if not identity_spot_checked:
        failures.append(f"[{label}] identity-rotation spot-check (Visuals) did not verify")

    # Mesh object naming + node membership + counts + triangle budget.
    mesh_count = len(assembly.objects)
    expected_count = EXPECTED_MESH_COUNTS[recipe_obj.kind]
    if mesh_count != expected_count:
        failures.append(f"[{label}] mesh count {mesh_count} != expected {expected_count}")

    total_triangles = 0
    for name, obj in assembly.objects.items():
        if obj.name != name:
            failures.append(f"[{label}] objects dict key {name!r} != obj.name {obj.name!r}")
        node_dotted = name.rsplit(".", 1)[0]
        node_path = node_dotted.replace(".", "/")
        if node_path not in actual_paths:
            failures.append(f"[{label}] mesh {name} implies unknown node path {node_path}")
        _positions, _normals, _uvs, triangles, _bmin, _bmax = scene.collect_mesh_data(obj)
        total_triangles += len(triangles)

    lo, hi = TRIANGLE_BUDGETS[recipe_obj.kind]
    if not (lo <= total_triangles <= hi):
        failures.append(f"[{label}] combined triangles {total_triangles} not in [{lo}, {hi}]")

    # Manifest integration.
    m = manifest_module.Manifest()
    assembly.fill_manifest(m)
    data = m.to_dict()
    if len(data["nodes"]) != len(assembly.nodes):
        failures.append(
            f"[{label}] manifest node count {len(data['nodes'])} != {len(assembly.nodes)}")
    if len(data["meshes"]) != mesh_count:
        failures.append(
            f"[{label}] manifest mesh count {len(data['meshes'])} != {mesh_count}")
    manifest_node_paths = {n["path"] for n in data["nodes"]}
    for mesh_entry in data["meshes"]:
        if mesh_entry["node"] not in manifest_node_paths:
            failures.append(
                f"[{label}] manifest mesh {mesh_entry['name']} references unknown node "
                f"{mesh_entry['node']}")


run_case("wagon", WAGON_RECIPE_PATH)
run_case("locomotive", LOCOMOTIVE_RECIPE_PATH)

if failures:
    print("BPY TESTS FAIL:", "; ".join(failures))
    sys.exit(1)
print("BPY TESTS OK")
sys.exit(0)
