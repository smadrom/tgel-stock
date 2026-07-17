"""Task 15 full-build, validation, determinism and atomic-publish contract.

This is intentionally a RED-first test.  Task 15 owns ``build.py`` and
``tgel_stock.validate``; until those production modules exist, the imports
below must fail under Blender with a non-zero process exit.

The expensive production 4096 texture build is deliberately NOT part of this
test.  Geometry determinism is exercised at the production 4096 atlas setting
without bakes for both recipes.  Full orchestration is exercised twice for the
wagon through the internal test-only 512 setting.  Public-CLI 4096 acceptance
remains a separate Task 15 gate.
"""

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import bpy  # noqa: E402

# RED boundary: both modules are intentionally absent before Task 15
# production implementation starts.
import build as build_module  # noqa: E402
from tgel_stock import assemble  # noqa: E402
from tgel_stock import export  # noqa: E402
from tgel_stock import lod  # noqa: E402
from tgel_stock import recipe as recipe_module  # noqa: E402
from tgel_stock import scene  # noqa: E402
from tgel_stock import uvmap  # noqa: E402
from tgel_stock import validate  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
WAGON_RECIPE_PATH = ROOT / "recipes" / "basic-box-wagon.rollingstock.json"
LOCOMOTIVE_RECIPE_PATH = (
    ROOT / "recipes" / "basic-diesel-locomotive.rollingstock.json")

PRODUCTION_ATLAS_PX = 4096
TEST_ATLAS_PX = 512
BLENDER_VERSION = "5.1.2"
SCHEMA = "tgel.rollingstock.manifest.v2"
LOD_SUFFIX_RE = re.compile(r"__LOD[12]$")

CASES = (
    {
        "label": "wagon",
        "recipe": WAGON_RECIPE_PATH,
        "model_id": "rolling-stock.wagon.40ft-box-v2",
        "kind": "wagon",
        "mesh_count": 200,
        "lod0_count": 94,
        "lod1_count": 94,
        "lod2_count": 12,
        "lengthOverCouplers": 13.5128,
        "bodyLength": 12.7508,
        "width": 3.2512,
        "height": 4.4196,
        "wheelRadius": 0.4191,
        "bogieCentreOffset": 4.699,
        "bogieWheelbase": 1.6764,
        "bogiePivotHeight": 0.96,
    },
    {
        "label": "locomotive",
        "recipe": LOCOMOTIVE_RECIPE_PATH,
        "model_id": "rolling-stock.locomotive.road-switcher-v2",
        "kind": "locomotive",
        "mesh_count": 173,
        "lod0_count": 77,
        "lod1_count": 77,
        "lod2_count": 19,
        "lengthOverCouplers": 17.1196,
        "bodyLength": 16.0,
        "width": 3.1242,
        "height": 4.4196,
        "wheelRadius": 0.508,
        "bogieCentreOffset": 4.7244,
        "bogieWheelbase": 2.7432,
        "bogiePivotHeight": 1.10,
    },
)

REQUIRED_META_KEYS = {
    "schema",
    "modelId",
    "kind",
    "recipeDigest",
    "scriptDigest",
    "blenderVersion",
    "atlasResolution",
    "lengthOverCouplers",
    "bodyLength",
    "width",
    "height",
    "trackGauge",
    "wheelBackToBack",
    "wheelWidth",
    "wheelRadius",
    "couplerHeight",
    "couplerPivotToFace",
    "bogieCentreOffset",
    "bogieWheelbase",
    "bogiePivotHeight",
}

EXPECTED_TEXTURES = {
    "albedo": ("albedo.png", "sRGB"),
    "normal": ("normal.png", "Non-Color"),
    "mask": ("mask.png", "Non-Color"),
}

failures = []


def fail(label, message):
    failures.append(f"[{label}] {message}")


def expect_raises(label, exception_type, action):
    try:
        action()
    except exception_type:
        return
    except Exception as exc:  # noqa: BLE001 - exact wrong outcome is evidence.
        fail(
            label,
            f"raised {type(exc).__name__}, expected {exception_type.__name__}: {exc}")
        return
    fail(label, f"did not raise {exception_type.__name__}")


def is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value))


def file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def manifest_mesh_map(document):
    meshes = document.get("meshes")
    if not isinstance(meshes, list):
        return {}
    return {
        entry.get("name"): entry
        for entry in meshes
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }


def stable_mesh_projection(document):
    """Fields that must match cross-build; texture hashes are not involved."""
    result = {}
    for name, entry in manifest_mesh_map(document).items():
        result[name] = {
            "node": entry.get("node"),
            "vertexCount": entry.get("vertexCount"),
            "triangleCount": entry.get("triangleCount"),
            "boundsMin": entry.get("boundsMin"),
            "boundsMax": entry.get("boundsMax"),
            "semanticHash": entry.get("semanticHash"),
        }
    return result


def validate_meta(label, document, case, atlas_px):
    meta = document.get("meta")
    if not isinstance(meta, dict):
        fail(label, "manifest meta is not an object")
        return

    missing = sorted(REQUIRED_META_KEYS - set(meta))
    if missing:
        fail(label, f"manifest meta missing keys: {missing}")
        return

    exact = {
        "schema": SCHEMA,
        "modelId": case["model_id"],
        "kind": case["kind"],
        "blenderVersion": BLENDER_VERSION,
        "atlasResolution": [atlas_px, atlas_px],
        "trackGauge": 1.435,
        "wheelBackToBack": 1.348,
        "wheelWidth": 0.135,
        "couplerHeight": 0.860,
        "couplerPivotToFace": 0.600,
    }
    exact.update({
        key: case[key]
        for key in (
            "lengthOverCouplers",
            "bodyLength",
            "width",
            "height",
            "wheelRadius",
            "bogieCentreOffset",
            "bogieWheelbase",
            "bogiePivotHeight",
        )
    })
    for key, expected in exact.items():
        actual = meta.get(key)
        if isinstance(expected, float):
            try:
                matches = abs(float(actual) - expected) <= 1e-6
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            fail(label, f"meta.{key}={actual!r}, expected {expected!r}")

    for digest_key in ("recipeDigest", "scriptDigest"):
        if not is_sha256(meta.get(digest_key)):
            fail(label, f"meta.{digest_key} is not lowercase SHA-256")


def validate_nodes_and_lod_mapping(label, document, case):
    nodes = document.get("nodes")
    if not isinstance(nodes, list):
        fail(label, "nodes is not a list")
        return
    if len(nodes) != 28:
        fail(label, f"node count {len(nodes)} != 28")
    paths = [node.get("path") for node in nodes if isinstance(node, dict)]
    if paths != sorted(paths):
        fail(label, "node table is not sorted by path")
    if len(paths) != len(set(paths)):
        fail(label, "node table contains duplicate paths")
    path_set = set(paths)

    mesh_map = manifest_mesh_map(document)
    if len(mesh_map) != case["mesh_count"]:
        fail(
            label,
            f"mesh count {len(mesh_map)} != {case['mesh_count']}")
        return
    if list(mesh_map) != sorted(mesh_map):
        fail(label, "mesh table is not sorted by name")

    lod0 = {
        name: entry for name, entry in mesh_map.items()
        if not LOD_SUFFIX_RE.search(name)
    }
    lod1 = {
        name: entry for name, entry in mesh_map.items()
        if name.endswith(lod.LOD1_SUFFIX)
    }
    lod2 = {
        name: entry for name, entry in mesh_map.items()
        if name.endswith(lod.LOD2_SUFFIX)
    }
    observed = (len(lod0), len(lod1), len(lod2))
    expected = (
        case["lod0_count"], case["lod1_count"], case["lod2_count"])
    if observed != expected:
        fail(label, f"LOD counts {observed} != {expected}")

    for name, entry in mesh_map.items():
        node = entry.get("node")
        if node not in path_set:
            fail(label, f"mesh {name} references unknown node {node!r}")
        if not is_sha256(entry.get("semanticHash")):
            fail(label, f"mesh {name} semanticHash is not lowercase SHA-256")

        base_name = LOD_SUFFIX_RE.sub("", name)
        base = lod0.get(base_name)
        if base is None:
            fail(label, f"mesh {name} has no LOD0 base {base_name}")
            continue
        if entry.get("node") != base.get("node"):
            fail(
                label,
                f"mesh {name} node {entry.get('node')!r} != "
                f"base node {base.get('node')!r}")

    for base_name, base in lod0.items():
        if base_name + lod.LOD1_SUFFIX not in lod1:
            fail(label, f"missing LOD1 for {base_name}")
        expects_lod2 = base.get("triangleCount", 0) >= 200
        has_lod2 = base_name + lod.LOD2_SUFFIX in lod2
        if has_lod2 != expects_lod2:
            fail(
                label,
                f"LOD2 eligibility mismatch for {base_name}: "
                f"triangles={base.get('triangleCount')} hasLOD2={has_lod2}")


def validate_geometry_snapshot(label, document, case):
    if not isinstance(document, dict):
        fail(label, f"snapshot type {type(document).__name__} is not dict")
        return
    validate_meta(label, document, case, PRODUCTION_ATLAS_PX)
    validate_nodes_and_lod_mapping(label, document, case)


def run_geometry_determinism_gate():
    for case in CASES:
        label = f"geometry-{case['label']}"
        first = build_module.build_geometry_snapshot(
            str(case["recipe"]), atlas_px=PRODUCTION_ATLAS_PX)
        second = build_module.build_geometry_snapshot(
            str(case["recipe"]), atlas_px=PRODUCTION_ATLAS_PX)
        validate_geometry_snapshot(label + "-first", first, case)
        validate_geometry_snapshot(label + "-second", second, case)

        if first.get("nodes") != second.get("nodes"):
            fail(label, "fresh-build node tables differ")
        first_meshes = stable_mesh_projection(first)
        second_meshes = stable_mesh_projection(second)
        if first_meshes != second_meshes:
            differing = sorted(
                name for name in set(first_meshes) | set(second_meshes)
                if first_meshes.get(name) != second_meshes.get(name))
            fail(
                label,
                f"fresh-build semantic mesh records differ for "
                f"{len(differing)}/{case['mesh_count']}: {differing[:8]}")
        print(
            f"[full_build_test] {case['label']} geometry determinism "
            f"meshes={len(first_meshes)} atlas={PRODUCTION_ATLAS_PX}",
            flush=True)


def run_export_mesh_name_gate():
    """Proves FBX Geometry IDs exactly match every manifest/object name."""
    for case in CASES:
        label = f"export-mesh-names-{case['label']}"
        scene.reset()
        recipe_obj = recipe_module.load(str(case["recipe"]))
        assembly = assemble.build_vehicle(recipe_obj)
        uvmap.unwrap_and_pack(
            assembly.objects, atlas_px=PRODUCTION_ATLAS_PX)
        objects = lod.build_lods(assembly.objects)

        if len(objects) != case["mesh_count"]:
            fail(label, f"object count {len(objects)} != {case['mesh_count']}")
            continue
        mismatched_before = sorted(
            name for name, obj in objects.items()
            if obj.data.name != name)

        if case["kind"] == "wagon":
            collision_target = next(
                (name for name in mismatched_before
                 if bpy.data.meshes.get(name) is None),
                None)
            if collision_target is None:
                fail(label, "fixture has no free mismatched mesh-name target")
            else:
                original_names = {
                    name: obj.data.name for name, obj in objects.items()}
                collision = bpy.data.meshes.new(collision_target)
                if collision.name != collision_target:
                    fail(
                        label,
                        f"collision fixture became {collision.name!r}, "
                        f"expected {collision_target!r}")
                else:
                    expect_raises(
                        label + "-external-collision",
                        ValueError,
                        lambda: build_module.prepare_export_mesh_names(objects))
                    changed = sorted(
                        name for name, obj in objects.items()
                        if obj.data.name != original_names[name])
                    if changed:
                        fail(
                            label,
                            f"collision preflight mutated {len(changed)} meshes: "
                            f"{changed[:8]}")
                bpy.data.meshes.remove(collision)

        try:
            build_module.prepare_export_mesh_names(objects)
            build_module.assert_export_mesh_names(objects)
        except Exception as exc:  # noqa: BLE001 - positive gate evidence.
            fail(label, f"valid names raised {type(exc).__name__}: {exc}")
            continue

        mismatched_after = sorted(
            name for name, obj in objects.items()
            if obj.data.name != name)
        if mismatched_after:
            fail(
                label,
                f"{len(mismatched_after)} export mesh names still differ: "
                f"{mismatched_after[:8]}")
        print(
            f"[full_build_test] {case['label']} export mesh names "
            f"objects={len(objects)} corrected={len(mismatched_before)}",
            flush=True)


def _find_node_index(assembly, path):
    for index, node in enumerate(assembly.nodes):
        if node.path == path:
            return index
    raise AssertionError(f"Test fixture node missing: {path}")


def run_assembly_validator_gate():
    label = "validate-assembly"
    scene.reset()
    recipe_obj = recipe_module.load(str(WAGON_RECIPE_PATH))
    assembly = assemble.build_vehicle(recipe_obj)
    uvmap.unwrap_and_pack(assembly.objects, atlas_px=PRODUCTION_ATLAS_PX)

    try:
        validate.validate_assembly(assembly, recipe_obj)
    except Exception as exc:  # noqa: BLE001 - positive contract evidence.
        fail(label, f"valid assembly raised {type(exc).__name__}: {exc}")
        return

    original_nodes = list(assembly.nodes)

    body = assembly.objects["Visuals.Body.BoxBody"]
    body_name = body.name
    original_body_node = assembly.mesh_nodes[body_name]
    assembly.mesh_nodes[body_name] = "Interaction"
    expect_raises(
        label + "-body-wrong-known-node",
        ValueError,
        lambda: validate.validate_assembly(assembly, recipe_obj))
    assembly.mesh_nodes[body_name] = original_body_node

    tread_name = (
        "Visuals.RunningGear.Bogie__front."
        "Wheelset__front_leading.WheelLeft.TreadAndFlange")
    original_tread_node = assembly.mesh_nodes[tread_name]
    assembly.mesh_nodes[tread_name] = (
        "Visuals/RunningGear/Bogie__front/Wheelset__front_leading")
    expect_raises(
        label + "-wheel-tread-wrong-known-node",
        ValueError,
        lambda: validate.validate_assembly(assembly, recipe_obj))
    assembly.mesh_nodes[tread_name] = original_tread_node

    parent_probe = bpy.data.objects.new("TGEL.Task15.ParentProbe", None)
    bpy.context.scene.collection.objects.link(parent_probe)
    original_body_world = body.matrix_world.copy()
    body.parent = parent_probe
    expect_raises(
        label + "-parented-flat-export-mesh",
        ValueError,
        lambda: validate.validate_assembly(assembly, recipe_obj))
    body.parent = None
    body.matrix_world = original_body_world
    bpy.data.objects.remove(parent_probe, do_unlink=True)

    assembly.nodes = [
        node for node in original_nodes if node.path != "Interaction"]
    expect_raises(
        label + "-missing-node",
        ValueError,
        lambda: validate.validate_assembly(assembly, recipe_obj))
    assembly.nodes = list(original_nodes)

    interaction_index = _find_node_index(assembly, "Interaction")
    interaction_node = assembly.nodes[interaction_index]
    assembly.nodes[interaction_index] = replace(
        interaction_node, local_position=(100.0, 100.0, 100.0))
    expect_raises(
        label + "-interaction-root-translation",
        ValueError,
        lambda: validate.validate_assembly(assembly, recipe_obj))
    assembly.nodes = list(original_nodes)

    markers_index = _find_node_index(assembly, "Markers")
    markers_node = assembly.nodes[markers_index]
    assembly.nodes[markers_index] = replace(
        markers_node, local_position=(1.0, 0.0, 0.0))
    expect_raises(
        label + "-markers-root-translation",
        ValueError,
        lambda: validate.validate_assembly(assembly, recipe_obj))
    assembly.nodes = list(original_nodes)

    visuals_index = _find_node_index(assembly, "Visuals")
    visuals_node = assembly.nodes[visuals_index]
    assembly.nodes[visuals_index] = replace(
        visuals_node,
        local_rotation_quat=(0.0, 0.3826834324, 0.0, 0.9238795325))
    expect_raises(
        label + "-forbidden-root-rotation",
        ValueError,
        lambda: validate.validate_assembly(assembly, recipe_obj))
    assembly.nodes = list(original_nodes)

    wheel_path = (
        "Visuals/RunningGear/Bogie__front/"
        "Wheelset__front_leading/WheelLeft")
    wheel_index = _find_node_index(assembly, wheel_path)
    wheel_node = assembly.nodes[wheel_index]
    assembly.nodes[wheel_index] = replace(
        wheel_node, local_position=(-0.7000, *wheel_node.local_position[1:]))
    expect_raises(
        label + "-back-to-back",
        ValueError,
        lambda: validate.validate_assembly(assembly, recipe_obj))
    assembly.nodes = list(original_nodes)

    mirrored_tread = assembly.objects[tread_name]
    original_tread_x = [vertex.co.x for vertex in mirrored_tread.data.vertices]
    for vertex in mirrored_tread.data.vertices:
        vertex.co.x = -vertex.co.x
    mirrored_tread.data.update()
    expect_raises(
        label + "-mirrored-flange",
        ValueError,
        lambda: validate.validate_assembly(assembly, recipe_obj))
    for vertex, original_x in zip(
            mirrored_tread.data.vertices, original_tread_x):
        vertex.co.x = original_x
    mirrored_tread.data.update()

    original_tread_radial = [
        (vertex.co.y, vertex.co.z)
        for vertex in mirrored_tread.data.vertices
    ]
    for vertex in mirrored_tread.data.vertices:
        vertex.co.y *= 0.5
        vertex.co.z *= 0.5
    mirrored_tread.data.update()
    expect_raises(
        label + "-half-radius-wheel",
        ValueError,
        lambda: validate.validate_assembly(assembly, recipe_obj))
    for vertex, (original_y, original_z) in zip(
            mirrored_tread.data.vertices, original_tread_radial):
        vertex.co.y = original_y
        vertex.co.z = original_z
    mirrored_tread.data.update()

    face_path = "Couplers/CouplerPivot__front/CouplerFace"
    face_index = _find_node_index(assembly, face_path)
    face_node = assembly.nodes[face_index]
    assembly.nodes[face_index] = replace(
        face_node, local_position=(0.0, 0.0, 0.550))
    expect_raises(
        label + "-coupler-plane",
        ValueError,
        lambda: validate.validate_assembly(assembly, recipe_obj))
    assembly.nodes = list(original_nodes)

    vertex = body.data.vertices[0]
    original_coordinate = vertex.co.copy()
    vertex.co = scene.to_blender((
        recipe_obj.width / 2.0 + 0.010,
        recipe_obj.height / 2.0,
        0.0,
    ))
    body.data.update()
    expect_raises(
        label + "-envelope",
        ValueError,
        lambda: validate.validate_assembly(assembly, recipe_obj))
    vertex.co = original_coordinate
    body.data.update()

    body_length_vertex = body.data.vertices[0]
    original_body_length_coordinate = body_length_vertex.co.copy()
    body_length_vertex.co = scene.to_blender((
        0.0,
        recipe_obj.height / 2.0,
        recipe_obj.body_length / 2.0 + 0.100,
    ))
    body.data.update()
    expect_raises(
        label + "-body-length-envelope",
        ValueError,
        lambda: validate.validate_assembly(assembly, recipe_obj))
    body_length_vertex.co = original_body_length_coordinate
    body.data.update()

    floor_vertex = body.data.vertices[0]
    original_floor_coordinate = floor_vertex.co.copy()
    floor_vertex.co = scene.to_blender((0.0, -0.020, 0.0))
    body.data.update()
    expect_raises(
        label + "-body-below-railhead",
        ValueError,
        lambda: validate.validate_assembly(assembly, recipe_obj))
    floor_vertex.co = original_floor_coordinate
    body.data.update()

    nan_vertex = body.data.vertices[0]
    original_nan_coordinate = nan_vertex.co.copy()
    nan_vertex.co.x = float("nan")
    body.data.update()
    expect_raises(
        label + "-nan-vertex",
        ValueError,
        lambda: validate.validate_assembly(assembly, recipe_obj))
    nan_vertex.co = original_nan_coordinate
    body.data.update()

    door = assembly.objects["Visuals.Body.DoorLeft"]
    door.data.calc_loop_triangles()
    exterior_triangles = [
        triangle for triangle in door.data.loop_triangles
        if abs(float(door.data.polygons[triangle.polygon_index].normal.x)) >= 0.8
        and float(
            door.data.polygons[triangle.polygon_index].center.x
            * door.data.polygons[triangle.polygon_index].normal.x) > 1e-6
        and float(triangle.area) > 1e-12
    ]
    collapsed_triangle = max(
        exterior_triangles,
        key=lambda triangle: (float(triangle.area), -triangle.index))
    door_uv = door.data.uv_layers.active.data
    collapsed_loops = tuple(collapsed_triangle.loops)
    original_triangle_uvs = [door_uv[index].uv.copy() for index in collapsed_loops]
    collapsed_coordinate = original_triangle_uvs[0].copy()
    for loop_index in collapsed_loops:
        door_uv[loop_index].uv = collapsed_coordinate
    expect_raises(
        label + "-collapsed-visible-uv-triangle",
        ValueError,
        lambda: validate.validate_assembly(assembly, recipe_obj))
    for loop_index, original_uv in zip(collapsed_loops, original_triangle_uvs):
        door_uv[loop_index].uv = original_uv

    uv_object = assembly.objects["Visuals.Body.BoxBody"]
    uv_layer = uv_object.data.uv_layers.active
    uv_object.data.uv_layers.remove(uv_layer)
    expect_raises(
        label + "-missing-uv",
        ValueError,
        lambda: validate.validate_assembly(assembly, recipe_obj))


class InjectedAtomicFailure(RuntimeError):
    pass


def _staging_entries(parent, destination_name):
    prefix = f".{destination_name}.staging-"
    return sorted(path.name for path in parent.iterdir() if path.name.startswith(prefix))


def run_atomic_output_gate(temp_root):
    label = "atomic-output"
    destination = temp_root / "failed-output"

    def failing_writer(stage):
        stage = Path(stage)
        (stage / "partial.txt").write_text("partial", encoding="utf-8")
        raise InjectedAtomicFailure("injected after partial staging write")

    expect_raises(
        label + "-injected",
        InjectedAtomicFailure,
        lambda: build_module.atomic_write_directory(destination, failing_writer))
    if destination.exists():
        fail(label, "injected failure published the destination")
    leftovers = _staging_entries(temp_root, destination.name)
    if leftovers:
        fail(label, f"injected failure left staging directories: {leftovers}")

    existing = temp_root / "existing-output"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("preserve-me", encoding="utf-8")
    writer_called = [False]

    def forbidden_writer(_stage):
        writer_called[0] = True

    expect_raises(
        label + "-existing",
        FileExistsError,
        lambda: build_module.atomic_write_directory(existing, forbidden_writer))
    if writer_called[0]:
        fail(label, "existing destination was rejected only after writer ran")
    if not sentinel.exists() or sentinel.read_text(encoding="utf-8") != "preserve-me":
        fail(label, "existing destination sentinel was changed")
    leftovers = _staging_entries(temp_root, existing.name)
    if leftovers:
        fail(label, f"existing-destination rejection left staging: {leftovers}")

    base_recipe = json.loads(WAGON_RECIPE_PATH.read_text(encoding="utf-8"))
    traversal_model_id = "../outside"
    absolute_model_id = str((temp_root / "absolute-outside").resolve())
    malicious_cases = (
        ("traversal", traversal_model_id, temp_root / "outside"),
        ("absolute", absolute_model_id, temp_root / "absolute-outside"),
    )
    frozen_recipe_cases = (
        ("enlarged-width", ("width",), 4.0),
        ("nan-wheel-radius", ("wheelRadius",), float("nan")),
        ("changed-seed", ("seed",), base_recipe["seed"] + 1),
        ("fractional-seed", ("seed",), base_recipe["seed"] + 0.5),
        ("changed-livery", ("livery", "body", 0), 0.197),
        ("extra-livery-key", ("livery", "unauthorized"), [0.1, 0.2, 0.3]),
        ("nan-livery", ("livery", "stencil", 1), float("nan")),
    )
    heavy_calls = {"assemble": 0, "export": 0}
    original_assemble = build_module.assemble.build_vehicle
    original_export = build_module.export.export_fbx

    def forbidden_assemble(*_args, **_kwargs):
        heavy_calls["assemble"] += 1
        raise RuntimeError("malicious recipe reached vehicle assembly")

    def forbidden_export(*_args, **_kwargs):
        heavy_calls["export"] += 1
        raise RuntimeError("malicious recipe reached FBX export")

    build_module.assemble.build_vehicle = forbidden_assemble
    build_module.export.export_fbx = forbidden_export
    try:
        for case_name, model_id, escaped_stem in malicious_cases:
            recipe_document = dict(base_recipe)
            recipe_document["modelId"] = model_id
            malicious_recipe = temp_root / f"{case_name}-recipe.json"
            malicious_recipe.write_text(
                json.dumps(recipe_document), encoding="utf-8")
            malicious_destination = temp_root / f"{case_name}-output"

            expect_raises(
                label + f"-{case_name}-model-id",
                ValueError,
                lambda recipe_path=malicious_recipe,
                       destination=malicious_destination:
                    build_module.run_build(
                        str(recipe_path),
                        str(destination),
                        atlas_px=TEST_ATLAS_PX))

            if malicious_destination.exists():
                fail(
                    label,
                    f"{case_name} modelId published destination")
            leftovers = _staging_entries(
                temp_root, malicious_destination.name)
            if leftovers:
                fail(
                    label,
                    f"{case_name} modelId left staging: {leftovers}")
            escaped_paths = (
                Path(f"{escaped_stem}.fbx"),
                Path(f"{escaped_stem}.manifest.json"),
            )
            escaped = [str(path) for path in escaped_paths if path.exists()]
            if escaped:
                fail(
                    label,
                    f"{case_name} modelId wrote escaped artifacts: {escaped}")

        for case_name, field_path, replacement in frozen_recipe_cases:
            recipe_document = json.loads(json.dumps(base_recipe))
            target = recipe_document
            for field in field_path[:-1]:
                target = target[field]
            target[field_path[-1]] = replacement
            invalid_recipe = temp_root / f"{case_name}-recipe.json"
            invalid_recipe.write_text(
                json.dumps(recipe_document), encoding="utf-8")
            invalid_destination = temp_root / f"{case_name}-output"

            expect_raises(
                label + f"-{case_name}",
                ValueError,
                lambda recipe_path=invalid_recipe,
                       destination=invalid_destination:
                    build_module.run_build(
                        str(recipe_path),
                        str(destination),
                        atlas_px=TEST_ATLAS_PX))

            if invalid_destination.exists():
                fail(label, f"{case_name} recipe published destination")
            leftovers = _staging_entries(
                temp_root, invalid_destination.name)
            if leftovers:
                fail(label, f"{case_name} recipe left staging: {leftovers}")
    finally:
        build_module.assemble.build_vehicle = original_assemble
        build_module.export.export_fbx = original_export

    if heavy_calls != {"assemble": 0, "export": 0}:
        fail(label, f"invalid frozen recipe reached heavy work: {heavy_calls}")


def _save_constant_png(path, width, height, rgba):
    image = bpy.data.images.new(
        name=f"TGEL.FullBuildTest.{path.stem}.{width}",
        width=width,
        height=height,
        alpha=True,
        float_buffer=False,
    )
    try:
        image.generated_type = 'BLANK'
        image.generated_color = rgba
        image.filepath_raw = str(path)
        image.file_format = 'PNG'
        image.save()
    finally:
        bpy.data.images.remove(image)


def validate_full_output(label, output_dir, case):
    expected_files = {
        f"{case['model_id']}.fbx",
        "albedo.png",
        "normal.png",
        "mask.png",
        f"{case['model_id']}.manifest.json",
    }
    actual_files = {
        path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        fail(
            label,
            f"output file set differs; missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}")

    for filename in expected_files:
        path = output_dir / filename
        if not path.is_file() or path.stat().st_size <= 0:
            fail(label, f"output {filename} missing or empty")

    manifest_path = output_dir / f"{case['model_id']}.manifest.json"
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - malformed output is test evidence.
        fail(label, f"manifest could not be read: {type(exc).__name__}: {exc}")
        return {}

    validate_meta(label, document, case, TEST_ATLAS_PX)
    validate_nodes_and_lod_mapping(label, document, case)

    textures = document.get("textures")
    if not isinstance(textures, list):
        fail(label, "textures is not a list")
        return document
    texture_map = {
        entry.get("name"): entry
        for entry in textures
        if isinstance(entry, dict)
    }
    if set(texture_map) != set(EXPECTED_TEXTURES):
        fail(label, f"texture names {sorted(texture_map)} != {sorted(EXPECTED_TEXTURES)}")
    for name, (filename, color_space) in EXPECTED_TEXTURES.items():
        entry = texture_map.get(name)
        if entry is None:
            continue
        expected = {
            "file": filename,
            "resolution": [TEST_ATLAS_PX, TEST_ATLAS_PX],
            "colorSpace": color_space,
        }
        for key, value in expected.items():
            if entry.get(key) != value:
                fail(label, f"texture {name}.{key}={entry.get(key)!r}, expected {value!r}")
        texture_path = output_dir / filename
        if entry.get("sha256") != file_sha256(texture_path):
            fail(label, f"texture {name} manifest SHA-256 mismatch")

    return document


def run_texture_validator_negative_gates(output_dir, document):
    label = "validate-textures"
    texture_paths = {
        entry["name"]: str(output_dir / entry["file"])
        for entry in document.get("textures", [])
        if isinstance(entry, dict) and "name" in entry and "file" in entry
    }
    try:
        validate.validate_textures(
            texture_paths, expected_resolution=TEST_ATLAS_PX)
    except Exception as exc:  # noqa: BLE001 - positive contract evidence.
        fail(label, f"valid texture set raised {type(exc).__name__}: {exc}")
        return

    missing = dict(texture_paths)
    missing["mask"] = str(output_dir / "does-not-exist.png")
    expect_raises(
        label + "-missing",
        ValueError,
        lambda: validate.validate_textures(
            missing, expected_resolution=TEST_ATLAS_PX))

    wrong_size_path = output_dir.parent / "wrong-size-mask.png"
    _save_constant_png(wrong_size_path, 16, 16, (0.5, 0.5, 0.5, 0.5))
    wrong_size = dict(texture_paths)
    wrong_size["mask"] = str(wrong_size_path)
    expect_raises(
        label + "-size",
        ValueError,
        lambda: validate.validate_textures(
            wrong_size, expected_resolution=TEST_ATLAS_PX))

    constant_path = output_dir.parent / "constant-roughness-mask.png"
    _save_constant_png(
        constant_path,
        TEST_ATLAS_PX,
        TEST_ATLAS_PX,
        (0.5, 0.25, 0.10, 0.62))
    constant = dict(texture_paths)
    constant["mask"] = str(constant_path)
    expect_raises(
        label + "-constant-roughness",
        ValueError,
        lambda: validate.validate_textures(
            constant, expected_resolution=TEST_ATLAS_PX))


def run_full_orchestration_gate(temp_root):
    case = CASES[0]
    outputs = (temp_root / "wagon-run-a", temp_root / "wagon-run-b")
    documents = []
    for index, output in enumerate(outputs):
        build_module.run_build(
            str(case["recipe"]),
            str(output),
            atlas_px=TEST_ATLAS_PX)
        documents.append(validate_full_output(
            f"full-wagon-{index + 1}", output, case))

    first, second = documents
    if first.get("nodes") != second.get("nodes"):
        fail("full-wagon-repeat", "full-build node tables differ")
    if stable_mesh_projection(first) != stable_mesh_projection(second):
        first_meshes = stable_mesh_projection(first)
        second_meshes = stable_mesh_projection(second)
        differing = sorted(
            name for name in set(first_meshes) | set(second_meshes)
            if first_meshes.get(name) != second_meshes.get(name))
        fail(
            "full-wagon-repeat",
            f"semantic mesh records differ for {len(differing)}/200: "
            f"{differing[:8]}")

    # Intentionally do NOT compare texture hashes across builds.  Each output
    # was individually checked against its own manifest snapshot above.
    if documents:
        run_texture_validator_negative_gates(outputs[0], documents[0])
    print(
        f"[full_build_test] wagon full orchestration runs=2 "
        f"atlas={TEST_ATLAS_PX} files=5",
        flush=True)


def run_export_contract_gate():
    label = "fbx-export-contract"
    if export.FBX_KWARGS.get("use_triangles") is not False:
        fail(
            label,
            "FBX exporter must keep use_triangles=False; TGEL owns exact-name "
            "temporary triangulation and rollback")
    if not hasattr(export, "_invoke_fbx_export"):
        fail(label, "missing injected-operator seam _invoke_fbx_export")
        return
    if not (0.0 < export.MAX_CORNER_NORMAL_DELTA <= 1e-3):
        fail(
            label,
            "corner-normal tolerance must stay within the frozen 1e-3 bound")

    scene.reset()
    fixture_name = "TGEL.ExportContract.ExactLoopTriangles"
    fixture_mesh = bpy.data.meshes.new(fixture_name)
    fixture_mesh.from_pydata(
        (
            (-0.312, 0.1, 0.3),
            (-0.312, 0.1, -0.15),
            (-0.3110865653, 0.0954077989, -0.15),
            (-0.3110865653, 0.0954077989, 0.3),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (1.5, 1.0, 0.0),
            (3.0, 0.0, 0.0),
            (3.5, 0.0, 0.0),
        ),
        ((7, 8),),
        ((0, 1, 2, 3), (4, 5, 6)))
    obj = bpy.data.objects.new(fixture_name, fixture_mesh)
    bpy.context.scene.collection.objects.link(obj)
    materials = (
        bpy.data.materials.new("TGEL.ExportContract.MaterialA"),
        bpy.data.materials.new("TGEL.ExportContract.MaterialB"),
    )
    for material in materials:
        fixture_mesh.materials.append(material)
    fixture_mesh.polygons[0].material_index = 0
    fixture_mesh.polygons[0].use_smooth = False
    fixture_mesh.polygons[1].material_index = 1
    fixture_mesh.polygons[1].use_smooth = True
    fixture_uv = fixture_mesh.uv_layers.new(name="TGEL.ExportContract.UV")
    for loop_index, uv_loop in enumerate(fixture_uv.data):
        uv_loop.uv = (
            float(loop_index + 1) / 16.0,
            float((loop_index * 7) % 17) / 17.0,
        )
    fixture_mesh.update()
    original = obj.data
    original_name = original.name
    original_pointer = original.as_pointer()
    original_snapshot = scene.collect_mesh_data(obj)
    original_mesh_ids = {
        (mesh.as_pointer(), mesh.name) for mesh in bpy.data.meshes}
    original.calc_loop_triangles()
    source_triangles = tuple(original.loop_triangles)
    expected_triangle_vertices = tuple(
        tuple(triangle.vertices) for triangle in source_triangles)
    expected_triangle_uvs = tuple(
        tuple(float(component) for component in fixture_uv.data[loop_index].uv)
        for triangle in source_triangles
        for loop_index in triangle.loops)
    expected_material_indices = tuple(
        original.polygons[triangle.polygon_index].material_index
        for triangle in source_triangles)
    expected_smooth_flags = tuple(
        original.polygons[triangle.polygon_index].use_smooth
        for triangle in source_triangles)
    expected_corner_normals = tuple(
        tuple(
            float(component)
            for component in original.corner_normals[loop_index].vector)
        for triangle in source_triangles
        for loop_index in triangle.loops)
    expected_loose_edges = tuple(sorted(
        tuple(sorted(edge.vertices))
        for edge in original.edges if edge.is_loose))

    class InjectedExportFailure(BaseException):
        pass

    calls = []
    original_invoke = export._invoke_fbx_export

    def inspect_temporary_mesh():
        mesh = obj.data
        if mesh is original:
            fail(label, "exporter saw the original source mesh")
        if mesh.name != obj.name:
            fail(label, f"temporary exact name {mesh.name!r} != {obj.name!r}")
        if any(len(polygon.vertices) != 3 for polygon in mesh.polygons):
            fail(label, "exporter saw a non-triangle polygon")
        actual_triangle_vertices = tuple(
            tuple(polygon.vertices) for polygon in mesh.polygons)
        if actual_triangle_vertices != expected_triangle_vertices:
            fail(
                label,
                "temporary ordered triangle tuples differ from source "
                f"loop triangles: {actual_triangle_vertices!r} != "
                f"{expected_triangle_vertices!r}")
        actual_triangle_uvs = tuple(
            tuple(float(component) for component in uv_loop.uv)
            for uv_loop in mesh.uv_layers.active.data)
        if actual_triangle_uvs != expected_triangle_uvs:
            fail(label, "temporary per-corner UV sequence differs from source")
        actual_material_indices = tuple(
            polygon.material_index for polygon in mesh.polygons)
        if actual_material_indices != expected_material_indices:
            fail(label, "temporary per-triangle material sequence differs")
        actual_smooth_flags = tuple(
            polygon.use_smooth for polygon in mesh.polygons)
        if actual_smooth_flags != expected_smooth_flags:
            fail(label, "temporary per-triangle smooth flags differ")
        actual_corner_normals = tuple(
            tuple(float(component) for component in corner.vector)
            for corner in mesh.corner_normals)
        maximum_normal_delta = max(
            (abs(actual - expected)
             for actual_row, expected_row in zip(
                 actual_corner_normals, expected_corner_normals)
             for actual, expected in zip(actual_row, expected_row)),
            default=0.0)
        if (len(actual_corner_normals) != len(expected_corner_normals) or
                not all(
                    math.isfinite(component)
                    for normal in actual_corner_normals
                    for component in normal) or
                maximum_normal_delta > export.MAX_CORNER_NORMAL_DELTA):
            fail(
                label,
                "temporary corner normals exceed frozen tolerance: "
                f"max delta {maximum_normal_delta}")
        actual_loose_edges = tuple(sorted(
            tuple(sorted(edge.vertices))
            for edge in mesh.edges if edge.is_loose))
        if actual_loose_edges != expected_loose_edges:
            fail(label, "temporary loose-edge sequence differs from source")

    def inspect_failed_export(_path):
        calls.append(obj.data.as_pointer())
        inspect_temporary_mesh()
        raise InjectedExportFailure("injected inside FBX operator seam")

    export._invoke_fbx_export = inspect_failed_export
    try:
        with tempfile.TemporaryDirectory(prefix="tgel-export-contract-") as temp_dir:
            expect_raises(
                label + "-injected",
                InjectedExportFailure,
                lambda: export.export_fbx(
                    [obj], str(Path(temp_dir) / "injected.fbx")))
    finally:
        export._invoke_fbx_export = original_invoke

    if len(calls) != 1:
        fail(label, f"injected exporter calls {len(calls)} != 1")
    if obj.data is not original or original.as_pointer() != original_pointer:
        fail(label, "source mesh pointer was not restored")
    if original.name != original_name:
        fail(label, f"source mesh name {original.name!r} != {original_name!r}")
    if scene.collect_mesh_data(obj) != original_snapshot:
        fail(label, "source geometry changed after injected export failure")
    final_mesh_ids = {
        (mesh.as_pointer(), mesh.name) for mesh in bpy.data.meshes}
    if final_mesh_ids != original_mesh_ids:
        fail(label, "mesh ID/name set changed after injected export failure")

    successful_calls = []

    def inspect_successful_export(_path):
        successful_calls.append(obj.data.as_pointer())
        inspect_temporary_mesh()
        return {"FINISHED"}

    export._invoke_fbx_export = inspect_successful_export
    try:
        with tempfile.TemporaryDirectory(prefix="tgel-export-contract-") as temp_dir:
            export.export_fbx(
                [obj], str(Path(temp_dir) / "successful.fbx"))
    except Exception as exc:  # noqa: BLE001 - positive operator seam evidence.
        fail(label, f"successful operator seam raised {type(exc).__name__}: {exc}")
    finally:
        export._invoke_fbx_export = original_invoke

    if len(successful_calls) != 1:
        fail(label, f"successful exporter calls {len(successful_calls)} != 1")
    if obj.data is not original or original.as_pointer() != original_pointer:
        fail(label, "source mesh pointer was not restored after successful export")
    if original.name != original_name:
        fail(
            label,
            f"source mesh name after successful export {original.name!r} != "
            f"{original_name!r}")
    if scene.collect_mesh_data(obj) != original_snapshot:
        fail(label, "source geometry changed after successful export")
    successful_mesh_ids = {
        (mesh.as_pointer(), mesh.name) for mesh in bpy.data.meshes}
    if successful_mesh_ids != original_mesh_ids:
        fail(label, "mesh ID/name set changed after successful export")


test_args = (
    sys.argv[sys.argv.index("--") + 1:]
    if "--" in sys.argv else [])
run_export_contract_gate()
if test_args == ["--export-contract-gate-only"]:
    pass
elif test_args == ["--export-mesh-name-gate-only"]:
    run_export_mesh_name_gate()
elif test_args:
    fail("arguments", f"unknown test arguments: {test_args}")
else:
    run_export_mesh_name_gate()
    run_geometry_determinism_gate()
    run_assembly_validator_gate()

    with tempfile.TemporaryDirectory(prefix="tgel-task15-full-build-") as temp_dir:
        temp_root = Path(temp_dir)
        run_atomic_output_gate(temp_root)
        run_full_orchestration_gate(temp_root)

if failures:
    print("BPY TESTS FAIL:", "; ".join(failures), flush=True)
    sys.exit(1)

print("BPY TESTS OK", flush=True)
sys.exit(0)
