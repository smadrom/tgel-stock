"""LOD generation plus exact fresh-build determinism contracts.

Task 14 still proves repeated DECIMATE runs fed the exact same packed LOD0.
Task 15's corrective boundary additionally proves two independent complete
``assemble -> UV -> LOD`` builds have identical exact semantic hashes for every
LOD0/LOD1/LOD2 mesh; hash fields and tolerance are intentionally unchanged.
"""

import hashlib
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import bpy  # noqa: E402

from tgel_stock import assemble  # noqa: E402
from tgel_stock import canonical  # noqa: E402
from tgel_stock import lod  # noqa: E402
from tgel_stock import recipe as recipe_module  # noqa: E402
from tgel_stock import scene  # noqa: E402
from tgel_stock import uvmap  # noqa: E402


WAGON_RECIPE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "recipes",
    "basic-box-wagon.rollingstock.json")
LOCOMOTIVE_RECIPE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "recipes",
    "basic-diesel-locomotive.rollingstock.json")

LOD1_SUFFIX = "__LOD1"
LOD2_SUFFIX = "__LOD2"
LOD2_SOURCE_TRIANGLE_FLOOR = 200
UV_SLACK = 1e-4

# Tasks 1-13 are accepted and frozen.  These counts make Task 14 fail loudly
# if it accidentally compensates for an upstream topology change.
CASES = (
    {
        "label": "wagon",
        "path": WAGON_RECIPE_PATH,
        "base_objects": 94,
        "lod2_objects": 12,
        # Four accepted corner steps are now connected 28-triangle U-frames
        # instead of detached 12-triangle tread boxes.
        "lod0_triangles": 11948,
        "lod1_triangles": 4716,
        "lod2_triangles": 786,
        "lod0_budget": 15000,
        "corrected_vertices": {},
        "cleanup_census": {},
    },
    {
        "label": "locomotive",
        "path": LOCOMOTIVE_RECIPE_PATH,
        "base_objects": 77,
        "lod2_objects": 19,
        "lod0_triangles": 16956,
        "lod1_triangles": 6751,
        "lod2_triangles": 1442,
        "lod0_budget": 30000,
        "corrected_vertices": {
            "Visuals.Body.CabGlass__LOD1": 7,
        },
        "cleanup_census": {
            "Visuals.Body.CabGlass__LOD1": (4, 2),
        },
    },
)

failures = []


def fail(label, message):
    failures.append(f"[{label}] {message}")


def triangle_count(obj):
    """Returns rendered triangles, not polygon count (n-gons count as n-2)."""
    obj.data.calc_loop_triangles()
    return len(obj.data.loop_triangles)


def semantic_hash(obj):
    positions, normals, uvs, triangles, _bounds_min, _bounds_max = (
        scene.collect_mesh_data(obj))
    return canonical.geometry_hash(positions, normals, uvs, triangles)


def matrix_signature(obj):
    return tuple(round(float(value), 8) for row in obj.matrix_local for value in row)


def material_signature(obj):
    return tuple(slot.material for slot in obj.material_slots)


def validate_non_face_cleanup_fixture():
    """Exercises isolated vertices, wire edges and a face-to-face wire edge."""
    mesh = bpy.data.meshes.new("TGEL.LODTest.NonFaceCleanup")
    materials = [
        bpy.data.materials.new("TGEL.LODTest.NonFaceCleanup.0"),
        bpy.data.materials.new("TGEL.LODTest.NonFaceCleanup.1"),
    ]
    try:
        mesh.from_pydata(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (2.0, 1.0, 0.0),
                (4.0, 0.0, 0.0),
                (5.0, 0.0, 0.0),
                (6.0, 0.0, 0.0),
            ],
            [(0, 3), (6, 7)],
            [(0, 1, 2), (3, 4, 5)],
        )
        for material in materials:
            mesh.materials.append(material)
        mesh.polygons[0].material_index = 0
        mesh.polygons[0].use_smooth = False
        mesh.polygons[1].material_index = 1
        mesh.polygons[1].use_smooth = True
        uv_layer = mesh.uv_layers.new(name="UVMap")
        for index, loop_uv in enumerate(uv_layer.data):
            loop_uv.uv = ((index % 3) / 2.0, (index // 3) / 2.0)

        before = lod._rendered_face_signature(mesh)
        removed_vertices, removed_edges = lod._remove_non_face_geometry(mesh)
        if (removed_vertices, removed_edges) != (3, 2):
            fail("cleanup-fixture", (
                "cleanup counts "
                f"{removed_vertices}/{removed_edges} != 3/2"))
        if len(mesh.vertices) != 6 or len(mesh.polygons) != 2:
            fail("cleanup-fixture", (
                f"cleanup result vertices/polygons "
                f"{len(mesh.vertices)}/{len(mesh.polygons)} != 6/2"))
        if lod._rendered_face_signature(mesh) != before:
            fail("cleanup-fixture", "rendered face signature changed")
        second = lod._remove_non_face_geometry(mesh)
        if second != (0, 0):
            fail("cleanup-fixture", f"clean-mesh rerun was not a no-op: {second}")
    finally:
        bpy.data.meshes.remove(mesh)
        for material in materials:
            bpy.data.materials.remove(material)


def uv_contract(obj):
    """Returns atlas bounds and total UV triangle area, or an error string."""
    mesh = obj.data
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        return None, f"{obj.name} has no active UV layer"
    if len(uv_layer.data) != len(mesh.loops):
        return None, (
            f"{obj.name} UV loop count {len(uv_layer.data)} != "
            f"mesh loop count {len(mesh.loops)}")

    coordinates = [
        (float(loop_uv.uv[0]), float(loop_uv.uv[1]))
        for loop_uv in uv_layer.data
    ]
    if not coordinates:
        return None, f"{obj.name} has an empty UV layer"
    if not all(math.isfinite(component) for uv in coordinates for component in uv):
        return None, f"{obj.name} has non-finite UV coordinates"
    if not all(
            -UV_SLACK <= component <= 1.0 + UV_SLACK
            for uv in coordinates for component in uv):
        return None, f"{obj.name} has UV coordinates outside [0,1]"

    mesh.calc_loop_triangles()
    total_area = 0.0
    for triangle in mesh.loop_triangles:
        uv0, uv1, uv2 = (
            uv_layer.data[loop_index].uv for loop_index in triangle.loops)
        total_area += abs(
            (float(uv1[0]) - float(uv0[0]))
            * (float(uv2[1]) - float(uv0[1]))
            - (float(uv2[0]) - float(uv0[0]))
            * (float(uv1[1]) - float(uv0[1]))) * 0.5
    if total_area <= 1e-12:
        return None, f"{obj.name} has no non-degenerate UV triangle area"

    us = [uv[0] for uv in coordinates]
    vs = [uv[1] for uv in coordinates]
    return {
        "name": uv_layer.name,
        "min_u": min(us),
        "max_u": max(us),
        "min_v": min(vs),
        "max_v": max(vs),
        "area": total_area,
    }, None


def remove_variants(result, base_names):
    """Removes one generated pass so the exact same LOD0 can be rerun."""
    for name in sorted(set(result) - set(base_names), reverse=True):
        obj = result[name]
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def build_lods_with_cleanup_census(objects):
    """Runs production LOD generation while recording every actual cleanup."""
    original = lod._remove_non_face_geometry
    census = {}

    def tracked(mesh):
        removed = original(mesh)
        if removed != (0, 0):
            if mesh.name in census:
                raise RuntimeError(
                    f"cleanup ran twice for generated mesh {mesh.name}")
            census[mesh.name] = removed
        return removed

    lod._remove_non_face_geometry = tracked
    try:
        result = lod.build_lods(objects)
    finally:
        lod._remove_non_face_geometry = original
    return result, census


def validate_cleanup_census(label, actual, expected, context):
    if actual != expected:
        fail(label, f"{context}: cleanup census {actual} != {expected}")


def expected_names(base_names, base_triangles):
    names = set(base_names)
    names.update(name + LOD1_SUFFIX for name in base_names)
    names.update(
        name + LOD2_SUFFIX for name in base_names
        if base_triangles[name] >= LOD2_SOURCE_TRIANGLE_FLOOR)
    return names


def assert_sources_unchanged(label, input_mapping, base_names, snapshots, context):
    if tuple(input_mapping.keys()) != base_names:
        fail(label, f"{context}: input mapping keys/order changed")
    for name in base_names:
        source = input_mapping.get(name)
        snapshot = snapshots[name]
        if source is not snapshot["object"]:
            fail(label, f"{context}: source object identity changed for {name}")
            continue
        if source.data is not snapshot["data"]:
            fail(label, f"{context}: source mesh identity changed for {name}")
        if semantic_hash(source) != snapshot["hash"]:
            fail(label, f"{context}: source semantic hash changed for {name}")


def validate_collision_preflight(
        label, input_mapping, base_names, base_triangles, snapshots):
    """Proves all target names are checked before the first clone is made."""
    all_variant_names = (
        expected_names(base_names, base_triangles) - set(base_names))
    # Collide with the final source, so an implementation that checks lazily
    # after each clone would leave almost a complete partial result behind.
    collision_name = base_names[-1] + LOD1_SUFFIX

    collision_mesh = bpy.data.meshes.new(collision_name + ".ObjectCollisionMesh")
    collision_object = bpy.data.objects.new(collision_name, collision_mesh)
    bpy.context.scene.collection.objects.link(collision_object)
    try:
        try:
            lod.build_lods(input_mapping)
            fail(label, "object-name collision did not raise ValueError")
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001 - record the exact wrong API outcome.
            fail(label, (
                f"object-name collision raised {type(exc).__name__}, expected ValueError"))

        unexpected_objects = sorted(
            name for name in all_variant_names
            if name != collision_name and bpy.data.objects.get(name) is not None)
        if unexpected_objects:
            fail(label, (
                "object collision left partial LOD objects: "
                f"{unexpected_objects}"))
        assert_sources_unchanged(
            label, input_mapping, base_names, snapshots, "object collision")
    finally:
        bpy.data.objects.remove(collision_object, do_unlink=True)
        if collision_mesh.users == 0:
            bpy.data.meshes.remove(collision_mesh)

    collision_mesh = bpy.data.meshes.new(collision_name)
    try:
        try:
            lod.build_lods(input_mapping)
            fail(label, "mesh-data-name collision did not raise ValueError")
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001 - record the exact wrong API outcome.
            fail(label, (
                f"mesh-data-name collision raised {type(exc).__name__}, "
                "expected ValueError"))

        unexpected_objects = sorted(
            name for name in all_variant_names
            if bpy.data.objects.get(name) is not None)
        if unexpected_objects:
            fail(label, (
                "mesh-data collision left partial LOD objects: "
                f"{unexpected_objects}"))
        unexpected_meshes = sorted(
            name for name in all_variant_names
            if name != collision_name and bpy.data.meshes.get(name) is not None)
        if unexpected_meshes:
            fail(label, (
                "mesh-data collision left partial LOD meshes: "
                f"{unexpected_meshes}"))
        assert_sources_unchanged(
            label, input_mapping, base_names, snapshots, "mesh-data collision")
    finally:
        if collision_mesh.users == 0:
            bpy.data.meshes.remove(collision_mesh)


def validate_result(
        label, result, base_names, base_triangles, snapshots,
        corrected_vertices):
    expected = expected_names(base_names, base_triangles)
    actual = set(result)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        fail(label, f"result names differ; missing={missing}, extra={extra}")

    for name in base_names:
        if name not in result:
            continue
        source = result[name]
        snapshot = snapshots[name]
        if source is not snapshot["object"]:
            fail(label, f"source object identity changed for {name}")
        if source.data is not snapshot["data"]:
            fail(label, f"source mesh identity changed for {name}")
        if semantic_hash(source) != snapshot["hash"]:
            fail(label, f"source semantic hash changed for {name}")
        if matrix_signature(source) != snapshot["matrix"]:
            fail(label, f"source transform changed for {name}")
        if material_signature(source) != snapshot["materials"]:
            fail(label, f"source materials changed for {name}")

        variant_names = [name + LOD1_SUFFIX]
        if base_triangles[name] >= LOD2_SOURCE_TRIANGLE_FLOOR:
            variant_names.append(name + LOD2_SUFFIX)
        elif name + LOD2_SUFFIX in result:
            fail(label, f"LOD2 was not dropped for sub-200-triangle source {name}")

        for variant_name in variant_names:
            variant = result.get(variant_name)
            if variant is None:
                fail(label, f"missing {variant_name}")
                continue
            if variant.name != variant_name:
                fail(label, f"object name {variant.name!r} != key {variant_name!r}")
            if variant.data.name != variant_name:
                fail(label, (
                    f"mesh data name {variant.data.name!r} != {variant_name!r}"))
            if variant.data is source.data:
                fail(label, f"{variant_name} shares mesh data with LOD0")
            if not variant.users_collection:
                fail(label, f"{variant_name} is not linked to a collection")
            if len(variant.modifiers) != 0:
                fail(label, f"{variant_name} retains unapplied modifiers")
            if matrix_signature(variant) != snapshot["matrix"]:
                fail(label, f"{variant_name} did not preserve object transform")
            if material_signature(variant) != snapshot["materials"]:
                fail(label, f"{variant_name} did not preserve materials")
            non_triangles = [
                polygon.index for polygon in variant.data.polygons
                if len(polygon.vertices) != 3
            ]
            if non_triangles:
                fail(label, (
                    f"{variant_name} contains {len(non_triangles)} non-triangle polygons"))
            loose_edges = [
                tuple(edge.vertices) for edge in variant.data.edges if edge.is_loose
            ]
            face_vertices = {
                vertex_index
                for polygon in variant.data.polygons
                for vertex_index in polygon.vertices
            }
            unreferenced_vertices = sorted(
                set(range(len(variant.data.vertices))) - face_vertices)
            if loose_edges or unreferenced_vertices:
                fail(label, (
                    f"{variant_name} contains non-face geometry: "
                    f"looseEdges={loose_edges} "
                    f"unreferencedVertices={unreferenced_vertices}"))
            corrected_vertex_count = corrected_vertices.get(variant_name)
            if (corrected_vertex_count is not None
                    and len(variant.data.vertices) != corrected_vertex_count):
                fail(label, (
                    f"{variant_name} vertices {len(variant.data.vertices)} != "
                    f"corrected contract {corrected_vertex_count}"))

            variant_uv, uv_error = uv_contract(variant)
            if uv_error:
                fail(label, uv_error)
                continue
            source_uv = snapshot["uv"]
            if variant_uv["name"] != source_uv["name"]:
                fail(label, (
                    f"{variant_name} UV layer {variant_uv['name']!r} != "
                    f"source {source_uv['name']!r}"))
            for lower in ("min_u", "min_v"):
                if variant_uv[lower] < source_uv[lower] - UV_SLACK:
                    fail(label, f"{variant_name} atlas lower bound escaped source")
            for upper in ("max_u", "max_v"):
                if variant_uv[upper] > source_uv[upper] + UV_SLACK:
                    fail(label, f"{variant_name} atlas upper bound escaped source")

        lod1_name = name + LOD1_SUFFIX
        if lod1_name in result:
            lod1_triangles = triangle_count(result[lod1_name])
            if not (0 < lod1_triangles < base_triangles[name]):
                fail(label, (
                    f"{lod1_name} triangles {lod1_triangles} not in "
                    f"(0, {base_triangles[name]})"))
        lod2_name = name + LOD2_SUFFIX
        if lod2_name in result and lod1_name in result:
            lod2_triangles = triangle_count(result[lod2_name])
            lod1_triangles = triangle_count(result[lod1_name])
            if not (0 < lod2_triangles < lod1_triangles):
                fail(label, (
                    f"{lod2_name} triangles {lod2_triangles} not in "
                    f"(0, {lod1_triangles})"))


def run_case(case):
    label = case["label"]
    scene.reset()
    recipe_obj = recipe_module.load(case["path"])
    assembly = assemble.build_vehicle(recipe_obj)
    uvmap.unwrap_and_pack(assembly.objects)

    # Exercise material preservation without running Task 13's expensive bake.
    probe_material = bpy.data.materials.new(f"TGEL.LODTest.{label}")
    for obj in assembly.objects.values():
        obj.data.materials.clear()
        obj.data.materials.append(probe_material)

    # Identity-only snapshots can miss a clone that constructs a fresh object;
    # give one representative source a deliberate non-identity transform.
    transform_probe = next(iter(assembly.objects.values()))
    transform_probe.location = (0.125, -0.25, 0.5)
    transform_probe.rotation_mode = 'XYZ'
    transform_probe.rotation_euler = (0.1, -0.2, 0.3)
    transform_probe.scale = (1.1, 0.9, 1.05)
    # Matrix properties are dependency-graph evaluated.  Freeze the authored
    # transform into matrix_local before taking the non-mutation snapshot.
    bpy.context.view_layer.update()

    input_mapping = assembly.objects
    base_names = tuple(input_mapping.keys())
    base_triangles = {
        name: triangle_count(obj) for name, obj in input_mapping.items()
    }
    snapshots = {}
    for name, obj in input_mapping.items():
        uv_state, uv_error = uv_contract(obj)
        if uv_error:
            fail(label, uv_error)
            continue
        snapshots[name] = {
            "object": obj,
            "data": obj.data,
            "hash": semantic_hash(obj),
            "matrix": matrix_signature(obj),
            "materials": material_signature(obj),
            "uv": uv_state,
        }

    if len(base_names) != case["base_objects"]:
        fail(label, (
            f"base object count {len(base_names)} != {case['base_objects']}"))
    lod2_count = sum(
        count >= LOD2_SOURCE_TRIANGLE_FLOOR for count in base_triangles.values())
    if lod2_count != case["lod2_objects"]:
        fail(label, f"LOD2-eligible object count {lod2_count} != {case['lod2_objects']}")

    lod0_total = sum(base_triangles.values())
    if lod0_total != case["lod0_triangles"]:
        fail(label, f"LOD0 triangles {lod0_total} != {case['lod0_triangles']}")
    if lod0_total > case["lod0_budget"]:
        fail(label, f"LOD0 triangles {lod0_total} exceed {case['lod0_budget']}")
    if len(snapshots) != len(base_names):
        return

    validate_collision_preflight(
        label, input_mapping, base_names, base_triangles, snapshots)

    first, first_cleanup = build_lods_with_cleanup_census(input_mapping)
    validate_cleanup_census(
        label, first_cleanup, case["cleanup_census"], "first same-input pass")
    if first is input_mapping:
        fail(label, "build_lods returned the input mapping instead of a new mapping")
    if tuple(input_mapping.keys()) != base_names:
        fail(label, "build_lods mutated the input mapping")
    validate_result(
        label, first, base_names, base_triangles, snapshots,
        case["corrected_vertices"])

    lod1_total = sum(
        triangle_count(first[name + LOD1_SUFFIX]) for name in base_names)
    lod2_total = sum(
        triangle_count(first[name + LOD2_SUFFIX])
        for name in base_names
        if base_triangles[name] >= LOD2_SOURCE_TRIANGLE_FLOOR)
    if lod1_total != case["lod1_triangles"]:
        fail(label, f"LOD1 triangles {lod1_total} != {case['lod1_triangles']}")
    if lod2_total != case["lod2_triangles"]:
        fail(label, f"LOD2 triangles {lod2_total} != {case['lod2_triangles']}")
    if lod1_total > 0.5 * lod0_total:
        fail(label, f"LOD1 ratio {lod1_total / lod0_total:.6f} exceeds 0.5")
    if lod2_total > 0.2 * lod0_total:
        fail(label, f"LOD2 ratio {lod2_total / lod0_total:.6f} exceeds 0.2")

    variant_names = sorted(set(first) - set(base_names))
    first_hashes = {name: semantic_hash(first[name]) for name in variant_names}
    remove_variants(first, base_names)

    # Same packed LOD0, second independent DECIMATE pass.  This isolates Task
    # 14 from known fresh-pack/fresh-build nondeterminism owned by Task 15.
    second, second_cleanup = build_lods_with_cleanup_census(input_mapping)
    validate_cleanup_census(
        label, second_cleanup, case["cleanup_census"],
        "second same-input pass")
    validate_result(
        label, second, base_names, base_triangles, snapshots,
        case["corrected_vertices"])
    second_names = sorted(set(second) - set(base_names))
    second_hashes = {name: semantic_hash(second[name]) for name in second_names}
    if second_names != variant_names:
        fail(label, "same-input rerun produced a different variant name set")
    for name in variant_names:
        if first_hashes[name] != second_hashes.get(name):
            fail(label, f"same-input semantic hash changed for {name}")

    digest_payload = "\n".join(
        f"{name}|{second_hashes[name]}" for name in second_names).encode("utf-8")
    digest = hashlib.sha256(digest_payload).hexdigest()
    print(
        f"[lod_test] {label}: base={len(base_names)} lod2Eligible={lod2_count} "
        f"returned={len(second)} tris={lod0_total}/{lod1_total}/{lod2_total} "
        f"ratios={lod1_total / lod0_total:.6f}/{lod2_total / lod0_total:.6f} "
        f"sameInputVariantDigest={digest}",
        flush=True)


def fresh_semantic_snapshot(case):
    """Builds one production-shaped mesh graph from a clean Blender scene."""
    scene.reset()
    recipe_obj = recipe_module.load(case["path"])
    assembly = assemble.build_vehicle(recipe_obj)
    uvmap.unwrap_and_pack(assembly.objects)
    result, cleanup_census = build_lods_with_cleanup_census(assembly.objects)
    hashes = {name: semantic_hash(obj) for name, obj in sorted(result.items())}
    digest_payload = "\n".join(
        f"{name}|{hashes[name]}" for name in sorted(hashes)).encode("utf-8")
    return hashes, hashlib.sha256(digest_payload).hexdigest(), cleanup_census


def validate_fresh_build_determinism(case):
    label = case["label"]
    first, first_digest, first_cleanup = fresh_semantic_snapshot(case)
    second, second_digest, second_cleanup = fresh_semantic_snapshot(case)
    validate_cleanup_census(
        label, first_cleanup, case["cleanup_census"], "first fresh build")
    validate_cleanup_census(
        label, second_cleanup, case["cleanup_census"], "second fresh build")
    expected_count = case["base_objects"] * 2 + case["lod2_objects"]
    if len(first) != expected_count or len(second) != expected_count:
        fail(label, (
            f"fresh result counts {len(first)}/{len(second)} != "
            f"{expected_count}"))
    if set(first) != set(second):
        fail(label, "fresh builds produced different mesh-name sets")
    differing = sorted(
        name for name in set(first) & set(second)
        if first[name] != second[name])
    if differing:
        fail(label, (
            f"fresh semantic hashes differ for {len(differing)}/{expected_count} "
            f"meshes; first={differing[:5]}"))
    print(
        f"[lod_test] {label}: freshDigests={first_digest}/{second_digest} "
        f"differing={len(differing)}/{expected_count}",
        flush=True)


validate_non_face_cleanup_fixture()

for case in CASES:
    validate_fresh_build_determinism(case)

for case in CASES:
    run_case(case)

if failures:
    print("BPY TESTS FAIL:", "; ".join(failures))
    sys.exit(1)
print("BPY TESTS OK")
sys.exit(0)
