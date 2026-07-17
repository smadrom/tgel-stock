import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from tgel_stock import assemble  # noqa: E402
from tgel_stock import materials  # noqa: E402
from tgel_stock import recipe as recipe_module  # noqa: E402
from tgel_stock import scene  # noqa: E402
from tgel_stock import uvmap  # noqa: E402

UV_SLACK = 1e-4

WAGON_RECIPE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "recipes", "basic-box-wagon.rollingstock.json")
LOCOMOTIVE_RECIPE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "recipes", "basic-diesel-locomotive.rollingstock.json")

# Floors are the controller-adjudicated production contract at atlas_px=4096.
# Coverage is a regression guard; density and overlap are the primary gates.
COVERAGE_FLOOR = 0.05
MAX_OVERLAP_CEILING = 0.01
DENSITY_FLOOR_PX_PER_M = 45.0
STENCIL_TEXEL_FLOOR = 20
OCCUPIED_COVERAGE_CEILING = 0.40
# Each accepted corner U-frame has ten hard-separated faces instead of the
# detached box's six, adding four islands at each of four wagon corners.
EXPECTED_ISLAND_COUNTS = {"wagon": 619, "locomotive": 763}
ISLAND_RATIO_RTOL = 0.02
ISLAND_AREA_ATOL = 1e-10
SHELF_TOLERANCE = 2e-6

failures = []


class _InjectedCanonicalRemoveFailure(RuntimeError):
    pass


class _MeshesProxy:
    def __init__(self, collection, rejected_mesh):
        self._collection = collection
        self._rejected_mesh = rejected_mesh

    def __getattr__(self, name):
        return getattr(self._collection, name)

    def remove(self, mesh, *args, **kwargs):
        if mesh is self._rejected_mesh:
            raise _InjectedCanonicalRemoveFailure(
                "injected while removing canonical source mesh")
        return self._collection.remove(mesh, *args, **kwargs)


class _DataProxy:
    def __init__(self, data, rejected_mesh):
        self._data = data
        self.meshes = _MeshesProxy(data.meshes, rejected_mesh)

    def __getattr__(self, name):
        return getattr(self._data, name)


class _BpyProxy:
    def __init__(self, module, rejected_mesh):
        self._module = module
        self.data = _DataProxy(module.data, rejected_mesh)

    def __getattr__(self, name):
        return getattr(self._module, name)


def _id_snapshot(collection):
    return tuple(sorted((item.as_pointer(), item.name) for item in collection))


def _validate_canonical_rollback(label, obj, geometry_before):
    original_mesh = obj.data
    original_name = original_mesh.name
    meshes_before = _id_snapshot(bpy.data.meshes)
    objects_before = _id_snapshot(bpy.data.objects)
    original_scene_bpy = scene.bpy
    scene.bpy = _BpyProxy(bpy, original_mesh)
    try:
        try:
            scene.canonicalize_mesh(obj)
        except _InjectedCanonicalRemoveFailure:
            pass
        except Exception as exc:  # noqa: BLE001 - exact wrong rollback outcome.
            failures.append(
                f"[{label}] canonical rollback raised {type(exc).__name__}: {exc}")
        else:
            failures.append(f"[{label}] canonical rollback injection did not raise")
    finally:
        scene.bpy = original_scene_bpy

    if obj.data is not original_mesh or original_mesh.name != original_name:
        failures.append(f"[{label}] canonical rollback did not restore source mesh")
    if _id_snapshot(bpy.data.meshes) != meshes_before:
        failures.append(f"[{label}] canonical rollback changed mesh datablock set")
    if _id_snapshot(bpy.data.objects) != objects_before:
        failures.append(f"[{label}] canonical rollback changed object datablock set")
    _validate_preserved_geometry(label, obj.name, obj, geometry_before)


def _validate_actual_island_adjacency():
    mesh = bpy.data.meshes.new("TGEL.UVActualIslandProbe")
    obj = bpy.data.objects.new("TGEL.UVActualIslandProbe", mesh)
    bpy.context.scene.collection.objects.link(obj)
    try:
        mesh.from_pydata(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
             (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            (),
            ((0, 1, 2), (0, 2, 3)),
        )
        mesh.update(calc_edges=True)
        uv_layer = mesh.uv_layers.new(name="UVMap")
        source_uv = ((0.0, 0.0), (1.0, 0.0),
                     (1.0, 1.0), (0.0, 1.0))
        for polygon in mesh.polygons:
            for loop_index in polygon.loop_indices:
                vertex_index = mesh.loops[loop_index].vertex_index
                uv_layer.data[loop_index].uv = source_uv[vertex_index]
        continuous = uvmap._actual_uv_island_faces(obj)
        if continuous != ((0, 1),):
            failures.append(
                f"[island-probe] continuous shared edge produced {continuous}")

        for loop_index in mesh.polygons[1].loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            if vertex_index in (0, 2):
                u, v = uv_layer.data[loop_index].uv
                uv_layer.data[loop_index].uv = (u + 2.0, v)
        discontinuous = uvmap._actual_uv_island_faces(obj)
        if discontinuous != ((0,), (1,)):
            failures.append(
                f"[island-probe] UV-discontinuous shared edge produced "
                f"{discontinuous}")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        if bpy.data.meshes.get(mesh.name) is mesh:
            bpy.data.meshes.remove(mesh)


def _validate_stencil_footprints(label, objects, atlas_px=1024):
    text = "TGEL 40-102\nCAP 50T\nBLT 7-26"
    glyph_triangles = materials._bfont_triangles(text)
    for name in ("Visuals.Body.DoorLeft", "Visuals.Body.DoorRight"):
        obj = objects[name]
        projections = materials._exterior_side_projections(obj)
        if len(projections) != 1:
            failures.append(
                f"[{label}] {name} has {len(projections)} exterior projections")
            continue
        for projection in projections:
            corners = materials._projection_uv(
                projection,
                np.asarray(((0.0, 0.0), (1.0, 0.0),
                            (1.0, 1.0), (0.0, 1.0))))
            projection_min = np.min(corners, axis=0)
            projection_max = np.max(corners, axis=0)
            projection_px = (projection_max - projection_min) * atlas_px
            rasterized = set()
            for triangle in glyph_triangles:
                raster = materials._raster_triangle(
                    materials._projection_uv(projection, triangle),
                    atlas_px, atlas_px)
                if raster is not None:
                    rasterized.update(zip(raster[0].tolist(), raster[1].tolist()))
            print(
                f"[stencil] {name} sign={projection[0]:+d} "
                f"projection_px=({projection_px[0]:.6f},"
                f"{projection_px[1]:.6f}) raster_texels={len(rasterized)}",
                flush=True)
            if len(rasterized) < STENCIL_TEXEL_FLOOR:
                failures.append(
                    f"[{label}] {name} stencil raster has {len(rasterized)} "
                    f"texels, expected >= {STENCIL_TEXEL_FLOOR}")


def _validate_occupied_coverage(label, objects, atlas_px=1024):
    occupied = np.zeros((atlas_px, atlas_px), dtype=np.bool_)
    for name in sorted(objects):
        for x, y, *_values in materials._iter_object_texels(
                objects[name], atlas_px, atlas_px):
            occupied[y, x] = True
    coverage = float(np.count_nonzero(occupied)) / float(occupied.size)
    print(f"[occupied] {label} coverage={coverage:.4f}", flush=True)
    if coverage >= OCCUPIED_COVERAGE_CEILING:
        failures.append(
            f"[{label}] occupied coverage {coverage} is not strictly below "
            f"{OCCUPIED_COVERAGE_CEILING}")


def _canonical_cycle(values):
    values = tuple(values)
    return min(values[index:] + values[:index] for index in range(len(values)))


def _rounded_coordinate(coordinate):
    return tuple(round(component, 8) for component in coordinate)


def _canonical_index_cycle(indices, coordinates):
    candidates = []
    indices = tuple(indices)
    for start in range(len(indices)):
        cycle = indices[start:] + indices[:start]
        rounded = tuple(_rounded_coordinate(coordinates[index]) for index in cycle)
        exact = tuple(coordinates[index] for index in cycle)
        candidates.append((rounded, exact, cycle))
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]


def _matrix_signature(obj):
    return tuple(float(value) for row in obj.matrix_local for value in row)


def _geometry_signature(obj):
    """Index-independent exact geometry signature that preserves winding."""
    coordinates = [
        tuple(float(component) for component in vertex.co)
        for vertex in obj.data.vertices
    ]
    faces = []
    for polygon in obj.data.polygons:
        coordinate_cycle = tuple(coordinates[index] for index in polygon.vertices)
        faces.append((
            _canonical_cycle(coordinate_cycle),
            int(polygon.material_index),
            bool(polygon.use_smooth),
        ))
    edges = [
        tuple(sorted((coordinates[edge.vertices[0]], coordinates[edge.vertices[1]])))
        for edge in obj.data.edges
    ]
    xs = [coordinate[0] for coordinate in coordinates]
    ys = [coordinate[1] for coordinate in coordinates]
    zs = [coordinate[2] for coordinate in coordinates]
    return {
        "object": obj,
        "mesh_name": obj.data.name,
        "coordinates": tuple(sorted(coordinates)),
        "faces": tuple(sorted(faces)),
        "edges": tuple(sorted(edges)),
        "bounds": (
            (min(xs), min(ys), min(zs)),
            (max(xs), max(ys), max(zs)),
        ),
        "matrix": _matrix_signature(obj),
        "materials": tuple(obj.data.materials),
    }


def _validate_preserved_geometry(label, name, obj, before, require_canonical=False):
    after = _geometry_signature(obj)
    for key in (
            "object", "mesh_name", "coordinates", "faces", "edges",
            "bounds", "matrix", "materials"):
        if after[key] != before[key]:
            failures.append(f"[{label}] {name} changed preserved {key}")

    if require_canonical:
        coordinates = [
            tuple(float(component) for component in vertex.co)
            for vertex in obj.data.vertices
        ]
        expected_coordinates = sorted(
            coordinates, key=lambda coordinate: (
                _rounded_coordinate(coordinate), coordinate))
        if coordinates != expected_coordinates:
            failures.append(f"[{label}] {name} vertex order is not canonical")

        face_records = [
            (
                tuple(polygon.vertices),
                int(polygon.material_index),
                bool(polygon.use_smooth),
            )
            for polygon in obj.data.polygons
        ]
        for cycle, _material_index, _use_smooth in face_records:
            if cycle != _canonical_index_cycle(cycle, coordinates):
                failures.append(
                    f"[{label}] {name} face cycle does not preserve canonical start")
                break

        def face_sort_key(record):
            cycle, material_index, use_smooth = record
            rounded = tuple(_rounded_coordinate(coordinates[index]) for index in cycle)
            exact = tuple(coordinates[index] for index in cycle)
            return (len(cycle), rounded, exact, material_index, use_smooth)

        if face_records != sorted(face_records, key=face_sort_key):
            failures.append(f"[{label}] {name} face order is not canonical")


def _packed_island_records(objects):
    records = []
    for name in sorted(objects):
        obj = objects[name]
        mesh = obj.data
        uv_layer = mesh.uv_layers.active
        mesh.calc_loop_triangles()
        triangles_by_polygon = [[] for _polygon in mesh.polygons]
        for triangle in mesh.loop_triangles:
            triangles_by_polygon[triangle.polygon_index].append(triangle)
        for face_indices in uvmap._actual_uv_island_faces(obj):
            loop_indices = tuple(
                loop_index
                for polygon_index in face_indices
                for loop_index in mesh.polygons[polygon_index].loop_indices)
            coordinates = np.asarray(
                [uv_layer.data[index].uv[:] for index in loop_indices],
                dtype=np.float64)
            area_3d = 0.0
            area_uv = 0.0
            for polygon_index in face_indices:
                for triangle in triangles_by_polygon[polygon_index]:
                    area_3d += float(triangle.area)
                    uv0, uv1, uv2 = (
                        uv_layer.data[index].uv for index in triangle.loops)
                    area_uv += uvmap._uv_triangle_area(uv0, uv1, uv2)
            records.append({
                "key": (name, face_indices),
                "minimum": np.min(coordinates, axis=0),
                "maximum": np.max(coordinates, axis=0),
                "area_3d": area_3d,
                "area_uv": area_uv,
            })
    return records


def _validate_allocator_contract(label, objects, allocation):
    records = _packed_island_records(objects)
    expected_count = EXPECTED_ISLAND_COUNTS[label]
    if len(records) != expected_count:
        failures.append(
            f"[{label}] packed island count {len(records)} != {expected_count}")
    if allocation["island_count"] != len(records):
        failures.append(
            f"[{label}] allocation island count {allocation['island_count']} "
            f"!= measured {len(records)}")

    expected_ratio = allocation["multiplier"] ** 2
    worst_ratio_error = (0.0, None, 0.0)
    ratio_failure = None
    for record in records:
        if record["area_3d"] <= 0.0:
            failures.append(f"[{label}] {record['key']} has non-positive 3D area")
            continue
        expected_uv_area = record["area_3d"] * expected_ratio
        ratio = record["area_uv"] / record["area_3d"]
        if expected_uv_area > ISLAND_AREA_ATOL:
            relative_error = abs(ratio - expected_ratio) / expected_ratio
            if relative_error > worst_ratio_error[0]:
                worst_ratio_error = (relative_error, record["key"], ratio)
        if (ratio_failure is None
                and not np.isclose(
                    record["area_uv"], expected_uv_area,
                    rtol=ISLAND_RATIO_RTOL, atol=ISLAND_AREA_ATOL)):
            ratio_failure = (record["key"], ratio, expected_uv_area)
    print(
        f"[island-ratio] {label} worstRelative={worst_ratio_error[0]:.6f} "
        f"key={worst_ratio_error[1]}",
        flush=True)
    if ratio_failure is not None:
        failures.append(
            f"[{label}] {ratio_failure[0]} UV/3D ratio {ratio_failure[1]} "
            f"does not match multiplier squared {expected_ratio}; expected "
            f"UV area {ratio_failure[2]}")

    padding = allocation["padding"]
    for record in records:
        minimum = record["minimum"]
        maximum = record["maximum"]
        if np.any(minimum < padding - SHELF_TOLERANCE):
            failures.append(
                f"[{label}] {record['key']} violates lower atlas padding")
            break
        if np.any(maximum > 1.0 - padding + SHELF_TOLERANCE):
            failures.append(
                f"[{label}] {record['key']} violates upper atlas padding")
            break

    for left_index, left in enumerate(records):
        left_min, left_max = left["minimum"], left["maximum"]
        for right in records[left_index + 1:]:
            right_min, right_max = right["minimum"], right["maximum"]
            separated = (
                right_min[0] - left_max[0] >= padding - SHELF_TOLERANCE
                or left_min[0] - right_max[0] >= padding - SHELF_TOLERANCE
                or right_min[1] - left_max[1] >= padding - SHELF_TOLERANCE
                or left_min[1] - right_max[1] >= padding - SHELF_TOLERANCE
            )
            if not separated:
                failures.append(
                    f"[{label}] shelf rectangles {left['key']} and "
                    f"{right['key']} lack {padding} padding")
                return


def run_case(label, recipe_path):
    scene.reset()
    recipe_obj = recipe_module.load(recipe_path)
    assembly = assemble.build_vehicle(recipe_obj)

    for name, obj in assembly.objects.items():
        coordinates = [tuple(float(c) for c in vertex.co) for vertex in obj.data.vertices]
        if len(coordinates) != len(set(coordinates)):
            failures.append(f"[{label}] {name} has duplicate vertex coordinates")

    transform_probe = next(iter(assembly.objects.values()))
    material_a = bpy.data.materials.new(f"TGEL.UVCanonicalizationTest.{label}.A")
    material_b = bpy.data.materials.new(f"TGEL.UVCanonicalizationTest.{label}.B")
    transform_probe.data.materials.append(material_a)
    transform_probe.data.materials.append(material_b)
    for index, polygon in enumerate(transform_probe.data.polygons):
        polygon.material_index = index % 2
        polygon.use_smooth = index % 3 == 0
    transform_probe.location = (0.125, -0.25, 0.5)
    transform_probe.rotation_mode = 'XYZ'
    transform_probe.rotation_euler = (0.1, -0.2, 0.3)
    transform_probe.scale = (1.1, 1.1, 1.1)
    bpy.context.view_layer.update()

    geometry_original = {
        name: _geometry_signature(obj) for name, obj in assembly.objects.items()
    }

    _validate_canonical_rollback(
        label,
        transform_probe,
        geometry_original[transform_probe.name])

    allocation = uvmap.unwrap_and_pack(assembly.objects)
    _validate_allocator_contract(label, assembly.objects, allocation)

    if label == "wagon":
        _validate_stencil_footprints(label, assembly.objects)
        _validate_occupied_coverage(label, assembly.objects)

    for name, obj in assembly.objects.items():
        _validate_preserved_geometry(
            label, name, obj, geometry_original[name], require_canonical=True)
        uv_layer = obj.data.uv_layers.active
        if uv_layer is None:
            failures.append(f"[{label}] {name} has no active UV layer")
            continue
        for loop_uv in uv_layer.data:
            u, v = loop_uv.uv[0], loop_uv.uv[1]
            if not (-UV_SLACK <= u <= 1.0 + UV_SLACK):
                failures.append(f"[{label}] {name} UV.u {u} outside [0,1] (+/- slack)")
            if not (-UV_SLACK <= v <= 1.0 + UV_SLACK):
                failures.append(f"[{label}] {name} UV.v {v} outside [0,1] (+/- slack)")

    if any(f.startswith(f"[{label}]") for f in failures):
        return

    report = uvmap.report(assembly.objects)
    coverage = report["coverage"]
    max_overlap = report["max_overlap"]
    density = report["density_px_per_m"]

    if coverage < COVERAGE_FLOOR:
        failures.append(f"[{label}] coverage {coverage} < floor {COVERAGE_FLOOR}")
    if max_overlap > MAX_OVERLAP_CEILING:
        failures.append(f"[{label}] max_overlap {max_overlap} > ceiling {MAX_OVERLAP_CEILING}")
    if density < DENSITY_FLOOR_PX_PER_M:
        failures.append(
            f"[{label}] density_px_per_m {density} < floor {DENSITY_FLOOR_PX_PER_M}")

    print(f"[{label}] islands={allocation['island_count']} "
          f"padding={allocation['padding']:.4f} "
          f"fit={allocation['fit_multiplier']:.12f} "
          f"cap={allocation['cap_multiplier']:.12f} "
          f"used={allocation['multiplier']:.12f} "
          f"coverage={coverage:.4f} max_overlap={max_overlap:.4f} "
          f"density_px_per_m={density:.1f}")


_validate_actual_island_adjacency()
run_case("wagon", WAGON_RECIPE_PATH)
run_case("locomotive", LOCOMOTIVE_RECIPE_PATH)

if failures:
    print("BPY TESTS FAIL:", "; ".join(failures))
    sys.exit(1)
print("BPY TESTS OK")
sys.exit(0)
