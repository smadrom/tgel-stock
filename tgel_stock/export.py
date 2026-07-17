from contextlib import contextmanager

import bpy

from . import assert_clean_path
from . import canonical
from . import scene

FBX_KWARGS = dict(
    use_selection=True,
    apply_unit_scale=True,
    apply_scale_options="FBX_SCALE_ALL",
    axis_forward="-Z",
    axis_up="Y",
    use_space_transform=True,
    bake_space_transform=True,
    object_types={"MESH"},
    use_mesh_modifiers=True,
    # TGEL supplies explicit exact-name triangle-only mesh copies below.
    # Blender's flag creates its own ``.001`` datablocks and exports those
    # temporary names, breaking Unity's manifest-name lookup contract.
    use_triangles=False,
    add_leaf_bones=False,
    bake_anim=False,
    use_custom_props=False,
)

# Blender re-normalizes custom corner normals on the temporary triangle mesh.
# Keep the representation-only drift below roughly 0.057 degrees while the
# source mesh and its manifest semantic hash remain byte-for-byte untouched.
MAX_CORNER_NORMAL_DELTA = 1e-3


def _mesh_snapshot(obj):
    positions, normals, uvs, triangles, bounds_min, bounds_max = (
        scene.collect_mesh_data(obj))
    return {
        "semanticHash": canonical.geometry_hash(
            positions, normals, uvs, triangles),
        "boundsMin": tuple(bounds_min),
        "boundsMax": tuple(bounds_max),
        "materials": tuple(
            material.as_pointer() if material is not None else None
            for material in obj.data.materials),
        "activeUv": (
            obj.data.uv_layers.active.name
            if obj.data.uv_layers.active is not None else None),
    }


def _mesh_id_state():
    return {
        (mesh.as_pointer(), mesh.name)
        for mesh in bpy.data.meshes
    }


def _preflight_triangle_export(objects):
    objects = tuple(objects)
    if not objects:
        raise ValueError("FBX export requires at least one mesh object")

    object_pointers = [obj.as_pointer() for obj in objects]
    if len(set(object_pointers)) != len(object_pointers):
        raise ValueError("FBX export objects must be unique")

    data_pointers = []
    exact_names = []
    records = []
    for index, obj in enumerate(objects):
        if obj.type != "MESH" or obj.data is None:
            raise ValueError(
                f"FBX export object is not a mesh: {getattr(obj, 'name', obj)!r}")
        mesh = obj.data
        if mesh.users != 1:
            raise ValueError(
                f"FBX export mesh must be single-user: {mesh.name} users={mesh.users}")
        if mesh.name != obj.name:
            raise ValueError(
                f"FBX export object/data name mismatch: {obj.name!r}/{mesh.name!r}")
        if mesh.shape_keys is not None:
            raise ValueError(f"FBX export shape keys are unsupported: {obj.name}")
        if len(obj.modifiers) != 0:
            raise ValueError(f"FBX export modifiers are unsupported: {obj.name}")
        if len(obj.vertex_groups) != 0:
            raise ValueError(f"FBX export vertex groups are unsupported: {obj.name}")
        if len(mesh.color_attributes) != 0:
            raise ValueError(
                f"FBX export color attributes are unsupported: {obj.name}")
        if any(slot.link == "OBJECT" for slot in obj.material_slots):
            raise ValueError(
                f"FBX export object-linked material slot is unsupported: {obj.name}")
        if any(material is None for material in mesh.materials):
            raise ValueError(
                f"FBX export empty material slot is unsupported: {obj.name}")

        data_pointers.append(mesh.as_pointer())
        exact_names.append(mesh.name)
        records.append({
            "obj": obj,
            "original": mesh,
            "originalPointer": mesh.as_pointer(),
            "exactName": mesh.name,
            "sourceName": f"TGEL.ExportSource.{index:04d}",
            "cleanupName": f"TGEL.ExportCleanup.{index:04d}",
            "snapshot": _mesh_snapshot(obj),
            "temporary": None,
        })

    if len(set(data_pointers)) != len(data_pointers):
        raise ValueError("FBX export mesh datablocks must be unique")
    if len(set(exact_names)) != len(exact_names):
        raise ValueError("FBX export mesh datablock names must be unique")

    reserved_names = {
        name
        for record in records
        for name in (record["sourceName"], record["cleanupName"])
    }
    if reserved_names.intersection(exact_names):
        raise ValueError("FBX export exact names collide with reserved names")
    collisions = sorted(
        name for name in reserved_names
        if bpy.data.meshes.get(name) is not None)
    if collisions:
        raise ValueError(
            f"FBX export reserved mesh-name collision: {collisions}")
    return objects, records


def _triangulate_copy(record):
    original = record["original"]
    original.calc_loop_triangles()
    source_triangles = tuple(original.loop_triangles)
    source_positions = tuple(
        tuple(float(component) for component in vertex.co)
        for vertex in original.vertices)
    source_loose_edges = tuple(
        tuple(edge.vertices) for edge in original.edges if edge.is_loose)
    source_loose_edge_keys = tuple(sorted(
        tuple(sorted(edge_vertices)) for edge_vertices in source_loose_edges))
    source_seam_edge_keys = frozenset(
        tuple(sorted(edge.vertices))
        for edge in original.edges if edge.use_seam)
    source_sharp_edge_keys = frozenset(
        tuple(sorted(edge.vertices))
        for edge in original.edges if edge.use_edge_sharp)
    source_triangle_vertices = tuple(
        tuple(triangle.vertices) for triangle in source_triangles)
    source_triangle_loops = tuple(
        tuple(triangle.loops) for triangle in source_triangles)
    source_material_indices = tuple(
        original.polygons[triangle.polygon_index].material_index
        for triangle in source_triangles)
    source_smooth_flags = tuple(
        original.polygons[triangle.polygon_index].use_smooth
        for triangle in source_triangles)
    source_corner_normals = tuple(
        tuple(
            float(component)
            for component in original.corner_normals[source_loop].vector)
        for triangle_loops in source_triangle_loops
        for source_loop in triangle_loops)
    source_uv_layers = tuple(
        (
            layer.name,
            tuple(
                tuple(float(component) for component in layer.data[source_loop].uv)
                for triangle_loops in source_triangle_loops
                for source_loop in triangle_loops),
        )
        for layer in original.uv_layers)
    source_active_uv_index = (
        original.uv_layers.active_index if original.uv_layers else -1)

    temporary = bpy.data.meshes.new(record["exactName"])
    record["temporary"] = temporary
    temporary.from_pydata(
        source_positions, source_loose_edges, source_triangle_vertices)
    for edge in temporary.edges:
        edge_key = tuple(sorted(edge.vertices))
        edge.use_seam = edge_key in source_seam_edge_keys
        edge.use_edge_sharp = edge_key in source_sharp_edge_keys
    for material in original.materials:
        temporary.materials.append(material)
    for polygon, material_index, use_smooth in zip(
            temporary.polygons,
            source_material_indices,
            source_smooth_flags):
        polygon.material_index = material_index
        polygon.use_smooth = use_smooth
    for layer_name, source_uvs in source_uv_layers:
        temporary_layer = temporary.uv_layers.new(name=layer_name)
        if temporary_layer.name != layer_name:
            raise RuntimeError(
                f"Temporary export UV layer lost exact name: "
                f"{temporary_layer.name!r} != {layer_name!r}")
        if len(temporary_layer.data) != len(source_uvs):
            raise RuntimeError(
                f"Temporary export UV loop count {len(temporary_layer.data)} != "
                f"source triangle UV count {len(source_uvs)} for "
                f"{record['exactName']}")
        for uv_loop, source_uv in zip(temporary_layer.data, source_uvs):
            uv_loop.uv = source_uv
    if source_active_uv_index >= 0:
        temporary.uv_layers.active_index = source_active_uv_index
    temporary.update()
    temporary.normals_split_custom_set(source_corner_normals)
    temporary.update()

    if temporary.name != record["exactName"]:
        raise RuntimeError(
            f"Temporary export mesh lost exact name: "
            f"{temporary.name!r} != {record['exactName']!r}")
    record["obj"].data = temporary

    if temporary.as_pointer() == record["originalPointer"]:
        raise RuntimeError("Temporary export mesh reused the source pointer")
    if len(temporary.vertices) != len(original.vertices):
        raise RuntimeError(
            f"Triangulation changed vertex count for {record['exactName']}")
    if tuple(
            tuple(float(component) for component in vertex.co)
            for vertex in temporary.vertices) != source_positions:
        raise RuntimeError(
            f"Triangulation changed vertex coordinates for {record['exactName']}")
    temporary_loose_edge_keys = tuple(sorted(
        tuple(sorted(edge.vertices))
        for edge in temporary.edges if edge.is_loose))
    if temporary_loose_edge_keys != source_loose_edge_keys:
        raise RuntimeError(
            f"Triangulation changed loose edges for {record['exactName']}")
    if frozenset(
            tuple(sorted(edge.vertices))
            for edge in temporary.edges if edge.use_seam
            ) != source_seam_edge_keys:
        raise RuntimeError(
            f"Triangulation changed seam edges for {record['exactName']}")
    if frozenset(
            tuple(sorted(edge.vertices))
            for edge in temporary.edges if edge.use_edge_sharp
            ) != source_sharp_edge_keys:
        raise RuntimeError(
            f"Triangulation changed sharp edges for {record['exactName']}")
    if any(len(polygon.vertices) != 3 for polygon in temporary.polygons):
        raise RuntimeError(
            f"Triangulation left a non-triangle polygon for {record['exactName']}")
    temporary_triangle_vertices = tuple(
        tuple(polygon.vertices) for polygon in temporary.polygons)
    if temporary_triangle_vertices != source_triangle_vertices:
        raise RuntimeError(
            f"Triangulated ordered vertex tuples differ from source loop "
            f"triangles for {record['exactName']}")
    if tuple(
            material.as_pointer() if material is not None else None
            for material in temporary.materials) != record["snapshot"]["materials"]:
        raise RuntimeError(
            f"Triangulation changed material slots for {record['exactName']}")
    if bool(temporary.uv_layers.active) != bool(record["snapshot"]["activeUv"]):
        raise RuntimeError(
            f"Triangulation changed UV presence for {record['exactName']}")
    if tuple(layer.name for layer in temporary.uv_layers) != tuple(
            layer_name for layer_name, _source_uvs in source_uv_layers):
        raise RuntimeError(
            f"Triangulation changed UV layers for {record['exactName']}")
    for temporary_layer, (layer_name, source_uvs) in zip(
            temporary.uv_layers, source_uv_layers):
        temporary_uvs = tuple(
            tuple(float(component) for component in uv_loop.uv)
            for uv_loop in temporary_layer.data)
        if temporary_uvs != source_uvs:
            raise RuntimeError(
                f"Triangulation changed ordered UV values in {layer_name!r} "
                f"for {record['exactName']}")
    if tuple(
            polygon.material_index for polygon in temporary.polygons
            ) != source_material_indices:
        raise RuntimeError(
            f"Triangulation changed face materials for {record['exactName']}")
    if tuple(
            polygon.use_smooth for polygon in temporary.polygons
            ) != source_smooth_flags:
        raise RuntimeError(
            f"Triangulation changed smooth flags for {record['exactName']}")
    temporary_corner_normals = tuple(
        tuple(float(component) for component in corner.vector)
        for corner in temporary.corner_normals)
    maximum_normal_delta = max(
        (abs(actual - expected)
         for actual_row, expected_row in zip(
             temporary_corner_normals, source_corner_normals)
         for actual, expected in zip(actual_row, expected_row)),
        default=0.0)
    if (len(temporary_corner_normals) != len(source_corner_normals) or
            maximum_normal_delta > MAX_CORNER_NORMAL_DELTA):
        raise RuntimeError(
            f"Triangulation changed corner normals for {record['exactName']} "
            f"(max delta {maximum_normal_delta})")

    _positions, _normals, _uvs, _triangles, bounds_min, bounds_max = (
        scene.collect_mesh_data(record["obj"]))
    if (tuple(bounds_min) != record["snapshot"]["boundsMin"] or
            tuple(bounds_max) != record["snapshot"]["boundsMax"]):
        raise RuntimeError(
            f"Triangulation changed bounds for {record['exactName']}")


def _rollback_triangle_export(records, mesh_state_before):
    errors = []
    for record in records:
        temporary = record["temporary"]
        if temporary is None:
            continue
        try:
            temporary.name = record["cleanupName"]
            if temporary.name != record["cleanupName"]:
                errors.append(
                    f"cleanup name {temporary.name!r} != "
                    f"{record['cleanupName']!r}")
        except Exception as exc:  # noqa: BLE001 - aggregate rollback evidence.
            errors.append(f"temporary rename failed: {type(exc).__name__}: {exc}")

    for record in records:
        try:
            record["obj"].data = record["original"]
            record["original"].name = record["exactName"]
            if record["original"].name != record["exactName"]:
                errors.append(
                    f"source name {record['original'].name!r} != "
                    f"{record['exactName']!r}")
        except Exception as exc:  # noqa: BLE001 - aggregate rollback evidence.
            errors.append(f"source restore failed: {type(exc).__name__}: {exc}")

    for record in records:
        temporary = record["temporary"]
        if temporary is None:
            continue
        try:
            bpy.data.meshes.remove(temporary)
            record["temporary"] = None
        except Exception as exc:  # noqa: BLE001 - aggregate rollback evidence.
            errors.append(f"temporary removal failed: {type(exc).__name__}: {exc}")

    try:
        bpy.context.view_layer.update()
    except Exception as exc:  # noqa: BLE001 - aggregate rollback evidence.
        errors.append(f"view-layer restore failed: {type(exc).__name__}: {exc}")

    for record in records:
        try:
            if record["obj"].data is not record["original"]:
                errors.append(f"source pointer not restored: {record['exactName']}")
            if record["original"].as_pointer() != record["originalPointer"]:
                errors.append(f"source ID changed: {record['exactName']}")
            if _mesh_snapshot(record["obj"]) != record["snapshot"]:
                errors.append(f"source geometry changed: {record['exactName']}")
        except Exception as exc:  # noqa: BLE001 - aggregate rollback evidence.
            errors.append(f"source audit failed: {type(exc).__name__}: {exc}")
    if _mesh_id_state() != mesh_state_before:
        errors.append("global mesh ID/name set changed")
    return errors


@contextmanager
def _temporary_triangulated_meshes(objects):
    objects, records = _preflight_triangle_export(objects)
    mesh_state_before = _mesh_id_state()
    body_error = None
    try:
        for record in records:
            record["original"].name = record["sourceName"]
            if record["original"].name != record["sourceName"]:
                raise RuntimeError(
                    f"Could not reserve source mesh name {record['sourceName']}")
        for record in records:
            _triangulate_copy(record)
        bpy.context.view_layer.update()
        yield objects
    except BaseException as exc:  # noqa: BLE001 - rollback includes interrupts.
        body_error = exc

    rollback_errors = _rollback_triangle_export(records, mesh_state_before)
    if rollback_errors:
        rollback_error = RuntimeError(
            "FBX export rollback failed: " + "; ".join(rollback_errors))
        if body_error is not None:
            raise rollback_error from body_error
        raise rollback_error
    if body_error is not None:
        raise body_error.with_traceback(body_error.__traceback__)


def _invoke_fbx_export(path):
    return bpy.ops.export_scene.fbx(filepath=path, **FBX_KWARGS)


def export_fbx(objects, path):
    assert_clean_path(path)
    with _temporary_triangulated_meshes(objects) as export_objects:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in export_objects:
            obj.select_set(True)
        result = _invoke_fbx_export(path)
        if "FINISHED" not in result:
            raise RuntimeError(f"FBX export did not finish: {result}")
