"""Deterministic LOD duplication and Blender DECIMATE application (Task 14).

``build_lods`` is deliberately non-mutating with respect to its input mapping
and LOD0 objects.  Every LOD0 mesh receives an independently-owned LOD1 mesh
at collapse ratio 0.40.  Meshes whose LOD0 rendered triangle count is at
least 200 additionally receive LOD2 at ratio 0.12; all smaller LOD2 parts are
dropped by that source-count rule without first constructing a disposable
clone.

The decimator is deterministic for an identical already-packed input mesh.
Fresh assembly/UV-pack determinism is a wider Task 15 concern and is not
claimed here.
"""

import bmesh
import bpy


LOD1_SUFFIX = "__LOD1"
LOD2_SUFFIX = "__LOD2"
LOD1_RATIO = 0.40
LOD2_RATIO = 0.12
LOD2_SOURCE_TRIANGLE_FLOOR = 200
_MODIFIER_NAME = "TGEL_LOD_DECIMATE"


def _triangle_count(obj):
    obj.data.calc_loop_triangles()
    return len(obj.data.loop_triangles)


def _source_items(objects):
    """Validates and snapshots the source mapping before any Blender writes."""
    items = list(objects.items())
    for key, obj in items:
        if obj is None or obj.type != 'MESH' or obj.data is None:
            raise ValueError(f"LOD source {key!r} is not a mesh object")
        if key != obj.name:
            raise ValueError(
                f"LOD source mapping key {key!r} != object name {obj.name!r}")
        if len(obj.modifiers) != 0:
            raise ValueError(
                f"LOD source {key!r} has unapplied modifiers; expected frozen LOD0")
    return items


def _target_specs(items):
    specs = []
    for source_name, source in items:
        specs.append((source_name + LOD1_SUFFIX, source, LOD1_RATIO))
        if _triangle_count(source) >= LOD2_SOURCE_TRIANGLE_FLOOR:
            specs.append((source_name + LOD2_SUFFIX, source, LOD2_RATIO))
    return specs


def _preflight_collisions(specs):
    """Checks every output ID before the first object or mesh is cloned."""
    target_names = [target_name for target_name, _source, _ratio in specs]
    if len(target_names) != len(set(target_names)):
        raise ValueError("LOD target names are not unique")
    for target_name in target_names:
        if bpy.data.objects.get(target_name) is not None:
            raise ValueError(f"LOD object target already exists: {target_name}")
        if bpy.data.meshes.get(target_name) is not None:
            raise ValueError(f"LOD mesh target already exists: {target_name}")


def _remove_created(created):
    """Best-effort rollback for a failed Blender modifier application."""
    for record in reversed(created):
        obj = record[0]
        mesh = record[1]
        if obj is not None and bpy.data.objects.get(obj.name) is obj:
            bpy.data.objects.remove(obj, do_unlink=True)
        if (mesh is not None and mesh.users == 0
                and bpy.data.meshes.get(mesh.name) is mesh):
            bpy.data.meshes.remove(mesh)


def _rendered_face_signature(mesh):
    """Returns the exact polygon-owned data that cleanup must not change.

    Coordinates rather than vertex indices are recorded because removing
    non-face vertices is allowed to compact Blender's vertex array.  Polygon
    order and each polygon's loop order remain significant, so winding is
    covered as well as topology.  Every UV layer is included rather than only
    the active layer.
    """
    polygons = tuple(
        (
            tuple(
                tuple(float(component) for component in mesh.vertices[index].co)
                for index in polygon.vertices
            ),
            int(polygon.material_index),
            bool(polygon.use_smooth),
        )
        for polygon in mesh.polygons
    )
    uv_layers = tuple(
        (
            layer.name,
            tuple(
                tuple(
                    tuple(float(component) for component in layer.data[index].uv)
                    for index in polygon.loop_indices
                )
                for polygon in mesh.polygons
            ),
        )
        for layer in mesh.uv_layers
    )
    return polygons, uv_layers


def _remove_non_face_geometry(mesh):
    """Drops post-DECIMATE wire/isolated geometry without touching faces.

    Most generated LOD meshes have no such geometry and deliberately avoid a
    BMesh round-trip.  For the exceptional case, an exact before/after face
    signature guards the frozen rendered contract.
    """
    face_vertices = {
        vertex_index
        for polygon in mesh.polygons
        for vertex_index in polygon.vertices
    }
    unreferenced_vertices = (
        set(range(len(mesh.vertices))) - face_vertices
    )
    loose_edge_count = sum(edge.is_loose for edge in mesh.edges)
    if not unreferenced_vertices and loose_edge_count == 0:
        return 0, 0

    before = _rendered_face_signature(mesh)
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        loose_edges = [edge for edge in bm.edges if not edge.link_faces]
        for edge in loose_edges:
            bm.edges.remove(edge)
        unused_vertices = [vertex for vertex in bm.verts if not vertex.link_faces]
        for vertex in unused_vertices:
            bm.verts.remove(vertex)
        bm.to_mesh(mesh)
    finally:
        bm.free()

    mesh.update()
    remaining_face_vertices = {
        vertex_index
        for polygon in mesh.polygons
        for vertex_index in polygon.vertices
    }
    remaining_unreferenced = (
        set(range(len(mesh.vertices))) - remaining_face_vertices
    )
    remaining_loose = [edge.index for edge in mesh.edges if edge.is_loose]
    if remaining_unreferenced or remaining_loose:
        raise RuntimeError(
            "LOD non-face cleanup was incomplete: "
            f"looseEdges={remaining_loose}, "
            f"unreferencedVertices={sorted(remaining_unreferenced)}")
    if _rendered_face_signature(mesh) != before:
        raise RuntimeError("LOD non-face cleanup changed rendered face data")
    return len(unreferenced_vertices), loose_edge_count


def _clone_and_decimate(source, target_name, ratio, created):
    # Object.copy initially shares source.data.  Record the object immediately
    # so even an exceptional mesh-data copy cannot leak a partial Blender ID.
    clone = source.copy()
    record = [clone, None]
    created.append(record)

    mesh = source.data.copy()
    clone.data = mesh
    record[1] = mesh

    clone.name = target_name
    mesh.name = target_name
    if clone.name != target_name or mesh.name != target_name:
        raise RuntimeError(
            f"Blender renamed LOD target {target_name!r} to "
            f"object={clone.name!r}, mesh={mesh.name!r}")

    collections = tuple(source.users_collection)
    if not collections:
        collections = (bpy.context.scene.collection,)
    for collection in collections:
        collection.objects.link(clone)

    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    clone.select_set(True)
    bpy.context.view_layer.objects.active = clone

    modifier = clone.modifiers.new(name=_MODIFIER_NAME, type='DECIMATE')
    modifier.decimate_type = 'COLLAPSE'
    modifier.ratio = ratio
    # Without this Blender 5.1.2 leaves many n-gons.  Explicit triangulation
    # keeps manifest triangle counts aligned with FBX/Unity interpretation.
    modifier.use_collapse_triangulate = True
    outcome = bpy.ops.object.modifier_apply(modifier=modifier.name)
    if outcome != {'FINISHED'}:
        raise RuntimeError(
            f"Failed to apply DECIMATE to {target_name}: {sorted(outcome)}")

    _remove_non_face_geometry(clone.data)
    clone.data.update()
    clone.data.calc_loop_triangles()
    if clone.name != target_name or clone.data.name != target_name:
        raise RuntimeError(
            f"Applied LOD target names changed for {target_name!r}: "
            f"object={clone.name!r}, mesh={clone.data.name!r}")
    return clone


def build_lods(objects):
    """Returns a new name->object map containing LOD0 plus generated LODs.

    The input mapping and every input object/mesh remain unchanged.  All
    output object and mesh-data names are exact; a pre-existing Blender object
    or mesh with any target name is a hard error rather than an automatic
    ``.001`` suffix.  Any failure after cloning begins rolls back every clone
    created by this call and then re-raises the original exception.
    """
    items = _source_items(objects)
    specs = _target_specs(items)
    _preflight_collisions(specs)

    result = dict(items)
    created = []
    try:
        for target_name, source, ratio in specs:
            result[target_name] = _clone_and_decimate(
                source, target_name, ratio, created)
    except Exception:
        _remove_created(created)
        raise
    return result
