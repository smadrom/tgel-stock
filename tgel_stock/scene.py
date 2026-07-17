import bmesh
import bpy


CANONICAL_COORD_DIGITS = 8
_ALLOWED_PRE_UV_ATTRIBUTES = {
    "position",
    ".edge_verts",
    ".corner_vert",
    ".corner_edge",
    ".select_vert",
    ".select_edge",
    ".select_poly",
    "sharp_face",
    "material_index",
}


def to_blender(v):
    return (-v[0], -v[2], v[1])


def to_unity(v):
    return (-v[0], v[2], -v[1])


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def mesh_object(name, bm):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _rounded_coordinate(coordinate):
    return tuple(round(float(component), CANONICAL_COORD_DIGITS) for component in coordinate)


def _canonical_face_cycle(indices, exact_coordinates):
    """Cyclically rotates a face to a stable start without reversing winding."""
    candidates = []
    for start in range(len(indices)):
        cycle = tuple(indices[start:] + indices[:start])
        rounded = tuple(_rounded_coordinate(exact_coordinates[index]) for index in cycle)
        exact = tuple(exact_coordinates[index] for index in cycle)
        candidates.append((rounded, exact, cycle))
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))


def canonicalize_mesh(obj):
    """Rebuilds one generated pre-UV mesh in deterministic index order.

    Current generated meshes have unique exact vertex coordinates and no
    loose/custom edge data.  Vertices are sorted by rounded then exact local
    coordinates.  Each polygon is only cyclically rotated (never reversed),
    then polygons are sorted by length and their rounded/exact coordinate
    sequence.  Coordinates, winding, topology, bounds, object transforms,
    material-slot identity/order, per-face material indices and smooth flags
    are thereby preserved while Blender's otherwise allocation-dependent
    vertex/face order is removed before UV operators run.
    """
    if obj is None or obj.type != 'MESH' or obj.data is None:
        raise ValueError(f"Cannot canonicalize non-mesh object: {obj!r}")
    mesh = obj.data
    if mesh.users != 1:
        raise ValueError(
            f"Canonical mesh {obj.name!r} must be single-user, got {mesh.users}")
    if obj.vertex_groups:
        raise ValueError(f"Canonical mesh {obj.name!r} has unsupported vertex groups")
    if mesh.shape_keys is not None:
        raise ValueError(f"Canonical mesh {obj.name!r} has unsupported shape keys")
    unexpected_attributes = sorted(
        attribute.name for attribute in mesh.attributes
        if attribute.name not in _ALLOWED_PRE_UV_ATTRIBUTES)
    if unexpected_attributes:
        raise ValueError(
            f"Canonical mesh {obj.name!r} has unsupported attributes: "
            f"{unexpected_attributes}")
    if mesh.uv_layers:
        raise ValueError(f"Canonical mesh {obj.name!r} already has UV layers")
    if mesh.color_attributes:
        raise ValueError(f"Canonical mesh {obj.name!r} has color attributes")
    if any(edge.is_loose for edge in mesh.edges):
        raise ValueError(f"Canonical mesh {obj.name!r} has loose edges")
    if any(edge.use_seam or edge.use_edge_sharp for edge in mesh.edges):
        raise ValueError(f"Canonical mesh {obj.name!r} has authored edge flags")

    exact_coordinates = [
        tuple(float(component) for component in vertex.co)
        for vertex in mesh.vertices
    ]
    if len(exact_coordinates) != len(set(exact_coordinates)):
        raise ValueError(
            f"Canonical mesh {obj.name!r} has duplicate vertex coordinates")

    vertex_order = sorted(
        range(len(exact_coordinates)),
        key=lambda index: (
            _rounded_coordinate(exact_coordinates[index]),
            exact_coordinates[index],
        ))
    old_to_new = {
        old_index: new_index for new_index, old_index in enumerate(vertex_order)
    }
    sorted_coordinates = [exact_coordinates[index] for index in vertex_order]

    face_records = []
    for polygon in mesh.polygons:
        remapped = [old_to_new[index] for index in polygon.vertices]
        rounded, exact, cycle = _canonical_face_cycle(remapped, sorted_coordinates)
        face_records.append({
            "vertices": cycle,
            "rounded": rounded,
            "exact": exact,
            "material_index": int(polygon.material_index),
            "use_smooth": bool(polygon.use_smooth),
        })
    face_records.sort(key=lambda record: (
        len(record["vertices"]),
        record["rounded"],
        record["exact"],
        record["material_index"],
        record["use_smooth"],
    ))

    materials = tuple(mesh.materials)
    old_name = mesh.name
    temporary_old_name = old_name + ".PreCanonical"
    new_name = old_name + ".Canonical"
    if bpy.data.meshes.get(temporary_old_name) is not None:
        raise ValueError(f"Canonical temporary mesh already exists: {temporary_old_name}")
    if bpy.data.meshes.get(new_name) is not None:
        raise ValueError(f"Canonical replacement mesh already exists: {new_name}")
    new_mesh = bpy.data.meshes.new(new_name)
    old_renamed = False
    swapped = False
    try:
        new_mesh.from_pydata(
            sorted_coordinates, [],
            [record["vertices"] for record in face_records])
        new_mesh.update(calc_edges=True)
        for material in materials:
            new_mesh.materials.append(material)
        for polygon, record in zip(new_mesh.polygons, face_records):
            polygon.material_index = record["material_index"]
            polygon.use_smooth = record["use_smooth"]
        new_mesh.update()

        mesh.name = temporary_old_name
        if mesh.name != temporary_old_name:
            raise RuntimeError(
                f"Blender renamed canonical source temporary {temporary_old_name!r} "
                f"to {mesh.name!r}")
        old_renamed = True
        new_mesh.name = old_name
        if new_mesh.name != old_name:
            raise RuntimeError(
                f"Blender renamed canonical mesh {old_name!r} to {new_mesh.name!r}")
        obj.data = new_mesh
        swapped = True
        if mesh.users != 0:
            raise RuntimeError(
                f"Canonical source mesh {old_name!r} still has {mesh.users} users")
        bpy.data.meshes.remove(mesh)
    except Exception:
        if swapped and bpy.data.meshes.get(mesh.name) is mesh:
            obj.data = mesh
        if bpy.data.meshes.get(new_mesh.name) is new_mesh:
            bpy.data.meshes.remove(new_mesh)
        if old_renamed and bpy.data.meshes.get(mesh.name) is mesh:
            mesh.name = old_name
        raise


def box(name, centre_unity, size_unity):
    """Axis-aligned box authored in vehicle space."""
    bm = bmesh.new()
    cx, cy, cz = centre_unity
    sx, sy, sz = (s * 0.5 for s in size_unity)
    verts = []
    for dx in (-sx, sx):
        for dy in (-sy, sy):
            for dz in (-sz, sz):
                verts.append(bm.verts.new(to_blender((cx + dx, cy + dy, cz + dz))))
    bm.verts.ensure_lookup_table()
    faces = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    for face in faces:
        bm.faces.new([verts[i] for i in face])
    bm.normal_update()
    return mesh_object(name, bm)


def collect_mesh_data(obj):
    mesh = obj.data
    mesh.calc_loop_triangles()
    positions = [to_unity(v.co) for v in mesh.vertices]
    normals = [to_unity(v.normal) for v in mesh.vertices]
    uv_layer = mesh.uv_layers.active
    uvs = [tuple(d.uv) for d in uv_layer.data] if uv_layer else []
    triangles = [tuple(t.vertices) for t in mesh.loop_triangles]
    xs = [p[0] for p in positions] or [0.0]
    ys = [p[1] for p in positions] or [0.0]
    zs = [p[2] for p in positions] or [0.0]
    return (positions, normals, uvs, triangles,
            (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))
