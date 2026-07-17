"""Deterministic UV unwrap and shared-atlas island allocation.

Blender owns the angle-based shape of each UV island.  Atlas placement does
not use Blender's allocation-order-dependent multi-object packer: canonical
seam islands are normalized to the same UV-area-per-square-metre scale, then
their bounding rectangles are placed by a stable global shelf allocator.
The resulting atlas has consistent texel density without starving a large
surface merely because its object also contains hundreds of small islands.
"""

import math

import bmesh
import bpy
import numpy as np

from . import scene

SEAM_ANGLE_THRESHOLD = math.radians(66)
UNWRAP_MARGIN = 0.001
PACK_PADDING = 0.002
GEOMETRIC_UV_AREA_TARGET = 0.30
UV_CONNECT_EPS = 1e-7
SHELF_BINARY_HIGH = 5.0
SHELF_BINARY_STEPS = 50
REPORT_GRID = 256
DEGENERATE_AREA_EPS = 1e-12


def _ensure_uv_layer(obj):
    if obj.data.uv_layers.active is None:
        obj.data.uv_layers.new(name="UVMap")


def _mark_seams(obj):
    """Marks seams on ``obj`` (must already be the sole selected+active
    object in edit mode) from dihedral face angle."""
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    for edge in bm.edges:
        try:
            angle = edge.calc_face_angle()
        except ValueError:
            # Boundary edge (0 or 1 linked face): no dihedral angle to test.
            continue
        edge.seam = angle > SEAM_ANGLE_THRESHOLD
    bmesh.update_edit_mesh(obj.data)


def _unwrap_one(obj):
    """Marks seams and runs an isolated angle-based unwrap on ``obj``."""
    _ensure_uv_layer(obj)

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode='EDIT')
    if len(obj.data.polygons) > 0:
        _mark_seams(obj)
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=UNWRAP_MARGIN)
        bpy.ops.uv.select_all(action='SELECT')
        bpy.ops.uv.average_islands_scale(scale_uv=True, shear=True)
    bpy.ops.object.mode_set(mode='OBJECT')


def _actual_uv_island_faces(obj):
    """Returns canonical face tuples for actual, not merely seam, UV islands.

    Angle unwrap can introduce an implicit cut on a smooth closed surface.
    Faces therefore join only when their shared edge is not a marked seam and
    both endpoint UVs agree within a tight fixed tolerance.
    """
    mesh = obj.data
    uv_data = mesh.uv_layers.active.data
    edge_faces = [[] for _edge in mesh.edges]
    edge_face_uvs = [{} for _edge in mesh.edges]
    for polygon in mesh.polygons:
        polygon_loops = tuple(polygon.loop_indices)
        for offset, loop_index in enumerate(polygon_loops):
            next_loop_index = polygon_loops[(offset + 1) % len(polygon_loops)]
            edge_index = mesh.loops[loop_index].edge_index
            edge_faces[edge_index].append(polygon.index)
            edge_face_uvs[edge_index][polygon.index] = {
                mesh.loops[loop_index].vertex_index:
                    tuple(float(value) for value in uv_data[loop_index].uv),
                mesh.loops[next_loop_index].vertex_index:
                    tuple(float(value) for value in uv_data[next_loop_index].uv),
            }

    def uv_continuous(edge_index, left, right):
        left_uvs = edge_face_uvs[edge_index][left]
        right_uvs = edge_face_uvs[edge_index][right]
        if left_uvs.keys() != right_uvs.keys():
            return False
        return all(
            max(abs(left_uvs[vertex][axis] - right_uvs[vertex][axis])
                for axis in (0, 1)) <= UV_CONNECT_EPS
            for vertex in left_uvs)

    visited = set()
    islands = []
    for seed in range(len(mesh.polygons)):
        if seed in visited:
            continue
        visited.add(seed)
        members = []
        stack = [seed]
        while stack:
            polygon_index = stack.pop()
            members.append(polygon_index)
            polygon = mesh.polygons[polygon_index]
            neighbours = []
            for loop_index in polygon.loop_indices:
                edge_index = mesh.loops[loop_index].edge_index
                if mesh.edges[edge_index].use_seam:
                    continue
                neighbours.extend(
                    neighbour for neighbour in edge_faces[edge_index]
                    if neighbour != polygon_index
                    and uv_continuous(edge_index, polygon_index, neighbour))
            for neighbour in sorted(set(neighbours), reverse=True):
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)
        islands.append(tuple(sorted(members)))
    return tuple(islands)


def _island_items(mesh_objects):
    """Builds stable metre-normalized rectangle records for every UV island."""
    items = []
    for obj in mesh_objects:
        mesh = obj.data
        uv_layer = mesh.uv_layers.active
        if uv_layer is None:
            raise ValueError(f"Unwrapped object {obj.name!r} has no UV layer")
        mesh.calc_loop_triangles()
        triangles_by_polygon = [[] for _polygon in mesh.polygons]
        for triangle in mesh.loop_triangles:
            triangles_by_polygon[triangle.polygon_index].append(triangle)

        for face_indices in _actual_uv_island_faces(obj):
            loop_indices = tuple(
                loop_index
                for polygon_index in face_indices
                for loop_index in mesh.polygons[polygon_index].loop_indices)
            uv_area = 0.0
            area_3d = 0.0
            for polygon_index in face_indices:
                for triangle in triangles_by_polygon[polygon_index]:
                    area_3d += float(triangle.area)
                    uv0, uv1, uv2 = (
                        uv_layer.data[loop_index].uv
                        for loop_index in triangle.loops)
                    uv_area += _uv_triangle_area(uv0, uv1, uv2)
            if area_3d <= DEGENERATE_AREA_EPS or uv_area <= DEGENERATE_AREA_EPS:
                raise ValueError(
                    f"Degenerate UV island {obj.name!r} faces {face_indices}: "
                    f"area_3d={area_3d}, uv_area={uv_area}")

            coordinates = np.asarray(
                [uv_layer.data[index].uv[:] for index in loop_indices],
                dtype=np.float64)
            if not np.all(np.isfinite(coordinates)):
                raise ValueError(
                    f"Non-finite UV island {obj.name!r} faces {face_indices}")
            minimum = np.min(coordinates, axis=0)
            maximum = np.max(coordinates, axis=0)
            normalizer = math.sqrt(area_3d / uv_area)
            width = float((maximum[0] - minimum[0]) * normalizer)
            height = float((maximum[1] - minimum[1]) * normalizer)
            if width <= 0.0 or height <= 0.0:
                raise ValueError(
                    f"Zero-extent UV island {obj.name!r} faces {face_indices}")

            rotate = height > width + 1e-12
            packed_width, packed_height = (
                (height, width) if rotate else (width, height))
            items.append({
                "key": (obj.name, face_indices[0], face_indices),
                "object": obj,
                "face_indices": face_indices,
                "loop_indices": loop_indices,
                "minimum": minimum,
                "normalizer": normalizer,
                "source_width": width,
                "rotate": rotate,
                "width": packed_width,
                "height": packed_height,
                "area": area_3d,
            })

    items.sort(key=lambda item: (
        -round(item["height"], 12),
        -round(item["width"], 12),
        -round(item["area"], 12),
        item["key"],
    ))
    return items


def _try_shelf_layout(items, multiplier, padding):
    x = padding
    y = padding
    row_height = 0.0
    placements = {}
    for item in items:
        width = item["width"] * multiplier
        height = item["height"] * multiplier
        if x + width + padding > 1.0:
            x = padding
            y += row_height + padding
            row_height = 0.0
        if x + width + padding > 1.0:
            return None
        if y + height + padding > 1.0:
            return None
        placements[item["key"]] = (x, y)
        x += width + padding
        row_height = max(row_height, height)
    return placements


def _shelf_layout(items, padding):
    low = 0.0
    high = SHELF_BINARY_HIGH
    placements = _try_shelf_layout(items, low, padding)
    for _step in range(SHELF_BINARY_STEPS):
        midpoint = (low + high) * 0.5
        candidate = _try_shelf_layout(items, midpoint, padding)
        if candidate is None:
            high = midpoint
        else:
            low = midpoint
            placements = candidate
    if placements is None:
        raise RuntimeError("Deterministic UV shelf allocation found no layout")
    return placements, low


def _allocate_shared_atlas(mesh_objects):
    padding = PACK_PADDING
    items = _island_items(mesh_objects)
    fit_placements, fit_multiplier = _shelf_layout(items, padding)
    total_area = sum(item["area"] for item in items)
    cap_multiplier = math.sqrt(GEOMETRIC_UV_AREA_TARGET / total_area)
    multiplier = min(fit_multiplier, cap_multiplier)
    if multiplier == fit_multiplier:
        placements = fit_placements
    else:
        placements = _try_shelf_layout(items, multiplier, padding)
        if placements is None:
            raise RuntimeError(
                "Capped deterministic UV shelf layout unexpectedly failed")
    for item in items:
        obj = item["object"]
        x, y = placements[item["key"]]
        uv_layer = obj.data.uv_layers.active
        minimum = item["minimum"]
        normalizer = item["normalizer"]
        source_width = item["source_width"]
        for loop_index in item["loop_indices"]:
            source = uv_layer.data[loop_index].uv
            local_u = (float(source[0]) - minimum[0]) * normalizer
            local_v = (float(source[1]) - minimum[1]) * normalizer
            if item["rotate"]:
                local_u, local_v = local_v, source_width - local_u
            uv_layer.data[loop_index].uv = (
                x + local_u * multiplier,
                y + local_v * multiplier,
            )
    return {
        "fit_multiplier": fit_multiplier,
        "cap_multiplier": cap_multiplier,
        "multiplier": multiplier,
        "island_count": len(items),
        "padding": padding,
    }


def unwrap_and_pack(objects, atlas_px=4096, texel_per_m=220):
    """Canonicalizes, unwraps and allocates a deterministic shared atlas."""
    if atlas_px <= 0:
        raise ValueError(f"atlas_px must be positive, got {atlas_px}")
    if texel_per_m <= 0:
        raise ValueError(f"texel_per_m must be positive, got {texel_per_m}")
    mesh_objects = sorted(objects.values(), key=lambda obj: obj.name)
    if not mesh_objects:
        return

    for obj in mesh_objects:
        scene.canonicalize_mesh(obj)
        _unwrap_one(obj)
    return _allocate_shared_atlas(mesh_objects)


def _uv_triangle_area(uv0, uv1, uv2):
    return abs(
        (uv1[0] - uv0[0]) * (uv2[1] - uv0[1])
        - (uv2[0] - uv0[0]) * (uv1[1] - uv0[1])) / 2.0


def _rasterize_triangle(counts, uv0, uv1, uv2, grid):
    x0, y0 = uv0[0] * grid, uv0[1] * grid
    x1, y1 = uv1[0] * grid, uv1[1] * grid
    x2, y2 = uv2[0] * grid, uv2[1] * grid

    min_x = max(int(math.floor(min(x0, x1, x2))), 0)
    max_x = min(int(math.ceil(max(x0, x1, x2))), grid)
    min_y = max(int(math.floor(min(y0, y1, y2))), 0)
    max_y = min(int(math.ceil(max(y0, y1, y2))), grid)
    if min_x >= max_x or min_y >= max_y:
        return

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-12:
        return  # Degenerate triangle in UV space (zero area).

    xs = np.arange(min_x, max_x, dtype=np.float64) + 0.5
    ys = np.arange(min_y, max_y, dtype=np.float64) + 0.5
    gx, gy = np.meshgrid(xs, ys)

    w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
    w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
    w2 = 1.0 - w0 - w1

    eps = -1e-6
    inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
    counts[min_y:max_y, min_x:max_x] += inside.astype(np.int32)


def report(objects, atlas_px=4096, grid=REPORT_GRID):
    """Rasterizes every mesh's active-UV-layer triangles into a ``grid`` x
    ``grid`` texel-count array and returns ``{"coverage": ...,
    "max_overlap": ..., "density_px_per_m": ...}``:

    - ``coverage``: fraction of texels claimed by >= 1 triangle;
    - ``max_overlap``: fraction of ALL texels (not just covered ones)
      claimed by >= 2 triangles;
    - ``density_px_per_m``: median over all triangles of
      ``sqrt(uv_area / 3d_area) * atlas_px`` (triangles with degenerate 3D
      area are skipped).

    Deterministic: no randomness.
    """
    counts = np.zeros((grid, grid), dtype=np.int32)
    densities = []

    for obj in objects.values():
        mesh = obj.data
        uv_layer = mesh.uv_layers.active
        if uv_layer is None:
            continue
        mesh.calc_loop_triangles()
        for tri in mesh.loop_triangles:
            l0, l1, l2 = tri.loops
            uv0 = uv_layer.data[l0].uv
            uv1 = uv_layer.data[l1].uv
            uv2 = uv_layer.data[l2].uv
            _rasterize_triangle(counts, uv0, uv1, uv2, grid)
            area_3d = tri.area
            if area_3d > DEGENERATE_AREA_EPS:
                uv_area = _uv_triangle_area(uv0, uv1, uv2)
                densities.append(math.sqrt(uv_area / area_3d) * atlas_px)

    total = grid * grid
    coverage = float(np.count_nonzero(counts >= 1)) / total
    max_overlap = float(np.count_nonzero(counts >= 2)) / total
    density = float(np.median(densities)) if densities else 0.0
    return {
        "coverage": coverage,
        "max_overlap": max_overlap,
        "density_px_per_m": density,
    }
