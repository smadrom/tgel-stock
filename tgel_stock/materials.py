"""Deterministic livery and procedural weathering atlas composition.

Task 13 turns Task 12's shared AO/curvature bakes plus the assembly UVs into
the two colour textures consumed by the runtime material:

* ``albedo.png`` -- muted per-part livery colours, deterministic +/-6 percent
  value variation, a static AO/height/noise dirt floor, and bfont-generated
  reporting/capacity/build stencils on the side-door UVs;
* ``mask.png`` -- R=AO, G=curvature/wear, B=upward-facing snow, A=roughness.

The atlas composer rasterizes the already-authored UV triangles directly.
That is equivalent to an emission bake for these generated, constant/analytic
signals, avoids another expensive Cycles pass, and is deterministic at texel
level.  Stencil glyph geometry still comes from Blender's bundled ``bfont``:
a temporary FONT object is converted to triangles, projected into a known
side-panel UV rectangle, then removed.

Production/default resolution is 4096 (Task 11's accepted progress-ledger
override).  Both public calls expose an optional ``atlas_px=4096`` parameter
and reject inputs at another size.  Focused gates explicitly pass their lower
test resolution, so no test-sized texture can silently enter production.

Blender 5.1.2 applies a display transform while reading PNG pixels unless the
image is first marked ``Non-Color``.  ``_load_raw_png`` deliberately performs
that assignment before its first pixel read.
"""

import hashlib
import os
import re

import bpy
import numpy as np

from . import assert_clean_path

DEFAULT_ATLAS_PX = 4096
ALBEDO_FILENAME = "albedo.png"
MASK_FILENAME = "mask.png"

STATIC_DIRT_STRENGTH = 0.35
VALUE_VARIATION = 0.06
SNOW_PADDING = 0.03
PRODUCTION_DILATION_PX = 16

_FRAME_TOKENS = ("Frame", "Underframe", "CentreSill", "EndSill", "Pilot")
_RUNNING_GEAR_TOKENS = (
    "Bogie", "Wheel", "Axle", "Spring", "Bolster", "SideFrame", "Axlebox",
    "Coupler", "Knuckle", "Shank", "CutLever",
)
_ROUGHNESS = {
    "body": 0.62,
    "glass": 0.45,
    "runningGear": 0.78,
    # Task 13 freezes only three bases.  Frame follows running gear; roof
    # follows body instead of inventing uncontracted fourth/fifth values.
    "frame": 0.78,
    "roof": 0.62,
}


def _part_category(name):
    """Returns the Task 13 livery rule for one contract object name."""
    short_name = name.rsplit(".", 1)[-1]
    # The assembly hierarchy is the strongest source of truth.  In
    # particular Hub, TreadAndFlange and CrossMember short names contain no
    # token from the prose rule but are unambiguously running gear.
    if name.startswith("Visuals.RunningGear.") or name.startswith("Couplers."):
        return "runningGear"
    # SideFrame is explicitly a running-gear part even though its spelling
    # also contains "Frame", so the more specific rule has precedence.
    if any(token in short_name for token in _RUNNING_GEAR_TOKENS):
        return "runningGear"
    if any(token in short_name for token in _FRAME_TOKENS):
        return "frame"
    if "Roof" in short_name:
        return "roof"
    if "CabGlass" in short_name:
        return "glass"
    return "body"


def _part_value_variation(seed, name):
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)
    return (unit * 2.0 - 1.0) * VALUE_VARIATION


def _wear_multiplier(name):
    """Art-direction emphasis for frequently handled/exposed parts."""
    short_name = name.rsplit(".", 1)[-1]
    if "Handrail" in short_name:
        return 1.55
    if "Door" in short_name or "Step" in short_name:
        return 1.35
    return 1.0


def _rain_streak_signal(longitudinal, side_facing, lower, seed):
    """Narrow seed-phased vertical streaks on vehicle side faces.

    The stripe phase depends only on longitudinal vehicle position, so one
    stripe stays coherent from roofline to sill; ``lower`` makes its deposit
    stronger toward the bottom.  Non-side-facing texels receive exactly zero.
    """
    longitudinal = np.asarray(longitudinal, dtype=np.float32)
    side_facing = np.clip(np.asarray(side_facing, dtype=np.float32), 0.0, 1.0)
    lower = np.clip(np.asarray(lower, dtype=np.float32), 0.0, 1.0)
    seed_phase = (int(seed) & 0xFFFF) * 0.00137
    primary = np.power(
        0.5 + 0.5 * np.sin(longitudinal * 11.73 + seed_phase), 10.0)
    secondary = np.power(
        0.5 + 0.5 * np.sin(longitudinal * 19.19 - seed_phase * 0.61), 14.0)
    stripes = np.maximum(primary, secondary)
    return side_facing * lower * stripes


def _part_colour(recipe, name):
    category = _part_category(name)
    base = np.asarray(recipe.livery[category], dtype=np.float32)
    varied = base * (1.0 + _part_value_variation(recipe.seed, name))
    return category, np.clip(varied, 0.0, 1.0)


def _material_name(model_id, name):
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return f"TGEL.Livery.{safe_model}.{safe_name}"


def _assign_part_material(obj, model_id, category, colour):
    """Assigns a real Principled material in addition to baking the atlas."""
    name = _material_name(model_id, obj.name)
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is None:
        principled = next(
            (node for node in material.node_tree.nodes
             if node.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if principled is None:
        principled = material.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (
        float(colour[0]), float(colour[1]), float(colour[2]), 1.0)
    principled.inputs["Roughness"].default_value = _ROUGHNESS[category]
    obj.data.materials.clear()
    obj.data.materials.append(material)


def _load_raw_png(path):
    assert_clean_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    image = bpy.data.images.load(path, check_existing=False)
    try:
        # Contractual ordering: set data colorspace BEFORE any pixel access.
        image.colorspace_settings.name = 'Non-Color'
        width, height = image.size
        channels = image.channels
        flat = np.empty(width * height * channels, dtype=np.float32)
        image.pixels.foreach_get(flat)
        raw = flat.reshape(height, width, channels)
        if channels == 4:
            return raw.copy()
        result = np.ones((height, width, 4), dtype=np.float32)
        result[:, :, :min(3, channels)] = raw[:, :, :min(3, channels)]
        return result
    finally:
        bpy.data.images.remove(image)


def _save_png(path, pixels, non_color, channel_packed=False):
    assert_clean_path(path)
    height, width, channels = pixels.shape
    if channels != 4:
        raise ValueError(f"Expected RGBA pixels, got {channels} channels")
    image = bpy.data.images.new(
        f"TGEL.{os.path.basename(path)}", width, height,
        alpha=True, float_buffer=False)
    try:
        image.colorspace_settings.name = 'Non-Color' if non_color else 'sRGB'
        if channel_packed:
            # The alpha component is independent roughness data, never
            # transparency.  This prevents Blender from premultiplying RGB.
            image.alpha_mode = 'CHANNEL_PACKED'
        image.pixels.foreach_set(
            np.ascontiguousarray(pixels, dtype=np.float32).reshape(-1))
        image.update()
        image.filepath_raw = path
        image.file_format = 'PNG'
        image.save()
    finally:
        bpy.data.images.remove(image)


def _dilation_radius(atlas_px):
    """Matches Task 12's frozen 16-texel bake margin at every resolution."""
    if atlas_px <= 0:
        raise ValueError(f"atlas_px must be positive, got {atlas_px}")
    return PRODUCTION_DILATION_PX


def _dilate_unoccupied(pixels, occupied, radius, channels):
    """Nearest-front dilation that never overwrites an occupied UV texel.

    Every iteration grows exactly one texel from the previous frontier.  A
    fixed neighbour priority resolves equal-distance fronts deterministically;
    because only the original/free padding is written, islands cannot bleed
    over one another's authored texels.
    """
    valid = occupied.copy()
    height, width = valid.shape
    directions = (
        (0, -1), (0, 1), (-1, 0), (1, 0),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )
    for _step in range(radius):
        claimed = np.zeros_like(valid)
        for dy, dx in directions:
            if dy >= 0:
                source_y = slice(0, height - dy)
                target_y = slice(dy, height)
            else:
                source_y = slice(-dy, height)
                target_y = slice(0, height + dy)
            if dx >= 0:
                source_x = slice(0, width - dx)
                target_x = slice(dx, width)
            else:
                source_x = slice(-dx, width)
                target_x = slice(0, width + dx)

            source_valid = valid[source_y, source_x]
            target_free = ~(valid[target_y, target_x] | claimed[target_y, target_x])
            take = source_valid & target_free
            if not np.any(take):
                continue
            for channel in channels:
                destination = pixels[target_y, target_x, channel]
                source = pixels[source_y, source_x, channel]
                destination[take] = source[take]
            claimed[target_y, target_x][take] = True
        if not np.any(claimed):
            break
        valid |= claimed



def _pixel_noise(x, y, seed, salt=0):
    """Small stable integer hash, vectorized over texel coordinates."""
    value = (
        np.asarray(x, dtype=np.uint64) * np.uint64(0x9E3779B1)
        + np.asarray(y, dtype=np.uint64) * np.uint64(0x85EBCA77)
        + np.uint64((int(seed) + int(salt)) & 0xFFFFFFFF)
        * np.uint64(0xC2B2AE3D)) & np.uint64(0xFFFFFFFF)
    value ^= value >> np.uint64(16)
    value = (value * np.uint64(0x7FEB352D)) & np.uint64(0xFFFFFFFF)
    value ^= value >> np.uint64(15)
    value = (value * np.uint64(0x846CA68B)) & np.uint64(0xFFFFFFFF)
    value ^= value >> np.uint64(16)
    return value.astype(np.float32) / np.float32(0xFFFFFFFF)


def _raster_triangle(uv, width, height):
    """Returns covered integer texels and barycentric weights for one UV tri."""
    if not np.all(np.isfinite(uv)):
        return None
    u0, u1, u2 = uv
    denominator = (
        (u1[1] - u2[1]) * (u0[0] - u2[0])
        + (u2[0] - u1[0]) * (u0[1] - u2[1]))
    if abs(float(denominator)) < 1e-12:
        return None

    x_min = max(0, int(np.ceil(float(np.min(uv[:, 0]) * width - 0.5))))
    x_max = min(width - 1, int(np.floor(float(np.max(uv[:, 0]) * width - 0.5))))
    y_min = max(0, int(np.ceil(float(np.min(uv[:, 1]) * height - 0.5))))
    y_max = min(height - 1, int(np.floor(float(np.max(uv[:, 1]) * height - 0.5))))
    if x_min > x_max or y_min > y_max:
        return None

    x_grid, y_grid = np.meshgrid(
        np.arange(x_min, x_max + 1, dtype=np.int32),
        np.arange(y_min, y_max + 1, dtype=np.int32))
    u = (x_grid.astype(np.float64) + 0.5) / width
    v = (y_grid.astype(np.float64) + 0.5) / height
    w0 = (
        (u1[1] - u2[1]) * (u - u2[0])
        + (u2[0] - u1[0]) * (v - u2[1])) / denominator
    w1 = (
        (u2[1] - u0[1]) * (u - u2[0])
        + (u0[0] - u2[0]) * (v - u2[1])) / denominator
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
    if not np.any(inside):
        return None
    return (
        x_grid[inside], y_grid[inside],
        w0[inside].astype(np.float32),
        w1[inside].astype(np.float32),
        w2[inside].astype(np.float32),
    )


def _iter_object_texels(obj, width, height):
    mesh = obj.data
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        raise ValueError(f"{obj.name} has no active UV layer")
    mesh.calc_loop_triangles()
    uv_data = uv_layer.data
    corner_normals = mesh.corner_normals

    for triangle in mesh.loop_triangles:
        uv = np.asarray(
            [uv_data[loop_index].uv[:] for loop_index in triangle.loops],
            dtype=np.float64)
        raster = _raster_triangle(uv, width, height)
        if raster is None:
            continue
        x, y, w0, w1, w2 = raster
        weights = (w0, w1, w2)
        # Blender Z is vehicle/world up under the frozen scene mapping.
        heights = np.asarray(
            [mesh.vertices[index].co.z for index in triangle.vertices],
            dtype=np.float32)
        longitudinal = np.asarray(
            [-mesh.vertices[index].co.y for index in triangle.vertices],
            dtype=np.float32)
        up_normals = np.asarray(
            [corner_normals[index].vector.z for index in triangle.loops],
            dtype=np.float32)
        side_normals = np.asarray(
            [abs(corner_normals[index].vector.x) for index in triangle.loops],
            dtype=np.float32)
        height_values = sum(weights[i] * heights[i] for i in range(3))
        longitudinal_values = sum(
            weights[i] * longitudinal[i] for i in range(3))
        up_values = sum(weights[i] * up_normals[i] for i in range(3))
        side_values = sum(weights[i] * side_normals[i] for i in range(3))
        yield (
            x, y, height_values, up_values, side_values,
            longitudinal_values)


def _exterior_side_projections(obj):
    """Returns one deterministic exterior side-face projection per X sign.

    Packed UV islands may be rotated, so each projection is an affine fit
    from physical side coordinates ``(vehicle longitudinal, height, 1)`` to
    UV.  The exterior predicate rejects a door slab's hidden inward face;
    a centred body such as LongHood naturally retains both +/-X sides.
    """
    mesh = obj.data
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        return ()
    uv_data = uv_layer.data
    candidates = {-1: [], 1: []}
    for polygon in mesh.polygons:
        if abs(float(polygon.normal.x)) < 0.8:
            continue
        if float(polygon.center.x * polygon.normal.x) <= 1e-6:
            continue
        sign = 1 if polygon.normal.x > 0.0 else -1
        physical_rows = []
        uvs = []
        for loop_index in polygon.loop_indices:
            vertex = mesh.vertices[mesh.loops[loop_index].vertex_index]
            physical_rows.append((-vertex.co.y, vertex.co.z, 1.0))
            uvs.append(uv_data[loop_index].uv[:])
        physical = np.asarray(physical_rows, dtype=np.float64)
        uvs = np.asarray(uvs, dtype=np.float64)
        affine, _residuals, rank, _singular = np.linalg.lstsq(
            physical, uvs, rcond=None)
        if rank < 3:
            continue
        fitted = physical @ affine
        if float(np.max(np.abs(fitted - uvs))) > 1e-5:
            continue
        physical_min = np.min(physical[:, :2], axis=0)
        physical_max = np.max(physical[:, :2], axis=0)
        physical_extent = physical_max - physical_min
        if np.any(physical_extent <= 1e-6):
            continue
        area = float(polygon.area)
        candidates[sign].append((
            area, polygon.index, physical_min, physical_max, affine))

    result = []
    for sign in (-1, 1):
        if not candidates[sign]:
            continue
        # Larger area wins; equal-area ties use the lowest stable polygon
        # index rather than depending on max() encounter order.
        _area, _index, physical_min, physical_max, affine = max(
            candidates[sign], key=lambda item: (item[0], -item[1]))
        extent = physical_max - physical_min
        result.append((
            sign,
            physical_min + extent * (0.08, 0.12),
            physical_max - extent * (0.08, 0.12),
            affine,
        ))
    return tuple(result)


def _projection_uv(projection, normalized_points):
    """Maps normalized glyph/rectangle points through one side projection."""
    sign, physical_min, physical_max, affine = projection
    normalized = np.asarray(normalized_points, dtype=np.float64).copy()
    # From the exterior, vehicle-longitudinal + reads in the opposite screen
    # direction on the +X side.  Flip there so reporting marks are readable,
    # not mirrored; -X retains the direct physical mapping.
    if sign > 0:
        normalized[:, 0] = 1.0 - normalized[:, 0]
    physical = physical_min + normalized * (physical_max - physical_min)
    homogeneous = np.column_stack((physical, np.ones(len(physical))))
    return homogeneous @ affine


def _bfont_triangles(text):
    """Builds normalized 2D glyph triangles from Blender's bundled bfont."""
    curve = bpy.data.curves.new("TGEL.Stencil.BFont", type='FONT')
    curve.body = text
    curve.align_x = 'LEFT'
    curve.align_y = 'BOTTOM'
    curve.size = 1.0
    curve.space_line = 1.15
    curve.extrude = 0.0
    curve.bevel_depth = 0.0
    curve.fill_mode = 'FRONT'
    obj = bpy.data.objects.new("TGEL.Stencil.Text", curve)
    bpy.context.scene.collection.objects.link(obj)
    mesh = None
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        mesh = bpy.data.meshes.new_from_object(
            obj.evaluated_get(depsgraph), depsgraph=depsgraph)
        mesh.calc_loop_triangles()
        points = np.asarray([(vertex.co.x, vertex.co.y) for vertex in mesh.vertices])
        if points.size == 0:
            return []
        point_min = np.min(points, axis=0)
        extent = np.max(points, axis=0) - point_min
        if np.any(extent <= 1e-8):
            return []
        normalized = (points - point_min) / extent
        return [
            np.asarray([normalized[index] for index in triangle.vertices])
            for triangle in mesh.loop_triangles
        ]
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.curves.remove(curve)
        if mesh is not None:
            bpy.data.meshes.remove(mesh)


def _overlay_stencils(albedo, occupied, objects, recipe):
    if recipe.kind == "wagon":
        text = "TGEL 40-102\nCAP 50T\nBLT 7-26"
        suffixes = ("DoorLeft", "DoorRight")
    else:
        text = "TGEL 2115\nBLT 7-26"
        suffixes = ("LongHood",)

    glyph_triangles = _bfont_triangles(text)
    stencil = np.asarray(recipe.livery["stencil"], dtype=np.float32)
    height, width = albedo.shape[:2]
    for name in sorted(objects):
        if not any(name.endswith(suffix) for suffix in suffixes):
            continue
        for projection in _exterior_side_projections(objects[name]):
            for triangle in glyph_triangles:
                uv = _projection_uv(projection, triangle)
                raster = _raster_triangle(uv, width, height)
                if raster is None:
                    continue
                x, y = raster[0], raster[1]
                albedo[y, x, :3] = stencil
                albedo[y, x, 3] = 1.0
                occupied[y, x] = True


def bake_albedo(objects, recipe, out_dir, atlas_px=DEFAULT_ATLAS_PX):
    """Writes and returns ``albedo.png`` for a post-UV vehicle assembly.

    The Task 13 signature has no explicit AO parameter, so the static dirt
    layer consumes Task 12's conventional ``out_dir/ao.png`` output.
    """
    assert_clean_path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ao_path = os.path.join(out_dir, "ao.png")
    ao = _load_raw_png(ao_path)
    height, width = ao.shape[:2]
    if (width, height) != (atlas_px, atlas_px):
        raise ValueError(
            f"AO resolution {(width, height)} != requested "
            f"{(atlas_px, atlas_px)}")

    body_colour = np.asarray(recipe.livery["body"], dtype=np.float32)
    albedo = np.empty((height, width, 4), dtype=np.float32)
    albedo[:, :, :3] = body_colour
    albedo[:, :, 3] = 1.0
    occupied = np.zeros((height, width), dtype=bool)

    vehicle_height = max(float(recipe.height), 1e-6)
    for name in sorted(objects):
        obj = objects[name]
        category, colour = _part_colour(recipe, name)
        # Persisting these generated materials is intentional Task 13 output;
        # this routine does not use or mutate Blender selection state.
        _assign_part_material(obj, recipe.model_id, category, colour)
        for (
                x, y, height_values, _up, side_values,
                longitudinal_values) in _iter_object_texels(
                    obj, width, height):
            cavity = np.clip(1.0 - ao[y, x, 0], 0.0, 1.0)
            lower = np.clip(1.0 - height_values / vehicle_height, 0.0, 1.0)
            noise = _pixel_noise(x, y, recipe.seed, salt=17)
            base_dirt = (
                cavity * (0.25 + 0.75 * lower) * (0.65 + 0.35 * noise))
            rain_streaks = _rain_streak_signal(
                longitudinal_values, side_values, lower, recipe.seed)
            dirt = STATIC_DIRT_STRENGTH * np.clip(
                base_dirt + 0.35 * rain_streaks * (0.75 + 0.25 * noise),
                0.0, 1.0)
            albedo[y, x, :3] = colour[None, :] * (1.0 - dirt[:, None])
            albedo[y, x, 3] = 1.0
            occupied[y, x] = True

    _overlay_stencils(albedo, occupied, objects, recipe)
    _dilate_unoccupied(
        albedo, occupied, _dilation_radius(atlas_px), channels=(0, 1, 2, 3))
    path = os.path.join(out_dir, ALBEDO_FILENAME)
    _save_png(path, np.clip(albedo, 0.0, 1.0), non_color=False)
    return path


def bake_weathering_masks(
        objects, recipe, ao_path, curvature_path, out_dir,
        atlas_px=DEFAULT_ATLAS_PX):
    """Writes and returns packed ``mask.png`` for a post-UV assembly."""
    assert_clean_path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ao = _load_raw_png(ao_path)
    curvature = _load_raw_png(curvature_path)
    if ao.shape[:2] != curvature.shape[:2]:
        raise ValueError(
            f"AO/curvature resolution mismatch: {ao.shape[:2]} vs "
            f"{curvature.shape[:2]}")
    height, width = ao.shape[:2]
    if (width, height) != (atlas_px, atlas_px):
        raise ValueError(
            f"AO/curvature resolution {(width, height)} != requested "
            f"{(atlas_px, atlas_px)}")

    ao_channel = np.clip(ao[:, :, 0], 0.0, 1.0)
    curvature_channel = np.clip(curvature[:, :, 0], 0.0, 1.0)
    edge_bias = 0.35 + 0.65 * (1.0 - ao_channel)
    wear = np.power(curvature_channel, 2.2) * edge_bias

    packed = np.empty((height, width, 4), dtype=np.float32)
    packed[:, :, 0] = ao_channel                         # R: copied AO
    packed[:, :, 1] = np.clip(wear, 0.0, 1.0)           # G: edge wear
    packed[:, :, 2] = SNOW_PADDING                       # B: UV padding
    packed[:, :, 3] = _ROUGHNESS["body"]                # A: padding
    occupied = np.zeros((height, width), dtype=bool)

    for name in sorted(objects):
        obj = objects[name]
        category = _part_category(name)
        base_roughness = _ROUGHNESS[category]
        wear_multiplier = _wear_multiplier(name)
        for (
                x, y, _height_values, up_values, _side,
                _longitudinal) in _iter_object_texels(obj, width, height):
            local_ao = ao_channel[y, x]
            local_wear = np.clip(wear[y, x] * wear_multiplier, 0.0, 1.0)
            packed[y, x, 1] = local_wear
            up = np.clip(up_values, 0.0, 1.0)
            packed[y, x, 2] = up * up * (1.0 - local_ao * 0.5)

            dirt = np.clip(1.0 - local_ao, 0.0, 1.0)
            noise = (_pixel_noise(x, y, recipe.seed, salt=29) - 0.5) * 0.10
            roughness = (
                base_roughness - 0.20 * local_wear + 0.15 * dirt + noise)
            packed[y, x, 3] = np.clip(roughness, 0.0, 1.0)
            occupied[y, x] = True

    # AO already carries Task 12's 16 px bake margin.  Dilate Task 13's
    # object-emphasized wear plus snow/roughness into the same padding.
    _dilate_unoccupied(
        packed, occupied, _dilation_radius(atlas_px), channels=(1, 2, 3))
    path = os.path.join(out_dir, MASK_FILENAME)
    _save_png(
        path, np.clip(packed, 0.0, 1.0), non_color=True,
        channel_packed=True)
    return path
