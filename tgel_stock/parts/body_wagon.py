"""Parametric wagon body part builders (PS-1b box wagon).

Space convention (controller-adjudicated, per Task 8 brief): all meshes are
authored Body-local, which is vehicle-root-local (the Body node sits at
identity under Visuals). Railhead is y=0; +Z is front (the A-end); the body
is symmetric about x=0 and z=0. The B-end (brake end) is -Z, where
``BrakeWheel`` is mounted.

Layout is anchored on the frozen wagon envelope (``recipe.width``,
``recipe.height``, ``recipe.body_length``) for anchors that scale with it:
the whole roof y-chain is derived from ``recipe.height`` inside
``build_meshes`` (walk top = height - 1e-3, peak = walk top - riser -
plank, eave = peak - fixed gable rise), and end-anchored parts derive from
``recipe.body_length``. Blockout literals (rib spacing, door/rail sizes,
plank/riser thicknesses, ladder/brake-wheel dimensions) stay module
constants per the brief.

Envelope guards (``_assert_envelope``, run at the end of ``build_meshes``):
overall width <= recipe.width/2 + 1e-4; every mesh's max_y <=
recipe.height + 1e-4; |z| <= recipe.body_length/2 + 1e-4 for every part
EXCEPT ``BrakeWheel``. ``BrakeWheel`` sits at z = -body_length/2 - 0.05 on
the B-end coupler face; that 0.05 m overhang is coupled-face clearance --
length_over_couplers (13.5128) leaves 0.05 m plus wheel/torus thickness of
slack beyond body_length (12.7508) -- not part of the body envelope proper,
so it is checked separately against ``recipe.length_over_couplers / 2``.

Documented design decisions (brief left these unspecified):
- ``Underframe`` length (12.45) is not given by the brief; it is set to
  match ``Floor``'s run length (12.45), since the floor plate physically
  rests on the underframe sills.
- ``EndSillFront``/``EndSillRear`` are inset 0.02 m from the exact
  body-length boundary (rather than sitting flush on it) to keep a safety
  margin against the z-envelope guard instead of relying on exact float
  equality.
- ``GabledRoof`` is built flush with ``BoxBody``'s side walls (no eave
  overhang): the side ribs already consume nearly the full width budget
  (BoxBody half-width 1.575 + rib depth 0.05 = 1.625, against the frozen
  half-width of 1.6256 -- 0.0006 m of slack), so the roof cannot afford any
  additional overhang without breaching the width guard.
- Interface count: the brief's running part-list totals 38 named parts by
  literal enumeration (4 chassis/shell parts + 16 side ribs + 6 door/rail
  parts + 3 roof parts + 2 ladders + 1 brake wheel + 2 end sills + 4 stirrup
  steps), not the "40 named parts" the plan text claims. This module and its
  test build/assert exactly those 38 keys; see the test's EXPECTED_KEYS set.
"""

import math

import bmesh
import mathutils

from .. import scene

# ---------------------------------------------------------------------------
# Underframe: 2.80 wide x 0.28 deep @ y 1.02, fish-belly taper toward ends.
# ---------------------------------------------------------------------------
UNDERFRAME_WIDTH = 2.80
UNDERFRAME_HEIGHT = 0.28
UNDERFRAME_Y = 1.02
UNDERFRAME_LENGTH = 12.45  # matches Floor's run length; see module docstring
UNDERFRAME_END_DEPTH_FACTOR = 0.5  # end stations taper to 50% of full depth
UNDERFRAME_STATIONS = (-1.0, -0.5, 0.0, 0.5, 1.0)  # t in [-1, 1] along length

# ---------------------------------------------------------------------------
# CentreSill / Floor: plain boxes, blockout literals.
# ---------------------------------------------------------------------------
CENTRE_SILL_Y = 0.78
CENTRE_SILL_SIZE = (0.42, 0.32, 12.30)

FLOOR_Y = 1.145
FLOOR_SIZE = (3.06, 0.05, 12.45)

# ---------------------------------------------------------------------------
# BoxBody: 3.15 wide x 2.85 tall @ y 2.595, len 12.40, inset side-panel bays
# (face inset 0.02) with beveled vertical corners (0.02).
# ---------------------------------------------------------------------------
BOX_BODY_WIDTH = 3.15
BOX_BODY_HEIGHT = 2.85
BOX_BODY_Y = 2.595
BOX_BODY_LENGTH = 12.40
BOX_BODY_INSET = 0.02
BOX_BODY_BEVEL = 0.02
BOX_BODY_PANEL_MARGIN_Y = 0.08
BOX_BODY_PANEL_MARGIN_Z = 0.05

RIB_Y0 = BOX_BODY_Y - BOX_BODY_HEIGHT / 2.0
RIB_Y1 = BOX_BODY_Y + BOX_BODY_HEIGHT / 2.0

# ---------------------------------------------------------------------------
# SideRibLeft00..07 / SideRibRight00..07: hat-profile ribs, 0.05x0.10
# footprint, full body height, at z = +/-{1.45, 2.85, 4.20, 5.55}.
# ---------------------------------------------------------------------------
RIB_Z_ANCHORS = (1.45, 2.85, 4.20, 5.55)
RIB_FOOTPRINT_DEPTH = 0.05  # outward from the wall face, in x
RIB_FOOTPRINT_WIDTH = 0.10  # along the wall, in z

# ---------------------------------------------------------------------------
# DoorLeft/DoorRight: raised panels with X-brace relief and a handle bar.
# DoorRail{Left,Right}{Top,Bottom}: plain rail boxes above/below the door.
# ---------------------------------------------------------------------------
DOOR_WIDTH = 2.45  # along z
DOOR_HEIGHT = 2.70  # along y
DOOR_Y_CENTRE = BOX_BODY_Y
DOOR_PANEL_THICKNESS = 0.03
# Brace/handle centre offsets are budgeted (offset + radius) against the
# same tight width-envelope slack SideRib uses (BoxBody half-width 1.575 +
# 0.05 <= recipe.width/2 = 1.6256, leaving only ~0.0007 m spare beyond the
# rib depth) -- these numbers include the rod radius so the true outward
# extent stays under that budget; see _assert_envelope.
DOOR_BRACE_OFFSET = 0.025
DOOR_BRACE_RADIUS = 0.018
DOOR_HANDLE_OFFSET = 0.030
DOOR_HANDLE_RADIUS = 0.014
DOOR_HANDLE_Z_INSET = 0.35
DOOR_HANDLE_HALF_HEIGHT = 0.25
ROD_SEGMENTS = 16

DOOR_RAIL_LENGTH = DOOR_WIDTH + 0.30
DOOR_RAIL_HEIGHT = 0.08
DOOR_RAIL_DEPTH = 0.04
DOOR_RAIL_GAP = 0.02

# ---------------------------------------------------------------------------
# GabledRoof + RoofWalk (0.60 wide plank run on 0.05 risers) + RoofRibs
# (9 transverse ridges in one mesh).
# ---------------------------------------------------------------------------
# The roof y-chain is DERIVED from recipe.height inside build_meshes (the
# blockout's roof-walk top sat exactly at the envelope height, so deriving
# restores intent): walk_top = recipe.height - ROOF_WALK_TOP_CLEARANCE;
# peak = walk_top - riser - plank; eave = peak - ROOF_GABLE_RISE. Only the
# pitch (rise) and x/z literals stay as constants.
ROOF_WALK_TOP_CLEARANCE = 1e-3  # keeps real headroom under the height guard
ROOF_GABLE_RISE = 0.28  # peak height above the eave line (fixed pitch)
ROOF_LENGTH = 12.40

ROOF_WALK_WIDTH = 0.60
ROOF_WALK_THICKNESS = 0.03
ROOF_WALK_RISER_HEIGHT = 0.05
ROOF_WALK_RISER_SIZE = (0.10, ROOF_WALK_RISER_HEIGHT, 0.08)
ROOF_WALK_LENGTH = 12.40
ROOF_WALK_RISER_COUNT = 6
ROOF_WALK_RISER_END_MARGIN = 0.5

ROOF_RIB_COUNT = 9
ROOF_RIB_HALF_WIDTH = 0.35
ROOF_RIB_BUMP = 0.035
ROOF_RIB_END_MARGIN = 0.3

# ---------------------------------------------------------------------------
# LadderFrontLeft / LadderRearRight: 2 stiles + 4 rungs, rod r 0.014.
# ---------------------------------------------------------------------------
LADDER_ROD_RADIUS = 0.014
LADDER_Y0 = 0.60
LADDER_Y1 = 4.00
LADDER_STILE_HALF_SPACING = 0.20
LADDER_OUTWARD_OFFSET = 0.03
LADDER_Z_INSET = 0.60
LADDER_RUNG_COUNT = 4

# ---------------------------------------------------------------------------
# BrakeWheel: 0.55-diameter spoked hand wheel (torus rim + 5 spokes + boss),
# mounted at y ~= 3.3, z = -body_length/2 - 0.05 (B-end).
# ---------------------------------------------------------------------------
BRAKE_WHEEL_DIAMETER = 0.55
BRAKE_WHEEL_TUBE_RADIUS = 0.02
BRAKE_WHEEL_MAIN_SEGMENTS = 32
BRAKE_WHEEL_TUBE_SEGMENTS = 12
BRAKE_WHEEL_SPOKE_RADIUS = 0.015
BRAKE_WHEEL_SPOKE_COUNT = 5
BRAKE_WHEEL_BOSS_RADIUS = 0.06
BRAKE_WHEEL_BOSS_DEPTH = 0.05
BRAKE_WHEEL_Y = 3.3
BRAKE_WHEEL_Z_OVERHANG = 0.05  # beyond -body_length/2; see module docstring

# ---------------------------------------------------------------------------
# EndSillFront/EndSillRear: heavy buffer-beam boxes at the underframe ends.
# ---------------------------------------------------------------------------
END_SILL_SIZE = (2.80, 0.30, 0.25)
END_SILL_Y = 1.02
END_SILL_MARGIN = 0.02  # inset from the exact z envelope boundary

# ---------------------------------------------------------------------------
# StirrupStep{FrontLeft,FrontRight,RearLeft,RearRight}: U-shaped steps under
# the corners.  The hanger centres continue the ladder stile spacing and the
# hanger tops meet the body/floor seam; a tread-only box would visibly float
# above railhead with no structural connection to the wagon.
# ---------------------------------------------------------------------------
STIRRUP_STEP_DEPTH = 0.04
STIRRUP_STEP_TREAD_HEIGHT = 0.06
STIRRUP_STEP_X_OFFSET = 0.02
STIRRUP_STEP_Y = 0.25
STIRRUP_STEP_HANGER_WIDTH = 0.04
STIRRUP_STEP_HALF_SPAN = (
    LADDER_STILE_HALF_SPACING + STIRRUP_STEP_HANGER_WIDTH / 2.0)
STIRRUP_STEP_TOP_Y = RIB_Y0
STIRRUP_STEP_Z_INSET = LADDER_Z_INSET


def _finalize(bm):
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    bmesh.ops.holes_fill(bm, edges=bm.edges[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()


def _box_bmesh(bm, centre_unity, size_unity):
    """Appends an axis-aligned box (vehicle space) into an existing bm."""
    cx, cy, cz = centre_unity
    sx, sy, sz = (s * 0.5 for s in size_unity)
    verts = []
    for dx in (-sx, sx):
        for dy in (-sy, sy):
            for dz in (-sz, sz):
                verts.append(bm.verts.new(scene.to_blender((cx + dx, cy + dy, cz + dz))))
    bm.verts.ensure_lookup_table()
    faces = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    for face in faces:
        bm.faces.new([verts[i] for i in face])


def _append_capsule_segment(bm, p0, p1, radius, segments):
    """Appends a closed cylinder between two arbitrary vehicle-space points."""
    v0 = mathutils.Vector(p0)
    v1 = mathutils.Vector(p1)
    axis = v1 - v0
    length = axis.length
    if length < 1e-9:
        return
    direction = axis / length
    reference = mathutils.Vector((0.0, 1.0, 0.0))
    if abs(direction.dot(reference)) > 0.999:
        reference = mathutils.Vector((1.0, 0.0, 0.0))
    e1 = direction.cross(reference).normalized()
    e2 = direction.cross(e1).normalized()

    ring_a = []
    ring_b = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        offset = e1 * (radius * math.cos(theta)) + e2 * (radius * math.sin(theta))
        ring_a.append(bm.verts.new(scene.to_blender(tuple(v0 + offset))))
        ring_b.append(bm.verts.new(scene.to_blender(tuple(v1 + offset))))
    bm.verts.ensure_lookup_table()
    for i in range(segments):
        j = (i + 1) % segments
        bm.faces.new((ring_a[i], ring_a[j], ring_b[j], ring_b[i]))
    bm.faces.new(ring_a)
    bm.faces.new(tuple(reversed(ring_b)))


def _append_disc_cylinder(bm, centre_unity, radius, height, axis, segments):
    """Adds a closed cylinder into bm, centred at centre_unity.

    ``axis`` selects which vehicle-space axis the cylinder's length runs
    along ('x', 'y' or 'z').
    """
    cx, cy, cz = centre_unity
    half = height * 0.5
    ring_a = []
    ring_b = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        c = radius * math.cos(theta)
        s = radius * math.sin(theta)
        if axis == "y":
            pa = (cx + c, cy - half, cz + s)
            pb = (cx + c, cy + half, cz + s)
        elif axis == "x":
            pa = (cx - half, cy + c, cz + s)
            pb = (cx + half, cy + c, cz + s)
        else:  # axis == "z"
            pa = (cx + c, cy + s, cz - half)
            pb = (cx + c, cy + s, cz + half)
        ring_a.append(bm.verts.new(scene.to_blender(pa)))
        ring_b.append(bm.verts.new(scene.to_blender(pb)))
    bm.verts.ensure_lookup_table()
    for i in range(segments):
        j = (i + 1) % segments
        bm.faces.new((ring_a[i], ring_a[j], ring_b[j], ring_b[i]))
    bm.faces.new(ring_a)
    bm.faces.new(tuple(reversed(ring_b)))


def _append_prism(bm, profile, axis, lo, hi, offset=(0.0, 0.0, 0.0)):
    """Extrudes a closed 2D polygon along a vehicle-space axis, into bm.

    ``profile`` is a list of (p, q) pairs in the plane perpendicular to
    ``axis``: for axis='y' the pair maps to (x, z); for axis='z' it maps to
    (x, y); for axis='x' it maps to (y, z). ``offset`` is a vehicle-space
    translation applied to the two perpendicular coordinates.
    """
    ox, oy, oz = offset
    ring_lo = []
    ring_hi = []
    for p, q in profile:
        if axis == "y":
            lo_pt = (p + ox, lo, q + oz)
            hi_pt = (p + ox, hi, q + oz)
        elif axis == "z":
            lo_pt = (p + ox, q + oy, lo)
            hi_pt = (p + ox, q + oy, hi)
        else:  # axis == "x"
            lo_pt = (lo, p + oy, q + oz)
            hi_pt = (hi, p + oy, q + oz)
        ring_lo.append(bm.verts.new(scene.to_blender(lo_pt)))
        ring_hi.append(bm.verts.new(scene.to_blender(hi_pt)))
    bm.verts.ensure_lookup_table()
    n = len(profile)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((ring_lo[i], ring_lo[j], ring_hi[j], ring_hi[i]))
    bm.faces.new(ring_lo)
    bm.faces.new(tuple(reversed(ring_hi)))


def _append_torus(bm, centre_unity, axis_unity, big_radius, small_radius, main_segments, tube_segments):
    """Adds a closed torus into bm via the same revolve technique as
    wheel.py's tread profile: a small closed tube-circle, positioned
    ``big_radius`` away from the axis in the half-plane spanned by a
    perpendicular direction and the axis itself, spun 360 degrees around
    the axis through ``centre_unity``.
    """
    cx, cy, cz = centre_unity
    axis = mathutils.Vector(axis_unity).normalized()
    reference = mathutils.Vector((0.0, 1.0, 0.0))
    if abs(axis.dot(reference)) > 0.999:
        reference = mathutils.Vector((1.0, 0.0, 0.0))
    e1 = axis.cross(reference).normalized()

    centre = mathutils.Vector((cx, cy, cz))
    profile_verts = []
    for j in range(tube_segments):
        phi = 2.0 * math.pi * j / tube_segments
        offset = e1 * (big_radius + small_radius * math.cos(phi)) + axis * (small_radius * math.sin(phi))
        point = centre + offset
        profile_verts.append(bm.verts.new(scene.to_blender(tuple(point))))
    bm.verts.ensure_lookup_table()
    profile_edges = [
        bm.edges.new((profile_verts[i], profile_verts[(i + 1) % tube_segments]))
        for i in range(tube_segments)
    ]
    spin_axis = scene.to_blender(tuple(axis))
    spin_cent = scene.to_blender((cx, cy, cz))
    bmesh.ops.spin(
        bm, geom=profile_verts + profile_edges, cent=spin_cent, axis=spin_axis,
        angle=math.radians(360.0), steps=main_segments, use_duplicate=False)


def _rib_z_positions():
    return sorted([-z for z in RIB_Z_ANCHORS] + list(RIB_Z_ANCHORS))


def _box_body_bay_bounds(hz):
    return [-hz] + _rib_z_positions() + [hz]


def _build_underframe():
    """5-station loft with a rectangular cross-section that tapers from
    full depth at the centre to half depth at the ends -- a visible
    fish-belly silhouette, not a plain box.
    """
    bm = bmesh.new()
    hx = UNDERFRAME_WIDTH / 2.0
    half_len = UNDERFRAME_LENGTH / 2.0
    y_top = UNDERFRAME_Y + UNDERFRAME_HEIGHT / 2.0

    rings = []
    for t in UNDERFRAME_STATIONS:
        z = t * half_len
        depth_factor = UNDERFRAME_END_DEPTH_FACTOR + (1.0 - UNDERFRAME_END_DEPTH_FACTOR) * (1.0 - abs(t))
        depth = UNDERFRAME_HEIGHT * depth_factor
        y_bottom = y_top - depth
        ring = [
            bm.verts.new(scene.to_blender((-hx, y_bottom, z))),
            bm.verts.new(scene.to_blender((hx, y_bottom, z))),
            bm.verts.new(scene.to_blender((hx, y_top, z))),
            bm.verts.new(scene.to_blender((-hx, y_top, z))),
        ]
        rings.append(ring)
    bm.verts.ensure_lookup_table()

    for i in range(len(rings) - 1):
        a, b = rings[i], rings[i + 1]
        for k in range(4):
            j = (k + 1) % 4
            bm.faces.new((a[k], a[j], b[j], b[k]))
    bm.faces.new(tuple(reversed(rings[0])))
    bm.faces.new(rings[-1])

    _finalize(bm)
    return scene.mesh_object("Underframe", bm)


def _append_side_wall_bays(bm, x_face, inward_sign, y_top, y_bottom, bounds):
    """Appends N inset bay panels (a picture-frame ring + recessed floor
    per bay) covering one BoxBody side wall face, following the hand-rolled
    pocket construction used for the bogie side frames (bmesh.ops
    inset_individual proved unreliable under remove_doubles there).
    """
    for i in range(len(bounds) - 1):
        z0, z1 = bounds[i], bounds[i + 1]
        outer = [
            bm.verts.new(scene.to_blender((x_face, y_bottom, z0))),
            bm.verts.new(scene.to_blender((x_face, y_bottom, z1))),
            bm.verts.new(scene.to_blender((x_face, y_top, z1))),
            bm.verts.new(scene.to_blender((x_face, y_top, z0))),
        ]
        pocket_x = x_face + inward_sign * BOX_BODY_INSET
        my = min(BOX_BODY_PANEL_MARGIN_Y, (y_top - y_bottom) * 0.5 - 1e-4)
        mz = min(BOX_BODY_PANEL_MARGIN_Z, (z1 - z0) * 0.5 - 1e-4)
        inner = [
            bm.verts.new(scene.to_blender((pocket_x, y_bottom + my, z0 + mz))),
            bm.verts.new(scene.to_blender((pocket_x, y_bottom + my, z1 - mz))),
            bm.verts.new(scene.to_blender((pocket_x, y_top - my, z1 - mz))),
            bm.verts.new(scene.to_blender((pocket_x, y_top - my, z0 + mz))),
        ]
        bm.verts.ensure_lookup_table()
        for k in range(4):
            j = (k + 1) % 4
            bm.faces.new((outer[k], outer[j], inner[j], inner[k]))
        bm.faces.new(inner)


def _find_vertical_corner_edges(bm, hx, y_bottom, y_top, hz):
    """Locates the 4 full-height vertical edges at the box's extreme
    (+/-hx, +/-hz) corners, by vertex position (post remove_doubles, each
    corner column collapses to a single edge shared by the end wall and the
    outermost bay panel).
    """
    targets = [(sx * hx, sz * hz) for sx in (-1.0, 1.0) for sz in (-1.0, 1.0)]
    found = []
    for edge in bm.edges:
        p0 = scene.to_unity(edge.verts[0].co)
        p1 = scene.to_unity(edge.verts[1].co)
        if abs(p0[0] - p1[0]) > 1e-6 or abs(p0[2] - p1[2]) > 1e-6:
            continue
        for tx, tz in targets:
            if abs(p0[0] - tx) < 1e-4 and abs(p0[2] - tz) < 1e-4:
                ys = sorted((p0[1], p1[1]))
                if abs(ys[0] - y_bottom) < 1e-4 and abs(ys[1] - y_top) < 1e-4:
                    found.append(edge)
                break
    return found


def _build_box_body():
    hx = BOX_BODY_WIDTH / 2.0
    hz = BOX_BODY_LENGTH / 2.0
    y_top = BOX_BODY_Y + BOX_BODY_HEIGHT / 2.0
    y_bottom = BOX_BODY_Y - BOX_BODY_HEIGHT / 2.0
    bounds = _box_body_bay_bounds(hz)

    bm = bmesh.new()
    _append_side_wall_bays(bm, -hx, 1.0, y_top, y_bottom, bounds)
    _append_side_wall_bays(bm, hx, -1.0, y_top, y_bottom, bounds)

    for z in (hz, -hz):
        corners = [
            bm.verts.new(scene.to_blender((-hx, y_bottom, z))),
            bm.verts.new(scene.to_blender((hx, y_bottom, z))),
            bm.verts.new(scene.to_blender((hx, y_top, z))),
            bm.verts.new(scene.to_blender((-hx, y_top, z))),
        ]
        bm.verts.ensure_lookup_table()
        bm.faces.new(corners)

    # Built as a z-segmented strip (matching `bounds`) rather than one quad:
    # the side-wall bay panels above subdivide their top/bottom boundary
    # edge into 9 segments at the same z positions, and a single
    # unsegmented cap quad would leave that boundary line a mismatched
    # 1-edge-vs-9-edges seam (a boundary edge on each side, patched by
    # holes_fill into a degenerate zero-area sliver face).
    for y in (y_top, y_bottom):
        for i in range(len(bounds) - 1):
            z0, z1 = bounds[i], bounds[i + 1]
            corners = [
                bm.verts.new(scene.to_blender((-hx, y, z0))),
                bm.verts.new(scene.to_blender((hx, y, z0))),
                bm.verts.new(scene.to_blender((hx, y, z1))),
                bm.verts.new(scene.to_blender((-hx, y, z1))),
            ]
            bm.verts.ensure_lookup_table()
            bm.faces.new(corners)

    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    bmesh.ops.holes_fill(bm, edges=bm.edges[:])

    bm.edges.ensure_lookup_table()
    corner_edges = _find_vertical_corner_edges(bm, hx, y_bottom, y_top, hz)
    assert len(corner_edges) == 4, f"BoxBody: expected 4 corner edges, found {len(corner_edges)}"
    bmesh.ops.bevel(
        bm, geom=corner_edges, offset=BOX_BODY_BEVEL, offset_type="OFFSET",
        segments=2, affect="EDGES")

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()
    return scene.mesh_object("BoxBody", bm)


def _hat_profile_local():
    """Closed hat-channel hexagon: (v, u) pairs where v is the outward
    offset from the wall in [0, RIB_FOOTPRINT_DEPTH] and u is the
    tangential offset along the wall in [-half_w, half_w]. Base flanges sit
    flush at v=0; the crown rises to v=RIB_FOOTPRINT_DEPTH.
    """
    half_w = RIB_FOOTPRINT_WIDTH / 2.0
    crown_w = half_w * 0.4
    flange_depth = RIB_FOOTPRINT_DEPTH * 0.3
    crown_depth = RIB_FOOTPRINT_DEPTH
    return (
        (0.0, -half_w),
        (flange_depth, -half_w),
        (crown_depth, -crown_w),
        (crown_depth, crown_w),
        (flange_depth, half_w),
        (0.0, half_w),
    )


def _build_side_rib(name, x_face, outward_sign, z_centre, y0, y1):
    bm = bmesh.new()
    profile = [(outward_sign * v, u) for v, u in _hat_profile_local()]
    _append_prism(bm, profile, axis="y", lo=y0, hi=y1, offset=(x_face, 0.0, z_centre))
    _finalize(bm)
    return scene.mesh_object(name, bm)


def _build_door(name, x_face, outward_sign):
    """Raised door panel: a shallow slab plus an X-brace (two diagonal rod
    segments) and a horizontal handle bar, all merged into one mesh (the
    same spatially-overlapping-shells composition used by bogie.py's
    Bolster/Axlebox parts).
    """
    bm = bmesh.new()
    hz = DOOR_WIDTH / 2.0
    hy = DOOR_HEIGHT / 2.0

    panel_centre_x = x_face + outward_sign * (DOOR_PANEL_THICKNESS / 2.0)
    _box_bmesh(bm, (panel_centre_x, DOOR_Y_CENTRE, 0.0), (DOOR_PANEL_THICKNESS, DOOR_HEIGHT, DOOR_WIDTH))

    brace_x = x_face + outward_sign * DOOR_BRACE_OFFSET
    bottom_left = (brace_x, DOOR_Y_CENTRE - hy, -hz)
    bottom_right = (brace_x, DOOR_Y_CENTRE - hy, hz)
    top_left = (brace_x, DOOR_Y_CENTRE + hy, -hz)
    top_right = (brace_x, DOOR_Y_CENTRE + hy, hz)
    _append_capsule_segment(bm, bottom_left, top_right, DOOR_BRACE_RADIUS, ROD_SEGMENTS)
    _append_capsule_segment(bm, bottom_right, top_left, DOOR_BRACE_RADIUS, ROD_SEGMENTS)

    handle_x = x_face + outward_sign * DOOR_HANDLE_OFFSET
    handle_z = hz - DOOR_HANDLE_Z_INSET
    _append_capsule_segment(
        bm,
        (handle_x, DOOR_Y_CENTRE - DOOR_HANDLE_HALF_HEIGHT, handle_z),
        (handle_x, DOOR_Y_CENTRE + DOOR_HANDLE_HALF_HEIGHT, handle_z),
        DOOR_HANDLE_RADIUS, ROD_SEGMENTS)

    _finalize(bm)
    return scene.mesh_object(name, bm)


def _door_rail_top_y():
    return DOOR_Y_CENTRE + DOOR_HEIGHT / 2.0 + DOOR_RAIL_GAP + DOOR_RAIL_HEIGHT / 2.0


def _door_rail_bottom_y():
    return DOOR_Y_CENTRE - DOOR_HEIGHT / 2.0 - DOOR_RAIL_GAP - DOOR_RAIL_HEIGHT / 2.0


def _build_door_rail(name, x_face, outward_sign, y_centre):
    centre_x = x_face + outward_sign * (DOOR_RAIL_DEPTH / 2.0)
    return scene.box(name, (centre_x, y_centre, 0.0), (DOOR_RAIL_DEPTH, DOOR_RAIL_HEIGHT, DOOR_RAIL_LENGTH))


def _build_gabled_roof(peak_y, eave_y, box_top_y):
    """Gable prism with a derived peak height (see build_meshes).

    When the derived eave line sits above the BoxBody top (the normal case:
    the eave follows the peak down by the fixed ROOF_GABLE_RISE while the
    box top is a blockout literal), the profile grows a short vertical
    fascia band from the box top up to the eave so the roof stays seated on
    the body instead of floating above it.
    """
    bm = bmesh.new()
    hx = BOX_BODY_WIDTH / 2.0
    if eave_y - box_top_y > 1e-6:
        profile = (
            (-hx, box_top_y), (-hx, eave_y), (0.0, peak_y),
            (hx, eave_y), (hx, box_top_y),
        )
    else:
        profile = ((-hx, eave_y), (0.0, peak_y), (hx, eave_y))
    _append_prism(bm, profile, axis="z", lo=-ROOF_LENGTH / 2.0, hi=ROOF_LENGTH / 2.0)
    _finalize(bm)
    return scene.mesh_object("GabledRoof", bm)


def _build_roof_walk(peak_y):
    bm = bmesh.new()
    plank_y = peak_y + ROOF_WALK_RISER_HEIGHT + ROOF_WALK_THICKNESS / 2.0
    _box_bmesh(bm, (0.0, plank_y, 0.0), (ROOF_WALK_WIDTH, ROOF_WALK_THICKNESS, ROOF_WALK_LENGTH))

    riser_y = peak_y + ROOF_WALK_RISER_HEIGHT / 2.0
    half_span = ROOF_WALK_LENGTH / 2.0 - ROOF_WALK_RISER_END_MARGIN
    for i in range(ROOF_WALK_RISER_COUNT):
        t = i / (ROOF_WALK_RISER_COUNT - 1)
        z = -half_span + t * (2.0 * half_span)
        _box_bmesh(bm, (0.0, riser_y, z), ROOF_WALK_RISER_SIZE)

    _finalize(bm)
    return scene.mesh_object("RoofWalk", bm)


def _build_roof_ribs(peak_y):
    bm = bmesh.new()
    profile = ((0.0, -ROOF_RIB_BUMP), (ROOF_RIB_BUMP, 0.0), (0.0, ROOF_RIB_BUMP))
    half_span = ROOF_WALK_LENGTH / 2.0 - ROOF_RIB_END_MARGIN
    for i in range(ROOF_RIB_COUNT):
        t = i / (ROOF_RIB_COUNT - 1)
        z = -half_span + t * (2.0 * half_span)
        _append_prism(
            bm, profile, axis="x", lo=-ROOF_RIB_HALF_WIDTH, hi=ROOF_RIB_HALF_WIDTH,
            offset=(0.0, peak_y, z))
    _finalize(bm)
    return scene.mesh_object("RoofRibs", bm)


def _build_ladder(name, x_face, outward_sign, z_centre):
    bm = bmesh.new()
    stile_x = x_face + outward_sign * LADDER_OUTWARD_OFFSET

    for dz in (-LADDER_STILE_HALF_SPACING, LADDER_STILE_HALF_SPACING):
        _append_capsule_segment(
            bm, (stile_x, LADDER_Y0, z_centre + dz), (stile_x, LADDER_Y1, z_centre + dz),
            LADDER_ROD_RADIUS, ROD_SEGMENTS)

    for i in range(LADDER_RUNG_COUNT):
        t = (i + 1) / (LADDER_RUNG_COUNT + 1)
        y = LADDER_Y0 + t * (LADDER_Y1 - LADDER_Y0)
        _append_capsule_segment(
            bm, (stile_x, y, z_centre - LADDER_STILE_HALF_SPACING),
            (stile_x, y, z_centre + LADDER_STILE_HALF_SPACING),
            LADDER_ROD_RADIUS, ROD_SEGMENTS)

    _finalize(bm)
    return scene.mesh_object(name, bm)


def _build_brake_wheel(z_centre):
    bm = bmesh.new()
    centre = (0.0, BRAKE_WHEEL_Y, z_centre)
    big_radius = BRAKE_WHEEL_DIAMETER / 2.0

    _append_torus(
        bm, centre, (0.0, 0.0, 1.0), big_radius, BRAKE_WHEEL_TUBE_RADIUS,
        BRAKE_WHEEL_MAIN_SEGMENTS, BRAKE_WHEEL_TUBE_SEGMENTS)

    for i in range(BRAKE_WHEEL_SPOKE_COUNT):
        theta = 2.0 * math.pi * i / BRAKE_WHEEL_SPOKE_COUNT
        rim_point = (
            centre[0] + big_radius * math.cos(theta),
            centre[1] + big_radius * math.sin(theta),
            centre[2],
        )
        _append_capsule_segment(bm, centre, rim_point, BRAKE_WHEEL_SPOKE_RADIUS, 12)

    _append_disc_cylinder(bm, centre, BRAKE_WHEEL_BOSS_RADIUS, BRAKE_WHEEL_BOSS_DEPTH, "z", 16)

    _finalize(bm)
    return scene.mesh_object("BrakeWheel", bm)


def _build_end_sill(name, z_sign, half_length):
    outer_face = half_length - END_SILL_MARGIN
    centre_z = z_sign * (outer_face - END_SILL_SIZE[2] / 2.0)
    return scene.box(name, (0.0, END_SILL_Y, centre_z), END_SILL_SIZE)


def _build_stirrup_step(name, x_sign, z_sign, hx, half_length):
    centre_x = x_sign * (hx + STIRRUP_STEP_X_OFFSET)
    centre_z = z_sign * (half_length - STIRRUP_STEP_Z_INSET)
    x0 = centre_x - STIRRUP_STEP_DEPTH / 2.0
    x1 = centre_x + STIRRUP_STEP_DEPTH / 2.0
    tread_bottom = STIRRUP_STEP_Y - STIRRUP_STEP_TREAD_HEIGHT / 2.0
    tread_top = STIRRUP_STEP_Y + STIRRUP_STEP_TREAD_HEIGHT / 2.0
    outer_z = STIRRUP_STEP_HALF_SPAN
    inner_z = outer_z - STIRRUP_STEP_HANGER_WIDTH

    # One closed concave profile, extruded through X, produces a single
    # manifold U-frame: two hangers and the tread.  The hanger centres are at
    # +/-LADDER_STILE_HALF_SPACING, so the two laddered corners read as one
    # continuous access assembly instead of a loose block on the ground.
    profile = (
        (tread_bottom, -outer_z),
        (STIRRUP_STEP_TOP_Y, -outer_z),
        (STIRRUP_STEP_TOP_Y, -inner_z),
        (tread_top, -inner_z),
        (tread_top, inner_z),
        (STIRRUP_STEP_TOP_Y, inner_z),
        (STIRRUP_STEP_TOP_Y, outer_z),
        (tread_bottom, outer_z),
    )
    bm = bmesh.new()
    _append_prism(
        bm,
        profile,
        axis="x",
        lo=min(x0, x1),
        hi=max(x0, x1),
        offset=(0.0, 0.0, centre_z))
    _finalize(bm)
    return scene.mesh_object(name, bm)


def _assert_envelope(meshes, recipe):
    """Guards the produced meshes against the frozen wagon envelope.

    BrakeWheel is intentionally excluded from the width/height/length
    checks below and checked separately against length_over_couplers: see
    the module docstring for why its 0.05 m B-end overhang is coupled-face
    clearance, not body length.
    """
    width_limit = recipe.width / 2.0 + 1e-4
    height_limit = recipe.height + 1e-4
    length_limit = recipe.body_length / 2.0 + 1e-4

    for name, obj in meshes.items():
        _p, _n, _u, _t, bounds_min, bounds_max = scene.collect_mesh_data(obj)
        max_abs_x = max(abs(bounds_min[0]), abs(bounds_max[0]))
        assert max_abs_x <= width_limit, f"{name} exceeds width envelope: |x|={max_abs_x}"
        assert bounds_max[1] <= height_limit, f"{name} exceeds height envelope: y={bounds_max[1]}"
        if name == "BrakeWheel":
            continue
        max_abs_z = max(abs(bounds_min[2]), abs(bounds_max[2]))
        assert max_abs_z <= length_limit, f"{name} exceeds body-length envelope: |z|={max_abs_z}"

    _p, _n, _u, _t, bw_min, bw_max = scene.collect_mesh_data(meshes["BrakeWheel"])
    brake_limit = recipe.length_over_couplers / 2.0 + 1e-4
    max_abs_z = max(abs(bw_min[2]), abs(bw_max[2]))
    assert max_abs_z <= brake_limit, f"BrakeWheel exceeds coupled-face clearance: |z|={max_abs_z}"


def build_meshes(recipe):
    """38 Body-local meshes for one PS-1b box-wagon body.

    See the module docstring for the space convention, the envelope guards
    applied before returning, and the disclosed 38-vs-"40"-named-parts
    count.
    """
    half_length = recipe.body_length / 2.0
    hx = BOX_BODY_WIDTH / 2.0

    meshes = {}

    meshes["Underframe"] = _build_underframe()
    meshes["CentreSill"] = scene.box("CentreSill", (0.0, CENTRE_SILL_Y, 0.0), CENTRE_SILL_SIZE)
    meshes["Floor"] = scene.box("Floor", (0.0, FLOOR_Y, 0.0), FLOOR_SIZE)
    meshes["BoxBody"] = _build_box_body()

    for i, z in enumerate(_rib_z_positions()):
        meshes[f"SideRibLeft{i:02d}"] = _build_side_rib(
            f"SideRibLeft{i:02d}", -hx, -1.0, z, RIB_Y0, RIB_Y1)
        meshes[f"SideRibRight{i:02d}"] = _build_side_rib(
            f"SideRibRight{i:02d}", hx, 1.0, z, RIB_Y0, RIB_Y1)

    meshes["DoorLeft"] = _build_door("DoorLeft", -hx, -1.0)
    meshes["DoorRight"] = _build_door("DoorRight", hx, 1.0)
    meshes["DoorRailLeftTop"] = _build_door_rail("DoorRailLeftTop", -hx, -1.0, _door_rail_top_y())
    meshes["DoorRailLeftBottom"] = _build_door_rail("DoorRailLeftBottom", -hx, -1.0, _door_rail_bottom_y())
    meshes["DoorRailRightTop"] = _build_door_rail("DoorRailRightTop", hx, 1.0, _door_rail_top_y())
    meshes["DoorRailRightBottom"] = _build_door_rail(
        "DoorRailRightBottom", hx, 1.0, _door_rail_bottom_y())

    # Roof y-chain, derived from the recipe envelope (the blockout's
    # roof-walk top sat exactly at height 4.4196, so this restores intent):
    # walk top a hair under the envelope, then peak and eave follow down.
    roof_walk_top = recipe.height - ROOF_WALK_TOP_CLEARANCE
    roof_peak_y = roof_walk_top - ROOF_WALK_RISER_HEIGHT - ROOF_WALK_THICKNESS
    roof_eave_y = roof_peak_y - ROOF_GABLE_RISE
    box_top_y = BOX_BODY_Y + BOX_BODY_HEIGHT / 2.0

    meshes["GabledRoof"] = _build_gabled_roof(roof_peak_y, roof_eave_y, box_top_y)
    meshes["RoofWalk"] = _build_roof_walk(roof_peak_y)
    meshes["RoofRibs"] = _build_roof_ribs(roof_peak_y)

    front_ladder_z = half_length - LADDER_Z_INSET
    meshes["LadderFrontLeft"] = _build_ladder("LadderFrontLeft", -hx, -1.0, front_ladder_z)
    meshes["LadderRearRight"] = _build_ladder("LadderRearRight", hx, 1.0, -front_ladder_z)

    brake_z = -(half_length + BRAKE_WHEEL_Z_OVERHANG)
    meshes["BrakeWheel"] = _build_brake_wheel(brake_z)

    meshes["EndSillFront"] = _build_end_sill("EndSillFront", 1.0, half_length)
    meshes["EndSillRear"] = _build_end_sill("EndSillRear", -1.0, half_length)

    for x_sign, z_sign, suffix in (
        (-1.0, 1.0, "FrontLeft"), (1.0, 1.0, "FrontRight"),
        (-1.0, -1.0, "RearLeft"), (1.0, -1.0, "RearRight"),
    ):
        meshes[f"StirrupStep{suffix}"] = _build_stirrup_step(
            f"StirrupStep{suffix}", x_sign, z_sign, hx, half_length)

    _assert_envelope(meshes, recipe)
    return meshes
