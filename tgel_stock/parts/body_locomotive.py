"""Parametric locomotive body part builders (GP9-envelope diesel).

Space convention (controller-adjudicated, per Task 9 brief): all meshes are
authored Body-local, which is vehicle-root local (the Body node sits at
identity under Visuals), matching body_wagon.py's convention. Railhead is
y=0; +Z is front (the short-hood/cab-forward end); the body is symmetric
about x=0 except where explicitly noted (none of these parts are
x-asymmetric).

Layout chain, derived from ``recipe`` inside ``build_meshes``: half_length =
recipe.body_length / 2. Frame length is pinned to recipe.body_length exactly
and Walkway length to recipe.body_length - 1.5 (a fixed end-clearance
margin) -- both reproduce the blockout's literal 16.0 / 14.5 when
recipe.body_length = 16.0. ShortHood and LongHood z-centres are derived from
half_length via fixed end-clearance margins (SHORT_HOOD_FRONT_MARGIN,
LONG_HOOD_REAR_MARGIN); Cab's z-centre and z-length are then derived as the
gap between LongHood's front face and ShortHood's rear face. All three
derivations reproduce the blockout's literal anchors (4.90 / -2.80 / 2.05)
exactly at the frozen recipe. CabRoof's crowned peak is derived from
recipe.height (see ``_build_cab_roof``), pinned to +/-0.01 by the test.
Frame/Walkway widths use ``recipe.width`` directly (not a module constant)
since both sit exactly at the width envelope boundary.

Envelope guards (``_assert_envelope``, run at the end of ``build_meshes``):
overall |x| <= recipe.width/2 + 1e-4 for every part INCLUDING the
handrails; every mesh's max_y <= recipe.height + 1e-4 INCLUDING the horn;
|z| <= recipe.body_length/2 + 1e-4 for every part. Unlike body_wagon.py
there is no BrakeWheel-style exemption here: every part derives its anchors
to fit inside the plain envelope.

Documented design decisions (brief left these unspecified or in conflict):
- Part count: the brief's running prose says "20 parts", but its own
  literal enumeration counts to 21 (see this module's build_meshes
  docstring and the test's EXPECTED_KEYS comment) -- the same kind of
  prose-vs-enumeration mismatch documented in body_wagon.py ("40" vs the
  actual 38).
- FuelTank: the blockout anchor (centre y 0.665, height 0.73) would put the
  tank's underside at y=0.30, below the 0.4 m per-mesh floor the task
  instructions impose on every part. The tank's top surface is instead
  pinned just under the Frame's underside (FUEL_TANK_TOP_Y = 1.02, versus
  Frame's own underside at 1.04) and its height is trimmed so the underside
  sits at FUEL_TANK_BOTTOM_Y = 0.42 (comfortably above the 0.4 floor)
  instead of the blockout's 0.30.
- Horn: the blockout anchor's y (4.45) already exceeds recipe.height
  (4.4196) before any trumpet geometry is added, and the brief itself flags
  "keep below height limit" as a caveat. The horn cluster's base is instead
  seated on the (derived) CabRoof peak line and its trumpets are built low
  and shallow so the tallest point stays at recipe.height - 0.02.
- RadiatorFanFront/Rear and ExhaustStack keep the blockout's literal y
  (3.59 / 3.79): both already sit at or below LongHood's derived top surface
  (LONG_HOOD_Y + LONG_HOOD_HEIGHT/2 = 3.55), so neither floats above the
  hood; ExhaustStack's is additionally re-derived here as hood_top_y +
  height/2 (an exact match to the literal) to make that seating explicit.
- CabGlass is authored as 4 single-sided flat quads (glass panes), not
  boxes: this is a deliberate, disclosed exception to the manifold check
  (a flat single-layer pane has boundary-only edges by construction) that
  mirrors body_wagon.py's disclosed BrakeWheel exemption from the
  body-length envelope check; see the test file's MANIFOLD_EXEMPT comment.
- LongHood's "5 inset door panels ... with hinge lines" narrative detail is
  built as 5 genuinely recessed picture-frame bays (matching body_wagon's
  BoxBody bay technique) but the separate "hinge line" rods called out in
  the fuller narrative are omitted to keep this already-the-richest task's
  scope bounded; the louvers, top bevel, fans, exhaust and door bays already
  give LongHood far more than a plain-box silhouette.
"""

import math

import bmesh
import mathutils

from .. import scene

# ---------------------------------------------------------------------------
# Frame: full-width structural sill, 0.28 deep @ y 1.18, length = body_length
# (derived), with a 2-step pilot relief at both ends (staircase profile, not
# a plain slab end).
# ---------------------------------------------------------------------------
FRAME_Y = 1.18
FRAME_HEIGHT = 0.28
FRAME_PILOT_STEP_LENGTH = 0.3  # two steps of this length = 0.6 m pilot run
FRAME_PILOT_STEP_DROP = 0.06  # y-drop per step (two steps => 0.12 total)

# ---------------------------------------------------------------------------
# Walkway: full-width running board @ y 1.37, z-offset -0.05, length =
# body_length - 1.5 (derived), with a thin raised edge lip on both sides.
# ---------------------------------------------------------------------------
WALKWAY_Y = 1.37
WALKWAY_HEIGHT = 0.10
WALKWAY_Z_OFFSET = -0.05
WALKWAY_LENGTH_MARGIN = 1.5
WALKWAY_LIP_WIDTH = 0.02
WALKWAY_LIP_HEIGHT = 0.03
WALKWAY_LIP_X_EPSILON = 0.004  # keeps the lip's outer face off the walkway's own edge
WALKWAY_LIP_Z_EPSILON = 0.01  # keeps the lip's ends off the walkway's own end edges

# ---------------------------------------------------------------------------
# FuelTank: rounded-corner tank (bevel the 4 long edges), 2.20 wide x 4.60
# long, seated just under the Frame with two filler-cap discs on top. See
# module docstring for the y-chain deviation from the blockout literal.
# ---------------------------------------------------------------------------
FUEL_TANK_WIDTH = 2.20
FUEL_TANK_LENGTH = 4.60
FUEL_TANK_TOP_Y = 1.02
FUEL_TANK_BOTTOM_Y = 0.42
FUEL_TANK_BEVEL = 0.12
FUEL_TANK_BEVEL_SEGMENTS = 12
FUEL_TANK_CAP_RADIUS = 0.10
FUEL_TANK_CAP_HEIGHT = 0.05
FUEL_TANK_CAP_Z_OFFSET = 1.0
FUEL_TANK_CAP_SEGMENTS = 20

# ---------------------------------------------------------------------------
# LongHood: 2.65 wide x 2.23 tall, length 7.40, z-centre derived from
# half_length via LONG_HOOD_REAR_MARGIN; rounded top (bevel), 5 inset door
# bays per side, 3 louver relief strips per side.
# ---------------------------------------------------------------------------
LONG_HOOD_WIDTH = 2.65
LONG_HOOD_HEIGHT = 2.23
LONG_HOOD_Y = 2.435
LONG_HOOD_LENGTH = 7.40
LONG_HOOD_REAR_MARGIN = 1.50
LONG_HOOD_BEVEL = 0.25
LONG_HOOD_BEVEL_SEGMENTS = 10
LONG_HOOD_DOOR_COUNT = 5
LONG_HOOD_DOOR_INSET = 0.015
LONG_HOOD_DOOR_MARGIN_Y = 0.20
LONG_HOOD_DOOR_MARGIN_Z = 0.10
LONG_HOOD_LOUVER_COUNT = 3
LONG_HOOD_LOUVER_SIZE = (0.02, 0.06, 0.55)  # (outward depth, height, z-run)
LONG_HOOD_LOUVER_Y = LONG_HOOD_Y - 0.35

RADIATOR_FAN_RADIUS = 0.66
RADIATOR_FAN_HEIGHT = 0.16
RADIATOR_FAN_Y = 3.59
RADIATOR_FAN_FRONT_Z = -4.10
RADIATOR_FAN_REAR_Z = -5.05
RADIATOR_FAN_DISC_SEGMENTS = 80
RADIATOR_FAN_BLADE_COUNT = 8
RADIATOR_FAN_BLADE_RADIUS = 0.03
RADIATOR_FAN_BLADE_HUB_RADIUS = 0.06
RADIATOR_FAN_BLADE_OFFSET = 0.58
RADIATOR_FAN_BLADE_SEGMENTS = 16

EXHAUST_STACK_RADIUS = 0.12
EXHAUST_STACK_HEIGHT = 0.48
EXHAUST_STACK_Z = -3.50
EXHAUST_STACK_SEGMENTS = 80

# ---------------------------------------------------------------------------
# Cab: 3.00 wide x 3.03 tall, z-length derived from the LongHood/ShortHood
# gap; four walls built as genuine through-hole "picture frame" solids
# (outer perimeter + inner window tunnel + front/back annulus caps) so the
# window openings are real cut-outs, not applied decals. CabGlass fills the
# four openings with single-sided inset panes.
# ---------------------------------------------------------------------------
CAB_WIDTH = 3.00
CAB_HEIGHT = 3.03
CAB_Y = 2.835
CAB_WALL_THICKNESS = 0.08
CAB_WINDOW_Y_OFFSET = 3.47 - CAB_Y  # 0.635, beltline offset from cab centre
CAB_SIDE_WINDOW_WIDTH = 0.74  # along z
CAB_SIDE_WINDOW_HEIGHT = 1.16  # along y
CAB_END_WINDOW_WIDTH = 1.82  # along x
CAB_END_WINDOW_HEIGHT = 0.74  # along y
CAB_GLASS_INSET = 0.03
CAB_CORNER_EPSILON = 0.003  # see _build_cab: keeps side/end wall corners from exactly coinciding
# Roof overhang lip half-extent additions (dx, dy, dz) added to CAB_WIDTH
# and cab length; dx is kept small (0.03, not the more generous 0.10 an
# unconstrained overhang would use) because CAB_WIDTH (3.00) already
# consumes most of the width envelope (recipe.width/2 = 1.5621), leaving
# only ~0.06 m of overhang budget once the wall itself is accounted for.
CAB_ROOF_LIP = (0.03, 0.05, 0.10)
CAB_NUMBER_BOARD_SIZE = (0.60, 0.12, 0.03)

CABROOF_LENGTH = 2.48
CABROOF_THICKNESS = 0.20
CABROOF_CROWN_RISE = 0.03
CABROOF_TOP_CLEARANCE = 1e-3

# ---------------------------------------------------------------------------
# ShortHood: 2.65 wide x 1.93 tall, length 3.40, z-centre derived from
# half_length via SHORT_HOOD_FRONT_MARGIN; top bevel + one inset door relief
# panel on the +Z face.
# ---------------------------------------------------------------------------
SHORT_HOOD_WIDTH = 2.65
SHORT_HOOD_HEIGHT = 1.93
SHORT_HOOD_Y = 2.285
SHORT_HOOD_LENGTH = 3.40
SHORT_HOOD_FRONT_MARGIN = 1.40
SHORT_HOOD_BEVEL = 0.15
SHORT_HOOD_BEVEL_SEGMENTS = 10
SHORT_HOOD_DOOR_DEPTH = 0.02
SHORT_HOOD_DOOR_HALF_WIDTH = 0.55
SHORT_HOOD_DOOR_HALF_HEIGHT = 0.60
SHORT_HOOD_DOOR_Y_OFFSET = -0.20  # panel centred slightly below hood centre

# ---------------------------------------------------------------------------
# Headlights: twin-lens housings at the hood ends.
# ---------------------------------------------------------------------------
HEADLIGHT_Y = 2.80
# The front/rear z values are deliberately NOT symmetric (6.63 vs -6.53):
# both come from the accepted blockout and reflect real body asymmetry.
# Each headlight seats on its own hood's end face, and the hoods are not
# mirror images: the short hood's front face sits at z = +6.60
# (short_hood_z 4.90 + half-length 1.70) while the long hood's rear face
# sits at z = -6.50 (long_hood_z -2.80 - half-length 3.70). The task
# dispatch's word "mirrored" was sloppy shorthand; the literal blockout
# values govern (controller-adjudicated).
HEADLIGHT_FRONT_Z = 6.63
HEADLIGHT_REAR_Z = -6.53
HEADLIGHT_WIDTH = 0.24
HEADLIGHT_HEIGHT = 0.18
HEADLIGHT_DEPTH = 0.10
HEADLIGHT_BEVEL = 0.02
HEADLIGHT_BEVEL_SEGMENTS = 3
HEADLIGHT_LENS_OFFSET = 0.065
HEADLIGHT_LENS_RADIUS = 0.055
HEADLIGHT_LENS_PROTRUSION = 0.04
HEADLIGHT_LENS_SEGMENTS = 48

# ---------------------------------------------------------------------------
# Horn: 3-trumpet cluster seated on the cab roof peak; kept low/shallow so
# it stays under the height envelope (see module docstring).
# ---------------------------------------------------------------------------
HORN_X = 0.3
HORN_Z = 1.6
HORN_TOP_MARGIN = 0.02  # stay this far under recipe.height
HORN_BASE_SIZE = (0.16, 0.05, 0.12)
HORN_TRUMPET_RISE = 0.06
HORN_TRUMPET_FORWARD = 0.10
HORN_TRUMPET_MOUTH_RADIUS = 0.035
HORN_TRUMPET_THROAT_RADIUS = 0.014
HORN_TRUMPET_X_OFFSETS = (-0.08, 0.0, 0.08)
HORN_TRUMPET_SEGMENTS = 24

# ---------------------------------------------------------------------------
# Bell: small bell shape under the frame edge.
# ---------------------------------------------------------------------------
BELL_X = 0.9
BELL_Y = 1.0
BELL_Z = 5.2
BELL_MOUTH_RADIUS = 0.09
BELL_WAIST_RADIUS = 0.03
BELL_TOP_RADIUS = 0.015
BELL_HALF_HEIGHT = 0.10
BELL_WAIST_T = 0.7  # fraction of the way up where the waist sits
BELL_SEGMENTS = 48

# ---------------------------------------------------------------------------
# Handrails: full-length rod runs on stanchions, along both walkway edges.
# ---------------------------------------------------------------------------
HANDRAIL_X_INSET = 0.06  # clears the width envelope with the rod radius
HANDRAIL_Y = 2.0
HANDRAIL_RADIUS = 0.016
HANDRAIL_Z0 = -7.0
HANDRAIL_Z1 = 7.0
HANDRAIL_STANCHION_SPACING = 1.5
HANDRAIL_STANCHION_RADIUS = 0.012
HANDRAIL_STANCHION_TOP_Y = WALKWAY_Y + WALKWAY_HEIGHT / 2.0
HANDRAIL_END_WRAP_X_FACTOR = 0.5
HANDRAIL_END_WRAP_Y_DROP = 0.28
HANDRAIL_END_WRAP_Z_EXTRA = 0.25
HANDRAIL_MAIN_SEGMENTS = 40
HANDRAIL_STANCHION_SEGMENTS = 32
HANDRAIL_END_SEGMENTS = 40

# ---------------------------------------------------------------------------
# Steps: 3-step wells at the four corners.
# ---------------------------------------------------------------------------
STEP_X = 1.2
STEP_Z = 7.2
STEP_TREAD_SIZE = (0.5, 0.04, 0.25)
STEP_TOP_Y = 1.32
STEP_BOTTOM_Y = 0.55
STEP_COUNT = 3
STEP_Z_STAGGER = 0.20
STEP_PLATE_THICKNESS = 0.03


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


def _append_cone_frustum(bm, p0, r0, p1, r1, segments):
    """Appends a closed, capped frustum (or cone, r1 near 0) between two
    arbitrary vehicle-space points with independent end radii -- the same
    ring-sweep technique as ``_append_capsule_segment``, generalised to a
    tapering radius (used for the horn trumpets and the bell)."""
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
        ca, sa = math.cos(theta), math.sin(theta)
        offset_a = e1 * (r0 * ca) + e2 * (r0 * sa)
        offset_b = e1 * (r1 * ca) + e2 * (r1 * sa)
        ring_a.append(bm.verts.new(scene.to_blender(tuple(v0 + offset_a))))
        ring_b.append(bm.verts.new(scene.to_blender(tuple(v1 + offset_b))))
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


def _find_edges(bm, predicate):
    """Returns bm edges whose two endpoints (converted to unity-space
    coordinates) satisfy ``predicate(p0, p1)``."""
    bm.edges.ensure_lookup_table()
    return [
        e for e in bm.edges
        if predicate(scene.to_unity(e.verts[0].co), scene.to_unity(e.verts[1].co))
    ]


def _append_inset_bays(bm, x_face, inward_sign, y_top, y_bottom, bounds, inset, margin_y, margin_z):
    """Appends N inset bay panels (a picture-frame ring + recessed floor
    per bay) covering one x-face wall, split into z-segments at ``bounds``
    -- the same hand-rolled pocket construction body_wagon.py's BoxBody
    uses (bmesh.ops.inset_individual proved unreliable there).
    """
    for i in range(len(bounds) - 1):
        z0, z1 = bounds[i], bounds[i + 1]
        outer = [
            bm.verts.new(scene.to_blender((x_face, y_bottom, z0))),
            bm.verts.new(scene.to_blender((x_face, y_bottom, z1))),
            bm.verts.new(scene.to_blender((x_face, y_top, z1))),
            bm.verts.new(scene.to_blender((x_face, y_top, z0))),
        ]
        pocket_x = x_face + inward_sign * inset
        my = min(margin_y, (y_top - y_bottom) * 0.5 - 1e-4)
        mz = min(margin_z, (z1 - z0) * 0.5 - 1e-4)
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


def _rect_corners(centre_p, centre_q, half_p, half_q):
    return [
        (centre_p - half_p, centre_q - half_q),
        (centre_p + half_p, centre_q - half_q),
        (centre_p + half_p, centre_q + half_q),
        (centre_p - half_p, centre_q + half_q),
    ]


def _axis_point(axis, p, q, val):
    return (val, p, q) if axis == "x" else (p, q, val)


def _append_window_wall(bm, axis, face_val, thickness, inward_sign, outer_rect, inner_rect):
    """Appends a closed "picture frame" solid -- a thin wall slab with a
    genuine rectangular hole through it -- into bm. ``outer_rect`` /
    ``inner_rect`` are (centre_p, centre_q, half_p, half_q) in the plane
    perpendicular to ``axis`` (p, q map the same way as ``_append_prism``).
    The result is a closed manifold "rectangular donut": an outer
    perimeter tube, an inner (hole) tube, and front/back annulus caps --
    used for Cab's real window openings (no booleans).
    """
    outer = _rect_corners(*outer_rect)
    inner = _rect_corners(*inner_rect)
    inner_face_val = face_val + inward_sign * thickness

    def ring(pts, val):
        return [bm.verts.new(scene.to_blender(_axis_point(axis, p, q, val))) for p, q in pts]

    outer_front = ring(outer, face_val)
    outer_back = ring(outer, inner_face_val)
    inner_front = ring(inner, face_val)
    inner_back = ring(inner, inner_face_val)
    bm.verts.ensure_lookup_table()

    n = 4
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((outer_front[i], outer_front[j], outer_back[j], outer_back[i]))
        bm.faces.new((inner_front[j], inner_front[i], inner_back[i], inner_back[j]))
        bm.faces.new((outer_front[i], outer_front[j], inner_front[j], inner_front[i]))
        bm.faces.new((outer_back[j], outer_back[i], inner_back[i], inner_back[j]))


def _append_pocket_panel(bm, axis, face_val, inward_sign, outer_rect, inner_rect, depth):
    """Appends a closed "picture frame" solid with a BLIND recess (a floor,
    not a through hole) -- used for ShortHood's single door relief panel,
    where ``outer_rect`` spans the whole face so this fully replaces it.
    """
    outer = _rect_corners(*outer_rect)
    inner = _rect_corners(*inner_rect)
    pocket_val = face_val + inward_sign * depth

    outer_v = [bm.verts.new(scene.to_blender(_axis_point(axis, p, q, face_val))) for p, q in outer]
    inner_v = [bm.verts.new(scene.to_blender(_axis_point(axis, p, q, pocket_val))) for p, q in inner]
    bm.verts.ensure_lookup_table()
    for k in range(4):
        j = (k + 1) % 4
        bm.faces.new((outer_v[k], outer_v[j], inner_v[j], inner_v[k]))
    bm.faces.new(inner_v)


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------
def _frame_profile(y0, y1, half_len):
    """Closed (y, z) staircase profile: flat bottom, and a top that steps
    down twice at each end (a 2-step pilot relief), not a plain slab end.
    """
    step_len = FRAME_PILOT_STEP_LENGTH
    drop = FRAME_PILOT_STEP_DROP
    z_a = half_len - 2.0 * step_len
    z_b = half_len - step_len
    y_mid = y1 - drop
    y_low = y1 - 2.0 * drop
    return [
        (y0, -half_len),
        (y0, half_len),
        (y_low, half_len),
        (y_low, z_b),
        (y_mid, z_b),
        (y_mid, z_a),
        (y1, z_a),
        (y1, -z_a),
        (y_mid, -z_a),
        (y_mid, -z_b),
        (y_low, -z_b),
        (y_low, -half_len),
    ]


def _build_frame(width, half_len):
    y0 = FRAME_Y - FRAME_HEIGHT / 2.0
    y1 = FRAME_Y + FRAME_HEIGHT / 2.0
    bm = bmesh.new()
    profile = _frame_profile(y0, y1, half_len)
    _append_prism(bm, profile, axis="x", lo=-width / 2.0, hi=width / 2.0)
    _finalize(bm)
    return scene.mesh_object("Frame", bm)


# ---------------------------------------------------------------------------
# Walkway
# ---------------------------------------------------------------------------
def _build_walkway(width, length):
    bm = bmesh.new()
    _box_bmesh(bm, (0.0, WALKWAY_Y, WALKWAY_Z_OFFSET), (width, WALKWAY_HEIGHT, length))

    # The lip's outer edge is kept a hair inside the walkway's own edge
    # (rather than exactly flush) and its z-run a hair shorter than the
    # walkway's, so none of its 4 boundary edges land exactly on the
    # walkway box's own edges: an exactly-flush lip would share full edges
    # (not just touch a face) with the walkway box, which remove_doubles
    # merges into a 3+-face non-manifold seam (the same failure mode as
    # the Cab wall corners; see CAB_CORNER_EPSILON).
    lip_y = WALKWAY_Y + WALKWAY_HEIGHT / 2.0 + WALKWAY_LIP_HEIGHT / 2.0
    half_width = width / 2.0
    lip_length = length - 2.0 * WALKWAY_LIP_Z_EPSILON
    for sign in (-1.0, 1.0):
        lip_x = sign * (half_width - WALKWAY_LIP_WIDTH / 2.0 - WALKWAY_LIP_X_EPSILON)
        _box_bmesh(bm, (lip_x, lip_y, WALKWAY_Z_OFFSET), (WALKWAY_LIP_WIDTH, WALKWAY_LIP_HEIGHT, lip_length))

    _finalize(bm)
    return scene.mesh_object("Walkway", bm)


# ---------------------------------------------------------------------------
# FuelTank
# ---------------------------------------------------------------------------
def _build_fuel_tank():
    hx = FUEL_TANK_WIDTH / 2.0
    hz = FUEL_TANK_LENGTH / 2.0
    height = FUEL_TANK_TOP_Y - FUEL_TANK_BOTTOM_Y
    centre_y = (FUEL_TANK_TOP_Y + FUEL_TANK_BOTTOM_Y) / 2.0

    bm = bmesh.new()
    _box_bmesh(bm, (0.0, centre_y, 0.0), (FUEL_TANK_WIDTH, height, FUEL_TANK_LENGTH))
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    # bevel's OFFSET direction depends on each adjacent face's winding; see
    # the note in _build_long_hood for how this was discovered.
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()

    long_edges = _find_edges(
        bm,
        lambda p0, p1: (
            abs(p0[1] - p1[1]) < 1e-6
            and abs(abs(p0[0]) - hx) < 1e-4 and abs(abs(p1[0]) - hx) < 1e-4
            and (p0[0] > 0) == (p1[0] > 0)
        ),
    )
    assert len(long_edges) == 4, f"FuelTank: expected 4 long edges, found {len(long_edges)}"
    bmesh.ops.bevel(
        bm, geom=long_edges, offset=FUEL_TANK_BEVEL, offset_type="OFFSET",
        segments=FUEL_TANK_BEVEL_SEGMENTS, affect="EDGES")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()

    cap_y = FUEL_TANK_TOP_Y + FUEL_TANK_CAP_HEIGHT / 2.0
    for z in (-FUEL_TANK_CAP_Z_OFFSET, FUEL_TANK_CAP_Z_OFFSET):
        _append_disc_cylinder(
            bm, (0.0, cap_y, z), FUEL_TANK_CAP_RADIUS, FUEL_TANK_CAP_HEIGHT, "y",
            FUEL_TANK_CAP_SEGMENTS)

    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()
    return scene.mesh_object("FuelTank", bm)


# ---------------------------------------------------------------------------
# LongHood
# ---------------------------------------------------------------------------
def _build_long_hood(z_centre):
    hx = LONG_HOOD_WIDTH / 2.0
    hz = LONG_HOOD_LENGTH / 2.0
    y_top = LONG_HOOD_Y + LONG_HOOD_HEIGHT / 2.0
    y_bottom = LONG_HOOD_Y - LONG_HOOD_HEIGHT / 2.0
    z0 = z_centre - hz
    z1 = z_centre + hz
    step = LONG_HOOD_LENGTH / LONG_HOOD_DOOR_COUNT
    bounds = [z0 + i * step for i in range(LONG_HOOD_DOOR_COUNT + 1)]

    bm = bmesh.new()
    _append_inset_bays(
        bm, -hx, 1.0, y_top, y_bottom, bounds,
        LONG_HOOD_DOOR_INSET, LONG_HOOD_DOOR_MARGIN_Y, LONG_HOOD_DOOR_MARGIN_Z)
    _append_inset_bays(
        bm, hx, -1.0, y_top, y_bottom, bounds,
        LONG_HOOD_DOOR_INSET, LONG_HOOD_DOOR_MARGIN_Y, LONG_HOOD_DOOR_MARGIN_Z)

    for z in (z0, z1):
        corners = [
            bm.verts.new(scene.to_blender((-hx, y_bottom, z))),
            bm.verts.new(scene.to_blender((hx, y_bottom, z))),
            bm.verts.new(scene.to_blender((hx, y_top, z))),
            bm.verts.new(scene.to_blender((-hx, y_top, z))),
        ]
        bm.verts.ensure_lookup_table()
        bm.faces.new(corners)

    for y in (y_top, y_bottom):
        for i in range(len(bounds) - 1):
            zz0, zz1 = bounds[i], bounds[i + 1]
            corners = [
                bm.verts.new(scene.to_blender((-hx, y, zz0))),
                bm.verts.new(scene.to_blender((hx, y, zz0))),
                bm.verts.new(scene.to_blender((hx, y, zz1))),
                bm.verts.new(scene.to_blender((-hx, y, zz1))),
            ]
            bm.verts.ensure_lookup_table()
            bm.faces.new(corners)

    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    bmesh.ops.holes_fill(bm, edges=bm.edges[:])
    # bevel's OFFSET direction depends on each adjacent face's winding, so
    # normals must be made consistent BEFORE beveling (discovered via a
    # failing envelope check: without this, the top-edge bevel moved
    # outward by exactly LONG_HOOD_BEVEL instead of inward).
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()

    top_edges = _find_edges(
        bm,
        lambda p0, p1: (
            abs(p0[1] - y_top) < 1e-4 and abs(p1[1] - y_top) < 1e-4
            and abs(abs(p0[0]) - hx) < 1e-4 and abs(abs(p1[0]) - hx) < 1e-4
            and (p0[0] > 0) == (p1[0] > 0)
        ),
    )
    expected_top_edges = 2 * LONG_HOOD_DOOR_COUNT
    assert len(top_edges) == expected_top_edges, (
        f"LongHood: expected {expected_top_edges} top edges, found {len(top_edges)}")
    bmesh.ops.bevel(
        bm, geom=top_edges, offset=LONG_HOOD_BEVEL, offset_type="OFFSET",
        segments=LONG_HOOD_BEVEL_SEGMENTS, affect="EDGES")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()

    louver_span = LONG_HOOD_LOUVER_SIZE[2]
    louver_pitch = LONG_HOOD_LENGTH / (LONG_HOOD_LOUVER_COUNT + 1)
    for x_face, outward in ((-hx, -1.0), (hx, 1.0)):
        for i in range(LONG_HOOD_LOUVER_COUNT):
            z = z0 + louver_pitch * (i + 1)
            centre_x = x_face + outward * (LONG_HOOD_LOUVER_SIZE[0] / 2.0)
            _box_bmesh(
                bm, (centre_x, LONG_HOOD_LOUVER_Y, z),
                (LONG_HOOD_LOUVER_SIZE[0], LONG_HOOD_LOUVER_SIZE[1], louver_span))

    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()
    return scene.mesh_object("LongHood", bm)


def _build_radiator_fan(name, z_centre, hood_top_y):
    bm = bmesh.new()
    centre = (0.0, RADIATOR_FAN_Y, z_centre)
    assert centre[1] - RADIATOR_FAN_HEIGHT / 2.0 <= hood_top_y + 1e-6, (
        f"{name}: fan does not seat on the hood top (floats above it)")
    _append_disc_cylinder(
        bm, centre, RADIATOR_FAN_RADIUS, RADIATOR_FAN_HEIGHT, "y", RADIATOR_FAN_DISC_SEGMENTS)

    blade_y = centre[1] + RADIATOR_FAN_HEIGHT / 2.0
    for i in range(RADIATOR_FAN_BLADE_COUNT):
        theta = 2.0 * math.pi * i / RADIATOR_FAN_BLADE_COUNT
        ct, st = math.cos(theta), math.sin(theta)
        # Every blade lies in the horizontal plane, so _append_capsule_segment
        # always derives the same vertical e2 basis vector for all of them;
        # two opposite-direction blades (e.g. i and i+4 of 8) then have
        # mirrored e1 vectors, which makes their p0-end rings the exact same
        # set of points (just reversed) -- an exact duplicate ring that
        # remove_doubles merges into a non-manifold 3-face seam. Starting
        # each blade from a small hub offset in its own direction (instead
        # of the literal shared centre point) gives every blade a distinct
        # p0, breaking that coincidence.
        hub = (
            centre[0] + RADIATOR_FAN_BLADE_HUB_RADIUS * ct,
            blade_y,
            centre[2] + RADIATOR_FAN_BLADE_HUB_RADIUS * st,
        )
        tip = (
            centre[0] + RADIATOR_FAN_BLADE_OFFSET * ct,
            blade_y,
            centre[2] + RADIATOR_FAN_BLADE_OFFSET * st,
        )
        _append_capsule_segment(bm, hub, tip, RADIATOR_FAN_BLADE_RADIUS, RADIATOR_FAN_BLADE_SEGMENTS)

    _finalize(bm)
    return scene.mesh_object(name, bm)


def _disc_part(name, centre, radius, height, segments):
    bm = bmesh.new()
    _append_disc_cylinder(bm, centre, radius, height, "y", segments)
    _finalize(bm)
    return scene.mesh_object(name, bm)


def _build_exhaust_stack(hood_top_y):
    centre_y = hood_top_y + EXHAUST_STACK_HEIGHT / 2.0
    return _disc_part(
        "ExhaustStack", (0.0, centre_y, EXHAUST_STACK_Z), EXHAUST_STACK_RADIUS,
        EXHAUST_STACK_HEIGHT, EXHAUST_STACK_SEGMENTS)


# ---------------------------------------------------------------------------
# Cab + CabGlass
# ---------------------------------------------------------------------------
def _build_cab(z_centre, length_z):
    hx = CAB_WIDTH / 2.0
    hy = CAB_HEIGHT / 2.0
    hz = length_z / 2.0
    y_top = CAB_Y + hy
    y_bottom = CAB_Y - hy
    z_front = z_centre + hz
    z_rear = z_centre - hz
    window_y = CAB_Y + CAB_WINDOW_Y_OFFSET

    bm = bmesh.new()

    # The side walls' outer rect is shrunk a hair short of the true z
    # extent (rather than reaching z_front/z_rear exactly) so its top/
    # bottom corners don't land exactly on the end walls' corners: at the
    # box's 4 true vertical corners, a side wall's outer-ring edge and an
    # end wall's outer-ring edge would otherwise be the literal same edge
    # (same two endpoints), and remove_doubles would merge them into one
    # edge shared by faces from both walls -- a non-manifold 3+-face seam.
    side_outer = (CAB_Y, z_centre, hy, hz - CAB_CORNER_EPSILON)
    side_inner = (window_y, z_centre, CAB_SIDE_WINDOW_HEIGHT / 2.0, CAB_SIDE_WINDOW_WIDTH / 2.0)
    _append_window_wall(bm, "x", -hx, CAB_WALL_THICKNESS, 1.0, side_outer, side_inner)
    _append_window_wall(bm, "x", hx, CAB_WALL_THICKNESS, -1.0, side_outer, side_inner)

    end_outer = (0.0, CAB_Y, hx, hy)
    end_inner = (0.0, window_y, CAB_END_WINDOW_WIDTH / 2.0, CAB_END_WINDOW_HEIGHT / 2.0)
    _append_window_wall(bm, "z", z_front, CAB_WALL_THICKNESS, -1.0, end_outer, end_inner)
    _append_window_wall(bm, "z", z_rear, CAB_WALL_THICKNESS, 1.0, end_outer, end_inner)

    # Roof overhang lip: a thin raised rim just below the roofline, slightly
    # proud of the wall faces on all four sides.
    lip_dx, lip_dy, lip_dz = CAB_ROOF_LIP
    lip_y = y_top - lip_dy / 2.0
    _box_bmesh(bm, (0.0, lip_y, z_centre), (CAB_WIDTH + 2.0 * lip_dx, lip_dy, length_z + 2.0 * lip_dz))

    # Number-board boxes, front AND rear top (per brief: "number-board
    # boxes front/rear top"), mirrored in z about the cab centre; each
    # sits proud of its end wall by its own depth.
    nb_w, nb_h, nb_d = CAB_NUMBER_BOARD_SIZE
    nb_y = y_top - nb_h * 0.7
    _box_bmesh(bm, (0.0, nb_y, z_front + nb_d / 2.0), CAB_NUMBER_BOARD_SIZE)
    _box_bmesh(bm, (0.0, nb_y, z_rear - nb_d / 2.0), CAB_NUMBER_BOARD_SIZE)

    _finalize(bm)
    return scene.mesh_object("Cab", bm), (y_bottom, y_top, z_front, z_rear, window_y, hx)


def _build_cab_glass(cab_geo):
    y_bottom, y_top, z_front, z_rear, window_y, hx = cab_geo
    bm = bmesh.new()

    def quad(axis, val, centre_p, centre_q, half_p, half_q):
        pts = _rect_corners(centre_p, centre_q, half_p, half_q)
        verts = [bm.verts.new(scene.to_blender(_axis_point(axis, p, q, val))) for p, q in pts]
        bm.verts.ensure_lookup_table()
        bm.faces.new(verts)

    z_mid = (z_front + z_rear) / 2.0
    quad("x", -hx + CAB_GLASS_INSET, window_y, z_mid,
         CAB_SIDE_WINDOW_HEIGHT / 2.0, CAB_SIDE_WINDOW_WIDTH / 2.0)
    quad("x", hx - CAB_GLASS_INSET, window_y, z_mid,
         CAB_SIDE_WINDOW_HEIGHT / 2.0, CAB_SIDE_WINDOW_WIDTH / 2.0)
    quad("z", z_front - CAB_GLASS_INSET, 0.0, window_y,
         CAB_END_WINDOW_WIDTH / 2.0, CAB_END_WINDOW_HEIGHT / 2.0)
    quad("z", z_rear + CAB_GLASS_INSET, 0.0, window_y,
         CAB_END_WINDOW_WIDTH / 2.0, CAB_END_WINDOW_HEIGHT / 2.0)

    bm.normal_update()
    return scene.mesh_object("CabGlass", bm)


def _build_cab_roof(recipe_height, z_centre, width):
    hx = width / 2.0
    peak_y = recipe_height - CABROOF_TOP_CLEARANCE
    edge_y = peak_y - CABROOF_CROWN_RISE
    bottom_y = edge_y - CABROOF_THICKNESS

    bm = bmesh.new()
    profile = ((-hx, bottom_y), (hx, bottom_y), (hx, edge_y), (0.0, peak_y), (-hx, edge_y))
    _append_prism(
        bm, profile, axis="z", lo=z_centre - CABROOF_LENGTH / 2.0, hi=z_centre + CABROOF_LENGTH / 2.0)
    _finalize(bm)
    return scene.mesh_object("CabRoof", bm)


# ---------------------------------------------------------------------------
# ShortHood
# ---------------------------------------------------------------------------
def _build_short_hood(z_centre):
    hx = SHORT_HOOD_WIDTH / 2.0
    hy = SHORT_HOOD_HEIGHT / 2.0
    hz = SHORT_HOOD_LENGTH / 2.0
    y_top = SHORT_HOOD_Y + hy
    y_bottom = SHORT_HOOD_Y - hy
    z_front = z_centre + hz
    z_rear = z_centre - hz

    bm = bmesh.new()
    for x in (-hx, hx):
        corners = [
            bm.verts.new(scene.to_blender((x, y_bottom, z_rear))),
            bm.verts.new(scene.to_blender((x, y_bottom, z_front))),
            bm.verts.new(scene.to_blender((x, y_top, z_front))),
            bm.verts.new(scene.to_blender((x, y_top, z_rear))),
        ]
        bm.verts.ensure_lookup_table()
        bm.faces.new(corners if x < 0 else list(reversed(corners)))

    for y in (y_top, y_bottom):
        corners = [
            bm.verts.new(scene.to_blender((-hx, y, z_rear))),
            bm.verts.new(scene.to_blender((hx, y, z_rear))),
            bm.verts.new(scene.to_blender((hx, y, z_front))),
            bm.verts.new(scene.to_blender((-hx, y, z_front))),
        ]
        bm.verts.ensure_lookup_table()
        bm.faces.new(corners if y > SHORT_HOOD_Y else list(reversed(corners)))

    rear = [
        bm.verts.new(scene.to_blender((-hx, y_bottom, z_rear))),
        bm.verts.new(scene.to_blender((hx, y_bottom, z_rear))),
        bm.verts.new(scene.to_blender((hx, y_top, z_rear))),
        bm.verts.new(scene.to_blender((-hx, y_top, z_rear))),
    ]
    bm.verts.ensure_lookup_table()
    bm.faces.new(list(reversed(rear)))

    outer_front = (0.0, SHORT_HOOD_Y, hx, hy)
    inner_front = (
        0.0, SHORT_HOOD_Y + SHORT_HOOD_DOOR_Y_OFFSET,
        SHORT_HOOD_DOOR_HALF_WIDTH, SHORT_HOOD_DOOR_HALF_HEIGHT)
    _append_pocket_panel(bm, "z", z_front, -1.0, outer_front, inner_front, SHORT_HOOD_DOOR_DEPTH)

    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    bmesh.ops.holes_fill(bm, edges=bm.edges[:])
    # bevel's OFFSET direction depends on each adjacent face's winding; see
    # the note in _build_long_hood for how this was discovered.
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()

    top_edges = _find_edges(
        bm,
        lambda p0, p1: (
            abs(p0[1] - y_top) < 1e-4 and abs(p1[1] - y_top) < 1e-4
            and abs(abs(p0[0]) - hx) < 1e-4 and abs(abs(p1[0]) - hx) < 1e-4
            and (p0[0] > 0) == (p1[0] > 0)
        ),
    )
    assert len(top_edges) == 2, f"ShortHood: expected 2 top edges, found {len(top_edges)}"
    bmesh.ops.bevel(
        bm, geom=top_edges, offset=SHORT_HOOD_BEVEL, offset_type="OFFSET",
        segments=SHORT_HOOD_BEVEL_SEGMENTS, affect="EDGES")

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()
    return scene.mesh_object("ShortHood", bm)


# ---------------------------------------------------------------------------
# Headlights
# ---------------------------------------------------------------------------
def _build_headlight(name, z_face, facing_sign):
    hx = HEADLIGHT_WIDTH / 2.0
    hz = HEADLIGHT_DEPTH / 2.0
    cz = z_face - facing_sign * hz

    bm = bmesh.new()
    _box_bmesh(bm, (0.0, HEADLIGHT_Y, cz), (HEADLIGHT_WIDTH, HEADLIGHT_HEIGHT, HEADLIGHT_DEPTH))
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    # bevel's OFFSET direction depends on each adjacent face's winding; see
    # the note in _build_long_hood for how this was discovered.
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()

    vertical_edges = _find_edges(
        bm,
        lambda p0, p1: (
            abs(p0[0] - p1[0]) < 1e-6 and abs(p0[2] - p1[2]) < 1e-6
            and abs(abs(p0[0]) - hx) < 1e-4 and abs(abs(p0[2] - cz) - hz) < 1e-4
        ),
    )
    if len(vertical_edges) == 4:
        bmesh.ops.bevel(
            bm, geom=vertical_edges, offset=HEADLIGHT_BEVEL, offset_type="OFFSET",
            segments=HEADLIGHT_BEVEL_SEGMENTS, affect="EDGES")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()

    lens_z = z_face + facing_sign * (HEADLIGHT_LENS_PROTRUSION * 0.4)
    for dx in (-HEADLIGHT_LENS_OFFSET, HEADLIGHT_LENS_OFFSET):
        _append_disc_cylinder(
            bm, (dx, HEADLIGHT_Y, lens_z), HEADLIGHT_LENS_RADIUS, HEADLIGHT_LENS_PROTRUSION, "z",
            HEADLIGHT_LENS_SEGMENTS)

    _finalize(bm)
    return scene.mesh_object(name, bm)


# ---------------------------------------------------------------------------
# Horn
# ---------------------------------------------------------------------------
def _build_horn(recipe_height, cabroof_peak_y):
    # The trumpet tip ring's radius can add extra height beyond the
    # centreline point (its axis is inclined, not purely horizontal, so
    # part of the mouth-radius ring can sit above the tip centre) --
    # budget for the full radius on top of the centreline rise so the
    # actual mesh, not just its centreline, stays under the envelope.
    ceiling = recipe_height - HORN_TOP_MARGIN
    y_base = min(cabroof_peak_y, ceiling - HORN_TRUMPET_RISE - HORN_TRUMPET_MOUTH_RADIUS)
    max_y = y_base + HORN_TRUMPET_RISE + HORN_TRUMPET_MOUTH_RADIUS
    assert max_y <= ceiling + 1e-6, f"Horn: max_y {max_y} breaches the height envelope margin"

    bm = bmesh.new()
    _box_bmesh(bm, (HORN_X, y_base - HORN_BASE_SIZE[1] / 2.0, HORN_Z), HORN_BASE_SIZE)
    for dx in HORN_TRUMPET_X_OFFSETS:
        p0 = (HORN_X + dx, y_base, HORN_Z)
        p1 = (HORN_X + dx * 0.6, y_base + HORN_TRUMPET_RISE, HORN_Z + HORN_TRUMPET_FORWARD)
        _append_cone_frustum(
            bm, p0, HORN_TRUMPET_THROAT_RADIUS, p1, HORN_TRUMPET_MOUTH_RADIUS, HORN_TRUMPET_SEGMENTS)

    _finalize(bm)
    return scene.mesh_object("Horn", bm)


# ---------------------------------------------------------------------------
# Bell
# ---------------------------------------------------------------------------
def _build_bell():
    """Built as one 3-station ring loft (the same technique body_wagon.py's
    Underframe fish-belly taper uses), NOT two independently-capped cone
    frustums butted together: the bell's axis is purely vertical for both
    the mouth->waist and waist->top legs, so two separate
    ``_append_cone_frustum`` calls would derive the identical e1/e2 basis
    for both and produce an exact duplicate ring at the shared waist point
    -- a non-manifold seam, the same failure mode documented in
    coupler.py's CUT_LEVER_POINTS comment. A single loft has only the two
    true end caps (mouth, top), with no duplicated internal ring.
    """
    bottom_y = BELL_Y - BELL_HALF_HEIGHT
    waist_y = bottom_y + 2.0 * BELL_HALF_HEIGHT * BELL_WAIST_T
    top_y = BELL_Y + BELL_HALF_HEIGHT
    stations = ((bottom_y, BELL_MOUTH_RADIUS), (waist_y, BELL_WAIST_RADIUS), (top_y, BELL_TOP_RADIUS))

    bm = bmesh.new()
    rings = []
    for y, r in stations:
        ring = []
        for i in range(BELL_SEGMENTS):
            theta = 2.0 * math.pi * i / BELL_SEGMENTS
            x = BELL_X + r * math.cos(theta)
            z = BELL_Z + r * math.sin(theta)
            ring.append(bm.verts.new(scene.to_blender((x, y, z))))
        rings.append(ring)
    bm.verts.ensure_lookup_table()

    for i in range(len(rings) - 1):
        a, b = rings[i], rings[i + 1]
        for k in range(BELL_SEGMENTS):
            j = (k + 1) % BELL_SEGMENTS
            bm.faces.new((a[k], a[j], b[j], b[k]))
    bm.faces.new(tuple(reversed(rings[0])))
    bm.faces.new(rings[-1])

    _finalize(bm)
    return scene.mesh_object("Bell", bm)


# ---------------------------------------------------------------------------
# Handrails
# ---------------------------------------------------------------------------
def _stanchion_z_positions():
    positions = []
    z = HANDRAIL_Z0
    while z <= HANDRAIL_Z1 + 1e-6:
        positions.append(z)
        z += HANDRAIL_STANCHION_SPACING
    return positions


def _build_handrail(name, x_sign, width):
    x = x_sign * (width / 2.0 - HANDRAIL_X_INSET)

    bm = bmesh.new()
    _append_capsule_segment(
        bm, (x, HANDRAIL_Y, HANDRAIL_Z0), (x, HANDRAIL_Y, HANDRAIL_Z1),
        HANDRAIL_RADIUS, HANDRAIL_MAIN_SEGMENTS)

    for z in _stanchion_z_positions():
        _append_capsule_segment(
            bm, (x, HANDRAIL_STANCHION_TOP_Y, z), (x, HANDRAIL_Y, z),
            HANDRAIL_STANCHION_RADIUS, HANDRAIL_STANCHION_SEGMENTS)

    for z_end, z_sign in ((HANDRAIL_Z0, -1.0), (HANDRAIL_Z1, 1.0)):
        wrap_x = x * HANDRAIL_END_WRAP_X_FACTOR
        wrap_y = HANDRAIL_Y - HANDRAIL_END_WRAP_Y_DROP
        wrap_z = z_end + z_sign * HANDRAIL_END_WRAP_Z_EXTRA
        _append_capsule_segment(
            bm, (x, HANDRAIL_Y, z_end), (wrap_x, wrap_y, wrap_z),
            HANDRAIL_RADIUS, HANDRAIL_END_SEGMENTS)

    _finalize(bm)
    return scene.mesh_object(name, bm)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def _build_steps(name, x_sign, z_sign):
    x_centre = x_sign * STEP_X
    z_centre = z_sign * STEP_Z

    bm = bmesh.new()
    tread_span = (STEP_TOP_Y - STEP_BOTTOM_Y) / (STEP_COUNT - 1)
    for i in range(STEP_COUNT):
        y = STEP_TOP_Y - i * tread_span
        z = z_centre + z_sign * STEP_Z_STAGGER * i
        _box_bmesh(bm, (x_centre, y, z), STEP_TREAD_SIZE)

    plate_height = (STEP_TOP_Y - STEP_BOTTOM_Y) + 0.10
    plate_y = (STEP_TOP_Y + STEP_BOTTOM_Y) / 2.0
    plate_z_run = STEP_Z_STAGGER * (STEP_COUNT - 1) + STEP_TREAD_SIZE[2] + 0.05
    plate_z = z_centre + z_sign * (STEP_Z_STAGGER * (STEP_COUNT - 1) / 2.0)
    for side in (-1.0, 1.0):
        plate_x = x_centre + side * (STEP_TREAD_SIZE[0] / 2.0 - STEP_PLATE_THICKNESS / 2.0)
        _box_bmesh(
            bm, (plate_x, plate_y, plate_z),
            (STEP_PLATE_THICKNESS, plate_height, plate_z_run))

    _finalize(bm)
    return scene.mesh_object(name, bm)


def _assert_envelope(meshes, recipe):
    width_limit = recipe.width / 2.0 + 1e-4
    height_limit = recipe.height + 1e-4
    length_limit = recipe.body_length / 2.0 + 1e-4

    for name, obj in meshes.items():
        _p, _n, _u, _t, bounds_min, bounds_max = scene.collect_mesh_data(obj)
        max_abs_x = max(abs(bounds_min[0]), abs(bounds_max[0]))
        assert max_abs_x <= width_limit, f"{name} exceeds width envelope: |x|={max_abs_x}"
        assert bounds_max[1] <= height_limit, f"{name} exceeds height envelope: y={bounds_max[1]}"
        max_abs_z = max(abs(bounds_min[2]), abs(bounds_max[2]))
        assert max_abs_z <= length_limit, f"{name} exceeds body-length envelope: |z|={max_abs_z}"


def build_meshes(recipe):
    """21 Body-local meshes for one GP9-envelope diesel locomotive body.

    See the module docstring for the space convention, the derivation chain
    (Frame/Walkway lengths, ShortHood/LongHood/Cab z-positions, CabRoof's
    crowned peak), the envelope guards applied before returning, and the
    disclosed 21-vs-"20"-named-parts count.
    """
    half_length = recipe.body_length / 2.0

    meshes = {}

    meshes["Frame"] = _build_frame(recipe.width, half_length)

    walkway_length = recipe.body_length - WALKWAY_LENGTH_MARGIN
    meshes["Walkway"] = _build_walkway(recipe.width, walkway_length)

    meshes["FuelTank"] = _build_fuel_tank()

    short_hood_z = half_length - SHORT_HOOD_FRONT_MARGIN - SHORT_HOOD_LENGTH / 2.0
    long_hood_z = -(half_length - LONG_HOOD_REAR_MARGIN - LONG_HOOD_LENGTH / 2.0)

    meshes["LongHood"] = _build_long_hood(long_hood_z)
    hood_top_y = LONG_HOOD_Y + LONG_HOOD_HEIGHT / 2.0
    meshes["RadiatorFanFront"] = _build_radiator_fan("RadiatorFanFront", RADIATOR_FAN_FRONT_Z, hood_top_y)
    meshes["RadiatorFanRear"] = _build_radiator_fan("RadiatorFanRear", RADIATOR_FAN_REAR_Z, hood_top_y)
    meshes["ExhaustStack"] = _build_exhaust_stack(hood_top_y)

    long_hood_front = long_hood_z + LONG_HOOD_LENGTH / 2.0
    short_hood_rear = short_hood_z - SHORT_HOOD_LENGTH / 2.0
    cab_z = (long_hood_front + short_hood_rear) / 2.0
    cab_length_z = short_hood_rear - long_hood_front
    assert cab_length_z > 0.0, "Cab: LongHood/ShortHood gap is non-positive"

    meshes["Cab"], cab_geo = _build_cab(cab_z, cab_length_z)
    meshes["CabGlass"] = _build_cab_glass(cab_geo)
    meshes["CabRoof"] = _build_cab_roof(recipe.height, cab_z, recipe.width)

    meshes["ShortHood"] = _build_short_hood(short_hood_z)

    meshes["HeadlightFront"] = _build_headlight("HeadlightFront", HEADLIGHT_FRONT_Z, 1.0)
    meshes["HeadlightRear"] = _build_headlight("HeadlightRear", HEADLIGHT_REAR_Z, -1.0)

    cabroof_peak_y = recipe.height - CABROOF_TOP_CLEARANCE
    meshes["Horn"] = _build_horn(recipe.height, cabroof_peak_y)
    meshes["Bell"] = _build_bell()

    meshes["HandrailLeft"] = _build_handrail("HandrailLeft", -1.0, recipe.width)
    meshes["HandrailRight"] = _build_handrail("HandrailRight", 1.0, recipe.width)

    for x_sign, z_sign, suffix in (
        (-1.0, 1.0, "FrontLeft"), (1.0, 1.0, "FrontRight"),
        (-1.0, -1.0, "RearLeft"), (1.0, -1.0, "RearRight"),
    ):
        meshes[f"Steps{suffix}"] = _build_steps(f"Steps{suffix}", x_sign, z_sign)

    _assert_envelope(meshes, recipe)
    return meshes
