"""Parametric unbranded knuckle coupler part builder.

Space convention (controller-adjudicated): all coupler meshes are authored
CouplerPivot-local. Local +Z points OUTWARD (away from the vehicle, toward
the far end of the train) and the pulling-face plane -- where the knuckle
jaw closes against the mating coupler -- sits at local z = +0.600
(``recipe.COUPLER_PIVOT_TO_FACE``). Local x is lateral, local y is vertical
(coupler-head "thickness" direction), matching the shared vehicle-space
convention used by the other part builders. The node table (Task 10) mounts
the pivot at coupler height and yaw-180s the rear coupler; this module only
authors the canonical +Z-outward orientation.

Layout (from the design brief):

- DraftGear: box centred (0, 0, -0.08), size (0.38, 0.30, 0.26); the
  yoke/housing that anchors the shank into the underframe.
- Shank: 0.18x0.18 section running z in [0, 0.49], widened slightly toward
  the head end (a hand-built 8-vert tapered prism, not a plain box).
- Head: an extruded closed 10-point 2D contour (top view: x lateral,
  z forward) tracing the guard arm and knuckle jaw, extruded to
  y in [-0.14, +0.14] and beveled along its vertical corner edges only,
  so the flat z=0.600 pulling face (three contour points sit exactly on
  that plane) is left untouched by the bevel.
- KnucklePin: a vertical (y-axis) cylinder near the jaw throat.
- CutLever: a bent rod (three straight capsule segments in one mesh)
  running from the head side back toward the frame.
"""

import math

import bmesh
import mathutils

from .. import recipe
from .. import scene

DRAFT_GEAR_CENTRE = (0.0, 0.0, -0.08)
DRAFT_GEAR_SIZE = (0.38, 0.30, 0.26)

SHANK_Z0 = 0.0
SHANK_Z1 = recipe.COUPLER_PIVOT_TO_FACE - 0.11  # 0.49
SHANK_HALF_BASE = (0.09, 0.09)  # draft-gear end: plain 0.18x0.18 section
SHANK_HALF_HEAD = (0.11, 0.10)  # head end: slightly widened taper

# Pulling-face plane; the contour's three foremost points lie exactly on it.
_FACE_Z = recipe.COUPLER_PIVOT_TO_FACE

HEAD_CONTOUR = (
    (-0.17, 0.40),
    (-0.17, 0.52),
    (-0.10, _FACE_Z),
    (-0.02, _FACE_Z),
    (0.05, 0.56),
    (0.09, _FACE_Z),
    (0.15, 0.58),
    (0.17, 0.50),
    (0.13, 0.44),
    (0.06, 0.40),
)
HEAD_Y_HALF = 0.14
HEAD_BEVEL_OFFSET = 0.015
HEAD_BEVEL_SEGMENTS = 1

# The brief's literal "(0.09, 0.0->0.10, 0.50)" is interpreted (per task
# instructions) as: pin axis at x=0.09, z=0.50, spanning vertically. Centred
# at y=0.10 (span y in [0.00, 0.20]) so the pin pokes a clearly visible
# 0.06 m above the head's y=+0.14 top, while staying well inside the
# |y| <= 0.30 bound.
KNUCKLE_PIN_CENTRE = (0.09, 0.10, 0.50)
KNUCKLE_PIN_RADIUS = 0.032
KNUCKLE_PIN_HEIGHT = 0.20
KNUCKLE_PIN_SEGMENTS = 24

CUT_LEVER_POINTS = (
    # The brief's polyline includes a (0.30, 0.10, 0.00) midpoint between
    # the second and last points, but it is exactly colinear with them (all
    # three share x=0.30, y=0.10, straight along -z). Building it as its
    # own capsule segment would butt two shells together with identical
    # orientation, so remove_doubles merges their end caps into a
    # duplicate-face seam (a non-manifold edge). It carries no bend
    # information, so it is collapsed here into a single straight leg.
    (0.17, 0.10, 0.52),
    (0.30, 0.10, 0.30),
    (0.30, 0.10, -0.15),
)
CUT_LEVER_RADIUS = 0.012
CUT_LEVER_SEGMENTS = 16


def _finalize(bm):
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    bmesh.ops.holes_fill(bm, edges=bm.edges[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()


def _build_draft_gear():
    return scene.box("DraftGear", DRAFT_GEAR_CENTRE, DRAFT_GEAR_SIZE)


def _build_shank():
    """Hand-built 8-vert prism: a plain square base tapering to a wider top.

    z=SHANK_Z0 (draft-gear end) uses SHANK_HALF_BASE (a plain 0.18x0.18
    section); z=SHANK_Z1 (head end) uses the slightly wider SHANK_HALF_HEAD,
    giving the shank a subtle taper toward the head.
    """
    bm = bmesh.new()
    hx0, hy0 = SHANK_HALF_BASE
    hx1, hy1 = SHANK_HALF_HEAD
    base = [
        bm.verts.new(scene.to_blender((-hx0, -hy0, SHANK_Z0))),
        bm.verts.new(scene.to_blender((hx0, -hy0, SHANK_Z0))),
        bm.verts.new(scene.to_blender((hx0, hy0, SHANK_Z0))),
        bm.verts.new(scene.to_blender((-hx0, hy0, SHANK_Z0))),
    ]
    top = [
        bm.verts.new(scene.to_blender((-hx1, -hy1, SHANK_Z1))),
        bm.verts.new(scene.to_blender((hx1, -hy1, SHANK_Z1))),
        bm.verts.new(scene.to_blender((hx1, hy1, SHANK_Z1))),
        bm.verts.new(scene.to_blender((-hx1, hy1, SHANK_Z1))),
    ]
    bm.verts.ensure_lookup_table()

    for i in range(4):
        j = (i + 1) % 4
        bm.faces.new((base[i], base[j], top[j], top[i]))
    bm.faces.new(tuple(reversed(base)))
    bm.faces.new(top)

    _finalize(bm)
    return scene.mesh_object("Shank", bm)


def _build_head():
    """Extruded knuckle-head contour, beveled on its vertical corner edges.

    Only the 10 vertical ("side") edges -- each shared by exactly the two
    side quads either side of a contour point -- are beveled. The front and
    back cap boundary edges (which include the three contour points sitting
    exactly on the z=0.600 pulling-face plane) are left untouched, so the
    flat pulling face is unaffected by the bevel.
    """
    bm = bmesh.new()
    count = len(HEAD_CONTOUR)
    front = []
    back = []
    for x, z in HEAD_CONTOUR:
        front.append(bm.verts.new(scene.to_blender((x, HEAD_Y_HALF, z))))
        back.append(bm.verts.new(scene.to_blender((x, -HEAD_Y_HALF, z))))
    bm.verts.ensure_lookup_table()

    side_edges = [bm.edges.new((front[i], back[i])) for i in range(count)]

    for i in range(count):
        j = (i + 1) % count
        bm.faces.new((front[i], front[j], back[j], back[i]))
    bm.faces.new(front)
    bm.faces.new(tuple(reversed(back)))

    bmesh.ops.bevel(
        bm,
        geom=side_edges,
        offset=HEAD_BEVEL_OFFSET,
        offset_type="OFFSET",
        segments=HEAD_BEVEL_SEGMENTS,
        affect="EDGES",
    )

    _finalize(bm)
    return scene.mesh_object("Head", bm)


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


def _build_knuckle_pin():
    bm = bmesh.new()
    cx, cy, cz = KNUCKLE_PIN_CENTRE
    half = KNUCKLE_PIN_HEIGHT * 0.5
    _append_capsule_segment(
        bm, (cx, cy - half, cz), (cx, cy + half, cz),
        KNUCKLE_PIN_RADIUS, KNUCKLE_PIN_SEGMENTS)
    _finalize(bm)
    return scene.mesh_object("KnucklePin", bm)


def _build_cut_lever():
    """Bent rod as a segmented sweep: one capsule cylinder per polyline leg.

    Each leg is its own closed manifold shell; legs share exact endpoint
    positions so consecutive shells butt together with no gap. This keeps
    the whole lever a single mesh object without needing a mitred join.
    """
    bm = bmesh.new()
    for i in range(len(CUT_LEVER_POINTS) - 1):
        _append_capsule_segment(
            bm, CUT_LEVER_POINTS[i], CUT_LEVER_POINTS[i + 1],
            CUT_LEVER_RADIUS, CUT_LEVER_SEGMENTS)
    _finalize(bm)
    return scene.mesh_object("CutLever", bm)


def build_meshes():
    """Five CouplerPivot-local meshes for one unbranded knuckle coupler.

    Authored for the canonical +Z-outward orientation only; the node table
    (Task 10) places/mirrors the rear coupler.
    """
    return {
        "DraftGear": _build_draft_gear(),
        "Shank": _build_shank(),
        "Head": _build_head(),
        "KnucklePin": _build_knuckle_pin(),
        "CutLever": _build_cut_lever(),
    }
