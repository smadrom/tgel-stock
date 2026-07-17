"""Parametric rail wheel and wheelset part builders.

Space convention (controller-adjudicated): wheel meshes are authored
CENTRE-ORIGIN and untranslated in wheel-local space. The axle runs along
local +X, the tread-and-flange mesh spans x in [-WHEEL_WIDTH/2,
+WHEEL_WIDTH/2] = [-0.0675, +0.0675], with the flange side at +0.0675 and
the outer (field-side) tread face at -0.0675. Placement lives ONLY in the
node table: the assembler (Task 10) places the WheelLeft/WheelRight nodes
at node-local x = -/+(BACK_TO_BACK + WHEEL_WIDTH)/2 = -/+0.7415, with a
180 degree yaw on the right, so the flange faces land at
-/+(0.7415 - 0.0675) = -/+0.674 and back-to-back comes out at the frozen
1.348 m.

The lathe profile below is written with x measured from the FLANGE face
(0.0 at the flange tip, increasing toward the tread's outer face at
WHEEL_WIDTH), the same numbering the design brief uses. Each profile point
maps into centre-origin wheel-local space once via
``WHEEL_WIDTH / 2 - profile_x``.
"""

import math

import bmesh

from .. import recipe
from .. import scene

SEGMENTS_DEFAULT = 32
# Hub boss radius 0.115 (controller-amended; supersedes the brief's 0.15,
# which poked out through the dished web). It sits below the web minimum
# radius (0.12) so the boss wall stays inside the web surface, and above
# the hub bore cap radius (0.085) so it still reads as a raised hub ring.
HUB_BOSS_RADIUS = 0.115
AXLE_VISUAL_RADIUS = 0.055


def _profile_points(radius):
    """Lathe profile as (x_from_flange, r) pairs, flange tip to hub bore.

    x is measured from the flange face (0.0) toward the tread's outer face
    (WHEEL_WIDTH) and is confined to [0, WHEEL_WIDTH] so the revolved
    tread-and-flange mesh spans exactly +/-WHEEL_WIDTH/2 after the
    centre-origin mapping (see module docstring).
    """
    return (
        (0.000, radius + 0.025),          # flange tip
        (0.008, radius + 0.016),          # flange root fillet, bevel 1
        (0.022, radius + 0.003),          # flange root fillet, bevel 2
        (0.043, radius),                  # tread start
        (0.135, radius - 0.092 * 0.05),   # tread end / outer face, 1:20 conicity
        (0.135, radius - 0.060),          # outer face, straight down the rim
        (0.120, radius - 0.075),          # rim inner, chamfered back inboard
        (0.0675, 0.12),                   # dished web down to hub
        (0.040, 0.085),                   # hub bore edge
    )


def _append_tread_and_flange(bm, radius, segments):
    """Revolves the lathe profile into bm, centre-origin, axle along +X."""
    profile_verts = []
    for profile_x, r in _profile_points(radius):
        local_x = recipe.WHEEL_WIDTH / 2.0 - profile_x
        profile_verts.append(bm.verts.new(scene.to_blender((local_x, r, 0.0))))
    bm.verts.ensure_lookup_table()

    profile_edges = [
        bm.edges.new((profile_verts[i], profile_verts[i + 1]))
        for i in range(len(profile_verts) - 1)
    ]

    spin_axis = scene.to_blender((1.0, 0.0, 0.0))
    bmesh.ops.spin(
        bm,
        geom=profile_verts + profile_edges,
        cent=(0.0, 0.0, 0.0),
        axis=spin_axis,
        angle=math.radians(360.0),
        steps=segments,
        use_duplicate=False,
    )


def _append_ring_cylinder(bm, centre_x, radius, width, segments):
    """Adds a closed cylinder (side quads + two ngon caps) into bm."""
    half = width * 0.5
    ring_a = []
    ring_b = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        y = radius * math.cos(theta)
        z = radius * math.sin(theta)
        ring_a.append(bm.verts.new(scene.to_blender((centre_x - half, y, z))))
        ring_b.append(bm.verts.new(scene.to_blender((centre_x + half, y, z))))
    bm.verts.ensure_lookup_table()
    for i in range(segments):
        j = (i + 1) % segments
        bm.faces.new((ring_a[i], ring_a[j], ring_b[j], ring_b[i]))
    bm.faces.new(ring_a)
    bm.faces.new(tuple(reversed(ring_b)))


def _append_hub_boss(bm, segments):
    """Adds the hub boss cylinder, centred on the origin (mid-width).

    Width WHEEL_WIDTH + 0.03 makes the boss protrude 15 mm past both the
    flange face (+WHEEL_WIDTH/2) and the outer face (-WHEEL_WIDTH/2),
    symmetrically.
    """
    _append_ring_cylinder(bm, 0.0, HUB_BOSS_RADIUS, recipe.WHEEL_WIDTH + 0.03, segments)


def _finalize(bm):
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    bmesh.ops.holes_fill(bm, edges=bm.edges[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()


def build_wheel(name, radius, segments=SEGMENTS_DEFAULT):
    """One wheel: 1:20 conical tread, 25 mm flange, dished web, hub boss.

    Centre-origin wheel-local space: axle along local +X, flange side at
    x=+WHEEL_WIDTH/2, outer tread face at x=-WHEEL_WIDTH/2. The hub boss
    protrudes 15 mm past both faces, keeping the mesh symmetric about the
    origin.
    """
    bm = bmesh.new()
    _append_tread_and_flange(bm, radius, segments)
    _append_hub_boss(bm, segments)
    _finalize(bm)
    return scene.mesh_object(name, bm)


def _build_tread_and_flange_object(name, radius, segments):
    bm = bmesh.new()
    _append_tread_and_flange(bm, radius, segments)
    _finalize(bm)
    return scene.mesh_object(name, bm)


def _build_hub_object(name, segments):
    bm = bmesh.new()
    _append_hub_boss(bm, segments)
    _finalize(bm)
    return scene.mesh_object(name, bm)


def _build_axle_visual(name, seat_offset, segments):
    bm = bmesh.new()
    bm_width = seat_offset * 2.0
    _append_ring_cylinder(bm, 0.0, AXLE_VISUAL_RADIUS, bm_width, segments)
    _finalize(bm)
    return scene.mesh_object(name, bm)


def build_wheelset_meshes(prefix, radius, segments=SEGMENTS_DEFAULT):
    """Five node-local meshes for one wheelset.

    Wheel meshes are centre-origin and UNTRANSLATED; WheelLeft.* and
    WheelRight.* are identical meshes. Placement lives only in the node
    table: the assembler (Task 10) puts the wheel nodes at node-local
    x = -/+(BACK_TO_BACK + WHEEL_WIDTH)/2 = -/+0.7415 with a 180 degree
    yaw on the right (flange faces at -/+0.674, back-to-back 1.348 m).
    AxleVisual is the one mesh with intrinsic extent: a cylinder along X
    spanning the wheel-centre seats at +/-0.7415.
    """
    seat_offset = (recipe.BACK_TO_BACK + recipe.WHEEL_WIDTH) / 2.0

    axle = _build_axle_visual(f"{prefix}.AxleVisual", seat_offset, segments)

    left_tread = _build_tread_and_flange_object(
        f"{prefix}.WheelLeft.TreadAndFlange", radius, segments)
    left_hub = _build_hub_object(f"{prefix}.WheelLeft.Hub", segments)

    right_tread = _build_tread_and_flange_object(
        f"{prefix}.WheelRight.TreadAndFlange", radius, segments)
    right_hub = _build_hub_object(f"{prefix}.WheelRight.Hub", segments)

    return {
        "AxleVisual": axle,
        "WheelLeft.TreadAndFlange": left_tread,
        "WheelLeft.Hub": left_hub,
        "WheelRight.TreadAndFlange": right_tread,
        "WheelRight.Hub": right_hub,
    }
