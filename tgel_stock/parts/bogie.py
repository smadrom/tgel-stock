"""Parametric bogie frame part builders.

Space convention (controller-adjudicated): all frame meshes are authored
Frame-local, i.e. bogie-pivot-local. The Frame node sits at identity under
the Bogie pivot, and the pivot itself is mounted at ``pivot_height`` above
the railhead by the node table, so railhead level in this local space is at
y = -pivot_height. Nothing in these meshes may dip below that line.

Layout table (blockout-proven anchors; x = across the track, y = up,
z = along the track, Left = negative x, Leading = positive z):

- Bolster: centre (0, 0, 0), size (2.25, 0.18, 0.36), with a centre-plate
  disc merged into the middle.
- SideFrameLeft/SideFrameRight: centre (-/+1.00, frameY, 0) where
  frameY = wheel_radius + 0.19 - pivot_height; outer size
  (0.18, 0.34, wheelbase + 0.42); modelled as an extruded U-profile with
  inset relief faces (no booleans).
- CrossMemberFront/CrossMemberRear: centres (0, frameY, +/-wheelbase/2),
  size (2.05, 0.16, 0.18); plain boxes.
- AxleboxLeft/RightLeading/Trailing: centres (-/+1.00, axleboxY,
  +/-wheelbase/2) with axleboxY = wheel_radius - pivot_height, size
  (0.30, 0.30, 0.34); a front cover disc is merged onto the outboard face.
- SpringLeft/RightFront/Rear: centres (-/+1.00, springY, +/-0.34) with
  springY = wheel_radius + 0.41 - pivot_height; coil-suggesting fluted
  cylinders, r=0.08, h=0.30.
"""

import math

import bmesh

from .. import scene

SEGMENTS_DEFAULT = 16
# 0.16 keeps the disc fully inside the bolster's fore/aft half-extent
# (0.36 / 2 = 0.18); the blockout's r~0.20 overhung the block by 0.02.
CENTRE_PLATE_RADIUS = 0.16
CENTRE_PLATE_HEIGHT = 0.05
AXLEBOX_COVER_RADIUS = 0.10
AXLEBOX_COVER_HEIGHT = 0.04
SPRING_RADIUS = 0.08
SPRING_HEIGHT = 0.30
SIDE_FRAME_INSET_DEPTH = 0.045
SIDE_FRAME_INSET_MARGIN = 0.10


def _append_disc_cylinder(bm, centre_unity, radius, height, axis, segments):
    """Adds a closed cylinder into bm, centred at centre_unity.

    ``axis`` selects which vehicle-space axis the cylinder's length runs
    along: 'y' for a disc lying flat (used for centre plates / covers set
    into a horizontal face) or 'z' for a disc facing along the track
    (axlebox outboard covers face +/-x, so this helper always builds the
    cylinder with its axis along the given vehicle-space axis and callers
    orient centres/offsets accordingly).
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


def _finalize(bm):
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    bmesh.ops.holes_fill(bm, edges=bm.edges[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()


def _box_bmesh(bm, centre_unity, size_unity):
    """Appends an axis-aligned box (vehicle space) into an existing bm.

    Not a duplicate of scene.box: it exists for parts that merge extra
    geometry into the same mesh (bolster centre plate, axlebox covers,
    side-frame recess). Simple standalone boxes must use scene.box.
    """
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


def _build_bolster(centre_y, segments=SEGMENTS_DEFAULT):
    """Bolster box with a centre-plate disc merged onto its underside."""
    bm = bmesh.new()
    _box_bmesh(bm, (0.0, centre_y, 0.0), (2.25, 0.18, 0.36))
    plate_centre = (0.0, centre_y - 0.18 * 0.5 - CENTRE_PLATE_HEIGHT * 0.5, 0.0)
    _append_disc_cylinder(bm, plate_centre, CENTRE_PLATE_RADIUS, CENTRE_PLATE_HEIGHT, "y", segments)
    _finalize(bm)
    return scene.mesh_object("Bolster", bm)


def _build_side_frame(name, centre_x, centre_y, length_z):
    """Extruded U-profile side frame: outer box with a recessed inner panel.

    Built as a box, then the inner (track-centreline-facing) long face is
    replaced with a shallow recessed panel: a picture-frame ring of 4 quads
    running from the original face boundary in to a smaller rectangle set
    back by ``SIDE_FRAME_INSET_DEPTH``, closed off by a floor quad. This is
    modelled explicitly with hand-placed vertices (no booleans, and no
    reliance on bmesh.ops.inset_individual's ring/face return semantics,
    which turned out to alias the original boundary verts and collapse
    under remove_doubles) -- the resulting geometry suggests the pedestal
    openings of a cast side frame while staying a single closed manifold.
    """
    size_x, size_y, size_z = 0.18, 0.34, length_z
    sy, sz = size_y * 0.5, size_z * 0.5
    bm = bmesh.new()
    _box_bmesh(bm, (centre_x, centre_y, 0.0), (size_x, size_y, size_z))
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # Locate the inner (track-centreline-facing) long face by the actual
    # unity-space x position of its verts. Position is unambiguous and
    # correct for both SideFrameLeft (x<0, inner face at larger x) and
    # SideFrameRight (x>0, inner face at smaller x) -- unlike face.normal,
    # which is not yet populated on a freshly built bmesh.
    inner_sign = 1.0 if centre_x < 0.0 else -1.0
    inner_x = centre_x + inner_sign * size_x * 0.5
    inner_face = None
    for face in bm.faces:
        if all(abs(scene.to_unity(v.co)[0] - inner_x) < 1e-6 for v in face.verts):
            inner_face = face
            break
    if inner_face is None:
        raise RuntimeError(f"{name}: could not locate inner side-frame face")

    boundary_verts = list(inner_face.verts)
    bm.faces.remove(inner_face)

    margin = min(SIDE_FRAME_INSET_MARGIN, sy * 0.5, sz * 0.5)
    pocket_x = inner_x - inner_sign * SIDE_FRAME_INSET_DEPTH

    pocket_verts = []
    for bv in boundary_verts:
        _outer_x, oy, oz = scene.to_unity(bv.co)
        iy = oy - margin if oy > centre_y else oy + margin
        iz = oz - margin if oz > 0.0 else oz + margin
        pocket_verts.append(bm.verts.new(scene.to_blender((pocket_x, iy, iz))))
    bm.verts.ensure_lookup_table()

    count = len(boundary_verts)
    for i in range(count):
        j = (i + 1) % count
        bm.faces.new((boundary_verts[i], boundary_verts[j], pocket_verts[j], pocket_verts[i]))
    bm.faces.new(pocket_verts)

    _finalize(bm)
    return scene.mesh_object(name, bm)


def _build_cross_member(name, centre_z, centre_y):
    return scene.box(name, (0.0, centre_y, centre_z), (2.05, 0.16, 0.18))


def _build_axlebox(name, centre_x, centre_y, centre_z, segments=SEGMENTS_DEFAULT):
    """Axlebox with a front cover disc merged onto the outboard face."""
    size = (0.30, 0.30, 0.34)
    bm = bmesh.new()
    _box_bmesh(bm, (centre_x, centre_y, centre_z), size)
    outboard_sign = -1.0 if centre_x < 0.0 else 1.0
    cover_centre = (centre_x + outboard_sign * (size[0] * 0.5 + AXLEBOX_COVER_HEIGHT * 0.5),
                     centre_y, centre_z)
    _append_disc_cylinder(bm, cover_centre, AXLEBOX_COVER_RADIUS, AXLEBOX_COVER_HEIGHT, "x", segments)
    _finalize(bm)
    return scene.mesh_object(name, bm)


def _build_spring(name, centre_x, centre_y, centre_z, segments=SEGMENTS_DEFAULT):
    """Coil-suggesting fluted cylinder: stacked ring segments with a flute.

    Modelled as a short stack of slightly-varying-radius rings (a simple,
    deterministic "coil" silhouette) rather than a plain cylinder, kept
    well under the 1500-triangle-per-spring budget.
    """
    coils = 4
    bm = bmesh.new()
    half = SPRING_HEIGHT * 0.5
    rings = []
    for i in range(coils + 1):
        t = i / coils
        y = centre_y - half + t * SPRING_HEIGHT
        flute = 1.0 if i % 2 == 0 else 0.88
        ring = []
        for s in range(segments):
            theta = 2.0 * math.pi * s / segments
            x = centre_x + SPRING_RADIUS * flute * math.cos(theta)
            z = centre_z + SPRING_RADIUS * flute * math.sin(theta)
            ring.append(bm.verts.new(scene.to_blender((x, y, z))))
        rings.append(ring)
    bm.verts.ensure_lookup_table()

    for i in range(coils):
        ring_a = rings[i]
        ring_b = rings[i + 1]
        for s in range(segments):
            j = (s + 1) % segments
            bm.faces.new((ring_a[s], ring_a[j], ring_b[j], ring_b[s]))
    bm.faces.new(rings[0])
    bm.faces.new(tuple(reversed(rings[-1])))

    _finalize(bm)
    return scene.mesh_object(name, bm)


def build_frame_meshes(wheelbase, wheel_radius, pivot_height):
    """Thirteen Frame-local meshes for one bogie frame assembly.

    All meshes are authored in Frame-local (bogie-pivot-local) space; the
    Frame node sits at identity under the Bogie pivot, which the node table
    mounts at ``pivot_height`` above the railhead, so railhead level here is
    y = -pivot_height. Every mesh's min_y stays at or above that line.
    """
    frame_y = wheel_radius + 0.19 - pivot_height
    axlebox_y = wheel_radius - pivot_height
    spring_y = wheel_radius + 0.41 - pivot_height
    side_frame_length = wheelbase + 0.42
    half_wheelbase = wheelbase / 2.0

    meshes = {}

    meshes["Bolster"] = _build_bolster(0.0)

    meshes["SideFrameLeft"] = _build_side_frame(
        "SideFrameLeft", -1.00, frame_y, side_frame_length)
    meshes["SideFrameRight"] = _build_side_frame(
        "SideFrameRight", 1.00, frame_y, side_frame_length)

    meshes["CrossMemberFront"] = _build_cross_member(
        "CrossMemberFront", half_wheelbase, frame_y)
    meshes["CrossMemberRear"] = _build_cross_member(
        "CrossMemberRear", -half_wheelbase, frame_y)

    meshes["AxleboxLeftLeading"] = _build_axlebox(
        "AxleboxLeftLeading", -1.00, axlebox_y, half_wheelbase)
    meshes["AxleboxRightLeading"] = _build_axlebox(
        "AxleboxRightLeading", 1.00, axlebox_y, half_wheelbase)
    meshes["AxleboxLeftTrailing"] = _build_axlebox(
        "AxleboxLeftTrailing", -1.00, axlebox_y, -half_wheelbase)
    meshes["AxleboxRightTrailing"] = _build_axlebox(
        "AxleboxRightTrailing", 1.00, axlebox_y, -half_wheelbase)

    meshes["SpringLeftFront"] = _build_spring(
        "SpringLeftFront", -1.00, spring_y, 0.34)
    meshes["SpringLeftRear"] = _build_spring(
        "SpringLeftRear", -1.00, spring_y, -0.34)
    meshes["SpringRightFront"] = _build_spring(
        "SpringRightFront", 1.00, spring_y, 0.34)
    meshes["SpringRightRear"] = _build_spring(
        "SpringRightRear", 1.00, spring_y, -0.34)

    return meshes
