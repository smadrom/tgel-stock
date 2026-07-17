"""High-poly bake-source generation (Task 12).

``build_bake_sources`` gives every mesh object in an (post-UV) assembly a
hidden high-poly duplicate that downstream Cycles baking (``bake.py``) reads
detail from via selected-to-active:

- a ``BEVEL`` modifier (width 0.012, segments 2, angle-limited to 52 deg) on
  every duplicate, rounding hard edges so normal/curvature bakes have real
  detail to capture;
- an additional ``SUBSURF`` modifier (level 1) on parts whose short name
  (the segment after the last ``.`` in the object's dotted contract name --
  e.g. ``LongHood`` in ``Visuals.Body.LongHood``) ends with one of
  ``SUBSURF_SUFFIXES`` -- the organic/curved parts (tank, hood tops);
- bolt-greeble icospheres (radius 0.008) merged directly into the
  duplicate's own mesh, instanced on a DETERMINISTIC lattice derived from
  the part's own local bounding-box edges. Positions use a golden-ratio
  additive (Weyl) sequence seeded by the object's index in the sorted key
  order of ``objects`` -- deterministic and reproducible across runs with no
  ``random`` module and no Date-like API, per the task brief.

Every duplicate is linked ONLY into a dedicated ``BakeSources`` collection
(never the default scene collection) with ``hide_render = True`` so it never
contributes to a normal render or export; ``bake.py`` briefly flips
``hide_render`` off for the one source object it is actively baking from,
then restores it. ``cleanup_bake_sources`` removes the collection and every
object/mesh it owns once baking is done.
"""

import math

import bmesh
import bpy
from mathutils import Matrix

BEVEL_WIDTH = 0.012
BEVEL_SEGMENTS = 2
BEVEL_ANGLE_LIMIT = math.radians(52)

SUBSURF_SUFFIXES = ("LongHood", "FuelTank", "GabledRoof", "Head")
SUBSURF_LEVELS = 1

GOLDEN_RATIO = 0.6180339887498949  # (sqrt(5) - 1) / 2

BOLT_RADIUS = 0.008
BOLT_ICOSPHERE_SUBDIVISIONS = 1
BOLTS_PER_OBJECT = 6
# Bolts sit inset from the bounding-box edge endpoints so they never land
# exactly on a corner (where three edges meet and greebles would overlap).
BOLT_EDGE_MARGIN = 0.12
# Skip bolt greebles on parts smaller than the bolts themselves (nuts,
# thin rods) -- a lattice on a part this small would just be noise.
BOLT_MIN_PART_EXTENT = BOLT_RADIUS * 4.0

BAKE_SOURCES_COLLECTION = "BakeSources"
BAKE_SOURCE_SUFFIX = ".hipoly"


def _get_or_create_collection(name):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def _short_name(full_name):
    return full_name.rsplit(".", 1)[-1]


def _weyl_sequence(seed_index, count):
    """Deterministic values in [0, 1) via a golden-ratio additive (Weyl)
    sequence seeded by ``seed_index``: no ``random`` module, no Date-like
    API -- just ``frac(seed_index * phi + i * phi)`` for i in
    ``range(count)``, which is uniformly distributed and fully
    reproducible."""
    base = (seed_index * GOLDEN_RATIO) % 1.0
    return [(base + (i + 1) * GOLDEN_RATIO) % 1.0 for i in range(count)]


def _duplicate_object(obj):
    dup_mesh = obj.data.copy()
    dup = obj.copy()
    dup.data = dup_mesh
    dup.name = obj.name + BAKE_SOURCE_SUFFIX
    dup.modifiers.clear()
    return dup


def _add_bevel_modifier(obj):
    mod = obj.modifiers.new(name="TGEL_Bevel", type='BEVEL')
    mod.width = BEVEL_WIDTH
    mod.segments = BEVEL_SEGMENTS
    mod.limit_method = 'ANGLE'
    mod.angle_limit = BEVEL_ANGLE_LIMIT
    return mod


def _add_subsurf_modifier(obj):
    mod = obj.modifiers.new(name="TGEL_Subsurf", type='SUBSURF')
    mod.levels = SUBSURF_LEVELS
    mod.render_levels = SUBSURF_LEVELS
    return mod


def _mesh_local_bounds(mesh):
    if len(mesh.vertices) == 0:
        return None
    xs = [v.co.x for v in mesh.vertices]
    ys = [v.co.y for v in mesh.vertices]
    zs = [v.co.z for v in mesh.vertices]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _bbox_edges(bmin, bmax):
    """The 12 axis-aligned edges of the box spanned by ``bmin``/``bmax``, as
    (p0, p1) point-tuple pairs -- the "panel/rib edge" lattice bolts are
    instanced along."""
    xs, ys, zs = (bmin[0], bmax[0]), (bmin[1], bmax[1]), (bmin[2], bmax[2])
    edges = []
    for y in ys:
        for z in zs:
            edges.append(((xs[0], y, z), (xs[1], y, z)))
    for x in xs:
        for z in zs:
            edges.append(((x, ys[0], z), (x, ys[1], z)))
    for x in xs:
        for y in ys:
            edges.append(((x, y, zs[0]), (x, y, zs[1])))
    return edges


def _append_icosphere(bm, centre):
    bmesh.ops.create_icosphere(
        bm, subdivisions=BOLT_ICOSPHERE_SUBDIVISIONS, radius=BOLT_RADIUS,
        matrix=Matrix.Translation(centre), calc_uvs=False)


def _add_bolt_greebles(dup, seed_index):
    """Merges deterministic bolt-greeble icospheres directly into ``dup``'s
    mesh, positioned along its own bounding-box edge lattice."""
    bounds = _mesh_local_bounds(dup.data)
    if bounds is None:
        return
    bmin, bmax = bounds
    extent = max(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
    if extent < BOLT_MIN_PART_EXTENT:
        return

    edges = _bbox_edges(bmin, bmax)
    offsets = _weyl_sequence(seed_index, BOLTS_PER_OBJECT)

    bm = bmesh.new()
    bm.from_mesh(dup.data)

    for t in offsets:
        edge_index = int(t * len(edges)) % len(edges)
        p0, p1 = edges[edge_index]
        edge_t = BOLT_EDGE_MARGIN + t * (1.0 - 2.0 * BOLT_EDGE_MARGIN)
        centre = tuple(p0[axis] + (p1[axis] - p0[axis]) * edge_t for axis in range(3))
        _append_icosphere(bm, centre)

    bm.to_mesh(dup.data)
    bm.free()
    dup.data.update()


def build_bake_sources(objects, seed=0):
    """Builds one hidden high-poly duplicate per mesh in ``objects`` (a
    ``{name: bpy Object}`` map, as in ``Assembly.objects``), returning a map
    of the same keys to the duplicate bpy Objects. Duplicates live in the
    ``BakeSources`` collection with ``hide_render = True``; see the module
    docstring for the bevel/subsurf/bolt-greeble treatment and
    ``cleanup_bake_sources`` for teardown.

    ``seed`` offsets the bolt-greeble Weyl sequence (callers with a recipe
    in hand pass ``recipe.seed``); each part additionally indexes the
    sequence by its position in the sorted key order, so lattices stay
    deterministic AND distinct per part for any fixed seed."""
    collection = _get_or_create_collection(BAKE_SOURCES_COLLECTION)
    sources = {}

    for index, name in enumerate(sorted(objects.keys())):
        obj = objects[name]
        dup = _duplicate_object(obj)
        collection.objects.link(dup)
        dup.hide_render = True

        _add_bevel_modifier(dup)

        short = _short_name(name)
        if short.endswith(SUBSURF_SUFFIXES):
            _add_subsurf_modifier(dup)

        _add_bolt_greebles(dup, seed + index)

        sources[name] = dup

    return sources


def _purge_bake_temp_materials():
    """Removes every throwaway ``TGEL.BakeTemp.*`` material ``bake.py``
    created (their embedded node trees go with them). ``bake_detail``
    restores each mesh's original material slots before returning, so these
    materials are unreferenced by the time cleanup runs."""
    from .bake import BAKE_TEMP_MATERIAL_PREFIX

    for material in [m for m in bpy.data.materials
                     if m.name.startswith(BAKE_TEMP_MATERIAL_PREFIX)]:
        bpy.data.materials.remove(material, do_unlink=True)


def cleanup_bake_sources():
    """Removes the ``BakeSources`` collection, every high-poly object and
    mesh it owns, and every throwaway ``TGEL.BakeTemp.*`` bake material.
    Safe to call even if ``build_bake_sources``/``bake_detail`` were never
    called (no-op)."""
    _purge_bake_temp_materials()

    collection = bpy.data.collections.get(BAKE_SOURCES_COLLECTION)
    if collection is None:
        return

    for obj in list(collection.objects):
        mesh = obj.data
        collection.objects.unlink(obj)
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    bpy.data.collections.remove(collection, do_unlink=True)
