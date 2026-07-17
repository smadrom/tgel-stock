"""Cycles CPU detail baking (Task 12): normal / AO / curvature atlas bakes.

``bake_detail`` bakes three whole-atlas images shared across EVERY object in
``objects`` (a post-UV, post-``highpoly.build_bake_sources`` assembly): a
tangent-space ``normal.png``, an ``ao.png``, and a Geometry-Pointiness
``curvature.png``. Each is ONE shared image that every object's high-poly
source (``sources``, from ``highpoly.build_bake_sources``) bakes onto in
turn via selected-to-active.

Material isolation: baking needs an active Image Texture node on the
target and a Pointiness->Emission shading path on the source, but objects
may already carry REAL materials (Task 13 authors them), and mesh copies
share material datablocks -- mutating whatever sits in slot 0 would
silently corrupt a fleet material for every referencing object. So
``bake_detail`` NEVER touches pre-existing materials: it creates dedicated
throwaway materials under the reserved ``TGEL.BakeTemp.`` prefix, swaps
them into every material slot only for the duration of the bake (recording
each mesh's original slot contents), and restores the originals in the same
``finally`` block that restores ``hide_render`` flags.
``highpoly.cleanup_bake_sources()`` purges all ``TGEL.BakeTemp.*``
materials afterwards, leaving zero orphaned datablocks.

Empirically verified against pinned Blender 5.1.2 (--background):

- ``bpy.ops.object.bake`` runs headless with no context override; it
  returns ``{'FINISHED'}`` and writes into the active Image Texture node of
  the ACTIVE object's material.
- Selected-to-active AO: the active (target) object does NOT occlude its
  own coincident high-poly source (an isolated pair bakes AO ~= 1.0 on a
  convex part), but ANY other render-visible object does. Assembly meshes
  are node-local and all sit piled at the Blender origin, so leaving
  sibling parts render-visible drives AO to 0.000 across the entire island
  set (measured). ``bake_detail`` therefore hides every object from render
  and unhides exactly the {target, source} pair for each bake.
- ``Image.save()`` on a float-buffer image writes a 16-bit PNG (verified by
  IHDR bit-depth inspection), so all three maps are saved 16-bit; the
  interface requires 16-bit for the normal map, and 16-bit AO/curvature is
  a harmless superset.
- Images load back with an sRGB colorspace by default and ``Image.pixels``
  returns colorspace-converted values; consumers (and the test) must set
  ``colorspace_settings.name = 'Non-Color'`` on loaded images to read the
  raw stored values.

Whole-atlas statistics and ``use_clear`` (disclosed deviation): the packed
atlas has ~15% UV coverage (Task 11's measured reality), so a black cleared
background would swamp any whole-image statistic (a correct neutral-blue
normal atlas would average B ~= 0.15 against black). Instead of letting the
first bake clear the image, every atlas is PRE-FILLED with its neutral
padding color (normal: (0.5, 0.5, 1.0) tangent identity; AO / curvature:
0.5 mid-gray -- unused texels are never sampled beyond the 16 px margin, so
their value is cosmetic) and ALL bakes run with ``use_clear=False``.

Engine: Cycles, ``device='CPU'``, ``use_denoising=False``. ``samples`` is
16 for the normal and curvature (Pointiness-as-Emission) passes; AO is
noisier per sample so it runs at ``AO_SAMPLES`` = 32 (the brief suggested
64; 32 is the brief's own disclosed wall-clock fallback, and per-part AO on
isolated bake pairs converges fast).
"""

import os

import bpy

from . import assert_clean_path

DEVICE = 'CPU'
DEFAULT_SAMPLES = 16
AO_SAMPLES = 32
USE_DENOISING = False

USE_SELECTED_TO_ACTIVE = True
CAGE_EXTRUSION = 0.03
BAKE_MARGIN = 16
NORMAL_SPACE = 'TANGENT'

NORMAL_FILENAME = "normal.png"
AO_FILENAME = "ao.png"
CURVATURE_FILENAME = "curvature.png"

NORMAL_NEUTRAL = (0.5, 0.5, 1.0, 1.0)
GRAY_NEUTRAL = (0.5, 0.5, 0.5, 1.0)

# Reserved prefix for throwaway bake materials. bake_detail restores every
# mesh's original slot contents before returning, and
# highpoly.cleanup_bake_sources() purges all materials under this prefix.
BAKE_TEMP_MATERIAL_PREFIX = "TGEL.BakeTemp."

_IMAGE_NODE_NAME = "TGEL_BakeTarget"


def _configure_cycles(scene):
    scene.render.engine = 'CYCLES'
    scene.cycles.device = DEVICE
    scene.cycles.use_denoising = USE_DENOISING
    scene.render.bake.use_selected_to_active = USE_SELECTED_TO_ACTIVE
    scene.render.bake.cage_extrusion = CAGE_EXTRUSION
    scene.render.bake.margin = BAKE_MARGIN
    scene.render.bake.normal_space = NORMAL_SPACE


def _make_bake_target_material(name):
    """A dedicated throwaway target material whose active node is the Image
    Texture node Blender writes IMAGE_TEXTURES-target bakes into."""
    material = bpy.data.materials.new(name=f"{BAKE_TEMP_MATERIAL_PREFIX}Target.{name}")
    material.use_nodes = True
    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    node.name = _IMAGE_NODE_NAME
    material.node_tree.nodes.active = node
    return material


def _make_pointiness_material(name):
    """A dedicated throwaway source material whose ONLY shading path is
    Geometry.Pointiness -> Emission -> Material Output -- baked as EMIT for
    the curvature pass. Harmless for the NORMAL/AO passes, which read
    geometry, not the source's shading."""
    material = bpy.data.materials.new(
        name=f"{BAKE_TEMP_MATERIAL_PREFIX}Pointiness.{name}")
    material.use_nodes = True
    node_tree = material.node_tree
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)

    output = node_tree.nodes.new("ShaderNodeOutputMaterial")
    emission = node_tree.nodes.new("ShaderNodeEmission")
    geometry = node_tree.nodes.new("ShaderNodeNewGeometry")
    node_tree.links.new(geometry.outputs["Pointiness"], emission.inputs["Color"])
    node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _assign_bake_material(obj, material):
    """Swaps ``material`` into EVERY material slot of ``obj``'s mesh
    (IMAGE_TEXTURES bakes write through the active node of each slot's
    material, so all slots must point at the bake material) and returns the
    original slot contents for ``_restore_materials``: a list of the
    original per-slot materials, or None when the mesh had no slots and one
    was appended."""
    mesh = obj.data
    if len(mesh.materials) == 0:
        mesh.materials.append(material)
        return None
    originals = [mesh.materials[i] for i in range(len(mesh.materials))]
    for i in range(len(mesh.materials)):
        mesh.materials[i] = material
    return originals


def _restore_materials(obj, originals):
    mesh = obj.data
    if originals is None:
        mesh.materials.clear()
        return
    for i, original in enumerate(originals):
        mesh.materials[i] = original


def _new_atlas_image(name, atlas_px, fill_color):
    image = bpy.data.images.new(name, atlas_px, atlas_px, alpha=False, float_buffer=True)
    image.colorspace_settings.name = 'Non-Color'
    image.generated_color = fill_color
    return image


def _run_bake_pass(names, objects, sources, image_nodes, image, bake_type, label):
    baked = 0
    for name in names:
        target = objects.get(name)
        source = sources.get(name)
        if target is None or source is None:
            continue
        if len(target.data.polygons) == 0 or len(source.data.polygons) == 0:
            continue

        image_nodes[name].image = image

        bpy.ops.object.select_all(action='DESELECT')
        target.hide_render = False
        source.hide_render = False
        try:
            source.select_set(True)
            target.select_set(True)
            bpy.context.view_layer.objects.active = target

            result = bpy.ops.object.bake(
                type=bake_type,
                target='IMAGE_TEXTURES',
                use_selected_to_active=USE_SELECTED_TO_ACTIVE,
                cage_extrusion=CAGE_EXTRUSION,
                margin=BAKE_MARGIN,
                normal_space=NORMAL_SPACE,
                use_clear=False,
            )
            if 'FINISHED' not in result:
                raise RuntimeError(
                    f"bpy.ops.object.bake({bake_type}) on '{name}' returned {result}")
        finally:
            source.select_set(False)
            target.select_set(False)
            target.hide_render = True
            source.hide_render = True

        baked += 1
        print(f"[bake] {label} {baked}: {name}", flush=True)


def bake_detail(objects, sources, out_dir, atlas_px=4096):
    """Bakes normal/AO/curvature whole-atlas images shared across every
    object in ``objects``, reading high-poly detail from ``sources`` (as
    returned by ``highpoly.build_bake_sources``) via selected-to-active.
    Writes ``normal.png`` (16-bit, tangent-space), ``ao.png`` and
    ``curvature.png`` into ``out_dir`` and returns
    ``{"normal": path, "ao": path, "curvature": path}``.

    Each bake renders with ONLY the current {target, source} pair visible
    (see module docstring: assembly meshes are piled at the origin, so
    per-pair isolation is the only sane AO semantics); every object's
    original ``hide_render`` flag AND original material slots are restored
    before returning (bakes run against dedicated ``TGEL.BakeTemp.*``
    materials; pre-existing materials are never mutated -- see the module
    docstring).

    Note: the SCENE is deliberately left configured for Cycles CPU baking
    on return (render engine, bake settings, sample counts) -- this is a
    batch bake tool and callers (Tasks 13/15) run it inside throwaway
    --background sessions, so no scene-settings restore is attempted.
    """
    assert_clean_path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    scene = bpy.context.scene
    _configure_cycles(scene)

    names = sorted(objects.keys())

    original_hide = {}
    original_materials = []
    image_nodes = {}
    try:
        for obj_map in (objects, sources):
            for obj in obj_map.values():
                original_hide[obj.name] = obj.hide_render
                obj.hide_render = True

        for name in names:
            target = objects.get(name)
            source = sources.get(name)
            if target is None or source is None:
                continue
            target_material = _make_bake_target_material(name)
            image_nodes[name] = target_material.node_tree.nodes[_IMAGE_NODE_NAME]
            original_materials.append(
                (target, _assign_bake_material(target, target_material)))
            source_material = _make_pointiness_material(name)
            original_materials.append(
                (source, _assign_bake_material(source, source_material)))

        normal_image = _new_atlas_image("TGEL_Normal", atlas_px, NORMAL_NEUTRAL)
        ao_image = _new_atlas_image("TGEL_AO", atlas_px, GRAY_NEUTRAL)
        curvature_image = _new_atlas_image("TGEL_Curvature", atlas_px, GRAY_NEUTRAL)

        scene.cycles.samples = DEFAULT_SAMPLES
        _run_bake_pass(names, objects, sources, image_nodes, normal_image,
                       'NORMAL', "normal")

        scene.cycles.samples = AO_SAMPLES
        _run_bake_pass(names, objects, sources, image_nodes, ao_image, 'AO', "ao")

        scene.cycles.samples = DEFAULT_SAMPLES
        _run_bake_pass(names, objects, sources, image_nodes, curvature_image,
                       'EMIT', "curvature")
    finally:
        for obj, originals in original_materials:
            _restore_materials(obj, originals)
        for obj_map in (objects, sources):
            for obj in obj_map.values():
                if obj.name in original_hide:
                    obj.hide_render = original_hide[obj.name]

    paths = {}
    for key, image, filename in (
        ("normal", normal_image, NORMAL_FILENAME),
        ("ao", ao_image, AO_FILENAME),
        ("curvature", curvature_image, CURVATURE_FILENAME),
    ):
        path = os.path.join(out_dir, filename)
        assert_clean_path(path)
        image.filepath_raw = path
        image.file_format = 'PNG'
        image.save()
        paths[key] = path

    return paths
