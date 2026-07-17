"""Wagon-chain livery and packed-weathering texture test (Task 13).

Production detail/material atlases are 4096 px (the Task 11 progress-ledger
override).  As in ``bake_test.py``, this focused background-Blender gate uses
1024 px so the Cycles CPU portion stays practical while exercising the same
UV, detail-bake and material-composition paths.

Every PNG is switched to ``Non-Color`` before ``Image.pixels`` is read.  That
ordering is a measured Blender 5.1.2 requirement: otherwise Blender silently
applies the file's display transform and the numeric channel gates are false.
"""

import os
import inspect
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from tgel_stock import assemble  # noqa: E402
from tgel_stock import bake  # noqa: E402
from tgel_stock import highpoly  # noqa: E402
from tgel_stock import materials  # noqa: E402
from tgel_stock import recipe as recipe_module  # noqa: E402
from tgel_stock import scene  # noqa: E402
from tgel_stock import uvmap  # noqa: E402

ATLAS_PX = 1024
SAMPLE_COUNT = 1000

WAGON_RECIPE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "recipes",
    "basic-box-wagon.rollingstock.json")
LOCOMOTIVE_RECIPE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "recipes",
    "basic-diesel-locomotive.rollingstock.json")
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "artifacts",
    "rolling-stock-generation", "materials_test")

failures = []

# Cheap contract guards run before the expensive 282-pair detail bake.
classification_cases = {
    "Visuals.RunningGear.Bogie__front.Wheelset__front_leading.WheelLeft.Hub":
        "runningGear",
    "Visuals.RunningGear.Bogie__front.Frame.CrossMemberFront": "runningGear",
    "Visuals.RunningGear.Bogie__rear.Frame.SideFrameLeft": "runningGear",
    "Visuals.Body.Underframe": "frame",
    "Visuals.Body.GabledRoof": "roof",
    "Visuals.Body.CabGlass": "glass",
}
for part_name, expected_category in classification_cases.items():
    actual_category = materials._part_category(part_name)
    if actual_category != expected_category:
        failures.append(
            f"category {part_name}: {actual_category} != {expected_category}")

if materials.DEFAULT_ATLAS_PX != 4096:
    failures.append(
        f"DEFAULT_ATLAS_PX {materials.DEFAULT_ATLAS_PX} != production 4096")
for function in (materials.bake_albedo, materials.bake_weathering_masks):
    default = inspect.signature(function).parameters["atlas_px"].default
    if default != 4096:
        failures.append(f"{function.__name__} atlas_px default {default} != 4096")

expected_roughness = {
    "body": 0.62,
    "roof": 0.62,
    "glass": 0.45,
    "runningGear": 0.78,
    "frame": 0.78,
}
if materials._ROUGHNESS != expected_roughness:
    failures.append(
        f"roughness bases {materials._ROUGHNESS} != {expected_roughness}")

if not (
        materials._wear_multiplier("Visuals.Body.DoorLeft") > 1.0
        and materials._wear_multiplier("Visuals.Body.HandrailLeft") > 1.0
        and materials._wear_multiplier("Visuals.Body.StirrupStepFrontLeft") > 1.0
        and materials._wear_multiplier("Visuals.Body.BoxBody") == 1.0):
    failures.append("Door/Handrail/Step wear multipliers are not emphasized")

streak_positions = np.linspace(-2.0, 2.0, 17, dtype=np.float32)
streak_side = materials._rain_streak_signal(
    streak_positions, np.ones_like(streak_positions),
    np.ones_like(streak_positions), 190702)
streak_repeat = materials._rain_streak_signal(
    streak_positions, np.ones_like(streak_positions),
    np.ones_like(streak_positions), 190702)
streak_roof = materials._rain_streak_signal(
    streak_positions, np.zeros_like(streak_positions),
    np.ones_like(streak_positions), 190702)
if not np.array_equal(streak_side, streak_repeat):
    failures.append("rain-streak signal is not deterministic for fixed seed")
if not (float(np.std(streak_side)) > 0.10):
    failures.append("rain-streak side signal lacks longitudinal variation")
if not np.all(streak_roof == 0.0):
    failures.append("rain-streak signal leaked onto non-side-facing texels")

if failures:
    print("BPY TESTS FAIL:", "; ".join(failures))
    sys.exit(1)


def _load_image_array(path):
    image = bpy.data.images.load(path)
    try:
        # Ordering is contractual: set raw-data colorspace before pixel reads.
        image.colorspace_settings.name = 'Non-Color'
        width, height = image.size
        channels = image.channels
        if channels != 4:
            raise AssertionError(
                f"{os.path.basename(path)} reload has {channels} channels, expected RGBA")
        pixels = np.empty(width * height * channels, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        return pixels.reshape(height, width, channels)
    finally:
        bpy.data.images.remove(image)


def _surface_samples(obj, image, normal_predicate, count, seed):
    """Samples ``image`` through the UV triangles of selected object faces."""
    mesh = obj.data
    mesh.calc_loop_triangles()
    uv_data = mesh.uv_layers.active.data
    candidates = []
    for triangle in mesh.loop_triangles:
        polygon = mesh.polygons[triangle.polygon_index]
        if not normal_predicate(polygon.normal):
            continue
        candidates.append(np.array(
            [uv_data[loop_index].uv[:] for loop_index in triangle.loops],
            dtype=np.float64))

    if not candidates:
        raise AssertionError(f"No matching UV triangles on {obj.name}")

    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    values = []
    while len(values) < count:
        uv_triangle = candidates[len(values) % len(candidates)]
        # Keep samples away from triangle borders so integer texel lookup
        # cannot fall outside the rasterized face because of rounding.
        a, b = rng.uniform(0.08, 0.84, size=2)
        if a + b > 0.92:
            a, b = 0.92 - a, 0.92 - b
        c = 1.0 - a - b
        uv = a * uv_triangle[0] + b * uv_triangle[1] + c * uv_triangle[2]
        x = int(np.clip(np.floor(uv[0] * width), 0, width - 1))
        y = int(np.clip(np.floor(uv[1] * height), 0, height - 1))
        values.append(image[y, x])
    return np.asarray(values)


def _occupied_uv_mask(objects, width, height):
    occupied = np.zeros((height, width), dtype=bool)
    for name in sorted(objects):
        for x, y, _height, _up, _side, _longitudinal in materials._iter_object_texels(
                objects[name], width, height):
            occupied[y, x] = True
    return occupied


def _bright_texels_in_projection(projection, luminance):
    """Counts bright stencil texels inside one affine exterior-face region."""
    corners = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=np.float64)
    uv_corners = materials._projection_uv(projection, corners)
    height, width = luminance.shape
    covered = np.zeros((height, width), dtype=bool)
    for indices in ((0, 1, 2), (0, 2, 3)):
        raster = materials._raster_triangle(
            uv_corners[list(indices)], width, height)
        if raster is not None:
            covered[raster[1], raster[0]] = True
    return int(np.count_nonzero(covered & (luminance > 0.60)))


def _projection_orientation_is_correct(projection):
    """Pins readable physical left/right orientation independently of pixels."""
    sign, physical_min, physical_max, affine = projection
    mapped = materials._projection_uv(
        projection, np.asarray(((0.0, 0.5), (1.0, 0.5))))
    mid_height = (physical_min[1] + physical_max[1]) * 0.5
    if sign > 0:
        longitudinal = (physical_max[0], physical_min[0])
    else:
        longitudinal = (physical_min[0], physical_max[0])
    expected_physical = np.asarray((
        (longitudinal[0], mid_height, 1.0),
        (longitudinal[1], mid_height, 1.0),
    ))
    expected_uv = expected_physical @ affine
    return bool(np.allclose(mapped, expected_uv, atol=1e-8, rtol=0.0))


t_start = time.time()
# Prove the single-mesh locomotive side target gets both +/-X projections,
# then reset so its geometry cannot occlude the wagon's isolated AO pairs.
scene.reset()
locomotive_recipe = recipe_module.load(LOCOMOTIVE_RECIPE_PATH)
locomotive = assemble.build_vehicle(locomotive_recipe)
uvmap.unwrap_and_pack(locomotive.objects, atlas_px=ATLAS_PX)
long_hood_projections = materials._exterior_side_projections(
    locomotive.objects["Visuals.Body.LongHood"])
if tuple(item[0] for item in long_hood_projections) != (-1, 1):
    failures.append(
        f"LongHood stencil side signs "
        f"{tuple(item[0] for item in long_hood_projections)} != (-1, 1)")
if not all(_projection_orientation_is_correct(item)
           for item in long_hood_projections):
    failures.append("LongHood stencil projection orientation is mirrored")
if failures:
    print("BPY TESTS FAIL:", "; ".join(failures))
    sys.exit(1)

scene.reset()
recipe_obj = recipe_module.load(WAGON_RECIPE_PATH)
assembly = assemble.build_vehicle(recipe_obj)
print(f"[materials_test] assembled ({len(assembly.objects)} meshes) "
      f"t={time.time() - t_start:.1f}s", flush=True)

uvmap.unwrap_and_pack(assembly.objects, atlas_px=ATLAS_PX)
print(f"[materials_test] unwrapped+packed t={time.time() - t_start:.1f}s", flush=True)

for door_name in ("Visuals.Body.DoorLeft", "Visuals.Body.DoorRight"):
    door_projections = materials._exterior_side_projections(
        assembly.objects[door_name])
    if len(door_projections) != 1:
        failures.append(
            f"{door_name} has {len(door_projections)} exterior stencil "
            f"projections, expected 1")
    elif not _projection_orientation_is_correct(door_projections[0]):
        failures.append(f"{door_name} exterior stencil orientation is mirrored")
if failures:
    print("BPY TESTS FAIL:", "; ".join(failures))
    sys.exit(1)

sources = highpoly.build_bake_sources(assembly.objects, seed=recipe_obj.seed)
os.makedirs(OUT_DIR, exist_ok=True)
detail_paths = bake.bake_detail(
    assembly.objects, sources, OUT_DIR, atlas_px=ATLAS_PX)
highpoly.cleanup_bake_sources()
print(f"[materials_test] detail baked t={time.time() - t_start:.1f}s", flush=True)

albedo_path = materials.bake_albedo(
    assembly.objects, recipe_obj, OUT_DIR, atlas_px=ATLAS_PX)

# Public material-assignment contract: every generated part keeps one real,
# vehicle-namespaced Principled material with its exact categorized values.
for object_name, obj in assembly.objects.items():
    if len(obj.data.materials) != 1 or obj.data.materials[0] is None:
        failures.append(
            f"{object_name} has {len(obj.data.materials)} material slots, expected 1")
        continue
    material = obj.data.materials[0]
    if material.name.startswith(bake.BAKE_TEMP_MATERIAL_PREFIX):
        failures.append(f"{object_name} retained temporary material {material.name}")
    if recipe_obj.model_id not in material.name:
        failures.append(
            f"{object_name} material {material.name} lacks model id namespace")
    principled = next(
        (node for node in material.node_tree.nodes
         if node.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if principled is None:
        failures.append(f"{object_name} material lacks Principled BSDF")
        continue
    category, expected_colour = materials._part_colour(recipe_obj, object_name)
    actual_colour = np.asarray(
        principled.inputs["Base Color"].default_value[:3], dtype=np.float32)
    actual_roughness = float(principled.inputs["Roughness"].default_value)
    if not np.allclose(actual_colour, expected_colour, atol=1e-6, rtol=0.0):
        failures.append(
            f"{object_name} base colour {actual_colour} != {expected_colour}")
    if not np.isclose(
            actual_roughness, materials._ROUGHNESS[category],
            atol=1e-6, rtol=0.0):
        failures.append(
            f"{object_name} roughness {actual_roughness} != "
            f"{materials._ROUGHNESS[category]}")
if failures:
    print("BPY TESTS FAIL:", "; ".join(failures))
    sys.exit(1)

mask_path = materials.bake_weathering_masks(
    assembly.objects, recipe_obj, detail_paths["ao"],
    detail_paths["curvature"], OUT_DIR, atlas_px=ATLAS_PX)
print(f"[materials_test] materials baked t={time.time() - t_start:.1f}s", flush=True)

for label, path in (("albedo", albedo_path), ("mask", mask_path)):
    if not os.path.isfile(path):
        failures.append(f"{label}.png missing at {path}")

if not failures:
    albedo = _load_image_array(albedo_path)
    mask = _load_image_array(mask_path)
    ao = _load_image_array(detail_paths["ao"])

    for label, array in (("albedo", albedo), ("mask", mask)):
        height, width = array.shape[:2]
        if (width, height) != (ATLAS_PX, ATLAS_PX):
            failures.append(
                f"{label}.png resolution {(width, height)} != "
                f"{(ATLAS_PX, ATLAS_PX)}")

    luminance = (
        0.2126 * albedo[:, :, 0]
        + 0.7152 * albedo[:, :, 1]
        + 0.0722 * albedo[:, :, 2])
    albedo_std = float(np.std(luminance))
    albedo_mean = float(np.mean(luminance))
    roughness_std = float(np.std(mask[:, :, 3]))
    snow_mean = float(np.mean(mask[:, :, 2]))
    occupied = _occupied_uv_mask(
        assembly.objects, albedo.shape[1], albedo.shape[0])
    coverage = float(np.mean(occupied))
    occupied_luminance_std = float(np.std(luminance[occupied]))
    occupied_roughness_std = float(np.std(mask[:, :, 3][occupied]))
    ao_copy_mae = float(np.mean(np.abs(mask[:, :, 0] - ao[:, :, 0])))
    occupied_wear_std = float(np.std(mask[:, :, 1][occupied]))
    stencil_texels = int(np.count_nonzero(luminance[occupied] > 0.60))
    door_stencil_texels = {}
    for door_name in ("Visuals.Body.DoorLeft", "Visuals.Body.DoorRight"):
        projection = materials._exterior_side_projections(
            assembly.objects[door_name])[0]
        door_stencil_texels[door_name] = _bright_texels_in_projection(
            projection, luminance)
    print(
        f"[materials_test] albedo meanL={albedo_mean:.4f} "
        f"stdL={albedo_std:.4f}; roughness std={roughness_std:.4f}; "
        f"snow mean={snow_mean:.4f}", flush=True)
    print(
        f"[materials_test] occupied coverage={coverage:.4f} "
        f"albedo stdL={occupied_luminance_std:.4f}; "
        f"roughness std={occupied_roughness_std:.4f}; "
        f"mask R/AO mae={ao_copy_mae:.6f}; "
        f"wear std={occupied_wear_std:.4f}; "
        f"stencil texels={stencil_texels}; "
        f"door stencils={door_stencil_texels}", flush=True)

    if not (albedo_std > 0.02):
        failures.append(f"albedo luminance std {albedo_std} <= 0.02")
    if not (0.05 < albedo_mean < 0.5):
        failures.append(
            f"albedo mean luminance {albedo_mean} outside (0.05, 0.5)")
    if not (roughness_std > 0.02):
        failures.append(f"mask roughness std {roughness_std} <= 0.02")
    if not (0.02 < snow_mean < 0.6):
        failures.append(f"mask snow mean {snow_mean} outside (0.02, 0.6)")
    if not (0.05 < coverage < 0.40):
        failures.append(f"occupied UV coverage {coverage} outside (0.05, 0.40)")
    if not (occupied_luminance_std > 0.02):
        failures.append(
            f"occupied albedo luminance std {occupied_luminance_std} <= 0.02")
    if not (occupied_roughness_std > 0.02):
        failures.append(
            f"occupied roughness std {occupied_roughness_std} <= 0.02")
    if not (ao_copy_mae < 0.005):
        failures.append(f"mask R / AO mean absolute error {ao_copy_mae} >= 0.005")
    if not (occupied_wear_std > 0.005):
        failures.append(
            f"occupied mask G std {occupied_wear_std} <= 0.005")
    if not (stencil_texels >= 20):
        failures.append(
            f"only {stencil_texels} bright stencil texels found, expected >= 20")
    for door_name, bright_count in door_stencil_texels.items():
        if bright_count < 20:
            failures.append(
                f"{door_name} exterior has only {bright_count} bright stencil "
                f"texels, expected >= 20")

    roof = assembly.objects["Visuals.Body.GabledRoof"]
    side = assembly.objects["Visuals.Body.BoxBody"]
    roof_samples = _surface_samples(
        roof, mask, lambda normal: normal.z > 0.5, SAMPLE_COUNT, 1301)
    side_samples = _surface_samples(
        side, mask, lambda normal: abs(normal.x) > 0.8, SAMPLE_COUNT, 1302)
    roof_snow = float(np.mean(roof_samples[:, 2]))
    side_snow = float(np.mean(side_samples[:, 2]))
    print(f"[materials_test] snow roof={roof_snow:.4f} "
          f"side={side_snow:.4f} delta={roof_snow - side_snow:.4f}",
          flush=True)
    if not (roof_snow - side_snow > 0.15):
        failures.append(
            f"roof-side snow delta {roof_snow - side_snow} <= 0.15")

print(f"[materials_test] total wall-clock t={time.time() - t_start:.1f}s", flush=True)
if failures:
    print("BPY TESTS FAIL:", "; ".join(failures))
    sys.exit(1)
print("BPY TESTS OK")
sys.exit(0)
