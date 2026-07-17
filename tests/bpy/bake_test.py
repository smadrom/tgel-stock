"""Wagon-chain high-poly + Cycles detail-bake test (Task 12).

Resolution note (disclosed per task brief): production atlas/bake
resolution is 4096 (Task 11's controller-adjudicated default, carried into
``bake.bake_detail``'s ``atlas_px`` default for Tasks 13/15). This TEST
passes ``atlas_px=1024`` to keep --background Cycles CPU wall-clock sane;
``bake_detail``'s bake logic is resolution-independent, so 1024 exercises
the exact same code paths production 4096 would.

Only the WAGON is baked here (not the locomotive) to keep wall-clock sane,
per the task brief's "wagon chain" scope.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from tgel_stock import assemble  # noqa: E402
from tgel_stock import bake  # noqa: E402
from tgel_stock import highpoly  # noqa: E402
from tgel_stock import recipe as recipe_module  # noqa: E402
from tgel_stock import scene  # noqa: E402
from tgel_stock import uvmap  # noqa: E402

ATLAS_PX = 1024

WAGON_RECIPE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "recipes", "basic-box-wagon.rollingstock.json")

OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..",
    "artifacts", "rolling-stock-generation", "bake_test")

failures = []


def _load_image_array(path):
    img = bpy.data.images.load(path)
    try:
        # Loaded PNGs default to an sRGB colorspace and Image.pixels returns
        # colorspace-CONVERTED values (empirically verified against Blender
        # 5.1.2); Non-Color reads back the raw stored texel values the bake
        # wrote.
        img.colorspace_settings.name = 'Non-Color'
        width, height = img.size
        channels = img.channels
        arr = np.array(img.pixels[:], dtype=np.float64)
        arr = arr.reshape(height, width, channels)
        return arr
    finally:
        bpy.data.images.remove(img)


t_start = time.time()
scene.reset()
recipe_obj = recipe_module.load(WAGON_RECIPE_PATH)
assembly = assemble.build_vehicle(recipe_obj)
print(f"[bake_test] assembled ({len(assembly.objects)} meshes) "
      f"t={time.time() - t_start:.1f}s", flush=True)
uvmap.unwrap_and_pack(assembly.objects, atlas_px=ATLAS_PX)
print(f"[bake_test] unwrapped+packed t={time.time() - t_start:.1f}s", flush=True)

sources = highpoly.build_bake_sources(assembly.objects)
print(f"[bake_test] bake sources built t={time.time() - t_start:.1f}s", flush=True)

os.makedirs(OUT_DIR, exist_ok=True)
paths = bake.bake_detail(assembly.objects, sources, OUT_DIR, atlas_px=ATLAS_PX)
print(f"[bake_test] baked t={time.time() - t_start:.1f}s", flush=True)

for key in ("normal", "ao", "curvature"):
    if key not in paths:
        failures.append(f"bake_detail result missing key '{key}'")

if not failures:
    for key, path in paths.items():
        if not os.path.isfile(path):
            failures.append(f"{key} PNG missing at {path}")
            continue

        arr = _load_image_array(path)
        height, width = arr.shape[0], arr.shape[1]
        if (width, height) != (ATLAS_PX, ATLAS_PX):
            failures.append(
                f"{key} PNG resolution {(width, height)} != {(ATLAS_PX, ATLAS_PX)}")

        if key == "normal":
            mean_r = float(np.mean(arr[:, :, 0]))
            mean_g = float(np.mean(arr[:, :, 1]))
            mean_b = float(np.mean(arr[:, :, 2]))
            std_r = float(np.std(arr[:, :, 0]))
            print(f"[bake_test] normal meanR={mean_r:.4f} meanG={mean_g:.4f} "
                  f"meanB={mean_b:.4f} stdR={std_r:.4f}", flush=True)
            if not (0.45 <= mean_r <= 0.55):
                failures.append(f"normal mean R {mean_r} outside [0.45, 0.55]")
            if not (0.45 <= mean_g <= 0.55):
                failures.append(f"normal mean G {mean_g} outside [0.45, 0.55]")
            if not (mean_b > 0.7):
                failures.append(f"normal mean B {mean_b} <= 0.7")
            # The (0.5, 0.5, 1.0) neutral prefill alone would satisfy the
            # mean gates; real baked bevel/greeble normals must add R
            # variance (an untouched prefill image has std exactly 0).
            if not (std_r > 0.005):
                failures.append(
                    f"normal std R {std_r} <= 0.005 (looks like bare prefill)")
        elif key == "ao":
            mean_v = float(np.mean(arr[:, :, 0]))
            std_v = float(np.std(arr[:, :, 0]))
            print(f"[bake_test] ao mean={mean_v:.4f} std={std_v:.4f}", flush=True)
            if not (0.2 < mean_v < 0.98):
                failures.append(f"ao mean {mean_v} outside (0.2, 0.98)")
            if not (std_v > 0.01):
                failures.append(f"ao std {std_v} <= 0.01 (looks constant)")
        elif key == "curvature":
            std_v = float(np.std(arr[:, :, 0]))
            print(f"[bake_test] curvature std={std_v:.4f}", flush=True)
            if not (std_v > 0.005):
                failures.append(f"curvature std {std_v} <= 0.005 (looks constant)")

# bake_detail must have restored every target's original material slots
# (wagon parts carry no materials pre-bake, so nothing may reference a
# TGEL.BakeTemp.* material after it returns).
for name, obj in assembly.objects.items():
    for slot_material in obj.data.materials:
        if slot_material is not None and slot_material.name.startswith(
                bake.BAKE_TEMP_MATERIAL_PREFIX):
            failures.append(f"{name} still references bake material {slot_material.name}")

# cleanup_bake_sources must remove the BakeSources collection, every
# high-poly object/mesh it contains, and every TGEL.BakeTemp.* material.
highpoly.cleanup_bake_sources()
if bpy.data.collections.get(highpoly.BAKE_SOURCES_COLLECTION) is not None:
    failures.append("BakeSources collection still present after cleanup_bake_sources()")
leftover = [o for o in bpy.data.objects if o.name.endswith(highpoly.BAKE_SOURCE_SUFFIX)]
if leftover:
    failures.append(f"cleanup_bake_sources left {len(leftover)} high-poly object(s) behind")
leftover_materials = [m.name for m in bpy.data.materials
                      if m.name.startswith(bake.BAKE_TEMP_MATERIAL_PREFIX)]
if leftover_materials:
    failures.append(
        f"cleanup_bake_sources left {len(leftover_materials)} TGEL.BakeTemp.* material(s)")

print(f"[bake_test] total wall-clock t={time.time() - t_start:.1f}s", flush=True)
if failures:
    print("BPY TESTS FAIL:", "; ".join(failures))
    sys.exit(1)
print("BPY TESTS OK")
sys.exit(0)
