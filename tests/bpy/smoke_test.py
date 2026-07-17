import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import bpy  # noqa: E402
from tgel_stock import BLENDER_REQUIRED_VERSION, assert_clean_path  # noqa: E402

failures = []
if bpy.app.version_string != BLENDER_REQUIRED_VERSION:
    failures.append(f"Blender {bpy.app.version_string} != required {BLENDER_REQUIRED_VERSION}")
try:
    assert_clean_path("C:/Games/RC2/whatever.fbx")
    failures.append("assert_clean_path accepted a forbidden path")
except ValueError:
    pass

if failures:
    print("SMOKE FAIL:", "; ".join(failures))
    sys.exit(1)
print("SMOKE OK")
sys.exit(0)
