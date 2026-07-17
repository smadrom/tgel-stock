import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tgel_stock import scene  # noqa: E402

TOLERANCE = 1e-4

failures = []

scene.reset()
front = scene.box("Probe.Front", (0.0, 0.5, 3.0), (0.2, 1.0, 0.4))
right = scene.box("Probe.Right", (1.5, 0.1, 0.0), (0.4, 0.2, 0.2))

_, _, _, _, _, front_max = scene.collect_mesh_data(front)
_, _, _, _, _, right_max = scene.collect_mesh_data(right)

if abs(front_max[2] - 3.2) > TOLERANCE:
    failures.append(f"Probe.Front bounds max z {front_max[2]} != 3.2")
if abs(front_max[1] - 1.0) > TOLERANCE:
    failures.append(f"Probe.Front bounds max y {front_max[1]} != 1.0")
if abs(right_max[0] - 1.7) > TOLERANCE:
    failures.append(f"Probe.Right bounds max x {right_max[0]} != 1.7")

if failures:
    print("BPY TESTS FAIL:", "; ".join(failures))
    sys.exit(1)
print("BPY TESTS OK")
sys.exit(0)
