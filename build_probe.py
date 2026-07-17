import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tgel_stock import scene, export  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "artifacts", "rolling-stock-generation", "probe")


def main():
    scene.reset()
    front = scene.box("Probe.Front", (0.0, 0.5, 3.0), (0.2, 1.0, 0.4))
    right = scene.box("Probe.Right", (1.5, 0.1, 0.0), (0.4, 0.2, 0.2))
    os.makedirs(OUT, exist_ok=True)
    export.export_fbx([front, right], os.path.join(OUT, "space-probe.fbx"))
    print("PROBE OK")


main()
