import json
from dataclasses import dataclass, field

from . import assert_clean_path

GAUGE = 1.435
BACK_TO_BACK = 1.348
WHEEL_WIDTH = 0.135
COUPLER_HEIGHT = 0.860
COUPLER_PIVOT_TO_FACE = 0.600

_REQUIRED = (
    "modelId", "kind", "lengthOverCouplers", "bodyLength", "width", "height",
    "wheelRadius", "bogieCentreOffset", "bogieWheelbase", "bogiePivotHeight",
    "seed", "livery",
)


@dataclass(frozen=True)
class Recipe:
    model_id: str
    kind: str
    length_over_couplers: float
    body_length: float
    width: float
    height: float
    wheel_radius: float
    bogie_centre_offset: float
    bogie_wheelbase: float
    bogie_pivot_height: float
    seed: int
    livery: dict = field(default_factory=dict)


def load(path: str) -> Recipe:
    assert_clean_path(path)
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    missing = [key for key in _REQUIRED if key not in data]
    if missing:
        raise ValueError(f"Recipe {path} missing fields: {missing}")
    if data["kind"] not in ("locomotive", "wagon"):
        raise ValueError(f"Unknown kind: {data['kind']}")
    return Recipe(
        model_id=data["modelId"],
        kind=data["kind"],
        length_over_couplers=float(data["lengthOverCouplers"]),
        body_length=float(data["bodyLength"]),
        width=float(data["width"]),
        height=float(data["height"]),
        wheel_radius=float(data["wheelRadius"]),
        bogie_centre_offset=float(data["bogieCentreOffset"]),
        bogie_wheelbase=float(data["bogieWheelbase"]),
        bogie_pivot_height=float(data["bogiePivotHeight"]),
        seed=int(data["seed"]),
        livery={k: list(map(float, v)) for k, v in data["livery"].items()},
    )
