
from ..elements import VectorizedElement
from ..scales import length_params, color_params
from ..coordinates import CtxLenType, mm
from ..config import ConfigKey

def points(x, y, color=ConfigKey("pointcolor")):
    return VectorizedElement(
        "circle", {
            "cx": length_params("x", x, CtxLenType.Pos),
            "cy": length_params("y", y, CtxLenType.Pos),
            "r": ConfigKey("pointsize"),
            "fill": color_params("color", color)
        }
    )
