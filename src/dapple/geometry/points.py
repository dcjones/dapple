
from ..elements import VectorizedElement
from ..scales import length_params, color_params
from ..coordinates import CtxLenType
from ..config import ConfigKey

def points(x, y, color=ConfigKey("pointcolor"), size=ConfigKey("pointsize")):
    return VectorizedElement(
        "circle", {
            "cx": length_params("x", x, CtxLenType.Pos),
            "cy": length_params("y", y, CtxLenType.Pos),
            "r": size,
            "fill": color_params("color", color)
        }
    )
