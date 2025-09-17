
from ..elements import VectorizedElement
from ..scales import UnscaledValues
from ..coordinates import CtxLenType, mm
from ..config import ConfigKey

def points(x, y, color):
    return VectorizedElement(
        "circle", {
            "cx": UnscaledValues("x", x, CtxLenType.Pos),
            "cy": UnscaledValues("y", y, CtxLenType.Pos),
            "r": ConfigKey("pointsize"),
            "fill": UnscaledValues("color", color)}
    )
