
from ..elements import VectorizedElement
from ..scales import UnscaledValues
from ..coordinates import CtxLenType

# Ok, a few issues we have to resolve:
# - We want to support non-scaling certain special types. I think we handle this
#   in the scaling logic though.
# - Need to figure how working out minimum maximum values for aesthetics. (I.e. whole bounds updating logic
# from Dapple.jl. I think we have to support custom bounds calculations, bbut

def points(x, y, color):
    return VectorizedElement(
        "circle", {
            "x": UnscaledValues("x", x, CtxLenType.Pos),
            "y": UnscaledValues("y", y, CtxLenType.Pos),
            "color": UnscaledValues("color", color)}
    )
