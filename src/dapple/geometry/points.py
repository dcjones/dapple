
from typing import override

from ..config import ConfigKey
from ..coordinates import (
    AbsLengths,
    CoordBounds,
    CtxLenType,
    Lengths,
    ResolveContext,
    resolve,
)
from ..elements import Element, VectorizedElement
from ..scales import color_params, length_params, shape_params
from ..shapes import Shapes, build_shape_paths


def points(
    x,
    y,
    color=ConfigKey("pointcolor"),
    size=ConfigKey("pointsize"),
    shape=None,
):
    """
    Draw points (markers) at the given x, y positions.

    Args:
        x: X coordinates of points.
        y: Y coordinates of points.
        color: Point colors, passed through the color scale (default config pointcolor).
        size: Point size (default config pointsize).
        shape: Optional values passed through a discrete shape scale to vary the
            marker shape. When omitted, all points are drawn as circles.
    """
    if shape is None:
        return VectorizedElement(
            "circle",
            {
                "cx": length_params("x", x, CtxLenType.Pos),
                "cy": length_params("y", y, CtxLenType.Pos),
                "r": length_params("size", size, CtxLenType.Vec),
                "fill": color_params("color", color),
            },
        )

    return Points(x, y, color, size, shape)


class Points(Element):
    """
    Points geometry that varies marker shape according to a discrete shape scale.

    Unlike the plain circle-based points, this resolves each marker to an SVG
    ``path`` whose ``d`` is generated from the point's position, size, and shape.
    """

    def __init__(self, x, y, color, size, shape):
        super().__init__(tag="dapple:points")
        self.attrib = {
            "cx": length_params("x", x, CtxLenType.Pos),
            "cy": length_params("y", y, CtxLenType.Pos),
            "r": length_params("size", size, CtxLenType.Vec),
            "fill": color_params("color", color),
            "dapple:shape": shape_params("shape", shape),
        }

    @override
    def update_bounds(self, bounds: CoordBounds):
        x = self.get_as("cx", Lengths)
        y = self.get_as("cy", Lengths)
        r = self.get_as("r", Lengths)

        bounds.update(x - r)
        bounds.update(y - r)
        bounds.update(x + r)
        bounds.update(y + r)

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        resolved = resolve(self.attrib, ctx)

        cx = resolved.pop("cx")
        cy = resolved.pop("cy")
        r = resolved.pop("r")
        shapes = resolved.pop("dapple:shape")

        assert isinstance(cx, AbsLengths)
        assert isinstance(cy, AbsLengths)
        assert isinstance(r, AbsLengths)
        assert isinstance(shapes, Shapes)

        d = build_shape_paths(shapes.indices, cx.values, cy.values, r.values)

        return VectorizedElement("path", {"d": d, **resolved})
