
from typing import Optional, override

import numpy as np

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
    x_jitter: Optional[Lengths] = None,
    y_jitter: Optional[Lengths] = None,
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
        x_jitter: Optional length (e.g. ``mm(2)``). Each point's x position is
            shifted by this length times a uniform random value in [-1, 1].
        y_jitter: Like ``x_jitter``, but for the y position.
    """
    if shape is None and x_jitter is None and y_jitter is None:
        return VectorizedElement(
            "circle",
            {
                "cx": length_params("x", x, CtxLenType.Pos),
                "cy": length_params("y", y, CtxLenType.Pos),
                "r": length_params("size", size, CtxLenType.Vec),
                "fill": color_params("color", color),
            },
        )

    return Points(x, y, color, size, shape, x_jitter, y_jitter)


class Points(Element):
    """
    Points geometry that varies marker shape according to a discrete shape scale.

    Unlike the plain circle-based points, this resolves each marker to an SVG
    ``path`` whose ``d`` is generated from the point's position, size, and shape.
    Without a shape, markers resolve to circles. Optional jitter lengths displace
    each point by a random fraction in [-1, 1] of the jitter.
    """

    def __init__(
        self,
        x,
        y,
        color,
        size,
        shape=None,
        x_jitter: Optional[Lengths] = None,
        y_jitter: Optional[Lengths] = None,
    ):
        super().__init__(tag="dapple:points")
        self.attrib = {
            "cx": length_params("x", x, CtxLenType.Pos),
            "cy": length_params("y", y, CtxLenType.Pos),
            "r": length_params("size", size, CtxLenType.Vec),
            "fill": color_params("color", color),
        }
        if shape is not None:
            self.attrib["dapple:shape"] = shape_params("shape", shape)
        if x_jitter is not None:
            self.attrib["dapple:xjitter"] = x_jitter
        if y_jitter is not None:
            self.attrib["dapple:yjitter"] = y_jitter
        self.seed = int(np.random.SeedSequence().generate_state(1)[0])

    @override
    def update_bounds(self, bounds: CoordBounds):
        x = self.get_as("cx", Lengths)
        y = self.get_as("cy", Lengths)
        r = self.get_as("r", Lengths)

        xj = self.attrib.get("dapple:xjitter")
        if isinstance(xj, Lengths):
            r_x = r + xj
        else:
            r_x = r
        yj = self.attrib.get("dapple:yjitter")
        if isinstance(yj, Lengths):
            r_y = r + yj
        else:
            r_y = r

        bounds.update(x - r_x)
        bounds.update(y - r_y)
        bounds.update(x + r_x)
        bounds.update(y + r_y)

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        resolved = resolve(self.attrib, ctx)

        cx = resolved.pop("cx")
        cy = resolved.pop("cy")
        r = resolved.pop("r")
        shapes = resolved.pop("dapple:shape", None)
        xjitter = resolved.pop("dapple:xjitter", None)
        yjitter = resolved.pop("dapple:yjitter", None)

        assert isinstance(cx, AbsLengths)
        assert isinstance(cy, AbsLengths)
        assert isinstance(r, AbsLengths)

        rng = np.random.default_rng(self.seed)
        n = len(cx.values)
        cxv = cx.values
        cyv = cy.values
        if xjitter is not None:
            assert isinstance(xjitter, AbsLengths)
            cxv = cxv + xjitter.values * rng.uniform(-1.0, 1.0, n)
        if yjitter is not None:
            assert isinstance(yjitter, AbsLengths)
            cyv = cyv + yjitter.values * rng.uniform(-1.0, 1.0, n)

        if shapes is None:
            return VectorizedElement(
                "circle",
                {"cx": AbsLengths(cxv), "cy": AbsLengths(cyv), "r": r, **resolved},
            )

        assert isinstance(shapes, Shapes)
        d = build_shape_paths(shapes.indices, cxv, cyv, r.values)

        return VectorizedElement("path", {"d": d, **resolved})
