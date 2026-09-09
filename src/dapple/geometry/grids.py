from ..elements import Element, VectorizedElement
from ..coordinates import ResolveContext, Lengths, vh, vw
from ..layout import Position
from ..config import ConfigKey

class XGrids(Element):
    """
    Draw vertical grid lines behind the plot at each x-axis tick.
    """

    def __init__(
            self,
            stroke=ConfigKey("grid_stroke"),
            stroke_width=ConfigKey("grid_stroke_width"),
            dasharray=ConfigKey("grid_stroke_dasharray"),
    ):
        attrib: dict[str, object] = {
            "dapple:position": Position.Below,
            "stroke": stroke,
            "stroke-width": stroke_width,
            "stroke-dasharray": dasharray,
        }
        super().__init__("dapple:xgrids", attrib)

    def resolve(self, ctx: ResolveContext) -> Element:
        if "x" not in ctx.scales:
            return Element("g")

        x_scale = ctx.scales["x"]
        _x_labels, x_ticks = x_scale.ticks()
        assert isinstance(x_ticks, Lengths)

        g = Element(
            "g", {
                "stroke": self.attrib["stroke"],
                "stroke-width": self.attrib["stroke-width"],
                "stroke-dasharray": self.attrib["stroke-dasharray"],
            }
        )
        g.append(
            VectorizedElement(
                "line", {
                    "x1": x_ticks,
                    "x2": x_ticks,
                    "y1": vh(0),
                    "y2": vh(1),
                }
            ))

        return g.resolve(ctx)

def xgrids(*args, **kwargs):
    """
    Draw vertical grid lines at each x-axis tick.

    Included in plots by default; pass a call to this to override its
    appearance.

    Args:
        stroke: Grid line color (default: config "grid_stroke").
        stroke_width: Grid line width (default: config "grid_stroke_width").
        dasharray: SVG dash pattern for the grid lines
            (default: config "grid_stroke_dasharray").
    """
    return XGrids(*args, **kwargs)

class YGrids(Element):
    """
    Draw horizontal grid lines behind the plot at each y-axis tick.
    """

    def __init__(
            self,
            stroke=ConfigKey("grid_stroke"),
            stroke_width=ConfigKey("grid_stroke_width"),
            dasharray=ConfigKey("grid_stroke_dasharray"),
    ):
        attrib: dict[str, object] = {
            "dapple:position": Position.Below,
            "stroke": stroke,
            "stroke-width": stroke_width,
            "stroke-dasharray": dasharray,
        }
        super().__init__("dapple:ygrids", attrib)

    def resolve(self, ctx: ResolveContext) -> Element:
        if "y" not in ctx.scales:
            return Element("g")

        y_scale = ctx.scales["y"]
        _y_labels, y_ticks = y_scale.ticks()
        assert isinstance(y_ticks, Lengths)

        g = Element(
            "g", {
                "stroke": self.attrib["stroke"],
                "stroke-width": self.attrib["stroke-width"],
                "stroke-dasharray": self.attrib["stroke-dasharray"],
            }
        )
        g.append(
            VectorizedElement(
                "line", {
                    "x1": vw(0),
                    "x2": vw(1),
                    "y1": y_ticks,
                    "y2": y_ticks,
                }
            ))

        return g.resolve(ctx)

def ygrids(*args, **kwargs):
    """
    Draw horizontal grid lines at each y-axis tick.

    Included in plots by default; pass a call to this to override its
    appearance.

    Args:
        stroke: Grid line color (default: config "grid_stroke").
        stroke_width: Grid line width (default: config "grid_stroke_width").
        dasharray: SVG dash pattern for the grid lines
            (default: config "grid_stroke_dasharray").
    """
    return YGrids(*args, **kwargs)
