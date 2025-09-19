from xml.etree.ElementTree import Element
from ..elements import ResolvableElement, VectorizedElement
from ..coordinates import Resolvable, ResolveContext, Lengths, vh
from ..layout import Position
from ..config import ConfigKey

class XGrids(ResolvableElement):
    def __init__(
            self,
            stroke=ConfigKey("grid_stroke"),
            stroke_width=ConfigKey("grid_stroke_width"),
            dasharray=ConfigKey("grid_stroke_dasharray"),
    ):
        attrib = {
            "dapple:position": Position.Below,
            "stroke": stroke,
            "stroke-width": stroke_width,
            "stroke-dasharray": dasharray,
        }
        super().__init__("dapple:xgrids", attrib) # type: ignore

    def resolve(self, ctx: ResolveContext) -> Element:
        assert "x" in ctx.scales

        x_scale = ctx.scales["x"]
        _x_labels, x_ticks = x_scale.ticks()
        assert isinstance(x_ticks, Lengths)

        g = ResolvableElement(
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
    return XGrids(*args, **kwargs)

class YGrids(ResolvableElement):
    def __init__(
            self,
            stroke=ConfigKey("grid_stroke"),
            stroke_width=ConfigKey("grid_stroke_width"),
            dasharray=ConfigKey("grid_stroke_dasharray"),
    ):
        attrib = {
            "dapple:position": Position.Below,
            "stroke": stroke,
            "stroke-width": stroke_width,
            "stroke-dasharray": dasharray,
        }
        super().__init__("dapple:ygrids", attrib) # type: ignore

    def resolve(self, ctx: ResolveContext) -> Element:
        assert "y" in ctx.scales

        y_scale = ctx.scales["y"]
        _y_labels, y_ticks = y_scale.ticks()
        assert isinstance(y_ticks, Lengths)

        g = ResolvableElement(
            "g", {
                "stroke": self.attrib["stroke"],
                "stroke-width": self.attrib["stroke-width"],
                "stroke-dasharray": self.attrib["stroke-dasharray"],
            }
        )
        g.append(
            VectorizedElement(
                "line", {
                    "x1": vh(0),
                    "x2": vh(1),
                    "y1": y_ticks,
                    "y2": y_ticks,
                }
            ))

        return g.resolve(ctx)

def ygrids(*args, **kwargs):
    return YGrids(*args, **kwargs)
