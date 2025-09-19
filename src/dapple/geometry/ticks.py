from xml.etree.ElementTree import Element
from ..elements import ResolvableElement, VectorizedElement
from ..coordinates import AbsLengths, Resolvable, ResolveContext, Lengths, vh, vw, mm
from ..layout import Position
from ..config import ConfigKey

class XTicks(ResolvableElement):
    """
    Draw tick marks along the bottom margin of the plot for the x-axis.
    """
    def __init__(
            self,
            stroke=ConfigKey("tick_stroke"),
            stroke_width=ConfigKey("tick_stroke_width"),
            tick_length=ConfigKey("tick_length"),
    ):
        attrib = {
            "dapple:position": Position.BottomLeft,
            "stroke": stroke,
            "stroke-width": stroke_width,
            "tick_length": tick_length,
        }
        super().__init__("dapple:xticks", attrib) # type: ignore

    def resolve(self, ctx: ResolveContext) -> Element:
        assert "x" in ctx.scales

        x_scale = ctx.scales["x"]
        _x_labels, x_ticks = x_scale.ticks()
        assert isinstance(x_ticks, Lengths)

        print(x_ticks)

        # Get resolved tick length from attributes
        tick_length = self.attrib["tick_length"]
        if isinstance(tick_length, Resolvable):
            tick_length = tick_length.resolve(ctx)

        g = ResolvableElement(
            "g", {
                "stroke": self.attrib["stroke"],
                "stroke-width": self.attrib["stroke-width"],
            }
        )

        # Add the main axis line
        g.append(
            ResolvableElement(
                "line", {
                    "x1": vw(0),
                    "x2": vw(1),
                    "y1": vh(0),
                    "y2": vh(0),
                }
            ))

        # Add tick marks - going downward from the axis
        g.append(
            VectorizedElement(
                "line", {
                    "x1": x_ticks,
                    "x2": x_ticks,
                    "y1": vh(0),
                    "y2": tick_length,
                }
            ))

        return g.resolve(ctx)

    def abs_bounds(self) -> tuple[AbsLengths, AbsLengths]:
        tick_length = self.attrib["tick_length"]
        assert isinstance(tick_length, AbsLengths)

        return (mm(0), tick_length)

def xticks(*args, **kwargs):
    return XTicks(*args, **kwargs)

class YTicks(ResolvableElement):
    """
    Draw tick marks along the left margin of the plot for the y-axis.
    """
    def __init__(
            self,
            stroke=ConfigKey("tick_stroke"),
            stroke_width=ConfigKey("tick_stroke_width"),
            tick_length=ConfigKey("tick_length"),
    ):
        attrib = {
            "dapple:position": Position.LeftTop,
            "stroke": stroke,
            "stroke-width": stroke_width,
            "tick_length": tick_length,
        }
        super().__init__("dapple:yticks", attrib) # type: ignore

    def resolve(self, ctx: ResolveContext) -> Element:
        assert "y" in ctx.scales

        y_scale = ctx.scales["y"]
        _y_labels, y_ticks = y_scale.ticks()
        assert isinstance(y_ticks, Lengths)

        # Get resolved tick length from attributes
        tick_length = self.attrib["tick_length"]
        assert isinstance(tick_length, AbsLengths)

        g = ResolvableElement(
            "g", {
                "stroke": self.attrib["stroke"],
                "stroke-width": self.attrib["stroke-width"],
            }
        )

        # Add the main axis line
        g.append(
            ResolvableElement(
                "line", {
                    "x1": tick_length,
                    "x2": tick_length,
                    "y1": y_ticks[0],
                    "y2": y_ticks[-1],
                }
            ))

        # Add tick marks - going leftward from the axis
        # Create negative tick length by using -tick_length for x2
        g.append(
            VectorizedElement(
                "line", {
                    "x1": vw(0),
                    "x2": tick_length,
                    "y1": y_ticks,
                    "y2": y_ticks,
                }
            ))

        return g.resolve(ctx)

    def abs_bounds(self) -> tuple[AbsLengths, AbsLengths]:
        tick_length = self.attrib["tick_length"]
        assert isinstance(tick_length, AbsLengths)

        return (tick_length, mm(0))


def yticks(*args, **kwargs):
    return YTicks(*args, **kwargs)
