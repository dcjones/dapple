from ..elements import Element, VectorizedElement, Path
from ..coordinates import AbsLengths, Resolvable, ResolveContext, Lengths, vh, vw, mm
from ..layout import Position
from ..config import ConfigKey

from typing import override


class XTicks(Element):
    """
    Draw tick marks along the bottom margin of the plot for the x-axis.
    """

    def __init__(
        self,
        stroke=ConfigKey("tick_stroke"),
        stroke_width=ConfigKey("tick_stroke_width"),
        tick_length=ConfigKey("tick_length"),
    ):
        attrib: dict[str, object] = {
            "dapple:position": Position.BottomLeft,
            "dapple:padding-top": ConfigKey("padding_std"),
            "dapple:padding-bottom": ConfigKey("padding_min"),
            "dapple:padding-left": ConfigKey("padding_nil"),
            "dapple:padding-right": ConfigKey("padding_nil"),
            "stroke": stroke,
            "stroke-width": stroke_width,
            "tick_length": tick_length,
        }
        super().__init__("dapple:xticks", attrib)  # type: ignore

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        if "x" not in ctx.scales:
            return Element("g")

        x_scale = ctx.scales["x"]
        _x_labels, x_ticks = x_scale.ticks()
        assert isinstance(x_ticks, Lengths)

        # Get resolved tick length from attributes
        tick_length = self.attrib["tick_length"]
        if isinstance(tick_length, Resolvable):
            tick_length = tick_length.resolve(ctx)
        assert isinstance(tick_length, AbsLengths)

        g = Element(
            "g",
            {
                "stroke": self.attrib["stroke"],
                "stroke-width": self.attrib["stroke-width"],
            },
        )

        # Add axis line + end ticks as a single path when we have at least two ticks
        if len(x_ticks) >= 2:
            # Resolve to absolute coordinates to build the combined path
            x_res = x_ticks.resolve(ctx)
            assert isinstance(x_res, AbsLengths)
            x0 = x_res[0].scalar_value()
            xN = x_res[-1].scalar_value()
            y_axis = vh(0).resolve(ctx).scalar_value()
            y_tick = tick_length.scalar_value()

            # Path: left tick -> axis -> right axis end -> right tick
            g.append(
                Path(
                    mm([x0, x0, xN, xN]),
                    mm([y_tick, y_axis, y_axis, y_tick]),
                    fill="none",
                )
            )

            # Interior tick marks (exclude the two ends), computed from resolved positions
            interior_vals = x_res.values[1:-1]
            if len(interior_vals) > 0:
                interior_x = mm(interior_vals)
                g.append(
                    VectorizedElement(
                        "line",
                        {
                            "x1": interior_x,
                            "x2": interior_x,
                            "y1": vh(0),
                            "y2": tick_length,
                        },
                    )
                )
        else:
            # Fallback: draw axis line (only if we have at least one tick) and all ticks
            if len(x_ticks) >= 1:
                g.append(
                    Element(
                        "line",
                        {
                            "x1": x_ticks[0],
                            "x2": x_ticks[-1],
                            "y1": vh(0),
                            "y2": vh(0),
                        },
                    )
                )
            g.append(
                VectorizedElement(
                    "line",
                    {
                        "x1": x_ticks,
                        "x2": x_ticks,
                        "y1": vh(0),
                        "y2": tick_length,
                    },
                )
            )

        return g.resolve(ctx)

    @override
    def abs_bounds(self) -> tuple[AbsLengths, AbsLengths]:
        tick_length = self.attrib["tick_length"]
        assert isinstance(tick_length, AbsLengths)

        return (mm(0), tick_length)


def xticks(*args, **kwargs):
    return XTicks(*args, **kwargs)


class YTicks(Element):
    """
    Draw tick marks along the left margin of the plot for the y-axis.
    """

    def __init__(
        self,
        stroke=ConfigKey("tick_stroke"),
        stroke_width=ConfigKey("tick_stroke_width"),
        tick_length=ConfigKey("tick_length"),
    ):
        attrib: dict[str, object] = {
            "dapple:position": Position.LeftTop,
            "dapple:padding-top": ConfigKey("padding_nil"),
            "dapple:padding-bottom": ConfigKey("padding_nil"),
            "dapple:padding-left": ConfigKey("padding_min"),
            "dapple:padding-right": ConfigKey("padding_std"),
            "stroke": stroke,
            "stroke-width": stroke_width,
            "tick_length": tick_length,
        }
        super().__init__("dapple:yticks", attrib)  # type: ignore

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        if "y" not in ctx.scales:
            return Element("g")

        y_scale = ctx.scales["y"]
        _y_labels, y_ticks = y_scale.ticks()
        assert isinstance(y_ticks, Lengths)

        # Get resolved tick length from attributes
        tick_length = self.attrib["tick_length"]
        assert isinstance(tick_length, AbsLengths)

        g = Element(
            "g",
            {
                "stroke": self.attrib["stroke"],
                "stroke-width": self.attrib["stroke-width"],
            },
        )

        # Add axis line + end ticks as a single path when we have at least two ticks
        if len(y_ticks) >= 2:
            # Resolve to absolute coordinates to build the combined path
            y_res = y_ticks.resolve(ctx)
            assert isinstance(y_res, AbsLengths)
            y0 = y_res[0].scalar_value()
            yN = y_res[-1].scalar_value()
            x_axis = tick_length.scalar_value()
            x_left = vw(0).resolve(ctx).scalar_value()

            # Path: left tick -> axis -> far axis end -> far tick
            g.append(
                Path(
                    mm([x_left, x_axis, x_axis, x_left]),
                    mm([y0, y0, yN, yN]),
                    fill="none",
                )
            )

            # Interior tick marks (exclude the two ends), computed from resolved positions
            interior_vals = y_res.values[1:-1]
            if len(interior_vals) > 0:
                interior_y = mm(interior_vals)
                g.append(
                    VectorizedElement(
                        "line",
                        {
                            "x1": vw(0),
                            "x2": tick_length,
                            "y1": interior_y,
                            "y2": interior_y,
                        },
                    )
                )
        else:
            # Fallback: draw axis line (only if we have at least one tick) and all ticks
            if len(y_ticks) >= 1:
                g.append(
                    Element(
                        "line",
                        {
                            "x1": tick_length,
                            "x2": tick_length,
                            "y1": y_ticks[0],
                            "y2": y_ticks[-1],
                        },
                    )
                )
            g.append(
                VectorizedElement(
                    "line",
                    {
                        "x1": vw(0),
                        "x2": tick_length,
                        "y1": y_ticks,
                        "y2": y_ticks,
                    },
                )
            )

        return g.resolve(ctx)

    @override
    def abs_bounds(self) -> tuple[AbsLengths, AbsLengths]:
        tick_length = self.attrib["tick_length"]
        assert isinstance(tick_length, AbsLengths)

        return (tick_length, mm(0))


def yticks(*args, **kwargs):
    return YTicks(*args, **kwargs)
