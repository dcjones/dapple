from ..elements import Element, VectorizedElement, RawText
from ..coordinates import (
    AbsLengths,
    CoordBounds,
    ResolveContext,
    Lengths,
    Resolvable,
    Serializable,
    mm,
    vw,
    vh,
)
from ..layout import Position
from ..config import ConfigKey
from ..textextents import Font
from ..scales import ScaleSet, Scale

from typing import override, final
import numpy as np


class RotateTransforms(Resolvable, Serializable):
    """
    Generates a list of rotation transform strings for vectorized elements.
    Each transform rotates around the corresponding (x, y) coordinate.
    """

    angle: float
    x: Lengths
    y: Lengths

    def __init__(self, angle: float, x: Lengths, y: Lengths):
        self.angle = angle
        self.x = x
        self.y = y

    @override
    def resolve(self, ctx: ResolveContext) -> "RotateTransforms":
        """Resolve the x and y coordinates to absolute lengths."""
        from ..coordinates import resolve

        resolved_x = resolve(self.x, ctx)
        resolved_y = resolve(self.y, ctx)
        return RotateTransforms(self.angle, resolved_x, resolved_y)

    @override
    def serialize(self) -> list[str] | None:
        # At serialization time, x and y should be resolved to AbsLengths
        if isinstance(self.x, AbsLengths) and isinstance(self.y, AbsLengths):
            x_vals = self.x.serialize()
            y_vals = self.y.serialize()

            # Handle both scalar and vector cases
            if isinstance(x_vals, str):
                x_list: list[str] = [x_vals]
            else:
                x_list = x_vals if x_vals is not None else []

            if isinstance(y_vals, str):
                y_list: list[str] = [y_vals]
            else:
                y_list = y_vals if y_vals is not None else []

            # If x is scalar but y is vector, repeat x for each y
            if len(x_list) == 1 and len(y_list) > 1:
                x_list = x_list * len(y_list)

            # If y is scalar but x is vector, repeat y for each x
            if len(y_list) == 1 and len(x_list) > 1:
                y_list = y_list * len(x_list)

            return [f"rotate({self.angle}, {x}, {y})" for x, y in zip(x_list, y_list)]
        else:
            # Not yet resolved, return None to skip serialization
            return None


class XLabel(Element):
    """
    Draw a text label along the bottom of the plot for the x-axis.
    """

    def __init__(
        self,
        text: str,
        font_family=ConfigKey("label_font_family"),
        font_size=ConfigKey("label_font_size"),
        fill=ConfigKey("label_fill"),
    ):
        attrib: dict[str, object] = {
            "dapple:position": Position.BottomCenter,
            "dapple:padding-top": ConfigKey("padding_std"),
            "text": text,
            "font_family": font_family,
            "font_size": font_size,
            "fill": fill,
        }
        super().__init__("dapple:xlabel", attrib)  # type: ignore

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        text = self.attrib["text"]
        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]
        fill = self.attrib["fill"]

        assert isinstance(text, str)
        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)

        # Get text extents using Font
        font = Font(font_family, font_size)
        _text_width, _text_height, text_baseline = font.get_extents_with_baseline(text)

        x_scale = ctx.scales["x"]
        _x_labels, x_ticks = x_scale.ticks()
        assert isinstance(x_ticks, Lengths)

        # Center the text horizontally using text-anchor
        x = 0.5 * (x_ticks[0] + x_ticks[-1])
        y = text_baseline  # Position from top of the space allocated

        text_element = Element(
            "text",
            {
                "x": x,
                "y": y,
                "font-family": font_family,
                "font-size": font_size,
                "fill": fill,
                "text-anchor": "middle",  # Center horizontally
            },
        )
        text_element.text = text

        return text_element.resolve(ctx)

    @override
    def abs_bounds(self) -> tuple[AbsLengths, AbsLengths]:
        text = self.attrib["text"]
        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]

        assert isinstance(text, str)
        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)

        font = Font(font_family, font_size)
        text_width, text_height = font.get_extents(text)

        return (text_width, text_height)


def xlabel(text: str, *args, **kwargs):
    return XLabel(text, *args, **kwargs)


class YLabel(Element):
    """
    Draw a rotated text label along the left side of the plot for the y-axis.
    """

    def __init__(
        self,
        text: str,
        font_family=ConfigKey("label_font_family"),
        font_size=ConfigKey("label_font_size"),
        fill=ConfigKey("label_fill"),
    ):
        attrib: dict[str, object] = {
            "dapple:position": Position.LeftCenter,
            "dapple:padding-right": ConfigKey("padding_std"),
            "text": text,
            "font_family": font_family,
            "font_size": font_size,
            "fill": fill,
        }
        super().__init__("dapple:ylabel", attrib)  # type: ignore

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        text = self.attrib["text"]
        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]
        fill = self.attrib["fill"]

        assert isinstance(text, str)
        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)

        # Get text extents using Font
        font = Font(font_family, font_size)
        _text_width, text_height = font.get_extents(text)

        y_scale = ctx.scales["y"]
        _y_labels, y_ticks = y_scale.ticks()
        assert isinstance(y_ticks, Lengths)

        # Position the text centered vertically and rotated -90 degrees
        x = text_height  # Distance from left edge
        y = 0.5 * (y_ticks[0] + y_ticks[-1])  # Center vertically

        # Resolve coordinates for the transform attribute
        x_resolved = x.resolve(ctx)
        y_resolved = y.resolve(ctx)

        text_element = Element(
            "text",
            {
                "x": x,
                "y": y,
                "font-family": font_family,
                "font-size": font_size,
                "fill": fill,
                "text-anchor": "middle",
                "transform": f"rotate(-90, {x_resolved.serialize()}, {y_resolved.serialize()})",
            },
        )
        text_element.text = text

        return text_element.resolve(ctx)

    @override
    def abs_bounds(self) -> tuple[AbsLengths, AbsLengths]:
        text = self.attrib["text"]
        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]

        assert isinstance(text, str)
        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)

        font = Font(font_family, font_size)
        text_width, text_height = font.get_extents(text)

        # When rotated -90°, text height becomes the width requirement
        return (text_height, text_width)


def ylabel(text: str, *args, **kwargs):
    return YLabel(text, *args, **kwargs)


class Title(Element):
    """
    Draw a text title at the top center of the plot.
    """

    def __init__(
        self,
        text: str,
        font_family=ConfigKey("title_font_family"),
        font_size=ConfigKey("title_font_size"),
        fill=ConfigKey("title_fill"),
    ):
        attrib: dict[str, object] = {
            "dapple:position": Position.TopCenter,
            "dapple:padding-bottom": ConfigKey("padding_std"),
            "text": text,
            "font_family": font_family,
            "font_size": font_size,
            "fill": fill,
        }
        super().__init__("dapple:title", attrib)  # type: ignore

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        text = self.attrib["text"]
        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]
        fill = self.attrib["fill"]

        assert isinstance(text, str)
        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)

        # Get text extents using Font
        font = Font(font_family, font_size)
        _text_width, text_height = font.get_extents(text)

        # Center the text horizontally using text-anchor
        x = vw(0.5)
        y = text_height  # Position from top of the space allocated

        text_element = Element(
            "text",
            {
                "x": x,
                "y": y,
                "font-family": font_family,
                "font-size": font_size,
                "fill": fill,
                "text-anchor": "middle",  # Center horizontally
            },
        )
        text_element.text = text

        return text_element.resolve(ctx)

    @override
    def abs_bounds(self) -> tuple[AbsLengths, AbsLengths]:
        text = self.attrib["text"]
        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]

        assert isinstance(text, str)
        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)

        font = Font(font_family, font_size)
        text_width, text_height = font.get_extents(text)

        return (text_width, text_height)


def title(text: str, *args, **kwargs):
    return Title(text, *args, **kwargs)


@final
class XTickLabels(Element):
    """
    Draw tick labels along the bottom margin of the plot for the x-axis.
    """

    root: Element | None
    tick_labels = list[str] | None
    tick_positions = Lengths | None

    def __init__(
        self,
        font_family=ConfigKey("tick_label_font_family"),
        font_size=ConfigKey("tick_label_font_size"),
        fill=ConfigKey("tick_label_fill"),
        rotate: bool = False,
    ):
        attrib: dict[str, object] = {
            "dapple:position": Position.BottomLeft,
            "font_family": font_family,
            "font_size": font_size,
            "fill": fill,
            "dapple:rotate": rotate,
        }
        super().__init__("dapple:xticklabels", attrib)  # type: ignore
        self._tick_labels = None
        self.root = None
        self.tick_labels = None
        self.tick_positions = None

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        assert self.root is not None
        return self.root.resolve(ctx)

    @override
    def apply_scales(self, scales: ScaleSet):
        """
        Once we are informed of the scales, we can acess the ticks and generate
        all the geometry.
        """

        if "x" not in scales:
            self.root = Element("g")
            return

        x_scale = scales["x"]
        assert isinstance(x_scale, Scale)
        tick_labels, tick_positions = x_scale.ticks()
        assert isinstance(tick_positions, Lengths)

        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]
        fill = self.attrib["fill"]
        rotate = self.attrib["dapple:rotate"]

        g_attrib = {
            "font-family": font_family,
            "font-size": font_size,
            "fill": fill,
        }

        if rotate:
            g_attrib["text-anchor"] = "end"
            g_attrib["dominant-baseline"] = "middle"
        else:
            g_attrib["text-anchor"] = "middle"  # Center horizontally
            g_attrib["dominant-baseline"] = "hanging"

        g = Element("g", g_attrib)

        text_attrib: dict[str, object] = {
            "x": tick_positions,
        }

        if rotate:
            # Create a transform for each label that rotates around its anchor point
            text_attrib["y"] = vh(0)
            text_attrib["transform"] = RotateTransforms(-90, tick_positions, vh(0))
        else:
            text_attrib["y"] = vh(0)

        g.append(
            VectorizedElement(
                "text",
                text_attrib,
                *map(RawText, tick_labels),
            )
        )
        self.root = g
        self.tick_labels = tick_labels
        self.tick_positions = tick_positions

    @override
    def update_bounds(self, bounds: CoordBounds):
        if self.tick_positions is None or self.tick_labels is None:
            return

        assert isinstance(self.tick_labels, np.ndarray)
        assert isinstance(self.tick_positions, Lengths)

        if len(self.tick_labels) == 0 or len(self.tick_positions) == 0:
            return

        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]
        rotate = self.attrib["dapple:rotate"]

        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)

        font = Font(font_family, font_size)

        l0 = self.tick_labels[0]
        assert isinstance(l0, str)
        l0_width, l0_height = font.get_extents(str(l0))
        x0 = self.tick_positions[0]

        ln = self.tick_labels[-1]
        assert isinstance(ln, str)
        ln_width, ln_height = font.get_extents(str(ln))
        xn = self.tick_positions[-1]

        if rotate:
            # When rotated 90 degrees, height becomes width in the x-direction
            # add a little slop
            l0_height += mm(1)
            ln_height += mm(1)

            bounds.update(x0 - 0.5 * l0_height)
            bounds.update(x0 + 0.5 * l0_height)
            bounds.update(xn - 0.5 * ln_height)
            bounds.update(xn + 0.5 * ln_height)
        else:
            bounds.update(x0 - 0.5 * l0_width)
            bounds.update(x0 + 0.5 * l0_width)
            bounds.update(xn - 0.5 * ln_width)
            bounds.update(xn + 0.5 * ln_width)

    @override
    def abs_bounds(self) -> tuple[AbsLengths, AbsLengths]:
        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]
        rotate = self.attrib["dapple:rotate"]

        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)

        font = Font(font_family, font_size)

        if self.tick_labels is not None:
            # Calculate precise bounds based on actual tick labels
            max_width = mm(0)
            max_height = mm(0)

            for label in self.tick_labels:
                label_width, label_height = font.get_extents(str(label))
                if label_width.scalar_value() > max_width.scalar_value():
                    max_width = label_width
                if label_height.scalar_value() > max_height.scalar_value():
                    max_height = label_height

            if rotate:
                # When rotated 90 degrees, width and height are swapped
                return (max_height, max_width)
            else:
                return (max_width, max_height)
        else:
            # Fall back to estimate based on a typical label
            typical_width, typical_height = font.get_extents("0.00")
            if rotate:
                return (typical_height, typical_width)
            else:
                return (typical_width, typical_height)


def xticklabels(*args, **kwargs):
    return XTickLabels(*args, **kwargs)


@final
class YTickLabels(Element):
    """
    Draw tick labels along the left margin of the plot for the y-axis.
    """

    root: Element | None
    tick_labels = list[str] | None
    tick_positions = Lengths | None

    def __init__(
        self,
        font_family=ConfigKey("tick_label_font_family"),
        font_size=ConfigKey("tick_label_font_size"),
        fill=ConfigKey("tick_label_fill"),
        rotate: bool = False,
    ):
        attrib: dict[str, object] = {
            "dapple:position": Position.LeftTop,
            "font_family": font_family,
            "font_size": font_size,
            "fill": fill,
            "dapple:rotate": rotate,
        }
        super().__init__("dapple:yticklabels", attrib)  # type: ignore
        self.root = None
        self.tick_labels = None
        self.tick_positions = None

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        assert self.root is not None
        return self.root.resolve(ctx)

    @override
    def apply_scales(self, scales: ScaleSet):
        """
        Once we are informed of the scales, we can acess the ticks and generate
        all the geometry.
        """
        if "y" not in scales:
            self.root = Element("g")
            return

        y_scale = scales["y"]
        assert isinstance(y_scale, Scale)
        tick_labels, tick_positions = y_scale.ticks()
        assert isinstance(tick_positions, Lengths)

        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]
        fill = self.attrib["fill"]
        rotate = self.attrib["dapple:rotate"]

        g_attrib = {
            "font-family": font_family,
            "font-size": font_size,
            "fill": fill,
        }

        if rotate:
            g_attrib["text-anchor"] = "middle"
            g_attrib["dominant-baseline"] = "auto"
        else:
            g_attrib["text-anchor"] = "end"  # Right-align text
            g_attrib["dominant-baseline"] = "middle"  # Center vertically

        g = Element("g", g_attrib)

        text_attrib: dict[str, object] = {
            "x": vw(1),
            "y": tick_positions,
        }

        if rotate:
            # Create a transform for each label that rotates around its anchor point
            text_attrib["transform"] = RotateTransforms(-90, vw(1), tick_positions)

        g.append(
            VectorizedElement(
                "text",
                text_attrib,
                *map(RawText, tick_labels),
            )
        )
        self.root = g
        self.tick_labels = tick_labels
        self.tick_positions = tick_positions

    @override
    def update_bounds(self, bounds: CoordBounds):
        if self.tick_positions is None or self.tick_labels is None:
            return

        assert isinstance(self.tick_labels, np.ndarray)
        assert isinstance(self.tick_positions, Lengths)

        if len(self.tick_labels) == 0 or len(self.tick_positions) == 0:
            return

        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]
        rotate = self.attrib["dapple:rotate"]

        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)

        font = Font(font_family, font_size)

        l0 = self.tick_labels[0]
        assert isinstance(l0, str)
        l0_width, l0_height = font.get_extents(str(l0))
        y0 = self.tick_positions[0]

        ln = self.tick_labels[-1]
        assert isinstance(ln, str)
        ln_width, ln_height = font.get_extents(str(ln))
        yn = self.tick_positions[-1]

        if rotate:
            # When rotated 90 degrees, width becomes height in the y-direction
            # add a little slop
            l0_width += mm(1)
            ln_width += mm(1)

            bounds.update(y0 - 0.5 * l0_width)
            bounds.update(y0 + 0.5 * l0_width)
            bounds.update(yn - 0.5 * ln_width)
            bounds.update(yn + 0.5 * ln_width)
        else:
            # add a little slop, since centering vertically doesn't necessarily put it
            # on the centroid
            l0_height += mm(1)
            ln_height += mm(1)

            bounds.update(y0 - 0.5 * l0_height)
            bounds.update(y0 + 0.5 * l0_height)
            bounds.update(yn - 0.5 * ln_height)
            bounds.update(yn + 0.5 * ln_height)

    @override
    def abs_bounds(self) -> tuple[AbsLengths, AbsLengths]:
        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]
        rotate = self.attrib["dapple:rotate"]

        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)

        font = Font(font_family, font_size)

        if self.tick_labels is not None:
            # Calculate precise bounds based on actual tick labels
            max_width = mm(0)
            max_height = mm(0)

            for label in self.tick_labels:
                label_width, label_height = font.get_extents(str(label))
                if label_width.scalar_value() > max_width.scalar_value():
                    max_width = label_width
                if label_height.scalar_value() > max_height.scalar_value():
                    max_height = label_height

            if rotate:
                # When rotated 90 degrees, width and height are swapped
                return (max_height, max_width)
            else:
                return (max_width, max_height)
        else:
            # Fall back to estimate based on a typical label
            typical_width, typical_height = font.get_extents("0.00")
            if rotate:
                return (typical_height, typical_width)
            else:
                return (typical_width, typical_height)


def yticklabels(*args, **kwargs):
    return YTickLabels(*args, **kwargs)
