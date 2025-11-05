from ..elements import Element, VectorizedElement, pad, Path
from ..coordinates import (
    AbsLengths,
    CtxLengths,
    CtxLenType,
    CoordTransform,
    Resolvable,
    ResolveContext,
    Lengths,
    vh,
    vhv,
    vw,
    cy,
    mm,
)
from ..layout import Position
from ..config import ConfigKey
from ..textextents import Font
from ..colors import Colors
from ..scales import ScaleDiscreteColor, ScaleContinuousColor
from .bars import Bar

from typing import override, cast
import numpy as np


class Key(Element):
    """
    Draw a color key/legend that shows colors and their labels.
    By default positioned to the right of the plot.
    """

    def __init__(
        self,
        font_family=ConfigKey("tick_label_font_family"),
        font_size=ConfigKey("tick_label_font_size"),
        font_weight=ConfigKey("tick_label_font_weight"),
        fill=ConfigKey("tick_label_fill"),
        square_size=ConfigKey("key_square_size"),
        spacing=ConfigKey("key_spacing"),
        gradient_width=ConfigKey("key_gradient_width"),
        stroke_width=ConfigKey("tick_stroke_width"),
        tick_length=ConfigKey("tick_length"),
        position=ConfigKey(key="key_position"),
    ):
        attrib: dict[str, object] = {
            "dapple:position": position,
            "dapple:padding-left": ConfigKey("padding_std"),
            "dapple:padding-right": ConfigKey("padding_nil"),
            "dapple:padding-top": ConfigKey("padding_nil"),
            "dapple:padding-bottom": ConfigKey("padding_nil"),
            "font_family": font_family,
            "font_size": font_size,
            "font_weight": font_weight,
            "fill": fill,
            "square_size": square_size,
            "spacing": spacing,
            "gradient_width": gradient_width,
            "stroke-width": stroke_width,
            "tick_length": tick_length,
        }
        super().__init__("dapple:key", attrib)  # type: ignore
        self._color_scale = None
        self._labels = None
        self._colors = None

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        if self._color_scale is None:
            # No color scale found, return empty group
            return Element("g", {})

        if isinstance(self._color_scale, ScaleDiscreteColor):
            return self._resolve_discrete(ctx)
        elif isinstance(self._color_scale, ScaleContinuousColor):
            return self._resolve_continuous(ctx)
        else:
            # Unknown scale type, return empty group
            return Element("g", {})

    def _resolve_discrete(self, ctx: ResolveContext) -> Element:
        """Resolve discrete color scale into squares with labels."""
        assert self._labels is not None
        assert self._colors is not None

        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]
        font_weight = self.attrib["font_weight"]
        fill = self.attrib["fill"]
        square_size = self.attrib["square_size"]
        spacing = self.attrib["spacing"]

        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)
        assert isinstance(square_size, AbsLengths)
        assert isinstance(spacing, AbsLengths)

        g = Element(
            "g",
            {
                "font-family": font_family,
                "font-size": font_size,
                "font-weight": font_weight,
                "fill": fill,
            },
        )

        # Calculate positions for each key item
        current_y_val = 0.0
        square_size_val = square_size.scalar_value()
        spacing_val = spacing.scalar_value()

        # Create arrays for vectorized rendering
        y_positions_vals = []
        for i in range(len(self._labels)):
            y_positions_vals.append(current_y_val)
            current_y_val += square_size_val + spacing_val

        # Create AbsLengths array for y positions
        y_positions_array = np.array(y_positions_vals)
        y_positions = mm(y_positions_array)

        # Draw color squares using VectorizedElement
        g.append(
            VectorizedElement(
                "rect",
                {
                    "x": mm(0),
                    "y": y_positions,
                    "width": square_size,
                    "height": square_size,
                    "fill": self._colors,
                },
            )
        )

        # Draw labels - text content cannot be easily vectorized, so use individual elements
        for i, label in enumerate(self._labels):
            text_element = Element(
                "text",
                {
                    "x": mm(square_size_val + spacing_val),
                    "y": mm(y_positions_vals[i] + square_size_val * 0.5),
                    "dominant-baseline": "middle",
                },
            )
            text_element.text = str(label)
            g.append(text_element)

        return g.resolve(ctx)

    def _resolve_continuous(self, ctx: ResolveContext) -> Element:
        """Resolve continuous color scale into gradient with tick marks."""
        assert self._color_scale is not None

        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]
        font_weight = self.attrib["font_weight"]
        fill = self.attrib["fill"]
        spacing = self.attrib["spacing"]
        gradient_width = self.attrib["gradient_width"]
        stroke_width = self.attrib["stroke-width"]
        tick_length = self.attrib["tick_length"]

        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)
        assert isinstance(spacing, AbsLengths)
        assert isinstance(gradient_width, AbsLengths)
        assert isinstance(stroke_width, AbsLengths)
        assert isinstance(tick_length, AbsLengths)

        font = Font(font_family, font_size)

        # Get tick information from the continuous scale
        tick_labels, tick_positions = self._color_scale.ticks()

        # Create gradient definition
        gradient_id = "key-gradient"
        gradient_height = vhv(1)  # Use full plot height
        spacing_val = spacing.scalar_value()

        # TODO: Fucke. We need to use the bar geometry!

        g = Element(
            "g",
            {
                "font-family": font_family,
                "font-size": font_size,
                "font-weight": font_weight,
                "fill": fill,
                "dapple:coords": {"vh": CoordTransform(vh(-1), vh(1))},
            },
        )

        # Create defs element for gradient
        defs = Element("defs")
        linear_gradient = Element(
            "linearGradient",
            {
                "id": gradient_id,
                "x1": "0%",
                "y1": "100%",
                "x2": "0%",
                "y2": "0%",
            },
        )

        # Add color stops based on the colormap
        # For continuous scales, we'll create a smooth gradient
        n_stops = 20
        for i in range(n_stops):
            position = i / (n_stops - 1)
            # Map position to color using the scale's colormap
            if hasattr(self._color_scale, "colormap"):
                colormap = self._color_scale.colormap
                # Call the colormap with a numpy array
                color_rgba = colormap(np.array([position]))
                # Convert to Colors object and serialize
                colors_obj = Colors(color_rgba)
                color_hex = colors_obj.serialize()
                if isinstance(color_hex, list) and len(color_hex) > 0:
                    color_hex = color_hex[0]
            else:
                color_hex = "#000000"  # fallback

            stop = Element(
                "stop",
                {
                    "offset": f"{position * 100}%",
                    "stop-color": color_hex,
                },
            )
            linear_gradient.append(stop)

        defs.append(linear_gradient)
        g.append(defs)

        # Draw gradient rectangle
        gradient_rect = Bar(
            mm(0),
            vh(0),
            gradient_width,
            gradient_height,
            fill=f"url(#{gradient_id})",
        )

        g.append(gradient_rect)

        # Calculate tick positions using vh coordinates
        tick_positions_vals = []
        for i in range(len(tick_labels)):
            if len(tick_labels) > 1:
                tick_positions_vals.append(i / (len(tick_labels) - 1))
            else:
                tick_positions_vals.append(0.5)

        # Create Lengths array for tick positions
        tick_y_positions = CtxLengths(
            np.array(tick_positions_vals), "vh", CtxLenType.Pos
        )

        # Draw axis line + end ticks as a single path when we have at least two ticks
        if len(tick_y_positions) >= 2:
            gw = gradient_width.scalar_value()
            tl = tick_length.scalar_value()
            combined_x = mm([gw + tl, gw, gw, gw + tl])
            combined_y = CtxLengths(
                np.array([0.0, 0.0, 1.0, 1.0]), "vh", CtxLenType.Pos
            )

            g.append(
                Path(
                    combined_x,
                    combined_y,
                    **{"stroke": "black", "stroke-width": stroke_width, "fill": "none"},
                )
            )

            # Interior tick marks (exclude the two ends)
            interior_vals = tick_y_positions.values[1:-1]
            interior_y = CtxLengths(
                interior_vals, tick_y_positions.unit, tick_y_positions.typ
            )
            if len(interior_y) > 0:
                g.append(
                    VectorizedElement(
                        "line",
                        {
                            "x1": gradient_width,
                            "x2": gradient_width + tick_length,
                            "y1": interior_y,
                            "y2": interior_y,
                            "stroke": "black",
                            "stroke-width": stroke_width,
                        },
                    )
                )
        else:
            # Fallback: draw axis line and all ticks individually
            g.append(
                Element(
                    "line",
                    attrib={
                        "x1": gradient_width,
                        "x2": gradient_width,
                        "y1": vh(0),
                        "y2": vh(1),
                        "stroke": "black",
                        "stroke-width": stroke_width,
                    },
                )
            )

            g.append(
                VectorizedElement(
                    "line",
                    {
                        "x1": gradient_width,
                        "x2": gradient_width + tick_length,
                        "y1": tick_y_positions,
                        "y2": tick_y_positions,
                        "stroke": "black",
                        "stroke-width": stroke_width,
                    },
                )
            )

        for label, y_pos in zip(tick_labels, tick_y_positions):
            text_element = Element(
                "text",
                {
                    "x": gradient_width + tick_length + spacing,
                    "y": y_pos,
                    "dominant-baseline": "middle",
                },
            )
            text_element.text = str(label)
            g.append(text_element)

        top_padding = mm(0)
        bottom_padding = mm(0)
        nudge = mm(1)
        if len(tick_y_positions) > 1:
            if tick_y_positions[0] == vh(0):
                _text_width, text_height = font.get_extents(str(tick_labels[0]))
                bottom_padding = 0.5 * text_height + nudge
            if tick_y_positions[-1] == vh(1):
                _text_width, text_height = font.get_extents(str(tick_labels[-1]))
                top_padding = 0.5 * text_height + nudge

        return pad(
            g,
            top=cast(AbsLengths, top_padding),
            bottom=cast(AbsLengths, bottom_padding),
        ).resolve(ctx)

    @override
    def apply_scales(self, scales):
        """Store color scale information for precise rendering."""
        if "color" in scales:
            self._color_scale = scales["color"]
            if isinstance(self._color_scale, ScaleDiscreteColor):
                self._labels, self._colors = self._color_scale.ticks()
            elif isinstance(self._color_scale, ScaleContinuousColor):
                # For continuous scales, we'll get tick info during resolve
                pass

    @override
    def abs_bounds(self) -> tuple[AbsLengths, AbsLengths]:
        if self._color_scale is None:
            return mm(0), mm(0)

        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]
        square_size = self.attrib["square_size"]
        spacing = self.attrib["spacing"]
        gradient_width = self.attrib.get("gradient_width", mm(20))

        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)
        assert isinstance(square_size, AbsLengths)
        assert isinstance(spacing, AbsLengths)
        assert isinstance(gradient_width, AbsLengths)

        font = Font(font_family, font_size)

        if (
            isinstance(self._color_scale, ScaleDiscreteColor)
            and self._labels is not None
        ):
            # Calculate bounds for discrete scale
            max_text_width_val = 0.0
            total_height_val = 0.0

            square_size_val = square_size.scalar_value()
            spacing_val = spacing.scalar_value()

            for i, label in enumerate(self._labels):
                text_width, text_height = font.get_extents(str(label))
                text_width_val = text_width.scalar_value()
                if text_width_val > max_text_width_val:
                    max_text_width_val = text_width_val

                # Add square size plus spacing for each item
                total_height_val += square_size_val
                if i < len(self._labels) - 1:  # No spacing after last item
                    total_height_val += spacing_val

            # Calculate total width: square + spacing + text
            total_width_val = square_size_val + spacing_val + max_text_width_val

            return (mm(total_width_val), mm(total_height_val))

        elif isinstance(self._color_scale, ScaleContinuousColor):
            # Calculate bounds for continuous scale
            gradient_width_val = gradient_width.scalar_value()
            tick_length_val = 3.0
            spacing_val = spacing.scalar_value()

            # Measure actual tick labels to get precise width
            max_text_width_val = 0.0
            total_text_height_val = 0.0

            # Get tick labels from scale to measure them
            tick_labels, _ = self._color_scale.ticks()

            for label in tick_labels:
                text_width, text_height = font.get_extents(str(label))
                text_width_val = text_width.scalar_value()
                text_height_val = text_height.scalar_value()

                if text_width_val > max_text_width_val:
                    max_text_width_val = text_width_val

                total_text_height_val += text_height_val

            total_width_val = (
                gradient_width_val + tick_length_val + spacing_val + max_text_width_val
            )
            return (mm(total_width_val), mm(total_text_height_val))

        else:
            return mm(0), mm(0)


def key(*args, **kwargs):
    """Create a Key geometry."""
    return Key(*args, **kwargs)
