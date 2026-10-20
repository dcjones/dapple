from typing import cast, override

import numpy as np
from cmap import Colormap
from numpy.typing import NDArray

from ..colors import Colors
from ..config import ConfigKey
from ..coordinates import (
    AbsLengths,
    CoordTransform,
    CtxLengths,
    CtxLenType,
    Lengths,
    Resolvable,
    ResolveContext,
    cy,
    mm,
    translate,
    vh,
    vhv,
    vw,
)
from ..elements import Element, Path, VectorizedElement, pad
from ..layout import Position
from ..scales import ScaleContinuousColor, ScaleDiscreteColor, ScaleDiscreteShape
from ..shapes import ShapePaths, shape_path_d
from ..textextents import Font
from .bars import Bar


def _continuous_key_layout(
    scale: ScaleContinuousColor,
) -> tuple[NDArray[np.str_], NDArray[np.float64], list[tuple[float, str]]]:
    """
    Compute tick labels, tick positions, and gradient stops for a continuous
    color key.

    The gradient spans the union of the scale's range and its ticks, so ticks
    falling outside the data range are still drawn at their true positions.

    Returns:
        A tuple ``(labels, fractions, stops)`` where ``fractions`` gives each
        tick's position along the gradient in ``[0, 1]`` (0 at the minimum)
        and ``stops`` is a list of ``(offset, color_hex)`` gradient stops.
    """
    labels, ticks = scale.ticks()
    assert isinstance(ticks, CtxLengths)
    assert scale.min is not None and scale.max is not None
    tick_vals = np.asarray(ticks.values, dtype=np.float64)

    lo = float(scale.min)
    hi = float(scale.max)
    if len(tick_vals) > 0:
        lo = min(lo, float(tick_vals.min()))
        hi = max(hi, float(tick_vals.max()))

    if hi > lo:
        fractions = (tick_vals - lo) / (hi - lo)
    else:
        fractions = np.full(len(tick_vals), 0.5)

    n_stops = 20
    offsets = np.linspace(0.0, 1.0, n_stops)
    colormap = scale.colormap
    assert isinstance(colormap, Colormap)
    stop_colors = Colors(colormap(scale.normalize(lo + offsets * (hi - lo))))
    color_hexes = stop_colors.serialize()
    assert isinstance(color_hexes, list)
    stops = [(float(o), str(c)) for o, c in zip(offsets, color_hexes)]

    return labels, fractions, stops


def _gradient_def(gradient_id: str, stops: list[tuple[float, str]]) -> Element:
    """Build a bottom-to-top ``linearGradient`` element from color stops."""
    linear_gradient = Element(
        "linearGradient",
        {"id": gradient_id, "x1": "0%", "y1": "100%", "x2": "0%", "y2": "0%"},
    )
    for offset, color_hex in stops:
        linear_gradient.append(
            Element("stop", {"offset": f"{offset * 100}%", "stop-color": color_hex})
        )
    return linear_gradient


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
        exclude: list | None = None,
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
        self._shape_scale = None
        self._shape_labels = None
        self._shapes = None
        self._exclude: set = set(exclude) if exclude is not None else set()

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        has_shape = self._shape_scale is not None
        color = self._color_scale

        if color is None and not has_shape:
            # No scale found, return empty group
            return Element("g", {})

        # Shape scale only.
        if color is None and has_shape:
            return self._resolve_shape(ctx)

        # Color scale only (preserve the original standalone behavior).
        if not has_shape:
            if isinstance(color, ScaleDiscreteColor):
                return self._resolve_discrete(ctx)
            elif isinstance(color, ScaleContinuousColor):
                return self._resolve_continuous(ctx)
            else:
                return Element("g", {})

        # Both color and shape scales are present: build a combined key.
        return self._stack_sections(self._combined_sections()).resolve(ctx)

    def _should_merge(self) -> bool:
        """
        Merge the color and shape guides into one section when they encode the
        same discrete variable, i.e. they share identical (ordered) labels.
        """
        if not isinstance(self._color_scale, ScaleDiscreteColor):
            return False
        if self._labels is None or self._shape_labels is None:
            return False
        return [str(x) for x in self._labels] == [str(x) for x in self._shape_labels]

    def _combined_sections(self) -> list[tuple[Element, float, float]]:
        """
        Build the ordered list of legend sections for the combined key, each as
        an ``(element, width_mm, height_mm)`` tuple in local (top-left) mm coords.
        """
        if self._should_merge():
            # One row per category, drawing each shape glyph in its color.
            return [self._marker_section(list(self._labels), self._colors, self._shapes)]

        sections: list[tuple[Element, float, float]] = []

        # Shape guide first.
        assert self._shape_labels is not None
        sections.append(
            self._marker_section(list(self._shape_labels), None, self._shapes)
        )

        # Then the color guide.
        if isinstance(self._color_scale, ScaleDiscreteColor):
            assert self._labels is not None
            sections.append(
                self._marker_section(list(self._labels), self._colors, None)
            )
        elif isinstance(self._color_scale, ScaleContinuousColor):
            sections.append(self._continuous_section())

        return sections

    def _section_gap(self) -> float:
        spacing = self.attrib["spacing"]
        assert isinstance(spacing, AbsLengths)
        return 2.0 * spacing.scalar_value()

    def _stack_sections(self, sections: list[tuple[Element, float, float]]) -> Element:
        """Stack legend sections vertically, separated by ``_section_gap``."""
        gap = self._section_gap()
        g = Element("g")
        y = 0.0
        for element, _w, h in sections:
            g.append(
                Element("g", {"transform": translate(mm(0), mm(y))}, element)
            )
            y += h + gap
        return g

    def _filtered_labels(self):
        """Return labels with excluded entries removed."""
        assert self._labels is not None
        if not self._exclude:
            return list(self._labels)
        return [lbl for lbl in self._labels if str(lbl) not in self._exclude and lbl not in self._exclude]

    def _filtered_labels_colors(self):
        """Return labels and colors with excluded entries removed."""
        assert self._labels is not None
        assert self._colors is not None
        if not self._exclude:
            return list(self._labels), self._colors
        mask = np.array([str(lbl) not in self._exclude and lbl not in self._exclude for lbl in self._labels])
        labels = [lbl for lbl, keep in zip(self._labels, mask) if keep]
        colors = Colors(self._colors.values[mask, :])
        return labels, colors

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

        # Reverse so the first stack segment (bottom of chart) appears at the
        # bottom of the key, matching the visual stacking order.
        filtered_labels, filtered_colors = self._filtered_labels_colors()
        labels = filtered_labels[::-1]
        colors = Colors(filtered_colors.values[::-1, :])

        # Create arrays for vectorized rendering
        y_positions_vals = []
        for i in range(len(labels)):
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
                    "fill": colors,
                },
            )
        )

        # Draw labels - text content cannot be easily vectorized, so use individual elements
        for i, label in enumerate(labels):
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

    def _resolve_shape(self, ctx: ResolveContext) -> Element:
        """Resolve discrete shape scale into markers with labels."""
        assert self._shape_labels is not None
        section, _w, _h = self._marker_section(
            list(self._shape_labels), None, self._shapes
        )
        return section.resolve(ctx)

    def _marker_section(
        self, labels: list, colors, shapes
    ) -> tuple[Element, float, float]:
        """
        Build one discrete legend section as an ``(element, width, height)`` tuple.

        Swatches are drawn as:
          * colored squares when only ``colors`` is given (plain color legend),
          * shape glyphs when ``shapes`` is given, filled with ``colors`` if
            supplied (merged color+shape legend) or the default fill otherwise.
        """
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

        # Filter out excluded labels, keeping colors/shapes aligned.
        keep = [
            i
            for i, lbl in enumerate(labels)
            if str(lbl) not in self._exclude and lbl not in self._exclude
        ]
        labels = [labels[i] for i in keep]
        if colors is not None:
            colors = Colors(colors.values[keep, :])
        shape_indices = shapes.indices[keep] if shapes is not None else None

        square_size_val = square_size.scalar_value()
        spacing_val = spacing.scalar_value()
        radius = 0.4 * square_size_val

        n = len(labels)
        y_positions_vals = [i * (square_size_val + spacing_val) for i in range(n)]

        g = Element(
            "g",
            {
                "font-family": font_family,
                "font-size": font_size,
                "font-weight": font_weight,
                "fill": fill,
            },
        )

        if n > 0:
            if shapes is None:
                assert colors is not None
                g.append(
                    VectorizedElement(
                        "rect",
                        {
                            "x": mm(0),
                            "y": mm(np.array(y_positions_vals)),
                            "width": square_size,
                            "height": square_size,
                            "fill": colors,
                        },
                    )
                )
            else:
                assert shape_indices is not None
                ds = [
                    shape_path_d(
                        int(shape_indices[i]),
                        square_size_val * 0.5,
                        y_positions_vals[i] + square_size_val * 0.5,
                        radius,
                    )
                    for i in range(n)
                ]
                swatch_fill = colors if colors is not None else fill
                g.append(
                    VectorizedElement(
                        "path", {"d": ShapePaths(ds), "fill": swatch_fill}
                    )
                )

        font = Font(font_family, font_size)
        max_text_width_val = 0.0
        for i, label in enumerate(labels):
            text_width, _text_height = font.get_extents(str(label))
            max_text_width_val = max(max_text_width_val, text_width.scalar_value())

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

        width_val = square_size_val + spacing_val + max_text_width_val
        height_val = n * square_size_val + max(n - 1, 0) * spacing_val
        return g, width_val, height_val

    def _continuous_section(self) -> tuple[Element, float, float]:
        """
        Build a finite-height continuous color gradient section for stacking with
        other guides, as an ``(element, width, height)`` tuple.
        """
        assert isinstance(self._color_scale, ScaleContinuousColor)

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
        tick_labels, tick_fracs, stops = _continuous_key_layout(self._color_scale)
        n = len(tick_labels)

        max_text_width_val = 0.0
        max_text_height_val = 0.0
        for label in tick_labels:
            tw, th = font.get_extents(str(label))
            max_text_width_val = max(max_text_width_val, tw.scalar_value())
            max_text_height_val = max(max_text_height_val, th.scalar_value())

        spacing_val = spacing.scalar_value()
        gradient_width_val = gradient_width.scalar_value()
        tick_length_val = tick_length.scalar_value()

        # Tall enough that the closest pair of tick labels doesn't collide,
        # within reason when ticks are very unevenly spaced.
        gap = max(2.0 * max_text_height_val, spacing_val)
        min_frac_gap = float(np.min(np.diff(tick_fracs))) if n > 1 else 0.0
        max_height_val = 3.0 * max((n - 1) * gap, gap)
        gradient_height_val = (
            min(gap / min_frac_gap, max_height_val) if min_frac_gap > 0 else gap
        )
        top_pad = 0.5 * max_text_height_val

        g = Element(
            "g",
            {
                "font-family": font_family,
                "font-size": font_size,
                "font-weight": font_weight,
                "fill": fill,
            },
        )

        gradient_id = f"key-gradient-{id(self)}"
        defs = Element("defs")
        defs.append(_gradient_def(gradient_id, stops))
        g.append(defs)

        # Gradient rectangle (min at the bottom, max at the top).
        g.append(
            Element(
                "rect",
                {
                    "x": mm(0),
                    "y": mm(top_pad),
                    "width": gradient_width,
                    "height": mm(gradient_height_val),
                    "fill": f"url(#{gradient_id})",
                },
            )
        )

        # Tick y positions: tick 0 (minimum) sits at the bottom.
        def tick_y(i: int) -> float:
            return top_pad + gradient_height_val * (1.0 - float(tick_fracs[i]))

        for i, label in enumerate(tick_labels):
            y = tick_y(i)
            g.append(
                Element(
                    "line",
                    {
                        "x1": gradient_width,
                        "x2": gradient_width + tick_length,
                        "y1": mm(y),
                        "y2": mm(y),
                        "stroke": fill,
                        "stroke-width": stroke_width,
                    },
                )
            )
            text_element = Element(
                "text",
                {
                    "x": gradient_width + tick_length + spacing,
                    "y": mm(y),
                    "dominant-baseline": "middle",
                },
            )
            text_element.text = str(label)
            g.append(text_element)

        width_val = (
            gradient_width_val + tick_length_val + spacing_val + max_text_width_val
        )
        height_val = gradient_height_val + max_text_height_val
        return g, width_val, height_val

    def _resolve_continuous(self, ctx: ResolveContext) -> Element:
        """Resolve continuous color scale into gradient with tick marks."""
        assert isinstance(self._color_scale, ScaleContinuousColor)

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

        tick_labels, tick_fracs, stops = _continuous_key_layout(self._color_scale)

        # Create gradient definition
        gradient_id = f"key-gradient-{id(self)}"
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

        defs = Element("defs")
        defs.append(_gradient_def(gradient_id, stops))
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

        tick_y_positions = CtxLengths(tick_fracs, "vh", CtxLenType.Pos)
        ticks_at_ends = (
            len(tick_fracs) >= 2
            and np.isclose(tick_fracs[0], 0.0)
            and np.isclose(tick_fracs[-1], 1.0)
        )

        # Draw axis line + end ticks as a single path when ticks sit at both ends
        if ticks_at_ends:
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
        # Labels near either end of the gradient overhang it by up to half
        # their height.
        end_frac = 0.1
        if len(tick_fracs) > 1:
            if tick_fracs[0] < end_frac:
                _text_width, text_height = font.get_extents(str(tick_labels[0]))
                bottom_padding = 0.5 * text_height + nudge
            if tick_fracs[-1] > 1.0 - end_frac:
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

        if "shape" in scales and isinstance(scales["shape"], ScaleDiscreteShape):
            self._shape_scale = scales["shape"]
            self._shape_labels, self._shapes = self._shape_scale.ticks()

    @override
    def abs_bounds(self) -> tuple[AbsLengths, AbsLengths]:
        if self._color_scale is None:
            if self._shape_scale is not None:
                return self._shape_abs_bounds()
            return mm(0), mm(0)

        # Combined color + shape key: measure the stacked sections.
        if self._shape_scale is not None:
            sections = self._combined_sections()
            if not sections:
                return mm(0), mm(0)
            gap = self._section_gap()
            width_val = max(w for _el, w, _h in sections)
            height_val = sum(h for _el, _w, h in sections) + gap * (len(sections) - 1)
            return mm(width_val), mm(height_val)

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

            labels = self._filtered_labels()
            for i, label in enumerate(labels):
                text_width, text_height = font.get_extents(str(label))
                text_width_val = text_width.scalar_value()
                if text_width_val > max_text_width_val:
                    max_text_width_val = text_width_val

                # Add square size plus spacing for each item
                total_height_val += square_size_val
                if i < len(labels) - 1:  # No spacing after last item
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


    def _shape_abs_bounds(self) -> tuple[AbsLengths, AbsLengths]:
        assert self._shape_labels is not None
        _el, width_val, height_val = self._marker_section(
            list(self._shape_labels), None, self._shapes
        )
        return mm(width_val), mm(height_val)


def key(*args, **kwargs):
    """Create a Key geometry."""
    return Key(*args, **kwargs)
