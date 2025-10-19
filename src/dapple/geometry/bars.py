import numpy as np
from typing import Optional, Any, override

from ..elements import Element, VectorizedElement
from ..scales import length_params, color_params
from ..coordinates import (
    CtxLenType,
    AbsLengths,
    Lengths,
    ResolveContext,
    CoordBounds,
    resolve,
    mm,
    cxv,
    cyv,
)
from ..config import ConfigKey


class Bar(Element):
    """
    Wrapper element for SVG rect that handles negative widths/heights.

    SVG rect elements require non-negative width and height values. This element
    resolves to a VectorizedElement with adjusted positions and positive dimensions.
    """

    def __init__(self, x, y, width, height, **kwargs):
        super().__init__(tag="dapple:bar")
        self.attrib = {"x": x, "y": y, "width": width, "height": height, **kwargs}

    @override
    def update_bounds(self, bounds: CoordBounds):
        """
        Update bounds to include rectangle extents.

        If a 'dapple:nudge' absolute length is present, expand bounds by that
        amount on all sides so neighboring rects can slightly overlap (avoids
        hairline gaps).
        """
        x = self.get_as("x", Lengths)
        y = self.get_as("y", Lengths)
        width = self.get_as("width", Lengths)
        height = self.get_as("height", Lengths)

        nudge = self.attrib.get("dapple:nudge")
        if nudge is not None:
            bounds.update(x - nudge)
            bounds.update(y - nudge)
            bounds.update(x + width + nudge)
            bounds.update(y + height + nudge)
        else:
            bounds.update(x)
            bounds.update(y)
            bounds.update(x + width)
            bounds.update(y + height)

    def resolve(self, ctx: ResolveContext) -> Element:
        """Resolve to VectorizedElement rect with corrected position and positive dimensions."""
        resolved_attrib = resolve(self.attrib, ctx)

        x = resolved_attrib.pop("x")
        y = resolved_attrib.pop("y")
        width = resolved_attrib.pop("width")
        height = resolved_attrib.pop("height")
        nudge = resolved_attrib.pop("dapple:nudge", None)

        # Handle negative widths by adjusting x position
        if isinstance(width, AbsLengths):
            # Vectorized case
            x_vals = (
                x.values
                if isinstance(x, AbsLengths)
                else np.full(len(width.values), x.scalar_value())
            )
            w_vals = width.values

            # Where width is negative, adjust x and flip width sign
            x_adjusted = np.where(w_vals < 0, x_vals + w_vals, x_vals)
            w_adjusted = np.abs(w_vals)

            x = AbsLengths(x_adjusted)
            width = AbsLengths(w_adjusted)
        elif hasattr(width, "scalar_value"):
            # Scalar case
            w_val = width.scalar_value()
            if w_val < 0:
                x = mm(x.scalar_value() + w_val)
                width = mm(-w_val)

        # Handle negative heights by adjusting y position
        if isinstance(height, AbsLengths):
            # Vectorized case
            y_vals = (
                y.values
                if isinstance(y, AbsLengths)
                else np.full(len(height.values), y.scalar_value())
            )
            h_vals = height.values

            # Where height is negative, adjust y and flip height sign
            y_adjusted = np.where(h_vals < 0, y_vals + h_vals, y_vals)
            h_adjusted = np.abs(h_vals)

            y = AbsLengths(y_adjusted)
            height = AbsLengths(h_adjusted)
        elif hasattr(height, "scalar_value"):
            # Scalar case
            h_val = height.scalar_value()
            if h_val < 0:
                y = mm(y.scalar_value() + h_val)
                height = mm(-h_val)

        # Apply absolute nudge (if provided) to slightly enlarge rectangles
        if nudge is not None:
            if isinstance(nudge, AbsLengths):
                nudge.assert_scalar()
                nv = nudge.scalar_value()
            else:
                nv = float(nudge)

            # Expand rectangle by nv on all sides
            if isinstance(x, AbsLengths):
                x = AbsLengths(x.values - nv)
            else:
                x = mm(x.scalar_value() - nv)

            if isinstance(y, AbsLengths):
                y = AbsLengths(y.values - nv)
            else:
                y = mm(y.scalar_value() - nv)

            if isinstance(width, AbsLengths):
                width = AbsLengths(width.values + 2 * nv)
            else:
                width = mm(width.scalar_value() + 2 * nv)

            if isinstance(height, AbsLengths):
                height = AbsLengths(height.values + 2 * nv)
            else:
                height = mm(height.scalar_value() + 2 * nv)

        # Return VectorizedElement with corrected values
        return VectorizedElement(
            "rect",
            {"x": x, "y": y, "width": width, "height": height, **resolved_attrib},
        )


def _prepare_bar_data(primary, primary_min, primary_max, default_width, axis_name: str):
    """
    Helper function to prepare position and width data for bars.

    Args:
        primary: Center position (e.g., x for vertical bars, y for horizontal bars)
        primary_min: Minimum position (e.g., xmin for vertical bars)
        primary_max: Maximum position (e.g., xmax for vertical bars)
        default_width: Default width/height to use when primary is specified (in data units)
        axis_name: Name of the axis ('x' or 'y') for error messages

    Returns:
        Tuple of (position, width) where position is the left/bottom edge
    """
    # Check that either primary or (primary_min and primary_max) are specified
    has_primary = primary is not None
    has_range = primary_min is not None and primary_max is not None

    if has_primary and has_range:
        raise ValueError(
            f"Specify either {axis_name} or ({axis_name}min and {axis_name}max), not both"
        )
    elif not has_primary and not has_range:
        raise ValueError(
            f"Must specify either {axis_name} or both {axis_name}min and {axis_name}max"
        )

    if has_primary:
        # Center the bar on the primary position with default width
        # Convert to array for arithmetic
        primary = length_params(axis_name, primary, CtxLenType.Pos)

        # position = primary - default_width/2
        # width = default_width (as a constant)
        position = primary - 0.5 * default_width
        return position, default_width
    else:
        # Use the explicit range
        primary_min = length_params(axis_name, primary_min, CtxLenType.Pos)
        primary_max = length_params(axis_name, primary_max, CtxLenType.Pos)

        position = primary_min
        width = primary_max - primary_min
        return position, width


def vertical_bars(
    x: Optional[Any] = None,
    y: Any = None,
    xmin: Optional[Any] = None,
    xmax: Optional[Any] = None,
    color: Optional[Any] = None,
):
    """
    Create vertical bars extending from the x-axis (y=0).

    Args:
        x: X positions for bar centers (use with default width)
        y: Y values (heights) for bars, extending from 0
        xmin: Left edges of bars (use with xmax)
        xmax: Right edges of bars (use with xmin)
        color: Fill color for bars (optional)

    Returns:
        Element containing the bar geometry

    Examples:
        # Bars centered at x positions with default width
        vertical_bars(x=[1, 2, 3], y=[10, 20, 15])

        # Bars with explicit x ranges
        vertical_bars(y=[10, 20, 15], xmin=[0.5, 1.5, 2.5], xmax=[1.5, 2.5, 3.5])
    """
    if y is None:
        raise ValueError("y parameter is required")

    # Prepare x position and width (default width is 1.0 in data coordinates)
    x_pos, bar_width = _prepare_bar_data(x, xmin, xmax, cxv(1), "x")

    # Convert y to array
    y_array = np.asarray(y)

    # Bars extend from 0 to y
    # rect.y should be min(0, y) and height should be |y|
    # We'll let the Bar element handle negative heights
    bar_y = np.zeros_like(y_array)
    bar_height = y_array

    # Build attributes dict
    attrib = {
        "x": x_pos,
        "y": length_params("y", bar_y, CtxLenType.Pos),
        "width": bar_width,
        "height": length_params("y", bar_height, CtxLenType.Vec),
    }

    if color is not None:
        attrib["fill"] = color_params("color", color)
    else:
        attrib["fill"] = ConfigKey("barcolor")

    return Bar(**attrib)


def horizontal_bars(
    y: Optional[Any] = None,
    x: Any = None,
    ymin: Optional[Any] = None,
    ymax: Optional[Any] = None,
    color: Optional[Any] = None,
):
    """
    Create horizontal bars extending from the y-axis (x=0).

    Args:
        y: Y positions for bar centers (use with default height)
        x: X values (widths) for bars, extending from 0
        ymin: Bottom edges of bars (use with ymax)
        ymax: Top edges of bars (use with ymin)
        color: Fill color for bars (optional)

    Returns:
        Element containing the bar geometry

    Examples:
        # Bars centered at y positions with default height
        horizontal_bars(y=[1, 2, 3], x=[10, 20, 15])

        # Bars with explicit y ranges
        horizontal_bars(x=[10, 20, 15], ymin=[0.5, 1.5, 2.5], ymax=[1.5, 2.5, 3.5])
    """
    if x is None:
        raise ValueError("x parameter is required")

    # Prepare y position and height (default height is 1.0 in data coordinates)
    y_pos, bar_height = _prepare_bar_data(y, ymin, ymax, 1.0, "y")

    # Convert x to array
    x_array = np.asarray(x)

    # Bars extend from 0 to x
    # rect.x should be min(0, x) and width should be |x|
    # We'll let the Bar element handle negative widths
    bar_x = np.zeros_like(x_array)
    bar_width = x_array

    # Build attributes dict
    attrib = {
        "x": length_params("x", bar_x, CtxLenType.Pos),
        "y": y_pos,
        "width": length_params("x", bar_width, CtxLenType.Vec),
        "height": bar_height,
    }

    if color is not None:
        attrib["fill"] = color_params("color", color)
    else:
        attrib["fill"] = ConfigKey("barcolor")

    return Bar(**attrib)
