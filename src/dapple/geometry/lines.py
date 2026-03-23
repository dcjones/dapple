import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from numbers import Number
from typing import Any, override

import numpy as np

from ..config import ConfigKey
from ..coordinates import (
    AbsLengths,
    CtxLenType,
    Lengths,
    ResolveContext,
    Serializable,
    cy,
    cyv,
    resolve,
)
from ..elements import Element, Path, PathData, VectorizedElement
from ..scales import UnscaledValues, color_params, length_params


def _adaptive_sample_function(
    func: Callable,
    xmin: float,
    xmax: float,
    max_points: int = 2000,
    tolerance: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adaptively sample a function using the adaptive library.

    This provides robust adaptive sampling that handles cases where simple
    collinearity tests fail (e.g., np.sin from -10 to 10 where midpoint is collinear).
    """
    try:
        import adaptive
    except ImportError:
        raise ImportError(
            "adaptive library is required for function plotting. Install with: pip install adaptive"
        )

    # Create learner for 1D function
    learner = adaptive.Learner1D(func, bounds=(xmin, xmax))

    # Run adaptive sampling with a reasonable goal
    # Use either the tolerance or a reasonable number of points, whichever comes first
    goal_func = lambda l: l.loss() < tolerance or l.npoints > max_points

    try:
        adaptive.runner.simple(learner, goal=goal_func)
    except Exception:
        # If function evaluation fails, return empty arrays
        return np.array([]), np.array([])

    # Extract points and sort by x-coordinate
    points = learner.data
    if not points:
        return np.array([]), np.array([])

    # Sort points by x-coordinate
    sorted_points = sorted(points.items())
    x_vals = np.array([p[0] for p in sorted_points])
    y_vals = np.array([p[1] for p in sorted_points])

    return x_vals, y_vals


def _sample_function(func: Callable, xmin: float, xmax: float) -> tuple[Any, Any]:
    """Sample a function and return coordinate data suitable for Path element."""
    x_vals, y_vals = _adaptive_sample_function(func, xmin, xmax)

    if len(x_vals) < 2:
        # Return empty arrays if sampling failed
        return [], []

    return x_vals.tolist(), y_vals.tolist()


def line(
    x=None,
    y=None,
    color=ConfigKey("linecolor"),
    width=ConfigKey("linestroke"),
    xmin=None,
    xmax=None,
):
    """
    Plot a single line.

    Args:
        x: X coordinates (iterable) or None for function plotting
        y: Y coordinates (iterable) or function for function plotting
        color: Line color (scalar or ConfigKey, defaults to "linecolor")
        xmin: Minimum x value for function plotting (optional)
        xmax: Maximum x value for function plotting (optional)
    """
    # Function plotting mode
    if x is None or xmin is not None:
        if xmin is None or xmax is None:
            raise ValueError("Function plotting requires both xmin and xmax")

        # Sample the function
        x_data, y_data = _sample_function(y, xmin, xmax)

        if len(x_data) < 2:
            # Return empty group if sampling failed
            return Element("g")
    else:
        # Regular coordinate plotting
        x_data, y_data = x, y

    return Path(
        length_params("x", x_data, CtxLenType.Pos),
        length_params("y", y_data, CtxLenType.Pos),
        stroke=color_params("color", color),
        fill="none",
        **{"stroke-width": width},
    )


def lines(
    x=None,
    y=None,
    color=ConfigKey("linecolor"),
    width=ConfigKey("linestroke"),
    group=None,
    xmin=None,
    xmax=None,
):
    """
    Plot lines with support for multiple patterns.

    Args:
        x: X coordinates/data or None for function plotting
        y: Y coordinates/data or function for function plotting
        color: Line color(s) - can be single value or array
        group: Grouping variable for multiple lines (optional)
        xmin: Minimum x value for function plotting (optional)
        xmax: Maximum x value for function plotting (optional)

    Notes:
        - Lines are grouped by unique (group, color) pairs when both are provided
        - When only group is provided, lines are grouped by group values
        - When only color is provided, lines are grouped by color values
        - When neither is provided, creates a single line
    """

    assert isinstance(width, (ConfigKey, Lengths))
    if isinstance(width, Lengths):
        assert width.isscalar()

    # Function plotting
    if x is None or xmin is not None:
        if xmin is None or xmax is None:
            raise ValueError("Function plotting requires both xmin and xmax")

        # Sample the function and create single path
        x_data, y_data = _sample_function(y, xmin, xmax)

        if len(x_data) < 2:
            # Return empty group if sampling failed
            return Element("g")

        return Path(
            length_params("x", x_data, CtxLenType.Pos),
            length_params("y", y_data, CtxLenType.Pos),
            stroke=color_params("color", color),
            fill="none",
            **{"stroke-width": width},
        )

    # Multiple lines with grouping (by group, color, or both)
    elif group is not None or _has_multiple_values(color):
        return _create_grouped_paths(x, y, color, group, width)

    # Single line (same as line() function)
    else:
        return line(x, y, color, width)


class SegmentsElement(Element):
    """
    Element representing one or more line segments, optionally with arrowheads.
    """

    def __init__(
        self,
        x1,
        y1,
        x2,
        y2,
        color=ConfigKey("linecolor"),
        stroke_width=ConfigKey("linestroke"),
        arrow=False,
    ):
        super().__init__("dapple:segments")
        self.attrib = {
            "x1": length_params("x", x1, CtxLenType.Pos),
            "y1": length_params("y", y1, CtxLenType.Pos),
            "x2": length_params("x", x2, CtxLenType.Pos),
            "y2": length_params("y", y2, CtxLenType.Pos),
            "stroke": color_params("color", color),
            "stroke-width": stroke_width,
            "arrow": arrow,
        }

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        res = resolve(self.attrib, ctx)
        x1 = res["x1"]
        y1 = res["y1"]
        x2 = res["x2"]
        y2 = res["y2"]
        stroke = res["stroke"]
        stroke_width = res["stroke-width"]
        arrow = res["arrow"]

        assert isinstance(x1, AbsLengths)
        assert isinstance(y1, AbsLengths)
        assert isinstance(x2, AbsLengths)
        assert isinstance(y2, AbsLengths)

        main_line = VectorizedElement(
            "line",
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "stroke": stroke,
                "stroke-width": stroke_width,
            },
        )

        if arrow is False:
            return main_line

        # Calculate arrowheads in absolute coordinates (mm)
        # Handle broadcasting to common length
        nels = max(len(x1), len(y1), len(x2), len(y2))

        def _get_values(l: AbsLengths) -> np.ndarray:
            if len(l) == nels:
                return l.values
            return np.repeat(l.values, nels)

        vx1 = _get_values(x1)
        vy1 = _get_values(y1)
        vx2 = _get_values(x2)
        vy2 = _get_values(y2)

        dx = vx2 - vx1
        dy = vy2 - vy1
        length = np.sqrt(dx**2 + dy**2)

        # Avoid division by zero
        mask = length > 1e-6
        ux = np.zeros_like(dx)
        uy = np.zeros_like(dy)
        ux[mask] = dx[mask] / length[mask]
        uy[mask] = dy[mask] / length[mask]

        # Arrowhead parameters
        arrow_len = 2.0  # 2mm
        angle = np.pi / 6  # 30 degrees
        ca = np.cos(angle)
        sa = np.sin(angle)

        # Arrowhead leg 1
        lx1 = vx2 - arrow_len * (ux * ca - uy * sa)
        ly1 = vy2 - arrow_len * (ux * sa + uy * ca)

        # Arrowhead leg 2
        lx2 = vx2 - arrow_len * (ux * ca + uy * sa)
        ly2 = vy2 - arrow_len * (-ux * sa + uy * ca)

        # If arrow is a vector, mask the legs
        if isinstance(arrow, (np.ndarray, list)):
            arrow_mask = np.asarray(arrow)
            if arrow_mask.shape != (nels,):
                arrow_mask = np.broadcast_to(arrow_mask, (nels,))
            
            final_mask = mask & arrow_mask
            lx1[~final_mask] = vx2[~final_mask]
            ly1[~final_mask] = vy2[~final_mask]
            lx2[~final_mask] = vx2[~final_mask]
            ly2[~final_mask] = vy2[~final_mask]
        else:
            # Scalar arrow=True
            lx1[~mask] = vx2[~mask]
            ly1[~mask] = vy2[~mask]
            lx2[~mask] = vx2[~mask]
            ly2[~mask] = vy2[~mask]

        leg1 = VectorizedElement(
            "line",
            {
                "x1": x2,
                "y1": y2,
                "x2": AbsLengths(lx1),
                "y2": AbsLengths(ly1),
                "stroke": stroke,
                "stroke-width": stroke_width,
            },
        )
        leg2 = VectorizedElement(
            "line",
            {
                "x1": x2,
                "y1": y2,
                "x2": AbsLengths(lx2),
                "y2": AbsLengths(ly2),
                "stroke": stroke,
                "stroke-width": stroke_width,
            },
        )

        return Element("g", {}, main_line, leg1, leg2)


def segments(x1, y1, x2, y2, color=ConfigKey("linecolor"), arrow=False) -> Element:
    return SegmentsElement(x1, y1, x2, y2, color, ConfigKey("linestroke"), arrow)


def density(
    x,
    y=None,
    color=ConfigKey("linecolor"),
    bw_method=None,
    weights=None,
    xmin=None,
    xmax=None,
    width=ConfigKey("linestroke"),
    normalize=False,
    clip=1e-3,
    markers=None,
):
    """
    Plot a kernel density estimate.

    This computes a kernel density estimate from the given `x` data using
    scipy's `gaussian_kde` and plots it as a line.

    Args:
        x: Input data (1D array-like).
        y: Optional y-offset. If provided, the density plot is shifted by this amount
           along the y-axis. This is useful for creating ridgeline plots.
        color: Line color.
        bw_method: Bandwidth estimation method for `gaussian_kde`. Can be
                   'scott', 'silverman', a scalar, or a callable.
        weights: Weights for each data point in `x`.
        xmin: Minimum x value for plotting. Defaults to min of `x`.
        xmax: Maximum x value for plotting. Defaults to max of `x`.
        normalize: If True, normalize the density so the maximum value is 1.
        clip: Threshold density value for automatic range determination.
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        raise ImportError(
            "scipy is required for density plotting. Install with: pip install scipy"
        )

    x_vals = np.asarray(x)
    if x_vals.ndim != 1:
        raise ValueError("x must be a 1D array-like object.")

    if x_vals.size < 2:
        return Element("g")

    kde = gaussian_kde(x_vals, bw_method=bw_method, weights=weights)

    if xmin is None or xmax is None:
        dmin = float(np.min(x_vals))
        dmax = float(np.max(x_vals))

        if dmax <= dmin:
            pad = 1.0 if dmin == 0.0 else abs(dmin) * 1e-6
            dmin -= pad
            dmax += pad

        std = float(np.std(x_vals))
        if not math.isfinite(std) or std == 0.0:
            std = max(1e-3, (dmax - dmin) if dmax > dmin else 1.0)

        if xmin is None:
            cur = dmin
            left_limit = dmin - 10.0 * std
            while kde(cur)[0] > clip and cur > left_limit:
                cur -= std
            xmin = float(cur)

        if xmax is None:
            cur = dmax
            right_limit = dmax + 10.0 * std
            while kde(cur)[0] > clip and cur < right_limit:
                cur += std
            xmax = float(cur)

    x_arr, y_arr = _adaptive_sample_function(lambda v: kde(v)[0], xmin, xmax)
    if len(x_arr) < 2:
        return Element("g")

    normalization_factor = 1.0
    if normalize:
        normalization_factor = np.max(y_arr)
        if normalization_factor > 0:
            y_arr /= normalization_factor
        else:
            normalization_factor = 1.0

    if y is not None:
        if np.ndim(y) == 0:
            y_baseline = [y] * len(x_arr)
        else:
            raise ValueError("y must be a scalar when plotting density")

        y_vals = length_params("y", y_baseline, CtxLenType.Pos) + cyv(y_arr)
        main_line = line(x=x_arr, y=y_vals, color=color, width=width)
    else:
        main_line = line(x=x_arr, y=y_arr, color=color, width=width)

    if markers is None:
        return main_line

    marker_vals = np.atleast_1d(markers)
    marker_heights = kde(marker_vals)

    if normalize:
        marker_heights /= normalization_factor

    if y is not None:
        # Use y as baseline
        y_pos_base = length_params("y", [y] * len(marker_vals), CtxLenType.Pos)
        y_pos_top = y_pos_base + cyv(marker_heights)
        marker_lines = segments(
            marker_vals, y_pos_base, marker_vals, y_pos_top, color=color
        )
    else:
        marker_lines = segments(
            marker_vals,
            np.zeros_like(marker_vals),
            marker_vals,
            marker_heights,
            color=color,
        )

    return Element("g", {}, main_line, marker_lines)


def _has_multiple_values(color) -> bool:
    """Check if color parameter represents multiple values for grouping."""
    if isinstance(color, ConfigKey):
        return False  # Single config key
    if hasattr(color, "__len__") and not isinstance(color, str):
        try:
            return len(color) > 1
        except:
            return False
    return False


def _create_grouped_paths(x, y, color, group, width):
    """
    Create multiple Path elements by grouping input data before scaling.

    This handles the grouping logic upfront, avoiding complex resolution-time
    string conversions and working with raw input values.
    """

    # Convert inputs to lists for easier handling
    x_vals = list(x) if hasattr(x, "__iter__") else [x]
    y_vals = list(y) if hasattr(y, "__iter__") else [y]

    # Handle group data
    if group is not None:
        if hasattr(group, "__iter__") and not isinstance(group, str):
            group_vals = list(group)
        else:
            group_vals = [group] * len(x_vals)
    else:
        group_vals = ["default"] * len(x_vals)

    # Handle color data
    if isinstance(color, ConfigKey):
        color_vals = [color] * len(x_vals)
    elif hasattr(color, "__iter__") and not isinstance(color, str):
        color_vals = list(color)
    else:
        color_vals = [color] * len(x_vals)

    # Ensure all arrays are the same length
    min_len = min(len(x_vals), len(y_vals), len(group_vals), len(color_vals))
    x_vals = x_vals[:min_len]
    y_vals = y_vals[:min_len]
    group_vals = group_vals[:min_len]
    color_vals = color_vals[:min_len]

    # Group points by (group, color) pairs using original values
    grouped_data = defaultdict(lambda: {"x": [], "y": [], "color": None})

    for i in range(min_len):
        # Create group key using original values, making them hashable
        if isinstance(color_vals[i], ConfigKey):
            color_key = f"ConfigKey({color_vals[i].key})"
        elif isinstance(color_vals[i], str):
            color_key = color_vals[i]
        else:
            color_key = id(color_vals[i])  # Use id for objects

        group_key = (group_vals[i], color_key)

        grouped_data[group_key]["x"].append(x_vals[i])
        grouped_data[group_key]["y"].append(y_vals[i])
        if grouped_data[group_key]["color"] is None:
            grouped_data[group_key]["color"] = color_vals[i]

    # Create Path elements for each group
    if len(grouped_data) == 0:
        return Element("g")  # Empty container

    if len(grouped_data) == 1:
        # Single group - return Path directly
        group_data = list(grouped_data.values())[0]
        if len(group_data["x"]) < 2:
            return Element("g")

        return Path(
            length_params("x", group_data["x"], CtxLenType.Pos),
            length_params("y", group_data["y"], CtxLenType.Pos),
            stroke=color_params("color", group_data["color"]),
            fill="none",
            **{"stroke-width": width},
        )
    else:
        # Multiple groups - return container with Path children
        container = Element("g")

        for group_data in grouped_data.values():
            if len(group_data["x"]) < 2:
                continue  # Skip groups with insufficient points

            path = Path(
                length_params("x", group_data["x"], CtxLenType.Pos),
                length_params("y", group_data["y"], CtxLenType.Pos),
                stroke=color_params("color", group_data["color"]),
                fill="none",
                **{"stroke-width": width},
            )

            container.children.append(path)

        return container
