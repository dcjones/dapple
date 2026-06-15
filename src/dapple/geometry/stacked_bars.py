from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional, override

import numpy as np

from ..colors import Colors
from ..coordinates import CtxLenType, Lengths, cxv, cyv
from ..elements import Element
from ..scales import ScaleSet, UnscaledExpr, UnscaledValues, color_params, length_params
from .bars import Bar


def _unique_in_order(values: list[Any]) -> list[Any]:
    seen: OrderedDict[Any, None] = OrderedDict()
    for v in values:
        if v not in seen:
            seen[v] = None
    return list(seen.keys())


def _build_stacked_pivot(
    pos_values: list[Any],
    data_values: list[float],
    color_values: list[Any],
    unique_pos: list[Any],
    unique_colors: list[Any],
) -> dict[Any, dict[Any, float]]:
    """
    Aggregate data into a pivot table keyed by (position, color).

    Multiple observations sharing the same (position, color) pair are summed.
    """
    pivot: dict[Any, dict[Any, float]] = {
        p: {c: 0.0 for c in unique_colors} for p in unique_pos
    }
    for p, d, c in zip(pos_values, data_values, color_values):
        pivot[p][c] += d
    return pivot


@dataclass
class _StackedPositionExpr(UnscaledExpr):
    """
    Lazily computes the starting offset of one layer of a stacked bar.

    The stacking order follows the order of ``color_unit``'s discrete scale,
    which is only known once that scale has been fit and finalized. This
    defers the offset computation to ``accept_scale``, which runs after
    finalization.
    """

    unit: str
    typ: CtxLenType
    color_unit: str
    color_val: Any
    pivot: dict[Any, dict[Any, float]]
    unique_pos: list[Any]
    unique_colors: list[Any]

    def __len__(self) -> int:
        return len(self.unique_pos)

    @override
    def accept_fit(self, scaleset: ScaleSet) -> None:
        # Fit using each stack's full range, independent of layer order, so
        # the scale's bounds cover the top of the stack regardless of which
        # color ends up on top.
        totals = [sum(self.pivot[p].values()) for p in self.unique_pos]
        scaleset[self.unit].fit_values(
            UnscaledValues(self.unit, [0.0, *totals], self.typ)
        )

    @override
    def accept_scale(self, scaleset: ScaleSet) -> Lengths | Colors:
        color_map = getattr(scaleset.get(self.color_unit), "map", None)
        if color_map is not None:
            ordered_colors = sorted(self.unique_colors, key=lambda c: color_map[c])
        else:
            ordered_colors = sorted(self.unique_colors)

        below = ordered_colors[: ordered_colors.index(self.color_val)]
        starts = [sum(self.pivot[p][c] for c in below) for p in self.unique_pos]

        return scaleset[self.unit].scale_values(
            UnscaledValues(self.unit, starts, self.typ)
        )

    @override
    def accept_visitor(self, visitor: Callable[[UnscaledValues], Any]) -> None:
        _ = visitor


def stacked_vertical_bars(
    x: Any,
    y: Any,
    color: Any,
    *,
    width: float = 0.8,
    normalize: Optional[float] = None,
) -> Element:
    """
    Create stacked vertical bars grouped by x position and segmented by color.

    Each unique value in ``color`` becomes one horizontal layer of the stack.
    Layers are stacked bottom-to-top in the order assigned by the plot's
    discrete color scale (which defaults to sorting values, but can be
    customized via ``order_by`` or an explicit ``values`` mapping). Multiple
    observations with the same ``(x, color)`` pair are summed before stacking.

    Args:
        x: Grouping position per observation (the x-axis category for each stack).
        y: Value per observation.
        color: Color group per observation (determines the stack segments).
        width: Bar width expressed in context-x units (default 0.8).
        normalize: When given, each stack's total is rescaled so that it equals
            this value.  Use ``normalize=100`` for percent-stacked bars or
            ``normalize=1`` for proportion-stacked bars.

    Returns:
        A container ``Element`` holding one ``Bar`` per unique color group.

    Example::

        stacked_vertical_bars(
            x=["A", "A", "B", "B"],
            y=[10, 20, 15, 25],
            color=["cats", "dogs", "cats", "dogs"],
        )

        # Percent-stacked
        stacked_vertical_bars(
            x=["A", "A", "B", "B"],
            y=[10, 20, 15, 25],
            color=["cats", "dogs", "cats", "dogs"],
            normalize=100,
        )
    """
    x_list: list[Any] = list(x)
    y_arr = np.asarray(list(y), dtype=float)
    color_list: list[Any] = list(color)

    n = len(x_list)
    if len(y_arr) != n or len(color_list) != n:
        raise ValueError("x, y, and color must all have the same length")

    unique_x = _unique_in_order(x_list)
    unique_colors = _unique_in_order(color_list)

    pivot = _build_stacked_pivot(
        x_list, y_arr.tolist(), color_list, unique_x, unique_colors
    )

    if normalize is not None:
        for xi in unique_x:
            total = sum(pivot[xi].values())
            if total != 0.0:
                scale = normalize / total
                for ci in unique_colors:
                    pivot[xi][ci] *= scale

    bar_width = cxv(width)
    container = Element("g")

    for color_val in unique_colors:
        y_vals = [pivot[xi][color_val] for xi in unique_x]

        # Compute the left edge of bars: center x minus half-width
        x_pos = length_params("x", unique_x, CtxLenType.Pos) - 0.5 * bar_width

        bar = Bar(
            x=x_pos,
            y=_StackedPositionExpr(
                "y", CtxLenType.Pos, "color", color_val, pivot, unique_x, unique_colors
            ),
            width=bar_width,
            height=length_params("y", y_vals, CtxLenType.Vec),
            fill=color_params("color", color_val),
        )
        container.append(bar)

    return container


def stacked_horizontal_bars(
    y: Any,
    x: Any,
    color: Any,
    *,
    width: float = 0.8,
    normalize: Optional[float] = None,
) -> Element:
    """
    Create stacked horizontal bars grouped by y position and segmented by color.

    Each unique value in ``color`` becomes one vertical layer of the stack.
    Layers are stacked left-to-right in the order assigned by the plot's
    discrete color scale (which defaults to sorting values, but can be
    customized via ``order_by`` or an explicit ``values`` mapping). Multiple
    observations with the same ``(y, color)`` pair are summed before stacking.

    Args:
        y: Grouping position per observation (the y-axis category for each stack).
        x: Value per observation.
        color: Color group per observation (determines the stack segments).
        width: Bar height expressed in context-y units (default 0.8).
        normalize: When given, each stack's total is rescaled so that it equals
            this value.  Use ``normalize=100`` for percent-stacked bars or
            ``normalize=1`` for proportion-stacked bars.

    Returns:
        A container ``Element`` holding one ``Bar`` per unique color group.

    Example::

        stacked_horizontal_bars(
            y=["A", "A", "B", "B"],
            x=[10, 20, 15, 25],
            color=["cats", "dogs", "cats", "dogs"],
        )

        # Proportion-stacked
        stacked_horizontal_bars(
            y=["A", "A", "B", "B"],
            x=[10, 20, 15, 25],
            color=["cats", "dogs", "cats", "dogs"],
            normalize=1,
        )
    """
    y_list: list[Any] = list(y)
    x_arr = np.asarray(list(x), dtype=float)
    color_list: list[Any] = list(color)

    n = len(y_list)
    if len(x_arr) != n or len(color_list) != n:
        raise ValueError("y, x, and color must all have the same length")

    unique_y = _unique_in_order(y_list)
    unique_colors = _unique_in_order(color_list)

    pivot = _build_stacked_pivot(
        y_list, x_arr.tolist(), color_list, unique_y, unique_colors
    )

    if normalize is not None:
        for yi in unique_y:
            total = sum(pivot[yi].values())
            if total != 0.0:
                scale = normalize / total
                for ci in unique_colors:
                    pivot[yi][ci] *= scale

    bar_height = cyv(width)
    container = Element("g")

    for color_val in unique_colors:
        x_vals = [pivot[yi][color_val] for yi in unique_y]

        # Compute the bottom edge of bars: center y minus half-height
        y_pos = length_params("y", unique_y, CtxLenType.Pos) - 0.5 * bar_height

        bar = Bar(
            x=_StackedPositionExpr(
                "x", CtxLenType.Pos, "color", color_val, pivot, unique_y, unique_colors
            ),
            y=y_pos,
            width=length_params("x", x_vals, CtxLenType.Vec),
            height=bar_height,
            fill=color_params("color", color_val),
        )
        container.append(bar)

    return container
