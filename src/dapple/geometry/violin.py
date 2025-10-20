from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from typing import Any, Optional, Callable, Tuple, List, Union, override

import numpy as np

from ..elements import Element, Path, PathData
from ..scales import (
    UnscaledExpr,
    UnscaledValues,
    length_params,
    color_params,
)
from ..coordinates import (
    CtxLenType,
    ResolveContext,
    CoordBounds,
    Lengths,
    AbsLengths,
    cxv,
    cyv,
)
from ..config import ConfigKey


# ---- Utilities ----------------------------------------------------------------


def _unique_in_order(values: Iterable[Any]) -> list[Any]:
    seen = OrderedDict()
    for v in values:
        if v not in seen:
            seen[v] = None
    return list(seen.keys())


def _group_indices(keys: list[Any]) -> dict[Any, list[int]]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    return groups


def _safe_kde_1d(data: np.ndarray, grid: np.ndarray, bw_method, weights) -> np.ndarray:
    """
    Compute gaussian_kde density on the given grid.
    """
    from scipy.stats import gaussian_kde  # type: ignore

    kde = gaussian_kde(data, bw_method=bw_method, weights=weights)
    dens = kde(grid)
    return dens


# ---- Concat Expressions --------------------------------------------------------


class VerticalViolinElement(Element):
    """
    Element holding separate left/right x segments and a shared y vector for a vertical violin.
    Concatenation and closing are performed at resolve-time on AbsLengths.
    """

    def __init__(self, x_left: Lengths, x_right: Lengths, y: Lengths, **kwargs: object):
        super().__init__(
            "dapple:violin", {"x_left": x_left, "x_right": x_right, "y": y, **kwargs}
        )

    @override
    def update_bounds(self, bounds: CoordBounds):
        xl = self.get_as("x_left", Lengths)
        xr = self.get_as("x_right", Lengths)
        y = self.get_as("y", Lengths)
        bounds.update(xl.unmin())
        bounds.update(xr.unmax())
        bounds.update(y.unmin())
        bounds.update(y.unmax())

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        # Resolve attributes to absolute values
        resolved = super().resolve(ctx)
        xl = resolved.attrib.pop("x_left")
        xr = resolved.attrib.pop("x_right")
        y = resolved.attrib.pop("y")

        assert isinstance(xl, AbsLengths)
        assert isinstance(xr, AbsLengths)
        assert isinstance(y, AbsLengths)

        x_vals = np.concatenate([xl.values, xr.values[::-1], xl.values[0:1]])
        y_vals = np.concatenate([y.values, y.values[::-1], y.values[0:1]])

        path = Element(
            "path",
            {"d": PathData(AbsLengths(x_vals), AbsLengths(y_vals)), **resolved.attrib},
        )
        return path


class HorizontalViolinElement(Element):
    """
    Element holding separate lower/upper y segments and a shared x vector for a horizontal violin.
    Concatenation and closing are performed at resolve-time on AbsLengths.
    """

    def __init__(self, x: Lengths, y_low: Lengths, y_high: Lengths, **kwargs: object):
        super().__init__(
            "dapple:violin", {"x": x, "y_low": y_low, "y_high": y_high, **kwargs}
        )

    @override
    def update_bounds(self, bounds: CoordBounds):
        x = self.get_as("x", Lengths)
        yl = self.get_as("y_low", Lengths)
        yh = self.get_as("y_high", Lengths)
        bounds.update(x.unmin())
        bounds.update(x.unmax())
        bounds.update(yl.unmin())
        bounds.update(yh.unmax())

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        resolved = super().resolve(ctx)
        x = resolved.attrib.pop("x")
        yl = resolved.attrib.pop("y_low")
        yh = resolved.attrib.pop("y_high")

        assert isinstance(x, AbsLengths)
        assert isinstance(yl, AbsLengths)
        assert isinstance(yh, AbsLengths)

        x_vals = np.concatenate([x.values, x.values[::-1], x.values[0:1]])
        y_vals = np.concatenate([yl.values, yh.values[::-1], yl.values[0:1]])

        path = Element(
            "path",
            {"d": PathData(AbsLengths(x_vals), AbsLengths(y_vals)), **resolved.attrib},
        )
        return path


# ---- Color mapping -------------------------------------------------------------


def _colors_for_groups(
    groups_in_order: list[Any],
    color: Any,
    total_points: int,
    group_indices: dict[Any, list[int]],
) -> dict[Any, Any]:
    """
    Determine per-group fill parameter.

    - If color is None: ConfigKey('barcolor') for all
    - If color length == num groups: map in order
    - If color length == num points: take the first occurrence per group
    - Else: treat as scalar
    """
    if color is None:
        return {g: ConfigKey("barcolor") for g in groups_in_order}

    if isinstance(color, (str, ConfigKey)):
        return {g: color for g in groups_in_order}

    try:
        col_list = list(color)  # type: ignore[arg-type]
    except Exception:
        return {g: color for g in groups_in_order}

    if len(col_list) == len(groups_in_order):
        return {g: col_list[i] for i, g in enumerate(groups_in_order)}

    if len(col_list) == total_points:
        out: dict[Any, Any] = {}
        for g in groups_in_order:
            idxs = group_indices.get(g, [])
            out[g] = col_list[idxs[0]] if len(idxs) > 0 else col_list[0]
        return out

    # Fallback scalar
    return (
        {g: col_list[0] for g in groups_in_order}
        if len(col_list) > 0
        else {g: ConfigKey("barcolor") for g in groups_in_order}
    )


# ---- Public API ----------------------------------------------------------------


def vertical_violin(
    x: Iterable[Any],
    y: Iterable[float],
    *,
    width: float = 0.9,
    color: Optional[Any] = None,
    bw_method: Any = None,
    weights: Optional[Iterable[float]] = None,
    n: int = 200,
) -> Element:
    """
    Draw vertical violins for distributions of y grouped by x.
    One violin per unique x (in order of first appearance).

    Args:
        x: Group values along x-axis (categorical or numeric).
        y: Sample values.
        width: Maximum full width in x data units (vector).
        color: Fill color(s). Scalar, per-group, or per-sample.
        bw_method: Bandwidth for gaussian_kde.
        weights: Optional per-sample weights.
        n: Number of grid points for KDE evaluation.
    """
    x_list = list(x)
    y_list = list(map(float, y))
    if len(x_list) != len(y_list):
        raise ValueError("x and y must have the same length")

    w_list = list(weights) if weights is not None else None
    if w_list is not None and len(w_list) != len(y_list):
        w_list = None

    groups = _unique_in_order(x_list)
    gidx = _group_indices(x_list)
    fills = _colors_for_groups(groups, color, len(x_list), gidx)

    container = Element("g")

    for g in groups:
        idxs = gidx.get(g, [])
        if len(idxs) < 2:
            continue

        data = np.asarray([y_list[i] for i in idxs], dtype=float)
        ws = (
            np.asarray([w_list[i] for i in idxs], dtype=float)
            if w_list is not None
            else None
        )

        # Grid over data extent with guard for zero-span
        dmin = float(np.min(data))
        dmax = float(np.max(data))
        if not (math.isfinite(dmin) and math.isfinite(dmax)):
            continue
        if dmax <= dmin:
            pad = 1.0 if dmin == 0.0 else abs(dmin) * 1e-6
            dmin -= pad
            dmax += pad

        grid = np.linspace(dmin, dmax, int(max(2, n)))
        dens = _safe_kde_1d(data, grid, bw_method, ws)
        dens_max = float(np.max(dens)) if dens.size > 0 else 0.0
        if dens_max <= 0.0 or not math.isfinite(dens_max):
            continue
        half = 0.5 * width * (dens / dens_max)

        # Build UnscaledExpr vectors
        y_vec = length_params("y", grid, CtxLenType.Pos)
        x_center = length_params("x", [g] * len(grid), CtxLenType.Pos)
        hw_x = cxv(half)

        x_left = x_center - hw_x
        x_right = x_center + hw_x

        path = VerticalViolinElement(
            x_left=x_left,
            x_right=x_right,
            y=y_vec,
            fill=color_params("color", fills[g]),
            **{
                "stroke": ConfigKey("linecolor"),
                "stroke-width": ConfigKey("linestroke"),
            },
        )
        container.append(path)

    if len(container) == 1:
        return container[0]
    return container


def horizontal_violin(
    x: Iterable[float],
    y: Iterable[Any],
    *,
    width: float = 0.9,
    color: Optional[Any] = None,
    bw_method: Any = None,
    weights: Optional[Iterable[float]] = None,
    n: int = 200,
) -> Element:
    """
    Draw horizontal violins for distributions of x grouped by y.
    One violin per unique y (in order of first appearance).

    Args:
        x: Sample values.
        y: Group values along y-axis (categorical or numeric).
        width: Maximum full height in y data units (vector).
        color: Fill color(s). Scalar, per-group, or per-sample.
        bw_method: Bandwidth for gaussian_kde.
        weights: Optional per-sample weights.
        n: Number of grid points for KDE evaluation.
    """
    x_list = list(map(float, x))
    y_list = list(y)
    if len(x_list) != len(y_list):
        raise ValueError("x and y must have the same length")

    w_list = list(weights) if weights is not None else None
    if w_list is not None and len(w_list) != len(x_list):
        w_list = None

    groups = _unique_in_order(y_list)
    gidx = _group_indices(y_list)
    fills = _colors_for_groups(groups, color, len(y_list), gidx)

    container = Element("g")

    for g in groups:
        idxs = gidx.get(g, [])
        if len(idxs) < 2:
            continue

        data = np.asarray([x_list[i] for i in idxs], dtype=float)
        ws = (
            np.asarray([w_list[i] for i in idxs], dtype=float)
            if w_list is not None
            else None
        )

        dmin = float(np.min(data))
        dmax = float(np.max(data))
        if not (math.isfinite(dmin) and math.isfinite(dmax)):
            continue
        if dmax <= dmin:
            pad = 1.0 if dmin == 0.0 else abs(dmin) * 1e-6
            dmin -= pad
            dmax += pad

        grid = np.linspace(dmin, dmax, int(max(2, n)))
        dens = _safe_kde_1d(data, grid, bw_method, ws)
        dens_max = float(np.max(dens)) if dens.size > 0 else 0.0
        if dens_max <= 0.0 or not math.isfinite(dens_max):
            continue
        half = 0.5 * width * (dens / dens_max)

        x_vec = length_params("x", grid, CtxLenType.Pos)
        y_center = length_params("y", [g] * len(grid), CtxLenType.Pos)
        hw_y = cyv(half)

        y_low = y_center - hw_y
        y_high = y_center + hw_y

        path = HorizontalViolinElement(
            x=x_vec,
            y_low=y_low,
            y_high=y_high,
            fill=color_params("color", fills[g]),
            **{
                "stroke": ConfigKey("linecolor"),
                "stroke-width": ConfigKey("linestroke"),
            },
        )
        container.append(path)

    if len(container) == 1:
        return container[0]
    return container


# Alias
violin = vertical_violin
