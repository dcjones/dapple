from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Optional, override

import numpy as np

from ..colors import Colors
from ..config import ConfigKey
from ..coordinates import (
    AbsLengths,
    CtxLenType,
    Lengths,
    ResolveContext,
    cxv,
    cyv,
    resolve,
)
from ..elements import Element, Path
from ..geometry.bars import Bar
from ..scales import UnscaledExpr, UnscaledValues, color_params, length_params
from .lines import _adaptive_sample_function

# ---- Utilities ----------------------------------------------------------------


def _unique_in_order(values: Iterable[Any]) -> list[Any]:
    seen = OrderedDict()
    for v in values:
        if v not in seen:
            seen[v] = None
    return list(seen.keys())


def _group_indices(keys: Sequence[Any]) -> dict[Any, list[int]]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    return groups


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

    if isinstance(color, (str, ConfigKey, Colors)):
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


# ---- Orientation Specification ------------------------------------------------


@dataclass(frozen=True)
class _OrientationSpec:
    center_unit: str
    value_unit: str
    offset_factory: Any
    is_vertical: bool

    def center_scalar(self, group: Any) -> UnscaledExpr | Lengths:
        return length_params(self.center_unit, [group], CtxLenType.Pos)

    def value_series(self, values: Sequence[float]) -> UnscaledExpr | Lengths:
        return length_params(self.value_unit, values, CtxLenType.Pos)

    def value_scalar(self, value: float) -> UnscaledExpr | Lengths:
        return length_params(self.value_unit, [value], CtxLenType.Pos)


_VERTICAL_SPEC = _OrientationSpec(
    center_unit="x",
    value_unit="y",
    offset_factory=cxv,
    is_vertical=True,
)

_HORIZONTAL_SPEC = _OrientationSpec(
    center_unit="y",
    value_unit="x",
    offset_factory=cyv,
    is_vertical=False,
)


class ViolinQuartileOverlay(Element):
    """
    Custom element that renders quartile bar and median line after values have been
    resolved to absolute coordinates.
    """

    def __init__(
        self,
        orientation: str,
        center: Lengths,
        q_low: Lengths,
        q_high: Lengths,
        q_med: Lengths,
        width: ConfigKey,
        fill: UnscaledExpr | Colors,
        stroke: ConfigKey,
        stroke_width: ConfigKey,
    ):
        super().__init__(
            "dapple:violin_quartile",
            {
                "orientation": orientation,
                "center": center,
                "q_low": q_low,
                "q_high": q_high,
                "q_med": q_med,
                "width": width,
                "fill": fill,
                "stroke": stroke,
                "stroke-width": stroke_width,
            },
        )

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        orientation = self.attrib["orientation"]
        center = resolve(self.attrib["center"], ctx)
        q_low = resolve(self.attrib["q_low"], ctx)
        q_high = resolve(self.attrib["q_high"], ctx)
        q_med = resolve(self.attrib["q_med"], ctx)
        width = resolve(self.attrib["width"], ctx)
        fill = resolve(self.attrib["fill"], ctx)
        stroke = resolve(self.attrib["stroke"], ctx)
        stroke_width = resolve(self.attrib["stroke-width"], ctx)

        assert isinstance(center, AbsLengths)
        assert isinstance(q_low, AbsLengths)
        assert isinstance(q_high, AbsLengths)
        assert isinstance(q_med, AbsLengths)
        assert isinstance(width, AbsLengths)

        if width.scalar_value() <= 0:
            return Element("g")

        center_val = center.scalar_value()
        q_low_val = q_low.scalar_value()
        q_high_val = q_high.scalar_value()
        q_med_val = q_med.scalar_value()
        width_val = width.scalar_value()
        half_width = 0.5 * width_val

        if orientation == "vertical":
            bar = Bar(
                x=AbsLengths(np.array([center_val - half_width])),
                y=AbsLengths(np.array([min(q_low_val, q_high_val)])),
                width=AbsLengths(np.array([width_val])),
                height=AbsLengths(np.array([abs(q_high_val - q_low_val)])),
                fill=fill,
            ).resolve(ctx)
            line_attribs = {
                "x1": center_val - half_width,
                "y1": q_med_val,
                "x2": center_val + half_width,
                "y2": q_med_val,
                "stroke": stroke,
                "stroke-width": stroke_width,
                "stroke-linecap": "square",
            }
        else:
            bar = Bar(
                x=AbsLengths(np.array([min(q_low_val, q_high_val)])),
                y=AbsLengths(np.array([center_val - half_width])),
                width=AbsLengths(np.array([abs(q_high_val - q_low_val)])),
                height=AbsLengths(np.array([width_val])),
                fill=fill,
            ).resolve(ctx)
            line_attribs = {
                "x1": q_med_val,
                "y1": center_val - half_width,
                "x2": q_med_val,
                "y2": center_val + half_width,
                "stroke": stroke,
                "stroke-width": stroke_width,
                "stroke-linecap": "square",
            }

        return Element("g", {}, bar, Element("line", line_attribs))


# ---- Core geometry construction ----------------------------------------------


def _compute_violin_path(
    group: Any,
    data: np.ndarray,
    spec: _OrientationSpec,
    width: float,
    fill_value: Any,
    clip: float,
    bw_method: Any,
    weights: Optional[np.ndarray],
) -> Element | None:
    if data.size < 2:
        return None

    dmin = float(np.min(data))
    dmax = float(np.max(data))
    if not (math.isfinite(dmin) and math.isfinite(dmax)):
        return None

    if dmax <= dmin:
        pad = 1.0 if dmin == 0.0 else abs(dmin) * 1e-6
        dmin -= pad
        dmax += pad

    from scipy.stats import gaussian_kde

    kde = gaussian_kde(data, bw_method=bw_method, weights=weights)

    std = float(np.std(data))
    if not math.isfinite(std) or std == 0.0:
        std = max(1e-3, (dmax - dmin) if dmax > dmin else 1.0)

    cur = dmin
    left_limit = dmin - 10.0 * std
    while kde(cur)[0] > clip and cur > left_limit:
        cur -= std
    left_bound = float(cur)

    cur = dmax
    right_limit = dmax + 10.0 * std
    while kde(cur)[0] > clip and cur < right_limit:
        cur += std
    right_bound = float(cur)

    xs, ys = _adaptive_sample_function(lambda v: kde(v)[0], left_bound, right_bound)
    if len(xs) < 2:
        return None

    dens = np.asarray(ys)
    values = np.asarray(xs, dtype=float)
    dens_max = float(np.max(dens)) if dens.size > 0 else 0.0
    if dens_max <= 0.0 or not math.isfinite(dens_max):
        return None

    half = 0.5 * width * (dens / dens_max)

    center = spec.center_scalar(group)
    path_offsets = np.concatenate([-half, half[::-1]])
    path_values = np.concatenate([values, values[::-1]])

    if spec.is_vertical:
        path_x = center + spec.offset_factory(path_offsets)
        path_y = spec.value_series(path_values)
    else:
        path_x = spec.value_series(path_values)
        path_y = center + spec.offset_factory(path_offsets)

    fill_expr = color_params("color", fill_value)

    path = Path(
        path_x,
        path_y,
        fill=fill_expr,
        stroke=ConfigKey("linecolor"),
        closed=True,
        **{"stroke-width": ConfigKey("violin_median_stroke_width")},
    )

    q1 = float(np.quantile(data, 0.25))
    q2 = float(np.quantile(data, 0.50))
    q3 = float(np.quantile(data, 0.75))

    q_low_val = min(q1, q3)
    q_high_val = max(q1, q3)

    q_low = spec.value_scalar(q_low_val)
    q_high = spec.value_scalar(q_high_val)
    q_med = spec.value_scalar(q2)
    center_scalar = center

    elements: list[Element] = [path]

    box_lightness = 0.4 if spec.is_vertical else 0.12
    box_fill = color_params(
        "color",
        fill_value,
        transform=lambda c: c.modulate_lightness(box_lightness),
    )

    overlay = ViolinQuartileOverlay(
        "vertical" if spec.is_vertical else "horizontal",
        center=center_scalar,
        q_low=q_low,
        q_high=q_high,
        q_med=q_med,
        width=ConfigKey("violin_bar_width"),
        fill=box_fill,
        stroke=ConfigKey("linecolor"),
        stroke_width=ConfigKey("violin_median_stroke_width"),
    )

    elements.append(overlay)

    if len(elements) == 1:
        return path

    return Element("g", {}, *elements)


def _violin_impl(
    groups: Sequence[Any],
    data: Sequence[float],
    spec: _OrientationSpec,
    *,
    width: float,
    color: Optional[Any],
    bw_method: Any,
    weights: Optional[Sequence[float]],
    clip: float,
) -> Element:
    groups_in_order = _unique_in_order(groups)
    group_indices = _group_indices(groups)
    fills = _colors_for_groups(groups_in_order, color, len(groups), group_indices)

    weights_array = np.asarray(weights, dtype=float) if weights is not None else None
    data_array = np.asarray(data, dtype=float)

    container = Element("g")

    for group in groups_in_order:
        idxs = group_indices.get(group, [])
        if len(idxs) < 2:
            continue

        group_data = data_array[idxs]
        group_weights = weights_array[idxs] if weights_array is not None else None

        violin_el = _compute_violin_path(
            group,
            group_data,
            spec,
            width,
            fills[group],
            clip,
            bw_method,
            group_weights,
        )

        if violin_el is not None:
            container.append(violin_el)

    if len(container) == 1:
        return container[0]
    return container


# ---- Public API ----------------------------------------------------------------


def vertical_violin(
    x: Any,
    y: Iterable[float],
    *,
    width: float = 0.9,
    color: Optional[Any] = None,
    bw_method: Any = None,
    weights: Optional[Iterable[float]] = None,
    clip: float = 1e-3,
) -> Element:
    """
    Draw vertical violins for distributions of y grouped by x.
    One violin per unique x (in order of first appearance).
    """
    y_list = list(map(float, y))
    if isinstance(x, (str, bytes)):
        x_list = [x] * len(y_list)
    else:
        try:
            x_list = list(x)
        except TypeError:
            x_list = [x] * len(y_list)
        else:
            if len(x_list) == 1 and len(y_list) > 1:
                x_list = x_list * len(y_list)

    if len(x_list) != len(y_list):
        raise ValueError("x and y must have the same length")

    w_list = list(weights) if weights is not None else None
    if w_list is not None and len(w_list) != len(y_list):
        w_list = None

    return _violin_impl(
        x_list,
        y_list,
        _VERTICAL_SPEC,
        width=width,
        color=color,
        bw_method=bw_method,
        weights=w_list,
        clip=clip,
    )


def horizontal_violin(
    x: Iterable[float],
    y: Any,
    *,
    width: float = 0.9,
    color: Optional[Any] = None,
    bw_method: Any = None,
    weights: Optional[Iterable[float]] = None,
    clip: float = 1e-3,
) -> Element:
    """
    Draw horizontal violins for distributions of x grouped by y.
    One violin per unique y (in order of first appearance).
    """
    x_list = list(map(float, x))
    if isinstance(y, (str, bytes)):
        y_list = [y] * len(x_list)
    else:
        try:
            y_list = list(y)
        except TypeError:
            y_list = [y] * len(x_list)
        else:
            if len(y_list) == 1 and len(x_list) > 1:
                y_list = y_list * len(x_list)

    if len(x_list) != len(y_list):
        raise ValueError("x and y must have the same length")

    w_list = list(weights) if weights is not None else None
    if w_list is not None and len(w_list) != len(x_list):
        w_list = None

    return _violin_impl(
        y_list,
        x_list,
        _HORIZONTAL_SPEC,
        width=width,
        color=color,
        bw_method=bw_method,
        weights=w_list,
        clip=clip,
    )


# Alias
violin = vertical_violin
