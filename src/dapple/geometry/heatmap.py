from __future__ import annotations

import numpy as np
from typing import Any, Iterable
from numpy.typing import ArrayLike

from ..coordinates import CtxLenType, cxv, cyv, Lengths
from ..scales import color_params, UnscaledValues
from .bars import Bar
from ..config import ConfigKey


def _ensure_1d_positions(values: Iterable[Any], expected_len: int, axis_name: str):
    """
    Convert a collection of positions into a flat Python list and validate its length.
    """
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(f"{axis_name} positions must be a 1D sequence")
        seq = values.tolist()
    elif isinstance(values, Lengths):
        if len(values) != expected_len:
            raise ValueError(
                f"Expected {expected_len} {axis_name} positions, got {len(values)}"
            )
        seq = [values[i] for i in range(len(values))]
    else:
        try:
            seq = list(values)
        except TypeError as err:
            raise ValueError(f"{axis_name} positions must be iterable") from err

        if len(seq) != expected_len:
            raise ValueError(
                f"Expected {expected_len} {axis_name} positions, got {len(seq)}"
            )

    return seq


def _default_positions(n: int) -> np.ndarray:
    """Default to unit-spaced positions when none are provided."""
    return np.arange(n, dtype=np.float64)


def heatmap(
    color: ArrayLike,
    x: Iterable[Any] | None = None,
    y: Iterable[Any] | None = None,
    exclude_diagonal: bool = False,
) -> Bar:
    """
    Draw colored squares from a matrix of values.

    Args:
        color: 2D matrix of values mapped to the color scale.
        x: Optional 1D sequence giving column positions. Defaults to unit spacing.
        y: Optional 1D sequence giving row positions. Defaults to unit spacing.
        exclude_diagonal: When true, omit squares where row index equals column index.

    Returns:
        VectorizedElement containing the heatmap geometry.
    """
    color_array = np.asarray(color)
    if color_array.ndim != 2:
        raise ValueError("color must be a 2D matrix")

    n_rows, n_cols = color_array.shape

    x_centers = (
        _default_positions(n_cols).tolist()
        if x is None
        else _ensure_1d_positions(x, n_cols, "x")
    )
    y_centers = (
        _default_positions(n_rows).tolist()
        if y is None
        else _ensure_1d_positions(y, n_rows, "y")
    )

    x_centers_flat = [x_centers[col] for _row in range(n_rows) for col in range(n_cols)]
    y_centers_flat = [y_centers[row] for row in range(n_rows) for _col in range(n_cols)]
    color_vals = color_array.reshape(-1)

    if exclude_diagonal:
        row_indices = np.repeat(np.arange(n_rows, dtype=np.int64), n_cols)
        col_indices = np.tile(np.arange(n_cols, dtype=np.int64), n_rows)
        mask = row_indices != col_indices

        x_centers_flat = [val for val, keep in zip(x_centers_flat, mask) if keep]
        y_centers_flat = [val for val, keep in zip(y_centers_flat, mask) if keep]
        color_vals = color_vals[mask]

    x_centers_expr = UnscaledValues("x", x_centers_flat, CtxLenType.Pos)
    y_centers_expr = UnscaledValues("y", y_centers_flat, CtxLenType.Pos)

    return Bar(
        x=x_centers_expr - cxv(0.5),
        y=y_centers_expr - cyv(0.5),
        width=cxv(1.0),
        height=cyv(1.0),
        fill=color_params("color", color_vals),
        stroke="none",
        **{"shape-rendering": "crispEdges", "dapple:nudge": ConfigKey("heatmap_nudge")},
    )
