from __future__ import annotations

from typing import Any, Iterable, override

import numpy as np
from numpy.typing import ArrayLike

from ..colors import Colors
from ..config import ConfigKey
from ..coordinates import CoordBounds, CtxLenType, Lengths, ResolveContext, cxv, cyv, mm
from ..elements import Element
from ..scales import (
    UnscaledBinaryOp,
    UnscaledExpr,
    UnscaledValues,
    color_params,
    length_params,
)
from .bars import Bar


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


class Heatmap(Element):
    def __init__(
        self,
        color: ArrayLike,
        x: Iterable[Any] | None = None,
        y: Iterable[Any] | None = None,
        x0: Iterable[Any] | None = None,
        x1: Iterable[Any] | None = None,
        y0: Iterable[Any] | None = None,
        y1: Iterable[Any] | None = None,
        exclude_diagonal: bool = False,
    ):
        super().__init__("dapple:heatmap")

        color_array = np.asarray(color)
        if color_array.ndim != 2:
            raise ValueError("color must be a 2D matrix")

        n_rows, n_cols = tuple(map(int, color_array.shape))

        x0_lens: Lengths | UnscaledExpr
        x1_lens: Lengths | UnscaledExpr
        col_positions = None
        if x0 is not None and x1 is not None:
            if x is not None:
                raise ValueError("x cannot be specified when x0 and x1 are provided")
            x0_lens_ = length_params("x", x0, CtxLenType.Pos)
            assert isinstance(x0_lens_, (Lengths, UnscaledValues))
            x1_lens_ = length_params("x", x1, CtxLenType.Pos)
            assert isinstance(x1_lens_, (Lengths, UnscaledValues))
            x0_lens = x0_lens_
            x1_lens = x1_lens_
        elif x is not None:
            if x0 is not None or x1 is not None:
                raise ValueError("x0 and x1 cannot be specified when x is provided")
            col_positions = length_params("x", x, CtxLenType.Pos)
            assert isinstance(col_positions, (Lengths, UnscaledValues))
            x0_lens = col_positions - 0.5 * cxv
            x1_lens = col_positions + 0.5 * cxv
        else:
            col_positions = length_params("x", np.arange(n_cols), CtxLenType.Pos)
            assert isinstance(col_positions, (Lengths, UnscaledValues))
            x0_lens = col_positions - 0.5 * cxv
            x1_lens = col_positions + 0.5 * cxv

        y0_lens: Lengths | UnscaledExpr
        y1_lens: Lengths | UnscaledExpr
        row_positions = None
        if y0 is not None and y1 is not None:
            if y is not None:
                raise ValueError("y cannot be specified when y0 and y1 are provided")
            y0_lens_ = length_params("y", y0, CtxLenType.Pos)
            assert isinstance(y0_lens_, (Lengths, UnscaledValues))
            y1_lens_ = length_params("y", y1, CtxLenType.Pos)
            assert isinstance(y1_lens_, (Lengths, UnscaledValues))
            y0_lens = y0_lens_
            y1_lens = y1_lens_
        elif y is not None:
            if y0 is not None or y1 is not None:
                raise ValueError("y0 and y1 cannot be specified when y is provided")
            row_positions = length_params("y", y, CtxLenType.Pos)
            assert isinstance(row_positions, (Lengths, UnscaledValues))
            y0_lens = row_positions - 0.5 * cyv
            y1_lens = row_positions + 0.5 * cyv
        else:
            row_positions = length_params("y", np.arange(n_rows), CtxLenType.Pos)
            assert isinstance(row_positions, (Lengths, UnscaledValues))
            y0_lens = row_positions - 0.5 * cyv
            y1_lens = row_positions + 0.5 * cyv

        if len(x0_lens) != n_cols or len(x1_lens) != n_cols:
            raise ValueError(
                "x arguments must have the same length as the number of columns"
            )

        if len(y0_lens) != n_rows or len(y1_lens) != n_rows:
            raise ValueError(
                "y arguments must have the same length as the number of rows"
            )

        self.attrib = {
            "exclude_diagonal": exclude_diagonal,
            "x0": x0_lens,
            "x1": x1_lens,
            "y0": y0_lens,
            "y1": y1_lens,
            "fill": color_params("color", color_array.reshape(-1)),
            "dapple:nudge": ConfigKey("heatmap_nudge"),
        }

        if exclude_diagonal:
            if isinstance(x0_lens, UnscaledValues) and isinstance(
                y0_lens, UnscaledValues
            ):
                x_grid, y_grid = np.meshgrid(
                    np.asarray(x0_lens.values), np.asarray(y0_lens.values)
                )
                self.attrib["mask"] = (x_grid != y_grid).flatten()
            if col_positions is not None and row_positions is not None:
                col_grid, row_grid = np.meshgrid(col_positions, row_positions)
                self.attrib["mask"] = (col_grid != row_grid).flatten()
            else:
                raise ValueError("Cannot exclude diagonal for pre-scaled positions")

    @override
    def update_bounds(self, bounds: CoordBounds):
        bounds.update(self.get_as("x0", Lengths))
        bounds.update(self.get_as("x1", Lengths))
        bounds.update(self.get_as("y0", Lengths))
        bounds.update(self.get_as("y1", Lengths))

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        x0 = self.get_as("x0", Lengths).resolve(ctx).values
        x1 = self.get_as("x1", Lengths).resolve(ctx).values
        y0 = self.get_as("y0", Lengths).resolve(ctx).values
        y1 = self.get_as("y1", Lengths).resolve(ctx).values

        assert len(x0) == len(x1) and len(y0) == len(y1)
        nrows = len(y0)
        ncols = len(x0)

        fill = self.get_as("fill", Colors)
        assert nrows * ncols == len(fill)

        (x0_grid, y0_grid) = np.meshgrid(x0, y0)
        (x1_grid, y1_grid) = np.meshgrid(x1, y1)

        x0 = x0_grid.flatten()
        x1 = x1_grid.flatten()
        y0 = y0_grid.flatten()
        y1 = y1_grid.flatten()

        if "mask" in self.attrib:
            mask = self.get_as("mask", np.ndarray)
            x0 = x0[mask]
            x1 = x1[mask]
            y0 = y0[mask]
            y1 = y1[mask]
            fill = fill[mask]

        nudge = self.get_as("dapple:nudge", Lengths)

        return Bar(
            mm(x0),
            mm(y0),
            x1=mm(x1),
            y1=mm(y1),
            fill=fill,
            nudge=None,
            stroke="none",
            **{
                "shape-rendering": "crispEdges",
                "dapple:nudge": nudge.resolve(ctx),
            },
        ).resolve(ctx)


def heatmap(
    color: ArrayLike,
    x: Iterable[Any] | None = None,
    y: Iterable[Any] | None = None,
    x0: Iterable[Any] | None = None,
    x1: Iterable[Any] | None = None,
    y0: Iterable[Any] | None = None,
    y1: Iterable[Any] | None = None,
    exclude_diagonal: bool = False,
) -> Heatmap:
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

    return Heatmap(
        color, x=x, y=y, x0=x0, x1=x1, y0=y0, y1=y1, exclude_diagonal=exclude_diagonal
    )
