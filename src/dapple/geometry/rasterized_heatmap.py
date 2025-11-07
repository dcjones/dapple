from numbers import Real
from typing import Any, Iterable, override

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..colors import Colors
from ..config import ConfigKey
from ..coordinates import (
    CoordBounds,
    CtxLenType,
    Lengths,
    ResolveContext,
    cxv,
    cyv,
    mm,
    resolve,
)
from ..elements import Element
from ..moderngl_utils import calculate_dpi_size, render_rectangles_to_texture
from ..scales import UnscaledValues, color_params, length_params
from .image import ImageElement


class RasterizedHeatmap(Element):
    def __init__(
        self,
        color: ArrayLike,
        x: Iterable[Any] | None = None,
        y: Iterable[Any] | None = None,
        exclude_diagonal: bool = False,
        dpi=ConfigKey("rasterize_dpi"),
    ):
        super().__init__("dapple:rasterized_heatmap")

        color_array = np.asarray(color)
        if color_array.ndim != 2:
            raise ValueError("color must be a 2D matrix")

        n_rows, n_cols = color_array.shape

        col_positions = (
            length_params("x", np.arange(n_cols), CtxLenType.Pos)
            if x is None
            else length_params("x", x, CtxLenType.Pos)
        )
        assert isinstance(col_positions, Lengths) or isinstance(
            col_positions, UnscaledValues
        )

        row_positions = (
            length_params("y", np.arange(n_rows), CtxLenType.Pos)
            if y is None
            else length_params("y", y, CtxLenType.Pos)
        )
        assert isinstance(row_positions, Lengths) or isinstance(
            row_positions, UnscaledValues
        )

        self.attrib = {
            "exclude_diagonal": exclude_diagonal,
            "x0": col_positions - 0.5 * cxv,
            "x1": col_positions + 0.5 * cxv,
            "y0": row_positions - 0.5 * cyv,
            "y1": row_positions + 0.5 * cyv,
            "fill": color_params("color", color_array.reshape(-1)),
            "dpi": dpi,
        }

        if exclude_diagonal:
            if not isinstance(col_positions, UnscaledValues) or not isinstance(
                row_positions, UnscaledValues
            ):
                raise ValueError("Cannot exclude diagonal for per-scaled positions")

            x, y = np.meshgrid(
                np.asarray(col_positions.values), np.asarray(row_positions.values)
            )

            self.attrib["mask"] = (x != y).flatten()

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

        dpi = self.get("dpi")
        assert isinstance(dpi, Real)
        dpi = float(dpi)

        x_range = (
            min(x0.min(), x1.min()),
            max(x0.max(), x1.max()),
        )
        y_range = (
            min(y0.min(), y1.min()),
            max(y0.max(), y1.max()),
        )

        width_mm = x_range[1] - x_range[0]
        height_mm = y_range[1] - y_range[0]
        width_pixels, height_pixels = calculate_dpi_size(width_mm, height_mm, dpi)

        # Ensure minimum dimensions
        width_pixels = max(width_pixels, 32)
        height_pixels = max(height_pixels, 32)

        ncols = len(x0)
        nrows = len(y0)

        x = np.minimum(x0, x1)
        y = np.minimum(y0, y1)

        w = np.maximum(x0, x1) - x
        h = np.maximum(y0, y1) - y

        (x, y) = np.meshgrid(x, y)
        (w, h) = np.meshgrid(w, h)

        x = x.flatten()
        y = y.flatten()
        w = w.flatten()
        h = h.flatten()

        if "mask" in self.attrib:
            mask = self.get_as("mask", np.ndarray)
            x = x[mask]
            y = y[mask]
            w = w[mask]
            h = h[mask]
            fill = fill[mask]

        texture_data = render_rectangles_to_texture(
            x,
            y,
            w,
            h,
            fill.values,
            width_pixels,
            height_pixels,
            x_range=x_range,
            y_range=y_range,
        )

        image_elem = ImageElement(
            x=mm(x_range[0]),
            y=mm(y_range[0]),
            width=mm(width_mm),
            height=mm(height_mm),
            data=texture_data,
        )

        return image_elem.resolve(ctx)

        # TODO: Ok, have to figure out how to rasterize rects with moderngl


def rasterized_heatmap(
    color: ArrayLike,
    x: Iterable[Any] | None = None,
    y: Iterable[Any] | None = None,
    exclude_diagonal: bool = False,
) -> RasterizedHeatmap:
    return RasterizedHeatmap(color, x, y, exclude_diagonal)
