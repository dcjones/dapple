from numbers import Real
from typing import Any, Iterable, override

import numpy as np
from numpy.typing import ArrayLike

from ..colors import Colors
from ..config import ConfigKey
from ..coordinates import Lengths, ResolveContext, mm
from ..moderngl_utils import calculate_dpi_size, render_rectangles_to_texture
from .heatmap import Heatmap
from .image import ImageElement


class RasterizedHeatmap(Heatmap):
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
        dpi=ConfigKey("rasterize_dpi"),
    ):
        super().__init__(
            color,
            x=x,
            y=y,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            exclude_diagonal=exclude_diagonal,
        )
        self.tag = "dapple:rasterized_heatmap"
        self.attrib["dpi"] = dpi

    @override
    def resolve(self, ctx: ResolveContext) -> "Element":
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


def rasterized_heatmap(
    color: ArrayLike,
    x: Iterable[Any] | None = None,
    y: Iterable[Any] | None = None,
    x0: Iterable[Any] | None = None,
    x1: Iterable[Any] | None = None,
    y0: Iterable[Any] | None = None,
    y1: Iterable[Any] | None = None,
    exclude_diagonal: bool = False,
) -> RasterizedHeatmap:
    """
    Draw colored squares from a matrix of values, rendered to an embedded image
    rather than to individual SVG rectangles.

    Equivalent to `heatmap`, but far cheaper for large matrices. The raster
    resolution is taken from the config's "rasterize_dpi".

    Args:
        color: 2D matrix of values mapped to the color scale.
        x: Optional 1D sequence giving column center positions. Defaults to unit
            spacing. Cannot be combined with `x0`/`x1`.
        y: Optional 1D sequence giving row center positions. Defaults to unit
            spacing. Cannot be combined with `y0`/`y1`.
        x0: Optional 1D sequence giving each column's left edge, used with `x1`
            to draw squares of varying width.
        x1: Optional 1D sequence giving each column's right edge.
        y0: Optional 1D sequence giving each row's top edge, used with `y1` to
            draw squares of varying height.
        y1: Optional 1D sequence giving each row's bottom edge.
        exclude_diagonal: When true, omit squares where row index equals column
            index.

    Returns:
        A `RasterizedHeatmap` element.
    """
    return RasterizedHeatmap(
        color, x=x, y=y, x0=x0, x1=x1, y0=y0, y1=y1, exclude_diagonal=exclude_diagonal
    )
