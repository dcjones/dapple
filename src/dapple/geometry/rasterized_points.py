import numpy as np
from typing import Optional, Union, Tuple
from ..elements import Element
from ..scales import length_params, color_params
from ..coordinates import CtxLenType, ResolveContext, AbsLengths, mm
from ..config import ConfigKey
from ..moderngl_utils import render_points_to_texture, calculate_dpi_size
from ..colors import Colors
from .image import ImageElement


def rasterized_points(
    x,
    y,
    color=ConfigKey("pointcolor"),
    size=ConfigKey("pointsize"),
    dpi=ConfigKey("rasterize_dpi"),
):
    """
    Create rasterized points geometry using ModernGL for high-performance rendering
    of large point datasets.

    Args:
        x: X coordinates of points
        y: Y coordinates of points
        color: Point colors (default uses config pointcolor)
        size: Point size (default uses config pointsize)
        dpi: Dots per inch for rasterization (default 150.0)

    Returns:
        RasterizedPointsElement: Element that renders points to a rasterized image
    """
    return RasterizedPointsElement(x, y, color, size, dpi)


class RasterizedPointsElement(Element):
    """
    Element that rasterizes points using ModernGL and embeds the result as an image.
    This is useful for plots with very large numbers of points where individual
    SVG elements would be inefficient.
    """

    def __init__(self, x, y, color, size, dpi):
        attrib: dict[str, object] = {
            "x": length_params("x", x, CtxLenType.Pos),
            "y": length_params("y", y, CtxLenType.Pos),
            "color": color_params("color", color),
            "size": size,
            "dpi": dpi,
        }

        super().__init__("dapple:rasterized_points", attrib)

    def resolve(self, ctx: ResolveContext) -> Element:
        """
        Resolve the rasterized points by rendering them to a texture and
        creating an image element.
        """
        # Resolve coordinate and color parameters

        resolved = super().resolve(ctx)

        x = resolved.attrib["x"]
        y = resolved.attrib["y"]
        size = resolved.attrib["size"]
        color = resolved.attrib["color"]
        dpi = resolved.attrib["dpi"]

        assert isinstance(x, AbsLengths)
        assert isinstance(y, AbsLengths)
        assert isinstance(size, AbsLengths)
        assert isinstance(color, Colors)
        assert isinstance(dpi, int | float)

        npoints = max(len(x), len(y), len(color), len(size))

        if len(x) != npoints:
            x = x.repeat_scalar(npoints)

        if len(y) != npoints:
            y = y.repeat_scalar(npoints)

        if len(color) != npoints:
            color = color.repeat_scalar(npoints)

        point_size = size.scalar_value()

        # Convert point size from mm to pixels at the given DPI
        point_size_pixels = (point_size / 25.4) * dpi

        # Calculate data bounds
        x_min, x_max = x.values.min(), x.values.max()
        y_min, y_max = y.values.min(), y.values.max()

        # Add some padding for points at the edges
        padding = point_size * 2  # padding in mm
        x_range = (x_min - padding, x_max + padding)
        y_range = (y_min - padding, y_max + padding)

        # Calculate image dimensions based on data range and DPI
        width_mm = x_range[1] - x_range[0]
        height_mm = y_range[1] - y_range[0]
        width_pixels, height_pixels = calculate_dpi_size(width_mm, height_mm, dpi)

        # Ensure minimum dimensions
        width_pixels = max(width_pixels, 32)
        height_pixels = max(height_pixels, 32)

        # Render points to texture
        texture_data = render_points_to_texture(
            x_coords=x.values,
            y_coords=y.values,
            colors=color.values,
            point_size=2 * point_size_pixels,  # radius to diameter
            width=width_pixels,
            height=height_pixels,
            x_range=x_range,
            y_range=y_range,
        )

        # Create image element with the rasterized data
        image_elem = ImageElement(
            x=mm(x_range[0]),
            y=mm(y_range[0]),
            width=mm(width_mm),
            height=mm(height_mm),
            data=texture_data,
        )

        return image_elem.resolve(ctx)
