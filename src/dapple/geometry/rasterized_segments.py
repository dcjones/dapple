from __future__ import annotations

from typing import override

import numpy as np

from ..colors import Colors
from ..config import ConfigKey
from ..coordinates import AbsLengths, CtxLenType, ResolveContext, mm
from ..elements import Element
from ..moderngl_utils import calculate_dpi_size, render_lines_to_texture
from ..scales import color_params, length_params
from .image import ImageElement

__all__ = ("rasterized_segments", "RasterizedSegmentsElement")


def rasterized_segments(
    x1,
    y1,
    x2,
    y2,
    color=ConfigKey("linecolor"),
    stroke_width=ConfigKey("linestroke"),
    dpi=ConfigKey("rasterize_dpi"),
    arrow=False,
) -> Element:
    """
    Create a rasterized collection of line segments rendered with ModernGL.

    Args:
        x1: Starting x-coordinates for each segment.
        y1: Starting y-coordinates for each segment.
        x2: Ending x-coordinates for each segment.
        y2: Ending y-coordinates for each segment.
        color: Segment colors (scalar or per-segment).
        stroke_width: Segment stroke width in millimeters.
        dpi: Resolution (dots per inch) for rasterization.
        arrow: Whether to draw arrowheads at the end of each segment.

    Returns:
        RasterizedSegmentsElement configured with the provided data.
    """
    return RasterizedSegmentsElement(x1, y1, x2, y2, color, stroke_width, dpi, arrow)


class RasterizedSegmentsElement(Element):
    """
    Element that rasterizes sets of line segments into an image using ModernGL.
    """

    def __init__(self, x1, y1, x2, y2, color, stroke_width, dpi, arrow=False):
        attrib: dict[str, object] = {
            "x1": length_params("x", x1, CtxLenType.Pos),
            "y1": length_params("y", y1, CtxLenType.Pos),
            "x2": length_params("x", x2, CtxLenType.Pos),
            "y2": length_params("y", y2, CtxLenType.Pos),
            "color": color_params("color", color),
            "stroke-width": stroke_width,
            "dpi": dpi,
            "arrow": arrow,
        }
        super().__init__("dapple:rasterized_segments", attrib)

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        resolved = super().resolve(ctx)

        x1 = resolved.attrib["x1"]
        y1 = resolved.attrib["y1"]
        x2 = resolved.attrib["x2"]
        y2 = resolved.attrib["y2"]
        color = resolved.attrib["color"]
        stroke_w = resolved.attrib["stroke-width"]
        dpi = resolved.attrib["dpi"]
        arrow = resolved.attrib["arrow"]

        assert isinstance(x1, AbsLengths)
        assert isinstance(y1, AbsLengths)
        assert isinstance(x2, AbsLengths)
        assert isinstance(y2, AbsLengths)
        assert isinstance(color, Colors)

        if not isinstance(stroke_w, AbsLengths):
            raise TypeError("stroke-width must resolve to AbsLengths")
        if not stroke_w.isscalar():
            raise ValueError("stroke-width must be a scalar length")

        if not isinstance(dpi, (int, float)):
            raise TypeError("dpi must resolve to a numeric value")
        dpi_value = float(dpi)

        nsegments = max(len(x1), len(y1), len(x2), len(y2))
        if nsegments == 0:
            return Element("g")

        def _broadcast(values: AbsLengths, count: int, name: str) -> AbsLengths:
            if len(values) == count:
                return values
            if len(values) == 1:
                return values.repeat_scalar(count)
            raise ValueError(
                f"{name} length mismatch: expected {count}, found {len(values)}"
            )

        x1 = _broadcast(x1, nsegments, "x1")
        y1 = _broadcast(y1, nsegments, "y1")
        x2 = _broadcast(x2, nsegments, "x2")
        y2 = _broadcast(y2, nsegments, "y2")

        p0 = np.column_stack(
            [
                np.asarray(x1.values, dtype=np.float32),
                np.asarray(y1.values, dtype=np.float32),
            ]
        )
        p1 = np.column_stack(
            [
                np.asarray(x2.values, dtype=np.float32),
                np.asarray(y2.values, dtype=np.float32),
            ]
        )

        lengths = np.linalg.norm(p1 - p0, axis=1)
        valid_mask = lengths > 1e-6
        if not np.any(valid_mask):
            return Element("g")

        p0 = p0[valid_mask]
        p1 = p1[valid_mask]
        # Update nsegments and broadcast valid mask to other arrays if needed
        # but here we only need valid_mask for color_array and later for arrows
        segment_count = p0.shape[0]

        if color.isscalar():
            color_array = color.repeat_scalar(nsegments).values.astype(np.float32)
        elif len(color) == nsegments:
            color_array = color.values.astype(np.float32)
        else:
            raise ValueError(
                "Color length mismatch: expected scalar or one color per segment"
            )
        color_array = color_array[valid_mask, :]

        segments = np.stack([p0, p1], axis=1)

        if arrow is not False:
            # Handle vectorized arrow
            if isinstance(arrow, (np.ndarray, list)):
                arrow_mask = np.asarray(arrow)
                if arrow_mask.shape != (nsegments,):
                    arrow_mask = np.broadcast_to(arrow_mask, (nsegments,))
                arrow_mask = arrow_mask[valid_mask]
            else:
                arrow_mask = np.ones(segment_count, dtype=bool)

            if np.any(arrow_mask):
                # Calculate arrowheads
                diff = p1 - p0
                seg_lengths = np.linalg.norm(diff, axis=1, keepdims=True)
                # seg_lengths > 1e-6 is already guaranteed by valid_mask
                u = diff / seg_lengths

                arrow_len = 2.0  # 2mm
                angle = np.pi / 6  # 30 degrees
                ca = np.cos(angle)
                sa = np.sin(angle)

                # Leg 1 unit vector
                l1_u = np.column_stack(
                    [u[:, 0] * ca - u[:, 1] * sa, u[:, 0] * sa + u[:, 1] * ca]
                )
                p1_l1 = p1 - arrow_len * l1_u

                # Leg 2 unit vector
                l2_u = np.column_stack(
                    [u[:, 0] * ca + u[:, 1] * sa, -u[:, 0] * sa + u[:, 1] * ca]
                )
                p1_l2 = p1 - arrow_len * l2_u

                # Filter by arrow_mask
                p1_arr = p1[arrow_mask]
                p1_l1_arr = p1_l1[arrow_mask]
                p1_l2_arr = p1_l2[arrow_mask]
                color_arr_arr = color_array[arrow_mask]

                leg1_segs = np.stack([p1_arr, p1_l1_arr], axis=1)
                leg2_segs = np.stack([p1_arr, p1_l2_arr], axis=1)

                segments = np.concatenate([segments, leg1_segs, leg2_segs], axis=0)
                color_array = np.concatenate(
                    [color_array, color_arr_arr, color_arr_arr], axis=0
                )

        stroke_mm = stroke_w.scalar_value()
        half_width_mm = stroke_mm * 0.5
        pad = max(half_width_mm, 0.5)

        all_x = np.concatenate([p0[:, 0], p1[:, 0]])
        all_y = np.concatenate([p0[:, 1], p1[:, 1]])

        x_min = float(all_x.min() - pad)
        x_max = float(all_x.max() + pad)
        y_min = float(all_y.min() - pad)
        y_max = float(all_y.max() + pad)

        width_mm = max(x_max - x_min, 1e-6)
        height_mm = max(y_max - y_min, 1e-6)

        width_px, height_px = calculate_dpi_size(width_mm, height_mm, dpi_value)
        width_px = max(width_px, 32)
        height_px = max(height_px, 32)

        line_width_pixels = (stroke_mm / 25.4) * dpi_value

        texture_data = render_lines_to_texture(
            segments=segments,
            colors=color_array,
            line_width=line_width_pixels,
            width=width_px,
            height=height_px,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
        )

        image_elem = ImageElement(
            x=mm(x_min),
            y=mm(y_min),
            width=mm(width_mm),
            height=mm(height_mm),
            data=texture_data,
        )

        return image_elem.resolve(ctx)
