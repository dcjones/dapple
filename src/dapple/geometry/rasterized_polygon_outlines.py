import numpy as np
from typing import Iterable, List

from ..elements import Element
from ..scales import length_params, color_params
from ..coordinates import CtxLenType, ResolveContext, AbsLengths, mm
from ..config import ConfigKey
from ..moderngl_utils import render_lines_to_texture, calculate_dpi_size
from ..colors import Colors
from .image import ImageElement


def rasterized_polygon_outlines(
    polygons,
    color=ConfigKey("linecolor"),
    stroke_width=ConfigKey("linestroke"),
    dpi=ConfigKey("rasterize_dpi"),
):
    """
    Create rasterized polygon outlines geometry using ModernGL.

    Args:
        polygons: A shapely Polygon, MultiPolygon, or an iterable of those.
        color: Stroke color(s). Can be scalar or one per polygon or per segment.
        stroke_width: Stroke width (AbsLengths or ConfigKey) in mm units.
        dpi: Dots per inch for rasterization.

    Returns:
        RasterizedPolygonOutlinesElement: Element that rasterizes polygon edges to an image.
    """
    return RasterizedPolygonOutlinesElement(polygons, color, stroke_width, dpi)


class RasterizedPolygonOutlinesElement(Element):
    """
    Element that rasterizes outlines of shapely polygons using ModernGL and embeds the result as an image.
    The polygon edges are converted to line segments, stored as UnscaledValues for coordinates, and drawn
    with a configurable stroke width (in mm) converted to pixels using DPI.
    """

    def __init__(self, polygons, color, stroke_width, dpi):
        # Import shapely lazily
        try:
            import shapely.geometry as _geom
        except Exception as e:
            raise ImportError(
                "rasterized_polygon_outlines requires shapely. Install with: pip install shapely"
            ) from e

        # Normalize input to a flat list of Polygons (MultiPolygon and iterables supported)
        poly_list: List = []
        self._flatten_polygons(polygons, poly_list)

        # Collect segments for each polygon (including interior rings)
        segments: List[np.ndarray] = []
        segs_per_poly: List[int] = []

        for poly in poly_list:
            if not isinstance(poly, _geom.Polygon):
                continue

            nseg = 0

            # Exterior ring segments
            ext_coords = np.asarray(poly.exterior.coords, dtype=np.float64)
            n_from_ext = self._append_ring_segments(ext_coords, segments)
            nseg += n_from_ext

            # Interior rings (holes) segments
            for ring in poly.interiors:
                int_coords = np.asarray(ring.coords, dtype=np.float64)
                n_from_int = self._append_ring_segments(int_coords, segments)
                nseg += n_from_int

            segs_per_poly.append(nseg)

        segment_count = len(segments)
        if segment_count == 0:
            attrib: dict[str, object] = {
                "x": length_params("x", [], CtxLenType.Pos),
                "y": length_params("y", [], CtxLenType.Pos),
                "segment_count": 0,
                "segs_per_poly": segs_per_poly,
                "color": color_params("color", color),
                "stroke-width": stroke_width,
                "dpi": dpi,
            }
            super().__init__("dapple:rasterized_polygon_outlines", attrib)
            return

        seg_np = np.stack(segments, axis=0).astype(np.float32)  # (S, 2, 2)
        x_vals = seg_np[:, :, 0].reshape(-1)
        y_vals = seg_np[:, :, 1].reshape(-1)

        attrib = {
            "x": length_params("x", x_vals, CtxLenType.Pos),
            "y": length_params("y", y_vals, CtxLenType.Pos),
            "segment_count": segment_count,
            "segs_per_poly": segs_per_poly,
            "color": color_params("color", color),
            "stroke-width": stroke_width,
            "dpi": dpi,
        }
        super().__init__("dapple:rasterized_polygon_outlines", attrib)

    def resolve(self, ctx: ResolveContext) -> Element:
        """
        Resolve by rendering pre-collected polygon edge segments into a texture and
        creating an image element.
        """
        resolved = super().resolve(ctx)

        x = resolved.attrib["x"]
        y = resolved.attrib["y"]
        segment_count = resolved.attrib["segment_count"]
        color = resolved.attrib["color"]
        stroke_w = resolved.attrib["stroke-width"]
        dpi = resolved.attrib["dpi"]
        segs_per_poly = resolved.attrib.get("segs_per_poly", None)

        assert isinstance(x, AbsLengths)
        assert isinstance(y, AbsLengths)
        if segment_count <= 0:
            return Element("g")

        assert isinstance(color, Colors)
        assert isinstance(stroke_w, AbsLengths)
        assert isinstance(dpi, (int, float))
        assert isinstance(segment_count, int)

        verts = np.column_stack([x.values, y.values]).astype(np.float32)
        if verts.shape[0] != 2 * segment_count:
            raise ValueError(
                f"Expected {2 * segment_count} vertices but found {verts.shape[0]}"
            )
        segments_np = verts.reshape((segment_count, 2, 2))

        # Bounds in mm
        x_min = float(np.min(segments_np[:, :, 0]))
        x_max = float(np.max(segments_np[:, :, 0]))
        y_min = float(np.min(segments_np[:, :, 1]))
        y_max = float(np.max(segments_np[:, :, 1]))

        # Convert stroke width from mm to pixels at given DPI
        stroke_mm = stroke_w.scalar_value()
        line_width_pixels = (stroke_mm / 25.4) * float(dpi)

        # Pad ranges to avoid clipping thick strokes
        pad = max(stroke_mm, 0.5)  # at least a small pad
        x_min -= pad
        x_max += pad
        y_min -= pad
        y_max += pad

        width_mm = x_max - x_min
        height_mm = y_max - y_min

        # Minimum image size constraints
        width_px, height_px = calculate_dpi_size(
            max(width_mm, 1e-6), max(height_mm, 1e-6), float(dpi)
        )
        width_px = max(width_px, 32)
        height_px = max(height_px, 32)

        # Colors per segment
        if len(color) == segment_count:
            per_segment_colors = color.values.astype(np.float32)
        elif color.isscalar():
            per_segment_colors = np.repeat(
                color.values.astype(np.float32), segment_count, axis=0
            )
        elif isinstance(segs_per_poly, list) and len(color) == len(segs_per_poly):
            repeats = np.array(segs_per_poly, dtype=int)
            per_segment_colors = np.repeat(
                color.values.astype(np.float32), repeats, axis=0
            )
            if per_segment_colors.shape[0] != segment_count:
                raise ValueError(
                    "Expanded per-polygon colors did not match segment count."
                )
        else:
            raise ValueError(
                f"Color length mismatch: expected scalar, {segment_count} (per segment), or {len(segs_per_poly) if isinstance(segs_per_poly, list) else 'N'} (per polygon)."
            )

        # Render to texture
        texture_data = render_lines_to_texture(
            segments=segments_np,
            colors=per_segment_colors,
            line_width=line_width_pixels,
            width=width_px,
            height=height_px,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
        )

        # Image element
        image_elem = ImageElement(
            x=mm(x_min),
            y=mm(y_min),
            width=mm(width_mm),
            height=mm(height_mm),
            data=texture_data,
        )

        return image_elem.resolve(ctx)

    @staticmethod
    def _flatten_polygons(geom, out: List) -> None:
        """
        Flatten input geometry into a list of shapely Polygons.
        Supports Polygon, MultiPolygon, or iterables of those.
        """
        try:
            import shapely.geometry as _geom
        except Exception:
            return

        if geom is None:
            return

        if isinstance(geom, _geom.Polygon):
            out.append(geom)
        elif isinstance(geom, _geom.MultiPolygon):
            for g in geom.geoms:
                if isinstance(g, _geom.Polygon):
                    out.append(g)
        elif isinstance(geom, Iterable) and not isinstance(geom, (str, bytes)):
            for g in geom:
                RasterizedPolygonOutlinesElement._flatten_polygons(g, out)
        else:
            raise TypeError(
                "polygons must be a shapely Polygon, MultiPolygon, or an iterable of those."
            )

    @staticmethod
    def _append_ring_segments(coords: np.ndarray, segments: List[np.ndarray]) -> int:
        """
        Append segments for a closed ring defined by `coords`.
        coords should include the closing point (last == first).
        Returns number of segments appended.
        """
        if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 2:
            return 0

        # Build segments between consecutive points; include closing edge if present
        n = coords.shape[0]
        count = 0
        for i in range(n - 1):
            p0 = coords[i]
            p1 = coords[i + 1]
            if np.allclose(p0, p1):
                continue
            segments.append(np.stack([p0, p1], axis=0))
            count += 1
        return count
