"""Geometry for hexagonal binning (hexbin) plots."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ..colors import Colors
from ..config import ConfigKey
from ..coordinates import (
    AbsLengths,
    CoordBounds,
    CtxLenType,
    Lengths,
    ResolveContext,
    Serializable,
    mm,
)
from ..elements import Element, VectorizedElement
from ..moderngl_utils import calculate_dpi_size, render_triangles_to_texture
from ..scales import color_params, length_params
from .image import ImageElement

__all__ = ["hexbin"]


def _hex_vertices(
    cx: np.ndarray,
    cy: np.ndarray,
    radius_x: float,
    radius_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the six vertices of pointy-top hexagons centered at (cx, cy).

    When ``radius_x`` equals ``radius_y``, the hexagons are regular. When they
    differ, hexagons are anisotropically scaled (stretched along one axis).

    Args:
        cx: 1D array of hex center x-coordinates.
        cy: 1D array of hex center y-coordinates.
        radius_x: Circumradius in the x-direction.
        radius_y: Circumradius in the y-direction.

    Returns:
        Tuple ``(x_verts, y_verts)`` where each has shape ``(n_hexagons, 6)``
        containing the x and y coordinates of the six vertices.
    """
    # Pointy-top: first vertex at top (angle = -pi/2), then every 60 degrees
    angles = np.arange(6) * (np.pi / 3.0) - np.pi / 2.0
    dx = radius_x * np.cos(angles)
    dy = radius_y * np.sin(angles)

    x_verts = cx[:, np.newaxis] + dx[np.newaxis, :]
    y_verts = cy[:, np.newaxis] + dy[np.newaxis, :]

    return x_verts, y_verts


def _hexbin_counts(
    x: np.ndarray,
    y: np.ndarray,
    bins_x: int,
    bins_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Compute hexagonal bin counts for 2D data.

    Uses a pointy-top hexagonal grid with axial coordinates. When ``bins_x``
    and ``bins_y`` differ, the data is pre-scaled so that the hex grid has
    approximately the requested number of bins along each axis, then the
    resulting hex centers and vertices are transformed back to the original
    coordinate space, producing anisotropically stretched hexagons.

    Args:
        x: 1D array of x-coordinates.
        y: 1D array of y-coordinates.
        bins_x: Approximate number of hexagons spanning the x-axis.
        bins_y: Approximate number of hexagons spanning the y-axis.

    Returns:
        Tuple of ``(cx, cy, counts, radius_x, radius_y)`` where ``cx`` and
        ``cy`` are 1D arrays of hexagon center coordinates, ``counts`` is a
        1D array of point counts per hexagon, and ``radius_x``/``radius_y``
        are the hexagon circumradii in each direction.
    """
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    # For pointy-top hexagons:
    #   horizontal spacing between column centers = sqrt(3) * radius
    #   vertical spacing between row centers = 1.5 * radius
    # So bins_x ~ x_range / (sqrt(3) * radius) and bins_y ~ y_range / (1.5 * radius)
    radius_x = x_range / (np.sqrt(3) * bins_x)
    radius_y = y_range / (1.5 * bins_y)

    if radius_x <= 0:
        radius_x = 1.0
    if radius_y <= 0:
        radius_y = 1.0

    # When bins_x != bins_y, we work in a normalized coordinate space where
    # the hex grid is regular, then transform back. The normalization scales
    # the data so that a single radius works for both axes.
    # We choose the uniform radius as the geometric mean, and scale the data
    # accordingly.
    radius = (radius_x * radius_y) ** 0.5

    # Scale factors to normalize the data so a regular hex grid with `radius`
    # produces approximately bins_x columns and bins_y rows
    scale_x = radius / radius_x
    scale_y = radius / radius_y

    # Transform data to normalized space
    x_norm = (x - x_min) * scale_x
    y_norm = (y - y_min) * scale_y

    # Convert to fractional axial coordinates (q, r) for pointy-top hexagons
    q_f = (np.sqrt(3) / 3.0 * x_norm - 1.0 / 3.0 * y_norm) / radius
    r_f = (2.0 / 3.0 * y_norm) / radius

    # Cube coordinates for rounding: q + r + s = 0
    s_f = -q_f - r_f

    q = np.round(q_f).astype(np.int64)
    r = np.round(r_f).astype(np.int64)
    s = np.round(s_f).astype(np.int64)

    # Fix rounding to ensure q + r + s == 0 (cube coordinate constraint)
    q_diff = np.abs(q - q_f)
    r_diff = np.abs(r - r_f)
    s_diff = np.abs(s - s_f)

    fix_q = (q_diff > r_diff) & (q_diff > s_diff)
    fix_r = (~fix_q) & (r_diff > s_diff)
    fix_s = (~fix_q) & (~fix_r)

    q[fix_q] = -r[fix_q] - s[fix_q]
    r[fix_r] = -q[fix_r] - s[fix_r]
    s[fix_s] = -q[fix_s] - r[fix_s]

    # Convert axial coordinates back to normalized Cartesian coordinates
    cx_norm = radius * (np.sqrt(3) * q + np.sqrt(3) / 2.0 * r)
    cy_norm = radius * (3.0 / 2.0 * r)

    # Transform back to original coordinate space
    cx = x_min + cx_norm / scale_x
    cy = y_min + cy_norm / scale_y

    # Count points per hex using unique (q, r) keys
    keys = q.astype(np.int64) * 1000003 + r.astype(np.int64)
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique_keys))

    # Get unique hex centers using the first occurrence for each key
    _, first_idx = np.unique(inverse, return_index=True)
    cx_unique = cx[first_idx]
    cy_unique = cy[first_idx]

    return cx_unique, cy_unique, counts.astype(np.float64), radius_x, radius_y


class HexPoints(Serializable):
    """
    Serializable that produces SVG polygon ``points`` attribute strings.

    Each hexagon is defined by 6 vertices stored in absolute coordinate arrays.
    On serialization, one ``points`` string per hexagon is produced.
    """

    def __init__(self, x_verts: AbsLengths, y_verts: AbsLengths, hex_count: int):
        self.x_verts = x_verts
        self.y_verts = y_verts
        self.hex_count = hex_count

    def serialize(self) -> list[str]:
        x = self.x_verts.values
        y = self.y_verts.values
        result: list[str] = []
        for i in range(self.hex_count):
            start = i * 6
            end = start + 6
            points_str = " ".join(f"{x[j]:.3f},{y[j]:.3f}" for j in range(start, end))
            result.append(points_str)
        return result


class Hexbin(Element):
    """
    Element representing a hexagonal binning plot.

    Each hexagon is rendered as an SVG polygon with a fill color determined
    by the bin count.
    """

    def __init__(
        self,
        cx: np.ndarray,
        cy: np.ndarray,
        counts: np.ndarray,
        radius_x: float,
        radius_y: float,
    ):
        super().__init__("dapple:hexbin")

        n = len(cx)
        assert len(cy) == n
        assert len(counts) == n

        # Compute hex vertices (6 per hexagon)
        x_verts, y_verts = _hex_vertices(cx, cy, radius_x, radius_y)

        # Flatten: each hexagon has 6 vertices, stored sequentially
        x_flat = x_verts.flatten()
        y_flat = y_verts.flatten()

        self.attrib = {
            "x": length_params("x", x_flat, CtxLenType.Pos),
            "y": length_params("y", y_flat, CtxLenType.Pos),
            "fill": color_params("color", counts),
            "hex_count": n,
            "radius_x": radius_x,
            "radius_y": radius_y,
        }

    @override
    def update_bounds(self, bounds: CoordBounds):
        bounds.update(self.get_as("x", Lengths))
        bounds.update(self.get_as("y", Lengths))

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        x = self.get_as("x", Lengths).resolve(ctx)
        y = self.get_as("y", Lengths).resolve(ctx)
        fill = self.get_as("fill", Colors)
        hex_count = self.attrib["hex_count"]
        assert isinstance(hex_count, int)

        hex_points = HexPoints(x, y, hex_count)

        return VectorizedElement(
            "polygon",
            {
                "points": hex_points,
                "fill": fill,
                "stroke": "none",
            },
        )


class RasterizedHexbin(Element):
    """
    Element representing a rasterized hexagonal binning plot.

    Hexagons are triangulated and rendered to a texture using ModernGL,
    then embedded as an image element.
    """

    def __init__(
        self,
        cx: np.ndarray,
        cy: np.ndarray,
        counts: np.ndarray,
        radius_x: float,
        radius_y: float,
        dpi=ConfigKey("rasterize_dpi"),
    ):
        super().__init__("dapple:rasterized_hexbin")

        n = len(cx)
        assert len(cy) == n
        assert len(counts) == n

        # Compute hex vertices
        x_verts, y_verts = _hex_vertices(cx, cy, radius_x, radius_y)

        # Triangulate each hexagon into 6 triangles (fan from center)
        # For each hexagon, 6 triangles: (center, v_i, v_{i+1}) for i in 0..5
        n_triangles = n * 6

        # Build triangle vertices using vectorized numpy operations
        # Centers repeated 6 times per hexagon
        cx_rep = np.repeat(cx, 6)
        cy_rep = np.repeat(cy, 6)

        # Vertex indices: for each hexagon, vertices 0..5 then (1..5, 0)
        v_idx = np.tile(np.arange(6), n)
        v_next_idx = np.tile((np.arange(6) + 1) % 6, n)

        # Map to flat vertex arrays
        hex_offsets = np.repeat(np.arange(n), 6) * 6
        flat_v = hex_offsets + v_idx
        flat_v_next = hex_offsets + v_next_idx

        x_flat = x_verts.flatten()
        y_flat = y_verts.flatten()

        # Build triangle coordinate arrays: (n_triangles * 3)
        tri_x = np.empty(n_triangles * 3)
        tri_y = np.empty(n_triangles * 3)

        # Vertex 0 of each triangle: center
        tri_x[0::3] = cx_rep
        tri_y[0::3] = cy_rep
        # Vertex 1: hex vertex i
        tri_x[1::3] = x_flat[flat_v]
        tri_y[1::3] = y_flat[flat_v]
        # Vertex 2: hex vertex (i+1) % 6
        tri_x[2::3] = x_flat[flat_v_next]
        tri_y[2::3] = y_flat[flat_v_next]

        self.attrib = {
            "x": length_params("x", tri_x, CtxLenType.Pos),
            "y": length_params("y", tri_y, CtxLenType.Pos),
            "fill": color_params("color", counts),
            "triangle_count": n_triangles,
            "dpi": dpi,
        }

    @override
    def update_bounds(self, bounds: CoordBounds):
        bounds.update(self.get_as("x", Lengths))
        bounds.update(self.get_as("y", Lengths))

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        x = self.get_as("x", Lengths).resolve(ctx).values
        y = self.get_as("y", Lengths).resolve(ctx).values
        fill = self.get_as("fill", Colors)
        triangle_count = self.attrib["triangle_count"]
        assert isinstance(triangle_count, int)

        dpi = self.get("dpi")
        assert isinstance(dpi, Real)
        dpi = float(dpi)

        if triangle_count <= 0:
            return Element("g")

        # Build triangles array
        verts = np.column_stack([x, y]).astype(np.float32)
        triangles_np = verts.reshape((triangle_count, 3, 2))

        # Compute bounds
        x_min = float(np.min(triangles_np[:, :, 0]))
        x_max = float(np.max(triangles_np[:, :, 0]))
        y_min = float(np.min(triangles_np[:, :, 1]))
        y_max = float(np.max(triangles_np[:, :, 1]))

        width_mm = x_max - x_min
        height_mm = y_max - y_min

        if width_mm <= 0.0:
            x_min -= 0.5
            x_max += 0.5
            width_mm = x_max - x_min
        if height_mm <= 0.0:
            y_min -= 0.5
            y_max += 0.5
            height_mm = y_max - y_min

        width_px, height_px = calculate_dpi_size(
            max(width_mm, 1e-6), max(height_mm, 1e-6), dpi
        )
        width_px = max(width_px, 32)
        height_px = max(height_px, 32)

        # Colors: one per hexagon, expand to one per triangle (6 triangles per hex)
        hex_count = triangle_count // 6
        if len(fill) == hex_count:
            per_triangle_colors = np.repeat(fill.values.astype(np.float32), 6, axis=0)
        elif fill.isscalar():
            per_triangle_colors = np.repeat(
                fill.values.astype(np.float32), triangle_count, axis=0
            )
        else:
            raise ValueError(
                f"Color length mismatch: expected {hex_count} (per hexagon) "
                f"or scalar, got {len(fill)}"
            )

        texture_data = render_triangles_to_texture(
            triangles=triangles_np,
            colors=per_triangle_colors,
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


def _normalize_bins(bins: int | Sequence[int]) -> tuple[int, int]:
    """
    Normalize the ``bins`` argument into a pair of positive integers.

    Args:
        bins: Either a single integer applied to both axes or a sequence of
            two integers.

    Returns:
        A tuple ``(bins_x, bins_y)`` with strictly positive integers.

    Raises:
        TypeError: If bins are not integers or a pair of integers.
        ValueError: If any bin count is non-positive or a sequence has the
            wrong length.
    """
    if isinstance(bins, Integral):
        bins_x = bins_y = int(bins)
    else:
        if not isinstance(bins, Sequence) or isinstance(bins, (str, bytes)):
            raise TypeError("bins must be an integer or a pair of integers")
        if len(bins) != 2:
            raise ValueError("bins sequence must contain exactly two integers")

        bins_x, bins_y = bins
        if not isinstance(bins_x, Integral) or not isinstance(bins_y, Integral):
            raise TypeError("bins sequence must contain integers")

        bins_x = int(bins_x)
        bins_y = int(bins_y)

    if bins_x <= 0 or bins_y <= 0:
        raise ValueError("bin counts must be positive")

    return bins_x, bins_y


def hexbin(
    x: ArrayLike,
    y: ArrayLike,
    bins: int | Sequence[int] = 10,
    *,
    rasterize: bool = False,
):
    """
    Render a hexagonal binning plot.

    Points are binned into hexagonal cells and the count in each cell is
    mapped to the fill color scale.

    Args:
        x: Sequence of x coordinates for input samples.
        y: Sequence of y coordinates for input samples.
        bins: Number of hexagons for each axis. Either a single integer
            applied to both dimensions or a pair ``(bins_x, bins_y)``.
            Defaults to ``10``.
        rasterize: When ``True``, returns a rasterized hexbin for improved
            performance with large bin counts. Defaults to ``False``.

    Returns:
        A hexbin or rasterized hexbin element representing the binned counts.

    Raises:
        ValueError: If ``x`` and ``y`` differ in length.
        TypeError: If ``bins`` is not an integer or a pair of integers.
    """
    x_array = np.asarray(x, dtype=np.float64).ravel()
    y_array = np.asarray(y, dtype=np.float64).ravel()

    if x_array.shape != y_array.shape:
        raise ValueError("x and y must contain the same number of elements")

    bins_x, bins_y = _normalize_bins(bins)

    cx, cy, counts, radius_x, radius_y = _hexbin_counts(
        x_array, y_array, bins_x, bins_y
    )

    if rasterize:
        return RasterizedHexbin(cx, cy, counts, radius_x, radius_y)
    else:
        return Hexbin(cx, cy, counts, radius_x, radius_y)
