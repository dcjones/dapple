import numpy as np
from typing import Iterable, List

from ..elements import Element
from ..scales import length_params, color_params
from ..coordinates import CtxLenType, ResolveContext, AbsLengths, mm
from ..config import ConfigKey
from ..moderngl_utils import render_triangles_to_texture, calculate_dpi_size
from ..colors import Colors
from .image import ImageElement


def rasterized_polygons(
    polygons, color=ConfigKey("linecolor"), dpi=ConfigKey("rasterize_dpi")
):
    """
    Create rasterized polygon geometry using ModernGL.

    Args:
        polygons: A shapely Polygon, MultiPolygon, or an iterable of those.
        color: Fill color(s). Can be a scalar color or one per polygon.
        dpi: Dots per inch for rasterization.

    Returns:
        RasterizedPolygonsElement: Element that rasterizes polygons to an image.
    """
    return RasterizedPolygonsElement(polygons, color, dpi)


class RasterizedPolygonsElement(Element):
    """
    Element that rasterizes shapely polygons using ModernGL and embeds the result as an image.
    The polygons are triangulated on the CPU (ear clipping for simple polygons) and rendered
    as filled triangles.
    """

    def __init__(self, polygons, color, dpi):
        try:
            import shapely.geometry as _geom
        except Exception as e:
            raise ImportError(
                "rasterized_polygons requires shapely. Install with: pip install shapely"
            ) from e

        # Normalize input to a flat list of Polygons
        poly_list: List = []
        self._flatten_polygons(polygons, poly_list)

        triangles: List[np.ndarray] = []
        tris_per_poly: List[int] = []
        for poly in poly_list:
            if not isinstance(poly, _geom.Polygon):
                continue

            # For now, only support polygons without holes
            if len(poly.interiors) > 0:
                raise NotImplementedError(
                    "rasterized_polygons currently does not support holes in polygons."
                )

            # Extract exterior coordinates (drop closing duplicate point)
            coords = np.asarray(poly.exterior.coords, dtype=np.float64)
            if coords.shape[0] < 4:
                # fewer than 3 unique vertices
                continue
            coords = coords[:-1, :]  # drop repeated last=first

            # Use raw polygon coordinates; scaling happens later in resolve
            pts = coords.astype(np.float64)
            if pts.shape[0] < 3:
                continue

            tris = _triangulate_polygon_earclip(pts)
            if len(tris) == 0:
                continue

            triangles.extend(tris)
            tris_per_poly.append(len(tris))

        if len(triangles) == 0:
            # Nothing to draw; still create an empty container with proper attributes
            attrib: dict[str, object] = {
                "x": length_params("x", [], CtxLenType.Pos),
                "y": length_params("y", [], CtxLenType.Pos),
                "triangle_count": 0,
                "tris_per_poly": tris_per_poly,
                "color": color_params("color", color),
                "dpi": dpi,
            }
            super().__init__("dapple:rasterized_polygons", attrib)
            return

        triangles_np = np.stack(triangles, axis=0).astype(np.float32)  # (T, 3, 2)
        triangle_count = int(triangles_np.shape[0])
        x_vals = triangles_np[:, :, 0].reshape(-1)
        y_vals = triangles_np[:, :, 1].reshape(-1)

        attrib: dict[str, object] = {
            "x": length_params("x", x_vals, CtxLenType.Pos),
            "y": length_params("y", y_vals, CtxLenType.Pos),
            "triangle_count": triangle_count,
            "tris_per_poly": tris_per_poly,
            "color": color_params("color", color),
            "dpi": dpi,
        }
        super().__init__("dapple:rasterized_polygons", attrib)

    def resolve(self, ctx: ResolveContext) -> Element:
        """
        Resolve by rendering pre-triangulated polygon vertices into a texture and
        creating an image element.
        """
        resolved = super().resolve(ctx)

        x = resolved.attrib["x"]
        y = resolved.attrib["y"]
        triangle_count = resolved.attrib["triangle_count"]
        color = resolved.attrib["color"]
        dpi = resolved.attrib["dpi"]
        tris_per_poly = resolved.attrib.get("tris_per_poly", None)

        assert isinstance(x, AbsLengths)
        assert isinstance(y, AbsLengths)
        if triangle_count <= 0:
            return Element("g")
        assert isinstance(color, Colors)
        assert isinstance(dpi, (int, float))
        assert isinstance(triangle_count, int)

        if triangle_count <= 0:
            return Element("g")

        # Build triangles from resolved absolute coordinates (mm)
        verts = np.column_stack([x.values, y.values]).astype(np.float32)
        if verts.shape[0] != 3 * triangle_count:
            raise ValueError(
                f"Expected {3 * triangle_count} vertices but found {verts.shape[0]}"
            )
        triangles_np = verts.reshape((triangle_count, 3, 2))

        # Compute bounds in mm
        x_min = float(np.min(triangles_np[:, :, 0]))
        x_max = float(np.max(triangles_np[:, :, 0]))
        y_min = float(np.min(triangles_np[:, :, 1]))
        y_max = float(np.max(triangles_np[:, :, 1]))

        width_mm = x_max - x_min
        height_mm = y_max - y_min

        # Guard against zero-width/height ranges used for normalization in the renderer
        if width_mm <= 0.0:
            pad = 0.5  # mm
            x_min -= pad
            x_max += pad
            width_mm = x_max - x_min
        if height_mm <= 0.0:
            pad = 0.5  # mm
            y_min -= pad
            y_max += pad
            height_mm = y_max - y_min

        # Minimum dimensions guard
        width_px, height_px = calculate_dpi_size(
            max(width_mm, 1e-6), max(height_mm, 1e-6), float(dpi)
        )
        width_px = max(width_px, 32)
        height_px = max(height_px, 32)

        # Colors per triangle
        if len(color) == triangle_count:
            per_triangle_colors = color.values.astype(np.float32)
        elif color.isscalar():
            per_triangle_colors = np.repeat(
                color.values.astype(np.float32), triangle_count, axis=0
            )
        elif isinstance(tris_per_poly, list) and len(color) == len(tris_per_poly):
            repeats = np.array(tris_per_poly, dtype=int)
            per_triangle_colors = np.repeat(
                color.values.astype(np.float32), repeats, axis=0
            )
            if per_triangle_colors.shape[0] != triangle_count:
                raise ValueError(
                    "Expanded per-polygon colors did not match triangle count."
                )
        else:
            raise ValueError(
                f"Color length mismatch: expected scalar, {triangle_count} (per triangle), or {len(tris_per_poly) if isinstance(tris_per_poly, list) else 'N'} (per polygon)."
            )

        # Render triangles to texture
        texture_data = render_triangles_to_texture(
            triangles=triangles_np,
            colors=per_triangle_colors,
            width=width_px,
            height=height_px,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
        )

        # Create image element with the rasterized data
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
            # The caller will surface a proper error before getting here normally
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
                RasterizedPolygonsElement._flatten_polygons(g, out)
        else:
            raise TypeError(
                "polygons must be a shapely Polygon, MultiPolygon, or an iterable of those."
            )


def _colors_for_triangles(
    colors: Colors, tris_per_poly: List[int], total_tris: int
) -> np.ndarray:
    """
    Expand provided Colors into a per-triangle RGBA array.

    - If colors is scalar, repeat to all triangles.
    - If colors has length equal to number of polygons, assign triangles from each polygon that color.
    """
    if colors.isscalar():
        rgba = colors.values.astype(np.float32)
        return np.repeat(rgba, total_tris, axis=0)

    n_polys = len(tris_per_poly)
    if len(colors) == n_polys:
        out = np.zeros((total_tris, 4), dtype=np.float32)
        k = 0
        for i, ntri in enumerate(tris_per_poly):
            if ntri == 0:
                continue
            out[k : k + ntri, :] = colors.values[i, :].astype(np.float32)
            k += ntri
        return out

    raise ValueError(
        f"Color length mismatch: expected scalar or {n_polys} colors (one per polygon), got {len(colors)}."
    )


def _triangulate_polygon_earclip(vertices: np.ndarray) -> List[np.ndarray]:
    """
    Triangulate a simple polygon (no holes) using ear clipping.
    vertices: (N, 2) array in CCW or CW order. Returns list of triangles (3, 2).

    This is a straightforward O(N^2) implementation suitable for typical plot polygons.
    """
    # Remove duplicate consecutive points and ensure at least 3 unique vertices
    verts = _remove_duplicate_consecutive(vertices)
    if verts.shape[0] < 3:
        return []

    # Ensure CCW orientation for consistency
    if _signed_area(verts) < 0.0:
        verts = verts[::-1, :]

    n = verts.shape[0]
    V = list(range(n))  # indices into verts
    triangles: List[np.ndarray] = []

    count_guard = 0
    max_iters = 3 * n  # simple guard to avoid infinite loops on degeneracy

    while len(V) > 3 and count_guard < max_iters:
        ear_found = False
        m = len(V)
        for i in range(m):
            i0 = V[(i - 1) % m]
            i1 = V[i]
            i2 = V[(i + 1) % m]

            a = verts[i0]
            b = verts[i1]
            c = verts[i2]

            if not _is_convex(a, b, c):
                continue

            # Check if any other vertex lies in triangle (a,b,c)
            has_inside = False
            for j in range(m):
                idx = V[j]
                if idx in (i0, i1, i2):
                    continue
                p = verts[idx]
                if _point_in_triangle(p, a, b, c):
                    has_inside = True
                    break

            if not has_inside:
                # Ear found
                triangles.append(np.array([a, b, c], dtype=np.float64))
                del V[i]
                ear_found = True
                break

        if not ear_found:
            # Degenerate or self-intersecting polygon; try to fan triangulate remaining as fallback
            # This is a last-resort attempt; may fail for highly degenerate inputs.
            if len(V) >= 3:
                base = V[0]
                for k in range(1, len(V) - 1):
                    triangles.append(
                        np.array(
                            [verts[base], verts[V[k]], verts[V[k + 1]]],
                            dtype=np.float64,
                        )
                    )
                V = []
            break

        count_guard += 1

    if len(V) == 3:
        a = verts[V[0]]
        b = verts[V[1]]
        c = verts[V[2]]
        triangles.append(np.array([a, b, c], dtype=np.float64))

    return triangles


def _remove_duplicate_consecutive(pts: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Remove exact or near-duplicate consecutive points (including closing duplicate)."""
    if pts.shape[0] <= 1:
        return pts
    out = [pts[0]]
    for i in range(1, pts.shape[0]):
        if np.linalg.norm(pts[i] - out[-1]) > tol:
            out.append(pts[i])
    # Also ensure first != last
    if len(out) >= 2 and np.linalg.norm(out[0] - out[-1]) <= tol:
        out.pop()
    return np.asarray(out)


def _signed_area(pts: np.ndarray) -> float:
    """Signed area (positive for CCW)."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _is_convex(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-12) -> bool:
    """Check convexity of angle at b for CCW polygon by cross product sign."""
    ab = b - a
    bc = c - b
    cross = ab[0] * bc[1] - ab[1] * bc[0]
    return cross > eps


def _point_in_triangle(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-12
) -> bool:
    """Barycentric technique to test if point lies inside triangle ABC."""
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < eps:
        return False

    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom

    return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)
