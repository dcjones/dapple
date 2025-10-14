import pytest
import numpy as np

import shapely.geometry as geom

from dapple.geometry.rasterized_polygons import (
    rasterized_polygons,
    RasterizedPolygonsElement,
)
from dapple.colors import color
from dapple.config import ConfigKey
from dapple.coordinates import (
    AbsCoordTransform,
    mm,
    ResolveContext,
    CoordSet,
    AbsLengths,
)
from dapple.scales import (
    ScaleSet,
    UnscaledExpr,
    UnscaledValues,
    xcontinuous,
    ycontinuous,
)
from dapple.occupancy import Occupancy
from dapple.elements import Element


class TestRasterizedPolygonsCreation:
    def test_rasterized_polygons_creation(self):
        rect1 = geom.box(0, 0, 10, 10)
        rect2 = geom.box(15, 5, 25, 20)

        element = rasterized_polygons([rect1, rect2])

        assert isinstance(element, RasterizedPolygonsElement)
        assert element.tag == "dapple:rasterized_polygons"

    def test_rasterized_polygons_with_colors(self):
        rects = [geom.box(0, 0, 10, 10), geom.box(20, 0, 30, 15)]
        colors = ["red", "green"]

        element = rasterized_polygons(rects, color=colors)

        # color should be stored as UnscaledValues for the scaling pipeline
        assert element.attrib["color"] == UnscaledValues("color", colors)

    def test_rasterized_polygons_with_custom_dpi(self):
        rects = [geom.box(0, 0, 10, 10)]
        custom_dpi = 300.0

        element = rasterized_polygons(rects, dpi=custom_dpi)

        assert element.attrib["dpi"] == custom_dpi

    def test_rasterized_polygons_defaults(self):
        rects = [geom.box(0, 0, 10, 10)]

        element = rasterized_polygons(rects)

        assert isinstance(element.attrib["color"], ConfigKey)
        assert element.attrib["color"].key == "linecolor"
        assert isinstance(element.attrib["dpi"], ConfigKey)
        assert element.attrib["dpi"].key == "rasterize_dpi"


class TestRasterizedPolygonsResolve:
    def _create_test_context_from_geoms(self, geoms):
        """
        Create a ResolveContext that maps data bounds to a 100x100 mm viewport.
        """
        # Flatten possible MultiPolygons
        polys = []
        for g in geoms:
            if isinstance(g, geom.MultiPolygon):
                polys.extend(list(g.geoms))
            else:
                polys.append(g)

        if len(polys) == 0:
            xmin = ymin = 0.0
            xmax = ymax = 1.0
        else:
            # Compute aggregate bounds
            xs_min = []
            ys_min = []
            xs_max = []
            ys_max = []
            for p in polys:
                bxmin, bymin, bxmax, bymax = p.bounds
                xs_min.append(bxmin)
                ys_min.append(bymin)
                xs_max.append(bxmax)
                ys_max.append(bymax)
            xmin = float(min(xs_min))
            ymin = float(min(ys_min))
            xmax = float(max(xs_max))
            ymax = float(max(ys_max))

        xspan = max(xmax - xmin, 1.0)
        yspan = max(ymax - ymin, 1.0)

        coords = CoordSet(
            {
                "x": AbsCoordTransform(100.0 / xspan, xmin),
                "y": AbsCoordTransform(100.0 / yspan, ymin),
            }
        )
        scales = ScaleSet(
            {
                "x": xcontinuous(),
                "y": ycontinuous(),
            }
        )
        occupancy = Occupancy(mm(100), mm(100))
        return ResolveContext(coords, scales, occupancy)

    def test_element_resolve_creates_image(self):
        rect1 = geom.box(10, 10, 60, 40)
        rect2 = geom.box(30, 20, 45, 55)

        element = rasterized_polygons(
            [rect1, rect2], color=color(["#66c2a5", "#fc8d62"]), dpi=150.0
        )

        ctx = self._create_test_context_from_geoms([rect1, rect2])

        # Apply scales to UnscaledValues before resolve, matching project pattern
        element.rewrite_attributes_inplace(
            lambda _key, obj: obj.accept_scale(ctx.scales)
            if obj.unit in ("x", "y")
            else obj,
            UnscaledValues,
        )

        resolved = element.resolve(ctx)

        assert isinstance(resolved, Element)
        assert resolved.tag == "image"
        assert "href" in resolved.attrib
        assert "x" in resolved.attrib
        assert "y" in resolved.attrib
        assert "width" in resolved.attrib
        assert "height" in resolved.attrib
        assert isinstance(resolved.attrib["x"], AbsLengths)
        assert isinstance(resolved.attrib["y"], AbsLengths)
        assert isinstance(resolved.attrib["width"], AbsLengths)
        assert isinstance(resolved.attrib["height"], AbsLengths)
        assert resolved.attrib["href"].startswith("data:image/png;base64,")

    def test_different_dpi_values(self):
        rects = [geom.box(10, 10, 60, 40)]
        element_low = rasterized_polygons(rects, color=color("red"), dpi=72.0)
        element_high = rasterized_polygons(rects, color=color("red"), dpi=300.0)

        ctx = self._create_test_context_from_geoms(rects)
        element_low.rewrite_attributes_inplace(
            lambda _key, obj: obj.accept_scale(ctx.scales)
            if obj.unit in ("x", "y")
            else obj,
            UnscaledValues,
        )
        element_high.rewrite_attributes_inplace(
            lambda _key, obj: obj.accept_scale(ctx.scales)
            if obj.unit in ("x", "y")
            else obj,
            UnscaledValues,
        )

        resolved_low = element_low.resolve(ctx)
        resolved_high = element_high.resolve(ctx)

        assert isinstance(resolved_low, Element)
        assert isinstance(resolved_high, Element)
        assert resolved_low.tag == "image"
        assert resolved_high.tag == "image"

        # Different DPI should yield different encoded image
        href_low = resolved_low.attrib["href"]
        href_high = resolved_high.attrib["href"]
        assert href_low != href_high

    def test_empty_input(self):
        element = rasterized_polygons([])

        ctx = self._create_test_context_from_geoms([])
        element.rewrite_attributes_inplace(
            lambda _key, obj: obj.accept_scale(ctx.scales)
            if obj.unit in ("x", "y")
            else obj,
            UnscaledValues,
        )
        resolved = element.resolve(ctx)

        # With no triangles, it should return an empty group
        assert isinstance(resolved, Element)
        assert resolved.tag == "g"

    def test_multipolygon_input(self):
        poly1 = geom.box(0, 0, 20, 20)
        poly2 = geom.box(25, 5, 40, 15)
        mp = geom.MultiPolygon([poly1, poly2])

        element = rasterized_polygons(
            mp, color=color(["#1b9e77", "#d95f02"]), dpi=150.0
        )

        ctx = self._create_test_context_from_geoms([mp])
        element.rewrite_attributes_inplace(
            lambda _key, obj: obj.accept_scale(ctx.scales)
            if obj.unit in ("x", "y")
            else obj,
            UnscaledValues,
        )
        resolved = element.resolve(ctx)

        assert isinstance(resolved, Element)
        assert resolved.tag == "image"
        assert resolved.attrib["href"].startswith("data:image/png;base64,")

    def test_polygon_with_hole_works(self):
        # A square with a hole inside - should work with mapbox-earcut
        shell = [(0, 0), (50, 0), (50, 50), (0, 50), (0, 0)]
        hole = [(10, 10), (40, 10), (40, 40), (10, 40), (10, 10)]
        p = geom.Polygon(shell, holes=[hole])

        # Should not raise - holes are now supported
        element = rasterized_polygons([p], color=color("blue"), dpi=150.0)

        # Verify element was created
        assert element.tag == "dapple:rasterized_polygons"
        assert "x" in element.attrib
        assert "y" in element.attrib

        # Verify it resolves to an image
        ctx = self._create_test_context_from_geoms([p])

        # Apply scales to UnscaledValues before resolve, matching project pattern
        element.rewrite_attributes_inplace(
            lambda _key, obj: obj.accept_scale(ctx.scales)
            if obj.unit in ("x", "y")
            else obj,
            UnscaledValues,
        )

        resolved = element.resolve(ctx)
        assert resolved.tag == "image"

    def test_color_length_mismatch_raises_on_resolve(self):
        # Pass 3 colors for 2 polygons -> should error at resolve time
        rect1 = geom.box(0, 0, 10, 10)
        rect2 = geom.box(20, 0, 30, 10)
        colors = ["#ff0000", "#00ff00", "#0000ff"]

        element = rasterized_polygons([rect1, rect2], color=color(colors), dpi=150.0)

        ctx = self._create_test_context_from_geoms([rect1, rect2])
        element.rewrite_attributes_inplace(
            lambda _key, obj: obj.accept_scale(ctx.scales)
            if obj.unit in ("x", "y")
            else obj,
            UnscaledValues,
        )

        with pytest.raises(ValueError):
            _ = element.resolve(ctx)


if __name__ == "__main__":
    pytest.main([__file__])
