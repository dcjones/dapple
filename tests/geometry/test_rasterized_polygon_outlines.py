import pytest
import numpy as np

import shapely.geometry as geom

from dapple.geometry.rasterized_polygon_outlines import (
    rasterized_polygon_outlines,
    RasterizedPolygonOutlinesElement,
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
    UnscaledValues,
    xcontinuous,
    ycontinuous,
)
from dapple.occupancy import Occupancy
from dapple.elements import Element


class TestRasterizedPolygonOutlinesCreation:
    def test_rasterized_polygon_outlines_creation(self):
        rect1 = geom.box(0, 0, 10, 10)
        rect2 = geom.box(15, 5, 25, 20)

        element = rasterized_polygon_outlines([rect1, rect2])

        assert isinstance(element, RasterizedPolygonOutlinesElement)
        assert element.tag == "dapple:rasterized_polygon_outlines"

    def test_rasterized_polygon_outlines_defaults(self):
        rects = [geom.box(0, 0, 10, 10)]
        element = rasterized_polygon_outlines(rects)

        # Defaults should be config keys
        assert isinstance(element.attrib["color"], ConfigKey)
        assert element.attrib["color"].key == "linecolor"

        assert "stroke-width" in element.attrib
        assert isinstance(element.attrib["stroke-width"], ConfigKey)
        assert element.attrib["stroke-width"].key == "linestroke"

        assert isinstance(element.attrib["dpi"], ConfigKey)
        assert element.attrib["dpi"].key == "rasterize_dpi"

    def test_rasterized_polygon_outlines_with_custom_params(self):
        rects = [geom.box(0, 0, 10, 10)]
        element = rasterized_polygon_outlines(
            rects, color=color("red"), stroke_width=mm(0.8), dpi=200.0
        )

        assert element.attrib["dpi"] == 200.0
        assert element.attrib["stroke-width"] == mm(0.8)


class TestRasterizedPolygonOutlinesResolve:
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
        scales = ScaleSet({"x": xcontinuous(), "y": ycontinuous()})
        occupancy = Occupancy(mm(100), mm(100))
        return ResolveContext(coords, scales, occupancy)

    def _scale_xy_only(self, element, ctx):
        # Scale only length UnscaledValues (x and y), leave color alone
        element.rewrite_attributes_inplace(
            lambda _k, obj: obj.accept_scale(ctx.scales)
            if isinstance(obj, UnscaledValues) and obj.unit in ("x", "y")
            else obj,
            UnscaledValues,
        )
        return element

    def test_resolve_per_polygon_colors(self):
        rect1 = geom.box(10, 10, 60, 40)
        rect2 = geom.box(30, 20, 45, 55)

        element = rasterized_polygon_outlines(
            [rect1, rect2],
            color=color(["#66c2a5", "#fc8d62"]),
            stroke_width=mm(0.6),
            dpi=150.0,
        )

        ctx = self._create_test_context_from_geoms([rect1, rect2])
        self._scale_xy_only(element, ctx)
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

    def test_dpi_variation_changes_output(self):
        rect = geom.box(10, 10, 60, 40)

        e_low = rasterized_polygon_outlines(
            [rect], color=color("red"), stroke_width=mm(1.0), dpi=72.0
        )
        e_high = rasterized_polygon_outlines(
            [rect], color=color("red"), stroke_width=mm(1.0), dpi=300.0
        )

        ctx = self._create_test_context_from_geoms([rect])
        self._scale_xy_only(e_low, ctx)
        self._scale_xy_only(e_high, ctx)

        r_low = e_low.resolve(ctx)
        r_high = e_high.resolve(ctx)

        assert isinstance(r_low, Element) and r_low.tag == "image"
        assert isinstance(r_high, Element) and r_high.tag == "image"

        assert r_low.attrib["href"] != r_high.attrib["href"]

    def test_stroke_width_affects_output(self):
        rect = geom.box(0, 0, 50, 50)

        # Same DPI, different stroke widths should alter the output
        e_thin = rasterized_polygon_outlines(
            [rect], color=color("#333333"), stroke_width=mm(0.2), dpi=150.0
        )
        e_thick = rasterized_polygon_outlines(
            [rect], color=color("#333333"), stroke_width=mm(2.0), dpi=150.0
        )

        ctx = self._create_test_context_from_geoms([rect])
        self._scale_xy_only(e_thin, ctx)
        self._scale_xy_only(e_thick, ctx)

        r_thin = e_thin.resolve(ctx)
        r_thick = e_thick.resolve(ctx)

        assert isinstance(r_thin, Element) and r_thin.tag == "image"
        assert isinstance(r_thick, Element) and r_thick.tag == "image"

        assert r_thin.attrib["href"] != r_thick.attrib["href"]

    def test_empty_input(self):
        element = rasterized_polygon_outlines([])

        ctx = self._create_test_context_from_geoms([])
        self._scale_xy_only(element, ctx)

        resolved = element.resolve(ctx)
        assert isinstance(resolved, Element)
        assert resolved.tag == "g"

    def test_polygon_with_hole_is_supported(self):
        shell = [(0, 0), (50, 0), (50, 50), (0, 50), (0, 0)]
        hole = [(10, 10), (40, 10), (40, 40), (10, 40), (10, 10)]
        p = geom.Polygon(shell, holes=[hole])

        element = rasterized_polygon_outlines(
            [p], color=color("black"), stroke_width=mm(0.7), dpi=150.0
        )

        ctx = self._create_test_context_from_geoms([p])
        self._scale_xy_only(element, ctx)
        resolved = element.resolve(ctx)

        assert isinstance(resolved, Element)
        assert resolved.tag == "image"
        assert resolved.attrib["href"].startswith("data:image/png;base64,")

    def test_color_length_mismatch_raises_on_resolve(self):
        rect1 = geom.box(0, 0, 10, 10)
        rect2 = geom.box(20, 0, 30, 10)
        # 3 colors for 2 polygons should mismatch after expansion
        colors = ["#ff0000", "#00ff00", "#0000ff"]

        element = rasterized_polygon_outlines(
            [rect1, rect2], color=color(colors), stroke_width=mm(1.0), dpi=150.0
        )

        ctx = self._create_test_context_from_geoms([rect1, rect2])
        self._scale_xy_only(element, ctx)

        with pytest.raises(ValueError):
            _ = element.resolve(ctx)


if __name__ == "__main__":
    pytest.main([__file__])
