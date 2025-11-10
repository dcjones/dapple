import numpy as np
import pytest

from dapple.colors import Colors, color
from dapple.config import ConfigKey
from dapple.coordinates import (
    AbsCoordTransform,
    AbsLengths,
    CoordSet,
    ResolveContext,
    mm,
)
from dapple.elements import Element
from dapple.geometry.rasterized_segments import (
    RasterizedSegmentsElement,
    rasterized_segments,
)
from dapple.occupancy import Occupancy
from dapple.scales import (
    ScaleSet,
    UnscaledValues,
    xcontinuous,
    ycontinuous,
)


class TestRasterizedSegments:
    """Tests for the rasterized_segments geometry."""

    def _create_context(self, xs, ys) -> ResolveContext:
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)

        if xs.size == 0:
            xmin = xmax = 0.0
        else:
            xmin = float(xs.min())
            xmax = float(xs.max())

        if ys.size == 0:
            ymin = ymax = 0.0
        else:
            ymin = float(ys.min())
            ymax = float(ys.max())

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

    def _scale_position_params(
        self, element: RasterizedSegmentsElement, ctx: ResolveContext
    ) -> RasterizedSegmentsElement:
        element.rewrite_attributes_inplace(
            lambda _k, obj: obj.accept_scale(ctx.scales)
            if isinstance(obj, UnscaledValues) and obj.unit in ("x", "y")
            else obj,
            UnscaledValues,
        )
        return element

    def test_rasterized_segments_creation(self):
        x1 = np.array([0.0, 10.0])
        y1 = np.array([0.0, 5.0])
        x2 = np.array([5.0, 15.0])
        y2 = np.array([5.0, 10.0])

        element = rasterized_segments(x1, y1, x2, y2)

        assert isinstance(element, RasterizedSegmentsElement)
        assert element.tag == "dapple:rasterized_segments"

    def test_rasterized_segments_defaults(self):
        element = rasterized_segments([0, 1], [0, 1], [1, 2], [1, 2])

        assert isinstance(element.attrib["color"], ConfigKey)
        assert element.attrib["color"].key == "linecolor"

        assert isinstance(element.attrib["stroke-width"], ConfigKey)
        assert element.attrib["stroke-width"].key == "linestroke"

        assert isinstance(element.attrib["dpi"], ConfigKey)
        assert element.attrib["dpi"].key == "rasterize_dpi"

    def test_rasterized_segments_custom_parameters(self):
        stroke = mm(0.6)
        dpi = 220.0
        element = rasterized_segments(
            [0, 5],
            [0, 5],
            [5, 10],
            [5, 10],
            color=color("red"),
            stroke_width=stroke,
            dpi=dpi,
        )

        color_attr = element.attrib["color"]
        assert isinstance(color_attr, Colors)
        np.testing.assert_array_equal(
            color_attr.values,
            color("red").values,
        )
        assert element.attrib["stroke-width"] == stroke
        assert element.attrib["dpi"] == dpi

    def test_resolve_creates_image_with_per_segment_colors(self):
        x1 = np.array([0.0, 20.0, 40.0])
        y1 = np.array([0.0, 5.0, 10.0])
        x2 = np.array([10.0, 30.0, 60.0])
        y2 = np.array([20.0, 25.0, 35.0])
        colors = color(["#ff0000", "#00ff00", "#0000ff"])

        element = rasterized_segments(
            x1,
            y1,
            x2,
            y2,
            color=colors,
            stroke_width=mm(1.0),
            dpi=180.0,
        )

        xs = np.concatenate([x1, x2])
        ys = np.concatenate([y1, y2])
        ctx = self._create_context(xs, ys)
        self._scale_position_params(element, ctx)

        resolved = element.resolve(ctx)

        assert isinstance(resolved, Element)
        assert resolved.tag == "image"
        assert isinstance(resolved.attrib["x"], AbsLengths)
        assert isinstance(resolved.attrib["y"], AbsLengths)
        assert isinstance(resolved.attrib["width"], AbsLengths)
        assert isinstance(resolved.attrib["height"], AbsLengths)
        assert resolved.attrib["href"].startswith("data:image/png;base64,")

    def test_scalar_color_broadcast(self):
        x1 = np.array([0.0, 5.0, 10.0])
        y1 = np.array([0.0, 5.0, 10.0])
        x2 = np.array([5.0, 10.0, 15.0])
        y2 = np.array([10.0, 15.0, 20.0])

        element = rasterized_segments(
            x1,
            y1,
            x2,
            y2,
            color=color("blue"),
            stroke_width=mm(0.8),
            dpi=150.0,
        )

        xs = np.concatenate([x1, x2])
        ys = np.concatenate([y1, y2])
        ctx = self._create_context(xs, ys)
        self._scale_position_params(element, ctx)

        resolved = element.resolve(ctx)
        assert isinstance(resolved, Element)
        assert resolved.tag == "image"

    def test_color_length_mismatch_raises(self):
        x1 = np.array([0.0, 5.0, 10.0])
        y1 = np.array([0.0, 5.0, 10.0])
        x2 = np.array([5.0, 10.0, 15.0])
        y2 = np.array([10.0, 15.0, 20.0])
        colors = color(["#ff0000", "#00ff00"])  # only two colors for three segments

        element = rasterized_segments(
            x1,
            y1,
            x2,
            y2,
            color=colors,
            stroke_width=mm(0.5),
            dpi=150.0,
        )

        xs = np.concatenate([x1, x2])
        ys = np.concatenate([y1, y2])
        ctx = self._create_context(xs, ys)
        self._scale_position_params(element, ctx)

        with pytest.raises(ValueError):
            element.resolve(ctx)

    def test_degenerate_segments_returns_empty_group(self):
        x = np.array([0.0, 10.0, 20.0])
        y = np.array([0.0, 10.0, 20.0])

        element = rasterized_segments(
            x,
            y,
            x,
            y,
            color=color("black"),
            stroke_width=mm(0.5),
            dpi=150.0,
        )

        xs = np.concatenate([x, x])
        ys = np.concatenate([y, y])
        ctx = self._create_context(xs, ys)
        self._scale_position_params(element, ctx)

        resolved = element.resolve(ctx)
        assert isinstance(resolved, Element)
        assert resolved.tag == "g"

    def test_different_dpi_affects_output(self):
        x1 = np.array([0.0, 20.0])
        y1 = np.array([0.0, 10.0])
        x2 = np.array([30.0, 40.0])
        y2 = np.array([25.0, 35.0])

        element_low = rasterized_segments(
            x1,
            y1,
            x2,
            y2,
            color=color("purple"),
            stroke_width=mm(1.2),
            dpi=72.0,
        )

        element_high = rasterized_segments(
            x1,
            y1,
            x2,
            y2,
            color=color("purple"),
            stroke_width=mm(1.2),
            dpi=300.0,
        )

        xs = np.concatenate([x1, x2])
        ys = np.concatenate([y1, y2])
        ctx = self._create_context(xs, ys)

        self._scale_position_params(element_low, ctx)
        self._scale_position_params(element_high, ctx)

        resolved_low = element_low.resolve(ctx)
        resolved_high = element_high.resolve(ctx)

        assert isinstance(resolved_low, Element)
        assert isinstance(resolved_high, Element)
        assert resolved_low.tag == "image"
        assert resolved_high.tag == "image"
        assert resolved_low.attrib["href"] != resolved_high.attrib["href"]


if __name__ == "__main__":
    pytest.main([__file__])
