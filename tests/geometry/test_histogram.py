import io
import re

import numpy as np

from dapple import plot
from dapple.coordinates import CoordBounds, CtxLenType, ctxlengths, mm
from dapple.geometry.histogram import histogram
from dapple.geometry.bars import Bar
from dapple.config import ConfigKey
from dapple.colors import color


class TestHistogramGeometry:
    def test_histogram_matches_numpy_defaults(self):
        data = np.array([0.5, 1.0, 1.5, 2.0, 3.25, 3.75, 4.0], dtype=float)

        bars = histogram(data)

        assert isinstance(bars, Bar)

        expected_counts, expected_edges = np.histogram(data)

        heights = np.asarray(bars.attrib["height"].values)
        left_edges = np.asarray(bars.attrib["x"].values)

        np.testing.assert_array_equal(heights, expected_counts)
        np.testing.assert_allclose(left_edges, expected_edges[:-1])

    def test_histogram_with_explicit_bins_sequence(self):
        data = np.array([0.2, 0.8, 1.3, 2.7, 2.9, 3.4, 3.6], dtype=float)
        bin_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

        bars = histogram(data, bins=bin_edges)
        expected_counts, _ = np.histogram(data, bins=bin_edges)

        heights = np.asarray(bars.attrib["height"].values)
        left_edges = np.asarray(bars.attrib["x"].values)

        np.testing.assert_array_equal(heights, expected_counts)
        np.testing.assert_allclose(left_edges, bin_edges[:-1])

    def test_histogram_with_density_and_weights(self):
        data = np.array([0.0, 0.5, 1.0, 1.5, 1.5, 2.0, 2.5, 3.0], dtype=float)
        weights = np.array([1, 2, 1, 2, 1, 3, 1, 1], dtype=float)

        bars = histogram(data, bins=4, density=True, weights=weights)
        expected_counts, _ = np.histogram(data, bins=4, density=True, weights=weights)

        heights = np.asarray(bars.attrib["height"].values, dtype=float)

        np.testing.assert_allclose(heights, expected_counts)

    def test_histogram_supports_string_bins(self):
        data = np.linspace(-2.0, 2.0, 20)

        bars = histogram(data, bins="fd")
        expected_counts, expected_edges = np.histogram(data, bins="fd")

        heights = np.asarray(bars.attrib["height"].values)
        left_edges = np.asarray(bars.attrib["x"].values)

        np.testing.assert_array_equal(heights, expected_counts)
        np.testing.assert_allclose(left_edges, expected_edges[:-1])

    def test_histogram_color_configuration(self):
        data = np.random.default_rng(1).normal(size=30)

        bars = histogram(data, bins=5, color=color("#ff5733"))

        fill = bars.attrib["fill"]
        assert fill is not None
        serialized = fill.serialize()
        assert isinstance(serialized, str)
        assert serialized.lower() == "#ff5733"

    def test_histogram_uses_default_barcolor_when_color_unspecified(self):
        data = np.random.default_rng(2).normal(size=30)

        bars = histogram(data, bins=5)

        fill = bars.attrib["fill"]
        assert isinstance(fill, ConfigKey)
        assert fill.key == "barcolor"


class TestHistogramBounds:
    def test_bounds_stay_within_bin_edges(self):
        """
        Bars built from explicit ranges must not produce coordinate bounds
        outside the bin edges. A width expressed as `xmax - xmin` used to be
        bounded by `min(xmax) - max(xmin)`, which pulled the x bounds far to
        the left of the data and pushed the origin into the middle of the plot.
        """
        rng = np.random.default_rng(0)
        data = np.exp(rng.standard_normal(1000))
        counts, edges = np.histogram(data)

        bars = Bar(
            x=ctxlengths(edges[:-1], "x", CtxLenType.Pos),
            y=ctxlengths(np.zeros(len(counts)), "y", CtxLenType.Pos),
            x1=ctxlengths(edges[1:], "x", CtxLenType.Pos),
            height=ctxlengths(counts.astype(float), "y", CtxLenType.Vec),
        )

        bounds = CoordBounds()
        bars.update_bounds(bounds)

        xs = [c.x for c in bounds.constraints if c.x is not None]
        assert xs
        assert min(xs) >= edges[0] - 1e-9
        assert max(xs) <= edges[-1] + 1e-9

    def test_resolved_bars_span_bin_edges(self):
        data = np.array([0.5, 1.0, 1.5, 2.0, 3.25, 3.75, 4.0], dtype=float)
        _, edges = np.histogram(data)

        pl = plot(histogram(data))
        out = io.StringIO()
        pl.svg(mm(100), mm(60), out)
        svg = out.getvalue()

        rect_attrs = re.findall(r"<rect([^>]*)/>", svg)
        assert rect_attrs
        assert not any("x1=" in attrs or "y1=" in attrs for attrs in rect_attrs)

        rects = [
            {k: float(v) for k, v in re.findall(r'(x|width)="([-0-9.]+)"', attrs)}
            for attrs in rect_attrs
        ]
        rects = [r for r in rects if "x" in r and "width" in r]
        assert len(rects) == len(edges) - 1

        widths = [r["width"] for r in rects]
        np.testing.assert_allclose(widths, widths[0], rtol=1e-3)

        span = max(r["x"] + r["width"] for r in rects) - min(r["x"] for r in rects)
        np.testing.assert_allclose(span, widths[0] * len(rects), rtol=1e-3)
