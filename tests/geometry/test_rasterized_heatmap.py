import numpy as np
import pytest

from dapple import inch, plot
from dapple.geometry import rasterized_heatmap
from dapple.geometry.rasterized_heatmap import RasterizedHeatmap


class TestRasterizedHeatmap:
    def test_rasterized_heatmap_creation(self):
        m, n = (5, 10)
        color = np.random.randn(m, n)
        x = np.arange(n)
        y = np.arange(m)

        el = rasterized_heatmap(color, x, y)
        assert isinstance(el, RasterizedHeatmap)
        assert el.tag == "dapple:rasterized_heatmap"

    def test_render_heatmap(self):
        m, n = (5, 10)
        color = np.random.randn(m, n)
        x = np.arange(n)
        y = np.arange(m)

        el = rasterized_heatmap(color, x, y)
        plot(el).svg(4 * inch, 4 * inch)
