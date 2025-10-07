import numpy as np

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
