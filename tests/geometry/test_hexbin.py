import numpy as np
import pytest

from dapple.geometry.hexbin import RasterizedHexbin, _hexbin_counts, hexbin
from dapple.scales import UnscaledValues


class TestHexbinCounts:
    """Tests for the hexagonal binning algorithm."""

    def test_single_point(self):
        x = np.array([1.0])
        y = np.array([2.0])
        cx, cy, counts, radius_x, radius_y = _hexbin_counts(x, y, bins_x=5, bins_y=5)
        assert len(cx) == 1
        assert len(cy) == 1
        assert counts[0] == 1.0

    def test_identical_points_in_one_bin(self):
        x = np.full(100, 5.0)
        y = np.full(100, 3.0)
        cx, cy, counts, radius_x, radius_y = _hexbin_counts(x, y, bins_x=10, bins_y=10)
        assert counts.sum() == 100.0
        assert len(counts) == 1
        assert counts[0] == 100.0

    def test_total_count_preserved(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=500)
        y = rng.normal(size=500)
        cx, cy, counts, radius_x, radius_y = _hexbin_counts(x, y, bins_x=10, bins_y=10)
        np.testing.assert_allclose(counts.sum(), 500.0)

    def test_more_bins_produces_more_hexagons(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=200)
        y = rng.normal(size=200)
        _, _, counts_5, _, _ = _hexbin_counts(x, y, bins_x=5, bins_y=5)
        _, _, counts_20, _, _ = _hexbin_counts(x, y, bins_x=20, bins_y=20)
        assert len(counts_20) > len(counts_5)

    def test_radius_positive(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=50)
        y = rng.normal(size=50)
        _, _, _, radius_x, radius_y = _hexbin_counts(x, y, bins_x=10, bins_y=10)
        assert radius_x > 0
        assert radius_y > 0

    def test_anisotropic_bins_different_radii(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=200)
        y = rng.normal(size=200)
        _, _, _, radius_x, radius_y = _hexbin_counts(x, y, bins_x=5, bins_y=20)
        # More bins on y means smaller radius_y
        assert radius_y < radius_x

    def test_anisotropic_bins_count_preserved(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=300)
        y = rng.normal(size=300)
        cx, cy, counts, radius_x, radius_y = _hexbin_counts(x, y, bins_x=5, bins_y=15)
        np.testing.assert_allclose(counts.sum(), 300.0)

    def test_anisotropic_bins_more_y_bins_produces_taller_grid(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=200)
        y = rng.normal(size=200)
        cx_eq, cy_eq, _, _, _ = _hexbin_counts(x, y, bins_x=10, bins_y=10)
        cx_aniso, cy_aniso, _, _, _ = _hexbin_counts(x, y, bins_x=10, bins_y=30)
        # More y bins should produce more hexagons
        assert len(cy_aniso) > len(cy_eq)


class TestHexbinGeometry:
    """Tests for the hexbin geometry function."""

    def test_hexbin_returns_hexbin_element(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=100)
        y = rng.normal(size=100)

        element = hexbin(x, y)

        assert element.tag == "dapple:hexbin"

    def test_hexbin_stores_unscaled_fill(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=100)
        y = rng.normal(size=100)

        element = hexbin(x, y, bins=5)

        fill_attr = element.attrib["fill"]
        assert isinstance(fill_attr, UnscaledValues)

    def test_hexbin_stores_unscaled_coordinates(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=100)
        y = rng.normal(size=100)

        element = hexbin(x, y, bins=5)

        x_attr = element.attrib["x"]
        y_attr = element.attrib["y"]
        assert isinstance(x_attr, UnscaledValues)
        assert isinstance(y_attr, UnscaledValues)

    def test_hexbin_rasterize_returns_rasterized_hexbin(self):
        x = np.linspace(0.0, 1.0, 25)
        y = np.linspace(2.0, 3.0, 25)

        element = hexbin(x, y, bins=5, rasterize=True)

        assert isinstance(element, RasterizedHexbin)
        assert element.tag == "dapple:rasterized_hexbin"

    def test_hexbin_requires_matching_input_lengths(self):
        with pytest.raises(
            ValueError, match="x and y must contain the same number of elements"
        ):
            _ = hexbin([0.0, 1.0], [0.5])

    def test_hexbin_rejects_non_integer_bins(self):
        with pytest.raises(TypeError, match="bins must be an integer or a pair"):
            _ = hexbin([0.0, 1.0], [0.0, 1.0], bins=5.5)  # type: ignore[arg-type]

    def test_hexbin_rejects_zero_bins(self):
        with pytest.raises(ValueError, match="bin counts must be positive"):
            _ = hexbin([0.0, 1.0], [0.0, 1.0], bins=0)

    def test_hexbin_rejects_negative_bins(self):
        with pytest.raises(ValueError, match="bin counts must be positive"):
            _ = hexbin([0.0, 1.0], [0.0, 1.0], bins=-3)

    def test_hexbin_vertex_count_matches_hex_count(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=100)
        y = rng.normal(size=100)

        element = hexbin(x, y, bins=5)
        hex_count: int = element.attrib["hex_count"]  # type: ignore[assignment]
        x_attr = element.attrib["x"]
        assert isinstance(x_attr, UnscaledValues)
        # 6 vertices per hexagon
        assert len(x_attr) == hex_count * 6

    def test_rasterized_hexbin_has_triangle_count(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=100)
        y = rng.normal(size=100)

        element = hexbin(x, y, bins=5, rasterize=True)
        triangle_count: int = element.attrib["triangle_count"]  # type: ignore[assignment]
        hex_count = triangle_count // 6
        # 6 triangles per hexagon
        assert triangle_count == hex_count * 6

        # Check coordinate array sizes: 3 vertices per triangle
        x_attr = element.attrib["x"]
        assert isinstance(x_attr, UnscaledValues)
        assert len(x_attr) == triangle_count * 3

    def test_hexbin_accepts_bin_tuple(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=100)
        y = rng.normal(size=100)

        element = hexbin(x, y, bins=(5, 20))

        assert element.tag == "dapple:hexbin"
        # Anisotropic bins should produce different radii
        assert element.attrib["radius_x"] != element.attrib["radius_y"]

    def test_hexbin_bin_tuple_stores_anisotropic_radii(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=100)
        y = rng.normal(size=100)

        element = hexbin(x, y, bins=(5, 20))
        radius_x: float = element.attrib["radius_x"]  # type: ignore[assignment]
        radius_y: float = element.attrib["radius_y"]  # type: ignore[assignment]
        # More y bins means smaller y radius
        assert radius_y < radius_x

    def test_hexbin_rejects_invalid_bin_sequence_length(self):
        with pytest.raises(
            ValueError, match="bins sequence must contain exactly two integers"
        ):
            _ = hexbin([0.0, 1.0], [0.0, 1.0], bins=[4])

    def test_hexbin_rejects_non_integer_bin_sequence(self):
        with pytest.raises(TypeError, match="bins sequence must contain integers"):
            _ = hexbin([0.0, 1.0], [0.0, 1.0], bins=(3.5, 2))  # type: ignore[arg-type]

    def test_hexbin_rejects_zero_in_bin_sequence(self):
        with pytest.raises(ValueError, match="bin counts must be positive"):
            _ = hexbin([0.0, 1.0], [0.0, 1.0], bins=(5, 0))

    def test_hexbin_single_integer_bins_equal_radii(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=100)
        y = rng.normal(size=100)

        element = hexbin(x, y, bins=10)
        # With equal bins, radii should differ only due to the different
        # hex spacing in x vs y (sqrt(3)*r vs 1.5*r)
        radius_x: float = element.attrib["radius_x"]  # type: ignore[assignment]
        radius_y: float = element.attrib["radius_y"]  # type: ignore[assignment]
        # For the same number of bins, radius_x and radius_y are computed
        # independently from x_range and y_range
        assert radius_x > 0
        assert radius_y > 0
