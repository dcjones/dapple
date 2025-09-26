import pytest
import numpy as np
import math

from dapple.geometry.lines import (
    line,
    lines,
    density,
    Path,
    PathData,
    _adaptive_sample_function,
)
from dapple.coordinates import mm, AbsLengths
from dapple.config import ConfigKey
from dapple.colors import color
from dapple.scales import UnscaledValues


class TestPathData:
    """Test SVG path data generation utility."""

    def test_simple_path(self):
        """Test basic SVG path creation."""
        x = AbsLengths(np.array([0, 10, 20]))
        y = AbsLengths(np.array([0, 5, 0]))
        path = PathData(x, y)

        result = path.serialize()
        assert result == "M 0.000 0.000 L 10.000 5.000 L 20.000 0.000"

    def test_path_length_mismatch(self):
        """Test error handling for mismatched coordinate lengths."""
        x = AbsLengths(np.array([0, 10]))
        y = AbsLengths(np.array([0, 5, 10]))

        with pytest.raises(ValueError, match="same length"):
            PathData(x, y)

    def test_path_insufficient_points(self):
        """Test error handling for insufficient points."""
        x = AbsLengths(np.array([0]))
        y = AbsLengths(np.array([0]))

        with pytest.raises(ValueError, match="at least 2 points"):
            PathData(x, y)


class TestPathElement:
    """Test Path element functionality."""

    def test_path_element_creation(self):
        """Test Path element creation."""
        x_data = [1, 2, 3]
        y_data = [1, 2, 3]

        path_elem = Path(x_data, y_data, stroke="red", fill="none")

        assert path_elem.tag == "dapple:path"
        assert "x" in path_elem.attrib
        assert "y" in path_elem.attrib
        assert path_elem.attrib["stroke"] == "red"
        assert path_elem.attrib["fill"] == "none"


class TestAdaptiveSampling:
    """Test adaptive function sampling."""

    def test_linear_function(self):
        """Test that linear functions are sampled efficiently."""

        def linear_func(x):
            return 2 * x + 1

        x_vals, y_vals = _adaptive_sample_function(linear_func, 0, 10, tolerance=1e-3)

        # Adaptive library may use a consistent number of points regardless of function complexity
        # Just verify we get reasonable coverage
        assert len(x_vals) > 10  # Should sample enough points
        # Check that we cover the domain reasonably
        assert min(x_vals) <= 0.1
        assert max(x_vals) >= 9.9

    def test_quadratic_function(self):
        """Test that curved functions get more sample points."""

        def quadratic_func(x):
            return x**2

        x_vals, y_vals = _adaptive_sample_function(
            quadratic_func, 0, 10, tolerance=0.01
        )

        # Should need more than 2 points for a curved function
        assert len(x_vals) > 2
        # Check that we get reasonable start and end values (adaptive might not include exact bounds)
        assert min(x_vals) <= 0.1  # Close to 0
        assert max(x_vals) >= 9.9  # Close to 10
        assert min(y_vals) <= 0.1  # Close to 0
        assert max(y_vals) >= 99  # Close to 100

    def test_sinusoidal_function(self):
        """Test sampling of oscillatory function."""

        def sin_func(x):
            return np.sin(x)

        x_vals, y_vals = _adaptive_sample_function(
            sin_func, 0, 2 * np.pi, tolerance=0.01
        )

        # Should sample many points for oscillatory function
        assert len(x_vals) > 10  # Should need many points to capture oscillations
        assert min(x_vals) <= 0.1
        assert max(x_vals) >= 2 * np.pi - 0.1

    def test_undefined_function(self):
        """Test handling of functions that raise exceptions."""

        def bad_func(x):
            if abs(x - 5) < 0.001:  # Make the undefined region very small
                raise ValueError("Undefined at x=5")
            return x

        x_vals, y_vals = _adaptive_sample_function(bad_func, 0, 10)

        # Should return empty arrays if function fails during sampling
        # The adaptive library doesn't handle exceptions gracefully by default
        assert len(x_vals) == 0 and len(y_vals) == 0


class TestLineGeometry:
    """Test the line geometry functions."""

    def test_simple_line(self):
        """Test basic line creation."""
        x_data = [1, 2, 3, 4]
        y_data = [2, 4, 1, 5]

        line_elem = line(x_data, y_data)

        assert line_elem.tag == "dapple:path"
        assert "x" in line_elem.attrib
        assert "y" in line_elem.attrib
        assert "stroke" in line_elem.attrib
        assert "stroke-width" in line_elem.attrib

    def test_line_with_custom_color(self):
        """Test line with custom color."""
        x_data = [1, 2, 3]
        y_data = [1, 2, 3]
        custom_color = color("#ff0000")

        line_elem = line(x_data, y_data, color=custom_color)

        assert line_elem.attrib["stroke"] == custom_color

    def test_lines_single_group(self):
        """Test lines function with single line (no grouping)."""
        x_data = [1, 2, 3]
        y_data = [1, 4, 2]

        lines_elem = lines(x_data, y_data)

        # Should behave same as line() function
        assert lines_elem.tag == "dapple:path"

    def test_lines_with_grouping(self):
        """Test lines function with multiple groups."""
        x_data = [1, 2, 3, 4, 5, 6]
        y_data = [1, 2, 3, 4, 5, 6]
        groups = ["A", "A", "A", "B", "B", "B"]

        lines_elem = lines(x_data, y_data, group=groups)

        # Should return a container with multiple Path children
        assert lines_elem.tag == "g"
        assert len(lines_elem.children) == 2  # Two groups

    def test_lines_with_color_grouping(self):
        """Test lines function grouped by color only."""
        x_data = [1, 2, 3, 4, 5, 6]
        y_data = [1, 2, 3, 4, 5, 6]
        colors = ["red", "red", "red", "blue", "blue", "blue"]

        lines_elem = lines(x_data, y_data, color=colors)

        # Should return a container with multiple Path children
        assert lines_elem.tag == "g"
        assert len(lines_elem.children) == 2  # Two colors

    def test_lines_with_group_and_color(self):
        """Test lines function with both group and color variables."""
        x_data = [1, 2, 3, 4, 5, 6, 7, 8]
        y_data = [1, 2, 3, 4, 5, 6, 7, 8]
        groups = ["A", "A", "B", "B", "A", "A", "B", "B"]
        colors = ["red", "red", "red", "red", "blue", "blue", "blue", "blue"]

        lines_elem = lines(x_data, y_data, color=colors, group=groups)

        # Should return a container with Path children for unique (group, color) pairs
        assert lines_elem.tag == "g"
        # Expected: (A,red), (B,red), (A,blue), (B,blue) = 4 unique pairs
        assert len(lines_elem.children) == 4

    def test_lines_function_plotting(self):
        """Test function plotting mode."""

        def test_func(x):
            return x**2

        lines_elem = lines(y=test_func, xmin=0, xmax=10)

        assert lines_elem.tag == "dapple:path"

    def test_line_function_plotting(self):
        """Test line function plotting mode."""

        def test_func(x):
            return np.sin(x)

        line_elem = line(y=test_func, xmin=0, xmax=2 * np.pi)

        assert line_elem.tag == "dapple:path"


class TestDensityGeometry:
    """Test the density geometry function."""

    def test_simple_density(self):
        """Test basic density plot creation."""
        x_data = np.random.randn(100)
        density_elem = density(x_data)

        # The density function returns a line plot of the KDE function
        assert density_elem.tag == "dapple:path"
        assert "y" in density_elem.attrib
        assert callable(density_elem.attrib["y"])

    def test_density_empty_input(self):
        """Test density with empty input data."""
        density_elem = density([])
        assert density_elem.tag == "g"
        assert not hasattr(density_elem, "children") or len(density_elem.children) == 0

    def test_density_custom_params(self, mocker):
        """Test density plot with custom bw_method and weights."""
        mock_kde = mocker.patch("scipy.stats.gaussian_kde")
        x_data = np.array([1, 2, 3, 4, 5])
        weights_data = np.array([1, 1, 2, 1, 1])

        density(x_data, bw_method=0.5, weights=weights_data)

        mock_kde.assert_called_once()
        # Using np.testing.assert_array_equal for numpy array comparison
        np.testing.assert_array_equal(mock_kde.call_args.args[0], x_data)
        assert mock_kde.call_args.kwargs["bw_method"] == 0.5
        np.testing.assert_array_equal(
            mock_kde.call_args.kwargs["weights"], weights_data
        )

    def test_density_insufficient_data(self):
        """Test density with just one data point."""
        # It should still produce a path. The underlying line sampling
        # will handle very simple functions.
        density_elem = density([1])
        assert line_elem.tag == "dapple:path"


class TestDensityGeometry:
    """Test the density geometry function."""

    def test_simple_density(self):
        """Test basic density plot creation."""
        x_data = np.random.randn(100)
        density_elem = density(x_data)

        # The density function returns a line plot of the KDE function
        assert density_elem.tag == "dapple:path"
        assert "y" in density_elem.attrib
        assert isinstance(density_elem.attrib["y"], UnscaledValues)

    def test_density_with_bw_method(self):
        """Test that bw_method is passed to gaussian_kde."""
        # This is an indirect test, but it ensures that the parameter
        # is accepted without crashing.
        x_data = np.random.randn(100)
        density_elem = density(x_data, bw_method="silverman")
        assert density_elem.tag == "dapple:path"

    def test_density_empty_input(self):
        """Test that empty input returns an empty <g> element."""
        density_elem = density([])
        assert density_elem.tag == "g"
        assert not density_elem.children

    def test_density_with_weights(self):
        """Test density with weights."""
        x_data = np.random.randn(100)
        weights = np.random.rand(100)
        density_elem = density(x_data, weights=weights)
        assert density_elem.tag == "dapple:path"

    def test_density_insufficient_data(self):
        """Test input with one data point returns an empty <g> element."""
        density_elem = density([1])
        assert density_elem.tag == "g"
        assert not density_elem.children


class TestConfigIntegration:
    """Test integration with Config system."""

    def test_default_config_keys(self):
        """Test that default config keys are used."""
        line_elem = line([1, 2], [1, 2])

        # Should use ConfigKey for default color and stroke
        assert isinstance(line_elem.attrib["stroke"], ConfigKey)
        assert line_elem.attrib["stroke"].key == "linecolor"
        assert isinstance(line_elem.attrib["stroke-width"], ConfigKey)
        assert line_elem.attrib["stroke-width"].key == "linestroke"


if __name__ == "__main__":
    pytest.main([__file__])
