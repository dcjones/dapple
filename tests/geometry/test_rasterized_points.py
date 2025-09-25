import pytest
import numpy as np
from dapple.geometry.rasterized_points import rasterized_points, RasterizedPointsElement
from dapple.colors import color
from dapple.config import ConfigKey
from dapple.coordinates import (
    AbsCoordSet,
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


class TestRasterizedPoints:
    """Test the rasterized_points geometry function and RasterizedPointsElement class."""

    def test_rasterized_points_creation(self):
        """Test basic creation of rasterized points."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])

        element = rasterized_points(x, y)

        assert isinstance(element, RasterizedPointsElement)
        assert element.tag == "dapple:rasterized_points"

    def test_rasterized_points_with_colors(self):
        """Test rasterized points with color specification."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        colors = ["red", "green", "blue"]

        element = rasterized_points(x, y, color=colors)

        assert isinstance(element, RasterizedPointsElement)
        assert element.attrib["color"] == UnscaledValues("color", colors)

    def test_rasterized_points_with_custom_dpi(self):
        """Test rasterized points with custom DPI."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        custom_dpi = 300.0

        element = rasterized_points(x, y, dpi=custom_dpi)

        assert element.attrib["dpi"] == custom_dpi

    def test_rasterized_points_defaults(self):
        """Test that default parameters are set correctly."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])

        element = rasterized_points(x, y)

        # Check default values
        assert isinstance(element.attrib["color"], ConfigKey)
        assert element.attrib["color"].key == "pointcolor"
        assert isinstance(element.attrib["size"], ConfigKey)
        assert element.attrib["size"].key == "pointsize"
        assert isinstance(element.attrib["dpi"], ConfigKey)
        assert element.attrib["dpi"].key == "rasterize_dpi"


class TestRasterizedPointsElement:
    """Test the RasterizedPointsElement class directly."""

    def create_test_context(self, x, y):
        """Create a test resolve context."""

        xmin = x.min() if len(x) > 0 else 0.0
        ymin = y.min() if len(y) > 0 else 0.0
        xspan = max(x.max() - x.min(), 1.0) if len(x) > 0 else 1.0
        yspan = max(y.max() - y.min(), 1.0) if len(y) > 0 else 1.0

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
        """Test that resolving creates an image element."""
        x = np.array([10.0, 20.0, 30.0])
        y = np.array([10.0, 20.0, 30.0])

        element = RasterizedPointsElement(
            x=x,
            y=y,
            color=color("blue"),
            size=mm(2.0),
            dpi=100.0,
        )

        ctx = self.create_test_context(x, y)
        element.rewrite_attributes_inplace(
            lambda _key, obj: obj.accept_scale(ctx.scales), UnscaledExpr
        )
        resolved = element.resolve(ctx)

        # Should resolve to an image element
        assert isinstance(resolved, Element)
        assert resolved.tag == "image"
        assert "href" in resolved.attrib
        assert "x" in resolved.attrib
        assert "y" in resolved.attrib
        assert "width" in resolved.attrib
        assert "height" in resolved.attrib

    def test_element_resolve_with_vector_data(self):
        """Test resolving with vector coordinate data."""
        # Create multiple points
        n_points = 100
        x = np.random.randn(n_points) * 10 + 50
        y = np.random.randn(n_points) * 10 + 50
        colors_list = ["red", "green", "blue"] * (n_points // 3 + 1)
        colors_list = colors_list[:n_points]

        element = RasterizedPointsElement(
            x=x,
            y=y,
            color=color(colors_list),
            size=mm(1.5),
            dpi=150.0,
        )

        ctx = self.create_test_context(x, y)
        element.rewrite_attributes_inplace(
            lambda _key, obj: obj.accept_scale(ctx.scales), UnscaledExpr
        )
        resolved = element.resolve(ctx)

        # Should successfully resolve to an image
        assert isinstance(resolved, Element)
        assert resolved.tag == "image"
        assert resolved.attrib["href"].startswith("data:image/png;base64,")

    def test_different_dpi_values(self):
        """Test that different DPI values produce different image sizes."""
        x = np.array([10.0, 20.0, 30.0])
        y = np.array([10.0, 20.0, 30.0])

        # Test with low DPI
        element_low = RasterizedPointsElement(
            x=x,
            y=y,
            color=color("red"),
            size=mm(1.0),
            dpi=72.0,
        )

        # Test with high DPI
        element_high = RasterizedPointsElement(
            x=x,
            y=y,
            color=color("red"),
            size=mm(1.0),
            dpi=300.0,
        )

        ctx = self.create_test_context(x, y)
        element_low.rewrite_attributes_inplace(
            lambda _key, obj: obj.accept_scale(ctx.scales), UnscaledExpr
        )
        element_high.rewrite_attributes_inplace(
            lambda _key, obj: obj.accept_scale(ctx.scales), UnscaledExpr
        )
        resolved_low = element_low.resolve(ctx)
        resolved_high = element_high.resolve(ctx)

        # Both should resolve successfully
        assert isinstance(resolved_low, Element)
        assert isinstance(resolved_high, Element)
        assert resolved_low.tag == "image"
        assert resolved_high.tag == "image"

        # The base64 data should be different (different image sizes)
        href_low = resolved_low.attrib["href"]
        href_high = resolved_high.attrib["href"]
        assert href_low != href_high

    def test_single_point(self):
        """Test with a single point."""
        x = np.array([15.0])
        y = np.array([25.0])

        element = RasterizedPointsElement(
            x=x,
            y=y,
            color=color("purple"),
            size=mm(3.0),
            dpi=150.0,
        )

        ctx = self.create_test_context(x, y)
        element.rewrite_attributes_inplace(
            lambda _key, obj: obj.accept_scale(ctx.scales), UnscaledExpr
        )
        resolved = element.resolve(ctx)

        assert isinstance(resolved, Element)
        assert resolved.tag == "image"

    def test_large_dataset(self):
        """Test with a larger dataset to ensure performance is reasonable."""
        # Create a moderately large dataset
        n_points = 10000
        x = np.random.randn(n_points) * 100
        y = np.random.randn(n_points) * 100

        element = RasterizedPointsElement(
            x=x,
            y=y,
            color=color("orange"),
            size=mm(0.5),
            dpi=150.0,
        )

        ctx = self.create_test_context(x, y)
        element.rewrite_attributes_inplace(
            lambda _key, obj: obj.accept_scale(ctx.scales), UnscaledExpr
        )

        # This should complete without errors (performance test)
        import time

        start_time = time.time()
        resolved = element.resolve(ctx)
        end_time = time.time()

        assert isinstance(resolved, Element)
        assert resolved.tag == "image"

        # Should complete in reasonable time (less than 5 seconds)
        assert (end_time - start_time) < 5.0

    def test_empty_arrays(self):
        """Test behavior with empty coordinate arrays."""
        x = np.array([])
        y = np.array([])

        element = RasterizedPointsElement(
            x=x,
            y=y,
            color=color("red"),
            size=mm(1.0),
            dpi=150.0,
        )

        ctx = self.create_test_context(x, y)
        element.rewrite_attributes_inplace(
            lambda _key, obj: obj.accept_scale(ctx.scales), UnscaledExpr
        )

        # Should handle empty arrays gracefully
        try:
            resolved = element.resolve(ctx)
            # If it doesn't raise an exception, it should still create an image
            assert isinstance(resolved, Element)
        except (ValueError, IndexError):
            # It's acceptable to raise an error for empty arrays
            pass


if __name__ == "__main__":
    pytest.main([__file__])
