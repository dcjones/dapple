import pytest
import numpy as np
import base64
from io import BytesIO
from PIL import Image

from dapple.geometry.image import image, ImageElement
from dapple.coordinates import mm, CtxLenType, ResolveContext, CoordSet, AbsCoordSet
from dapple.scales import ScaleSet, length_params
from dapple.occupancy import Occupancy


class TestImageGeometry:
    """Test suite for image geometry functionality."""

    def test_image_function_returns_image_element(self):
        """Test that image function returns an ImageElement."""
        data = np.zeros((10, 10), dtype=np.uint8)
        result = image(mm(0), mm(0), mm(10), mm(10), data)
        assert isinstance(result, ImageElement)
        assert result.tag == "image"

    def test_grayscale_image_2d_array(self):
        """Test creating image from 2D grayscale numpy array."""
        # Create a simple 3x3 grayscale image
        data = np.array([
            [0, 128, 255],
            [64, 192, 32],
            [255, 0, 128]
        ], dtype=np.uint8)

        img = image(mm(0), mm(0), mm(10), mm(10), data)

        # Check that href is a data URL
        href = img.attrib["href"]
        assert href.startswith("data:image/png;base64,")

        # Decode and verify the PNG
        base64_data = href.split(",")[1]
        png_bytes = base64.b64decode(base64_data)
        pil_img = Image.open(BytesIO(png_bytes))

        assert pil_img.mode == 'L'  # Grayscale
        assert pil_img.size == (3, 3)

    def test_rgb_image_3d_array(self):
        """Test creating image from 3D RGB numpy array."""
        # Create a simple 2x2 RGB image
        data = np.array([
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]]
        ], dtype=np.uint8)

        img = image(mm(5), mm(10), mm(20), mm(15), data)

        # Check that href is a data URL
        href = img.attrib["href"]
        assert href.startswith("data:image/png;base64,")

        # Decode and verify the PNG
        base64_data = href.split(",")[1]
        png_bytes = base64.b64decode(base64_data)
        pil_img = Image.open(BytesIO(png_bytes))

        assert pil_img.mode == 'RGB'
        assert pil_img.size == (2, 2)

    def test_rgba_image_4_channel_array(self):
        """Test creating image from 4-channel RGBA numpy array."""
        # Create a simple 2x2 RGBA image with transparency
        data = np.array([
            [[255, 0, 0, 255], [0, 255, 0, 128]],
            [[0, 0, 255, 64], [255, 255, 0, 0]]
        ], dtype=np.uint8)

        img = image(mm(0), mm(0), mm(30), mm(25), data)

        # Check that href is a data URL
        href = img.attrib["href"]
        assert href.startswith("data:image/png;base64,")

        # Decode and verify the PNG
        base64_data = href.split(",")[1]
        png_bytes = base64.b64decode(base64_data)
        pil_img = Image.open(BytesIO(png_bytes))

        assert pil_img.mode == 'RGBA'
        assert pil_img.size == (2, 2)

    def test_image_parameters_stored_correctly(self):
        """Test that x, y, width, height parameters are stored with correct types."""
        data = np.zeros((5, 5), dtype=np.uint8)
        img = image(mm(10), mm(20), mm(30), mm(40), data)

        # Check parameter types and units
        x_param = img.attrib["x"]
        y_param = img.attrib["y"]
        width_param = img.attrib["width"]
        height_param = img.attrib["height"]

        # These should be length parameters, not raw values
        assert hasattr(x_param, 'resolve') or isinstance(x_param, type(mm(10)))
        assert hasattr(y_param, 'resolve') or isinstance(y_param, type(mm(20)))
        assert hasattr(width_param, 'resolve') or isinstance(width_param, type(mm(30)))
        assert hasattr(height_param, 'resolve') or isinstance(height_param, type(mm(40)))

    def test_normalization_of_float_data(self):
        """Test that float arrays get normalized to 0-255 range."""
        # Create float data in range [0, 1]
        data = np.array([
            [0.0, 0.5, 1.0],
            [0.25, 0.75, 0.1]
        ], dtype=np.float64)

        img = image(mm(0), mm(0), mm(10), mm(10), data)

        # Should not raise an error and should create valid PNG
        href = img.attrib["href"]
        assert href.startswith("data:image/png;base64,")

        # Decode and verify it worked
        base64_data = href.split(",")[1]
        png_bytes = base64.b64decode(base64_data)
        pil_img = Image.open(BytesIO(png_bytes))

        assert pil_img.mode == 'L'
        assert pil_img.size == (3, 2)

    def test_resolve_method(self):
        """Test that resolve method works correctly."""
        data = np.ones((3, 3), dtype=np.uint8) * 128
        img = image(mm(5), mm(10), mm(15), mm(20), data)

        # Create minimal resolve context
        coords = CoordSet()
        scales = ScaleSet()
        occupancy = Occupancy(mm(100), mm(100))
        ctx = ResolveContext(coords, scales, occupancy)

        resolved = img.resolve(ctx)

        assert resolved.tag == "image"
        assert "href" in resolved.attrib
        assert resolved.attrib["href"].startswith("data:image/png;base64,")

    def test_invalid_array_dimensions(self):
        """Test error handling for invalid array dimensions."""
        # 1D array should raise error
        with pytest.raises(ValueError, match="data must be 2D .* or 3D .*"):
            data_1d = np.array([1, 2, 3], dtype=np.uint8)
            image(mm(0), mm(0), mm(10), mm(10), data_1d)

        # 4D array should raise error
        with pytest.raises(ValueError, match="data must be 2D .* or 3D .*"):
            data_4d = np.zeros((2, 2, 3, 4), dtype=np.uint8)
            image(mm(0), mm(0), mm(10), mm(10), data_4d)

    def test_invalid_3d_array_channels(self):
        """Test error handling for 3D arrays with wrong number of channels."""
        # 3D array with 2 channels should raise error
        with pytest.raises(ValueError, match="3D arrays must have 3 .* or 4 .* channels"):
            data = np.zeros((5, 5, 2), dtype=np.uint8)
            image(mm(0), mm(0), mm(10), mm(10), data)

        # 3D array with 5 channels should raise error
        with pytest.raises(ValueError, match="3D arrays must have 3 .* or 4 .* channels"):
            data = np.zeros((5, 5, 5), dtype=np.uint8)
            image(mm(0), mm(0), mm(10), mm(10), data)

    def test_non_numpy_array_input(self):
        """Test error handling for non-numpy array input."""
        with pytest.raises(ValueError, match="data must be a numpy array"):
            regular_list = [[1, 2], [3, 4]]
            image(mm(0), mm(0), mm(10), mm(10), regular_list)

    def test_large_image_array(self):
        """Test handling of larger image arrays."""
        # Create a 50x50 RGB image with a gradient
        height, width = 50, 50
        data = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                data[y, x, 0] = int(255 * x / width)  # Red gradient
                data[y, x, 1] = int(255 * y / height)  # Green gradient
                data[y, x, 2] = 128  # Constant blue

        img = image(mm(0), mm(0), mm(50), mm(50), data)

        # Verify the image was created successfully
        href = img.attrib["href"]
        assert href.startswith("data:image/png;base64,")

        # Decode and verify
        base64_data = href.split(",")[1]
        png_bytes = base64.b64decode(base64_data)
        pil_img = Image.open(BytesIO(png_bytes))

        assert pil_img.mode == 'RGB'
        assert pil_img.size == (50, 50)
