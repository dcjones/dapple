import re
import pytest
import numpy as np
import base64
from io import BytesIO, StringIO
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
        """Test that the image corners are stored as absolute positions.

        The extent is stored as the far corner (x + width, y + height) rather
        than as a relative width/height vector, so that continuous scales fit
        the image's true extent. The width/height are recovered during resolve.
        """
        data = np.zeros((5, 5), dtype=np.uint8)
        img = image(mm(10), mm(20), mm(30), mm(40), data)

        x_param = img.attrib["x"]
        y_param = img.attrib["y"]
        x1_param = img.attrib["dapple:x1"]
        y1_param = img.attrib["dapple:y1"]

        # width/height are not stored up front; they are derived in resolve.
        assert "width" not in img.attrib
        assert "height" not in img.attrib

        # These should be length parameters, not raw values
        assert hasattr(x_param, 'resolve') or isinstance(x_param, type(mm(10)))
        assert hasattr(y_param, 'resolve') or isinstance(y_param, type(mm(20)))
        assert hasattr(x1_param, 'resolve') or isinstance(x1_param, type(mm(40)))
        assert hasattr(y1_param, 'resolve') or isinstance(y1_param, type(mm(60)))

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

    def test_far_from_origin_does_not_include_origin(self):
        """Continuous scales should fit the image extent, not the origin.

        Regression test: previously the width/height were stored as a relative
        vector and fed into scale fitting as if they were coordinates, which
        dragged the axes back toward (0, 0) for an image placed far away.
        """
        from dapple import plot, xcontinuous, ycontinuous

        data = np.zeros((10, 10, 3), dtype=np.uint8)
        data[:, :, 0] = 255

        pl = plot(
            image(x=1000, y=1000, width=100, height=100, data=data),
            xcontinuous(),
            ycontinuous(),
        )
        pl.svg(400, 400, StringIO())

        scales = pl.attrib["dapple:scaleset"]
        for unit in ("x", "y"):
            scale = scales[unit]
            assert scale.min == 1000
            assert scale.max == 1100

    def test_flipped_axis_emits_positive_dimensions(self):
        """A flipped axis must not produce a negative SVG width/height.

        Regression test: the image previously baked the axis flip into a
        transform but left the height negative, which is invalid SVG. Renderers
        that clamp the negative dimension drew the image shifted by a full
        height. The extent must be positive and the image must map onto its
        true data rectangle.
        """
        from dapple import plot, xcontinuous, ycontinuous
        from dapple.geometry.points import points

        data = np.zeros((20, 20, 3), dtype=np.uint8)
        data[:, :, 2] = 255

        # Anchor a domain of [1000, 1800] on both axes so the image occupies a
        # sub-rectangle rather than the whole plot.
        pl = plot(
            points(x=[1000, 1800], y=[1000, 1800]),
            image(x=1400, y=1200, width=200, height=200, data=data),
            xcontinuous(),
            ycontinuous(),
        )
        out = StringIO()
        pl.svg(300, 300, out)  # y is flipped by default
        svg = out.getvalue()

        img_line = next(l for l in svg.splitlines() if "<image" in l)

        width = float(re.search(r'width="(-?[\d.]+)"', img_line).group(1))
        height = float(re.search(r'height="(-?[\d.]+)"', img_line).group(1))
        assert width > 0
        assert height > 0

        # Map the image's local box through its transform and confirm the
        # screen extent matches the [1400, 1600] x [1200, 1400] data rectangle.
        m = re.search(r"matrix\(([^)]*)\)", img_line)
        assert m is not None
        a, _b, _c, d, e, f = [float(v) for v in m.group(1).split(",")]
        xs = [e + a * lx for lx in (0.0, width)]
        ys = [f + d * ly for ly in (0.0, height)]
        screen_x = (min(xs), max(xs))
        screen_y = (min(ys), max(ys))

        # Reference points: circles at data (1000, 1000) and (1800, 1800).
        pts = []
        for l in svg.splitlines():
            cx = re.search(r'cx="([\d.]+)"', l)
            cy = re.search(r'cy="([\d.]+)"', l)
            if cx and cy:
                pts.append((float(cx.group(1)), float(cy.group(1))))
        (cx_lo, cy_lo), (cx_hi, cy_hi) = sorted(pts)

        # Data -> screen linear maps derived from the two anchor points.
        def to_screen_x(v):
            return cx_lo + (cx_hi - cx_lo) * (v - 1000) / (1800 - 1000)

        def to_screen_y(v):
            return cy_lo + (cy_hi - cy_lo) * (v - 1000) / (1800 - 1000)

        exp_x = tuple(sorted((to_screen_x(1400), to_screen_x(1600))))
        exp_y = tuple(sorted((to_screen_y(1200), to_screen_y(1400))))

        assert screen_x[0] == pytest.approx(exp_x[0], abs=0.5)
        assert screen_x[1] == pytest.approx(exp_x[1], abs=0.5)
        assert screen_y[0] == pytest.approx(exp_y[0], abs=0.5)
        assert screen_y[1] == pytest.approx(exp_y[1], abs=0.5)

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
