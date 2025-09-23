import numpy as np
import base64
from io import BytesIO
from PIL import Image
from ..elements import Element
from ..scales import length_params
from ..coordinates import CtxLenType, ResolveContext
from typing import override


def image(x, y, width, height, data):
    """
    Create an SVG image element from numpy array data.

    Args:
        x: X position of the image
        y: Y position of the image
        width: Width of the image
        height: Height of the image
        data: Numpy array containing pixel data

    Returns:
        Element: SVG image element with PNG data URL
    """
    return ImageElement(x, y, width, height, data)


class ImageElement(Element):
    """
    SVG image element that encodes numpy array data as PNG data URL.
    """

    def __init__(self, x, y, width, height, data):
        # Convert numpy array to PNG data URL
        data_url = self._numpy_to_png_data_url(data)

        attrib = {
            "x": length_params("x", x, CtxLenType.Pos),
            "y": length_params("y", y, CtxLenType.Pos),
            "width": length_params("x", width, CtxLenType.Vec),
            "height": length_params("y", height, CtxLenType.Vec),
            "href": data_url,
        }

        super().__init__("image", attrib)

    def _numpy_to_png_data_url(self, data):
        """Convert numpy array to PNG data URL."""
        # Ensure data is in the right format
        if not isinstance(data, np.ndarray):
            raise ValueError("data must be a numpy array")

        # Handle different array shapes and types
        if data.ndim == 2:
            # Grayscale image
            if data.dtype != np.uint8:
                # Normalize to 0-255 range
                data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
            image_array = data
        elif data.ndim == 3:
            # RGB or RGBA image
            if data.shape[2] == 3:
                # RGB
                if data.dtype != np.uint8:
                    data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
                image_array = data
            elif data.shape[2] == 4:
                # RGBA
                if data.dtype != np.uint8:
                    data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
                image_array = data
            else:
                raise ValueError("3D arrays must have 3 (RGB) or 4 (RGBA) channels")
        else:
            raise ValueError("data must be 2D (grayscale) or 3D (RGB/RGBA) array")

        # Create PIL Image
        if image_array.ndim == 2:
            pil_image = Image.fromarray(image_array)
        elif image_array.shape[2] == 3:
            pil_image = Image.fromarray(image_array)
        else:  # RGBA
            pil_image = Image.fromarray(image_array)

        # Convert to PNG bytes
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        png_bytes = buffer.getvalue()

        # Encode as base64 data URL
        base64_data = base64.b64encode(png_bytes).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        """Resolve the image element to pure SVG."""
        attrib = {}
        for key, value in self.attrib.items():
            if hasattr(value, 'resolve'):
                attrib[key] = value.resolve(ctx)
            else:
                attrib[key] = value

        return Element("image", attrib)
