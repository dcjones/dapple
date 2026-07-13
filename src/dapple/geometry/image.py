import base64
from io import BytesIO
from typing import override

import numpy as np
from PIL import Image

from ..coordinates import AbsLengths, AbsTransform, CtxLenType, ResolveContext
from ..elements import Element
from ..scales import length_params


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

        # Store the far corner (x + width, y + height) as an absolute position
        # rather than storing width/height as a relative vector. If the extent
        # were stored as a vector, scale fitting would treat the width/height
        # magnitudes as coordinates, dragging the axes toward the origin when
        # the image is placed far from (0, 0). The width/height are recovered
        # from the two corners in `resolve`.
        attrib = {
            "x": length_params("x", x, CtxLenType.Pos),
            "y": length_params("y", y, CtxLenType.Pos),
            "dapple:x1": length_params("x", x + width, CtxLenType.Pos),
            "dapple:y1": length_params("y", y + height, CtxLenType.Pos),
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
                data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(
                    np.uint8
                )
            image_array = data
        elif data.ndim == 3:
            # RGB or RGBA image
            if data.shape[2] == 3:
                # RGB
                if data.dtype != np.uint8:
                    data = (
                        (data - data.min()) / (data.max() - data.min()) * 255
                    ).astype(np.uint8)
                image_array = data
            elif data.shape[2] == 4:
                # RGBA
                if data.dtype != np.uint8:
                    data = (
                        (data - data.min()) / (data.max() - data.min()) * 255
                    ).astype(np.uint8)
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
        pil_image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        # Encode as base64 data URL
        base64_data = base64.b64encode(png_bytes).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        root = super().resolve(ctx)

        x = root.attrib["x"]
        y = root.attrib["y"]
        x1 = root.attrib["dapple:x1"]
        y1 = root.attrib["dapple:y1"]

        assert isinstance(x, AbsLengths)
        assert isinstance(y, AbsLengths)
        assert isinstance(x1, AbsLengths)
        assert isinstance(y1, AbsLengths)

        # Recover the width/height from the two absolute corners. When an axis
        # is flipped this difference is negative; the flip handling below
        # rewrites the position and dimensions so SVG never sees a negative
        # width/height. (The dapple:* corner attributes are stripped after
        # resolution.)
        width = x1 - x
        height = y1 - y
        assert isinstance(width, AbsLengths)
        assert isinstance(height, AbsLengths)
        root.attrib["width"] = width
        root.attrib["height"] = height

        xflipped = "x" in ctx.coords and ctx.coords["x"].scale < 0
        yflipped = "y" in ctx.coords and ctx.coords["y"].scale < 0

        if xflipped or yflipped:
            x = x.scalar_value()
            y = y.scalar_value()
            width = width.scalar_value()
            height = height.scalar_value()

            t = AbsTransform(
                -1.0 if xflipped else 1.0,
                0.0,
                0.0,
                -1.0 if yflipped else 1.0,
                x + width if xflipped else x,
                y + height if yflipped else y,
            )

            root.attrib["transform"] = t.serialize()
            root.attrib["x"] = "0"
            root.attrib["y"] = "0"

        return root
