
"""
Text extent computation using HarfBuzz for text shaping and FreeType for glyph metrics.

This module provides a Font class that can accurately compute bounding boxes for
text strings by combining HarfBuzz's advanced text shaping capabilities with
FreeType's precise glyph metrics.
"""

from matplotlib import font_manager
from .coordinates import AbsoluteLengths
from typing import Tuple
import uharfbuzz as hb
import freetype

class Font:
    """
    A font class that provides accurate text measurement using HarfBuzz and FreeType.

    This class combines HarfBuzz for text shaping (handling complex scripts, ligatures,
    kerning, etc.) with FreeType for precise glyph metrics to compute accurate bounding
    boxes for text strings.

    Attributes:
        properties: Matplotlib font properties for font matching
        font_path: Path to the actual font file being used
        font: HarfBuzz font object for text shaping
        ft_face: FreeType face object for glyph metrics
        size_mm: Font size in millimeters
    """
    properties: font_manager.FontProperties
    font_path: str
    font: hb.Font
    ft_face: freetype.Face
    size_mm: float

    def __init__(self, family: str, size: AbsoluteLengths):
        """
        Initialize a Font object with the specified family and size.

        Args:
            family: Font family name (e.g., "Arial", "DejaVu Sans")
            size: Font size as an AbsoluteLengths object in millimeters

        Raises:
            ValueError: If size is not a scalar value
            RuntimeError: If font file cannot be loaded
        """
        size.assert_scalar()
        self.size_mm = size.values[0]

        if self.size_mm <= 0:
            raise ValueError(f"Font size must be positive, got {self.size_mm}mm")

        PTS_PER_MM = 2.83464567
        pt_size = self.size_mm * PTS_PER_MM

        self.properties = font_manager.FontProperties(
            family=family,
            size=pt_size
        )
        self.font_path = font_manager.findfont(self.properties)

        try:
            # Initialize HarfBuzz
            blob = hb.Blob.from_file_path(self.font_path)
            hb_face = hb.Face(blob)
            self.font = hb.Font(hb_face)

            # Set font size in HarfBuzz (in font units)
            self.font.scale = (int(pt_size * 64), int(pt_size * 64))  # 26.6 fixed point format

            # Initialize FreeType
            self.ft_face = freetype.Face(self.font_path)
            # Set character size in points (width, height, horizontal DPI, vertical DPI)
            self.ft_face.set_char_size(int(pt_size * 64), 0, 72, 72)  # 26.6 fixed point format
        except Exception as e:
            raise RuntimeError(f"Failed to load font '{family}' from '{self.font_path}': {e}")

    def get_extents(self, text: str) -> Tuple[AbsoluteLengths, AbsoluteLengths]:
        """
        Compute the bounding box for the given text.

        This method uses HarfBuzz for advanced text shaping (handling complex scripts,
        ligatures, kerning, etc.) and FreeType for precise glyph metrics to compute
        an accurate bounding box.

        Args:
            text: The text string to measure

        Returns:
            A tuple of (width, height) as AbsoluteLengths objects in millimeters.
            The bounding box represents the minimal rectangle that would contain
            all visible pixels of the rendered text.

        Note:
            - Empty strings return (0, 0)
            - The bounding box may be smaller than the advance width for fonts
              with overhanging characters
            - Complex scripts and ligatures are properly handled through HarfBuzz
        """
        if not text:
            return AbsoluteLengths(0.0), AbsoluteLengths(0.0)

        try:
            # Create HarfBuzz buffer
            buf = hb.Buffer()
            buf.add_str(text)
            buf.guess_segment_properties()

            # Shape the text
            hb.shape(self.font, buf)

            # Get glyph info and positions
            glyph_infos = buf.glyph_infos
            glyph_positions = buf.glyph_positions

            if not glyph_infos:
                return AbsoluteLengths(0.0), AbsoluteLengths(0.0)
        except Exception as e:
            raise RuntimeError(f"Failed to shape text '{text}': {e}")

        # Get bounding box by examining each glyph
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')

        current_x = 0
        current_y = 0

        try:
            for i, (info, pos) in enumerate(zip(glyph_infos, glyph_positions)):
                glyph_id = info.codepoint

                # Load glyph in FreeType to get metrics
                self.ft_face.load_glyph(glyph_id, freetype.FT_LOAD_DEFAULT)
                glyph = self.ft_face.glyph

                # Get glyph metrics
                metrics = glyph.metrics

                # Skip glyphs with no visible content (like spaces)
                if metrics.width == 0 or metrics.height == 0:
                    current_x += pos.x_advance
                    current_y += pos.y_advance
                    continue

                # Apply position offsets from HarfBuzz
                glyph_x = current_x + pos.x_offset
                glyph_y = current_y + pos.y_offset

                # Calculate actual glyph bounds using bitmap metrics
                # FreeType metrics are in 26.6 fixed point format
                glyph_min_x = glyph_x + metrics.horiBearingX
                glyph_max_x = glyph_x + metrics.horiBearingX + metrics.width
                glyph_min_y = glyph_y + metrics.horiBearingY - metrics.height
                glyph_max_y = glyph_y + metrics.horiBearingY

                # Update overall bounds
                min_x = min(min_x, glyph_min_x)
                max_x = max(max_x, glyph_max_x)
                min_y = min(min_y, glyph_min_y)
                max_y = max(max_y, glyph_max_y)

                # Advance position
                current_x += pos.x_advance
                current_y += pos.y_advance
        except Exception as e:
            raise RuntimeError(f"Failed to compute glyph metrics: {e}")

        # Handle case where no glyphs have visible bounds
        if min_x == float('inf'):
            min_x = max_x = 0
        if min_y == float('inf'):
            min_y = max_y = 0

        # Convert from font units to millimeters
        # HarfBuzz uses 26.6 fixed point format (divide by 64)
        # Then convert from points to mm
        MM_PER_PT = 1.0 / 2.83464567

        width_pts = (max_x - min_x) / 64.0
        height_pts = (max_y - min_y) / 64.0

        width_mm = width_pts * MM_PER_PT
        height_mm = height_pts * MM_PER_PT

        return AbsoluteLengths(width_mm), AbsoluteLengths(height_mm)
