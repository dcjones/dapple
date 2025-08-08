
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import ElementTree, Element
from numbers import Number
from typing import Union, TextIO, Optional, BinaryIO
import sys

from dapple.coordinates import Resolvable, AbsCoordSet, AbsCoordTransform, Lengths, AbsLengths, abslengths
from dapple.occupancy import Occupancy
from dapple.clipboard import copy_svg, ClipboardError
from dapple.export import svg_to_png, svg_to_pdf, ExportError


# Figuring some things out here:
#   - We should be able to build our tree using xml.etree.ElementTree, which
#     is pretty much our XML representation.
#
#

from .treemap import treemap
from .scales import Scale
from .coordinates import Resolvable
from .elements import ResolvableElement


class Plot(ResolvableElement):
    def __init__(self):
        super().__init__("dapple:plot")

    def resolve(self, coords: AbsCoordSet, occupancy: Occupancy) -> Union[Lengths, Element]:
        # TODO: Do a lot of stuff here.
        #   - Figure out scale, coords, layouts.
        #   - Traverse and rewrite the tree.
        #   - Return that rewritten tree.
        pass

    def svg(self, width: Union[AbsLengths, Number], height: Union[AbsLengths, Number], output: Optional[Union[str, TextIO]]=None, clip: bool=False):
        if not isinstance(width, AbsLengths):
            width = abslengths(width)
        width.assert_scalar()

        if not isinstance(height, AbsLengths):
            height = abslengths(height)
        height.assert_scalar()

        coords = {
            "vw": AbsCoordTransform(width.scalar_value(), 0.0),
            "vh": AbsCoordTransform(height.scalar_value(), 0.0)
        }
        occupancy = Occupancy(width, height)

        resolved_plot = self.resolve(coords, occupancy)
        assert isinstance(resolved_plot, Element)

        svg_root = Element("svg")
        svg_root.set("width", str(width.scalar_value()))
        svg_root.set("height", str(height.scalar_value()))
        svg_root.set("xmlns", "http://www.w3.org/2000/svg")
        svg_root.append(resolved_plot)

        if clip:
            svg_string = ET.tostring(svg_root, encoding="unicode", method="xml")
            try:
                copy_svg(svg_string)
            except ClipboardError as e:
                print(f"Warning: Failed to copy SVG to clipboard: {e}", file=sys.stderr)

        if output is not None:
            tree = ElementTree(svg_root)
            if isinstance(output, str):
                tree.write(output, encoding="unicode", xml_declaration=True)
            else:
                tree.write(output, encoding="unicode", xml_declaration=True)

        return svg_root

    def png(self, width: Union[AbsLengths, Number], height: Union[AbsLengths, Number],
            output: Optional[Union[str, BinaryIO]]=None, dpi: int=96,
            pixel_width: Optional[int]=None, pixel_height: Optional[int]=None) -> Optional[bytes]:
        """
        Export plot as PNG using Inkscape.

        Args:
            width: Width of the plot (in absolute units)
            height: Height of the plot (in absolute units)
            output: Output destination - can be:
                - A string path to write the PNG file
                - A file-like object opened in binary mode
                - None to return the PNG data as bytes
            dpi: Resolution in dots per inch (default: 96)
            pixel_width: Optional width in pixels (overrides SVG dimensions)
            pixel_height: Optional height in pixels (overrides SVG dimensions)

        Returns:
            PNG data as bytes if output is None, otherwise None

        Raises:
            ExportError: If Inkscape is not available or conversion fails
        """
        # First generate the SVG
        svg_root = self.svg(width, height)
        svg_string = ET.tostring(svg_root, encoding="unicode", method="xml")

        # Convert to PNG using Inkscape
        try:
            return svg_to_png(svg_string, output=output, dpi=dpi,
                            width=pixel_width, height=pixel_height)
        except ExportError as e:
            print(f"Error exporting to PNG: {e}", file=sys.stderr)
            raise

    def pdf(self, width: Union[AbsLengths, Number], height: Union[AbsLengths, Number],
            output: Optional[Union[str, BinaryIO]]=None, text_to_path: bool=False) -> Optional[bytes]:
        """
        Export plot as PDF using Inkscape.

        Args:
            width: Width of the plot (in absolute units)
            height: Height of the plot (in absolute units)
            output: Output destination - can be:
                - A string path to write the PDF file
                - A file-like object opened in binary mode
                - None to return the PDF data as bytes
            text_to_path: Convert text to paths (default: False)

        Returns:
            PDF data as bytes if output is None, otherwise None

        Raises:
            ExportError: If Inkscape is not available or conversion fails
        """
        # First generate the SVG
        svg_root = self.svg(width, height)
        svg_string = ET.tostring(svg_root, encoding="unicode", method="xml")

        # Convert to PDF using Inkscape
        try:
            return svg_to_pdf(svg_string, output=output, text_to_path=text_to_path)
        except ExportError as e:
            print(f"Error exporting to PDF: {e}", file=sys.stderr)
            raise


def plot(*args, **kwargs) -> Plot:
    """
    Plot constructor interface.
    """
    pl = Plot()

    for arg in args:
        if isinstance(arg, Element):
            pl.append(arg)
        elif isinstance(arg, Scale):
            # TODO: I guess we keep scales in an attribute?
            pass
        else:
            raise TypeError(f"Unsupported type for plot argument: {type(arg)}")

    for (k, v)in kwargs.items():
        # TODO: not sure what keyword args this actually supports...
        pass

    return pl
