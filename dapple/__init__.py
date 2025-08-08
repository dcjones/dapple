
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import ElementTree, Element
from numbers import Number
from typing import Union, TextIO, Optional
import sys

from dapple.coordinates import Resolvable, AbsCoordSet, AbsCoordTransform, Lengths, AbsLengths, abslengths
from dapple.occupancy import Occupancy
from dapple.clipboard import copy_svg, ClipboardError


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

    def svg(self, width: Union[AbsLengths, Number], height: Union[AbsLengths, Number], output: Optional[str, TextIO]=None, clip: bool=False):
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
