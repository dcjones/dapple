
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import ElementTree, Element
from numbers import Number
from typing import Union, TextIO, Optional, BinaryIO
from enum import Enum
import numpy as np
import sys

from .coordinates import Resolvable, AbsCoordSet, AbsCoordTransform, Lengths, AbsLengths, ResolveContext, Lengths, abslengths
from .coordinates import mm, cm, pt, inch
from .coordinates import cx, cxv, cy, cyv, cw, cwv, ch, chv
from .occupancy import Occupancy
from .clipboard import copy_svg, ClipboardError
from .scales import UnscaledValues, UnscaledExpr, ScaleSet, ScaleContinuousColor, ScaleDiscreteColor, ScaleContinuousLength, ScaleDiscreteLength
from .export import svg_to_png, svg_to_pdf, ExportError
from .defaults import DEFAULTS
from .scales import Scale
from .coordinates import Resolvable, CoordBounds
from .elements import ResolvableElement, ViewportElement, traverse_attributes, rewrite_attributes, abs_bounds, viewport


class Position(Enum):
    """
    Represents relative positioning of a plot element, to be arranged by the by
    the layout function.
    """

    Default = 1 # place in the panel, with unspecified order

    Above = 2
    AboveTopLeft = 3
    AboveTopRight = 4
    AboveBottomLeft = 5
    AboveBottomRight = 6

    Below = 7
    BelowTopLeft = 8
    BelowTopRight = 9
    BelowBottomLeft = 10
    BelowBottomRight = 11

    BottomLeft = 12
    BottomCenter = 13
    BottomRight = 14

    TopLeft = 15
    TopCenter = 16
    TopRight = 17

    LeftTop = 18
    LeftCenter = 19
    LeftBottom = 20

    RightTop = 21
    RightCenter = 22
    RightBottom = 23

    def isabove(self) -> bool:
        return self in [Position.Above, Position.AboveTopLeft, Position.AboveTopRight, Position.AboveBottomLeft, Position.AboveBottomRight]

    def isbelow(self) -> bool:
        return self in [Position.Below, Position.BelowTopLeft, Position.BelowTopRight, Position.BelowBottomLeft, Position.BelowBottomRight]

    def isbottom(self) -> bool:
        return self in [Position.BottomLeft, Position.BottomCenter, Position.BottomRight]

    def istop(self) -> bool:
        return self in [Position.TopLeft, Position.TopCenter, Position.TopRight]

    def isleft(self) -> bool:
        return self in [Position.LeftTop, Position.LeftCenter, Position.LeftBottom]

    def isright(self) -> bool:
        return self in [Position.RightTop, Position.RightCenter, Position.RightBottom]

    def offset(self, width: AbsLengths, height: AbsLengths) -> tuple[Lengths, Lengths]:
        """
        Give the size of the element in absolute units, return the position it should be placed at
        with respect to the cell its placed in in the grid layout.
        """

        match self:
            case Position.Default:
                return mm(0), mm(0)
            case Position.Above:
                return mm(0), mm(0)
            case Position.Below:
                return mm(0), mm(0)
            case Position.AboveTopLeft:
                return mm(0), mm(0)
            case Position.AboveTopRight:
                return cw(1) - width, mm(0)
            case Position.AboveBottomLeft:
                return mm(0), ch(1) - height
            case Position.AboveBottomRight:
                return cw(1) - width, ch(1) - height
            case Position.BelowTopLeft:
                return mm(0), mm(0)
            case Position.BelowTopRight:
                return cw(1) - width, mm(0)
            case Position.BelowBottomLeft:
                return mm(0), ch(1) - height
            case Position.BelowBottomRight:
                return cw(1) - width, ch(1) - height
            case Position.BottomLeft:
                return mm(0), mm(0)
            case Position.BottomCenter:
                return cw(0.5) - 0.5*width, mm(0)
            case Position.BottomRight:
                return cw(1) - width, mm(0)
            case Position.TopLeft:
                return mm(0), mm(0)
            case Position.TopCenter:
                return cw(0.5) - 0.5*width, mm(0)
            case Position.TopRight:
                return cw(1) - width, mm(0)
            case Position.LeftTop:
                return mm(0), mm(0)
            case Position.LeftCenter:
                return mm(0), ch(0.5) - 0.5*height
            case Position.LeftBottom:
                return mm(0), ch(1) - height
            case Position.RightTop:
                return mm(0), mm(0)
            case Position.RightCenter:
                return mm(0), ch(0.5) - 0.5*height
            case Position.RightBottom:
                return mm(0), ch(1) - height
            case _:
                raise ValueError(f"Invalid position: {self}")

class Plot(ResolvableElement):
    def __init__(self):
        # TODO:
        # - default scales and such

        super().__init__("dapple:plot")

    def resolve(self, ctx: ResolveContext) -> Element:
        """
        The root resolve function. This does a certain about of set up before
        recursively calling resolve on the rest of the tree.
        """

        # Set up default scales
        scaleset = self.attrib.get("dapple:scaleset", ScaleSet())
        assert isinstance(scaleset, dict)

        all_numeric = dict()
        def update_all_numeric_values(values: UnscaledValues):
            all_numeric[values.unit] = all_numeric.get(values.unit, True) & values.all_numeric()

        def update_all_numeric_expr(_attr, values: UnscaledExpr):
            values.accept_visitor(update_all_numeric_values)

        traverse_attributes(self, update_all_numeric_expr, UnscaledExpr)

        for (unit, numeric) in all_numeric.items():
            if unit in scaleset:
                continue

            if unit == "color":
                if numeric:
                    scaleset[unit] = ScaleContinuousColor(unit, colormap=DEFAULTS["continuous_cmap"])
                else:
                    scaleset[unit] = ScaleDiscreteColor(unit, colormap=DEFAULTS["discrete_cmap"])
            elif unit == "shape":
                raise Exception("shape scale not yet implemented")
            else:
                if numeric:
                    scaleset[unit] = ScaleContinuousLength(unit)
                else:
                    scaleset[unit] = ScaleDiscreteLength(unit)

        # Fit and apply scales
        def fit_expr(_attr, expr: UnscaledExpr):
            expr.accept_fit(scaleset)

        traverse_attributes(self, fit_expr, UnscaledExpr)

        def scale_expr(_attr, expr: UnscaledExpr):
            return expr.accept_scale(scaleset)

        # Layout plot
        els = list(rewrite_attributes(self, scale_expr, UnscaledExpr))
        width = mm(ctx.coords["vw"].scale)
        height = mm(ctx.coords["vh"].scale)
        root = self.layout(els, width, height)

        # Fit coordinates
        bounds = CoordBounds()
        def update_bounds(_attr, expr: Lengths):
            bounds.update(expr)
        traverse_attributes(root, update_bounds, Lengths)
        coordset = bounds.solve()

        for child in root:
            assert isinstance(child, ViewportElement)
            child.merge_coords(coordset)

        # Resolve children
        #   - run resolve on the root node

        # TODO: assert there are no "dapple:" attributes or tags
        # (Maybe we have to strip these?)
        pass

    def layout(self, els: list[Element], width: AbsLengths, height: AbsLengths) -> Element:
        """
        Arrange child elements in a grid, based on the "dapple:position" attribute.
        """

        nleft = 0
        nright = 0
        ntop = 0
        nbottom = 0
        for child in self:
            position = child.attrib.get("dapple:position", Position.Default)
            assert isinstance(position, Position)

            if position.isleft():
                nleft += 1
            elif position.isright():
                nright += 1
            elif position.istop():
                ntop += 1
            elif position.isbottom():
                nbottom += 1

        nrow = 1 + ntop + nbottom
        ncol = 1 + nleft + nright

        # REMEMBER: 0-based indexes!!!

        grid = np.full((nrow, ncol), None, dtype=object)
        i_focus = ntop
        j_focus = nleft

        unpositioned_nodes = []
        above_nodes = []
        below_nodes = []

        next_left = j_focus - 1
        next_right = j_focus + 1
        next_top = i_focus - 1
        next_bottom = i_focus + 1

        for child in els:
            position = child.attrib.get("dapple:position", Position.Default)
            assert isinstance(position, Position)

            if position == Position.Default:
                unpositioned_nodes.append(child)
            elif position == Position.Below:
                below_nodes.append(child)
            elif position.isbelow():
                wbound, hbound = abs_bounds(child)
                xoff, yoff = position.offset(wbound, hbound)
                childvp = viewport([child], x=xoff, y=yoff, width=wbound, height=hbound)
                below_nodes.append(childvp)
            elif position == Position.Above:
                above_nodes.append(child)
            elif position.isabove():
                wbound, hbound = abs_bounds(child)
                xoff, yoff = position.offset(wbound, hbound)
                childvp = viewport([child], x=xoff, y=yoff, width=wbound, height=hbound)
                above_nodes.append(childvp)
            elif position.isleft():
                grid[i_focus, next_left] = child
                next_left -= 1
            elif position.isright():
                grid[i_focus, next_right] = child
                next_right += 1
            elif position.istop():
                grid[next_top, j_focus] = child
                next_top -= 1
            elif position.isbottom():
                grid[next_bottom, j_focus] = child
                next_bottom += 1

        assert next_left == -1
        assert next_right == ncol
        assert next_top == -1
        assert next_bottom == nrow

        panel_nodes = sum([below_nodes, unpositioned_nodes, above_nodes], []) # concatenate
        grid[i_focus, j_focus] = viewport(panel_nodes)
        grid[i_focus, j_focus].attrib["dapple:track-occupancy"] = True

        return self._arrange_children(grid, i_focus, j_focus, width, height)

    def _arrange_children(self, grid: np.ndarray, i_focus: int, j_focus: int, width: AbsLengths, height: AbsLengths) -> Element:
        nrows, ncols = grid.shape

        def cell_abs_bounds(cell: Optional[Element]):
            if cell is None:
                return (0.0, 0.0)
            else:
                l, u = abs_bounds(cell)
                return (l.scalar_value(), u.scalar_value())

        widths, heights = np.vectorize(cell_abs_bounds)(grid)

        row_heights = heights.max(axis=1)
        col_widths = widths.max(axis=0)

        total_width = width.scalar_value()
        total_height = height.scalar_value()

        if row_heights.sum() > total_height and col_widths.sum() > total_width:
            print("Warning: Insufficient height and width to draw the plot.")
        elif row_heights.sum() > total_height:
            print("Warning: Insufficient height to draw the plot.")
        elif col_widths.sum() > total_width:
            print("Warning: Insufficient width to draw the plot.")

        focus_height = total_height - row_heights.sum()
        focus_width = total_width - col_widths.sum()

        root = Element("g")
        y = 0.0
        for (i, row_height) in enumerate(row_heights):
            x = 0.0

            if i == i_focus:
                vp_height = focus_height
            else:
                vp_height = row_heights[i]

            for (j, col_width) in enumerate(col_widths):
                if grid[i, j] is None:
                    continue

                if j == j_focus:
                    vp_width = focus_width
                else:
                    vp_width = col_widths[i]

                root.append(viewport(
                    [grid[i,j]],
                    x=mm(x),
                    y=mm(y),
                    width=mm(vp_width),
                    height=mm(vp_height),
                ))

        return root


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

        scales = self.attrib.get("dapple:scaleset", ScaleSet())
        assert isinstance(scales, dict)

        ctx = ResolveContext(
            coords,
            scales,
            occupancy
        )

        resolved_plot = self.resolve(ctx)
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
