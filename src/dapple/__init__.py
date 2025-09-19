
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import ElementTree, Element
from numbers import Number
from typing import Union, TextIO, Optional, BinaryIO
import numpy as np
import sys

from .coordinates import Resolvable, Serializable, AbsCoordSet, AbsCoordTransform, Lengths, AbsLengths, ResolveContext, Lengths, abslengths
from .coordinates import mm, cm, pt, inch
from .coordinates import cx, cxv, cy, cyv, vw, vwv, vh, vhv
from .occupancy import Occupancy
from .clipboard import copy_svg, ClipboardError
from .scales import UnscaledValues, UnscaledExpr, ScaleSet, ScaleContinuousColor, ScaleDiscreteColor, ScaleContinuousLength, ScaleDiscreteLength
from .export import svg_to_png, svg_to_pdf, ExportError
from .defaults import DEFAULTS
from .scales import Scale
from .coordinates import Resolvable, CoordBounds
from .config import Config, ConfigKey
from .layout import Position
from .elements import ResolvableElement, ViewportElement, \
    delete_attributes_inplace, traverse_attributes, traverse_elements, rewrite_attributes, \
    rewrite_attributes_inplace, abs_bounds, viewport



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

        config = self.get("dapple:config")
        if config is None:
            config = Config()
        assert isinstance(config, Config)

        # Set up default scales
        # TODO: Probably need to do a deep copy here to avoid modifying the underlying plot on resolve
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
                    scaleset[unit] = ScaleContinuousColor(unit)
                else:
                    scaleset[unit] = ScaleDiscreteColor(unit)
            elif unit == "shape":
                raise Exception("shape scale not yet implemented")
            else:
                if numeric:
                    scaleset[unit] = ScaleContinuousLength(unit)
                else:
                    scaleset[unit] = ScaleDiscreteLength(unit)
        config.replace_keys(scaleset)

        # Fit and apply scales
        def fit_expr(_attr, expr: UnscaledExpr):
            expr.accept_fit(scaleset)

        traverse_attributes(self, fit_expr, UnscaledExpr)

        # Layout plot
        root_configed = rewrite_attributes(
            self, lambda k, v: config.get(v), ConfigKey)

        def scale_expr(_attr, expr: UnscaledExpr):
            return expr.accept_scale(scaleset)

        root_scaled =  rewrite_attributes(root_configed,  scale_expr, UnscaledExpr)

        els = list(root_scaled)

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
        ctx.scales = scaleset
        svg_root = root.resolve(ctx)

        # Strip any lingering dapple-specific attributes
        delete_attributes_inplace(svg_root, lambda k, v: k.startswith("dapple:"))

        # Convert serializable attributes to strings
        rewrite_attributes_inplace(svg_root, lambda k, v: v.serialize(), Serializable)
        delete_attributes_inplace(svg_root, lambda k, v: v is None)

        return svg_root

    def layout(self, els: list[Element], width: AbsLengths, height: AbsLengths) -> ResolvableElement:
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

    def _arrange_children(self, grid: np.ndarray, i_focus: int, j_focus: int, width: AbsLengths, height: AbsLengths) -> ResolvableElement:
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

        root = ResolvableElement("g")
        y = 0.0
        for (i, row_height) in enumerate(row_heights):
            if i == i_focus:
                vp_height = focus_height
            else:
                vp_height = row_heights[i]

            x = 0.0
            for (j, col_width) in enumerate(col_widths):
                if j == j_focus:
                    vp_width = focus_width
                else:
                    vp_width = col_widths[j]

                if grid[i, j] is not None:
                    root.append(viewport(
                        [grid[i, j]],
                        x=mm(x),
                        y=mm(y),
                        width=mm(vp_width),
                        height=mm(vp_height),
                    ))
                x += vp_width

            y += vp_height

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
        width_value = width.scalar_value()
        height_value = height.scalar_value()
        svg_root.set("width", f"{width_value:.3f}mm")
        svg_root.set("height", f"{height_value:.3f}mm")
        svg_root.set("viewBox", f"0 0 {width_value:.3f} {height_value:.3f}")
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
            if pl.get("dapple:scaleset") is None:
                pl.set("dapple:scaleset", arg.unit, arg) # type: ignore
            else:
                scaleset = pl.get("dapple:scaleset")
                assert isinstance(scaleset, dict)
                scaleset[arg.unit] = arg
        elif isinstance(arg, Config):
            pl.set("dapple:config", arg) # type: ignore
        else:
            raise TypeError(f"Unsupported type for plot argument: {type(arg)}")

    for (k, v)in kwargs.items():
        # TODO: not sure what keyword args this actually supports...
        pass

    return pl
