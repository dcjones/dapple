from numbers import Number
from typing import TextIO, BinaryIO, Callable, Collection, override
from io import StringIO
import numpy as np
import sys
from pathlib import Path

from .elements import Element, viewport
from .geometry import xgrids, ygrids, xticks, yticks, xticklabels, yticklabels, key
from .coordinates import (
    CoordBounds,
    Serializable,
    AbsCoordSet,
    AbsCoordTransform,
    Lengths,
    AbsLengths,
    ResolveContext,
    Lengths,
    abslengths,
    mm,
    cm,
    pt,
    inch,
    cx,
    cxv,
    cy,
    cyv,
    vw,
    vwv,
    vh,
    vhv,
)
from .occupancy import Occupancy
from .clipboard import copy_svg, ClipboardError
from .scales import (
    UnscaledValues,
    UnscaledExpr,
    ScaleSet,
    ScaleContinuousColor,
    ScaleDiscreteColor,
    ScaleContinuousLength,
    ScaleDiscreteLength,
)
from .export import svg_to_png, svg_to_pdf, ExportError
from .scales import Scale
from .config import Config, ConfigKey, default_config
from .layout import Position


class Plot(Element):
    def __init__(
        self,
        defaults: Collection[Callable[[], Element]] = (
            xgrids,
            ygrids,
            xticks,
            yticks,
            xticklabels,
            yticklabels,
        ),
    ):
        super().__init__("dapple:plot")

        for default in defaults:
            default_el = default()
            assert isinstance(default_el, Element)
            default_el.attrib["dapple:default_element"] = True
            self.append(default_el)

    @override
    def append(self, child: Element):
        replace_index: int | None = None
        for i, other_child in enumerate(self.children):
            if other_child.tag == child.tag and other_child.attrib.get(
                "dapple:default_element", False
            ):
                replace_index = i
                break

        if replace_index is not None:
            self.children[replace_index] = child
        else:
            super().append(child)

    @override
    def resolve(self, ctx: ResolveContext) -> Element:
        """
        The root resolve function. This does a certain about of set up before
        recursively calling resolve on the rest of the tree.
        """

        config = self.get("dapple:config")
        if config is None:
            config = default_config()
        assert isinstance(config, Config)

        # Set up default scales
        # TODO: Probably need to do a deep copy here to avoid modifying the underlying plot on resolve
        scaleset = self.attrib.get("dapple:scaleset", ScaleSet())
        assert isinstance(scaleset, dict)

        all_numeric: dict[str, bool] = dict()

        def update_all_numeric_values(values: UnscaledValues):
            all_numeric[values.unit] = (
                all_numeric.get(values.unit, True) & values.all_numeric()
            )

        def update_all_numeric_expr(_attr, values: UnscaledExpr):
            values.accept_visitor(update_all_numeric_values)

        self.traverse_attributes(update_all_numeric_expr, UnscaledExpr)

        for unit, numeric in all_numeric.items():
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

        self.traverse_attributes(fit_expr, UnscaledExpr)

        for scale in scaleset.values():
            scale.finalize()

        # Layout plot
        root_configed = self.rewrite_attributes(lambda k, v: config.get(v), ConfigKey)

        def scale_expr(_attr, expr: object):
            assert isinstance(expr, UnscaledExpr)
            return expr.accept_scale(scaleset)

        root_scaled = root_configed.rewrite_attributes(scale_expr, UnscaledExpr)

        def scale_elements(el: Element):
            el.apply_scales(scaleset)

        root_scaled.traverse_elements(scale_elements)

        els = list(root_scaled)

        width = mm(ctx.coords["vw"].scale)
        height = mm(ctx.coords["vh"].scale)
        root = self.layout(els, width, height, config)

        # Fit coordinates
        bounds = CoordBounds()
        bounds.update_from_ticks(scaleset)
        root.update_bounds(bounds)
        coordset = bounds.solve(set(["y"]))

        for child in root:
            grandchild = child[0]
            grandchild.merge_coords(coordset)

        # Resolve children
        ctx.scales = scaleset
        svg_root = root.resolve(ctx)

        # Strip any lingering dapple-specific attributes
        svg_root.delete_attributes_inplace(lambda k, v: k.startswith("dapple:"))

        return svg_root

    def layout(
        self, els: list[Element], width: AbsLengths, height: AbsLengths, config: Config
    ) -> Element:
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
                wbound, hbound = child.abs_bounds()
                xoff, yoff = position.offset(wbound, hbound)
                childvp = viewport([child], x=xoff, y=yoff, width=wbound, height=hbound)
                below_nodes.append(childvp)
            elif position == Position.Above:
                above_nodes.append(child)
            elif position.isabove():
                wbound, hbound = child.abs_bounds()
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

        panel_nodes = sum(
            [below_nodes, unpositioned_nodes, above_nodes], []
        )  # concatenate
        grid[i_focus, j_focus] = viewport(panel_nodes)
        grid[i_focus, j_focus].attrib["dapple:track-occupancy"] = True

        return self._arrange_children(grid, i_focus, j_focus, width, height, config)

    def _arrange_children(
        self,
        grid: np.ndarray,
        i_focus: int,
        j_focus: int,
        width: AbsLengths,
        height: AbsLengths,
        config: Config,
    ) -> Element:
        nrows, ncols = grid.shape

        def cell_abs_bounds(cell: Element | None):
            if cell is None:
                return (0.0, 0.0)
            else:
                l, u = cell.abs_bounds()
                return (l.scalar_value(), u.scalar_value())

        widths, heights = np.vectorize(cell_abs_bounds)(grid)

        def scalar_or_zero(value: object):
            if value is None:
                return 0.0
            elif isinstance(value, AbsLengths):
                return value.scalar_value()
            else:
                raise ValueError(f"Unexpected type {type(value)}")

        def cell_padding(cell: Element | None):
            if cell is None:
                return (0.0, 0.0, 0.0, 0.0)
            else:
                t = scalar_or_zero(cell.attrib.get("dapple:padding-top"))
                b = scalar_or_zero(cell.attrib.get("dapple:padding-bottom"))
                l = scalar_or_zero(cell.attrib.get("dapple:padding-left"))
                r = scalar_or_zero(cell.attrib.get("dapple:padding-right"))

                return (t, r, b, l)

        pad_t, pad_r, pad_b, pad_l = np.vectorize(cell_padding)(grid)

        row_pad_ts = pad_t.max(axis=1)
        row_pad_bs = pad_b.max(axis=1)
        col_pad_ls = pad_l.max(axis=0)
        col_pad_rs = pad_r.max(axis=0)

        row_heights = heights.max(axis=1) + row_pad_ts + row_pad_bs
        col_widths = widths.max(axis=0) + col_pad_ls + col_pad_rs

        total_width = width.scalar_value()
        total_height = height.scalar_value()

        if row_heights.sum() > total_height and col_widths.sum() > total_width:
            print("Warning: Insufficient height and width to draw the plot.")
        elif row_heights.sum() > total_height:
            print("Warning: Insufficient height to draw the plot.")
        elif col_widths.sum() > total_width:
            print("Warning: Insufficient width to draw the plot.")

        focus_height = total_height - row_heights.sum() + row_heights[i_focus]
        focus_width = total_width - col_widths.sum() + col_widths[j_focus]

        root = Element("g")
        y = 0.0
        for i, (row_height, row_pad_t, row_pad_b) in enumerate(
            zip(row_heights, row_pad_ts, row_pad_bs)
        ):
            if i == i_focus:
                vp_height = focus_height
            else:
                vp_height = row_heights[i]

            x = 0.0
            for j, (col_width, col_pad_l, col_pad_r) in enumerate(
                zip(col_widths, col_pad_ls, col_pad_rs)
            ):
                if j == j_focus:
                    vp_width = focus_width
                else:
                    vp_width = col_widths[j]

                if grid[i, j] is not None:
                    root.append(
                        viewport(
                            [viewport([grid[i, j]])],
                            x=mm(x + col_pad_l),
                            y=mm(y + row_pad_t),
                            width=mm(vp_width - col_pad_l - col_pad_r),
                            height=mm(vp_height - row_pad_t - row_pad_b),
                        )
                    )
                x += vp_width

            y += vp_height

        return root

    def svg(
        self,
        width: AbsLengths | Number,
        height: AbsLengths | Number,
        output: None | str | Path | TextIO = None,
        clip: bool = False,
    ):
        """
        Given an absolute plot size, resolve the the plot into a pure SVG.
        """

        if not isinstance(width, AbsLengths):
            width = abslengths(width)
        width.assert_scalar()

        if not isinstance(height, AbsLengths):
            height = abslengths(height)
        height.assert_scalar()

        coords = {
            "vw": AbsCoordTransform(width.scalar_value(), 0.0),
            "vh": AbsCoordTransform(height.scalar_value(), 0.0),
        }
        occupancy = Occupancy(width, height)

        scales = self.attrib.get("dapple:scaleset", ScaleSet())
        assert isinstance(scales, dict)

        ctx = ResolveContext(coords, scales, occupancy)

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

        # TODO: We should output xml declaration

        if clip:
            svg_content = svg_root._repr_svg_()
            try:
                copy_svg(svg_content)
            except ClipboardError as e:
                print(f"Warning: Failed to copy SVG to clipboard: {e}", file=sys.stderr)

        if output is not None:
            if isinstance(output, Path):
                with output.open("w") as output_file:
                    svg_root.serialize(output_file)
            elif isinstance(output, str):
                with open(output, "w") as output_file:
                    svg_root.serialize(output_file)
            else:
                svg_root.serialize(output)

        return svg_root

    def png(
        self,
        width: AbsLengths | Number,
        height: AbsLengths | Number,
        output: None | str | BinaryIO = None,
        dpi: int = 96,
        pixel_width: None | int = None,
        pixel_height: None | int = None,
    ) -> None | bytes:
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
        buf = StringIO()
        svg_root.serialize(buf)
        svg_string = buf.getvalue()

        # Convert to PNG using Inkscape
        try:
            return svg_to_png(
                svg_string,
                output=output,
                dpi=dpi,
                width=pixel_width,
                height=pixel_height,
            )
        except ExportError as e:
            print(f"Error exporting to PNG: {e}", file=sys.stderr)
            raise

    def pdf(
        self,
        width: AbsLengths | Number,
        height: AbsLengths | Number,
        output: None | str | BinaryIO = None,
        text_to_path: bool = False,
    ) -> None | bytes:
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
        buf = StringIO()
        svg_root.serialize(buf)
        svg_string = buf.getvalue()

        # Convert to PDF using Inkscape
        try:
            return svg_to_pdf(svg_string, output=output, text_to_path=text_to_path)
        except ExportError as e:
            print(f"Error exporting to PDF: {e}", file=sys.stderr)
            raise

    def _repr_svg_(self) -> str:
        config = self.get_as("dapple:config", Config, lambda: default_config())
        return self.svg(config.plot_width, config.plot_height)._repr_svg_()


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
                pl.set("dapple:scaleset", {})

            scaleset = pl.get("dapple:scaleset")
            assert isinstance(scaleset, dict)
            scaleset[arg.unit] = arg
        elif isinstance(arg, Config):
            pl.set("dapple:config", arg)
        else:
            raise TypeError(f"Unsupported type for plot argument: {type(arg)}")

    for k, v in kwargs.items():
        # TODO: not sure what keyword args this actually supports...
        pass

    return pl
