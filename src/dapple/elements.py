from .coordinates import (
    CoordTransform,
    CoordBounds,
    Resolvable,
    CoordSet,
    AbsLengths,
    Lengths,
    ResolveContext,
    Serializable,
    resolve,
    mm,
    vw,
    vwv,
    vh,
    vhv,
    translate,
)
from .colors import Colors
from .scales import ScaleSet
from typing import (
    cast,
    final,
    Any,
    TextIO,
    Callable,
    TypeVar,
    override,
)
from collections.abc import Iterable
from io import StringIO
from itertools import repeat
from copy import copy

AttrType = TypeVar("AttrType")


class Element(Resolvable):
    """
    This is an XML Element representation that mimics the standard library etree
    Element, but permits non-string attribute values.
    """

    tag: str
    attrib: dict[str, object]
    text: str | list[str] | None
    children: list["Element"]

    def __init__(
        self,
        tag: str = "g",
        attrib: dict[str, object] | None = None,
        *args: "Element",
        **kwargs: object,
    ):
        self.tag = tag
        self.text = None
        self.attrib = attrib if attrib is not None else dict()
        for key, value in kwargs.items():
            self.attrib[key] = value

        self.children = []
        for arg in args:
            if isinstance(arg, RawText):
                self.text = arg.data
            else:
                self.children.append(arg)

    def __getitem__(self, index: int) -> "Element":
        return self.children[index]

    def __len__(self) -> int:
        return len(self.children)

    def __iter__(self):
        return iter(self.children)

    def clear(self):
        self.text = None
        self.attrib = dict()
        self.children = []

    def set(self, key: str, value: object):
        self.attrib[key] = value

    def get(self, key: str, default: object | None = None) -> object | None:
        return self.attrib.get(key, default)

    def get_as(
        self,
        key: str,
        expected_type: type[AttrType],
        default_fn: None | Callable[[], AttrType] = None,
    ) -> AttrType:
        value = self.attrib.get(key)
        if value is None and default_fn is not None:
            value = default_fn()
        if value is None:
            raise KeyError(f"Attribute '{key}' not found")
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Attribute '{key}' is not of type {expected_type.__name__}"
            )
        return value

    def append(self, child: "Element"):
        self.children.append(child)

    def isxml(self) -> bool:
        allstr = True
        for _key, value in self.attrib.items():
            if not isinstance(value, str):
                allstr = False
                break

        return allstr and all(child.isxml() for child in self.children)

    def assert_xml(self):
        if not self.isxml():
            raise ValueError("Element is not pure XML")

    @override
    def __repr__(self):
        return f"<{self.tag} {self.attrib}>"

    def similar(self, attrib: dict[str, object]) -> "Element":
        return Element(self.tag, attrib)

    @override
    def resolve(self, ctx: ResolveContext) -> "Element":
        """
        Convert the resolvable element into a regular svg element by providing
        context on absolute sizes ond occupancy.
        """

        attrib = {k: resolve(v, ctx) for (k, v) in self.attrib.items()}

        if "dapple:coords" in attrib:
            coords = attrib["dapple:coords"]
            assert isinstance(coords, dict)

            child_coords = copy(ctx.coords)
            child_coords.update(coords)
            child_ctx = ResolveContext(child_coords, ctx.scales, ctx.occupancy)
        else:
            child_ctx = ctx

        el = self.similar(attrib)
        el.text = self.text

        for child in self:
            el.append(resolve(child, child_ctx))

        return el

    def update_bounds(self, bounds: CoordBounds):
        # Special-cases handling particular SVG elements
        match self.tag:
            case "circle":
                x = self.get_as("cx", Lengths)
                y = self.get_as("cy", Lengths)
                r = self.get_as("r", Lengths)

                bounds.update(x - r)
                bounds.update(y - r)
                bounds.update(x + r)
                bounds.update(y + r)
            case "rect":
                x = self.get_as("x", Lengths)
                y = self.get_as("y", Lengths)
                w = self.get_as("width", Lengths)
                h = self.get_as("height", Lengths)

                bounds.update(x)
                bounds.update(y)
                bounds.update(x + w)
                bounds.update(y + h)
            case _:
                for _attr, value in self.attrib.items():
                    if isinstance(value, Lengths):
                        bounds.update(value)

        for child in self:
            child.update_bounds(bounds)

    def serialize(self, output: TextIO, indent: int = 0):
        _ = output.write(" " * indent)
        _ = output.write(f"<{self.tag} ")
        for key, value in self.attrib.items():
            if isinstance(value, Serializable):
                value = value.serialize()
            if value is not None:
                _ = output.write(f'{key}="{value}" ')

        if len(self) == 0 and self.text is None:
            _ = output.write("/>\n")
        else:
            _ = output.write(">\n")
            if self.text is not None:
                assert isinstance(self.text, str)
                _ = output.write(self.text)
            for child in self:
                child.serialize(output, indent + 2)
            _ = output.write(" " * indent)
            _ = output.write(f"</{self.tag}>\n")

    def _repr_svg_(self) -> str:
        buf = StringIO()
        self.serialize(buf)
        return buf.getvalue()

    def delete_attributes_inplace(self, predicate: Callable[[str, Any], bool]):
        """
        Delete attributes that match a particular predicate.
        """

        self.attrib = {k: v for k, v in self.attrib.items() if not predicate(k, v)}

        for child in self:
            child.delete_attributes_inplace(predicate)

    def traverse_attributes(
        self, visitor: Callable[[str, Any], None], filter_type: type | None
    ):
        """
        Simple functional tree traversal for element trees.
        """

        for attr, value in self.attrib.items():
            if filter_type is None or isinstance(value, filter_type):
                visitor(attr, value)

        for child in self:
            child.traverse_attributes(visitor, filter_type)

    def traverse_elements(self, visitor: Callable[["Element"], None]):
        """
        Simple functional tree traversal for element trees.
        """

        visitor(self)
        for child in self:
            child.traverse_elements(visitor)

    def rewrite_attributes_inplace(
        self, visitor: Callable[[str, object], object], filter_type: type | None = None
    ):
        """
        Rewrite an element tree by applying a function to every elements attribute (optionally, only of a particular type)
        """

        for attr, value in self.attrib.items():
            if filter_type is None or isinstance(value, filter_type):
                self.attrib[attr] = visitor(attr, value)
            else:
                self.attrib[attr] = value

        for child in self:
            child.rewrite_attributes_inplace(visitor, filter_type)

    def rewrite_attributes(
        self, visitor: Callable[[str, object], object], filter_type: type | None = None
    ):
        """
        Rewrite an element tree by applying a function to every elements attribute (optionally, only of a particular type)
        """

        el_rewrite = copy(self)
        el_rewrite.clear()

        el_rewrite.text = self.text
        el_rewrite.attrib = {}
        for attr, value in self.attrib.items():
            if filter_type is None or isinstance(value, filter_type):
                el_rewrite.attrib[attr] = visitor(attr, value)
            else:
                el_rewrite.attrib[attr] = value

        for child in self:
            el_rewrite.append(child.rewrite_attributes(visitor, filter_type))

        return el_rewrite

    def abs_bounds(self) -> tuple[AbsLengths, AbsLengths]:
        """
        Elements can optionally supply lower bound in absolute units, for the amount
        of space they need. This is only necessary for elements positioned on the
        sides of plots, where we need to know how much space to reserve.

        We assume ConfigKeys have been all replaced by this point.
        """

        return mm(0), mm(0)

    def apply_scales(self, _scales: ScaleSet):
        """
        Most elements don't need to implement this, but e.g. if an element needs
        to know what the ticks are, this is a useful way to get that info.
        """
        pass

    def merge_coords(self, new_coords: CoordSet):
        if "dapple:coords" not in self.attrib:
            self.set("dapple:coords", copy(new_coords))
        else:
            coords = cast(CoordSet, self.get_as("dapple:coords", dict))
            coords.update(new_coords)


class RawText(Element):
    """Special quasi-element used entirely for constructing Elements with text contents."""

    data: str

    def __init__(self, text: str):
        self.data = text
        super().__init__("dapple:__text__")

    @override
    def serialize(self, output: TextIO, indent: int = 0) -> None:
        raise NotImplementedError("RawText elements should not be serialized directly.")


@final
class VectorizedElement(Element):
    """
    An element that is is given vector attribute values and is serialized into
    multiple elements. Useful for more efficiently dealing with plot geometry.
    """

    def __init__(
        self,
        tag: str,
        attrib: dict[str, object] | None = None,
        *args: Element,
        **kwargs: object,
    ):
        texts: list[str] = []
        nontext_args: list[Element] = []
        for arg in args:
            if isinstance(arg, RawText):
                texts.append(arg.data)
            else:
                nontext_args.append(arg)

        super().__init__(tag, attrib, *nontext_args, **kwargs)

        if len(texts) > 0:
            self.text = texts

    @override
    def similar(self, attrib: dict[str, object]) -> "VectorizedElement":
        return VectorizedElement(self.tag, attrib)

    @override
    def serialize(self, output: TextIO, indent: int = 0):
        nels = 1
        for _key, value in self.attrib.items():
            if isinstance(value, (AbsLengths, Colors)):
                if len(value) > 1:
                    if nels != len(value):
                        if nels > 1:
                            raise Exception(
                                "VectorizedElement attribute has inconsistent lengths"
                            )
                        nels = len(value)

        texts: Iterable[str | None]
        if self.text is not None:
            assert isinstance(self.text, list)
            if len(self.text) != 1 and len(self.text) != nels:
                raise Exception("VectorizedElement text has inconsistent lengths")

            if len(self.text) == 1:
                texts = repeat(self.text[0], nels)
            else:
                texts = self.text

        else:
            texts = repeat(None, nels)

        keys: list[str] = []
        value_iters: list[Iterable[str]] = []
        for key, value in self.attrib.items():
            keys.append(key)
            if isinstance(value, Serializable):
                svalue = value.serialize()
                if isinstance(svalue, str):
                    value_iters.append(repeat(svalue, nels))
                elif isinstance(svalue, list):
                    value_iters.append(svalue)
                else:
                    assert svalue is None
            else:
                value_iters.append(repeat(str(value), nels))

        for text, *values in zip(texts, *value_iters):
            _ = output.write(" " * indent)
            _ = output.write(f"<{self.tag} ")
            for k, v in zip(keys, values):
                _ = output.write(f'{k}="{v}" ')

            if len(self) == 0 and text is None:
                _ = output.write("/>\n")
            else:
                _ = output.write(">\n")
                if text is not None:
                    assert isinstance(text, str)
                    _ = output.write(text)
                for child in self:
                    child.serialize(output, indent + 2)
                _ = output.write(" " * indent)
                _ = output.write(f"</{self.tag}>\n")


class PathData(Serializable):
    """Generates SVG path data string from resolved coordinates."""

    def __init__(self, x_coords: AbsLengths, y_coords: AbsLengths):
        if len(x_coords.values) < 2 and len(y_coords.values) < 2:
            raise ValueError("Path must have at least 2 points")

        if len(x_coords.values) != len(y_coords.values) and not (
            x_coords.isscalar() or y_coords.isscalar()
        ):
            raise ValueError(
                "PathData must x and y coordinates must be the same length"
            )

        self.x_coords = x_coords
        self.y_coords = y_coords

    def serialize(self) -> str:
        """Generate SVG path string from coordinates."""
        if len(self.x_coords.values) == 0:
            return ""

        # Convert coordinates to strings
        path_parts = []

        # Move to first point
        x0, y0 = self.x_coords.values[0], self.y_coords.values[0]
        path_parts.append(f"M {x0:.3f} {y0:.3f}")

        x_it = (
            repeat(self.x_coords.values[0])
            if self.x_coords.isscalar()
            else self.x_coords.values[1:]
        )

        y_it = (
            repeat(self.y_coords.values[0])
            if self.y_coords.isscalar()
            else self.y_coords.values[1:]
        )

        # Line to subsequent points
        for x, y in zip(x_it, y_it):
            path_parts.append(f"L {x:.3f} {y:.3f}")

        return " ".join(path_parts)


class Path(Element):
    """Path element that resolves to SVG path with proper coordinate handling."""

    def __init__(self, x_coords, y_coords, **kwargs):
        super().__init__(tag="dapple:path")
        self.attrib = {"x": x_coords, "y": y_coords, **kwargs}

    def resolve(self, ctx: ResolveContext) -> Element:
        """Resolve to SVG path element."""

        resolved_attrib = resolve(self.attrib, ctx)

        x_coords = resolved_attrib.pop("x")
        y_coords = resolved_attrib.pop("y")

        # Generate path data
        path_data = PathData(x_coords, y_coords)

        # Create SVG path element with resolved attributes
        return Element("path", {"d": path_data, **resolved_attrib})


def viewport(
    children: Iterable[Element],
    x: Lengths | None = None,
    y: Lengths | None = None,
    width: Lengths | None = None,
    height: Lengths | None = None,
) -> Element:
    if x is None:
        x = mm(0)
    if y is None:
        y = mm(0)

    if width is None:
        width = vw(1) - x
    if height is None:
        height = vh(1) - y

    return Element(
        "g",
        {
            "dapple:coords": {
                "vw": CoordTransform(width, mm(0)),
                "vh": CoordTransform(height, mm(0)),
            },
            "transform": translate(x, y),
        },
        *children,
    )


def pad(
    el: Element,
    padding: None | AbsLengths = None,
    top: None | AbsLengths = None,
    right: None | AbsLengths = None,
    bottom: None | AbsLengths = None,
    left: None | AbsLengths = None,
) -> Element:
    if padding is None:
        padding = mm(0)

    if top is None:
        top = padding
    if right is None:
        right = padding
    if bottom is None:
        bottom = padding
    if left is None:
        left = padding

    return viewport([el], left, top, vwv(1) - left - right, vhv(1) - top - bottom)
