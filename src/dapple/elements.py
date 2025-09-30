from .coordinates import (
    CoordTransform,
    CoordBounds,
    Resolvable,
    CoordSet,
    AbsCoordSet,
    AbsLengths,
    Lengths,
    Transform,
    ResolveContext,
    Serializable,
    resolve,
    mm,
    vw,
    vh,
    translate,
)
from .occupancy import Occupancy
from .colors import Colors
from .scales import ScaleSet
from typing import (
    Any,
    Collection,
    TextIO,
    Callable,
    Optional,
    Iterable,
    TypeVar,
    override,
)
from itertools import repeat
from functools import singledispatch
from copy import copy

AttrType = TypeVar("AttrType")


class Element(Resolvable):
    """
    This is an XML Element representation that mimics the standard library etree
    Element, but permits non-string attribute values.
    """

    tag: str
    attrib: dict[str, object]
    text: str | None
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

    def get_as(self, key: str, expected_type: type[AttrType]) -> AttrType:
        value = self.attrib.get(key)
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
                # TODO: For this to work we need to support scalar + vector length operations :(

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
            _ = output.write(f"/>\n")
        else:
            _ = output.write(f">\n")
            if self.text is not None:
                _ = output.write(self.text)
            for child in self:
                child.serialize(output, indent + 2)
            _ = output.write(" " * indent)
            _ = output.write(f"</{self.tag}>\n")

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

    def apply_scales(self, scales: ScaleSet):
        """
        Most elements don't need to implement this, but e.g. if an element needs
        to know what the ticks are, this is a useful way to get that info.
        """
        pass

    def merge_coords(self, new_coords: CoordSet):
        if "dapple:coords" not in self.attrib:
            self.set("dapple:coords", copy(new_coords))
        else:
            coords = self.get("dapple:coords")
            coords.update(new_coords)


class VectorizedElement(Element):
    """
    An element that is is given vector attribute values and is serialized into
    multiple elements. Useful for more efficiently dealing with plot geometry.
    """

    def __init__(self, tag: str, attrib={}, **extra):
        super().__init__(tag, attrib, **extra)

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

        keys = []
        value_iters = []
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

        for values in zip(*value_iters):
            _ = output.write(" " * indent)
            _ = output.write(f"<{self.tag} ")
            for k, v in zip(keys, values):
                _ = output.write(f'{k}="{v}" ')

            if len(self) == 0:
                _ = output.write(f"/>\n")
            else:
                _ = output.write(f">\n")
                for child in self:
                    child.serialize(output, indent + 2)
                _ = output.write(" " * indent)
                _ = output.write(f"</{self.tag}>\n")


def viewport(
    children: Iterable[Element],
    x: Lengths = mm(0),
    y: Lengths = mm(0),
    width: Optional[Lengths] = None,
    height: Optional[Lengths] = None,
) -> Element:
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
