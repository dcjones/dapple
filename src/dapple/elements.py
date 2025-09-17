
from xml.etree.ElementTree import Element
from .coordinates import CoordTransform, Resolvable, CoordSet, AbsCoordSet, AbsLengths, Lengths, Transform, ResolveContext, resolve, mm, cw, ch, translate
from .occupancy import Occupancy
from .colors import Colors
from . import svg
from typing import Any, Collection, Callable, Optional, Iterable
from itertools import cycle
from functools import singledispatch
from copy import copy
from abc import ABC, abstractmethod


def delete_attributes_inplace(el: Element, predicate: Callable[[str, Any], bool]):
    """
    Delete attributes that match a particular predicate.
    """

    el.attrib = {k: v for k, v in el.attrib.items() if not predicate(k, v)}

    for child in el:
        delete_attributes_inplace(child, predicate)


def traverse_attributes(el: Element, visitor: Callable[[str, Any], None], filter_type: Optional[type]):
    """
    Simple functional tree traversal for element trees.
    """

    for attr, value in el.attrib.items():
        if filter_type is None or isinstance(value, filter_type):
            visitor(attr, value)

    for child in el:
        traverse_attributes(child, visitor, filter_type)


def rewrite_attributes_inplace(el: Element, visitor: Callable[[str, Any], Any], filter_type: Optional[type]=None):
    """
    Rewrite an element tree by applying a function to every elements attribute (optionally, only of a particular type)
    """

    for attr, value in el.attrib.items():
        if filter_type is None or isinstance(value, filter_type):
            el.attrib[attr] = visitor(attr, value)
        else:
            el.attrib[attr] = value

    for child in el:
        rewrite_attributes_inplace(child, visitor, filter_type)


def rewrite_attributes(el: Element, visitor: Callable[[str, Any], Any], filter_type: Optional[type]=None):
    """
    Rewrite an element tree by applying a function to every elements attribute (optionally, only of a particular type)
    """

    el_rewrite = copy(el)
    el_rewrite.clear()

    el_rewrite.text = el.text
    el_rewrite.attrib = {}
    for attr, value in el.attrib.items():
        if filter_type is None or isinstance(value, filter_type):
            el_rewrite.attrib[attr] = visitor(attr, value)
        else:
            el_rewrite.attrib[attr] = value

    for child in el:
        el_rewrite.append(rewrite_attributes(child, visitor, filter_type))

    return el_rewrite


def abs_bounds(el: Element) -> tuple[AbsLengths, AbsLengths]:
    """
    Elements can optionally supply lower bound in absolute units, for the amount
    of space they need. This is only necessary for elements positioned on the
    sides of plots, where we need to know how much space to reserve.
    """

    if isinstance(el, ResolvableElement):
        return el.abs_bounds()
    elif isinstance(el, Element):
        return mm(0), mm(0)
    else:
        raise TypeError(f"Unsupported element type: {type(el)}")


class ResolvableElement(Element, Resolvable):
    """
    An XML element that can be resolved (unually into another element).
    """

    def __init__(self, tag: str, attrib={}, **extra):
        super().__init__(tag, attrib, **extra)

    def resolve(self, ctx: ResolveContext) -> Element:
        """
        Convert the resolvable element into a regular svg element by providing
        context on absolute sizes ond occupancy.
        """

        attrib = {k: resolve(v, ctx) for (k, v) in self.attrib.items()}
        el = Element(self.tag, attrib)

        for child in self:
            el.append(resolve(child, ctx))

        return el

    def abs_bounds(self) -> tuple[AbsLengths, AbsLengths]:
        return mm(0), mm(0)


class VectorizedElement(ResolvableElement):
    """
    An element that is is given vector attribute values and is resolved into multiple single lengths.
    """

    def __init__(self, tag: str, attrib={}, **extra):
        super().__init__(tag, attrib, **extra)

    def resolve(self, ctx: ResolveContext) -> Element:
        # TODO: Actually, I think we shouldn't expand this when we resolve. Instead
        # we should do this when we write to xml.

        g = svg.g()

        keys = []
        value_iters = []
        veclen = None
        for (key, value) in self.attrib.items():
            keys.append(key)
            if isinstance(value, (Lengths, Colors, Collection)):
                if veclen is None:
                    veclen = len(value)
                else:
                    assert veclen == len(value), f"VectorizedElement attribute {key} has length {len(value)}, but previous attribute has length {veclen}"
            else:
                value_iters.append(cycle(value))

        if veclen is None:
            raise ValueError("VectorizedElement must have vector attributes")

        resolved_children = [
            resolve(child, ctx)
            for child in self
        ]

        for values in zip(*value_iters):
            attrib = {k: v for (k, v) in zip(keys, values)}
            el = Element(self.tag, attrib)
            el.extend(resolved_children)
            g.append(el)

        return g


class ViewportElement(ResolvableElement):
    """
    A special container element used as a shortcut for defining the special `vw`
    and `vh` units. The primary use case is to define a particular region using
    a bounding box (though bounds are not anywhere enforced), and setting up
    geometry relative to that bounding box using (vw, vh) units.
    """

    def __init__(self, x: Lengths, y: Lengths, width: Lengths, height: Lengths):
        x.assert_scalar()
        y.assert_scalar()
        width.assert_scalar()
        height.assert_scalar()

        # The idea here is to push the translation part to an actual svg
        # transform. That way everything gets translated the same way and we
        # don't have to prefix everything with a 0vw + ..., 0vh + ...
        attribs = {
            "dapple:coords": {
                "vw": CoordTransform(width, mm(0)),
                "vh": CoordTransform(height, mm(0)),
            },


            # TODO: Ahhh, this has to be converted to as SVG transform string
            "transform": translate(x, y)
        }

        super().__init__("g", attribs)

    def merge_coords(self, new_coords: CoordSet):
        if "dapple:coords" not in self.attrib:
            self.set("dapple:coords", copy(new_coords))
        else:
            coords = self.get("dapple:coords")
            assert isinstance(coords, dict)
            coords.update(new_coords)


def viewport(children: Iterable[Element], x: Lengths=mm(0), y: Lengths=mm(0), width: Optional[Lengths]=None, height: Optional[Lengths]=None) -> ViewportElement:
    if width is None:
        width = cw(1) - x
    if height is None:
        height = ch(1) - y

    vp = ViewportElement(x, y, width, height)
    for el in children:
        vp.append(el)
    return vp
