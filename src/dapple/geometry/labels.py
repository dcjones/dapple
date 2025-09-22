from ..elements import Element
from ..coordinates import AbsLengths, ResolveContext, mm, vw, vh
from ..layout import Position
from ..config import ConfigKey, Config
from ..textextents import Font


class XLabel(Element):
    """
    Draw a text label along the bottom of the plot for the x-axis.
    """
    def __init__(
            self,
            text: str,
            font_family=ConfigKey("label_font_family"),
            font_size=ConfigKey("label_font_size"),
            fill=ConfigKey("label_fill"),
    ):
        attrib: dict[str, object] = {
            "dapple:position": Position.BottomCenter,
            "text": text,
            "font_family": font_family,
            "font_size": font_size,
            "fill": fill,
        }
        super().__init__("dapple:xlabel", attrib) # type: ignore

    def resolve(self, ctx: ResolveContext) -> Element:
        text = self.attrib["text"]
        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]
        fill = self.attrib["fill"]

        assert isinstance(text, str)
        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)

        # Get text extents using Font
        font = Font(font_family, font_size)
        text_width, text_height = font.get_extents(text)

        # Center the text horizontally using text-anchor
        x = vw(0.5)
        y = text_height  # Position from top of the space allocated

        text_element = Element(
            "text", {
                "x": x,
                "y": y,
                "font-family": font_family,
                "font-size": font_size,
                "fill": fill,
                "text-anchor": "middle",  # Center horizontally
            }
        )
        text_element.text = text

        return text_element.resolve(ctx)

    def abs_bounds(self) -> tuple[AbsLengths, AbsLengths]:
        text = self.attrib["text"]
        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]

        assert isinstance(text, str)
        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)

        font = Font(font_family, font_size)
        text_width, text_height = font.get_extents(text)

        return (text_width, text_height)


def xlabel(text: str, *args, **kwargs):
    return XLabel(text, *args, **kwargs)


class YLabel(Element):
    """
    Draw a rotated text label along the left side of the plot for the y-axis.
    """
    def __init__(
            self,
            text: str,
            font_family=ConfigKey("label_font_family"),
            font_size=ConfigKey("label_font_size"),
            fill=ConfigKey("label_fill"),
    ):
        attrib: dict[str, object] = {
            "dapple:position": Position.LeftCenter,
            "text": text,
            "font_family": font_family,
            "font_size": font_size,
            "fill": fill,
        }
        super().__init__("dapple:ylabel", attrib) # type: ignore

    def resolve(self, ctx: ResolveContext) -> Element:
        text = self.attrib["text"]
        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]
        fill = self.attrib["fill"]

        assert isinstance(text, str)
        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)

        # Get text extents using Font
        font = Font(font_family, font_size)
        text_width, text_height = font.get_extents(text)

        # Position the text centered vertically and rotated -90 degrees
        x = text_height  # Distance from left edge
        y = vh(0.5)  # Center vertically

        # Resolve coordinates for the transform attribute
        x_resolved = x.resolve(ctx)
        y_resolved = y.resolve(ctx)

        text_element = Element(
            "text", {
                "x": x,
                "y": y,
                "font-family": font_family,
                "font-size": font_size,
                "fill": fill,
                "text-anchor": "middle",
                "transform": f"rotate(-90, {x_resolved.serialize()}, {y_resolved.serialize()})",
            }
        )
        text_element.text = text

        return text_element.resolve(ctx)

    def abs_bounds(self) -> tuple[AbsLengths, AbsLengths]:
        text = self.attrib["text"]
        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]

        assert isinstance(text, str)
        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)

        font = Font(font_family, font_size)
        text_width, text_height = font.get_extents(text)

        # When rotated -90°, text height becomes the width requirement
        return (text_height, text_width)


def ylabel(text: str, *args, **kwargs):
    return YLabel(text, *args, **kwargs)


class Title(Element):
    """
    Draw a text title at the top center of the plot.
    """
    def __init__(
            self,
            text: str,
            font_family=ConfigKey("title_font_family"),
            font_size=ConfigKey("title_font_size"),
            fill=ConfigKey("title_fill"),
    ):
        attrib: dict[str, object] = {
            "dapple:position": Position.TopCenter,
            "text": text,
            "font_family": font_family,
            "font_size": font_size,
            "fill": fill,
        }
        super().__init__("dapple:title", attrib) # type: ignore

    def resolve(self, ctx: ResolveContext) -> Element:
        text = self.attrib["text"]
        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]
        fill = self.attrib["fill"]

        assert isinstance(text, str)
        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)

        # Get text extents using Font
        font = Font(font_family, font_size)
        text_width, text_height = font.get_extents(text)

        # Center the text horizontally using text-anchor
        x = vw(0.5)
        y = text_height  # Position from top of the space allocated

        text_element = Element(
            "text", {
                "x": x,
                "y": y,
                "font-family": font_family,
                "font-size": font_size,
                "fill": fill,
                "text-anchor": "middle",  # Center horizontally
            }
        )
        text_element.text = text

        return text_element.resolve(ctx)

    def abs_bounds(self) -> tuple[AbsLengths, AbsLengths]:
        text = self.attrib["text"]
        font_family = self.attrib["font_family"]
        font_size = self.attrib["font_size"]

        assert isinstance(text, str)
        assert isinstance(font_family, str)
        assert isinstance(font_size, AbsLengths)

        font = Font(font_family, font_size)
        text_width, text_height = font.get_extents(text)

        return (text_width, text_height)


def title(text: str, *args, **kwargs):
    return Title(text, *args, **kwargs)
