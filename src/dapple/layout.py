
from enum import Enum
from .coordinates import AbsLengths, Lengths, mm, vw, vh

class Position(Enum):
    """
    Represents relative positioning of a plot element, to be arranged by the
    layout function.

    Two types of positioning are supported:
    1. Z-layer positioning (Above/Below): Elements are layered behind or in front
       of the main plot content, but within the same spatial area.
    2. Spatial positioning (Top/Bottom/Left/Right): Elements are positioned
       outside the main plot area in dedicated margin spaces.

    Use spatial positions (BottomCenter, LeftCenter, etc.) for axis labels and
    titles that need their own space. Use z-layer positions (Above, Below) for
    annotations and overlays within the plot area.
    """

    Default = 1 # place in the panel, with unspecified order

    # Z-layer positions - same spatial area as plot, different z-order
    Above = 2      # In front of main plot content (higher z-layer)
    AboveTopLeft = 3
    AboveTopRight = 4
    AboveBottomLeft = 5
    AboveBottomRight = 6

    Below = 7      # Behind main plot content (lower z-layer)
    BelowTopLeft = 8
    BelowTopRight = 9
    BelowBottomLeft = 10
    BelowBottomRight = 11

    # Spatial positions - outside main plot area in margin spaces
    BottomLeft = 12    # Bottom margin, left-aligned
    BottomCenter = 13  # Bottom margin, center-aligned (ideal for x-axis labels)
    BottomRight = 14   # Bottom margin, right-aligned

    TopLeft = 15       # Top margin, left-aligned
    TopCenter = 16     # Top margin, center-aligned (ideal for titles)
    TopRight = 17      # Top margin, right-aligned

    LeftTop = 18       # Left margin, top-aligned
    LeftCenter = 19    # Left margin, center-aligned (ideal for y-axis labels)
    LeftBottom = 20    # Left margin, bottom-aligned

    RightTop = 21      # Right margin, top-aligned
    RightCenter = 22   # Right margin, center-aligned
    RightBottom = 23   # Right margin, bottom-aligned

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
                return vw(1) - width, mm(0)
            case Position.AboveBottomLeft:
                return mm(0), vh(1) - height
            case Position.AboveBottomRight:
                return vw(1) - width, vh(1) - height
            case Position.BelowTopLeft:
                return mm(0), mm(0)
            case Position.BelowTopRight:
                return vw(1) - width, mm(0)
            case Position.BelowBottomLeft:
                return mm(0), vh(1) - height
            case Position.BelowBottomRight:
                return vw(1) - width, vh(1) - height
            case Position.BottomLeft:
                return mm(0), mm(0)
            case Position.BottomCenter:
                return vw(0.5) - 0.5*width, mm(0)
            case Position.BottomRight:
                return vw(1) - width, mm(0)
            case Position.TopLeft:
                return mm(0), mm(0)
            case Position.TopCenter:
                return vw(0.5) - 0.5*width, mm(0)
            case Position.TopRight:
                return vw(1) - width, mm(0)
            case Position.LeftTop:
                return mm(0), mm(0)
            case Position.LeftCenter:
                return mm(0), vh(0.5) - 0.5*height
            case Position.LeftBottom:
                return mm(0), vh(1) - height
            case Position.RightTop:
                return mm(0), mm(0)
            case Position.RightCenter:
                return mm(0), vh(0.5) - 0.5*height
            case Position.RightBottom:
                return mm(0), vh(1) - height
            case _:
                raise ValueError(f"Invalid position: {self}")
