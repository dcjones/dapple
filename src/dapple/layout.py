
from enum import Enum
from .coordinates import AbsLengths, Lengths, mm, vw, vh

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
