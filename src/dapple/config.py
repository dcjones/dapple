from dataclasses import dataclass, field
from typing import Any
from .colors import Colors, color
from .coordinates import AbsLengths, mm
from cmap import Colormap


@dataclass
class ConfigKey:
    """
    Reprent a value that should be looked up in the a `Config`
    instance when the plot is resolved.
    """

    key: str


@dataclass
class ChooseTicksParams:
    k_min: int
    k_max: int
    k_ideal: int
    granularity_weight: float
    simplicity_weight: float
    coverage_weight: float
    niceness_weight: float


@dataclass
class Config:
    pointsize: AbsLengths = field(default_factory=lambda: mm(0.4))
    pointcolor: Colors = field(default_factory=lambda: color("#333333"))
    discrete_cmap: Colormap = field(default_factory=lambda: Colormap("colorcet:cet_c6"))
    continuous_cmap: Colormap = field(
        default_factory=lambda: Colormap("colorcet:cet_l20")
    )
    tick_coverage: str = "sub"
    tick_params: ChooseTicksParams = field(
        default_factory=lambda: ChooseTicksParams(
            k_min=2,
            k_max=10,
            k_ideal=5,
            granularity_weight=1 / 4,
            simplicity_weight=1 / 6,
            coverage_weight=1 / 2,
            niceness_weight=1 / 4,
        )
    )
    grid_stroke_width: AbsLengths = field(default_factory=lambda: mm(0.2))
    grid_stroke: Colors = field(default_factory=lambda: color("#dddddd"))
    linecolor: Colors = field(default_factory=lambda: color("#333333"))
    linestroke: AbsLengths = field(default_factory=lambda: mm(0.3))
    barcolor: Colors = field(default_factory=lambda: color("#333333"))
    grid_stroke_dasharray: str = "1"
    tick_stroke_width: AbsLengths = field(default_factory=lambda: mm(0.4))
    tick_stroke: Colors = field(default_factory=lambda: color("#333333"))
    tick_length: AbsLengths = field(default_factory=lambda: mm(2.0))

    # Layout configuration
    # Three predefined padding values are used, representing (typically) no padding,
    # minimal padding, and standard padding.
    padding_nil: AbsLengths = field(default_factory=lambda: mm(0))
    padding_min: AbsLengths = field(default_factory=lambda: mm(0.5))
    padding_std: AbsLengths = field(default_factory=lambda: mm(1.5))

    # Label configuration
    label_font_family: str = "DejaVu Sans"
    label_font_size: AbsLengths = field(default_factory=lambda: mm(3.5))
    label_fill: Colors = field(default_factory=lambda: color("#333333"))

    # Title configuration
    title_font_family: str = "DejaVu Sans"
    title_font_size: AbsLengths = field(default_factory=lambda: mm(4.5))
    title_fill: Colors = field(default_factory=lambda: color("#333333"))

    # Tick label configuration
    tick_label_font_family: str = "DejaVu Sans"
    tick_label_font_size: AbsLengths = field(default_factory=lambda: mm(2.5))
    tick_label_fill: Colors = field(default_factory=lambda: color("#333333"))

    # Key configuration
    key_square_size: AbsLengths = field(default_factory=lambda: mm(4))
    key_spacing: AbsLengths = field(default_factory=lambda: mm(1))
    key_gradient_width: AbsLengths = field(default_factory=lambda: mm(4))

    # Rasterization configuration
    rasterize_dpi: float = 150.0

    def get(self, key: ConfigKey) -> Any:
        return getattr(self, key.key)

    def replace_keys(self, obj: object):
        """
        General puprose method to replace ConfigKey fields in any object with
        their associated value in the Config.
        """

        if isinstance(obj, ConfigKey):
            return self.get(obj)
        elif hasattr(obj, "__dict__"):
            for attr_name, attr_value in obj.__dict__.items():
                if isinstance(attr_value, ConfigKey):
                    setattr(obj, attr_name, self.get(attr_value))
                else:
                    self.replace_keys(attr_value)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self.replace_keys(item)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, ConfigKey):
                    obj[key] = self.get(value)
                else:
                    self.replace_keys(value)
        return obj
