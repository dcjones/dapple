from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from pathlib import Path
import os
import tomllib
from .colors import Colors, color, distinguishable_colors
from .coordinates import AbsLengths, mm, cm, pt, inch
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


def _get_config_paths() -> list[Path]:
    """Returns list of paths to check for config files, in order of priority."""
    paths: list[Path] = []

    # 1. Current directory
    paths.append(Path.cwd() / ".dapplerc.toml")

    # 2. Home directory
    home = Path.home()
    paths.append(home / ".dapplerc.toml")

    # 3. XDG config directory
    xdg_config = Path(os.environ.get("XDG_CONFIG_HOME", str(home / ".config")))
    paths.append(Path(xdg_config) / "dapple" / "config.toml")

    return paths


def _load_config_file() -> dict[str, object] | None:
    """Load config from file if it exists."""
    for path in _get_config_paths():
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return tomllib.load(f)
            except Exception as e:
                import warnings

                warnings.warn(f"Failed to load config from {path}: {e}")
    return None


def _parse_config_value(key: str, value: object) -> object:
    """Convert config file values to proper types."""
    # Handle length values (strings like "2mm", "0.5cm")
    if isinstance(value, str):
        if value.endswith("mm"):
            return mm(float(value[:-2]))
        elif value.endswith("cm"):
            return cm(float(value[:-2]))
        elif value.endswith("pt"):
            return pt(float(value[:-2]))
        elif value.endswith("inch"):
            return inch(float(value[:-4]))

    # Handle color values (strings like "#333333")
    if key.endswith("color") or key.endswith("fill") or key.endswith("stroke"):
        if isinstance(value, str):
            return color(value)

    # Handle colormap values
    if key.endswith("cmap"):
        if isinstance(value, str):
            return Colormap(value)

    # Handle nested ChooseTicksParams
    if key == "tick_params" and isinstance(value, dict):
        return ChooseTicksParams(**value)

    return value


@dataclass
class Config:
    plot_width: AbsLengths = field(default_factory=lambda: mm(100))
    plot_height: AbsLengths = field(default_factory=lambda: mm(75))
    pointsize: AbsLengths = field(default_factory=lambda: mm(0.4))
    pointcolor: Colors = field(default_factory=lambda: color("#333333"))
    # discrete_cmap: Colormap = field(default_factory=lambda: Colormap("colorcet:cet_c6"))
    discrete_cmap: Colormap | Callable[[int], Colormap] = distinguishable_colors
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
    label_font_weight: str = "normal"
    label_fill: Colors = field(default_factory=lambda: color("#333333"))

    # Title configuration
    title_font_family: str = "DejaVu Sans"
    title_font_size: AbsLengths = field(default_factory=lambda: mm(4.8))
    title_font_weight: str = "normal"
    title_fill: Colors = field(default_factory=lambda: color("#333333"))

    # Tick label configuration
    tick_label_font_family: str = "DejaVu Sans"
    tick_label_font_size: AbsLengths = field(default_factory=lambda: mm(2.5))
    tick_label_font_weight: str = "normal"
    tick_label_fill: Colors = field(default_factory=lambda: color("#333333"))

    # Key configuration
    key_square_size: AbsLengths = field(default_factory=lambda: mm(2.5))
    key_spacing: AbsLengths = field(default_factory=lambda: mm(1))
    key_gradient_width: AbsLengths = field(default_factory=lambda: mm(4))

    # Heatmap configuration
    heatmap_nudge: AbsLengths = field(default_factory=lambda: mm(0.05))

    # Rasterization configuration
    rasterize_dpi: float = 150.0

    @staticmethod
    def load(path: Optional[Path | str] = None) -> "Config":
        """
        Load config from a file. If no path is provided, searches standard locations.
        """
        if path is not None:
            # Load from specific file
            path = Path(path)
            if not path.exists():
                # File doesn't exist, return defaults
                return Config()
            with open(path, "rb") as f:
                config_data = tomllib.load(f)
        else:
            # Load from standard locations
            config_data = _load_config_file()
            if config_data is None:
                # No config file found, return defaults
                return Config()

        config = Config()
        for key, value in config_data.items():
            if hasattr(config, key):
                parsed_value = _parse_config_value(key, value)
                setattr(config, key, parsed_value)

        return config

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


# Global default config loaded from file
_default_config: Optional[Config] = None


def default_config() -> Config:
    """
    Returns the default config, loading from file if not already loaded.
    This is cached so the file is only read once per session.
    """
    global _default_config
    if _default_config is None:
        _default_config = Config.load()
    return _default_config
