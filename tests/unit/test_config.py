import pytest
import tempfile
from pathlib import Path
from dapple.config import (
    Config,
    default_config,
    _load_config_file,
    _parse_config_value,
    ChooseTicksParams,
)
from dapple.coordinates import mm, cm, pt, inch
from dapple.colors import color
from cmap import Colormap


def test_config_default_values():
    """Test that Config() returns default values without loading from file."""
    config = Config()
    assert config.plot_width == mm(100)
    assert config.plot_height == mm(75)
    assert config.pointsize == mm(0.4)
    assert config.label_font_family == "DejaVu Sans"
    assert config.label_font_weight == "normal"
    assert config.title_font_weight == "normal"
    assert config.tick_label_font_weight == "normal"


def test_config_load_from_file():
    """Test loading config from a specific file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("""
plot_width = "120mm"
plot_height = "90mm"
pointsize = "0.5cm"
pointcolor = "#2E86AB"
discrete_cmap = "colorcet:cet_c7"
label_font_family = "Helvetica"
label_font_weight = "bold"
rasterize_dpi = 200.0
        """)
        f.flush()
        temp_path = Path(f.name)

    try:
        config = Config.load(temp_path)
        assert config.plot_width == mm(120)
        assert config.plot_height == mm(90)
        assert config.pointsize == cm(0.5)
        assert str(config.pointcolor) == str(color("#2E86AB"))
        assert config.discrete_cmap.name == "colorcet:cet_c7"
        assert config.label_font_family == "Helvetica"
        assert config.label_font_weight == "bold"
        assert config.rasterize_dpi == 200.0
    finally:
        temp_path.unlink()


def test_config_load_no_file():
    """Test that Config.load() returns defaults when no file exists."""
    config = Config.load("/nonexistent/path/config.toml")
    # Should return defaults without error
    assert config.plot_width == mm(100)


def test_parse_config_value_lengths():
    """Test parsing length values from strings."""
    assert _parse_config_value("plot_width", "10mm") == mm(10)
    assert _parse_config_value("plot_width", "2.5cm") == cm(2.5)
    assert _parse_config_value("plot_width", "12pt") == pt(12)
    assert _parse_config_value("plot_width", "1inch") == inch(1)


def test_parse_config_value_colors():
    """Test parsing color values from strings."""
    color_keys = ["pointcolor", "linecolor", "grid_stroke", "label_fill", "tick_stroke"]
    for key in color_keys:
        result = _parse_config_value(key, "#FF0000")
        assert str(result) == str(color("#FF0000"))


def test_parse_config_value_colormaps():
    """Test parsing colormap values from strings."""
    result = _parse_config_value("discrete_cmap", "viridis")
    assert isinstance(result, Colormap)
    assert result.name == "bids:viridis"


def test_parse_config_value_tick_params():
    """Test parsing nested tick_params."""
    tick_data = {
        "k_min": 3,
        "k_max": 8,
        "k_ideal": 5,
        "granularity_weight": 0.5,
        "simplicity_weight": 0.3,
        "coverage_weight": 0.6,
        "niceness_weight": 0.2,
    }
    result = _parse_config_value("tick_params", tick_data)
    assert isinstance(result, ChooseTicksParams)
    assert result.k_min == 3
    assert result.k_max == 8
    assert result.granularity_weight == 0.5


def test_parse_config_value_passthrough():
    """Test that non-special values pass through unchanged."""
    assert _parse_config_value("tick_coverage", "sub") == "sub"
    assert _parse_config_value("rasterize_dpi", 150.0) == 150.0
    assert _parse_config_value("grid_stroke_dasharray", "1,2") == "1,2"


def test_default_config_caching():
    """Test that default_config() returns the same instance on repeated calls."""
    # Reset the global cache first
    import dapple.config

    dapple.config._default_config = None

    config1 = default_config()
    config2 = default_config()
    assert config1 is config2


def test_config_with_partial_settings():
    """Test loading a config file with only some settings."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("""
plot_width = "150mm"
pointcolor = "#123456"
        """)
        f.flush()
        temp_path = Path(f.name)

    try:
        config = Config.load(temp_path)
        # Modified values
        assert config.plot_width == mm(150)
        assert str(config.pointcolor) == str(color("#123456"))
        # Default values for unspecified settings
        assert config.plot_height == mm(75)
        assert config.pointsize == mm(0.4)
    finally:
        temp_path.unlink()


def test_config_invalid_key_ignored():
    """Test that invalid config keys are ignored gracefully."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("""
plot_width = "100mm"
invalid_key = "should_be_ignored"
another_invalid = 42
        """)
        f.flush()
        temp_path = Path(f.name)

    try:
        config = Config.load(temp_path)
        assert config.plot_width == mm(100)
        assert not hasattr(config, "invalid_key")
        assert not hasattr(config, "another_invalid")
    finally:
        temp_path.unlink()
