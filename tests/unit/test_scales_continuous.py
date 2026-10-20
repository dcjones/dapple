
import numpy as np
import pytest
from dapple.scales import (
    ScaleContinuousColor,
    ScaleContinuousLength,
    ScaleContinuousSize,
    UnscaledValues,
    colorcontinuous,
    xcontinuous,
    sizecontinuous,
)
from dapple.coordinates import mm, LengthsSequence

def test_scale_continuous_color_min_max():
    # Use factory
    scale = colorcontinuous(colormap="RdBu", min=0, max=10)
    assert scale.min == 0
    assert scale.max == 10
    
    uv = UnscaledValues("color", [2, 5, 8])
    scale.fit_values(uv)
    assert scale.min == 0
    assert scale.max == 10
    
    # Expand
    scale.fit_values(UnscaledValues("color", [-5, 15]))
    assert scale.min == -5
    assert scale.max == 15

def test_scale_continuous_color_mid():
    # Mid only
    scale = colorcontinuous(colormap="RdBu", mid=0)
    scale.fit_values(UnscaledValues("color", [1, 5]))
    assert scale.min == -5
    assert scale.max == 5
    
    # Mid and min/max
    scale = colorcontinuous(colormap="RdBu", min=-1, max=1, mid=0)
    scale.fit_values(UnscaledValues("color", [0.5]))
    assert scale.min == -1
    assert scale.max == 1
    
    scale.fit_values(UnscaledValues("color", [2]))
    assert scale.min == -2
    assert scale.max == 2

def test_scale_continuous_length_min_max():
    # xcontinuous factory uses ScaleContinuousLength
    scale = xcontinuous(min=10, max=20)
    assert scale.min == 10
    assert scale.max == 20
    
    uv = UnscaledValues("x", [15])
    scale.fit_values(uv)
    assert scale.min == 10
    assert scale.max == 20
    
    scale.fit_values(UnscaledValues("x", [25]))
    assert scale.max == 25

def test_scale_continuous_size_min_max():
    # sizecontinuous factory uses ScaleContinuousSize
    scale = sizecontinuous(min=0, max=100)
    assert scale.min == 0
    assert scale.max == 100
    
    uv = UnscaledValues("size", [50])
    scale.fit_values(uv)
    assert scale.min == 0
    assert scale.max == 100
    
    scale.fit_values(UnscaledValues("size", [150]))
    assert scale.max == 150


def test_tick_coverage_parses_strings():
    from dapple.scales import TickCoverage

    assert xcontinuous("sub").tick_coverage == TickCoverage.StrictSub
    assert xcontinuous("Super").tick_coverage == TickCoverage.StrictSuper
    assert xcontinuous("flexible").tick_coverage == TickCoverage.Flexible
    assert xcontinuous(TickCoverage.StrictSub).tick_coverage == TickCoverage.StrictSub


def test_tick_coverage_rejects_invalid_string():
    with pytest.raises(ValueError, match="Invalid tick coverage"):
        xcontinuous("everything")


def test_tick_coverage_from_config_is_parsed():
    import numpy as np
    from dataclasses import replace

    from dapple.config import Config
    from dapple.scales import TickCoverage

    data = UnscaledValues("x", np.array([0.3, 9.7]))
    for coverage, expected in [
        ("sub", TickCoverage.StrictSub),
        ("super", TickCoverage.StrictSuper),
    ]:
        config = replace(Config(), tick_coverage=coverage)
        scale = xcontinuous()
        config.replace_keys(scale)
        scale.fit_values(data)
        _, ticks = scale.ticks()
        assert scale.tick_coverage == expected
        if expected == TickCoverage.StrictSub:
            assert ticks.values[0] >= 0.3 and ticks.values[-1] <= 9.7
        else:
            assert ticks.values[0] <= 0.3 and ticks.values[-1] >= 9.7


def test_default_tick_coverage_is_flexible():
    from dapple.config import Config
    from dapple.scales import TickCoverage

    scale = xcontinuous()
    Config().replace_keys(scale)
    scale.fit_values(UnscaledValues("x", [0.3, 9.7]))
    scale.ticks()
    assert scale.tick_coverage == TickCoverage.Flexible
