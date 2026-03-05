
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
