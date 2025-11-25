import pytest

from dapple import plot
from dapple.coordinates import mm
from dapple.geometry.labels import TextLabels, labels
from dapple.scales import xcontinuous, ycontinuous


def test_labels_creation():
    """Test basic creation of labels element."""
    l = labels(x=[1, 2], y=[3, 4], text=["A", "B"])
    assert isinstance(l, TextLabels)
    assert l.tag == "text"
    assert l.attrib["text-anchor"] == "middle"
    assert l.attrib["dominant-baseline"] == "middle"
    assert l.text == ["A", "B"]


def test_labels_single_text():
    """Test labels with a single string."""
    l = labels(x=[1], y=[2], text="Hello")
    assert l.text == ["Hello"]


def test_labels_custom_style():
    """Test labels with custom styling parameters."""
    l = labels(
        x=[1],
        y=[1],
        text=["A"],
        anchor="start",
        alignment_baseline="hanging",
        font_size=mm(5),
        font_family="Arial",
    )
    assert l.attrib["text-anchor"] == "start"
    assert l.attrib["dominant-baseline"] == "hanging"
    assert l.attrib["font-size"] == mm(5)
    assert l.attrib["font-family"] == "Arial"


def test_labels_integration():
    """Test full integration in a plot."""
    p = plot(
        labels(x=[0, 1], y=[0, 1], text=["Start", "End"]), xcontinuous(), ycontinuous()
    )

    # Render to SVG
    output = str(p.svg(width=mm(100), height=mm(100)))

    # Check that text elements are present
    assert "Start" in output
    assert "End" in output
    assert 'text-anchor="middle"' in output
    assert 'dominant-baseline="middle"' in output


def test_labels_alignment():
    """Test that different alignments are passed through correctly."""
    l_start = labels(x=[0], y=[0], text="A", anchor="start")
    assert l_start.attrib["text-anchor"] == "start"

    l_end = labels(x=[0], y=[0], text="A", anchor="end")
    assert l_end.attrib["text-anchor"] == "end"

    l_base = labels(x=[0], y=[0], text="A", alignment_baseline="central")
    assert l_base.attrib["dominant-baseline"] == "central"
