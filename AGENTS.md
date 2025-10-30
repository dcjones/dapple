
# Overview

This is a python plotting library called Dapple. The core concept is to build a
SVG element tree that contains additional information such as non-standard
coordinates and elements. This tree is then transformed ("resolved") into a
pure-SVG tree.

# Plotting

Plots are constructed with the `plot` function and drawn with the `Plot.svg`
method, which expects an absolute width and height as well as on output sink.

The typical way of defining a plot is to pass various elements and scales to `plot`, like
`pl = plot(line(x=[0, 1], y=[0, 1]), xcontinuous(), ycontinuous(), title("Hello"))`

But elements can also be appended after the plot is instantiated, like:
`pl.append(points(x=[0, 1], y=[0, 1]))`

# Development Commands

- **Run tests**: `uv run pytest`
- **Install dependencies**: `uv sync`
- **Run Python with project dependencies**: `uv run python`

# Core components

- **Elements (`elements.py`)**: Base `Element` class that mimics XML etree Elements but permits non-string attribute values. Elements are `Resolvable` objects that can be transformed during the plotting process.

  * `VectorizedElement` implements a scheme for generating many of the same type of element in sequence. For efficiency reasons, this should be preferred when possible to instantiating many Elements with the same tag.
  * `Path` is resolved to a SVG path element, but holds coordinates in a way that is legible to the plotting pipeline.
  * `viewport` constructs a 'g' Element with special attributes describing positioning and coordinate transformations.

- **Plot Resolution Pipeline (`plot.py`)**:
  1. `ConfigKey` (`config.py`) objects in element attributes are replaced using values stored in a  `Config` object.
  2. `UnscaledValues` objects wrapping raw data are converted to `Lengths` or `Colors` according to what `Scale` objects are attached the plot.
  3. Plot is laid out, computing the positioning of geometry like labels, titles, and axis ticks.
  4. Coordinates transform of `Lengths` into `AbsLengths` (absolute mm coordinates). This requires first determining how to transform the scaled values. Custom Elements must implement the `update_bounds` function to inform this system of possible minimum and maximum positions..

  * Because plot element trees go through these transformations, it's critical that Elements store
important parameters such as colors and lengths in the attribute dictionary, rather than as members of the class, so they can be discovered and rewritten by this pipeline.
  * To be discovered and mapped to meaningful units, these attributes should initially be wrapped in `UnscaledValues` objects. The `length_params` and `color_params` functions provide shortcuts for doing this.

- **Coordinates (`coordinates.py`)**: Handles coordinate systems and transformations. Provides coordinate units like `mm`, `cm`, `pt`, `inch`, `cx`, `cy`, `vw`, `vh` for absolute and relative positioning.

- **Geometry (`geometry/`)**: Contains specific plot elements:
  - `grids.py`: Grid lines and background elements
  - `labels.py`, `ticks.py`: Axis labeling and tick marks
  - `points.py`, `rasterized_points.py`: Point plotting implementations
  - `lines.py`: Basic lines and univariate function plotting.
  - `key.py`: Legend/key functionality
  - `image.py`: Image handling

- **Scales (`scales.py`)**: Handles data scaling for continuous/discrete color and length mappings

- **Export (`export.py`)**: SVG to PNG/PDF conversion utilities

# Basic guidelines when writing code.

  * In `Elements` subtypes parameters should be stored in the `attrib` dict
    for them to be correctly processed by the resolve transforms.
  * Use the "dapple:" prefix for tag names of non-SVG element types and
    for non-SVG attributes of SVG element types.
  * Do write docstrings, but don't write external documentation files unless
    explicitly asked to.
  * Do write tests when new features are added. Tests belong in the `tests`
    directory.
  * Use type annotations whenever possible.
  * Don't add comments to code that is fairly straightforward.
  * Put simple examples of new features in the `debug` directory.
  * `AbsLengths` objects are constructed with the `mm` function.
  * SVG rect elements don't support negative widths or height which can conflict
    with plot coordinates being flipped by default. Prefer the `Bar` element defined in `bars.py`.
