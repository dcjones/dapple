
# Overview

This is a python plotting library called Dapple. The core concept is to build a
SVG element tree that contains additional information such as non-standard
coordinates and elements. This tree is then transformed ("resolved") into a
pure-SVG tree.

# Plotting

The full plotting pass, which is implemented in `plot.py`, works like so:
  1. `ConfigKey` objects in element attributes are replaced using
     values from a `Config` instance.
  2. `UnscaledValues`, which hold raw data to be plotted, are converted
     into`Lengths` or `Colors` according to which scales are present.
  3. Coordinates are applied to transform `Lengths` into `AbsLengths` which
     given absolute coordinates and sizes in millimeters.

Plots are constructed with the `plot` function and drawn with the `Plot.svg`
method, which expects an absolute width and height as well as on output sink.

# Development Commands

- **Run tests**: `uv run pytest`
- **Install dependencies**: `uv sync`
- **Run Python with project dependencies**: `uv run python`

# Core components

- **Elements (`elements.py`)**: Base `Element` class that mimics XML etree Elements but permits non-string attribute values. Elements are `Resolvable` objects that can be transformed during the plotting process.

- **Plot Resolution Pipeline (`plot.py`)**:
  1. `ConfigKey` objects in element attributes are replaced using `Config` values
  2. `UnscaledValues` (raw data) are converted to `Lengths` or `Colors` according to scales
  3. Coordinates transform `Lengths` into `AbsLengths` (absolute mm coordinates)

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
  * Prefer `VectorizedElement` to constructing many similar `Element` objects.
  * `AbsLengths` objects are constructed with the `mm` function.
