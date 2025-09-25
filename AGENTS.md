This is a python plotting library called Dapple. The core concept is to build a
SVG element tree that contains additional information such as non-standard
coordinates and elements. This tree is then transformed ("resolved") into a
pure-SVG tree.

The full plotting pass, which is implemented in `plot.py`, works like so:
  1. `ConfigKey` objects in element attributes are replaced using
     values from a `Config` instance.
  2. `UnscaledValues`, which hold raw data to be plotted, are converted
     into`Lengths` or `Colors` according to which scales are present.
  3. Coordinates are applied to transform `Lengths` into `AbsLengths` which
     given absolute coordinates and sizes in millimeters.

Follow some basic guidelines when writing code.

  * In `Elements` subtypes parameters should be stored in the `attrib` dict
    for them to be correctly processed by the resolve transforms.
  * Use the "dapple:" prefix for tag names of non-SVG element types and
    for non-SVG attributes of SVG element types.
  * Do write docstrings, but don't write external documentation files unless
    explicitly asked to.
  * Do write tests when new features are added. Tests belong in the `tests`
    directory and are run with `uv run pytest`.
  * Use type annotations whenever possible.
  * Don't add comments to code that is fairly straightforward.
  * Put simple examples of new features in the `debug` directory.
  * Prefer `VectorizedElement` to constructing many similar `Element` objects.
  * `AbsLengths` objects are constructed with the `mm` function.
