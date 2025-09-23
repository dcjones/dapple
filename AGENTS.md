This is a python plotting library called Dapple. The core concept is to build a
SVG element tree that contains additional information such as non-standard
coordinates and elements. This tree is then transformed ("resolved") into a
pure-SVG tree.

Follow some basic guidelines when writing code.

  * Do write docstrings, but don't write external documentation files unless
    explicitly asked to.
  * Do write tests when new features are added. Tests belong in the `tests`
    directory and are run with `uv run pytest`.
  * Use type annotations whenever possible.
  * Don't add comments to code that is fairly straightforward.
  * Put simple examples of features in the `debug` directory.
  * Prefer `VectorizedElement` to constructing many similar `Element` objects.
  * `AbsLengths` objects are constructed with `mm`.
