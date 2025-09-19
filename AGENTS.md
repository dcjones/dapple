This is a python plotting library called Dapple. The core concept is to build a
SVG element tree (with the ElementTree API) that contains additional information
such as non-standard coordinates and elements. This tree is then transformed
("resolved") into a pure-SVG tree.

Follow some basic guidelines when writing code.

  * Do write docstrings, but don't write external documentation files unless
    explicitly asked to.
  * Do write tests when new features are added. Tests belong in the `tests`
    directory and are run with `uv run pytest`.
  * Use type annotations whenever possible.
  * Don't add comments to code that is fairly straightforward.
