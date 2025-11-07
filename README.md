
# Dapple

This in an experimental python plotting package that's in active development. At
this point it's a just personal side project, but one I am actively dogfooding.
There are two major goals:

  1. Build on ideas I figured out writing
     [Gadfly](https://github.com/GiovineItalia/Gadfly.jl) (and another Julia
     plotting library which I wrote and used, but never released.)
  2. Experiment with LLM agents to accelerate implementation of specific features.

# Design

You can think of dapple as a compiler that compiles a superset of SVG down to
normal, standards compliant SVG. The core idea is that plots are declared as if
they were an SVG document. With elements representing plot geometry attached at
will. This allows the freedom of reusing parts of the plot, and not having to
set up an output target in advance.

In addition to defining custom elements as needed, we add to SVG a sophisticated
coordinate system that's solved using linear programming once the plot is rendered.

# Features

A lot of what I prioritize implementing are features that help with precise
layout and typography and those that are useful in my work.

  * Font discovery and extents calculations.
  * Optional GPU rasterization of points, polygons, and such.
  * Support for many color schemes and optimally distinguishable discrete color scales.

Not yet implemented, but high priority.
  * Subplots
  * Point size scales.
  * Non-linear scales
  * Fancier label formatting
  * Label placement optimization
