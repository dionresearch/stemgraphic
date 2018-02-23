""" stemgraphic.num.

BRAND NEW in V.0.5.0!

Stemgraphic provides a complete set of functions to handle everything related to stem-and-leaf plots. num is a
module of the stemgraphic package to handle numerical variables.

This module structure is new as of v.0.5.0 to match the addition of stemgraphic.alpha.

The shorthand from previous versions of stemgraphic is still available and defaults to the numerical functions:

  from stemgraphic import stem_graphic, stem_text, heatmap

"""

from .text import (
    stem_data,
    stem_dot,
    stem_text
)

from .graphic import (
    heatmap,
    stem_graphic
)
