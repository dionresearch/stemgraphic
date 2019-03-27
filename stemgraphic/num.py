""" stemgraphic.num.

BRAND NEW in V.0.5.0!

Stemgraphic provides a complete set of functions to handle everything related to stem-and-leaf plots. num is a
module of the stemgraphic package to handle numerical variables.

This module structure is new as of v.0.5.0 to match the addition of stemgraphic.alpha.

The shorthand from previous versions of stemgraphic is still available and defaults to the numerical functions:

  from stemgraphic import stem_graphic, stem_text, heatmap

"""

from .text import (
    heatmatrix,
    heatmap as text_heatmap,
    quantize,
    stem_data,
    stem_dot as text_dot,
    stem_hist as text_hist,
    stem_tally,
    stem_text,
)

from .graphic import density_plot, heatmap, leaf_scatter, stem_graphic
