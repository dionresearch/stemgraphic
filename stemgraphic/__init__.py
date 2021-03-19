#!/usr/bin/env python3
"""stem_graphic.

Package implementing a complete toolkit for text and a graphical stem-and-leaf plots and other visualizations adapted
to stem-and-leaf pair values, such as heatmaps and sunburst charts.

It also handles very large data sets through scaling, sampling, trimming and other techniques.

See research paper ( http://artchiv.es/pydata2016/stemgraphic ) for more technical details.

A command line utility was installed along with the package, allowing to process excel or csv
files. See: stem -h
"""
# flake8: noqa F401
from .aliases import stem_hist, stem_kde, stem_line, stem_dot, stem_symmetric_dot  # noqa F401
from .graphic import stem_graphic, heatmap  # noqa F401
from .text import stem_data, stem_text, heatmatrix  # noqa F401
from .helpers import dd  # noqa F401
from .stopwords import *  # noqa F401

__version__ = "0.9.1"