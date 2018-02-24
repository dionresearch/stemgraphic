#!/usr/bin/env python3
"""stem_graphic

Package implementing a complete toolkit for text and a graphical stem-and-leaf plots and other visualizations adapted
to stem-and-leaf pair values, such as heatmaps and sunburst charts.

It also handles very large data sets through scaling, sampling, trimming and other techniques.

See research paper ( http://artchiv.es/pydata2016/stemgraphic ) for more technical details.

A command line utility was installed along with the package, allowing to process excel or csv
files. See: stem -h
"""
from .aliases import stem_hist, stem_kde, stem_line
from .graphic import stem_graphic
from .text import stem_data, stem_dot, stem_text
from .helpers import dd
from .stopwords import *
