#!/usr/bin/env python3
"""stem_graphic

Module implementing a text and a graphical stem-and-leaf plot function. stem_graphic provides
horizontal, vertical or mirrored layouts, sorted in ascending or descending order, with sane
default settings for the visuals, legend, median and outliers.

It also handles very large data sets through scaling, sampling, trimming and other techniques.

See research paper () for more technical details.

A command line utility was installed along with the module, allowing to process excel or csv
files. See: stem -h
"""
from .aliases import stem_hist, stem_kde, stem_line
from .graphic import stem_graphic
from .text import stem_data, stem_dot, stem_text
from .helpers import dd
