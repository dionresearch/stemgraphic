# stemgraphic


# Overview

John Tukey’s stem-and-leaf plot first appeared in 1970. Although very useful back then, it cannot handle 
more than 300 data points and is completely text-based. Stemgraphic is a very easy to use python package 
providing a solution to these limitations (no size limit, graphical tool). It also supports **categorical**
and **text** as input.

A typical stem_graphic output:

  ![stem_graphic example](https://github.com/dionresearch/stemgraphic/blob/master/png/test_rosetta.png?raw=true)

For an in depth look at the algorithms and the design of stemgraphic, see

  [Stemgraphic: A Stem-and-Leaf Plot for the Age of Big Data](https://github.com/fdion/stemgraphic/raw/master/doc/stemgraphic%20A%20Stem-and-Leaf%20Plot%20for%20the%20Age%20of%20Big%20Data.pdf)

Documentation is available as pdf [stemgraphic.pdf](http://stemgraphic.org/doc/stemgraphic.pdf)
and [online](http://stemgraphic.org/doc/) html.

The official website of stemgraphic is: http://stemgraphic.org

A Stem-and-leaf Timeline: [timeline pdf](http://artchiv.es/pydata2016/timeline) on artchiv.es 

See also:
[Are you smarter than a fifth grader?](https://www.linkedin.com/pulse/you-smarter-than-fifth-grader-francois-dion/)


# Installation

Stemgraphic requires docopt, matplotlib and pandas. Optionally, having Scipy installed will give you secondary plots 
and Dask (see requirements_dev.txt for all needed to run all the functional tests) will allow for out of core, big data
visualization. See more python packages that can be installed for more functionality in the section
"Optional Requirements".


Installation is simple:

    pip3 install -U stemgraphic  

or from this cloned repository, in the package root:

    python3 setup.py install
    
If you only have python3, pip3 and python3 are probably going to be pip and python. At this time,
we do not have a conda package yet, but you can install everything else with conda, and then pip install stemgraphic.

# Optional requirements

You can pip install these modules for additional functionality:

- dask (for distributed computing)
- pysixel (for graphics in the text console)
- python-levenshtein (for distance metric)
- scipy (for marginal plots)


# Command line

stemgraphic comes with a command line tool:

    stem -h

    Stem.
    
    Stem and leaf plot from a csv or excel spreadsheet using best defaults. Can do text (text and dot) or graphic (kde,
    graphic, hist, line).
    
    Usage:
        stem <input> [-c <column>] [-d] [-f] [-k <file>] [-o <file>] [-p <percent>] [-r <random>] [-s <server>] [-t <type>] [-u] [-w]
        stem -h | --help
        stem --version
    
    Options:
        -h --help    Show this screen.
        -c <column>  column index
        -d           describe the data
        -f           force dask
        -k <file>    persist sample to file (.csv, .pkl)
        -o <file>    output file (.txt, .png) or stdout
        -p <percent> trim data on both ends (ex: 0.2)
        -r <random>  random_state seed (ex: 42)
        -s <server>  head node for distributed cluster
        -t <type>    alternate type of distribution plot
        -u           use all data (default: 300 on text, 900 on graphics)
        -w           wide format (horizontal)
        --version

A typical command line output:

  ![text heatmap example](https://github.com/dionresearch/stemgraphic/raw/master/png/text_heatmap_in_terminal.png)
    
An example Sixel graphics in the terminal:

  ![heatmap example in terminal](https://github.com/dionresearch/stemgraphic/raw/master/png/graphic_heatmap_in_terminal.png)

The supported graphic chart types (-t):

- dot
- graphic (default - stem_graphic plot)
- heatmap
- hist
- kde
- line

The supported text chart types (-t):

- heatmatrix
- text (stem_text plot)
- text_dot
- text_hist
- text_heatmap
- tally

# Latest changes

## Version 0.9.1

- fixes for `pandas` >= 1.0
- passing aggregation from small_multiples and default to False
- handle categorical filtered out on density plots

## Version 0.9.0

- bugfix when no index in translate_representation
- reformat alpha.py with black (2 changes)
- added log scale support with cufflinks in 3d (bug was fixed)
- matplotlib log scale in 3d still not working, however
- added html renders of all the demo notebooks
- fixed some warnings (pandas and matplotlib)
- fixed color palette for interactive plots when comparing 2 sources
- fixed marker size on default view (5 x increase)
- bugfix on command line stem

## Version 0.8.3

- text mode heatmatrix
- text mode heatmap (heatmatrix without 0 values, compact format)
- symmetric stem_dot option to center the dots
- stem_symmetric_dot alias
- improved documentation
- stem_hist, text histogram
- stem_tally, text tally chart
- charset support for stem_text
- charset support for heatmap, heatmatrix
- heatmap for alpha
- heatmatrix for alpha
- unicode digit charsets added:
     'arabic',
     'arabic_r',
     'bold',
     'circled',
     'default',
     'doublestruck',
     'fullwidth',
     'gurmukhi',
     'mono',
     'nko',
     'rod',
     'roman',
     'sans',
     'sansbold',
     'square',
     'subscript',
     'tamil'



## Version 0.8.2

- bugfix on min/max values from command line
- silence warning from matplotlib on tight_layout
- Alignment issue on title for back to back stem-and-leaf plots
- bugfix on dot plot number of dots
- Added symmetric dot plot option and alias since I was working on dot plot

## Version 0.8.1

- command line output improved: description of data more elaborate
- leaf_scatter plot added
- stem_text support for flip_axes
- stem_dot support flip_axes
- stem_dot defaults marker to unicode circle
- added support for dot for command line stem (stem -t dot)

## Version 0.7.5

- Bugfix for issue 12, -0 stem not showing in certain cases 

## Version 0.7.4

- Bugfix for stem_text with plain list (df and numpy are ok)

## Version 0.7.2

- Bugfix for secondary plot calculation

## Version 0.7.0

- Made Levenshtein module optional
- Small Multiples support

## Version 0.6.2

- Bugfix for VERSION

## Version 0.6.1

- back-to-back stem-and-leaf plots can use predefined axes (secondary ax added)
- added quantize function (basically a round trip number->stem-and-leaf->number))
- density_plot added for numerical values with stem-and-leaf quantization and sampling
- density_plot also support multiple secondary plots like box, violin, rug, strip
- notebook demoing density_plot
- notebook demoing comparison of violin, box and stem-and-leaf for certain distributions

## Version 0.6.0

Version bump to 0.6 due to order of params changing. Shouldn't affect using named args

Major code change and expansion for num.stem_graphic including:
- back-to-back stem-and-leaf plots
- allows comparison of very skewed data
- bug fix (rounding issue) due to python precision
- better stem handling 
- alpha down to 10% for bars
- median alpha can be specified
- stems can be hidden
- added title option, besides the legend

Other changes:
- More notebook examples
- added leaf_skip, stem_skip to a few functions missing them
- heatmap_grid bugfix
- added reverse to a few functions missing it
- improved documentation
- matrix_difference ord param added added
- ngram_data now properly defaults to case insensitive
- switched magenta to 'C4' - compatible with mpl styles now
- functions to read/write .npy and .pkl files
- more unicode typographical glyphs added to the list of non alpha


## Version 0.5.3

- scatter 3d support
- added 3rd source to compare (in 3d) with scatter plots
- more scatter plot fixes
- some warnings added to deal with 3d and log scale issues
- added fig_xy to scatter - useful to quickly adjust figsize in a notebook
- added normalize, percentage and whole (integer) to scatter
- added alpha to scatter

## Version 0.5.2

- added documentation for scatter plots
- added jitter to scatter plots
- added log scale to scatter plots
- more notebooks

## Version 0.5.1

- stem_text legend fix
- missed adding the code for scatter plots
- more notebooks

## Version 0.5.0

Major new release.

- All 0.4.0 private changes were merged
- new module stemgraphic.alpha:
  - n-gram support
  - stem_graphic supporting categorical
  - stem_graphic supporting text
  - stem_text supporting categorical
  - stem_text supporting text
  - stem command line supporting categorical when column specified
  - heatmap for n-grams
  - heatmap grid to compare multiple text sources
  - Frobenius norm on diff matrices
  - radar plot with Levenshtein distance
  - frequency plot (bar, barh, hist, area, pie)
  - sunburst char
  - interactive charts with cufflinks
- new module stemgraphic.num to match .alpha
- stop word dictionaries for English, Spanish and French
- Massively improved documentation of modules and functions
- Improved HTML documentation
- Improved PDF documentation

## Version 0.4.0

Internal release for customer.

- Added Heatmap

- Basic PDF documentation

- Quickstart notebook

## Version 0.3.7

Matploblib 2.0 compatibility

## Version 0.3.6

- Persist sample from command line tool (-k filename.pkl or -k filename.csv).

- Windows compatible bat file wrapper (stem.bat).

- Added full command line access to dask distributed server (-d, -s, use file in '' when using glob / wildcard).

- For operations with dask, performance has been increased by 25% in this latest release, by doing a compute
once of min, max and count all at once. Count replaces len(x).


Added the companion PDF as it will be presented at PyData Carolinas 2016.


# TODO

- multivariate support
- provide support for secondary plots with dask
- automatic dense layout
- add a way to provide an alternate function to the sampling
- support for spark rdds and/or sparkling pandas
- create a bokeh version. Ideally rbokeh too.
- add unit tests
- add feather, hdf5 etc support, particularly on sample persistence
- more charts
- more examples