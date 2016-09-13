stemgraphic
===========

Overview
========

John Tukey’s stem-and-leaf plot first appeared in 1970. Although very
useful back then, it cannot handle more than 300 data points and is
completely text-based. Stemgraphic is a very easy to use python package
providing a solution to these limitations.

A typical stem\_graphic output:

.. figure:: https://github.com/fdion/stemgraphic/raw/master/png/test_rosetta.png
   :alt: stem\_graphic example

   stem\_graphic example

For an in depth look at

Installation
============

Stemgraphic requires docopt, matplotlib and pandas. Optionally, having
Scipy installed will give you secondary plots and Dask (see
requirements.txt for all needed to run all the functional tests) will
allow for out of core, big data visualization.

Installation is simple:

::

    pip3 install -U stemgraphic  

or from this cloned repository, in the package root:

::

    python3 setup.py install

Latest changes
==============

For operations with dask, performance has been increased by 25% in this
latest release, by doing a compute once of min, max and count all at
once. Count replaces len(x).

Added the companion PDF as it will be presented at PyData Carolinas
2016.

TODO
====

Plenty... but to start:

-  back to back and scale calculation
-  multivariate support
-  provide support for secondary plots with dask
-  automatic dense layout
-  add a way to provide an alternate function to the sampling
-  support for spark rdds and/or sparkling pandas
-  create a bokeh version. Ideally rbokeh too.
-  interactive version based on the above
-  add unit tests
-  add feather, hdf5 etc support, particularly on sample persistence
