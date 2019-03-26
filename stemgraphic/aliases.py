"""Handy aliases for stem_graphic options."""
from .graphic import stem_graphic
from .text import stem_dot


def stem_hist(
    x,
    aggregation=False,
    alpha=1,
    asc=True,
    column=None,
    color="b",
    delimiter_color="r",
    display=300,
    flip_axes=True,
    legend_pos="short",
    outliers=False,
    trim=False,
):
    """ stem_hist

    stem_hist builds a graphical histogram matching the stem-and-leaf plot, with the numbers hidden, as shown on the
    cover of the companion brochure.

    :param legend_pos:
    :param x: list, numpy array, time series, pandas or dask dataframe
    :param aggregation: Boolean for sum, else specify function
    :param alpha: opacity of the bars, median and outliers, defaults to 15%
    :param asc: stem sorted in ascending order, defaults to True
    :param column: specify which column (string or number) of the dataframe to use,
                   else the first numerical is selected
    :param color: the bar facecolor
    :param delimiter_color: color of the line between aggregate and stem and stem and leaf
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param flip_axes: X becomes Y and Y becomes X
    :param outliers: this is NOP, for compatibility
    :param trim: this is NOP, for compatibility
    :return: matplotlib figure and axes instance
    """
    font_settings = {"fontsize": 8}
    return stem_graphic(
        x,
        alpha=alpha,
        aggregation=aggregation,
        asc=asc,
        bar_color=color,
        break_on=10,
        column=column,
        delimiter_color=delimiter_color,
        display=display,
        flip_axes=flip_axes,
        font_kw=font_settings,
        leaf_color=color,
        legend_pos=legend_pos,
        median_color=None,
        outliers=False,
    )


def stem_kde(x, **kw_args):
    """stem_kde buils a stem-and-leaf plot and adds an overlaid kde as secondary plot.

    :param x:  list, numpy array, time series, pandas or dask dataframe
    :param kw_args:
    :return: matplotlib figure and axes instance
    """
    kw_args["secondary_plot"] = "overlay_kde"
    return stem_graphic(x, **kw_args)


def stem_line(
    x,
    aggregation=False,
    alpha=0,
    asc=True,
    column=None,
    color="k",
    delimiter_color="r",
    display=300,
    flip_axes=True,
    outliers=False,
    secondary_plot=None,
    trim=False,
):
    """stem_line builds a stem-and-leaf plot with lines instead of bars.

    :param x: list, numpy array, time series, pandas or dask dataframe
    :param aggregation: Boolean for sum, else specify function
    :param alpha: opacity of the bars, median and outliers, defaults to 15%
    :param asc: stem sorted in ascending order, defaults to True
    :param column: specify which column (string or number) of the dataframe to use,
                   else the first numerical is selected
    :param color: the color of the line
    :param delimiter_color: color of the line between aggregate and stem and stem and leaf
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param flip_axes: X becomes Y and Y becomes X
    :param outliers:
    :param secondary_plot: One or more of 'dot', 'kde', 'margin_kde', 'rug' in a comma delimited string or None
    :param trim: this is NOP, for compatibility
    :return: matplotlib figure and axes instance
    """
    return stem_graphic(
        x,
        alpha=alpha,
        aggregation=aggregation,
        asc=asc,
        bar_color=None,
        break_on=10,
        column=column,
        delimiter_color=delimiter_color,
        display=display,
        flip_axes=flip_axes,
        leaf_color=color,
        legend_pos="short",
        median_color=None,
        outliers=outliers,
        secondary_plot=secondary_plot,
        underline_color=color,
    )


def stem_symmetric_dot(x, **kw_args):
    """ stem_symmetric_dot

    stem_symmetric_dot builds a symmetric stem dot plot

    **Example**:

        .. sourcecode:: python

            stem_symmetric_dot(diamonds.price)

    **Output**:

        .. sourcecode:: python

            326
                ¡
              0 | ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
              1 |            ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
              2 |                     ●●●●●●●●●●●●●●●●●●●●●●●●●●
              3 |                    ●●●●●●●●●●●●●●●●●●●●●●●●●●●●
              4 |                   ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
              5 |                       ●●●●●●●●●●●●●●●●●●●●●●●
              6 |                        ●●●●●●●●●●●●●●●●●●●●
              7 |                           ●●●●●●●●●●●●●●●
              8 |                                 ●●●
              9 |                              ●●●●●●●●
             10 |                                ●●●●
             11 |                               ●●●●●●●
             12 |                               ●●●●●●
             13 |                                 ●●
             14 |                                 ●●
             15 |                                 ●●●
             16 |                                 ●●●
             17 |                              ●●●●●●●●
             18 |                                  ●
                !
            18823
            Scale:
            18|6 => 18.6x1000 = 18600.0

    :param x:  list, numpy array, time series, pandas or dask dataframe
    :param kw_args: keyword args to stem_dot
    :return:
    """
    kw_args["symmetric"] = True
    return stem_dot(x, **kw_args)
