import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from .helpers import key_calc, legend, min_max_count, dd
from .text import stem_data


def stem_graphic(df, alpha=0.15, aggregation=True, asc=True, ax=None, bar_color='b', bar_outline=None,
                 break_on=None, column=None, compact=False, delimiter_color='r', display=900, flip_axes=False,
                 font_kw=None, leaf_color='k', leaf_order=1, legend_pos='best', median_color='magenta', mirror=False,
                 outliers=None, outliers_color='r', persistence=None, primary_kw=None, random_state=None, scale=None,
                 secondary_kw=None, secondary_plot=None, trim=False, trim_blank=True, underline_color=None,
                 unit='', zoom=None):
    """ A graphical stem and leaf plot.


    :param df: list, numpy array, time series, pandas or dask dataframe
    :param aggregation: Boolean for sum, else specify function
    :param alpha: opacity of the bars, median and outliers, defaults to 15%
    :param asc: stem sorted in ascending order, defaults to True
    :param ax: matplotlib axes instance, usually from a figure or other plot
    :param bar_color: the fill color of the bar representing the leaves
    :param bar_outline: the outline color of the bar representing the leaves
    :param break_on: force a break of the leaves at x in (5, 10), defaults to 10
    :param column: specify which column (string or number) of the dataframe to use,
                   else the first numerical is selected
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param delimiter_color: color of the line between aggregate and stem and stem and leaf
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param flip_axes: X becomes Y and Y becomes X
    :param font_kw: keyword dictionary, font parameters
    :param leaf_color: font color of the leaves
    :param leaf_order: how many leaf digits per data point to display, defaults to 1
    :param legend_pos: One of 'top', 'bottom', 'best' or None, defaults to 'best'.
    :param median_color: color of the box representing the median
    :param mirror: mirror the plot in the axis of the delimiters
    :param outliers: display outliers - these are from the full data set, not the sample. Defaults to Auto
    :param outliers_color: background color for the outlier boxes
    :param persistence: filename. save sampled data to disk, either as pickle (.pkl) or csv (any other extension)
    :param primary_kw: stem-and-leaf plot additional arguments
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param scale: force a specific scale for building the plot. Defaults to None (automatic).
    :param secondary_kw: any matplotlib keyword supported by .plot(), for the secondary plot
    :param secondary_plot: One or more of 'dot', 'kde', 'margin_kde', 'rug' in a comma delimited string or None
    :param trim: ranges from 0 to 0.5 (50%) to remove from each end of the data set, defaults to None
    :param trim_blank: remove the blank between the delimiter and the first leaf, defaults to True
    :param underline_color: color of the horizontal line under the leaves, None for no display
    :param unit: specify a string for the unit ('$', 'Kg'...). Used for outliers and for legend, defaults to ''
    :param zoom: zoom level, on top of calculated scale (+1, -1 etc)
    :return: matplotlib figure and axes instance
    """
    try:
        cols = len(df.columns)
    except AttributeError:
        # wasn't a multi column data frame, might be a list
        cols = 1
    if cols > 1:
        if column is None:
            # We have to figure out the first numerical column on our own
            start_at = 1 if df.columns[0] == 'id' else 0
            for i in range(start_at, len(df.columns)):
                if df.dtypes[i] in ('int64', 'float64'):
                    column = i
                    break
        if dd:
            df = df[df.columns.values[column]]
        else:
            df = df.ix[:, column]

    if font_kw is None:
        font_kw = {}
    if primary_kw is None:
        primary_kw = {}
    base_fontsize = font_kw.get('fontsize', 12)
    aggr_fontsize = font_kw.get('aggr_fontsize', base_fontsize - 2)
    aggr_fontweight = font_kw.get('aggr_fontweight', 'normal')
    aggr_facecolor = font_kw.get('aggr_facecolor', None)
    aggr_fontcolor = font_kw.get('aggr_color', 'k')

    stem_fontsize = font_kw.get('stem_fontsize', base_fontsize)
    stem_fontweight = font_kw.get('stem_fontweight', 'normal')
    stem_facecolor = font_kw.get('stem_facecolor', None)
    stem_fontcolor = font_kw.get('stem_color', 'k')

    pad = primary_kw.get('pad', 1.5)
    if outliers is None:
        outliers = True
        remove_duplicate = True
    else:
        outliers = outliers
        remove_duplicate = False
    leaf_alpha = 1
    if leaf_color is None:
        leaf_color = 'k'
        leaf_alpha = 0

    min_val, max_val, total_rows = min_max_count(df)

    scale_factor, pair, rows = stem_data(df, break_on=break_on, column=column, compact=compact,
                                         display=display, omin=min_val, omax=max_val, outliers=False,
                                         persistence=persistence, random_state=random_state, scale=scale,
                                         total_rows=total_rows, trim=trim, zoom=zoom)
    max_leaves = len(max(rows, key=len))
    if max_leaves > display / 3:
        # more than 1/3 on a single stem, let's try one more time
        if random_state:
            random_state += 1
        scale_factor2, pair2, rows2 = stem_data(df, break_on=break_on, column=column, compact=compact,
                                                display=display, omin=min_val, omax=max_val, outliers=False,
                                                persistence=persistence, random_state=random_state, scale=scale,
                                                total_rows=total_rows, trim=trim, zoom=zoom)
        max_leaves2 = len(max(rows2, key=len))
        if max_leaves2 < max_leaves:
            max_leaves = max_leaves2
            scale_factor = scale_factor2
            pair = pair2
            rows = rows2

    spread = (max_val - min_val)
    half_spread = spread / 2

    st, lf = pair.split('|')
    n = display if total_rows > display else total_rows
    fig = None
    if flip_axes:
        height = max_leaves / 2.2 + 3
        if height < 20:
            height = 20
        width = len(rows) + 3
    else:
        width = max_leaves / 2.2 + 3
        if width < 20:
            width = 20
        height = len(rows) + 3
    if ax is None:
        fig = plt.figure(figsize=((width / 4), (height / 4)))
        ax = fig.add_axes((0.05, 0.05, 0.9, 0.9),
                          aspect='equal', frameon=False,
                          xlim=(-1, width + 0.05),
                          ylim=(-1, height + 0.05))
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
    offset = 0
    if mirror:
        plt.gca().invert_yaxis() if flip_axes else plt.gca().invert_xaxis()
        offset = 0.5
    if not asc:
        plt.gca().invert_xaxis() if flip_axes else plt.gca().invert_yaxis()

    tot = 0
    min_s = 99999999
    med = None
    first_val = None
    for cnt, item in enumerate(rows):
        stem, leaf = item.split('|')
        if trim_blank:
            leaf = leaf.strip()
        if stem != '    ':
            stem = stem.strip()
            last_stem = int(stem)
            if int(stem) < min_s:
                min_s = last_stem
        if first_val is None:
            first_val = key_calc(stem, leaf[0 if asc else -1], scale_factor)

        tot += int(len(leaf.strip()))  # TODO: currently only valid if leaf order is 1
        if tot > n / 2 and med is None and median_color is not None:
            med = abs(tot - n / 2 - len(leaf.strip()))
            if flip_axes:
                ax.text(cnt, 2.5 + med / 2.23, '_', fontsize=base_fontsize, color=leaf_color,
                        bbox={'facecolor': median_color, 'alpha': alpha, 'pad': pad}, alpha=leaf_alpha,
                        ha='left', va='top' if mirror else 'bottom', rotation=90)
            else:
                ax.text(2.5 + med / 2.23, cnt + (asc == False), '_', fontsize=base_fontsize, color=leaf_color,
                        bbox={'facecolor': median_color, 'alpha': alpha, 'pad': pad}, alpha=leaf_alpha,
                        ha='left', va='bottom')
        if flip_axes:
            if aggregation:
                ax.text(cnt + offset, 0, tot, fontsize=aggr_fontsize, rotation=90, color=aggr_fontcolor,
                        bbox={'facecolor': aggr_facecolor, 'alpha': alpha, 'pad': pad} if aggr_facecolor is not None
                        else {},
                        fontweight=aggr_fontweight, va='center', ha='right' if mirror else 'left')
            # STEM
            ax.text(cnt + offset, 1.5, stem, fontweight=stem_fontweight, color=stem_fontcolor,
                    bbox={'facecolor': stem_facecolor, 'alpha': alpha, 'pad': pad} if stem_facecolor is not None
                    else {},
                    fontsize=stem_fontsize, va='center', ha='right' if mirror else 'left')

            # LEAF
            ax.text(cnt, 2.1, leaf[::-1] if mirror else leaf, fontsize=base_fontsize,  color=leaf_color,
                    ha='left', va='top' if mirror else 'bottom', rotation=90, alpha=leaf_alpha,
                    bbox={'facecolor': bar_color, 'edgecolor': bar_outline, 'alpha': alpha, 'pad': pad})

        else:
            if aggregation:
                ax.text(-0.5, cnt + 0.5, tot, fontsize=aggr_fontsize, color=aggr_fontcolor,
                        bbox={'facecolor': aggr_facecolor, 'alpha': alpha, 'pad': pad} if aggr_facecolor is not None
                        else {},
                        fontweight=aggr_fontweight, va='center', ha='right' if mirror else 'left')
            # STEM
            ax.text(2.4, cnt + 0.5, stem, fontweight=stem_fontweight, color=stem_fontcolor,
                    bbox={'facecolor': stem_facecolor, 'alpha': alpha, 'pad': pad} if stem_facecolor is not None
                    else {},
                    fontsize=stem_fontsize, va='center', ha='left' if mirror else 'right')

            # LEAF
            ax.text(2.6, cnt + 0.5, leaf[::-1] if mirror else leaf, fontsize=base_fontsize,
                    va='center', ha='right' if mirror else 'left', color=leaf_color, alpha=leaf_alpha,
                    bbox={'facecolor': bar_color, 'edgecolor': bar_outline, 'alpha': alpha, 'pad': pad})
            if underline_color:
                ax.hlines(cnt, 2.6, 2.6 + len(leaf)/2, color=underline_color)
    last_val = key_calc(last_stem, leaf, scale_factor)
    if remove_duplicate and (np.isclose(first_val, min_val) or np.isclose(first_val, max_val))\
                        and (np.isclose(last_val, min_val) or np.isclose(last_val, max_val)):
        outliers = False
    cur_font = FontProperties()
    if flip_axes:
        ax.hlines(2, min_s, min_s + 1 + cnt, color=delimiter_color, alpha=0.7)
        if aggregation:
            ax.hlines(1, min_s, min_s + 1 + cnt, color=delimiter_color, alpha=0.7)
        if outliers:
            ax.text(min_s - 1.5, 1.5, '{} {}'.format(min_val, unit), fontsize=base_fontsize, rotation=90,
                    va='center', ha='left' if asc else 'right',
                    bbox={'facecolor': 'red', 'alpha': alpha, 'pad': 2})
            ax.text(min_s + cnt + 1.6, 1.5, '{} {}'.format(max_val, unit), fontsize=base_fontsize, rotation=90,
                    va='center', ha='left' if asc else 'right',
                    bbox={'facecolor': 'red', 'alpha': alpha, 'pad': 2})
            ax.hlines(1.5, min_s, min_s - 0.5, color=delimiter_color, alpha=0.7)
            ax.hlines(1.5, min_s + 1 + cnt, min_s + 1.5 + cnt, color=delimiter_color, alpha=0.7)
        legend(ax, width, min_s + cnt, asc, flip_axes, mirror, st, lf,
               scale_factor, delimiter_color, aggregation, cur_font, n, legend_pos, unit)

    else:
        if aggregation:
            ax.vlines(1, 0, 1 + cnt, color=delimiter_color, alpha=0.7)
        ax.vlines(2.4, 0, 1 + cnt, color=delimiter_color, alpha=0.7)
        if outliers:
            ax.text(1.5, -1, '{} {}'.format(min_val, unit), fontsize=base_fontsize,
                    va='center', ha='center',
                    bbox={'facecolor': outliers_color, 'alpha': alpha, 'pad': 2})
            ax.text(1.5, cnt + 2, '{} {}'.format(max_val, unit), fontsize=12,
                    va='center', ha='center',
                    bbox={'facecolor': outliers_color, 'alpha': alpha, 'pad': 2})
            ax.vlines(1.5, -0.5, 0, color=delimiter_color, alpha=0.7)
            ax.vlines(1.5, 1 + cnt, 1.5 + cnt, color=delimiter_color, alpha=0.7)
        legend(ax, width, cnt, asc, flip_axes, mirror, st, lf,
               scale_factor, delimiter_color, aggregation, cur_font, n, legend_pos, unit)

    if secondary_plot is not None:
        secondary_kw = secondary_kw or {'alpha': 0.5}
        start_at = 1.5
        from scipy.stats import gaussian_kde
        try:
            y = df.dropna()
        except AttributeError:
            y = df
        gkde = gaussian_kde(y)
        ind = np.linspace(min_val, int((int(lf)/10+int(st))*int(scale_factor)), len(rows)*10)
        if 'overlay_kde' in secondary_plot:
            if flip_axes:
                ax.plot((ind/scale_factor) + 0.01 if asc else -1,
                        0.9+start_at+gkde.evaluate(ind)*scale_factor*width*6, **secondary_kw)
            else:
                ax.plot(0.9+start_at+gkde.evaluate(ind)*scale_factor*width*6,
                        (ind/scale_factor) + 0.01 if asc else -1, **secondary_kw)
        elif 'kde' in secondary_plot:
            if flip_axes:
                ax.plot((ind/scale_factor) + 0.01 if asc else -1,
                        start_at + gkde.evaluate(ind)*scale_factor*width*6/width, **secondary_kw)
            else:
                ax.plot(start_at + gkde.evaluate(ind)*scale_factor*width*6/width,
                        (ind/scale_factor) + 0.01 if asc else -1, **secondary_kw)
        if 'rug' in secondary_plot:
            y = df.sample(frac=display/total_rows).dropna()

            if flip_axes:
                ax.plot((y/scale_factor) + 0.01 if asc else -1, y*0+1.2,
                        '|', color='k', **secondary_kw)
            else:
                ax.plot(y*0+1.2, (y/scale_factor) + 0.01 if asc else -1,
                        '_', color='k', **secondary_kw)
        elif secondary_plot == 'dot':
            y = df.sample(frac=display/total_rows).dropna()

            if flip_axes:
                ax.plot((y/scale_factor) + 0.01 if asc else -1, y*0+1.2,
                        'o', markeredgewidth=1, markerfacecolor='None', markeredgecolor='k', **secondary_kw)
            else:
                ax.plot(y*0+1.2, (y/scale_factor) + 0.01 if asc else -1,
                        'o', markeredgewidth=1, markerfacecolor='None', markeredgecolor='k', **secondary_kw)
    return fig, ax
