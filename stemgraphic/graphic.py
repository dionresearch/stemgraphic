""" Stemgraphic.graphic

Stemgraphic provides a complete set of functions to handle everything related to stem-and-leaf plots.
Stemgraphic.graphic is a module implementing a graphical stem-and-leaf plot function and a stem-and-leaf heatmap plot
function for numerical data.
"""
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import seaborn as sns
from warnings import warn

from .helpers import key_calc, legend, min_max_count, dd
from .text import stem_data


def heatmap(df, annotate=False, asFigure=False, ax=None, column=None, compact=False, display=900,
            interactive=True, persistence=None, random_state=None, scale=None,
            trim=False, trim_blank=True, unit='', zoom=None):
    """ heatmap

        The heatmap displays the same underlying data as the stem-and-leaf plot, but instead of stacking the leaves,
        they are left in their respective columns. Row '42' and Column '7' would have the count of numbers starting
        with '427' of the given scale.

        The heatmap is useful to look at patterns. For distribution, stem_graphic is better suited.

    :param df: list, numpy array, time series, pandas or dask dataframe
    :param annotate: display annotations (Z) on heatmap
    :param asFigure: return plot as plotly figure (for web applications)
    :param ax:  matplotlib axes instance, usually from a figure or other plot
    :param column: specify which column (string or number) of the dataframe to use,
                   else the first numerical is selected
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param interactive: if cufflinks is loaded, renders as interactive plot in notebook
    :param persistence: filename. save sampled data to disk, either as pickle (.pkl) or csv (any other extension)
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param scale: force a specific scale for building the plot. Defaults to None (automatic).
    :param trim: ranges from 0 to 0.5 (50%) to remove from each end of the data set, defaults to None
    :param trim_blank: remove the blank between the delimiter and the first leaf, defaults to True
    :param unit:  specify a string for the unit ('$', 'Kg'...). Used for outliers and for legend, defaults to ''
    :param zoom: zoom level, on top of calculated scale (+1, -1 etc)
    :return: count matrix, scale and matplotlib ax or figure if interactive and asFigure are True
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

    min_val, max_val, total_rows = min_max_count(df)

    scale_factor, pair, rows = stem_data(df, break_on=None, column=column, compact=compact,
                                         display=display, leaf_order=1, omin=min_val, omax=max_val,
                                         outliers=False, persistence=persistence, random_state=random_state,
                                         scale=scale, total_rows=total_rows, trim=trim, zoom=zoom)
    max_leaves = len(max(rows, key=len))

    if max_leaves > display / 3:
        # more than 1/3 on a single stem, let's try one more time
        if random_state:
            random_state += 1
        scale_factor2, pair2, rows2 = stem_data(df, break_on=None, column=column, compact=compact,
                                                display=display, leaf_order=1, omin=min_val, omax=max_val,
                                                outliers=False, persistence=persistence, random_state=random_state,
                                                scale=scale, total_rows=total_rows, trim=trim, zoom=zoom)
        max_leaves2 = len(max(rows2, key=len))
        if max_leaves2 < max_leaves:
            max_leaves = max_leaves2
            scale_factor = scale_factor2
            pair = pair2
            rows = rows2

    split_rows = [i.split('|') for i in rows]

    # redo the leaves in a matrix form
    # this should be refactored as an option for stem_data, like rows_only for ngram_data
    matrix = []
    for stem, leaves in split_rows:
        row_count = [stem]
        for num in '0123456789':
            row_count.append(leaves.count(num))
        matrix.append(row_count)

    num_matrix = pd.DataFrame(matrix, columns=['stem', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    num_matrix.set_index('stem', inplace=True)
    if trim_blank:
        num_matrix.applymap(lambda x: x.strip() if type(x) is str else x)

    title = 'Stem-and-leaf heatmap ({} x {} {})'.format(pair.replace('|', '.'), scale_factor, unit)
    if interactive:
        try:
            fig = num_matrix.iplot(kind='heatmap', asFigure=asFigure, title=title)
        except AttributeError:
            if ax is None:
                fig, ax = plt.subplots(figsize=(9, 9))
                plt.yticks(rotation=0)
            ax.set_title(title)
            sns.heatmap(num_matrix, annot=annotate, ax=ax)
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 12))
            plt.yticks(rotation=0)
        ax.set_title(title)
        sns.heatmap(num_matrix, annot=annotate, ax=ax)
    return num_matrix, scale_factor, fig if asFigure else ax


def stem_graphic(df, df2=None, aggregation=True, alpha=0.1, asc=True, ax=None, bar_color='C0', bar_outline=None,
                 break_on=None, column=None, combined=None, compact=False, delimiter_color='C3', display=900,
                 figure_only=True, flip_axes=False, font_kw=None, leaf_color='k', leaf_order=1, legend_pos='best',
                 median_alpha=0.25, median_color='C4', mirror=False, outliers=None, outliers_color='C3', persistence=None,
                 primary_kw=None, random_state=None, scale=None,  secondary=False, secondary_kw=None,
                 secondary_plot=None, show_stem=True, title=None, trim=False, trim_blank=True, underline_color=None,
                 unit='', zoom=None):
    """ stem_graphic

    A graphical stem and leaf plot. stem_graphic provides horizontal, vertical or mirrored layouts, sorted in
    ascending or descending order, with sane default settings for the visuals, legend, median and outliers.

    :param df: list, numpy array, time series, pandas or dask dataframe
    :param df2: string, filename, url, list, numpy array, time series, pandas or dask dataframe (optional).
                for back 2 back stem-and-leaf plots
    :param aggregation: Boolean for sum, else specify function
    :param alpha: opacity of the bars, median and outliers, defaults to 10%
    :param asc: stem sorted in ascending order, defaults to True
    :param ax: matplotlib axes instance, usually from a figure or other plot
    :param bar_color: the fill color of the bar representing the leaves
    :param bar_outline: the outline color of the bar representing the leaves
    :param break_on: force a break of the leaves at x in (5, 10), defaults to 10
    :param column: specify which column (string or number) of the dataframe to use,
                   else the first numerical is selected
    :param combined: list (specific subset to automatically include, say, for comparisons), or None
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param delimiter_color: color of the line between aggregate and stem and stem and leaf
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param figure_only: bool if True (default) returns matplotlib (fig,ax), False returns (fig,ax,df)
    :param flip_axes: X becomes Y and Y becomes X
    :param font_kw: keyword dictionary, font parameters
    :param leaf_color: font color of the leaves
    :param leaf_order: how many leaf digits per data point to display, defaults to 1
    :param legend_pos: One of 'top', 'bottom', 'best' or None, defaults to 'best'.
    :param median_alpha: opacity of median and outliers, defaults to 25%
    :param median_color: color of the box representing the median
    :param mirror: mirror the plot in the axis of the delimiters
    :param outliers: display outliers - these are from the full data set, not the sample. Defaults to Auto
    :param outliers_color: background color for the outlier boxes
    :param persistence: filename. save sampled data to disk, either as pickle (.pkl) or csv (any other extension)
    :param primary_kw: stem-and-leaf plot additional arguments
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param scale: force a specific scale for building the plot. Defaults to None (automatic).
    :param secondary: bool if True, this is a secondary plot - mostly used for back-to-back plots
    :param secondary_kw: any matplotlib keyword supported by .plot(), for the secondary plot
    :param secondary_plot: One or more of 'dot', 'kde', 'margin_kde', 'rug' in a comma delimited string or None
    :param show_stem: bool if True (default) displays the stems
    :param title: string to display as title
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

    scale_factor, pair, rows, _, stems = stem_data(df, break_on=break_on, column=column, compact=compact,
                                                   display=display, full=True, leaf_order=leaf_order, omin=min_val,
                                                   omax=max_val, outliers=False, persistence=persistence,
                                                   random_state=random_state, scale=scale, total_rows=total_rows,
                                                   trim=trim, zoom=zoom)

    max_leaves = len(max(rows, key=len))

    if max_leaves > display / 3:
        # more than 1/3 on a single stem, let's try one more time
        if random_state:
            random_state += 1
        scale_factor2, pair2, rows2, stems2 = stem_data(df, break_on=break_on, column=column, compact=compact,
                                                        display=display, full=True, leaf_order=leaf_order, omin=min_val,
                                                        omax=max_val, outliers=False, persistence=persistence,
                                                        random_state=random_state, scale=scale, total_rows=total_rows,
                                                        trim=trim, zoom=zoom)
        max_leaves2 = len(max(rows2, key=len))
        if max_leaves2 < max_leaves:
            max_leaves = max_leaves2
            scale_factor = scale_factor2
            pair = pair2
            rows = rows2
            stems = stems2

    st, lf = pair.split('|')
    n = display if total_rows > display else total_rows
    fig = None
    ax1 = None

    if flip_axes:
        height = max_leaves
        if height < 20:
            height = 20
        width = len(rows) + 3
    else:
        width = max_leaves
        if width < 20:
            width = 20
        height = len(rows) + 3
    if combined is None:
        combined = stems

    aggr_offset = -0.5
    aggr_line_offset = 1
    if df2 is not None:
        if flip_axes:
            warn("Error: flip_axes is not available with back to back stem-and-leaf plots.")
            return None

        min_val_df2, max_val_df2, total_rows = min_max_count(df2)
        scale_factor_df2, _, _, rows_df2, stems_df2 = stem_data(df2, break_on=break_on, column=column, compact=compact,
                                                                display=display, full=True, leaf_order=leaf_order,
                                                                omin=min_val_df2, omax=max_val_df2, outliers=outliers,
                                                                persistence=persistence, random_state=random_state,
                                                                scale=scale, total_rows=total_rows, trim=trim,
                                                                zoom=zoom)

        compact_combined = list(set(stems + stems_df2))
        combined_min = min(compact_combined)
        combined_max = max(compact_combined)
        if compact:
            combined = compact_combined
        else:
            combined = list(np.arange(combined_min, combined_max, 0.5 if break_on==5 else 1))
        cnt_offset_df2 = 0
        while stems[cnt_offset_df2] not in stems_df2 and cnt_offset_df2 < len(stems):
            cnt_offset_df2 += 1
        max_leaves_df2 = len(max(rows_df2, key=len))
        total_width = max_leaves +  max_leaves_df2 # / 2.2 + 3
        if total_width < 20:
            total_width = 20
        total_height = combined_max + 1 - combined_min  #cnt_offset_df2 + len(stems_df2)

        fig, (ax1, ax) = plt.subplots(1, 2, sharey=True, figsize=((total_width / 4), (total_height / 4)))

        plt.box(on=None)
        ax1.axes.get_yaxis().set_visible(False)
        ax1.axes.get_xaxis().set_visible(False)
        ax1.set_xlim(-1, total_width + 0.05)
        ax1.set_ylim(-1, total_height + 0.05)
        _ = stem_graphic(df2,  # NOQA
                         alpha=alpha, ax=ax1, aggregation=mirror and aggregation, asc=asc, bar_color=bar_color,
                         bar_outline=bar_outline, break_on=break_on, column=column,
                         combined=combined, compact=compact, delimiter_color=delimiter_color, display=display,
                         flip_axes=False, legend_pos=None,
                         median_alpha=median_alpha, median_color=median_color,
                         mirror=not mirror, outliers=outliers,
                         random_state=random_state, secondary=True, secondary_kw=secondary_kw,
                         secondary_plot=secondary_plot, show_stem=True, trim=trim,
                         trim_blank=trim_blank, underline_color=underline_color, unit=unit, zoom=zoom)

    else:
        total_width = width
        total_height = height
    if ax is None:
        fig = plt.figure(figsize=((width / 4), (total_height / 4)))
        ax = fig.add_axes((0.05, 0.05, 0.9, 0.9),
                          aspect='equal', frameon=False,
                          xlim=(-1, width + 0.05),
                          ylim=(-1, height + 0.05))
    plt.box(on=None)
    ax.axis('off')
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    # Title
    if df2 is not None or secondary:
        title_offset = -2 if mirror else 4
    else:
        title_offset = 0 if mirror else 2

    if title:
        if flip_axes:
            ax.set_title(title, y=title_offset)
        else:
            ax.set_title(title, x=title_offset)

    # Offsets
    offset = 0
    if ax1 is not None:
        aggr_offset = -3.8
        aggr_line_offset = -0.5

    if mirror:
        ax.set_ylim(ax.get_ylim()[::-1]) if flip_axes else ax.set_xlim(ax.get_xlim()[::-1])
        offset = -2 if secondary else 0.5
    if not asc:
        ax.set_xlim(ax.get_xlim()[::-1]) if flip_axes else ax.set_ylim(ax.get_ylim()[::-1])

    tot = 0
    min_s = 99999999
    med = None
    first_val = None

    cnt_offset = 0
    while combined[cnt_offset] not in stems and cnt_offset < len(stems):
        cnt_offset += 1

    for item_num, item in enumerate(rows):
        cnt = item_num + cnt_offset
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
                        bbox={'facecolor': median_color, 'alpha': median_alpha, 'pad': pad}, alpha=leaf_alpha,
                        ha='left', va='top' if mirror else 'bottom', rotation=90)
            else:
                ax.text(2.5 + med / 2.23, cnt + (asc == False), '_', fontsize=base_fontsize, color=leaf_color,  # NOQA
                        bbox={'facecolor': median_color, 'alpha': median_alpha, 'pad': pad}, alpha=leaf_alpha,
                        ha='left', va='bottom')
        if flip_axes:
            if aggregation and not (df2 is not None and mirror):
                ax.text(cnt + offset, 0, tot, fontsize=aggr_fontsize, rotation=90, color=aggr_fontcolor,
                        bbox={'facecolor': aggr_facecolor, 'alpha': alpha, 'pad': pad} if aggr_facecolor is not None
                        else {'alpha': 0},
                        fontweight=aggr_fontweight, va='center', ha='right' if mirror else 'left')
            # STEM
            if show_stem:
                ax.text(cnt + offset, 1.5, stem, fontweight=stem_fontweight, color=stem_fontcolor,
                        bbox={'facecolor': stem_facecolor, 'alpha': alpha, 'pad': pad} if stem_facecolor is not None
                        else {'alpha': 0},
                        fontsize=stem_fontsize, va='center', ha='right' if mirror else 'left')

            # LEAF
            ax.text(cnt, 2.1, leaf[::-1] if mirror else leaf, fontsize=base_fontsize,  color=leaf_color,
                    ha='left', va='top' if mirror else 'bottom', rotation=90, alpha=leaf_alpha,
                    bbox={'facecolor': bar_color, 'edgecolor': bar_outline, 'alpha': alpha, 'pad': pad})

        else:
            if aggregation and not (df2 is not None and mirror):
                ax.text(aggr_offset, cnt + 0.5, tot, fontsize=aggr_fontsize, color=aggr_fontcolor,
                        bbox={'facecolor': aggr_facecolor, 'alpha': alpha, 'pad': pad} if aggr_facecolor is not None
                        else {'alpha': 0},
                        fontweight=aggr_fontweight, va='center', ha='right' if mirror else 'left')
            # STEM
            if show_stem:
                ax.text(2.4, cnt + 0.5, stem, fontweight=stem_fontweight, color=stem_fontcolor,
                        bbox={'facecolor': stem_facecolor, 'alpha': alpha, 'pad': pad} if stem_facecolor is not None
                        else {'alpha': 0},
                        fontsize=stem_fontsize, va='center', ha='left' if mirror else 'right')

            # LEAF
            ax.text(2.6, cnt + 0.5, leaf[::-1] if mirror else leaf, fontsize=base_fontsize,
                    va='center', ha='right' if mirror else 'left', color=leaf_color, alpha=leaf_alpha,
                    bbox={'facecolor': bar_color, 'edgecolor': bar_outline, 'alpha': alpha, 'pad': pad})
            if underline_color:
                ax.hlines(cnt, 2.6, 2.6 + len(leaf)/2, color=underline_color)
    last_val = key_calc(last_stem, leaf, scale_factor)
    if remove_duplicate and (np.isclose(first_val, min_val) or np.isclose(first_val, max_val))\
                        and (np.isclose(last_val, min_val) or np.isclose(last_val, max_val)):  # NOQA
        outliers = False
    cur_font = FontProperties()
    if flip_axes:
        ax.hlines(2, min_s, min_s + 1 + cnt, color=delimiter_color, alpha=0.7)
        if aggregation and not (df2 is not None and mirror):
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
        line_length = 1 + cnt if (ax1 is None) or df2 is not None else 1 + max(stems)
        if aggregation and not (df2 is not None and mirror):
            ax.vlines(aggr_line_offset, cnt_offset, line_length, color=delimiter_color, alpha=0.7)
        if show_stem:
            ax.vlines(2.4, cnt_offset, line_length, color=delimiter_color, alpha=0.7)
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
    if flip_axes:
        ax.plot(0, total_height)
    else:
        ax.plot(total_width, 0)
    if figure_only:
        return fig, ax
    else:
        return fig, ax, df
