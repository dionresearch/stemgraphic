# -*- coding: utf-8 -*-
""" stemgraphic.alpha.


BRAND NEW in V.0.5.0!

Stemgraphic provides a complete set of functions to handle everything related to stem-and-leaf plots. alpha is a
module of the stemgraphic package to add support for categorical and text variables.

The module also adds functionality to handle whole words, beside stem-and-leaf bigrams and n-grams.

For example, for the word "alabaster":

With word_ functions, we can look at the word frequency in a text, or compare it through a distance function
(default to Levenshtein) to other words in a corpus

With stem_ functions, we can look at the fundamental stem-and-leaf, stem would be 'a' and leaf would be 'l', for
a bigram 'al'. With a stem_order of 1 and a leaf_order of 2, we would have 'a' and 'la', for a trigram 'ala', so
on and so forth.

"""
from math import radians
import re
import unicodedata
from urllib.request import urlopen
from warnings import warn

import Levenshtein
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from .helpers import stack_columns, CHAR_FILTER, LETTERS, NON_ALPHA


def add_missing_letters(mat, stem_order, leaf_order, letters=None):
    """ Add missing stems based on LETTERS. defaults to a-z alphabet.

    :param mat: matrix to modify
    :param stem_order: how many stem characters per data point to display, defaults to 1
    :param leaf_order: how many leaf characters per data point to display, defaults to 1
    :param letters: letters that must be present as stems
    :return: the modified matrix
    """
    if letters is None:
        letters = LETTERS
    if stem_order > 1 or leaf_order > 1:
        # No change, can't force every missing leaf bigram
        return mat

    previous_col = 0
    for letter in letters:
        if letter not in mat.columns:
            mat.insert(previous_col + 1, letter, np.NaN)
            previous_col += 1
        else:
            previous_col = list(mat.columns.values).index(letter)
    return mat


# noinspection PyPep8Naming
def heatmap(src, alpha_only=False, annotate=False, asFigure=False, ax=None, caps=False, compact=True,  # NOQA
            display=None, interactive=True, leaf_order=1, leaf_skip=0, random_state=None, stem_order=1,
            stem_skip=0, stop_words=None):
    """ The heatmap displays the same underlying data as the stem-and-leaf plot, but instead of stacking the leaves,
     they are left in their respective columns. Row 'a' and Column 'b' would have the count of words starting
     with 'ab'. The heatmap is useful to look at patterns. For distribution, stem\_graphic is better suited.

    :param src: string, filename, url, list, numpy array, time series, pandas or dask dataframe
    :param alpha_only: only use stems from a-z alphabet
    :param annotate: display annotations (Z) on heatmap
    :param asFigure: return plot as plotly figure (for web applications)
    :param ax:  matplotlib axes instance, usually from a figure or other plot
    :param caps: bool, True to be case sensitive
    :param compact: remove empty stems
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param interactive: if cufflinks is loaded, renders as interactive plot in notebook
    :param leaf_order: how many leaf characters per data point to display, defaults to 1
    :param leaf_skip: how many leaf characters to skip, defaults to 0 - useful w/shared bigrams: 'wol','wor','woo'
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param stem_order: how many stem characters per data point to display, defaults to 1
    :param stem_skip: how many stem characters to skip, defaults to 0 - useful to zoom in on a single root letter
    :param stop_words: stop words to remove. None (default), list or builtin EN (English), ES (Spanish) or FR (French)
    :return:
    """

    _, alpha_matrix, _ = ngram_data(
        src,
        alpha_only=alpha_only,
        caps=caps,
        compact=compact,
        display=display,
        leaf_order=leaf_order,
        leaf_skip=leaf_skip,
        rows_only=False,
        random_state=random_state,
        stem_order=stem_order,
        stem_skip=stem_skip,
        stop_words=stop_words
    )
    if not compact:
        alpha_matrix.word = add_missing_letters(alpha_matrix.word, stem_order, leaf_order)
    if isinstance(src, str):
        title = 'stem-and-leaf heatmap for {}'.format(src)
    else:
        title = 'stem-and-leaf heatmap'
    if interactive:
        try:
            fig = alpha_matrix.word.T.iplot(kind='heatmap', asFigure=asFigure, title=title)
        except AttributeError:
            if ax is None:
                fig, ax = plt.subplots(figsize=(20, 16))
            ax.set_title(title)
            sns.heatmap(alpha_matrix.word, annot=annotate, ax=ax)
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 16))
        ax.set_title(title)
        sns.heatmap(alpha_matrix.word, annot=annotate, ax=ax)
    return alpha_matrix, ax


# noinspection PyUnboundLocalVariable
def heatmap_grid(src1, src2, src3=None, src4=None, alpha_only=True, annot=False, caps=False, center=0, cmap=None,
                 display=1000, leaf_order=1, leaf_skip=0, random_state=None, reverse=False, robust=False, stem_order=1,
                 stem_skip=0, stop_words=None, threshold=0):
    """ heatmap_grid.

    With stem_graphic, it is possible to directly compare two different sources. In the case of a heatmap,
    two different data sets cannot be visualized directly on a single heatmap. For this task, we designed
    heatmap_grid to adapt to the number of sources to build a layout. It can take from 2 to 4 different source.

    With 2 sources, a square grid will be generated, allowing for horizontal and vertical comparisons,
    with an extra heatmap showing the difference between the two matrices. It also computes a norm for that
    difference matrix. The smaller the value, the closer the two heatmaps are.

    With 3 sources, it builds a triangular grid, with each source heatmap in a corner and the difference between
    each pair in between.

    Finally, with 4 sources, a 3 x 3 grid is built, each source in a corner and the
    difference between each pair in between, with the center expressing the difference between top left
    and bottom right diagonal.

    :param src1: string, filename, url, list, numpy array, time series, pandas or dask dataframe (required)
    :param src2: string, filename, url, list, numpy array, time series, pandas or dask dataframe (required)
    :param src3: string, filename, url, list, numpy array, time series, pandas or dask dataframe (optional)
    :param src4: string, filename, url, list, numpy array, time series, pandas or dask dataframe (optional)
    :param alpha_only: only use stems from a-z alphabet
    :param annot: display annotations (Z) on heatmap
    :param caps: bool, True to be case sensitive, defaults to False, recommended for comparisons.
    :param center: the center of the divergent color map for the difference heatmaps
    :param cmap: color map for difference heatmap or None (default) to use the builtin red / blue divergent map
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param leaf_order: how many leaf characters per data point to display, defaults to 1
    :param leaf_skip: how many leaf characters to skip, defaults to 0 - useful w/shared bigrams: 'wol','wor','woo'
    :param robust: reduce effect of outliers on difference heatmap
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param stem_order: how many stem characters per data point to display, defaults to 1
    :param stem_skip: how many stem characters to skip, defaults to 0 - useful to zoom in on a single root letter
    :param stop_words: stop words to remove. None (default), list or builtin EN (English), ES (Spanish) or FR (French)
    :param threshold: absolute value minimum count difference for a difference heatmap element to be visible
    :return:
    """
    res1, alpha1, x1 = ngram_data(src1, alpha_only=alpha_only, display=display, stem_order=stem_order,
                                  leaf_order=leaf_order, leaf_skip=leaf_skip, random_state=random_state, rows_only=False,
                                  stop_words=stop_words, reverse=reverse,
                                  caps=caps)
    res2, alpha2, x2 = ngram_data(src2, alpha_only=alpha_only, display=display, stem_order=stem_order,
                                  leaf_order=leaf_order, leaf_skip=leaf_skip, random_state=random_state, rows_only=False,
                                  stop_words=stop_words, reverse=reverse,
                                  caps=caps)
    alpha1 = add_missing_letters(alpha1.word, stem_order, leaf_order)
    alpha2 = add_missing_letters(alpha2.word, stem_order, leaf_order)

    if src3 is not None:
        res3, alpha3, x3 = ngram_data(src3, alpha_only=alpha_only, display=display, stem_order=stem_order,
                                      leaf_order=leaf_order, leaf_skip=leaf_skip, random_state=random_state, rows_only=False,
                                      stop_words=stop_words, caps=caps, reverse=reverse)
        alpha3 = add_missing_letters(alpha3.word, stem_order, leaf_order)

    if src4 is not None:
        res4, alpha4, x4 = ngram_data(src4, alpha_only=alpha_only, display=display, stem_order=stem_order,
                                      leaf_order=leaf_order, leaf_skip=leaf_skip, random_state=random_state, rows_only=False,
                                      stop_words=stop_words, caps=caps, reverse=reverse)
        alpha4 = add_missing_letters(alpha4.word, stem_order, leaf_order)

    diff1, norm1, ratio1 = matrix_difference(alpha1, alpha2, thresh=threshold)

    mvmin = alpha1.min().min()
    mvmax = alpha1.max().max()

    if src3:
        # noinspection PyUnboundLocalVariable
        diff2, norm2, ratio2 = matrix_difference(alpha1, alpha3, thresh=threshold)
        diff3, norm3, ratio3 = matrix_difference(alpha2, alpha3, thresh=threshold)

    if src4:
        # noinspection PyUnboundLocalVariable
        diff4, norm4, ratio4 = matrix_difference(alpha2, alpha4, thresh=threshold)
        diff5, norm5, ratio5 = matrix_difference(alpha3, alpha4, thresh=threshold)
        diff6, norm6, ratio6 = matrix_difference(alpha1, alpha4, thresh=threshold)

    if cmap is None:
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

    if center is not None:
        data = diff1
        calc_data = data[~np.isnan(data)]
        vmin = np.percentile(calc_data, 2) if robust else calc_data.min().min()
        vmax = np.percentile(calc_data, 98) if robust else calc_data.max().max()
        vrange = max(vmax - center, center - vmin)
        normalize = mpl.colors.Normalize(center - vrange, center + vrange)
        cmin, cmax = normalize([vmin, vmax])
        cc = np.linspace(cmin, cmax, 256)
        cmap = mpl.colors.ListedColormap(cmap(cc))

    if src3 is None and src4 is None:
        fig, ((ax1, ax3), (ax4, ax2)) = plt.subplots(2, 2, figsize=(20, 16))
        sns.heatmap(alpha1, annot=annot, ax=ax1, vmin=mvmin, vmax=mvmax, square=True)
    else:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(20, 20))
        sns.heatmap(alpha1, annot=annot, ax=ax1, vmin=mvmin, vmax=mvmax, square=True)

    ax1.set_title(src1)

    if src3:
        ax1.text(1.3, 0.7, '->',
                 size=12,
                 horizontalalignment='center',
                 verticalalignment='center',
                 rotation='horizontal',
                 transform=ax1.transAxes)

        ax1.text(0.3, -0.1, '->',
                 size=12,
                 horizontalalignment='center',
                 verticalalignment='center',
                 rotation=-90,
                 transform=ax1.transAxes)

    if src4:
        ax1.text(1.3, -0.1, '->',
                 size=12,
                 horizontalalignment='center',
                 verticalalignment='center',
                 rotation=-45,
                 transform=ax1.transAxes)
        # noinspection PyUnboundLocalVariable
        ax5.text(1.3, -0.1, '->',
                 size=12,
                 horizontalalignment='center',
                 verticalalignment='center',
                 rotation=-45,
                 transform=ax5.transAxes)

    ax2.set_title('changes ({})'.format(norm1 / ratio1))
    # noinspection PyUnboundLocalVariable,PyUnboundLocalVariable
    sns.heatmap(diff1, annot=True if norm1 < 100 else False, ax=ax2, vmin=vmin, vmax=vmax, cmap=cmap, square=True)
    ax2.text(1.3, 0.7, '->',
             size=12,
             horizontalalignment='center',
             verticalalignment='center',
             rotation='horizontal',
             transform=ax2.transAxes)

    ax3.set_title(src2)
    sns.heatmap(alpha2, ax=ax3, vmin=mvmin, vmax=mvmax, square=True)
    if src3:
        # noinspection PyUnboundLocalVariable,PyUnboundLocalVariable
        ax4.set_title('changes ({})'.format(norm2 / ratio2))
        # noinspection PyUnboundLocalVariable
        sns.heatmap(diff2, annot=True if norm2 < 100 else False, ax=ax4, vmin=vmin, vmax=vmax, cmap=cmap, square=True)
        ax4.text(0.3, -0.1, '->',
                 size=12,
                 horizontalalignment='center',
                 verticalalignment='center',
                 rotation=-90,
                 transform=ax4.transAxes)

        # noinspection PyUnboundLocalVariable
        ax7.set_title(src3)
        sns.heatmap(alpha3, annot=annot, ax=ax7, vmin=mvmin, vmax=mvmax)

        if src4:
            # noinspection PyUnboundLocalVariable,PyUnboundLocalVariable
            ax5.set_title('changes ({})'.format(norm6 / ratio6))
            # noinspection PyUnboundLocalVariable
            sns.heatmap(diff6, annot=True if norm6 < 100 else False, ax=ax5, vmin=vmin, vmax=vmax, cmap=cmap,
                        square=True)
        else:
            # noinspection PyUnboundLocalVariable,PyUnboundLocalVariable
            ax5.set_title('changes ({})'.format(norm3 / ratio3))
            # noinspection PyUnboundLocalVariable
            sns.heatmap(diff3, annot=True if norm3 < 100 else False, ax=ax5, vmin=vmin, vmax=vmax, cmap=cmap,
                        square=True)
    else:
        ax4.set_title(src2)
        sns.heatmap(alpha2, ax=ax4, vmin=mvmin, vmax=mvmax, square=True)
    if src4:
        ax3.text(0.7, -0.1, '->',
                 size=12,
                 horizontalalignment='center',
                 verticalalignment='center',
                 rotation=-90,
                 transform=ax3.transAxes)
        # noinspection PyUnboundLocalVariable,PyUnboundLocalVariable,PyUnboundLocalVariable
        ax6.set_title('changes ({})'.format(norm4 / ratio4))
        # noinspection PyUnboundLocalVariable
        sns.heatmap(diff4, annot=True if norm4 < 100 else False, ax=ax6, vmin=vmin, vmax=vmax, cmap=cmap, square=True)
        ax6.text(0.7, -0.1, '->',
                 size=12,
                 horizontalalignment='center',
                 verticalalignment='center',
                 rotation=-90,
                 transform=ax6.transAxes)
        ax7.text(1.3, 0.3, '->',
                 size=12,
                 horizontalalignment='center',
                 verticalalignment='center',
                 rotation='horizontal',
                 transform=ax7.transAxes)
        # noinspection PyUnboundLocalVariable
        ax8.text(1.3, 0.3, '->',
                 size=12,
                 horizontalalignment='center',
                 verticalalignment='center',
                 rotation='horizontal',
                 transform=ax8.transAxes)
        # noinspection PyUnboundLocalVariable
        ax9.set_title(src4)
        sns.heatmap(alpha4, annot=annot, ax=ax9, vmin=mvmin, vmax=mvmax)

        # noinspection PyUnboundLocalVariable,PyUnboundLocalVariable
        ax8.set_title('changes ({})'.format(norm5 / ratio5))
        # noinspection PyUnboundLocalVariable
        sns.heatmap(diff5, annot=True if norm5 < 100 else False, ax=ax8, vmin=vmin, vmax=vmax, cmap=cmap, square=True)
    elif src3:
        # noinspection PyUnboundLocalVariable
        ax6.axis('off')
        # noinspection PyUnboundLocalVariable
        ax8.axis('off')
        # noinspection PyUnboundLocalVariable
        ax9.axis('off')
    return fig


def matrix_difference(mat1, mat2, thresh=0, ord=None):
    """ matrix_difference

    :param mat1: first heatmap dataframe
    :param mat2: second heatmap dataframe
    :param thresh: : absolute value minimum count difference for a difference heatmap element to be visible
    :return: difference matrix, norm and ratio of the sum of the first matrix over the second
    """
    tot1 = mat1.sum().sum()
    tot2 = mat2.sum().sum()
    ratio = tot1 / tot2

    diff = mat1.fillna(-999999).subtract(mat2.fillna(0) * ratio, fill_value=0).reindex_like(mat1).astype(int)
    diff = diff.replace(-999999, np.NaN)
    diff[diff < -999999] = diff[diff < -999999] + 999999
    diff[(diff >= 0) & (diff <= thresh)] = np.NaN
    diff[(diff < 0) & (diff >= -thresh)] = np.NaN
    norm = np.linalg.norm(diff.fillna(0), ord=ord)
    return diff, norm, ratio


def ngram_data(df, alpha_only=False, ascending=True, binary=False, break_on=None, caps=False,
               char_filter=None, column=None, compact=False, display=750, leaf_order=1, leaf_skip=0,
               persistence=None, random_state=None, remove_accents=False, reverse=False,
               rows_only=True, sort_by='len', stem_order=1, stem_skip=0, stop_words=None):
    """ ngram_data

    This is the main text ingestion function for stemgraphic.alpha. It is used by most of the visualizations. It
    can also be used directly, to feed a pipeline, for example.

    If selected (rows_only=False), the returned dataframe includes in each row a single word, the stem, the leaf and
    the ngram (stem + leaf) - the index is the 'token' position in the original source:

        word    stem 	leaf 	ngram
    12 	salut   s       a       sa
    13 	chéri   c       h       ch

    :param df: list, numpy array, series, pandas or dask dataframe
    :param alpha_only: only use stems from a-z alphabet (NA on dataframe)
    :param ascending: bool if the sort is ascending
    :param binary: bool if True forces counts to 1 for anything greater than 0
    :param break_on: letter on which to break a row, or None (default)
    :param caps: bool, True to be case sensitive, defaults to False, recommended for comparisons.(NA on dataframe)
    :param char_filter: list of characters to ignore. If None (default) CHAR_FILTER list will be used
    :param column: specify which column (string or number) of the dataframe to use, or group of columns (stems)
                   else the frame is assumed to only have one column with words.
    :param compact: remove empty stems
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param leaf_order: how many leaf characters per data point to display, defaults to 1
    :param leaf_skip: how many leaf characters to skip, defaults to 0 - useful w/shared bigrams: 'wol','wor','woo'
    :param persistence: will save the sampled datafrae to filename (with csv or pkl extension) or None
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param remove_accents: bool if True strips accents (NA on dataframe)
    :param rows_only: bool by default returns only the stem and leaf rows. If false, also the matrix and dataframe
    :param sort_by: default to 'len', can also be 'alpha'
    :param stem_order: how many stem characters per data point to display, defaults to 1
    :param stem_skip: how many stem characters to skip, defaults to 0 - useful to zoom in on a single root letter
    :param stop_words: stop words to remove. None (default), list or builtin EN (English), ES (Spanish) or FR (French)
    :return: ordered rows if rows_only, else also returns the matrix and dataframe
    """
    if char_filter is None:
        char_filter = CHAR_FILTER
    if isinstance(df, str):
        # First check if it is a url
        if df[:7] in ['http://', 'https:/'] and len(df) < 2000:  # In theory 2048 is the max URL length
            data = urlopen(df)
            with data as r:
                lines = r.read().decode()  # utf8 for now
                linecontent = ''.join(lines)

        # Maybe filename passed, try to read a text file directly
        else:
            try:
                with open(df) as r:
                    lines = r.readlines()
                    linecontent = ' '.join(lines)
            except IOError:
                # not url or filename, we'll assume a content string then
                linecontent = df
        if remove_accents:
            normalized = unicodedata.normalize('NFKD', linecontent)
            if normalized != linecontent:
                linecontent = ''.join([c for c in normalized if not unicodedata.combining(c)])
        for ch in char_filter:
            if ch in linecontent:
                linecontent = linecontent.replace(ch, ',')
        if reverse:
            linecontent = linecontent[::-1]
        x = pd.DataFrame({
            'word': linecontent.replace(' ', ',').split(',')
        })
        x = x[x.word != '']
        if alpha_only:
            x = x[~x.word.str[:1].isin(NON_ALPHA)]
        if not caps:
            x.word = x.word.str.lower()
        if stop_words is not None:
            if not caps:
                stop_words = [x.lower() for x in stop_words]
            x = x[~x.word.isin(stop_words)]
        if column:
            x = x[x.word.str[:1].isin(column)]
        if display is None or display > x.word.shape[0]:
            x_s = x.reset_index()
        else:
            x_s = x.sample(n=display, random_state=random_state).reset_index()
    elif isinstance(df, list):
        x = pd.DataFrame({
            'word': df
        })
        if display is None or display > x.word.shape[0]:
            x_s = x
        else:
            x_s = x.sample(n=display, random_state=random_state).reset_index()
    else:
        try:
            x = df if column is None else df[column]
            if reverse:
                x = x.str[::-1]
        except KeyError:
            x = df.copy()
            if reverse:
                x = x.applymap(lambda r: r.str[::-1])
            if column:
                x = x[x.word.str[:1].isin(column)]
        if display is None or display > x.shape[0]:
            x_s = x
        else:
            x_s = x.sample(n=display, random_state=random_state).reset_index()

    if stem_order is None:
        stem_order = 1
    if leaf_order is None:
        leaf_order = 1

    x_s['stem'] = x_s.word.str[stem_skip:stem_skip + stem_order]
    offset = stem_skip + stem_order + leaf_skip
    x_s['leaf'] = x_s.word.str[offset:offset + leaf_order].str.ljust(leaf_order)
    x_s['ngram'] = x_s['stem'] + x_s['leaf']

    if persistence is not None:
        if persistence[-4:] == '.pkl':
            x_s.to_pickle(persistence)
        else:
            x_s.to_csv(persistence)  # TODO: add feather, hdf5 etc

    alpha_matrix = x_s.groupby(['stem', 'leaf']).count().unstack('leaf')
    if binary:
        alpha_matrix.astype(bool).astype(int)
    if compact:
        pass  # nothing to do ATM.
    if break_on is not None:
        # TODO: investigate if we need this down to ngram_data level, or if stem_text/stem_graphic level is ok
        pass
    rows = alpha_matrix[alpha_matrix.columns[0][0]].apply(stack_columns, axis=1)

    # Sorting
    if sort_by == 'len':
        rows = rows[rows.str.len().sort_values().index]
    ordered_rows = rows if ascending else rows[::-1]

    if rows_only:
        return ordered_rows
    else:
        return ordered_rows, alpha_matrix, x_s


def polar_word_plot(ax, word, words, label, min_dist, max_dist, metric, offset, step):
    """ polar_word_plot

    Utility function for radar plot.

    :param ax: matplotlib ax
    :param word: string, the reference word that will be placed in the middle
    :param words: list of words to compare
    :param label:  bool if True display words centered at coordinate
    :param min_dist: minimum distance based on metric to include a word for display
    :param max_dist: maximum distance for a given section
    :param metric: any metric function accepting two values and returning that metric in a range from 0 to x
    :param offset: where to start plotting in degrees
    :param step: how many degrees to step between plots
    :return:
    """
    for i, comp in enumerate(sorted(words)):
        dist = metric(word, comp)
        if dist > max_dist:
            max_dist = dist
        if dist >= min_dist:
            ax.plot((0, radians((i + 1) * step + offset)), (0, dist - 0.01))
        if label:
            t = ax.text(radians((i + 1) * step + offset), dist, comp, size=12, ha='center', va='center')
            t.set_bbox(dict(facecolor='white', alpha=0.3))
    return max_dist


def plot_sunburst_level(normalized, ax, label=True, level=0, offset=0, ngram=False, plot=True, stem=None, vis=0):
    """ plot_sunburst_level

    utility function for sunburst function.

    :param normalized:
    :param ax:
    :param label:
    :param level:
    :param ngram:
    :param offset:
    :param plot:
    :param stem:
    :param vis:
    :return:
    """
    total = len(normalized)

    heights = [level + 1] * total

    widths = normalized.values

    bottoms = [level + 0] * total
    values = np.cumsum([0 + offset] + list(widths[:-1]))
    if plot:
        rects = ax.bar(values, heights, widths, bottoms, linewidth=1,
                       edgecolor='white', align='edge')
    else:
        return values

    labels = normalized.index.values

    if level in (0, 0.4):
        fontsize = 16
    else:
        fontsize = 10
    if stem:
        labels = [stem + label for label in labels]  # for stem, this is ok, unless level is 0.4 (next statement)
        fontsize = 10
    if level > 0.4 and not ngram:
        labels = [i[1:] for i in labels]  # strip stem, label should be leaf only unless ngram requested

    # If label display is enabled, we got more work to do
    if label:
        for rect, label in zip(rects, labels):
            width = rect.get_width()
            x = rect.get_x() + width / 2
            y = (rect.get_y() + rect.get_height()) / 2 if level == 0.4 else level * 2.9
            if width > vis:
                ax.text(x, y, label, size=fontsize, color='k', ha='center', va='center')
    return values


def radar(word, comparisons, ascending=True, display=100, label=True, metric=None,
          min_distance=1, max_distance=None, random_state=None, sort_by='alpha'):
    """ radar

    The radar plot compares a reference word with a corpus. By default, it calculates the levenshtein
    distance between the reference word and each words in the corpus. An alternate distance or metric
    function can be provided. Each word is then plotted around the center based on 3 criteria.

    1) If the word length is longer, it is plotted on the left side, else on the right side.

    2) Distance from center is based on the distance function.

    3) the words are equidistant, and their order defined alphabetically or by count (only applicable
       if the corpus is a text and not a list of unique words, such as a password dictionary).

    Stem-and-leaf support is upcoming.

    :param word: string, the reference word that will be placed in the middle
    :param comparisons: external file, list or string or dataframe of words
    :param ascending: bool if the sort is ascending
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param label: bool if True display words centered at coordinate
    :param metric: Levenshtein (default), or any metric function accepting two values and returning that metric
    :param min_distance: minimum distance based on metric to include a word for display
    :param max_distance: maximum distance based on metric to include a word for display
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param sort_by: default to 'alpha', can also be 'len'
    :return:
    """
    if metric is None:
        metric = Levenshtein.distance
    # TODO: switch to ngram_data for stem-and-leaf support and better word support
    if isinstance(comparisons, str):
        with open(comparisons) as r:
            lines = r.readlines()
            linecontent = ' '.join(lines)
        df = pd.DataFrame({
            'word': linecontent.replace('\n', ',').replace('"', ',').replace(".", ',').replace(' ', ',').split(',')
        })
        x = df[df.word != ''].word.sample(n=display, random_state=random_state).tolist()
    else:
        x = comparisons

    fig, pol_ax = plt.subplots(1, 1, figsize=(15, 15), subplot_kw=dict(projection='polar'))
    pol_ax.grid(color='#dfdfdf')  # Color the grid
    pol_ax.set_theta_zero_location('N')  # Origin is at the top
    pol_ax.set_theta_direction(-1)  # Reverse the rotation
    pol_ax.set_rlabel_position(0)  # default is angled to the right. move it out of the way
    pol_ax.axes.get_xaxis().set_visible(False)
    word_len = len(word)

    if sort_by == 'alpha':
        high = sorted([i for i in x if len(i) > word_len])
        low = sorted([i for i in x if len(i) <= word_len])
    else:
        high = sorted([i for i in x if len(i) > word_len], key=len)
        low = sorted([i for i in x if len(i) <= word_len], key=len)
    if not ascending:
        high = high[::-1]
        low = low[::-1]
    numh = len(high)
    numl = len(low)
    max_dist = 0

    # This was initially in radians, but that's not readable for most people, so it
    # is in degrees. I convert to radians directly at the call for plot and text
    step = 180 / (numh + 1)
    offset = 180
    max_dist = polar_word_plot(pol_ax, word, high, label, min_distance, max_dist, metric, offset, step)

    step = 180 / (numl + 1)
    offset = 0

    max_dist = polar_word_plot(pol_ax, word, low, label, min_distance, max_dist, metric, offset, step)
    if max_distance is None:
        max_distance = max_dist
    pol_ax.set_ylim(0, max_distance)
    t = pol_ax.text(0, 0, word, ha='center', va='center', size=12)
    t.set_bbox(dict(facecolor='white', alpha=0.5))
    pol_ax.set_title('{} distance to {}'.format(metric, word))
    return pol_ax


def _scatter3d(df, x, y, z, s, color, ax, label=None,  alpha=0.5):
    """ _scatter3d

    Helper to make call to scatter3d a little more like the 2d

    :param df: data
    :param x: x var name
    :param y: y var name
    :param z: z var name
    :param s: size (list or scalar)
    :param color: color, sequence, or sequence of color
    :param ax: matplotlib ax
    :param label: label for legend
    :param alpha: alpha transparency
    :return:
    """

    xs = 0 if x == 0 else df[x]  # logic for projections
    ys = 0 if y in (0,100) else df[y]
    zs = 0 if z == 0 else df[z]
    ax.scatter(xs, ys, zs=zs, alpha=alpha, s=s, color=color, label=label)


def scatter(src1, src2, src3=None, alpha=0.5, alpha_only=True, ascending=True, asFigure=False, ax=None, caps=False,
            compact=True, display=None, fig_xy=None, interactive=True, jitter=False, label=False, leaf_order=1,
            leaf_skip=0, log_scale=True, normalize=None, percentage=None, project=False, project_only=False,
            random_state=None, sort_by='alpha', stem_order=1, stem_skip=0, stop_words=None, whole=False):
    """ scatter

    With 2 sources:

    Scatter compares the word frequency of two sources, on each axis. Each data point Z value is the word
    or stem-and-leaf value, while the X axis reflects that word/ngram count in one source and the Y axis
    reflect the same word/ngram count in the other source, in two different colors. If one word/ngram is more common
    on the first source it will be displayed in one color, and if it is more common in the second source, it
    will be displayed in a different color. The values that are the same for both sources will be displayed
    in a third color (default colors are blue, black and pink.

    With 3 sources:

    The scatter will compare in 3d the word frequency of three sources, on each axis. Each data point hover value is
    the word or stem-and-leaf value, while the X axis reflects that word/ngram count in the 1st source, the Y axis
    reflects the same word/ngram count in the 2nd source, and the Z axis the 3rd source, each in a different color.
    If one word/ngram is more common on the 1st source it will be displayed in one color, in the 2nd source as a
    second color and if it is more common in the 3rd source, it will be displayed in a third color.
    The values that are the same for both sources will be displayed in a 4th color (default colors are
    blue, black, purple and pink.

    In interactive mode, hovering the data point
    will give the precise counts on each axis along with the word itself, and filtering by category is done
    by clicking on the category in the legend. Double clicking a category will show only that category.

    :param src1: string, filename, url, list, numpy array, time series, pandas or dask dataframe
    :param src2: string, filename, url, list, numpy array, time series, pandas or dask dataframe
    :param src3: string, filename, url, list, numpy array, time series, pandas or dask dataframe, optional
    :param alpha:: opacity of the dots, defaults to 50%
    :param alpha_only: only use stems from a-z alphabet (NA on dataframe)
    :param ascending: word/stem count sorted in ascending order, defaults to True
    :param asFigure: return plot as plotly figure (for web applications)
    :param ax: matplotlib axes instance, usually from a figure or other plot
    :param caps: bool, True to be case sensitive, defaults to False, recommended for comparisons.(NA on dataframe)
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param fig_xy: tuple for matplotlib figsize, defaults to (20,20)
    :param interactive: if cufflinks is loaded, renders as interactive plot in notebook
    :param jitter: random noise added to help see multiple data points sharing the same coordinate
    :param label: bool if True display words centered at coordinate
    :param leaf_order: how many leaf digits per data point to display, defaults to 1
    :param leaf_skip: how many leaf characters to skip, defaults to 0 - useful w/shared bigrams: 'wol','wor','woo'
    :param log_scale: bool if True (default) uses log scale axes (NA in 3d due to open issues with mpl, cufflinks)
    :param normalize: bool if True normalize frequencies in src2 and src3 relative to src1 length
    :param percentage: coordinates in percentage of maximum word/ngram count (in non interactive mode)
    :param project: project src1/src2 and src1/src3 comparisons on X=0 and Z=0 planes
    :param project_only: only show the projection (NA if project is False)
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param sort_by: sort by 'alpha' (default) or 'count'
    :param stem_order: how many stem characters per data point to display, defaults to 1
    :param stem_skip: how many stem characters to skip, defaults to 0 - useful to zoom in on a single root letter
    :param stop_words: stop words to remove. None (default), list or builtin EN (English), ES (Spanish) or FR (French)
    :param whole: for normalized or percentage, use whole integer values (round)
    :return: matplotlib ax, dataframe with categories
    """

    alpha_matrix = []
    x = []
    filename = []
    for src in [src1, src2, src3]:
        if isinstance(src, str):
            filename1 = src[:96]
        else:
            filename1 = 'data'
        if src:
            _, alpha_matrix1, x1 = ngram_data(
                src,
                alpha_only=alpha_only,
                compact=compact,
                display=display,
                leaf_order=leaf_order,
                leaf_skip=leaf_skip,
                rows_only=False,
                random_state=random_state,
                sort_by=sort_by,
                stem_order=stem_order,
                stem_skip=stem_skip,
                stop_words=stop_words,
                caps=caps)
            alpha_matrix.append(alpha_matrix1)
            x.append(x1)
            filename.append(filename1)

    if stem_order is None and leaf_order is None:
        count_by = 'word'
    else:
        count_by = 'ngram'
    xy_ratio = len(x[0]) / len(x[1])
    if src3:
        xz_ratio = len(x[0]) / len(x[2])
        red = pd.concat([x[0][count_by].value_counts().rename('x'),
                         x[1][count_by].value_counts().rename('y'),
                         x[2][count_by].value_counts().rename('z')], axis=1)
        red.fillna(0)
        if normalize:
            red.y = red.y * xy_ratio
            red.z = red.z * xz_ratio
        max_count = red[['x', 'y', 'z']].abs().max().max()
    else:
        red = pd.concat([x[0][count_by].value_counts().rename('x'), x[1][count_by].value_counts().rename('y')], axis=1)
        if normalize:
            red.y = red.y * xy_ratio
        max_count = red[['x', 'y']].abs().max().max()

    title='{} vs{}{}{}'.format(filename[0],
                               '<br>' if interactive else '\n',
                               'normalized ' if normalize else '', filename[1])
    if src3:
        title = '{} vs {}'.format(title, filename[2])

    red.x.fillna(0)
    red.y.fillna(0)
    red.dropna(inplace=True)
    if percentage:
        red.x = red.x / max_count * 100
        red.y = red.y / max_count * 100
        if src3:
            red.z = red.z / max_count * 100

    if whole:
        red.x = red.x.round()
        red.y = red.y.round()
        if src3:
            red.z = red.z.round()

    red['diff1'] = red.x - red.y
    if src3:
        red['diff2'] = red.x - red.z
    red['categories'] = 'x'
    red.loc[(red['diff1'] < 0), 'categories'] = 'y'
    if src3:
        red.loc[(red['diff2'] < 0), 'categories'] = 'z'
        red.loc[(red['diff2'] == 0) & (red['diff1'] == 0), 'categories'] = '='
        red['hovertext'] = red.index.values + ' ' \
                           + red.x.astype(str) + ' ' + red.y.astype(str) + ' ' + red.z.astype(str)
    else:
        red.loc[(red['diff1'] == 0), 'categories'] = '='
        red['hovertext'] = red.x.astype(str) + ' ' + red.index.values + ' ' + red.y.astype(str)

    red['text'] = red.index.values
    red.sort_values(by='categories', inplace=True)
    if jitter:
        # varies slightly the values from their integer counts, but the hover will show the correct count pre jitter
        red['x'] = red['x'] + np.random.uniform(-0.25, 0.25, len(red))
        red['y'] = red['y'] + np.random.uniform(-0.25, 0.25, len(red))
        if src3:
            red['z'] = red['z'] + np.random.uniform(-0.25, 0.25, len(red))
    palette = ['pink', 'blue', 'gray', 'lightpurple']
    if len(red.categories.dropna().unique()) < 4:
        palette = palette[1:len(red.categories.dropna().unique())]
    if fig_xy == None:
        fig_xy=(10,10)
    if interactive:
        try:
            if src3:
                ax1 = red.iplot(kind='scatter3d', colors=palette,
                                x='x', y='y', z='z', categories='categories', title=title, opacity=alpha,
                                # can't use this until fixed: https://github.com/santosjorge/cufflinks/issues/87
                                # logx=log_scale, logy=log_scale, logz=log_scale,
                                size=red.index.str.len(), text='text' if label else 'hovertext', hoverinfo='text',
                                mode='markers+text' if label else 'markers', asFigure=asFigure)
            else:
                ax1 = red.iplot(kind='scatter', colors=palette, logx=log_scale, logy=log_scale, opacity=alpha,
                                x='x', y='y', categories='categories', title=title,
                                size=red.index.str.len(), text='text' if label else 'hovertext', hoverinfo='text',
                                mode='markers+text' if label else 'markers', asFigure=asFigure)
        except AttributeError:
            warn('Interactive plot requested, but cufflinks not loaded. Falling back to matplotlib.')
            interactive=False
            # in case %matplotlib notebook
            fig_xy = (10,10)

    if not interactive:
        if ax is None:
            if src3:
                fig = plt.figure(figsize=fig_xy)
                ax = fig.add_subplot(111, projection='3d')

                if not project_only:
                    _scatter3d(red[red.categories == 'x'], x='x', y='y', z='z', alpha=alpha,
                               s=red[red.categories == 'x'].index.str.len()*10, ax=ax, color='C0', label='x')
                    _scatter3d(red[red.categories == 'y'], x='x', y='y', z='z', alpha=alpha,
                               s=red[red.categories == 'y'].index.str.len()*10, ax=ax, color='k', label='y')
                    _scatter3d(red[red.categories == 'z'], x='x', y='y', z='z', alpha=alpha,
                               s=red[red.categories == 'z'].index.str.len()*10, ax=ax, color='C4', label='z')

                    if len(palette) == 4:
                        # we do have equal values
                        _scatter3d(red[red.categories == '='], x='x', y='y', z='z', alpha=alpha,
                                   s=red[red.categories == '='].index.str.len()*10, ax=ax, color='C3', label='=')
                if project:
                    _scatter3d(red[red.categories == 'x'], x='x', y='y', z=0, alpha=alpha,
                               s=red[red.categories == 'x'].index.str.len() * 10, ax=ax, color='C0')
                    _scatter3d(red[red.categories == 'y'], x='x', y='y', z=0, alpha=alpha,
                               s=red[red.categories == 'y'].index.str.len() * 10, ax=ax, color='k')

                    _scatter3d(red[red.categories == 'y'], x=0, y='y', z='z', alpha=alpha,
                               s=red[red.categories == 'y'].index.str.len() * 10, ax=ax, color='k')
                    _scatter3d(red[red.categories == 'z'], x=0, y='y', z='z', alpha=alpha,
                               s=red[red.categories == 'z'].index.str.len() * 10, ax=ax, color='C4')


            else:
                fig, ax = plt.subplots(1, 1, figsize=fig_xy)
                if label:
                    alpha=0.05
                red[red.categories == 'x'].plot(kind='scatter', x='x', y='y', color='C0', ax=ax, label='x',
                                                alpha=alpha, s=red[red.categories == 'x'].index.str.len() * 10)
                red[red.categories == 'y'].plot(ax=ax, kind='scatter', x='x', y='y', color='k', label='y',
                                                alpha=alpha, s=red[red.categories == 'y'].index.str.len() * 10)
                if len(palette) == 3:
                    red[red.categories == '='].plot(ax=ax, kind='scatter', x='x', y='y', color='C3', label='=',
                                                    alpha=alpha, s=red[red.categories == '='].index.str.len() * 10)

        if log_scale:
            if src3:
                warn("Log_scale is not working currently due to an issue in {}.".format(
                    'cufflinks' if interactive else 'matplotlib'))
                # matplotlib bug: https://github.com/matplotlib/matplotlib/issues/209
                # cufflinks bug: https://github.com/santosjorge/cufflinks/issues/87
            else:
                ax.set_xscale('log')
                ax.set_yscale('log')
        if label:
            if log_scale:
              warn("Labels do not currently work in log scale due to an incompatibility in matplotlib."
                    " Set log_scale=False to display text labels.")
            elif src3:
                for tx, ty, tz, tword in red[['x', 'y', 'z', 'text']].dropna().values:
                    ax.text(tx, ty, tword, zs=tz, va='center', ha='center')
            else:
                for tx, ty, tword in red[['x', 'y', 'text']].dropna().values:
                    if tx < 5 and ty < 5:
                        if np.random.random() > 0.90:
                            # very dense area usually, show roughly 15%, randomly
                            ax.text(tx, ty, tword, va='center', ha='center')
                    else:
                        ax.text(tx, ty, tword, va='center', ha='center')
        ax.set_title(title)
        ax.legend(loc='best')
        if not ascending:
            ax.invert_xaxis()
    return ax, red.drop(['hovertext'], axis=1)


def stem_scatter(src1, src2, src3=None, alpha=0.5, alpha_only=True, ascending=True, asFigure=False, ax=None, caps=False,
                 compact=True, display=None, fig_xy=None, interactive=True, jitter=False, label=False, leaf_order=1,
                 leaf_skip=0, log_scale=True, normalize=None, percentage=None, project=False, project_only=False,
                 random_state=None, sort_by='alpha', stem_order=1, stem_skip=0, stop_words=None, whole=False):
    """ stem_scatter

    stem_scatter compares the word frequency of two sources, on each axis. Each data point Z value is the word
    or stem-and-leaf value, while the X axis reflects that word/ngram count in one source and the Y axis
    reflect the same word/ngram count in the other source, in two different colors. If one word/ngram is more common
    on the first source it will be displayed in one color, and if it is more common in the second source, it
    will be displayed in a different color. The values that are the same for both sources will be displayed
    in a third color (default colors are blue, black and pink. In interactive mode, hovering the data point
    will give the precise counts on each axis along with the word itself, and filtering by category is done
    by clicking on the category in the legend.

    :param src1: string, filename, url, list, numpy array, time series, pandas or dask dataframe
    :param src2: string, filename, url, list, numpy array, time series, pandas or dask dataframe
    :param src3: string, filename, url, list, numpy array, time series, pandas or dask dataframe, optional
    :param alpha:: opacity of the dots, defaults to 50%
    :param alpha_only: only use stems from a-z alphabet (NA on dataframe)
    :param ascending: stem sorted in ascending order, defaults to True
    :param asFigure: return plot as plotly figure (for web applications)
    :param ax: matplotlib axes instance, usually from a figure or other plot
    :param caps: bool, True to be case sensitive, defaults to False, recommended for comparisons.(NA on dataframe)
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param fig_xy:  tuple for matplotlib figsize, defaults to (20,20)
    :param interactive: if cufflinks is loaded, renders as interactive plot in notebook
    :param jitter: random noise added to help see multiple data points sharing the same coordinate
    :param label: bool if True display words centered at coordinate
    :param leaf_order: how many leaf digits per data point to display, defaults to 1
    :param leaf_skip: how many leaf characters to skip, defaults to 0 - useful w/shared bigrams: 'wol','wor','woo'
    :param log_scale: bool if True (default) uses log scale axes (NA in 3d due to open issues with mpl, cufflinks)
    :param normalize: bool if True normalize frequencies in src2 and src3 relative to src1 length
    :param percentage: coordinates in percentage of maximum word/ngram count
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param sort_by: sort by 'alpha' (default) or 'count'
    :param stem_order: how many stem characters per data point to display, defaults to 1
    :param stem_skip: how many stem characters to skip, defaults to 0 - useful to zoom in on a single root letter
    :param stop_words: stop words to remove. None (default), list or builtin EN (English), ES (Spanish) or FR (French)
    :param whole: for normalized or percentage, use whole integer values (round)
    :return: matplotlib polar ax, dataframe
    """
    return scatter(src1=src1, src2=src2, src3=src3, alpha=alpha, alpha_only=alpha_only, asFigure=asFigure,
                   ascending=ascending, ax=ax, caps=caps, compact=compact, display=display, fig_xy=fig_xy,
                   interactive=interactive, jitter=jitter, label=label, leaf_order=leaf_order, leaf_skip=leaf_skip,
                   log_scale=log_scale, normalize=normalize, percentage=percentage, project=project,
                   project_only=project_only, random_state=random_state, sort_by=sort_by, stem_order=stem_order,
                   stem_skip=stem_skip,stop_words=stop_words, whole=whole)


def stem_text(df, aggr=False, alpha_only=True, ascending=True, binary=False, break_on=None, caps=True,
              column=None, compact=False, display=750,
              legend_pos='top', leaf_order=1, leaf_skip=0, persistence=None, remove_accents=False,
              reverse=False, rows_only=False, sort_by='len', stem_order=1, stem_skip=0,
              stop_words=None, random_state=None):
    """ stem_text

    Tukey's original stem-and-leaf plot was text, with a vertical delimiter to separate stem from
    leaves. Just as stemgraphic implements a text version of the plot for numbers,
    stemgraphic.alpha implements a text version for words. This type of plot serves a similar
    purpose as a stacked bar chart with each data point annotated.

    It also displays some basic statistics on the whole text (or subset if using column).

    :param df: list, numpy array, time series, pandas or dask dataframe
    :param aggr: bool if True display the aggregated count of leaves by row
    :param alpha_only: only use stems from a-z alphabet (NA on dataframe)
    :param ascending: bool if the sort is ascending
    :param binary: bool if True forces counts to 1 for anything greater than 0
    :param break_on: force a break of the leaves at that letter, the rest of the leaves will appear on the next line
    :param caps: bool, True to be case sensitive, defaults to False, recommended for comparisons.(NA on dataframe)
    :param column: specify which column (string or number) of the dataframe to use, or group of columns (stems)
                   else the frame is assumed to only have one column with words.
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param leaf_order: how many leaf characters per data point to display, defaults to 1
    :param leaf_skip: how many leaf characters to skip, defaults to 0 - useful w/shared bigrams: 'wol','wor','woo'
    :param legend_pos: where to put the legend: 'top' (default), 'bottom' or None
    :param persistence:  will save the sampled datafrae to  filename (with csv or pkl extension) or None
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param remove_accents: bool if True strips accents (NA on dataframe)
    :param reverse: bool if True look at words from right to left
    :param rows_only: by default returns only the stem and leaf rows. If false, also return the matrix and dataframe
    :param sort_by: default to 'len', can also be 'alpha'
    :param stem_order: how many stem characters per data point to display, defaults to 1
    :param stem_skip: how many stem characters to skip, defaults to 0 - useful to zoom in on a single root letter
    :param stop_words: stop words to remove. None (default), list or builtin EN (English), ES (Spanish) or FR (French)
    """

    # the rows will come back sorted from this call
    rows, alpha_matrix, x = ngram_data(df, alpha_only=alpha_only, ascending=ascending, binary=binary, break_on=break_on,
                                       caps=caps, column=column, compact=compact, display=display,
                                       leaf_order=leaf_order, leaf_skip=leaf_skip, persistence=persistence,
                                       random_state=random_state, remove_accents=remove_accents,
                                       reverse=reverse, rows_only=rows_only, sort_by=sort_by,
                                       stem_order=stem_order, stem_skip=stem_skip, stop_words=stop_words)

    if legend_pos == 'top':
        print('{}: \n{}\nsampled {:>4}\n'.format(column if column else '', x.word.describe(include='all'), display))

    cnt = 0
    find = re.compile("([{}-z?])".format(break_on))
    for i, val in enumerate(rows.index):
        leaves = rows[i]
        mask = '{:<' + str(stem_order) + '}| {}'
        if aggr:
            cnt += int(len(leaves) / leaf_order)
            mask = '{:<' + str(len(str(display))) + '}|{:<' + str(stem_order) + '}| {}'
        if break_on is not None:
            try:
                pos = re.search(find, leaves).start()
            except AttributeError:
                pos = 0
            if pos > 0:
                low = leaves[:pos]
                high = leaves[pos:]
            else:
                low = leaves
                high = ''
            if ascending:
                argsl = (cnt, val, low) if aggr else (val, low)
                argsh = (cnt, val, high) if aggr else (val, high)
            else:
                argsl = (cnt, val, high) if aggr else (val, high)
                argsh = (cnt, val, low) if aggr else (val, low)
        else:
            argsl = (cnt, val, leaves) if aggr else (val, leaves)
        print(mask.format(*argsl))
        if break_on:
            # noinspection PyUnboundLocalVariable
            print(mask.format(*argsh))

    if legend_pos is not None and legend_pos != 'top':
        print('Alpha stem and leaf {}: \n{}\nsampled {:>4}\n'.format(
            column if column else '', x.word.describe(include='all'), display))

    if rows_only:
        return rows
    else:
        return rows, alpha_matrix, x


# noinspection PyTypeChecker
def stem_graphic(df, df2=None, aggregation=True, alpha=0.1, alpha_only=True, ascending=False, ax=None, bar_color='C0',
                 bar_outline=None, break_on=None, caps=True, column=None, combined=None, compact=False,
                 delimiter_color='C3', display=750, figure_only=True, flip_axes=False,
                 font_kw=None, leaf_color='k', leaf_order=1, leaf_skip=0, legend_pos='best',
                 median_color='C4', mirror=False, persistence=None, primary_kw=None,
                 random_state=None, remove_accents=False, reverse=False, secondary=False,
                 show_stem=True, sort_by='len', stop_words=None, stem_order=1, stem_skip=0,
                 title=None, trim_blank=False, underline_color=None):
    """ stem_graphic

    The principal visualization of stemgraphic.alpha is stem_graphic. It offers all the
    options of stem\_text (3.1) and adds automatic title, mirroring, flipping of axes,
    export (to pdf, svg, png, through fig.savefig) and many more options to change the
    visual appearance of the plot (font size, color, background color, underlining and more).

    By providing a secondary text source, the plot will enable comparison through a back-to-back display


    :param df: string, filename, url, list, numpy array, time series, pandas or dask dataframe
    :param df2: string, filename, url, list, numpy array, time series, pandas or dask dataframe (optional).
                for back 2 back stem-and-leaf plots
    :param aggregation: Boolean for sum, else specify function
    :param alpha: opacity of the bars, median and outliers, defaults to 10%
    :param alpha_only: only use stems from a-z alphabet (NA on dataframe)
    :param ascending: stem sorted in ascending order, defaults to True
    :param ax: matplotlib axes instance, usually from a figure or other plot
    :param bar_color: the fill color of the bar representing the leaves
    :param bar_outline: the outline color of the bar representing the leaves
    :param break_on: force a break of the leaves at that letter, the rest of the leaves will appear on the next line
    :param caps: bool, True to be case sensitive, defaults to False, recommended for comparisons.(NA on dataframe)
    :param column: specify which column (string or number) of the dataframe to use, or group of columns (stems)
                   else the frame is assumed to only have one column with words.
    :param combined: list (specific subset to automatically include, say, for comparisons), or None
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param delimiter_color: color of the line between aggregate and stem and stem and leaf
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param figure_only: bool if True (default) returns matplotlib (fig,ax), False returns (fig,ax,df)
    :param flip_axes: X becomes Y and Y becomes X
    :param font_kw: keyword dictionary, font parameters
    :param leaf_color: font color of the leaves
    :param leaf_order: how many leaf digits per data point to display, defaults to 1
    :param leaf_skip: how many leaf characters to skip, defaults to 0 - useful w/shared bigrams: 'wol','wor','woo'
    :param legend_pos: One of 'top', 'bottom', 'best' or None, defaults to 'best'.
    :param median_color: color of the box representing the median
    :param mirror: mirror the plot in the axis of the delimiters
    :param persistence: filename. save sampled data to disk, either as pickle (.pkl) or csv (any other extension)
    :param primary_kw: stem-and-leaf plot additional arguments
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param remove_accents: bool if True strips accents (NA on dataframe)
    :param reverse: bool if True look at words from right to left
    :param secondary: bool if True, this is a secondary plot - mostly used for back-to-back plots
    :param show_stem: bool if True (default) displays the stems
    :param sort_by: default to 'len', can also be 'alpha'
    :param stem_order: how many stem characters per data point to display, defaults to 1
    :param stem_skip: how many stem characters to skip, defaults to 0 - useful to zoom in on a single root letter
    :param stop_words: stop words to remove. None (default), list or builtin EN (English), ES (Spanish) or FR (French)
    :param title: string, or None. When None and source is a file, filename will be used.
    :param trim_blank: remove the blank between the delimiter and the first leaf, defaults to True
    :param underline_color: color of the horizontal line under the leaves, None for no display
    :return: matplotlib figure and axes instance, and dataframe if figure_only is False
    """

    if isinstance(df, str) and title is None:
        title = df[:96]  # max 96 chars for title
    elif title is None:
        # still
        title = ''
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

    leaf_alpha = 1
    if leaf_color is None:
        leaf_color = 'k'
        leaf_alpha = 0

    rows, alpha_matrix, x = ngram_data(
        df,
        alpha_only=alpha_only,
        break_on=break_on,
        caps=caps,
        compact=compact,
        column=column,
        display=display,
        leaf_order=leaf_order,
        leaf_skip=leaf_skip,
        persistence=persistence,
        random_state=random_state,
        remove_accents=remove_accents,
        reverse=reverse,
        rows_only=False,
        sort_by=sort_by,
        stem_order=stem_order,
        stem_skip=stem_skip,
        stop_words=stop_words
    )

    if combined is not None:
        max_leaves = rows[combined].str.len().max()
    else:
        max_leaves = rows.str.len().max()

    if df2:
        if flip_axes:
            warn("Error: flip_axes is not available with back to back stem-and-leaf plots.")
            return None

        _ = ngram_data(
            df2,
            alpha_only=alpha_only,
            break_on=break_on,
            caps=caps,
            column=column,
            display=display,
            leaf_order=leaf_order,
            leaf_skip=leaf_skip,
            random_state=random_state,
            reverse=reverse,
            rows_only=False,
            sort_by=sort_by,
            stem_order=stem_order,
            stem_skip=stem_skip,
            stop_words=stop_words
        )

    fig = None
    if flip_axes:
        height = max_leaves / 8 + 3
        if height < 20:
            height = 20
        width = len(rows) + 3
    else:
        width = max_leaves / 8 + 3
        if width < 20:
            width = 20
        # if df2:
        #    width /= 2  # two charts, need to maximize ratio
        height = len(rows) + 3
    if combined is None:
        combined = rows.index
    else:
        height = len(combined)
        width = max_leaves / 8 + 3

    aggr_offset = -0.5
    aggr_line_offset = 1
    if df2:
        # values = res.index
        # combined = sorted(list(set(values.append(res2.index))))
        aggr_offset = -3.7
        aggr_line_offset = 0.2
        fig, (ax1, ax) = plt.subplots(1, 2, sharey=True, figsize=((width / 4), (height / 4)))
        # ax1 = fig.add_subplot(111, sharey=True)
        plt.box(on=None)
        ax1.axes.get_yaxis().set_visible(False)
        ax1.axes.get_xaxis().set_visible(False)
        ax1.set_xlim(-1, width + 0.05)
        ax1.set_ylim(-1, height + 0.05)
        _ = stem_graphic(df2,  # NOQA
                         ax=ax1, aggregation=mirror and aggregation, alpha_only=alpha_only, ascending=ascending,
                         break_on=break_on, column=column, combined=combined, display=display, flip_axes=False,
                         mirror=not mirror, reverse=reverse, secondary=True, random_state=random_state,
                         show_stem=True, stop_words=stop_words)

    if ax is None:
        fig = plt.figure(figsize=((width / 4), (height / 2)))
        ax = fig.add_axes((0.05, 0.05, 0.9, 0.9),
                          aspect='equal', frameon=False,
                          xlim=(-1, width + 0.05),
                          ylim=(-1, height + 0.05))
    plt.box(on=None)
    ax.axis('off')
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    if df2 or secondary:
        title_offset = -2 if mirror else 4
    else:
        title_offset = 0 if mirror else 2
    if flip_axes:
        ax.set_title(title, y=title_offset)
    else:
        ax.set_title(title, x=title_offset)

    offset = 0
    if mirror:
        ax.set_ylim(ax.get_ylim()[::-1]) if flip_axes else ax.set_xlim(ax.get_xlim()[::-1])
        offset = -2 if secondary else 0.5

    if not ascending:
        ax.set_xlim(ax.get_xlim()[::-1]) if flip_axes else ax.set_ylim(ax.get_ylim()[::-1])

    tot = 0
    min_s = 99999999

    mask = '{:>' + str(len(str(display))) + '}'

    for cnt, item in enumerate(combined):
        stem = item
        try:
            leaf = rows[item]
        except KeyError:
            leaf = ' '
        tot += int(len(leaf) / leaf_order)
        if trim_blank:
            leaf = leaf.strip()

        tot_display = mask.format(tot)
        if flip_axes:
            if aggregation and not (df2 and mirror):
                ax.text(cnt + offset, 0, tot_display, fontsize=aggr_fontsize, rotation=90, color=aggr_fontcolor,
                        bbox={'facecolor': aggr_facecolor, 'alpha': alpha, 'pad': pad} if aggr_facecolor is not None
                        else {'alpha': 0},
                        fontweight=aggr_fontweight, va='center', ha='right' if mirror else 'left')
            # STEM
            if show_stem:
                ax.text(cnt + offset, 1.5, stem, fontweight=stem_fontweight, color=stem_fontcolor, family='monospace',
                        bbox={'facecolor': stem_facecolor, 'alpha': alpha, 'pad': pad} if stem_facecolor is not None
                        else {'alpha': 0},
                        fontsize=stem_fontsize, va='center', ha='right' if mirror else 'left')

            # LEAF
            ax.text(cnt, 2.1, leaf[::-1] if mirror else leaf, fontsize=base_fontsize, color=leaf_color,
                    ha='left', va='top' if mirror else 'bottom', rotation=90, alpha=leaf_alpha, family='monospace',
                    bbox={'facecolor': bar_color, 'edgecolor': bar_outline, 'alpha': alpha, 'pad': pad})

        else:
            if aggregation and not (df2 and mirror):
                ax.text(aggr_offset, cnt + 0.5, tot_display, fontsize=aggr_fontsize, color=aggr_fontcolor,
                        bbox={'facecolor': aggr_facecolor, 'alpha': alpha, 'pad': pad} if aggr_facecolor is not None
                        else {'alpha': 0},
                        fontweight=aggr_fontweight, va='center', ha='right')  # if mirror else 'left')
            # STEM
            if show_stem:
                stem_offset = 2.2
                if secondary and not mirror:
                    stem_offset = -8
                elif df2 and mirror:
                    stem_offset = 2.1
                ax.text(stem_offset, cnt + 0.5, stem, fontweight=stem_fontweight, color=stem_fontcolor,
                        family='monospace',
                        bbox={'facecolor': stem_facecolor, 'alpha': alpha, 'pad': pad} if stem_facecolor is not None
                        else {'alpha': 0},
                        fontsize=stem_fontsize, va='center', ha='left' if mirror else 'right')

            # LEAF
            ax.text(2.6, cnt + 0.5, leaf[::-1] if mirror else leaf, fontsize=base_fontsize, family='monospace',
                    va='center', ha='right' if mirror else 'left', color=leaf_color, alpha=leaf_alpha,
                    bbox={'facecolor': bar_color, 'edgecolor': bar_outline, 'alpha': alpha, 'pad': pad})
            if underline_color:
                ax.hlines(cnt, 2.6, 2.6 + len(leaf) / 2, color=underline_color)

    if flip_axes:
        # noinspection PyUnboundLocalVariable
        ax.hlines(2, min_s, min_s + 1 + cnt, color=delimiter_color, alpha=0.7)
        if aggregation:
            ax.hlines(1, min_s, min_s + 1 + cnt, color=delimiter_color, alpha=0.7)

    else:
        if aggregation and not (df2 and mirror):
            # noinspection PyUnboundLocalVariable
            ax.vlines(aggr_line_offset, 0, 1 + cnt, color=delimiter_color, alpha=0.7)
        if show_stem:
            ax.vlines(2.4, 0, 1 + cnt, color=delimiter_color, alpha=0.7)
    if flip_axes:
        ax.plot(0, height)
    else:
        ax.plot(width, 0)
    if figure_only:
        return fig, ax
    else:
        return fig, ax, x


# noinspection PyPep8Naming
def stem_freq_plot(df, alpha_only=False, asFigure=False, column=None, compact=True, caps=False,  # NOQA
                   display=2600, interactive=True, kind='barh', leaf_order=1, leaf_skip=0, random_state=None,
                   stem_order=1, stem_skip=0, stop_words=None):
    """ stem_freq_plot

    Word frequency plot is the most common visualization in NLP. In this version it supports stem-and-leaf / n-grams.

    Each row is the stem, and similar leaves are grouped together and each different group is stacked
    in bar charts.

    Default is horizontal bar chart, but vertical, histograms, area charts and even pie charts are
    supported by this one visualization.


    :param df: string, filename, url, list, numpy array, time series, pandas or dask dataframe
    :param alpha_only: only use stems from a-z alphabet (NA on dataframe)
    :param asFigure: return plot as plotly figure (for web applications)
    :param column: specify which column (string or number) of the dataframe to use, or group of columns (stems)
                   else the frame is assumed to only have one column with words.
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param caps: bool, True to be case sensitive, defaults to False, recommended for comparisons.(NA on dataframe)
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param interactive: if cufflinks is loaded, renders as interactive plot in nebook
    :param kind: defaults to 'barh'. One of 'bar','barh','area','hist'. Non-interactive also supports 'pie'
    :param leaf_order: how many leaf digits per data point to display, defaults to 1
    :param leaf_skip: how many leaf characters to skip, defaults to 0 - useful w/shared bigrams: 'wol','wor','woo'
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param stem_order: how many stem characters per data point to display, defaults to 1
    :param stem_skip: how many stem characters to skip, defaults to 0 - useful to zoom in on a single root letter
    :param stop_words: stop words to remove. None (default), list or builtin EN (English), ES (Spanish) or FR (French)
    :return:
    """
    rows, alpha_matrix, x = ngram_data(
        df,
        alpha_only=alpha_only,
        caps=caps,
        compact=compact,
        display=display,
        leaf_order=leaf_order,
        leaf_skip=leaf_skip,
        rows_only=False,
        random_state=random_state,
        stem_order=stem_order,
        stem_skip=stem_skip,
        stop_words=stop_words,
    )

    if not interactive:
        plt.figure(figsize=(20, 20))
    if isinstance(df, str):
        title = 'stem-and-leaf stacked frequency for {}'.format(df)
    else:
        title = 'stem-and-leaf stacked frequency'
    if interactive:
        try:
            if column:
                # one or multiple "columns" specified, we filter those stems
                fig = alpha_matrix.loc[column].word.iplot(kind=kind, barmode='stack', asFigure=asFigure, title=title)
            else:
                alpha_matrix.word.iplot(kind=kind, barmode='stack', asFigure=asFigure, title=title)
        except AttributeError:
            warn('Interactive plot requested, but cufflinks not loaded. Falling back to matplotlib.')
            alpha_matrix.word.plot(kind=kind, stacked=True, legend=None, title=title)
    else:
        alpha_matrix.word.plot(kind=kind, stacked=True, legend=None, title=title)
    return x


def stem_sunburst(words, alpha_only=True, ascending=False, caps=False, compact=True, display=None, hole=True,
                  label=True, leaf_order=1, leaf_skip=0, median=True, ngram=False, random_state=None, sort_by='alpha',
                  statistics=True, stem_order=1, stem_skip=0, stop_words=None, top=0):
    """ stem_sunburst

    Stem-and-leaf based sunburst. See sunburst for details

    :param words: string, filename, url, list, numpy array, time series, pandas or dask dataframe
    :param alpha_only: only use stems from a-z alphabet (NA on dataframe)
    :param ascending: stem sorted in ascending order, defaults to True
    :param caps: bool, True to be case sensitive, defaults to False, recommended for comparisons.(NA on dataframe)
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param hole: bool if True (default) leave space in middle for statistics
    :param label: bool if True display words centered at coordinate
    :param leaf_order: how many leaf digits per data point to display, defaults to 1
    :param leaf_skip: how many leaf characters to skip, defaults to 0 - useful w/shared bigrams: 'wol','wor','woo'
    :param median: bool if True (default) display an origin and a median mark
    :param ngram: bool if True display full n-gram as leaf label
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param sort_by: sort by 'alpha' (default) or 'count'
    :param statistics: bool if True (default) displays statistics in center - hole has to be True
    :param stem_order: how many stem characters per data point to display, defaults to 1
    :param stem_skip: how many stem characters to skip, defaults to 0 - useful to zoom in on a single root letter
    :param stop_words: stop words to remove. None (default), list or builtin EN (English), ES (Spanish) or FR (French)
    :param top: how many different words to count by order frequency. If negative, this will be the least frequent
    :return:
    """
    return sunburst(words, alpha_only=alpha_only, ascending=ascending, caps=caps, compact=compact, display=display,
                    hole=hole, label=label, leaf_order=leaf_order, leaf_skip=leaf_skip, median=median, ngram=ngram,
                    random_state=random_state, sort_by=sort_by, statistics=statistics,
                    stem_order=stem_order, stem_skip=stem_skip, stop_words=stop_words, top=top)


# noinspection PyTypeChecker,PyTypeChecker,PyTypeChecker,PyTypeChecker,PyTypeChecker,PyTypeChecker
def sunburst(words, alpha_only=True, ascending=False, caps=False, compact=True, display=None, hole=True,
             label=True, leaf_order=1, leaf_skip=0, median=True, ngram=True, random_state=None, sort_by='alpha',
             statistics=True, stem_order=1, stem_skip=0, stop_words=None, top=40):
    """ sunburst

     Word sunburst charts are similar to pie or donut charts, but add some statistics
     in the middle of the chart, including the percentage of total words targeted for a given
    number of unique words (ie. top 50 words, 48\% coverage).

    With stem-and-leaf, the first level of the sunburst represents the stem and the second
    level subdivides each stem by leaves.

    :param words: string, filename, url, list, numpy array, time series, pandas or dask dataframe
    :param alpha_only: only use stems from a-z alphabet (NA on dataframe)
    :param ascending: stem sorted in ascending order, defaults to True
    :param caps: bool, True to be case sensitive, defaults to False, recommended for comparisons.(NA on dataframe)
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param hole: bool if True (default) leave space in middle for statistics
    :param label: bool if True display words centered at coordinate
    :param leaf_order: how many leaf digits per data point to display, defaults to 1
    :param leaf_skip: how many leaf characters to skip, defaults to 0 - useful w/shared bigrams: 'wol','wor','woo'
    :param median: bool if True (default) display an origin and a median mark
    :param ngram: bool if True (default) display full n-gram as leaf label
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param statistics: bool if True (default) displays statistics in center - hole has to be True
    :param sort_by: sort by 'alpha' (default) or 'count'
    :param stem_order: how many stem characters per data point to display, defaults to 1
    :param stem_skip: how many stem characters to skip, defaults to 0 - useful to zoom in on a single root letter
    :param stop_words: stop words to remove. None (default), list or builtin EN (English), ES (Spanish) or FR (French)
    :param top: how many different words to count by order frequency. If negative, this will be the least frequent
    :return: matplotlib polar ax, dataframe
    """
    if isinstance(words, str):
        filename = words
    else:
        filename = 'data'

    _, alpha_matrix, x = ngram_data(
        words,
        alpha_only=alpha_only,
        compact=compact,
        display=display,
        leaf_order=leaf_order,
        leaf_skip=leaf_skip,
        rows_only=False,
        random_state=random_state,
        sort_by=sort_by,
        stem_order=stem_order,
        stem_skip=stem_skip,
        stop_words=stop_words,
        caps=caps)

    fig, pol_ax = plt.subplots(1, 1, figsize=(12, 12), subplot_kw=dict(projection='polar'))
    pol_ax.grid(color='#dfdfdf')  # Color the grid

    pol_ax.set_theta_zero_location('N')  # we start at the top
    pol_ax.set_theta_direction(-1)  # and go clockwise
    pol_ax.set_rlabel_position(0)
    pol_ax.set_axis_off()
    if median:
        # start marker
        if leaf_order is None:
            pol_ax.plot((0, 0), (1.98, 2.02), color='r')
        else:
            pol_ax.plot((0, 0), (0.98, 1.02), color='r')
    if top < 0:
        sum_by_len = x.word.value_counts()[top:].sum()
    else:
        sum_by_len = x.word.value_counts()[:top].sum()
    sum_by_stem = alpha_matrix.word.T.sum()
    sum_of_sum = sum_by_stem.sum()
    qty_unique_ngrams = len(x.ngram.unique())

    if stem_order is None:
        if leaf_order is None:
            col = 'word'  # dealing with words
        else:
            col = 'ngram'  # partial stem and leaf, words are n-grams
            top = qty_unique_ngrams if top == 0 else top
        # We are dealing with words, then.
        d = np.pi * 2 / sum_by_len
        if top < 0:
            normalized = x[col].value_counts()[top:] * d
        else:
            normalized = x[col].value_counts()[:top] * d
        if sort_by == 'alpha':
            normalized.sort_index(inplace=True, ascending=ascending)
        elif sort_by == 'count':
            normalized.sort_values(inplace=True, ascending=ascending)
        # elif sort_by == 'len':
        #
        plot_sunburst_level(normalized, ax=pol_ax, label=label, level=0.5 if hole else 0, ngram=ngram)
        if median:
            pol_ax.plot((np.pi, np.pi), (0.48, 2.02), color='r')
        if hole and statistics:
            pol_ax.text(0, 0, '{} words\n{:.2f} %'.format(sum_by_len, sum_by_len / len(x.word) * 100),
                        size=20, ha='center', va='center')
        if top < 0:
            plt.figtext(.5, 0.95, 'Bottom {} {}s from'.format(abs(top), col), ha='center')
            pol_ax.set_title('{}'.format(filename), fontsize=8)
        else:
            plt.figtext(.5, 0.95, 'Top {} {}s from'.format(top, col), ha='center')
            pol_ax.set_title('{}'.format(filename), fontsize=8)
    else:
        # A variation of a stem-and-leaf polar plot
        d = np.pi * 2 / sum_of_sum
        normalized = alpha_matrix.word.T.sum() * d
        if sort_by == 'alpha':
            normalized.sort_index(inplace=True, ascending=ascending)
        elif sort_by == 'count':
            # default is alpha
            normalized.sort_values(inplace=True, ascending=ascending)
            # sum_by_stem.sort_values(inplace=True, ascending=ascending)
        hole_adjust = 0.4 if hole else 0
        values = plot_sunburst_level(normalized, ax=pol_ax, label=label, level=hole_adjust,
                                     ngram=ngram, plot=stem_order)
        if hole and statistics:
            pol_ax.text(0, 0, '{:.2f} %'.format(sum_of_sum / len(x.word) * 100), size=12, ha='center',
                        va='center')

        if leaf_order is not None:
            stems = list(normalized.index)
            for i, stem in enumerate(stems):
                try:
                    leaves = alpha_matrix.word.T[stem].fillna(0) * d
                    if sort_by == 'count':
                        leaves.sort_values(inplace=True, ascending=ascending)
                    plot_sunburst_level(leaves, offset=values[i],
                                        level=1 if stem_order else hole_adjust, ax=pol_ax,
                                        ngram=ngram,
                                        stem=stem, vis=0.001)
                except KeyError:
                    pass
                if stem_order:
                    plt.figtext(.5, 0.95, 'Stem-and-leaves from', ha='center')
                    pol_ax.set_title(filename, fontsize=8)
                    if median:
                        pol_ax.plot((np.pi, np.pi), (0, 1.02), color='r')
                else:
                    plt.figtext(.5, 0.95, 'Leaves from', ha='center')
                    pol_ax.set_title(filename, fontsize=8)
                    if median:
                        pol_ax.plot((np.pi, np.pi), (0, 1.02), color='r')
        else:
            plt.figtext(.5, 0.95, 'Stems from', ha='center')
            pol_ax.set_title(filename, fontsize=8)
            if median:
                pol_ax.plot((np.pi, np.pi), (0, 1.02), color='r')

    return pol_ax, x


# noinspection PyPep8Naming,PyTypeChecker,PyTypeChecker
def word_freq_plot(src, alpha_only=False, ascending=False, asFigure=False, caps=False, display=None,  # NOQA
                   interactive=True, kind='barh', random_state=None, sort_by='count', stop_words=None, top=100):
    """ word frequency bar chart.

    This function creates a classical word frequency bar chart.

    :param src: Either a filename including path, a url or a ready to process text in a dataframe or a tokenized format.
    :param alpha_only: words only if True, words and numbers if False
    :param ascending: stem sorted in ascending order, defaults to True
    :param asFigure: if interactive, the function will return a plotly figure instead of a matplotlib ax
    :param caps: keep capitalization (True, False)
    :param display: if specified, sample that quantity of words
    :param interactive: interactive graphic (True, False)
    :param kind: horizontal bar chart (barh) - also 'bar', 'area', 'hist' and non interactive 'kde' and 'pie'
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param sort_by: default to 'count', can also be 'alpha'
    :param stop_words: a list of words to ignore
    :param top: how many different words to count by order frequency. If negative, this will be the least frequent
    :return: text as dataframe and plotly figure or matplotlib ax
    """
    _, _, x = ngram_data(
        src,
        alpha_only=alpha_only,
        caps=caps,
        compact=True,
        display=display,
        leaf_order=None,
        rows_only=False,
        random_state=random_state,
        sort_by=sort_by,
        stem_order=None,
        stop_words=stop_words
    )

    if stop_words is not None:
        x = x[~x.word.isin(stop_words)]
    # if sort_by == 'alpha':
    #    x.sort_values(by='word', inplace=True, ascending=ascending)
    # elif sort_by == 'count':
    #    x.word.value_counts().sort_values(ascending=ascending, inplace=True)
    # elif sort_by == 'len':
    #    x = x[x.word.str.len().sort_values().index]
    if isinstance(src, str):
        if top < 0:
            title = 'Bottom {} word frequency for {}'.format(min(len(x.word.value_counts()), abs(top)), src)
        else:
            title = 'Top {} word frequency for {}'.format(min(len(x.word.value_counts()), top), src)
    else:
        title = 'word frequency'
    if interactive:
        try:
            if top < 0:
                figure = x.word.value_counts().sort_values(ascending=ascending)[top:].iplot(kind=kind,
                                                                                            asFigure=asFigure,
                                                                                            title=title)
            else:
                if sort_by == 'alpha':
                    figure = x.word.value_counts()[:top].sort_index(ascending=ascending).iplot(kind=kind,
                                                                                               asFigure=asFigure,
                                                                                               title=title)
                else:
                    figure = x.word.value_counts()[:top].sort_values(ascending=ascending).iplot(kind=kind,
                                                                                                asFigure=asFigure,
                                                                                                title=title)
        except AttributeError:
            warn('Interactive plot requested, but cufflinks not loaded. Falling back to matplotlib.')
            plt.figure(figsize=(20, 20))
            if top < 0:
                ax = x.word.value_counts()[top:].plot(kind=kind, title=title)
            else:
                if sort_by == 'alpha':
                    ax = x.word.value_counts()[:top].sort_index(ascending=ascending).plot(kind=kind, title=title)
                else:
                    ax = x.word.value_counts()[:top].sort_values(ascending=ascending).plot(kind=kind, title=title)
            figure = ax  # special case, requested interactive, but unavailable, so return matplotlib ax
    else:
        plt.figure(figsize=(20, 20))
        if top < 0:
            ax = x.word.value_counts()[top:].sort_values(ascending=ascending).plot(kind=kind, title=title)
        else:
            if sort_by == 'alpha':
                ax = x.word.value_counts()[:top].sort_index(ascending=ascending).plot(kind=kind, title=title)
            else:
                ax = x.word.value_counts()[:top].sort_values(ascending=ascending).plot(kind=kind, title=title)
    # noinspection PyUnboundLocalVariable,PyUnboundLocalVariable
    return x, figure if interactive else ax


def word_radar(word, comparisons, ascending=True, display=100, label=True, metric=None,
               min_distance=1, max_distance=None, random_state=None, sort_by='alpha'):
    """ word_radar

    Radar plot based on words. Currently, the only type of radar plot supported. See `radar' for more detail.

    :param word: string, the reference word that will be placed in the middle
    :param comparisons: external file, list or string or dataframe of words
    :param ascending: bool if the sort is ascending
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param label: bool if True display words centered at coordinate
    :param metric: any metric function accepting two values and returning that metric in a range from 0 to x
    :param min_distance: minimum distance based on metric to include a word for display
    :param max_distance: maximum distance based on metric to include a word for display
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param sort_by: default to 'alpha', can also be 'len'
    :return:
    """

    return radar(word, comparisons, ascending=ascending, display=display, label=label, metric=metric,
                 min_distance=min_distance, max_distance=max_distance, random_state=random_state, sort_by=sort_by)


def word_scatter(src1, src2, src3=None, alpha=0.5, alpha_only=True, ascending=True, asFigure=False, ax=None, caps=False,
                 compact=True, display=None, fig_xy=None, interactive=True, jitter=False, label=False,
                 leaf_order=None, leaf_skip=0, log_scale=True, normalize=None, percentage=None, random_state=None,
                 sort_by='alpha', stem_order=None, stem_skip=0, stop_words=None, whole=False):
    """ word_scatter

    Scatter compares the word frequency of two sources, on each axis. Each data point Z value is the word
    or stem-and-leaf value, while the X axis reflects that word count in one source and the Y axis re-
    flect the same word count in the other source, in two different colors. If one word is more common
    on the first source it will be displayed in one color, and if it is more common in the second source, it
    will be displayed in a different color. The values that are the same for both sources will be displayed
    in a third color (default colors are blue, black and pink. In interactive mode, hovering the data point
    will give the precise counts on each axis along with the word itself, and filtering by category is done
    by clicking on the category in the legend.

    :param src1: string, filename, url, list, numpy array, time series, pandas or dask dataframe
    :param src2: string, filename, url, list, numpy array, time series, pandas or dask dataframe
    :param src3: string, filename, url, list, numpy array, time series, pandas or dask dataframe, optional
    :param alpha: opacity of the bars, median and outliers, defaults to 10%
    :param alpha_only: only use stems from a-z alphabet (NA on dataframe)
    :param ascending: stem sorted in ascending order, defaults to True
    :param asFigure: return plot as plotly figure (for web applications)
    :param ax: matplotlib axes instance, usually from a figure or other plot
    :param caps: bool, True to be case sensitive, defaults to False, recommended for comparisons.(NA on dataframe)
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param fig_xy:  tuple for matplotlib figsize, defaults to (20,20)
    :param interactive: if cufflinks is loaded, renders as interactive plot in notebook
    :param jitter: random noise added to help see multiple data points sharing the same coordinate
    :param label: bool if True display words centered at coordinate
    :param leaf_order: how many leaf digits per data point to display, defaults to 1
    :param leaf_skip: how many leaf characters to skip, defaults to 0 - useful w/shared bigrams: 'wol','wor','woo'
    :param log_scale: bool if True (default) uses log scale axes
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param sort_by: sort by 'alpha' or 'count' (default)
    :param stem_order: how many stem characters per data point to display, defaults to 1
    :param stem_skip: how many stem characters to skip, defaults to 0 - useful to zoom in on a single root letter
    :param stop_words: stop words to remove. None (default), list or builtin EN (English), ES (Spanish) or FR (French)
    :param whole: for normalized or percentage, use whole integer values (round)
    :return: matplotlib polar ax, dataframe
    """
    return scatter(src1=src1, src2=src2, src3=src3, alpha=alpha, alpha_only=alpha_only, asFigure=asFigure,
                   ascending=ascending, ax=ax, caps=caps, compact=compact, display=display, fig_xy=fig_xy,
                   interactive=interactive, jitter=jitter, label=label, leaf_order=leaf_order, leaf_skip=leaf_skip,
                   log_scale=log_scale, normalize=normalize, percentage=percentage, random_state=random_state,
                   sort_by=sort_by, stem_order=stem_order, stem_skip=stem_skip,stop_words=stop_words, whole=whole)


def word_sunburst(words, alpha_only=True, ascending=False, caps=False, compact=True, display=None, hole=True,
                  label=True, leaf_order=None, leaf_skip=0, median=True, ngram=True, random_state=None, sort_by='alpha',
                  statistics=True, stem_order=None, stem_skip=0,  stop_words=None, top=40):
    """ word_sunburst

    Word based sunburst. See sunburst for details

    :param words: string, filename, url, list, numpy array, time series, pandas or dask dataframe
    :param alpha_only: only use stems from a-z alphabet (NA on dataframe)
    :param ascending: stem sorted in ascending order, defaults to True
    :param caps: bool, True to be case sensitive, defaults to False, recommended for comparisons.(NA on dataframe)
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param hole: bool if True (default) leave space in middle for statistics
    :param label: bool if True display words centered at coordinate
    :param leaf_order: how many leaf digits per data point to display, defaults to 1
    :param leaf_skip: how many leaf characters to skip, defaults to 0 - useful w/shared bigrams: 'wol','wor','woo'
    :param median: bool if True (default) display an origin and a median mark
    :param ngram: bool if True (default) display full n-gram as leaf label
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param statistics: bool if True (default) displays statistics in center - hole has to be True
    :param sort_by: sort by 'alpha' (default) or 'count'
    :param stem_order: how many stem characters per data point to display, defaults to 1
    :param stem_skip: how many stem characters to skip, defaults to 0 - useful to zoom in on a single root letter
    :param stop_words: stop words to remove. None (default), list or builtin EN (English), ES (Spanish) or FR (French)
    :param top: how many different words to count by order frequency. If negative, this will be the least frequent
    :return:
    """
    return sunburst(words, alpha_only=alpha_only, ascending=ascending, caps=caps, compact=compact, display=display,
                    hole=hole, label=label, leaf_order=leaf_order, leaf_skip=leaf_skip, median=median, ngram=ngram,
                    random_state=random_state, sort_by=sort_by, statistics=statistics,
                    stem_order=stem_order, stem_skip=stem_skip, stop_words=stop_words, top=top)
