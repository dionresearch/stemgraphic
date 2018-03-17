""" helpers.py

Helper functions for stemgraphic.
"""
import matplotlib.tri as tri
import numpy as np
import pandas as pd
import pickle
from warnings import warn
try:
    import dask.dataframe as dd
except ImportError:
    dd = False


def key_calc(stem, leaf, scale):
    """Calculates a value from a stem, a leaf and a scale.

    :param stem:
    :param leaf:
    :param scale:
    :return: calculated values
    """
    return (int(leaf) / 10 + int(stem)) * float(scale)


def legend(ax, x, y, asc, flip_axes, mirror, stem, leaf, scale, delimiter_color, aggregation=True, cur_font=None,
           display=10, pos='best', unit=''):
    """ legend

    Builds a graphical legend for numerical stem-and-leaf plots.

    :param display:
    :param cur_font:
    :param ax:
    :param x:
    :param y:
    :param pos:
    :param asc:
    :param flip_axes:
    :param mirror:
    :param stem:
    :param leaf:
    :param scale:
    :param delimiter_color:
    :param unit:
    :param aggregation:
    """

    if pos is None:
        return
    aggr_fontsize = cur_font.get_size() - 2
    if (mirror and not flip_axes) or (flip_axes and not asc):
        ha = 'right'
        formula = '{2}{1} =        x{0} = '
        offset = len(str(scale)) + 3.1 + (len(stem) + len(leaf)) / 1.7
        secondary = -2.5 if asc and flip_axes else -1.6
        key_text = 'Key: leaf|stem{}'.format('|aggr' if aggregation else '')
    else:
        ha = 'left'
        formula = '  =         x{0} = {1}{2}'
        offset = 3.1
        secondary = 0.1
        key_text = 'Key: {}stem|leaf'.format('aggr|' if aggregation else '')
    start_at = (len(stem)*2 + 11 + len(str(scale)) + len(leaf)) / 1.7
    if pos == 'short':
        ax.text(x - start_at, y + 2, ' x {}'.format(scale),
                va='center', ha=ha, fontproperties=cur_font)
    else:
        if aggregation:
            ax.text(x - start_at - 1, y + 2, key_text,
                    va='center', ha=ha, fontproperties=cur_font)
            ax.text(x - start_at - 2, y + 1, display,
                    fontsize=aggr_fontsize - 2, va='center', ha=ha)
        cur_font.set_weight('bold')
        ax.text(x - start_at - 1, y + 1, stem,
                va='center', ha=ha, fontproperties=cur_font)
        ax.text(x - start_at + (1 + len(leaf) + offset) / 1.7, y + 1, stem,
                va='center', ha=ha, fontproperties=cur_font)
        cur_font.set_weight('normal')
        ax.text(x - start_at + (len(stem) + len(leaf)) / 1.7, y + 1,
                formula.format(scale, key_calc(stem, leaf, scale), unit),
                va='center', ha=ha, fontproperties=cur_font)
        cur_font.set_style('italic')
        ax.text(x - start_at + 0.3, y + 1, leaf, bbox={'facecolor': 'C0', 'alpha': 0.15, 'pad': 2},
                va='center', ha=ha, fontproperties=cur_font)
        ax.text(x - start_at + (len(stem) + offset + len(leaf) + 0.6)/1.7 + secondary,
                y + 1, '.'+leaf, va='center', ha=ha, fontproperties=cur_font)

        if flip_axes:
            ax.vlines(x - start_at, y + 0.5, y + 1.5, color=delimiter_color, alpha=0.7)
            if aggregation:
                ax.vlines(x - start_at-1, y + 0.5, y + 1.5, color=delimiter_color, alpha=0.7)
        else:
            ax.vlines(x - start_at + 0.1, y + 0.5, y + 1.5, color=delimiter_color, alpha=0.7)
            if aggregation:
                ax.vlines(x - start_at - 1.1, y + 0.5, y + 1.5, color=delimiter_color, alpha=0.7)


def min_max_count(x, column=0):
    """ min_max_count

    Handles min, max and count. This works on numpy, lists, pandas and dask dataframes.

    :param column:
    :param x: list, numpy array, series, pandas or dask dataframe
    :return: min, max and count
    """
    if dd and type(x) in (dd.core.DataFrame, dd.core.Series):
        omin, omax, count = dd.compute(x.min(), x.max(), x.count())
    elif type(x) in (pd.DataFrame, pd.Series):
        omin = x.min()
        omax = x.max()
        count = len(x)
    else:
        omin = min(x)
        omax = max(x)
        count = len(x)

    return omin, omax, int(count)


def npy_save(path, array):
    if path[-4:] != '.npy':
        path += '.npy'
    with open(path, 'wb+') as f:
        np.save(f, array, allow_pickle=False)
    return path


def npy_load(path):
    if path[-4:] != '.npy':
        warn("Not a numpy NPY file.")
        return None
    return np.load(path)


def pkl_save(path, array):
    if path[-4:] != '.pkl':
        path += '.pkl'
    with open(path, 'wb+') as f:
        pickle.dump(array, f)
    return path


def pkl_load(path):
    if path[-4:] != '.pkl':
        warn("Not a PKL file.")
        return None
    with open(path, 'rb') as f:
        matrix = pickle.load(f)
    return matrix


def percentile(data, alpha):
    """ percentile

    :param data: list, numpy array, time series or pandas dataframe
    :param alpha: between 0 and 0.5 proportion to select on each side of the distribution
    :return: the actual value at that percentile
    """
    n = sorted(data)
    low = int(round(alpha * len(data) + 0.5))
    high = int(round((1-alpha) * len(data) + 0.5))
    return n[low - 1], n[high - 1]


def stack_columns(row):
    """ stack_columns

    stack multiple columns into a single stacked value
    :param row: a row of letters
    :return: stacked string
    """
    row = row.dropna()
    stack = ''
    for i, col in row.iteritems():
        stack += (str(i)*int(col))
    return stack


#: Typographical apostrophe - ex: I’m, l’arbre
APOSTROPHE = '’'

#: Straight quote mark - ex: 'INCONCEIVABLE'
QUOTE = '\''

#: Double straight quote mark
DOUBLE_QUOTE = '\"'

#: empty
EMPTY = b' '

#: for typesetting overlap
OVER = b'\xd6\xb1'

#: Characters to filter. Does a relatively good job on a majority of texts
#: '- ' and '–' is to skip quotes in many plays and dialogues in books, especially French.
CHAR_FILTER = [
    '\t', '\n', '\\', '/', '`', '*', '_', '{', '}', '[', ']', '(', ')', '<', '>',
    '#', '=', '+', '- ', '–', '.', ';', ':', '!', '?', '|', '$', QUOTE, DOUBLE_QUOTE, '…'
]


#: Similar purpose to CHAR_FILTER, ut keeps the period. The last word of each sentence will end with a '.'
#: Useful for manipulating the dataframe returned by the various visualizations and ngram_data,
#: to break down frequencies by sentence instead of the full text or list.
NO_PERIOD_FILTER = [
    '\t', '\n', '\\', '/', '`', '*', '_', '{', '}', '[', ']', '(', ')', '<', '>',
    '#', '=', '+', '- ', '–', ';', ':', '!', '?', '|', '$', QUOTE, DOUBLE_QUOTE
]


#: Default definition of standard letters
#: remove_accent has to be called explicitely for any of these letters to match their
#: accented counterparts
LETTERS = 'abcdefghijklmnopqrstuvwxyz'

#: List of non alpha characters. Temporary - I want to balance flexibility with convenience, but
#: still looking at options.
NON_ALPHA = [
    '-', '+', '/', '[', ']', '_', '£',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',
    ';',
    QUOTE, DOUBLE_QUOTE, APOSTROPHE, EMPTY, OVER,
    '?',
    '¡', '¿',  # spanish
    '«', '»',
    '“', '”',
    '-', '—',

]

