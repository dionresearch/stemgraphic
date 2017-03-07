import pandas as pd
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
    """

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
        ax.text(x - start_at + 0.3, y + 1, leaf, bbox={'facecolor': 'blue', 'alpha': 0.2, 'pad': 2},
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
    """Handles min, max and count. This works on numpy, lists, pandas and dask dataframes.

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


def percentile(data, alpha):
    """

    :param data: list, numpy array, time series or pandas dataframe
    :param alpha: between 0 and 0.5 proportion to select on each side of the distribution
    :return: the actual value at that percentile
    """
    n = sorted(data)
    l = int(round(alpha * len(data) + 0.5))
    h = int(round((1-alpha) * len(data) + 0.5))
    return n[l - 1], n[h - 1]
