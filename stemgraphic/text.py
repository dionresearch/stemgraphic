import math
import numpy as np
from operator import itemgetter

from .helpers import *


def stem_data(x, column=None, display=300, leaf_order=1, break_on=None, scale=None, compact=False, omin=None,
              omax=None, outliers=False,  persistence=None, random_state=None, total_rows=None, trim=False, zoom=None):
    """ Returns scale factor, key label and list of rows.

    :param x: list, numpy array, time series, pandas or dask dataframe
    :param column: specify which column (string or number) of the dataframe to use,
                   else the first numerical is selected
    :param display:
    :param leaf_order:
    :param break_on:
    :param scale:
    :param compact:
    :param outliers:
    :param omin:
    :param omax:
    :param persistence: persist sampled dataframe
    :param random_state:
    :param total_rows:
    :param trim:
    :param zoom:
    """
    rows = []
    # Multivariate or not
    try:
        cols = len(x.columns)
    except AttributeError:
        # wasn't a multi column data frame, might be a list
        cols = 1

    if cols > 1:
        if column is None:
            # We have to figure out the first numerical column on our own
            start_at = 1 if x.columns[0] == 'id' else 0
            for i in range(start_at, len(x.columns)):
                if x.dtypes[i] in ('int64', 'float64'):
                    column = i
                    break
        if dd:
            x = x[x.columns.values[column]]
        else:
            x = x.ix[:, column]

    # Sampling or not we need the absolute min/max
    if omin is None or omax is None or total_rows is None:
        omin, omax, total_rows = min_max_count(x, column)  # very expensive if on disk, don't do it twice

    n = total_rows
    if n == 0:
        return None
    elif n > display:
        try:
            x = x.sample(n=display, random_state=random_state).values
        except TypeError:
            # We are here due to dask not supporting n=. We'll use less precise frac instead
            frac = display / n
            x = x.sample(frac=frac, random_state=random_state).compute().values
        if persistence is not None:
                if persistence[-4:] == '.pkl':
                    x.to_pickle(persistence)
                else:
                    x.to_csv(persistence)  # TODO: add feather, hdf5 etc
        n = display

    if n <= 300:
        # Dixon
        lines = math.floor(10 * math.log(n, 10))
    else:
        # Velleman
        lines = math.floor(2 * math.sqrt(n))

    try:
        xmin = x.min()
        xmax = x.max()
    except AttributeError:
        xmin = min(x)
        xmax = max(x)

    try:
        spread = xmax - xmin
    except TypeError:
        print("Column data appears to be non numerical. Specify a numeric column.")
        return None

    # we will trim on the sample, or the whole data set
    lowest, highest = percentile(x, trim) if trim else xmin, xmax
    # scale_factor = as small as possible but lines * S must be >= spread
    r_value = spread / lines
    if scale:  # we were passed a scale, use it
        scale_factor = scale
    else:  # The bulk of the logic to figure out the best scaling and visualization
        scale_factor = pow(10, math.ceil(math.log(r_value, 10)))
        check = math.floor(xmax / scale_factor - xmin / scale_factor + 1)
        if check > lines:
            scale_factor *= 10
        elif (check < 7 and n >= 45) or check < 3:
            scale_factor /= 10  # 30 lines on avg, up to 60 some lines max by bumping the scale
        elif math.floor(check) * 2 <= lines + 1 and break_on is None:
            break_on = 5
        if zoom == -1 and break_on == 5:
            break_on = None
        elif zoom == -1:
            break_on = 5
            scale_factor /= 10
        elif zoom == 1 and break_on == 5:
            scale_factor *= 10
        elif zoom == 1:
            break_on = 5
            scale_factor *= 10

    if break_on is None:
        break_on = 10

    truncate_factor = scale_factor / pow(10, leaf_order)

    # Now that we have a scale, we are going to round to it, trim outliers and split stem and leaf
    rounded_data = [np.round(item / truncate_factor) * truncate_factor for item in x if lowest <= item <= highest]
    data = [math.modf(val / scale_factor) for val in rounded_data]
    sorted_data = sorted(data, key=itemgetter(1, 0))
    current_stem = None
    current_leaf = None
    previous_mod = 0
    row = ''
    sign_transition = False
    if xmin < 0 < xmax:
        sign_transition = True
    if outliers:
        row = '{}\n    ¡'.format(omin)
    for leaf, stem in sorted_data:
        if stem == current_stem:
            ileaf = round(leaf * 10)
            if sign_transition and stem == 0 and abs(leaf) == leaf:
                sign_transition = False
                rows.append(row)
                row = '{:>3} | '.format(int(stem))
            elif current_stem is not None and ileaf >= break_on == 5 and previous_mod > (ileaf % break_on):
                rows.append(row)
                row = '    | '
            elif leaf_order > 1:
                row += ' '
            previous_mod = (ileaf % break_on)
            row += str(round(abs(leaf), 1 + leaf_order))[2:leaf_order + 2]
        else:
            if row != '':
                rows.append(row)
            if current_stem is not None and not compact:
                if break_on == 5 and row[0:4] != '    ':
                    row = '    | '
                    rows.append(row)
                for missing in range(int(current_stem) + 1, int(stem)):
                    if int(current_stem) < 0 and missing == 0:
                        neg_zero = '{:>3} |'.format("-0")
                        rows.append(neg_zero)
                    empty_row = '{:>3} |'.format(missing)
                    rows.append(empty_row)
                    if break_on == 5:
                        rows.append('    | ')

            current_leaf = str(round(abs(leaf), 1 + leaf_order))[2:leaf_order + 2].zfill(leaf_order)
            if current_stem and int(current_leaf) >= break_on:
                row = '{:>3} | '.format(int(stem))
                rows.append(row)
                stem_ind = '   '
            else:
                stem_ind = int(stem)
            row = '{:>3} | {}'.format("-0" if stem == 0 and abs(leaf) != leaf else stem_ind, current_leaf)
            current_stem = stem

    # Potentially catching a last row
    rows.append(row)
    if outliers:
        rows.append('    !\n{}'.format(omax))
    key_label = "{}|{}".format(int(current_stem), current_leaf)
    return scale_factor, key_label, rows


def stem_dot(df, asc=True, break_on=None, column=None, compact=False, display=300,
             legend_pos='best', outliers=True, random_state=None, scale=None, trim=False, unit='', zoom=None):
    """

    :param df: list, numpy array, time series, pandas or dask dataframe
    :param asc:
    :param break_on:
    :param column: specify which column (string or number) of the dataframe to use,
                   else the first numerical is selected
    :param compact:
    :param display:
    :param legend_pos:
    :param outliers:
    :param random_state:
    :param scale:
    :param trim:
    :param unit:
    :param zoom:
    """
    x = df if column is None else df[column]
    scale, pair, rows = stem_data(x,  break_on=break_on, column=column, compact=compact,
                                  display=display, outliers=outliers, random_state=random_state,
                                  scale=scale, trim=trim, zoom=zoom)
    if legend_pos == 'top':
        st, lf = pair.split('|')
        print('Key: \n{} => {}.{}x{} = {} {}'.format(pair, st, lf, scale, key_calc(st, lf, scale), unit))

    ordered_rows = rows if asc else rows[::-1]
    for row in ordered_rows:
        try:
            st, lf = row.split('|')
            print("{}|{}".format(st, 'O' * len(lf)))
        except ValueError:
            # no pipe in row, print as is
            print(row)
    if legend_pos is not None and legend_pos != 'top':
        st, lf = pair.split('|')
        print('Scale: \n{} => {}.{}x{} = {} {}'.format(pair, st, lf, scale, key_calc(st, lf, scale), unit))


def stem_text(df, asc=True, break_on=None, column=None, compact=False, display=300,
              legend_pos='best', outliers=True, persistence=None,
              random_state=None, scale=None, trim=False, unit='', zoom=None):
    """

    :param df: list, numpy array, time series, pandas or dask dataframe
    :param asc:
    :param break_on:
    :param column: specify which column (string or number) of the dataframe to use,
                   else the first numerical is selected
    :param compact:
    :param display:
    :param legend_pos:
    :param outliers:
    :param persistence:
    :param random_state:
    :param scale:
    :param trim:
    :param unit:
    :param zoom:
    """
    x = df if column is None else df[column]
    scale, pair, rows = stem_data(x,  break_on=break_on, column=column, compact=compact,
                                  display=display, outliers=outliers, persistence=persistence,
                                  random_state=random_state, scale=scale, trim=trim, zoom=zoom)
    if legend_pos == 'top':
        st, lf = pair.split('|')
        print('Key: \n{} => {}.{}x{} = {} {}'.format(pair, st, lf, scale, key_calc(st, lf, scale), unit))

    ordered_rows = rows if asc else rows[::-1]
    for row in ordered_rows:
        print(row)
    if legend_pos is not None and legend_pos != 'top':
        st, lf = pair.split('|')
        print('Key: \n{} => {}.{}x{} = {} {}'.format(pair, st, lf, scale, key_calc(st, lf, scale), unit))
