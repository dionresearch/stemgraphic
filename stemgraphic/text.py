import math
import numpy as np
from operator import itemgetter
from warnings import warn

from .helpers import *


def heatmap(
    df,
    caps=None,
    charset=None,
    column=None,
    compact=True,
    display=900,
    flip_axes=False,
    leaf_order=1,
    outliers=None,
    persistence=None,
    random_state=None,
    scale=None,
    trim=False,
    trim_blank=True,
    unit="",
    zero_blank=True,
    zoom=None,
):
    """ heatmap

        The heatmap displays the same underlying data as the stem-and-leaf plot, but instead of stacking the leaves,
        they are left in their respective columns. Row '42' and Column '7' would have the count of numbers starting
        with '427' of the given scale. The difference with the heatmatrix is that by default it doesn't show zero
        values and it present a compact form by not showing whole empty rows either. Set compact = True to display
        those empty rows.

        The heatmap is useful to look at patterns. For distribution, stem_graphic is better suited.

        **Example**:

            .. sourcecode:: python

                heatmap(diamonds.carat, charset='bold');

        **Output**:

            .. sourcecode:: bash

                Stem-and-leaf heatmap (30.1 x 0.1 )
                       𝟎   𝟏   𝟐   𝟑   𝟒   𝟓  𝟔   𝟕   𝟖  𝟗
                stem
                  𝟐                𝟔   𝟒   𝟏  𝟑  𝟏𝟎   𝟑
                  𝟑   𝟒𝟖  𝟒𝟔  𝟑𝟑  𝟐𝟔  𝟐𝟏  𝟏𝟐  𝟖   𝟔  𝟏𝟏  𝟔
                  𝟒   𝟏𝟒  𝟐𝟎  𝟏𝟐  𝟏𝟎   𝟐   𝟒  𝟓   𝟑
                  𝟓   𝟏𝟗  𝟏𝟖  𝟏𝟐  𝟏𝟒   𝟗   𝟔  𝟔   𝟓   𝟑  𝟐
                  𝟔    𝟕       𝟑       𝟏   𝟒      𝟏
                  𝟕   𝟑𝟑  𝟏𝟒  𝟏𝟓  𝟏𝟏   𝟔   𝟔  𝟑   𝟑   𝟔  𝟒
                  𝟖    𝟒   𝟓           𝟏          𝟐
                  𝟗   𝟑𝟐   𝟕   𝟑   𝟏   𝟏   𝟏  𝟏   𝟑
                 𝟏𝟎   𝟐𝟓  𝟑𝟒  𝟏𝟐  𝟏𝟑   𝟕   𝟕  𝟓   𝟑   𝟏  𝟓
                 𝟏𝟏    𝟖   𝟏   𝟔   𝟓   𝟓   𝟒  𝟑       𝟑  𝟐
                 𝟏𝟐   𝟏𝟒   𝟓   𝟓   𝟓   𝟔   𝟏  𝟒   𝟑   𝟏  𝟏
                 𝟏𝟑    𝟏   𝟑   𝟐   𝟐   𝟏   𝟐  𝟏       𝟏
                 𝟏𝟒        𝟏                      𝟏
                 𝟏𝟓    𝟗  𝟏𝟐   𝟕   𝟔   𝟓   𝟓  𝟏       𝟏  𝟐
                 𝟏𝟔    𝟏               𝟏      𝟏
                 𝟏𝟕    𝟑   𝟒   𝟐   𝟑   𝟐   𝟏             𝟏
                 𝟏𝟖                𝟏                  𝟏
                 𝟏𝟗                    𝟏          𝟏
                 𝟐𝟎    𝟔   𝟖   𝟏   𝟏   𝟑      𝟏
                 𝟐𝟏    𝟐                   𝟏      𝟏   𝟏  𝟏
                 𝟐𝟐    𝟐                      𝟏          𝟏
                 𝟐𝟑    𝟏                                 𝟐
                 𝟑𝟎        𝟏


    :param df: list, numpy array, time series, pandas or dask dataframe
    :param charset: valid unicode digit character set, as returned by helpers.available_charsets()
    :param column: specify which column (string or number) of the dataframe to use,
                   else the first numerical is selected
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param flip_axes: wide format
    :param leaf_order: how many leaf digits per data point to display, defaults to 1
    :param outliers: for compatibility with other text plots
    :param persistence: filename. save sampled data to disk, either as pickle (.pkl) or csv (any other extension)
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param scale: force a specific scale for building the plot. Defaults to None (automatic).
    :param trim: ranges from 0 to 0.5 (50%) to remove from each end of the data set, defaults to None
    :param trim_blank: remove the blank between the delimiter and the first leaf, defaults to True
    :param unit:  specify a string for the unit ('$', 'Kg'...). Used for outliers and for legend, defaults to ''
    :param zero_blank: replace zero digit with space
    :param zoom: zoom level, on top of calculated scale (+1, -1 etc)
    :return: count matrix, scale
    """
    return heatmatrix(
        df,
        caps=caps,
        charset=charset,
        column=column,
        compact=compact,
        display=display,
        flip_axes=flip_axes,
        leaf_order=leaf_order,
        outliers=outliers,
        persistence=persistence,
        random_state=random_state,
        scale=scale,
        trim=trim,
        trim_blank=trim_blank,
        unit=unit,
        zero_blank=zero_blank,
        zoom=zoom,
    )


def heatmatrix(
    df,
    caps=None,
    charset=None,
    column=None,
    compact=False,
    display=900,
    flip_axes=False,
    leaf_order=1,
    outliers=None,
    persistence=None,
    random_state=None,
    scale=None,
    trim=False,
    trim_blank=True,
    unit="",
    zero_blank=False,
    zoom=None,
):
    """ heatmatrix

        The heatmatrix displays the same underlying data as the stem-and-leaf plot, but instead of stacking the leaves,
        they are left in their respective columns. Row '42' and Column '7' would have the count of numbers starting
        with '427' of the given scale.

        The heatmatrix is useful to look at patterns. For distribution, stem_graphic is better suited.

        **Example**:

            .. sourcecode:: python

                heatmatrix(diamonds.carat, charset='bold');

        **Output**:

            .. sourcecode:: bash

                Stem-and-leaf heatmap (24.0 x 0.1 )
                       𝟎   𝟏   𝟐   𝟑   𝟒   𝟓  𝟔   𝟕   𝟖  𝟗
                stem
                  𝟐    𝟏   𝟎   𝟏   𝟓   𝟒   𝟏  𝟓   𝟎   𝟒  𝟐
                  𝟑   𝟒𝟓  𝟒𝟎  𝟐𝟔  𝟏𝟕  𝟏𝟒   𝟕  𝟖   𝟒  𝟏𝟐  𝟕
                  𝟒   𝟑𝟎  𝟑𝟏  𝟏𝟖   𝟖   𝟑   𝟐  𝟑   𝟎   𝟏  𝟏
                  𝟓   𝟐𝟑  𝟐𝟎   𝟖   𝟓   𝟖  𝟏𝟑  𝟖   𝟔   𝟓  𝟕
                  𝟔    𝟔   𝟒   𝟐   𝟎   𝟑   𝟎  𝟎   𝟎   𝟎  𝟎
                  𝟕   𝟏𝟖  𝟐𝟐  𝟏𝟐   𝟕   𝟕   𝟖  𝟒   𝟑   𝟒  𝟐
                  𝟖    𝟓   𝟒   𝟑   𝟓   𝟎   𝟏  𝟓   𝟏   𝟎  𝟎
                  𝟗   𝟏𝟗  𝟏𝟒   𝟐   𝟐   𝟎   𝟎  𝟎   𝟎   𝟎  𝟎
                 𝟏𝟎   𝟐𝟖  𝟑𝟔  𝟏𝟎   𝟖   𝟗  𝟏𝟎  𝟏  𝟏𝟒   𝟒  𝟓
                 𝟏𝟏    𝟕   𝟒   𝟒   𝟑   𝟒   𝟎  𝟔   𝟏   𝟏  𝟏
                 𝟏𝟐   𝟏𝟏   𝟗   𝟗   𝟒   𝟕   𝟐  𝟏   𝟐   𝟐  𝟏
                 𝟏𝟑    𝟔   𝟏   𝟒   𝟐   𝟐   𝟎  𝟎   𝟎   𝟏  𝟎
                 𝟏𝟒    𝟎   𝟎   𝟎   𝟎   𝟏   𝟎  𝟎   𝟎   𝟎  𝟎
                 𝟏𝟓   𝟏𝟎  𝟏𝟔   𝟒   𝟑   𝟑   𝟓  𝟐   𝟑   𝟐  𝟏
                 𝟏𝟔    𝟐   𝟏   𝟏   𝟏   𝟎   𝟏  𝟎   𝟏   𝟎  𝟎
                 𝟏𝟕    𝟔   𝟓   𝟎   𝟏   𝟏   𝟏  𝟎   𝟎   𝟏  𝟏
                 𝟏𝟖    𝟏   𝟎   𝟏   𝟎   𝟎   𝟎  𝟎   𝟎   𝟎  𝟎
                 𝟏𝟗    𝟏   𝟏   𝟎   𝟎   𝟎   𝟎  𝟎   𝟎   𝟎  𝟎
                 𝟐𝟎    𝟑   𝟗   𝟒   𝟑   𝟐   𝟐  𝟏   𝟏   𝟏  𝟎
                 𝟐𝟏    𝟎   𝟏   𝟎   𝟎   𝟏   𝟎  𝟎   𝟏   𝟎  𝟎
                 𝟐𝟐    𝟎   𝟐   𝟏   𝟎   𝟎   𝟏  𝟎   𝟎   𝟎  𝟎
                 𝟐𝟑    𝟎   𝟎   𝟎   𝟎   𝟎   𝟎  𝟎   𝟎   𝟎  𝟎
                 𝟐𝟒    𝟏   𝟎   𝟏   𝟎   𝟐   𝟎  𝟎   𝟎   𝟎  𝟎


    :param df: list, numpy array, time series, pandas or dask dataframe
    :param column: specify which column (string or number) of the dataframe to use,
                   else the first numerical is selected
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param flip_axes: wide format
    :param leaf_order: how many leaf digits per data point to display, defaults to 1
    :param outliers: for compatibility with other text plots
    :param persistence: filename. save sampled data to disk, either as pickle (.pkl) or csv (any other extension)
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param scale: force a specific scale for building the plot. Defaults to None (automatic).
    :param trim: ranges from 0 to 0.5 (50%) to remove from each end of the data set, defaults to None
    :param trim_blank: remove the blank between the delimiter and the first leaf, defaults to True
    :param unit:  specify a string for the unit ('$', 'Kg'...). Used for outliers and for legend, defaults to ''
    :param zero_blank: replace zero digit with space
    :param zoom: zoom level, on top of calculated scale (+1, -1 etc)
    :return: count matrix, scale
    """
    if charset is None:
        charset = "default"
    try:
        cols = len(df.columns)
    except AttributeError:
        # wasn't a multi column data frame, might be a list
        cols = 1
    if cols > 1:
        if column is None:
            # We have to figure out the first numerical column on our own
            start_at = 1 if df.columns[0] == "id" else 0
            for i in range(start_at, len(df.columns)):
                if df.dtypes[i] in ("int64", "float64"):
                    column = i
                    break
        if dd:
            df = df[df.columns.values[column]]
        else:
            df = df.ix[:, column]

    min_val, max_val, total_rows = min_max_count(df)

    scale_factor, pair, rows = stem_data(
        df,
        break_on=10,
        column=column,
        compact=compact,
        display=display,
        leaf_order=leaf_order,
        omin=min_val,
        omax=max_val,
        outliers=False,
        persistence=persistence,
        random_state=random_state,
        scale=scale,
        total_rows=total_rows,
        trim=trim,
        zoom=zoom,
    )
    max_leaves = len(max(rows, key=len))

    if max_leaves > display / 3:
        # more than 1/3 on a single stem, let's try one more time
        if random_state:
            random_state += 1
        scale_factor2, pair2, rows2 = stem_data(
            df,
            break_on=None,
            column=column,
            compact=compact,
            display=display,
            leaf_order=1,
            omin=min_val,
            omax=max_val,
            outliers=False,
            persistence=persistence,
            random_state=random_state,
            scale=scale,
            total_rows=total_rows,
            trim=trim,
            zoom=zoom,
        )
        max_leaves2 = len(max(rows2, key=len))
        if max_leaves2 < max_leaves:
            max_leaves = max_leaves2
            scale_factor = scale_factor2
            pair = pair2
            rows = rows2

    split_rows = [i.split("|") for i in rows]

    # redo the leaves in a matrix form
    # this should be refactored as an option for stem_data, like rows_only for ngram_data
    matrix = []
    for stem, leaves in split_rows:
        row_count = [stem]
        for num in "0123456789":
            leaf_count = leaves.count(num)
            row_count.append(leaf_count)
        matrix.append(row_count)

    num_matrix = pd.DataFrame(
        matrix, columns=["stem", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    )
    num_matrix.set_index("stem", inplace=True)
    if flip_axes:
        num_matrix = num_matrix.T
    if trim_blank:
        num_matrix.applymap(lambda x: x.strip() if type(x) is str else x)

    title = "Stem-and-leaf heatmap ({} x {} {})".format(
        pair.replace("|", "."), scale_factor, unit
    )
    print(title)
    if charset:
        if charset not in available_charsets():
            warn("charset must be one of {}".format(available_charsets()))
            return
        num_matrix_text = str(num_matrix).split("\n")
        translated_num_matrix = [
            translate_representation(
                row, charset=charset, index=i, zero_blank=zero_blank
            )
            for i, row in enumerate(num_matrix_text)
        ]
        print("\n".join(translated_num_matrix))
    else:
        print(num_matrix)
    return num_matrix, scale_factor


def quantize(
    df,
    column=None,
    display=750,
    leaf_order=1,
    random_state=None,
    scale=None,
    trim=None,
    zoom=None,
):
    """ quantize

    Converts a series into stem-and-leaf and back into decimal. This has the potential effect of decimating (or
    truncating) values in a lossy way.

    :param df: list, numpy array, time series, pandas or dask dataframe
    :param column: specify which column (string or number) of the dataframe to use,
                   else the first numerical is selected
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param leaf_order: how many leaf digits per data point to display, defaults to 1
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param scale: force a specific scale for building the plot. Defaults to None (automatic).
    :param trim: ranges from 0 to 0.5 (50%) to remove from each end of the data set, defaults to None
    :param zoom: zoom level, on top of calculated scale (+1, -1 etc)
    :return: decimated df
    """
    x = df if column is None else df[column]
    scale, pair, rows, sorted_data, stems = stem_data(
        x,
        column=column,
        display=display,
        full=True,
        leaf_order=leaf_order,
        random_state=random_state,
        scale=scale,
        trim=trim,
        zoom=zoom,
    )

    values = [(stem + leaf) * scale for stem, leaf in sorted_data]
    return values


def stem_data(
    x,
    break_on=None,
    column=None,
    compact=False,
    display=300,
    full=False,
    leaf_order=1,
    omin=None,
    omax=None,
    outliers=False,
    persistence=None,
    random_state=None,
    scale=None,
    total_rows=None,
    trim=False,
    zoom=None,
):
    """ stem_data

    Returns scale factor, key label and list of rows.

    :param x: list, numpy array, time series, pandas or dask dataframe
    :param break_on: force a break of the leaves at x in (5, 10), defaults to 10
    :param column: specify which column (string or number) of the dataframe to use,
                   else the first numerical is selected
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param full: bool, if True returns all interim results including sorted data and stems
    :param leaf_order: how many leaf digits per data point to display, defaults to 1
    :param outliers: display outliers - these are from the full data set, not the sample. Defaults to Auto
    :param omin: float, if already calculated, helps speed up the process for large data sets
    :param omax: float, if already calculated, helps speed up the process for large data sets
    :param persistence: persist sampled dataframe
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param scale: force a specific scale for building the plot. Defaults to None (automatic)
    :param total_rows: int, if already calculated, helps speed up the process for large data sets
    :param trim: ranges from 0 to 0.5 (50%) to remove from each end of the data set, defaults to None
    :param zoom: zoom level, on top of calculated scale (+1, -1 etc)
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
            start_at = 1 if x.columns[0] == "id" else 0
            for i in range(start_at, len(x.columns)):
                if x.dtypes[i] in ("int64", "float64"):
                    column = i
                    break
        # if dd:
        #    x = x[x.columns.values[column]]
        # else:
        x = x.ix[:, column]

    # Sampling or not we need the absolute min/max
    if omin is None or omax is None or total_rows is None:
        omin, omax, total_rows = min_max_count(
            x, column
        )  # very expensive if on disk, don't do it twice

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
            if persistence[-4:] == ".pkl":
                pd.Dataframe(x).to_pickle(persistence)
            else:
                pd.Dataframe(x).to_csv(persistence)  # TODO: add feather, hdf5 etc
        n = display

    if n <= 300:
        # Dixon
        lines = math.floor(10 * math.log(n, 10))
    else:
        # Velleman
        lines = math.floor(2 * math.sqrt(n))

    try:
        x = x[~np.isnan(x)]
        xmin = x.min()
        xmax = x.max()
    except (AttributeError, TypeError):
        xmin = min(x)
        xmax = max(x)

    try:
        spread = xmax - xmin
    except TypeError:
        warn("Column data appears to be non numerical. Specify a numeric column.")
        return None

    # we will trim on the sample, or the whole data set
    lowest, highest = percentile(x, trim) if trim else xmin, xmax
    # scale_factor = as small as possible but lines * S must be >= spread
    if lines == 0:
        lines = 1
    r_value = spread / lines
    if scale:  # we were passed a scale, use it
        scale_factor = scale
    else:  # The bulk of the logic to figure out the best scaling and visualization
        try:
            scale_factor = pow(10, math.ceil(math.log(r_value, 10)))
        except ValueError:
            scale_factor = 1
        check = math.floor(xmax / scale_factor - xmin / scale_factor + 1)
        if check > lines:
            scale_factor *= 10
        elif (check < 7 and n >= 45) or check < 3:
            scale_factor /= (
                10
            )  # 30 lines on avg, up to 60 some lines max by bumping the scale
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
    rounded_data = [
        int(np.round(item / truncate_factor)) * truncate_factor
        for item in x
        if lowest <= item <= highest
    ]
    data = []
    for val in rounded_data:
        frac_part, int_part = math.modf(val / scale_factor)
        round_frac = round(frac_part, 2)
        if round_frac == 1:
            round_frac = 0.0
            int_part += 1.0
        data.append((round_frac, int_part))
    sorted_data = sorted(data, key=itemgetter(1, 0))
    stems = list(set([s for l, s in sorted_data]))
    current_stem = None
    current_leaf = None
    previous_mod = 0
    row = ""
    sign_transition = False
    if xmin < 0 < xmax:
        sign_transition = True
    if outliers:
        row = "{}\n    ¡".format(omin)

    neg_zero_leaf = len([l for l, s in sorted_data if -1 < l < 0])
    pos_zero_leaf = len([l for l, s in sorted_data if 0 < l < 1])
    neg_zero = None
    for leaf, stem in sorted_data:
        # leaf = round(f_leaf, 1 + leaf_order)
        if stem == current_stem:
            ileaf = round(leaf * 10)
            if sign_transition and stem == 0 and abs(leaf) == leaf:
                sign_transition = False
                rows.append(row)
                row = "{:>3} | ".format(int(stem))
            elif (
                current_stem is not None
                and ileaf >= break_on == 5
                and previous_mod > (ileaf % break_on)
            ):
                rows.append(row)
                row = "    | "
            elif leaf_order > 1:
                row += " "
            previous_mod = ileaf % break_on
            row += str(round(abs(leaf), 1 + leaf_order))[2 : leaf_order + 2]
        else:
            if row != "":
                rows.append(row)
            if current_stem is not None and not compact:
                if break_on == 5 and row[0:4] != "    ":
                    row = "    | "
                    rows.append(row)
                if current_stem == -0.0 and pos_zero_leaf == 0:
                    pos_zero = "{:>3} |".format("0")
                    rows.append(pos_zero)
                for missing in range(int(current_stem) + 1, int(stem)):
                    if int(current_stem) < 0 and missing == 0:
                        neg_zero = "{:>3} |".format("-0")
                        rows.append(neg_zero)
                    empty_row = "{:>3} |".format(missing)
                    rows.append(empty_row)
                    if break_on == 5:
                        rows.append("    | ")
                if (
                    neg_zero_leaf == 0
                    and neg_zero is None
                    and int(current_stem) < 0
                    and stem == 0
                ):
                    # special case where 0 is a stem, we have transition, but no -0 value
                    neg_zero = "{:>3} |".format("-0")
                    rows.append(neg_zero)

            current_leaf = str(round(abs(leaf), 1 + leaf_order))[
                2 : leaf_order + 2
            ].zfill(leaf_order)
            if current_stem and int(current_leaf) >= break_on:
                row = "{:>3} | ".format(int(stem))
                rows.append(row)
                stem_ind = "   "
            else:
                stem_ind = int(stem)
            row = "{:>3} | {}".format(
                "-0" if stem == 0 and abs(leaf) != leaf else stem_ind, current_leaf
            )
            current_stem = stem

    # Potentially catching a last row
    rows.append(row)
    if outliers:
        rows.append("    !\n{}".format(omax))
    key_label = "{}|{}".format(int(current_stem), current_leaf)
    if full:
        return scale_factor, key_label, rows, sorted_data, stems
    else:
        return scale_factor, key_label, rows


def stem_dot(
    df,
    asc=True,
    break_on=None,
    column=None,
    compact=False,
    display=300,
    flip_axes=False,
    leaf_order=1,
    legend_pos="best",
    marker=None,
    outliers=True,
    persistence=None,
    random_state=None,
    scale=None,
    symmetric=False,
    trim=False,
    unit="",
    zoom=None,
):
    """ stem_dot

    stem_dot builds a stem-and-leaf plot with dots instead of bars.

    **Example**:

        .. sourcecode:: python

            stem_dot(diamonds.price)

    **Output**:

        .. sourcecode:: bash

            326
                ¡
              0 |●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
              1 |●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
              2 |●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
              3 |●●●●●●●●●●●●●●●●●●●●●●●●●●●●
              4 |●●●●●●●●●●●●●●●●●●●●●●●●●●●
              5 |●●●●●●●●●●●●●●●
              6 |●●●●●●●●●
              7 |●●●
              8 |●●●●●
              9 |●●●●●●●
             10 |●●
             11 |●●●●
             12 |●●●●●
             13 |●●●●●
             14 |●●
             15 |●●●
             16 |●●
             17 |●●●●
                !
            18823
            Scale:
            17|1 => 17.1x1000 = 17100.0

    :param df: list, numpy array, time series, pandas or dask dataframe
    :param asc: stem sorted in ascending order, defaults to True
    :param break_on: force a break of the leaves at x in (5, 10), defaults to 10
    :param column: specify which column (string or number) of the dataframe to use,
                   else the first numerical is selected
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param flip_axes: bool, default is False
    :param legend_pos: One of 'top', 'bottom', 'best' or None, defaults to 'best'.
    :param marker: char, symbol to use as marker. '●' is default. Suggested alternatives: '*', '+', 'x', '.', 'o'
    :param outliers: display outliers - these are from the full data set, not the sample. Defaults to Auto
    :param persistence: filename. save sampled data to disk, either as pickle (.pkl) or csv (any other extension)
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param scale: force a specific scale for building the plot. Defaults to None (automatic).
    :param symmetric: if True, dot plot will be distributed on both side of a center line
    :param trim: ranges from 0 to 0.5 (50%) to remove from each end of the data set, defaults to None
    :param unit: specify a string for the unit ('$', 'Kg'...). Used for outliers and for legend, defaults to ''
    :param zoom: zoom level, on top of calculated scale (+1, -1 etc)
    """
    if marker is None:
        marker = "●"  # commonly used, but * could also be used
    x = df if column is None else df[column]
    scale, pair, rows = stem_data(
        x,
        break_on=break_on,
        column=column,
        compact=compact,
        display=display,
        leaf_order=leaf_order,
        outliers=outliers,
        random_state=random_state,
        scale=scale,
        trim=trim,
        zoom=zoom,
    )
    if legend_pos == "top":
        st, lf = pair.split("|")
        print(
            "Key: \n{} => {}.{}x{} = {} {}".format(
                pair, st, lf, scale, key_calc(st, lf, scale), unit
            )
        )

    ordered_rows = rows if asc else rows[::-1]
    max_len = len(max(ordered_rows, key=len))
    dot_rows = []
    for row in ordered_rows:
        try:
            st, lf = row.split("|")
            if symmetric:
                # pad spaces between the | and dots
                dot_rows.append(
                    "{}|{}{}".format(
                        st,
                        " " * int((max_len - len(lf)) / 2 - 1),
                        marker * (len(lf) - 1),
                    )
                )
            else:
                dot_rows.append("{}|{}".format(st, marker * (len(lf) - 1)))
        except ValueError:
            # no pipe in row, print as is
            dot_rows.append(row)

    if flip_axes:
        max_len = len(max(dot_rows, key=len))
        padded_rows = [
            row + (" " * (max_len - len(row))) for row in dot_rows if "|" in row
        ]
        flipped_rows = ["".join(chars) for chars in zip(*padded_rows)]
        ordered_rows = flipped_rows[::-1] if asc else flipped_rows
        print()
        for row in ordered_rows:
            if "|" in row:
                print(row.replace("|", "-") + "⇪")
            else:
                print(row)
    else:
        for row in dot_rows:
            print(row)
    if legend_pos is not None and legend_pos != "top":
        st, lf = pair.split("|")
        print(
            "Scale: \n{} => {}.{}x{} = {} {}".format(
                pair, st, lf, scale, key_calc(st, lf, scale), unit
            )
        )


def stem_hist(
    df,
    asc=True,
    break_on=None,
    column=None,
    compact=False,
    display=300,
    flip_axes=False,
    leaf_order=1,
    legend_pos="best",
    marker=None,
    outliers=True,
    persistence=None,
    random_state=None,
    scale=None,
    shade=None,
    symmetric=False,
    trim=False,
    unit="",
    zoom=None,
):
    """ stem_hist

    stem_hist builds a histogram matching the stem-and-leaf plot, with the numbers hidden, as shown on the
    cover of the companion brochure.

    **Example**:

        .. sourcecode:: python

            stem_hist(diamonds.price, shade='medium')

    **Output**:

        .. sourcecode:: bash

              0 |▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
              1 |▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
              2 |▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
              3 |▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
              4 |▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
              5 |▒▒▒▒▒▒▒▒▒▒▒▒▒▒
              6 |▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
              7 |▒▒▒▒▒▒▒▒▒▒▒▒
              8 |▒▒▒▒
              9 |▒▒▒▒
             10 |▒▒▒▒▒▒▒
             11 |▒▒
             12 |▒▒▒▒▒▒▒
             13 |▒▒▒▒
             14 |
             15 |▒
             16 |▒
             17 |
             18 |▒
            Scale:
            18|4 => 18.4x1000 = 18400.0

    :param df: list, numpy array, time series, pandas or dask dataframe
    :param asc: stem sorted in ascending order, defaults to True
    :param break_on: force a break of the leaves at x in (5, 10), defaults to 10
    :param column: specify which column (string or number) of the dataframe to use,
                   else the first numerical is selected
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param flip_axes: bool, default is False
    :param legend_pos: One of 'top', 'bottom', 'best' or None, defaults to 'best'.
    :param marker: char, symbol to use as marker. 'O' is default. Suggested alternatives: '*', '+', 'x', '.', 'o'
    :param outliers: display outliers - these are from the full data set, not the sample. Defaults to Auto
    :param persistence: filename. save sampled data to disk, either as pickle (.pkl) or csv (any other extension)
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param scale: force a specific scale for building the plot. Defaults to None (automatic).
    :param shade: shade of marker: 'none','light','medium','dark','full'
    :param symmetric: if True, dot plot will be distributed on both side of a center line
    :param trim: ranges from 0 to 0.5 (50%) to remove from each end of the data set, defaults to None
    :param unit: specify a string for the unit ('$', 'Kg'...). Used for outliers and for legend, defaults to ''
    :param zoom: zoom level, on top of calculated scale (+1, -1 etc)
    """
    if marker is None:
        if shade == "light":
            marker = "░"
        elif shade == "medium":
            marker = "▒"
        elif shade == "dark":
            marker = "▓"
        else:
            marker = "█"

    return stem_dot(
        df,
        asc=asc,
        break_on=break_on,
        column=column,
        display=display,
        flip_axes=flip_axes,
        legend_pos=legend_pos,
        marker=marker,
        outliers=False,
        persistence=persistence,
        random_state=random_state,
        scale=scale,
        symmetric=symmetric,
        trim=trim,
        unit=unit,
        zoom=zoom,
    )


def stem_tally(
    df,
    asc=True,
    break_on=None,
    column=None,
    compact=False,
    display=300,
    flip_axes=False,
    legend_pos="best",
    outliers=True,
    persistence=None,
    random_state=None,
    scale=None,
    symmetric=False,
    trim=False,
    unit="",
    zoom=None,
):
    """ stem_tally

    Stem-and-leaf plot using tally marks for leaf count, up to 5 per block.

    **Example**:

        .. sourcecode:: python

            stem_tally(diamonds.price)
            326
                ¡
              0 |卌卌卌卌卌卌卌卌卌卌卌卌卌卌卌𝍩
              1 |卌卌卌卌卌卌卌卌卌卌卌卌
              2 |卌卌卌卌卌卌𝍫
              3 |卌卌卌卌𝍩
              4 |卌卌卌卌卌𝍫
              5 |卌卌卌卌卌𝍩
              6 |卌卌卌𝍩
              7 |卌卌卌𝍩
              8 |卌卌𝍩
              9 |𝍫
             10 |𝍪
             11 |𝍬
             12 |卌𝍩
             13 |𝍬
             14 |𝍬
             15 |𝍫
             16 |𝍪
             17 |
             18 |𝍫
                !
            18823
            Key:
            18|3 => 18.3x1000 = 18300.0


    :param df: list, numpy array, time series, pandas or dask dataframe
    :param asc: stem sorted in ascending order, defaults to True
    :param break_on: force a break of the leaves at x in (5, 10), defaults to 10
    :param column: specify which column (string or number) of the dataframe to use,
                   else the first numerical is selected
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param flip_axes: bool, default is False
    :param legend_pos: One of 'top', 'bottom', 'best' or None, defaults to 'best'.
    :param outliers: display outliers - these are from the full data set, not the sample. Defaults to Auto
    :param persistence: filename. save sampled data to disk, either as pickle (.pkl) or csv (any other extension)
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param scale: force a specific scale for building the plot. Defaults to None (automatic).
    :param symmetric: if True, dot plot will be distributed on both side of a center line
    :param trim: ranges from 0 to 0.5 (50%) to remove from each end of the data set, defaults to None
    :param unit: specify a string for the unit ('$', 'Kg'...). Used for outliers and for legend, defaults to ''
    :param zoom: zoom level, on top of calculated scale (+1, -1 etc)
    """
    tally_map = {"0": "", "1": "𝍩", "2": "𝍪", "3": "𝍫", "4": "𝍬", "5": "卌"}

    x = df if column is None else df[column]
    scale, pair, rows = stem_data(
        x,
        break_on=break_on,
        column=column,
        compact=compact,
        display=display,
        outliers=outliers,
        persistence=persistence,
        random_state=random_state,
        scale=scale,
        trim=trim,
        zoom=zoom,
    )
    if legend_pos == "top":
        st, lf = pair.split("|")
        print(
            "Key: \n{} => {}.{}x{} = {} {}".format(
                pair, st, lf, scale, key_calc(st, lf, scale), unit
            )
        )

    ordered = rows if asc else rows[::-1]

    ordered_rows = []
    for row in ordered:
        try:
            row_stem, row_leaf = row.split("|")
            leaf_count = len(row_leaf)
            slash_tally = int((leaf_count - (leaf_count % 5)) / 5) * tally_map["5"]
            partial_tally = tally_map[str(leaf_count % 5)]
            combined = row_stem + "|" + slash_tally + partial_tally
            ordered_rows.append(combined)
        except ValueError:
            ordered_rows.append(row)

    max_len = len(max(ordered_rows, key=len))
    if symmetric:
        padded_rows = []
        for row in ordered_rows:
            try:
                st, lf = row.split("|")
                # pad spaces between the | and dots
                padded_rows.append(
                    "{}|{}{}{}".format(
                        st,
                        " " * int((max_len - len(lf)) / 2 - 1),
                        lf,
                        " " * int((max_len - len(lf)) / 2 - 1),
                    )
                )
            except ValueError:
                pass
    else:
        padded_rows = [
            row + (" " * (max_len - len(row))) for row in ordered_rows if "|" in row
        ]
    if flip_axes:
        flipped_rows = ["".join(chars) for chars in zip(*padded_rows)]
        ordered_rows = flipped_rows[::-1] if asc else flipped_rows
        print()
        for row in ordered_rows:
            if "|" in row:
                print(row.replace("|", "-") + "⇪")
            else:
                print(row)
    else:
        if symmetric:
            ordered_rows = padded_rows
        for row in ordered_rows:
            print(row)
    if legend_pos is not None and legend_pos != "top":
        st, lf = pair.split("|")
        print(
            "Key: \n{} => {}.{}x{} = {} {}".format(
                pair, st, lf, scale, key_calc(st, lf, scale), unit
            )
        )


def stem_text(
    df,
    asc=True,
    break_on=None,
    charset=None,
    column=None,
    compact=False,
    display=300,
    flip_axes=False,
    legend_pos="best",
    outliers=True,
    persistence=None,
    random_state=None,
    scale=None,
    symmetric=False,
    trim=False,
    unit="",
    zoom=None,
):
    """ stem_text.

    Classic text based stem-and-leaf plot.

    :param df: list, numpy array, time series, pandas or dask dataframe
    :param asc: stem sorted in ascending order, defaults to True
    :param break_on: force a break of the leaves at x in (5, 10), defaults to 10
    :param charset: (default to ascii), 'roman', 'rod', 'arabic', 'circled', 'circled_inverted'
    :param column: specify which column (string or number) of the dataframe to use,
                   else the first numerical is selected
    :param compact: do not display empty stem rows (with no leaves), defaults to False
    :param display: maximum number of data points to display, forces sampling if smaller than len(df)
    :param flip_axes: bool, default is False
    :param legend_pos: One of 'top', 'bottom', 'best' or None, defaults to 'best'.
    :param outliers: display outliers - these are from the full data set, not the sample. Defaults to Auto
    :param persistence: filename. save sampled data to disk, either as pickle (.pkl) or csv (any other extension)
    :param random_state: initial random seed for the sampling process, for reproducible research
    :param scale: force a specific scale for building the plot. Defaults to None (automatic).
    :param symmetric: if True, dot plot will be distributed on both side of a center line
    :param trim: ranges from 0 to 0.5 (50%) to remove from each end of the data set, defaults to None
    :param unit: specify a string for the unit ('$', 'Kg'...). Used for outliers and for legend, defaults to ''
    :param zoom: zoom level, on top of calculated scale (+1, -1 etc)
    """
    x = df if column is None else df[column]
    scale, pair, rows = stem_data(
        x,
        break_on=break_on,
        column=column,
        compact=compact,
        display=display,
        outliers=outliers,
        persistence=persistence,
        random_state=random_state,
        scale=scale,
        trim=trim,
        zoom=zoom,
    )
    if legend_pos == "top":
        st, lf = pair.split("|")
        print(
            "Key: \n{} => {}.{}x{} = {} {}".format(
                pair, st, lf, scale, key_calc(st, lf, scale), unit
            )
        )

    ordered_rows = rows if asc else rows[::-1]
    max_len = len(max(ordered_rows, key=len))
    if charset:
        if charset not in available_charsets():
            warn("charset must be one of {}".format(available_charsets()))
            return
        ordered_rows = [
            translate_representation(row, charset=charset) for row in ordered_rows
        ]
    if symmetric:
        padded_rows = []
        for row in ordered_rows:
            try:
                st, lf = row.split("|")
                # pad spaces between the | and dots
                padded_rows.append(
                    "{}|{}{}{}".format(
                        st,
                        " " * int((max_len - len(lf)) / 2 - 1),
                        lf,
                        " " * int((max_len - len(lf)) / 2 - 1),
                    )
                )
            except ValueError:
                pass
    else:
        padded_rows = [
            row + (" " * (max_len - len(row))) for row in ordered_rows if "|" in row
        ]
    if flip_axes:
        flipped_rows = ["".join(chars) for chars in zip(*padded_rows)]
        ordered_rows = flipped_rows[::-1] if asc else flipped_rows
        print()
        for row in ordered_rows:
            if "|" in row:
                print(row.replace("|", "-") + "⇪")
            else:
                print(row)
    else:
        if symmetric:
            ordered_rows = padded_rows
        for row in ordered_rows:
            print(row)
    if legend_pos is not None and legend_pos != "top":
        st, lf = pair.split("|")
        print(
            "Key: \n{} => {}.{}x{} = {} {}".format(
                pair, st, lf, scale, key_calc(st, lf, scale), unit
            )
        )
