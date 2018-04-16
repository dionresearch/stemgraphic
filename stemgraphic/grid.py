import stemgraphic.alpha as alpha
import stemgraphic.num as num
from stemgraphic.num import density_plot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from warnings import warn


def small_multiples(df, var, axes=None, bins=None, box=None, cols=None, col_labels=None, density=False,
                    density_fill=True,
                    display=750, fit=None, flip_axes=False,
                    hist=None, hues=None, hue_labels=None, jitter=None, legend='on', limit_var=False,
                    norm_hist=None,
                    plot_function=None, random_state=None, reverse=False, rows=None, row_labels=None, rug=None,
                    singular=True,
                    shared_stem=False, stem_display=True, stem_order=1, stem_skip=0, strip=None, x_min=0, x_max=None):
    if density:
        stem_display = None
        if df[var].dtypes == 'object':
            df[var + '_code'] = pd.Categorical(df[var])
            df[var + '_code'] = df[var + '_code'].cat.codes.astype('int64')
            var = var + '_code'

    legend_title = 'per {}'.format(cols) if cols else ''
    legend_title += ' per {}'.format(rows) if rows else ''
    legend_title += ' per {}'.format(hues) if hues else ''

    hue_categories = sorted(df[hues].dropna().unique()) if hues else ['all']
    row_categories = sorted(df[rows].dropna().unique()) if rows else ['all']
    col_categories = sorted(df[cols].dropna().unique()) if cols else ['all']

    hue_labels = hue_labels if hue_labels else hue_categories
    row_labels = row_labels if row_labels else row_categories
    col_labels = col_labels if col_labels else col_categories

    titles = [legend_title] + row_labels

    if legend == 'top':
        offset = 1
    else:
        offset = 0

    nb_rows = len(row_categories)
    nb_cols = len(col_categories)
    if nb_cols == 0:
        nb_cols = 1

    if shared_stem and stem_display:
        if nb_cols % 2 != 0:
            warn("Column variable has to have an even number of categories to use back to back stem-and-leaf plots.")
            return None
        adjustmentx = 2
        adjustmenty = nb_rows
        sharex = False
    else:
        adjustmentx = 1
        adjustmenty = nb_rows + 0.5
        sharex = False
    if axes is None:
        fig, (axes) = plt.subplots(nb_rows + offset,
                                   nb_cols, sharex=sharex,
                                   sharey=True, figsize=(nb_rows * 4 * adjustmentx, nb_cols * 4 * adjustmenty))
    plt.suptitle('Distribution of {}'.format(var), ha='center', fontsize=16)

    if nb_rows + offset > 1 and nb_cols > 1:
        multidim = 'xy'
        ax0 = axes[0][0]
        ax1 = axes[0][1]
    elif nb_rows + offset > 1:
        multidim = 'y'
        ax0 = axes[0]
        ax1 = None
    else:
        multidim = 'x'
        ax0 = axes[0]

    if legend == 'top':

        ax0.axis('off')
        ax0.axes.set_ylim(0, 0.01)
        ax0.set_title(legend_title, loc='left')

        for i, val in enumerate(hue_categories):
            ax0.scatter((i), (0.11), marker='s')  # outside viewing area, just to generate a legend with squares
        if legend == 'top':
            ax0.legend(hue_labels, ncol=3, loc='center', fontsize='medium', frameon=False)

    max_max = 0
    for k, colval in enumerate(col_categories):
        if k % 2 == 1 and shared_stem and stem_display:
            ax2 = ax
        if nb_cols > 1 and nb_rows - offset > 1:
            ax = [i[k] for i in axes]
        elif nb_cols > 1 and (nb_rows - offset) == 1:
            ax = list(axes[k:])
        elif nb_cols > 1:
            ax = list(axes[k])
        else:
            ax = axes
        max_peak = 0
        for j, rowval in enumerate(row_categories):

            max_peak = 0
            loc = 'center' if stem_display else 'left'
            if rows and cols:
                ax[j + offset].set_title(row_labels[j] + ' ' + col_labels[k], loc=loc, va='top')
            elif cols:
                ax[j + offset].set_title(col_labels[k], loc=loc, va='top')
            else:
                ax[j + offset].set_title(row_labels[j], loc=loc, va='top')
            if cols:
                col_filter = (df[cols] == colval)
            if rows:
                row_filter = (df[rows] == rowval)
            if rows and cols:
                full_filter = row_filter & col_filter
            elif rows:
                full_filter = row_filter
            elif cols:
                full_filter = col_filter
            to_plot = df[full_filter]

            if len(to_plot) > 0:
                if k == 0 and stem_display and shared_stem:
                    secondary_to_plot = to_plot
                elif stem_display:
                    if to_plot[var].dtype.name == 'object':
                        if shared_stem:
                            alpha.stem_graphic(to_plot[var].to_frame('word'),
                                               secondary_to_plot[var].to_frame('word'),
                                               ax=ax[j + offset], ax2=ax2[j + offset]);
                        else:
                            alpha.stem_graphic(to_plot[var].to_frame('word'), ax=ax[j + offset]);
                    else:
                        if shared_stem:
                            f, a = num.stem_graphic(to_plot, secondary_to_plot, ax=ax[j + offset],
                                                    ax2=ax2[j + offset], column=var, flip_axes=flip_axes);
                            # ax2[j+offset].set_xlim(ax2[j+offset].get_xlim()[::-1])
                        else:
                            f, a = num.stem_graphic(to_plot, ax=ax[j + offset], column=var, flip_axes=flip_axes);
                elif plot_function:
                    plot_function(to_plot, ax=ax[j + offset])
                else:

                    _, ax[j + offset], max_peak, _, _ = density_plot(to_plot, var=var, ax=ax[j + offset], bins=bins,
                                                                     box=box,
                                                                     density=density, density_fill=density_fill,
                                                                     display=display, fit=fit, fig_only=False,
                                                                     hist=hist, hues=hues, jitter=jitter,
                                                                     legend=False if legend == 'top' else legend,
                                                                     limit_var=limit_var,
                                                                     norm_hist=norm_hist, random_state=random_state,
                                                                     rug=rug, strip=strip,
                                                                     x_min=x_min, x_max=x_max)

                ax[j + offset].axes.get_yaxis().set_visible(False)
                ax[j + offset].axes.set_xlabel('')

            if max_peak > max_max:
                max_max = max_peak
            if limit_var:
                true_min = df[var][full_filter].dropna().min()
                true_max = df[var][full_filter].dropna().max()

                if stem_display:
                    if flip_axes:
                        ax[j + offset].set_xlim(true_min, true_max)
                        ax[j + offset].set_ylim(0, true_max * 2)
                    else:
                        ax[j + offset].set_xlim(0, true_max * 2)
                        ax[j + offset].set_ylim(true_min, true_max)

            if x_min and x_max:
                ax[j + offset].set_xlim(x_min, x_max)
            if density or hist or rug:
                ax[j + offset].set_ylim(0, max_max + 0.005)

    if legend != 'top' and legend:
        ax[0].legend(hue_labels, ncol=3, loc='upper right', fontsize='medium', frameon=False)

    plt.box(False)
    sns.despine(left=True, bottom=True, top=True, right=True);
    if not density or (shared_stem and stem_display):
        plt.tight_layout()
    return fig, axes, df[var][full_filter]
