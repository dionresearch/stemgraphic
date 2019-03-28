""" helpers.py

Helper functions for stemgraphic.
"""
from io import BytesIO
import numpy as np
import pandas as pd
import pickle
from warnings import warn

try:
    import dask.dataframe as dd
except ImportError:
    dd = False

try:
    import sixel
except ImportError:
    sixel = None


def jitter(data, scale):
    """Adds jitter to data, for display purpose

    :param data: numpy or pandas dataframe
    :param scale:
    :return:
    """
    return data + np.random.rand(len(data)) / (2 * scale)


def key_calc(stem, leaf, scale):
    """Calculates a value from a stem, a leaf and a scale.

    :param stem:
    :param leaf:
    :param scale:
    :return: calculated values
    """
    return (int(leaf) / 10 + int(stem)) * float(scale)


def legend(
    ax,
    x,
    y,
    asc,
    flip_axes,
    mirror,
    stem,
    leaf,
    scale,
    delimiter_color,
    aggregation=True,
    cur_font=None,
    display=10,
    pos="best",
    unit="",
):
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
        ha = "right"
        formula = "{2}{1} =        x{0} = "
        offset = len(str(scale)) + 3.1 + (len(stem) + len(leaf)) / 1.7
        secondary = -2.5 if asc and flip_axes else -1.6
        key_text = "Key: leaf|stem{}".format("|aggr" if aggregation else "")
    else:
        ha = "left"
        formula = "  =         x{0} = {1}{2}"
        offset = 3.1
        secondary = 0.1
        key_text = "Key: {}stem|leaf".format("aggr|" if aggregation else "")
    start_at = (len(stem) * 2 + 11 + len(str(scale)) + len(leaf)) / 1.7
    if pos == "short":
        ax.text(
            x - start_at,
            y + 2,
            " x {}".format(scale),
            va="center",
            ha=ha,
            fontproperties=cur_font,
        )
    else:
        if aggregation:
            ax.text(
                x - start_at - 1,
                y + 2,
                key_text,
                va="center",
                ha=ha,
                fontproperties=cur_font,
            )
            ax.text(
                x - start_at - 2,
                y + 1,
                display,
                fontsize=aggr_fontsize - 2,
                va="center",
                ha=ha,
            )
        cur_font.set_weight("bold")
        ax.text(
            x - start_at - 1, y + 1, stem, va="center", ha=ha, fontproperties=cur_font
        )
        ax.text(
            x - start_at + (1 + len(leaf) + offset) / 1.7,
            y + 1,
            stem,
            va="center",
            ha=ha,
            fontproperties=cur_font,
        )
        cur_font.set_weight("normal")
        ax.text(
            x - start_at + (len(stem) + len(leaf)) / 1.7,
            y + 1,
            formula.format(scale, key_calc(stem, leaf, scale), unit),
            va="center",
            ha=ha,
            fontproperties=cur_font,
        )
        cur_font.set_style("italic")
        ax.text(
            x - start_at + 0.3,
            y + 1,
            leaf,
            bbox={"facecolor": "C0", "alpha": 0.15, "pad": 2},
            va="center",
            ha=ha,
            fontproperties=cur_font,
        )
        ax.text(
            x - start_at + (len(stem) + offset + len(leaf) + 0.6) / 1.7 + secondary,
            y + 1,
            "." + leaf,
            va="center",
            ha=ha,
            fontproperties=cur_font,
        )

        if flip_axes:
            ax.vlines(x - start_at, y + 0.5, y + 1.5, color=delimiter_color, alpha=0.7)
            if aggregation:
                ax.vlines(
                    x - start_at - 1, y + 0.5, y + 1.5, color=delimiter_color, alpha=0.7
                )
        else:
            ax.vlines(
                x - start_at + 0.1, y + 0.5, y + 1.5, color=delimiter_color, alpha=0.7
            )
            if aggregation:
                ax.vlines(
                    x - start_at - 1.1,
                    y + 0.5,
                    y + 1.5,
                    color=delimiter_color,
                    alpha=0.7,
                )


def min_max_count(x, column=0):
    """ min_max_count

    Handles min, max and count. This works on numpy, lists, pandas and dask dataframes.

    :param x: list, numpy array, series, pandas or dask dataframe
    :param column: future use
    :return: min, max and count
    """
    if dd and type(x) in (dd.core.DataFrame, dd.core.Series):
        omin, omax, count = dd.compute(x.min(), x.max(), x.count())
    elif type(x) in (pd.DataFrame, pd.Series):
        try:
            omin = x.min().values[0]
            omax = x.max().values[0]
        except AttributeError:
            omin = x.min()
            omax = x.max()
        count = len(x)
    else:
        omin = min(x)
        omax = max(x)
        count = len(x)

    return omin, omax, int(count)


def na_count(x, column=0):
    """ min_max_count

        Handles min, max and count. This works on numpy, lists, pandas and dask dataframes.

        :param x: list, numpy array, series, pandas or dask dataframe
        :param column: future use
        :return: all numpy nan count
        """
    val_missing = x.isnull().sum()
    return val_missing


def npy_save(path, array):
    """ npy_save

    saves numpy array to npy file on disk.

    :param path: path where to save npy file
    :param array: numpy array
    :return: path
    """
    if path[-4:] != ".npy":
        path += ".npy"
    with open(path, "wb+") as f:
        np.save(f, array, allow_pickle=False)
    return path


def npy_load(path):
    """ npy_load

    load numpy array (npy) file from disk.

    :param path: path to pickle file
    :return: numpy array
    """
    if path[-4:] != ".npy":
        warn("Not a numpy NPY file.")
        return None
    return np.load(path)


def pkl_save(path, array):
    """ pkl_save

    saves matrix or dataframe to pkl file on disk.

    :param path: path where to save pickle file
    :param array: matrix (array) or dataframe
    :return: path
    """
    if path[-4:] != ".pkl":
        path += ".pkl"
    with open(path, "wb+") as f:
        pickle.dump(array, f)
    return path


def pkl_load(path):
    """ pkl_load

    load matrix or dataframe pickle (pkl) file from disk.

    :param path: path to pickle file
    :return: matrix or dataframe
    """
    if path[-4:] != ".pkl":
        warn("Not a PKL file.")
        return None
    with open(path, "rb") as f:
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
    high = int(round((1 - alpha) * len(data) + 0.5))
    return n[low - 1], n[high - 1]


def savefig(plt):
    """ savefig

    Allows displaying a matplotlib figure to the console terminal. This requires pysixel to be pip installed.
    It also requires a terminal with Sixel graphic support, like DEC with graphic support, Linux xterm (started
    with -ti 340), MLTerm (multilingual terminal, available on Windows, Linux etc).

    This is called by the command line stem tool when using -o stdout and can also be used in an ipython session.

    :param plt: matplotlib pyplot
    :return:
    """
    buf = BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    if sixel is None:
        warn("No sixel module available. Please install pysixel")
    writer = sixel.SixelWriter()
    writer.draw(buf)


def stack_columns(row):
    """ stack_columns

    stack multiple columns into a single stacked value

    :param row: a row of letters
    :return: stacked string
    """
    row = row.dropna()
    stack = ""
    for i, col in row.iteritems():
        stack += str(i) * int(col)
    return stack


#: Typographical apostrophe - ex: I’m, l’arbre
APOSTROPHE = "’"

#: Straight quote mark - ex: 'INCONCEIVABLE'
QUOTE = "'"

#: Double straight quote mark
DOUBLE_QUOTE = '"'

#: empty
EMPTY = b" "

#: for typesetting overlap
OVER = b"\xd6\xb1"

#: Characters to filter. Does a relatively good job on a majority of texts
#: '- ' and '–' is to skip quotes in many plays and dialogues in books, especially French.
CHAR_FILTER = [
    "\t",
    "\n",
    "\\",
    "/",
    "`",
    "*",
    "_",
    "{",
    "}",
    "[",
    "]",
    "(",
    ")",
    "<",
    ">",
    "#",
    "=",
    "+",
    "- ",
    "–",
    ".",
    ";",
    ":",
    "!",
    "?",
    "|",
    "$",
    QUOTE,
    DOUBLE_QUOTE,
    "…",
]


#: Similar purpose to CHAR_FILTER, ut keeps the period. The last word of each sentence will end with a '.'
#: Useful for manipulating the dataframe returned by the various visualizations and ngram_data,
#: to break down frequencies by sentence instead of the full text or list.
NO_PERIOD_FILTER = [
    "\t",
    "\n",
    "\\",
    "/",
    "`",
    "*",
    "_",
    "{",
    "}",
    "[",
    "]",
    "(",
    ")",
    "<",
    ">",
    "#",
    "=",
    "+",
    "- ",
    "–",
    ";",
    ":",
    "!",
    "?",
    "|",
    "$",
    QUOTE,
    DOUBLE_QUOTE,
]


#: Default definition of standard letters
#: remove_accent has to be called explicitly for any of these letters to match their
#: accented counterparts
LETTERS = "abcdefghijklmnopqrstuvwxyz"

#: List of non alpha characters. Temporary - I want to balance flexibility with convenience, but
#: still looking at options.
NON_ALPHA = [
    "-",
    "+",
    "/",
    "[",
    "]",
    "_",
    "£",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "0",
    "!",
    "@",
    "#",
    "$",
    "%",
    "^",
    "&",
    "*",
    "(",
    ")",
    ";",
    QUOTE,
    DOUBLE_QUOTE,
    APOSTROPHE,
    EMPTY,
    OVER,
    "?",
    "¡",
    "¿",  # spanish
    "«",
    "»",
    "“",
    "”",
    "-",
    "—",
]

#: Charset unicode digit mappings
mapping = {
    "arabic": {
        "0": "٠",
        "1": "١",
        "2": "٢",
        "3": "٣",
        "4": "٤",
        "5": "٥",
        "6": "٦",
        "7": "٧",
        "8": "٨",
        "9": "٩",
    },
    "arabic_r": {
        "0": "٠",
        "1": "١",
        "2": "٢",
        "3": "٣",
        "4": "٤",
        "5": "٥",
        "6": "٦",
        "7": "٧",
        "8": "٨",
        "9": "٩",
    },
    "bold": {
        "0": "𝟎",
        "1": "𝟏",
        "2": "𝟐",
        "3": "𝟑",
        "4": "𝟒",
        "5": "𝟓",
        "6": "𝟔",
        "7": "𝟕",
        "8": "𝟖",
        "9": "𝟗",
    },
    "circled": {
        "0": "⓪",
        "1": "①",
        "2": "②",
        "3": "③",
        "4": "④",
        "5": "⑤",
        "6": "⑥",
        "7": "⑦",
        "8": "⑧",
        "9": "⑨",
    },
    "default": {
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
    },
    "doublestruck": {
        "0": "𝟘",
        "1": "𝟙",
        "2": "𝟚",
        "3": "𝟛",
        "4": "𝟜",
        "5": "𝟝",
        "6": "𝟞",
        "7": "𝟟",
        "8": "𝟠",
        "9": "𝟡",
    },
    "fullwidth": {
        "0": "０",
        "1": "１",
        "2": "２",
        "3": "３",
        "4": "４",
        "5": "５",
        "6": "６",
        "7": "７",
        "8": "８",
        "9": "９",
    },
    "gurmukhi": {
        "0": "੦",
        "1": "੧",
        "2": "੨",
        "3": "੩",
        "4": "੪",
        "5": "੫",
        "6": "੬",
        "7": "੭",
        "8": "੮",
        "9": "੯",
    },
    "mono": {
        "0": "𝟶",
        "1": "𝟷",
        "2": "𝟸",
        "3": "𝟹",
        "4": "𝟺",
        "5": "𝟻",
        "6": "𝟼",
        "7": "𝟽",
        "8": "𝟾",
        "9": "𝟿",
    },
    "nko": {
        "0": "߀",
        "1": "߁",
        "2": "߂",
        "3": "߃",
        "4": "߄",
        "5": "߅",
        "6": "߆",
        "7": "߇",
        "8": "߈",
        "9": "߉",
    },
    "rod": {
        "0": "◯",  # U+25EF LARGE CIRCLE
        "1": "𝍩",
        "2": "𝍪",
        "3": "𝍫",
        "4": "𝍬",
        "5": "𝍭",
        "6": "𝍮",
        "7": "𝍯",
        "8": "𝍰",
        "9": "𝍱",
    },
    "roman": {
        "0": ".",  # No zero
        "1": "Ⅰ",
        "2": "Ⅱ",
        "3": "Ⅲ",
        "4": "Ⅳ",
        "5": "Ⅴ",
        "6": "Ⅵ",
        "7": "Ⅶ",
        "8": "Ⅷ",
        "9": "Ⅸ",
    },
    "sans": {
        "0": "𝟢",
        "1": "𝟣",
        "2": "𝟤",
        "3": "𝟥",
        "4": "𝟦",
        "5": "𝟧",
        "6": "𝟨",
        "7": "𝟩",
        "8": "𝟪",
        "9": "𝟫",
    },
    "sansbold": {
        "0": "𝟬",
        "1": "𝟭",
        "2": "𝟮",
        "3": "𝟯",
        "4": "𝟰",
        "5": "𝟱",
        "6": "𝟲",
        "7": "𝟳",
        "8": "𝟴",
        "9": "𝟵",
    },
    "square": {
        "0": "🞌",
        "1": "🞍",
        "2": "￭",
        "3": "⬛",
        "4": "🞓",
        "5": "🞒",
        "6": "🞑",
        "7": "🞐",
        "8": "🞏",
        "9": "🞎",
    },
    "subscript": {
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉",
    },
    "tamil": {
        "0": "௦",
        "1": "௧",
        "2": "௨",
        "3": "௩",
        "4": "௪",
        "5": "௫",
        "6": "௬",
        "7": "௭",
        "8": "௮",
        "9": "௯",
    },
}


#: Alphabet unicode mapping
alpha_mapping = {
    "boldsans": "𝗔𝗕𝗖𝗗𝗘𝗙𝗚𝗛𝗜𝗝𝗞𝗟𝗠𝗡𝗢𝗣𝗤𝗥𝗦𝗧𝗨𝗩𝗪𝗫𝗬𝗭𝗮𝗯𝗰𝗱𝗲𝗳𝗴𝗵𝗶𝗷𝗸𝗹𝗺𝗻𝗼𝗽𝗾𝗿𝘀𝘁𝘂𝘃𝘄𝘅𝘆𝘇",
    "bold": "𝐀𝐁𝐂𝐃𝐄𝐅𝐆𝐇𝐈𝐉𝐊𝐋𝐌𝐍𝐎𝐏𝐐𝐑𝐒𝐓𝐔𝐕𝐖𝐗𝐘𝐙𝐚𝐛𝐜𝐝𝐞𝐟𝐠𝐡𝐢𝐣𝐤𝐥𝐦𝐧𝐨𝐩𝐪𝐫𝐬𝐭𝐮𝐯𝐰𝐱𝐲𝐳",
    "circle": "ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ",
    "cursive": "𝒜𝐵𝒞𝒟𝐸𝐹𝒢𝐻𝐼𝒥𝒦𝐿𝑀𝒩𝒪𝒫𝒬𝑅𝒮𝒯𝒰𝒱𝒲𝒳𝒴𝒵𝒶𝒷𝒸𝒹𝑒𝒻𝑔𝒽𝒾𝒿𝓀𝓁𝓂𝓃𝑜𝓅𝓆𝓇𝓈𝓉𝓊𝓋𝓌𝓍𝓎𝓏",
    "default": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "doublestruck": "𝔸𝔹ℂ𝔻𝔼𝔽𝔾ℍ𝕀𝕁𝕂𝕃𝕄ℕ𝕆ℙℚℝ𝕊𝕋𝕌𝕍𝕎𝕏𝕐ℤ𝕒𝕓𝕔𝕕𝕖𝕗𝕘𝕙𝕚𝕛𝕜𝕝𝕞𝕟𝕠𝕡𝕢𝕣𝕤𝕥𝕦𝕧𝕨𝕩𝕪𝕫",
    "italicbold": "𝑨𝑩𝑪𝑫𝑬𝑭𝑮𝑯𝑰𝑱𝑲𝑳𝑴𝑵𝑶𝑷𝑸𝑹𝑺𝑻𝑼𝑽𝑾𝑿𝒀𝒁𝒂𝒃𝒄𝒅𝒆𝒇𝒈𝒉𝒊𝒋𝒌𝒍𝒎𝒏𝒐𝒑𝒒𝒓𝒔𝒕𝒖𝒗𝒘𝒙𝒚𝒛",
    "italicboldsans": "𝘼𝘽𝘾𝘿𝙀𝙁𝙂𝙃𝙄𝙅𝙆𝙇𝙈𝙉𝙊𝙋𝙌𝙍𝙎𝙏𝙐𝙑𝙒𝙓𝙔𝙕𝙖𝙗𝙘𝙙𝙚𝙛𝙜𝙝𝙞𝙟𝙠𝙡𝙢𝙣𝙤𝙥𝙦𝙧𝙨𝙩𝙪𝙫𝙬𝙭𝙮𝙯",
    "medieval": "𝔄𝔅ℭ𝔇𝔈𝔉𝔊ℌℑ𝔍𝔎𝔏𝔐𝔑𝔒𝔓𝔔ℜ𝔖𝔗𝔘𝔙𝔚𝔛𝔜ℨ𝔞𝔟𝔠𝔡𝔢𝔣𝔤𝔥𝔦𝔧𝔨𝔩𝔪𝔫𝔬𝔭𝔮𝔯𝔰𝔱𝔲𝔳𝔴𝔵𝔶𝔷",
    "medievalbold": "𝕬𝕭𝕮𝕯𝕰𝕱𝕲𝕳𝕴𝕵𝕶𝕷𝕸𝕹𝕺𝕻𝕼𝕽𝕾𝕿𝖀𝖁𝖂𝖃𝖄𝖅𝖆𝖇𝖈𝖉𝖊𝖋𝖌𝖍𝖎𝖏𝖐𝖑𝖒𝖓𝖔𝖕𝖖𝖗𝖘𝖙𝖚𝖛𝖜𝖝𝖞𝖟",
    "square": "🄰🄱🄲🄳🄴🄵🄶🄷🄸🄹🄺🄻🄼🄽🄾🄿🅀🅁🅂🅃🅄🅅🅆🅇🅈🅉🄰🄱🄲🄳🄴🄵🄶🄷🄸🄹🄺🄻🄼🄽🄾🄿🅀🅁🅂🅃🅄🅅🅆🅇🅈🅉",
    "square_inverted": "🅰🅱🅲🅳🅴🅵🅶🅷🅸🅹🅺🅻🅼🅽🅾🅿🆀🆁🆂🆃🆄🆅🆆🆇🆈🆉🅰🅱🅲🅳🅴🅵🅶🅷🅸🅹🅺🅻🅼🅽🅾🅿🆀🆁🆂🆃🆄🆅🆆🆇🆈🆉",
    "typewriter": "𝙰𝙱𝙲𝙳𝙴𝙵𝙶𝙷𝙸𝙹𝙺𝙻𝙼𝙽𝙾𝙿𝚀𝚁𝚂𝚃𝚄𝚅𝚆𝚇𝚈𝚉𝚊𝚋𝚌𝚍𝚎𝚏𝚐𝚑𝚒𝚓𝚔𝚕𝚖𝚗𝚘𝚙𝚚𝚛𝚜𝚝𝚞𝚟𝚠𝚡𝚢𝚣",
}


def square_scale():
    """ square_scale

    Ordered key for 0-9 mapping to squares from tiny filled square to large hollow square.

    :return: scale from 0 to 9
    """
    return "🞌 🞍 ￭ ⬛ 🞓 🞒 🞑 🞐 🞏 🞎"


def available_charsets():
    """ available_alpha_charsets

        All supported unicode digit charsets, such as 'doublestruck' where 0 looks like: 𝟘

        :return: list of charset names
        """
    return list(mapping.keys())


def available_alpha_charsets():
    """ available_alpha_charsets

    All supported unicode alphabet charsets, such as 'doublestruck' where A looks like: 𝔸

    :return: list of charset names
    """
    return list(alpha_mapping.keys())


def translate_alpha_representation(text, charset=None):
    """ translate_alpha_representation

    Replace the default (ASCII type) charset in a string with the equivalent in
    a different unicode charset.

    :param text: input string
    :param charset: unicode character set as defined by available_alpha_charsets
    :return: translated string
    """
    default = alpha_mapping["default"]
    lookup_charset = alpha_mapping[charset]

    lookup = dict(zip(default, lookup_charset))

    if charset == "arabic_r":
        if text[-1] != "|":
            text_string = text[::-1]
        else:
            text_string = text[text.find("|") - 1 :: -1] + text[-1:]
    else:
        text_string = text

    return "".join([lookup.get(c, c) for c in text_string])


def translate_representation(text, charset=None, index=None, zero_blank=None):
    """ translate_representation

    Replace the default (ASCII type) digit glyphs in a string with the equivalent in
    a different unicode charset.

    :param text: input string
    :param charset: unicode character set as defined by available_alpha_charsets
    :param index: correspond to which item in a list we are looking at, for zero_blank
    :param zero_blank: will blank 0 if True, unless we are looking at header (row index < 2)
    :return: translated string
    """
    lookup = mapping[charset]
    if charset == "arabic_r":
        if text[-1] != "|":
            text_string = text[::-1]
        else:
            text_string = text[text.find("|") - 1 :: -1] + text[-1:]
    else:
        text_string = text
    if index > 1 and zero_blank:
        text_string = text_string[:4] + text_string[4:].replace(" 0", "  ").replace(
            " nan", "    "
        )
    return "".join([lookup.get(c, c) for c in text_string])
