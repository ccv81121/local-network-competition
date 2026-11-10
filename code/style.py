"""Shared figure style: sans-serif (Arial) type, project palette and export."""
import contextlib

import matplotlib

matplotlib.use("Agg")
import matplotlib._mathtext as mathtext_layout  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.mathtext import MathTextParser  # noqa: E402

BEND = "#2C64A4"         # bending / bond-preserving channel, edge-sharing networks
NET = "#C2553A"          # network / bond-changing channel
CORNER = "#3a3a3a"       # corner-sharing networks
LOCAL = "#6b6b6b"        # local (onsite) terms
NEAR_BLACK = "#1F1D1B"
DARK_GREY = "#4E4A45"
GREY = "#8A847B"
LIGHT_GREY = "#D9D4CB"

TEXT_PT = 7.0
LETTER_PT = 10.0
HALO = dict(boxstyle="square,pad=0.12", fc="white", ec="none")   # label interrupts the line under it

RC_LA327 = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "custom", "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
            "mathtext.sf": "Arial", "font.size": TEXT_PT, "axes.linewidth": 0.6,
            "axes.spines.top": False, "axes.spines.right": False, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
            "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none",
            "svg.hashsalt": "localnetwork", "savefig.dpi": 600}   # images inside PDF/SVG at 600 dpi
RC = {**RC_LA327, "mathtext.default": "it", "axes.titlesize": TEXT_PT, "legend.frameon": False,
      "xtick.minor.size": 1.2, "ytick.minor.size": 1.2, "lines.markersize": 4.0}

SHRINK = 0.86            # size of sub- and superscripts relative to their base
SHRINK_LA327 = 0.7


@contextlib.contextmanager
def use(rc=RC, shrink=SHRINK):
    """Draw inside this context: rcParams and mathtext script size of one figure family."""
    old = mathtext_layout.SHRINK_FACTOR
    mathtext_layout.SHRINK_FACTOR = shrink
    MathTextParser._parse_cached.cache_clear()
    try:
        with matplotlib.rc_context():
            matplotlib.rcdefaults()
            matplotlib.rcParams.update(rc)
            yield
    finally:
        mathtext_layout.SHRINK_FACTOR = old
        MathTextParser._parse_cached.cache_clear()


def letter(fig, x, y, s):
    """Bold panel letter at figure coordinates (x, y), bottom-left aligned."""
    fig.text(x, y, s, fontsize=LETTER_PT, fontweight="bold", ha="left", va="bottom")


def legend(ax, handles, **kw):
    """Legend laid out on an 7-pt labels."""
    leg = ax.legend(handles=handles, fontsize=TEXT_PT, **kw)
    for t in leg.get_texts():
        t.set_fontsize(TEXT_PT)
    return leg


def save(fig, outdir, stem):
    """Write <stem>.pdf, .svg and .png (600 dpi) with a tight bounding box."""
    outdir.mkdir(parents=True, exist_ok=True)
    kw = dict(bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outdir / f"{stem}.pdf", metadata={"CreationDate": None}, **kw)
    fig.savefig(outdir / f"{stem}.svg", metadata={"Date": None}, **kw)
    fig.savefig(outdir / f"{stem}.png", dpi=600, **kw)
    plt.close(fig)
