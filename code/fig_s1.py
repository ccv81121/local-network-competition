"""Supplementary Fig. 1: band edges along compression, bands at 4 and 5 GPa with Sc d / Sb weights (a) and ordering
margins with and without SOC (b)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import style
from common import DATA, read_json
from style import LIGHT_GREY

SC_D = "#6B8E23"   # species colours, distinct from the BEND/NET/CORNER roles
SB = "#A0527C"
BANDS = "#B9B4AC"
Y_LETTER = 0.47 + 0.40 + 0.06


def panel_bands(fig, j, key, title):
    """Bands of one pressure on a broken energy axis; conduction-band dots coloured by the dominant species."""
    d = np.load(DATA[key])
    x, energy, weight = d["dist"], d["E"], d["W"]
    i_gamma = int(np.argmin(np.linalg.norm(d["k"], axis=1)))
    x0 = 0.08 + j * 0.17
    top = fig.add_axes([x0, 0.47, 0.15, 0.40])
    bot = fig.add_axes([x0, 0.20, 0.15, 0.22])
    for ax, lo, hi in ((top, 1.02, 1.26), (bot, -0.17, 0.03)):
        for b in range(148, 166):
            ax.plot(x, energy[:, b], color=BANDS, lw=0.6, zorder=1)
        for b in range(156, 166):                 # conduction bands
            sc, sb = weight[:, b, 0], weight[:, b, 1]
            keep = slice(None, None, 2)
            for wgt, col in ((sc, SC_D), (sb, SB)):
                dom = (wgt >= np.maximum(sc, sb))[keep]
                ax.scatter(x[keep][dom], energy[keep, b][dom], s=(wgt[keep][dom] * 18), color=col, lw=0, zorder=2)
        ax.axvline(x[i_gamma], color=LIGHT_GREY, lw=0.5, zorder=0)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(lo, hi)
    cb, vb = energy[:, 156], energy[:, 155]
    top.plot(x[int(np.argmin(cb))], cb.min(), "o", ms=5.5, mfc="none", mec="black", mew=0.9, zorder=4)
    bot.plot(x[int(np.argmax(vb))], vb.max(), "o", ms=5.5, mfc="none", mec="black", mew=0.9, zorder=4)
    top.set_title(title, pad=7)
    top.set_xticks([])
    top.spines["bottom"].set_visible(False)
    bot.spines["top"].set_visible(False)
    bot.set_xticks([x[0], x[i_gamma], x[-1]])
    bot.set_xticklabels(["R", r"$\mathrm{\Gamma}$", "Y"])
    top.set_yticks([1.05, 1.15, 1.25])
    bot.set_yticks([-0.1, 0.0])
    if j:
        top.set_yticklabels([])
        bot.set_yticklabels([])
    for ax, yy in ((top, 0.0), (bot, 1.0)):       # axis-break marks
        ax.plot([-0.04, 0.04], [yy - 0.03, yy + 0.03], transform=ax.transAxes, color="black", clip_on=False, lw=0.6)


def panel_margins(ax):
    """Valence and conduction margins along compression, PBE on the 6x6x6 mesh (filled) and with SOC (open squares)."""
    edges = sorted((r for r in read_json("path_edges") if r["mesh"] == 6), key=lambda r: r["P_GPa"])
    p_gpa = np.array([r["P_GPa"] for r in edges])
    lambda_v_mev = np.array([r["Lambda_V_meV"] for r in edges])
    lambda_c_mev = np.array([r["Lambda_C_meV"] for r in edges])
    soc = pd.read_csv(DATA["soc_margins"]).sort_values("P_GPa")
    p_soc, lambda_v_soc, lambda_c_soc = (soc["P_GPa"].to_numpy(float), soc["Lambda_V_SOC_meV"].to_numpy(float),
                                         soc["Lambda_C_SOC_meV"].to_numpy(float))
    ax.axhline(0, color=LIGHT_GREY, lw=0.6, zorder=0)
    for y, y_soc, m, lab in ((lambda_v_mev, lambda_v_soc, "o", r"$\mathrm{\Lambda}_{\mathrm{V}}$"),
                             (lambda_c_mev, lambda_c_soc, "D", r"$\mathrm{\Lambda}_{\mathrm{C}}$")):
        ax.plot(p_gpa, y, color="black", lw=0.9, zorder=2)
        ax.scatter(p_gpa, y, s=20, marker=m, color="black", edgecolor="white", lw=0.4, zorder=3)
        ax.plot(p_soc, y_soc, color="black", lw=0.8, ls=(0, (2.5, 1.5)), zorder=2)
        ax.scatter(p_soc, y_soc, s=22, marker="s", facecolor="white", edgecolor="black", lw=0.8, zorder=3)
        ax.text(-0.3, y[0], lab, ha="right", va="center")
    ax.text(7.5, lambda_c_soc[-1] - 14, "SOC", ha="center", va="top")
    ax.set_xlim(-1.4, 8.0)
    ax.set_ylim(-130, 95)
    ax.set_xticks([0, 2, 4, 6])
    ax.set_yticks([-100, -50, 0, 50])
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Margin (meV)")


def draw():
    fig = plt.figure(figsize=(7.0, 2.6))
    for j, (key, title) in enumerate((("fatband_p4", "4 GPa"), ("fatband_p5", "5 GPa"))):
        panel_bands(fig, j, key, title)
    fig.text(0.025, 0.535, r"$E-E_{\mathrm{VBM}}$ (eV)", rotation=90, va="center", ha="center")
    x_a = 0.08 - 0.07
    h = [Line2D([], [], marker="o", ls="", ms=4.5, color=SC_D, label=r"Sc $d$"),
         Line2D([], [], marker="o", ls="", ms=4.5, color=SB, label=r"Sb $p$–Se $s$")]
    fig.legend(handles=h, loc="lower left", ncol=2, bbox_to_anchor=(x_a + 0.035, Y_LETTER - 0.004),
               bbox_transform=fig.transFigure, borderaxespad=0, handletextpad=0.2, columnspacing=0.6, frameon=False)
    panel_margins(fig.add_axes([0.54, 0.20, 0.42, 0.67]))
    style.letter(fig, x_a, Y_LETTER, "a")
    style.letter(fig, 0.60 - 0.075 - 0.06, Y_LETTER, "b")
    return fig


def make(outdir):
    with style.use():
        style.save(draw(), outdir, "fig_s1")
