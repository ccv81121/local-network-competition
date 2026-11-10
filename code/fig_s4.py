"""Supplementary Fig. 4: geometric and property factors of the bond-preserving share for the x = 0.75 and x = 0.5 spinel steps,
with conditional four-state increments (crosses)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import style
from common import read_json
from style import BEND, CORNER, HALO, LIGHT_GREY, NEAR_BLACK

EDGE_SYSTEMS = {"k12_t45", "k12_05", "k08", "Layered_NaScInO2"}
SYSTEM_LABELS = {"k12_t45": "4–5", "k12_05": "0–5", "k08": "$x$ = 0.5",
                 "Layered_NaScInO2": r"Na(Sc,In)O$_{\mathrm{2}}$",
                 "CaTiO3": r"CaTiO$_{\mathrm{3}}$", "CaSnO3": r"CaSnO$_{\mathrm{3}}$",
                 "Ca2SnTiO6_rs": r"Ca$_{\mathrm{2}}$SnTiO$_{\mathrm{6}}$",
                 "Ca8Sn6Ti2O24": r"Ca$_{\mathrm{8}}$Sn$_{\mathrm{6}}$Ti$_{\mathrm{2}}$O$_{\mathrm{24}}$"}
LABEL_OFFSETS = {"k12_t45": (7, 8, "left"), "k12_05": (-7, -6, "right"), "k08": (6, 0, "left"),
                 "Layered_NaScInO2": (5, 0, "left"), "CaTiO3": (-6, 0, "right"), "CaSnO3": (8, 0, "left"),
                 "Ca2SnTiO6_rs": (-6, 0, "right"), "Ca8Sn6Ti2O24": (-6, 0, "right")}

def panel_factors(ax):
    """Geometric and property factors of the bond-preserving share; dashed lines: constant |phi/(1-phi)|, open markers: g_b/g_r < 0."""
    prepare = read_json("leverage_prepare")
    three_channel = {**read_json("three_channel"), **read_json("three_channel_all")}
    x = np.logspace(np.log10(0.02), np.log10(1.6), 50)
    for c, phi in ((0.01, "0.01"), (0.1, "0.1"), (1, "1"), (10, "10")):   # constant |phi/(1-phi)|
        ax.plot(x, c / x, color=LIGHT_GREY, lw=0.6, ls=(0, (2.5, 2)), zorder=0)
        xl = {"10": 0.05, "0.01": 0.022}.get(phi, 1.45)
        ax.text(xl, c / xl, phi, ha="left" if phi in ("10", "0.01") else "right", va="center", bbox=HALO, zorder=2)
    for k, r in three_channel.items():
        u_b, u_r = prepare[k]["ub_norm_A"], prepare[k]["uc_network_norm_A"]
        geometric = u_b / u_r
        electronic = (r["pred_meV"]["b"] / u_b) / (r["pred_meV"]["n"] / u_r)
        electronic_four_state = (r["meas_meV"]["b"] / u_b) / (r["meas_meV"]["n"] / u_r)
        edge = k in EDGE_SYSTEMS
        col = BEND if edge else CORNER
        m = "o" if edge else "s"
        ax.plot([geometric] * 2, [abs(electronic), abs(electronic_four_state)], color=col, lw=0.6, zorder=1)
        same_sign = electronic > 0                  # open markers: g_b/g_r < 0
        ax.scatter(geometric, abs(electronic), s=20, marker=m, facecolor=col if same_sign else "white",
                   edgecolor=col, lw=0.9, zorder=3)
        ax.scatter(geometric, abs(electronic_four_state), s=10, marker="x", color=NEAR_BLACK, lw=0.6, zorder=4)
        dx, dy, ha = LABEL_OFFSETS[k]
        ax.annotate(SYSTEM_LABELS[k], (geometric, abs(electronic)), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", bbox=HALO, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.02, 1.6)
    ax.set_ylim(1e-3, 400)
    ax.set_xticks([0.1, 1])
    ax.set_xticklabels(["0.1", "1"])
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1, 10, 100])
    ax.set_yticklabels(["0.001", "0.01", "0.1", "1", "10", "100"])
    ax.set_xlabel(r"Geometric factor $|u_b|/|u_r|$")
    ax.set_ylabel(r"Property factor $|g_b/g_r|$")
    h = [Line2D([], [], marker="o", ls="", ms=4, color=BEND, label="edge"),
         Line2D([], [], marker="s", ls="", ms=4, color=CORNER, label="corner"),
         Line2D([], [], marker="o", ls="", ms=4, mfc="white", mec=NEAR_BLACK, label="opposite signs"),
         Line2D([], [], marker="x", ls="", ms=3.5, color=NEAR_BLACK, label="four-state")]
    style.legend(ax, h, loc="lower left", ncol=1, handletextpad=0.1, labelspacing=0.2, borderaxespad=0.1)


def draw():
    fig = plt.figure(figsize=(3.6, 3.0))
    panel_factors(fig.add_axes([0.17, 0.15, 0.80, 0.80]))
    return fig


def make(outdir):
    with style.use():
        style.save(draw(), outdir, "fig_s4")
