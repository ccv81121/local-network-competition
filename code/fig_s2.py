"""Supplementary Fig. 2: substitution path, margins along Sc->Sb (a) and start / affine step / relaxation at the two
switching steps and the 4-5-GPa step (b)."""
import matplotlib.pyplot as plt
import pandas as pd

import style
from common import DATA, read_json
from style import LIGHT_GREY

STEPS = [("S05->S06", 0, "V", r"$S_{\mathrm{5}}\rightarrow S_{\mathrm{6}}$", (-20, 30), ["initial", "swap", "relax"]),
         ("S11->S12", 1, "C", r"$S_{\mathrm{11}}\rightarrow S_{\mathrm{12}}$", (-160, 45), ["initial", "swap", "relax"]),
         ("P4->P5", 1, "C", "4→5 GPa", (-30, 15), ["initial", "lattice", "relax"])]


def panel_path(ax):
    m = pd.read_csv(DATA["ordering_margins"]).sort_values("x")
    x, lambda_v_mev, lambda_c_mev = (m[k].to_numpy(float) for k in ("x", "Lambda_V_meV", "Lambda_C_meV"))
    x_of = m.set_index("state")["x"]
    ax.axhline(0, color=LIGHT_GREY, lw=0.6, zorder=0)
    for a, b in (("S05", "S06"), ("S11", "S12")):
        ax.axvline((x_of[a] + x_of[b]) / 2, color="black", lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.plot(x, lambda_v_mev, color="black", lw=0.9, zorder=2)
    ax.scatter(x, lambda_v_mev, s=18, marker="o", color="black", edgecolor="white", lw=0.4, zorder=3)
    ax.plot(x, lambda_c_mev, color="black", lw=0.9, zorder=2)
    ax.scatter(x, lambda_c_mev, s=20, marker="D", color="black", edgecolor="white", lw=0.4, zorder=3)
    ax.text(0.075, lambda_v_mev[0] + 25, r"$\mathrm{\Lambda}_{\mathrm{V}}$", ha="center", va="bottom")
    ax.text(0.075, lambda_c_mev[0] + 30, r"$\mathrm{\Lambda}_{\mathrm{C}}$", ha="center", va="bottom")
    x_switch_v, x_switch_c = (x_of["S05"] + x_of["S06"]) / 2, (x_of["S11"] + x_of["S12"]) / 2
    ax.text(x_switch_v + 0.015, -275, r"VBM $\rightarrow$ $\mathrm{\Gamma}$", ha="left", va="center")
    ax.text(x_switch_c + 0.015, -275, r"CBM $\rightarrow$ $q_{\mathrm{C}}$", ha="left", va="center")
    ax.set_xlim(0.03, 0.91)
    ax.set_ylim(-310, 210)
    ax.set_xticks([0.25, 0.5, 0.75])
    ax.set_yticks([-200, -100, 0, 100, 200])
    ax.set_xlabel(r"Sb fraction $x$ in Mg(Sc$_{\mathrm{1}-x}$Sb$_x$)$_{\mathrm{2}}$Se$_{\mathrm{4}}$")
    ax.set_ylabel("Margin (meV)")
    top = ax.secondary_xaxis("top")             # path structures S1 ... S14
    top.set_xticks(x)
    top.set_xticklabels([f"S{int(s[1:])}" for s in m["state"]], rotation=90, fontsize=6)
    top.tick_params(length=2, pad=1.5)


def panel_steps(axes):
    """Margin at the initial structure, after the swap or lattice step at fixed internal coordinates, and after relaxation."""
    split = read_json("path_split")
    rows = {r["step"]: r["Lambda_fixed"] for r in split["composition"] + split["pressure"]}
    for ax, (step, j, edge, title, ylim, xlabels) in zip(axes, STEPS):
        f = rows[step]
        mid = f["chem"][j] if "chem" in f else f["aff"][j]
        y = [f["start"][j], mid, f["end"][j]]
        ax.axhline(0, color=LIGHT_GREY, lw=0.6, zorder=0)
        ax.plot([0, 1, 2], y, color="black", lw=0.9, zorder=2)
        ax.scatter([0, 1, 2], y, s=20, marker="o" if edge == "V" else "D", color="black", edgecolor="white", lw=0.4,
                   zorder=3)
        ax.set_xlim(-0.35, 2.35)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(xlabels)
        ax.set_title(title, pad=3)
        ax.text(0.04 if edge == "V" else 0.96, 0.96, rf"$\mathrm{{\Lambda}}_{{\mathrm{{{edge}}}}}$",
                transform=ax.transAxes, ha="left" if edge == "V" else "right", va="top")
    axes[0].set_ylabel("Margin (meV)")


def draw():
    fig = plt.figure(figsize=(7.0, 2.5))
    axa = fig.add_axes([0.08, 0.20, 0.33, 0.66])
    axs = [fig.add_axes([0.52 + 0.165 * k, 0.20, 0.115, 0.66]) for k in range(3)]
    panel_path(axa)
    panel_steps(axs)
    y = 0.20 + 0.66 + 0.06
    style.letter(fig, 0.08 - 0.07, y, "a")
    style.letter(fig, 0.52 - 0.07, y, "b")
    return fig


def make(outdir):
    with style.use():
        style.save(draw(), outdir, "fig_s2")
