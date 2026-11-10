"""Supplementary Fig. 5 (from the earlier real-field figure): the Ti->Sn substitution in the 80-atom CaTiO3 cell as source, filter and response.
a, force f; b, relaxed displacement over the prediction H^+ f; c, bond-preserving part of the relaxed displacement;
d, decay of |u| from the Sn site; e, q-sector shares of the bond-preserving weight.
Colours: force ochre (as in the main field figure), prediction grey, relaxed black, bond-preserving part blue."""
import json

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import render_fields
import style
from common import _D
from style import BEND, GREY, NEAR_BLACK, NET

RESULT = _D / "field_sectors" / "dispersion" / "dispersion_result.json"
SECTORS = [("Gamma20", r"$\mathrm{\Gamma}$"), ("X20(1/2,0,0)", "X"), ("Z20(0,0,1/2)", "Z"), ("U20(1/2,0,1/2)", "U")]
PRED = "#B4B0A8"                                   # prediction: light warm grey, drawn under the relaxed field


def _tint(c, t):
    a = np.array(mc.to_rgb(c))
    return tuple(a + (1 - a) * t)


def panel_decay(ax):
    z = np.load(render_fields.SUBSTITUTION)
    act, pred = z["shells_act"], z["shells_pred"]
    d = np.linspace(0.8, 8.8, 100)
    (aa, xa), (ap, xp) = z["fit_act"], z["fit_pred"]
    ax.plot(d, ap * np.exp(-d / xp), color=GREY, lw=1.6, alpha=0.6, zorder=1, solid_capstyle="round")
    ax.plot(d, aa * np.exp(-d / xa), color=NEAR_BLACK, lw=0.8, zorder=2)
    ax.plot(pred[:, 0], pred[:, 1], "o", ms=5.0, color=PRED, zorder=3)
    ax.plot(act[:, 0], act[:, 1], "o", ms=3.2, mfc="white", mec=NEAR_BLACK, mew=0.8, zorder=4)
    d_atom, sp, sn = z["dist"], z["species"], int(z["sn"])
    shell = np.floor(d_atom / 1.0).astype(int)    # the 1 Angstrom shells of the average
    for k in range(len(act)):
        members = [sp[i] for i in range(len(sp)) if shell[i] == int(act[k, 0]) and i != sn]
        kinds = [e for e in ("Ca", "Ti", "O") if e in members]
        kinds.sort(key=lambda e: -members.count(e))
        ax.annotate(" ".join(kinds), (act[k, 0], max(act[k, 1], pred[k, 1])), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=5, color=NEAR_BLACK)
    ax.set_yscale("log")
    ax.set_xlim(0, 9.2)
    ax.set_ylim(4e-3, 0.25)
    ax.set_yticks([0.01, 0.1])
    ax.set_yticklabels(["0.01", "0.1"])
    ax.minorticks_off()
    ax.set_xlabel("Distance from Sn (Å)")
    ax.set_ylabel(r"Mean $|u|$ (Å)")
    h = [Line2D([], [], marker="o", ms=3.2, mfc="white", mec=NEAR_BLACK, color=NEAR_BLACK, lw=0.8,
                label=f"relaxed, decay {xa:.2f} Å"),
         Line2D([], [], marker="o", ms=5.0, color=PRED, mec=PRED, lw=1.6, alpha=0.9,
                label=rf"$H^{{\mathrm{{-1}}}}f$, decay {xp:.2f} Å")]
    style.legend(ax, h, loc="upper right", handletextpad=0.5, labelspacing=0.3, borderaxespad=0.1, handlelength=1.8)


def panel_sectors(ax):
    t4 = json.loads(RESULT.read_text(encoding="utf-8"))["T4"]
    x = np.arange(len(SECTORS))
    w = 0.25
    for k, (key, col, lab) in enumerate((("f", "#D4A02F", "force $f$"), ("u_pred", PRED, r"$H^{\mathrm{-1}}f$"),
                                         ("u_actual", NEAR_BLACK, "relaxed"))):
        d = t4[key]
        share = {s["q"]: 100 * s["bp_weight"] / d["s_total"] for s in d["q_sectors"]}
        ax.bar(x + (k - 1) * w, [share[q] for q, _ in SECTORS], w * 0.92, color=col, lw=0, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in SECTORS])
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0, 60)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("Bond-preserving weight (%)")
    ax.legend(loc="upper right", fontsize=style.TEXT_PT, handlelength=0.9, handletextpad=0.4, borderaxespad=0.0,
              labelspacing=0.25)


def draw():
    W, H = 7.0, 2.3
    fig = plt.figure(figsize=(W, H))
    box = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    axa = box(0.62, 0.42, 2.55, 1.70)
    axb = box(4.15, 0.42, 2.70, 1.70)
    panel_decay(axa)
    panel_sectors(axb)
    for s, x in (("a", 0.0), ("b", 3.55)):
        style.letter(fig, x / W, 2.2 / H, s)
    return fig


def make(outdir):
    with style.use():
        style.save(draw(), outdir, "fig_s5")
