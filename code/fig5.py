"""Supplementary Fig. 5 (fig_s6): explicit structural relations, double-perovskite out-of-fold plane (a), group neutralisation (b), removal
of shared primitives (c), the two path coordinates along Sc->Sb (d)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import style
from common import DATA
from style import BEND, GREY, LIGHT_GREY, LOCAL, NET

# role of each structural primitive: local preference q0,K / accommodation q* / network L / electronic coupling W_e
ROLE = {"BXB_angle_min": "L", "D_XBX_angle_disorder_norm": "q*", "elem_A_period": "q0", "B_d_filling_mean": "q0",
        "onsite_alignment_inverse": "W", "D_BX6_bond_cv": "q*", "XX_d1_std": "L", "XX_distance_anisotropy": "L",
        "BB_graph_shortest_path_mean": "L", "BX6_distortion_cv": "q*", "BXB_path_length_std": "L",
        "BX_distance_skew": "q*"}
ROLE_COLOR = {"q0": LOCAL, "q*": "#8B5A2B", "L": NET, "W": "black"}
SHORT = {"BXB_angle_min": r"$\theta_{BXB}^{\min}$", "D_XBX_angle_disorder_norm": r"$D_{XBX}$",
         "elem_A_period": r"$P_A$", "B_d_filling_mean": r"$n_{Bd}$",
         "onsite_alignment_inverse": r"$\mathrm{\Delta}\varepsilon_{\mathrm{on}}$", "D_BX6_bond_cv": r"$D_{BX6}$",
         "XX_d1_std": r"$\sigma_{d,XX}$", "XX_distance_anisotropy": r"$A_{d,XX}$",
         "BB_graph_shortest_path_mean": r"$\ell_{BB}$", "BX6_distortion_cv": r"$c_{BX6}$",
         "BXB_path_length_std": r"$\sigma_{\ell,BXB}$", "BX_distance_skew": r"$\gamma_{d,BX}$"}


def signed_side_rank(v):
    """Rank-scale each sign side to (0, 1], keeping sign and within-side order (display only)."""
    out = np.zeros_like(v, dtype=float)
    for s in (-1.0, 1.0):
        mk = s * v > 0
        if mk.any():
            r = pd.Series(np.abs(v[mk])).rank(method="average").to_numpy(float)
            out[mk] = s * (r - 0.5) / mk.sum()
    return out


def panel_dp_plane(ax):
    """Double-perovskite states in the plane of the out-of-fold valence and conduction coordinates."""
    d = pd.read_csv(DATA["dp_out_of_fold"])
    m_v, m_c = signed_side_rank(d.margin_V.to_numpy(float)), signed_side_rank(d.margin_C.to_numpy(float))
    ok = ((d.pred_V == d.A_V) & (d.pred_C == d.A_C)).to_numpy()
    ax.axhline(0, color="black", lw=0.5, zorder=1)
    ax.axvline(0, color="black", lw=0.5, zorder=1)
    ax.scatter(m_v[ok], m_c[ok], s=14, color=GREY, edgecolor="white", lw=0.3, zorder=3)
    ax.scatter(m_v[~ok], m_c[~ok], s=24, facecolor="white", edgecolor="black", lw=0.8, zorder=4)
    quadrant = {(1, 1): (0.55, 1.13, r"both at $k_D$"), (-1, 1): (-0.55, 1.13, r"CBM at $k_D$"),
                (-1, -1): (-0.55, -1.13, r"neither at $k_D$"), (1, -1): (0.55, -1.13, r"VBM at $k_D$")}
    for (sv, sc), (xq, yq, lab) in quadrant.items():
        n = int(((np.sign(d.A_V * 2 - 1) == sv) & (np.sign(d.A_C * 2 - 1) == sc)).sum())
        ax.text(xq, yq, f"{lab} ({n})", ha="center", va="center")
    ax.set_xlim(-1.27, 1.27)
    ax.set_ylim(-1.27, 1.27)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xticklabels(["−1", "0", "1"])
    ax.set_yticklabels(["−1", "0", "1"])
    ax.set_xlabel(r"Valence coordinate $m_{\mathrm{V}}$ (rank-scaled)", labelpad=1)
    ax.set_ylabel(r"Conduction coordinate $m_{\mathrm{C}}$ (rank-scaled)", labelpad=1)
    h = [Line2D([], [], marker="o", ls="", ms=3.6, color=GREY, label=f"both edges correct ({ok.sum()})"),
         Line2D([], [], marker="o", ls="", ms=4.2, mfc="white", mec="black",
                label=f"an edge misassigned ({(~ok).sum()})")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.45, -0.115), ncol=2, handletextpad=0.2,
              borderaxespad=0, labelspacing=0.3)


def panel_neutralisation(ax):
    """Loss of balanced accuracy when the local or the network primitives of a relation are neutralised."""
    g = pd.read_csv(DATA["group_neutralisation"]).set_index("ablation")
    relations = [(r"$\mathrm{\Pi}_{\mathrm{col}}$", "sp"), (r"$\mathrm{\Pi}_{\mathrm{V}}$", "V"),
                 (r"$\mathrm{\Pi}_{\mathrm{C}}$", "C")]
    yy = np.arange(3)[::-1]
    for i, (_, k) in enumerate(relations):
        local_loss, network_loss = -g.loc["GROUP_local", f"{k}_d_dev"], -g.loc["GROUP_network", f"{k}_d_dev"]
        ax.plot([network_loss, local_loss], [yy[i]] * 2, "-", color=LIGHT_GREY, lw=1.6, zorder=0)
        ax.plot(local_loss, yy[i], "o", color=LOCAL, ms=4.5)
        ax.plot(network_loss, yy[i], "o", color=NET, ms=4.5)
    ax.set_yticks(yy)
    ax.set_yticklabels([r for r, _ in relations])
    ax.set_ylim(-0.6, 2.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.35)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel("Loss of balanced accuracy")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color=LOCAL, label="local"),
                       Line2D([], [], ls="", marker="o", color=NET, label="network")],
              loc="upper left", bbox_to_anchor=(0.98, 1.0), ncol=1, handletextpad=0.1, borderaxespad=0,
              labelspacing=0.3, handlelength=1.0)


def panel_removal(ax):
    """Median loss on removing each primitive shared by all 30 near-optimal relations (top eight)."""
    a = pd.read_csv(DATA["feature_ablation"])
    top = a[a.parents == 30].sort_values("median_delta_BACC").head(8)
    yc = np.arange(len(top))
    loss = -top.median_delta_BACC.to_numpy()
    col = [ROLE_COLOR[ROLE[f]] for f in top.feature]
    ax.hlines(yc, 0, loss, color=col, lw=1.0)
    ax.scatter(loss, yc, s=14, c=col, zorder=3, lw=0)
    ax.set_yticks(yc)
    ax.set_yticklabels([SHORT[f] for f in top.feature])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.5)
    ax.set_xticks([0, 0.2, 0.4])
    ax.set_xlabel("Median loss on removal")
    ax.text(loss[0] + 0.025, 0, f"{loss[0]:.3f}", ha="left", va="center")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=3.5, color=ROLE_COLOR[r], label=lab) for r, lab in
                       (("q0", r"$q^0,K$"), ("q*", r"$q^*$"), ("L", r"$L$"), ("W", r"$W_e$"))],
              loc="lower right", ncol=2, handletextpad=0.05, columnspacing=0.6, borderaxespad=0.1, title="role",
              title_fontsize=7.0)


def panel_path(axes):
    """Valence and conduction coordinates m = tau - Pi along the Sc->Sb sequence; open marks: m <= 0."""
    p = pd.read_csv(DATA["path_terms"])
    k = np.arange(1, len(p) + 1)
    coordinates = (("Pi_V", "tau_V", r"$m_{\mathrm{V}}$", 5.5, "valence"),
                   ("Pi_C", "tau_C", r"$m_{\mathrm{C}}$", 11.5, "conduction"))
    for j, (ax, (pi, tau, lab, switch, name)) in enumerate(zip(axes, coordinates)):
        m = -(p[pi] - p[tau])
        ax.axvline(switch, color=GREY, lw=0.6, ls=(0, (2, 2)), zorder=0)
        ax.axhline(0, color=LIGHT_GREY, lw=0.6, zorder=0)
        pos = m > 0
        ax.plot(k, m, "-", color="black", lw=0.8, zorder=1)
        ax.scatter(k[pos], m[pos], s=13, c="black", zorder=2, lw=0)
        ax.scatter(k[~pos], m[~pos], s=13, facecolors="white", edgecolors="black", lw=0.7, zorder=2)
        ax.set_ylabel(lab)
        ax.set_xlim(0.5, 14.5)
        ax.set_xticks(k)
        ax.text(switch - 0.3 if j else switch + 0.3, 0.03, f"{name}\nreordering", transform=ax.get_xaxis_transform(),
                va="bottom", ha="right" if j else "left", linespacing=1.0)
    axes[0].set_xticklabels([])
    axes[0].set_ylim(-0.45, 1.3)
    axes[0].set_yticks([0, 0.5, 1.0])
    axes[1].set_xticklabels([str(i) if i % 2 else "" for i in k])
    axes[1].set_ylim(-0.42, 0.25)
    axes[1].set_yticks([-0.4, -0.2, 0, 0.2])
    axes[1].set_xlabel(r"Structure $S_k$ along Sc$\rightarrow$Sb")


def draw():
    fig = plt.figure(figsize=(7.2, 3.35))
    axb = fig.add_axes([0.505, 0.705, 0.165, 0.215])
    axc = fig.add_axes([0.505, 0.115, 0.165, 0.40])
    axd = [fig.add_axes([0.785, 0.56, 0.20, 0.36]), fig.add_axes([0.785, 0.115, 0.20, 0.39])]
    axa = fig.add_axes([0.075, 0.215, 0.33, 0.705])
    panel_dp_plane(axa)
    panel_neutralisation(axb)
    panel_removal(axc)
    panel_path(axd)
    for ax, s, x, y in ((axb, "b", -0.3, 1.1), (axc, "c", -0.3, 1.04), (axd[0], "d", -0.2, 1.04),
                        (axa, "a", -0.24, 1.03)):
        style.letter(fig, *fig.transFigure.inverted().transform(ax.transAxes.transform((x, y))), s)
    return fig


def make(outdir):
    with style.use():
        style.save(draw(), outdir, "fig_s6")
