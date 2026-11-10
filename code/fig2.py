"""Fig. 2: structural filtering, force and displacement fractions (a), bond-preserving amplitudes (b), octahedral tilts
under pressure (c), and symmetry sectors of CaTiO3 under compression and a radial source on one Ti (d)."""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import render_fields
import style
from common import DATA, read_json
from style import BEND, CORNER, GREY, HALO, LIGHT_GREY, DARK_GREY, NEAR_BLACK, NET

BAND = "#E9EEF3"

# (force fraction s_f, displacement fraction s_u) in % under pressure
EDGE_PRESSURE = [(0.54, 32.9), (0.29, 4.9), (0.47, 4.05), (0.0029, 0.052)]
CORNER_PRESSURE = [(10.1, 10.8), (11.6, 25.6), (7.9, 14.5), (11.2, 30.5)]
SUBSTITUTION = [(2.04, 2.04 * 13.1, "edge"), (0.64, 0.64 * 17.2, "edge"), (0.09, 24.8, "corner")]
AMPLITUDE_NAMES = {"MgSc0.5Sb1.5Se4 k12": "$x$ = 0.75", "MgScSbSe4 k08": "$x$ = 0.5",
                   "Cd(Ga,In)2O4": r"Cd(Ga,In)$_{\mathrm{2}}$O$_{\mathrm{4}}$",
                   "Na(Sc,In)O2 layered": r"Na(Sc,In)O$_{\mathrm{2}}$", "CaTiO3": r"CaTiO$_{\mathrm{3}}$",
                   "CaSnO3": r"CaSnO$_{\mathrm{3}}$", "Ca2SnTiO6": r"Ca$_{\mathrm{2}}$SnTiO$_{\mathrm{6}}$",
                   "Ca8Sn6Ti2O24": r"Ca$_{\mathrm{8}}$Sn$_{\mathrm{6}}$Ti$_{\mathrm{2}}$O$_{\mathrm{24}}$"}
# CaSnO3 single crystal, Zhao, Ross & Angel, Phys. Chem. Miner. 31, 299 (2004), Table 3; mean = (O1 + 2 O2)/3
EXP_P_GPA = np.array([0, 0.738, 1.50, 2.61, 3.94, 5.18, 5.98, 6.73])
EXP_ANGLE_O1 = np.array([147.04, 146.61, 146.49, 146.59, 146.50, 146.66, 146.58, 146.23])
EXP_ANGLE_O2 = np.array([147.87, 147.92, 147.67, 147.61, 147.43, 146.95, 147.06, 147.28])
EXP_SIGMA_O1 = np.array([.23, .22, .24, .21, .22, .19, .20, .18])
EXP_SIGMA_O2 = np.array([.13, .12, .13, .12, .12, .11, .10, .11])


def read_tilt():
    """Mean B-O-B angle against pressure for CaTiO3 and CaSnO3: {name: [[p_gpa, angle_deg], ...]}."""
    out, cur = {}, None
    for line in DATA["tilt_vs_pressure"].read_text().splitlines():
        if line.strip() in ("CaTiO3", "CaSnO3"):
            cur = line.strip()
            out[cur] = []
        elif "mean" in line:
            p_gpa = float(line.split("P=")[1].split("GPa")[0])
            angle = float(line.split("mean")[1].split()[0])
            out[cur].append((p_gpa, angle))
    return {k: np.array(v) for k, v in out.items()}


def panel_shares(ax):
    xs = np.logspace(-3, 2.2, 50)
    for e, lab in ((1, r"$\varepsilon=1$"), (10, "10"), (100, "100")):
        ax.plot(xs, e * xs, color=LIGHT_GREY, lw=0.6, ls="-" if e == 1 else (0, (2.5, 2)), zorder=0)
        x0 = {1: 0.04, 10: 0.02, 100: 0.0025}[e]
        ax.text(x0, e * x0, lab, ha="center", va="center", rotation=48, rotation_mode="anchor", bbox=HALO, zorder=2)
    ax.axhline(100, color=DARK_GREY, lw=0.6, zorder=0)
    ax.text(1.2e-3, 100 * 1.12, r"$\varepsilon\leq1/s_f$", va="bottom")
    for s_f, s_u in EDGE_PRESSURE:
        ax.scatter(s_f, s_u, s=22, marker="o", color=BEND, edgecolor="white", lw=0.4, zorder=3)
    for s_f, s_u in CORNER_PRESSURE:
        ax.scatter(s_f, s_u, s=18, marker="s", color=CORNER, edgecolor="white", lw=0.4, zorder=3)
    for s_f, s_u, topology in SUBSTITUTION:
        edge = topology == "edge"
        ax.scatter(s_f, s_u, s=26, marker="D", facecolor=BEND if edge else "white", edgecolor=BEND if edge else CORNER,
                   lw=0.9, zorder=4)
    ax.annotate("Ti→Sn\n278×", (0.09, 24.8), xytext=(-4, 0), textcoords="offset points", ha="right", va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xlim(1e-3, 60)
    ax.set_ylim(1e-2, 300)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10])
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1", "10"])
    ax.set_yticks([1e-2, 1e-1, 1, 10, 100])
    ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.set_xlabel(r"Force fraction $s_f$ (%)")
    ax.set_ylabel(r"Displacement fraction $s_u$ (%)")
    h = [Line2D([], [], marker="o", ls="", ms=4.2, color=BEND, label="edge, compression"),
         Line2D([], [], marker="s", ls="", ms=3.8, mfc=CORNER, mec=CORNER, mew=0.9, label="corner, compression"),
         Line2D([], [], marker="D", ls="", ms=3.8, mfc=BEND, mec=BEND, label="edge, substitution"),
         Line2D([], [], marker="D", ls="", ms=3.8, mfc="white", mec=CORNER, mew=0.9, label="corner, substitution")]
    style.legend(ax, h, loc="lower right", handletextpad=0.2, labelspacing=0.25, borderaxespad=0.1)


def panel_amplitudes(ax):
    """Direct term (open) and full response (filled) of the bond-preserving amplitude |u_b|/|u_r|."""
    rows = [r for r in read_json("bending_amplitude") if r["system"] in AMPLITUDE_NAMES]
    rows.sort(key=lambda r: (r["topology"] != "edge", -r["amplitude_ratio_up_over_uq"]))
    assert len(rows) == len(AMPLITUDE_NAMES)
    for i, r in enumerate(rows):
        edge = r["topology"] == "edge"
        c = BEND if edge else CORNER
        m = "o" if edge else "s"
        direct, full = r["block_direct_up_per_uq"], r["amplitude_ratio_up_over_uq"]
        ax.plot([direct, full], [i, i], color=c, lw=1.1, solid_capstyle="round", zorder=1)
        ax.scatter(direct, i, s=18, marker=m, facecolor="white", edgecolor=c, lw=0.9, zorder=3)
        ax.scatter(full, i, s=18, marker=m, color=c, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([AMPLITUDE_NAMES[r["system"]] for r in rows])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(0, 0.75)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel(r"$|u_b|\,/\,|u_r|$")
    ax.text(0.29, -0.75, "direct", ha="center", va="bottom")
    ax.text(0.70, -0.75, "full", ha="center", va="bottom")
    ax.set_ylim(7.5, -1.1)


def panel_tilts(ax):
    tilt = read_tilt()
    ax.axhline(0, color=LIGHT_GREY, lw=0.6, zorder=0)
    for name, m, lab in (("CaTiO3", "^", r"CaTiO$_{\mathrm{3}}$"), ("CaSnO3", "s", r"CaSnO$_{\mathrm{3}}$")):
        p_gpa, d_angle = tilt[name][:, 0], tilt[name][:, 1] - tilt[name][0, 1]
        ax.plot(p_gpa, d_angle, color=CORNER, lw=1.0, zorder=2)
        ax.scatter(p_gpa, d_angle, s=18, marker=m, color=CORNER, edgecolor="white", lw=0.4, zorder=3 if name == "CaSnO3" else 3.5)
        ax.text(20.6, d_angle[-1] + (0.13 if name == "CaSnO3" else -0.13), lab, va="center", ha="left")
    mean = (EXP_ANGLE_O1 + 2 * EXP_ANGLE_O2) / 3
    sigma = np.sqrt(EXP_SIGMA_O1 ** 2 + 4 * EXP_SIGMA_O2 ** 2) / 3
    ax.errorbar(EXP_P_GPA, mean - mean[0], yerr=sigma, fmt="o", ms=3.2, mfc="white", mec=NEAR_BLACK, ecolor=NEAR_BLACK,
                elinewidth=0.6, capsize=1.2, mew=0.7, zorder=4)
    pp = np.array([0, 10])
    for rate in (0.13, 0.16):
        ax.plot(pp, -rate * pp, color=NEAR_BLACK, lw=0.7, ls=(0, (1.2, 1.4)), zorder=1)
    ax.text(9.8, -1.78, r"CaTiO$_{\mathrm{3}}$ exp.", ha="right", va="center")
    ax.annotate(r"CaSnO$_{\mathrm{3}}$ exp.", (EXP_P_GPA[-1], mean[-1] - mean[0] + 0.1), xytext=(8.4, -0.12), ha="left",
                va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=NEAR_BLACK, shrinkA=1, shrinkB=1))
    ax.set_xlim(-0.8, 24.5)
    ax.set_ylim(-2.6, 0.45)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([-2, -1, 0])
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel(r"$\mathrm{\Delta}\theta_{B\mathrm{–O–}B}$ (°)" if NP_LABELS else r"$\mathrm{\Delta}\theta_{B\mathrm{-O-}B}$ (°)")


NP_LABELS = False                                   # set by fig_network: upright Mulliken symbols, en dashes


def panel_sectors(ax):
    """Share of each symmetry sector of CaTiO3 (Pnma) in the force of uniform compression and of a radial source on one
    TiO6 octahedron, and in the bond-preserving response to the radial source; B1g (shaded) is the rigid rotation."""
    d = json.loads((DATA["energy_bound"].parent / "closed_loop.json").read_text(encoding="utf-8"))["ca_tio3"]
    bs = d["breathing_sectors"]
    irreps = ["Ag", "B1g", "B2g", "B3g"]
    labels = ([r"$\mathrm{A}_{\mathrm{g}}$", r"$\mathrm{B}_{\mathrm{1g}}$", r"$\mathrm{B}_{\mathrm{2g}}$",
               r"$\mathrm{B}_{\mathrm{3g}}$"] if NP_LABELS else
              [r"$A_{\mathrm{g}}$", r"$B_{\mathrm{1g}}$", r"$B_{\mathrm{2g}}$", r"$B_{\mathrm{3g}}$"])
    comp = np.array([d["compression_f2_by_irrep"][k] for k in irreps])
    comp = 100 * comp / comp.sum()
    src = np.array([100 * bs[k]["f2_frac"] for k in irreps])
    resp = np.array([bs[k]["u2_net_frac"] * bs[k]["s_u"] for k in irreps])
    resp = 100 * resp / resp.sum()
    x = np.arange(4)
    ax.axvspan(0.5, 1.5, color=BAND, lw=0, zorder=0)
    w = 0.26
    ax.bar(x - w, comp, w, color=LIGHT_GREY, lw=0, label="force, compression", zorder=2)
    ax.bar(x, src, w, color=GREY, lw=0, label="force, radial source", zorder=2)
    ax.bar(x + w, resp, w, color=BEND, lw=0, label="response $u_b$, radial source", zorder=2)
    ax.text(1 + w, resp[1] + 1.5, "%.0f%%" % resp[1], ha="center", va="bottom", color=BEND)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0, 106)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Share (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0, pad=2)
    style.legend(ax, ax.get_legend_handles_labels()[0], loc="upper right", handlelength=0.9, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.2)


def draw():
    """2 x 2 grid: a ceiling, b amplitudes (top); c tilts, d symmetry sectors (bottom)."""
    W, H = 7.0, 4.6
    fig = plt.figure(figsize=(W, H))
    box = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])   # inches from the lower left
    row1, row2, h = 2.62, 0.38, 1.78
    axa = box(0.55, row1, 2.75, h)
    axb = box(4.30, row1, 2.62, h)
    axc = box(0.55, row2, 2.75, h)
    axd = box(3.82, row2, 3.10, h)
    panel_shares(axa)
    panel_amplitudes(axb)
    panel_tilts(axc)
    panel_sectors(axd)
    for s, x, y in (("a", 0.0, row1 + h + 0.12), ("b", 3.45, row1 + h + 0.12), ("c", 0.0, row2 + h + 0.12),
                    ("d", 3.45, row2 + h + 0.12)):
        style.letter(fig, x / W, y / H, s)
    return fig


def make(outdir):
    with style.use():
        style.save(draw(), outdir, "fig2")
