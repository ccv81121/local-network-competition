"""Fig. 3: electronic readout, channel scaling (a), local/network split of the conduction margin (b), four-state
construction (c), geometric and electronic factors (d), bond-preserving share of property changes (e)."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

import style
from common import DATA, read_json
from style import BEND, CORNER, DARK_GREY, GREY, HALO, LIGHT_GREY, NEAR_BLACK, NET

BAND = "#E9EEF3"
EDGE_SYSTEMS = {"k12_t45", "k12_05", "k08", "Layered_NaScInO2"}
SYSTEM_LABELS = {"k12_t45": "4–5", "k12_05": "0–5", "k08": "$x$ = 0.5",
                 "Layered_NaScInO2": r"Na(Sc,In)O$_{\mathrm{2}}$",
                 "CaTiO3": r"CaTiO$_{\mathrm{3}}$", "CaSnO3": r"CaSnO$_{\mathrm{3}}$",
                 "Ca2SnTiO6_rs": r"Ca$_{\mathrm{2}}$SnTiO$_{\mathrm{6}}$",
                 "Ca8Sn6Ti2O24": r"Ca$_{\mathrm{8}}$Sn$_{\mathrm{6}}$Ti$_{\mathrm{2}}$O$_{\mathrm{24}}$"}
LABEL_OFFSETS = {"k12_t45": (7, 8, "left"), "k12_05": (-7, -6, "right"), "k08": (6, 0, "left"),
                 "Layered_NaScInO2": (5, 0, "left"), "CaTiO3": (-6, 0, "right"), "CaSnO3": (8, 0, "left"),
                 "Ca2SnTiO6_rs": (-6, 0, "right"), "Ca8Sn6Ti2O24": (-6, 0, "right")}
LAMBDA_C_CA8_MEV = {"00": 73.77, "10": 75.06, "01": 70.75, "11": 72.06}   # four-state margins of Ca8Sn6Ti2O24
PHI_K08 = -0.0149                                                          # +0.114 / -7.661 meV
PHI_STRAIN = 0.1286
PHI_S5_S6, PHI_S11_S12 = -0.2546, 0.3458

Y_TOP = 0.645 + 0.275 + 0.02                      # letter line above the top row
Y_LOW = 0.095 + 0.38 + 0.012                       # letter line above the bottom row
X_LEFT = 0.075 - 0.07


def phi_order_averaged(lam):
    """Signed share of the bending (10) increment, averaged over the two addition orders."""
    return 0.5 * ((lam["10"] - lam["00"]) + (lam["11"] - lam["01"])) / (lam["11"] - lam["00"])


def phi_compression(row):
    """Order-averaged bond-preserving share from the four-state increments (b and c from state 00)."""
    b, c, t = row["dLambda_b_actual_meV"], row["dLambda_c_actual_meV"], row["dLambda_total_actual_meV"]
    return 0.5 * (b + (t - c)) / t


def panel_channels(ax, legend_anchor=(X_LEFT + 0.035, Y_TOP - 0.012)):
    """+-25% scaling of X-X hopping and X onsite terms in 24 Wannier Hamiltonians, response of the two margins."""
    w = pd.read_csv(DATA["wannier_endpoint_audit"])
    w["V"], w["C"] = w.chi_V_gamma_favored * 1e3, w.chi_C_gamma_favored * 1e3
    hopping, onsite = w[w.channel == "XX_network"], w[w.channel == "X_onsite"]
    n_hopping, n_onsite = int(hopping.opposite_sign.sum()), int(onsite.opposite_sign.sum())
    ax.axhline(0, color="black", lw=0.5, zorder=1)
    ax.axvline(0, color="black", lw=0.5, zorder=1)
    is_spinel = lambda d: d.family.isin(["spinel", "pressure"])   # pressure tasks are the x = 0.75 spinel
    for sel, marker, size in ((is_spinel, "o", 24), (lambda d: ~is_spinel(d), "s", 22)):
        o, h = onsite[sel(onsite)], hopping[sel(hopping)]
        ax.scatter(o.V, o.C, s=size, marker=marker, facecolor="none", edgecolor="black", lw=0.8, zorder=5)
        ax.scatter(h.V, h.C, s=size, marker=marker, color=NET, edgecolor="white", lw=0.4, zorder=4)
    ticks, labels = [-100, -10, -1, 0, 1, 10, 100], ["−100", "−10", "", "0", "", "10", "100"]
    x_labels = ["−100", "", "", "0", "", "", "100"]   # the x axis is too short for the +-10 labels
    for set_scale, set_lim, set_ticks, set_labels, labs in (
            (ax.set_xscale, ax.set_xlim, ax.set_xticks, ax.set_xticklabels, x_labels),
            (ax.set_yscale, ax.set_ylim, ax.set_yticks, ax.set_yticklabels, labels)):
        set_scale("symlog", linthresh=1)
        set_lim(-300, 300)
        set_ticks(ticks)
        set_labels(labs)
    ax.minorticks_off()
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\delta\mathrm{\Lambda}_{\mathrm{V}}$ (meV)")
    ax.set_ylabel(r"$\delta\mathrm{\Lambda}_{\mathrm{C}}$ (meV)")
    ax.text(0.04, 0.96, "opposite", transform=ax.transAxes, ha="left", va="top")
    ax.text(0.96, 0.04, "opposite", transform=ax.transAxes, ha="right", va="bottom")
    h = [Line2D([], [], marker="o", ls="", ms=4.5, color=NET, label=f"$X$–$X$ hopping ({n_hopping}/{len(hopping)})"),
         Line2D([], [], marker="o", ls="", ms=4.5, mfc="white", mec="black",
                label=f"$X$ on-site ({n_onsite}/{len(onsite)})"),
         Line2D([], [], marker="o", ls="", ms=4.5, mfc=GREY, mec="white", mew=0.4, label="spinel"),
         Line2D([], [], marker="s", ls="", ms=4.2, mfc=GREY, mec="white", mew=0.4, label="perovskite")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=legend_anchor,
              bbox_transform=ax.figure.transFigure, ncol=2, handletextpad=0.25, borderaxespad=0, columnspacing=1.5, labelspacing=0.4)


def panel_split(ax):
    """Local/network split of the 5-GPa conduction margin in two Wannier bases."""
    for k, key in enumerate(("wpart_projected", "wpart_mlwf")):
        v = read_json(key)["Lambda_C"]
        local_part, network_part, lambda_c_mev = v["local_L"], v["network_N"], v["Lambda_meV"]
        assert abs(local_part + network_part + v["A_link"] - lambda_c_mev) < 1e-6
        y = 1.25 * k
        ax.barh(y, local_part / 1e3, 0.45, color=GREY, lw=0)
        ax.barh(y, network_part / 1e3, 0.45, color=NET, lw=0)
        a_link = v["A_link"] / 1e3                  # A-X link: positive stacks beyond the network end, negative runs back
        if a_link >= 0:
            ax.barh(y, a_link, 0.45, left=network_part / 1e3, color=LIGHT_GREY, lw=0)
        else:
            ax.barh(y, a_link, 0.45, left=network_part / 1e3, facecolor="none", edgecolor=DARK_GREY, hatch="//////",
                    lw=0, zorder=3)
        ax.plot([lambda_c_mev / 1e3] * 2, [y - 0.30, y + 0.30], color=NEAR_BLACK, lw=1.0, zorder=5)   # margin
        ax.text(0.05, y + 0.28, f"{lambda_c_mev:.1f} meV", va="top", ha="left")
    ax.axvline(0, color=NEAR_BLACK, lw=0.5)
    ax.set_yticks([0, 1.25])
    ax.set_yticklabels(["projected", "max.\nlocalized"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-1.1, 1.1)
    ax.set_xticks([-1, 0, 1])
    ax.set_ylim(1.95, -0.75)
    ax.text(-0.5, -0.42, "local", ha="center")
    ax.text(0.5, -0.42, "network", ha="center")
    ax.set_xlabel(r"Part of $\mathrm{\Lambda}_{\mathrm{C}}$ (eV)")
    ax.spines["left"].set_visible(False)


def panel_four_states(fig, ax, insets=None):
    """Four-state construction with the force-balance prediction, and the two competing conduction states."""
    s = read_json("tight_summary")
    four, stars = s["four_state_k6"], s["force_balance_stars_k6"]
    names = ["state00", "state10", "state01", "state11"]
    energy = np.array([four["E_minus_E00_meV_per_cell"][n] for n in names])
    lambda_c = np.array([four["Lambda_C_meV"][n] for n in names])
    ax.axhline(0, color=LIGHT_GREY, lw=0.6, zorder=0)
    for a, b in ((0, 1), (2, 3)):
        ax.annotate("", xy=(energy[b], lambda_c[b]), xytext=(energy[a], lambda_c[a]),
                    arrowprops=dict(arrowstyle="-|>", color=BEND, lw=1.2, shrinkA=4, shrinkB=4 if b == 1 else 10, mutation_scale=8))   # 01 -> 11 stops short of the star
    for i, n in enumerate(names):
        lab = n[-2:]
        ax.scatter(energy[i], lambda_c[i], s=22, color=BEND if lab[0] == "1" else DARK_GREY, edgecolor="white", lw=0.5,
                   zorder=5)
        off = {"00": (7, 2), "10": (6, 0), "01": (-6, 0), "11": (7, 0)}[lab]
        ax.annotate(lab, (energy[i], lambda_c[i]), xytext=off, textcoords="offset points",
                    ha="center" if off[0] == 0 else ("left" if off[0] > 0 else "right"), va="center")
    affine, predicted = stars["r_aff"], stars["prediction_r_aff_plus_Hpf"]
    # star heights on the smearing of the four states (leverage_evaluation: r_aff margin; prediction = r_aff + predicted shift)
    ev = read_json("leverage_evaluation")["rows"]["k12_t45"]
    xa, ya = affine["E_minus_E00_meV"], ev["Lambda_raff_meV"]
    xp, yp = predicted["E_minus_E00_meV"], ev["Lambda_raff_meV"] + ev["dLambda_total_pred_meV"]
    # dashed shaft and a solid head (a dash pattern on the head outline breaks it)
    ax.annotate("", xy=(xp, yp), xytext=(xa, ya), arrowprops=dict(arrowstyle="-", color=DARK_GREY, lw=0.7,
                                                                  ls=(0, (2, 1.6)), shrinkA=4, shrinkB=8))
    xh, yh = xa + 0.9 * (xp - xa), ya + 0.9 * (yp - ya)
    ax.annotate("", xy=(xp, yp), xytext=(xh, yh), arrowprops=dict(arrowstyle="-|>", color=DARK_GREY, lw=0.7,
                                                                  shrinkA=0, shrinkB=5, mutation_scale=7))
    ax.scatter([xa], [ya], s=40, marker="*", facecolor="white", edgecolor=DARK_GREY, lw=0.7, zorder=4)
    ax.scatter([xp], [yp], s=110, marker="*", facecolor="none", edgecolor=NEAR_BLACK, lw=0.8, zorder=6)
    ax.annotate(r"$H^{-1}f$", (xp, yp), xytext=(0, 9), textcoords="offset points", ha="center", va="bottom")
    ax.text(0.05, 3.3, "bending", ha="left", va="center", color=BEND)
    ax.set_xlim(-8.6, 2.4)
    ax.set_ylim(-3.0, 9.6)
    ax.set_xticks([-8, -4, 0])
    ax.set_yticks([0, 4, 8])
    ax.set_xlabel(r"$E-E_{\mathrm{00}}$ (meV/cell)")
    ax.set_ylabel(r"$\mathrm{\Lambda}_{\mathrm{C}}$ (meV)")
    for k, (key, title) in enumerate((("wavefunction_gamma", r"$\mathrm{\Gamma}$ valley, Sc $d$"),
                                      ("wavefunction_r", r"R valley, Sb $p$–Se $s$"))):
        im = Image.open(DATA[key]).convert("RGBA")
        im = im.crop(im.getbbox())
        hgt = 0.15
        wid = hgt * (4.5 / 7.0) * im.width / im.height
        cx, y_first, step, title_y = 0.915, 0.765, 0.20, 0.925
        if insets is not None:                   # (centre x, first y, step, height, title y) in figure fractions
            cx, y_first, step, hgt, title_y = insets
            w_in, h_in = fig.get_size_inches()
            wid = hgt * (h_in / w_in) * im.width / im.height
        ia = fig.add_axes([cx - wid / 2, y_first - k * step, wid, hgt])
        ia.imshow(np.asarray(im))
        ia.set_axis_off()
        fig.text(cx, title_y - k * step, title, ha="center", va="bottom")


def _phase_marker(r):
    if r["control"] == "substitution":
        return "D"
    return "o" if r["connectivity"] == "edge" else "s"


def _phase_color(r):
    return BEND if r["connectivity"] == "edge" else CORNER


def four_state_property_factors():
    """Ratio of the four-state to the finite-difference property factor, for the two x = 0.75 steps."""
    prepare = read_json("leverage_prepare")
    three_channel = {**read_json("three_channel"), **read_json("three_channel_all")}
    out = {}
    for k in ("k12_t45", "k12_05"):
        r = three_channel[k]
        u_b, u_r = prepare[k]["ub_norm_A"], prepare[k]["uc_network_norm_A"]
        out[k] = ((r["meas_meV"]["b"] / u_b) / (r["meas_meV"]["n"] / u_r)) / ((r["pred_meV"]["b"] / u_b) /
                                                                            (r["pred_meV"]["n"] / u_r))
    return out


TC_KEYS = {"k12_t45": "k12_t45", "k12_05": "k12_05", "k08": "k08", "ca_tio3": "CaTiO3", "ca_sno3": "CaSnO3",
           "layered_na_sc_ino2": "Layered_NaScInO2", "ca2_sn_tio6_rs": "Ca2SnTiO6_rs", "ca8_sn6_ti2o24": "Ca8Sn6Ti2O24"}


def network_factors():
    """|u_b|/|u_r| with u_r the network complement of H^-1 f, and g_b/g_r from the predicted channel changes."""
    prepare = read_json("leverage_prepare")
    three_channel = {**read_json("three_channel"), **read_json("three_channel_all")}
    out = {}
    for k, t in TC_KEYS.items():
        u_b, u_r = prepare[t]["ub_norm_A"], prepare[t]["uc_network_norm_A"]
        r = three_channel[t]
        out[k] = (u_b / u_r, (r["pred_meV"]["b"] / u_b) / (r["pred_meV"]["n"] / u_r))
    return out


def panel_phase(ax, phase, ell=False):
    """Share phase diagram: geometric against property factor; dashed lines, constant share (odds 0.1, 1, 10)."""
    gx = np.logspace(-1.8, 0.2, 50)
    for odds in (0.1, 1.0, 10.0):
        ax.plot(gx, odds / gx, color=GREY, lw=0.8, ls=(0, (3, 1.6)) if odds != 1 else "-", zorder=1)
    ax.fill_between(gx, 1 / gx, 1e3, color=LIGHT_GREY, alpha=0.45, lw=0, zorder=0)
    phase = {k: dict(r) for k, r in phase.items()}
    for k, r in network_factors().items():          # compressions: same definition as Fig. 2b and Supp. Fig. 3
        phase[k]["geometric"], phase[k]["property"] = r
    for k, r in phase.items():
        same, col = r["property"] > 0, _phase_color(r)
        ax.plot(r["geometric"], abs(r["property"]), _phase_marker(r), ms=4.2, color=col,
                mfc=col if same else "white", mec=col, mew=1.0, zorder=3)
    for k, ratio in four_state_property_factors().items():
        ax.plot(phase[k]["geometric"], abs(phase[k]["property"] * ratio), "x", ms=3.6, color=NEAR_BLACK, mew=0.6, zorder=4)
    lab_off = {"k12_t45": (6, 9), "k12_05": (6, -9), "k08": (5, 0), "layered_na_sc_ino2": (5, 0), "ca_tio3": (-5, 0),
               "ca_sno3": (-5, 0), "ca2_sn_tio6_rs": (-5, 0), "ca8_sn6_ti2o24": (-6, 0), "i0506": (6, 0),
               "i1112": (-5, 0)}
    for k, r in phase.items():
        dx, dy = lab_off[k]
        lab = (r["label"].replace("x=", "$x$ = ").replace("->", r"$\rightarrow$").replace("S5", "$S_5$")
               .replace("S6", "$S_6$").replace("S11", "$S_{11}$").replace("S12", "$S_{12}$")
               .replace("Ca2SnTiO6", r"Ca$_{\mathrm{2}}$SnTiO$_{\mathrm{6}}$")
               .replace("Ca8Sn6Ti2O24", r"Ca$_{\mathrm{8}}$Sn$_{\mathrm{6}}$Ti$_{\mathrm{2}}$O$_{\mathrm{24}}$")
               .replace("CaTiO3", r"CaTiO$_{\mathrm{3}}$").replace("CaSnO3", r"CaSnO$_{\mathrm{3}}$")
               .replace("O2", r"O$_{\mathrm{2}}$")
               .replace("4-5", "4–5").replace("0-5", "0–5"))
        if k in ("k12_t45", "k12_05"):                  # two lines, so the labels stay clear of the bound lines
            lab = lab.replace(", ", "\n")
        ax.annotate(lab, (r["geometric"], abs(r["property"])), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", va="center", fontsize=6.0, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10 ** -1.8, 10 ** 0.2)
    ax.set_ylim(1e-3, 60)
    ax.set_xticks([0.1, 1])
    ax.set_xticklabels(["0.1", "1"])
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1, 10])
    ax.set_yticklabels(["0.001", "0.01", "0.1", "1", "10"])
    if ell:                                             # NP: the lines are contours of the leverage
        for odds, x0 in ((1.0, 0.03), (10.0, 0.25), (0.1, 0.09)):
            p1 = ax.transData.transform((x0, odds / x0))
            p2 = ax.transData.transform((1.5 * x0, odds / (1.5 * x0)))
            ang = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
            ax.text(x0, 1.3 * odds / x0, rf"$|\ell|={odds:g}$", rotation=ang, rotation_mode="anchor",
                    ha="left", va="bottom", fontsize=6.0, color=GREY, zorder=2)
    ax.set_xlabel(r"Geometric factor $|u_b|/|u_r|$")
    ax.set_ylabel(r"Property factor $|g_b/g_r|$")
    hs = [Line2D([], [], ls="none", marker="o", ms=4.2, color=BEND, label="edge sharing"),
          Line2D([], [], ls="none", marker="s", ms=4.2, color=CORNER, label="corner sharing"),
          Line2D([], [], ls="none", marker="D", ms=4.2, color=BEND, label="substitution"),
          Line2D([], [], ls="none", marker="o", ms=4.2, color=DARK_GREY, mfc="white", label="opposite signs"),
          Line2D([], [], ls="none", marker="x", ms=3.6, color=NEAR_BLACK, mew=0.6, label="four-state")]
    style.legend(ax, hs, loc="lower left", handletextpad=0.3, borderaxespad=0.3, labelspacing=0.25)


def panel_bound(ax, bound):
    """Energy share against its Cauchy-Schwarz bound for six compressed structures."""
    lim = 25.0
    ax.plot([0, lim], [0, lim], color=GREY, lw=0.8, ls=(0, (3, 1.6)), zorder=1)
    ax.fill_between([0, lim], [0, lim], [lim, lim], color=LIGHT_GREY, alpha=0.45, lw=0, zorder=0)
    for c, r in bound.items():
        edge = c in ("k12_t45", "k08")
        ax.plot(100 * r["bound"], 100 * r["e_b"], "o" if edge else "s", ms=4.2, color=BEND if edge else CORNER,
                zorder=3)
    for c, lab in (("k12_t45", "$x$ = 0.75"), ("k08", "$x$ = 0.5")):
        r = bound[c]
        ax.annotate(lab, (100 * r["bound"], 100 * r["e_b"]), xytext=(5, 0), textcoords="offset points", ha="left",
                    va="center")
    ax.text(0.04, 0.96, "excluded", transform=ax.transAxes, ha="left", va="top", color=DARK_GREY)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xticks([0, 10, 20])
    ax.set_xlabel(r"Bound $\sqrt{s_f\ \ s_u}\,/\kappa$ (%)")
    ax.set_ylabel(r"Energy share $e_b$ (%)")


def panel_bending_share(ax):
    """Signed bond-preserving share phi of property changes across controls and properties, sorted."""
    evaluation = read_json("leverage_evaluation")["rows"]
    polarization = read_json("polarization")["tot"]
    EDGE_M, CORNER_M, SUB_M = ("o", BEND), ("s", CORNER), ("D", BEND)       # as in panel d
    rows = [(r"$x$ = 0.75", phi_compression(evaluation["k12_t45"]), *EDGE_M),
            (r"$x$ = 0.5", PHI_K08, *EDGE_M),
            (r"Na(Sc,In)O$_{\mathrm{2}}$", phi_compression(evaluation["Layered_NaScInO2"]), *EDGE_M),
            (r"CaTiO$_{\mathrm{3}}$", phi_compression(evaluation["CaTiO3"]), *CORNER_M),
            (r"CaSnO$_{\mathrm{3}}$", phi_compression(evaluation["CaSnO3"]), *CORNER_M),
            (r"Ca$_{\mathrm{2}}$SnTiO$_{\mathrm{6}}$", phi_compression(evaluation["Ca2SnTiO6_rs"]), *CORNER_M),
            (r"Ca$_{\mathrm{8}}$Sn$_{\mathrm{6}}$Ti$_{\mathrm{2}}$O$_{\mathrm{24}}$",
             phi_order_averaged(LAMBDA_C_CA8_MEV), *CORNER_M),
            (r"$S_{\mathrm{11}}$ strain", PHI_STRAIN, "p", BEND),
            (r"$S_{\mathrm{5}}\rightarrow S_{\mathrm{6}}$", PHI_S5_S6, *SUB_M),
            (r"$S_{\mathrm{11}}\rightarrow S_{\mathrm{12}}$", PHI_S11_S12, *SUB_M),
            (r"polarization", polarization["phi"], "h", "black")]
    rows.sort(key=lambda r: -r[1])
    ax.axvspan(0, 100, color=BAND, lw=0, zorder=0)
    ax.axvline(0, color="black", lw=0.6, zorder=1)
    for i, (_, phi, m, col) in enumerate(rows):                  # filled: bending and rest push the same way (0 < phi < 1); open: opposite
        ax.plot(100 * phi, i, m, ms=4.4 if m in "sD" else 5.0, mfc=col if 0 < phi < 1 else "white", mec=col, mew=1.0,
                zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(-85, 130)
    ax.set_xticks([-50, 0, 50, 100])
    ax.set_xticklabels(["−50", "0", "50", "100"])
    ax.set_xlabel(r"Bond-preserving share $\phi$ (%)")


def draw():
    """Fig. 4: four panels, a channels, b four-state construction, c geometric and electronic factors,
    d bond-preserving share (the two-basis split and the energy-bound panels are left out)."""
    d = json.loads(DATA["energy_bound"].read_text(encoding="utf-8"))
    W, H = 7.0, 4.85
    fig = plt.figure(figsize=(W, H))
    box = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])    # inches from the lower left
    h_top, y_top, h_low, y_low = 1.65, 2.85, 1.75, 0.50
    axa = box(0.62, y_top, 1.65, h_top)
    axb = box(3.30, y_top, 2.30, h_top)
    axc = box(0.62, y_low, 3.10, h_low)
    axd = box(4.95, y_low, 2.00, h_low)
    panel_channels(axa, legend_anchor=(0.60 / W, (y_top + h_top + 0.04) / H))
    # valley isosurfaces right of b: centre x, first (upper) y, step, height, title y of the first, figure fractions
    ins_h = 0.75
    panel_four_states(fig, axb, insets=(6.30 / W, (y_top + 1.05) / H, 1.05 / H, ins_h / H,
                                         (y_top + 1.05 + ins_h + 0.02) / H))
    panel_phase(axc, d["phase"], ell=True)
    panel_bending_share(axd)
    ytop_l, ylow_l = (y_top + h_top + 0.10) / H, (y_low + h_low + 0.10) / H
    for lab, x, y in (("a", 0.0, ytop_l), ("b", 2.75, ytop_l), ("c", 0.0, ylow_l), ("d", 4.05, ylow_l)):
        style.letter(fig, x / W, y, lab)
    return fig


def make(outdir):
    with style.use():
        style.save(draw(), outdir, "fig3")
