"""Fig. 4: La3Ni2O7 under pressure and in films, one competition with two read-outs (gamma band and t_perp)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Patch, Polygon
from scipy.spatial import ConvexHull

import style
from common import DATA

COL = {"0 GPa": "#8A847B", "Pressure": "#2A8C7A", "Film": "#7A5C99"}
TINT = {"0 GPa": "#E8E5DF", "Pressure": "#C7E3DC", "Film": "#DDD2E8"}
R_CMAP = LinearSegmentedColormap.from_list("sl", [(0, "#fbfbfa"), (0.5, "#e4e1dc"), (0.75, "#a09b93"), (1, "#4a4640")])
FIT = "#222222"                                   # t_perp contours (b) and fit (e)

# 3x3 lattice grid (PBE+U, U = 4 eV): rows a_p = A0..A2, columns c = C0..C2; A0C1 is the 0-GPa state
A_P = np.array([3.8522, 3.804, 3.756])
C_AXIS = np.array([19.926, 20.376, 20.826])
R_U4_MEV = np.array([[78.0, -24.1, -222.2], [40.2, -108.5, -300.9], [-14.3, -195.7, -382.5]])
T_PERP = np.array([[1.0031, 0.9745, 0.9408], [1.0620, 1.0246, 0.9657], [1.1232, 1.0766, 0.9821]])
T_PERP_0 = 0.9745
END_LATTICE = {"P2": (3.7930, 19.983), "film": (3.756, 20.720)}    # relaxed pressure end state, film (a_p, c)

# affine step and internal relaxation of the gamma-band top E_gamma (meV, U = 4 eV; pbe: total without U) and of t_perp (%)
GAMMA_SHIFT_MEV = {"Pressure": dict(affine=29.0, relaxation=10.7, pbe=-9.0),
                   "Film": dict(affine=-247.8, relaxation=-72.8, pbe=-145.6)}
T_PERP_CHANGE_PCT = {"Pressure": dict(affine=5.07, relaxation=5.95), "Film": dict(affine=-3.99, relaxation=9.39)}
# Ni-O1-Ni straightening (deg) and relaxation part of the t_perp change (%) on the grid and in the film
STRAIGHTENING = {"A0C0": (0.12, -2.4), "A0C2": (-0.53, 1.1), "A1C1": (2.77, 4.6), "A2C1": (5.55, 9.3),
                 "A1C0": (3.41, 3.2), "A1C2": (1.53, 3.7), "A2C0": (8.06, 9.0), "A2C2": (3.54, 6.3),
                 "film": (5.27, 9.4)}
STRAIGHTENING_PRESSURE = (5.42, 5.95)            # sum of the 0 -> P1 and P1 -> P2 intervals, not fitted
STRUCTURES = {"0 GPa": "la327_p0", "Pressure": "la327_p2", "Film": "la327_film"}


def minus(v):
    return f"{v:g}".replace("-", "−")


MINUS = plt.FuncFormatter(lambda v, _: minus(v))


def read_poscar(path):
    """Lattice (rows), fractional coordinates and element symbols of a VASP 5 POSCAR/CONTCAR."""
    lines = path.read_text().splitlines()
    lattice = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)])
    counts = [int(x) for x in lines[6].split()]
    symbols = [s for s, n in zip(lines[5].split(), counts) for _ in range(n)]
    frac = np.array([[float(x) for x in lines[8 + i].split()[:3]] for i in range(sum(counts))])
    return lattice, frac, symbols


def oxygen_sites(lattice, frac, symbols):
    """Ni indices and O indices by site: O1 inner apical (two Ni along c), O2 outer apical (one Ni along c),
    Op planar. Neighbours within 2.4 A, minimum image by rounding (exact for the orthorhombic cells used here)."""
    ni = [i for i, s in enumerate(symbols) if s == "Ni"]
    out = {"Ni": ni, "O1": [], "O2": [], "Op": []}
    for i, s in enumerate(symbols):
        if s != "O":
            continue
        d = frac[ni] - frac[i]
        d = (d - np.round(d)) @ lattice
        near = [v for v in d if np.linalg.norm(v) < 2.4]
        axial = [v for v in near if abs(v[2]) > 0.8 * np.linalg.norm(v)]
        key = "O1" if len(near) == 2 and len(axial) == 2 else "O2" if len(near) == 1 and len(axial) == 1 else "Op"
        out[key].append(i)
    return out


def bilayer(path):
    """One bilayer (two NiO6 sharing O1) projected on the plane of the Ni-O1-Ni bend, in A relative to O1.
    Returns the two octahedra (lower first) and the Ni-O1-Ni angle in degrees."""
    lattice, frac, symbols = read_poscar(path)
    sites = oxygen_sites(lattice, frac, symbols)
    o1 = sites["O1"][0]

    def rel(i, ref):
        d = frac[i] - frac[ref]
        return (d - np.round(d)) @ lattice

    ni = sorted(sites["Ni"], key=lambda i: np.linalg.norm(rel(i, o1)))[:2]
    octs = []
    for a in ni:
        ra = rel(a, o1)
        o2 = min(sites["O2"], key=lambda j: np.linalg.norm(rel(j, a)))
        op = sorted(sites["Op"], key=lambda j: np.linalg.norm(rel(j, a)))[:4]
        octs.append(dict(Ni=ra, O2=ra + rel(o2, a), Op=[ra + rel(j, a) for j in op]))
    mid = (octs[0]["Ni"] + octs[1]["Ni"]) / 2
    ez = np.array([0, 0, 1.0])
    side = -mid
    side[2] = 0
    ex = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else np.array([1.0, 0, 0])

    def project(v):
        return np.array([v @ ex, v @ ez])

    out = []
    for o in sorted(octs, key=lambda o: o["Ni"][2]):
        out.append(dict(Ni=project(o["Ni"]), O1=project(np.zeros(3)), O2=project(o["O2"]),
                        Op=[project(v) for v in o["Op"]], d_ni_o2=float(np.linalg.norm(o["O2"] - o["Ni"]))))
    v1, v2 = -octs[0]["Ni"], -octs[1]["Ni"]
    angle = float(np.degrees(np.arccos(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2))))
    return out, angle


def _lagrange(x, nodes):
    return np.array([np.prod([(x - nodes[m]) / (nodes[k] - nodes[m]) for m in range(3) if m != k], axis=0)
                     for k in range(3)])


def interpolate(grid, a_p, c):
    """Biquadratic interpolation through the 3x3 lattice grid."""
    return np.einsum("i...,ij,j...->...", _lagrange(a_p, A_P), grid, _lagrange(c, C_AXIS))


ORBITAL_CMAP = LinearSegmentedColormap.from_list("orb", ["#ffffff", "#dedad3", "#8f8a82", "#3a3733"])
ORBITAL_KEY = {"0 GPa": "p0", "Pressure": "p2", "Film": "film"}


def panel_bilayers(ax):
    """Bilayers on the plane of the Ni-O1-Ni bend with the charge of the interlayer d_z2 pairs at Gamma:
    left half, the gamma state (top = E_gamma); right half, its antibonding partner."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_anchor("NW")
    orb = np.load(DATA["la327_orbital"])
    uo, vo = orb["u"], orb["v"]
    vmax = max(float(orb[k].max()) for k in orb.files if k.startswith("rho"))
    left, right = (uo <= 0) & (uo >= -2.2), (uo >= 0) & (uo <= 2.2)
    vin = np.abs(vo) <= 4.5
    floor = 0.03 * vmax                          # below this the map is the white background
    for k, (name, key) in enumerate(STRUCTURES.items()):
        octs, angle = bilayer(DATA[key])
        dx = 4.2 * k
        for half, part in ((left, "gamma"), (right, "partner")):
            rho = orb[f"rho_{ORBITAL_KEY[name]}_{part}"][np.ix_(half, vin)].T
            rho = np.ma.masked_less(rho, floor)
            ax.imshow(rho, extent=(uo[half][0] + dx, uo[half][-1] + dx, vo[vin][0], vo[vin][-1]), origin="lower",
                      cmap=ORBITAL_CMAP, norm=PowerNorm(0.45, floor, 0.7 * vmax), interpolation="bilinear", zorder=0,
                      rasterized=True)
        for o in octs:
            pts = np.array([o["O1"], o["O2"]] + o["Op"]) + [dx, 0]
            hull = pts[ConvexHull(pts).vertices]
            ax.add_patch(Polygon(hull, closed=True, fc="none", ec=COL[name], lw=0.8, zorder=1))
            ni, o2 = o["Ni"] + [dx, 0], o["O2"] + [dx, 0]
            ax.plot(*zip(ni, o2), color="black", lw=1.1, solid_capstyle="round", zorder=3)
            ax.plot(*zip(ni, o["O1"] + [dx, 0]), color=COL[name], lw=0.9, zorder=3)
            for p in o["Op"]:
                ax.plot(*(p + [dx, 0]), "o", ms=3.2, mfc="white", mec="black", mew=0.6, zorder=4)
            ax.plot(*o2, "o", ms=4.2, mfc="white", mec="black", mew=0.7, zorder=4)
            ax.plot(*ni, "o", ms=5.5, mfc=COL[name], mec="black", mew=0.6, zorder=5)
            if o["O2"][1] > 0:
                mid = (ni + o2) / 2
                ax.annotate(f"{o['d_ni_o2']:.2f} Å", xy=mid, xytext=(mid[0] + 1.0, mid[1]), fontsize=7, ha="left",
                            va="center", zorder=6,
                            arrowprops=dict(arrowstyle="-", lw=0.5, color="black", shrinkA=1, shrinkB=0))
        o1 = octs[0]["O1"] + [dx, 0]
        ax.plot(*o1, "o", ms=4.2, mfc="white", mec="black", mew=0.7, zorder=4)
        v = octs[1]["Ni"] - octs[0]["O1"]
        th = np.degrees(np.arctan2(v[1], v[0]))
        ax.add_patch(Arc(o1, 1.3, 1.3, theta1=th, theta2=th + angle, color="black", lw=0.6, zorder=4))
        ax.text(o1[0] - 0.78, o1[1], f"{angle:.0f}°", fontsize=7, va="center", ha="right")
        ax.text(dx, -4.55, "8 GPa" if name == "Pressure" else name.lower() if name == "Film" else name, fontsize=7, ha="center", va="top",
                color="black")
        if k == 0:
            top_bond = (octs[1]["Ni"] + octs[1]["O2"]) / 2 + [dx, 0]
            ax.annotate("Ni–O2", xy=top_bond, xytext=(-3.9, top_bond[1]), fontsize=7, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="black", shrinkA=1, shrinkB=2))
            ax.annotate("Ni–O1–Ni", xy=(octs[0]["Ni"] + o1)[:] / 2 * 0 + (o1 + octs[0]["Ni"]) / 2, xytext=(-3.9, (o1[1] + octs[0]["Ni"][1]) / 2), fontsize=7, ha="left",
                        va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color="black", shrinkA=1, shrinkB=1))
    ax.set_xlim(-4.0, 10.6)
    ax.set_ylim(-4.95, 4.6)


def panel_drive_plane(fig, ax, cax):
    a_p0, c0 = A_P[0], C_AXIS[1]

    def compression(a_p):
        return 100 * (1 - a_p / a_p0)

    def c_strain(c):
        return 100 * (c / c0 - 1)

    x = np.linspace(0, 2.55, 300)
    z = np.linspace(-2.3, 2.6, 300)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    aa, cc = a_p0 * (1 - xx / 100), c0 * (1 + zz / 100)
    r_map = interpolate(R_U4_MEV, aa, cc)
    t_map = 100 * (interpolate(T_PERP, aa, cc) / T_PERP_0 - 1)
    im = ax.imshow(r_map.T, origin="lower", extent=(x[0], x[-1], z[0], z[-1]), aspect="auto", cmap=R_CMAP,
                   norm=TwoSlopeNorm(vmin=-400, vcenter=0, vmax=150), interpolation="bilinear")
    thin = ax.contour(xx, zz, r_map, levels=[-300, -200, -100], colors="black", alpha=0.22, linewidths=0.4,
                      linestyles="solid")
    ax.clabel(thin, fmt=lambda v: minus(v), fontsize=6, inline_spacing=2, manual=[(1.3, 2.1), (0.3, 1.8), (0.3, 0.7)])
    for t in thin.labelTexts:
        t.set_color("#6b6b6b")
    r_zero = ax.contour(xx, zz, r_map, levels=[0], colors="black", linewidths=0.9)
    ct = ax.contour(xx, zz, t_map, levels=[5, 10], colors=FIT, linewidths=0.9, linestyles="--")
    ax.clabel(ct, fmt=lambda v: f"+{v:g}%", fontsize=7, inline_spacing=3, manual=[(0.62, -1.6), (1.95, -1.3)])
    for t in ct.labelTexts:
        t.set_color(FIT)
    for name, state in (("Pressure", "P2"), ("Film", "film")):
        a_p, c = END_LATTICE[state]
        ax.annotate("", xy=(compression(a_p), c_strain(c)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.28", color=COL[name], lw=1.8,
                                    shrinkA=0, shrinkB=0))
    ax.plot(0, 0, "o", ms=4.5, mfc="white", mec="black", mew=0.8, zorder=5)
    # total change of E_gamma at the arrow heads (data/la3ni2o7/fig_la327_unified_data.csv, panel c totals)
    for name, state, (dx, dy, ha) in (("Pressure", "P2", (-0.12, -0.20, "right")), ("Film", "film", (-0.12, 0.30, "right"))):
        a_p, c = END_LATTICE[state]
        tot = sum(GAMMA_SHIFT_MEV[name][k] for k in ("affine", "relaxation"))
        ax.text(compression(a_p) + dx, c_strain(c) + dy, f"$\mathrm{{\Delta}}E_\gamma$ = {tot:+.0f} meV".replace("-", "−"), fontsize=7, ha=ha,
                va="center", color=COL[name], bbox=style.HALO, zorder=6)
    px, pz = compression(END_LATTICE["P2"][0]), c_strain(END_LATTICE["P2"][1])
    ax.text(0.55, 0.55 * pz / px + 0.08, "pressure", fontsize=7, ha="center", va="bottom",
            rotation=np.degrees(np.arctan2(pz, px)), rotation_mode="anchor", transform_rotates_text=True)
    ax.text(1.55, 1.27, "film", fontsize=7, rotation=16, ha="center", va="center")
    seg = max(r_zero.allsegs[0], key=len)
    i = int(np.argmin(abs(seg[:, 0] - 2.05)))
    x0, y0 = seg[i]
    slope = (seg[i + 3, 1] - seg[i - 3, 1]) / (seg[i + 3, 0] - seg[i - 3, 0])
    ax.text(x0, y0 + 0.12, "$E_\\gamma$ = 0", fontsize=7, ha="center", va="bottom", rotation=np.degrees(np.arctan(slope)),
            rotation_mode="anchor", transform_rotates_text=True)
    ax.text(0.05, -1.15, "$\\gamma$ above $E_\\mathrm{F}$", fontsize=7)
    ax.text(0.05, 2.3, "$\\gamma$ below $E_\\mathrm{F}$", fontsize=7, va="center")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(z[0], z[-1])
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.yaxis.set_major_formatter(MINUS)
    ax.set_xlabel("In-plane compression (%)", labelpad=1)
    ax.set_ylabel("$c$-axis strain (%)", labelpad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    cb = fig.colorbar(im, cax=cax, ticks=[-400, -200, 0, 100], format=MINUS)
    cb.outline.set_linewidth(0.5)
    cax.tick_params(length=2, labelsize=7, pad=1)
    cax.set_title("$E_\\gamma$ (meV)", fontsize=7, pad=4)


def waterfall(ax, data, ylim, yticks, fmt, pbe=None):
    """Affine step (dark), internal relaxation (light) and total (dashed) for pressure and film."""
    ax.set_ylim(*ylim)
    span = ylim[1] - ylim[0]
    for i, (name, d) in enumerate(data.items()):
        x0 = i * 1.25
        a, n = d["affine"], d["relaxation"]
        ax.bar(x0, a, 0.42, color=COL[name], linewidth=0)
        ax.bar(x0 + 0.5, n, 0.42, bottom=a, color=TINT[name], linewidth=0)
        ax.plot([x0 + 0.21, x0 + 0.29], [a, a], color="black", lw=0.5)
        total = a + n
        ax.plot([x0 - 0.21, x0 + 0.71], [total] * 2, color="black", lw=0.8, ls=(0, (2, 1)))
        up = total > 0
        ax.text(x0 + 0.25, total + (0.03 if up else -0.03) * span, fmt(total), ha="center",
                va="bottom" if up else "top", fontsize=7)
        if pbe is not None:
            ax.plot(x0 + 0.25, pbe[name], "D", ms=3.5, mfc="white", mec="black", mew=0.7, zorder=4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks([0.25, 1.5], [n.lower() for n in data])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.35, 2.05)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(MINUS)


def panel_straightening(ax):
    xs = np.array([v[0] for v in STRAIGHTENING.values()])
    ys = np.array([v[1] for v in STRAIGHTENING.values()])
    slope, intercept = np.polyfit(xs, ys, 1)
    r = np.corrcoef(xs, ys)[0, 1]
    xl = np.linspace(-1, 8.6, 50)
    ax.plot(xl, slope * xl + intercept, color=FIT, lw=1.0, zorder=2)
    for name, (x, y) in STRAIGHTENING.items():
        film = name == "film"
        if film:                                  # hollow ring on top: the grid point at the same place stays visible
            ax.plot(x, y, "o", ms=8, mfc="none", mec=COL["Film"], mew=1.3, zorder=5)
        else:
            ax.plot(x, y, "o", ms=4, mfc="white", mec="black", mew=0.7, zorder=4)
    ax.plot(*STRAIGHTENING_PRESSURE, "o", ms=5, mfc=COL["Pressure"], mec=COL["Pressure"], zorder=4)
    ax.axhline(0, color="black", lw=0.5, ls=(0, (1, 1.5)))
    ax.legend(handles=[Line2D([], [], marker="o", ms=4, mfc="white", mec="black", ls="none", label="grid points"),
                       Line2D([], [], marker="o", ms=5, mfc=COL["Pressure"], mec=COL["Pressure"], ls="none",
                              label="pressure"),
                       Line2D([], [], marker="o", ms=7, mfc="none", mec=COL["Film"], mew=1.3, ls="none", label="film"),
                       Line2D([], [], color=FIT, lw=1.0, label=f"fit ($r$ = {r:.2f})")],
              loc="upper left", frameon=False, fontsize=7, handlelength=1.4, handletextpad=0.4, borderaxespad=0.2,
              labelspacing=0.3)
    ax.set_xlim(-1, 8.6)
    ax.set_ylim(-4, 14)
    ax.set_yticks([0, 5, 10])
    ax.set_xticks([0, 4, 8])
    ax.yaxis.set_major_formatter(MINUS)
    ax.xaxis.set_major_formatter(MINUS)
    ax.set_xlabel("Relaxation straightening (°)", labelpad=1)
    ax.set_ylabel("Relaxation $\\mathrm{\\Delta}t_\\perp$ (%)", labelpad=1)


def draw(np_variant=False):
    fig = plt.figure(figsize=(7.0, 4.9))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.75, 1], hspace=0.34, left=0.06, right=0.955, bottom=0.085,
                             top=0.93)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.5, 1, 0.035], wspace=0.14)
    bot = outer[1].subgridspec(1, 3, wspace=0.42)
    axa, axb, cax = (fig.add_subplot(top[i]) for i in range(3))
    axc, axd, axe = (fig.add_subplot(bot[i]) for i in range(3))

    panel_bilayers(axa)
    panel_drive_plane(fig, axb, cax)
    waterfall(axc, GAMMA_SHIFT_MEV, (-385, 85), [-300, -150, 0], lambda v: ("+" if v > 0 else "") + minus(round(v)),
              pbe=None if np_variant else {n: d["pbe"] for n, d in GAMMA_SHIFT_MEV.items()})
    axc.set_ylabel("$\\mathrm{\\Delta}E_\\gamma$ (meV)", labelpad=1)
    axc.legend(handles=[Patch(fc="#555555", label="affine part" if np_variant else "affine step"), Patch(fc="#cfcfcf", label="relaxation"),
                        Line2D([], [], color="black", lw=0.8, ls=(0, (2, 1)), label="total")]
               + ([] if np_variant else [Line2D([], [], marker="D", ms=3.5, mfc="white", mec="black", ls="none",
                                                label=r"without $U_{\mathrm{eff}}$")]),
               loc="lower left", frameon=False, fontsize=7, handlelength=1.2, handletextpad=0.4, borderaxespad=0.2,
               labelspacing=0.3)
    waterfall(axd, T_PERP_CHANGE_PCT, (-6, 14), [-5, 0, 5, 10], lambda v: f"+{v:.1f}%".replace("+11.0%", "+11%"))
    axd.set_ylabel("$\\mathrm{\\Delta}t_\\perp$ (%)", labelpad=1)
    axc.text(0.98, 0.97, "Ni–O2", transform=axc.transAxes, ha="right", va="top", fontsize=7)
    axd.text(0.98, 0.97, "Ni–O1–Ni", transform=axd.transAxes, ha="right", va="top", fontsize=7)
    panel_straightening(axe)

    fig.canvas.draw()                             # final axes positions (panel a has a fixed aspect)
    pos = {k: ax.get_position() for k, ax in zip("abcde", (axa, axb, axc, axd, axe))}
    x_left = pos["c"].x0 - 0.05
    y_top = max(pos["a"].y1, pos["b"].y1) + 0.012
    y_bot = max(pos[q].y1 for q in "cde") + 0.012
    for lab in "abcde":
        x = x_left if lab in "ac" else pos[lab].x0 - 0.05
        style.letter(fig, x, y_top if lab in "ab" else y_bot, lab)
    return fig


def make(outdir):
    """Fig. 3: panel c without the no-U_eff diamonds."""
    with style.use():
        style.save(draw(np_variant=True), outdir, "fig4")
