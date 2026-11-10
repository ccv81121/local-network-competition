"""Fig. 1: local-network competition as a closed loop, lattice competition (a-d) and electronic competition (e-h)."""
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon

import style
from common import arrow, read_json
from fig2 import CORNER_PRESSURE, EDGE_PRESSURE, SUBSTITUTION
from style import BEND, CORNER, GREY, LIGHT_GREY, LOCAL, NET, NEAR_BLACK

INK = "black"
TEAL, GOLD = "#2C64A4", "#8A847B"        # the two cations with different local preferences
NAVY, RUST = "#263746", "#C2553A"
LIGAND, LIGAND_BACK = "#DCE5EA", "#CDD7DE"
FS = style.TEXT_PT

# one grid for all eight panels: four equal column boxes, two rows, one equation line per row
COLX = (0.005, 0.257, 0.509, 0.761)               # no row titles: rows are named in the caption
BOXX = tuple(c + 0.03 for c in COLX)
BOXW = 0.19
FIG_H = 3.44                                   # inches; layout below is set in inches
BOXH = 0.943 / FIG_H
BOXY = (2.20 / FIG_H, 0.60 / FIG_H)
EQC = (1.97 / FIG_H, 0.29 / FIG_H)
# bending in Mg(Sc1-xSbx)2Se4, x = 0.75: share (%) of the force, the displacement and the band-edge change
SPINEL_SHARES = (("force", 0.54), ("displacement", 32.9), ("band\nedge", 90.0))


def _octahedron_projection(view=(0.55, 0.45, 0.70)):
    """Orthographic projection of a regular octahedron seen along `view`: vertices [+x, -x, +y, -y, +z, -z] in the
    screen plane (+x ligand on the horizontal axis) and their depths (positive = towards the viewer)."""
    v = np.asarray(view, dtype=float)
    v = v / np.linalg.norm(v)
    ex = np.array([1.0, 0.0, 0.0])
    u1 = ex - ex.dot(v) * v
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(v, u1)
    verts = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    return np.column_stack([verts @ u1, verts @ u2]), verts @ v


def _mix(color, other, weight):
    a, b = np.array(mcolors.to_rgb(color)), np.array(mcolors.to_rgb(other))
    return tuple((1.0 - weight) * a + weight * b)


def octahedron(ax, cx, cy, scale, center_color, face_color, angle=0.0, shared_side=None, sx=1.0, sy=1.0,
               center_offset=(0.0, 0.0), face_alpha=0.20):
    """Octahedron with depth cues: rear faces faint, rear bonds tinted, front faces, silhouette, front bonds and atoms
    on top. `sx`/`sy` stretch the preferred geometry; with `shared_side` the bridging ligand sits at the origin."""
    proj, depth = _octahedron_projection()
    if shared_side == "left":
        proj = proj * np.array([-1.0, 1.0])
    base = proj * 0.60 * np.array([sx, sy])
    t = np.deg2rad(angle)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    vertices = base @ rot.T * scale + np.array([cx, cy])
    if shared_side in ("right", "left"):
        vertices[0] = (0.0, 0.0)
    ccx = cx + center_offset[0] * scale
    ccy = cy + center_offset[1] * scale
    front_vertex = depth > 0

    faces = [(i, j, k) for i in (0, 1) for j in (2, 3) for k in (4, 5)]
    for face in [f for f in faces if depth[list(f)].sum() < 0]:
        ax.add_patch(Polygon(vertices[list(face)], closed=True, facecolor=mcolors.to_rgba(face_color, 0.16),
                             edgecolor="none", zorder=2))
    rear_bond = _mix(center_color, "white", 0.55)
    for idx in np.flatnonzero(~front_vertex):
        vx, vy = vertices[idx]
        ax.plot([ccx, vx], [ccy, vy], color=rear_bond, lw=1.2, solid_capstyle="round", zorder=2.5)
    for n, face in enumerate([f for f in faces if depth[list(f)].sum() > 0]):
        ax.add_patch(Polygon(vertices[list(face)], closed=True, facecolor=mcolors.to_rgba(face_color, face_alpha),
                             edgecolor=mcolors.to_rgba(NAVY, 0.9), linewidth=0.5, zorder=3 + n * 0.02))
    order = np.argsort(np.arctan2(*(proj[:, ::-1]).T))
    ax.add_patch(Polygon(vertices[order], closed=True, facecolor="none", edgecolor=NAVY, linewidth=0.7, zorder=3.5))
    for idx in np.flatnonzero(front_vertex):
        vx, vy = vertices[idx]
        ax.plot([ccx, vx], [ccy, vy], color=center_color, lw=1.55, alpha=0.94, solid_capstyle="round", zorder=4.2)

    r = 0.070 * scale
    for idx, (x, y) in enumerate(vertices):
        front = front_vertex[idx]
        ax.add_patch(Circle((x, y), r if front else 0.93 * r, facecolor=LIGAND if front else LIGAND_BACK,
                            edgecolor=NAVY, linewidth=0.62 if front else 0.5, zorder=6 if front else 5.8))
        ax.add_patch(Ellipse((x - 0.018 * scale, y + 0.022 * scale), 0.030 * scale, 0.020 * scale, facecolor="white",
                             edgecolor="none", alpha=0.75 if front else 0.45, zorder=6.2 if front else 5.9))
    ax.add_patch(Circle((ccx, ccy), 0.115 * scale, facecolor=center_color, edgecolor="none", zorder=7))
    ax.add_patch(Ellipse((ccx - 0.033 * scale, ccy + 0.040 * scale), 0.070 * scale, 0.045 * scale, facecolor="white",
                         edgecolor="none", alpha=0.28, zorder=7.2))


def octahedron_pair(parent_ax, bounds, shared):
    """Two octahedra with different local preferences, isolated or sharing one ligand (with the opposing drives)."""
    ax = parent_ax.inset_axes(bounds)
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-0.72, 0.72)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if not shared:
        octahedron(ax, -0.52, 0.0, 0.86, TEAL, "#D6E1EF", angle=-3, sx=0.88, sy=1.12)
        octahedron(ax, 0.52, 0.0, 0.86, GOLD, "#ECE8E1", angle=3, sx=1.12, sy=0.88, center_offset=(0.06, 0.04))
        return
    octahedron(ax, -0.50, 0.0, 0.92, TEAL, "#D6E1EF", angle=-2, shared_side="right", sx=0.92, sy=1.08)
    octahedron(ax, 0.50, 0.0, 0.92, GOLD, "#ECE8E1", angle=2, shared_side="left", sx=1.08, sy=0.92,
               center_offset=(0.05, 0.03))
    ax.add_patch(Circle((0, 0), 0.072, facecolor=LIGAND, edgecolor=NAVY, linewidth=0.65, zorder=8))
    ax.add_patch(Circle((0, 0), 0.125, facecolor="none", edgecolor=RUST, linewidth=0.7, alpha=0.65, zorder=8.5))
    for p0, p1, c in (((-0.05, 0.11), (-0.13, 0.33), TEAL), ((0.05, -0.11), (0.13, -0.33), GOLD)):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6, color=c, lw=1.0, shrinkA=0,
                                     shrinkB=0, zorder=9))


def plane(fig, rect, xmin=-0.42, b_right=False):
    """b-r displacement plane; r (bond-changing) in grey; b_right puts the b label right of its axis."""
    ax = fig.add_axes(rect)
    ax.set_xlim(xmin, 1.12)
    ax.set_ylim(-0.12, 0.80)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_anchor("N")
    arrow(ax, (0, 0), (1.08, 0), LOCAL, lw=0.7, ms=5)
    arrow(ax, (0, 0), (0, 0.76), BEND, lw=0.7, ms=5)
    ax.text(1.09, -0.05, r"$\mathbf{r}$", ha="right", va="top")
    ax.text(0.04 if b_right else -0.04, 0.76, r"$\mathbf{b}$", ha="left" if b_right else "right", va="top")
    return ax


def spines_off(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def box(i, row, dx=0.0, dw=0.0):
    return [BOXX[i] + dx, BOXY[row], BOXW + dw, BOXH]


def equation(fig, i, row, s):
    fig.text(BOXX[i] + BOXW / 2, EQC[row], s, ha="center", va="center")


def link(fig, p0, p1, label, label_pos):
    fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transSubfigure, arrowstyle="-|>", mutation_scale=7, color=INK,
                                   lw=0.7, shrinkA=0, shrinkB=0))
    fig.text(*label_pos, label, ha="center", va="center", fontsize=FS)


def header(fig, rect):
    """Field-theory band: source -> filter -> read-out (no panel letter)."""
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    paper = "#F2F0EB"
    specs = (("Source", r"preferred geometry $q^{\mathrm{0}}$ and force $f$", "pressure, strain: uniform, totally symmetric",
              "substitution: local, every symmetry sector", paper, "#8A847B", 0.005, 0.335),
             ("Filter", "compatibility of shared ligands", r"zero modes $\mathbf{b}$ of the bond network",
              r"local mode $k$ passed as $\mathrm{1}/(\mathrm{1}+\ell_k)$", _mix(NET, "white", 0.88), NET, 0.375, 0.31),
             ("Read-out", r"property gradient $g$,", r"bond-preserving share $\phi$", "", _mix(BEND, "white", 0.88), BEND,
              0.72, 0.275))
    for k, (title, l1, l2, l3, face, edge, x0, w) in enumerate(specs):
        ax.add_patch(FancyBboxPatch((x0, 0.0), w, 1.0, boxstyle="round,pad=0,rounding_size=0.02", fc=face, ec=edge,
                                    lw=0.7))
        lines = [t for t in (l1, l2, l3) if t]
        ax.text(x0 + 0.012, 0.9, title, fontsize=7, fontweight="bold", ha="left", va="top")
        for i, t in enumerate(lines):
            ax.text(x0 + 0.012, 0.65 - 0.215 * i, t, fontsize=6.5, ha="left", va="top")
        if k < 2:
            x1 = x0 + w
            arrow(ax, (x1 + 0.003, 0.5), (x1 + 0.03, 0.5), INK, lw=0.9, ms=6)


def draw(np_variant=False):
    top = plt.figure(figsize=(7.05, FIG_H))
    fig = top

    # a: local preferences meet at a shared ligand
    axa = fig.add_axes(box(0, 0, dx=-0.03, dw=0.03))
    axa.set_axis_off()
    axa.set_xlim(0, 1)
    axa.set_ylim(0, 1)
    octahedron_pair(axa, (0.00, 0.20, 0.46, 0.66), shared=False)
    octahedron_pair(axa, (0.54, 0.20, 0.46, 0.66), shared=True)
    arrow(axa, (0.465, 0.53), (0.535, 0.53), INK, lw=0.7, ms=5)
    axa.text(0.23, 0.09, r"$q^{\mathrm{0}},\ K$", ha="center")
    axa.text(0.77, 0.09, r"network $L$", ha="center")
    equation(fig, 0, 0, r"$H=K+L$" if np_variant else r"$w_k^{*}=\dfrac{w_k^{\mathrm{0}}}{\mathrm{1}+\ell_k}$")

    # b: quasi-static path
    axb = fig.add_axes(box(1, 0))
    spines_off(axb)
    axb.set_xlim(-0.08, 1.12)
    axb.set_ylim(-0.12, 1.05)
    axb.set_xlabel("lattice change", labelpad=2)
    axb.set_ylabel("internal\ncoordinates", labelpad=2, linespacing=1.0)
    p0, pa, pr = np.array([0.05, 0.10]), np.array([0.58, 0.10]), np.array([0.58, 0.80])
    axb.plot([p0[0], pr[0]], [p0[1], pr[1]], color=LIGHT_GREY, lw=0.8, ls=(0, (2.5, 1.5)), zorder=1)
    arrow(axb, p0, pa, INK, lw=1.1, ms=6)
    arrow(axb, pa, pr, INK, lw=1.1, ms=6, ls=(0, (2.5, 1.5)))
    arrow(axb, pa, pa + np.array([0.13, 0.42]), INK, lw=0.9, ms=5)
    for p, c in ((p0, "white"), (pa, INK), (pr, INK)):
        axb.plot(*p, "o", ms=3.4, mfc=c, mec=INK, mew=0.7, zorder=5)
    axb.text(p0[0], p0[1] - 0.06, "initial", ha="center", va="top")
    axb.text(pa[0], pa[1] - 0.06, "start", ha="center", va="top")
    axb.text(pr[0] + 0.05, pr[1], "relaxed", ha="left", va="center")
    axb.text(pa[0] + 0.15, pa[1] + 0.42, r"$f$", ha="left", va="center")
    axb.text(pa[0] - 0.04, 0.45, r"$u$", ha="right", va="center")
    equation(fig, 1, 0, r"$\mathrm{\Delta}P=\mathrm{\Delta}P_{\mathrm{aff}}+\mathrm{\Delta}P_{\mathrm{rel}}$")

    # c: limit, enrichment bounded by 1/s_f
    b0 = box(2, 0)
    axc = fig.add_axes([b0[0] + 0.03, b0[1] + 0.06, b0[2] - 0.03, b0[3] - 0.06])   # room for the tick labels
    spines_off(axc)
    s_f = np.logspace(-5, 0, 50)
    axc.fill_between(s_f, 1 / s_f, 3e5, color=LIGHT_GREY, alpha=0.35, lw=0, zorder=1)
    axc.plot(s_f, 1 / s_f, color=INK, lw=1.0, zorder=3)
    # computed structures of Fig. 2a: (s_f, s_u) in %, enrichment = s_u / s_f
    if np_variant:       # same scheme as Fig. 2a (panel_shares), larger markers
        c_styles = ((EDGE_PRESSURE, dict(marker="o", ms=4.2, mfc=BEND, mec="white", mew=0.4)),
                    (CORNER_PRESSURE, dict(marker="s", ms=3.6, mfc=CORNER, mec="white", mew=0.4)))
    else:
        c_styles = ((EDGE_PRESSURE, dict(marker="o", ms=3.2, mfc=BEND, mec="white", mew=0.3)),
                    (CORNER_PRESSURE, dict(marker="s", ms=2.8, mfc="white", mec=CORNER, mew=0.7)))
    for pts, kw in c_styles:
        for sf, su in pts:
            axc.plot(sf / 100, su / sf, ls="", zorder=4, **kw)
    for sf, su, topo in SUBSTITUTION:
        edge = topo == "edge"
        axc.plot(sf / 100, su / sf, ls="", marker="D", ms=4.6 if np_variant else 3.0, mfc=BEND if edge else "white",
                 mec=BEND if edge else CORNER, mew=0.9 if np_variant else 0.7, zorder=4)
    axc.set_xscale("log")
    axc.set_yscale("log")
    axc.set_xlim(1e-5, 1)
    axc.set_ylim(0.8, 3e5)
    axc.minorticks_off()
    axc.set_xticks([1e-4, 1e-2, 1])
    axc.set_xticklabels(["0.01", "1", "100"])               # s_f in %, as in Fig. 2a
    axc.set_yticks([1, 10, 100, 1000])
    axc.set_yticklabels(["1", "10", "100", "1000"])
    axc.tick_params(labelsize=6.5, length=2.0, pad=2.0)
    axc.set_xlabel(r"force fraction $s_f$ (%)", labelpad=2)
    axc.set_ylabel(r"enrichment $\epsilon$", labelpad=2)
    axc.annotate("278×", (0.0009, 24.8 / 0.09), xytext=(-3, 0), textcoords="offset points", ha="right", va="center")
    axc.text(1.5e-3, 6e4, "unreachable", ha="center", va="center")
    (x1, x2) = (3e-5, 3e-4)
    p1, p2 = axc.transData.transform([(x1, 1 / x1), (x2, 1 / x2)])
    ang = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    axc.annotate("ceiling", (10 ** -4.2, 10 ** 4.2), xytext=(0, -2), textcoords="offset points", rotation=ang,
                 rotation_mode="anchor", ha="center", va="top")
    equation(fig, 2, 0, r"$\epsilon=\dfrac{s_u}{s_f}\leq\dfrac{\mathrm{1}}{s_f}$" if np_variant else
             r"$\epsilon=\dfrac{s_u}{s_f}\ \leq\ \dfrac{\mathrm{1}}{s_f}$")

    # d: distance, force balance with direct and coupled terms
    # x = 0.75, 4-5 GPa (data/bending_amplitude/amplitude.json, data/four_state_leverage/prepare_report.json):
    # |u_b|/|u_r| = 0.6767 (full), 0.2747 (direct, H_bb^-1 f_b alone); |f_b|/|f| = sqrt(drive share 0.00544)
    amp = [r for r in read_json("bending_amplitude") if r["system"] == "MgSc0.5Sb1.5Se4 k12"][0]
    ratio_full, ratio_direct = amp["amplitude_ratio_up_over_uq"], amp["block_direct_up_per_uq"]
    fb = np.sqrt(read_json("leverage_prepare")["k12_t45"]["drive_share"])
    f = 0.95 * np.array([np.sqrt(1 - fb ** 2), fb])
    u = 0.80 * np.array([1.0, ratio_full])
    axd = plane(fig, box(3, 0), b_right=np_variant)
    arrow(axd, (0, 0), f, INK, lw=1.2, ms=6)
    arrow(axd, (0, 0), u, INK, lw=1.2, ms=6, ls=(0, (2.5, 1.5)))
    axd.text(f[0], f[1] + 0.05, r"$f$", ha="center", va="bottom")
    axd.text(u[0] - 0.02, u[1] + 0.06, r"$u=H^{-\mathrm{1}}f$", ha="center", va="bottom")
    bd, bf, bw = 0.80 * ratio_direct, u[1], 0.07
    xdir, xfull = -0.33, -0.16                     # direct = open, full = filled (as in Fig. 2b)
    axd.add_patch(plt.Rectangle((xdir - bw / 2, 0), bw, bd, facecolor="white", edgecolor=BEND, lw=0.8, zorder=3))
    axd.add_patch(plt.Rectangle((xfull - bw / 2, 0), bw, bf, facecolor=BEND, edgecolor=BEND, lw=0.8, zorder=3))
    axd.plot([0, u[0]], [u[1], u[1]], color=LIGHT_GREY, lw=0.5, ls=(0, (1.5, 1.5)), zorder=1)
    axd.text(xdir, bd + 0.02, f"{ratio_direct:.2f}", ha="center", va="bottom")
    axd.text(xfull, bf + 0.02, f"{ratio_full:.2f}", ha="center", va="bottom")
    equation(fig, 3, 0, r"$u_b=H_{bb}^{-\mathrm{1}}f_b-H_{bb}^{-\mathrm{1}}H_{br}u_r$")

    # e: two states weigh onsite (local) and hopping (network) terms differently
    axe = fig.add_axes(box(0, 1, dx=-0.03, dw=0.03))
    axe.set_axis_off()
    axe.set_xlim(0, 1)
    axe.set_ylim(0, 0.84)
    for x0, name, yl, (d_local, d_network) in ((0.04, r"$\alpha$: $\Gamma$, Sc $d$", 0.40, (0.36, 0.09)),
                                               (0.54, r"$\beta$: R, Sb $p$–Se $s$", 0.56, (0.08, 0.36))):
        axe.plot([x0, x0 + 0.34], [yl, yl], color=INK, lw=1.5, solid_capstyle="butt")
        axe.text(x0 + 0.17, 0.0, name, ha="center", va="bottom")
        arrow(axe, (x0 + 0.08, yl), (x0 + 0.08, yl + d_local), LOCAL, lw=1.3, ms=6)
        arrow(axe, (x0 + 0.26, yl), (x0 + 0.26, yl - d_network), NET, lw=1.3, ms=6)
    axe.text(0.04 + 0.12, 0.40 + 0.30, "on-site\n(local)", ha="left", va="center", linespacing=1.0)
    axe.text(0.54 + 0.30, 0.56 - 0.30, "hopping\n(network)", ha="left", va="center", linespacing=1.0)
    equation(fig, 0, 1, r"$\mathrm{\Lambda}_e=\zeta_e(E_{e{,}\alpha}-E_{e{,}\beta})$" if np_variant else
             r"$\dfrac{d\mathrm{\Lambda}_e}{d\lambda}=W_e\,(JR_\lambda+c_\lambda)$")

    # f: the margin is the small difference of a local and a network part
    w = read_json("wpart_projected")["Lambda_C"]      # projected Wannier basis, 5 GPa, x = 0.75 (meV)
    vals = [w["local_L"], w["network_N"], w["AX_link"], w["Lambda_meV"]]
    axf = fig.add_axes(box(1, 1, dx=0.03, dw=-0.05))
    if not np_variant:
        axf.set_yscale("symlog", linthresh=10, linscale=0.5)
    axf.set_axis_off()
    if np_variant:
        panel_f_linear(fig, axf, vals)
    else:
        axf.bar(range(4), vals, 0.62, color=[LOCAL, NET, GREY, INK], lw=0)
        axf.axhline(0, color=INK, lw=0.6)
        axf.set_xlim(-0.6, 3.6)
        axf.set_ylim(*((-1400, 1400) if np_variant else (-9000, 9000)))
        names = ("local", "network", r"$A$–$X$" if np_variant else "A–X", r"$\mathrm{\Lambda}_{\mathrm{C}}$")
        if np_variant:                                  # linear axis: the margin is a sliver, labelled with a leader
            names = names[:3] + (None,)
            axf.plot([3, 3], [vals[3] + 30, -330], color=INK, lw=0.5, zorder=1)
            axf.text(3.55, -480, r"$\mathrm{\Lambda}_{\mathrm{C}}$ = " + f"{vals[3]:.1f} meV", ha="right", va="center",
                     fontsize=6.0)
        for i, (lab, v) in enumerate(zip(names, vals)):
            if lab is None:
                continue
            axf.annotate(lab, (i, v), xytext=(0, -2 if v < 0 else 2), textcoords="offset points",
                         ha="center", va="top" if v < 0 else "bottom", fontsize=6.0)
    equation(fig, 1, 1, r"$\mathrm{\Lambda}_{\mathrm{C}}=\mathrm{L}+\mathrm{N}+A\mathrm{–}X$" if np_variant else
             r"$\mathrm{\Lambda}_e=\zeta_e\,(E_{e,\alpha}-E_{e,\beta})$")

    # g: worth, two states read the same displacement differently
    axg = plane(fig, box(2, 1), xmin=-0.14)
    # x = 0.75, 4-5 GPa conduction margin: g_b, g_r = predicted b- and network-channel change over |u_b|, |u_r|
    # (data/three_channel/tc_result.json, data/four_state_leverage/prepare_report.json); direction to scale
    tc, prep = read_json("three_channel")["k12_t45"], read_json("leverage_prepare")["k12_t45"]
    g_b = tc["pred_meV"]["b"] / prep["ub_norm_A"] / 1e3                  # eV/A
    g_r = tc["pred_meV"]["n"] / prep["uc_network_norm_A"] / 1e3          # eV/A
    g = 0.78 * np.array([g_r, g_b]) / np.hypot(g_r, g_b)                 # direction (r, b), drawn at length 0.78
    ratio_g = ratio_full                                                 # |u_b|/|u_r| = 0.68 (panel d)
    u_g = 0.80 * np.array([1.0, ratio_g])
    arrow(axg, (0, 0), u_g, INK, lw=1.2, ms=6, ls=(0, (2.5, 1.5)))
    arrow(axg, (0, 0), g, LOCAL, lw=1.1, ms=6)
    axg.text(u_g[0] + 0.03, u_g[1] - 0.02, r"$u$", ha="left", va="top")
    axg.text(g[0] + 0.07, g[1] - 0.03, r"$g$", ha="left", va="top")
    equation(fig, 2, 1, (r"$\dfrac{\phi}{\mathrm{1}-\phi}=\dfrac{g_b}{g_r}\,\dfrac{|u_b|}{|u_r|}=\ell$" if np_variant else r"$\dfrac{\phi}{\mathrm{1}-\phi}=\dfrac{g_b}{g_r}\,\dfrac{|u_b|}{|u_r|}$"))

    # h: the loop on data
    axh = fig.add_axes(box(3, 1, dx=0.03, dw=-0.03))
    shares = [v for _, v in SPINEL_SHARES]
    cols = [LOCAL, BEND, INK]
    xs = [0.0, 1.35, 2.7]                                              # spaced so the 6-pt names do not touch
    axh.bar(xs, shares, 0.7, color=cols, lw=0)
    for x, v, c in zip(xs, shares, cols):                              # top edge at the true value, visible for 0.54
        axh.plot([x - 0.35, x + 0.35], [v, v], color=c, lw=1.4, solid_capstyle="butt", zorder=3)
    axh.set_ylim(0, 100)
    axh.set_xlim(-0.6, 3.3)
    axh.set_xticks([])
    axh.set_yticks([0, 50, 100])
    axh.tick_params(length=2, pad=1, labelsize=FS, colors=NEAR_BLACK)
    axh.set_ylabel("Share (%)", labelpad=1)
    for x, (lab, v) in zip(xs, SPINEL_SHARES):
        axh.text(x, v + 2, f"{v:g}", ha="center", va="bottom")
        axh.text(x, -4, lab, ha="center", va="top", linespacing=1.0, fontsize=6.0)
    equation(fig, 3, 1, r"$\mathrm{\Delta}\mathrm{\Lambda}_{e{,}\mathrm{rel}}=g^{\mathsf{T}}u$" if np_variant else
             r"$\mathrm{\Delta}\mathrm{\Lambda}_e=g^{\mathsf{T}}u$")

    # loop links: u from d into g between the rows; W_e, g and dLambda_e between the bottom panels
    xd, xg = BOXX[3] + BOXW / 2, BOXX[2] + BOXW / 2
    ys = EQC[0] - 0.12 / FIG_H
    yg = ys - 0.10 / FIG_H
    fig.add_artist(Line2D([xd, xd, xg], [ys, yg, yg], transform=fig.transSubfigure, color=INK, lw=0.7))
    fig.add_artist(FancyArrowPatch((xg, yg), (xg, BOXY[1] + BOXH + 0.01), transform=fig.transSubfigure,
                                   arrowstyle="-|>", mutation_scale=7, color=INK, lw=0.7, shrinkA=0, shrinkB=0))
    fig.text((xd + xg) / 2, yg + 0.008, r"$u$", ha="center", va="bottom")
    yt = BOXY[0] + 0.62 * BOXH
    for i, lab in ((1, r"$f$"), (2, r"$s_f$")):
        x1 = BOXX[i + 1] + (-0.027 if i == 1 else (-0.014 if np_variant else 0.012))   # NP: s_f arrow clear of 0.27
        x0 = BOXX[i] + BOXW + 0.004
        link(fig, (x0, yt), (x1, yt), lab, ((x0 + x1) / 2, yt + 0.03))
    ym = BOXY[1] + BOXH / 2
    dlam = r"$\mathrm{\Delta}\mathrm{\Lambda}_{e{,}\mathrm{rel}}$" if np_variant else r"$\mathrm{\Delta}\mathrm{\Lambda}_e$"
    for i, lab in ((0, r"$W_e$"), (1, r"$g$"), (2, dlam)):
        x0, x1 = BOXX[i] + BOXW + 0.004, BOXX[i + 1] + (0.012 if i < 2 else -0.035)
        link(fig, (x0, ym), (x1, ym), lab, ((x0 + x1) / 2, ym + 0.03))

    # each letter at the top-left of its own panel (content extent incl. axis labels), one baseline per row
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    bbs = [inv.transform_bbox(ax.get_tightbbox(fig.canvas.get_renderer())) for ax in (axa, axb, axc, axd, axe, axf, axg, axh)]
    rows = (max(b.y1 for b in bbs[:4]), max(b.y1 for b in bbs[4:]))
    for k, (lab, bb) in enumerate(zip("abcdefgh", bbs)):
        style.letter(fig, max(bb.x0 - 0.012, 0.0), rows[k // 4] + 0.004, lab)
    return top


def make(outdir):
    """Fig. 1 (H = K + L, enlarged markers in c, linear bars in f)."""
    with style.use():
        style.save(draw(np_variant=True), outdir, "fig1")


def panel_f_linear(fig, axf, vals):
    """Fig. 1f (NP): local (L) and network (N) parts on an eV axis, and L+N, the A-X part and the margin on a meV axis."""
    from matplotlib.lines import Line2D as _L2
    fmt = lambda v, d=0: (f"{v:+.{d}f}").replace("-", "−")
    # two small bar charts, each with its own meV axis: the large local and network parts, and the small sum,
    # the A-X part and the margin on an expanded scale (no magnification factor to decode)
    pos = axf.get_position()
    ln = vals[0] + vals[1]

    def style(ax, ticks, names, xpos, unit):
        ax.set_axis_on()
        ax.axhline(0, color=INK, lw=0.6)
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")))
        ax.tick_params(axis="y", labelsize=5.6, length=2, width=0.5, pad=1)
        ax.set_xticks([])
        ax.text(-0.02, 1.0, unit, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6)

    axf.set_position([pos.x0 + pos.width * 0.06, pos.y0 + pos.height * 0.12, pos.width * 0.40, pos.height * 0.76])
    axf.bar([0, 1.5], [v / 1000 for v in vals[:2]], 0.55, color=[LOCAL, NET], lw=0)   # eV
    axf.set_xlim(-0.7, 2.2)
    axf.set_ylim(-1.35, 1.35)
    style(axf, [-1, 0, 1], None, None, "eV")
    for x, v, name in ((0, vals[0] / 1000, "L"), (1.5, vals[1] / 1000, "N")):   # name at the end of its bar
        axf.text(x, v + (0.04 if v >= 0 else -0.04), name, ha="center", va="bottom" if v >= 0 else "top", fontsize=5.8)
    axi = fig.add_axes([pos.x0 + pos.width * 0.64, pos.y0 + pos.height * 0.12, pos.width * 0.50, pos.height * 0.76])
    axi.bar([0, 1.3, 2.6], [ln, vals[2], vals[3]], 0.5, color=[_mix(LOCAL, NET, 0.5), GREY, INK], lw=0)
    axi.set_xlim(-0.75, 3.1)
    axi.set_ylim(-27, 27)                         # same headroom over the ticks as the left axes (1.35x), zeros aligned
    style(axi, [-20, -10, 0, 10, 20], None, None, "meV")
    names = ("L+N", r"$A$–$X$", r"$\mathrm{\Lambda}_{\mathrm{C}}$")
    # only the margin carries a value (it matches the text); the parts are read from the axis, so rounded labels
    # never fail to add up and no label crowds another
    for x, v, name in zip((0, 1.3, 2.6), (ln, vals[2], vals[3]), names):
        if name.startswith(r"$\mathrm{\Lambda}"):
            axi.text(x, v + 0.8, fmt(v, 1), ha="center", va="bottom", fontsize=5.6)
            axi.text(x, v + 6.6, name, ha="center", va="bottom", fontsize=5.8)
        elif v >= 0:
            axi.text(x, v + 0.8, name, ha="center", va="bottom", fontsize=5.8)
        else:
            axi.text(x, v - 0.8, name, ha="center", va="top", fontsize=5.8)
