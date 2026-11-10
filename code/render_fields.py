"""Real-space renders of the bond-preserving response: CaTiO3 layer under a radial source on one Ti (rigid rotations
about b) and the shared-edge hinge of x = 0.75 under compression."""
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import FancyArrowPatch

from common import _D
from render import Scene, view_matrix
from style import BEND, CORNER, DARK_GREY, GREY, LIGHT_GREY, NET

FIELDS = _D / "field_sectors" / "response_fields.npz"
SPINEL = _D / "four_state_leverage" / "calculations" / "k12_t45"


def _read_poscar(path):
    t = path.read_text().split("\n"); s = float(t[1]); L = s * np.array([[float(x) for x in t[i].split()] for i in (2, 3, 4)])
    el, n = t[5].split(), [int(x) for x in t[6].split()]
    fr = np.array([[float(x) for x in t[8 + i].split()[:3]] for i in range(sum(n))])
    return L, fr, np.repeat(el, n)


def _rotation_about(axis, ti, ligands, field):
    """Least-squares rigid rotation (rad) of one octahedron about axis, from the displacements of its ligands."""
    A = np.vstack([np.array([[0, d[2], -d[1]], [-d[2], 0, d[0]], [d[1], -d[0], 0]]) for _, d in ligands])
    y = np.concatenate([field[o] - field[ti] for o, _ in ligands])
    w = np.linalg.lstsq(-A, y, rcond=None)[0]
    return float(w @ axis)


def catio3_layer(ax, part="u_b", scale=70.0, cells=((0, 0), (1, 0), (0, 1), (1, 1)), arcs=True):
    """One TiO6 layer of CaTiO3 seen along b; arrows: the chosen part of the response to a radial source on the
    red Ti of every cell; arcs: sense of the rigid rotation about b."""
    z = np.load(FIELDS); L, fr, sp, bonds, j0 = z["lattice"], z["frac"], z["species"], z["bonds"], int(z["driven"])
    field = z[part]
    oct_ = {}
    for i, k, *img in bonds:
        for c, o, s in ((i, k, 1), (k, i, -1)):
            if sp[c] == "Ti":
                oct_.setdefault(c, []).append((o, (fr[o] - fr[c] + s * np.array(img)) @ L))
    layer = [t for t in oct_ if abs(fr[t][1] - fr[j0][1]) < 0.1]
    bax = L[1] / np.linalg.norm(L[1])
    sc = Scene(view_matrix((0.08, -1.0, 0.12), up=(0, 0, 1)))
    col = BEND if part == "u_b" else NET
    rings = []
    for ca, cc in cells:
        t = np.array([ca, 0, cc]) @ L
        for ti in layer:
            cen = fr[ti] @ L + t; pts = [cen + d for _, d in oct_[ti]]
            drv = ti == j0
            sc.polyhedron(pts, "#F3DDD6" if drv else "#E6EAEF", "#9A6B5F" if drv else "#7D8590", alpha=0.6, lw=0.3)
            sc.atom(cen, 0.26, "#B5533C" if drv else "#5B6470", lw=0.3)
            for (o, _), p in zip(oct_[ti], pts):
                sc.atom(p, 0.15, "white", lw=0.35)
                sc.arrow(p, scale * field[o], col, lw=0.8, head=2.2, min_len=0.12)
            if arcs:
                rings.append((sc.p(cen)[:2], _rotation_about(bax, ti, oct_[ti], field)))
    sc.draw(ax)
    for xy, w in rings:
        if abs(w) < 1e-4:
            continue
        a0, a1 = (200, 340) if w > 0 else (340, 200)
        th = np.radians(np.linspace(a0, a1, 30)); r = 0.62
        pts = np.c_[xy[0] + r * np.cos(th), xy[1] + r * np.sin(th)]
        ax.add_patch(FancyArrowPatch(pts[-3], pts[-1], arrowstyle="-|>,head_length=0.18,head_width=0.10",
                                     color=DARK_GREY, lw=0.6, mutation_scale=10, zorder=4))
        ax.plot(pts[:-1, 0], pts[:-1, 1], color=DARK_GREY, lw=0.6, zorder=4)


def spinel_hinge(ax, scale=90.0, pair=(14, 20), cutoff=3.2):
    L, fr, sp = _read_poscar(SPINEL / "raff" / "POSCAR"); ub = np.load(SPINEL / "u_b.npy").reshape(-1, 3)
    c0 = fr[pair[0]]
    def near(i):
        d = fr - fr[i]; d -= np.round(d); r = np.linalg.norm(d @ L, axis=1)
        return [(j, (fr[i] + d[j]) @ L) for j in np.flatnonzero((r < cutoff) & (r > 0.1)) if sp[j] == "Se"]
    base = {pair[0]: (c0 - c0) @ L}
    d = fr[pair[1]] - c0; d -= np.round(d); base[pair[1]] = d @ L
    sh = (fr[pair[0]]) @ L
    ligs = {}
    for i in pair:
        ligs[i] = [(j, r - fr[i] @ L + base[i]) for j, r in near(i)]
    shared = set(j for j, _ in ligs[pair[0]]) & set(j for j, _ in ligs[pair[1]])
    axis = base[pair[1]] / np.linalg.norm(base[pair[1]])
    edge = np.mean([r for j, r in ligs[pair[0]] if j in shared], axis=0)
    sc = Scene(view_matrix(np.cross(axis, edge - base[pair[1]] / 2) + 0.25 * axis, up=edge - base[pair[1]] / 2))
    for i, colr in zip(pair, ("#E4E8EE", "#E4E8EE")):
        sc.polyhedron([r for _, r in ligs[i]], colr, DARK_GREY, alpha=0.55, lw=0.35)
        sc.atom(base[i], 0.34, CORNER, lw=0.35); sc.arrow(base[i], scale * ub[i], BEND, lw=1.0, head=3.0)
        for j, r in ligs[i]:
            sc.atom(r, 0.24, "#F4E3B2" if j in shared else "white", lw=0.4)
            sc.arrow(r, scale * ub[j], BEND, lw=1.0, head=3.0)
    sc.draw(ax)
    return sp[list(pair)]


SUBSTITUTION = _D / "field_sectors" / "dispersion" / "real_field.npz"


def substitution_layer(ax, layers, window=5.7):
    """The TiO6 layer through the Sn site of the 80-atom CaTiO3 Ti->Sn cell, seen along b, centred on Sn.
    layers: list of (field, scale, color, lw, head); each per-atom vector field (force f in eV/A or displacement in A)
    is drawn times scale, in the given order (later on top)."""
    z = np.load(SUBSTITUTION)
    L, fr, sp, sn = z["lattice"], z["frac"], z["species"], int(z["sn"])
    B = [i for i, s in enumerate(sp) if s in ("Ti", "Sn")]
    O = np.array([i for i, s in enumerate(sp) if s == "O"])
    yb = L[1] / np.linalg.norm(L[1])
    sc = Scene(view_matrix((0.06, -1.0, 0.10), up=(0, 0, 1)))
    sites = {}
    imgs = [np.array([a, b, c]) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    for j in B:
        for im in imgs:
            r = (fr[j] - fr[sn] + im - np.round(fr[j] - fr[sn])) @ L
            if abs(r @ yb) > 1.0 or abs(r[0]) > window or abs(r[2]) > window:
                continue
            d = fr[O] - fr[j]; d -= np.round(d); dc = d @ L
            near = np.argsort(np.linalg.norm(dc, axis=1))[:6]
            pts = [r + dc[k] for k in near]
            is_sn = sp[j] == "Sn"
            sc.polyhedron(pts, "#F6E3DC" if is_sn else "#F0F2F5", "#C08A7C" if is_sn else "#A9B0B9", alpha=0.75, lw=0.3)
            sc.atom(r, 0.24, "#B5533C" if is_sn else "#8C939C", lw=0.25)
            sites[(j, tuple(np.round(r, 2)))] = (j, r)
            for k, p in zip(near, pts):
                sc.atom(p, 0.13, "white", lw=0.3)
                sites[(int(O[k]), tuple(np.round(p, 2)))] = (int(O[k]), p)
    for field, scale, color, lw, head in layers:
        vec = z[field]
        for j, r in sites.values():
            sc.arrow(r, scale * vec[j], color, lw=lw, head=head, min_len=0.24)
    sc.draw(ax)
    ax.set_xlim(-window - 0.6, window + 0.6)
    ax.set_ylim(-window - 0.6, window + 0.6)


BOND_MAX = 2.4        # A, Ti/Sn-O bond cutoff (3D) for the drawn framework


def layer_atoms(window=6.6):
    """Atoms of the TiO2 layer through Sn (|y| < 1 A, periodic images) in the in-plane frame (x, z) centred on Sn:
    list of (index, species, x, z) and the Ti/Sn-O bonds as pairs of in-plane points."""
    z = np.load(SUBSTITUTION)
    L, fr, sp, sn = z["lattice"], z["frac"], z["species"], int(z["sn"])
    atoms = []
    imgs = [np.array([a, 0, c]) for a in (-2, -1, 0, 1, 2) for c in (-2, -1, 0, 1, 2)]
    for j in range(len(sp)):
        if sp[j] == "Ca":
            continue
        d0 = fr[j] - fr[sn]
        d0 = d0 - np.round(d0)
        for im in imgs:
            r = (d0 + im) @ L
            if abs(r[1]) <= 1.0 and abs(r[0]) <= window and abs(r[2]) <= window:
                atoms.append((j, str(sp[j]), r))
    bonds = []
    for a, (_, sa, ra) in enumerate(atoms):
        for _, sb, rb in atoms[a + 1:]:
            if (sa == "O") != (sb == "O") and np.linalg.norm(ra - rb) < BOND_MAX:
                bonds.append(((ra[0], ra[2]), (rb[0], rb[2])))
    return atoms, bonds


def field_quiver(ax, field, color, scale, ref, ref_label, window=6.0, min_frac=0.03):
    """In-plane (x, z) vector of every layer atom as an arrow of length scale*|v| (one scale per panel), on the drawn
    Ti/Sn-O framework. Vectors below min_frac of the panel maximum are omitted. ref: reference arrow value.
    Returns the largest in-plane magnitude."""
    z = np.load(SUBSTITUTION)
    vec = z[field]
    atoms, bonds = layer_atoms(window + 0.6)
    for p, q in bonds:
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#D3D5D8", lw=0.6, solid_capstyle="round", zorder=1)
    xs, ys, us, vs = [], [], [], []
    for j, k, r in atoms:
        xs.append(r[0]); ys.append(r[2]); us.append(vec[j][0]); vs.append(vec[j][2])
    xs, ys, us, vs = map(np.array, (xs, ys, us, vs))
    mag = np.hypot(us, vs)
    vmax = float(mag.max())
    keep = ((mag >= min_frac * vmax) & (np.abs(xs) <= window) & (np.abs(ys) <= window)
            & (np.abs(xs + scale * us) <= window) & (np.abs(ys + scale * vs) <= window))   # whole arrow inside the panel
    ax.quiver(xs[keep], ys[keep], us[keep], vs[keep], angles="xy", scale_units="xy", scale=1.0 / scale, width=0.0085,
              headwidth=3.8, headlength=4.2, headaxislength=3.6, color=color, zorder=4, pivot="tail")
    for (_, k, r) in atoms:
        x, y = r[0], r[2]
        if abs(x) > window or abs(y) > window:
            continue
        if k == "Sn":
            ax.plot(x, y, "o", ms=4.6, mfc="#1F1D1B", mec="white", mew=0.6, zorder=6)
        elif k == "Ti":
            ax.plot(x, y, "o", ms=3.0, mfc="white", mec="#6E747C", mew=0.6, zorder=5)
        else:
            ax.plot(x, y, "o", ms=1.8, mfc="#9A9FA6", mec="none", zorder=5)
    x0, y0 = -window + 0.2, -window - 0.95           # reference arrow, in the row below the panel
    ax.annotate("", xy=(x0 + ref * scale, y0), xytext=(x0, y0), annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.14", color=color, lw=1.0,
                                shrinkA=0, shrinkB=0, mutation_scale=7))
    ax.text(x0 + ref * scale + 0.3, y0, ref_label, ha="left", va="center", fontsize=6.0, color=color, clip_on=False)
    ax.set_xlim(-window, window)
    ax.set_ylim(-window, window)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return vmax, int(keep.sum()), int(len(keep))


def field_continuous(ax, field, cmap, line_color, window=6.0, sigma=0.7, npix=241, gamma=0.85, density=0.7,
                     coherence=0.6, floor=0.06):
    """Continuous in-plane field on the TiO2 layer through Sn. Colour: per-atom magnitude |v| smoothed by a Gaussian
    kernel (sigma, A), in the field's units. Lines: streamlines of the kernel-averaged vector, drawn only where the
    neighbouring atoms move together (|<v>| >= coherence * <|v|>) and the magnitude exceeds floor of its maximum, so
    that no line is drawn where the kernel averages opposed vectors. Returns the image and the per-atom maximum."""
    import matplotlib.colors as mcol
    z = np.load(SUBSTITUTION)
    vec = z[field]
    atoms, bonds = layer_atoms(window + 2.0)
    pts = np.array([(r[0], r[2]) for _, _, r in atoms])
    vv = np.array([(vec[j][0], vec[j][2]) for j, _, _ in atoms])
    mag = np.hypot(vv[:, 0], vv[:, 1])
    vmax = float(mag[(np.abs(pts[:, 0]) <= window) & (np.abs(pts[:, 1]) <= window)].max())
    g = np.linspace(-window, window, npix)
    X, Y = np.meshgrid(g, g)
    w = np.exp(-((X[..., None] - pts[:, 0]) ** 2 + (Y[..., None] - pts[:, 1]) ** 2) / (2 * sigma ** 2))
    ws = w.sum(-1) + 1e-12
    U, V = (w * vv[:, 0]).sum(-1) / ws, (w * vv[:, 1]).sum(-1) / ws
    M = (w * mag).sum(-1) / ws
    im = ax.imshow(M, extent=(-window, window, -window, window), origin="lower", cmap=cmap,
                   norm=mcol.PowerNorm(gamma, 0, vmax), interpolation="bilinear", zorder=0, rasterized=True)
    ok = (np.hypot(U, V) >= coherence * M) & (M >= floor * vmax)
    Um, Vm = np.where(ok, U, np.nan), np.where(ok, V, np.nan)
    lw = 0.4 + 0.8 * np.clip(M / vmax, 0, 1) ** 0.5
    ax.streamplot(g, g, Um, Vm, density=density, color=line_color, linewidth=lw, arrowsize=0.6, arrowstyle="-|>",
                  minlength=0.2, broken_streamlines=True, zorder=2)
    for p, q in bonds:
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#8A9099", lw=0.4, alpha=0.45, zorder=1)
    for (_, k, r) in atoms:
        x, y = r[0], r[2]
        if abs(x) > window or abs(y) > window:
            continue
        if k == "Sn":
            ax.plot(x, y, "o", ms=4.4, mfc="#1F1D1B", mec="white", mew=0.6, zorder=4)
        elif k == "Ti":
            ax.plot(x, y, "o", ms=2.8, mfc="white", mec="#4A5058", mew=0.6, zorder=3)
        else:
            ax.plot(x, y, "o", ms=1.6, mfc="#4A5058", mec="white", mew=0.2, zorder=3)
    ax.set_xlim(-window, window)
    ax.set_ylim(-window, window)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im, vmax, float(ok.mean())
