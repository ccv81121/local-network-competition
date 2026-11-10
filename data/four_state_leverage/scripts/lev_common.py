"""Shared definitions for the property-energy leverage analysis.
H, f, r_aff, bonds and the bond-preserving projector as in enrichment_all.py / sector_force_balance_analyze.py /
bending_shares.py and the corner-system analysis; four-state construction as sector_force_balance_prepare.py;
margin read-out as phi_readout.py (TRIM best / second best, edge with the smaller |Lambda| at state11) or, for the
corner systems, the published four-state read-out with fixed named valleys.
The work folders listed below ($WORK/..., $WORK/spinel/...) are on the cluster and not deposited."""
import json, re
from pathlib import Path
import numpy as np

PRL_ROOT = Path("$WORK")
MECHANISM_WORK_DIR = PRL_ROOT / "PRL_MECHANISM_TEST_20261014" / "work"
TIGHT_SPINEL_DIR = PRL_ROOT / "PRL_TIGHT_SPINEL_20261015"
CORNER_WORK_DIR = PRL_ROOT / "PRL_CORNER_FOURSTATE_20261012" / "work"
LEVERAGE_DIR = Path("$WORK/spinel/LEVERAGE_20261023")

# name: (Hessian type, work directory with meta.json, four-state directory, group)
SYSTEMS = {
    "k12_t45": ("full_tight", TIGHT_SPINEL_DIR / "fb_tight", TIGHT_SPINEL_DIR / "work/Spinel/P4_P5/fourstate", "A"),
    "k12_05": ("symsec", MECHANISM_WORK_DIR / "Spinel_k12_tight/P0_P5",
               MECHANISM_WORK_DIR / "Spinel_k12_tight/P0_P5/fourstate", "A"),
    "k08": ("symsec", MECHANISM_WORK_DIR / "Spinel_k08/P0_P5", MECHANISM_WORK_DIR / "Spinel_k08/P0_P5/fourstate", "A"),
    "Layered_NaScInO2": ("symsec", MECHANISM_WORK_DIR / "Layered_NaScInO2/P0_P5",
                         MECHANISM_WORK_DIR / "Layered_NaScInO2/P0_P5/fourstate", "A"),
    "CaTiO3": ("full", CORNER_WORK_DIR / "CaTiO3/P0_P5", CORNER_WORK_DIR / "CaTiO3/P0_P5/fourstate", "A"),
    "CaSnO3": ("full", CORNER_WORK_DIR / "CaSnO3/P0_P5", CORNER_WORK_DIR / "CaSnO3/P0_P5/fourstate", "A"),
    "Ca2SnTiO6_rs": ("symsec", MECHANISM_WORK_DIR / "Ca2SnTiO6_rs/P0_P5",
                     MECHANISM_WORK_DIR / "Ca2SnTiO6_rs/P0_P5/fourstate", "B"),
    "Ca8Sn6Ti2O24": ("symsec", MECHANISM_WORK_DIR / "Ca8Sn6Ti2O24/P0_P5",
                     MECHANISM_WORK_DIR / "Ca8Sn6Ti2O24/P0_P5/fourstate", "B"),
}

# margins of the published corner four-state read-out: (band edge, k1, k2), Lambda = E_V(k1) - E_V(k2)
CORNER_MARGIN = {"CaTiO3": ("V", "G", "U"), "CaSnO3": ("V", "U", "Z")}
TRIM_POINTS = {"G": (0, 0, 0), "X": (.5, 0, 0), "Y": (0, .5, 0), "Z": (0, 0, .5), "S": (.5, .5, 0), "U": (.5, 0, .5),
               "T": (0, .5, .5), "R": (.5, .5, .5)}


def load_system(name):
    """Return dict with L2, f1, f2, species, counts, N, net, bonds, Q, u_pred, F (drive, 3N), u_act, H-info."""
    h_type, work_dir, _, _ = SYSTEMS[name]
    meta = json.loads((work_dir / "meta.json").read_text())
    sp, cnt = list(meta["species"]), list(meta["counts"])
    species = np.repeat(sp, cnt)
    N = len(species)
    delta = meta["delta_A"]
    T = np.zeros((3 * N, 3))
    for a in range(3):
        T[a::3, a] = 1 / np.sqrt(N)
    Q = np.eye(3 * N) - T @ T.T
    out = {"name": name, "path": h_type, "species": sp, "counts": cnt, "N": N, "Q": Q}
    if h_type in ("full", "full_tight"):
        if h_type == "full_tight":
            def read_forces(c):
                return np.load(work_dir / "forces" / f"{c}.npy")

            L2 = np.array(meta["lattice_5GPa"])
            f1 = np.array(meta["frac_start"])
            f2 = np.array(meta["frac_5GPa"])
            bonds = json.loads((work_dir / "inputs" / "pressure_bond_accommodation.json").read_text())["bonds"]
            net = np.flatnonzero(np.isin(species, ["Sc", "Sb", "Se"]))
        else:
            def read_forces(c):
                return np.load(work_dir / "hessian" / c / "forces.npy")

            L2 = np.array(meta["lattice_P2"])
            f1 = np.array(meta["frac_start"])
            f2 = np.array(meta["frac_P2"])
            bonds = meta["bonds"]
            net = np.array(sorted({b[0] for b in bonds} | {b[1] for b in bonds}))
        H = np.zeros((3 * N, 3 * N))
        for i in range(N):
            for a, ax in enumerate("xyz"):
                H[3 * i + a] = -(read_forces(f"d{i:02d}{ax}p") - read_forces(f"d{i:02d}{ax}m")).ravel() / (2 * delta)
        H = 0.5 * (H + H.T)
        w, V = np.linalg.eigh(Q @ H @ Q)
        keep = np.argsort(np.abs(w))[3:]
        w, V = w[keep], V[:, keep]
        F_raw = read_forces("ref").ravel()
        F = Q @ F_raw
        u_pred = V @ ((V.T @ F) / w)
        out.update(H=H, Hpinv=V @ np.diag(1 / w) @ V.T, F_share=F_raw if h_type == "full_tight" else F)
        # enrichment_all.py uses raw f0 for the tight drive share and Q f0 for the corner drive share
    else:
        S = np.load(work_dir / "S.npy")

        def read_forces(c):
            return np.load(work_dir / "hessian" / c / "forces.npy").ravel()

        L2 = np.array(meta["lattice_P2"])
        f1 = np.array(meta["frac_start"])
        f2 = np.array(meta["frac_P2"])
        bonds = meta["bonds"]
        net = np.array(meta["network"])
        ns = S.shape[1]
        Hs = np.zeros((ns, ns))
        for k in range(ns):
            Hs[k] = -(S.T @ (read_forces(f"s{k:03d}p") - read_forces(f"s{k:03d}m"))) / (2 * delta)
        Hs = 0.5 * (Hs + Hs.T)
        F = read_forces("ref")
        fs = S.T @ F
        us = np.linalg.solve(Hs, fs)
        u_pred = S @ us
        out.update(S=S, Hs=Hs, Hs_saved=np.load(work_dir / "Hs.npy"), fs=fs, us=us, F_share=F)
    df = f2 - f1
    df -= np.round(df)
    u_act = Q @ (df @ L2).ravel()
    out.update(L2=L2, f1=f1, f2=f2, net=net, bonds=bonds, F=F, u_pred=u_pred, u_act=u_act, meta=meta)
    return out


def kernel_rows(f1, L2, bonds, net):
    col = {int(a): c for c, a in enumerate(net)}
    vec = np.array([(f1[j] - f1[i] + np.array(img)) @ L2 for i, j, img in bonds])
    J = np.zeros((len(bonds), 3 * len(net)))
    for r, (i, j, _) in enumerate(bonds):
        e = vec[r] / np.linalg.norm(vec[r])
        J[r, 3 * col[i]:3 * col[i] + 3] = -e
        J[r, 3 * col[j]:3 * col[j] + 3] = e
    _, s, Vt = np.linalg.svd(J, full_matrices=False)
    return Vt[: int(np.sum(s > s[0] * 1e-10))]


def share(u, Rb, net):
    v = u.reshape(-1, 3)[net]
    v = v - v.mean(axis=0)
    v = v.ravel()
    p = v - Rb.T @ (Rb @ v)
    return float(p @ p / (v @ v))


def split_b(u, Rb, net, N):
    """u_b = P u : bond-preserving (kernel) part of the centred network displacement, zero on non-network atoms."""
    v = u.reshape(-1, 3)[net]
    v = v - v.mean(axis=0)
    v = v.ravel()
    p = v - Rb.T @ (Rb @ v)
    ub = np.zeros((N, 3))
    ub[net] = p.reshape(-1, 3)
    return ub.ravel()


def write_poscar(path, title, lat, sp, n, fr):
    out = [title, "1.0"] + ["  ".join(f"{x:.16f}" for x in v) for v in lat]
    out += ["  ".join(sp), "  ".join(map(str, n)), "Direct"] + ["  ".join(f"{x:.16f}" for x in f % 1.0) for f in fr]
    Path(path).write_bytes(("\n".join(out) + "\n").encode())


def read_poscar(path):
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1])
    lat = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)]) * scale
    sp = lines[5].split()
    n = [int(x) for x in lines[6].split()]
    mode_line = 7 + (lines[7].strip()[0] in "sS")
    fr = np.array([[float(x) for x in lines[mode_line + 1 + i].split()[:3]] for i in range(sum(n))])
    if lines[mode_line].strip()[0] not in "dD":
        fr = fr @ np.linalg.inv(lat)
    return lat, sp, n, fr % 1.0


def read_eig(d):
    lines = (Path(d) / "EIGENVAL").read_text().splitlines()
    ne, nk, nb = map(int, lines[5].split())
    n_occ = ne // 2
    i = 6
    K, V, C = [], [], []
    for _ in range(nk):
        while not lines[i].strip():
            i += 1
        K.append(list(map(float, lines[i].split()[:3])))
        i += 1
        bands = [float(lines[i + j].split()[1]) for j in range(nb)]
        i += nb
        V.append(bands[n_occ - 1])
        C.append(bands[n_occ])
    toten = float(re.findall(r"free  energy   TOTEN  =\s+(-?\d+\.\d+)", (Path(d) / "OUTCAR").read_text())[-1])
    return np.array(K), np.array(V), np.array(C), toten


def kindex(K, q):
    for i, k in enumerate(K):
        if (np.allclose((k - q) - np.round(k - q), 0, atol=1e-6)
                or np.allclose((k + q) - np.round(k + q), 0, atol=1e-6)):
            return i
    raise KeyError(q)


def trim_list(K):
    def same_k(a, b):
        return (np.allclose((K[a] - K[b]) - np.round(K[a] - K[b]), 0, atol=1e-6)
                or np.allclose((K[a] + K[b]) - np.round(K[a] + K[b]), 0, atol=1e-6))

    trim_ids = [i for i, q in enumerate(K) if np.allclose(2 * q - np.round(2 * q), 0, atol=1e-6)]
    trim_ids = [i for n, i in enumerate(trim_ids) if not any(same_k(i, j) for j in trim_ids[:n])]
    assert len(trim_ids) == 8, len(trim_ids)
    return trim_ids


def phi_margin_from_state11(d11):
    """phi_readout.py rule: best and second-best TRIM at state11, edge with smaller |Lambda|."""
    K, V, C, _ = read_eig(d11)
    trim_ids = trim_list(K)
    order_v = sorted(trim_ids, key=lambda k: -V[k])
    order_c = sorted(trim_ids, key=lambda k: C[k])
    LV = 1000 * (V[order_v[0]] - V[order_v[1]])
    LC = 1000 * (C[order_c[1]] - C[order_c[0]])
    edge = "V" if abs(LV) <= abs(LC) else "C"
    edge_idx, rival_idx = (order_v[0], order_v[1]) if edge == "V" else (order_c[0], order_c[1])
    return {"edge": edge, "k_edge": K[edge_idx].tolist(), "k_rival": K[rival_idx].tolist(),
            "LambdaV_state11_meV": LV, "LambdaC_state11_meV": LC,
            "rule": "phi_readout.py: TRIM best vs second best at state11, edge with smaller |Lambda|"}


def margin_def(name):
    """Fixed margin of the system's published four-state readout."""
    fourstate_dir = SYSTEMS[name][2]
    if name in CORNER_MARGIN:
        e, k1, k2 = CORNER_MARGIN[name]
        return {"edge": e, "k_edge": list(TRIM_POINTS[k1]), "k_rival": list(TRIM_POINTS[k2]), "label": f"{k1}-{k2}",
                "rule": "readout_fs.py / readout_trim.py: Lambda = E_V(k1) - E_V(k2)"}
    return phi_margin_from_state11(fourstate_dir / "state11")


def Lambda(d, margin):
    """Margin in meV with the phi_readout sign convention: V: E_V(edge)-E_V(rival); C: E_C(rival)-E_C(edge).
    Corner: E_V(k1)-E_V(k2), as in the published corner four-state read-out."""
    K, V, C, toten = read_eig(d)
    edge_idx, rival_idx = kindex(K, np.array(margin["k_edge"])), kindex(K, np.array(margin["k_rival"]))
    lam = 1000 * (V[edge_idx] - V[rival_idx]) if margin["edge"] == "V" else 1000 * (C[rival_idx] - C[edge_idx])
    return lam, toten
