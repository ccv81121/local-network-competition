"""Exact partition of band-edge margins of a Wannier Hamiltonian into local and network parts.

E_n(k) = sum_g <psi_nk|H_g(k)|psi_nk>  (H linear in its matrix elements; degenerate subspaces trace-averaged).
Groups (by atomic sites of the basis functions, all R kept):
  G1 onsite   R=0, same atom (all orbitals of that atom)                     -> local
  G2 BX_oct   B to its 6 nearest X                                           -> local
  G3 AX_link  A to its 4 nearest X                                           -> separate
  G4 network  everything else: XX, BB, BX_remote, AA, AB, AX_remote          -> network
Margins (fixed k): Lambda_C = E_C(R) - E_C(Gamma); Lambda_V = E_V(Gamma) - max_X-type E_V,
X-type = (1/2,0,0), (0,1/2,0), (0,0,1/2).

TASKDIR holds basis_map.json; the Wannier subdirectory holds wannier90_hr.dat and POSCAR. Validation against DFT uses
EIGENVAL in the same subdirectory when present. The result is printed and written to wpart.json in that subdirectory.

The Wannier90 files (wannier90_hr.dat, EIGENVAL) are not deposited; wpart.json is the result.

usage: python wannier_margin_split.py TASKDIR [WANNIER_SUBDIR]
"""
import json
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(sys.argv[1])
WANNIER_DIR = TASK_DIR / (sys.argv[2] if len(sys.argv) > 2 else "03_wannier_validated")
basis = json.loads((TASK_DIR / "basis_map.json").read_text())["basis"]
nw = len(basis)


def read_hr(p):
    with open(p) as fh:
        fh.readline()
        n = int(fh.readline())
        nr = int(fh.readline())
        deg = []
        while len(deg) < nr:
            deg += [int(x) for x in fh.readline().split()]
        dat = np.loadtxt(fh)
    assert n == nw and dat.shape[0] == nr * n * n
    R = dat[::n * n, :3].astype(int)
    H = (dat[:, 5] + 1j * dat[:, 6]).reshape(nr, n, n).transpose(0, 2, 1)  # file loops m fastest: H[r, m, n]
    return R, np.array(deg), H


def read_lat(p):
    lines = p.read_text().splitlines()
    scale = float(lines[1].split()[0])
    return np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]) * scale


R, deg, H = read_hr(WANNIER_DIR / "wannier90_hr.dat")
lat = read_lat(WANNIER_DIR / "POSCAR")
site = np.array([b["canonical_site_index"] for b in basis])
role = np.array([b["role"] for b in basis])
frac = np.array([b["frac"] for b in basis])
elem = np.array([b["element"] for b in basis])
sites = sorted(set(site))
site_frac = {s: frac[site == s][0] for s in sites}
site_role = {s: role[site == s][0] for s in sites}


def near(center_role, n_neigh):
    """Set of (s, t, image) for the n_neigh nearest X of every site of center_role.

    image is the integer cell of t relative to s."""
    out = set()
    x_sites = [t for t in sites if site_role[t] == "X"]
    for s in (s for s in sites if site_role[s] == center_role):
        candidates = []
        for t in x_sites:
            d = site_frac[t] - site_frac[s]
            img = -np.round(d)
            dc = (d + img) @ lat
            candidates.append((np.linalg.norm(dc), t, tuple(img.astype(int))))
        candidates.sort()
        assert candidates[n_neigh][0] - candidates[n_neigh - 1][0] > 0.05, f"ambiguous shell at site {s}"
        for d, t, img in candidates[:n_neigh]:
            out.add((s, t, img))
    return out


BX_PAIRS = near("B", 6)
AX_PAIRS = near("A", 4)
# group label per (R, m, n): H[r, m, n] = <m,0|H|n,R>; vector from m to n = R + f_n - f_m
group_label = np.full(H.shape, 3, dtype=np.int8)
site_m, site_n = site[:, None], site[None, :]
for r, Rv in enumerate(R):
    group_r = np.full((nw, nw), 3, dtype=np.int8)
    if not Rv.any():
        group_r[site_m == site_n] = 0
    for (s, t, img) in BX_PAIRS | AX_PAIRS:
        g = 1 if (s, t, img) in BX_PAIRS else 2
        # B/A at home cell, X at cell img -> R = img (s->t); and reverse t->s at R = -img
        if tuple(Rv) == img:
            group_r[np.ix_(site == s, site == t)] = g
        if tuple(Rv) == tuple(-np.array(img)):
            group_r[np.ix_(site == t, site == s)] = g
    group_label[r] = group_r
GROUP_NAMES = ["onsite", "BX_oct", "AX_link", "network"]


def hk(k, mask=None):
    ph = np.exp(2j * np.pi * (R @ k)) / deg
    Hm = H if mask is None else np.where(mask, H, 0)
    return np.tensordot(ph, Hm, axes=(0, 0))


def partition(k, nocc, eref=None):
    Hk = hk(k)
    Hk = 0.5 * (Hk + Hk.conj().T)
    w, V = np.linalg.eigh(Hk)
    parts = [hk(k, group_label == g) for g in range(4)]
    fine = {}
    onsite_mask = group_label == 0
    for rl in ("A", "B", "X"):
        fine[f"onsite_{rl}"] = hk(k, onsite_mask & (role[None, :, None] == rl))
    net = group_label == 3
    for name, test in (("net_XX", lambda a, b: a == "X" and b == "X"), ("net_BB", lambda a, b: a == "B" and b == "B"),
                       ("net_BX_remote", lambda a, b: {a, b} == {"B", "X"}),
                       ("net_A_other", lambda a, b: "A" in (a, b))):
        M = np.array([[test(ra, rb) for rb in role] for ra in role])
        fine[name] = hk(k, net & M[None])
    out = {}
    for band, idx in (("V", nocc - 1), ("C", nocc)):
        # pick the Wannier state that matches the DFT edge state (robust to in-gap ghosts)
        if eref is not None:
            idx = int(np.argmin(np.abs(w - eref[nocc - 1 if band == "V" else nocc])))
        e = w[idx]
        sel = np.flatnonzero(np.abs(w - e) < 1e-4)
        P = V[:, sel]

        def mean_trace(M):
            return float(np.real(np.trace(P.conj().T @ M @ P)) / len(sel))

        out[band] = {"E": float(e), "deg": int(len(sel)), **{GROUP_NAMES[g]: mean_trace(parts[g]) for g in range(4)},
                     **{fine_name: mean_trace(v) for fine_name, v in fine.items()}}
        out[band]["sum_check"] = out[band]["E"] - sum(out[band][n] for n in GROUP_NAMES)
    return out, w


eigenval_path = WANNIER_DIR / "EIGENVAL"
dft_ref = []
if eigenval_path.exists():
    lines = eigenval_path.read_text().splitlines()
    ne, nk, nb = map(int, lines[5].split())
    nocc = ne // 2
    i = 7
    for _ in range(nk):
        while not lines[i].strip():
            i += 1
        q = np.array(list(map(float, lines[i].split()[:3])))
        i += 1
        dft_ref.append((q, np.array([float(lines[i + j].split()[1]) for j in range(nb)])))
        i += nb
else:
    nocc = int(json.loads((WANNIER_DIR / "VALIDATION_REPORT.json").read_text())["dft"]["nocc"])


def dft_at(qt):
    for q, e in dft_ref:
        if np.allclose((q - qt) - np.round(q - qt), 0, atol=1e-6):
            return e


K_POINTS = {"G": np.zeros(3), "R": np.array([.5, .5, .5]), "X1": np.array([.5, 0, 0]), "X2": np.array([0, .5, 0]),
            "X3": np.array([0, 0, .5])}
partitions = {k: partition(q, nocc, dft_at(q) if dft_ref else None)[0] for k, q in K_POINTS.items()}
kx = max(("X1", "X2", "X3"), key=lambda k: partitions[k]["V"]["E"])
keys = GROUP_NAMES + ["onsite_A", "onsite_B", "onsite_X", "net_XX", "net_BB", "net_BX_remote", "net_A_other", "E"]
LamC = {n: 1000 * (partitions["R"]["C"][n] - partitions["G"]["C"][n]) for n in keys}
LamV = {n: 1000 * (partitions["G"]["V"][n] - partitions[kx]["V"][n]) for n in keys}


def summarize(L):
    return {"Lambda_meV": L["E"], "local_L": L["onsite"] + L["BX_oct"], "network_N": L["network"],
            "A_link": L["AX_link"], **L}


out = {"task": str(TASK_DIR), "wannier": WANNIER_DIR.name, "nocc": nocc, "X_type_VB_rival": kx,
       "Lambda_C": summarize(LamC), "Lambda_V": summarize(LamV),
       "max_sum_check_eV": max(abs(partitions[k][b]["sum_check"]) for k in partitions for b in "VC"),
       "n_BX_pairs": len(BX_PAIRS), "n_AX_pairs": len(AX_PAIRS)}
if dft_ref:
    errs, ghosts = [], []
    for q, e in dft_ref:  # all 64 mesh points
        w = np.linalg.eigvalsh(0.5 * (hk(q) + hk(q).conj().T))
        # every DFT edge-24 level matched
        errs.append(max(float(np.min(np.abs(w - x))) for x in e[nocc - 12:nocc + 12]))
        # Wannier levels inside the DFT gap
        ghosts.append(int(np.sum((w > e[nocc - 1] + 0.02) & (w < e[nocc] - 0.02))))
    out["validation_edge24_match_max_abs_eV"] = max(errs)
    out["ingap_ghost_states_max"] = max(ghosts)
    out["ingap_ghost_kpoints"] = int(sum(g > 0 for g in ghosts))
    eG, eR, eX = dft_at(K_POINTS["G"]), dft_at(K_POINTS["R"]), dft_at(K_POINTS[kx])
    out["DFT_Lambda_C_meV"] = 1000 * (eR[nocc] - eG[nocc])
    out["DFT_Lambda_V_meV"] = 1000 * (eG[nocc - 1] - eX[nocc - 1])
    out["valid"] = bool(out["validation_edge24_match_max_abs_eV"] <= 0.02 and out["ingap_ghost_states_max"] == 0)
json.dump(out, open(WANNIER_DIR / "wpart.json", "w"), indent=1)
print(json.dumps(out, indent=1))
