"""Electronic factor of La3Ni2O7: decompose the readout R = (top of the d_z2 bonding band at Gamma) - E_F into
the Wannier matrix-element classes that carry it.

For two states A, B of one interval, H(l) = H_A + l (H_B - H_A) and, at fixed electron number,
    dR/dl = <top| dH(Gamma) |top> - <dH>_FS,   <dH>_FS = sum_nk f'(e_nk) <nk|dH(k)|nk> / sum_nk f'(e_nk),
which is integrated over l (Gauss-Legendre). dH is split by masks on (R, m, n) into onsite and hopping classes, so the
class contributions add up exactly to the model's R_B - R_A. E_F at every l comes from the 6x6x2 Wannier mesh with
Gaussian smearing at the electron number of state A. The arbitrary absolute reference of each VASP run is removed by
shifting all onsite energies of dH by one constant so that its onsite trace vanishes (onsite changes are relative to
the mean onsite change).

Needs wannier90_hr.dat, wannier90.eig, OUTCAR, POSCAR, basis_map.json and DONE of every state in
<wannier dir>/calculations/<i>/<state>/ (VASP-wannier90 outputs, not deposited).
usage: python wannier_decomposition.py <wannier dir> <interval> [<interval> ...]     writes <wannier dir>/wannier_<i>.json
(interval label P1, P2, S1 or S2; directories and file names use the lower-case label)
"""
import json
import sys
from pathlib import Path

import numpy as np
from multiprocessing import Pool

SMEARING = 0.05  # eV, Gaussian smearing for the Fermi-surface average
N_GAUSS_POINTS = 12  # Gauss-Legendre points along l
K_MESH = [(i / 6, j / 6, k / 2) for i in range(6) for j in range(6) for k in range(2)]
STATE_PAIRS = [("00", "10"), ("01", "11"), ("00", "01"), ("10", "11"), ("00", "01ip"), ("00", "01ap")]


def load(d):
    d = Path(d)
    lines = (d / "wannier90_hr.dat").read_text().split("\n")
    nw, nr = int(lines[1]), int(lines[2])
    nh = 3 + (nr + 14) // 15
    deg = np.array(" ".join(lines[3:nh]).split(), int)
    a = np.array(" ".join(lines[nh:]).split(), float).reshape(-1, 7)
    R = a[::nw * nw, :3].astype(int)
    H = (a[:, 5] + 1j * a[:, 6]).reshape(nr, nw, nw).transpose(0, 2, 1) / deg[:, None, None]  # H[R, m, n]
    pos = (d / "POSCAR").read_text().split("\n")
    lat = np.array([p.split() for p in pos[2:5]], float)
    nat = sum(map(int, pos[6].split()))
    frac = np.array([p.split()[:3] for p in pos[8:8 + nat]], float)
    basis = json.loads((d / "basis_map.json").read_text())["basis"]
    efw = json.loads((d / "basis_map.json").read_text())["efermi_window"]
    efermi = float([line for line in (d / "OUTCAR").read_text().split("\n") if "E-fermi" in line][-1].split()[2])
    return dict(R=R, H=H, lat=lat, frac=frac, basis=basis, ef=efermi, efw=efw)


def hk(R, H, k):
    ph = np.exp(2j * np.pi * (R @ np.asarray(k)))
    return np.tensordot(ph, H, 1)


def classes(S):
    """Masks over (R, m, n) for the matrix-element classes (nearest Ni-O bonds by distance)."""
    b, R, lat, frac = S["basis"], S["R"], S["lat"], S["frac"]
    at = np.array([x["atom"] for x in b])
    site = np.array([x["site"] for x in b])
    orb = np.array([x["orbital"] for x in b])
    tau = frac[at] @ lat
    shift = R @ lat
    dist = np.linalg.norm(tau[None, None, :, :] + shift[:, None, None, :] - tau[None, :, None, :], axis=-1)
    R0 = np.all(R == 0, axis=1)[:, None, None]
    same = (at[:, None] == at[None, :])[None] & R0
    ni = site == "Ni"

    def pair(s1, s2):
        return (np.outer(s1, s2) | np.outer(s2, s1))[None]

    nn = dist < 2.4
    m = {}
    m["onsite Ni d_z2"] = same & np.outer(orb == "dz2", orb == "dz2")[None] & ni[:, None][None]
    m["onsite Ni other d"] = same & ni[:, None][None] & ~m["onsite Ni d_z2"]
    for o in ("O1_inner", "O2_outer", "O_inplane"):
        m[f"onsite {o}"] = same & (site == o)[:, None][None]
    for o, name in (("O1_inner", "Ni-O1 (inner apical)"), ("O2_outer", "Ni-O2 (outer apical)"),
                    ("O_inplane", "Ni-O in-plane")):
        m[f"hop {name}"] = pair(ni, site == o) & nn
    m["hop Ni-O farther"] = pair(ni, ~ni) & ~nn
    m["hop O-O"] = pair(~ni, ~ni) & ~same
    m["hop Ni-Ni"] = pair(ni, ni) & ~same
    tot = sum(x.astype(int) for x in m.values())
    assert tot.max() == 1 and tot.min() == 1, "classes must partition H"
    return m, dict(orb=orb, site=site, at=at, dist=dist)


def fermi(e, N):
    from scipy.special import erfc
    from scipy.optimize import brentq

    def electron_count(mu):  # electrons per cell, spin-degenerate
        return (erfc((e - mu) / SMEARING) / 2).sum() * 2 / len(K_MESH)

    return brentq(lambda mu: electron_count(mu) - N, e.min() - 1, e.max() + 1)


def gamma_states(H0, info, ef):
    e, v = np.linalg.eigh(H0)
    dz2 = (np.abs(v[(info["orb"] == "dz2") & (info["site"] == "Ni")]) ** 2).sum(0)
    o1 = (np.abs(v[(info["orb"] == "pz") & (info["site"] == "O1_inner")]) ** 2).sum(0)
    near = np.where((e > ef - 0.7) & (e < ef + 0.6) & (dz2 > 0.35))[0]
    bond = [i for i in near if o1[i] < 0.05]
    anti = [i for i in near if o1[i] >= 0.05]
    return e, v, bond, anti, dz2, o1


def readout(H, R, info, N):
    Hk = [hk(R, H, k) for k in K_MESH]
    ek = np.array([np.linalg.eigvalsh(x) for x in Hk])
    ef = fermi(ek.ravel(), N)
    e, v, bond, anti, dz2, o1 = gamma_states(Hk[0], info, ef)
    top = max(bond, key=lambda i: e[i])
    return dict(ef=ef, R=e[top] - ef, top=top, e=e, v=v, bond=bond, anti=anti, Hk=Hk,
                split=(np.mean(e[anti]) - np.mean(e[bond])) if anti else None,
                dz2=dz2[top], o1=o1[top], ek=ek)


def nelec(S, info):
    ek = np.array([np.linalg.eigvalsh(hk(S["R"], S["H"], k)) for k in K_MESH])
    from scipy.special import erfc
    return (erfc((ek - S["ef"]) / SMEARING) / 2).sum() * 2 / len(K_MESH)


def decompose(A, B, info, masks, N):
    assert np.array_equal(A["R"], B["R"]), "R lists differ"
    dH = B["H"] - A["H"]
    R0 = np.where(np.all(A["R"] == 0, axis=1))[0][0]
    c = np.trace(dH[R0]).real / dH.shape[1]
    dH[R0] -= c * np.eye(dH.shape[1])  # remove the absolute reference shift
    x, w = np.polynomial.legendre.leggauss(N_GAUSS_POINTS)
    x, w = (x + 1) / 2, w / 2
    dHk = {name: [hk(A["R"], np.where(mk, dH, 0), kk) for kk in K_MESH] for name, mk in masks.items()}
    contrib = {k: 0.0 for k in masks}
    path = []
    for l, wl in zip(x, w):
        r = readout(A["H"] + l * dH, A["R"], info, N)
        path.append([float(l), 1e3 * r["R"], [1e3 * (r["e"][i] - r["ef"]) for i in r["bond"]],
                     [1e3 * (r["e"][i] - r["ef"]) for i in r["anti"]]])
        vt = r["v"][:, r["top"]]
        fp = np.exp(-((r["ek"] - r["ef"]) / SMEARING) ** 2)
        evs = [np.linalg.eigh(h)[1] for h in r["Hk"]]
        for name in masks:
            g = (vt.conj() @ dHk[name][0] @ vt).real
            fs = sum((fp[ik] * np.einsum("in,ij,jn->n", evs[ik].conj(), dHk[name][ik], evs[ik]).real).sum()
                     for ik in range(len(K_MESH)))
            contrib[name] += wl * (g - fs / fp.sum())
    return contrib, c, path


def common(S, ref):
    """Put every state on one R list in the cell labelling of ref. An atom written one lattice vector T away from its
    position in ref (wrapped coordinate) has its Wannier functions at tau + T, so H_B[R, m, n] belongs to
    R' = R + T_n - T_m in ref's labelling: a diagonal phase gauge, eigenvalues unchanged."""
    at = np.array([x["atom"] for x in ref["basis"]])
    tab = {}
    for s, X in S.items():
        d = X["frac"] - ref["frac"]
        X["frac"] = ref["frac"].copy() if s != "00" else X["frac"]
        # VASP's Wannier90 interface already labels the wrapped atoms by their home-cell position (the Ni-O46 bond of
        # P1/11 sits at the same R as in P1/10), so no relabelling is needed; only the R lists are merged.
        T = 0 * np.round(d).astype(int)[at]
        dT = T[None, :, :] - T[:, None, :]  # [m, n, 3] = T_n - T_m
        H = {}
        for iR, R in enumerate(X["R"]):
            Rp = R[None, None, :] + dT
            for key in {tuple(v) for v in Rp.reshape(-1, 3)}:
                mk = np.all(Rp == key, axis=-1)
                H.setdefault(key, np.zeros_like(X["H"][0]))[mk] += X["H"][iR][mk]
        tab[s] = H
    keys = sorted(set().union(*tab.values()))
    for s, X in S.items():
        X["R"] = np.array(keys)
        X["H"] = np.array([tab[s].get(k, np.zeros_like(X["H"][0])) for k in keys])


JOB_DATA = None


def one_pair(ab):
    S, info, masks, N = JOB_DATA
    return decompose(S[ab[0]], S[ab[1]], info, masks, N)


def main():
    top, intervals = Path(sys.argv[1]), sys.argv[2:]
    for interval_name in intervals:
        dirs = {s: top / "calculations" / interval_name.lower() / s for s in ["00", "10", "01", "11", "01ip", "01ap"]
                if (top / "calculations" / interval_name.lower() / s / "DONE").exists()}
        if interval_name in ("P1", "S1") and (top / "calculations/p0start/00/DONE").exists():
            dirs["start"] = top / "calculations/p0start/00"
        with Pool(len(dirs)) as pool:
            S = dict(zip(dirs, pool.map(load, dirs.values())))
        ref = S["00"]
        for s, X in S.items():
            assert [b["atom"] for b in X["basis"]] == [b["atom"] for b in ref["basis"]]
            d = X["frac"] - ref["frac"]
            assert np.abs(d - np.round(d)).max() < 0.05, f"{s}: atom order differs"
        common(S, ref)
        masks, info = classes(ref)
        out = dict(interval=interval_name, smearing=SMEARING, states={}, pairs={})
        N = nelec(ref, info)
        out["N_model"] = N
        eig = np.array((top / "calculations" / interval_name.lower() / "00" / "wannier90.eig").read_text().split(),
                       float).reshape(-1, 3)
        dft_g = eig[eig[:, 1] == 1, 2]
        r0 = readout(ref["H"], ref["R"], info, N)
        win = (dft_g > ref["efw"] - 10) & (dft_g < ref["efw"] + 0.5)
        out["gamma_check_meV"] = float(1e3 * np.abs(np.sort(r0["e"])[: win.sum()] - np.sort(dft_g[win])).max())
        print(interval_name, "N", round(N, 3), "Gamma check (meV)", round(out["gamma_check_meV"], 3), flush=True)
        for s, X in S.items():
            r = readout(X["H"], X["R"], info, N)
            out["states"][s] = dict(R_meV=1e3 * r["R"], split_meV=None if r["split"] is None else 1e3 * r["split"],
                                    n_bond=len(r["bond"]), n_anti=len(r["anti"]), top_dz2=float(r["dz2"]),
                                    top_O1pz=float(r["o1"]), ef_model=r["ef"], ef_dft=X["ef"])
        pairs = [(a, b) for a, b in list(STATE_PAIRS) + [("start", "00")] if a in S and b in S]
        global JOB_DATA
        JOB_DATA = (S, info, masks, N)
        with Pool(len(pairs)) as pool:
            res = pool.map(one_pair, pairs)
        for (a, b), (c, shift, path) in zip(pairs, res):
            ends = out["states"][b]["R_meV"] - out["states"][a]["R_meV"]
            out["pairs"][f"{b}-{a}"] = dict(total_meV=1e3 * sum(c.values()), ends_meV=ends, ref_shift_eV=shift,
                                             classes_meV={k: 1e3 * v for k, v in c.items()}, path=path)
            print(interval_name, f"{b}-{a}", round(1e3 * sum(c.values()), 2), "ends", round(ends, 2), flush=True)
        (top / f"wannier_{interval_name.lower()}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
