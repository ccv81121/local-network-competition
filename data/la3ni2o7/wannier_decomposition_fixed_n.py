"""Electronic factor of La3Ni2O7: decompose the readout R = (top of the d_z2 bonding band at Gamma) - E_F into
the Wannier matrix-element classes that carry it.

For two states A, B of one interval, H(l) = H_A + l (H_B - H_A) and, at fixed electron number,
    dR/dl = <top| dH(Gamma) |top> - <dH>_FS,   <dH>_FS = sum_nk f'(e_nk) <nk|dH(k)|nk> / sum_nk f'(e_nk),
which is integrated over l (Gauss-Legendre). dH is split by masks on (R, m, n) into onsite and hopping classes, so the
class contributions add up exactly to the model's R_B - R_A. Every hopping is placed on its nearest periodic image by
atom positions (fold_atoms) and matrix-element classes are assigned after this folding. E_F at every l comes from a
12x12x3 mesh, Gaussian smearing 0.03 eV, at N = 228 electrons (28 O 2p6 + 8 Ni d7.5; the 6x6x2 count of the DFT
occupations gives 227.9-228.0). The arbitrary absolute reference of each VASP run is removed by
shifting all onsite energies of dH by one constant so that its onsite trace vanishes (onsite changes are relative to
the mean onsite change).

Needs wannier90_hr.dat, wannier90.eig, OUTCAR, POSCAR, basis_map.json and DONE of every state in
<wannier dir>/calculations/<i>/<state>/ (VASP-wannier90 outputs, not deposited).
usage: python wannier_decomposition_fixed_n.py <wannier dir> <interval> [<interval> ...]
writes <wannier dir>/wannier4_<i>.json (interval label P1, P2, S1 or S2; directories and file names use the lower-case label)
"""
import json
import sys
from pathlib import Path

import numpy as np
from multiprocessing import Pool

SMEARING = 0.03  # eV, Gaussian smearing for the Fermi-surface average
N_GAUSS_POINTS = 12  # Gauss-Legendre points along l
K_MESH = [(i / 12, j / 12, k / 3) for i in range(12) for j in range(12) for k in range(3)]
WIGNER_SEITZ = None  # optional Wigner-Seitz corrections (wsvec); not set, images come from fold_atoms
STATE_PAIRS = [("00", "10"), ("01", "11"), ("00", "01"), ("10", "11"), ("00", "01ip"), ("00", "01ap")]


def load(d):
    d = Path(d)
    cache = d / "H_cache.npz"
    if cache.exists():
        z = np.load(cache)
        R, H = z["R"], z["H"]
    else:
        head = (d / "wannier90_hr.dat").read_text().split("\n", 3)
        nw, nr = int(head[1]), int(head[2])
        nums = np.array(head[3].split(), float)
        deg = nums[:nr].astype(int)
        a = nums[nr:].reshape(-1, 7)
        R = a[::nw * nw, :3].astype(int)
        H = (a[:, 5] + 1j * a[:, 6]).reshape(nr, nw, nw).transpose(0, 2, 1) / deg[:, None, None]  # H[R, m, n]
        np.savez(cache, R=R, H=H)
    pos = (d / "POSCAR").read_text().split("\n")
    lat = np.array([p.split() for p in pos[2:5]], float)
    nat = sum(map(int, pos[6].split()))
    frac = np.array([p.split()[:3] for p in pos[8:8 + nat]], float)
    basis = json.loads((d / "basis_map.json").read_text())["basis"]
    efw = json.loads((d / "basis_map.json").read_text())["efermi_window"]
    import subprocess
    efermi = float(subprocess.run(["grep", "E-fermi", str(d / "OUTCAR")], capture_output=True,
                                  text=True).stdout.split("\n")[-2].split()[2])
    return dict(R=R, H=H, lat=lat, frac=frac, basis=basis, ef=efermi, efw=efw)


def hk(R, H, k):
    k = np.asarray(k)
    out = np.tensordot(np.exp(2j * np.pi * (R @ k)), H, 1)
    if WIGNER_SEITZ is not None and np.any(k):  # exact at Gamma without corrections
        iR, m, n, RT, w = WIGNER_SEITZ
        c = np.zeros(len(iR), complex)
        np.add.at(c, w[1], w[0] * np.exp(2j * np.pi * (RT @ k)))
        c -= np.exp(2j * np.pi * (R[iR] @ k))
        np.add.at(out, (m, n), H[iR, m, n] * c)
    return out


def wsvec(d, R):
    """Elements whose Wigner-Seitz image is not just R: (iR, m, n), and for every image R+T its weight and owner."""
    a = np.array((Path(d) / "wannier90_wsvec.dat").read_text().split(chr(10), 1)[1].split(), int)
    idx = {tuple(r): i for i, r in enumerate(R)}
    p, el, RT, wt, own = 0, [], [], [], []
    while p < len(a):
        r, m, n, cnt = a[p:p + 3], a[p + 3] - 1, a[p + 4] - 1, a[p + 5]
        T = a[p + 6:p + 6 + 3 * cnt].reshape(cnt, 3)
        p += 6 + 3 * cnt
        if cnt == 1 and not T.any():
            continue
        own += [len(el)] * cnt
        el.append((idx[tuple(r)], m, n))
        RT.append(r[None, :] + T)
        wt += [1 / cnt] * cnt
    el = np.array(el)
    return el[:, 0], el[:, 1], el[:, 2], np.vstack(RT), (np.array(wt), np.array(own))


def classes(S):
    """Masks over (R, m, n) for the matrix-element classes (nearest Ni-O bonds by distance)."""
    b, R, lat, frac = S["basis"], S["R"], S["lat"], S["frac"]
    at = np.array([x["atom"] for x in b])
    site = np.array([x["site"] for x in b])
    orb = np.array([x["orbital"] for x in b])
    tau = (frac[at] + S.get("wf_off", 0)) @ lat  # WF positions (fold_atoms sets wf_off)
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
    return 228.0


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
    ctop = {k: 0.0 for k in masks}
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
            ctop[name] += wl * g
    return contrib, c, path, ctop


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


def fold(S, ws):
    """Fold the Wigner-Seitz images into the R list once (H[R] -> sum_T w H at R+T), in place for every state, so
    every Fourier sum is a single matrix product and every element sits at its true lattice vector. Exact at Gamma
    and on the 6x6x2 mesh."""
    iR, m, n, RT, (wt, own) = ws
    R = next(iter(S.values()))["R"]
    keys = np.unique(np.vstack([R, RT]), axis=0)
    kidx = {tuple(r): i for i, r in enumerate(keys)}
    at_R = np.array([kidx[tuple(r)] for r in R])
    at_RT = np.array([kidx[tuple(r)] for r in RT])
    for X in S.values():
        assert np.array_equal(X["R"], R)
        Hn = np.zeros((len(keys),) + X["H"].shape[1:], complex)
        Hn[at_R] = X["H"]
        np.subtract.at(Hn, (at_R[iR], m, n), X["H"][iR, m, n])
        np.add.at(Hn, (at_RT, m[own], n[own]), wt * X["H"][iR[own], m[own], n[own]])
        X["H"], X["R"] = Hn, keys


def wf_offsets(X):
    """Lattice offset of every Wannier function relative to its atom. Some O WFs sit one lattice vector away
    from the POSCAR position of their atom (gauge of the VASP-wannier90 interface for atoms on a cell face), which an
    even supercell along c cannot undo. Ni WFs are taken at their atoms; each O WF gets the offset T (|T_i| <= 1) that
    puts its strongest hopping to a Ni d orbital at a nearest-neighbour distance (< 2.4 A)."""
    b = X["basis"]
    at = np.array([x["atom"] for x in b])
    ni = np.array([x["site"] == "Ni" for x in b])
    tau = X["frac"][at]
    T = np.zeros((len(b), 3), int)
    cands = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    for n in np.where(~ni)[0]:
        blk = np.abs(X["H"][:, ni, n])  # [R, Ni orbitals]
        iR, im = np.unravel_index(np.argmax(blk), blk.shape)
        m = np.where(ni)[0][im]
        d = [np.linalg.norm((tau[n] + t + X["R"][iR] - tau[m]) @ X["lat"]) for t in cands]
        k = int(np.argmin(d))
        assert d[k] < 2.4, (n, d[k])
        T[n] = cands[k]
    for a in set(at[~ni]):  # all p orbitals of one atom agree
        rows = T[at == a]
        assert (rows == rows[0]).all(), (a, rows)
    return T


def fold_atoms(S, ref, nk=(6, 6, 2)):
    """Assign every hopping to its nearest periodic image by ATOM positions. wannier90's use_ws_distance
    chooses images by the Wannier centres, which are unreliable here (projected WFs, num_iter = 0, two k points along
    c: centres up to 0.39 in fractional c away from their atoms), and it placed some nearest-neighbour Ni-O hoppings
    (1.4 eV) on images 2c away. Raw H(R)/ndegen is summed per class R mod nk (exact on the k mesh), and each class is
    put on its representative(s) R + T (T in the nk supercell) of smallest |tau_n + R - tau_m|, ties shared equally.
    The same assignment (from ref positions) is applied to every state, in place."""
    nk = np.array(nk)
    at = np.array([x["atom"] for x in ref["basis"]])
    # The VASP-wannier90 interface places each Wannier function at its atom's fractional position wrapped to
    # [-0.5, 0.5) (checked on P0start/00, P2/11 and the film structure); with these positions all strong Ni-O
    # hoppings (> 0.3 eV) are nearest neighbours, and the remaining far copies are exactly half of them (the c-boundary
    # split of ndegen 2), which the class sum below restores.
    off = -(ref["frac"][at] >= 0.5).astype(int)
    tau = ref["frac"][at] + off
    ref["wf_off"] = off
    R = ref["R"]
    cls = {}
    for iR, r in enumerate(R):
        cls.setdefault(tuple(np.mod(r, nk)), []).append(iR)
    shifts = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]) * nk
    new_R, weights = [], []  # per class: candidate R list and weight [cand, m, n]
    for c, members in cls.items():
        base = np.array(c)
        cand = np.array([cc for cc in {tuple(x) for x in (base + shifts)}])
        dvec = tau[None, None, :, :] + cand[:, None, None, :] - tau[None, :, None, :]
        dist = np.linalg.norm(dvec @ ref["lat"], axis=-1)  # [cand, m, n]
        best = dist.min(axis=0)
        w = (dist < best[None] + 1e-3).astype(float)
        w /= w.sum(axis=0, keepdims=True)
        new_R.append(cand)
        weights.append((members, w))
    keys = np.unique(np.vstack(new_R), axis=0)
    kidx = {tuple(r): i for i, r in enumerate(keys)}
    for X in S.values():
        assert np.array_equal(X["R"], R)
        Hn = np.zeros((len(keys),) + X["H"].shape[1:], complex)
        for (members, w), cand in zip(weights, new_R):
            Hc = X["H"][members].sum(axis=0)
            for ic, r in enumerate(cand):
                Hn[kidx[tuple(r)]] += w[ic] * Hc
        X["H"], X["R"] = Hn, keys


JOB_DATA = None
PROCS = int(__import__("os").environ.get("PROCS", "160"))


def one_point(task):
    """One Gauss-Legendre point of one pair (the parallel unit): class terms of dR/dl at l, unweighted."""
    a, b, il = task
    S, info, masks, N = JOB_DATA
    A, B = S[a], S[b]
    dH = B["H"] - A["H"]
    R0 = np.where(np.all(A["R"] == 0, axis=1))[0][0]
    c = np.trace(dH[R0]).real / dH.shape[1]
    dH[R0] -= c * np.eye(dH.shape[1])  # remove the absolute reference shift
    x, w = np.polynomial.legendre.leggauss(N_GAUSS_POINTS)
    l, wl = (x[il] + 1) / 2, w[il] / 2
    r = readout(A["H"] + l * dH, A["R"], info, N)
    vt = r["v"][:, r["top"]]
    fp = np.exp(-((r["ek"] - r["ef"]) / SMEARING) ** 2)
    evs = [np.linalg.eigh(h)[1] for h in r["Hk"]]
    top, fs = {}, {}
    for name, mk in masks.items():
        dHc = np.where(mk, dH, 0)
        top[name] = (vt.conj() @ hk(A["R"], dHc, K_MESH[0]) @ vt).real
        fs[name] = sum((fp[ik] * np.einsum("in,ij,jn->n", evs[ik].conj(), hk(A["R"], dHc, K_MESH[ik]),
                                           evs[ik]).real).sum()
                       for ik in range(len(K_MESH))) / fp.sum()
    point = [float(l), 1e3 * r["R"], [1e3 * (r["e"][i] - r["ef"]) for i in r["bond"]],
             [1e3 * (r["e"][i] - r["ef"]) for i in r["anti"]]]
    return a, b, wl, top, fs, c, point


def one_state(s):
    S, info, masks, N = JOB_DATA
    X = S[s]
    r = readout(X["H"], X["R"], info, N)
    return s, dict(R_meV=1e3 * r["R"], split_meV=None if r["split"] is None else 1e3 * r["split"],
                   n_bond=len(r["bond"]), n_anti=len(r["anti"]), top_dz2=float(r["dz2"]),
                   top_O1pz=float(r["o1"]), ef_model=r["ef"], ef_dft=X["ef"])


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
        fold_atoms(S, ref)
        # Classes are assigned after folding, by the distance of each image. On the stored R (6x6x2 mesh: c-images 2c
        # apart) some nearest Ni-O1 hoppings sit at the far image and would be counted as "Ni-O farther"; class totals
        # are unaffected, the split between classes is.
        masks, info = classes(ref)
        out = dict(interval=interval_name, smearing=SMEARING, states={}, pairs={})
        N = nelec(ref, info)
        out["N_model"] = N
        eig = np.array((top / "calculations" / interval_name.lower() / "00" / "wannier90.eig").read_text().split(),
                       float).reshape(-1, 3)
        dft_g = eig[eig[:, 1] == 1, 2]
        r0 = readout(ref["H"], ref["R"], info, N)
        e6 = np.array([np.linalg.eigvalsh(hk(ref["R"], ref["H"], (i / 6, j / 6, k / 2))) for i in range(6)
                       for j in range(6) for k in range(2)])
        e6d = np.sort(eig[:, 2].reshape(72, -1), axis=1)

        def frozen_levels(e):
            return np.sort(e[(e > ref["efw"] - 9.9) & (e < ref["efw"] + 0.4)])

        out["mesh_check_meV"] = float(1e3 * max(
            min(np.abs(frozen_levels(e6[q]) - frozen_levels(e6d[p])).max()
                if len(frozen_levels(e6[q])) == len(frozen_levels(e6d[p])) else 9 for p in range(72))
            for q in range(72)))
        print(interval_name, "6x6x2 check with WS (meV)", round(out["mesh_check_meV"], 3), flush=True)
        win = (dft_g > ref["efw"] - 10) & (dft_g < ref["efw"] + 0.5)
        out["gamma_check_meV"] = float(1e3 * np.abs(np.sort(r0["e"])[: win.sum()] - np.sort(dft_g[win])).max())
        print(interval_name, "N", round(N, 3), "Gamma check (meV)", round(out["gamma_check_meV"], 3), flush=True)
        pairs = [(a, b) for a, b in list(STATE_PAIRS) + [("start", "00")] if a in S and b in S]
        global JOB_DATA
        JOB_DATA = (S, info, masks, N)
        tasks = [(a, b, il) for a, b in pairs for il in range(N_GAUSS_POINTS)]
        with Pool(min(PROCS, len(tasks) + len(S))) as pool:
            st = pool.map_async(one_state, list(S))
            res = pool.map(one_point, tasks, chunksize=1)
            out["states"] = dict(st.get())
        for a, b in pairs:
            rows = [x for x in res if x[0] == a and x[1] == b]
            tp = {k: sum(x[2] * x[3][k] for x in rows) for k in masks}
            ef = {k: sum(x[2] * x[4][k] for x in rows) for k in masks}
            c = {k: tp[k] - ef[k] for k in masks}
            ends = out["states"][b]["R_meV"] - out["states"][a]["R_meV"]
            out["pairs"][f"{b}-{a}"] = dict(total_meV=1e3 * sum(c.values()), ends_meV=ends, ref_shift_eV=rows[0][5],
                                             classes_meV={k: 1e3 * v for k, v in c.items()},
                                             top_meV={k: 1e3 * v for k, v in tp.items()},
                                             ef_meV={k: 1e3 * v for k, v in ef.items()},
                                             path=sorted(x[6] for x in rows))
            print(interval_name, f"{b}-{a}", round(1e3 * sum(c.values()), 2), "ends", round(ends, 2), flush=True)
        (top / f"wannier4_{interval_name.lower()}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
