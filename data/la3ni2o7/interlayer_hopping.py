"""Interlayer coupling t_perp of La3Ni2O7 from the Wannier Hamiltonians, free of the c-axis aliasing of the O functions.

With two k points along c the projected O1 p_z functions are not localized along c, so real-space O hoppings along c
are not unique; the Ni functions sit inside the cell and are well defined. On every mesh k point (where H(k) is exact)
the O p orbitals are folded out (Lowdin, at energy E; all O p states lie well below E_F, so the Lowdin self-energy is
regular) and the Ni d block is projected on one local d_z2 per Ni, axis along its own Ni-O1 bond. The 8x8 effective
Hamiltonians are Fourier-inverted on the 6x6x2 mesh and every Ni pair is placed at its nearest image (Ni positions).
t_perp = |H_eff| between the two Ni of a bilayer (sharing an O1).
Needs H_cache.npz, POSCAR, basis_map.json and OUTCAR in each Wannier directory (VASP-wannier90 outputs, not deposited).
usage: python interlayer_hopping.py name=wan_dir [...]"""
import sys
from pathlib import Path

import numpy as np

import wannier_decomposition_fixed_n as fixed_n

SQRT3 = np.sqrt(3)
MESH = [(i / 6, j / 6, k / 2) for i in range(6) for j in range(6) for k in range(2)]


def local_dz2_coefficients(n):
    nx, ny, nz = n
    return np.array([(3 * nz**2 - 1) / 2, SQRT3 * nx * nz, SQRT3 * ny * nz, SQRT3 / 2 * (nx**2 - ny**2),
                     SQRT3 * nx * ny])


def tperp(d, energies):
    X = fixed_n.load(Path(d))
    b = X["basis"]
    at = np.array([x["atom"] for x in b])
    site = np.array([x["site"] for x in b])
    lat, frac = X["lat"], X["frac"] - (X["frac"] >= 0.5)  # WF positions: atoms wrapped to [-0.5, 0.5)
    ni = sorted({a for a, s in zip(at, site) if s == "Ni"})
    o1 = sorted({a for a, s in zip(at, site) if s == "O1_inner"})
    first = {a: int(np.where(at == a)[0][0]) for a in ni}

    def minimum_image(v):
        return v - np.round(v)

    P = np.zeros((len(b), len(ni)))
    partner = {}
    for c, a in enumerate(ni):
        dv = [minimum_image(frac[o] - frac[a]) @ lat for o in o1]
        k = int(np.argmin([np.linalg.norm(v) for v in dv]))
        P[first[a]:first[a] + 5, c] = local_dz2_coefficients(dv[k] / np.linalg.norm(dv[k]))
        partner.setdefault(o1[k], []).append(c)
    pairs = [tuple(v) for v in partner.values()]
    assert all(len(p) == 2 for p in pairs) and len(pairs) == len(ni) // 2, pairs
    d_indices = np.where(site == "Ni")[0]
    p_indices = np.where(site != "Ni")[0]
    Pd = P[d_indices]  # local d_z2 coefficients within the d block
    out, qmax = {}, -1e9
    for E in energies:
        Hk = []
        for k in MESH:
            H = fixed_n.hk(X["R"], X["H"], k)
            A, B, C = H[np.ix_(d_indices, d_indices)], H[np.ix_(d_indices, p_indices)], H[np.ix_(p_indices, p_indices)]
            qmax = max(qmax, np.linalg.eigvalsh(C).max())
            He = A + B @ np.linalg.solve(E * np.eye(len(C)) - C, B.conj().T)
            Hk.append(Pd.T @ He @ Pd)
        Hk = np.array(Hk)
        tau = frac[ni]
        vals = []
        for (i, j) in pairs:
            dv0 = tau[j] - tau[i]
            cands = [np.round(-dv0) + np.array(t) for t in
                     [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]]
            Rb = min(cands, key=lambda r: np.linalg.norm((dv0 + r) @ lat))
            ph = np.exp(-2j * np.pi * (np.array(MESH) @ Rb))
            vals.append(abs((ph * Hk[:, i, j]).mean()))
        out[E] = float(np.mean(vals))
    ang = []
    for o, (i, j) in partner.items():
        vi, vj = minimum_image(frac[o] - frac[ni[i]]) @ lat, minimum_image(frac[o] - frac[ni[j]]) @ lat
        ang.append(np.degrees(np.arccos(vi @ vj / np.linalg.norm(vi) / np.linalg.norm(vj))))
    return out, X["ef"], float(np.mean(ang)), qmax


if __name__ == "__main__":
    print("state        t_perp(E_F)  t_perp(E_F-0.5)  t_perp(E_F+0.5)   Ni-O1-Ni (deg)", flush=True)
    for arg in sys.argv[1:]:
        name, d = arg.split("=")
        ef = fixed_n.load(Path(d))["ef"]
        r, _, ang, qmax = tperp(d, [ef, ef - 0.5, ef + 0.5])
        v = list(r.values())
        print(f"{name:12s} {v[0]:.4f}       {v[1]:.4f}           {v[2]:.4f}            {ang:.2f}   "
              f"(top of O-p block E_F{qmax - ef:+.2f} eV)", flush=True)
