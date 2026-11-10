"""Ni-O bond network of La3Ni2O7 (48-atom Amam cell) and the projectors used by the force-balance and four-state
scripts.

Bonds: each Ni with its six nearest O (minimum image, cutoff 2.6 A), 48 bonds. A bond is apical when its vector is
mostly along c (inner O1 shared by the two layers, or outer O2 toward the LaO rock-salt layer), in-plane otherwise.
Coordinates are Cartesian, 3N in POSCAR order. The bond-preserving space b is the null space of the bond-length
Jacobian C with the La coordinates fixed and the overall translation removed (bond-preserving network motion with
the A sites fixed).
"""
import numpy as np


def bonds(atoms, cutoff=2.6):
    sym = np.array(atoms.get_chemical_symbols())
    lat, pos = atoms.cell.array, atoms.positions
    ni, o_atoms = np.where(sym == "Ni")[0], np.where(sym == "O")[0]
    out = []
    for i in ni:
        d = pos[o_atoms] - pos[i]
        frac = d @ np.linalg.inv(lat)
        d = (frac - np.round(frac)) @ lat
        r = np.linalg.norm(d, axis=1)
        for k in np.argsort(r)[:6]:
            assert r[k] < cutoff, (i, r[k])
            v = d[k]
            kind = "apical" if abs(v[2]) > np.hypot(v[0], v[1]) else "inplane"
            out.append(dict(ni=int(i), o=int(o_atoms[k]), vec=v, length=float(r[k]), kind=kind))
    return out


def jacobian(atoms, bond_list):
    n = len(atoms)
    C = np.zeros((len(bond_list), 3 * n))
    for row, b in enumerate(bond_list):
        e = b["vec"] / b["length"]
        C[row, 3 * b["o"]:3 * b["o"] + 3] = e
        C[row, 3 * b["ni"]:3 * b["ni"] + 3] = -e
    return C


def translation_free(n):
    T = np.zeros((3, 3 * n))
    for a in range(3):
        T[a, a::3] = 1 / np.sqrt(n)
    return np.eye(3 * n) - T.T @ T


def network_mask(atoms):
    sym = np.array(atoms.get_chemical_symbols())
    return np.repeat(sym != "La", 3)


def bond_preserving_projector(atoms, bond_list, tol=1e-8):
    """Orthogonal projector onto b = ker(C) with La fixed, in translation-free coordinates."""
    C = jacobian(atoms, bond_list)
    mask = network_mask(atoms)
    Cn = C[:, mask]
    _, s, Vt = np.linalg.svd(Cn)
    rank = int((s > tol * s[0]).sum())
    K = np.zeros((3 * len(atoms), Vt.shape[0] - rank))
    K[mask] = Vt[rank:].T  # null-space basis embedded with zero La rows
    Q = translation_free(len(atoms))
    K = Q @ K
    U, s2, _ = np.linalg.svd(K, full_matrices=False)
    U = U[:, s2 > 1e-8]
    return U @ U.T, U.shape[1], C


def min_norm_bond_part(C, mask, u, rows):
    """Minimum-norm network displacement (La fixed) that reproduces C u on `rows` and leaves the other bonds
    unchanged."""
    target = np.zeros(C.shape[0])
    target[rows] = (C @ u)[rows]
    x = np.zeros_like(u)
    x[mask] = np.linalg.pinv(C[:, mask]) @ target
    return x
