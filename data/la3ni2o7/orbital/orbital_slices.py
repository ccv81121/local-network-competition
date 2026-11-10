"""Charge of the interlayer d_z2 pairs at Gamma (U = 4 eV readout settings) on the plane of the Ni-O1-Ni bend.

gamma state: bands 215+216, the bonding pair whose top is E_gamma (no O1 p_z weight); partner: bands 217+218
(antibonding, with O1 p_z). PARCHG from LPARD runs on the SCF WAVECAR of the drive-grid structures p0, p2 and film
(server $WORK/ORB327_20261104/<s>/parchg; inputs and band choice in README.md). The plane is the one of
code/fig4.py bilayer(): centred on O1, horizontal axis towards the side of the bend, vertical axis c.
usage: python orbital_slices.py <dir with p0/ p2/ film/ each holding parchg/PARCHG.021[5-8].ALLK>
Writes slices.npz (u, v in A; rho_<s>_gamma, rho_<s>_partner in e/A^3)."""
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent


def read(fn):
    f = open(fn); t = [f.readline() for _ in range(7)]; s = float(t[1])
    L = s * np.array([[float(x) for x in t[i].split()] for i in (2, 3, 4)]); el = t[5].split(); n = [int(x) for x in t[6].split()]
    f.readline(); fr = np.array([[float(x) for x in f.readline().split()[:3]] for _ in range(sum(n))])
    f.readline(); g = [int(x) for x in f.readline().split()]; N = int(np.prod(g)); v = []
    while len(v) < N:
        v.extend(f.readline().split())
    return L, fr, list(np.repeat(el, n)), np.array(v[:N], float).reshape(g[2], g[1], g[0]).transpose(2, 1, 0)


def plane(L, fr, sp):
    """O1, the two Ni of its bilayer and the in-plane axis towards the side of the bend (as fig4.bilayer)."""
    ni = [i for i, s in enumerate(sp) if s == "Ni"]
    rel = lambda i, r: ((fr[i] - fr[r]) - np.round(fr[i] - fr[r])) @ L
    for o in (i for i, s in enumerate(sp) if s == "O"):
        d = [rel(i, o) for i in ni]; near = [v for v in d if np.linalg.norm(v) < 2.4]
        if len(near) == 2 and all(abs(v[2]) > 0.8 * np.linalg.norm(v) for v in near):
            break
    pair = sorted(ni, key=lambda i: np.linalg.norm(rel(i, o)))[:2]
    side = -(rel(pair[0], o) + rel(pair[1], o)) / 2; side[2] = 0
    ex = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else np.array([1.0, 0, 0])
    return fr[o] @ L, ex, np.array([0, 0, 1.0])


def main(root):
    u, v = np.linspace(-2.6, 2.6, 131), np.linspace(-4.8, 4.8, 241)
    U, V = np.meshgrid(u, v, indexing="ij"); out = dict(u=u, v=v)
    for s in ("p0", "p2", "film"):
        rho = {b: read(Path(root) / s / "parchg" / f"PARCHG.0{b}.ALLK") for b in (215, 216, 217, 218)}
        L, fr, sp, _ = rho[215]; c0, ex, ez = plane(L, fr, sp)
        frac = (c0 + U[..., None] * ex + V[..., None] * ez).reshape(-1, 3) @ np.linalg.inv(L); vol = abs(np.linalg.det(L))
        for name, bands in (("gamma", (215, 216)), ("partner", (217, 218))):
            r = sum(rho[b][3] for b in bands); g = np.array(r.shape)
            out[f"rho_{s}_{name}"] = (map_coordinates(r, (frac % 1.0 * g).T, order=1, mode="grid-wrap").reshape(U.shape) / vol).astype(np.float32)
    np.savez_compressed(HERE / "slices.npz", **out)
    print({k: float(x.max()) for k, x in out.items() if k.startswith("rho")})


if __name__ == "__main__":
    main(sys.argv[1])
