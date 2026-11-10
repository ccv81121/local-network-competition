"""Structure predicted by H+ f at the tight r_aff (poscar_pred_tight).
H: central-difference force constants of ../force_balance_reference (forces/, meta.json), acoustic modes projected out;
f: drive at the tight r_aff (force_balance/forces/ref.npy); r_aff = tight 4-GPa fractional coordinates
(spinel/p4/relax/CONTCAR) in the tight 5-GPa lattice (spinel/p5/relax/CONTCAR); the header of static_k6_pred/raff/POSCAR
is reused. Writes poscar_pred_tight to the current directory.
usage: python predicted_structure.py"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FB = HERE.parent / "force_balance_reference"
meta = json.loads((FB / "meta.json").read_text())
species = np.repeat(meta["species"], meta["counts"])
N = len(species)
D = meta["delta_A"]


def force(name):
    return np.load(FB / "forces" / f"{name}.npy")


H = np.zeros((3 * N, 3 * N))
for i in range(N):
    for a, ax in enumerate("xyz"):
        H[3 * i + a] = -(force(f"d{i:02d}{ax}p") - force(f"d{i:02d}{ax}m")).ravel() / (2 * D)
H = 0.5 * (H + H.T)
T = np.zeros((3 * N, 3))
for a in range(3):
    T[a::3, a] = 1 / np.sqrt(N)
Q = np.eye(3 * N) - T @ T.T
w, V = np.linalg.eigh(Q @ H @ Q)
keep = np.argsort(np.abs(w))[3:]
w, V = w[keep], V[:, keep]


def Hp(f):
    return V @ ((V.T @ (Q @ f)) / w)


def read(path):
    L = Path(path).read_text().splitlines()
    s = float(L[1])
    lat = np.array([[float(x) for x in L[i].split()] for i in (2, 3, 4)]) * s
    sp = L[5].split()
    n = [int(x) for x in L[6].split()]
    h = 7 + (L[7].strip()[0] in "sS")
    fr = np.array([[float(x) for x in L[h + 1 + i].split()[:3]] for i in range(sum(n))])
    if L[h].strip()[0] not in "dD":
        fr = fr @ np.linalg.inv(lat)
    assert sp == list(meta["species"]) and n == list(meta["counts"]), (sp, n)
    return lat, fr % 1.0


_, f4t = read(HERE / "spinel/p4/relax/CONTCAR")
L5t, _ = read(HERE / "spinel/p5/relax/CONTCAR")
ft = np.load(HERE / "force_balance/forces/ref.npy").ravel()
up = Hp(ft).reshape(-1, 3)
frac = (f4t + up @ np.linalg.inv(L5t)) % 1.0
header = (HERE / "static_k6_pred/raff/POSCAR").read_text().splitlines()[:7]
header[0] = "predicted by H+ f at tight r_aff"
Path("poscar_pred_tight").write_text(
    "\n".join(header + ["Direct"] + ["  ".join(f"{x:.16f}" for x in r) for r in frac]) + "\n")
print("u_pred norm", float(np.linalg.norm(up)))
