"""Bending shares of the drive, the predicted response and the relaxed displacement between the tight 4- and 5-GPa
structures.
The full force-constant matrix and bond list of the earlier pressure set (FORCE_BALANCE_DIR/forces) are used unchanged;
drive f_t = forces at r_aff,t (tight 4-GPa fractional coordinates in the tight 5-GPa lattice); u_t = tight 4 -> 5 GPa
displacement in the tight 5-GPa lattice. The same quantities for the earlier endpoints (key "round5") and their
residual forces are also reported.
Reads FORCE_BALANCE_DIR/{meta.json,forces,inputs/pressure_bond_accommodation.json} and, from TIGHT_DIR,
spinel/p{4,5}/relax/{CONTCAR,OUTCAR_c1} and static_raff/OUTCAR (the OUTCARs are not deposited); writes
TIGHT_DIR/bending_shares.json.
usage: python bending_shares.py FORCE_BALANCE_DIR TIGHT_DIR   (../force_balance_reference and this folder)"""
import json, sys
from pathlib import Path
import numpy as np

FORCE_BALANCE_DIR, TIGHT_DIR = Path(sys.argv[1]), Path(sys.argv[2])
meta = json.loads((FORCE_BALANCE_DIR / "meta.json").read_text())
species = np.repeat(meta["species"], meta["counts"])
N = len(species)
delta = meta["delta_A"]


def read_forces(name):
    return np.load(FORCE_BALANCE_DIR / "forces" / f"{name}.npy")


H = np.zeros((3 * N, 3 * N))
for i in range(N):
    for a, ax in enumerate("xyz"):
        H[3 * i + a] = -(read_forces(f"d{i:02d}{ax}p") - read_forces(f"d{i:02d}{ax}m")).ravel() / (2 * delta)
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


bonds = json.loads((FORCE_BALANCE_DIR / "inputs" / "pressure_bond_accommodation.json").read_text())["bonds"]
net = np.flatnonzero(np.isin(species, ["Sc", "Sb", "Se"]))
net_column = {int(a): c for c, a in enumerate(net)}


def read(path):
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1])
    lattice = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)]) * scale
    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    header = 7 + (lines[7].strip()[0] in "sS")
    frac = np.array([[float(x) for x in lines[header + 1 + i].split()[:3]] for i in range(sum(counts))])
    if lines[header].strip()[0] not in "dD":
        frac = frac @ np.linalg.inv(lattice)
    assert elements == list(meta["species"]) and counts == list(meta["counts"]), (elements, counts)
    return lattice, frac % 1.0


def outcar_forces(path, block=-1):
    block_lines = Path(path).read_text().split("TOTAL-FORCE (eV/Angst)")[1:][block].splitlines()[2:2 + N]
    return np.array([[float(x) for x in line.split()[3:6]] for line in block_lines]).ravel()


def kernel(f_start, L):
    bond_vectors = np.array([(f_start[j] - f_start[i] + np.array(img)) @ L for i, j, img in bonds])
    J = np.zeros((len(bonds), 3 * len(net)))
    for r, (i, j, _) in enumerate(bonds):
        e = bond_vectors[r] / np.linalg.norm(bond_vectors[r])
        J[r, 3 * net_column[i]:3 * net_column[i] + 3] = -e
        J[r, 3 * net_column[j]:3 * net_column[j] + 3] = e
    _, s, Vt = np.linalg.svd(J, full_matrices=False)
    return Vt[: int(np.sum(s > s[0] * 1e-10))]


def preserving_share(u, B):
    v = u.reshape(-1, 3)[net]
    v = v - v.mean(axis=0)
    v = v.ravel()
    p = v - B.T @ (B @ v)
    return float(p @ p / (v @ v))


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def report(tag, L5, f4, f5, f0):
    df = f5 - f4
    df -= np.round(df)
    u = Q @ (df @ L5).ravel()
    B = kernel(f4, L5)
    up = Hp(f0)
    return {"case": tag, "drive_share": preserving_share(f0, B), "pred_share": preserving_share(up, B),
            "actual_share": preserving_share(u, B),
            "cos_pred_actual": cos(up, u), "cos_force_actual": cos(Q @ f0, u), "u_norm_A": float(np.linalg.norm(u)),
            "pred_norm_A": float(np.linalg.norm(up)),
            "drive_rms_eVA": float(np.sqrt(np.mean((f0.reshape(-1, 3) ** 2).sum(1))))}


out = {}
L5o = np.array(meta["lattice_5GPa"])
f4o = np.array(meta["frac_start"])
f5o = np.array(meta["frac_5GPa"])
out["round5"] = report("round5 (as published)", L5o, f4o, f5o, read_forces("ref").ravel())
L4t, f4t = read(TIGHT_DIR / "spinel/p4/relax/CONTCAR")
L5t, f5t = read(TIGHT_DIR / "spinel/p5/relax/CONTCAR")
ft = outcar_forces(TIGHT_DIR / "static_raff/OUTCAR")
out["tight"] = report("tight endpoints", L5t, f4t, f5t, ft)
for pressure in (4, 5):
    # first ionic step of the tight run = residual force at the earlier endpoint
    r = outcar_forces(TIGHT_DIR / f"spinel/p{pressure}/relax/OUTCAR_c1", 0)
    out[f"round5_P{pressure}_residual_rms_eVA"] = float(np.sqrt(np.mean((r.reshape(-1, 3) ** 2).sum(1))))
    out[f"round5_P{pressure}_residual_max_eVA"] = float(np.linalg.norm(r.reshape(-1, 3), axis=1).max())
out["lattice_change_5GPa_rel"] = float(np.linalg.norm(L5t - L5o) / np.linalg.norm(L5o))
(TIGHT_DIR / "bending_shares.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
