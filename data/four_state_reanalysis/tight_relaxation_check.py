"""Tight-relaxation check of the spinel redirection with the sensitivity of the actual share to residual forces.

Uses the full Hessian and bond list of the earlier (round5) pressure structures unchanged; new drive f_t = forces at
r_aff,t (tight 4-GPa fractional coords in tight 5-GPa lattice); u_t = tight 4 -> 5 GPa displacement in the tight 5-GPa
lattice. Also reports the round5 numbers recomputed with the same code, and the residual forces of both round5
endpoints.

Reads FB_DIR/{meta.json,forces,inputs/pressure_bond_accommodation.json} and, from TIGHT_DIR, spinel/p{4,5}/relax/
{CONTCAR,OUTCAR,OUTCAR_c1} and static_raff/OUTCAR (the OUTCARs are not deposited).
usage: python tight_relaxation_check.py FB_DIR TIGHT_DIR   (../force_balance_reference and ../spinel_pressure_path)
"""
import json, sys
from pathlib import Path
import numpy as np

FORCE_BALANCE_DIR, TIGHT_DIR = Path(sys.argv[1]), Path(sys.argv[2])
meta = json.loads((FORCE_BALANCE_DIR / "meta.json").read_text())
species = np.repeat(meta["species"], meta["counts"])
N = len(species)
DELTA_A = meta["delta_A"]


def load_forces(name):
    return np.load(FORCE_BALANCE_DIR / "forces" / f"{name}.npy")


H = np.zeros((3 * N, 3 * N))
for i in range(N):
    for a, ax in enumerate("xyz"):
        H[3 * i + a] = -(load_forces(f"d{i:02d}{ax}p") - load_forces(f"d{i:02d}{ax}m")).ravel() / (2 * DELTA_A)
H = 0.5 * (H + H.T)
T = np.zeros((3 * N, 3))
for a in range(3):
    T[a::3, a] = 1 / np.sqrt(N)
Q = np.eye(3 * N) - T @ T.T
w, V = np.linalg.eigh(Q @ H @ Q)
keep_modes = np.argsort(np.abs(w))[3:]
w, V = w[keep_modes], V[:, keep_modes]


def apply_h_pinv(f):
    return V @ ((V.T @ (Q @ f)) / w)


bonds = json.loads((FORCE_BALANCE_DIR / "inputs" / "pressure_bond_accommodation.json").read_text())["bonds"]
net = np.flatnonzero(np.isin(species, ["Sc", "Sb", "Se"]))
col = {int(a): c for c, a in enumerate(net)}


def read_poscar(path):
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1])
    lat = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)]) * scale
    names = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    header_row = 7 + (lines[7].strip()[0] in "sS")
    frac = np.array([[float(x) for x in lines[header_row + 1 + i].split()[:3]] for i in range(sum(counts))])
    if lines[header_row].strip()[0] not in "dD":
        frac = frac @ np.linalg.inv(lat)
    assert names == list(meta["species"]) and counts == list(meta["counts"]), (names, counts)
    return lat, frac % 1.0


def outcar_forces(path, block=-1):
    block_lines = Path(path).read_text().split("TOTAL-FORCE (eV/Angst)")[1:][block].splitlines()[2:2 + N]
    return np.array([[float(x) for x in line.split()[3:6]] for line in block_lines]).ravel()


def kernel(f_start, L):
    vec = np.array([(f_start[j] - f_start[i] + np.array(img)) @ L for i, j, img in bonds])
    J = np.zeros((len(bonds), 3 * len(net)))
    for r, (i, j, _) in enumerate(bonds):
        e = vec[r] / np.linalg.norm(vec[r])
        J[r, 3 * col[i]:3 * col[i] + 3] = -e
        J[r, 3 * col[j]:3 * col[j] + 3] = e
    _, s, Vt = np.linalg.svd(J, full_matrices=False)
    return Vt[: int(np.sum(s > s[0] * 1e-10))]


def pres(u, B):
    v = u.reshape(-1, 3)[net]
    v = v - v.mean(axis=0)
    v = v.ravel()
    p = v - B.T @ (B @ v)
    return float(p @ p / (v @ v))


def cosine(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def report(tag, L5, f4, f5, f0):
    df = f5 - f4
    df -= np.round(df)
    u = Q @ (df @ L5).ravel()
    B = kernel(f4, L5)
    up = apply_h_pinv(f0)
    return {"case": tag, "drive_share": pres(f0, B), "pred_share": pres(up, B), "actual_share": pres(u, B),
            "cos_pred_actual": cosine(up, u), "cos_force_actual": cosine(Q @ f0, u),
            "u_norm_A": float(np.linalg.norm(u)), "pred_norm_A": float(np.linalg.norm(up)),
            "drive_rms_eVA": float(np.sqrt(np.mean((f0.reshape(-1, 3) ** 2).sum(1))))}


out = {}
L5o = np.array(meta["lattice_5GPa"])
f4o = np.array(meta["frac_start"])
f5o = np.array(meta["frac_5GPa"])
out["round5"] = report("round5 (as published)", L5o, f4o, f5o, load_forces("ref").ravel())
L4t, f4t = read_poscar(TIGHT_DIR / "spinel/p4/relax/CONTCAR")
L5t, f5t = read_poscar(TIGHT_DIR / "spinel/p5/relax/CONTCAR")
ft = outcar_forces(TIGHT_DIR / "static_raff/OUTCAR")
out["tight"] = report("tight endpoints", L5t, f4t, f5t, ft)
for P in (4, 5):
    # first ionic step of the tight run = residual at the round5 endpoint
    r = outcar_forces(TIGHT_DIR / f"spinel/p{P}/relax/OUTCAR_c1", 0)
    out[f"round5_P{P}_residual_rms_eVA"] = float(np.sqrt(np.mean((r.reshape(-1, 3) ** 2).sum(1))))
    out[f"round5_P{P}_residual_max_eVA"] = float(np.linalg.norm(r.reshape(-1, 3), axis=1).max())
out["lattice_change_5GPa_rel"] = float(np.linalg.norm(L5t - L5o) / np.linalg.norm(L5o))
print(json.dumps(out, indent=1))

# sensitivity of the actual share to the residual forces of the tight endpoints
B = kernel(f4t, L5t)
df = f5t - f4t
df -= np.round(df)
u = Q @ (df @ L5t).ravel()
r4 = outcar_forces(TIGHT_DIR / "spinel/p4/relax/OUTCAR")
r5 = outcar_forces(TIGHT_DIR / "spinel/p5/relax/OUTCAR")
d4, d5 = apply_h_pinv(r4), apply_h_pinv(r5)
sens = {"tight_P4_final_Fmax": float(np.linalg.norm(r4.reshape(-1, 3), axis=1).max()),
        "tight_P5_final_Fmax": float(np.linalg.norm(r5.reshape(-1, 3), axis=1).max()),
        "|H+ r4|_A": float(np.linalg.norm(d4)), "|H+ r5|_A": float(np.linalg.norm(d5)),
        "|u_t|_A": float(np.linalg.norm(u)),
        "share_u": pres(u, B), "share_u_corrected(u+d5-d4)": pres(u + d5 - d4, B),
        "share_u+d5+d4": pres(u + d5 + d4, B), "share_u-d5-d4": pres(u - d5 - d4, B)}
# worst case: residual displacement of same norm fully along bond-preserving dir aligned
v = u.reshape(-1, 3)[net]
v = v - v.mean(0)
v = v.ravel()
pp = v - B.T @ (B @ v)
e = pp / np.linalg.norm(pp)
eps = np.linalg.norm(d4) + np.linalg.norm(d5)
for s in (1, -1):
    w2 = v + s * eps * e
    p2 = w2 - B.T @ (B @ w2)
    sens["worst_share_sign%+d" % s] = float(p2 @ p2 / (w2 @ w2))
print("SENS", json.dumps(sens, indent=1))
