"""After the ISIF=2 relaxation: split the network displacement s1 -> relax into bond-changing and bond-preserving parts
(first-order B-X bond Jacobian, six nearest X of every B in the s1 geometry, network centroid fixed; as in
sector_force_balance_prepare.py) and write the four readout states. Also reports the bond-preserving share of u and of
the drive (s1 forces, first ionic step).
CASE_DIR holds s1/, relax/CONTCAR and relax/OUTCAR_c1 of the VASP runs (not deposited).
usage: python alloy_decomposition.py CASE_DIR"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

CASE_DIR = Path(sys.argv[1])
meta = json.loads((CASE_DIR / "meta.json").read_text())
sys.path.insert(0, str(Path(__file__).resolve().parent))
from alloy_prepare import read_poscar, STATIC_INCAR  # noqa: E402


def frames(outcar, N):
    blocks = Path(outcar).read_text().split("TOTAL-FORCE (eV/Angst)")[1:]
    return [np.array([[float(x) for x in line.split()[3:6]] for line in t.splitlines()[2:2 + N]]) for t in blocks]


lat, sp, f1 = read_poscar(CASE_DIR / "s1" / "POSCAR")
latr, spr, fr = read_poscar(CASE_DIR / "relax" / "CONTCAR")
assert sp == spr and np.allclose(lat, latr, atol=1e-6)
N = len(sp)
sp = np.array(sp)
net = np.flatnonzero(np.isin(sp, meta["B"] + [meta["X"]]))
B_idx = np.flatnonzero(np.isin(sp, meta["B"]))
X_idx = np.flatnonzero(sp == meta["X"])
df = fr - f1
df -= np.round(df)
disp = df @ lat
shift = disp[net].mean(axis=0)
col = {int(a): k for k, a in enumerate(net)}
cart_s1 = f1 @ lat
bonds = []
for b in B_idx:
    dv = f1[X_idx] - f1[b]
    dv -= np.round(dv)
    dc = dv @ lat
    dist = np.linalg.norm(dc, axis=1)
    for k in np.argsort(dist)[:6]:
        bonds.append((int(b), int(X_idx[k]), dc[k], float(dist[k])))
J = np.zeros((len(bonds), 3 * len(net)))
for row, (b, x, v, length) in enumerate(bonds):
    e = v / length
    J[row, 3 * col[b]:3 * col[b] + 3] = -e
    J[row, 3 * col[x]:3 * col[x] + 3] = e
_, s, Vt = np.linalg.svd(J, full_matrices=False)
R = Vt[: int(np.sum(s > s[0] * 1e-10))]


def split(v):
    v = v.reshape(-1, 3)[net]
    v = v - v.mean(axis=0)
    v = v.ravel()
    par = R.T @ (R @ v)
    return v, v - par, par


u, u_bp, u_bc = split(disp)
F = frames(CASE_DIR / "relax" / "OUTCAR_c1", N)[0]
f, f_bp, f_bc = split(F)


def norm_share(a, b):
    return float(b @ b / (a @ a))


geo = {"case": meta["case"], "n_atoms": N, "n_network": int(len(net)), "n_bonds": len(bonds),
       "bond_rank": int(R.shape[0]), "bond_preserving_dof": 3 * len(net) - int(R.shape[0]),
       "bond_lengths_min_max_A": [min(b[3] for b in bonds), max(b[3] for b in bonds)],
       "u_rms_A": float(np.sqrt(u @ u / len(net))), "u_max_A": float(np.abs(u).max()),
       "A_rms_A": float(np.sqrt(np.mean(np.sum((disp[sp == meta["A"]] - shift) ** 2, axis=1)))),
       "u_bond_preserving_share": norm_share(u, u_bp), "drive_bond_preserving_share": norm_share(f, f_bp),
       "enrichment_actual_over_drive": norm_share(u, u_bp) / norm_share(f, f_bp),
       "drive_max_force_eV_A": float(np.abs(F).max()),
       "n_ionic_cycle1": len(frames(CASE_DIR / "relax" / "OUTCAR_c1", N))}
inv = np.linalg.inv(lat)
is_A = sp == meta["A"]
base = cart_s1.copy()
base[is_A] = cart_s1[is_A] + disp[is_A] - shift
states = {"state00": np.zeros_like(u), "state10": u_bp, "state01": u_bc, "state11": u}
for name, du in states.items():
    cart = base.copy()
    cart[net] = cart_s1[net] + du.reshape(-1, 3)
    frs = cart @ inv
    d = CASE_DIR / "fourstate" / name
    d.mkdir(parents=True, exist_ok=True)
    el = list(dict.fromkeys(sp.tolist()))
    n = [int(np.sum(sp == e)) for e in el]
    poscar_lines = ([f"{meta['case']} {name}", "1.0"] + ["  ".join(f"{x:.12f}" for x in v) for v in lat]
                    + ["  ".join(el), "  ".join(map(str, n)), "Direct"])
    poscar_lines += ["  ".join(f"{x:.12f}" for x in row) for row in frs]
    (d / "POSCAR").write_text("\n".join(poscar_lines) + "\n")
    (d / "INCAR").write_text(f"SYSTEM = {meta['case']} {name}\n" + STATIC_INCAR)
    shutil.copy(CASE_DIR / "s1" / "KPOINTS", d / "KPOINTS")
    shutil.copy(CASE_DIR / "s1" / "POTCAR", d / "POTCAR")
(CASE_DIR / "geometry.json").write_text(json.dumps(geo, indent=1))
print(json.dumps(geo, indent=1))
