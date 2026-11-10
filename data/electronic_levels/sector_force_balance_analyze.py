"""Symmetric-sector force-balance prediction for one interval (inputs from sector_force_balance_prepare.py --write).
H_s[k,l] = -S_l . (F(+d S_k) - F(-d S_k)) / 2d ; f_s = S^T F(r_aff) ; u_pred = S H_s^-1 f_s.
Reports the bond-preserving share of the drive, of the prediction and of the actual relaxation, cosine, energy, and the
symmetric-sector stiffness spectrum with the bond-preserving content of each mode.
Reads meta.json, S.npy and hessian/*/forces.npy of the case folder; writes poscar_pred, Hs.npy and
force_balance_sym.json there.
usage: python sector_force_balance_analyze.py CASE_DIR   (a sector_force_balance/<system>/p0_p5 folder)"""
import json, sys
from pathlib import Path
import numpy as np

CASE_DIR = Path(sys.argv[1])
meta = json.loads((CASE_DIR / "meta.json").read_text())
S = np.load(CASE_DIR / "S.npy")
L2 = np.array(meta["lattice_P2"])
f1 = np.array(meta["frac_start"])
f2 = np.array(meta["frac_P2"])
delta = meta["delta_A"]
species = np.repeat(meta["species"], meta["counts"])
N = len(species)
net = np.array(meta["network"])
bonds = meta["bonds"]


def read_forces(c):
    return np.load(CASE_DIR / "hessian" / c / "forces.npy").ravel()


def read_energy(c):
    return json.loads((CASE_DIR / "hessian" / c / "result.json").read_text())["E_eV"]


ns = S.shape[1]
Hs = np.zeros((ns, ns))
for k in range(ns):
    Hs[k] = -(S.T @ (read_forces(f"s{k:03d}p") - read_forces(f"s{k:03d}m"))) / (2 * delta)
asym = float(np.linalg.norm(Hs - Hs.T) / np.linalg.norm(Hs))
Hs = 0.5 * (Hs + Hs.T)
F0 = read_forces("ref")
fs = S.T @ F0
sym_resid = float(np.linalg.norm(F0 - S @ fs) / np.linalg.norm(F0))
w, V = np.linalg.eigh(Hs)
u_sector = np.linalg.solve(Hs, fs)
u_pred = S @ u_sector
df = f2 - f1
df -= np.round(df)
u_act = (df @ L2).ravel()
T = np.zeros((3 * N, 3))
for a in range(3):
    T[a::3, a] = 1 / np.sqrt(N)
u_act = u_act - T @ (T.T @ u_act)
net_column = {int(a): c for c, a in enumerate(net)}
bond_vectors = np.array([(f1[j] - f1[i] + np.array(img)) @ L2 for i, j, img in bonds])
J = np.zeros((len(bonds), 3 * len(net)))
for r, (i, j, _) in enumerate(bonds):
    e = bond_vectors[r] / np.linalg.norm(bond_vectors[r])
    J[r, 3 * net_column[i]:3 * net_column[i] + 3] = -e
    J[r, 3 * net_column[j]:3 * net_column[j] + 3] = e
_, s, Vt = np.linalg.svd(J, full_matrices=False)
bond_basis = Vt[: int(np.sum(s > s[0] * 1e-10))]


def network_part(u):
    v = u.reshape(-1, 3)[net]
    v = v - v.mean(axis=0)
    return v.ravel()


def preserving_share(u):
    v = network_part(u)
    p = v - bond_basis.T @ (bond_basis @ v)
    return float(p @ p / (v @ v)) if v @ v > 1e-30 else 0.0


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


c = V.T @ fs
response = c / w
modes = [{"k_eV_A2": round(float(w[m]), 4), "bond_preserving": round(preserving_share(S @ V[:, m]), 4),
          "drive_share": round(float(c[m] ** 2 / (c @ c)), 4),
          "response_share": round(float(response[m] ** 2 / (response @ response)), 4)} for m in range(ns)]
out = {"material": CASE_DIR.parent.name, "interval": CASE_DIR.name, "n_symmetric_dof": ns, "H_asymmetry": asym,
       "n_negative": int(np.sum(w < 0)),
       "drive_symmetry_residual": sym_resid,
       "bond_preserving_share": {"drive": preserving_share(F0), "predicted": preserving_share(u_pred),
                                 "actual": preserving_share(u_act)},
       "cosine_pred_vs_actual": cos(u_pred, u_act), "cosine_drive_vs_actual": cos(F0 - T @ (T.T @ F0), u_act),
       "norm_ratio_pred_over_actual": float(np.linalg.norm(u_pred) / np.linalg.norm(u_act)),
       "harmonic_energy_lowering_meV": float(-0.5 * fs @ u_sector * 1000), "modes_soft_to_stiff": modes}
if (CASE_DIR / "hessian" / "r2" / "result.json").exists():
    out["dft_E_P2_minus_E_aff_meV"] = (read_energy("r2") - read_energy("ref")) * 1000
frac_pred = f1 + u_pred.reshape(-1, 3) @ np.linalg.inv(L2)
lines = ["predicted by symmetric-sector force balance", "1.0"] + ["  ".join(f"{x:.16f}" for x in v) for v in L2]
lines += (["  ".join(meta["species"]), "  ".join(map(str, meta["counts"])), "Direct"]
          + ["  ".join(f"{x:.16f}" for x in r % 1.0) for r in frac_pred])
(CASE_DIR / "poscar_pred").write_bytes(("\n".join(lines) + "\n").encode())
np.save(CASE_DIR / "Hs.npy", Hs)
(CASE_DIR / "force_balance_sym.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "modes_soft_to_stiff"}, indent=1))
for m in modes[:8]:
    print(m)
