"""Zero-parameter force-balance prediction of the 4->5 GPa internal accommodation.

Inputs (all computed at the start structure r_aff = 4-GPa fractional coordinates in the 5-GPa lattice):
  f0 : DFT forces at r_aff (the drive created by the imposed lattice change)
  H  : DFT force-constant matrix at r_aff (central differences), i.e. K + L
Prediction: u_pred = H^+ f0 on the nontranslational subspace (one harmonic force-balance step).
Nothing is fitted; the actual 5-GPa relaxed structure is used only for comparison.
Reads meta.json, forces/ and the bond list (inputs/pressure_bond_accommodation.json); writes
outputs/force_balance_prediction.json and outputs/poscar_pred (the predicted structure).
usage: python analyze.py [FORCE_BALANCE_DIR]   (default: the folder above scripts/)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" if (ROOT / "outputs").is_dir() else ROOT
BONDS_FILE = next(p for p in (ROOT / "pressure_bond_accommodation.json",
                              ROOT / "inputs" / "pressure_bond_accommodation.json") if p.exists())
meta = json.loads((ROOT / "meta.json").read_text())
L5 = np.array(meta["lattice_5GPa"])
f4 = np.array(meta["frac_start"])
f5 = np.array(meta["frac_5GPa"])
species = np.repeat(meta["species"], meta["counts"])
N = len(species)
delta = meta["delta_A"]


def read_forces(name):
    return np.load(ROOT / "forces" / f"{name}.npy")


H = np.zeros((3 * N, 3 * N))
for i in range(N):
    for a, ax in enumerate("xyz"):
        H[3 * i + a] = -(read_forces(f"d{i:02d}{ax}p") - read_forces(f"d{i:02d}{ax}m")).ravel() / (2 * delta)
asym = float(np.linalg.norm(H - H.T) / np.linalg.norm(H))
H = 0.5 * (H + H.T)
T = np.zeros((3 * N, 3))
for a in range(3):
    T[a::3, a] = 1 / np.sqrt(N)
Q = np.eye(3 * N) - T @ T.T
asr = float(np.linalg.norm(H @ T) / np.linalg.norm(H))
w, V = np.linalg.eigh(Q @ H @ Q)
keep = np.argsort(np.abs(w))[3:]  # drop the three translations
w_k, V_k = w[keep], V[:, keep]

f0 = read_forces("ref").ravel()
f0_q = Q @ f0
u_pred = V_k @ ((V_k.T @ f0_q) / w_k)

df = f5 - f4
df -= np.round(df)
u_act = (df @ L5).ravel()
u_act = Q @ u_act


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def rms(u, mask=None):
    v = u.reshape(-1, 3) if mask is None else u.reshape(-1, 3)[mask]
    return float(np.sqrt(np.mean(np.sum(v ** 2, axis=1))))


# bond-preserving decomposition: B spans the row space of the bond-length Jacobian J (bond-changing motion)
bonds = json.loads(BONDS_FILE.read_text())["bonds"]
net = np.flatnonzero(np.isin(species, ["Sc", "Sb", "Se"]))
net_column = {int(a): c for c, a in enumerate(net)}
bond_vectors = np.array([(f4[j] - f4[i] + np.array(img)) @ L5 for i, j, img in bonds])
J = np.zeros((len(bonds), 3 * len(net)))
for r, (i, j, _) in enumerate(bonds):
    e = bond_vectors[r] / np.linalg.norm(bond_vectors[r])
    J[r, 3 * net_column[i]:3 * net_column[i] + 3] = -e
    J[r, 3 * net_column[j]:3 * net_column[j] + 3] = e
_, s, Vt = np.linalg.svd(J, full_matrices=False)
B = Vt[: int(np.sum(s > s[0] * 1e-10))]


def net_part(u):
    v = u.reshape(-1, 3)[net]
    v = v - v.mean(axis=0)
    return v.ravel()


def preserving_fraction(u):
    v = net_part(u)
    stretching = B.T @ (B @ v)
    preserving = v - stretching
    return float(preserving @ preserving / (v @ v))


# effective stiffness of the network on the two subspaces (network block of H, translation removed)
idx = np.concatenate([[3 * a, 3 * a + 1, 3 * a + 2] for a in net])
Hn = H[np.ix_(idx, idx)]
P_str = B.T @ B
Tn = np.zeros((3 * len(net), 3))
for a in range(3):
    Tn[a::3, a] = 1 / np.sqrt(len(net))
P_pre = np.eye(3 * len(net)) - P_str - Tn @ Tn.T


def mean_stiff(P):
    w_, V_ = np.linalg.eigh(P)
    basis = V_[:, w_ > 0.5]
    ev = np.linalg.eigvalsh(basis.T @ Hn @ basis)
    return float(np.mean(ev)), float(np.min(ev)), float(np.max(ev))


out = {
    "force_constants": {"relative_asymmetry_before_symmetrization": asym, "acoustic_sum_rule_residual": asr,
                        "nontranslational_eigenvalues_eV_A2": {"min": float(w_k.min()), "max": float(w_k.max()),
                                                               "n_negative": int(np.sum(w_k < 0))}},
    "drive": {"rms_force_eV_A": rms(f0_q), "max_force_eV_A": float(np.abs(f0_q.reshape(-1, 3)).max())},
    "displacement": {
        "cosine_pred_vs_actual": cos(u_pred, u_act),
        "norm_ratio_pred_over_actual": float(np.linalg.norm(u_pred) / np.linalg.norm(u_act)),
        "rms_actual_A": rms(u_act), "rms_pred_A": rms(u_pred),
        "rms_error_A": rms(u_pred - u_act),
        "by_species": {sp: {"rms_actual_A": rms(u_act, species == sp), "rms_pred_A": rms(u_pred, species == sp),
                            "cosine": cos(u_pred.reshape(-1, 3)[species == sp].ravel(),
                                          u_act.reshape(-1, 3)[species == sp].ravel())}
                       for sp in meta["species"]},
        "network_cosine": cos(net_part(u_pred), net_part(u_act)),
    },
    "bond_preserving_fraction": {"actual": preserving_fraction(u_act), "predicted": preserving_fraction(u_pred),
                                 "of_the_drive_force": preserving_fraction(f0_q)},
    "bond_length_changes": {"pearson_pred_vs_actual": float(np.corrcoef(J @ net_part(u_pred),
                                                                        J @ net_part(u_act))[0, 1]),
                            "rms_actual_A": float(np.sqrt(np.mean((J @ net_part(u_act)) ** 2))),
                            "rms_pred_A": float(np.sqrt(np.mean((J @ net_part(u_pred)) ** 2)))},
    "network_stiffness_eV_A2": {"bond_changing_subspace_mean_min_max": mean_stiff(P_str),
                                "bond_preserving_subspace_mean_min_max": mean_stiff(P_pre)},
    "harmonic_energy_lowering_meV_per_cell": {"predicted": float(-0.5 * f0_q @ u_pred * 1000)},
}
if (ROOT / "forces" / "r5.npy").exists():
    E = (json.loads((ROOT / "results" / "ref.json").read_text())["E_eV"],
         json.loads((ROOT / "results" / "r5.json").read_text())["E_eV"])
    out["harmonic_energy_lowering_meV_per_cell"]["dft_E5_minus_Eaff"] = (E[1] - E[0]) * 1000
    out["residual_force_at_actual_5GPa_rms_eV_A"] = rms(Q @ read_forces("r5").ravel())
frac_pred = f4 + (u_pred.reshape(-1, 3) @ np.linalg.inv(L5))
lines = ["predicted by force balance", "1.0"] + ["  ".join(f"{x:.16f}" for x in v) for v in L5]
lines += (["  ".join(meta["species"]), "  ".join(map(str, meta["counts"])), "Direct"]
          + ["  ".join(f"{x:.16f}" for x in r) for r in frac_pred])
(OUTPUT_DIR / "poscar_pred").write_bytes(("\n".join(lines) + "\n").encode())  # LF endings for VASP
(OUTPUT_DIR / "force_balance_prediction.json").write_bytes((json.dumps(out, indent=1) + "\n").encode())
print(json.dumps(out, indent=1))
