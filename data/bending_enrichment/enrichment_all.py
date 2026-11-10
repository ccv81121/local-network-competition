"""Bond-preserving enrichment for every system with a finite-difference force-constant matrix.
Share(x) = |P x|^2/|x|^2 on the centred B-X network part of x, P = projector onto the kernel of the first-order
B-X bond Jacobian (bond vectors from the start fractional coordinates in the end lattice), as in
spinel_pressure_path/bending_shares.py and sector_force_balance_analyze.py.
Enrichment = share(response) / share(drive f), for the predicted H^-1 f and for the actual relaxation u.
Code paths:
  full     - full 3N x 3N central-difference H, translations removed (tight spinel, CaTiO3, CaSnO3)
  symsec   - symmetric-sector H_s = -S^T dF/d(S) (sector_force_balance/; k08, Ca2SnTiO6, Ca8Sn6Ti2O24)
Reads the force constants and structures of the neighbouring folders (paths below); writes enrichment_all_systems.csv
and enrichment_all_systems.json next to the script and prints one line per system (numpy only).
usage: python enrichment_all.py"""
import csv, json
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[0]
FORCE_BALANCE_DIR = DATA_DIR / "force_balance_reference"
TIGHT_FORCE_BALANCE_DIR = DATA_DIR / "spinel_pressure_path" / "force_balance"
FULL_FORCE_BALANCE_DIR = SCRIPT_DIR / "full_force_balance"
SECTOR_FORCE_BALANCE_DIR = SCRIPT_DIR / "sector_force_balance"


def kernel(f_start, L, bonds, net):
    col = {int(a): c for c, a in enumerate(net)}
    vec = np.array([(f_start[j] - f_start[i] + np.array(img)) @ L for i, j, img in bonds])
    J = np.zeros((len(bonds), 3 * len(net)))
    for r, (i, j, _) in enumerate(bonds):
        e = vec[r] / np.linalg.norm(vec[r])
        J[r, 3 * col[i]:3 * col[i] + 3] = -e
        J[r, 3 * col[j]:3 * col[j] + 3] = e
    _, s, Vt = np.linalg.svd(J, full_matrices=False)
    return Vt[: int(np.sum(s > s[0] * 1e-10))], J


def share(u, B, net):
    v = u.reshape(-1, 3)[net]
    v = v - v.mean(axis=0)
    v = v.ravel()
    p = v - B.T @ (B @ v)
    return float(p @ p / (v @ v))


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def full_H(read_forces, N, delta):
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
    return Q, V[:, keep], w[keep]


def finish(name, topo, path, interval, L, f1, f2, f0, u_pred, bonds, net, Q, extra):
    df = f2 - f1
    df -= np.round(df)
    u = Q @ (df @ L).ravel()
    B, _ = kernel(f1, L, bonds, net)
    drive, predicted, actual = share(f0, B, net), share(u_pred, B, net), share(u, B, net)
    return {"system": name, "topology": topo, "interval_GPa": interval, "H_type": path,
            "n_atoms": len(f1), "n_network": len(net), "n_bonds": len(bonds),
            "bond_preserving_dof": 3 * len(net) - B.shape[0],
            "drive_share": drive, "predicted_share": predicted, "actual_share": actual,
            "enrichment_predicted": predicted / drive, "enrichment_actual": actual / drive,
            "cos_pred_actual": cos(u_pred, u), "u_norm_A": float(np.linalg.norm(u)), **extra}


rows = []

# tight spinel MgSc0.5Sb1.5Se4 (k12), 4->5 GPa: full H of force_balance_reference + tight drive
# (as the "tight" entry of bending_shares.py)
meta = json.loads((TIGHT_FORCE_BALANCE_DIR / "meta.json").read_text())
species = np.repeat(meta["species"], meta["counts"])
N = len(species)
Q, V, w = full_H(lambda c: np.load(FORCE_BALANCE_DIR / "forces" / f"{c}.npy"), N, meta["delta_A"])
f0 = np.load(TIGHT_FORCE_BALANCE_DIR / "forces" / "ref.npy").ravel()
bonds = json.loads((FORCE_BALANCE_DIR / "inputs" / "pressure_bond_accommodation.json").read_text())["bonds"]
net = np.flatnonzero(np.isin(species, ["Sc", "Sb", "Se"]))
rows.append(finish("MgSc0.5Sb1.5Se4 (k12, tight)", "edge-sharing spinel", "full", "4-5",
                   np.array(meta["lattice_5GPa"]), np.array(meta["frac_start"]), np.array(meta["frac_5GPa"]),
                   f0, V @ ((V.T @ (Q @ f0)) / w), bonds, net, Q, {}))

# CaTiO3, CaSnO3 (Pnma), 0->5 GPa: full H
for system, case_name in (("CaTiO3", "ca_tio3"), ("CaSnO3", "ca_sno3")):
    case_dir = FULL_FORCE_BALANCE_DIR / case_name / "p0_p5"
    meta = json.loads((case_dir / "meta.json").read_text())
    N = sum(meta["counts"])

    def read_forces(c):
        return np.load(case_dir / "hessian" / c / "forces.npy")

    Q, V, w = full_H(read_forces, N, meta["delta_A"])
    f0 = Q @ read_forces("ref").ravel()
    bonds = meta["bonds"]
    net = np.array(sorted({b[0] for b in bonds} | {b[1] for b in bonds}))
    rows.append(finish(system, "corner-sharing perovskite", "full", "0-5", np.array(meta["lattice_P2"]),
                       np.array(meta["frac_start"]), np.array(meta["frac_P2"]), f0, V @ ((V.T @ f0) / w), bonds, net,
                       Q, {}))

# sector_force_balance systems, 0->5 GPa: symmetric-sector H (as sector_force_balance_analyze.py)
for system, label, topo in (("spinel_k08", "MgScSbSe4 (k08 ordering)", "edge-sharing spinel"),
                            ("ca2_sn_tio6_rs", "Ca2SnTiO6 (rock-salt)", "corner-sharing perovskite"),
                            ("ca8_sn6_ti2o24", "Ca8Sn6Ti2O24", "corner-sharing perovskite")):
    case_dir = SECTOR_FORCE_BALANCE_DIR / system / "p0_p5"
    meta = json.loads((case_dir / "meta.json").read_text())
    S = np.load(case_dir / "S.npy")
    N = sum(meta["counts"])
    delta = meta["delta_A"]

    def read_forces(c):
        return np.load(case_dir / "hessian" / c / "forces.npy").ravel()

    ns = S.shape[1]
    Hs = np.zeros((ns, ns))
    for k in range(ns):
        Hs[k] = -(S.T @ (read_forces(f"s{k:03d}p") - read_forces(f"s{k:03d}m"))) / (2 * delta)
    Hs = 0.5 * (Hs + Hs.T)
    drive_forces = read_forces("ref")
    fs = S.T @ drive_forces
    T = np.zeros((3 * N, 3))
    for a in range(3):
        T[a::3, a] = 1 / np.sqrt(N)
    Q = np.eye(3 * N) - T @ T.T
    rows.append(finish(label, topo, "symsec", "0-5", np.array(meta["lattice_P2"]), np.array(meta["frac_start"]),
                       np.array(meta["frac_P2"]), drive_forces, S @ np.linalg.solve(Hs, fs), meta["bonds"],
                       np.array(meta["network"]), Q,
                       {"n_symmetric_dof": ns,
                        "drive_symmetry_residual": float(np.linalg.norm(drive_forces - S @ fs)
                                                         / np.linalg.norm(drive_forces)),
                        "n_negative_eig": int(np.sum(np.linalg.eigvalsh(Hs) < 0))}))

keys = list(rows[0]) + ["n_symmetric_dof", "drive_symmetry_residual", "n_negative_eig"]
with open(SCRIPT_DIR / "enrichment_all_systems.csv", "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=keys)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
(SCRIPT_DIR / "enrichment_all_systems.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False),
                                                        encoding="utf-8")
for r in rows:
    print(f"{r['system']:34s} {r['H_type']:6s} drive {100*r['drive_share']:6.2f}%  "
          f"pred {100*r['predicted_share']:6.2f}%  act {100*r['actual_share']:6.2f}%  "
          f"E_pred {r['enrichment_predicted']:6.2f}  E_act {r['enrichment_actual']:6.2f}  "
          f"cos {r['cos_pred_actual']:.4f}")
