"""Bond-preserving enrichment for the two alloy intervals (no new DFT).
kernel() and share() are identical to bending_enrichment/enrichment_all.py; bonds and network from
alloy_leverage.geometry (six nearest Se of each Sc/Sb, bond vectors from the F_aff fractional coordinates in the S_b
lattice).
Class: E1 eps_act >= 9; E2 3 <= eps_act < 9; E3 eps_act < 3.
Needs the path-split POSCARs of the cluster (alloy_leverage.geometry) and runs/<iv>/{u_pred.npy,geom.json} of ../alloy_leverage
(not deposited).
usage: python enrichment_factor.py
"""
import json, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
ALLOY_LEVERAGE_DIR = Path(__file__).resolve().parents[1] / "alloy_leverage"
sys.path.insert(0, str(ALLOY_LEVERAGE_DIR))
from alloy_leverage import INTERVALS, geometry

OUTPUT_DIR = Path(__file__).resolve().parent


def kernel(f_start, L, bonds, net):
    col = {int(atom): c for c, atom in enumerate(net)}
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
    v_bp = v - B.T @ (B @ v)
    return float(v_bp @ v_bp / (v @ v))


results = {}
for iv in INTERVALS:
    geom = geometry(iv)
    run_dir = ALLOY_LEVERAGE_DIR / "runs" / iv
    f, up = np.load(run_dir / "f.npy"), np.load(run_dir / "u_pred.npy")
    B, _ = kernel(geom["fF"], geom["L"], geom["bonds"], geom["net"])
    drive_share, predicted_share, actual_share = (share(f, B, geom["net"]), share(up, B, geom["net"]),
                                                  share(geom["u_act"], B, geom["net"]))
    enrichment_class = "E1" if actual_share / drive_share >= 9 else ("E2" if actual_share / drive_share >= 3 else "E3")
    results[iv] = {"n_network": len(geom["net"]), "n_bonds": len(geom["bonds"]),
                   "bond_preserving_dof": 3 * len(geom["net"]) - B.shape[0],
                   "drive_share": drive_share, "predicted_share": predicted_share, "actual_share": actual_share,
                   "enrichment_predicted": predicted_share / drive_share,
                   "enrichment_actual": actual_share / drive_share, "class": enrichment_class,
                   "GEOM_u_act_b_share_reference": json.loads((run_dir / "geom.json").read_text())["u_act_b_share"]}
    print(f"{iv} drive {100*drive_share:.3f}%  pred {100*predicted_share:.2f}%  "
          f"act {100*actual_share:.2f}%  eps_pred {predicted_share/drive_share:.2f}  "
          f"eps_act {actual_share/drive_share:.2f}  {enrichment_class}")
(OUTPUT_DIR / "eps_result.json").write_text(json.dumps(results, indent=1))
