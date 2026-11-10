"""Geometric factor |u_b|/|u_r| (Ni/O part, La excluded; four_state_build.py definition) of the force-balance
prediction u_pred = B^T H^-1 F_ref against the actual relaxation, per interval."""
import json
from pathlib import Path

import numpy as np
from ase.io import read

import la327_network as net
from force_balance_analyze import block, load

HERE = Path(__file__).parent


def split(u, aff):
    bond_list = net.bonds(aff)
    C = net.jacobian(aff, bond_list)
    mask = net.network_mask(aff)
    Cn = C[:, mask]
    uN = np.where(mask, u, 0.0)
    ur = np.zeros_like(u)
    ur[mask] = np.linalg.pinv(Cn) @ (Cn @ uN[mask])
    ub = uN - ur
    return np.linalg.norm(ub), np.linalg.norm(ur), np.linalg.norm(u - uN)


modes = json.loads((HERE / "force_balance" / "modes.json").read_text())
for name, meta in modes.items():
    aff = read(HERE / "force_balance" / "calculations" / name.lower() / "ref" / "POSCAR")
    ref = load(name, "ref")
    B, H, _ = block(name, meta, "")
    f = ref["f"] - np.repeat(ref["f"].reshape(-1, 3).mean(0)[None], len(aff), 0).ravel()
    u_pred = B.T @ np.linalg.solve((H + H.T) / 2, B @ f)
    u_act = np.array(meta["u_actual"])
    ba, ra, la = split(u_act, aff)
    bp, rp, lp = split(u_pred, aff)
    print(f"{name}: actual |u_b| {ba:.4f} |u_r| {ra:.4f} |u_La| {la:.4f} G {ba / ra:.3f} | "
          f"pred |u_b| {bp:.4f} |u_r| {rp:.4f} |u_La| {lp:.4f} G {bp / rp:.3f} | "
          f"G dev {100 * (bp / rp / (ba / ra) - 1):+.1f}%  |u_b| ratio {bp / ba:.3f}  |u_r| ratio {rp / ra:.3f}")
