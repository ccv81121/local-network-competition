"""Bending-stiffness check. Bending carries a small share of ||H||^2, so the three bending stiffnesses are also
fitted on the bond-preserving (stretch-free) block of H alone, and the full-fit ratios are bootstrapped over atoms.
Writes outputs/bend_block_fit.json; local_stiffness_model.json is regenerated unchanged (checked by hash).
usage: python bend_block_fit.py"""
import contextlib
import hashlib
import io
import json
import runpy
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
output_files = sorted((ROOT / "outputs").glob("*"))
before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output_files}
sys.argv = [sys.argv[0]]
with contextlib.redirect_stdout(io.StringIO()):
    model = runpy.run_path(str(ROOT / "scripts" / "local_stiffness_model.py"), run_name="x")
after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output_files}
assert before == after, "local model outputs changed"

types, G, p, target, Q, coords = model["types"], model["G"], model["p"], model["target"], model["Q"], model["coords"]
species = model["species"]
target_norm2 = np.linalg.norm(target) ** 2
Gq = [Q @ Gt @ Q for Gt in G]
Hb = sum(pi * Gt for pi, Gt, t in zip(p, Gq, types) if "bend" in t)

# bond-preserving subspace: null space of all metal-Se stretch rows, translations removed
stretch_rows = np.array([r for t in types if "stretch" in t for r in coords[t]])
_, s, Vt = np.linalg.svd(stretch_rows)
null_basis = Vt[int(np.sum(s > s[0] * 1e-10)):].T
T = np.kron(np.ones((len(species), 1)), np.eye(3)) / np.sqrt(len(species))
null_basis = null_basis - T @ (T.T @ null_basis)
U, sv, _ = np.linalg.svd(null_basis, full_matrices=False)
preserving_basis = U[:, sv > 1e-8]
target_block = preserving_basis.T @ target @ preserving_basis
bend_types = [t for t in types if "bend" in t]
A = np.stack([(preserving_basis.T @ G[types.index(t)] @ preserving_basis).ravel() for t in bend_types], axis=1)
block_bend, _ = nnls(A, target_block.ravel())
block_r2 = 1 - np.linalg.norm(A @ block_bend - target_block.ravel()) ** 2 / np.linalg.norm(target_block) ** 2

rng = np.random.default_rng(0)
ratios = []
for _ in range(200):
    drawn_atoms = np.sort(rng.choice(len(species), len(species), replace=True))
    idx = np.concatenate([np.arange(3 * i, 3 * i + 3) for i in drawn_atoms])
    A_boot = np.stack([Gt[np.ix_(idx, idx)].ravel() for Gt in Gq], axis=1)
    boot_fit, _ = nnls(A_boot, target[np.ix_(idx, idx)].ravel())
    ratios.append([boot_fit[types.index(f"{x} bend")] / boot_fit[types.index(f"{x} stretch")]
                   for x in ("Sc", "Sb", "Mg")])
ratios = np.array(ratios)

out = {
    "share_of_H_norm2": {t: float(np.linalg.norm(pi * Gt) ** 2 / target_norm2) for t, pi, Gt in zip(types, p, Gq)},
    "bend_share_of_H_norm2": float(np.linalg.norm(Hb) ** 2 / target_norm2),
    "full_fit_residual": float(1 - model["r2"]),
    "bond_preserving_block_dim": int(preserving_basis.shape[1]),
    "block_fit_bend_eV_A2": {t: float(x) for t, x in zip(bend_types, block_bend)},
    "block_fit_R2": float(block_r2),
    "bend_to_stretch_block_fit": {x: float(block_bend[bend_types.index(f"{x} bend")] / p[types.index(f"{x} stretch")])
                                  for x in ("Sc", "Sb", "Mg")},
    "bend_to_stretch_full_fit": {x: float(p[types.index(f"{x} bend")] / p[types.index(f"{x} stretch")])
                                 for x in ("Sc", "Sb", "Mg")},
    "bootstrap_atoms_200_full_fit_ratio_95": {x: np.percentile(ratios[:, k], [2.5, 97.5]).tolist()
                                              for k, x in enumerate(("Sc", "Sb", "Mg"))},
}
(ROOT / "outputs" / "bend_block_fit.json").write_bytes((json.dumps(out, indent=1) + "\n").encode())
print(json.dumps(out, indent=1))
