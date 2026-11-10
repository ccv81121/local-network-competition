"""Mode-resolved view of the force-balance response.

For the 10, 20, 30 and 45 softest nontranslational modes of H: the largest eigenvalue, the shares of the drive and of
the response carried by these modes, and their mean bond-preserving fraction.
Runs analyze.py on this folder; writes outputs/soft_mode_shares.json.
usage: python soft_modes.py
"""
import json, runpy, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.argv = [sys.argv[0], str(ROOT)]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    analysis = runpy.run_path(str(ROOT / "scripts" / "analyze.py"))
w, V, fq, preserving_fraction = analysis["w_k"], analysis["V_k"], analysis["f0_q"], analysis["preserving_fraction"]
order = np.argsort(w)
w, V = w[order], V[:, order]
drive_coeff = V.T @ fq
response_coeff = drive_coeff / w
out = {}
for n in (10, 20, 30, 45):
    out[n] = {"max_eigenvalue_eV_A2": float(w[n - 1]),
              "drive_share": float(np.sum(drive_coeff[:n] ** 2) / np.sum(drive_coeff ** 2)),
              "response_share": float(np.sum(response_coeff[:n] ** 2) / np.sum(response_coeff ** 2)),
              "mean_bond_preserving_fraction_of_modes": float(np.mean([preserving_fraction(V[:, k])
                                                                       for k in range(n)]))}
(ROOT / "outputs" / "soft_mode_shares.json").write_bytes((json.dumps(out, indent=1) + "\n").encode())
print(json.dumps(out, indent=1))
