"""Control: response of the local model to the same drive when the bend/stretch anisotropy is removed or varied.

The three bending stiffnesses are scaled by 0.5, 1, 2 and 4, or set equal to the stretching stiffness of the same ion
("isotropic"); the bond-preserving fraction of the force-balance response is reported for each case.
Runs local_stiffness_model.py on this folder; writes outputs/anisotropy_control.json.
usage: python anisotropy_control.py
"""
import contextlib, io, json, runpy, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.argv = [sys.argv[0], str(ROOT)]
with contextlib.redirect_stdout(io.StringIO()):
    model = runpy.run_path(str(ROOT / "scripts" / "local_stiffness_model.py"))
p, G, types, Q, f, preserving_fraction = model["p"], model["G"], model["types"], model["Q"], model["f"], model["pf"]
out = {}
for scale in (0.5, 1.0, 2.0, 4.0, "isotropic"):
    stiffnesses = []
    for t, pi in zip(types, p):
        if "bend" in t:
            stretch_stiffness = p[types.index(t.replace("bend", "stretch"))]
            stiffnesses.append(stretch_stiffness if scale == "isotropic" else pi * scale)
        else:
            stiffnesses.append(pi)
    Hm = Q @ sum(stiffness * Gt for stiffness, Gt in zip(stiffnesses, G)) @ Q
    w, V = np.linalg.eigh(Hm)
    keep = np.argsort(np.abs(w))[3:]
    w, V = w[keep], V[:, keep]
    u = V @ ((V.T @ f) / w)
    out[str(scale)] = {"bond_preserving_fraction_of_response": preserving_fraction(u)}
(ROOT / "outputs" / "anisotropy_control.json").write_bytes((json.dumps(out, indent=1) + "\n").encode())
print(json.dumps(out, indent=1))
