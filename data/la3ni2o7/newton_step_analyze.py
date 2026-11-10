"""Modified Newton step u2 = B^T H^-1 B Q f(r1) with the force-balance H of force_balance_analyze.py, and the refined
displacement u1 + u2 against the actual relaxation (cosine, amplitude ratio, geometric factor |u_b|/|u_r|).
Needs force_balance/forces/<i>_newton1.dat (interval in lower case). Prints one line per interval and writes
force_balance/newton_step_result.json."""
import json
from pathlib import Path

import numpy as np
from ase.io import read

from force_balance_analyze import load
from force_balance_geometric_factor import split
from newton_step_prepare import predicted

HERE = Path(__file__).parent
FORCE_BALANCE_DIR = HERE / "force_balance"

modes = json.loads((FORCE_BALANCE_DIR / "modes.json").read_text())
out = {}
for name, meta in modes.items():
    aff = read(FORCE_BALANCE_DIR / "calculations" / name.lower() / "ref" / "POSCAR")
    r1 = load(name, "newton1")
    if r1 is None:
        print(name, "newton1 missing")
        continue
    B, H, u1 = predicted(name, meta, aff, load(name, "ref"))
    f1 = r1["f"] - np.repeat(r1["f"].reshape(-1, 3).mean(0)[None], len(aff), 0).ravel()
    u2 = B.T @ np.linalg.solve(H, B @ f1)
    u = np.array(meta["u_actual"])
    ba, ra, _ = split(u, aff)
    res = {}
    for tag, v in (("u1", u1), ("u1+u2", u1 + u2)):
        bp, rp, _ = split(v, aff)
        res[tag] = dict(cos=float(v @ u / np.linalg.norm(v) / np.linalg.norm(u)),
                        amp=float(np.linalg.norm(v) / np.linalg.norm(u)),
                        G_dev=float(bp / rp / (ba / ra) - 1), ub_ratio=float(bp / ba), ur_ratio=float(rp / ra))
    res["f1_over_f0"] = float(np.linalg.norm(B @ f1) / np.linalg.norm(B @ (load(name, "ref")["f"])))
    out[name] = res
    print(name, "  ".join(f"{t}: cos {d['cos']:.4f} amp {d['amp']:.3f} G {100 * d['G_dev']:+.1f}% "
                          f"ub {d['ub_ratio']:.3f} ur {d['ur_ratio']:.3f}"
                          for t, d in res.items() if t != "f1_over_f0"),
          f" |Bf1|/|Bf0| {res['f1_over_f0']:.3f}")
(FORCE_BALANCE_DIR / "newton_step_result.json").write_text(json.dumps(out, indent=1))
