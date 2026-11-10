"""T2b evaluation: central-difference |dPi/deps| of the fixed relations along bond-preserving (bp) and
bond-changing (bc) directions, from features.csv produced by displace_structures.py. The scores are evaluated with the
numerical contract of ../t1_t2_ablation.py (executed up to the baseline scoring). Writes sensitivity_derivatives.csv
and sensitivity_results.json next to this script.
usage: python sensitivity_eval.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
T1_T2_SOURCE = (HERE.parent / "t1_t2_ablation.py").read_text(encoding="utf8")
namespace = {"__file__": str(HERE.parent / "t1_t2_ablation.py"), "__name__": "t2b"}
# Numerical contract and scores() of the ablation script; the ablation itself is not run.
exec(T1_T2_SOURCE[:T1_T2_SOURCE.index("base, Tb = scores(raw0)")], namespace)
raw0, ids, feat, scores = namespace["raw0"], list(namespace["ids"]), namespace["feat"], namespace["scores"]
features_table = pd.read_csv(HERE / "features.csv")
DISPLACEMENT_STEP = 0.01
out = {"per_structure": {}, "pooled": {}}
records = []
for sample_id, grp in features_table.groupby("sample_id"):
    row_index = ids.index(sample_id)
    ref = grp[grp.subspace == "ref"][feat].to_numpy()[0]
    assert np.allclose(ref, raw0[row_index], rtol=1e-9, atol=1e-12, equal_nan=True), sample_id

    def displaced_scores(vals):
        raw = raw0.copy()
        raw[row_index] = vals
        s = scores(raw)[0]
        return {k: float(s[k][row_index]) for k in s}

    for (sub, sd), q in grp[grp.subspace != "ref"].groupby(["subspace", "seed"]):
        plus = displaced_scores(q[q.sign == 1][feat].to_numpy()[0])
        minus = displaced_scores(q[q.sign == -1][feat].to_numpy()[0])
        records.append({"sample_id": sample_id, "subspace": sub, "seed": sd,
                        **{k: abs(plus[k] - minus[k]) / (2 * DISPLACEMENT_STEP) for k in plus}})
derivatives = pd.DataFrame(records)
derivatives.to_csv(HERE / "sensitivity_derivatives.csv", index=False)
for k in ("sp", "V", "C"):
    for sample_id, q in derivatives.groupby("sample_id"):
        bp, bc = q[q.subspace == "bp"][k].median(), q[q.subspace == "bc"][k].median()
        out["per_structure"].setdefault(sample_id, {})[k] = {"bp_median": bp, "bc_median": bc,
                                                              "ratio": bp / bc if bc else None}
    bp, bc = (derivatives[derivatives.subspace == "bp"][k].median(),
              derivatives[derivatives.subspace == "bc"][k].median())
    out["pooled"][k] = {"bp_median": bp, "bc_median": bc, "ratio": bp / bc if bc else None}
pooled_ratio = out["pooled"]["sp"]["ratio"]
out["T2b_verdict"] = "holds" if pooled_ratio >= 2 else ("falsified" if pooled_ratio <= 1 else "undetermined")
(HERE / "sensitivity_results.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
