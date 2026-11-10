"""Recompute the 29 relation features of S05, S06, S11, S12 from their structures (descriptors of
symbolic_regression.inputs.spinel_descriptors) and compare them with the stored feature matrix of the spinel relations
(output recomputed_feature_comparison.json); prerequisite of the T2b sensitivity.
usage: python compare_recomputed_features.py RAW.npy IDS.npy RETAINED_NAMES.json OUT.json
(RAW.npy = raw_features_681x29.npy, IDS.npy = sample_ids.npy, both in spinel_relations/colocation_relation)
Reads ../../descriptor_tables/structures/ (spinel_structures.csv, elements_primary.csv, elements_extended.csv)."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from symbolic_regression.inputs import spinel_descriptors as sd  # noqa: E402

STRUCTURES = REPO / "data" / "descriptor_tables" / "structures"
raw = np.load(sys.argv[1])
ids = list(np.load(sys.argv[2]).astype(str))
names = json.loads(Path(sys.argv[3]).read_text())
table = sd.read_structures(STRUCTURES / "spinel_structures.csv")
elements = sd.load_elements(STRUCTURES / "elements_primary.csv", STRUCTURES / "elements_extended.csv")
targets = ["mp-1001019_Sb_k%02d" % n for n in (5, 6, 11, 12)]
out = {"key_column": sd.ID_COLUMN, "rows": {}}
for t in targets:
    row = table[table[sd.ID_COLUMN].astype(str) == t]
    assert len(row) == 1, (t, len(row))
    row = row.iloc[0]
    structure = sd.structure(row)
    endpoint = str(row["endpoint_canonicalized"]).lower() == "true"
    features = sd.descriptors(row, sd.supercell(structure), elements)
    reference = raw[ids.index(t)]
    recomputed = np.array([float(features[n]) for n in names])
    relative_error = np.abs(recomputed - reference) / np.maximum(np.abs(reference), 1e-12)
    out["rows"][t] = {"role": "primitive_canonical_endpoint" if endpoint else "canonical_interior",
                      "nsites": int(structure.nsites), "max_rel_err": float(np.nanmax(relative_error)),
                      "worst": names[int(np.nanargmax(relative_error))],
                      "n_nan_mismatch": int(np.sum(np.isnan(recomputed) != np.isnan(reference))),
                      "per_feature": {n: [float(a), float(b)] for n, a, b in zip(names, reference, recomputed)}}
    print(t, out["rows"][t]["role"], structure.nsites, "max_rel", out["rows"][t]["max_rel_err"], out["rows"][t]["worst"])
out["gate_pass"] = all(r["max_rel_err"] <= 1e-6 and r["n_nan_mismatch"] == 0 for r in out["rows"].values())
Path(sys.argv[4]).write_text(json.dumps(out, indent=1))
print("GATE_PASS", out["gate_pass"])
