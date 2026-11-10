"""Band-edge margins for s0, s1 and the four states; split into chemistry, A-cation, bond-changing and bond-preserving
parts.
Margin (as in phi_readout.py): best and second-best of the eight TRIM points, chosen at state11 and held fixed.
Both edges reported; the switching edge is the one with the smaller |Lambda| at state11.
CASE_DIR holds the EIGENVAL and OUTCAR of the VASP runs (not deposited; results/<case>/readout.json is the result).
usage: python alloy_readout.py CASE_DIR"""
import json
import re
import sys
from pathlib import Path

import numpy as np

CASE_DIR = Path(sys.argv[1])
STATE_DIRS = {"s0": CASE_DIR / "s0", "s1": CASE_DIR / "s1",
              **{s: CASE_DIR / "fourstate" / s for s in ("state00", "state10", "state01", "state11")}}


def read_band_edges(d):
    lines = (d / "EIGENVAL").read_text().splitlines()
    ne, nk, nb = map(int, lines[5].split())
    n_occ = ne // 2
    i = 6
    K, V, C = [], [], []
    for _ in range(nk):
        while not lines[i].strip():
            i += 1
        K.append(list(map(float, lines[i].split()[:3])))
        i += 1
        bands = [float(lines[i + j].split()[1]) for j in range(nb)]
        i += nb
        V.append(bands[n_occ - 1])
        C.append(bands[n_occ])
    toten = float(re.findall(r"free  energy   TOTEN  =\s+(-?\d+\.\d+)", (d / "OUTCAR").read_text())[-1])
    return np.array(K), np.array(V), np.array(C), toten


BAND_EDGES = {s: read_band_edges(d) for s, d in STATE_DIRS.items()}
K = BAND_EDGES["s0"][0]
for s in BAND_EDGES:
    assert np.allclose(BAND_EDGES[s][0], K), s


def same_k(a, b):
    return (np.allclose((K[a] - K[b]) - np.round(K[a] - K[b]), 0, atol=1e-6)
            or np.allclose((K[a] + K[b]) - np.round(K[a] + K[b]), 0, atol=1e-6))


trim_indices = [i for i, q in enumerate(K) if np.allclose(2 * q - np.round(2 * q), 0, atol=1e-6)]
trim_indices = [i for n, i in enumerate(trim_indices) if not any(same_k(i, j) for j in trim_indices[:n])]
assert len(trim_indices) == 8, len(trim_indices)
_, V11, C11, _ = BAND_EDGES["state11"]
v_edge, v_rival = sorted(trim_indices, key=lambda k: -V11[k])[:2]
c_edge, c_rival = sorted(trim_indices, key=lambda k: C11[k])[:2]
Lam = {"V": lambda s: 1000 * (BAND_EDGES[s][1][v_edge] - BAND_EDGES[s][1][v_rival]),
       "C": lambda s: 1000 * (BAND_EDGES[s][2][c_rival] - BAND_EDGES[s][2][c_edge])}
out = {"dir": str(CASE_DIR), "k": {"V_edge": K[v_edge].tolist(), "V_rival": K[v_rival].tolist(),
                                   "C_edge": K[c_edge].tolist(), "C_rival": K[c_rival].tolist()}}
for edge, lam in Lam.items():
    chem = lam("s1") - lam("s0")
    a_cation = lam("state00") - lam("s1")
    bp = lam("state10") - lam("state00")
    bc = lam("state01") - lam("state00")
    rel = lam("state11") - lam("state00")
    out[edge] = {"Lambda_meV": {s: round(lam(s), 2) for s in BAND_EDGES}, "d_chem": round(chem, 2),
                 "d_A": round(a_cation, 2), "d_bc": round(bc, 2),
                 "d_bp": round(bp, 2), "d_relax_network": round(rel, 2), "additivity_residual": round(rel - bc - bp, 2),
                 "d_total": round(lam("state11") - lam("s0"), 2),
                 "R_rel": round(abs(bc + bp) / (abs(chem) + abs(bc + bp)), 4) if abs(chem) + abs(bc + bp) > 0 else None,
                 "phi_sub": round(abs(bp) / (abs(bp) + abs(bc)), 4) if abs(bp) + abs(bc) > 0 else None}
out["switching_edge"] = "V" if abs(Lam["V"]("state11")) <= abs(Lam["C"]("state11")) else "C"


def edge_summary(s):
    return {"VBM_k": K[int(np.argmax(BAND_EDGES[s][1]))].tolist(),
            "CBM_k": K[int(np.argmin(BAND_EDGES[s][2]))].tolist(),
            "gap_eV": round(float(BAND_EDGES[s][2].min() - BAND_EDGES[s][1].max()), 4)}


out["edges"] = {s: edge_summary(s) for s in ("s0", "s1", "state11")}
out["E_rel_meV"] = {s: round(1000 * (BAND_EDGES[s][3] - BAND_EDGES["state00"][3]), 2)
                    for s in ("state00", "state10", "state01", "state11")}
out["E_relax_from_s1_meV"] = round(1000 * (BAND_EDGES["state11"][3] - BAND_EDGES["s1"][3]), 2)
(CASE_DIR / "readout.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
