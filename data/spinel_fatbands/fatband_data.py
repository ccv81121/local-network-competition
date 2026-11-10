"""Orbital-resolved bands along R-Gamma-Y at 4 and 5 GPa (tight structures) from PROCAR (LORBIT=11).

Reads raw/<p>/PROCAR for p4 and p5 (PROCAR not deposited); the k-distance uses the lattice of raw/p4/POSCAR (4 GPa)
and of the state11 POSCAR of spinel_pressure_path (5 GPa). Writes fatband_<p>.npz with k-distance, energies (relative to the VBM) and
Sc-d / Sb / Se weights, and fatband_summary.json with VBM/CBM locations and the Gamma-R conduction margin on this path.

usage: python fatband_data.py
"""
import json
import re
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
ION_ELEMENTS = ["Mg"] * 8 + ["Sb"] * 12 + ["Sc"] * 4 + ["Se"] * 32
NOCC = 156


def lattice(p):
    lines = p.read_text().split("\n")
    return np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)])


def parse(procar):
    txt = procar.read_text()
    k_blocks = re.split(r"\n k-point\s+\d+ :", txt)[1:]
    ks, E, weights = [], [], []
    for blk in k_blocks:
        ks.append([float(x) for x in re.findall(r"-?\d+\.\d+", blk.split("\n", 1)[0])[:3]])
        e_k, w_k = [], []
        for match in re.finditer(r"band\s+(\d+) # energy\s+(\S+).*?\nion[^\n]*\n(.*?)\ntot", blk, re.S):
            rows = np.array([[float(x) for x in r.split()[1:10]] for r in match.group(3).split("\n")[:56]])
            tot = rows.sum()

            def ion_indices(el):
                return [i for i, n in enumerate(ION_ELEMENTS) if n == el]

            w_k.append([rows[ion_indices("Sc"), 4:9].sum() / tot, rows[ion_indices("Sb"), 0:4].sum() / tot,
                        rows[ion_indices("Se"), 0:4].sum() / tot])
            e_k.append(float(match.group(2)))
        E.append(e_k)
        weights.append(w_k)
    return np.array(ks), np.array(E), np.array(weights)


summary = {}
for P in ("P4", "P5"):
    ks, E, weights = parse(HERE / "raw" / P.lower() / "PROCAR")
    poscar_path = (HERE / "raw" / "p4" / "POSCAR" if P == "P4"
                   else HERE.parent / "spinel_pressure_path" / "fourstate" / "state11" / "POSCAR")
    rec = np.linalg.inv(lattice(poscar_path)).T
    kc = ks @ rec
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(kc, axis=0), axis=1))])
    vbm = E[:, NOCC - 1].max()
    E_rel = E - vbm
    ivb, icb = int(np.argmax(E[:, NOCC - 1])), int(np.argmin(E[:, NOCC]))
    iG = int(np.argmin(np.linalg.norm(ks, axis=1)))
    iR = 0
    summary[P] = {"VBM_k": ks[ivb].tolist(), "CBM_k": ks[icb].tolist(), "gap_eV": float(E[icb, NOCC] - vbm),
                  "Lambda_C_R_minus_Gamma_meV": float((E[iR, NOCC] - E[iG, NOCC]) * 1000),
                  "CBM_Gamma_weights_ScD_Sb_Se": weights[iG, NOCC].round(3).tolist(),
                  "CBM_R_weights_ScD_Sb_Se": weights[iR, NOCC].round(3).tolist()}
    np.savez_compressed(HERE / f"fatband_{P.lower()}.npz", k=ks, dist=dist, E=E_rel, W=weights)
    print(P, json.dumps(summary[P]))
json.dump(summary, open(HERE / "fatband_summary.json", "w"), indent=1)
