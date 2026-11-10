"""T2b displaced-structure features. Writes features.csv: one row per (structure, subspace, seed, sign) with the 29
relation features of the reference structure and of structures displaced by +/- DISPLACEMENT_STEP (1 A maximum atomic
displacement per unit step) along random bond-preserving (bp) and bond-changing (bc) directions of the B-X network.
Features: symbolic_regression.inputs.spinel_descriptors; structures and element tables from
../../descriptor_tables/structures/ (spinel_structures.csv, elements_primary.csv, elements_extended.csv).
usage: python displace_structures.py RETAINED_NAMES.json OUT.csv"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from symbolic_regression.inputs import spinel_descriptors as sd  # noqa: E402

STRUCTURES = REPO / "data" / "descriptor_tables" / "structures"
names = json.loads(Path(sys.argv[1]).read_text())
DISPLACEMENT_STEP = 0.01
table = sd.read_structures(STRUCTURES / "spinel_structures.csv")
elements = sd.load_elements(STRUCTURES / "elements_primary.csv", STRUCTURES / "elements_extended.csv")


def null_and_row_space(M, tol=1e-9):
    _, s, Vt = np.linalg.svd(M)
    r = int(np.sum(s > tol * s[0]))
    return Vt[r:].T, Vt[:r].T


rows = []
for t in ["mp-1001019_Sb_k%02d" % n for n in (5, 6, 11, 12)]:
    row = table[table[sd.ID_COLUMN].astype(str) == t].iloc[0]
    st = sd.structure(row)
    N = st.nsites
    symbols = list(st.symbols)
    b_sites = [i for i in range(N) if symbols[i] in ("Sc", "Sb")]
    x_sites = [i for i in range(N) if symbols[i] == "Se"]
    network_sites = b_sites + x_sites
    dist, vec = sd.minimum_image(st)
    bonds = []
    for b in b_sites:
        nb = [j for j in np.argsort(dist[b]) if symbols[j] == "Se"][:6]
        for j in nb:
            bonds.append((b, int(j), vec[b, j]))
    # network coordinates: 3*len(network_sites); centroid removed
    site_column = {a: k for k, a in enumerate(network_sites)}
    J = np.zeros((len(bonds), 3 * len(network_sites)))
    for r, (i, j, v) in enumerate(bonds):
        e = v / np.linalg.norm(v)
        J[r, 3 * site_column[i]:3 * site_column[i] + 3] = -e
        J[r, 3 * site_column[j]:3 * site_column[j] + 3] = e
    T = np.zeros((3 * len(network_sites), 3))
    for a in range(3):
        T[a::3, a] = 1 / np.sqrt(len(network_sites))
    Q, _ = null_and_row_space(T.T)  # centroid-free basis
    kerQ, rowQ = null_and_row_space(J @ Q)
    Pb = Q @ kerQ
    Pc = Q @ rowQ
    print(t, "N", N, "bonds", len(bonds), "dim bp", Pb.shape[1], "dim bc", Pc.shape[1], "leak",
          float(np.abs(J @ Pb).max()), flush=True)
    for sub, P, seeds in (("bp", Pb, range(0, 20)), ("bc", Pc, range(100, 120))):
        for seed in seeds:
            c = np.random.default_rng(seed).standard_normal(P.shape[1])
            d = P @ c
            d = d.reshape(-1, 3)
            d = d / np.linalg.norm(d, axis=1).max()  # max atomic displacement = 1 A per unit eps
            for direction in (+1, -1):
                dc = np.zeros((N, 3))
                dc[network_sites] = direction * DISPLACEMENT_STEP * d
                s2 = sd.Structure(st.lattice.copy(), st.symbols, (st.frac + dc @ np.linalg.inv(st.lattice)) % 1.0)
                f = sd.descriptors(row, sd.supercell(s2), elements)
                rows.append({"sample_id": t, "subspace": sub, "seed": seed, "sign": direction,
                             **{n: float(f[n]) for n in names}})
    f0 = sd.descriptors(row, sd.supercell(st), elements)
    rows.append({"sample_id": t, "subspace": "ref", "seed": -1, "sign": 0, **{n: float(f0[n]) for n in names}})
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
print("DONE", len(rows))
