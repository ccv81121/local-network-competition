"""Electronic partition phi from a four-state directory.
Primary margin: at state11, edge and rival are the best and second-best of the eight TRIM points for VB and CB; the edge
with the smaller |Lambda| is used, with the k-points held fixed for all states.
phi = |dLambda_perp| / (|dLambda_perp| + |dLambda_par|), dLambda_perp = Lambda(state10) - Lambda(state00),
dLambda_par = Lambda(state01) - Lambda(state00).
Secondary phi_all: norm of the change of all VB (rel. to state00 VBM k) and CB (rel. to state00 CBM k) energies.
Reads state00, state10, state01, state11 (EIGENVAL, OUTCAR, KPOINTS) of the four-state directory; writes phi.json
there.
EIGENVAL and OUTCAR are not deposited; fourstate/phi.json is the result.
usage: python phi_readout.py FOURSTATE_DIR"""
import json, re, sys
from pathlib import Path
import numpy as np

FOURSTATE_DIR = Path(sys.argv[1])


def read(state_dir):
    lines = (state_dir / "EIGENVAL").read_text().splitlines()
    n_electrons, n_kpoints, n_bands = map(int, lines[5].split())
    n_occ = n_electrons // 2
    i = 6
    kpoints, V, C = [], [], []
    for _ in range(n_kpoints):
        while not lines[i].strip():
            i += 1
        kpoints.append(list(map(float, lines[i].split()[:3])))
        i += 1
        bands = [float(lines[i + j].split()[1]) for j in range(n_bands)]
        i += n_bands
        V.append(bands[n_occ - 1])
        C.append(bands[n_occ])
    energy = float(re.findall(r"free  energy   TOTEN  =\s+(-?\d+\.\d+)", (state_dir / "OUTCAR").read_text())[-1])
    return np.array(kpoints), np.array(V), np.array(C), energy


states = {s: read(FOURSTATE_DIR / s) for s in ("state00", "state10", "state01", "state11")}
kpoints = states["state00"][0]
for s in states:
    assert np.allclose(states[s][0], kpoints)


# k-points a and b (indices) are equal up to a reciprocal-lattice vector or inversion
def same_k(a, b):
    return (np.allclose((kpoints[a] - kpoints[b]) - np.round(kpoints[a] - kpoints[b]), 0, atol=1e-6)
            or np.allclose((kpoints[a] + kpoints[b]) - np.round(kpoints[a] + kpoints[b]), 0, atol=1e-6))


MESH = [int(x) for x in (FOURSTATE_DIR / "state00" / "KPOINTS").read_text().splitlines()[3].split()[:3]]


def mesh_key(q):
    return tuple(int(round(x * n)) % n for x, n in zip(q, MESH))


TRIM = [i for i, q in enumerate(kpoints) if np.allclose(2 * q - np.round(2 * q), 0, atol=1e-6)]
TRIM = [i for n, i in enumerate(TRIM) if not any(same_k(i, j) for j in TRIM[:n])]
assert len(TRIM) == 8, len(TRIM)


def margin(E, sign):
    # edge and rival: best and second-best of the eight TRIM points
    order = sorted(TRIM, key=lambda k: -sign * E[k])
    return order[0], order[1]


_, V11, C11, _ = states["state11"]
kv, kvr = margin(V11, +1)
kc, kcr = margin(C11, -1)


def lambda_v(s):
    return 1000 * (states[s][1][kv] - states[s][1][kvr])


def lambda_c(s):
    return 1000 * (states[s][2][kcr] - states[s][2][kc])


edge = "V" if abs(lambda_v("state11")) <= abs(lambda_c("state11")) else "C"
lambda_edge = lambda_v if edge == "V" else lambda_c
d10 = lambda_edge("state10") - lambda_edge("state00")
d01 = lambda_edge("state01") - lambda_edge("state00")
d11 = lambda_edge("state11") - lambda_edge("state00")
phi = abs(d10) / (abs(d10) + abs(d01))
_, V0, C0, _ = states["state00"]
v0 = np.argmax(V0)
c0 = np.argmin(C0)


def level_vector(s):
    return np.concatenate([states[s][1] - states[s][1][v0], states[s][2] - states[s][2][c0]]) * 1000


a10 = np.linalg.norm(level_vector("state10") - level_vector("state00"))
a01 = np.linalg.norm(level_vector("state01") - level_vector("state00"))
out = {"dir": str(FOURSTATE_DIR), "edge": edge, "k_edge": kpoints[kv if edge == "V" else kc].tolist(),
       "k_rival": kpoints[kvr if edge == "V" else kcr].tolist(),
       "Lambda_meV": {s: round(lambda_edge(s), 2) for s in states}, "dLambda_perp_meV": round(d10, 2),
       "dLambda_par_meV": round(d01, 2),
       "dLambda_total_meV": round(d11, 2), "additivity_residual_meV": round(d11 - d10 - d01, 2), "phi": round(phi, 4),
       "phi_all": round(a10 / (a10 + a01), 4),
       "E_rel_meV": {s: round(1000 * (states[s][3] - states["state00"][3]), 2) for s in states},
       "LambdaV_state11_meV": round(lambda_v("state11"), 2), "LambdaC_state11_meV": round(lambda_c("state11"), 2)}
(FOURSTATE_DIR / "phi.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
