"""Four-state readout: Lambda_C = E_C(1/2,1/2,1/2) - E_C(Gamma) and total energies of state00..state11.
Reads EIGENVAL and OUTCAR of state00, state10, state01, state11 in each directory (not deposited; toten.txt and
tight_summary.json hold the results) and prints, per directory,
Lambda_C of every state (meV), d10 = Lambda_C(state10) - Lambda_C(state00), d01 likewise for state01,
phi = |d10| / (|d10| + |d01|) and the total energies relative to state00 (meV).
usage: python fourstate_margins.py FOURSTATE_DIR ..."""
import sys, re, numpy as np
from pathlib import Path


def read_state(state_dir):
    lines = (state_dir / "EIGENVAL").read_text().splitlines()
    n_electrons, n_kpoints, n_bands = map(int, lines[5].split())
    n_occ = n_electrons // 2
    i = 6
    out = {}
    for _ in range(n_kpoints):
        while not lines[i].strip():
            i += 1
        k = tuple(np.round(np.abs(list(map(float, lines[i].split()[:3]))), 6))
        i += 1
        bands = [float(lines[i + j].split()[1]) for j in range(n_bands)]
        i += n_bands
        out[k] = (bands[n_occ - 1], bands[n_occ])
    energy = float(re.findall(r"free  energy   TOTEN  =\s+(-?\d+\.\d+)", (state_dir / "OUTCAR").read_text())[-1])
    return out, energy


for root in sys.argv[1:]:
    states = {s: read_state(Path(root) / s) for s in ("state00", "state10", "state01", "state11")}
    gamma_k = (0.0, 0.0, 0.0)
    r_point = (0.5, 0.5, 0.5)
    lambda_c = {s: 1000 * (states[s][0][r_point][1] - states[s][0][gamma_k][1]) for s in states}
    d10 = lambda_c["state10"] - lambda_c["state00"]
    d01 = lambda_c["state01"] - lambda_c["state00"]
    print(root, {s: round(v, 2) for s, v in lambda_c.items()},
          "d10=%.2f d01=%.2f phi=%.3f" % (d10, d01, abs(d10) / (abs(d10) + abs(d01))),
          "Erel", {s: round(1000 * (states[s][1] - states["state00"][1]), 2) for s in states})
