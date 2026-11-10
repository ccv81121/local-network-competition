"""Band-edge readout along the tight pressure path (definitions as in the paper).
Lambda_C = E_C(1/2,1/2,1/2) - E_C(Gamma)  (>0: CBM at Gamma)
Lambda_V = E_V(Gamma) - E_V(q_V), q_V = highest non-Gamma valence point on the mesh
(also reported: axis points (1/m,0,0) etc.)
Reads EIGENVAL of every converged (DONE) static_k*/p* run in TIGHT_DIR (EIGENVAL and DONE are not deposited); writes
TIGHT_DIR/path_edges.json.
usage: python path_edges.py TIGHT_DIR"""
import json, sys
from pathlib import Path
import numpy as np

TIGHT_DIR = Path(sys.argv[1])
out = []
for run_dir in sorted(TIGHT_DIR.glob("static_k*/p*"), key=lambda p: (p.parent.name, float(p.name[1:]))):
    if not (run_dir / "DONE").exists():
        continue
    lines = (run_dir / "EIGENVAL").read_text().splitlines()
    n_electrons, n_kpoints, n_bands = map(int, lines[5].split())
    n_occ = n_electrons // 2
    i = 6
    kpoints, V, C = [], [], []
    for _ in range(n_kpoints):
        while not lines[i].strip():
            i += 1
        kpoints.append(np.array(list(map(float, lines[i].split()[:3]))))
        i += 1
        b = [float(lines[i + j].split()[1]) for j in range(n_bands)]
        i += n_bands
        V.append(b[n_occ - 1])
        C.append(b[n_occ])
    kpoints = np.array(kpoints)
    V = np.array(V)
    C = np.array(C)

    def find_k(q):
        return [n for n, k in enumerate(kpoints)
                if np.allclose((k - q) - np.round(k - q), 0, atol=1e-6)
                or np.allclose((k + q) - np.round(k + q), 0, atol=1e-6)][0]

    gamma_index = find_k(np.zeros(3))
    r_index = find_k(np.array([.5, .5, .5]))
    mesh = int(run_dir.parent.name.split("k")[1])
    non_gamma = [n for n in range(n_kpoints) if n != gamma_index]
    q_v_index = max(non_gamma, key=lambda n: V[n])
    axes = {f"({'/'.join(['1/%d' % mesh if a == b else '0' for b in range(3)])})":
            round(1000 * (V[gamma_index] - V[find_k(np.eye(3)[a] / mesh)]), 2) for a in range(3)}
    gap = C.min() - V.max()
    out.append({"mesh": mesh, "P_GPa": float(run_dir.name[1:]),
                "Lambda_C_meV": round(1000 * (C[r_index] - C[gamma_index]), 2),
                "Lambda_V_meV": round(1000 * (V[gamma_index] - V[q_v_index]), 2),
                "qV": kpoints[q_v_index].round(4).tolist(), "Lambda_V_axes_meV": axes,
                "VBM_k": kpoints[int(V.argmax())].round(4).tolist(),
                "CBM_k": kpoints[int(C.argmin())].round(4).tolist(),
                "gap_eV": round(gap, 4),
                "direct_at_Gamma": bool(V.argmax() == gamma_index and C.argmin() == gamma_index)})
(TIGHT_DIR / "path_edges.json").write_text(json.dumps(out, indent=1))
for r in out:
    print(r["mesh"], r["P_GPa"], "LC", r["Lambda_C_meV"], "LV", r["Lambda_V_meV"], r["qV"], "VBM", r["VBM_k"], "CBM",
          r["CBM_k"], "gap", r["gap_eV"])
