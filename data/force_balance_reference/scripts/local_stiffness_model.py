"""Local-octahedron decomposition of the DFT force constants.

H_model = sum_c p_type(c) g_c g_c^T, where each internal coordinate c is a metal-Se bond length or a cis
Se-M-Se angle of one coordination polyhedron, and g_c is its Cartesian gradient at the start structure.
The six stiffnesses p (stretch and bend for Sc, Sb, Mg) are local; the shared-ligand network enters only
through the assembly map (which polyhedra share each Se). The p are obtained by non-negative least squares
on the translation-projected DFT H. The model is then tested on what it was not fitted to: the soft-mode
subspace and the force-balance response to the same drive f.
Runs analyze.py on this folder; writes outputs/local_stiffness_model.json and outputs/poscar_local_model.
usage: python local_stiffness_model.py
"""
import contextlib
import io
import json
import runpy
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.argv = [sys.argv[0], str(ROOT)]
with contextlib.redirect_stdout(io.StringIO()):
    analysis = runpy.run_path(str(ROOT / "scripts" / "analyze.py"))
H, Q, f, u_dft, u_act = analysis["H"], analysis["Q"], analysis["f0_q"], analysis["u_pred"], analysis["u_act"]
L5, f4, species, pf = analysis["L5"], analysis["f4"], analysis["species"], analysis["preserving_fraction"]
N = len(species)
R = f4 @ L5


def neighbors(i, cand, n):
    out = []
    for j in cand:
        d = f4[j] - f4[i]
        for img in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 3):
            v = (d - np.round(d) + img) @ L5
            out.append((np.linalg.norm(v), j, v))
    out.sort(key=lambda x: x[0])
    return out[:n]


se_atoms = np.flatnonzero(species == "Se")
coords = {t: [] for t in ("Sc stretch", "Sb stretch", "Mg stretch", "Sc bend", "Sb bend", "Mg bend")}
for i in range(N):
    sp = species[i]
    if sp == "Se":
        continue
    nb = neighbors(i, se_atoms, 6 if sp in ("Sc", "Sb") else 4)
    for r, j, v in nb:  # bond stretch
        e = v / r
        row = np.zeros(3 * N)
        row[3 * i:3 * i + 3] -= e
        row[3 * j:3 * j + 3] += e
        coords[f"{sp} stretch"].append(row)
    rbar = np.mean([x[0] for x in nb])
    for a in range(len(nb)):
        for b in range(a + 1, len(nb)):
            (ra, ja, va), (rb, jb, vb) = nb[a], nb[b]
            ua, ub = va / ra, vb / rb
            c = ua @ ub
            if sp in ("Sc", "Sb") and c < -0.5:  # trans pair of the octahedron: not an independent bend
                continue
            s = np.sqrt(1 - c * c)
            ga = -(ub - c * ua) / (ra * s)  # d theta / d r_a
            gb = -(ua - c * ub) / (rb * s)
            row = np.zeros(3 * N)
            row[3 * ja:3 * ja + 3] += ga * rbar
            row[3 * jb:3 * jb + 3] += gb * rbar  # length-scaled angle
            row[3 * i:3 * i + 3] -= (ga + gb) * rbar
            coords[f"{sp} bend"].append(row)

types = list(coords)
G = [sum(np.outer(r, r) for r in coords[t]) for t in types]
target = Q @ H @ Q
A = np.stack([(Q @ Gt @ Q).ravel() for Gt in G], axis=1)
p, _ = nnls(A, target.ravel())
Hm = sum(pi * Gt for pi, Gt in zip(p, G))
Hm_q = Q @ Hm @ Q
r2 = 1 - np.linalg.norm(Hm_q - target) ** 2 / np.linalg.norm(target) ** 2


def spectrum(M):
    w, V = np.linalg.eigh(M)
    keep = np.argsort(np.abs(w))[3:]
    w, V = w[keep], V[:, keep]
    order = np.argsort(w)
    return w[order], V[:, order]


wd, Vd = spectrum(target)
wm, Vm = spectrum(Hm_q)
n_soft = 20
overlap = float(np.linalg.norm(Vd[:, :n_soft].T @ Vm[:, :n_soft]) ** 2 / n_soft)
u_m = Vm @ ((Vm.T @ f) / wm)


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


# stretch-only model: its zero modes are exactly the bond-preserving (floppy) motions of the stretch network
Gs = sum(pi * Gt for pi, Gt, t in zip(p, G, types) if "stretch" in t)
ws = np.linalg.eigvalsh(Q @ Gs @ Q)

out = {
    "parameters_eV_per_A2": {t: float(pi) for t, pi in zip(types, p)},
    "bend_to_stretch_ratio": {s: float(p[types.index(f"{s} bend")] / p[types.index(f"{s} stretch")])
                              for s in ("Sc", "Sb", "Mg")},
    "n_internal_coordinates": {t: len(coords[t]) for t in types},
    "fit_R2_frobenius": float(r2),
    "softest_eigenvalues_eV_A2": {"dft": wd[:n_soft].round(3).tolist(), "model": wm[:n_soft].round(3).tolist()},
    "soft_subspace_overlap_20": overlap,
    "model_soft_modes_bond_preserving_fraction_mean": float(np.mean([pf(Vm[:, k]) for k in range(n_soft)])),
    "response": {"cos_model_vs_dft_response": cos(u_m, u_dft), "cos_model_vs_actual": cos(u_m, u_act),
                 "bond_preserving_fraction_model": pf(u_m), "bond_preserving_fraction_dft": pf(u_dft),
                 "norm_ratio_model_over_dft": float(np.linalg.norm(u_m) / np.linalg.norm(u_dft))},
    "stretch_only_zero_modes": int(np.sum(np.abs(ws) < 1e-6 * np.abs(ws).max()) - 3),
}
(ROOT / "outputs" / "local_stiffness_model.json").write_bytes((json.dumps(out, indent=1) + "\n").encode())
print(json.dumps(out, indent=1))

# structure predicted by the local model, for the K6 band-edge readout
frac_m = f4 + u_m.reshape(-1, 3) @ np.linalg.inv(L5)
meta = json.loads((ROOT / "meta.json").read_text())
lines = ["predicted by local polyhedral stiffness model", "1.0"] + ["  ".join(f"{x:.16f}" for x in v) for v in L5]
lines += (["  ".join(meta["species"]), "  ".join(map(str, meta["counts"])), "Direct"]
          + ["  ".join(f"{x:.16f}" for x in r) for r in frac_m])
(ROOT / "outputs" / "poscar_local_model").write_bytes(("\n".join(lines) + "\n").encode())
