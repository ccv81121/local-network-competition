"""(1) Cubic-perovskite B-X bond-only compatibility matrix C(q): dim null C(q), self-stress count.
(2) Tilted Pnma: B-O-B bend angles and the bent-chain prediction f2(Gamma)/f2(R) = <tan^2(delta/2)>
for central breathing (uniform tension on a bent chain leaves a net O force 2t sin(delta/2))."""
import json, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
D = HERE.parents[1] / "data"
# (1) cubic: atoms B(0), Ox(1/2,0,0), Oy, Oz ; bonds B(R)-Oa(R) and Oa(R)-B(R+a)
pos = np.array([[0, 0, 0], [.5, 0, 0], [0, .5, 0], [0, 0, .5]])
def C(q):
    rows = []
    for a in range(3):
        e = np.eye(3)[a]
        r1 = np.zeros(12, complex); r1[0:3] -= e; r1[3 + 3 * a:6 + 3 * a] += e          # B(0)-Oa(0)
        r2 = np.zeros(12, complex); r2[3 + 3 * a:6 + 3 * a] -= e; r2[0:3] += e * np.exp(2j * np.pi * q[a])  # Oa(0)-B(a)
        rows += [r1, r2]
    return np.array(rows)
res = {}
for name, q in {"Gamma": (0, 0, 0), "X": (.5, 0, 0), "M": (.5, .5, 0), "R": (.5, .5, .5), "generic": (.13, .29, .41), "plane_qz0": (.13, .29, 0)}.items():
    s = np.linalg.svd(C(q), compute_uv=False); rk = int(np.sum(s > 1e-9))
    res[name] = {"dim_null_C": 12 - rk, "self_stress": 6 - rk}
print("cubic", res)
# (2) bend angles
out = {}
for cd in ("ca_tio3", "ca_sno3"):
    m = json.loads((D / "bending_enrichment/full_force_balance" / cd / "p0_p5/meta.json").read_text())
    L = np.array(m["lattice_P2"]); fr = np.array(m["frac_start"]); bonds = m["bonds"]
    Bset = set(np.flatnonzero(np.repeat(m["species"], m["counts"]) == m["species"][1]).tolist())
    vec = {}
    for i, k, img in bonds:
        b, o, s = (i, k, 1) if i in Bset else (k, i, -1)
        v = (fr[k] - fr[i] + np.array(img)) @ L * (1 if i == b else -1)   # vector O - B
        vec.setdefault(o, []).append(v)
    d = [np.degrees(np.pi - np.arccos(-np.dot(*v) / np.linalg.norm(v[0]) / np.linalg.norm(v[1]))) for v in vec.values() if len(v) == 2]
    d = 180.0 - np.array(d)   # bend delta = 180 - (B-O-B angle)
    out[cd] = {"bend_deg_mean": float(d.mean()), "bend_deg_range": [float(d.min()), float(d.max())],
               "tan2_half_mean": float(np.mean(np.tan(np.radians(d) / 2) ** 2))}
print(out)
cl = json.loads((HERE / "closed_loop.json").read_text())
for cd in out:
    s = cl[cd]["breathing_sectors"]
    print(cd, "f2 Ag/B3g(R) =", round(s["Ag"]["f2_frac"] / s["B3g"]["f2_frac"], 4), " predicted <tan^2(d/2)> =", round(out[cd]["tan2_half_mean"], 4))
json.dump({"cubic": res, "bend": out}, open(HERE / "selfstress.json", "w"), indent=1)
