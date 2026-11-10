"""Real-space response fields for the figure renders (numpy, read-only on data/).
CaTiO3 (Pnma, 20 atoms, p0_p5 force constants): response u = H^+ f to a pure breathing force on one Ti (the drive of
closed_loop.py), split into its bond-preserving part P_b u and the remainder. Writes response_fields.npz and prints the
share of the bond-preserving part that is a rigid rotation of the octahedra and its B1g weight."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
src = (HERE / "closed_loop.py").read_text()
head, body = src.split("def perovskite(cd):")[0], src.split("def perovskite(cd):")[1]
exec(compile((HERE / "checks.py").read_text().split("rows = []")[0], "checks_defs", "exec"))
exec(compile("import spglib\n" + head.split("out = {}")[1].split("# ---------- validation")[0] + "def sym_ops" + head.split("def sym_ops")[1], "defs", "exec"))

cd = "ca_tio3"
d = D / "bending_enrichment/full_force_balance" / cd / "p0_p5"
m = json.loads((d / "meta.json").read_text())
N = sum(m["counts"]); spc = np.repeat(m["species"], m["counts"])
H = full_H(lambda c: np.load(d / "hessian" / c / "forces.npy"), N, m["delta_A"])
bonds = m["bonds"]; net = np.array(sorted({b[0] for b in bonds} | {b[1] for b in bonds}))
L = np.array(m["lattice_P2"]); frac = np.array(m["frac_start"]); Bs = np.flatnonzero(spc == "Ti")
T = np.zeros((3 * N, 3))
for a in range(3):
    T[a::3, a] = 1 / np.sqrt(N)
Q = np.eye(3 * N) - T @ T.T; Hq = Q @ H @ Q
w, V = np.linalg.eigh(Hq); k = np.argsort(np.abs(w))[3:]; w, V = w[k], V[:, k]
Hp = lambda x: V @ ((V.T @ x) / w)
J = jac(frac, L, bonds, net)
_, s, Vt = np.linalg.svd(J); rank = int(np.sum(s > s[0] * 1e-10)); Nb = Vt[rank:].T
Tn = np.zeros((3 * len(net), 3))
for a in range(3):
    Tn[a::3, a] = 1 / np.sqrt(len(net))
Nb = Nb - Tn @ (Tn.T @ Nb); Ub, sb, _ = np.linalg.svd(Nb, full_matrices=False); Nb = Ub[:, sb > 1e-8]
idx = np.concatenate([[3 * a, 3 * a + 1, 3 * a + 2] for a in net])
B = np.zeros((3 * N, Nb.shape[1])); B[idx] = Nb; Pb = B @ B.T
j = Bs[0]
tj = np.array([1.0 if (i == j or kk == j) else 0.0 for i, kk, _ in bonds])
f = np.zeros(3 * N); f[idx] = -J.T @ tj; f = Q @ f
u = Hp(f); ub = Pb @ u
cen = np.zeros(3 * N); cen[idx] = 1.0
ur = (u - ub) * cen
# rigid rotations of each octahedron about its Ti (closed_loop.py construction)
rot = []
for jj in Bs:
    for a in range(3):
        e = np.eye(3)[a]; v = np.zeros(3 * N)
        for i, kk, img in bonds:
            if i == jj or kk == jj:
                o = kk if i == jj else i
                dvec = (frac[o] - frac[jj] + (np.array(img) if i == jj else -np.array(img))) @ L
                v[3 * o:3 * o + 3] += np.cross(e, dvec)
        rot.append(Q @ v)
Prot = np.linalg.svd(np.array(rot).T, full_matrices=False)[0]
rot_share = float(np.sum((Prot.T @ ub) ** 2) / (ub @ ub))
np.savez(HERE / "response_fields.npz", lattice=L, frac=frac, species=spc, bonds=np.array([[b[0], b[1], *b[2]] for b in bonds]),
         driven=j, f=f.reshape(-1, 3), u=u.reshape(-1, 3), u_b=ub.reshape(-1, 3), u_r=ur.reshape(-1, 3))
cn = lambda x: float(x[idx] @ x[idx])
print(f"driven Ti {j}; s_u = {cn(ub) / cn(u * cen):.4f}; rigid-rotation share of P_b u = {rot_share:.4f}")
