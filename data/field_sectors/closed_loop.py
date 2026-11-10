"""Closed-loop field-theory tests on existing force constants (numpy + spglib, read-only on data/).
Symmetry-sector (generalised Bloch) decomposition of drives and responses in CaTiO3 / CaSnO3 (Pnma, 20 atoms),
sum rule for the single-site breathing drive, sector-resolved floppy dimension, Schur stiffness and
rigid-octahedron (RUM-like) content. Writes closed_loop.json; prints a compact summary."""
import json
from pathlib import Path
import numpy as np
import spglib

HERE = Path(__file__).resolve().parent
src = (HERE / "checks.py").read_text().split("rows = []")[0]
exec(compile(src, "checks_defs", "exec"))          # jac, full_H, analyse, D (data dir)
out = {}

# ---------- validation: reproduce known numbers ----------
m = json.loads((D / "spinel_pressure_path/force_balance/meta.json").read_text())
sp = np.repeat(m["species"], m["counts"]); N = len(sp)
Hk = full_H(lambda c: np.load(D / "force_balance_reference/forces" / f"{c}.npy"), N, m["delta_A"])
fk = np.load(D / "spinel_pressure_path/force_balance/forces/ref.npy").ravel()
bk = json.loads((D / "force_balance_reference/inputs/pressure_bond_accommodation.json").read_text())["bonds"]
netk = np.flatnonzero(np.isin(sp, ["Sc", "Sb", "Se"]))
rk = analyse("k12", Hk, fk, np.array(m["frac_start"]), np.array(m["lattice_5GPa"]), bk, netk, N)
out["validate_k12_s_f"] = rk["s_f"]


def sym_ops(L, frac, num):
    ds = spglib.get_symmetry_dataset((L, frac, num), symprec=2e-3)
    ops = []
    Linv = np.linalg.inv(L)
    for W, t in zip(ds.rotations, ds.translations):
        new = frac @ W.T + t
        perm = np.empty(len(frac), int)
        for i, x in enumerate(new):
            d = frac - x; d -= np.round(d)
            perm[i] = int(np.argmin(np.linalg.norm(d @ L, axis=1)))
        Rc = L.T @ W @ Linv.T                      # cartesian (column vectors)
        Dg = np.zeros((3 * len(frac),) * 2)
        for i, j in enumerate(perm):
            Dg[3 * j:3 * j + 3, 3 * i:3 * i + 3] = Rc
        ops.append((W, Dg))
    return ds.number, ops


def mmm_irreps(ops):
    """1D irreps of mmm from rotation parts (orthorhombic, diagonal W)."""
    lab = {}
    for name, (cx, cy, p) in {"Ag": (1, 1, 1), "B1g": (-1, -1, 1), "B2g": (-1, 1, 1), "B3g": (1, -1, 1),
                              "Au": (1, 1, -1), "B1u": (-1, -1, -1), "B2u": (-1, 1, -1), "B3u": (1, -1, -1)}.items():
        chi = []
        for W, _ in ops:
            det = round(np.linalg.det(W)); Pr = np.rint(W * det).astype(int)
            dg = tuple(np.diag(Pr))
            c = {(1, 1, 1): 1, (1, -1, -1): cx, (-1, 1, -1): cy, (-1, -1, 1): cx * cy}[dg]
            chi.append(c * (p if det < 0 else 1))
        lab[name] = np.array(chi)
    return lab


def perovskite(cd):
    d = D / "bending_enrichment/full_force_balance" / cd / "p0_p5"
    m = json.loads((d / "meta.json").read_text())
    N = sum(m["counts"]); spc = np.repeat(m["species"], m["counts"])
    rd = lambda c: np.load(d / "hessian" / c / "forces.npy")
    H = full_H(rd, N, m["delta_A"]); f0 = rd("ref").ravel()
    bonds = m["bonds"]; net = np.array(sorted({b[0] for b in bonds} | {b[1] for b in bonds}))
    L = np.array(m["lattice_P2"]); frac = np.array(m["frac_start"])
    Bs = np.flatnonzero(spc == m["species"][1])
    res = {"validate": analyse(cd, H, f0, frac, L, bonds, net, N, None, Bs)}
    num = np.array([{"Ca": 20, "Ti": 22, "Sn": 50, "O": 8}[s] for s in spc])
    sg, ops = sym_ops(L, frac, num)
    res["spacegroup"] = int(sg); res["n_ops"] = len(ops)
    # translation-free space, pseudo-inverse
    T = np.zeros((3 * N, 3))
    for a in range(3):
        T[a::3, a] = 1 / np.sqrt(N)
    Q = np.eye(3 * N) - T @ T.T
    Hq = Q @ H @ Q
    res["H_asym_commutator"] = float(max(np.linalg.norm(Dg @ Hq - Hq @ Dg) for _, Dg in ops) / np.linalg.norm(Hq))
    w, V = np.linalg.eigh(Hq); k = np.argsort(np.abs(w))[3:]; w, V = w[k], V[:, k]
    Hp = lambda x: V @ ((V.T @ x) / w)
    # b basis (same construction as analyse)
    J = jac(frac, L, bonds, net)
    _, s, Vt = np.linalg.svd(J); rank = int(np.sum(s > s[0] * 1e-10)); Nb = Vt[rank:].T
    Tn = np.zeros((3 * len(net), 3))
    for a in range(3):
        Tn[a::3, a] = 1 / np.sqrt(len(net))
    Nb = Nb - Tn @ (Tn.T @ Nb); Ub, sb, _ = np.linalg.svd(Nb, full_matrices=False); Nb = Ub[:, sb > 1e-8]
    idx = np.concatenate([[3 * a, 3 * a + 1, 3 * a + 2] for a in net])
    B = np.zeros((3 * N, Nb.shape[1])); B[idx] = Nb; Pb = B @ B.T
    Qn = np.zeros((3 * N, 3 * N)); Qn[np.ix_(idx, idx)] = np.eye(len(idx)) - Tn @ Tn.T   # centred network
    res["Pb_commutator"] = float(max(np.linalg.norm(Dg @ Pb - Pb @ Dg) for _, Dg in ops))
    lab = mmm_irreps(ops)
    P = {g: sum(c * Dg for c, (_, Dg) in zip(chi, ops)) / len(ops) for g, chi in lab.items()}
    res["proj_completeness"] = float(np.linalg.norm(sum(P.values()) - np.eye(3 * N)))
    bnum = lambda x: float(x @ Pb @ x); nnum = lambda x: float(x @ Qn @ x)
    # breathing drives (pure central, s_f = 0)
    fj = []
    for j in Bs:
        tj = np.array([1.0 if (i == j or kk == j) else 0.0 for i, kk, _ in bonds])
        x = np.zeros(3 * N); x[idx] = -J.T @ tj; fj.append(Q @ x)
    rB = frac[Bs] @ L
    sect = {}
    f1 = fj[0]; u1 = Hp(f1)
    for g in lab:
        fg = P[g] @ f1
        if np.linalg.norm(fg) < 1e-8 * np.linalg.norm(f1):
            continue
        ug = Hp(fg)
        pat = np.array([fg @ x for x in fj]); pat = np.sign(np.round(pat / np.abs(pat).max(), 6))
        sect[g] = {"pattern_on_B": pat.tolist(), "f2_frac": float(fg @ fg / (f1 @ f1)),
                   "u2_net_frac": nnum(ug) / nnum(u1), "s_u": bnum(ug) / nnum(ug),
                   "compliance": float(np.linalg.norm(ug) / np.linalg.norm(fg))}
    res["single_site_direct_s_u"] = bnum(u1) / nnum(u1)
    res["single_site_sumrule_s_u"] = sum(v["s_u"] * v["u2_net_frac"] for v in sect.values())
    res["sum_of_u2_fracs"] = sum(v["u2_net_frac"] for v in sect.values())
    res["breathing_sectors"] = sect
    res["B_positions_cart"] = np.round(rB, 3).tolist()
    # compression drive sector weights
    fq = Q @ f0
    res["compression_f2_by_irrep"] = {g: float(fq @ P[g] @ fq / (fq @ fq)) for g in lab}
    # sector-resolved floppy dimension, Schur stiffness, rigid-rotation (RUM-like) content
    ev, Rr = np.linalg.eigh(Q - Pb); R = Rr[:, ev > 0.5]
    rot = []
    for j in Bs:
        for a in range(3):
            e = np.eye(3)[a]; v = np.zeros(3 * N)
            for i, kk, img in bonds:
                if i == j or kk == j:
                    o = kk if i == j else i
                    dvec = (frac[o] - frac[j] + (np.array(img) if i == j else -np.array(img))) @ L
                    v[3 * o:3 * o + 3] += np.cross(e, dvec)
            rot.append(Q @ v)
    rot = np.array(rot).T
    per = {}
    for g in lab:
        Bg = np.linalg.svd(P[g] @ B, full_matrices=False); Bg = Bg[0][:, Bg[1] > 0.5]
        Rg = np.linalg.svd(P[g] @ R, full_matrices=False); Rg = Rg[0][:, Rg[1] > 0.5]
        Ug = np.linalg.svd(P[g] @ rot, full_matrices=False); Ug = Ug[0][:, Ug[1] > 1e-6 * Ug[1].max()] if Ug[1].max() > 1e-8 else np.zeros((3 * N, 0))
        row = {"dim_total": int(round(np.trace(P[g] @ Q))), "dim_b": Bg.shape[1], "dim_r": Rg.shape[1], "dim_rot": Ug.shape[1]}
        if Ug.shape[1]:
            row["rot_b_content"] = np.round(np.linalg.eigvalsh(Ug.T @ Pb @ Ug), 3).tolist()
        if Bg.shape[1]:
            Hbb = Bg.T @ Hq @ Bg
            if Rg.shape[1]:
                Hbr = Bg.T @ Hq @ Rg; Hrr = Rg.T @ Hq @ Rg
                Sb = Hbb - Hbr @ np.linalg.solve(Hrr, Hbr.T)
            else:
                Sb = Hbb
            lam, vec = np.linalg.eigh(Sb)
            soft = Bg @ vec[:, 0]
            row["Sb_min"] = float(lam[0]); row["Sb_eigs_low3"] = np.round(lam[:3], 3).tolist()
            Prot = np.linalg.svd(rot, full_matrices=False)[0]
            row["soft_rot_overlap"] = float(np.sum((Prot.T @ soft) ** 2))
        per[g] = row
    Prot = np.linalg.svd(rot, full_matrices=False)[0]
    res["rot_baseline_overlap_random_b"] = float(np.trace(Prot.T @ Pb @ Prot) / B.shape[1])
    res["sectors"] = per
    return res


for cd in ("ca_tio3", "ca_sno3"):
    out[cd] = perovskite(cd)

(HERE / "closed_loop.json").write_text(json.dumps(out, indent=1, default=float))
print("k12 s_f", round(out["validate_k12_s_f"], 5))
for cd in ("ca_tio3", "ca_sno3"):
    r = out[cd]; v = r["validate"]
    print(f"== {cd}: SG {r['spacegroup']} ops {r['n_ops']} s_f {v['s_f']:.4f} [H,D] {r['H_asym_commutator']:.1e} [Pb,D] {r['Pb_commutator']:.1e} complete {r['proj_completeness']:.1e}")
    print(f"  single-site s_u direct {r['single_site_direct_s_u']:.4f} sumrule {r['single_site_sumrule_s_u']:.4f} (sum w {r['sum_of_u2_fracs']:.4f})")
    for g, s in r["breathing_sectors"].items():
        print(f"   {g}: pat {s['pattern_on_B']} f2 {s['f2_frac']:.3f} u2 {s['u2_net_frac']:.3f} s_u {s['s_u']:.4f} c {s['compliance']:.3f}")
    print("  compression f2:", {g: round(x, 4) for g, x in r["compression_f2_by_irrep"].items() if x > 1e-4})
    for g, s in r["sectors"].items():
        print("  ", g, {k: s[k] for k in s})
    print("  rot baseline", round(r["rot_baseline_overlap_random_b"], 3))
