"""Field-theory / Schur checks on existing force constants (numpy only, read-only on data/).
Prints compact numbers; writes checks.json next to this script."""
import json
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parents[2] / "data"
OUT = Path(__file__).resolve().parent


def jac(frac, L, bonds, net):
    col = {int(a): c for c, a in enumerate(net)}
    J = np.zeros((len(bonds), 3 * len(net)))
    for r, (i, j, img) in enumerate(bonds):
        v = (frac[j] - frac[i] + np.array(img)) @ L
        e = v / np.linalg.norm(v)
        J[r, 3 * col[i]:3 * col[i] + 3] = -e
        J[r, 3 * col[j]:3 * col[j] + 3] = e
    return J


def full_H(read, N, delta):
    H = np.zeros((3 * N, 3 * N))
    for i in range(N):
        for a, ax in enumerate("xyz"):
            H[3 * i + a] = -(read(f"d{i:02d}{ax}p") - read(f"d{i:02d}{ax}m")).ravel() / (2 * delta)
    return 0.5 * (H + H.T)


def analyse(name, H, f, frac, L, bonds, net, N, u_act=None, Bsites=None):
    T = np.zeros((3 * N, 3))
    for a in range(3):
        T[a::3, a] = 1 / np.sqrt(N)
    Q = np.eye(3 * N) - T @ T.T
    H = Q @ H @ Q
    f = Q @ f
    w, V = np.linalg.eigh(H)
    keep = np.argsort(np.abs(w))[3:]
    w, V = w[keep], V[:, keep]
    Hp = lambda x: V @ ((V.T @ x) / w)
    J = jac(frac, L, bonds, net)
    # null(C) on network coords, minus network translation, embedded in 3N
    _, s, Vt = np.linalg.svd(J)
    rank = int(np.sum(s > s[0] * 1e-10))
    Nb = Vt[rank:].T                                    # 3n x k
    Tn = np.zeros((3 * len(net), 3))
    for a in range(3):
        Tn[a::3, a] = 1 / np.sqrt(len(net))
    Nb = Nb - Tn @ (Tn.T @ Nb)
    Ub, sb, _ = np.linalg.svd(Nb, full_matrices=False)
    Nb = Ub[:, sb > 1e-8]
    idx = np.concatenate([[3 * a, 3 * a + 1, 3 * a + 2] for a in net])
    B = np.zeros((3 * N, Nb.shape[1]))
    B[idx] = Nb
    # complement r in translation-free space
    W, ev = np.linalg.eigh(Q - B @ B.T)
    R = ev[:, W > 0.5]
    Pb = B @ B.T

    def s_net(x):  # code definition: network atoms, centred
        v = x.reshape(-1, 3)[net]
        v = (v - v.mean(0)).ravel()
        return float(np.sum((Nb.T @ v) ** 2) / (v @ v))

    u = Hp(f)
    res = {"system": name, "dim_b": B.shape[1], "dim_r": R.shape[1]}
    res["s_f"] = s_net(f)
    res["s_u_pred"] = s_net(u)
    res["eps_pred"] = res["s_u_pred"] / res["s_f"]
    if u_act is not None:
        res["s_u_act"] = s_net(u_act)
    # (1) least-squares bond-tension residual == s_f
    v = f.reshape(-1, 3)[net]
    v = (v - v.mean(0)).ravel()
    t, *_ = np.linalg.lstsq(J.T, v, rcond=None)
    res["lsq_residual_minus_s_f"] = float(np.sum((v - J.T @ t) ** 2) / (v @ v)) - res["s_f"]
    # (2) Schur form in full space
    Hbb, Hbr, Hrr = B.T @ H @ B, B.T @ H @ R, R.T @ H @ R
    fb, fr = B.T @ f, R.T @ f
    ftr = -Hbr @ np.linalg.solve(Hrr, fr)               # force transmitted through the coupling
    Sb = Hbb - Hbr @ np.linalg.solve(Hrr, Hbr.T)       # Schur complement (b-stiffness, r relaxed)
    ub = np.linalg.solve(Sb, fb + ftr)
    res["schur_check"] = float(np.linalg.norm(B @ ub - Pb @ u) / np.linalg.norm(Pb @ u))
    sf_full = fb @ fb / (f @ f)
    su_full = ub @ ub / (u @ u)
    feff = fb + ftr
    res["sf_full"], res["su_full"], res["eps_full"] = float(sf_full), float(su_full), float(su_full / sf_full)
    res["tau"] = float(feff @ feff / (fb @ fb))         # transmission amplification of the b-force
    res["ftr_over_fb"] = float(np.linalg.norm(ftr) / np.linalg.norm(fb))
    res["cos_fb_ftr"] = float(fb @ ftr / np.linalg.norm(fb) / np.linalg.norm(ftr))
    cb = np.linalg.norm(ub) / np.linalg.norm(feff)
    c = np.linalg.norm(u) / np.linalg.norm(f)
    res["rho"] = float((cb / c) ** 2)                   # eps_full = tau * rho exactly
    res["eps_check"] = float(res["tau"] * res["rho"] / res["eps_full"] - 1)
    # crude a-priori: mean stiffnesses, no coupling
    kb = np.trace(Hbb) / Hbb.shape[0]
    kall = np.trace(H) / (3 * N - 3)
    res["k_b_mean"], res["k_r_mean"] = float(kb), float(np.trace(Hrr) / Hrr.shape[0])
    res["eps_naive_(kall/kb)^2"] = float((kall / kb) ** 2)
    # direct-only (no coupling) and coupling-only estimates
    ud = np.linalg.solve(Hbb, fb)
    res["eps_direct_only"] = float((ud @ ud / (u @ u)) / sf_full)
    # (3) spectral distribution over eigenmodes of H
    p = np.sum((B.T @ V) ** 2, axis=0)                 # b-content of each mode
    fn = V.T @ f
    a_f, a_u = fn ** 2, (fn / w) ** 2
    res["sf_diag"] = float(np.sum(p * a_f) / a_f.sum())
    res["su_diag_no_cross"] = float(np.sum(p * a_u) / a_u.sum())
    order = np.argsort(w)
    n10 = max(1, len(w) // 10)
    soft = order[:n10]
    res["softest10pct_share_of_u2"] = float(a_u[soft].sum() / a_u.sum())
    res["softest10pct_share_of_f2"] = float(a_f[soft].sum() / a_f.sum())
    res["softest10pct_mean_b_content"] = float(p[soft].mean())
    res["mean_b_content_all"] = float(p.mean())
    # (4) model drives: single-site and uniform central breathing of the B cations (f = -C^T t, s_f = 0)
    if Bsites is not None:
        out = []
        tot = np.zeros(3 * N)
        for j in Bsites:
            tj = np.array([1.0 if (i == j or k == j) else 0.0 for i, k, _ in bonds])
            fn_ = np.zeros(3 * N)
            fn_[idx] = -J.T @ tj
            fn_ = Q @ fn_
            tot += fn_
            uj = Hp(fn_)
            out.append(s_net(uj))
        res["breathing_single_s_f"] = s_net(fn_)
        res["breathing_single_s_u_mean"] = float(np.mean(out))
        res["breathing_single_s_u_range"] = [float(min(out)), float(max(out))]
        res["breathing_uniform_s_u"] = s_net(Hp(tot))
        if len(Bsites) == 4:
            pats = {}
            fj = []
            for j in Bsites:
                tj = np.array([1.0 if (i == j or k == j) else 0.0 for i, k, _ in bonds])
                x = np.zeros(3 * N); x[idx] = -J.T @ tj; fj.append(Q @ x)
            for sg in ((1, 1, 1, 1), (1, 1, -1, -1), (1, -1, 1, -1), (1, -1, -1, 1)):
                fp = sum(c * x for c, x in zip(sg, fj))
                up = Hp(fp)
                pats[str(sg)] = [s_net(up), float(np.linalg.norm(up) / np.linalg.norm(fp))]
            res["breathing_patterns_s_u_compliance"] = pats
            res["Ti_frac"] = [list(map(float, np.round(frac[j], 3))) for j in Bsites]
    return res


rows = []
# k12 tight, 4->5 GPa
m = json.loads((D / "spinel_pressure_path/force_balance/meta.json").read_text())
sp = np.repeat(m["species"], m["counts"]); N = len(sp)
H = full_H(lambda c: np.load(D / "force_balance_reference/forces" / f"{c}.npy"), N, m["delta_A"])
f0 = np.load(D / "spinel_pressure_path/force_balance/forces/ref.npy").ravel()
bonds = json.loads((D / "force_balance_reference/inputs/pressure_bond_accommodation.json").read_text())["bonds"]
net = np.flatnonzero(np.isin(sp, ["Sc", "Sb", "Se"]))
L = np.array(m["lattice_5GPa"]); fs, fe = np.array(m["frac_start"]), np.array(m["frac_5GPa"])
du = fe - fs; du -= np.round(du)
rows.append(analyse("k12 x=0.75 4-5GPa", H, f0, fs, L, bonds, net, N, (du @ L).ravel(),
                    np.flatnonzero(np.isin(sp, ["Sc", "Sb"]))))
H_k12 = H
# CaTiO3, CaSnO3
for nm, cd in (("CaTiO3 0-5GPa", "ca_tio3"), ("CaSnO3 0-5GPa", "ca_sno3")):
    d = D / "bending_enrichment/full_force_balance" / cd / "p0_p5"
    m = json.loads((d / "meta.json").read_text())
    N = sum(m["counts"]); sp = np.repeat(m["species"], m["counts"])
    rd = lambda c: np.load(d / "hessian" / c / "forces.npy")
    H = full_H(rd, N, m["delta_A"])
    f0 = rd("ref").ravel()
    bonds = m["bonds"]
    net = np.array(sorted({b[0] for b in bonds} | {b[1] for b in bonds}))
    L = np.array(m["lattice_P2"]); fs, fe = np.array(m["frac_start"]), np.array(m["frac_P2"])
    du = fe - fs; du -= np.round(du)
    rows.append(analyse(nm, H, f0, fs, L, bonds, net, N, (du @ L).ravel(),
                        np.flatnonzero(sp == m["species"][1])))
# i1112 substitution (approximate start geometry: S12 parent minus H^+f), H and f deposited
txt = (D / "alloy_local_geometry/inputs/k12_parent_contcar.vasp").read_text().split("\n")
L = np.array([[float(x) for x in txt[k].split()] for k in (2, 3, 4)]) * float(txt[1])
names, cnt = txt[5].split(), [int(x) for x in txt[6].split()]
N = sum(cnt); sp = np.repeat(names, cnt)
fr = np.array([[float(x) for x in txt[8 + i].split()[:3]] for i in range(N)])
for iv in ("i1112",):
    H = np.load(D / "alloy_leverage/runs" / iv / "H.npy"); f = np.load(D / "alloy_leverage/runs" / iv / "f.npy").ravel()
    Hs = 0.5 * (H + H.T)
    w_, V_ = np.linalg.eigh(Hs); k_ = np.argsort(np.abs(w_))[3:]
    up = V_[:, k_] @ ((V_[:, k_].T @ f) / w_[k_])
    fa = fr - (up.reshape(-1, 3) @ np.linalg.inv(L))   # approx start = end - u_pred
    Bi = [i for i in range(N) if sp[i] in ("Sc", "Sb")]; X = [i for i in range(N) if sp[i] == "Se"]
    bnd = []
    for i in Bi:
        dd = fa[X] - fa[i]; img = -np.round(dd)
        r = np.linalg.norm((dd + img) @ L, axis=1)
        for k in np.argsort(r)[:6]:
            bnd.append((i, X[k], img[k].astype(int).tolist()))
    netw = np.array(sorted(Bi + X))
    rows.append(analyse(f"{iv} substitution (approx geom)", Hs, f, fa, L, bnd, netw, N, None, None))

(OUT / "checks.json").write_text(json.dumps(rows, indent=1))
for r in rows:
    print("==", r["system"])
    print("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items() if k != "system"))
