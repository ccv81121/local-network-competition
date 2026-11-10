"""Analysis of DISPERSION_20261104 strictly against prereg_dispersion.md (T1-T4). python -I analyse_dispersion_v1.py
Inputs in raw/ (forces from OUTCAR, POSCARs). Output dispersion_result.json. No threshold/cell/sector definition is tuned."""
import json
from pathlib import Path
import numpy as np, spglib
from scipy.optimize import curve_fit
HERE = Path(__file__).resolve().parent; RAW = HERE / "raw"


def rd_poscar(p):
    L = open(p).read().split("\n"); lat = np.array([[float(x) for x in L[i].split()] for i in (2, 3, 4)])
    el, cnt = L[5].split(), [int(x) for x in L[6].split()]; n = sum(cnt)
    fr = np.array([[float(x) for x in L[8 + i].split()[:3]] for i in range(n)])
    return lat, [e for e, c in zip(el, cnt) for _ in range(c)], fr


OUT = {}
lat, sp0, fr0 = rd_poscar(RAW / "POSCAR_ref"); N = len(sp0); sp0 = np.array(sp0)
meta = json.loads((RAW / "meta.json").read_text()); DEL = meta["delta_A"]; reps = meta["reps"]
# ---- symmetry
nums = {"Ca": 20, "Ti": 22, "O": 8}
ds = spglib.get_symmetry_dataset((lat, fr0, [nums[s] for s in sp0]), symprec=1e-4)
rots, trs = np.array(ds.rotations), np.array(ds.translations); nops = len(rots)
perm = []
for R, t in zip(rots, trs):
    x = fr0 @ R.T + t; p = np.empty(N, int)
    for i in range(N):
        d = fr0 - x[i]; d -= np.round(d); k = np.argmin(np.linalg.norm(d @ lat, axis=1))
        assert np.linalg.norm(d[k] @ lat) < 2e-3 and sp0[k] == sp0[i]; p[i] = k
    assert len(set(p)) == N; perm.append(p)
LT = lat.T; Rc = [LT @ R @ np.linalg.inv(LT) for R in rots]
OUT["n_ops"] = nops
# ---- H from displacements
F = {j: np.loadtxt(RAW / f"F_{j}_tmp") for j in ["ref"] + [f"d{r:02d}{a}{s}" for r in reps for a in "xyz" for s in "pm"]}
dmax = []
for r in reps:
    _, _, f2 = rd_poscar(RAW / f"POSCAR_d{r:02d}xp"); dd = f2 - fr0; dd -= np.round(dd); dmax.append(float(np.abs(dd @ lat).max()))
OUT["displacement_amplitude_A_from_POSCAR"] = dmax
cols = {}
for r in reps:
    C = np.zeros((3 * N, 3))
    for a, ax in enumerate("xyz"):
        C[:, a] = -(F[f"d{r:02d}{ax}p"] - F[f"d{r:02d}{ax}m"]).ravel() / (2 * DEL)
    cols[r] = C
H = np.zeros((3 * N, 3 * N)); H_list = {i: [] for i in range(N)}
for r in reps:
    for g in range(nops):
        i = perm[g][r]; Rg = Rc[g]; blk = np.zeros((3 * N, 3))
        for j in range(N):
            blk[3 * perm[g][j]:3 * perm[g][j] + 3] = Rg @ cols[r][3 * j:3 * j + 3] @ Rg.T
        H_list[i].append(blk)
spread = 0.0; scale = float(np.abs(np.concatenate([c.ravel() for c in cols.values()])).max())
for i in range(N):
    B = np.array(H_list[i]); assert len(B) > 0
    H[:, 3 * i:3 * i + 3] = B.mean(0); spread = max(spread, float(np.abs(B - B.mean(0)).max()))
OUT["H_max_abs_eVA2"] = scale; OUT["rep_fill_spread_max_eVA2"] = spread; OUT["rep_fill_spread_rel"] = spread / scale
OUT["asymmetry_rel_Fro_before_symmetrising"] = float(np.linalg.norm(H - H.T) / np.linalg.norm(H))
Hs = 0.5 * (H + H.T)
S = Hs.reshape(N, 3, N, 3).sum(axis=2)
OUT["ASR_row_sum_max_abs_eVA2"] = float(np.abs(S).max()); OUT["ASR_row_sum_rel_to_Hmax"] = float(np.abs(S).max() / scale)
OUT["ASR_row_sum_Fro_over_H_Fro"] = float(np.linalg.norm(S) / np.linalg.norm(Hs))
Tm = np.zeros((3 * N, 3))
for a in range(3): Tm[a::3, a] = 1 / np.sqrt(N)
Q = np.eye(3 * N) - Tm @ Tm.T
HQ = Q @ Hs @ Q
OUT["lowest_6_eig_Hs_eVA2"] = [float(x) for x in np.linalg.eigvalsh(Hs)[:6]]
wq = np.linalg.eigvalsh(HQ); wk = wq[np.argsort(np.abs(wq))[3:]]
OUT["HQ_3_smallest_abs_eig"] = [float(x) for x in np.sort(np.abs(wq))[:3]]
OUT["HQ_lowest_nontrivial_eig_eVA2"] = float(wk.min()); OUT["HQ_n_negative_nontrivial"] = int((wk < 0).sum())
Hd = Hs.copy()
for i in range(N): Hd[3 * i:3 * i + 3, 3 * i:3 * i + 3] -= S[i]
Hd = 0.5 * (Hd + Hd.T)
# ---- reorder to s1 (substituted Ti = s0 idx16 -> Sn at s1 idx31)
lat1, sp1, fr1 = rd_poscar(RAW / "s1_POSCAR"); sp1 = np.array(sp1)
assert np.allclose(lat1, lat, atol=1e-6)
s1_of_s0 = np.empty(N, int)
for i in range(N):
    d = fr1 - fr0[i]; d -= np.round(d); k = np.argmin(np.linalg.norm(d @ lat, axis=1)); assert np.linalg.norm(d[k] @ lat) < 5e-3; s1_of_s0[i] = k
OUT["s1_index_of_s0_idx16"] = int(s1_of_s0[16])
OUT["s0_to_s1_max_pos_mismatch_A"] = float(max(np.linalg.norm(((fr1[s1_of_s0[i]] - fr0[i]) - np.round(fr1[s1_of_s0[i]] - fr0[i])) @ lat) for i in range(N)))
P = np.zeros((N, N)); P[s1_of_s0, np.arange(N)] = 1
P3 = np.kron(P, np.eye(3))
HQ1, Hd1 = P3 @ HQ @ P3.T, P3 @ Hd @ P3.T


def pinv_u(Hm, f):
    w, V = np.linalg.eigh(0.5 * (Hm + Hm.T)); k = np.argsort(np.abs(w))[3:]; w, V = w[k], V[:, k]
    return V @ ((V.T @ f) / w)


# ---- drive, actual response
Fa = np.loadtxt(RAW / "F_alloy_c1"); f = Fa.ravel(); fc = Q @ f
OUT["drive_net_force_eVA"] = [float(x) for x in Fa.sum(0)]
_, spr, frr = rd_poscar(RAW / "CONTCAR"); assert list(spr) == list(sp1)
du = frr - fr1; du -= np.round(du); uact = du @ lat; uact_c = (uact - uact.mean(0)).ravel()
upred = pinv_u(HQ1, fc)
OUT["u_rms_pred_A"] = float(np.sqrt(np.mean(np.sum(upred.reshape(-1, 3) ** 2, 1))))
OUT["u_rms_act_A_all80_centroid_removed"] = float(np.sqrt(np.mean(np.sum(uact_c.reshape(-1, 3) ** 2, 1))))
B_idx = np.flatnonzero(np.isin(sp1, ["Ti", "Sn"])); X_idx = np.flatnonzero(sp1 == "O"); net = np.flatnonzero(np.isin(sp1, ["Ti", "Sn", "O"]))
col = {int(a): k for k, a in enumerate(net)}; bonds = []
for b in B_idx:
    dv = fr1[X_idx] - fr1[b]; dv -= np.round(dv); dc = dv @ lat; dist_ = np.linalg.norm(dc, axis=1)
    for k in np.argsort(dist_)[:6]: bonds.append((int(b), int(X_idx[k]), dc[k], float(dist_[k])))
J = np.zeros((len(bonds), 3 * len(net)))
for r, (b, x, v, l) in enumerate(bonds):
    e = v / l; J[r, 3 * col[b]:3 * col[b] + 3] = -e; J[r, 3 * col[x]:3 * col[x] + 3] = e
_, sv, Vt = np.linalg.svd(J, full_matrices=False); Rr = Vt[:int(np.sum(sv > sv[0] * 1e-10))]
Kn = np.eye(3 * len(net)) - Rr.T @ Rr
idx = np.concatenate([[3 * a, 3 * a + 1, 3 * a + 2] for a in net])


def center_net(x):
    v = x.reshape(-1, 3)[net]; return (v - v.mean(0)).ravel()


def s_net(x):
    v = center_net(x); return float(np.sum((Kn @ v) ** 2) / (v @ v))


OUT["bond_rank"] = int(Rr.shape[0])
OUT["s_f_check_ref_0.089pct"] = 100 * s_net(f); OUT["s_u_act_check_ref_24.8pct"] = 100 * s_net(uact.ravel())
# ---- sensitivity: negative modes of H
w1, V1 = np.linalg.eigh(0.5 * (HQ1 + HQ1.T)); k1 = np.argsort(np.abs(w1))[3:]; w1, V1 = w1[k1], V1[:, k1]; a1 = (V1.T @ fc) / w1
neg = w1 < 0
OUT["sens_negative_modes"] = {"n_neg": int(neg.sum()), "eigs": [float(x) for x in w1[neg]], "frac_of_u2_from_negative_modes": float(np.sum(a1[neg] ** 2) / np.sum(a1 ** 2)),
    "frac_of_f2_in_negative_modes": float(np.sum((V1.T @ fc)[neg] ** 2) / (fc @ fc))}
u_abs = V1 @ ((V1.T @ fc) / np.abs(w1)); u_pos = V1[:, ~neg] @ ((V1[:, ~neg].T @ fc) / w1[~neg])
OUT["sens_negative_modes"].update({"abs_inverse": {"cos_T1": float(u_abs @ uact_c / np.linalg.norm(u_abs) / np.linalg.norm(uact_c)), "s_u_pct": 0.0},
    "drop_negative": {"cos_T1": float(u_pos @ uact_c / np.linalg.norm(u_pos) / np.linalg.norm(uact_c)), "s_u_pct": 0.0}})
# ---- T1
nrm = np.linalg.norm
cosT1 = float(upred @ uact_c / nrm(upred) / nrm(uact_c))
ud = pinv_u(Hd1, fc)
cn = lambda a, b: float(center_net(a) @ center_net(b) / nrm(center_net(a)) / nrm(center_net(b)))
OUT["T1"] = {"cos_all80": cosT1, "norm_ratio_pred_over_act": float(nrm(upred) / nrm(uact_c)), "threshold": 0.90, "verdict": "PASS" if cosT1 >= 0.90 else "FAIL",
             "cos_pred_QHQ_vs_pred_diagASR": float(upred @ ud / nrm(upred) / nrm(ud)), "cos_all80_with_diagASR_H": float(ud @ uact_c / nrm(ud) / nrm(uact_c)),
             "cos_network_centred_only": cn(upred, uact.ravel())}
# ---- T2
s2 = 100 * s_net(upred)
OUT["T2"] = {"s_u_pred_pct": s2, "s_u_actual_pct": 100 * s_net(uact.ravel()), "window_pct": [17.4, 32.2], "verdict": "PASS" if 17.4 <= s2 <= 32.2 else "FAIL",
             "s_u_pred_diagASR_H_pct": 100 * s_net(ud), "s_f_pct": 100 * s_net(f), "enrichment_s_u_pred_over_s_f": s2 / (100 * s_net(f))}
# ---- T3
sn = int(np.flatnonzero(sp1 == "Sn")[0])


def mind(fr_all, ref, L):
    d = fr_all - ref; d = d - np.round(d); best = np.full(len(fr_all), 1e9)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for c in (-1, 0, 1): best = np.minimum(best, np.linalg.norm((d + [a, b, c]) @ L, axis=1))
    return best


dist = mind(fr1, fr1[sn], lat)


def shells(d, mag, width=1.0, drop_src=True):
    m = d > 1e-6 if drop_src else np.ones(len(d), bool)
    bins = np.floor(d[m] / width).astype(int); out = []
    for b in np.unique(bins):
        s = bins == b; out.append((float(d[m][s].mean()), float(mag[m][s].mean()), int(s.sum())))
    return np.array(out)


def fit_exp(sh):
    x, y = sh[:, 0], sh[:, 1]; res = {}
    try:
        p, _ = curve_fit(lambda d, A, xi: A * np.exp(-d / xi), x, y, p0=[y[0] * 2, 2.0], bounds=([0, 0.05], [1e3, 1e3]), maxfev=20000)
        r2 = 1 - np.sum((y - p[0] * np.exp(-x / p[1])) ** 2) / np.sum((y - y.mean()) ** 2); res["exp"] = {"A": float(p[0]), "xi_A": float(p[1]), "R2": float(r2)}
    except Exception as e: res["exp"] = {"error": str(e)}
    try:
        p, _ = curve_fit(lambda d, A, n: A * d ** (-n), x, y, p0=[y[0], 1.0], bounds=([0, 0], [1e4, 12]), maxfev=20000)
        r2 = 1 - np.sum((y - p[0] * x ** (-p[1])) ** 2) / np.sum((y - y.mean()) ** 2); res["power"] = {"A": float(p[0]), "n": float(p[1]), "R2": float(r2)}
    except Exception as e: res["power"] = {"error": str(e)}
    return res


mag_act = nrm(uact_c.reshape(-1, 3), axis=1); mag_pred = nrm(upred.reshape(-1, 3), axis=1)
shA, shP = shells(dist, mag_act), shells(dist, mag_pred)
T3 = {"binning": "1.0 A shells of minimum-image distance from the Sn site, Sn site itself excluded, all species averaged", "d_max_A": float(dist.max()),
      "actual": {"shells": shA.tolist(), **fit_exp(shA)}, "pred_supercell": {"shells": shP.tolist(), **fit_exp(shP)}}
# kernel: Fourier-interpolated H(q), minimum-image real-space force constants of the supercell (diag-ASR H, s1 ordering)
Nq = (6, 10, 6); HR = {}
for i in range(N):
    for j in range(N):
        raw = fr1[j] - fr1[i]; d0 = raw - np.round(raw)
        cands = [(nrm((d0 + np.array(m)) @ lat), m) for m in [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]]
        dmn = min(c[0] for c in cands); tie = [m for dd, m in cands if dd < dmn + 1e-4]
        for m in tie:
            Rint = tuple(np.round(d0 + np.array(m) - raw).astype(int))
            HR.setdefault(Rint, np.zeros((3 * N, 3 * N)))[3 * i:3 * i + 3, 3 * j:3 * j + 3] += Hd1[3 * i:3 * i + 3, 3 * j:3 * j + 3] / len(tie)
OUT["kernel_n_R_vectors"] = len(HR); OUT["kernel_sum_R_equals_Hd1_max_err"] = float(np.abs(sum(HR.values()) - Hd1).max())
qs = np.array([(a / Nq[0], b / Nq[1], c / Nq[2]) for a in range(Nq[0]) for b in range(Nq[1]) for c in range(Nq[2])])
Rkeys = list(HR); Rarr = np.array(Rkeys)
Uq = np.zeros((len(qs), 3 * N), complex); Uqa = np.zeros((len(qs), 3 * N), complex); nneg_q = 0; wmin = 1e9; herm = 0
for iq, q in enumerate(qs):
    Hq = sum(HR[k] * np.exp(2j * np.pi * (np.array(k) @ q)) for k in Rkeys)
    herm = max(herm, float(np.abs(Hq - Hq.conj().T).max())); Hq = 0.5 * (Hq + Hq.conj().T); w, V = np.linalg.eigh(Hq)
    if np.allclose(q, 0): k = np.argsort(np.abs(w))[3:]; w, V = w[k], V[:, k]
    else: wmin = min(wmin, float(w.min()))
    Uq[iq] = V @ ((V.conj().T @ fc) / w); Uqa[iq] = V @ ((V.conj().T @ fc) / np.abs(w)); nneg_q += int((w < 0).sum() > 0)
OUT["kernel_n_q_with_negative_eig"] = nneg_q; OUT["kernel_n_q"] = len(qs); OUT["kernel_min_eig_Hq_nonzero_q_eVA2"] = wmin; OUT["kernel_hermiticity_max"] = herm
Rgrid = np.array([(a, b, c) for a in range(Nq[0]) for b in range(Nq[1]) for c in range(Nq[2])])
Ureal = ((np.exp(2j * np.pi * (Rgrid @ qs.T)) / len(qs)) @ Uq).real.reshape(len(Rgrid), N, 3)
Lbig = lat * np.array(Nq)[:, None]
posBig = ((fr1[None, :, :] + Rgrid[:, None, :]) / np.array(Nq)).reshape(-1, 3)
dk = mind(posBig, fr1[sn] / np.array(Nq), Lbig); magk = nrm(Ureal.reshape(-1, 3), axis=1)
rng = dist.max()
# kernel sampled at the same atoms/range as the actual u (cell 0 images only) and extended range
msel = dk <= rng + 1e-9; shK = shells(dk[msel], magk[msel]); T3["kernel_same_range"] = {"shells": shK.tolist(), **fit_exp(shK)}
mext = dk <= 20.0; shKe = shells(dk[mext], magk[mext]); T3["kernel_to_20A"] = {"shells": shKe.tolist(), **fit_exp(shKe)}
T3["kernel_u_at_cell0_vs_supercell_pred_cos"] = float(Ureal[0].ravel() @ upred / nrm(Ureal[0].ravel()) / nrm(upred))
Ureal_a = ((np.exp(2j * np.pi * (Rgrid @ qs.T)) / len(qs)) @ Uqa).real.reshape(len(Rgrid), N, 3); magka = nrm(Ureal_a.reshape(-1, 3), axis=1)
shKa = shells(dk[msel], magka[msel]); T3["kernel_absinverse_same_range"] = {"shells": shKa.tolist(), **fit_exp(shKa)}
shKae = shells(dk[mext], magka[mext]); T3["kernel_absinverse_to_20A"] = {"shells": shKae.tolist(), **fit_exp(shKae)}
T3["actual_incl_Sn_site"] = fit_exp(shells(dist, mag_act, drop_src=False))
m_ = dist > 1e-6; T3["actual_per_atom_fit"] = fit_exp(np.c_[dist[m_], mag_act[m_]])
lim = 0.5 * float(lat.diagonal().min()); T3["xi_limit_0.5xshortest_edge_A"] = lim


def judge(r):
    e = r.get("exp", {})
    if "xi_A" not in e: return "ERR"
    return "INCONCLUSIVE" if (e["xi_A"] > lim or e["R2"] < 0.8) else "ok"


ja, jk = judge(T3["actual"]), judge(T3["kernel_same_range"]); T3["rule_actual"], T3["rule_kernel"] = ja, jk
xa, xk = T3["actual"]["exp"].get("xi_A"), T3["kernel_same_range"]["exp"].get("xi_A")
if "INCONCLUSIVE" in (ja, jk): v = "INCONCLUSIVE"
elif xa and xk: T3["xi_ratio_kernel_over_actual"] = xk / xa; v = "PASS" if abs(xk / xa - 1) <= 0.30 else "FAIL"
else: v = "ERR"
T3["verdict"] = v; OUT["T3"] = T3
OUT["sens_negative_modes"]["abs_inverse"]["s_u_pct"] = 100 * s_net(u_abs); OUT["sens_negative_modes"]["drop_negative"]["s_u_pct"] = 100 * s_net(u_pos)
# ---- T4
idr = [g for g in range(nops) if np.all(rots[g] == np.eye(3, dtype=int))]
OUT["T4_translations_frac"] = [list(np.round(trs[g] % 1, 4)) for g in idr]


def Umat(g):
    U = np.zeros((3 * N, 3 * N))
    for i in range(N): U[3 * perm[g][i]:3 * perm[g][i] + 3, 3 * i:3 * i + 3] = Rc[g]
    return U


Ug = [Umat(g) for g in range(nops)]
qk = lambda t: tuple(int(round(2 * x)) % 2 for x in t)
ks = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)]
names = {(0, 0, 0): "Gamma20", (1, 0, 0): "X20(1/2,0,0)", (0, 0, 1): "Z20(0,0,1/2)", (1, 0, 1): "U20(1/2,0,1/2)"}
Pi = {}
for k in ks:
    M = np.zeros((3 * N, 3 * N))
    for g in idr:
        tk = qk(trs[g]); M += (-1) ** (k[0] * tk[0] + k[1] * tk[1] + k[2] * tk[2]) * Ug[g]
    Pi[k] = M / len(idr)
Pirr = {}
for e in [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]:
    M = np.zeros((3 * N, 3 * N))
    for g in range(nops):
        d = np.diag(Rc[g]); sgn = 1
        for ax in range(3):
            if d[ax] < -0.5: sgn *= e[ax]
        M += sgn * Ug[g]
    Pirr[e] = M / nops
OUT["T4_checks"] = {"sum_Pi_q_minus_I": float(np.abs(sum(Pi.values()) - np.eye(3 * N)).max()),
                    "idempotent_max": float(max(np.abs(Pi[k] @ Pi[k] - Pi[k]).max() for k in ks)),
                    "commute_with_H_max": float(max(np.abs(Pi[k] @ Hs - Hs @ Pi[k]).max() for k in ks)),
                    "sum_Pi_irreps_minus_Pi_Gamma20": float(np.abs(sum(Pirr.values()) - Pi[(0, 0, 0)]).max())}
to1 = lambda M: P3 @ M @ P3.T
Pi1 = {k: to1(M) for k, M in Pi.items()}; Pir1 = {k: to1(M) for k, M in Pirr.items()}
Kfull = np.zeros((3 * N, 3 * N)); Kfull[np.ix_(idx, idx)] = Kn
OUT["T4_checks"]["kernel_proj_commute_with_Pi_max"] = float(max(np.abs(Kfull @ Pi1[k] - Pi1[k] @ Kfull).max() for k in ks))


def netvec(x):
    v = np.zeros(3 * N); v[idx] = center_net(x); return v


# rigid-octahedron rotation span, as closed_loop.py: for every B and axis, rotate its six O about B (single octahedron), network-centred; overlap of the bond-preserving part with this span
rotv = []
for j in B_idx:
    for a in range(3):
        e = np.eye(3)[a]; v = np.zeros(3 * N)
        for (b, x, vv, l) in bonds:
            if b == j: v[3 * x:3 * x + 3] += np.cross(e, vv)
        rotv.append(netvec(v))
rotv = np.array(rotv).T
Ur, sr, _ = np.linalg.svd(rotv, full_matrices=False); Trot = Ur[:, sr > 1e-6 * sr.max()]; Prot = Trot @ Trot.T
OUT["T4_rot_span_dim"] = int(Trot.shape[1]); OUT["T4_rot_random_bond_preserving_baseline"] = float(np.trace(Prot @ Kfull) / np.trace(Kfull))


def sector_table(x, label):
    v = netvec(x); tot = v @ v; rows = []
    for k in ks:
        pv = Pi1[k] @ v; w = pv @ pv / tot; pb = Kfull @ pv; c = pb @ pb / tot
        rot = float((Prot @ pb) @ (Prot @ pb) / (pb @ pb)) if pb @ pb > 1e-30 else None
        rows.append({"q": names[k], "pi": float(w), "s": float(c / w) if w > 1e-14 else None, "bp_weight": float(c), "bp_in_rigid_rotation_span": rot})
    irr = []
    for e, M in Pir1.items():
        pv = M @ v; w = pv @ pv / tot; pb = Kfull @ pv; c = pb @ pb / tot
        irr.append({"irrep_eps_xyz": list(e), "pi": float(w), "bp_weight": float(c), "s": float(c / w) if w > 1e-14 else None})
    return {"sum_pi": float(sum(r["pi"] for r in rows)), "sum_bp": float(sum(r["bp_weight"] for r in rows)), "s_total": s_net(x), "q_sectors": rows, "Gamma20_irreps": irr}


T4 = {"rigid_rotation_basis_dim": int(Trot.shape[1]), "f": sector_table(f, "f"), "u_pred": sector_table(upred, "u_pred"), "u_actual": sector_table(uact.ravel(), "u_actual")}
rows = T4["u_pred"]["q_sectors"]; dom = max(rows, key=lambda r: r["bp_weight"])
T4["dominant_q_for_bp_weight_u_pred"] = dom["q"]; T4["dominant_share_of_total_bp"] = dom["bp_weight"] / T4["u_pred"]["s_total"]
T4["dominant_rot_span_fraction"] = dom["bp_in_rigid_rotation_span"]
domA = max(T4["u_actual"]["q_sectors"], key=lambda r: r["bp_weight"]); T4["dominant_q_for_bp_weight_u_actual"] = domA["q"]
T4["verdict_rule_applied"] = ("PASS iff the dominant q sector of the bond-preserving weight of u_pred is not Gamma20 and >50% of its bond-preserving part lies in the "
                              "rigid-octahedral-rotation span (closed_loop.py definition: single-octahedron rotations about B, overlap of the bond-preserving part); FAIL otherwise. The 50% and the span definition are my operationalisation, not registered.")
T4["verdict"] = "FAIL" if dom["q"] == "Gamma20" else ("PASS" if (dom["bp_in_rigid_rotation_span"] or 0) > 0.5 else "FAIL")
OUT["T4"] = T4
(HERE / "dispersion_result.json").write_text(json.dumps(OUT, indent=1, default=float))
np.savez_compressed(HERE / "arrays.npz", H_s0=Hs, upred_s1=upred, uact_s1=uact_c, f_s1=f)
keys = ("n_ops", "displacement_amplitude_A_from_POSCAR", "rep_fill_spread_rel", "asymmetry_rel_Fro_before_symmetrising", "ASR_row_sum_rel_to_Hmax", "ASR_row_sum_Fro_over_H_Fro",
        "HQ_lowest_nontrivial_eig_eVA2", "HQ_n_negative_nontrivial", "s_f_check_ref_0.089pct", "s_u_act_check_ref_24.8pct", "s0_to_s1_max_pos_mismatch_A", "T1", "T2")
print(json.dumps({k: OUT[k] for k in keys}, indent=1, default=float))
print("T3", T3["verdict"], T3["rule_actual"], T3["rule_kernel"], T3["actual"]["exp"], T3["actual"]["power"], T3["kernel_same_range"]["exp"], T3["kernel_to_20A"]["exp"], "minHq", wmin, "herm", herm)
print(OUT["sens_negative_modes"], OUT["kernel_n_q_with_negative_eig"], OUT["kernel_n_q"], T3["kernel_absinverse_same_range"]["exp"], T3["kernel_absinverse_to_20A"]["exp"])
print(OUT["T4_rot_span_dim"], OUT["T4_rot_random_bond_preserving_baseline"])
print("T4", T4["verdict"], T4["dominant_q_for_bp_weight_u_pred"], T4["dominant_share_of_total_bp"], T4["dominant_rot_span_fraction"], T4["rigid_rotation_basis_dim"], OUT["T4_checks"])
