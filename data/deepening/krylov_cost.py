"""Noisy-Krylov cost test: conjugate gradient on H u = f (W coordinates of three_term_split.py) with a finite-difference matvec
model  matvec(v) = H v + (sigma_F/delta) * |v| * xi,  xi ~ N(0,1) per component, delta = 0.01 A (spec's concrete form; the
physical central-difference noise would be sigma_F*|v|/(sqrt2*delta), i.e. 1/sqrt2 smaller). sigma_F in {0,1e-4,1e-3} eV/A,
20 seeds each, k = 1..20 CG iterations (x0 = 0). CG is used since H is SPD in W (checked; else MINRES would be needed).
Per iteration: s_u(k) = old share (enrichment_all.py share function: network atoms, centred, kernel of B-X Jacobian) of the
embedded u_k, eps(k) = s_u(k)/s_f, cos(u_k, H^-1 f).  k_conv = smallest k such that |s_u(k')-s_u_exact| <= 0.1 s_u_exact
for all k' in [k,20] (None = not reached by k=20).  Cost: 2 force calls per matvec (central difference), so 2k calls.
Boundaries: sigma_F=0 is deterministic (seeds identical); CG breaks down if p^T H p <= 0 under noise (iterate frozen, flagged).
usage: python krylov_cost.py"""
import json
from pathlib import Path
import numpy as np
from three_term_split import load_systems, old_share_check, share, bond_basis, HERE

DELTA = 0.01
SIGMAS = (0.0, 1e-4, 1e-3)
KMAX = 20
NSEED = 20


def noisy_cg(Hs, f, sigma, rng):
    n = len(f)
    x = np.zeros(n); r = f.copy(); p = r.copy(); rs = r @ r
    xs, broke = [], False
    for _ in range(KMAX):
        if not broke:
            Ap = Hs @ p + (sigma / DELTA) * np.linalg.norm(p) * rng.standard_normal(n)
            d = p @ Ap
            if d <= 0 or rs == 0:
                broke = True
            else:
                al = rs / d
                x = x + al * p
                r = r - al * Ap
                rn = r @ r
                p = r + (rn / rs) * p
                rs = rn
        xs.append(x.copy())
    return xs, broke


def first_stable(ok):
    """smallest k (1-based) with ok[k-1:] all True, else None"""
    k = None
    for i in range(len(ok) - 1, -1, -1):
        if ok[i]:
            k = i + 1
        else:
            break
    return k


if __name__ == "__main__":
    out = []
    for s in load_systems():
        S, Hs = s["S"], s["Hs"]
        fs = S.T @ s["f"]
        sf, s_exact, Bo = old_share_check(s)
        a_ex = np.linalg.solve(Hs, fs)
        w = np.linalg.eigvalsh(Hs)
        B, R = bond_basis(s)
        Hbb, Hbr, Hrr = B.T @ Hs @ B, B.T @ Hs @ R, R.T @ Hs @ R
        wS = np.linalg.eigvalsh(Hbb - Hbr @ np.linalg.solve(Hrr, Hbr.T))
        rec = {"system": s["name"], "H_cond": float(w[-1] / w[0]), "H_min_eig": float(w[0]), "Sb_cond": float(wS[-1] / wS[0]),
               "s_f": sf, "s_u_exact": s_exact, "eps_exact": s_exact / sf, "dim_W": len(fs),
               "force_calls_full_H_recorded": s["n_jobs"], "ionic_steps_of_relaxation": "not recorded in meta.json/result.json",
               "sigma": {}}
        for sg in SIGMAS:
            ks, fin_su, fin_cos, brk = [], [], [], 0
            curve = np.zeros(KMAX)
            for seed in range(NSEED):
                xs, broke = noisy_cg(Hs, fs, sg, np.random.default_rng(1000 + seed))
                brk += broke
                su = np.array([share(S @ x, Bo, s["net"]) for x in xs])
                ok = np.abs(su - s_exact) <= 0.1 * s_exact
                ks.append(first_stable(ok))
                curve += su / NSEED
                x = xs[-1]
                fin_su.append(su[-1]); fin_cos.append(float(x @ a_ex / np.linalg.norm(x) / np.linalg.norm(a_ex)))
            valid = [k for k in ks if k is not None]
            rec["sigma"][str(sg)] = {
                "k_conv_median": float(np.median([k if k is not None else np.inf for k in ks])) if ks else None,
                "k_conv_worst": None if len(valid) < len(ks) else int(max(valid)),
                "n_seeds_never_converged": len(ks) - len(valid), "cg_breakdowns": int(brk),
                "force_calls_at_median_k": None if not valid or np.median([k if k is not None else np.inf for k in ks]) == np.inf
                else int(2 * np.median([k if k is not None else np.inf for k in ks])),
                "mean_s_u_k": [float(x) for x in curve], "eps_mean_k20": float(curve[-1] / sf),
                "cos_k20_median": float(np.median(fin_cos)), "cos_k20_min": float(np.min(fin_cos))}
        out.append(rec)
        print(f"{s['name'][:28]:28s} condH {rec['H_cond']:8.1f} condSb {rec['Sb_cond']:8.1f} dimW {len(fs):3d} Hcalls {s['n_jobs']:4d} "
              + " | ".join(f"s={k}: med {v['k_conv_median']} worst {v['k_conv_worst']} cos20 {v['cos_k20_median']:.3f}"
                           for k, v in rec["sigma"].items()))
    (HERE / "krylov_cost.json").write_text(json.dumps(out, indent=1))
