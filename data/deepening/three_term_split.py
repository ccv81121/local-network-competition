"""Three-term split of the bond-preserving component of H^-1 f from the start structure only (no DFT).
Space W: the space the existing code uses (enrichment_all.py): Cartesian 3N with translations removed (full H systems; basis
= eigenvectors of Q H Q with the 3 smallest-|eig| removed, H = diag(w) there) or the symmetric-sector basis S (Hs = -S^T dF/dS).
In W coordinates (a = S^T x):  B = null{[J; x_A=0] S} (bond-preserving network motions, A-cations fixed at 0; same definition as
bending_amplitude/symmetric_sector_blocks.py p_s), R = orthogonal complement of B in W.
  b = S_b^-1 (f_b - H_br H_rr^-1 f_r),  S_b = H_bb - H_br H_rr^-1 H_rb
  t1 = H_bb^-1 f_b (clamped), t2 = (S_b^-1 - H_bb^-1) f_b (softening), t3 = -S_b^-1 H_br H_rr^-1 f_r (force transmission)
Metric: Euclidean norms in W coordinates (orthonormal basis, so equal to Cartesian norms of the embedded vectors).
  sq fraction  |t_i|^2 / sum_j |t_j|^2  (sums to 1, ignores cross terms);  signed fraction t_i.b / b.b  (sums to 1).
Old posterior split: direct = H_bb^-1 f_b, coupled = -H_bb^-1 H_br u_r with u_r the R-part of the ACTUAL relaxation u.
s_f = old share of the drive (network atoms, centred, kernel of B-X Jacobian; = drive_share of enrichment_all_systems.csv);
sB_f = |f_b|^2/|f|^2 in W (A-cations counted in R, so differs from s_f).
Boundaries: exact when H_rr and H_bb are SPD (checked); dim B=0 -> system skipped; actual u only approximately in W (leak reported).
usage: python three_term_split.py"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
FB = DATA / "force_balance_reference"
TFB = DATA / "spinel_pressure_path" / "force_balance"
FULL = DATA / "bending_enrichment" / "full_force_balance"
SEC = DATA / "bending_enrichment" / "sector_force_balance"
AMP = DATA / "bending_amplitude" / "sector_force_balance"


def kernel(f_start, L, bonds, net):
    col = {int(a): c for c, a in enumerate(net)}
    vec = np.array([(f_start[j] - f_start[i] + np.array(img)) @ L for i, j, img in bonds])
    J = np.zeros((len(bonds), 3 * len(net)))
    for r, (i, j, _) in enumerate(bonds):
        e = vec[r] / np.linalg.norm(vec[r])
        J[r, 3 * col[i]:3 * col[i] + 3] = -e
        J[r, 3 * col[j]:3 * col[j] + 3] = e
    _, s, Vt = np.linalg.svd(J, full_matrices=False)
    return Vt[: int(np.sum(s > s[0] * 1e-10))]


def share(u, B, net):
    v = u.reshape(-1, 3)[net]
    v = v - v.mean(axis=0)
    v = v.ravel()
    p = v - B.T @ (B @ v)
    return float(p @ p / (v @ v))


def full_H(read_forces, N, delta):
    H = np.zeros((3 * N, 3 * N))
    for i in range(N):
        for a, ax in enumerate("xyz"):
            H[3 * i + a] = -(read_forces(f"d{i:02d}{ax}p") - read_forces(f"d{i:02d}{ax}m")).ravel() / (2 * delta)
    H = 0.5 * (H + H.T)
    T = np.zeros((3 * N, 3))
    for a in range(3):
        T[a::3, a] = 1 / np.sqrt(N)
    Q = np.eye(3 * N) - T @ T.T
    w, V = np.linalg.eigh(Q @ H @ Q)
    keep = np.argsort(np.abs(w))[3:]
    return Q, V[:, keep], w[keep]


def sector_H(case_dir, meta):
    S = np.load(case_dir / "S.npy")
    delta = meta["delta_A"]
    rf = lambda c: np.load(case_dir / "hessian" / c / "forces.npy").ravel()
    ns = S.shape[1]
    Hs = np.zeros((ns, ns))
    for k in range(ns):
        Hs[k] = -(S.T @ (rf(f"s{k:03d}p") - rf(f"s{k:03d}m"))) / (2 * delta)
    return S, 0.5 * (Hs + Hs.T), rf("ref")


def load_systems():
    """list of dicts: name, kind, S (3N x m orthonormal), Hs (m x m), f (3N drive), u_act (3N actual), L, f1, bonds, net, N,
    n_jobs (force calls recorded for H), ref (csv row or None)."""
    ref = {r["system"]: r for r in json.loads((DATA / "bending_enrichment" / "enrichment_all_systems.json").read_text())}
    out = []

    def act(f1, f2, L, Q):
        df = f2 - f1
        df -= np.round(df)
        return Q @ (df @ L).ravel()

    meta = json.loads((TFB / "meta.json").read_text())
    species = np.repeat(meta["species"], meta["counts"]); N = len(species)
    Q, V, w = full_H(lambda c: np.load(FB / "forces" / f"{c}.npy"), N, meta["delta_A"])
    f0 = np.load(TFB / "forces" / "ref.npy").ravel()
    bonds = json.loads((FB / "inputs" / "pressure_bond_accommodation.json").read_text())["bonds"]
    net = np.flatnonzero(np.isin(species, ["Sc", "Sb", "Se"]))
    L = np.array(meta["lattice_5GPa"]); f1 = np.array(meta["frac_start"])
    out.append(dict(name="MgSc0.5Sb1.5Se4 (k12, tight)", kind="full", S=V, Hs=np.diag(w), f=f0,
                    u_act=act(f1, np.array(meta["frac_5GPa"]), L, Q), L=L, f1=f1, bonds=bonds, net=net, N=N,
                    n_jobs=len(meta["jobs"]), ref=ref["MgSc0.5Sb1.5Se4 (k12, tight)"]))
    for system, case in (("CaTiO3", "ca_tio3"), ("CaSnO3", "ca_sno3")):
        d = FULL / case / "p0_p5"
        meta = json.loads((d / "meta.json").read_text()); N = sum(meta["counts"])
        rf = lambda c, d=d: np.load(d / "hessian" / c / "forces.npy")
        Q, V, w = full_H(rf, N, meta["delta_A"])
        f0 = Q @ rf("ref").ravel()
        bonds = meta["bonds"]; net = np.array(sorted({b[0] for b in bonds} | {b[1] for b in bonds}))
        L = np.array(meta["lattice_P2"]); f1 = np.array(meta["frac_start"])
        out.append(dict(name=system, kind="full", S=V, Hs=np.diag(w), f=f0,
                        u_act=act(f1, np.array(meta["frac_P2"]), L, Q), L=L, f1=f1, bonds=bonds, net=net, N=N,
                        n_jobs=len(meta["jobs"]), ref=ref[system]))
    for root, system, label in ((SEC, "spinel_k08", "MgScSbSe4 (k08 ordering)"), (SEC, "ca2_sn_tio6_rs", "Ca2SnTiO6 (rock-salt)"),
                                (SEC, "ca8_sn6_ti2o24", "Ca8Sn6Ti2O24"), (AMP, "spinel_cd_ga_ino4", "Cd(Ga,In)2O4"),
                                (AMP, "layered_na_sc_ino2", "Na(Sc,In)O2")):
        d = root / system / "p0_p5"
        if not (d / "S.npy").exists() or not (d / "hessian").exists():
            print("missing inputs, skipped:", label); continue
        meta = json.loads((d / "meta.json").read_text()); N = sum(meta["counts"])
        S, Hs, f0 = sector_H(d, meta)
        T = np.zeros((3 * N, 3))
        for a in range(3):
            T[a::3, a] = 1 / np.sqrt(N)
        Q = np.eye(3 * N) - T @ T.T
        L = np.array(meta["lattice_P2"]); f1 = np.array(meta["frac_start"])
        out.append(dict(name=label, kind="sector", S=S, Hs=Hs, f=f0,
                        u_act=act(f1, np.array(meta["frac_P2"]), L, Q), L=L, f1=f1, bonds=meta["bonds"],
                        net=np.array(meta["network"]), N=N, n_jobs=len(meta["jobs"]), ref=ref.get(label)))
    return out


def null_space(M, tol=1e-9):
    _, s, Vt = np.linalg.svd(M)
    r = int(np.sum(s > tol * s[0]))
    return Vt[r:].T


def bond_basis(sysd):
    """B (m x nb) and R (m x nr) orthonormal bases in W coordinates."""
    N, S = sysd["N"], sysd["S"]
    J = np.zeros((len(sysd["bonds"]), 3 * N))
    for r, (i, j, img) in enumerate(sysd["bonds"]):
        v = (sysd["f1"][j] - sysd["f1"][i] + np.array(img)) @ sysd["L"]
        e = v / np.linalg.norm(v)
        J[r, 3 * i:3 * i + 3] = -e
        J[r, 3 * j:3 * j + 3] = e
    netset = set(int(x) for x in sysd["net"])
    aidx = [3 * a + k for a in range(N) if a not in netset for k in range(3)]
    C = np.vstack([J, np.eye(3 * N)[aidx]]) @ S
    B = null_space(C)
    R = null_space(B.T) if B.shape[1] else np.eye(S.shape[1])
    return B, R


def old_share_check(sysd):
    Bo = kernel(sysd["f1"], sysd["L"], sysd["bonds"], sysd["net"])
    S, Hs = sysd["S"], sysd["Hs"]
    up = S @ np.linalg.solve(Hs, S.T @ sysd["f"])
    return share(sysd["f"], Bo, sysd["net"]), share(up, Bo, sysd["net"]), Bo


def analyse(sysd):
    S, Hs = sysd["S"], sysd["Hs"]
    B, R = bond_basis(sysd)
    if B.shape[1] == 0:
        return {"system": sysd["name"], "note": "dim B = 0"}
    fs = S.T @ sysd["f"]
    Hbb, Hbr, Hrr = B.T @ Hs @ B, B.T @ Hs @ R, R.T @ Hs @ R
    fb, fr = B.T @ fs, R.T @ fs
    X = np.linalg.solve(Hrr, np.column_stack([Hbr.T, fr]))
    Sb = Hbb - Hbr @ X[:, :-1]
    t1 = np.linalg.solve(Hbb, fb)
    t2 = np.linalg.solve(Sb, fb) - t1
    t3 = -np.linalg.solve(Sb, Hbr @ X[:, -1])
    b = t1 + t2 + t3
    a = np.linalg.solve(Hs, fs)
    err = float(np.linalg.norm(b - B.T @ a))
    n = np.array([t @ t for t in (t1, t2, t3)])
    ua = S.T @ sysd["u_act"]
    ub, ur = B.T @ ua, R.T @ ua
    c_old = -np.linalg.solve(Hbb, Hbr @ ur)
    old = {"direct_signed": float(t1 @ ub / (ub @ ub)), "coupled_signed": float(c_old @ ub / (ub @ ub)),
           "residual_signed": float((ub - t1 - c_old) @ ub / (ub @ ub))}
    ex = {"direct_signed": float(t1 @ b / (b @ b)), "coupled_signed": float((t2 + t3) @ b / (b @ b))}
    wH, wS = np.linalg.eigvalsh(Hs), np.linalg.eigvalsh(Sb)
    names = ("t1_clamped", "t2_softening", "t3_transmission")
    return {"system": sysd["name"], "kind": sysd["kind"], "dim_W": int(S.shape[1]), "dim_B": int(B.shape[1]),
            "H_min_eig": float(wH[0]), "H_max_eig": float(wH[-1]), "H_cond": float(wH[-1] / wH[0]) if wH[0] > 0 else None,
            "Sb_min_eig": float(wS[0]), "Sb_cond": float(wS[-1] / wS[0]) if wS[0] > 0 else None,
            "recon_err_b": err, "recon_rel": err / float(np.linalg.norm(b)),
            "sq_frac": dict(zip(names, map(float, n / n.sum()))),
            "signed_frac": {k: float(t @ b / (b @ b)) for k, t in zip(names, (t1, t2, t3))},
            "norm_t": [float(np.sqrt(x)) for x in n],
            "s_f_old": sysd["_sf"], "sB_f_W": float(fb @ fb / (fs @ fs)), "b_share_of_Hinvf": float(b @ b / (a @ a)),
            "exact_direct_coupled_signed": ex, "old_posterior_actual_u": old,
            "cos_u_act_Hinvf": float(ua @ a / np.linalg.norm(ua) / np.linalg.norm(a)),
            "u_act_leak_outside_W": float(np.linalg.norm(sysd["u_act"] - S @ ua) / np.linalg.norm(sysd["u_act"]))}


if __name__ == "__main__":
    res, chk = [], []
    for s in load_systems():
        sf, sp, _ = old_share_check(s)
        s["_sf"] = sf
        if s["ref"] is not None:
            d1, d2 = abs(sf - s["ref"]["drive_share"]), abs(sp - s["ref"]["predicted_share"])
            chk.append({"system": s["name"], "drive_share": sf, "predicted_share": sp, "abs_diff_drive": d1,
                        "abs_diff_pred": d2, "match_1e-6": bool(max(d1, d2) < 1e-6)})
        else:
            chk.append({"system": s["name"], "drive_share": sf, "predicted_share": sp, "note": "no csv reference"})
        res.append(analyse(s))
    (HERE / "three_term_split.json").write_text(json.dumps({"sanity": chk, "split": res}, indent=1))
    for c in chk: print("CHECK", json.dumps(c))
    for r in res:
        if "note" in r: print(r); continue
        print(f"{r['system'][:28]:28s} dimB {r['dim_B']:3d} sf {r['s_f_old']:.4f} b/Hf {r['b_share_of_Hinvf']:.3f} "
              f"sq {[round(x,3) for x in r['sq_frac'].values()]} signed {[round(x,3) for x in r['signed_frac'].values()]} "
              f"recon {r['recon_rel']:.1e} oldpost {[round(x,3) for x in r['old_posterior_actual_u'].values()]} Hmin {r['H_min_eig']:.3g}")
