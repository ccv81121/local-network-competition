"""Reproduce the enrichment shares, split u_pred = H^-1 f into u_b = P u_pred and
u_c = u_pred - u_b, harmonic energies, write the +-0.25|u_i| d_i POSCARs (and r_aff) with the four-state VASP inputs.
Also verifies that the four-state construction of sector_force_balance_prepare.py rebuilt from META reproduces the
existing four-state POSCARs.
Reads the cluster work folders of lev_common.SYSTEMS (not deposited).
usage: python lev_prepare.py   (writes under $WORK/spinel/LEVERAGE_20261023 only)"""
import json, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lev_common import (LEVERAGE_DIR, SYSTEMS, TIGHT_SPINEL_DIR, kernel_rows, load_system, margin_def, read_poscar,
                        share, split_b, write_poscar)
from lev_four_state_build import build_fourstate

DELTA_FRAC = 0.25
# reference shares (drive, predicted, actual) from bending_enrichment / bending_amplitude / edge_extension
REFERENCE_SHARES = {
    "k12_t45": (0.00544, 0.3141, 0.3291), "CaTiO3": (0.1008, 0.1220, 0.1080), "CaSnO3": (0.1158, 0.2704, 0.2562)}


def incar_body(p):
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip() and not l.strip().startswith("SYSTEM")]


def fdist(a, b, L):
    d = a - b
    d -= np.round(d)
    return float(np.abs(d @ L).max())


report = {}
LEVERAGE_DIR.joinpath("runs").mkdir(parents=True, exist_ok=True)
for name in SYSTEMS:
    system = load_system(name)
    _, work_dir, fourstate_dir, group = SYSTEMS[name]
    N, net, L2, f1 = system["N"], system["net"], system["L2"], system["f1"]
    Rb = kernel_rows(f1, L2, system["bonds"], net)
    shares = (share(system["F_share"], Rb, net), share(system["u_pred"], Rb, net), share(system["u_act"], Rb, net))
    cos_pred_actual = float(system["u_pred"] @ system["u_act"] / np.linalg.norm(system["u_pred"])
                            / np.linalg.norm(system["u_act"]))
    result = {"group": group, "H_type": system["path"], "N": N, "n_network": len(net), "drive_share": shares[0],
              "pred_share": shares[1], "actual_share": shares[2], "cos_pred_actual": cos_pred_actual}
    if system["path"] == "symsec":
        force_balance = json.loads((work_dir / "force_balance_sym.json").read_text())
        result["ref_shares"] = [force_balance["bond_preserving_share"][k] for k in ("drive", "predicted", "actual")]
        result["ref_cos"] = force_balance["cosine_pred_vs_actual"]
        result["ref_harmonic_meV"] = force_balance["harmonic_energy_lowering_meV"]
        result["Hs_vs_saved_rel"] = float(np.linalg.norm(system["Hs"] - system["Hs_saved"])
                                          / np.linalg.norm(system["Hs_saved"]))
    else:
        result["ref_shares"] = list(REFERENCE_SHARES[name])
    result["shares_match_3sig"] = bool(all(abs(a - b) <= 0.0005 * abs(b) + 5e-6
                                           for a, b in zip(shares, result["ref_shares"])))
    # split of the predicted response
    u_pred = system["u_pred"]
    ub = split_b(u_pred, Rb, net, N)
    uc = u_pred - ub
    F = system["F"]
    nb, nc = float(np.linalg.norm(ub)), float(np.linalg.norm(uc))
    Eb, Ec, Et = 0.5 * float(F @ ub), 0.5 * float(F @ uc), 0.5 * float(F @ u_pred)
    A_idx = np.setdiff1d(np.arange(N), net)
    ucA = uc.reshape(-1, 3)[A_idx].ravel()
    ucN = uc.reshape(-1, 3)[net].ravel()
    result.update({"u_pred_norm_A": float(np.linalg.norm(u_pred)), "ub_norm_A": nb, "uc_norm_A": nc,
                   "uc_A_cation_norm_A": float(np.linalg.norm(ucA)), "uc_network_norm_A": float(np.linalg.norm(ucN)),
                   "ub_share_of_upred_sq": nb ** 2 / float(u_pred @ u_pred), "ub_dot_uc": float(ub @ uc),
                   "E_b_meV": 1000 * Eb, "E_c_meV": 1000 * Ec, "E_total_meV": 1000 * Et,
                   "E_b_plus_E_c_minus_E_total_meV": 1000 * (Eb + Ec - Et),
                   "harmonic_energy_share_b": Eb / (Eb + Ec)})
    if system["path"] == "symsec":
        S = system["S"]
        leak = float(np.linalg.norm(ub - S @ (S.T @ ub)) / nb)
        result["ub_sector_leakage"] = leak
        if leak < 1e-6:
            bs, cs = S.T @ ub, S.T @ uc
            Hs = system["Hs"]
            result["quadratic_split_meV"] = {
                "b": 1000 * float(-F @ ub + 0.5 * bs @ Hs @ bs), "c": 1000 * float(-F @ uc + 0.5 * cs @ Hs @ cs),
                "cross_ub_H_uc": 1000 * float(bs @ Hs @ cs), "total": 1000 * float(-0.5 * system["fs"] @ system["us"])}
    else:
        H = system["H"]
        result["quadratic_split_meV"] = {
            "b": 1000 * float(-F @ ub + 0.5 * ub @ H @ ub), "c": 1000 * float(-F @ uc + 0.5 * uc @ H @ uc),
            "cross_ub_H_uc": 1000 * float(ub @ H @ uc), "total": 1000 * float(-F @ u_pred + 0.5 * u_pred @ H @ u_pred)}
    # four-state construction check (geometry only) and margin definition
    if system["path"] == "full_tight":
        fourstate_meta = json.loads((work_dir.parent / "work/Spinel/P4_P5/meta.json").read_text())
    else:
        fourstate_meta = system["meta"] if "network" in system["meta"] else dict(system["meta"], network=net.tolist())
    built_lat, built = build_fourstate(fourstate_meta)
    result["fourstate_rebuild_maxdev_A"] = {k: fdist(read_poscar(fourstate_dir / k / "POSCAR")[3], v, built_lat)
                                            for k, v in built.items()}
    result["fourstate_meta_vs_lev_raff_maxdev_A"] = fdist(np.array(fourstate_meta["frac_start"]), f1, L2)
    result["fourstate_lattice_vs_lev_lattice"] = float(np.abs(np.array(fourstate_meta["lattice_P2"]) - L2).max())
    margin = margin_def(name)
    result["margin"] = margin
    # POSCARs of the finite-difference statics
    inv = np.linalg.inv(L2)
    src = fourstate_dir / "state00"
    lat, sp, cnt, _ = read_poscar(src / "POSCAR")
    assert sp == system["species"] and cnt == system["counts"] and np.allclose(lat, L2, atol=1e-8), (name, sp, cnt)
    db, dc = ub / nb, uc / nc
    pts = {"bp": +DELTA_FRAC * nb * db, "bm": -DELTA_FRAC * nb * db, "cp": +DELTA_FRAC * nc * dc,
           "cm": -DELTA_FRAC * nc * dc, "raff": 0 * db}
    result["delta_b_A"], result["delta_c_A"] = DELTA_FRAC * nb, DELTA_FRAC * nc
    for tag, du in pts.items():
        d = LEVERAGE_DIR / "runs" / name / tag
        d.mkdir(parents=True, exist_ok=True)
        fr = f1 + du.reshape(-1, 3) @ inv
        write_poscar(d / "POSCAR",
                     f"{name} leverage {tag} (r_aff {'+-'[tag.endswith('m')] if tag != 'raff' else ''}0.25|u| d)",
                     L2, sp, cnt, fr)
        shutil.copy(src / "INCAR", d / "INCAR")
        shutil.copy(src / "KPOINTS", d / "KPOINTS")
        shutil.copy(src / "POTCAR", d / "POTCAR")
    np.save(LEVERAGE_DIR / "runs" / name / "u_pred.npy", u_pred)
    np.save(LEVERAGE_DIR / "runs" / name / "u_b.npy", ub)
    np.save(LEVERAGE_DIR / "runs" / name / "u_c.npy", uc)
    np.save(LEVERAGE_DIR / "runs" / name / "F_raff.npy", F)
    # existing r_aff readout with identical settings?
    existing_dir = {"k12_t45": TIGHT_SPINEL_DIR / "static_k6_pred/raff"}.get(name)
    if existing_dir is not None and (existing_dir / "EIGENVAL").exists():
        same = (incar_body(existing_dir / "INCAR") == incar_body(src / "INCAR")
                and Path(existing_dir / "KPOINTS").read_text().split()[3:6]
                == Path(src / "KPOINTS").read_text().split()[3:6])
        dev = fdist(read_poscar(existing_dir / "POSCAR")[3], f1, L2)
        result["existing_raff"] = {"dir": str(existing_dir), "identical_incar_kpoints": bool(same), "pos_maxdev_A": dev}
    report[name] = result
    print(name, json.dumps({k: v for k, v in result.items() if k not in ("margin",)}, default=float)[:1500], "\n")
(LEVERAGE_DIR / "prepare_report.json").write_text(json.dumps(report, indent=1, default=float))
