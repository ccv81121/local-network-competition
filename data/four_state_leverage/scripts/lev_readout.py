"""Margins and energies of the +-0.25|u_i| d_i statics; g_b, g_c, phi_pred (signed).
phi_pred = g_b|u_b| / (g_b|u_b| + g_c|u_c|),  g_i = [Lambda(+) - Lambda(-)] / (2 * 0.25 |u_i|)  (meV/A).
Reads the VASP outputs under LEVERAGE_DIR on the cluster (not deposited; calculations/<system>/leverage.json is the result).
usage: python lev_readout.py SYSTEM [SYSTEM ...]   -> runs/<SYSTEM>/leverage.json"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lev_common import LEVERAGE_DIR, Lambda

prepare_report = json.loads((LEVERAGE_DIR / "prepare_report.json").read_text())
for name in sys.argv[1:]:
    run_dir = LEVERAGE_DIR / "runs" / name
    entry = prepare_report[name]
    margin = entry["margin"]
    out = {"system": name, "margin": margin}
    lam, toten = {}, {}
    for tag in ("bp", "bm", "cp", "cm", "raff"):
        if (run_dir / tag / "DONE").exists():
            lam[tag], toten[tag] = Lambda(run_dir / tag, margin)
    out["Lambda_meV"] = lam
    out["TOTEN_eV"] = toten
    nb, nc = entry["ub_norm_A"], entry["uc_norm_A"]
    delta_b, delta_c = 0.25 * nb, 0.25 * nc
    if all(tag in lam for tag in ("bp", "bm", "cp", "cm")):
        gb = (lam["bp"] - lam["bm"]) / (2 * delta_b)
        gc = (lam["cp"] - lam["cm"]) / (2 * delta_c)
        Pb, Pc = gb * nb, gc * nc
        F = np.load(run_dir / "F_raff.npy")
        ub = np.load(run_dir / "u_b.npy")
        uc = np.load(run_dir / "u_c.npy")
        out.update({
            "g_b_meV_per_A": gb, "g_c_meV_per_A": gc, "dP_b_pred_meV": Pb, "dP_c_pred_meV": Pc,
            "dP_total_pred_meV": Pb + Pc, "phi_pred": Pb / (Pb + Pc), "phi_pred_abs": abs(Pb) / (abs(Pb) + abs(Pc)),
            "E_b_harm_meV": entry["E_b_meV"], "E_c_harm_meV": entry["E_c_meV"],
            "harmonic_energy_share_b": entry["harmonic_energy_share_b"],
            # finite-difference DFT energies along d_i: dE/ds = [E(+)-E(-)]/(2 delta), compared with -f.d_i
            "dft_dEds_b_meV_per_A": 1000 * (toten["bp"] - toten["bm"]) / (2 * delta_b),
            "harm_minus_f_dot_db_meV_per_A": -1000 * float(F @ ub) / nb,
            "dft_dEds_c_meV_per_A": 1000 * (toten["cp"] - toten["cm"]) / (2 * delta_c),
            "harm_minus_f_dot_dc_meV_per_A": -1000 * float(F @ uc) / nc,
            "dft_E_b_linear_meV": -0.5 * 1000 * (toten["bp"] - toten["bm"]) / (2 * delta_b) * nb,
            "dft_E_c_linear_meV": -0.5 * 1000 * (toten["cp"] - toten["cm"]) / (2 * delta_c) * nc})
        if "raff" in lam:
            out["Lambda_curvature_check_meV"] = {"b": (lam["bp"] + lam["bm"]) / 2 - lam["raff"],
                                                 "c": (lam["cp"] + lam["cm"]) / 2 - lam["raff"]}
            out["dft_E_second_diff_meV"] = {"b": 1000 * (toten["bp"] + toten["bm"] - 2 * toten["raff"]),
                                            "c": 1000 * (toten["cp"] + toten["cm"] - 2 * toten["raff"])}
    (run_dir / "leverage.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
