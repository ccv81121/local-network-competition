"""Agreement flags C1-C4 of predicted and four-state phi -> evaluation.json.
C1: phi_pred and phi on the same side of 0.5; C2: |phi_pred - phi| <= 0.15 where |dLambda_total| >= 3 meV;
C3: harmonic and four-state DFT energy shares within 0.10 (group A); C4: group B satisfies C1 and, where it applies, C2.
phi_actual: four-state phi in the convention of the published read-outs (spinel/layered: phi_readout.py
|d10|/(|d10|+|d01|); corner: signed d10/(d10+d01) with the published fixed valleys). Group B: phi_readout.py on the
re-built four states.
phi_pred: signed g.P H^-1 f / g.H^-1 f; the |.| form is reported alongside.
DFT four-state energy share (C3) = E10 / (E10 + E01) with E relative to state00.
Diagnostic: Lambda(state00) - Lambda(r_aff) = margin change carried by the A-cation motion
(+ network centroid shift) that the four-state construction puts into the baseline but u_c contains.
Reads LEVERAGE_DIR on the cluster (not deposited; ../evaluation.json is the result)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lev_common import CORNER_MARGIN, LEVERAGE_DIR, SYSTEMS, Lambda

SYSTEM_ORDER = ["k12_t45", "k12_05", "k08", "Layered_NaScInO2", "CaTiO3", "CaSnO3", "Ca2SnTiO6_rs", "Ca8Sn6Ti2O24"]
prepare_report = json.loads((LEVERAGE_DIR / "prepare_report.json").read_text())
rows = {}
for name in SYSTEM_ORDER:
    leverage = json.loads((LEVERAGE_DIR / "runs" / name / "leverage.json").read_text())
    margin = leverage["margin"]
    if name in CORNER_MARGIN:
        fourstate_dir = LEVERAGE_DIR / "A_fourstate_copy" / name
        lam = {state: Lambda(fourstate_dir / state, margin) for state in ("state00", "state10", "state01", "state11")}
        d10 = lam["state10"][0] - lam["state00"][0]
        d01 = lam["state01"][0] - lam["state00"][0]
        d11 = lam["state11"][0] - lam["state00"][0]
        E = {state: 1000 * (lam[state][1] - lam["state00"][1]) for state in lam}
        L00 = lam["state00"][0]
        phi = d10 / (d10 + d01)
        phi_abs = abs(d10) / (abs(d10) + abs(d01))
        src = "A_fourstate_copy (readout_fs margin, signed)"
    else:
        phi_data = json.loads(
            ((LEVERAGE_DIR / "runs" / name / "fourstate") if name.startswith("Ca2") or name.startswith("Ca8")
             else (LEVERAGE_DIR / "A_fourstate_copy" / name)).joinpath("phi.json").read_text())
        assert (phi_data["edge"] == margin["edge"] and np.allclose(phi_data["k_edge"], margin["k_edge"])
                and np.allclose(phi_data["k_rival"], margin["k_rival"]))
        d10, d01, d11 = phi_data["dLambda_perp_meV"], phi_data["dLambda_par_meV"], phi_data["dLambda_total_meV"]
        E = phi_data["E_rel_meV"]
        L00 = phi_data["Lambda_meV"]["state00"]
        phi_abs = phi_data["phi"]
        phi = d10 / (d10 + d01)
        src = "phi_readout.py"
    phi_ref = phi_abs if name not in CORNER_MARGIN else phi  # convention of the published read-outs
    eshare = E["state10"] / (E["state10"] + E["state01"])
    row = {"group": SYSTEMS[name][3], "margin": margin.get("label", margin["edge"]), "phi_actual": phi_ref,
           "phi_actual_signed": phi, "phi_actual_abs": phi_abs,
           "dLambda_b_actual_meV": d10, "dLambda_c_actual_meV": d01, "dLambda_total_actual_meV": d11,
           "dLambda_b_pred_meV": leverage["dP_b_pred_meV"], "dLambda_c_pred_meV": leverage["dP_c_pred_meV"],
           "dLambda_total_pred_meV": leverage["dP_total_pred_meV"],
           "phi_pred": leverage["phi_pred"], "phi_pred_abs": leverage["phi_pred_abs"],
           "E_share_harmonic": leverage["harmonic_energy_share_b"], "E_share_DFT_fourstate": eshare,
           "E_fourstate_meV": E, "source": src}
    row["C1"] = (row["phi_pred"] - 0.5) * (row["phi_actual"] - 0.5) > 0
    row["C2_applies"] = abs(d11) >= 3.0
    row["C2"] = abs(row["phi_pred"] - row["phi_actual"]) <= 0.15 if row["C2_applies"] else None
    row["C3"] = abs(row["E_share_harmonic"] - eshare) <= 0.10 if row["group"] == "A" else None
    row["abs_phi_pred_minus_phi"] = abs(row["phi_pred"] - row["phi_actual"])
    if "raff" in leverage["Lambda_meV"]:
        row["Lambda_raff_meV"] = leverage["Lambda_meV"]["raff"]
        row["diag_Lambda_state00_minus_raff_meV"] = L00 - leverage["Lambda_meV"]["raff"]
        row["diag_Lambda_curvature_meV"] = leverage.get("Lambda_curvature_check_meV")
    rows[name] = row
C1 = all(row["C1"] for row in rows.values())
C2 = all(row["C2"] for row in rows.values() if row["C2_applies"])
C3 = all(row["C3"] for row in rows.values() if row["C3"] is not None)
group_b_rows = [rows[name] for name in ("Ca2SnTiO6_rs", "Ca8Sn6Ti2O24")]
C4 = all(row["C1"] and (row["C2"] is None or row["C2"]) for row in group_b_rows)
C4_strict = all(row["C1"] and row["C2"] is True for row in group_b_rows)
if not C1:
    verdict = "C1 不通过：不进正文，现有框架不变"
elif C1 and C2 and C3 and C4:
    verdict = "C1-C4 全部通过：进正文"
elif C1 and C3:
    verdict = "C1、C3 通过，C2 或 C4 不通过：正文只写定性规律，定量对比放 SM"
else:
    verdict = "C1 通过但 C3 不通过：预注册结果表未列此情形"
out = {"rows": rows, "C1": C1, "C2": C2, "C3": C3, "C4": C4, "C4_if_C2_required_strictly": C4_strict,
       "verdict": verdict}
(LEVERAGE_DIR / "evaluation.json").write_text(
    json.dumps(out, indent=1, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x)))
for name, row in rows.items():
    print(f"{name:18s} phi={row['phi_actual']:+.3f} pred={row['phi_pred']:+.3f} "
          f"|dP|={abs(row['dLambda_total_actual_meV']):6.2f} "
          f"Eh={row['E_share_harmonic']:.3f} Ed={row['E_share_DFT_fourstate']:+.3f} "
          f"C1={row['C1']} C2={row['C2']} C3={row['C3']} "
          f"diag00-raff={row.get('diag_Lambda_state00_minus_raff_meV')}")
print("C1", C1, "C2", C2, "C3", C3, "C4", C4, "C4strict", C4_strict, "->", verdict)
