"""Three-channel closure for the six other leverage systems.
prepare: r_aff +- 0.25|u_A| d_A POSCARs (u_A = non-network part of u_c), INCAR/KPOINTS/POTCAR from the system's bp run.
eval   : g_A from the two statics (lev_common.Lambda, same margin rule), predicted shares b/n/A vs measured
         (b, n from the four states, A = Lambda(state00) - Lambda(r_aff), four_state_leverage/evaluation.json);
         agreement flags T0-T3.
Reads and writes the leverage folders on the cluster ($WORK/spinel): not deposited.
usage: python three_channel_all_network.py prepare|eval
"""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "four_state_leverage" / "scripts"))
from lev_common import read_poscar, write_poscar, Lambda

LEVERAGE_DIR = Path("$WORK/spinel/LEVERAGE_20261023")
OUTPUT_DIR = Path(__file__).resolve().parent
A_SITE_SPECIES = {"k08": "Mg", "CaTiO3": "Ca", "CaSnO3": "Ca", "Layered_NaScInO2": "Na", "Ca2SnTiO6_rs": "Ca",
                  "Ca8Sn6Ti2O24": "Ca"}
MAIN_SYSTEMS = ("k08", "CaTiO3", "CaSnO3")  # |dLambda_total| >= 3 meV: T1-T3; the others: T0 only
prepare_report = json.loads((LEVERAGE_DIR / "prepare_report.json").read_text())


def prepare():
    info = {}
    for name, a_species in A_SITE_SPECIES.items():
        run_dir = LEVERAGE_DIR / "runs" / name
        lat, sp, cnt, f1 = read_poscar(run_dir / "raff" / "POSCAR")
        specs = sum([[s] * c for s, c in zip(sp, cnt)], [])
        A = [i for i, s in enumerate(specs) if s == a_species]
        assert len(A) == prepare_report[name]["N"] - prepare_report[name]["n_network"], name
        uc = np.load(run_dir / "u_c.npy").reshape(-1, 3)
        uA = np.zeros_like(uc)
        uA[A] = uc[A]
        nA = float(np.linalg.norm(uA))
        assert abs(nA - prepare_report[name]["uc_A_cation_norm_A"]) < 1e-9, (name, nA)
        dA = uA / nA
        inv = np.linalg.inv(lat)
        info[name] = {"A_species": a_species, "uA_norm_A": nA, "delta_A_A": 0.25 * nA, "A_idx": A}
        for tag, sgn in (("ap", 1), ("am", -1)):
            d = OUTPUT_DIR / "runs" / name / tag
            d.mkdir(parents=True, exist_ok=True)
            write_poscar(d / "POSCAR", f"{name} threechannel-allnet {tag}", lat, sp, cnt,
                         f1 + (sgn * 0.25 * nA * dA) @ inv)
            for filename in ("INCAR", "KPOINTS", "POTCAR"):
                shutil.copy(run_dir / "bp" / filename, d / filename)
            info[name][tag + "_sha256"] = hashlib.sha256((d / "POSCAR").read_bytes()).hexdigest()
    (OUTPUT_DIR / "a_channel_steps.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def evaluate():
    evaluation_rows = json.loads((LEVERAGE_DIR / "evaluation.json").read_text())["rows"]
    a_channel_steps = json.loads((OUTPUT_DIR / "a_channel_steps.json").read_text())
    results = {}
    for name in A_SITE_SPECIES:
        evaluation, steps = evaluation_rows[name], a_channel_steps[name]
        margin = prepare_report[name]["margin"]
        lambda_ap, _ = Lambda(OUTPUT_DIR / "runs" / name / "ap", margin)
        lambda_am, _ = Lambda(OUTPUT_DIR / "runs" / name / "am", margin)
        gA = (lambda_ap - lambda_am) / (2 * steps["delta_A_A"])
        pA = gA * steps["uA_norm_A"]
        pred = {"b": evaluation["dLambda_b_pred_meV"], "n": evaluation["dLambda_c_pred_meV"] - pA, "A": pA}
        meas = {"b": evaluation["dLambda_b_actual_meV"], "n": evaluation["dLambda_c_actual_meV"],
                "A": evaluation["diag_Lambda_state00_minus_raff_meV"]}
        pred_share = {k: v / sum(pred.values()) for k, v in pred.items()}
        meas_share = {k: v / sum(meas.values()) for k, v in meas.items()}
        result = {"Lambda_ap_meV": lambda_ap, "Lambda_am_meV": lambda_am, "g_A_meV_per_A": gA, "pred_meV": pred,
                  "meas_meV": meas, "pred_share": pred_share, "meas_share": meas_share,
                  "total_pred_meV": sum(pred.values()), "total_meas_meV": sum(meas.values()),
                  "T0_signs": bool(all(np.sign(pred[k]) == np.sign(meas[k]) for k in pred))}
        if name in MAIN_SYSTEMS:
            result["T1"] = bool(all(abs(pred_share[k] - meas_share[k]) <= 0.10 for k in pred))
            result["T2"] = bool(abs(pred_share["A"] - meas_share["A"]) <= 0.10
                                and pred_share["A"] * meas_share["A"] > 0)
            result["T3"] = bool(abs(sum(pred.values()) - sum(meas.values()))
                                <= 0.10 * abs(sum(meas.values())))
        results[name] = result
    (OUTPUT_DIR / "tca_result.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    {"prepare": prepare, "eval": evaluate}[sys.argv[1]]()
