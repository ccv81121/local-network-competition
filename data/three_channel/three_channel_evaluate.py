"""Three-channel closure (bending b, network complement n, A site).
Predicted: dL_b = g_b|u_b| and dL_c = g_c|u_c| from the leverage test; dL_A = g_A|u_A| (new statics);
dL_n = dL_c - dL_A.
Measured: b and n from the four states, A = Lambda(state00) - Lambda(r_aff)
(four_state_leverage/evaluation.json).
usage: python three_channel_evaluate.py   (needs tc_readout.json and a_channel_steps.json beside this file)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALUATION_ROWS = json.loads((HERE.parent / "four_state_leverage" / "evaluation.json").read_text())["rows"]
READOUTS = json.loads((HERE / "tc_readout.json").read_text())
A_CHANNEL_STEPS = json.loads((HERE / "a_channel_steps.json").read_text())
results = {}
for name in ("k12_t45", "k12_05"):
    evaluation, readout, steps = EVALUATION_ROWS[name], READOUTS[name], A_CHANNEL_STEPS[name]
    gA = (readout["ap"]["Lambda_meV"] - readout["am"]["Lambda_meV"]) / (2 * steps["delta_A_A"])
    pA = gA * steps["uA_norm_A"]
    pb = evaluation["dLambda_b_pred_meV"]
    pn = evaluation["dLambda_c_pred_meV"] - pA
    mb, mn, mA = (evaluation["dLambda_b_actual_meV"], evaluation["dLambda_c_actual_meV"],
                  evaluation["diag_Lambda_state00_minus_raff_meV"])
    pred, meas = {"b": pb, "n": pn, "A": pA}, {"b": mb, "n": mn, "A": mA}
    pred_share = {k: v / sum(pred.values()) for k, v in pred.items()}
    meas_share = {k: v / sum(meas.values()) for k, v in meas.items()}
    T1 = all(abs(pred_share[k] - meas_share[k]) <= 0.10 for k in pred)
    T2 = abs(pred_share["A"] - meas_share["A"]) <= 0.10 and pred_share["A"] * meas_share["A"] > 0
    T3 = abs(sum(pred.values()) - sum(meas.values())) <= 0.10 * abs(sum(meas.values()))
    results[name] = {"g_A_meV_per_A": gA, "pred_meV": pred, "meas_meV": meas, "pred_share": pred_share,
                     "meas_share": meas_share, "total_pred_meV": sum(pred.values()),
                     "total_meas_meV": sum(meas.values()), "T1": T1, "T2": bool(T2), "T3": T3}
(HERE / "tc_result.json").write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1))
