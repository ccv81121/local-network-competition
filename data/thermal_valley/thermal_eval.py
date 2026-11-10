"""Thermal readout. Margins as in substitution_path_split/path_readout_local.py on the 4x4x4 mesh:
Lambda_C = E_C(R) - E_C(Gamma) with R = (1/2,1/2,1/2) fixed; Lambda_V = E_V(Gamma) - E_V(k_V*), k_V* fixed at the
reference's best rival. Pair mean <L> = (L(+u) + L(-u))/2; delta = <L> - L(reference static).
MC: mean and standard error over the 4 antithetic pair means.
Reported checks: Q1 compares |ZG(300 K, P5) - MC| with 2 SE; Q2 compares |delta_C(300 K, P5)| with 3 meV.
P_c(T): linear interpolation of <Lambda_C> between 4 and 5 GPa.
The EIGENVAL files (inputs/p4, inputs/p5, RUNROOT/jobs/<job>) and DONE are not deposited (configurations/<job>/OSZICAR is).
usage: python thermal_eval.py RUNROOT   (RUNROOT/jobs/<job>/EIGENVAL; references from inputs/p4, inputs/p5)"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RUN_DIR = Path(sys.argv[1])
path_readout_source = (HERE.parent / "substitution_path_split" / "path_readout_local.py").read_text(encoding="utf8")
namespace = {"__name__": "tv", "__file__": str(HERE.parent / "substitution_path_split" / "path_readout_local.py")}
saved_argv = sys.argv
sys.argv = ["x", "."]
exec(path_readout_source[:path_readout_source.index("res = {")], namespace)
sys.argv = saved_argv
eig, rivals, same_k = namespace["eig"], namespace["rivals"], namespace["same_k"]


def kidx(s, q):
    for i, p in enumerate(s["K"]):
        if np.allclose((p - q) - np.round(p - q), 0, atol=1e-6) or np.allclose((p + q) - np.round(p + q), 0, atol=1e-6):
            return i


ref = {p: eig(HERE / "inputs" / p.lower()) for p in ("P4", "P5")}
kvq = {p: ref[p]["K"][rivals(ref[p])[0]] for p in ref}
R = np.array([0.5, 0.5, 0.5])


def lam(s, p):
    return (1000 * (s["V"][s["G"]] - s["V"][kidx(s, kvq[p])]), 1000 * (s["C"][kidx(s, R)] - s["C"][s["G"]]))


L0 = {p: lam(ref[p], p) for p in ref}
out = {"reference": {p: {"Lambda_V": L0[p][0], "Lambda_C": L0[p][1], "kV_rival": kvq[p].tolist()} for p in ref},
       "pairs": {}}
missing = []
for d in sorted((RUN_DIR / "jobs").glob("*_p")):
    base = d.name[:-2]
    p = "P4" if "_P4" in base else "P5"
    dm = RUN_DIR / "jobs" / (base + "_m")
    if not ((d / "EIGENVAL").exists() and (dm / "EIGENVAL").exists() and (d / "DONE").exists()
            and (dm / "DONE").exists()):
        missing.append(base)
        continue
    lp, lm = lam(eig(d), p), lam(eig(dm), p)
    out["pairs"][base] = {"P": p, "L_plus": lp, "L_minus": lm, "mean_V": (lp[0] + lm[0]) / 2,
                          "mean_C": (lp[1] + lm[1]) / 2,
                          "delta_V": (lp[0] + lm[0]) / 2 - L0[p][0], "delta_C": (lp[1] + lm[1]) / 2 - L0[p][1],
                          "half_diff_C": (lp[1] - lm[1]) / 2}
out["missing_pairs"] = missing
pairs = out["pairs"]
mc_means = [pairs[f"mc{n}_T300_P5"]["mean_C"] for n in range(1, 5) if f"mc{n}_T300_P5" in pairs]
if len(mc_means) == 4 and "zg_all_T300_P5" in pairs:
    m, se = float(np.mean(mc_means)), float(np.std(mc_means, ddof=1) / 2)
    out["MC_T300_P5"] = {"mean_C": m, "se_C": se, "delta_C": m - L0["P5"][1], "pair_means": mc_means}
    out["Q1"] = {"zg_minus_mc": pairs["zg_all_T300_P5"]["mean_C"] - m, "two_se": 2 * se,
                 "pass": bool(abs(pairs["zg_all_T300_P5"]["mean_C"] - m) <= 2 * se)}
if "zg_all_T300_P5" in pairs:
    dC = pairs["zg_all_T300_P5"]["delta_C"]
    out["Q2"] = {"delta_C_300K_P5": dC, "reaches_3meV": bool(abs(dC) >= 3)}
out["Pc"] = {}
for T in (0, 150, 300):
    a, b = pairs.get(f"zg_all_T{T}_P4"), pairs.get(f"zg_all_T{T}_P5")
    if a and b:
        l4, l5 = a["mean_C"], b["mean_C"]
        pc = 4 + (0 - l4) / (l5 - l4)
        out["Pc"][f"{T}K"] = {"L4": l4, "L5": l5, "Pc_GPa": pc if 4 <= pc <= 5 else None,
                             "note": "within 4-5 GPa" if 4 <= pc <= 5 else ("above 5 GPa" if l5 < 0 else "below 4 GPa")}
l4s, l5s = L0["P4"][1], L0["P5"][1]
out["Pc"]["static"] = {"L4": l4s, "L5": l5s, "Pc_GPa": 4 + (0 - l4s) / (l5s - l4s)}
for p in ("P4", "P5"):
    a, b, c = pairs.get(f"zg_all_T300_{p}"), pairs.get(f"zg_bend_T300_{p}"), pairs.get(f"zg_stretch_T300_{p}")
    if a and b and c:
        out.setdefault("classes_300K", {})[p] = {"all": a["delta_C"], "bend": b["delta_C"], "stretch": c["delta_C"],
                                                 "bend_plus_stretch": b["delta_C"] + c["delta_C"]}
amplitude_delta_c = {k: pairs[k]["delta_C"]
                     for k in ("amp0.9_T300_P5", "zg_all_T300_P5", "amp1.1_T300_P5") if k in pairs}
if amplitude_delta_c:
    out["amplitude_sensitivity_delta_C"] = amplitude_delta_c
# same configuration computed on the GPU and the CPU machine
gpu_job, cpu_job = RUN_DIR / "jobs" / "xcheck_amp0.9_T300_P5_m", RUN_DIR / "jobs" / "amp0.9_T300_P5_m"
if (gpu_job / "DONE").exists() and (cpu_job / "DONE").exists():
    lx, ly = lam(eig(gpu_job), "P5"), lam(eig(cpu_job), "P5")
    dv, dc = lx[0] - ly[0], lx[1] - ly[1]
    out["cross_machine_check"] = {"gpu152": lx, "cpu105": ly, "dV": dv, "dC": dc,
                                  "pass_0p1meV": bool(abs(dv) <= 0.1 and abs(dc) <= 0.1)}
(HERE / "thermal_result.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1, default=float))
for k, v in pairs.items():
    print(f"{k:22s} C+ {v['L_plus'][1]:8.2f} C- {v['L_minus'][1]:8.2f} <C> {v['mean_C']:8.2f} "
          f"dC {v['delta_C']:7.2f}   dV {v['delta_V']:7.2f}")
