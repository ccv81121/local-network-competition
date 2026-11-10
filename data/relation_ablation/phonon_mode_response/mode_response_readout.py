"""T3 / T4 read-out of the frozen-phonon statics written by prepare_modes.py.
Margins as in path_readout.py: Lambda_V = E_V(G) - max_{k!=G} E_V(k), Lambda_C = min_{k!=G} E_C(k) - E_C(G); main
readout with the rival fixed at the reference structure's best rival, own-rival reported.
dL/dQ = (L+ - L-)/2Q, d2L/dQ2 = (L+ + L- - 2L0)/Q^2 (Q in sqrt(amu) A). Unit-energy deformation potential =
|d2L/dQ2| / omega^2. T4: <Q^2> = (hbar/2w) coth(hbar w / 2kT); dL_T = 1/2 sum d2 <Q^2>; sigma^2 = sum d1^2 <Q^2>,
over the sampled modes of each class.
The EIGENVAL files of the statics (<system>/<mode>_<sign>/, ref/) are not deposited; only their OSZICAR.
usage: python mode_response_readout.py"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PATH_SPLIT_DIR = HERE.parents[1] / "substitution_path_split"
sys.argv = ["x", "."]
path_readout_source = (PATH_SPLIT_DIR / "path_readout.py").read_text(encoding="utf8")
namespace = {"__name__": "t3"}
exec(path_readout_source[:path_readout_source.index("res = {")], namespace)
eig, rivals, lam, same_k = namespace["eig"], namespace["rivals"], namespace["lam"], namespace["same_k"]
MODE_DIR = HERE
SYSTEM_DIRECTORY = {"k12": "k12", "CaTiO3": "ca_tio3", "CaSnO3": "ca_sno3"}  # directory of each system on disk
MODES = {m["system"]: m for m in json.loads((MODE_DIR / "modes.json").read_text())}
EV, AMU, ANGSTROM, HBAR, KB = 1.602176634e-19, 1.66053906660e-27, 1e-10, 1.054571817e-34, 1.380649e-23
res = {"T3": {}, "T4": {}}
rows = []
for sysname, mode_set in MODES.items():
    ref = eig(MODE_DIR / SYSTEM_DIRECTORY[sysname] / "ref")
    kv, kc = rivals(ref)
    L0 = lam(ref, kv, kc)
    for sel in mode_set["selected"]:
        w2 = sel["w2_eV_A2_amu"]
        Q = sel["Q_amp_sqrtamu_A"]
        Lp, Lm, own = [], [], {}
        for tag in ("p", "m"):
            s = eig(MODE_DIR / SYSTEM_DIRECTORY[sysname] / f"{sel['label']}_{tag}")
            m = same_k(s, ref)
            (Lp if tag == "p" else Lm).append(lam(s, m(kv), m(kc)))
            own[tag] = lam(s, *rivals(s))
        Lp, Lm = Lp[0], Lm[0]
        w = np.sqrt(w2 * EV / (ANGSTROM ** 2 * AMU))  # rad/s
        r = {"system": sysname, "label": sel["label"], "class": sel["class"], "w2": w2,
             "freq_cm1": w / (2 * np.pi * 2.99792458e10),
             "bp_share": sel["bp_share"], "Q": Q}
        for e, j in (("V", 0), ("C", 1)):
            d1 = (Lp[j] - Lm[j]) / (2 * Q)
            d2 = (Lp[j] + Lm[j] - 2 * L0[j]) / Q ** 2
            r[f"L0_{e}"], r[f"Lp_{e}"], r[f"Lm_{e}"] = L0[j], Lp[j], Lm[j]
            r[f"d1_{e}"], r[f"d2_{e}"], r[f"ued_{e}"] = d1, d2, abs(d2) / w2
            r[f"own_p_{e}"], r[f"own_m_{e}"] = own["p"][j], own["m"][j]
        for T in (300.0, 10.0):
            # sqrt(amu) A squared
            q2 = HBAR / (2 * w) / np.tanh(HBAR * w / (2 * KB * T)) / (AMU * ANGSTROM ** 2)
            r[f"Q2_{int(T)}"] = q2
        rows.append(r)
for sysname in MODES:
    system_rows = [r for r in rows if r["system"] == sysname]
    res["T3"][sysname] = {}
    res["T4"][sysname] = {}
    for e in "VC":
        b = np.median([r[f"ued_{e}"] for r in system_rows if r["class"] == "bend"])
        s = np.median([r[f"ued_{e}"] for r in system_rows if r["class"] == "stretch"])
        res["T3"][sysname][e] = {"bend_median_ued": b, "stretch_median_ued": s, "ratio": b / s if s else None}
        for T in (300, 10):
            for cls in ("bend", "stretch"):
                rr = [r for r in system_rows if r["class"] == cls]
                res["T4"][sysname][f"{e}_{cls}_{T}K"] = {
                    "dL_T_meV": 0.5 * sum(r[f"d2_{e}"] * r[f"Q2_{T}"] for r in rr),
                    "sigma_meV": float(np.sqrt(sum(r[f"d1_{e}"] ** 2 * r[f"Q2_{T}"] for r in rr)))}
            bb, ss = (res["T4"][sysname][f"{e}_bend_{T}K"]["dL_T_meV"],
                      res["T4"][sysname][f"{e}_stretch_{T}K"]["dL_T_meV"])
            res["T4"][sysname][f"{e}_ratio_abs_{T}K"] = abs(bb) / abs(ss) if ss else None
rC = {k: res["T3"][k]["C"]["ratio"] for k in MODES}
res["T3a"] = {"k12_C_ratio": rC["k12"],
              "verdict": "holds" if rC["k12"] >= 2 else ("falsified" if rC["k12"] <= 1 else "undetermined")}
res["T3b"] = {"ratios_C": rC, "verdict": "holds" if rC["k12"] > max(rC["CaTiO3"], rC["CaSnO3"]) else
              ("falsified" if rC["k12"] < min(rC["CaTiO3"], rC["CaSnO3"]) else "undetermined")}
t4 = {k: res["T4"][k]["C_ratio_abs_300K"] for k in MODES}
res["T4a"] = {"k12_C_ratio_300K": t4["k12"],
              "verdict": "holds" if t4["k12"] >= 2 else ("falsified" if t4["k12"] <= 1 else "undetermined")}
res["T4b"] = {"ratios_C_300K": t4, "k12_largest": t4["k12"] > max(t4["CaTiO3"], t4["CaSnO3"])}
(HERE / "mode_response_results.json").write_text(json.dumps(res, indent=1, default=float))
import csv
with open(HERE / "mode_table.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(json.dumps({k: res[k] for k in ("T3a", "T3b", "T4a", "T4b")}, indent=1, default=float))
for k in MODES:
    print(k, json.dumps(res["T3"][k], default=float))
for k in MODES:
    print(k, {kk: (round(v["dL_T_meV"], 2), round(v["sigma_meV"], 2)) if isinstance(v, dict) else round(v, 2)
              for kk, v in res["T4"][k].items() if "300K" in kk})
