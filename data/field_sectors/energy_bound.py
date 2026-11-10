"""Energy-share bound e_b <= sqrt(s_f s_u) * |f_N||u_N| / (f.u)  (Cauchy-Schwarz on f.u_b = (P_b f_N).(P_b u_N)).
Inputs: four_state_leverage/calculations/<case>/{F_raff,u_b,u_pred}.npy and leverage.json; network s_f from
bending_enrichment/enrichment_all_systems.csv (drive_share, same drive at r_aff). Network atoms = nonzero rows of u_b."""
import csv, json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parents[1]
csvmap = {"k12_t45": "MgSc0.5Sb1.5Se4 (k12, tight)", "ca_tio3": "CaTiO3", "ca_sno3": "CaSnO3",
          "k08": "MgScSbSe4 (k08 ordering)", "ca2_sn_tio6_rs": "Ca2SnTiO6 (rock-salt)", "ca8_sn6_ti2o24": "Ca8Sn6Ti2O24"}
rows = {r["system"]: r for r in csv.DictReader(open(D / "bending_enrichment/enrichment_all_systems.csv", encoding="utf-8"))}
ev = json.load(open(D / "four_state_leverage/evaluation.json", encoding="utf-8"))["rows"]
evkey = {"k12_t45": "k12_t45", "ca_tio3": "CaTiO3", "ca_sno3": "CaSnO3", "k08": "k08",
         "ca2_sn_tio6_rs": "Ca2SnTiO6_rs", "ca8_sn6_ti2o24": "Ca8Sn6Ti2O24"}
out = {}
for c, sysname in csvmap.items():
    d = D / "four_state_leverage/calculations" / c
    f = np.load(d / "F_raff.npy"); ub = np.load(d / "u_b.npy"); u = np.load(d / "u_pred.npy")
    lev = json.load(open(d / "leverage.json", encoding="utf-8"))
    net = np.abs(ub.reshape(-1, 3)).sum(1) > 0
    m = np.repeat(net, 3)
    fN, uN = f[m], u[m]
    s_f = float(rows[sysname]["drive_share"])
    s_u = float(ub @ ub / (uN @ uN))
    e_b = float(f @ ub / (f @ u))
    gamma = float((f @ u) / (np.linalg.norm(fN) * np.linalg.norm(uN)))
    bound = np.sqrt(s_f * s_u) / gamma
    out[c] = dict(s_f=s_f, s_u=s_u, s_u_csv_pred=float(rows[sysname]["predicted_share"]), e_b=e_b,
                  e_b_leverage_json=lev.get("harmonic_energy_share_b"), gamma=gamma, bound=float(bound),
                  bound_su1=float(np.sqrt(s_f) / gamma), ratio=float(e_b / bound),
                  phi_actual=ev[evkey[c]]["phi_actual_signed"], E_share_DFT=ev[evkey[c]]["E_share_DFT_fourstate"])
    print(f"{c:16s} s_f {s_f:.4f} s_u {s_u:.3f} (csv {out[c]['s_u_csv_pred']:.3f}) e_b {e_b:.4f} (json {out[c]['e_b_leverage_json']:.4f}) "
          f"gamma {gamma:.3f} bound {bound:.4f} bound(s_u=1) {out[c]['bound_su1']:.3f} e/bound {e_b/bound:.2f} | phi {out[c]['phi_actual']:.3f} E_DFT {out[c]['E_share_DFT']:.3f}")
json.dump(out, open(Path(__file__).with_name("energy_bound.json"), "w"), indent=1)

# Proposition: phi/(1-phi) = (g_b/g_c)|u_b|/|u_c| >= (g_b/g_c) * e_b * cos(f,u) / sqrt(s_f_full)   (requires g_b/g_c > 0)
print("--- proposition: |u_b|/|u_c| >= e_b cos(f,u)/sqrt(s_f_full)")
for c in csvmap:
    d = D / "four_state_leverage/calculations" / c
    f = np.load(d / "F_raff.npy"); ub = np.load(d / "u_b.npy"); uc = np.load(d / "u_c.npy"); u = np.load(d / "u_pred.npy")
    net = np.repeat(np.abs(ub.reshape(-1, 3)).sum(1) > 0, 3)
    sff = out[c]["s_f"] * (f[net] @ f[net]) / (f @ f)
    cosfu = (f @ u) / np.linalg.norm(f) / np.linalg.norm(u)
    lhs = np.linalg.norm(ub) / np.linalg.norm(uc); rhs = out[c]["e_b"] * cosfu / np.sqrt(sff)
    lev = json.load(open(d / "leverage.json", encoding="utf-8")); gr = lev["g_b_meV_per_A"] / lev["g_c_meV_per_A"]
    out[c].update(s_f_full=float(sff), cos_fu=float(cosfu), ub_over_uc=float(lhs), prop_rhs=float(rhs), g_ratio=float(gr))
    print(f"{c:16s} |ub|/|uc| {lhs:.3f} >= {rhs:.3f}  ({'ok' if lhs >= rhs - 1e-12 else 'FAIL'})  g_b/g_c {gr:.2f}  phi_pred {lev['phi_pred']:.3f}")
json.dump(out, open(Path(__file__).with_name("energy_bound.json"), "w"), indent=1)

# Phase-diagram table (Supplementary Fig. 4b): geometric factor |u_b|/|u_c| and property factor g_b/g_c of the
# force-balance prediction, with predicted and four-state shares, for all eight compressions and two substitution steps.
ph = {}
names = {"k12_t45": "x=0.75, 4-5 GPa", "k12_05": "x=0.75, 0-5 GPa", "k08": "x=0.5", "layered_na_sc_ino2": "Na(Sc,In)O2",
         "ca_tio3": "CaTiO3", "ca_sno3": "CaSnO3", "ca2_sn_tio6_rs": "Ca2SnTiO6", "ca8_sn6_ti2o24": "Ca8Sn6Ti2O24"}
evk = {"k12_t45": "k12_t45", "k12_05": "k12_05", "k08": "k08", "layered_na_sc_ino2": "Layered_NaScInO2", "ca_tio3": "CaTiO3",
       "ca_sno3": "CaSnO3", "ca2_sn_tio6_rs": "Ca2SnTiO6_rs", "ca8_sn6_ti2o24": "Ca8Sn6Ti2O24"}
for c, lab in names.items():
    d = D / "four_state_leverage/calculations" / c
    ub = np.load(d / "u_b.npy"); uc = np.load(d / "u_c.npy")
    lev = json.load(open(d / "leverage.json", encoding="utf-8"))
    ph[c] = dict(label=lab, connectivity="edge" if c in ("k12_t45", "k12_05", "k08", "layered_na_sc_ino2") else "corner",
                 control="compression", geometric=float(np.linalg.norm(ub) / np.linalg.norm(uc)),
                 property=float(lev["g_b_meV_per_A"] / lev["g_c_meV_per_A"]), phi_pred=float(lev["phi_pred"]),
                 phi=float(ev[evk[c]]["phi_actual_signed"]))
al = json.load(open(D / "alloy_leverage/al_result.json", encoding="utf-8"))
for k, lab in (("i0506", "S5->S6"), ("i1112", "S11->S12")):
    e = al[k]["edge"]; pm = al[k]["pred_meta"]
    ph[k] = dict(label=lab, connectivity="edge", control="substitution", geometric=float(pm["u_b_norm"] / pm["u_c_norm"]),
                 property=float(al[k][e]["g_b"] / al[k][e]["g_c"]), phi_pred=float(al[k][e]["phi_pred"]),
                 phi=float(al[k][e]["phi_meas"]))
for k, r in ph.items():
    odds = r["geometric"] * r["property"]
    assert abs(odds / (1 + odds) - r["phi_pred"]) < 2e-3, (k, odds / (1 + odds), r["phi_pred"])
json.dump({"bound": out, "phase": ph}, open(Path(__file__).with_name("energy_bound.json"), "w"), indent=1)
print("phase table", len(ph), "cases; leverage identity reproduces phi_pred in all")
