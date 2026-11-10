"""Group neutralisation of local and network features in the fixed relations, with the canonical colocation labels for
Pi_sp. Identical to t1_t2_ablation.py except that Pi_sp is scored against the canonical labels
(true_binary_label == co_located, baseline BACC 0.923, the value quoted in the main text), so the ablation losses
share the baseline of the main metric. Only the two group ablations are run.
Reads the replay data of ../spinel_relations/colocation_relation. Output: one CSV, archived here as
neutralisation_ablation.csv.
usage: python neutralisation_ablation.py"""
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COLOCATION_DIR = HERE.parents[0] / "spinel_relations" / "colocation_relation"
DUAL_EDGE = runpy.run_path(str(COLOCATION_DIR / "dual_edge_validation.py"))
raw0 = np.load(COLOCATION_DIR / "raw_features_681x29.npy")
retained = json.loads((COLOCATION_DIR / "retained_feature_indices.json").read_text())
ids = np.load(COLOCATION_DIR / "sample_ids.npy").astype(str)
names = json.loads((COLOCATION_DIR / "formula_full_feature_names.json").read_text())
frozen_scales = np.load(COLOCATION_DIR / "feature_scales.npy")
best = json.loads((COLOCATION_DIR / "best_performance_formula.json").read_text())
state = json.loads((COLOCATION_DIR / "pi_s_decision_state.json").read_text())
node = json.loads(best["ast_json"])
epsilon = float(state.get("epsilon", 1e-6))
saved = pd.read_csv(COLOCATION_DIR / "reference_relation_scores.csv").set_index("sample_id").loc[ids]
dev = saved.partition.eq("DUAL_DEV_V1").to_numpy()
dev_442 = ~(np.load(COLOCATION_DIR / "mask_challenge_target.npy").astype(bool)
            | np.load(COLOCATION_DIR / "mask_challenge_guard.npy").astype(bool))
assert dev.sum() == 440 and dev_442.sum() == 442
REQUIRED_FEATURES = ["lattice_angle_anisotropy", "elem_A_d_electrons", "D_BXB_connectivity_disorder_proxy",
                     "XX_network_coherence", "alloy_first_ionization_kJmol_variance", "B_d_filling_mean",
                     "alloy_Z_variance", "BB_d2_mean", "BX_d_electrons_mismatch_norm",
                     "BB_graph_shortest_path_mean", "D_BX6_distortion_RMS", "elem_Bmix_Z"]
cols = np.array([retained.index(names.index(n)) for n in REQUIRED_FEATURES], int)
feat = [names[i] for i in retained]
NETWORK = {"BXB_angle_min", "BXB_path_length_std", "D_BXB_angle_disorder", "D_BXB_connectivity_disorder_proxy",
           "BB_d2_mean", "BB_graph_shortest_path_mean", "XX_d1_std", "XX_distance_anisotropy",
           "XX_t_pp_sigma_rel_mean", "XX_network_coherence"}
GLOBAL = {"lattice_angle_anisotropy"}
role = {f: ("network" if f in NETWORK else "global" if f in GLOBAL else "local") for f in feat}

# Baseline numerical contract: protected-ratio parameters, primitive scales and epsilons come from the development set.
xfs0 = np.zeros((len(ids), len(names)))
for r, o in enumerate(retained):
    xfs0[:, o] = raw0[:, r] / frozen_scales[o]
phi_dev0, RATIO_PARAMS_DEV = DUAL_EDGE["rc8_fit"](node, xfs0[dev].T, epsilon)
SHARED_IQR = float(np.quantile(phi_dev0, .75) - np.quantile(phi_dev0, .25))
_, RATIO_PARAMS_FULL = DUAL_EDGE["rc8_fit"](node, xfs0.T, epsilon)
PRIMITIVE_SCALES = DUAL_EDGE["robust_scales"](raw0[dev][:, cols])
fx0 = {n: np.nan_to_num(raw0[:, cols][:, i] / PRIMITIVE_SCALES[i], nan=0, posinf=1e6, neginf=-1e6)
       for i, n in enumerate(REQUIRED_FEATURES)}
EPS_V = 1e-6 * float(np.quantile(np.abs(fx0["XX_network_coherence"][dev] ** 2), .75))
EPS_C = 1e-6 * float(np.quantile(np.abs(fx0["alloy_Z_variance"][dev] * fx0["elem_Bmix_Z"][dev]), .75))


def signed_sqrt_product(a, b):
    return np.sign(a * b) * np.sqrt(np.abs(a * b))


def signed_square(a):
    return np.sign(a) * np.minimum(np.abs(a), 1e6) ** 2


def scores(raw):
    xfs = np.zeros((len(ids), len(names)))
    for r, o in enumerate(retained):
        xfs[:, o] = raw[:, r] / frozen_scales[o]
    pis = DUAL_EDGE["rc8_apply"](node, xfs.T, RATIO_PARAMS_DEV) / SHARED_IQR
    psp = DUAL_EDGE["rc8_apply"](node, xfs.T, RATIO_PARAMS_FULL)
    fx = {n: np.nan_to_num(raw[:, cols][:, i] / PRIMITIVE_SCALES[i], nan=0, posinf=1e6, neginf=-1e6)
          for i, n in enumerate(REQUIRED_FEATURES)}
    terms = {"V_shared": pis,
             "V_aniso_dA": 3 * fx["lattice_angle_anisotropy"] * fx["elem_A_d_electrons"],
             "V_conn_over_XX": (-fx["D_BXB_connectivity_disorder_proxy"]
                                / (np.abs(fx["XX_network_coherence"] ** 2) + EPS_V)),
             "V_ion_dfill": (-(1 / 6) * fx["alloy_first_ionization_kJmol_variance"]
                             * signed_square(fx["B_d_filling_mean"])),
             "C_varZ": -signed_square(fx["alloy_Z_variance"]) / 3,
             "C_BB_mismatch": 2 * signed_sqrt_product(
                 fx["BB_graph_shortest_path_mean"],
                 signed_sqrt_product(fx["BB_d2_mean"], fx["BX_d_electrons_mismatch_norm"])),
             "C_oct_over_Z": (signed_square(fx["D_BX6_distortion_RMS"])
                              / (np.abs(fx["alloy_Z_variance"] * fx["elem_Bmix_Z"]) + EPS_C))}
    v = terms["V_shared"] + terms["V_aniso_dA"] + terms["V_conn_over_XX"] + terms["V_ion_dfill"]
    c = terms["C_varZ"] + terms["C_BB_mismatch"] + terms["C_oct_over_Z"]
    return {"sp": psp, "V": v, "C": c}, terms


LABELS = {"sp": saved.true_binary_label.eq("co_located").to_numpy(int), "V": saved.V_true_Gamma.to_numpy(int),
          "C": saved.C_true_Gamma.to_numpy(int)}
FIT_MASK = {"sp": dev_442, "V": dev, "C": dev}
TAU_FROZEN = {"sp": float(best["threshold"]), "V": float(saved.tau_V.iloc[0]), "C": float(saved.tau_C.iloc[0])}


def bacc(y, pred):
    tp = np.mean(pred[y == 1]) if (y == 1).any() else np.nan
    tn = np.mean(~pred[y == 0]) if (y == 0).any() else np.nan
    return 0.5 * (tp + tn)


def refit(s, k):
    m = FIT_MASK[k]
    y = LABELS[k][m]
    u = np.unique(s[m])
    cand = np.concatenate([[u[0] - 1], 0.5 * (u[1:] + u[:-1]), [u[-1] + 1]])
    b = [bacc(y, s[m] <= t) for t in cand]
    return cand[int(np.nanargmax(b))], float(np.nanmax(b))


base, Tb = scores(raw0)
replay_error = {
    "Pi_S_shared": float(np.max(np.abs(base["V"] - Tb["V_aniso_dA"] - Tb["V_conn_over_XX"] - Tb["V_ion_dfill"]
                                       - saved.Pi_S_shared.to_numpy()))),
    "Pi_V": float(np.max(np.abs(base["V"] - saved.Pi_V_raw.to_numpy()))),
    "Pi_C": float(np.max(np.abs(base["C"] - saved.Pi_C_raw.to_numpy()))),
    "Pi_sp": float(np.max(np.abs(base["sp"] - saved.Pi_S_frozen_raw.to_numpy())))}
assert max(replay_error.values()) < 1e-9, replay_error
FROZEN_BACC = {k: bacc(LABELS[k][FIT_MASK[k]], base[k][FIT_MASK[k]] <= TAU_FROZEN[k]) for k in base}
assert (abs(FROZEN_BACC["sp"] - 0.9231219252758942) < 1e-12 and abs(FROZEN_BACC["V"] - 0.8732504637325047) < 1e-12
        and abs(FROZEN_BACC["C"] - 0.8665079365079364) < 1e-12), FROZEN_BACC
BASE_REFIT = {k: refit(base[k], k) for k in base}
MEDIANS = {k: np.median(raw0[FIT_MASK[k]], axis=0) for k in base}
rows = []


def ablate(label, idxs, kind):
    r = {"ablation": label, "kind": kind}
    for k in base:
        raw = raw0.copy()
        raw[:, idxs] = MEDIANS[k][idxs]
        s = scores(raw)[0][k]
        t, bd = refit(s, k)
        r[f"{k}_dev_bacc"] = bd
        r[f"{k}_d_dev"] = bd - BASE_REFIT[k][1]
        r[f"{k}_d_dev_frozen_tau"] = bacc(LABELS[k][FIT_MASK[k]], s[FIT_MASK[k]] <= TAU_FROZEN[k]) - FROZEN_BACC[k]
        perm = []
        for seed in range(200):
            raw = raw0.copy()
            idm = np.flatnonzero(FIT_MASK[k])
            p = np.random.default_rng(seed).permutation(idm)
            raw[np.ix_(idm, idxs)] = raw0[np.ix_(p, idxs)]
            perm.append(refit(scores(raw)[0][k], k)[1] - BASE_REFIT[k][1])
        r[f"{k}_d_dev_perm_median"] = float(np.median(perm))
    rows.append(r)


for g in ("local", "network"):
    ablate(f"GROUP_{g}", [i for i, f in enumerate(feat) if role[f] == g], "group")
abl = pd.DataFrame(rows)
abl.to_csv(HERE / "neutralisation_ablation.csv", index=False)
print(abl.T.to_string())
print("baseline refit", BASE_REFIT)
