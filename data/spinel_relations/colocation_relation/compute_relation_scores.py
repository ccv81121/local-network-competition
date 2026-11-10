"""Replay of the spinel single-coordinate and dual-edge relations (formula version 1.0.3) on the 681 retained states
from the stored arrays; writes the replay tables and relation_parameters.json to this folder.
No search, coefficient refinement or threshold fitting. Helper functions are loaded from dual_edge_validation.py.
usage: python compute_relation_scores.py
"""
from pathlib import Path
import json
import runpy
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
validation = runpy.run_path(str(HERE / "dual_edge_validation.py"))
raw = np.load(HERE / "raw_features_681x29.npy", allow_pickle=False)
retained = json.loads((HERE / "retained_feature_indices.json").read_text())
ids = np.load(HERE / "sample_ids.npy", allow_pickle=False).astype(str)
names = json.loads((HERE / "formula_full_feature_names.json").read_text())
dev_ids = set(pd.read_csv(HERE / "dual_dev_sample_ids.csv").sample_id.astype(str))
dev = np.array([s in dev_ids for s in ids])
assert int(dev.sum()) == 440
frozen_scales = np.load(HERE / "feature_scales.npy", allow_pickle=False)
best = json.loads((HERE / "best_performance_formula.json").read_text())
state = json.loads((HERE / "pi_s_decision_state.json").read_text())
node = json.loads(best["ast_json"])
epsilon = float(state.get("epsilon", 1e-6))
scaled_features = np.zeros((len(ids), len(names)), float)
for reduced_index, original_index in enumerate(retained):
    scaled_features[:, original_index] = raw[:, reduced_index] / frozen_scales[original_index]
phi_dev, parameters = validation["rc8_fit"](node, scaled_features[dev].T, epsilon)
phi_all = validation["rc8_apply"](node, scaled_features.T, parameters)
iqr = float(np.quantile(phi_dev, .75) - np.quantile(phi_dev, .25))
assert np.isfinite(iqr) and iqr > 1e-12
pi_shared = phi_all / iqr

required_features = ["lattice_angle_anisotropy", "elem_A_d_electrons",
                     "D_BXB_connectivity_disorder_proxy", "XX_network_coherence",
                     "alloy_first_ionization_kJmol_variance", "B_d_filling_mean",
                     "alloy_Z_variance", "BB_d2_mean", "BX_d_electrons_mismatch_norm",
                     "BB_graph_shortest_path_mean", "D_BX6_distortion_RMS", "elem_Bmix_Z"]
cols = np.array([retained.index(names.index(n)) for n in required_features], int)
s = validation["robust_scales"](raw[dev][:, cols])
scaled_primitives = np.nan_to_num(np.asarray(raw[:, cols], float) / s[None, :],
                                  nan=0, posinf=1e6, neginf=-1e6)
fx = {n: scaled_primitives[:, i] for i, n in enumerate(required_features)}
eps_v = 1e-6 * float(np.quantile(np.abs(fx["XX_network_coherence"][dev] ** 2), .75))
eps_c = 1e-6 * float(np.quantile(np.abs(fx["alloy_Z_variance"][dev] * fx["elem_Bmix_Z"][dev]), .75))
# Replayed raw coordinates: v = Pi_V, c = Pi_C (an edge is Gamma when its coordinate is <= tau).
v = (pi_shared + 3 * fx["lattice_angle_anisotropy"] * fx["elem_A_d_electrons"]
     - fx["D_BXB_connectivity_disorder_proxy"] / (np.abs(fx["XX_network_coherence"] ** 2) + eps_v)
     - (1 / 6) * fx["alloy_first_ionization_kJmol_variance"]
     * np.sign(fx["B_d_filling_mean"]) * np.minimum(np.abs(fx["B_d_filling_mean"]), 1e6) ** 2)


def signed_sqrt_product(a, b):
    return np.sign(a * b) * np.sqrt(np.abs(a * b))


c = (-np.sign(fx["alloy_Z_variance"]) * np.minimum(np.abs(fx["alloy_Z_variance"]), 1e6) ** 2 / 3
     + 2 * signed_sqrt_product(fx["BB_graph_shortest_path_mean"],
                               signed_sqrt_product(fx["BB_d2_mean"], fx["BX_d_electrons_mismatch_norm"]))
     + np.sign(fx["D_BX6_distortion_RMS"]) * np.minimum(np.abs(fx["D_BX6_distortion_RMS"]), 1e6) ** 2
     / (np.abs(fx["alloy_Z_variance"] * fx["elem_Bmix_Z"]) + eps_c))
saved = pd.read_csv(HERE / "reference_relation_scores.csv").set_index("sample_id").loc[ids]
errors = {key: float(np.max(np.abs(values - saved[key].to_numpy(float))))
          for key, values in [("Pi_S_shared", pi_shared), ("Pi_V_raw", v), ("Pi_C_raw", c)]}
assert max(errors.values()) < 1e-9, errors
tau_v = saved.tau_V.to_numpy(float)
tau_c = saved.tau_C.to_numpy(float)
assert np.array_equal(v <= tau_v, saved.V_pred_Gamma.to_numpy(bool))
assert np.array_equal(c <= tau_c, saved.C_pred_Gamma.to_numpy(bool))

used_features = set()


def collect_terminals(ast_node):
    if ast_node.get("op") == "terminal":
        used_features.add(int(ast_node["feature_index"]))
    for key in ["arg", "left", "right"]:
        if key in ast_node:
            collect_terminals(ast_node[key])


collect_terminals(node)
pd.DataFrame({"feature_name": required_features, "frozen_scale": s}).to_csv(
    HERE / "dual_12_primitive_scales.csv", index=False)
pd.DataFrame({"feature_index": sorted(used_features), "feature_name": [names[i] for i in sorted(used_features)],
              "frozen_scale": [frozen_scales[i] for i in sorted(used_features)]}).to_csv(
    HERE / "shared_19_primitive_scales.csv", index=False)
out = saved.reset_index()[["sample_id", "partition", "Pi_S_shared", "Pi_S_frozen_raw", "Pi_V_raw", "tau_V",
                           "V_pred_Gamma", "V_true_Gamma", "Pi_C_raw", "tau_C", "C_pred_Gamma", "C_true_Gamma"]].copy()
out["Pi_S_shared_replayed"] = pi_shared
out["Pi_V_replayed"] = v
out["Pi_C_replayed"] = c
out.to_csv(HERE / "relation_scores_681.csv", index=False)
historical = validation["rc8_q75_eval"](node, scaled_features.T, epsilon)
_, historical_parameters = validation["rc8_fit"](node, scaled_features.T, epsilon)
historical_error = float(np.max(np.abs(historical - saved.Pi_S_frozen_raw.to_numpy(float))))
assert historical_error < 1e-9, historical_error
old_dev = ~(np.load(HERE / "mask_challenge_target.npy", allow_pickle=False).astype(bool)
            | np.load(HERE / "mask_challenge_guard.npy", allow_pickle=False).astype(bool))
assert int(old_dev.sum()) == 442
legacy_y = np.load(HERE / "labels.npy", allow_pickle=False).astype(int)
canonical_y = saved.true_binary_label.eq("co_located").astype(int).to_numpy()
legacy_metrics = validation["binary_metrics"](legacy_y[old_dev], historical[old_dev],
                                              historical[old_dev] <= float(best["threshold"]), -1,
                                              float(best["threshold"]))
canonical_metrics = validation["binary_metrics"](canonical_y[old_dev], historical[old_dev],
                                                 historical[old_dev] <= float(best["threshold"]), -1,
                                                 float(best["threshold"]))
edge_metrics = {}
for partition, expected_n in [("DUAL_DEV_V1", 440), ("FROZEN_CHALLENGE_HOLDOUT", 224)]:
    mask = saved.partition.eq(partition).to_numpy()
    assert int(mask.sum()) == expected_n
    edge_metrics[partition] = {}
    for edge, score, threshold in [("V", v, float(tau_v[0])), ("C", c, float(tau_c[0]))]:
        edge_metrics[partition][edge] = validation["binary_metrics"](
            saved[edge + "_true_Gamma"].to_numpy(int)[mask], score[mask],
            score[mask] <= threshold, -1, threshold)
square_inputs = ["B_d_filling_mean", "alloy_Z_variance", "D_BX6_distortion_RMS"]
square_ranges = {n: {"min": float(fx[n].min()), "max": float(fx[n].max())} for n in square_inputs}
assert all(np.all(np.isfinite(fx[n])) and np.all((fx[n] >= 0) & (fx[n] <= 1e6)) for n in square_inputs)
record = {
    "authority_version": "1.0.3", "formula_id": best["formula_id"],
    "development_n": int(dev.sum()), "total_n": len(ids),
    "partition_counts": {k: int(v) for k, v in saved.partition.value_counts().items()},
    "shared_component_dual_IQR": iqr,
    "shared_component_primitive_scales": "shared_19_primitive_scales.csv",
    "shared_component_ast": "best_performance_formula.json::ast_json",
    "shared_component_dual_protected_ratio_parameters": parameters,
    "shared_component_dual_input_epsilon": epsilon,
    "shared_component_dual_definition": "rc8_apply(same frozen AST, frozen 19/214 scales, "
                                        "ratio eps fitted on 440 dev)/IQR(dev raw component)",
    "standalone_shared_definition": "same frozen AST/scales; ratio eps=epsilon*Q75(abs(denominator)) "
                                    "over full supplied protocol matrix; no dual IQR division",
    "standalone_shared_protected_ratio_parameters_full681": historical_parameters,
    "outer_V_eps": eps_v, "outer_C_eps": eps_c,
    "outer_ratio_definition": "a/(abs(b)+eps_e), eps_e=1e-6*Q75_dev(abs(b))",
    "outer_primitive_scale_definition": "dev IQR; if not finite or <=1e-12, 1.4826*MAD; "
                                        "then dev population std; then 1",
    "outer_primitive_scaling": "each of the 12 raw primitives is individually divided by its frozen scale "
                               "before products/powers; no centering",
    "max_absolute_replay_errors": errors,
    "max_standalone_component_replay_error": historical_error,
    "standalone_442_legacy_metrics": legacy_metrics,
    "standalone_442_primitive_canonical_metrics": canonical_metrics,
    "standalone_442_label_changes": int(np.sum(legacy_y[old_dev] != canonical_y[old_dev])),
    "frozen_edge_metrics_replayed": edge_metrics,
    "signed_square_equals_ordinary_square_on_all_681_inputs": True,
    "scaled_square_input_ranges": square_ranges,
    "frozen_edge_predictions_all_match": True,
    "V_threshold": float(tau_v[0]), "C_threshold": float(tau_c[0]),
    "numeric_semantics": "source nan_to_num(..., nan=0,posinf=1e6,neginf=-1e6); "
                         "signed square clips absolute base at1e6; preserved, not replaced",
    "scope": "parameter extraction and score replay only; no search, coefficient refit, threshold optimization, "
             "new predictions, DFT, or remote changes"
}
(HERE / "relation_parameters.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
