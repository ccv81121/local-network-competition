# Single- and double-perovskite relations

Edge-location relations for perovskite oxides: the double-perovskite (DP) valence/conduction relations evaluated out of fold on 85 states (grouped five-fold, by substitution trajectory), and the single-perovskite (SP) relation evaluated on nine BaSn1−xTixO3 states from their raw descriptors.

- DP: `dp_dual_oof_4state.csv` (per-state fold, margins, scores and four-state class), `dp_dual_oof_metrics.json` (BACC, AUC, four-state accuracy), `dp_out_of_fold_summary.json` (79 of 85 states with both edges correct), `dp_edge_targets.csv` (edge k points and colocation labels), `dp_state_counts.json` (state and target counts).
- SP: `compute_sp_s20_scores.py` reads `sp_s20_raw_inputs_full9.csv` and the descriptor scales `sp_s20_descriptor_scales.csv` and writes `sp_s20_scores_9_states.csv`; descriptor definitions in `sp_s20_feature_whitelist_and_operational_definitions.csv`, per-state edge labels in `basnti_9_state_edge_labels.csv`.
- The perovskite descriptors are computed from the relaxed structures by `python -m symbolic_regression describe --family sp|dp` (see `data/descriptor_tables/README.md`). `basnti_features.csv` and `basnti_features_new4.csv` hold the BaSn1−xTixO3 descriptors; elemental values absent from the element table are in `element_extensions.json`.
