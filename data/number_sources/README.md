# Source records of the perovskite relations, the spinel database and the optics settings

Byte copies of the original records behind the relation statistics, the spinel substitution database and the optical-response settings.

- `sp/`: single-perovskite relation on 136 states. `sp_relation_formulas.csv` (row 1: 6 descriptors, 10 operator nodes in `ast_json`, threshold 17.5066), `sp_out_of_fold_metrics.json` (out-of-fold BACC 0.99398, AUC 0.98818) and the per-state out-of-fold scores of the rank-3 relation `sp_rank3_out_of_fold_scores.csv`.
- `dp/`: double-perovskite relation on 85 states. `dp_search_record.json` (search settings and summary), `dp_dataset_summary.json` (85 states, five grouped folds, 182 source descriptors, 109 resolved; names in `full_feature_names.json`), `best_performance_formula.json` (11 descriptors, 30 operator nodes, threshold 7174.45, out-of-fold BACC 0.94235, AUC 0.94013), `top30_safe.csv`.
- `spinel_database/dataset.csv`: 1003 spinel states; 39 k = 0 parents, the other 964 grouped by (mp_id, substitute) form 62 substitution paths, 43 complete (k = 1…16) and 19 shorter.
- `optics/c005_p_mp-1001019_sb_k12_{2,5}gpa/`: INCAR and KPOINTS of the optical runs (LOPTICS, NBANDS 224, CSHIFT 0.05 eV, Γ-centred 4×4×4 mesh), with the task definition and reference band edges in `task_definition.json`.
