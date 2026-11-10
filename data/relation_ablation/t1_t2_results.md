# Neutralisation ablation and relation terms (T1, T2)

Source: `t1_t2_ablation.py` → `t1_t2_results.json`; T2b from `descriptor_sensitivity/sensitivity_eval.py` → `descriptor_sensitivity/sensitivity_results.json`.

## T1-R: replay of the Top30 subtree ablation
- The Top30 vector analysis (TOP_N 30, GREEDY_ROUNDS 8, FAMILY_JACCARD 0.55, MAX_FAMILY_REPS 10, PYTHONHASHSEED 56) was rerun from the stored search checkpoint. `TOP30_FEATURE_ABLATION_STABILITY.csv`, `TOP30_SINGLE_STEP_ABLATIONS.csv` (5256 rows), `TOP30_BASELINE_REPLAY.csv` and `TOP30_FEATURE_FREQUENCY.csv` are reproduced exactly.
- BXB_angle_min occurs in 30 formulas and is removable in 27. Removing it changes BACC by −0.3765 (median), −0.3781 (worst) and −0.0488 (best); no formula keeps its score. It is the most depended-on of the 52 features.

## T1: neutralisation in the fixed relations
Evaluation sets: Π<sub>sp</sub> on its 442 search structures with the protocol labels (threshold −114.17, BACC 0.925); Π<sub>V</sub>, Π<sub>C</sub> on the 440 double-coordinate development structures with the canonical edge labels (BACC 0.873 / 0.867). Medians for neutralisation and threshold refits are taken on the same sets.

Group neutralisation (ΔBACC):

| Relation | all local features | all network features |
|---|---|---|
| Π<sub>sp</sub> | −0.298 | −0.145 |
| Π<sub>V</sub> | −0.285 | −0.086 |
| Π<sub>C</sub> | −0.292 | −0.039 |

Largest single-feature losses (ΔBACC):
- Π<sub>sp</sub>: XX_distance_anisotropy (network) −0.125, D_BX6_bond_cv −0.098, BX6_distortion_cv −0.097. BXB_angle_min: −0.046 with refitted threshold, −0.068 with the original threshold, −0.143 as permutation median.
- Π<sub>V</sub>: elem_A_d_electrons −0.084, B_d_filling_mean −0.074, BX6_distortion_cv −0.061.
- Π<sub>C</sub>: D_BX6_distortion_RMS −0.144, alloy_Z_variance −0.113, BB_d2_mean (network) −0.028.

Reading:
- Within the single relation Π<sub>sp</sub>, BXB_angle_min is not the largest single loss (−0.046); across the Top30 formula family it is the largest dependence (−0.377). The subtree ablation measures how much the family relies on a feature; the neutralisation measures its weight within one formula.
- Both groups lower Π<sub>sp</sub> by more than 0.05 (local −0.298, network −0.145): the relations need both classes of information, the local class more strongly.
- Π<sub>V</sub> and Π<sub>C</sub> rely on different leading features, both mainly local: A-site d electrons, B-site d filling and octahedral distortion for the valence edge; octahedral distortion and composition variance for the conduction edge.

## T2: local and network terms
- T2a: features are scaled but not centred, so the network term −D_BXB/C_XX² is always negative and the local term −(1/6)Var(I1)·f_Bd² always ≤ 0; their signs are fixed by the formula structure, and Π<sub>C</sub> has no purely network term. A sign test near the boundary is therefore not informative; the changes of the terms along the path are.
- T2c, composition path (`t2c_path_terms.csv`, S01–S14 all in the development set):
  - Valence edge: Π<sub>V</sub> crosses its threshold at S04, two steps before the calculated switch S05→S06. The largest change along the path is in the local term V_ion_dfill (−0.01 to −1.07); the network term V_conn_over_XX stays between −0.04 and −0.10.
  - Conduction edge: Π<sub>C</sub> crosses its threshold between S10 and S11, one step before the calculated switch S11→S12. It first decreases and then increases (internal reversal), driven by the local term C_varZ (extremum at S08) and the mixed term C_BB_mismatch.

## T2b: sensitivity to bending and stretching
Twenty random directions in each of the bond-preserving (45-dimensional) and bond-changing (96-dimensional) subspaces of the canonical S05, S06, S11, S12 structures, ±0.01 Å maximum atomic displacement (leak |J·v| ≈ 1e−16); 324 structures in total, features recomputed with the feature pipeline (maximum relative deviation from the stored matrix 1.7e−13, `t2b/gate.json`).

| | bond-preserving median | bond-changing median | ratio |
|---|---|---|---|
| Π<sub>sp</sub> | 5.66 | 6.46 | 0.88 |
| Π<sub>V</sub> | 0.44 | 0.49 | 0.90 |
| Π<sub>C</sub> | 0.077 | 0.34 | 0.22 |

Per structure, the Π<sub>sp</sub> ratio is 1.43 (S05), 0.95 (S06), 0.88 (S11) and 0.47 (S12). The relations are not more sensitive to bond-preserving bending than to bond-changing stretching, and Π<sub>C</sub> is markedly more sensitive to stretching. The network features, including the shared-ligand angle, describe differences between structures (composition, distortion) rather than the response to bending about a given structure.
