# Frozen path statistics of the 1003-state spinel database

Copied 2026-11-08 from the archive ($ARCHIVE, handoff v2.0 of 2026-08-21, `data_authority/FINAL_CORE_FACTS.json`, key `spinel_paper`; and `01_core_freeze/02_EDGE_RESOLVED_RECONSTRUCTION/CANONICAL_TRAJECTORY_DENOMINATORS.csv`).

- 62 paths, 1003 states; 38 paths with at least one adjacent relation change, 25 with several; 9 of 13 endpoint-matched paths reverse internally; 4 of 43 fully sampled paths show strict mismatch-edge transfer.
- The script that produced 38 and 25 was not kept with the archive. Recounting from `dataset.csv` with the labels available there (is_direct, Bandgap Type, label_gap_state_stage1, label_phase_compact_stage1, signs of the two margins, ordering_margin_kind_canonical) reproduces 62 paths and 25 multiple changes for is_direct, but gives 32 rather than 38 paths with at least one change. 9 of 13 is reproduced by `four_state_reanalysis/spinel_913_cellconv.txt`.
