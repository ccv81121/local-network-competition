# Dispersive host force constants and the Ti->Sn response (SI Note 8, main text "With the dispersion ...")

- **Calculation:** 80-atom 2x1x2 Pnma CaTiO3 cell at the parent geometry of `alloy_local_geometry` case `ca_tio3_ti2_sn`.
  - 24 central displacements of ±0.01 Å plus a reference; the full H is rebuilt by the 32 operations.
  - Server: `$WORK/spinel/DISPERSION_20261104` (queue and worker scripts here; inputs from `build_dispersion_v1.py`, `s0_POSCAR.vasp`).
- **Analysis:** `analyse_dispersion_v1.py` writes `dispersion_result.json` and `arrays.npz` (H_s0, upred_s1, uact_s1, f_s1). It reads the forces and POSCARs in `raw/`.
- **Result:** `RESULT.md`, graded against the pre-registration `notes/np_submission/prereg_dispersion.md`.
  - T1 and T2 pass: cos 0.993; s_u 25.3% against 24.8%.
  - T3 is inconclusive for the bare kernel. The predicted u decays over 1.86 Å, the relaxed u over 1.71 Å.
  - T4: 96% of the bond-preserving weight is at the zone boundary. Rigid rotation makes up 0.52 at X, 0.92 at U and 0.26 at Z.
- `real_field.py` → `real_field.npz`: Ti->Sn fields for the real-field figure (drive, H^+f, relaxed u and its bond-preserving part, shell averages and fits).
