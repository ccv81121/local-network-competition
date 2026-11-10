# Deepening: three-term split and noisy-Krylov cost

Existing force constants only; no new DFT. Python with numpy, run from this folder with `OPENBLAS_NUM_THREADS=1`.

- `three_term_split.py` -> `three_term_split.json`. Loads H, drive f, actual relaxation u for the six systems of `bending_enrichment` plus Cd(Ga,In)2O4 and Na(Sc,In)O2 (inputs in `bending_amplitude/sector_force_balance`). First reproduces drive_share / predicted_share of `enrichment_all_systems.csv` (all six agree to <= 2e-15; the two extra systems have no csv row).
- `krylov_cost.py` -> `krylov_cost.json` (imports the loader of `three_term_split.py`).
- Space W: Cartesian, translations removed (full-H systems) or the symmetric-sector basis S. B = bond-preserving network motions in W with A-cations fixed at 0 (same definition as `symmetric_sector_blocks.py`), R = complement in W. Euclidean metric in W. t1 = H_bb^-1 f_b (clamped), t2 = (S_b^-1 - H_bb^-1) f_b (softening), t3 = -S_b^-1 H_br H_rr^-1 f_r (transmission); t1+t2+t3 equals the B-component of H^-1 f to 3e-14 relative (all systems). H is SPD in W for all systems.
- s_f is the old share of the drive (network atoms, centred); sB_f = |f_b|^2/|f|^2 in W differs because A-cations sit in R.

## Three-term split (squared fraction |t_i|^2/sum|t_j|^2 ; signed t_i.b/b.b)

| system | dim B | s_f | |b|^2/|H^-1f|^2 | sq t1/t2/t3 | signed t1/t2/t3 | old posterior direct/coupled (actual u) |
|---|---|---|---|---|---|---|
| k12 tight | 45 | 0.0054 | 0.280 | 0.289/0.189/0.521 | 0.285/0.268/0.447 | 0.260/0.739 |
| CaTiO3 | 21 | 0.1008 | 0.115 | 0.535/0.290/0.175 | 0.430/0.437/0.133 | 0.504/0.529 |
| CaSnO3 | 21 | 0.1158 | 0.218 | 0.666/0.276/0.058 | 0.730/0.474/-0.204 | 0.779/0.253 |
| k08 | 4 | 0.0029 | 0.059 | 0.365/0.104/0.531 | 0.431/0.152/0.417 | 0.456/0.628 |
| Ca2SnTiO6 | 3 | 0.0789 | 0.146 | 0.593/0.311/0.095 | 0.555/0.457/-0.012 | 0.633/0.420 |
| Ca8Sn6Ti2O24 | 9 | 0.1115 | 0.258 | 0.606/0.225/0.169 | 0.551/0.341/0.108 | 0.590/0.438 |
| Cd(Ga,In)2O4 | 4 | 0.0047 | 0.035 | 0.724/0.131/0.145 | 0.712/0.268/0.020 | 0.678/0.285 |
| Na(Sc,In)O2 | 4 | 0.00003 | 0.001 | 0.848/0.048/0.104 | 0.671/0.143/0.186 | 0.811/0.320 |

Old posterior split: direct = H_bb^-1 f_b, coupled = -H_bb^-1 H_br u_r with u_r the R-part of the actual relaxation; fractions are projections on the actual u_b and do not sum to 1 because the actual u differs from H^-1 f (cos 0.99-0.9996; the remainder is in `three_term_split.json`, `residual_signed`). The exact-prediction direct/coupled (t1 vs t2+t3) reproduces `symmetric_sector_blocks.json` (k12: 0.285/0.715).

## Noisy CG (matvec = Hv + (sigma_F/0.01 A)|v| xi, xi ~ N(0,1) per component; 20 seeds; k = 1..20)

k_conv = smallest k after which s_u stays within 10% (relative) of the exact s_u through k = 20. Median over seeds / worst; "inf/None" = some or all seeds leave the band again before k=20 (noisy CG keeps drifting). Force calls = 2 per iteration. Last column: calls to build H (recorded jobs).

| system | cond H | cond S_b | sigma 0 | sigma 1e-4 | sigma 1e-3 | H build calls |
|---|---|---|---|---|---|---|
| k12 tight | 88 | 33 | 11 / 11 | 10.5 / None | inf / None | 337 |
| CaTiO3 | 97 | 8.5 | 6 / 6 | 7 / 7 | inf / None | 122 |
| CaSnO3 | 41 | 10.8 | 4 / 4 | 5 / 6 | inf / None | 122 |
| k08 | 13.9 | 5.4 | 6 / 6 | 6 / 7 | inf / None | 28 |
| Ca2SnTiO6 | 21.7 | 3.4 | 6 / 6 | 6 / 7 | inf / None | 26 |
| Ca8Sn6Ti2O24 | 36.5 | 8.0 | 7 / 7 | 7 / 8 | 7.5 / None | 56 |
| Cd(Ga,In)2O4 | 8.3 | 3.8 | 4 / 4 | 4 / 6 | inf / None | 28 |
| Na(Sc,In)O2 | 72.6 | 17.6 | 10 / 10 | inf / None | inf / None | 42 |

At sigma 0, k iterations cost 2k force calls (k12: 22 calls vs 337). Ionic steps of the actual relaxation are not recorded in any meta.json / result.json of these inputs (result.json holds only electronic-step counts, energy, max force). The per-k curves, enrichment eps(k) at k=20 and cos(u_k, H^-1 f) are in `krylov_cost.json`. Note the noise model follows the spec literally; physical central-difference noise is smaller by 1/sqrt(2). sigma 1e-3 is far above the band for most systems; its failure is noise drift of the iterate (cos at k=20 is 0.61 for k12, 0.96-1.0 elsewhere).
