# Valley shifts in the four states of the two spinels

Lowest conduction level at Γ and at the rival point, and the valence maximum at Γ, in the four states (00, 10, 01, 11) of Mg(Sc<sub>1−x</sub>Sb<sub>x</sub>)<sub>2</sub>Se<sub>4</sub>; no new calculation.

- `x075/state*/EIGENVAL`: x = 0.75, tight 4→5 GPa four states (server `PRL_TIGHT_SPINEL_20261015/work/Spinel/P4_P5/fourstate`), rival R = (½,½,½).
- `x05/state*/EIGENVAL`: ordered x = 0.5, 0→5 GPa four states (server `PRL_MECHANISM_TEST_20261014/work/Spinel_k08/P0_P5/fourstate`), rival (0,0,½).
- `valley_eig.py DIR k1 k2 k3` → `valley_shifts.txt`. Λ_C reproduces the archived four-state values (0.34, 6.17, 0.99, 6.78 meV at x = 0.75; 20.35 meV in state 11 at x = 0.5). All four states of each crystal share one lattice, so the level shifts between states are compared directly.

| Adding bending (00→10 / 01→11), meV | Γ valley | rival valley | Λ_C |
|---|---|---|---|
| x = 0.75 (rival R) | −4.6 / −4.6 | +1.2 / +1.3 | +5.8 / +5.8 |
| x = 0.5 (rival (0,0,½)) | −5.2 / −5.6 | −5.2 / −5.4 | +0.1 / +0.2 |

## Orbital character of the x = 0.5 valleys (run 2026-11-08)

`x05_procar/`: the four x = 0.5 states rerun as fixed-ion SCF with the archived INCAR plus `LORBIT = 11` (server `$WORK/spinel/K08_PROCAR_20261108`, one GPU each, 13 min; `run_k08procar_v1.sh`, `INCAR_state11`). Their eigenvalues reproduce the archived four states exactly. `procar_valley.py RUNDIR` → `valley_weights.json`: weights of the lowest conduction band at Γ and (0,0,½). PROCAR files stay on the server.

| x = 0.5, state 11 | Sc d | Mg s | Se d | Sb p |
|---|---|---|---|---|
| Γ valley | 0.78 | 0.08 | 0.07 | 0.01 |
| (0,0,½) valley | 0.81 | 0.05 | 0.05 | 0.03 |

Both competing valleys at x = 0.5 are Sc d (0.78–0.81 in all four states). At x = 0.75 the Γ valley is Sc d (0.66) and the R valley Sb and Se (0.57 + 0.34; `spinel_fatbands/fatband_summary.json`).

## Columns of the four-state table (x = 0.75)

- Λ_V^(z) = E_V(Γ) − E_V(0,0,1/6) from `x075/state*/EIGENVAL` (156 occupied bands): 36.015, 37.785, 35.311, 37.070 meV for states 00, 10, 01, 11.
- max|f_i| = largest per-atom force norm in the last force block of the state OUTCARs (server `PRL_TIGHT_SPINEL_20261015/work/Spinel/P4_P5/fourstate/state*/OUTCAR`, not deposited): 0.0896, 0.0807, 0.0544, 0.0012 eV/Å.
