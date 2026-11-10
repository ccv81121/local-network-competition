# Force-balance prediction of the alloy accommodation

Two intervals of the Mg(Sc<sub>1−x</sub>Sb<sub>x</sub>)<sub>2</sub>Se<sub>4</sub> substitution path (56-atom cell): S₀₅→S₀₆ (valence edge, `i0506`) and S₁₁→S₁₂ (conduction edge, `i1112`). At the affine state F (species of S<sub>b</sub>, fractional coordinates of S<sub>a</sub> in the S<sub>b</sub> lattice) the Hessian H is built from ±0.01 Å force differences on all 168 coordinates; u<sub>pred</sub> = H⁺f, with rigid translations projected out, predicts the internal relaxation F → S<sub>b</sub>.

u<sub>pred</sub> and the actual displacement u<sub>act</sub> are split into a bending part u<sub>b</sub> (kernel of the first-order Sc/Sb–Se bond Jacobian, six nearest Se per cation) and a complementary part u<sub>c</sub>. Electronic factors g<sub>b</sub>, g<sub>c</sub> come from statics at F ± 0.25|u<sub>i</sub>| along each part; the measured four states are 10 = F + u<sub>act,b</sub>, 01 = F + u<sub>act,c</sub>. Readout: ISYM = 0 statics on a Γ-centred 4×4×4 mesh, margins against the competing k point of the end structure, held fixed for all states.

- `al_result.json`: per interval, four-state and finite-difference margins and total energies, cos(u<sub>pred</sub>, u<sub>act</sub>) (`L1_cos`), predicted and measured ΔΛ, bending shares φ<sub>pred</sub>, φ<sub>meas</sub>, additivity residuals; `L1`–`L3` are comparisons with fixed tolerances (cos ≥ 0.90; |ΔΛ<sub>pred</sub> − ΔΛ<sub>meas</sub>| ≤ max(30%, 3 meV); |φ<sub>pred</sub> − φ<sub>meas</sub>| ≤ 0.15).
- `geom.json`: network, bond count and bending share of u<sub>act</sub> for `i0506`.
- `runs/{i0506,i1112}/`: H at F (`H.npy`, 168 × 168, eV Å⁻²) and the drive f (`f.npy`, eV Å⁻¹); u<sub>pred</sub> = H⁺f with the translations removed.
- Produced by `alloy_leverage.py` (`prepare`, `stage2`, `eval`).
- A diagonal second-order correction from the same finite-difference points gives ΔΛ = 10.65 and 32.73 meV (measured 11.01 and 33.25) but bending shares −0.44 and 0.46 (measured −0.25 and 0.35); the four-state cross terms (+5.71, −5.26 meV) are of the same size, so the first-order values are the ones reported.
