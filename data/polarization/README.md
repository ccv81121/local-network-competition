# Polarization in the four-state construction

Four-state construction `k12_t45` of `four_state_leverage` (x = 0.75 spinel, 4 → 5 GPa, 5-GPa lattice, 56 atoms): states 00, 10 (+ bending), 01 (+ complementary), 11, the affine reference r<sub>aff</sub>, and finite-difference points bp, bm, cp, cm (±δ along u<sub>b</sub>, u<sub>c</sub>; δ<sub>b</sub> = 0.0069 Å, δ<sub>c</sub> = 0.0111 Å). The finite-difference points give the force-balance prediction φ<sub>pred</sub>; the states give the share φ = bend/(X₁₁ − X₀₀) and φ′ = bend/(X₁₁ − X<sub>raff</sub>), with bend = ½[(X₁₀ − X₀₀) + (X₁₁ − X₀₁)].

- Structures: `berry_phase/poscar_*`.
- Dipoles (VASP Berry phase): `berry_phase/dipoles_states.txt`, `berry_phase/dipoles_fd.txt`. The ionic part is recomputed on one branch from ZVAL-weighted positions.
- Results: channel derivatives `berry_phase/polarization_channel_derivatives.json`; projections on ê = (P₁₁ − P₀₀)/|P₁₁ − P₀₀| and shares `berry_phase/polarization_result.json`.
- Script: `scripts/polarization_shares.py` (`predict` for the finite-difference points, `measure` for the states).
