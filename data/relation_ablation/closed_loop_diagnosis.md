# Decomposition of the band-edge switches

Synthesis of the path split (`substitution_path_split`), the relation analyses (`t1_t2_results.md`), the force-constant analyses (`bending_amplitude`) and the frozen-phonon and order tests of this folder (`t3/t3_t4_results.json`, `order_test/d1_result.json`). Margins in meV, competing point fixed at the end state of each step.

## Where each switch crosses zero

| Switch | start | after substitution or compression | after structural adjustment | zero crossed |
|---|---|---|---|---|
| substitution S05→S06, valence | −11.5 | +10.3 | +21.2 | at the substitution |
| substitution S11→S12, conduction | +25.9 | −135.9 | −107.4 | at the substitution; the adjustment pulls back +33 without changing the sign |
| substitution S04→S05, valence | −197.4 | −6.8 | −18.5 | no switch; the adjustment moves the margin from the threshold back to −18.5 |
| pressure P4→P5, conduction | −21.5 | −1.3 (compression) | +6.8 | in the internal relaxation (order dependence below) |

## Observations

| | observation | source |
|---|---|---|
| A | the relations are not more sensitive to bending than to stretching (T2b ratio 0.88) | `t1_t2_results.md` |
| B | along substitution, the network relaxation of the conduction margin is small in most steps (2/13 steps of opposite sign) | `substitution_path_split` |
| C | under pressure, internal relaxation and compression have the same sign (0/5 steps of opposite sign) | `substitution_path_split` |
| D | the Wannier decomposition of the electronic channels depends on the basis | `electronic_levels` |

## Interpretation

**A.** In the database the bond angle is an outcome of the local–network competition rather than an independent bending coordinate.
- Over the 442 structures, about 85% of the variance of the minimum B–X–B angle is explained linearly by the local octahedral geometry, and 52% by composition alone.
- The structures differ mainly by composition, and along the substitution path the composition-driven band-edge change is dominated by the substitution itself. The relations therefore describe differences between compositions and are insensitive to bending at fixed composition, consistent with the path decomposition. The conduction relation Π<sub>C</sub> leans towards stretching (ratio 0.22), consistent with B.

**B.** The conduction switch is driven by local chemistry; the network pulls back only near the switch (+33 and +21 meV at S11→S12 and S12→S13).

**C.** Under pressure the competition appears in the forces, not as a cancellation of band-edge contributions: the network forces are almost pure bond tension (less than 1% along bending), the bending modes are soft and coupled to bond-length motion, about one third of the actual displacement is bending, and φ = 0.82 for x = 0.75 at 0→5 GPa. The internal relaxation amplifies the compression and is itself mainly bending.

**D.** Basis-free read-outs are used instead: the path split and the mode-resolved frozen-phonon deformation potentials (T3).

## Order test of the pressure switch (D1)

| state | Λ<sub>C</sub> |
|---|---|
| P4 | −21.54 |
| P4 fractional coordinates in the P5 lattice (compression first) | −1.31 |
| P5 fractional coordinates in the P4 lattice (relaxation first, X) | −13.32 |
| P5 | +6.85 |

The internal-relaxation contribution is +8.22 meV in reverse order and +8.16 meV in forward order (difference 0.06 meV), so the split is additive. Neither compression alone (−1.3 meV) nor relaxation alone (−13.3 meV) crosses zero: the 4→5 GPa conduction switch requires both, compression about 71% (+20.2) and the mainly bending internal relaxation about 29% (+8.2).

## Frozen-phonon deformation potentials (T3) and thermal renormalisation (T4)

| structure | Λ<sub>C</sub> bending | Λ<sub>C</sub> stretching | Λ<sub>C</sub> ratio | Λ<sub>V</sub> ratio | 300 K Λ<sub>C</sub> renormalisation, \|bending\|/\|stretching\| | 300 K Λ<sub>C</sub> fluctuation σ, bending / stretching (meV) |
|---|---|---|---|---|---|---|
| x = 0.75 (k12, 5 GPa) | 227 | 36 | 6.4 | 2.3 | 6.8 | 9.0 / 0.5 |
| CaTiO<sub>3</sub> | 243 | 50 | 4.8 | 16.8 | 0.36 | 10.7 / 15.0 |
| CaSnO<sub>3</sub> | 6.0 | 1.9 | 3.2 | 0.15 | 2.4 | 2.4 / 2.7 |

Deformation potential per unit energy |∂²Λ/∂Q²|/ω², median over the six modes of each class. Per unit vibrational energy, bending modes move the conduction edge more than stretching modes in all three structures (ratio 3–6). What distinguishes x = 0.75 is softness: its bending modes lie at 24–29 cm⁻¹ with ω² about 1/15 of the stretching modes, against 140–200 cm⁻¹ in CaTiO<sub>3</sub> and 95–150 cm⁻¹ in CaSnO<sub>3</sub>, so the same energy gives a much larger bending amplitude. Limitations: 12 Γ modes of the supercell and no Brillouin-zone integral, so only the bending/stretching ratios are meaningful; the 300 K amplitudes of the soft x = 0.75 bending modes are 2.0–2.7 times the sampled displacement (quadratic extrapolation, anharmonicity not included); several CaTiO<sub>3</sub> modes change the best competing point (own-point and fixed-point read-outs differ by up to about 27 meV), and the fixed point is used.

## Summary
1. Substitution: local chemistry alone switches the band edge. The network adjustment pulls back, generally at the valence edge and near the switch at the conduction edge; it can return a state near the threshold (S04→S05) but does not undo a switch.
2. Pressure: compression provides about 70% and internal relaxation about 30% of the conduction switch; the relaxation is mainly bending (φ = 0.82), and both parts are required.
3. Lattice: bending enters because it is soft and coupled to bond-length motion, not because the force points along it.
4. Statistics: the relations need local and network information; the bond angle is the structural imprint of the competition and describes differences between compositions, not motion at fixed composition.
5. Dynamics: the thermal renormalisation of the switching (conduction) edge of x = 0.75 is dominated by the soft bending modes (6.8 times stretching), unlike the corner-sharing structures (0.36 and 2.4).
