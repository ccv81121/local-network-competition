# Zero-point and thermal averages of the conduction margin: results

33 statics (ZG and Monte Carlo configurations at the tight 4- and 5-GPa structures); data in `thermal_result.json` and `runs/`. The amp0.9_T300_P5_m configuration computed on both machines gives dΛ_C = dΛ_V = 0.000 meV.

| Quantity | Value (meV) |
|---|---|
| Static Λ_C, 4 / 5 GPa | −21.54 / +6.85 (static P_c = 4.76 GPa) |
| ⟨Λ_C⟩ ZG, 0 K, 4 / 5 GPa | −36.41 / −10.81 |
| ⟨Λ_C⟩ ZG, 150 K, 4 / 5 GPa | −55.59 / −32.02 |
| ⟨Λ_C⟩ ZG, 300 K, 4 / 5 GPa | −82.13 / −61.90 |
| δΛ_C(300 K, 5 GPa), ZG | −68.74 |
| δΛ_C(300 K, 5 GPa), MC (4 antithetic pairs) | −76.44 ± 12.12 (SE) |
| δΛ_C 300 K, 5 GPa: bend only / stretch only / all | −17.95 / −19.27 / −68.74 |
| δΛ_C 300 K, 4 GPa: bend only / stretch only / all | −18.24 / −15.11 / −60.59 |
| Amplitude ×0.9 / ×1.0 / ×1.1 (5 GPa, 300 K) | −56.28 / −68.74 / −81.92 (∝ amplitude², quadratic regime) |

- ZG and MC agree: |ZG − MC| = 7.70 meV against 2 SE = 24.25 meV.
- ⟨Λ_C⟩ < 0 at 5 GPa for 0, 150 and 300 K, so the conduction-band crossing lies above 5 GPa at every temperature; zero-point motion alone moves Λ_C(5 GPa) from +6.85 to −10.81 meV.
- Vibrations lower E_C(R) relative to E_C(Γ), i.e. they oppose pressure, and the reordering pressure rises with temperature.
- At 5 GPa bend + stretch (−37.2 meV) differs from all modes (−68.7 meV). Single ZG configurations of a mode subset carry sampling noise of the order of the MC pair spread (pair means −45 to −103 meV), so this non-additivity is not resolved.
