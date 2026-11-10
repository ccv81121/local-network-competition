# La3Ni2O7: force balance along pressure and strain

## Construction

| Interval | Start → end (relaxed CONTCARs) | Internal displacement \|u\| |
|---|---|---|
| P1 | 0 GPa → 4 GPa | 0.236 Å |
| P2 | 4 GPa → 8 GPa | 0.252 Å |
| S1 | 0 → −1% biaxial | 0.102 Å |
| S2 | −1% → −2% biaxial | 0.087 Å |

- `ref`: r_aff = start fractional coordinates in the end lattice; its forces are the drive f. `end`: static calculation
  of the relaxed end structure (residual force, actual displacement u = r_end − r_aff).
- `sNN p/m`: ±0.01 Å per atom along each of the 10 totally symmetric internal modes (La 3, Ni 2, O 5), built from the
  16 Amam (Cmcm) operations found by spglib on r_aff; P2 also at ±0.02 Å (`x2`, linearity check). The actual
  displacements lie in this subspace (residual < 10⁻¹²). Modes, steps and u: `force_balance/modes.json`.
- Settings: PBE PAW (La, Ni_pv, O), ENCUT 520 eV, PREC Accurate, EDIFF 10⁻⁸, ISMEAR 1, SIGMA 0.05 eV, ISYM 0,
  Γ-centred 8×8×2, LREAL .FALSE., LASPH .TRUE.; ALGO Normal for `ref`, Fast for the displaced runs, which restart from the `ref` WAVECAR.
- Inputs: `force_balance_prepare.py`; job script `force_balance_job.sh`; forces in `force_balance/forces/`; analysis `force_balance_analyze.py`
  (→ `force_balance/force_balance_result.json`; `force_balance/force_balance_result_reference_only.json` uses the reference forces only) and
  `force_balance_geometric_factor.py` (geometric factor |u_b|/|u_r|).

## Predicted against actual relaxation (u_pred = H⁻¹f in the symmetric block)

| Interval | cos(u_pred, u) | Amplitude ratio | Geometric factor actual / predicted | Deviation | Harmonic energy predicted / actual (meV) |
|---|---|---|---|---|---|
| P1 | 0.9987 | 0.800 | 0.969 / 0.941 | −2.9% | −39.6 / −44.1 |
| P2 | 0.9990 | 0.733 | 1.159 / 1.158 | −0.2% | −27.8 / −33.0 |
| S1 | 0.9960 | 0.957 | 1.075 / 0.954 | −11.2% | −33.5 / −33.7 |
| S2 | 0.9922 | 0.983 | 1.104 / 0.883 | −20.0% | −27.1 / −27.0 |

- The direction is reproduced on all four intervals (cosine ≥ 0.99).
- Pressure path: the bond-preserving and bond-changing parts are underestimated by the same ratio (P1 0.787 and 0.810,
  P2 0.732 and 0.733), so the shape of the response is correct and only its amplitude is low. The softest symmetric
  eigenvalue drops from 1.718 (P1 start) to 1.341 eV/Å² (P2 start), a 22% softening on approach to the Fmmm transition
  (extrapolated to about 9.8 GPa). With the H of the P2 start, P1 gives amplitude ratio 1.015 and cosine 0.9993; with
  the mean of the two end-point H, 0.888 and 0.9999.
- Strain path: the bond-changing part is reproduced (1.01, 1.09), the bond-preserving part is underestimated by
  10–13%. Using the next interval's starting H leaves S1 at about −10% (that structure is not identical to the end
  structure, so this is an approximate test; the Newton step below is the direct one).
- Enrichment ε ≤ 1/s_f on all intervals: predicted ε 1.64, 1.27, 2.40, 2.41; actual 1.69, 1.28, 2.73, 3.05.
- Linearity (P2, ±0.02 Å): eigenvalues change by ≤ 0.5%; cosine between the two predicted displacements 0.9999996.
- Maximum residual force of the end structures: 0.0011–0.0019 eV/Å.

## Anharmonic residual

In the harmonic relation F_ref − H·u equals the end-structure force, which is about zero; the difference res gives the
error of the predicted displacement, u_pred − u = H⁻¹·res.

| Interval | \|res\|/\|F_ref\| | Secant / harmonic stiffness along u | Error / \|u\| | Error share in the two softest modes | cos(error, u) | Bond-preserving share of the error |
|---|---|---|---|---|---|---|
| P1 | 0.199 | 0.824 | 0.206 | 0.99 | −0.98 | 0.51 |
| P2 | 0.275 | 0.750 | 0.270 | 0.98 | −0.99 | 0.56 |
| S1 | 0.055 | 0.984 | 0.097 | 0.84 | −0.48 | 0.53 |
| S2 | 0.055 | 1.007 | 0.125 | 0.88 | −0.20 | 0.51 |

- Pressure: the error is antiparallel to u and lies in the two softest modes; the secant stiffness (0.82, 0.75 of the
  harmonic one) matches the amplitude ratios (0.80, 0.73). The amplitude deficit is the softening of the response
  direction on approach to the Fmmm transition.
- Strain: the stiffness along u is unchanged (0.98, 1.01), so the amplitude is correct. The residual (5.5% of the
  drive) lies mainly in soft modes nearly orthogonal to u, which amplify it to 10–13% of |u|, half of it bond
  preserving: third-order coupling tilts the response toward the soft modes.

## Modified Newton step

A second force calculation at r₁ = r_aff + u₁ gives u₂ = H⁻¹Q f(r₁) with the same H; u₁ + u₂ is compared with the actual
relaxation. Scripts `newton_step_prepare.py`, `newton_step_analyze.py`; result `force_balance/newton_step_result.json`.

| Interval | Residual at r₁ / starting force | Cosine u₁ → u₁+u₂ | Amplitude ratio u₁ → u₁+u₂ | Geometric-factor deviation u₁ → u₁+u₂ |
|---|---|---|---|---|
| P1 | 0.123 | 0.9987 → 0.9999 | 0.800 → 0.921 | −2.9% → −0.7% |
| P2 | 0.152 | 0.9990 → 0.9999 | 0.733 → 0.878 | −0.2% → −0.1% |
| S1 | 0.047 | 0.9960 → 0.9999 | 0.957 → 0.989 | −11.2% → −0.8% |
| S2 | 0.052 | 0.9922 → 0.9997 | 0.983 → 0.991 | −20.0% → −3.6% |

- Strain: one step removes the underestimate of the bond-preserving part (deviation within 4%, amplitude ratio about
  0.99), confirming its third-order origin.
- Pressure: the direction is exact (cosine 0.9999) and the amplitude ratio rises to 0.92 and 0.88; because the
  response softens along the path, a fixed-H iteration converges only gradually (a secant stiffness H̄ would be needed).
