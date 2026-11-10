# Epitaxial biaxial strain: results

ε is the in-plane biaxial strain (compression negative). Λ in meV; E_el is the elastic energy relative to ε = 0 in meV/atom. Raw results: `strain_result.json`, `fits.json`, `strain_supp_result.json`.

**k12, MgSc<sub>0.5</sub>Sb<sub>1.5</sub>Se<sub>4</sub>** (in-plane a<sub>0</sub> = 11.617 Å)

| ε (%) | −3 | −2 | −1 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| Λ_C | −140.2 | −122.2 | −114.8 | −112.3 | −120.0 | −131.5 | −148.7 |
| Λ_V | 27.3 | 33.7 | 42.8 | 54.7 | 36.5 | 17.2 | −9.2 |
| c*/c<sub>0</sub> | 1.036 | 1.025 | 1.013 | 1 | 0.987 | 0.972 | 0.956 |
| E_el | 8.8 | 3.8 | 0.9 | 0 | 0.9 | 3.5 | 7.6 |

**S11, MgSc<sub>0.625</sub>Sb<sub>1.375</sub>Se<sub>4</sub>** (in-plane a<sub>0</sub> = 11.540 Å)

| ε (%) | −3 | −2 | −1 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| Λ_C | −34.8 | −10.2 | +5.6 | +28.2 | +14.9 | +10.3 | −0.9 |
| Λ_V | 42.6 | 48.5 | 54.6 | 65.8 | 50.9 | 26.5 | −1.1 |
| c*/c<sub>0</sub> | 1.043 | 1.032 | 1.020 | 1 | 0.994 | 0.980 | 0.965 |
| E_el | 11.2 | 5.3 | 1.5 | 0 | −0.2 | 1.7 | 5.2 |

## Out-of-plane relaxation
All 12 strained c* lie inside the scanned range; five points were extended once beyond the scan edge (k12 −3, +2, +3%; S11 −3, −2, −1%). Two minima lie at the range edge (S11 +3%: c* = 0.96467 against the lower bound 0.96446; k12 +3%: the minimum is the extension point), so their c* is an extrapolation.

## Conduction-valley switching
- k12: strain of either sign makes Λ_C more negative, i.e. moves the conduction valley away from Γ. Λ_C has its maximum (≈ −112 meV) near ε = 0, with slopes −2.5 meV/% (compressive) and −7.7 meV/% (tensile); no strain reaches a crossing.
- S11 switches on both sides, in both cases from Γ to the non-Γ valley (direct to indirect gap):
  - compressive: ε_c = −1.35% (interpolated −1.3548%), in-plane a = 11.38 Å (mean in-plane lattice vector 11.5398 Å);
  - tensile: ε_c = +2.92%, a = 11.876 Å.
- `bracket()` in `strain.py` scans the tensile side first and returns the first sign change, so `strain_result.json` records only the tensile crossing (`S1_accessible: false`); the compressive crossing at |ε_c| < 2% is the one given above and in `strain_supp_result.json` (`e_c_neg`).
- The S11 ε = 0 structure is not at the in-plane energy minimum of these settings (E_el(+1%) = −0.2 meV/atom; quadratic minimum at ε ≈ +0.5%). Measured from that minimum the two critical strains are about −1.9% and +2.4%. For k12 the offset is +0.04%.
- Λ_V also switches under tension (valence maximum leaves Γ): k12 at +2.65%, S11 at +2.96%.

## Mechanism of the two S11 crossings

| Step | Enrichment (drive → realized share) | φ | Four-state Λ_C: s00 / s10 / s01 / s11 (meV) | Affine step | Internal relaxation |
|---|---|---|---|---|---|
| Compressive −1 → −2% | 17.7 (0.19% → 3.33%) | 0.13 | −25.9 / −23.7 / −12.0 / −10.2 | +5.6 → −25.9 (−31.5) | +15.6, opposite to the switch |
| Tensile +2 → +3% | 8.9 (0.18% → 1.60%) | −0.50 | −5.3 / −7.4 / +1.4 / −0.9 | +10.3 → −5.3 (−15.6) | +4.4, opposite to the switch |

- In both steps the affine (lattice) step alone carries the conduction valley through zero; the internal network relaxation pulls back but does not restore the sign.
- Compressive step: the network motion is still strongly enriched in the bond-preserving direction (17.7, within the 9–60 range of edge-sharing structures under pressure), but bending carries only 13% of the pull-back.
- Tensile step: additivity residual −0.3 meV; of the +4.4 meV pull-back, the complementary part gives +6.6 and bending −2.2 meV.
- Under hydrostatic pressure, by contrast, the internal relaxation acts with the lattice, carries 21–45% of the change, and bending provides 90% of it.

6×6×6 readouts (same INCAR as the 4×4×4 readout):

| Strain point | Λ_C, 4×4×4 → 6×6×6 (meV) | Λ_V, 4×4×4 → 6×6×6 (meV) |
|---|---|---|
| S11 −1% | +5.6 → +5.6 | 54.6 → 27.9 |
| S11 +3% | −0.9 → −1.0 | −1.1 → −10.9 |

Λ_C is mesh-converged and the two critical strains are unchanged. Λ_V decreases on 6×6×6 because that mesh contains k points at 1/6 along c*, where the valence maximum is higher; the 4×4×4 Λ_V values are upper bounds, and the valence switch occurs at smaller tensile strain. Readouts use ISMEAR = 0, SIGMA = 0.05, as in the path-split readouts.

## Γ and R level shifts
The non-Γ conduction minimum is R = (½,½,½) at every strain point of both systems, a single k point; the next conduction band at R lies 73–183 meV higher, so there is no degenerate valley that strain could split. The shifts of Γ and R were separated into a volume term and a shape (tetragonal) term, with the k-averaged Sc 3p semicore group (≈ −25 eV; 15 bands for S11) as reference and the volume term from a quadratic fit in ln V along the k12 pressure path P0–P6 (Sc 3s as reference gives the same split; the reference cancels in ΔΛ_C). Values in meV relative to ε = 0 of each system; cell volumes in `readout_eig/vols.txt`.

| System, ε | Δln V | ΔΛ_C | Volume term | Shape term | ΔE_C(Γ) actual / volume prediction | Shape term at R |
|---|---|---|---|---|---|---|
| k12 −3% | −0.025 | −27.8 | +25.8 | −53.6 | −37.0 / −36.3 | −54.3 |
| k12 −1% | −0.007 | −2.5 | +6.5 | −9.0 | −10.1 / −9.4 | −9.7 |
| k12 +1% | +0.006 | −7.7 | −5.1 | −2.6 | +6.3 / +7.6 | −3.9 |
| k12 +3% | +0.014 | −36.3 | −10.4 | −25.9 | +10.2 / +15.7 | −31.4 |
| S11 −3% | −0.019 | −63.0 | +18.7 | −81.8 | −26.6 / −26.6 | −81.8 |
| S11 −2% | −0.009 | −38.4 | +8.5 | −46.9 | −11.9 / −12.2 | −46.6 |
| S11 −1% | −0.001 | −22.6 | +0.4 | −23.0 | +1.4 / −0.6 | −21.1 |

- Under compression the Γ level follows the volume to within 2 meV in both systems; the shape term falls almost entirely on the R level and is negative.
- The S11 switching step −1 → −2% has volume term +8.1, shape term −23.9 and ΔΛ_C = −15.8 meV: the switch is the action of the shape change on the R valley.
- Along the pressure path P0 → P6 (Δln V = −0.098) Γ falls by 195 meV and R by only 47 meV; the pressure-induced reordering is the volume-driven descent of the Γ valley, with zero shape term.
- The S11 rows use the k12 volume coefficients; the S11 ε = 0 structure is offset by about 0.5% from the in-plane minimum (its volume nearly equals that of the −1% point). On the S11 tensile side Γ rises by about twice the volume prediction, so the volume picture is applied to the compressive side only.

## Summary
Epitaxial biaxial strain moves the conduction valley away from Γ for either sign: k12 (valley at R) moves further from the crossing, and S11 (valley at Γ) returns to the non-Γ valley at −1.4% compression or +2.9% tension. A direct gap requires isotropic compression; biaxial and hydrostatic drives that shorten the Se–Se distances alike give opposite read-outs. At 300 K vibrations lower Λ_C by about 69 meV (5 GPa, `../thermal_valley/`), so the conduction valley of unstrained S11 may already lie away from Γ at room temperature; the values here are static.
