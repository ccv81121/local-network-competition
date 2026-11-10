# La3Ni2O7: four-state readouts

Four-state structures from `four_state_build.py` (`four_state/pbe/<I>/<state>/POSCAR`, `four_state/fourstate.json`):
00 = r_aff, 10 = r_aff + u_b (bond preserving), 01 = r_aff + u_r (bond changing), 11 = r_end, 11La = r_aff + u_b + u_r
with La at r_aff, 01ip / 01ap = only the in-plane / apical bond changes. Readout inputs from `four_state_prepare.py`:
PBE, Γ-centred 12×12×3, ISMEAR −5, LORBIT 11 (`runs/`); PBE+U (Dudarev, U = 4 eV on Ni d) for 00, 10, 01, 11
(`pbe_u4/`); 19×19×5 for P2 00 and 11 (`pbe_k19/`). Job script `four_state_job.sh`; Γ-point PROCAR and E_F extracts in
`four_state/readout/`; analysis `four_state_readout.py` → `four_state/readout.json`.

## Readout definition

- Near E_F the Γ-point d_z² bands (tetragonal M point folded to Γ) come in two pairs: one without O1 p_z (d_z² about
  0.62–0.64, O2 p_z about 0.12) is the interlayer **bonding** γ band; the other carries O1 p_z of about 0.12–0.14 and is
  the **antibonding** band, since O1 p_z couples only to the odd combination of the two layers and pushes it up.
- **R = top of the bonding pair − E_F.** Also recorded: mean and splitting of the bonding pair (the splitting comes from
  tilt folding) and the bonding–antibonding splitting (measure of the interlayer hopping t⊥). On the pressure path the
  bonding top lies above E_F.
- The identification is the same (bands 215–218) in all 40 readouts: PBE and PBE+U, pressure and strain, k12 and k19.
- k-point convergence: R(11) − R(00) of P2 is −13.2 meV at 12×12×3 and −12.6 meV at 19×19×5.

## Four-state readouts (R, meV)

| | 00 | 10 | 01 | 11 | 11La | 01ip | 01ap | Internal total | Δ_b | Δ_c | Cross | φ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 PBE | +73.5 | +60.6 | +80.6 | +67.8 | +67.2 | +72.8 | +81.8 | −5.7 | −12.9 | +7.1 | +0.1 | +226% |
| P2 PBE | +69.8 | +52.8 | +73.8 | +56.6 | +56.9 | +68.8 | +75.0 | −13.2 | −17.1 | +4.0 | −0.1 | +130% |
| S1 PBE | +30.4 | +24.5 | +21.6 | +12.6 | +16.1 | +30.4 | +21.5 | −17.8 | −5.9 | −8.8 | −3.2 | +42% |
| S2 PBE | −80.8 | −87.8 | −100.3 | −115.6 | −107.3 | −82.3 | −98.7 | −34.9 | −7.1 | −19.6 | −8.2 | +32% |
| P1 +U | +2.7 | −23.3 | +26.9 | +16.3 | | | | +13.6 | −25.9 | +24.3 | +15.3 | −134% |
| P2 +U | +18.5 | −5.6 | +32.4 | +15.6 | | | | −2.9 | −24.1 | +13.8 | +7.4 | +706% |
| S1 +U | −156.0 | −171.2 | −179.5 | −205.6 | | | | −49.6 | −15.2 | −23.5 | −10.9 | +42% |
| S2 +U | −335.3 | −344.2 | −356.5 | −375.6 | | | | −40.4 | −9.0 | −21.3 | −10.1 | +35% |

φ is the bond-preserving share averaged over the two orders. φ > 100% means the two channels act in opposite
directions with the bond-preserving one larger than the total; for P2 +U the total is only −2.9 meV, so φ is not
meaningful there.

Bonding–antibonding splitting (meV), 00 → 10 → 01 → 11: P1 221 → 248 → 166 → 183; P2 188 → 225 → 147 → 178;
S1 271 → 288 → 311 → 338; S2 381 → 389 → 419 → 437. The bond-preserving motion increases the splitting (t⊥) on all four
intervals; the bond-changing motion decreases it under pressure and increases it under strain.

Affine step (00 of the interval minus 11 of the previous one): P2 PBE +69.8 − 67.8 = +2.0 meV, P2 +U +2.2 meV;
S2 PBE −80.8 − 12.6 = −93.4 meV. P1 and S1 start from the 0 GPa structure (`affine_step.py`, see `wannier_result.md`).

## E_F and band top separately (meV, relative to 00)

| | ΔE_F | Δ(band top, absolute) | Δ(band top − E_F) |
|---|---|---|---|
| P1 01ip (in-plane bonds only) | +23.7 | +23.0 | −0.7 |
| P2 01ip | +19.9 | +18.9 | −1.0 |
| P1 01ap (apical bonds only) | +46.0 | +54.3 | +8.3 |
| P2 01ap | +34.6 | +39.8 | +5.2 |

The absolute eigenvalue reference of each VASP run shifts with the structure, so only the difference is physical; the
comparison shows whether band top and E_F move together.

## Reading

1. The in-plane bonds shift the whole e_g manifold rigidly: band top and E_F (set by the in-plane d_x²−y² band) move
   together and R is unchanged. R measures the competition between the d_z² bonding level and the d_x²−y² sector; the
   in-plane bonds act on both, the apical bonds and the interlayer angle act on d_z² only.
2. At equal a_p the two drives differ. Pressure contracts both the plane and c; biaxial strain contracts the plane and
   expands c, so the apical bonds change in opposite directions in the affine step (P2 +2.0, S2 −93.4 meV). The drive
   felt by the d_z² readout is c and the apical bond, not a_p.
3. Under pressure the two channels oppose each other: straightening (bond preserving) increases t⊥ and lowers the
   bonding band, the shorter apical bond raises d_z² through the crystal field; φ = 226% and 130%. Under strain the
   shorter inner apical bond also increases t⊥, the channels act together and φ = 42% and 32%.
4. PBE+U changes the electronic factors, not the geometric ones: Δ_b and Δ_c grow with U (−12.9 → −25.9, +7.1 → +24.3
   meV on P1) on identical structures, and the net result and the shares change with them. The sign of each channel is
   the same as in PBE on P1, P2 and S1, while the net internal change of P1 changes sign (−5.7 → +13.6 meV).
5. On the strain path the internal relaxation lowers the band top (−17.8, −34.9 meV); the 4 GPa structure (P1/11,
   +67.8 meV) lies 55 meV above the −1% strained one (S1/11, +12.6 meV), mostly through the affine step.

All readouts refer to the Amam phase, non-magnetic PBE, with E_F as reference (E_F itself moves by 0–0.06 eV between
states).
