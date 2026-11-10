# La3Ni2O7: Wannier electronic factors, interlayer coupling and the drive-plane grid

## Wannier Hamiltonians

- `wannier_prepare.py` writes `wannier/calculations/<interval>/<state>` for the states 00, 10, 01, 01ip, 01ap and 11 of every
  interval, plus `calculations/p0start/00` (0 GPa structure). PBE, ISMEAR 1, SIGMA 0.05 eV, Γ-centred 6×6×2 full mesh (ISYM −1),
  NBANDS 288, projected Wannier functions from VASP's wannier90 interface: Ni d + O p, 124 functions, one projection
  per atom at its own position, z along c; outer window E_F − 10 to E_F + 4 eV, frozen window E_F − 10 to E_F + 0.5 eV.
  `basis_map.json` in each directory records orbital order and O class (O1 inner apical, O2 outer apical, in-plane O).
- `readout/p0start/{pbe,u4}`: 12×12×3 readout of the 0 GPa structure, baseline of the affine step of P1 and S1.
- Decomposition (`wannier_decomposition.py`, `wannier_decomposition_fixed_n.py`): between two states H(λ) = H_A + λΔH, integrated with
  12 Gauss–Legendre points; ΔH is split into matrix-element classes. The band-top part ⟨top|ΔH(Γ)|top⟩ is exact at Γ
  and agrees with the VASP band-top change within 1.5 meV in 24 comparisons. The E_F part is taken from the four-state VASP
  readouts (12×12×3, tetrahedron); the model E_F is sensitive to mesh and smearing near the flat d_z² pocket at Γ on the
  pressure path (4–6 meV) and is not used. Electron count N = 228; Wigner–Seitz images are folded into the R list.
- `wannier/wannier_table.md` (from `wannier_table.py`) tabulates the decomposition of `wannier/wannier3_<interval>.json`,
  in which part of the nearest-neighbour Ni–O hopping stored at the far c image is counted as "Ni–O farther". The class
  attribution below assigns every hopping to its nearest image by atom position (`wannier_decomposition_fixed_n.py`); class totals are
  identical in both.

## Class attribution of R (atom-wrapped Wannier positions, nearest image)

Four-state averages, meV (Δb = (10−00 + 11−01)/2, Δc = (01−00 + 11−10)/2); classes of the band top minus E_F at Γ.
Model totals reproduce the path ends to < 0.01 meV; mesh check 0.017 meV, Γ check 0.013 meV.

| Interval | Move | On d_z² | On d other | On O2 | Ni–O1 | Ni–O2 | Ni–O in-plane | Ni–O farther | O–O | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 | Δb | +0.5 | +1.7 | +2.9 | −6.7 | −2.3 | +1.2 | −2.0 | −7.3 | −12.5 |
| P1 | Δc | +4.2 | −6.9 | −5.5 | −0.2 | +10.0 | +6.6 | +4.1 | −4.6 | +9.5 |
| P2 | Δb | +1.6 | +2.5 | +1.5 | −11.3 | −2.3 | +1.6 | −2.0 | −8.3 | −18.5 |
| P2 | Δc | +3.8 | −7.5 | −3.5 | −0.7 | +8.1 | +6.4 | +4.6 | −3.6 | +8.9 |
| S1 | Δb | +1.3 | 0.0 | −0.8 | −3.6 | −0.7 | +0.5 | 0.0 | −2.2 | −6.1 |
| S1 | Δc | 0.0 | +0.2 | +3.5 | −1.6 | −6.8 | −1.4 | +0.9 | −1.6 | −7.1 |
| S2 | Δb | +1.9 | −2.2 | −1.7 | −7.7 | −1.3 | +1.0 | +0.6 | −1.7 | −12.6 |
| S2 | Δc | −1.0 | −1.2 | +7.4 | −10.0 | −15.5 | −6.8 | −3.1 | −2.2 | −28.0 |

- The "Ni–O farther" class is ≤ 2 meV. The straightening move (Δb) lowers the band top through the inner apical Ni–O1
  channel and O–O hopping, the channel that carries t⊥.
- The internal c move (Δc) acts through the outer apical Ni–O2 hopping with the sign of the Ni–O2 length change (+8 to
  +10 meV under pressure, −7 / −16 meV under strain), joined by the in-plane Ni–O hopping. In-plane-only part of the c
  move (01ip − 00): −0.5, +1.2, −0.1, −2.5 meV.
- In-plane bonds only (01ip, pressure): band top +24.4 and +19.1 meV, E_F +23.7 and +19.9 meV, R −0.7 and −1.0 meV; the
  shift comes mostly from the overall reference and onsite energies, a rigid shift of the e_g manifold.
- Bonding–antibonding splitting (model, meV): the bond-preserving move increases it on all intervals (P1 365 → 390,
  P2 350 → 384, S1 380 → 396, S2 381 → 390); the bond-changing move decreases it under pressure and increases it under
  strain, as in the four-state readouts.
- Affine step (`affine_step.py` → `wannier/affine.json`): VASP R of the 0 GPa start +65.6 meV (PBE), −24.1 meV
  (U = 4 eV). Wannier affine step P1 +8.7, S1 −34.0 meV against VASP +7.9, −35.2 meV.

## Full chain: total = affine step + internal relaxation (Δb + Δc + cross term), R in meV

| Path | Start R | Affine | Δb | Δc | End R | Total |
|---|---|---|---|---|---|---|
| PBE P1 (0 → P4) | +65.6 | +7.9 | −12.8 | +7.1 | +67.8 | +2.2 |
| PBE P2 | +67.8 | +2.0 | −17.1 | +3.9 | +56.6 | −11.2 |
| PBE S1 (0 → −1%) | +65.6 | −35.2 | −7.4 | −10.3 | +12.6 | −53.0 |
| PBE S2 (−1% → −2%) | +12.6 | −93.4 | −11.2 | −23.7 | −115.6 | −128.2 |
| U4 P1 | −24.1 | +26.8 | −18.3 | +31.9 | +16.3 | +40.4 |
| U4 P2 | +16.3 | +2.2 | −20.4 | +17.5 | +15.6 | −0.7 |
| U4 S1 | −24.1 | −131.9 | −20.6 | −28.9 | −205.6 | −181.5 |
| U4 S2 | −205.6 | −129.7 | −14.0 | −26.4 | −375.6 | −170.0 |

Cross terms are not listed separately; Δb and Δc are averages over the two orders.

Geometry (Å, °):

| State | a_p | c | Ni–O in-plane | Ni–O1 | Ni–O2 | ∠c (Ni–O1–Ni) | ∠ab |
|---|---|---|---|---|---|---|---|
| 0 GPa | 3.852 | 20.38 | 1.935 | 1.969 | 2.241 | 165.0 | 168.9 |
| P1/11 | 3.823 | 20.14 | 1.917 | 1.955 | 2.185 | 167.3 | 171.1 |
| P2/11 | 3.793 | 19.98 | 1.900 | 1.944 | 2.150 | 170.4 | 173.5 |
| S1/11 | 3.814 | 20.64 | 1.916 | 1.973 | 2.298 | 166.9 | 168.9 |
| S2/11 | 3.775 | 20.89 | 1.898 | 1.979 | 2.348 | 168.2 | 168.5 |

Wannier matrix elements (eV):

| State | t(d_z²–O1 p_z) | t(d_z²–O2 p_z) | ε(d_z²) − ε(O1 p_z) |
|---|---|---|---|
| 0 GPa | 0.709 | 0.992 | 2.761 |
| P1/11 | 0.744 | 1.092 | 2.793 |
| P2/11 | 0.778 | 1.169 | 2.823 |
| S1/00 → S1/11 | 0.691 → 0.717 | 0.974 → 0.947 | 2.732 → 2.805 |
| S2/11 | 0.719 | 0.911 | 2.852 |

The outer apical t(d_z²–O2 p_z) rises by 18% under pressure and falls by 8% under strain, following the Ni–O2 length
(2.24 → 2.15 Å and 2.24 → 2.35 Å). A real-space ratio t₁²/Δ is not a reliable interlayer measure here (c-axis aliasing
of the O functions, see below); t⊥ replaces it.

## Tilt check (`run_film_tilt.sh`, `run_tilt_readout.sh`; structures in `film_tilt/`, readout with `tilt_readout.py`)

Energies (non-magnetic PBE, eV/cell):
- −2% cell: tilted (strain/eta-2) −358.499, tilt-free (s2_fmmm, c relaxed to 20.73 Å) −358.315; tilted lower by 184 meV.
- Experimental film lattice (a_p 3.756 Å, c 20.72 Å; Wang et al., PRX 16, 031008): the tilted start relaxes to an
  interlayer angle of 171.0° (film_amam), −358.203; tilt-free (film_fmmm) −358.138; tilted lower by 65 meV.
- In PBE the tilt persists in the film, but the tilt direction is soft (film_amam moves from 168° to 171° during the
  relaxation). The c of the relaxed tilt-free structure agrees with the experimental c.

Readout R (meV, PBE / U4):

| Lattice | Tilted | Tilt-free |
|---|---|---|
| Experimental film | −80.0 / −344.7 (171°) | −59.7 / −329.4 |
| −2% (c free) | −115.6 / −375.6 (168°) | −31.1 / −287.4 |

On the experimental lattice the tilt changes R by only 15–20 meV; in the c-free −2% cell by 85 meV, because removing the
tilt there also shortens c and the outer apical bond (Ni–O2 2.348 → 2.291 Å).

## Interlayer coupling t⊥ (`interlayer_hopping.py`)

- VASP's wannier90 interface places every Wannier function at its atom's fractional position wrapped to [−0.5, 0.5).
  With two k points along c, hoppings across the c boundary are split equally between two images (ndegen = 2), so
  real-space O hoppings along c are not unique.
- t⊥: on every 6×6×2 k point all O p orbitals are folded out at E_F (Löwdin; the top of the O p bands lies 1.7–2.0 eV
  below E_F, so the self-energy is regular); the Ni d block is projected on the local d_z² of each Ni along its own
  Ni–O1 bond; after Fourier inversion t⊥ is the effective hopping between the two Ni of one bilayer. It is independent
  of the coordinate orientation and of the O function positions; the trends are the same at E_F ± 0.5 eV.

t⊥ (eV, at E_F): 0 GPa 0.975; 8 GPa (P2/11) 1.082 (+11.0%); −2% tilted (S2/11) 0.960 (−1.5%); −2% tilt-free 1.088
(+11.6%); experimental film tilted (171°) 1.027 (+5.4%); experimental film tilt-free 1.091 (+11.9%).

Four-state attribution of t⊥ (meV):

| Interval | Affine | Δb | Δc | Internal relaxation | Bond-preserving share |
|---|---|---|---|---|---|
| P1 | +28.2 | +25.9 | −2.5 | +23.3 | 111% |
| P2 | +21.2 | +37.3 | −2.7 | +34.6 | 108% |
| S1 | −22.1 | +20.1 | +18.3 | +38.4 | 52% |
| S2 | −46.0 | +9.0 | +6.2 | +15.2 | 59% |

- The interlayer coupling is carried by the bond-preserving straightening; under pressure its share exceeds 100% and
  the bond-changing channel is slightly negative.
- The affine step follows the drive: under pressure c shortens and the coupling grows; under strain c expands, the inner
  apical bond lengthens and the coupling weakens.
- The same bond-preserving motion gives φ = 226% and 130% for the γ band (opposite to the apical channel) and a share of
  about 110% (dominant) for t⊥: the readout is set by the character of the state.

## Relation to the leverage expression

In φ/(1−φ) = (g_b/g_n)(|u_b|/|u_n|), g_b comes from t⊥ and the inter-octahedral hopping and is negative on all four
intervals; g_n comes from the outer apical Ni–O2 σ hopping (with the d_z² onsite energy) and changes sign with the drive,
since pressure shortens and strain lengthens Ni–O2. This is the matrix-element origin of opposing channels under
pressure and cooperating channels under strain.

## Drive-plane grid (`run_grid.sh`; results in `drive_plane_grid/`)

Grid: a_p in {3.8522, 3.804, 3.756} Å × c in {19.926, 20.376, 20.826} Å, b/a = 1.018, ions relaxed from the Amam 0 GPa
fractions (A0C1 = 0 GPa). Affine states: 0 GPa fractions in each cell, readouts only. Reference R = 65.6 / −24.1 meV
(PBE / U4), t⊥ = 0.9745 eV. A2C1 was rerun to full convergence (A2C1r: R 11.2 vs 11.6 meV, t⊥ 1.0783 vs 1.0766 eV);
A2C0 converged to f_max 0.0095 eV/Å. A0C0 PBE: bonding and antibonding levels interleave at Γ, R is the upper bonding
level. The table is `drive_plane_grid/attribution.txt`; affine t⊥ values are in `drive_plane_grid/tperp_aff.txt`.

| Point | ΔR PBE = affine + internal | ΔR U4 = affine + internal | Δt⊥ % = affine + internal | Ni–O1–Ni |
|---|---|---|---|---|
| A0C0 (c −2.2%) | +43.0 = +33.5 + 9.5 | +102.1 = +86.4 + 15.7 | +2.9 = +5.3 − 2.4 | 164.8 |
| A0C2 (c +2.2%) | −62.4 = −34.2 − 28.2 | −198.1 = −126.5 − 71.6 | −3.5 = −4.6 + 1.1 | 164.8 |
| A1C1 (a_p −1.25%) | −25.5 = −15.0 − 10.5 | −84.4 = −68.0 − 16.4 | +5.1 = +0.5 + 4.6 | 167.9 |
| A2C1 (a_p −2.5%) | −54.0 = −38.4 − 15.6 | −171.6 = −143.2 − 28.4 | +10.5 = +1.2 + 9.3 | 170.9 |
| A1C0 | +11.2 = +19.9 − 8.7 | +64.3 = +54.0 + 10.3 | +9.0 = +5.7 + 3.2 | 168.2 |
| A1C2 | −118.3 = −59.3 − 59.0 | −276.8 = −198.5 − 78.3 | −0.9 = −4.6 + 3.7 | 167.0 |
| A2C0 | −23.3 = +5.6 − 28.9 | +9.8 = −1.0 + 10.8 | +15.3 = +6.3 + 9.0 | 173.1 |
| A2C2 | −184.4 = −117.1 − 67.3 | −358.4 = −268.1 − 90.3 | +0.8 = −5.5 + 6.3 | 169.2 |
| Film lattice (b/a 1) | −145.6 = −97.2 − 48.4 | −320.6 = −247.8 − 72.8 | +5.4 = −4.0 + 9.4 | 171.0 |

b/a check (b/a 1.018 against the independently relaxed states): film lattice R −81.1 vs −80.0 meV (PBE), −341.2 vs
−344.7 meV (U4), t⊥ 1.0097 vs 1.0271 eV; P2 lattice R 64.4 vs 56.6 and 20.1 vs 15.6 meV, t⊥ 1.0703 vs 1.0818 eV. The
grid reproduces the independent states within 1–2 points in t⊥, within 1 meV in R on the strain side and about 10 meV
on the pressure side (the b/a part).

The γ band follows the affine shape of the octahedron (60–85% of ΔR), with internal relaxation of the same sign. The
affine part of t⊥ follows c (+5% for c −2.2%, −4.6% for c +2.2%) and its internal part follows the straightening (+9% at
171–173°, +4.6% at 168°, about 0 at 165°). On the film lattice the affine −4.0% (c expansion) is overturned by the
straightening +9.4%; under pressure both parts increase t⊥.
