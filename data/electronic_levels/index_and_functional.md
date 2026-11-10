# Redirection index, electronic partition and functional dependence

All systems are compressed 0→5 GPa (PBE unless stated); four-state readouts by `phi_readout.py`, symmetric-sector force balance by `sector_force_balance_analyze.py`.

## Definitions
- **Redirection index ρ**: bond-preserving share of the predicted response H⁻¹f divided by the bond-preserving share of the drive f. The bond-preserving subspace is the kernel of the first-order B–X bond Jacobian at the fixed 5-GPa lattice; H is the symmetric-sector Hessian at r<sub>aff</sub> (the path keeps the space group, so this equals the full-Hessian result).
- **Electronic partition φ** = |ΔΛ_perp| / (|ΔΛ_perp| + |ΔΛ_par|), with ΔΛ_perp = Λ(state10) − Λ(state00) and ΔΛ_par = Λ(state01) − Λ(state00) from the four-state construction.
- **Primary margin Λ**: at state11, edge and rival are the best and second-best of the eight TRIM points; of VB and CB, the edge with the smaller |Λ| is used. For the spinel this is Λ_C = E_C(½½½) − E_C(Γ).
- **φ_all**: the same ratio with the Euclidean norm of the change of all VB energies (relative to the state00 VBM k point) and all CB energies (relative to the state00 CBM k point) on the edge mesh.

Systems: spinel k12 (x = 0.75), spinel k08 (28 atoms, P-4m2), rock-salt-ordered Ca<sub>2</sub>SnTiO<sub>6</sub> (20 atoms, P2<sub>1</sub>/c), Ca<sub>8</sub>Sn<sub>6</sub>Ti<sub>2</sub>O<sub>24</sub> (40 atoms, P2<sub>1</sub>/c), CaTiO<sub>3</sub>, CaSnO<sub>3</sub>. ρ of the new systems: k08 20.3 (5.9% / 0.29%; cos 0.9990), Ca<sub>2</sub>SnTiO<sub>6</sub> 2.11 (16.6% / 7.9%; cos 0.9993), Ca<sub>8</sub>Sn<sub>6</sub>Ti<sub>2</sub>O<sub>24</sub> 2.90 (32.3% / 11.2%; cos 0.9996).

## Results

| System | ρ | φ (TRIM) | φ_all | ΔΛ_perp / ΔΛ_par (meV) | H⁻¹f cosine |
|---|---|---|---|---|---|
| Spinel k12 | 26 | 0.97 | — | +15.2 / −0.4 | 0.89 |
| Spinel k08 | 20.3 | 0.008 | 0.25 | +0.06 / −7.82 | 0.9990 |
| Ca<sub>8</sub>Sn<sub>6</sub>Ti<sub>2</sub>O<sub>24</sub> | 2.90 | 0.30 | 0.44 | +1.29 / −3.01 | 0.9996 |
| CaSnO<sub>3</sub> | 2.25 | 0.013 | 0.08 | +0.16 / −12.70 | 0.9996 |
| Ca<sub>2</sub>SnTiO<sub>6</sub> rock-salt | 2.11 | 0.85 | 0.50 | −2.20 / +0.39 | 0.9993 |
| CaTiO<sub>3</sub> | 1.2 | 0.037 | 0.13 | −0.65 / +17.14 | 0.9994 |

- ρ values of the reference systems: k12 26 (79/3), CaTiO<sub>3</sub> 1.2 (12/10), CaSnO<sub>3</sub> 2.25 (27/12).
- The Spearman rank correlation between ρ and φ is 0.14: k08 has large ρ but φ = 0.008, the rock-salt small ρ but φ = 0.85. The redirection index does not fix where the electronic change lands; that also requires the electronic coupling W_e of each component (dΛ/dλ = W_e(JR + c)).
- The zero-parameter H⁻¹f prediction reaches cos ≥ 0.999 in all five symmetric-sector cases, and the predicted energy lowering agrees with DFT to within 3%.

## Functional dependence (r<sub>2</sub>SCAN)

**r<sub>2</sub>SCAN at the fixed PBE four-state geometries (F1)**

| System | Margin | ΔΛ_perp / ΔΛ_par, PBE (meV) | ΔΛ_perp / ΔΛ_par, r<sub>2</sub>SCAN (meV) | φ PBE → r<sub>2</sub>SCAN |
|---|---|---|---|---|
| Spinel | Λ_C = E_C(½½½) − E_C(Γ) | +15.18 / −0.41 | +18.56 / −2.45 | 0.974 → 0.883 |
| CaTiO<sub>3</sub> | Λ_V (Γ vs U) | −0.65 / +17.14 | −0.74 / +18.00 | 0.037 → 0.039 |
| CaSnO<sub>3</sub> | Λ_V (U vs Z) | +0.16 / −12.70 | −0.01 / −12.43 | 0.013 → 0.0005 |

- The spinel's absolute Λ_C shifts under r<sub>2</sub>SCAN (state00 −6.4 → −16.2 meV, state11 +8.2 → 0.0 meV); the increment carried by the bond-preserving component is +15.2 (PBE) and +18.6 meV (r<sub>2</sub>SCAN), same sign and magnitude.

**Relaxation, Hessian and four states entirely in r<sub>2</sub>SCAN (F2)**
- CaTiO<sub>3</sub>: H⁻¹f vs actual cosine 0.9996; energy lowering 3.70 predicted vs 3.62 meV DFT; ρ = 0.95 (PBE 1.2); φ = 0.032 (PBE 0.037).
- CaSnO<sub>3</sub>: cosine 0.9971; energy 5.72 predicted vs 5.76 meV DFT; ρ = 18.4% / 8.9% = 2.08 (PBE 2.25). The VBM stays at U, but the TRIM runner-up changes from Z (PBE) to Γ (r<sub>2</sub>SCAN), so the primary margin is a different pair of extrema, on which φ = 0.39 (ΔΛ_perp +1.41, ΔΛ_par +2.21 meV); φ_all is 0.065 (r<sub>2</sub>SCAN) vs 0.083 (PBE). The functional reorders near-degenerate runner-up valence extrema at the meV level without changing the structural competition or the band edge.
- The r<sub>2</sub>SCAN CaSnO<sub>3</sub> relaxations: P0 with residual forces 0.002–0.003 eV/Å and |stress| < 0.1 kB; P5 with EDIFFG = −3E-3 (r<sub>2</sub>SCAN grid noise prevents EDIFFG = 1E-3).
