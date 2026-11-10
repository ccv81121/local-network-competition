# Force balance on two further edge-sharing mixed-B networks

Structures: spinel Cd(Ga<sub>0.5</sub>In<sub>0.5</sub>)<sub>2</sub>O<sub>4</sub> (28 atoms, ordering of the Mg(Sc<sub>0.5</sub>Sb<sub>0.5</sub>)<sub>2</sub>Se<sub>4</sub> k08 cell) and layered α-NaFeO<sub>2</sub>-type Na(Sc<sub>0.5</sub>In<sub>0.5</sub>)O<sub>2</sub> (24 atoms, Sc/In stripe order), built by `make_structures.py` into `inputs/`.
Each was relaxed (PBE, ENCUT 520 eV, no U, non-spin-polarized, ISIF = 3) at 0, 4 and 5 GPa; the 0→5 GPa interval was analysed with symmetry-sector Hessians (displacement 0.03 Å) and four-state structures using `sector_force_balance_prepare.py`, `sector_force_balance_analyze.py` and `phi_readout.py` (in `../bending_enrichment/` and `../electronic_levels/`).
Results: `results/<system>/` (`meta.json`, `geometry.json`, `force_balance_sym.json`, `phi.json`). Shares are of the bond-preserving (bending) component; enrichment = realized share / drive share.

| Quantity (0→5 GPa) | Cd(Ga,In)<sub>2</sub>O<sub>4</sub> | Na(Sc,In)O<sub>2</sub> | Mg(Sc,Sb)<sub>2</sub>Se<sub>4</sub> k12 / k08 |
|---|---|---|---|
| Symmetric degrees of freedom | 13 | 20 | — |
| Negative eigenvalues | 0 | 0 | — |
| cos(H⁻¹f, u) | 0.99987 | 0.99990 | — |
| Bond-preserving share of the drive | 0.47% | 0.0029% | 0.54% / 0.29% |
| Bond-preserving share of the predicted displacement | 3.78% | 0.087% | — |
| Bond-preserving share of the realized displacement | 4.05% | 0.052% | 32.9% / 4.9% |
| Enrichment (realized / drive) | 8.6 | 17.8 | 60.5 / 16.8 |
| Harmonic / DFT energy lowering (meV) | −7.82 / −7.82 | −60.9 / −61.2 | — |

In the spinel the best and next-best valence TRIM points are symmetry-degenerate, so Λ<sub>V</sub> ≡ 0; with the first non-degenerate rival the valence margin changes 64.6 → 64.1 meV (bending −1.39, bond-changing +0.87 meV, φ = 0.62) and the conduction margin (≈1440 meV) has φ = 0.15. In the layered compound the valence margin changes 3.3 → 4.0 meV with φ = 0.08 (conduction φ = 0.09), and the energy lowering is almost entirely bond-changing (−61.25 meV).
