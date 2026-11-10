# Electronic read-outs on the earlier pressure structures

The exact local–network split of the 5-GPa conduction margin used in the manuscript is in `../spinel_wannier_soc/` (converged disentanglement); an earlier split on a basis with a spurious in-gap state is not included.
- `sector_force_balance/spinel_k08/p0_p5/`: symmetric-sector force balance (`sector_force_balance_analyze.py` → `force_balance_sym.json`, `geometry.json`, `meta.json`, `poscar_pred`) and four-state inputs with `fourstate/phi.json` (`phi_readout.py`; Γ-centred 6×6×6, Γ vs (0,0,½)): φ = 0.0082, ΔΛ_C bending +0.06 / complementary −7.82 / total −7.66 meV, state energies +0.53 (10) and −110.24 meV (01) per cell.
- `sector_force_balance/mg_sc2_se4/p4_p5/`: the same for MgSc<sub>2</sub>Se<sub>4</sub>, where the bond-preserving share vanishes by symmetry (1.1E-27), with its force constants (`S.npy`, `Hs.npy`, `hessian/*/forces.npy`). The k08 force constants are in `../bending_enrichment/sector_force_balance/spinel_k08/p0_p5/`.
- `relaxations/spinel_k08/`, `relaxations/mg_sc2_se4/`: relaxations at 0, 4 and 5 GPa (start structure `poscar_init`). `index_and_functional.md`: redirection index ρ of each system, electronic partition and r<sub>2</sub>SCAN tests across systems.
- Not included: Wannier90 mmn/amn/chk/hr/u files, CHGCAR, POTCAR, DOSCAR, PROCAR and large vasprun files.
