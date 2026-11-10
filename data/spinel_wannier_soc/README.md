# Exact local–network split and spin–orbit margins of Mg(Sc1−xSbx)2Se4, x = 0.75

Exact partition of the 5-GPa band-edge margins of a Wannier Hamiltonian into on-site, B–X octahedral, A–X link and network parts, and PBE+SOC margins at 4, 5, 6 and 7.5 GPa, all on the tightly relaxed structures of `spinel_pressure_path` (inputs copied to `inputs/`).

- Wannier basis: SCF at four-state state 11 (6×6×6 Γ mesh, Gaussian smearing 0.02 eV, ISYM 0); NSCF on the full 4×4×4 mesh at fixed density (EDIFF 1E-8, 286 bands) with Wannier90 3.1.0, 248 Wannier functions; projection centres and basis sites (`wannier/basis_map.json`) moved atom by atom from the earlier pressure-set basis (`make_inputs.py`, largest shift 0.014 Å), energy windows from the band extrema (`fill_windows.py`).
- Disentanglement iterated to convergence (dis_conv_tol 1e-10) and maximal localisation (num_iter 1000): `disentangle.sh` → `results/wannier_mlwf_converged/` (`wannier90.win`, `wpart.json`). The projection-only basis on the same subspace: `projected_wannier_basis.py` → `results/wannier_proj_converged/wpart.json`. Both bases reproduce the 24 DFT edge levels on all 64 k points (`validation_edge24_match_max_abs_eV`) and have no Wannier level inside the gap (`ingap_ghost_*`).
- Split definitions (`wannier_margin_split.py`): Λ_C = E_C(R) − E_C(Γ), Λ_V = E_V(Γ) − max E_V over the X-type points; E_n(k) = Σ_g ⟨ψ_nk|H_g(k)|ψ_nk⟩ over the hopping groups.
- SOC: PBE seed (4×4×4, ISMEAR −5, LMAXMIX 4) then fixed-density PBE+SOC (LSORBIT, SAXIS 0 0 1) prepared by `prepare.sh` and run by `run_soc.sh`; margins with the same definitions from `soc_margins.py` → `results/soc_margins.csv`.
- `run_wannier.sh` and `maximal_localization.sh` run the SCF/NSCF chain and an initial localisation from the NSCF disentanglement.
