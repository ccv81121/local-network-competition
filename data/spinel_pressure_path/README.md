# Pressure path of Mg(Sc1−xSbx)2Se4, x = 0.75

The 56-atom cell relaxed at 0, 2, 3, 4, 5 and 6 GPa to max |F| ≤ 0.001 eV/Å; band-edge margins along the path, the four-state construction between 4 and 5 GPa, and the force-balance prediction on these structures. `tight_summary.json` collects the derived numbers with their source files.

- Settings (`relax_tight.sh`): ISIF 3, PSTRESS = P, ENCUT 520 eV, EDIFFG −1E-3 eV/Å, EDIFF 1E-8 eV, ISMEAR −5, NBANDS 172, Γ-centred 3×3×3 mesh. Structures and INCARs in `spinel/p*/relax/`; the 7.5-GPa relaxation ended at the force noise floor and is not used.
- Margins Λ_C = E_C(½,½,½) − E_C(Γ) and Λ_V = E_V(Γ) − E_V(q_V), q_V the highest non-Γ valence point, on Γ-centred 4×4×4 and 6×6×6 static meshes (`static_k4/`, `static_k6/`): `path_edges.py` → `path_edges.json`.
- Four states at the 5-GPa lattice, 6×6×6 mesh: `fourstate/state{00,10,01,11}/` (POSCAR, INCAR, KPOINTS, total energy in `toten.txt`); margins and energies from `fourstate_margins.py`, collected in `tight_summary.json` (`four_state_k6`).
- `bending_shares.py` → `bending_shares.json`: bending shares of the drive, of the predicted response H⁺f and of the relaxed 4→5 GPa displacement, their cosines, and the residual forces of the earlier, less tightly relaxed 4- and 5-GPa endpoints (key `round5`).
- `force_balance/`: force balance with the drive computed at r_aff (tight 4-GPa fractional coordinates in the tight 5-GPa lattice; `forces/ref.npy`) and the force-constant matrix of the earlier pressure set (`../force_balance_reference`; `prepare_force_balance.py`, `meta.json`; results in `force_balance/outputs/`).
- `static_k6_pred/`: 6×6×6 readouts of r_aff and of the predicted structure `poscar_pred_tight` (`predicted_structure.py`).
- `geometry/`: geometric analyses of the same structures (`build_rows.py` writes `pressure_geometry_input_rows.csv`; `analyze_pressure_*.py`, `reconstruct_pressure_geometry.py`).
