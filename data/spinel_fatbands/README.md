# Orbital-resolved bands of Mg(Sc1−xSbx)2Se4, x = 0.75

PBE bands along R–Γ–Y at 4 and 5 GPa on the tightly relaxed structures of `spinel_pressure_path` (5 GPa: four-state state 11), with Sc-d, Sb and Se weights from PROCAR (LORBIT 11), and the real-space CBM wavefunctions at Γ and R at 5 GPa.

- `fatband_p4.npz`, `fatband_p5.npz`: k-distance, energies relative to the VBM and orbital weights; `fatband_summary.json`: VBM/CBM locations, gap, Λ_C = E_C(R) − E_C(Γ) on the path and the CBM orbital weights.
- `raw/p4/`, `raw/p5/`: INCAR and KPOINTS of the band runs (and the 4-GPa POSCAR), set up and run by `setup_and_run.sh`.
- `fatband_data.py` reads PROCAR and writes the arrays and summary; `wavefunction_cell.py` and `wavefunction_cluster.py` write the cube and VESTA files of the CBM states (rendered images in `rendered/`).
