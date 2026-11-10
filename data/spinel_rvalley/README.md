# RVALLEY_20261104
Test of the R-valley mixing: Mg(Sc0.25Sb0.75)2Se4 (k12, x=0.75), tight 5-GPa structure (four-state state11, 56 atoms,
Lambda_C = E_C(R)-E_C(Gamma) = +6.78 meV). 112-atom supercell A1=a1+a2, A2=a2+a3, A3=a3+a1 folds R=(1/2,1/2,1/2) onto Gamma.
Structures: state11 replicated + s*(-1)^(n1+n2+n3)*d_b, d_b = r(state10)-r(state00) (bending), rms over Sc/Sb/Se atoms
= 0, 0.01, 0.02, 0.04 A (folders a000, a010, a020, a040); Mg not displaced.
Settings: INCAR of state11 + NBANDS=344, LWAVE/LCHARG=.FALSE., ISYM=0. KPOINTS Gamma-centred 5x5x5 (spacing 0.0153 1/A vs 0.0148 for 6x6x6 of the 56-atom cell).
POTCAR = $WORK/PRL_TIGHT_SPINEL_20261015/work/Spinel/P4_P5/fourstate/state00/POTCAR. VASP 6.5.1 GPU, one job per GPU.
Prediction: split of the two lowest conduction levels at Gamma = sqrt(L1^2 + 4|M d|^2). Read out with readout_v1.py (EIGENVAL / vasprun.xml kept).
Run: run_rvalley_v1.sh GPU DIR ; completion: DONE / FAILED files in each folder.

## Archive note (2026-11-05)
- `levels.json` holds the two lowest conduction levels at Γ for a000–a040, read from EIGENVAL. The full run folders stay on the server.
- Used in SI Note 5 and in the main-text robustness sentence: the split is 6.78 → 6.72 meV at 0.04 Å, both levels shift together, and |M| is below about 12 meV/Å.
- Graded in `notes/np_submission/prereg_dispersion.md`: R2 passes, and R1 fails as registered because there is no positive M².
