# Octahedral tilt under pressure in CaTiO<sub>3</sub> and CaSnO<sub>3</sub>

Computed from the relaxed structures at 0, 5, 10, 15 and 20 GPa and the 0→5 GPa force constants of the corner-sharing perovskites; no further DFT calculations.
- `tilt_vs_p.py` → `tilt_vs_p.txt`: B–O–B angles and ⟨B–O⟩ versus pressure. Reads the relaxed structures in `inputs/ca_tio3/` and `inputs/ca_sno3/` (`p00_contcar` … `p20_contcar`, VASP CONTCAR at 0, 5, 10, 15, 20 GPa); run `python tilt_vs_p.py` in this folder.
- `tilt_force_balance.py` → `tilt_force_balance.txt`: force along the tilt coordinate, relaxed tilt stiffness, direct and coupled terms (0→5 GPa, fixed 5-GPa lattice).
- `compare_experiment.py` → `compare_experiment.txt`: weighted linear fit of the CaSnO<sub>3</sub> single-crystal data (Zhao, Ross, Angel, Phys. Chem. Minerals 31, 299 (2004), Table 3) compared with the calculated slopes per oxygen site.

| | Calculated dθ/dP (°/GPa, 0–5 GPa) | Experiment |
|---|---|---|
| CaSnO<sub>3</sub> Sn–O1–Sn | −0.100 | −0.059 ± 0.031 |
| CaSnO<sub>3</sub> Sn–O2–Sn | −0.112 | −0.130 ± 0.018 |
| CaSnO<sub>3</sub> mean (1:2) | −0.108 | −0.104 ± 0.016 |
| CaTiO<sub>3</sub> mean | −0.132 | −0.131 (Zhao bond-valence model, consistent with experiment to 10 GPa); −0.134 (Guennou et al., PRB 2010, bond-valence model); −0.160 (same work, Raman rotational mode) |

| 0→5 GPa (fixed 5-GPa lattice) | CaTiO<sub>3</sub> | CaSnO<sub>3</sub> |
|---|---|---|
| Angle change, predicted / actual (°) | −0.680 / −0.661 | −0.562 / −0.550 |
| Force along the tilt (eV/Å) | −0.191 | −0.201 |
| Tilt stiffness (eV/Å²) | 4.34 | 5.74 |
| Direct-term share | 106.5% | 94.7% |

Affine compression leaves the angles unchanged, so the tilt increment comes entirely from internal relaxation; the fixed-coordinate model of Zhao (2004) likewise predicts unchanged angles.

## Bare and relaxed tilt stiffness

`tilt_stiffness_split.py` → `tilt_stiffness_split.txt` (same force constants, no new DFT): bare stiffness e·H·e with every other coordinate clamped, and its split into on-site O terms and couplings between different O (the tilt direction moves only O).

| 0→5 GPa | CaTiO<sub>3</sub> | CaSnO<sub>3</sub> |
|---|---|---|
| Bare tilt stiffness (eV/Å²) | 5.01 | 6.50 |
| Softening by the other coordinates | 0.66 | 0.75 |
| Relaxed tilt stiffness | 4.34 | 5.74 |
| On-site O part / O–O coupling part of the bare stiffness | 7.96 / −2.96 | 8.57 / −2.08 |
