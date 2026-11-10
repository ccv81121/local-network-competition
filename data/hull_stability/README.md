# Convex-hull stability of the spinel substitution path

Energies above the Mg–Sc–Sb–Se convex hull at 0 GPa (E) and 5 GPa (H = E + PV) for the substitution-path structures S01–S14 and the x = 0.75 structure k12, with mixing energies against the two spinel end members and the ideal B-site mixing entropy; PBE, U = 0, ENCUT 520 eV, with a D3 check on the phases that fix the decomposition.
- Inputs: `hull_prepare.py` (competitors and spinel statics; reads `inputs/`: the S01–S14 starting structures `s01_poscar.vasp`…, the tight k12 structures at 0 and 5 GPa `k12_p0_contcar.vasp`, `k12_p5_contcar.vasp`, and the Sb<sub>2</sub>Se<sub>3</sub> CIF `sb2se3_cod_2310974.cif`; the POTCAR library of the cluster is not deposited), `sc2se3_enum.py` (Sc<sub>2</sub>Se<sub>3</sub> vacancy-ordering enumeration and CHGNet ranking, `sc2se3_enum_ranking.json`), `hull_prepare_sc2se3.py` (DFT relaxation of the lowest Sc<sub>2</sub>Se<sub>3</sub> candidates and elemental re-relaxations); job script `run_job.sh`.
- Results: `hull.py` → `hull_result.json`; D3 comparison `d3_result.json`; structure metadata `struct_info.json`.
- Tables and discussion: `result.md`.
