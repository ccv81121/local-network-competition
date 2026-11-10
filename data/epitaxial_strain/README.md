# Epitaxial biaxial strain of the spinel

(001) biaxial strain ε = −3 … +3% of the x = 0.75 structure k12 and of the substitution-path structure S11 (56-atom cells): out-of-plane lattice from a quadratic fit of E(c) over ISIF = 2 relaxations (Γ-centred 3×3×3), then a readout static (path-split INCAR, ISYM = 0, Γ-centred 4×4×4) for the margins Λ_C, Λ_V, the critical strains, the elastic energy and, for the switching steps, the four-state mechanism split (enrichment and bending share φ).
- `strain.py prepare|stage2|stage3|eval` → `strain_result.json`, `fits.json`, `brackets.json`; job script `strain_job.sh`.
- `strain_supp.py prepare|eval` → `strain_supp_result.json`: mechanism split of the compressive S11 crossing (−1 → −2%) and 6×6×6 readouts of the points with |Λ_C| < 10 meV.
- `readout_eig/`: EIGENVAL reader `read_eigenval.py` and cell volumes `vols.txt` used for the Γ/R level split.
- Tables and discussion: `result.md`.
