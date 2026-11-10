# Symmetry-sector (field) form of local-network competition (SI Note 8)

Numpy + spglib, read-only on other `data/` folders; each script runs in under a minute and writes its json next to itself.

- `checks.py` -> `checks.json`: validation of the force-balance analysis (`analyse`, `full_H`) on k12 4-5 GPa, CaTiO3/CaSnO3 0-5 GPa and the alloy-leverage runs. `closed_loop.py` imports its definitions.
- `closed_loop.py` -> `closed_loop.json`: space-group (Pnma, 20 atoms, 8 operations) sector decomposition of the single-Ti breathing drive and response in CaTiO3 and CaSnO3; sector sum rule (`single_site_sumrule_s_u` vs `single_site_direct_s_u`), per-sector `f2_frac`, `u2_net_frac`, `s_u`, Schur stiffness `Sb_min`, rigid-octahedron overlap `soft_rot_overlap`, Ag weight of uniform compression (`compression_f2_by_irrep`).
- `selfstress.py` -> `selfstress.json`: self-stress counts of the cubic B-X bond network at Gamma, X, M, R, and B-O-B bend angles with `tan2_half_mean` of the tilted cells; prints the Ag/B3g drive-weight ratio from `closed_loop.json` (run after it).

Inputs (all under `data/`): `spinel_pressure_path/force_balance/{meta.json,forces/ref.npy}`, `force_balance_reference/{forces/*.npy,inputs/pressure_bond_accommodation.json}`, `bending_enrichment/full_force_balance/{ca_tio3,ca_sno3}/p0_p5/{meta.json,hessian/*/forces.npy}`, `alloy_leverage/runs/<iv>/{H.npy,f.npy}`.

Rerun from the repository root (order matters):

    python -I data/field_sectors/checks.py
    python -I data/field_sectors/closed_loop.py
    python -I data/field_sectors/selfstress.py

The three jsons reproduce byte-identically. Numbers are used in Supplementary Note 8 and the main-text 95% (see `paper/numbers_ledger.csv`; the 95% is B1g `u2_net_frac*s_u` divided by the sum over sectors).

- `energy_bound.py` → `energy_bound.json`: the energy-share bound e_b ≤ (s_f s_u)^{1/2}/γ (Cauchy–Schwarz, main-text equation for the energy share; SI Note 8) and its corollary φ/(1−φ) ≥ (g_b/g_r) γ e_b s_f^{−1/2}, checked on the six compressed structures with force constants (`bound`). The table also has the geometric and property factors of eight compressions and two substitution steps for Supplementary Fig. 4b (`phase`), and checks that their product reproduces φ_pred. Inputs: `four_state_leverage/calculations/*/{F_raff,u_b,u_c,u_pred}.npy` and `leverage.json`, `bending_enrichment/enrichment_all_systems.csv`, `four_state_leverage/evaluation.json`, `alloy_leverage/al_result.json`. Run: `python -I data/field_sectors/energy_bound.py`.
- `response_fields.py` → `response_fields.npz`: real-space response of CaTiO3 to a breathing force on one Ti (bond-preserving part and remainder) for the Fig. 2e,f renders; s_u = 10.31%, rigid-rotation share of the bond-preserving part 97.5%.
- `dispersion/`: 80-atom dispersive H and the Ti->Sn closed-loop test (SI Note 8); see its README.
