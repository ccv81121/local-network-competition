# Descriptor tables of the symbolic search

Inputs from which `python -m symbolic_regression prepare --family <family>
--raw data/descriptor_tables/<family> --out <dir>` rebuilds the protocol and
space directories the published spinel, single-perovskite (SP) and
double-perovskite (DP) searches read. One row per structure; rows appear in
the order of the search.

`descriptors.csv` (all families): `sample_id`, then one column per
descriptor, in registry order. Values are written in shortest round-trip form.
`python -m symbolic_regression describe` recomputes them from `structures/`:

```bash
S=data/descriptor_tables/structures
python -m symbolic_regression describe --family spinel --structures $S/spinel_structures.csv \
    --samples data/descriptor_tables/spinel/samples.csv --out spinel.csv
python -m symbolic_regression describe --family spinel --structures $S/spinel_structures.csv \
    --samples data/descriptor_tables/spinel_edges/descriptors.csv --out spinel_edges.csv
python -m symbolic_regression describe --family sp --structures $S/perovskite_structures.csv --out sp.csv
python -m symbolic_regression describe --family dp --structures $S/perovskite_structures.csv --out dp.csv
```

The published spinel matrices passed once through a `%.17g` CSV read with
pandas' default float parser (a shift of a few units in the last place);
`describe` repeats that step. With numpy 2.4 and pandas 3.0 on Linux the four files equal the
`descriptors.csv` byte for byte. Other builds round differently in the last
digits; this shows more where rounding decides between equidistant
neighbours (A-site off-centring and X–B–X angles of 45 high-symmetry spinel
endpoints) or where acos is steep (angle deviations of nearly straight
perovskite bonds, ≤ 3 × 10⁻⁸ degree).

## structures/

`spinel_structures.csv`
: the 1003-structure spinel database, one row per structure (`Task
  Directory` = `sample_id`): site species (`A_site`, `B_original`,
  `substitute`, `X_site`), the embedded POSCAR (`spinel_poscar_gz_b64`,
  gzip + base64) and, for the canonicalised path endpoints
  (`endpoint_canonicalized`), the 14-atom primitive cell
  (`primitive_poscar_gz_b64`), with the band-structure columns of each state.

`elements_primary.csv`, `elements_extended.csv`
: element properties of the spinel descriptors (the extended table takes
  precedence).

`perovskite_structures.csv`
: the 221 SP and DP states (`family` = perovskite / double_perovskite):
  site species (`A_site`, `B_original`, `substitute`, `B_fixed`, `X_site`),
  the relaxed supercell of the mode-designed SP states
  (`POSCAR_relaxed_supercell`) and the 40-atom geometry of every state
  (`geometry_40atom_poscar_gz_b64`).

## spinel/ (442 development structures, 214 descriptors)

`samples.csv`
: `source_global_row_index` (row in the 1003-structure database),
  `sample_id`, `strict_label` (co_located / separated), `y` (1 = co-located
  band edges), `ordering_margin_eV` and `ordering_margin_kind` (the margin
  that decides hard-core membership), `is_metal_excluded`,
  `is_margin_unavailable`, `curve_id` (substitution curve), `mp_id`, `k`
  (step along the curve), `relation_component_id` (connected group of
  related structures), and descriptive tags `challenge_5d_soc`,
  `challenge_small_gap`, `challenge_structure_occupancy`,
  `challenge_rare_chemistry`, `challenge_multi_factor`.

`descriptor_definitions.csv`
: `name`, `definition`, `physical_sector`, `unit`, `dimension_signature`,
  `role_hint`, `redundancy_group`, `allowed_transforms`, and `minimum`,
  `maximum`, `median` over the full 1003-structure database (the minimum
  fixes each descriptor's domain).

## sp/ (136 structures, 111 descriptors) and dp/ (85 structures, 182 descriptors)

`samples.csv`
: `sample_id`, `curve_id`, `parent_formula` (the grouping of the folds),
  `mode_design_label` (set for multiconfiguration SP states) and `target`
  (1 = direct gap).

`dp/numerical_guard.csv`
: per descriptor, `allow_for_search` and `guard_reason` of the
  numerical-resolution guard, with the drifts that decided it: the largest
  scaled change under physically identical re-representations (species
  order, origin, axis permutation) and the 95th-percentile scaled change
  under 10⁻⁶ Å and 10⁻⁵ Å coordinate perturbations. Allowed are the active
  descriptors with drifts ≤ 10⁻³, 0.1 and 1, except `dp_oct_bond_std_contrast`;
  these 109 enter the DP search.

## spinel_edges/ (681 retained states, 214 descriptors; two-edge search)

`prepare --family spinel-edges --raw data/descriptor_tables/spinel_edges`
writes the inputs of `edges run` (spinel valence and conduction coordinates).
The 681 states are the 440 discovery states, 224 challenge states and 17
excluded states; all rows enter the protected-ratio offsets of Π_S in the
search, only the discovery rows its scores. Rows follow the search matrix.

`descriptors.csv`
: as above, for the 681 states.

`band_edges.csv`
: per state, copied cell for cell from the band-structure table:
  `sample_id`, `k` (step along the substitution curve),
  `endpoint_canonicalized` (curve endpoints), `VBM_kpoint_frac`,
  `CBM_kpoint_frac` (fractional k point of the edge in the path calculation),
  `primitive_VBM_kpoint_frac`, `primitive_CBM_kpoint_frac` (in the
  primitive-cell calculation, used for endpoints), and the Γ-versus-non-Γ
  margins `S_V_Gamma_minus_nonGamma_eV` = E_V(Γ) − E_V(non-Γ) and
  `S_C_nonGamma_minus_Gamma_eV` = E_C(non-Γ) − E_C(Γ) (positive when Γ wins).

`discovery_samples.csv`
: `sample_id` of the 440 discovery states, in search order.

`random_forest_importance.csv`
: `feature`, and the random-forest importances `valence`, `conduction` of
  each descriptor for the two edges (the sampling prior of the search).
