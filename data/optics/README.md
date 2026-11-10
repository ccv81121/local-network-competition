# Near-edge optical response

Absorption and ε₂ near the gap for edge-separated versus edge-colocated pairs: Mg(Sc1−xSbx)2Se4 (x = 0.75) at 2 and 5 GPa (broadening profiles c005 and c010), BaSn1−xTixO3 at x = 0.25 and 0.50, and double-perovskite pairs. Energies are measured from each spectrum's gap Eg (global sampled-k CBM minus VBM from EIGENVAL) and paired spectra are interpolated onto a common grid.

- `optical_integrals_recomputed_from_primary_grids.csv`: trapezoid integrals over 0–0.3 eV above Eg and their relative change within each pair.
- `optical_pair_aligned_common_grid.csv.gz`, `optical_pair_common_grid.csv.gz`: the paired spectra on the common relative-energy grid.
- `optical_eg_alignment.csv`, `optical_relative_energy_window_0p30.csv`: Eg, minimum vertical gap and edge k points per spectrum, and the grid rows of the 0.3-eV window; `optical_alignment_provenance.json`: definitions and source hashes.
- Run settings (LOPTICS, NBANDS 224, CSHIFT 0.05 eV, Γ-centred 4×4×4) are in `number_sources/optics/`.
