# Local–network competition in crystals

Code and data for the manuscript *Local–network competition in crystals*. Every figure is regenerated from the
data in `data/` by one module in `code/`.

## Layout

```
code/                 figure modules (table below), shared style and data paths
data/                 results, structures, VASP/Wannier90 inputs and analysis scripts, one folder per topic
figures/              the figures of the Nature Physics manuscript and its Supplementary Information
symbolic_regression/  the search that produced the descriptor relations (see its README)
```

## Usage

Python 3.11 with the packages in `requirements.txt`; the figures are pixel-reproducible with the pinned versions.

```
pip install -r requirements.txt
python code/make_figures.py              # all figures into figures/
python code/make_figures.py Fig3         # one figure (Fig1 … Fig4, SFig1 … SFig6)
```

| Figure | Module | File |
|---|---|---|
| Fig. 1 | `fig1.py` | `fig1` |
| Fig. 2 | `fig_network.py` | `fig_network` |
| Fig. 3 | `fig4.py` | `fig4` |
| Fig. 4 | `fig3.py` | `fig3` |
| Supplementary Fig. 1 | `fig_s5.py` | `fig_s5` |
| Supplementary Fig. 2 | `fig_s1.py` | `fig_s1` |
| Supplementary Fig. 3 | `fig_s4.py` | `fig_s4` |
| Supplementary Fig. 4 | `fig_s2.py` | `fig_s2` |
| Supplementary Fig. 5 | `fig5.py` | `fig_s6` |
| Supplementary Fig. 6 | `fig_s3.py` | `fig_s3` |

## Data

| Folder | Content |
|---|---|
| `spinel_pressure_path` | Tightly relaxed Mg(Sc<sub>1−x</sub>Sb<sub>x</sub>)<sub>2</sub>Se<sub>4</sub> (x = 0.75) along pressure: structures, band-edge margins, four-state construction, force-balance prediction |
| `spinel_wannier_soc` | Exact local–network split of the 5-GPa conduction margin (projected and maximally localized Wannier bases) and spin–orbit margins on the same structures |
| `spinel_fatbands` | Orbital-resolved bands at 4 and 5 GPa |
| `spinel_relations` | Single-coordinate colocation relation for spinels, ordering margins along substitution, descriptor ablation |
| `perovskite_relations` | Single- and double-perovskite relations and their out-of-fold records |
| `descriptor_tables` | Descriptor tables from which `python -m symbolic_regression prepare` rebuilds the inputs of the spinel, single- and double-perovskite searches |
| `number_sources` | Source records of the perovskite relations, the 1003-state spinel substitution database and the optics settings |
| `relation_ablation` | Group neutralisation of local and network descriptors, path terms |
| `four_state_leverage`, `four_state_reanalysis` | Four-state constructions and bending shares across structures |
| `three_channel` | Three-channel decomposition of the margin increments |
| `bending_amplitude`, `bending_enrichment` | Bending amplitude and enrichment of bond-preserving motion |
| `wannier_channel_response` | Margin responses to ±25% scaling of Wannier Hamiltonian channels |
| `electronic_levels` | Electronic read-outs on the earlier pressure structures |
| `force_balance_reference` | Force constants and force-balance response on the affine reference |
| `substitution_path_split` | Swap and relaxation contributions along substitution paths |
| `thermal_valley` | Zero-point and thermal averages of the conduction margin |
| `hull_stability`, `epitaxial_strain`, `tilt_pressure`, `edge_extension` | Hull energies, strain, octahedral tilt under pressure, further band edges |
| `alloy_leverage`, `alloy_local_geometry`, `alloy_newton` | Alloyed perovskites |
| `polarization`, `optics` | Polarization and near-edge optical responses |
| `la3ni2o7` | La<sub>3</sub>Ni<sub>2</sub>O<sub>7</sub> under pressure and strain: structures, read-outs, interlayer hopping |
| `rendered` | Pre-rendered wavefunction images used in Fig. 3 |

Large raw outputs (wavefunctions,
charge densities, Wannier90 matrices and Hamiltonians, full band read-outs) and POTCAR files are not included. The job scripts in `data/` were run on a
cluster; paths in them refer to that machine.

## Licence

Code: MIT (`LICENSE`). Data and figures: CC BY 4.0 (`LICENSE-DATA`).
