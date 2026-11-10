# Thermodynamic stability of Mg(Sc<sub>1−x</sub>Sb<sub>x</sub>)<sub>2</sub>Se<sub>4</sub>: results

At 0 GPa the x = 0.75 structure k12 (MgSc<sub>0.5</sub>Sb<sub>1.5</sub>Se<sub>4</sub>) lies 3.1 meV/atom above the convex hull (PBE, no van der Waals correction).

## 0 GPa
- Competing phases on the hull: Mg, Sc, Sb, Se, MgSe, ScSe, ScSb, Mg<sub>3</sub>Sb<sub>2</sub>, Sb<sub>2</sub>Se<sub>3</sub>, Sc<sub>2</sub>Se<sub>3</sub> (enumerated Fddd structure) and MgSc<sub>2</sub>Se<sub>4</sub>.
- The synthesized parent MgSc<sub>2</sub>Se<sub>4</sub> lies on the hull, stable by only 1.4 meV/atom against MgSe + Sc<sub>2</sub>Se<sub>3</sub>; the end member MgSb<sub>2</sub>Se<sub>4</sub> lies 9.4 meV/atom above it.
- Decomposition of k12: 0.25 MgSc<sub>2</sub>Se<sub>4</sub> + 0.21 MgSe + 0.54 Sb<sub>2</sub>Se<sub>3</sub> (atomic fractions).

| Structure | x(Sb) | E_hull (meV/atom) | Mixing energy vs. the two spinel end members (meV/atom) | Ideal mixing −TS at 1000 K (meV/atom) |
|---|---|---|---|---|
| S01 | 0.0625 | 0.1 | -0.5 | -5.8 |
| S02 | 0.1250 | -1.4 | -2.6 | -9.3 |
| S03 | 0.1875 | -2.1 | -3.8 | -11.9 |
| S04 | 0.2500 | -1.1 | -3.4 | -13.8 |
| S05 | 0.3125 | 1.8 | -1.1 | -15.3 |
| S06 | 0.3750 | 1.3 | -2.2 | -16.3 |
| S07 | 0.4375 | 1.1 | -3.0 | -16.9 |
| S08 | 0.5000 | 2.8 | -1.8 | -17.1 |
| S09 | 0.5625 | 2.2 | -3.1 | -16.9 |
| S10 | 0.6250 | 1.4 | -4.4 | -16.3 |
| S11 | 0.6875 | 2.1 | -4.4 | -15.3 |
| S12 | 0.7500 | 3.7 | -3.4 | -13.8 |
| S13 | 0.8125 | 5.0 | -2.6 | -11.9 |
| S14 | 0.8750 | 6.1 | -2.1 | -9.3 |
| k12 | 0.7500 | 3.1 | -3.9 | -13.8 |

- Along the substitution path E_hull lies between −2.1 and +6.1 meV/atom; S02–S04 are negative, i.e. these ordered structures are more stable than the listed competing phases.
- All mixing energies are negative (−0.5 to −4.4 meV/atom), and the ideal mixing entropy contributes −6 to −17 meV/atom at 1000 K, so the disordered solid solution is thermodynamically favoured at synthesis temperatures.

## 5 GPa (enthalpy H = E + PV)
- k12 lies 68.5 meV/atom above the hull and decomposes into Sc<sub>2</sub>Se<sub>3</sub> + MgSe + Sb<sub>2</sub>Se<sub>3</sub>; MgSc<sub>2</sub>Se<sub>4</sub> lies 58.8 and MgSb<sub>2</sub>Se<sub>4</sub> 80.9 meV/atom above it.
- Decomposition into rock-salt-based MgSe (octahedral Mg) + Sc<sub>2</sub>Se<sub>3</sub> lowers the volume by 1.9 Å<sup>3</sup>/atom (about 7%); at 5 GPa the PV term (about −60 meV/atom) outweighs the 1–3 meV/atom energy differences at 0 GPa.
- Under pressure the whole spinel family, including the synthesized parent, is therefore metastable against the denser rock-salt assemblage. The decomposition requires long-range cation diffusion and is kinetically blocked under room-temperature compression; this is a thermodynamic statement, not a predicted transition.

## Settings
- Sc<sub>2</sub>Se<sub>3</sub>: 495 vacancy arrangements in a Sc<sub>2</sub>S<sub>3</sub>-type rock-salt supercell, 21 after removal of duplicates; the lowest CHGNet candidate is Fddd, the space group of the known structure. The lowest four were relaxed with DFT; the lowest (c20, Fddd) lies on the hull. Ranking: `sc2se3_enum_ranking.json`.
- Sc (0 and 5 GPa) and Se (5 GPa) were relaxed with IBRION = 1 and EDIFF = 1E−7; the static of S08 used ISYM = 0, as did the original path statics.

Data: `hull_result.json`, `d3_result.json`, `struct_info.json`.

## Van der Waals (D3) sensitivity
The four phases that fix the decomposition of k12 (k12, MgSc<sub>2</sub>Se<sub>4</sub>, MgSe, Sb<sub>2</sub>Se<sub>3</sub>) were fully relaxed again with D3 (IVDW = 12); the decomposition fractions are those of PBE.

| | k12 relative to its decomposition products (meV/atom) | k12 volume per atom (Å<sup>3</sup>) |
|---|---|---|
| PBE | 3.1 | 27.96 |
| PBE + D3 | 22.8 | 26.82 |

D3 mainly stabilizes the chain compound Sb<sub>2</sub>Se<sub>3</sub>, raising k12 to 22.8 meV/atom above its decomposition products; the two treatments give a range of 3–23 meV/atom.
