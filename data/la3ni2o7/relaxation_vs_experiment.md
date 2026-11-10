# La3Ni2O7: relaxed structures compared with experiment

Structures: relaxed CONTCARs in `relaxation/` (non-magnetic PBE, 48-atom Amam cell, 8×8×2, EDIFFG −2×10⁻³ eV/Å;
`run_relaxation.sh`). Starting forces of the force-balance intervals: `force_balance/forces/`, analysed by `force_balance_analyze.py`. All
geometric quantities are computed with `four_state_build.geometry`; the experimental structure is Ling et al. (2000),
COD 1521521 (`poscar_exp`, written by `make_poscar.py`).

## 1. Ambient structure

| Quantity | Experiment (Ling 2000) | PBE 0 GPa | Deviation |
|---|---|---|---|
| Volume (Å³, 48 atoms) | 602.90 | 604.74 | +0.3% |
| a_p = √(a²+b²)/2 (Å) | 3.833 | 3.853 | +0.5% |
| c (Å) | 20.519 | 20.376 | −0.7% |
| Interlayer Ni–O1–Ni | 167.6° | 165.0° | −2.6° |
| In-plane Ni–O–Ni (mean) | 170.2° | 168.9° | −1.3° |
| Ni–O in-plane / inner apical / outer apical (Å) | 1.924 / 1.975 / 2.227 | 1.935 / 1.969 / 2.241 | +0.6% / −0.3% / +0.6% |

Experimental film lattice (Wang et al., PRX 16, 031008): a_p = 3.756 Å, c = 20.72 Å, i.e. c is +1.69% longer than the
PBE 0 GPa value c(0 GPa) = 20.376 Å.

## 2. Hydrostatic pressure and phase stability

| Pressure | a_p (Å) | c (Å) | Interlayer angle | In-plane angle | In-plane / inner / outer Ni–O (Å) |
|---|---|---|---|---|---|
| 0 GPa | 3.853 | 20.376 | 165.0° | 168.9° | 1.935 / 1.969 / 2.241 |
| 4 GPa | 3.823 | 20.140 | 167.3° | 171.1° | 1.917 / 1.955 / 2.185 |
| 8 GPa | 3.793 | 19.983 | 170.4° | 173.5° | 1.900 / 1.944 / 2.150 |

- H(Fmmm) − H(Amam) is +102 meV/cell at 4 GPa and +32 meV/cell at 8 GPa; linear extrapolation crosses zero at about
  9.8 GPa. The measured transition to Fmmm lies at about 10–14 GPa (Sun et al. 2023). The tilt-free Fmmm references are
  built by `make_fmmm.py` and relaxed by `run_fmmm_cpu.sh`.
- Within Amam the interlayer angle straightens by about 0.7° per GPa and the outer apical Ni–O2 bond shortens by about
  0.011 Å per GPa.

## 3. Biaxial compressive strain

| Strain | a_p (Å) | c (Å) | Interlayer angle | In-plane angle | In-plane / inner / outer Ni–O (Å) |
|---|---|---|---|---|---|
| 0 | 3.853 | 20.376 | 165.0° | 168.9° | 1.935 / 1.969 / 2.241 |
| −1% | 3.814 | 20.642 | 166.9° | 168.9° | 1.916 / 1.973 / 2.298 |
| −2% | 3.775 | 20.893 | 168.2° | 168.5° | 1.898 / 1.979 / 2.348 |

## 4. The two paths at equal a_p

At the a_p of the 8 GPa bulk structure (3.793 Å), with the strain data interpolated linearly between −1% and −2%:

| Quantity | Hydrostatic 8 GPa | Biaxial strain (interpolated, about −1.5%) | Difference |
|---|---|---|---|
| In-plane Ni–O | 1.900 Å | 1.906 Å | 0.3% |
| Interlayer Ni–O1–Ni | 170.4° | 167.6° | 2.8° |
| In-plane Ni–O–Ni | 173.5° | 168.7° | 4.8° |
| Outer apical Ni–O2 | 2.150 Å | 2.325 Å | 8% |
| c | 19.98 Å | 20.78 Å | 4% |

At equal a_p only the in-plane Ni–O bond length coincides; the tilt angles, the outer apical bond and c differ strongly.

## 5. Starting forces: bending share of the drive

s_f is the bond-preserving share of the affine drive force, 1/s_f the resulting upper bound on the enrichment,
s_u the bond-preserving share of the actual relaxation and ε = s_u/s_f.

| Interval | s_f | Bound 1/s_f | s_u | ε |
|---|---|---|---|---|
| P1 0→4 GPa | 27.6% | 3.6 | 46.6% | 1.7 |
| P2 4→8 GPa | 43.7% | 2.3 | 55.7% | 1.3 |
| S1 0→−1% | 16.5% | 6.1 | 45.2% | 2.7 |
| S2 −1→−2% | 14.5% | 6.9 | 44.3% | 3.1 |
