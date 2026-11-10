"""Build the two edge-sharing mixed-B structures Cd(Ga,In)2O4 and Na(Sc,In)O2.
1. Spinel_CdGaInO4: Cd(Ga0.5In0.5)2O4 on the k08 ordering of Mg(Sc0.5Sb0.5)2Se4 (Mg->Cd, Sc->Ga, Sb->In, Se->O),
   lattice scaled to a_cubic ~ 8.88 A (between CdGa2O4 8.59 and CdIn2O4 9.17).
2. Layered_NaScInO2: Na(Sc0.5In0.5)O2, alpha-NaFeO2 (R-3m) hexagonal cell doubled along a; in every layer one B
   site Sc and one In (stripe order). Starting a=3.24, c=16.35 A, O z=0.265.
Both are then relaxed by VASP (ISIF=3) at 0, 4, 5 GPa.
Reads inputs/poscar_k08_template; writes inputs/poscar_spinel_cd_ga_ino4 and inputs/poscar_layered_na_sc_ino2 and
prints their first lines."""
from pathlib import Path
import numpy as np

INPUT_DIR = Path(__file__).resolve().parent / "inputs"
template_lines = (INPUT_DIR / "poscar_k08_template").read_text().splitlines()
lat = np.array([[float(x) for x in template_lines[i].split()] for i in (2, 3, 4)]) * float(template_lines[1])
a_cub = np.sqrt(2) * lat[0, 0]
scale = 8.88 / ((a_cub + lat[2, 2]) / 2)
spinel_lines = ["Cd4 Ga4 In4 O16 (k08 ordering)", "1.0"] + ["  ".join(f"{x:.10f}" for x in v) for v in lat * scale]
spinel_lines += ["Cd Ga In O", "4 4 4 16"] + template_lines[7:8]
spinel_lines += ["  ".join(line.split()[:3]) for line in template_lines[8:8 + 28]]  # coordinates without the template species labels
assert template_lines[5].split() == ["Mg", "Sc", "Sb", "Se"] and template_lines[6].split() == ["4", "4", "4", "16"]
(INPUT_DIR / "poscar_spinel_cd_ga_ino4").write_text("\n".join(spinel_lines) + "\n")

a, c, z = 3.24, 16.35, 0.265
cen = [np.zeros(3), np.array([2 / 3, 1 / 3, 1 / 3]), np.array([1 / 3, 2 / 3, 2 / 3])]
B = [t for t in cen]
Na = [(t + [0, 0, 0.5]) % 1 for t in cen]
O = [(t + [0, 0, z]) % 1 for t in cen] + [(t - [0, 0, z]) % 1 for t in cen]


def supercell_site(f, k):
    return np.array([(f[0] + k) / 2, f[1], f[2]])


Sc = [supercell_site(f, 0) for f in B]
In = [supercell_site(f, 1) for f in B]
Nas = [supercell_site(f, k) for f in Na for k in (0, 1)]
Os = [supercell_site(f, k) for f in O for k in (0, 1)]
lat2 = np.array([[2 * a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
layered_lines = ["Na6 Sc3 In3 O12 (stripe order)", "1.0"] + ["  ".join(f"{x:.10f}" for x in v) for v in lat2]
layered_lines += ["Na Sc In O", "6 3 3 12", "Direct"] + ["  ".join(f"{x:.10f}" for x in f) for f in Nas + Sc + In + Os]
(INPUT_DIR / "poscar_layered_na_sc_ino2").write_text("\n".join(layered_lines) + "\n")
print(open(INPUT_DIR / "poscar_spinel_cd_ga_ino4").read()[:400])
print(open(INPUT_DIR / "poscar_layered_na_sc_ino2").read()[:300])
