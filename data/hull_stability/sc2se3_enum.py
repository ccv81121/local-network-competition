"""Sc2Se3 for the hull, built by enumeration (no experimental structure file is available).

Sc2Se3 is of the Sc2S3 type (Fddd, oF80; AFLOW prototype A3B2_oF80_70_fh_2e-001, Dismukes & White 1964): rock salt
with one third of the cation sites vacant and ordered. Its F-orthorhombic cell (2a0, sqrt2 a0, 3 sqrt2 a0) has a
primitive cell of 12 anion + 12 cation sites of rock salt. All C(12,4) = 495 vacancy patterns are enumerated,
duplicates removed with StructureMatcher, each candidate is relaxed with CHGNet (ranking only), and the lowest four
are relaxed with DFT (hull_prepare_sc2se3.py). The space group of each is reported.

Writes sc2se3/cand<k>.vasp (not deposited) and sc2se3_enum_ranking.json next to this script. Needs chgnet.
usage: python sc2se3_enum.py
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

OUT_DIR = Path(__file__).resolve().parent / "sc2se3"
OUT_DIR.mkdir(exist_ok=True)
A0 = 5.40
A, B, C = np.array([2, 0, 0.]), np.array([0, 1, 1.]), np.array([0, -3, 3.])  # orthorhombic axes in units of a0
PRIMITIVE_VECTORS = np.array([(B + C) / 2, (A + C) / 2, (A + B) / 2]) * A0  # F-centred primitive vectors
lat = Lattice(PRIMITIVE_VECTORS)
FCC_OFFSETS = [np.array(v) * 0.5 for v in itertools.product(range(-8, 9), repeat=3) if sum(v) % 2 == 0]


def sites(shift):
    frac_sites = []
    for v in FCC_OFFSETS:
        frac_site = lat.get_fractional_coords((v + shift) * A0) % 1.0
        frac_site[np.isclose(frac_site, 1.0, atol=1e-6)] = 0.0
        if not any(np.allclose(frac_site, g, atol=1e-4) for g in frac_sites):
            frac_sites.append(frac_site)
    return frac_sites


se, cat = sites(np.zeros(3)), sites(np.array([0.5, 0, 0]))
assert len(se) == 12 and len(cat) == 12, (len(se), len(cat))
candidates = []
for vac in itertools.combinations(range(12), 4):
    sc = [cat[i] for i in range(12) if i not in vac]
    candidates.append(Structure(lat, ["Sc"] * 8 + ["Se"] * 12, sc + se))
matcher = StructureMatcher(ltol=0.1, stol=0.1, angle_tol=2, primitive_cell=False, scale=False)
unique_candidates = [g[0] for g in matcher.group_structures(candidates)]
print("patterns", len(candidates), "unique", len(unique_candidates))

from chgnet.model import StructOptimizer
optimizer = StructOptimizer()
rows = []
for k, s in enumerate(unique_candidates):
    r = optimizer.relax(s, fmax=0.05, steps=400, relax_cell=True, verbose=False)
    fs = r["final_structure"]
    e = float(r["trajectory"].energies[-1]) / len(fs)
    sg = SpacegroupAnalyzer(fs, symprec=0.1).get_space_group_symbol()
    sg0 = SpacegroupAnalyzer(s, symprec=0.01).get_space_group_symbol()
    rows.append({"k": k, "E_chgnet_eV_atom": e, "sg_ideal": sg0, "sg_relaxed": sg})
    fs.to(filename=str(OUT_DIR / f"cand{k:03d}.vasp"), fmt="poscar")
rows.sort(key=lambda r: r["E_chgnet_eV_atom"])
e0 = rows[0]["E_chgnet_eV_atom"]
for r in rows:
    r["dE_meV_atom"] = 1e3 * (r["E_chgnet_eV_atom"] - e0)
(OUT_DIR.parent / "sc2se3_enum_ranking.json").write_text(json.dumps(rows, indent=1))
for r in rows[:8]:
    print(r)
