"""Build inputs for the Mg-Sc-Sb-Se convex hull (PBE, U=0, ENCUT 520, the POTCARs of the pressure-path calculations).

Competitors: full relaxation (two ISIF=3 rounds) + tetrahedron static, at 0 GPa and at PSTRESS = 50 kbar.
Spinel structures: one static each at the unified settings (S01-S14 relaxed 0-GPa ends of the substitution path;
k12 at 0 and 5 GPa).

Reads inputs/ (S01-S14 CHGNet-relaxed structures, k12 tight-relaxed 0- and 5-GPa structures, Sb2Se3 CIF COD 2310974);
the POTCAR library is read from the cluster (/data/vasp, licensed, not deposited).
usage: python hull_prepare.py
"""
import hashlib, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
from ase.build import bulk
from ase.spacegroup import crystal
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp import Poscar

OUT_DIR = Path(__file__).resolve().parent
POTCAR_LIBRARY = Path("/data/vasp/potpaw_PBE")
POTCAR_DIR_NAME = {"Mg": "Mg", "Sc": "Sc_sv", "Sb": "Sb", "Se": "Se"}
INPUT_DIR = OUT_DIR / "inputs"
METALS = {"Mg", "Sc", "Sb", "ScSb", "ScSe"}

INCAR_BASE = """PREC = Accurate
ENCUT = 520
EDIFF = 1E-6
ISPIN = 1
LREAL = .FALSE.
LASPH = .TRUE.
ADDGRID = .TRUE.
ALGO = Normal
NELM = 200
KSPACING = 0.20
KGAMMA = .TRUE.
LWAVE = .FALSE.
LCHARG = .FALSE.
"""


def smear(name, static):
    if name in METALS:
        return "ISMEAR = 1\nSIGMA = 0.10\n"
    return "ISMEAR = -5\n" if static else "ISMEAR = 0\nSIGMA = 0.02\n"


def potcar(structure, d):
    with open(d / "POTCAR", "wb") as fh:
        for el in [e.symbol for e in structure.composition.elements]:
            fh.write((POTCAR_LIBRARY / POTCAR_DIR_NAME[el] / "POTCAR").read_bytes())


def write(d, structure, name, kind, pstress, hashes):
    d.mkdir(parents=True, exist_ok=True)
    structure = structure.get_sorted_structure(key=lambda s: s.species_string)
    Poscar(structure, comment=f"hull {name} {kind} P{pstress}").write_file(d / "POSCAR")
    potcar(structure, d)
    pstress_line = f"PSTRESS = {pstress}\n" if pstress else ""
    if kind == "relax":
        relax_incar = (INCAR_BASE + smear(name, False) + "IBRION = 2\nISIF = 3\nNSW = 150\nEDIFFG = -0.005\n"
                       + pstress_line)
        (d / "INCAR.r").write_text(relax_incar)
        (d / "INCAR.s").write_text(INCAR_BASE + smear(name, True) + "IBRION = -1\nNSW = 0\n")
    else:
        (d / "INCAR.s").write_text(INCAR_BASE + smear(name, True) + "IBRION = -1\nNSW = 0\n")
    (d / "KIND").write_text(kind + "\n")
    hashes[str(d.relative_to(OUT_DIR))] = hashlib.sha256((d / "POSCAR").read_bytes()).hexdigest()


def atoms_to_structure(atoms):
    return AseAtomsAdaptor.get_structure(atoms)


def competitors():
    structures = {
        "Mg": atoms_to_structure(bulk("Mg", "hcp", a=3.21, c=5.21)),
        "Sc": atoms_to_structure(bulk("Sc", "hcp", a=3.309, c=5.273)),
        "Sb": atoms_to_structure(crystal("Sb", [(0, 0, 0.2336)], spacegroup=166,
                                         cellpar=[4.3084, 4.3084, 11.274, 90, 90, 120], primitive_cell=True)),
        "Se": atoms_to_structure(crystal("Se", [(0.2254, 0, 1 / 3)], spacegroup=152,
                                         cellpar=[4.366, 4.366, 4.954, 90, 90, 120])),
        "MgSe": atoms_to_structure(bulk("MgSe", "rocksalt", a=5.46)),
        "ScSe": atoms_to_structure(bulk("ScSe", "rocksalt", a=5.40)),
        "ScSb": atoms_to_structure(bulk("ScSb", "rocksalt", a=5.85)),
        "Mg3Sb2": atoms_to_structure(crystal(["Mg", "Mg", "Sb"],
                                             [(0, 0, 0), (1 / 3, 2 / 3, 0.6298), (1 / 3, 2 / 3, 0.2256)],
                                             spacegroup=164, cellpar=[4.573, 4.573, 7.229, 90, 90, 120])),
        "Sb2Se3": CifParser(INPUT_DIR / "sb2se3_cod_2310974.cif").parse_structures(primitive=False)[0],
    }
    s01 = Structure.from_file(INPUT_DIR / "s01_poscar.vasp")
    s01.replace_species({"Sb": "Sc"})
    s14 = Structure.from_file(INPUT_DIR / "s14_poscar.vasp")
    s14.replace_species({"Sc": "Sb"})
    structures["MgSc2Se4"] = s01.get_primitive_structure(tolerance=0.3)
    structures["MgSb2Se4"] = s14.get_primitive_structure(tolerance=0.3)
    return structures


def main():
    hashes = {"Sb2Se3_cif_sha256": hashlib.sha256((INPUT_DIR / "sb2se3_cod_2310974.cif").read_bytes()).hexdigest()}
    jobs = []
    info = {}
    for name, st in competitors().items():
        info[name] = {"formula": st.composition.reduced_formula, "natoms": len(st)}
        for pressure_kbar in (0, 50):
            d = OUT_DIR / "runs" / f"P{pressure_kbar}" / name
            write(d, st, name, "relax", pressure_kbar, hashes)
            jobs.append(str(d.relative_to(OUT_DIR)))
    for i in range(1, 15):
        st = Structure.from_file(INPUT_DIR / f"s{i:02d}_poscar.vasp")
        d = OUT_DIR / "runs" / "P0" / f"S{i:02d}"
        write(d, st, f"S{i:02d}", "static", 0, hashes)
        jobs.append(str(d.relative_to(OUT_DIR)))
    for pressure_kbar, src in ((0, "p0"), (50, "p5")):
        st = Structure.from_file(INPUT_DIR / f"k12_{src}_contcar.vasp")
        d = OUT_DIR / "runs" / f"P{pressure_kbar}" / "k12"
        write(d, st, "k12", "static", 0, hashes)
        jobs.append(str(d.relative_to(OUT_DIR)))
    (OUT_DIR / "hull_input_hashes.json").write_text(json.dumps(hashes, indent=1))
    (OUT_DIR / "jobs.txt").write_text("\n".join(jobs) + "\n")
    (OUT_DIR / "struct_info.json").write_text(json.dumps(info, indent=1))
    print(len(jobs), "jobs")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
