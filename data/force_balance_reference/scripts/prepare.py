"""Build inputs for the zero-parameter force-balance prediction.
Start structure r_aff: 4-GPa relaxed fractional coordinates in the actual 5-GPa lattice.
Hessian of r_aff by central finite differences (+-DELTA Angstrom on each of 168 Cartesian dof).
Reads inputs/poscar_4gpa and inputs/poscar_5gpa of the force_balance_reference folder and its POTCAR (VASP licence, not
deposited); writes cases/<name>/{POSCAR,INCAR,KPOINTS,POTCAR} for the reference ("ref") and the 336 displaced runs
(dNNxp, dNNxm, ...) and meta.json there."""
import json, hashlib, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DELTA = 0.01


def read(path):
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1])
    lattice = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)]) * scale
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    frac = np.array([[float(x) for x in lines[8 + i].split()[:3]] for i in range(sum(counts))])
    return lattice, species, counts, frac


def write(path, title, lattice, species, counts, frac):
    out = [title, "1.0"] + ["  ".join(f"{x:.16f}" for x in vector) for vector in lattice]
    out += (["  ".join(species), "  ".join(map(str, counts)), "Direct"]
            + ["  ".join(f"{x:.16f}" for x in row) for row in frac])
    Path(path).write_text("\n".join(out) + "\n")


lattice_4gpa, species, counts, frac_4gpa = read(ROOT / "inputs" / "poscar_4gpa")
lattice_5gpa, species_5gpa, counts_5gpa, frac_5gpa = read(ROOT / "inputs" / "poscar_5gpa")
assert species == species_5gpa and counts == counts_5gpa
inv_lattice_5gpa = np.linalg.inv(lattice_5gpa)
INCAR_TEMPLATE = """SYSTEM = force-balance finite difference
ENCUT = 520
PREC = Accurate
EDIFF = 1E-7
NELM = 180
NELMIN = 5
ALGO = {algo}
ISMEAR = -5
SIGMA = 0.05
IBRION = -1
NSW = 0
ISPIN = 1
ISYM = 0
LDAU = .FALSE.
LREAL = .FALSE.
LASPH = .TRUE.
ADDGRID = .TRUE.
LMAXMIX = 4
NBANDS = 172
ISTART = {istart}
ICHARG = {icharg}
LWAVE = {lwave}
LCHARG = .FALSE.
KPAR = 1
NCORE = 1
"""
KPOINTS_TEXT = "Gamma K3 (relaxation mesh)\n0\nGamma\n3 3 3\n0 0 0\n"
jobs = []


def case(name, frac, is_reference):
    case_dir = ROOT / "cases" / name
    case_dir.mkdir(parents=True, exist_ok=True)
    write(case_dir / "POSCAR", name, lattice_5gpa, species, counts, frac)
    (case_dir / "INCAR").write_text(INCAR_TEMPLATE.format(algo="Normal" if is_reference else "Fast",
                                                          istart=0 if is_reference else 1,
                                                          icharg=2 if is_reference else 0,
                                                          lwave=".TRUE." if is_reference else ".FALSE."))
    (case_dir / "KPOINTS").write_text(KPOINTS_TEXT)
    shutil.copy(ROOT / "POTCAR", case_dir / "POTCAR")
    jobs.append(name)


case("ref", frac_4gpa, True)
for i in range(sum(counts)):
    for a in range(3):
        for sign, tag in ((1, "p"), (-1, "m")):
            cart = frac_4gpa @ lattice_5gpa
            cart[i, a] += sign * DELTA
            case(f"d{i:02d}{'xyz'[a]}{tag}", cart @ inv_lattice_5gpa, False)
meta = {"delta_A": DELTA, "lattice_5GPa": lattice_5gpa.tolist(), "frac_start": frac_4gpa.tolist(),
        "frac_5GPa": frac_5gpa.tolist(),
        "species": species, "counts": counts, "jobs": jobs,
        "poscar_sha256": {k: hashlib.sha256((ROOT / "inputs" / k).read_bytes()).hexdigest()
                          for k in ("poscar_4gpa", "poscar_5gpa")},
        "potcar_sha256": hashlib.sha256((ROOT / "POTCAR").read_bytes()).hexdigest()}
(ROOT / "meta.json").write_text(json.dumps(meta, indent=1))
print(len(jobs), "cases")
