"""Build the finite-difference displacement set for the CaTiO3 80-atom (2x1x2 Pnma) force constants.
usage: python -I build_dispersion_v1.py s0_POSCAR.vasp OUTDIR
Host = s0 of the Ti->Sn run (parent geometry, parent lattice). Displaced atoms = one representative per
spglib symmetry orbit; +-0.01 A along x,y,z (Cartesian). Plus 'ref' (undisplaced). INCAR as the 20-atom Hessian reference
(ENCUT 520, PREC Accurate, EDIFF 1e-8, ISMEAR 0 SIGMA 0.02, ISYM 0, LREAL F, LASPH T, ADDGRID T) with ALGO Normal, ISTART 0.
KPOINTS Gamma 2x3x2 = 20-atom 4x3x4 scaled to the 2x1x2 supercell. Writes LF only."""
import sys, json, os
import numpy as np
import spglib

src, out = sys.argv[1], sys.argv[2]
DELTA = 0.01
L = open(src, newline="").read().split("\n")
assert "\r" not in "".join(L)
lat = np.array([[float(x) for x in L[i].split()] for i in (2, 3, 4)])
el, cnt = L[5].split(), [int(x) for x in L[6].split()]
assert L[7].strip().lower().startswith("d")
n = sum(cnt)
fr = np.array([[float(x) for x in L[8 + i].split()[:3]] for i in range(n)])
sp = [e for e, c in zip(el, cnt) for _ in range(c)]
nums = {"Ca": 20, "Ti": 22, "O": 8}
ds = spglib.get_symmetry_dataset((lat, fr, [nums[s] for s in sp]), symprec=1e-4)
eq = np.array(ds.equivalent_atoms) if not isinstance(ds, dict) else np.array(ds["equivalent_atoms"])
nops = len(ds.rotations) if not isinstance(ds, dict) else len(ds["rotations"])
reps = sorted(set(eq.tolist()))
print("n_ops", nops, "orbits", len(reps), "reps", reps, [sp[i] for i in reps])

INCAR = """SYSTEM = CaTiO3 80-atom 2x1x2 finite-difference force constants
ENCUT = 520
PREC = Accurate
EDIFF = 1E-8
NELM = 200
NELMIN = 5
ALGO = Normal
ISMEAR = 0
SIGMA = 0.02
IBRION = -1
NSW = 0
ISYM = 0
LREAL = .FALSE.
LASPH = .TRUE.
ADDGRID = .TRUE.
ISTART = 0
ICHARG = 2
LWAVE = .FALSE.
LCHARG = .FALSE.
"""
KPT = "Gamma force mesh 2x3x2 (20-atom 4x3x4 scaled to 2x1x2)\n0\nGamma\n2 3 2\n0 0 0\n"


def poscar(title, f):
    s = f"{title}\n1.0\n" + "".join("  %.12f  %.12f  %.12f\n" % tuple(v) for v in lat)
    s += "  ".join(el) + "\n" + "  ".join(map(str, cnt)) + "\nDirect\n"
    return s + "".join("  %.12f  %.12f  %.12f\n" % tuple(v) for v in f)


os.makedirs(out, exist_ok=True)
inv = np.linalg.inv(lat)
jobs = ["ref"]
cases = {"ref": fr}
for i in reps:
    for a in range(3):
        for sgn, tag in ((1, "p"), (-1, "m")):
            c = fr @ lat
            c[i, a] += sgn * DELTA
            nm = f"d{i:02d}{'xyz'[a]}{tag}"
            cases[nm] = c @ inv
            jobs.append(nm)
for nm, f in cases.items():
    d = os.path.join(out, nm)
    os.makedirs(d, exist_ok=True)
    for fn, txt in (("POSCAR", poscar(f"CaTiO3 80 {nm}", f)), ("INCAR", INCAR), ("KPOINTS", KPT)):
        with open(os.path.join(d, fn), "w", newline="\n") as fh:
            fh.write(txt)
with open(os.path.join(out, "joblist.txt"), "w", newline="\n") as fh:
    fh.write("\n".join(jobs) + "\n")
json.dump({"delta_A": DELTA, "reps": reps, "rep_species": [sp[i] for i in reps], "equivalent_atoms": eq.tolist(),
           "jobs": jobs, "n_atoms": n}, open(os.path.join(out, "meta.json"), "w"), indent=1)
print(len(jobs), "jobs")
