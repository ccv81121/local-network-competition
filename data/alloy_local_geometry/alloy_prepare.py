"""Alloying (local view): single isovalent substitution in a fixed parent cell; builds s0, s1 and relax inputs.
Substituted site = lowest-index site of that element in the parent POSCAR.
Reads the parent structures from inputs/; the POTCAR library (/data/vasp) of the cluster is licensed and not deposited.
usage: python alloy_prepare.py ROOT"""
import json
import sys
from pathlib import Path

import numpy as np

INPUT_DIR = Path(__file__).resolve().parent / "inputs"
POTCAR_LIBRARY = Path("/data/vasp/potpaw_PBE")
POTCAR_LABEL = {"Mg": "Mg", "Sb": "Sb", "Sc": "Sc_sv", "Se": "Se", "Ca": "Ca_sv", "Ti": "Ti_sv", "Sn": "Sn_d", "O": "O"}
FCC2CONV = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
CASES = {
    "k12_Sb2Sc": dict(parent=INPUT_DIR / "k12_parent_contcar.vasp", M=None, src="Sb", dst="Sc",
                      A="Mg", B=["Sc", "Sb"], X="Se", KF=(3, 3, 3), KE=(4, 4, 4)),
    "k12_Sc2Sb": dict(parent=INPUT_DIR / "k12_parent_contcar.vasp", M=None, src="Sc", dst="Sb",
                      A="Mg", B=["Sc", "Sb"], X="Se", KF=(3, 3, 3), KE=(4, 4, 4)),
    "MgSc2Se4_Sc2Sb": dict(parent=INPUT_DIR / "mg_sc2se4_parent_contcar.vasp", M=FCC2CONV,
                           src="Sc", dst="Sb", A="Mg", B=["Sc", "Sb"], X="Se", KF=(3, 3, 3), KE=(4, 4, 4)),
    "CaTiO3_Ti2Sn": dict(parent=INPUT_DIR / "ca_tio3_parent_contcar.vasp",
                         M=[[2, 0, 0], [0, 1, 0], [0, 0, 2]], src="Ti", dst="Sn", A="Ca", B=["Ti", "Sn"], X="O",
                         KF=(2, 3, 2), KE=(2, 4, 2)),
}
STATIC_INCAR = """ENCUT = 520
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
LWAVE = .FALSE.
LCHARG = .FALSE.
"""
RELAX_INCAR = """PREC = Accurate
ENCUT = 520
EDIFF = 1E-8
EDIFFG = -1E-3
IBRION = 2
ISIF = 2
NSW = 200
ISMEAR = 0
SIGMA = 0.02
LREAL = .FALSE.
LASPH = .TRUE.
LWAVE = .FALSE.
LCHARG = .FALSE.
NELM = 150
"""


def read_poscar(path):
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1].split()[0])
    lat = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]) * scale
    el = lines[5].split()
    n = [int(x) for x in lines[6].split()]
    i = 7
    if lines[i].strip()[0] in "sS":
        i += 1
    assert lines[i].strip()[0] in "dD", "Direct coordinates expected"
    fr = np.array([[float(x) for x in lines[i + 1 + k].split()[:3]] for k in range(sum(n))])
    return lat, [e for e, c in zip(el, n) for _ in range(c)], fr


def write_poscar(path, title, lat, sp, fr, order):
    idx = sorted(range(len(sp)), key=lambda i: (order.index(sp[i]), i))
    el = [e for e in order if e in sp]
    n = [sp.count(e) for e in el]
    out = ([title, "1.0"] + ["  ".join(f"{x:.12f}" for x in v) for v in lat]
           + ["  ".join(el), "  ".join(map(str, n)), "Direct"])
    out += ["  ".join(f"{x:.12f}" for x in fr[i]) for i in idx]
    Path(path).write_text("\n".join(out) + "\n")
    return [sp[i] for i in idx], fr[idx], el


def supercell(lat, sp, fr, M):
    M = np.array(M)
    Mi = np.linalg.inv(M)
    new, nsp = [], []
    rng = range(-3, 4)
    for t in ((a, b, c) for a in rng for b in rng for c in rng):
        g = (fr + np.array(t)) @ Mi
        for s, x in zip(sp, g):
            x = x - np.floor(x + 1e-8)
            if any(np.allclose(((x - y) + 0.5) % 1 - 0.5, 0, atol=1e-5) for y in new):
                continue
            new.append(x)
            nsp.append(s)
    assert len(new) == round(abs(np.linalg.det(M))) * len(sp)
    return M @ lat, nsp, np.array(new)


def potcar(path, el):
    Path(path).write_text("".join((POTCAR_LIBRARY / POTCAR_LABEL[e] / "POTCAR").read_text() for e in el))


def main():
    root = Path(sys.argv[1])
    work_dir = root / "Alloy_Local"
    work_dir.mkdir(exist_ok=True)
    for name, case in CASES.items():
        lat, sp, fr = read_poscar(case["parent"])
        if case["M"] is not None:
            lat, sp, fr = supercell(lat, sp, fr, case["M"])
        order0 = list(dict.fromkeys(sp))
        order1 = (order0 if case["dst"] in order0
                  else order0[:order0.index(case["src"]) + 1] + [case["dst"]] + order0[order0.index(case["src"]) + 1:])
        case_dir = work_dir / name
        # parent in its own order (sorted), then substitute the lowest-index src site of the sorted parent
        for sub in ("s0", "s1", "relax"):
            (case_dir / sub).mkdir(parents=True, exist_ok=True)
        sp0, fr0, el0 = write_poscar(case_dir / "s0" / "POSCAR", f"{name} s0 parent", lat, sp, fr, order0)
        isub = sp0.index(case["src"])
        sp1 = list(sp0)
        sp1[isub] = case["dst"]
        sp1s, fr1s, el1 = write_poscar(case_dir / "s1" / "POSCAR", f"{name} s1 substituted, parent geometry", lat, sp1,
                                       fr0, order1)
        write_poscar(case_dir / "relax" / "POSCAR", f"{name} relax start", lat, sp1, fr0, order1)
        for sub, el in (("s0", el0), ("s1", el1), ("relax", el1)):
            potcar(case_dir / sub / "POTCAR", el)
        for sub in ("s0", "s1"):
            (case_dir / sub / "INCAR").write_text(f"SYSTEM = {name} {sub}\n" + STATIC_INCAR)
            (case_dir / sub / "KPOINTS").write_text("Gamma edge mesh\n0\nGamma\n{} {} {}\n0 0 0\n".format(*case["KE"]))
        (case_dir / "relax" / "INCAR").write_text(f"SYSTEM = {name} relax ISIF=2\n" + RELAX_INCAR)
        (case_dir / "relax" / "KPOINTS").write_text("Gamma force mesh\n0\nGamma\n{} {} {}\n0 0 0\n".format(*case["KF"]))
        # substituted atom index in the s1 ordering
        order_idx = sorted(range(len(sp1)), key=lambda i: (order1.index(sp1[i]), i))
        isub1 = order_idx.index(isub)
        meta = {"case": name, "parent": "inputs/" + case["parent"].name, "supercell": case["M"], "src": case["src"],
                "dst": case["dst"], "sub_index_parent_sorted": isub, "sub_index_s1": isub1, "A": case["A"],
                "B": case["B"], "X": case["X"], "KF": case["KF"], "KE": case["KE"], "order_s0": el0,
                "order_s1": el1, "n_atoms": len(sp1), "lattice": lat.tolist(), "species_s1": sp1s,
                "sub_frac": fr1s[isub1].tolist()}
        (case_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        print(name, len(sp1), el0, "->", el1, "sub", case["src"], "->", case["dst"], "at", np.round(fr1s[isub1], 4))


if __name__ == "__main__":
    main()
