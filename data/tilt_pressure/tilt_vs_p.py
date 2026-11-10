"""B-O-B angles, mean B-O bond length and volume per formula unit of CaTiO3 and CaSnO3 at 0-20 GPa.

Inputs:  inputs/<material>/p<pressure>_contcar, the relaxed VASP structures (CONTCAR) at P = 0, 5, 10, 15, 20 GPa
         for material = ca_tio3, ca_sno3 (10 files).
Output:  one line per material and pressure on stdout (committed as tilt_vs_p.txt).
usage:   python tilt_vs_p.py   (no arguments)
"""
import numpy as np, itertools, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf8")
INPUT_DIR = Path(__file__).resolve().parent / "inputs"


def read(p):
    lines = open(p).read().split("\n")
    s = float(lines[1])
    lat = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)]) * s
    el = lines[5].split()
    n = [int(x) for x in lines[6].split()]
    i0 = 8 if lines[7].strip()[0] in "Dd" else 9
    fr = np.array([[float(x) for x in lines[i0 + k].split()[:3]] for k in range(sum(n))])
    sp = sum([[e] * c for e, c in zip(el, n)], [])
    return lat, fr, sp


def b_o_b_angles(lat, fr, sp, b_element):
    b_sites = [i for i, s in enumerate(sp) if s == b_element]
    o_sites = [i for i, s in enumerate(sp) if s == "O"]
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    angles, bond_lengths = [], []
    for o in o_sites:
        neighbours = []
        for b in b_sites:
            d = (fr[b] + shifts - fr[o]) @ lat
            r = np.linalg.norm(d, axis=1)
            for k in np.where(r < 2.4)[0]:
                neighbours.append(d[k])
                bond_lengths.append(r[k])
        if len(neighbours) == 2:
            c = neighbours[0] @ neighbours[1] / np.linalg.norm(neighbours[0]) / np.linalg.norm(neighbours[1])
            angles.append(np.degrees(np.arccos(c)))
    return np.array(angles), np.array(bond_lengths), abs(np.linalg.det(lat))


for mat, folder, b_element in (("CaTiO3", "ca_tio3", "Ti"), ("CaSnO3", "ca_sno3", "Sn")):
    print(mat)
    for pressure_gpa in (0, 5, 10, 15, 20):
        lat, fr, sp = read(INPUT_DIR / folder / f"p{pressure_gpa:02d}_contcar")
        a, b, V = b_o_b_angles(lat, fr, sp, b_element)
        u = sorted(set(np.round(a, 2)))
        print(f"  P={pressure_gpa:2d} GPa  V/f.u.={V/(len(sp)/5):6.2f}  B-O-B angles {u}  mean {a.mean():.2f}  "
              f"<B-O> {b.mean():.4f}")
