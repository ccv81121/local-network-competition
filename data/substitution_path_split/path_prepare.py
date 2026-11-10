"""Inputs for the basis-free split of band-edge margin changes along the two paths, DFT statics only.
Composition path S01..S14 (CHGNet-relaxed 56-atom cells, nested: S(n+1) = S(n) with one more Sc->Sb):
  chem  = species of S(n+1) on the S(n) geometry (lattice and fractional coordinates of S(n))
  aff   = species of S(n+1), fractional coordinates of S(n) in the lattice of S(n+1)
  end   = S(n+1) itself (recomputed with ISYM=0)
Pressure path (tight DFT relaxations 0,2,3,4,5,6,7.5 GPa):
  aff   = fractional coordinates of P(i) in the lattice of P(i+1); ends reuse static_k4 (same settings, ISYM=0)
Reads inputs/ (CHGNet-relaxed S01..S14 structures, tight 0..7.5-GPa relaxed structures, static INCAR of the tight run);
the POTCAR is read from the tight-spinel work folder on the cluster (licensed, not deposited).
usage: python path_prepare.py OUTROOT   (writes inputs and MAPPING.json; checks the atom correspondence)"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

OUT_ROOT = Path(sys.argv[1])
OUT_ROOT.mkdir(parents=True, exist_ok=True)
INPUT_DIR = Path(__file__).resolve().parent / "inputs"
POTCAR = Path("$WORK/PRL_TIGHT_SPINEL_20261015/Spinel/POTCAR")
ORDER = ["Mg", "Sb", "Sc", "Se"]
COMP_INCAR = """SYSTEM = composition path split static
ISTART = 0
ICHARG = 2
ENCUT = 520
PREC = Accurate
EDIFF = 1E-5
NSW = 0
IBRION = -1
ISMEAR = 0
SIGMA = 0.05
ISPIN = 1
LREAL = Auto
ALGO = Normal
NELM = 160
LASPH = .TRUE.
ADDGRID = .TRUE.
LWAVE = .FALSE.
LCHARG = .FALSE.
ISYM = 0
"""
KPT = "Gamma\n0\nGamma\n4 4 4\n0 0 0\n"


def read(p):
    L = Path(p).read_text().splitlines()
    s = float(L[1].split()[0])
    lat = np.array([[float(x) for x in L[i].split()[:3]] for i in (2, 3, 4)]) * s
    el = L[5].split()
    n = [int(x) for x in L[6].split()]
    i = 7
    if L[i].strip()[0] in "sS":
        i += 1
    fr = np.array([[float(x) for x in L[i + 1 + k].split()[:3]] for k in range(sum(n))])
    return lat, [e for e, c in zip(el, n) for _ in range(c)], fr


def write(d, title, lat, sp, fr, incar, potcar=POTCAR, kpt=KPT):
    d.mkdir(parents=True, exist_ok=True)
    idx = sorted(range(len(sp)), key=lambda i: (ORDER.index(sp[i]), i))
    el = [e for e in ORDER if e in sp]
    L = ([title, "1.0"] + ["  ".join(f"{x:.12f}" for x in v) for v in lat]
         + ["  ".join(el), "  ".join(str(sp.count(e)) for e in el), "Direct"])
    L += ["  ".join(f"{x:.12f}" for x in fr[i]) for i in idx]
    (d / "POSCAR").write_text("\n".join(L) + "\n")
    (d / "INCAR").write_text(incar)
    (d / "KPOINTS").write_text(kpt)
    shutil.copy(potcar, d / "POTCAR")


meta = {"composition": [], "pressure": []}
STRUCTURES = {n: read(INPUT_DIR / f"s{n:02d}_contcar.vasp") for n in range(1, 15)}
for n in range(1, 15):
    lat, sp, fr = STRUCTURES[n]
    write(OUT_ROOT / "comp" / f"S{n:02d}" / "end", f"S{n:02d} end", lat, sp, fr, COMP_INCAR)
for n in range(1, 14):
    (l0, s0, f0), (l1, s1, f1) = STRUCTURES[n], STRUCTURES[n + 1]
    # map every atom of S(n+1) to its counterpart in S(n) by minimum-image fractional distance
    # (cartesian in the S(n) lattice)
    d = f1[:, None, :] - f0[None, :, :]
    d -= np.round(d)
    dist = np.linalg.norm(d @ l0, axis=2)
    m = dist.argmin(axis=1)
    dmin = dist[np.arange(len(m)), m]
    second = np.sort(dist, axis=1)[:, 1]
    assert len(set(m)) == len(m), "mapping not one-to-one"
    changed = [i for i in range(len(m)) if s1[i] != s0[m[i]]]
    assert len(changed) == 1 and s0[m[changed[0]]] == "Sc" and s1[changed[0]] == "Sb", changed
    symbols_chem = list(s0)
    symbols_chem[m[changed[0]]] = "Sb"  # S(n) geometry, S(n+1) species
    base = OUT_ROOT / "comp" / f"step_{n:02d}_{n + 1:02d}"
    write(base / "chem", f"S{n:02d}->S{n + 1:02d} chem", l0, symbols_chem, f0, COMP_INCAR)
    write(base / "aff", f"S{n:02d}->S{n + 1:02d} aff", l1, symbols_chem, f0, COMP_INCAR)
    u = (f1 - f0[m])
    u -= np.round(u)
    meta["composition"].append({"step": f"S{n:02d}->S{n + 1:02d}",
                                "substituted_site_frac_Sn": f0[m[changed[0]]].tolist(),
                                "map_max_disp_A": float(dmin.max()), "map_min_second_A": float(second.min()),
                                "internal_disp_rms_A": float(np.sqrt(np.mean(np.sum((u @ l1) ** 2, axis=1)))),
                                "volume_change_pct": float(100 * (abs(np.linalg.det(l1))
                                                                  / abs(np.linalg.det(l0)) - 1))})
PRESSURE_LABELS = ["0", "2", "3", "4", "5", "6", "7.5"]
PRESSURE_INCAR = (INPUT_DIR / "static_k4_p4_incar").read_text().replace(
    "SYSTEM = tight path P=4 k4", "SYSTEM = pressure path affine state")
for a, b in zip(PRESSURE_LABELS[:-1], PRESSURE_LABELS[1:]):
    la, sa, fa = read(INPUT_DIR / f"p{a.replace('.', '_')}_contcar.vasp")
    lb, sb, fb = read(INPUT_DIR / f"p{b.replace('.', '_')}_contcar.vasp")
    assert sa == sb
    write(OUT_ROOT / "press" / f"step_P{a}_P{b}" / "aff", f"P{a}->P{b} aff", lb, sa, fa, PRESSURE_INCAR)
    u = fb - fa
    u -= np.round(u)
    meta["pressure"].append({"step": f"P{a}->P{b}",
                             "internal_disp_rms_A": float(np.sqrt(np.mean(np.sum((u @ lb) ** 2, axis=1)))),
                             "volume_change_pct": float(100 * (abs(np.linalg.det(lb))
                                                               / abs(np.linalg.det(la)) - 1))})
(OUT_ROOT / "MAPPING.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
