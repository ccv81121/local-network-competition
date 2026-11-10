"""Wannier inputs for the tight 5-GPa x = 0.75 structure (four-state state 11).

Built from the earlier pressure-set basis (PRL_MECHANISM_TEST_20261014/Wannier_Competition/basis_test) by moving every
projection centre and basis site to the same atom in the new structure. The energy windows stay placeholders
(@dis_win_min@ and the like); the server fills them from the SCF eigenvalues (fill_windows.py).
REFDIR holds old.POSCAR, state11.POSCAR, basis_map.json, edge12.win and mlwf_std.win; OUTDIR receives basis_map.json,
edge12.win.tmpl (num_iter 0) and mlwf.win.tmpl (num_iter 200).

REFDIR is a cluster folder (not deposited; wannier/basis_map.json is the resulting basis).
usage: python make_inputs.py REFDIR OUTDIR
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ref_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)


def poscar(p):
    lines = p.read_text().splitlines()
    scale = float(lines[1].split()[0])
    A = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]) * scale
    species, counts = lines[5].split(), [int(x) for x in lines[6].split()]
    i0 = 8 if lines[7].strip()[0] in "Dd" else 9
    F = np.array([[float(x) for x in lines[i0 + i].split()[:3]] for i in range(sum(counts))])
    return A, sum(([e] * k for e, k in zip(species, counts)), []), F


A0, elem_old, F0 = poscar(ref_dir / "old.POSCAR")
A1, elem_new, F1 = poscar(ref_dir / "state11.POSCAR")
assert elem_old == elem_new, "species order differs"
shift = F1 - F0
shift -= np.round(shift)
disp = np.linalg.norm(shift @ A1, axis=1)
print("atom-by-atom displacement old -> tight: max %.3f A, mean %.3f A" % (disp.max(), disp.mean()))
assert disp.max() < 0.2


def atom_of(f):
    delta = F0 - np.array(f)
    delta -= np.round(delta)
    dist = np.linalg.norm(delta @ A0, axis=1)
    i = int(np.argmin(dist))
    assert dist[i] < 1e-3, dist[i]
    return i


def new_frac(f):
    i = atom_of(f)
    g = F1[i].copy()
    g += np.round(np.array(f) - g)  # keep the old periodic image
    return g


basis_map = json.loads((ref_dir / "basis_map.json").read_text())
for b in basis_map["basis"]:
    b["frac"] = [float(x) for x in new_frac(b["frac"])]
basis_map["structure"] = "tight 5-GPa x = 0.75 (four-state state 11), PRL_TIGHT_SPINEL_20261015"
(out_dir / "basis_map.json").write_text(json.dumps(basis_map, indent=1))

PROJECTION_LINE = re.compile(r"^(\s*f=)([-\d.eE+]+),([-\d.eE+]+),([-\d.eE+]+)(:.*)$")


def remap_win(src, num_iter):
    lines = []
    for line in (ref_dir / src).read_text().splitlines():
        if line.startswith("num_bands") or line.startswith("mp_grid") or line.startswith("begin unit_cell_cart"):
            break  # VASP writes cell, atoms and k points itself
        match = PROJECTION_LINE.match(line)
        if match:
            new_f = new_frac([float(match.group(i)) for i in (2, 3, 4)])
            line = "%s%.12f,%.12f,%.12f%s" % (match.group(1), new_f[0], new_f[1], new_f[2], match.group(5))
        line = re.sub(r"^num_iter\s*=.*", "num_iter = %d" % num_iter, line)
        line = re.sub(r"^(dis_(win|froz)_(min|max))\s*=.*", r"\1 = @\1@", line)
        lines.append(line)
    return "\n".join(lines) + "\n"


(out_dir / "edge12.win.tmpl").write_text(remap_win("edge12.win", 0))
(out_dir / "mlwf.win.tmpl").write_text(remap_win("mlwf_std.win", 200))
for f in ("edge12.win.tmpl", "mlwf.win.tmpl"):
    t = (out_dir / f).read_text()
    print(f, t.count("f="), "projection lines;", t.count("@dis_"), "window placeholders")
