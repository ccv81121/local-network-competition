"""Convex hulls at 0 GPa (E) and 5 GPa (H = E + PV) from the static energies in runs/ (runs/P0, runs/P50 = 50 kbar).

Reports E_hull of k12 and S01-S14 with decomposition, mixing energy against the two spinel end members, and the ideal
B-site mixing entropy -TS (per atom) at 300 and 1000 K. Missing competitors are listed, not silently ignored.
The VASP runs in runs/ (OUTCAR, CONTCAR, DONE) are not deposited; hull_result.json is the result.

usage: python hull.py
"""
import json, re, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Structure

OUT_DIR = Path(__file__).resolve().parent
EV_A3_PER_GPA = 0.00624150907
PATH_STRUCTURES = [f"S{i:02d}" for i in range(1, 15)] + ["k12"]


def energy(d):
    text = (d / "OUTCAR").read_text()
    return float(re.findall(r"free  energy   TOTEN  =\s+(\S+)", text)[-1])


def load(pressure_kbar):
    rows = {}
    for d in sorted((OUT_DIR / "runs" / f"P{pressure_kbar}").iterdir()):
        if not (d / "DONE").exists():
            continue
        structure = Structure.from_file(d / ("CONTCAR" if (d / "INCAR.r").exists() else "POSCAR"))
        e = energy(d)
        enthalpy = e + (5 * EV_A3_PER_GPA * structure.volume if pressure_kbar == 50 else 0.0)
        rows[d.name] = {"comp": structure.composition, "E": e, "H": enthalpy, "V": structure.volume,
                        "n": len(structure)}
    return rows


def entropy_TS(x, T):
    s = -(x * np.log(x) + (1 - x) * np.log(1 - x)) if 0 < x < 1 else 0.0
    return -8.617333e-2 * T * s * 2 / 7  # meV per atom; B sites are 2/7 of atoms


hulls = {}
for pressure_kbar in (0, 50):
    rows = load(pressure_kbar)
    missing = [n for n in ("Mg", "Sc", "Sb", "Se", "MgSe", "ScSe", "ScSb", "Mg3Sb2", "Sb2Se3", "MgSc2Se4", "MgSb2Se4",
                           "Sc2Se3")
               if not any(k == n or k.startswith(n + "_") for k in rows)]
    competitor_entries = [PDEntry(r["comp"], r["H"], name=n) for n, r in rows.items() if n not in PATH_STRUCTURES]
    phase_diagram = PhaseDiagram(competitor_entries)
    out = {"missing_competitors": missing, "stable_competitors": sorted(e.name for e in phase_diagram.stable_entries)}
    end_sc = rows.get("MgSc2Se4")
    end_sb = rows.get("MgSb2Se4")
    for n in PATH_STRUCTURES:
        if n not in rows:
            continue
        r = rows[n]
        e = PDEntry(r["comp"], r["H"], name=n)
        dec, eh = phase_diagram.get_decomp_and_e_above_hull(e, allow_negative=True)
        x = r["comp"]["Sb"] / (r["comp"]["Sb"] + r["comp"]["Sc"])
        row = {"x_Sb": x, "E_hull_meV_per_atom": 1e3 * eh, "decomposition": {k.name: float(v) for k, v in dec.items()},
               "minus_TS_300K_meV": entropy_TS(x, 300), "minus_TS_1000K_meV": entropy_TS(x, 1000)}
        if end_sc and end_sb:
            ea, eb = end_sc["H"] / end_sc["n"], end_sb["H"] / end_sb["n"]
            row["E_mix_meV_per_atom"] = 1e3 * (r["H"] / r["n"] - ((1 - x) * ea + x * eb))
        out[n] = row
    for n in ("MgSc2Se4", "MgSb2Se4"):
        if n in rows:
            out[f"{n}_E_hull_meV_per_atom"] = 1e3 * phase_diagram.get_e_above_hull(
                PDEntry(rows[n]["comp"], rows[n]["H"], name=n), allow_negative=True)
    hulls[f"P{pressure_kbar}"] = out
(OUT_DIR / "hull_result.json").write_text(json.dumps(hulls, indent=1, default=float))
print(json.dumps(hulls, indent=1, default=float))
