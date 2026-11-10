"""Tilt-free starting structure (Fmmm reference) from the Amam cell of make_poscar.py: the y coordinates of La, Ni and
the apical O set to 1/4 (no octahedral tilt), and the two in-plane O heights set to their mean. Relaxed under the
same settings as the Amam endpoints to give H(Fmmm) - H(Amam) at 4 and 8 GPa. Writes poscar_fmmm_start."""
from pathlib import Path

src = Path(__file__).with_name("make_poscar.py").read_text(encoding="utf8")
zin = (0.10442 + 0.08837) / 2
src = src.replace('"La1": (0.25, 0.2534, 0.5), "La2": (0.25, 0.2599, 0.31886), "Ni1": (0.25, 0.2548, 0.09568)',
                  '"La1": (0.25, 0.25, 0.5), "La2": (0.25, 0.25, 0.31886), "Ni1": (0.25, 0.25, 0.09568)')
src = src.replace(
    '"O1": (0.25, 0.2938, 0.0), "O2": (0.25, 0.2189, 0.2038), "O3": (0.0, 0.5, 0.10442), "O4": (0.5, 0.0, 0.08837)',
    f'"O1": (0.25, 0.25, 0.0), "O2": (0.25, 0.25, 0.2038), "O3": (0.0, 0.5, {zin:.5f}), "O4": (0.5, 0.0, {zin:.5f})')
src = src.replace('with_name("poscar_exp")', 'with_name("poscar_fmmm_start")').replace(
    "Amam exp", "Fmmm start (tilt-free)")
assert src.count("poscar_fmmm_start") == 1
# make_poscar.py with the patched site table is run as it stands, so both structures come from the same code
exec(compile(src, "make_fmmm", "exec"), {"__file__": __file__})
