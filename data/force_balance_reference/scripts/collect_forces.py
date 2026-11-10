"""Accept a finite-difference point only if VASP finished normally and the SCF met EDIFF; store forces.
Reads vasp_exit_code.txt, OSZICAR and OUTCAR of one run (raw VASP output, not deposited; forces/ and results/ hold the
extracted values); writes forces.npy (56 x 3, eV/A) and result.json there.
usage: python collect_forces.py RUN_DIR"""
import json, re, sys
from pathlib import Path
import numpy as np

case_dir = Path(sys.argv[1])
assert (case_dir / "vasp_exit_code.txt").read_text().strip() == "0", "vasp exit"
osz = (case_dir / "OSZICAR").read_text().splitlines()
steps = [line for line in osz if re.match(r"^\s*(DAV|RMM|CG):", line)]
last = steps[-1].split()
n_el, dE = int(last[1]), abs(float(last[3]))
assert n_el < 180 and n_el >= 5 and dE < 1e-7, f"SCF not converged: n={n_el} dE={dE}"
outcar_text = (case_dir / "OUTCAR").read_text()
force_block = outcar_text[outcar_text.rindex("TOTAL-FORCE (eV/Angst)"):].splitlines()[2:]
forces = []
for line in force_block:
    if line.startswith(" ---"):
        break
    forces.append([float(x) for x in line.split()[3:6]])
forces = np.array(forces)
assert forces.shape == (56, 3)
energy = float(re.findall(r"free  energy   TOTEN  =\s+(-?\d+\.\d+)", outcar_text)[-1])
np.save(case_dir / "forces.npy", forces)
(case_dir / "result.json").write_text(json.dumps({"n_elec_steps": n_el, "last_dE": dE, "E_eV": energy,
                                                  "max_force": float(np.abs(forces).max())}))
print("ok", n_el, dE)
