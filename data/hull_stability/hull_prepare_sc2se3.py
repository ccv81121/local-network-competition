"""Second set of hull inputs: Sc2Se3 candidates (top 4 of the CHGNet ranking, DFT relaxed at 0 and 5 GPa) and
re-runs of the elemental relaxations that ended with ZBRENT bracketing errors (IBRION 2 -> 1, EDIFF 1E-7; otherwise
unchanged).

Needs sc2se3/cand<k>.vasp (from sc2se3_enum.py), hull_input_hashes.json and runs/ (from hull_prepare.py): not deposited.
usage: python hull_prepare_sc2se3.py
"""
import json, shutil, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hull_prepare import OUT_DIR, write
from pymatgen.core import Structure

hashes = json.loads((OUT_DIR / "hull_input_hashes.json").read_text())
jobs = []
ranking = json.loads((OUT_DIR / "sc2se3_enum_ranking.json").read_text())
for candidate in ranking[:4]:
    structure = Structure.from_file(OUT_DIR / "sc2se3" / f"cand{candidate['k']:03d}.vasp")
    for pressure_kbar in (0, 50):
        run_dir = OUT_DIR / "runs" / f"P{pressure_kbar}" / f"Sc2Se3_c{candidate['k']:02d}"
        write(run_dir, structure, "Sc2Se3", "relax", pressure_kbar, hashes)
        jobs.append(str(run_dir.relative_to(OUT_DIR)))
for j in ("P0/Sc", "P50/Sc", "P50/Se"):
    run_dir = OUT_DIR / "runs" / j
    keep_files = {"POSCAR", "POTCAR", "INCAR.r", "INCAR.s", "KIND"}
    if (run_dir / "POSCAR.r1").exists():
        shutil.copy(run_dir / "POSCAR.r1", run_dir / "POSCAR.r1.failedrun")
    for entry in run_dir.iterdir():
        if entry.name not in keep_files and entry.name != "POSCAR.r1.failedrun":
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    incar_text = ((run_dir / "INCAR.r").read_text()
                  .replace("IBRION = 2", "IBRION = 1").replace("EDIFF = 1E-6", "EDIFF = 1E-7"))
    (run_dir / "INCAR.r").write_text(incar_text)
    jobs.append(j if j.startswith("runs") else "runs/" + j)
(OUT_DIR / "hull_input_hashes.json").write_text(json.dumps(hashes, indent=1))
(OUT_DIR / "jobs2.txt").write_text("\n".join(jobs) + "\n")
print(jobs)
