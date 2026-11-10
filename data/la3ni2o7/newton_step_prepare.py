"""One modified Newton step of the La3Ni2O7 force balance.
For each interval, u1 = B^T H^-1 B f(r_aff) from the symmetric block of force_balance_analyze.py, and
r1 = r_aff + u1 in the end lattice. One SCF at r1 (same settings as the displaced runs, restarted from ../ref/WAVECAR)
gives f(r1); newton_step_analyze.py then forms u2 = B^T H^-1 B Q f(r1) with the same H.
Writes force_balance/calculations/<i>/newton1/{POSCAR,INCAR,KPOINTS} (LF) and force_balance/jobs_newton.txt.
"""
import json
from pathlib import Path

import numpy as np
from ase.io import read, write

from force_balance_analyze import block, load
from force_balance_prepare import INCAR_DISPLACED, KPOINTS_TEXT

HERE = Path(__file__).parent
FORCE_BALANCE_DIR = HERE / "force_balance"


def predicted(name, meta, aff, ref):
    B, H, missing = block(name, meta, "")
    assert not missing, (name, missing)
    f = ref["f"] - np.repeat(ref["f"].reshape(-1, 3).mean(0)[None], len(aff), 0).ravel()
    return B, (H + H.T) / 2, B.T @ np.linalg.solve((H + H.T) / 2, B @ f)


def main():
    modes = json.loads((FORCE_BALANCE_DIR / "modes.json").read_text())
    jobs = []
    for name, meta in modes.items():
        aff = read(FORCE_BALANCE_DIR / "calculations" / name.lower() / "ref" / "POSCAR")
        _, _, u1 = predicted(name, meta, aff, load(name, "ref"))
        at = aff.copy()
        at.positions = aff.positions + u1.reshape(-1, 3)
        case_dir = FORCE_BALANCE_DIR / "calculations" / name.lower() / "newton1"
        case_dir.mkdir(parents=True, exist_ok=True)
        write(case_dir / "POSCAR", at, format="vasp", direct=True, sort=False)
        p = case_dir / "POSCAR"
        p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
        (case_dir / "INCAR").write_text(f"SYSTEM = La3Ni2O7 stage1n {name} newton1\n" + INCAR_DISPLACED, newline="\n")
        (case_dir / "KPOINTS").write_text(KPOINTS_TEXT, newline="\n")
        jobs.append(f"calculations/{name.lower()}/newton1")
        print(f"{name}: |u1| = {np.linalg.norm(u1):.4f} A, max atom shift {np.abs(u1.reshape(-1, 3)).max():.4f} A")
    (FORCE_BALANCE_DIR / "jobs_newton.txt").write_text("\n".join(jobs) + "\n", newline="\n")


if __name__ == "__main__":
    main()
