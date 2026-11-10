"""Readout inputs for the four-state structures.

pbe/<i>/<state>     PBE, Gamma 12x12x3, ISMEAR -5, LORBIT 11; 00 from scratch, the other states restart from 00.
pbe_u4/<i>/<state>  PBE+U (Dudarev, U = 4 eV on Ni) for 00, 10, 01, 11; same restart scheme.
pbe_k19/p2/<state>  PBE at Gamma 19x19x5 for 00 and 11 (mesh convergence), from scratch.
Intervals are lower-case directories, state 11La is directory 11_la.
Writes INCAR/KPOINTS (LF) next to the POSCARs from four_state_build.py and the queues jobs_ref.txt, jobs_rest.txt.
"""
import shutil
from pathlib import Path

HERE = Path(__file__).parent
FOUR_STATE_DIR = HERE / "four_state"
INCAR_BASE = """PREC = Accurate
ENCUT = 520
EDIFF = 1E-7
ISMEAR = -5
LREAL = .FALSE.
LASPH = .TRUE.
NELM = 200
IBRION = -1
NSW = 0
LORBIT = 11
NEDOS = 3001
LCHARG = .FALSE.
"""
INCAR_SCRATCH = "ALGO = Normal\nLWAVE = .TRUE.\n"
INCAR_RESTART = "ALGO = Fast\nISTART = 1\nLWAVE = .FALSE.\n"
INCAR_PLUS_U = "LDAU = .TRUE.\nLDAUTYPE = 2\nLDAUL = -1 2 -1\nLDAUU = 0 4 0\nLDAUJ = 0 0 0\nLMAXMIX = 4\n"
STATES = ["00", "10", "01", "11", "11La", "01ip", "01ap"]
STATE_DIRECTORY = {"11La": "11_la"}  # directory name of a state on disk
U_STATES = ["00", "10", "01", "11"]


def put(d, title, incar, mesh):
    (d / "INCAR").write_bytes(f"SYSTEM = {title}\n{incar}".encode())
    (d / "KPOINTS").write_bytes(f"Gamma\n0\nGamma\n{mesh}\n0 0 0\n".encode())


def main():
    scratch_jobs, restart_jobs = [], []
    for interval in ["P1", "P2", "S1", "S2"]:
        for state in STATES:
            state_dir = FOUR_STATE_DIR / "pbe" / interval.lower() / STATE_DIRECTORY.get(state, state)
            put(state_dir, f"La3Ni2O7 readout {interval} {state}",
                INCAR_BASE + (INCAR_SCRATCH if state == "00" else INCAR_RESTART), "12 12 3")
            (scratch_jobs if state == "00" else restart_jobs).append(f"pbe/{interval.lower()}/{state_dir.name}")
        for state in U_STATES:
            state_dir = FOUR_STATE_DIR / "pbe_u4" / interval.lower() / state
            state_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(FOUR_STATE_DIR / "pbe" / interval.lower() / state / "POSCAR", state_dir / "POSCAR")
            put(state_dir, f"La3Ni2O7 readout U4 {interval} {state}",
                INCAR_BASE + INCAR_PLUS_U + (INCAR_SCRATCH if state == "00" else INCAR_RESTART), "12 12 3")
            (scratch_jobs if state == "00" else restart_jobs).append(f"pbe_u4/{interval.lower()}/{state}")
    k19_jobs = []
    for state in ["00", "11"]:
        state_dir = FOUR_STATE_DIR / "pbe_k19" / "p2" / state
        state_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(FOUR_STATE_DIR / "pbe" / "p2" / state / "POSCAR", state_dir / "POSCAR")
        put(state_dir, f"La3Ni2O7 readout k19 P2 {state}", INCAR_BASE + "ALGO = Normal\nLWAVE = .FALSE.\n", "19 19 5")
        k19_jobs.append(f"pbe_k19/p2/{state}")
    (FOUR_STATE_DIR / "jobs_ref.txt").write_bytes(("\n".join(scratch_jobs) + "\n").encode())
    # long k19 runs first
    (FOUR_STATE_DIR / "jobs_rest.txt").write_bytes(("\n".join(k19_jobs + restart_jobs) + "\n").encode())
    print(len(scratch_jobs), "from scratch,", len(k19_jobs) + len(restart_jobs), "after them")


if __name__ == "__main__":
    main()
