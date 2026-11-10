"""Read-only: displacement error implied by final residual forces of the P0/P5 relaxations,
dq = Hs^-1 S^T r (symmetric sector).

For every argument WORK_DIR,OUTCAR_P0,OUTCAR_P5: WORK_DIR holds meta.json, S.npy and Hs.npy of the sector force
balance; the OUTCARs are the endpoint relaxations (not deposited). Prints |u|, |du| = |S Hs^-1 S^T (r5 - r0)|, their ratio and the
largest residual force components.
usage: python residual_force_sensitivity.py WORK_DIR,OUTCAR_P0,OUTCAR_P5 [...]
"""
import numpy as np, json, sys
from pathlib import Path


def forces(path):
    text = Path(path).read_text().split("TOTAL-FORCE (eV/Angst)")[-1].splitlines()[2:]
    out = []
    for line in text:
        fields = line.split()
        if len(fields) < 6:
            break
        out.append([float(x) for x in fields[3:6]])
    return np.array(out).ravel()


for work_dir, outcar_p0, outcar_p5 in [a.split(",") for a in sys.argv[1:]]:
    work_dir = Path(work_dir)
    meta = json.loads((work_dir / "meta.json").read_text())
    S = np.load(work_dir / "S.npy")
    Hs = np.load(work_dir / "Hs.npy")
    lattice = np.array(meta["lattice_P2"])
    dfrac = np.array(meta["frac_P2"]) - np.array(meta["frac_start"])
    dfrac -= np.round(dfrac)
    u = (dfrac @ lattice).ravel()
    r0, r5 = forces(outcar_p0), forces(outcar_p5)
    du = S @ np.linalg.solve(Hs, S.T @ r5) - S @ np.linalg.solve(Hs, S.T @ r0)
    print(work_dir.parts[-2],
          "|u|=%.4f A  |du|=%.4f A  ratio=%.3f  Fmax(P0,P5)=%.4f,%.4f  S orthonormal err=%.1e"
          % (np.linalg.norm(u), np.linalg.norm(du), np.linalg.norm(du) / np.linalg.norm(u),
             np.abs(r0).max(), np.abs(r5).max(), np.abs(S.T @ S - np.eye(S.shape[1])).max()))
