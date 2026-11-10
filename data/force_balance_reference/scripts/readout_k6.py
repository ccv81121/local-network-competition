"""K6 fixed-ion readout for force-balance check structures; same margin definitions as the four-state collector.
Reads OSZICAR, OUTCAR, EIGENVAL (Gamma-centred 6x6x6 mesh) and vasp_exit_code.txt of one run (EIGENVAL and OUTCAR not
deposited); writes readout.json there:
convergence, total energy, Lambda_C = E_C(1/2,1/2,1/2) - E_C(Gamma), Lambda_V = E_V(Gamma) - E_V(0,0,1/6) in meV and the
k-points of the band edges.
usage: python readout_k6.py CASE_DIR"""
import json, re, sys
from pathlib import Path

case = Path(sys.argv[1]).resolve()
oszicar = [line.split() for line in (case / "OSZICAR").read_text().splitlines()
          if line.lstrip().startswith(("DAV:", "RMM:"))]
n_steps, dE, dB = int(oszicar[-1][1]), float(oszicar[-1][3]), float(oszicar[-1][4])
outcar_text = (case / "OUTCAR").read_text(errors="replace")
ok = int((case / "vasp_exit_code.txt").read_text()) == 0 and 5 <= n_steps < 180 and abs(dE) < 1e-7 and abs(dB) < 1e-7 \
    and "General timing and accounting" in outcar_text
eigenval_lines = (case / "EIGENVAL").read_text().splitlines()
nelect, nk, n_bands = map(int, eigenval_lines[5].split())
assert nelect == 312 and n_bands == 172
nocc = nelect // 2
pts = []
i = 6
for _ in range(nk):
    while not eigenval_lines[i].strip():
        i += 1
    k = list(map(float, eigenval_lines[i].split()[:3]))
    i += 1
    bands = [list(map(float, eigenval_lines[i + j].split())) for j in range(n_bands)]
    pts.append({"k": k, "EV": bands[nocc - 1][1], "EC": bands[nocc][1]})
    i += n_bands


def same_k(a, b):
    return all(abs((x - y) - round(x - y)) < 1e-6 for x, y in zip(a, b))


def point_at(k):
    return [p for p in pts if same_k(p["k"], k) or same_k([-x for x in p["k"]], k)][0]


gamma = point_at([0, 0, 0])
readout = {"case": case.name, "converged": ok, "n_elec_steps": n_steps, "last_dE": dE,
           "E_eV": float(re.findall(r"free\s+energy\s+TOTEN\s*=\s*([-+\d.Ee]+)", outcar_text)[-1]),
           "Lambda_C_meV": 1000 * (point_at([0.5, 0.5, 0.5])["EC"] - gamma["EC"]),
           "Lambda_V_meV": 1000 * (gamma["EV"] - point_at([0, 0, 1 / 6])["EV"]),
           "VBM_k": max(pts, key=lambda p: p["EV"])["k"], "CBM_k": min(pts, key=lambda p: p["EC"])["k"]}
(case / "readout.json").write_text(json.dumps(readout, indent=1))
print(json.dumps(readout))
