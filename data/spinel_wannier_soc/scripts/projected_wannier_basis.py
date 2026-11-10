"""Projected Wannier basis on the converged disentangled subspace (wannier/dis_v2, written by disentangle.sh).

U_k = A'_k (A'_k^+ A'_k)^(-1/2) with A'_k = U_opt,k^+ A_k, the projection-only basis that Wannier90 uses as the start of
maximal localisation (num_iter = 0). H(R) = (1/N_k) sum_k exp(-2 pi i k.R) H(k) on the Wigner-Seitz vectors of the
dis_v2 run. Consistency check: the same code with the run's own U_k reproduces its wannier90_hr.dat.
DIS_DIR holds the Wannier90 files of that run (eig, amn, u_dis.mat, u.mat, win, hr.dat) with POSCAR and EIGENVAL;
OUT_DIR (created) receives POSCAR, EIGENVAL, wannier90.win and the projected-basis wannier90_hr.dat.

The Wannier90 files of DIS_DIR are not deposited.
usage: python projected_wannier_basis.py DIS_DIR OUT_DIR
"""
import shutil
import sys
from pathlib import Path

import numpy as np

DIS_DIR, OUT_DIR = Path(sys.argv[1]), Path(sys.argv[2])


def read_mat(p, nr, nc):
    lines = p.read_text().split("\n")
    nk = int(lines[1].split()[0])
    out, ks, i = [], [], 3
    for _ in range(nk):
        while not lines[i].strip():
            i += 1
        ks.append([float(x) for x in lines[i].split()])
        i += 1
        v = np.array([[float(x) for x in lines[i + j].split()] for j in range(nr * nc)])
        i += nr * nc
        out.append((v[:, 0] + 1j * v[:, 1]).reshape(nc, nr).T)     # written column by column
    return np.array(ks), np.array(out)


eig = np.loadtxt(DIS_DIR / "wannier90.eig")
nb = int(eig[:, 0].max())
nk = int(eig[:, 1].max())
E = eig[:, 2].reshape(nk, nb)
A = np.zeros((nk, nb, 248), complex)
for row in np.loadtxt(DIS_DIR / "wannier90.amn", skiprows=2):
    A[int(row[2]) - 1, int(row[0]) - 1, int(row[1]) - 1] = row[3] + 1j * row[4]
ks, Uopt = read_mat(DIS_DIR / "wannier90_u_dis.mat", nb, 248)
_, Umlwf = read_mat(DIS_DIR / "wannier90_u.mat", 248, 248)
win = (DIS_DIR / "wannier90.win").read_text()
wmin = float(win.split("dis_win_min =")[1].split()[0])
wmax = float(win.split("dis_win_max =")[1].split()[0])
inw = (E >= wmin) & (E <= wmax)                     # Wannier90 packs the outer-window bands first in u_dis
hr_file = open(DIS_DIR / "wannier90_hr.dat")
hr_file.readline()
n = int(hr_file.readline())
nr = int(hr_file.readline())
deg = []
while len(deg) < nr:
    deg += [int(x) for x in hr_file.readline().split()]
dat = np.loadtxt(hr_file)
R = dat[::n * n, :3].astype(int)
Href = (dat[:, 5] + 1j * dat[:, 6]).reshape(nr, n, n).transpose(0, 2, 1)


def hr(U):
    Hk = []
    for k in range(nk):
        V = np.zeros((nb, 248), complex)
        V[inw[k]] = Uopt[k][: inw[k].sum()]
        V = V @ U[k]
        Hk.append(V.conj().T @ (E[k][:, None] * V))
    ph = np.exp(-2j * np.pi * (R @ ks.T)) / nk
    return np.tensordot(ph, np.array(Hk), axes=(1, 0))


err = np.abs(hr(Umlwf) - Href).max()
print("reproduce dis_v2 hr: max |dH| = %.2e eV" % err)
assert err < 1e-5
Up = []
for k in range(nk):
    Ap = np.zeros((nb, 248), complex)
    Ap[inw[k]] = Uopt[k][: inw[k].sum()]
    W, s, Vh = np.linalg.svd(Ap.conj().T @ A[k])
    Up.append(W @ Vh)
H = hr(np.array(Up))
OUT_DIR.mkdir()
for fn in ("POSCAR", "EIGENVAL", "wannier90.win"):
    shutil.copy(DIS_DIR / fn, OUT_DIR / fn)
with open(OUT_DIR / "wannier90_hr.dat", "w") as fh:
    fh.write(" projected basis on the dis_v2 subspace (proj_basis.py)\n%12d\n%12d\n" % (n, nr))
    for i in range(0, nr, 15):
        fh.write("".join("%5d" % d for d in deg[i:i + 15]) + "\n")
    for r in range(nr):
        for b in range(n):
            for a in range(n):
                fh.write("%5d%5d%5d%5d%5d%12.6f%12.6f\n" % (*R[r], a + 1, b + 1, H[r, a, b].real, H[r, a, b].imag))
print("written", OUT_DIR / "wannier90_hr.dat")
