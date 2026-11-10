"""Read a VASP EIGENVAL file: k points, band energies and number of electrons."""
import numpy as np


def read_eigenval(path):
    lines = open(path).read().split('\n')
    ne, nk, nb = map(int, lines[5].split())
    i = 7
    K = []
    E = []
    for _ in range(nk):
        K.append([float(x) for x in lines[i].split()[:3]])
        bands = [float(lines[i + 1 + j].split()[1]) for j in range(nb)]
        E.append(bands)
        i += nb + 2
    return np.array(K), np.array(E), ne
