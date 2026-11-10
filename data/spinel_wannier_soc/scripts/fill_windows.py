"""Fill the Wannier energy windows of a window template from the SCF eigenvalues.

The offsets are those of the earlier pressure-set basis (PRL_MECHANISM_TEST_20261014/.../edge12_gpu): outer window from
0.249362 eV below the lowest band to 8.834432 eV above the VBM, frozen window from 1.344289 eV below the VBM to
0.939468 eV above the CBM. The upper outer bound is referred to the VBM because the SCF run has 187 bands and the
NSCF run 286.

EIGENVAL is that of the SCF run (not deposited).
usage: python fill_windows.py EIGENVAL TEMPLATE OUT
"""
import sys

import numpy as np

lines = open(sys.argv[1]).read().split("\n")
ne, nk, nb = [int(x) for x in lines[5].split()]
nocc = ne // 2
E, i = [], 7
for k in range(nk):
    E.append([float(lines[i + 1 + j].split()[1]) for j in range(nb)])
    i += nb + 2
E = np.array(E)
windows = {"dis_win_min": E.min() - 0.249362, "dis_win_max": E[:, nocc - 1].max() + 8.834432,
           "dis_froz_min": E[:, nocc - 1].max() - 1.344289, "dis_froz_max": E[:, nocc].min() + 0.939468}
text = open(sys.argv[2]).read()
for k, v in windows.items():
    text = text.replace("@%s@" % k, "%.10f" % v)
assert "@" not in text
open(sys.argv[3], "w").write(text)
print("bands", nb, "occupied", nocc, " ".join("%s=%.4f" % kv for kv in windows.items()))
