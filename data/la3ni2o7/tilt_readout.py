"""Gamma readout of the tilt-check structures from a text file in the format of four_state/readout/*.txt (E_F line
followed by the Gamma PROCAR block). Same R definition as four_state_readout.readout, except that degenerate
bonding / antibonding pairs are allowed (no tilt: the folding splitting vanishes).
usage: python tilt_readout.py <readout.txt> <POSCAR>"""
import sys

import numpy as np

import four_state_readout as fs_readout
from affine_step import sites_from


def readout(txt, poscar):
    lines = open(txt).read().splitlines()
    ef = float(lines[0].split()[2])
    bands_list, current = [], None
    for line in lines[1:]:
        if line.startswith("band"):
            current = dict(e=float(line.split()[4]), w={})
            bands_list.append(current)
        elif current is not None and line[:5].strip().isdigit() and len(line.split()) > 10:
            fields = line.split()
            current["w"][int(fields[0]) - 1] = np.array([float(x) for x in fields[1:]])
    sites = sites_from(poscar)
    rows = []
    for k, b in enumerate(bands_list):

        def summed_weight(ions, orb):
            return float(sum(b["w"][i][fs_readout.ORBITAL_COLUMN[orb]] for i in ions))

        rows.append(dict(band=k + 1, e=b["e"] - ef, dz2=summed_weight(sites["Ni"], "dz2"),
                         O1pz=summed_weight(sites["O1"], "pz")))
    window = [r for r in rows if -0.8 <= r["e"] <= 0.6 and r["dz2"] >= 0.5]
    bond = sorted([r for r in window if r["O1pz"] < 0.03], key=lambda r: r["e"])[-2:]
    anti = sorted([r for r in window if r["O1pz"] >= 0.08], key=lambda r: r["e"])[:2]
    assert len(bond) == 2 and len(anti) == 2 and bond[-1]["e"] <= anti[0]["e"], (txt, window)
    bonding_energies, antibonding_energies = [r["e"] for r in bond], [r["e"] for r in anti]
    return dict(R=1e3 * bonding_energies[1], bond=[1e3 * x for x in bonding_energies],
                anti=[1e3 * x for x in antibonding_energies],
                ab_split=1e3 * (sum(antibonding_energies) / 2 - sum(bonding_energies) / 2), efermi=ef,
                bands=[r["band"] for r in bond + anti])


if __name__ == "__main__":
    print(readout(sys.argv[1], sys.argv[2]))
