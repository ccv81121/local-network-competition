"""Force-balance inputs for the tight endpoints: drive forces at r_aff (forces/ref.npy) and meta.json with the tight
lattice and fractional coordinates; the force-constant matrix of the earlier pressure set is reused.
Reads ../spinel/p{4,5}/relax/CONTCAR and ../static_raff/OUTCAR (last force block; not deposited) and the meta.json of
../../force_balance_reference; writes forces/ref.npy and meta.json in the current directory.
usage: python prepare_force_balance.py   (run from the force_balance folder)"""
import json, numpy as np, sys
sys.path.insert(0, "..")
FORCE_BALANCE_DIR = "../../force_balance_reference"
meta = json.load(open(FORCE_BALANCE_DIR + "/meta.json"))


def read(path):
    lines = open(path).read().splitlines()
    scale = float(lines[1])
    lattice = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)]) * scale
    counts = [int(x) for x in lines[6].split()]
    header = 7 + (lines[7].strip()[0] in "sS")
    frac = np.array([[float(x) for x in lines[header + 1 + i].split()[:3]] for i in range(sum(counts))])
    assert lines[5].split() == list(meta["species"])
    return lattice, frac % 1.0


L4, f4 = read("../spinel/p4/relax/CONTCAR")
L5, f5 = read("../spinel/p5/relax/CONTCAR")
N = sum(meta["counts"])
block_lines = open("../static_raff/OUTCAR").read().split("TOTAL-FORCE (eV/Angst)")[1:][-1].splitlines()[2:2 + N]
ft = np.array([[float(x) for x in line.split()[3:6]] for line in block_lines])
np.save("forces/ref.npy", ft)
meta["lattice_5GPa"] = L5.tolist()
meta["frac_start"] = f4.tolist()
meta["frac_5GPa"] = f5.tolist()
meta["note"] = "tight endpoints (PRL_TIGHT_SPINEL_20261015); H reused from Round5"
json.dump(meta, open("meta.json", "w"), indent=1)
print("ok", ft.shape)
