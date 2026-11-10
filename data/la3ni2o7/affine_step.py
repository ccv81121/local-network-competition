"""Affine step of La3Ni2O7 readouts: the 0-GPa start structure (wannier/readout/p0start/{pbe,u4}, 12x12x3, ISMEAR -5,
same R definition as four_state_readout.readout; reads wannier/readout/p0start/<tag>.txt, the E_F line and Gamma PROCAR
block of those runs, which is not deposited) and the three-part split of each path from its start:
    total = affine + internal relaxation,   internal = Delta_b (bond preserving) + Delta_c (bond changing) + cross.
P1 and S1 start at 0 GPa; P2 starts at P1/11 and S2 at S1/11. Writes wannier/affine.json."""
import json
from pathlib import Path

import numpy as np
from ase.io import read

import four_state_readout as fs_readout

HERE = Path(__file__).parent


def sites_from(poscar):
    atoms = read(poscar)
    symbols = atoms.get_chemical_symbols()
    ni_atoms = [i for i, s in enumerate(symbols) if s == "Ni"]
    out = {"Ni": ni_atoms, "O1": [], "O2": [], "Op": []}
    for i, s in enumerate(symbols):
        if s != "O":
            continue
        ni_vectors = atoms.get_distances(i, ni_atoms, mic=True, vector=True)
        near = [v for v in ni_vectors if np.linalg.norm(v) < 2.4]
        axial = [v for v in near if abs(v[2]) > 0.8 * np.linalg.norm(v)]
        key = "O1" if len(near) == 2 and len(axial) == 2 else "O2" if len(near) == 1 and len(axial) == 1 else "Op"
        out[key].append(i)
    return out


def p0_readout(tag):
    lines = (HERE / f"wannier/readout/p0start/{tag}.txt").read_text().splitlines()
    ef = float(lines[0].split()[2])
    bands_list, current = [], None
    for line in lines[1:]:
        if line.startswith("band"):
            current = dict(e=float(line.split()[4]), w={})
            bands_list.append(current)
        elif current is not None and line[:5].strip().isdigit() and len(line.split()) > 10:
            fields = line.split()
            current["w"][int(fields[0]) - 1] = np.array([float(x) for x in fields[1:]])
    sites = sites_from(HERE / "relaxation/pressure_p0.vasp")
    rows = []
    for k, b in enumerate(bands_list):

        def summed_weight(ions, orb):
            return float(sum(b["w"][i][fs_readout.ORBITAL_COLUMN[orb]] for i in ions))

        rows.append(dict(band=k + 1, e=b["e"] - ef, dz2=summed_weight(sites["Ni"], "dz2"),
                         O1pz=summed_weight(sites["O1"], "pz")))
    window = [r for r in rows if -0.8 <= r["e"] <= 0.6 and r["dz2"] >= 0.5]
    bond = sorted([r for r in window if r["O1pz"] < 0.03], key=lambda r: r["e"])[-2:]
    anti = sorted([r for r in window if r["O1pz"] >= 0.08], key=lambda r: r["e"])[:2]
    assert len(bond) == 2 and len(anti) == 2 and bond[-1]["e"] < anti[0]["e"], (tag, window)
    bonding_energies, antibonding_energies = [r["e"] for r in bond], [r["e"] for r in anti]
    return dict(R=bonding_energies[1], ab_split=sum(antibonding_energies) / 2 - sum(bonding_energies) / 2, efermi=ef,
                bands=[r["band"] for r in bond + anti])


def main():
    readouts = json.loads((HERE / "four_state/readout.json").read_text())
    out = {}
    for tag, tree in (("pbe", "runs"), ("u4", "runs_U")):
        p0 = p0_readout(tag)
        out[f"P0start_{tag}"] = p0
        start = {"P1": p0["R"], "S1": p0["R"]}
        for interval, prev in (("P2", "P1"), ("S2", "S1")):
            if f"{tree}/{prev}" in readouts:
                start[interval] = readouts[f"{tree}/{prev}"]["states"]["11"]["R"]
        for interval in ("P1", "P2", "S1", "S2"):
            rec = readouts.get(f"{tree}/{interval}")
            if rec is None or "phi" not in rec or interval not in start:
                continue
            s = rec["states"]
            r00, r11 = s["00"]["R"], s["11"]["R"]
            db = 0.5 * ((s["10"]["R"] - s["00"]["R"]) + (s["11"]["R"] - s["01"]["R"]))
            dc = 0.5 * ((s["01"]["R"] - s["00"]["R"]) + (s["11"]["R"] - s["10"]["R"]))
            out[f"{tag}/{interval}"] = dict(start=1e3 * start[interval], end=1e3 * r11,
                                            affine=1e3 * (r00 - start[interval]), internal=1e3 * (r11 - r00),
                                            d_b=1e3 * db, d_c=1e3 * dc, total=1e3 * (r11 - start[interval]))
    (HERE / "wannier/affine.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, {a: (round(b, 1) if isinstance(b, float) and abs(b) > 1 else b) for a, b in v.items()})


if __name__ == "__main__":
    main()
