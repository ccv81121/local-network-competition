"""Readouts at Gamma of the four-state structures (Amam cell; the tetragonal M point folds here).
Input: four_state/readout/<tree>__<I>__<state>.txt (E_F line followed by the Gamma PROCAR block of the VASP run; not
deposited) and the state POSCARs in four_state/pbe/<i>/<state>/ (interval in lower case, state 11La in 11_la).
Oxygen sites are classified geometrically (Ni neighbours within 2.4 A): O1 inner apical (two Ni along c), O2 outer
apical (one Ni along c), Op planar (two Ni in the ab plane).
  bands(tree, I, state): Gamma bands with energy relative to E_F and summed weights Ni dz2, Ni x2-y2, O1 pz, O2 pz,
  Op pxy.
  readout(...): R = top of the interlayer-bonding Ni dz2 pair at Gamma minus E_F; the O1 pz weight separates the
  bonding from the antibonding combination.
Run as a script: `python four_state_readout.py show <tree> <I> <state>` prints the band table near E_F;
`python four_state_readout.py` prints the readouts and four-state increments of every complete interval and writes
four_state/readout.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).parent
FOUR_STATE_DIR = HERE / "four_state"
ORBITAL_COLUMN = {"s": 0, "py": 1, "pz": 2, "px": 3, "dxy": 4, "dyz": 5, "dz2": 6, "dxz": 7, "x2-y2": 8}
STATES = ["00", "10", "01", "11", "11La", "01ip", "01ap"]
STATE_DIRECTORY = {"11La": "11_la"}  # directory name of a state on disk


def sites(interval, state):
    atoms = read(FOUR_STATE_DIR / "pbe" / interval.lower() / STATE_DIRECTORY.get(state, state) / "POSCAR")
    symbols = atoms.get_chemical_symbols()
    ni_atoms = [i for i, s in enumerate(symbols) if s == "Ni"]
    out = {"Ni": ni_atoms, "O1": [], "O2": [], "Op": []}
    for i, s in enumerate(symbols):
        if s != "O":
            continue
        ni_vectors = atoms.get_distances(i, ni_atoms, mic=True, vector=True)
        near = [v for v in ni_vectors if np.linalg.norm(v) < 2.4]
        axial = [v for v in near if abs(v[2]) > 0.8 * np.linalg.norm(v)]
        if len(near) == 2 and len(axial) == 2:
            out["O1"].append(i)
        elif len(near) == 1 and len(axial) == 1:
            out["O2"].append(i)
        elif len(near) == 2 and not axial:
            out["Op"].append(i)
        else:
            raise ValueError((interval, state, i, len(near), len(axial)))
    assert (len(out["O1"]), len(out["O2"]), len(out["Op"])) == (4, 8, 16), {k: len(v) for k, v in out.items()}
    return out


def parse(tree, interval, state):
    lines = (FOUR_STATE_DIR / "readout" / f"{tree}__{interval}__{state}.txt").read_text().splitlines()
    ef = float(lines[0].split()[2])
    bands, current = [], None
    for line in lines[1:]:
        if line.startswith("band"):
            current = dict(e=float(line.split()[4]), w={})
            bands.append(current)
        elif current is not None and line[:5].strip().isdigit() and len(line.split()) > 10:
            fields = line.split()
            current["w"][int(fields[0]) - 1] = np.array([float(x) for x in fields[1:]])
    return ef, bands


def bands(tree, interval, state):
    ef, parsed_bands = parse(tree, interval, state)
    st = sites(interval, state)
    rows = []
    for k, b in enumerate(parsed_bands):
        w = b["w"]

        def summed_weight(ions, orb):
            return float(sum(w[i][ORBITAL_COLUMN[orb]] for i in ions))

        rows.append(dict(band=k + 1, e=b["e"] - ef, dz2=summed_weight(st["Ni"], "dz2"),
                         x2y2=summed_weight(st["Ni"], "x2-y2"),
                         O1pz=summed_weight(st["O1"], "pz"), O2pz=summed_weight(st["O2"], "pz"),
                         Oppxy=summed_weight(st["Op"], "px") + summed_weight(st["Op"], "py")))
    return ef, rows


def readout(tree, interval, state):
    """Near E_F the Gamma bands with d_z2 weight ~0.6-0.7 come in two pairs: one without O1 p_z (interlayer bonding,
    the folded M-point gamma band) and one with O1 p_z ~0.12-0.14 (antibonding; O1 p_z couples only to the odd d_z2
    combination and pushes it up). R is the top of the bonding pair, which can lie above E_F; also returned: the pair
    mean, its splitting, and the bonding-antibonding splitting (t_perp proxy)."""
    ef, rows = bands(tree, interval, state)
    window = [r for r in rows if -0.8 <= r["e"] <= 0.6 and r["dz2"] >= 0.5]
    bond = sorted([r for r in window if r["O1pz"] < 0.03], key=lambda r: r["e"])[-2:]
    anti = sorted([r for r in window if r["O1pz"] >= 0.08], key=lambda r: r["e"])[:2]
    assert len(bond) == 2 and len(anti) == 2 and bond[-1]["e"] < anti[0]["e"], (tree, interval, state)
    bonding_energies, antibonding_energies = [r["e"] for r in bond], [r["e"] for r in anti]
    return dict(R=bonding_energies[1], bond=bonding_energies, anti=antibonding_energies,
                bond_mean=sum(bonding_energies) / 2, bond_split=bonding_energies[1] - bonding_energies[0],
                ab_split=sum(antibonding_energies) / 2 - sum(bonding_energies) / 2,
                bands=[r["band"] for r in bond + anti], efermi=ef)


def show(tree, interval, state):
    fermi_energy, rows = bands(tree, interval, state)
    print(f"{tree} {interval} {state}: E_F = {fermi_energy:.4f} eV")
    print(" band   E-E_F   Ni_dz2  Ni_x2y2  O1_pz  O2_pz  Op_pxy")
    for r in rows:
        if -2.0 <= r["e"] <= 1.0:
            print(f" {r['band']:4d} {r['e']:+7.3f}  {r['dz2']:.3f}   {r['x2y2']:.3f}   {r['O1pz']:.3f}  "
                  f"{r['O2pz']:.3f}  {r['Oppxy']:.3f}")
    print("readout:", readout(tree, interval, state))


def main():
    out = {}
    for tree, states in (("runs", STATES), ("runs_U", STATES[:4]), ("runs_k19", ["00", "11"])):
        for interval in ["P1", "P2", "S1", "S2"]:
            have = [s for s in states if (FOUR_STATE_DIR / "readout" / f"{tree}__{interval}__{s}.txt").exists()]
            if not have:
                continue
            readouts = {s: readout(tree, interval, s) for s in have}
            rec = dict(states=readouts)
            if all(s in readouts for s in ("00", "10", "01", "11")):
                r = {s: readouts[s]["R"] for s in readouts}
                db, dc = r["10"] - r["00"], r["01"] - r["00"]
                dint = r["11"] - r["10"] - r["01"] + r["00"]
                total = r["11"] - r["00"]
                phi = 0.5 * ((r["10"] - r["00"]) + (r["11"] - r["01"])) / total if abs(total) > 1e-9 else float("nan")
                rec.update(total=total, d_b=db, d_c=dc, d_int=dint, phi=phi)
                if "11La" in r:
                    rec["d_La"] = r["11"] - r["11La"]
                for s in ("01ip", "01ap"):
                    if s in r:
                        rec["d_" + s] = r[s] - r["00"]
            out[f"{tree}/{interval}"] = rec
            line = "  ".join(f"{s} {readouts[s]['R'] * 1e3:+8.1f}" for s in have)
            extra = ""
            if "phi" in rec:
                extra = (f" | total {rec['total'] * 1e3:+.1f} meV, d_b {rec['d_b'] * 1e3:+.1f}, "
                         f"d_c {rec['d_c'] * 1e3:+.1f}, int {rec['d_int'] * 1e3:+.1f}, phi {100 * rec['phi']:+.0f}%")
            print(f"{tree:8s} {interval}: {line} (meV){extra}")
    assert out, "no four_state/readout/*.txt found (not deposited); readout.json left untouched"
    (FOUR_STATE_DIR / "readout.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        show(*sys.argv[2:5])
    else:
        main()
