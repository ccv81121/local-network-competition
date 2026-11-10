"""Four-state structures for the La3Ni2O7 intervals.

With u = r_end - r_aff (internal, from force_balance/modes.json), split at the 48 Ni-O bonds (la327_network.py):
  u_La  La part; u_N the Ni/O part;
  u_b   first-order bond-preserving part of u_N (projection onto ker C with La fixed);
  u_r   bond-changing part u_N - u_b = C^+ C u_N, further split into the minimum-norm displacements x_ip, x_ap that
        reproduce the in-plane and the apical bond-length changes of u (x_ip + x_ap = u_r).
States (all at the end lattice):
  00 r_aff | 10 r_aff + u_b | 01 r_aff + u_r | 11 r_end | 11La r_aff + u_b + u_r (La frozen at r_aff)
  01ip r_aff + x_ip | 01ap r_aff + x_ap
Writes four_state/pbe/<i>/<state>/POSCAR (LF; interval in lower case, state 11La in directory 11_la), four_state/fourstate.json with component norms, bond-length changes and
the Ni-O-Ni angles and Ni-O bond lengths of every state. Prints checks: max bond change in state 10 (second order),
max mismatch of 01 bond changes against C u, and |x_ip + x_ap - u_r|.
"""
import json
from pathlib import Path

import numpy as np
from ase.io import read, write

import la327_network as net

HERE = Path(__file__).parent
FOUR_STATE_DIR = HERE / "four_state"
STATE_DIRECTORY = {"11La": "11_la"}  # directory name of a state on disk


def state_dir(interval, state):
    return FOUR_STATE_DIR / "pbe" / interval.lower() / STATE_DIRECTORY.get(state, state)


def geometry(atoms, bl):
    """Bond lengths by class and the Ni-O-Ni angles through each O bonded to two Ni."""
    lat, pos = atoms.cell.array, atoms.positions
    lengths = {"inplane": [], "apical_inner": [], "apical_outer": []}
    by_o = {}
    for b in bl:
        by_o.setdefault(b["o"], []).append(b)
    angles = {"interlayer": [], "inplane": []}
    for o, bs in by_o.items():
        vecs = []
        for b in bs:
            d = pos[o] - pos[b["ni"]]
            fr = d @ np.linalg.inv(lat)
            d = (fr - np.round(fr)) @ lat
            vecs.append(d)
            if b["kind"] == "inplane":
                lengths["inplane"].append(np.linalg.norm(d))
            else:
                lengths["apical_inner" if len(bs) == 2 else "apical_outer"].append(np.linalg.norm(d))
        if len(bs) == 2:
            a, c = -vecs[0], -vecs[1]  # O->Ni vectors
            ang = np.degrees(np.arccos(a @ c / np.linalg.norm(a) / np.linalg.norm(c)))
            angles["interlayer" if bs[0]["kind"] == "apical" else "inplane"].append(ang)
    return {k: float(np.mean(v)) for k, v in {**{f"len_{k}": v for k, v in lengths.items()},
                                              **{f"angle_{k}": v for k, v in angles.items()}}.items()}


def main():
    modes = json.loads((HERE / "force_balance" / "modes.json").read_text())
    res = {}
    for name, meta in modes.items():
        aff = read(HERE / "force_balance" / "calculations" / name.lower() / "ref" / "POSCAR")
        end = read(HERE / "force_balance" / "calculations" / name.lower() / "end" / "POSCAR")
        u = np.array(meta["u_actual"])
        bond_list = net.bonds(aff)
        C = net.jacobian(aff, bond_list)
        mask = net.network_mask(aff)
        Cn = C[:, mask]
        uN = np.where(mask, u, 0.0)
        uLa = u - uN
        ur = np.zeros_like(u)
        ur[mask] = np.linalg.pinv(Cn) @ (Cn @ uN[mask])
        ub = uN - ur
        inplane_rows = [i for i, b in enumerate(bond_list) if b["kind"] == "inplane"]
        apical_rows = [i for i, b in enumerate(bond_list) if b["kind"] == "apical"]
        x_ip = net.min_norm_bond_part(C, mask, uN, inplane_rows)
        x_ap = net.min_norm_bond_part(C, mask, uN, apical_rows)
        states = {"00": 0 * u, "10": ub, "01": ur, "11La": ub + ur, "01ip": x_ip, "01ap": x_ap}
        r = dict(norm_u=float(np.linalg.norm(u)), norm_ub=float(np.linalg.norm(ub)), norm_ur=float(np.linalg.norm(ur)),
                 norm_uLa=float(np.linalg.norm(uLa)), norm_x_ip=float(np.linalg.norm(x_ip)),
                 norm_x_ap=float(np.linalg.norm(x_ap)), s_u_network=float(ub @ ub / (u @ u)),
                 check_split=float(np.linalg.norm(x_ip + x_ap - ur)), states={})
        for tag, du in {**states, "11": None}.items():
            at = end.copy() if du is None else aff.copy()
            if du is not None:
                at.positions = aff.positions + du.reshape(-1, 3)
            d = state_dir(name, tag)
            d.mkdir(parents=True, exist_ok=True)
            write(d / "POSCAR", at, format="vasp", direct=True, sort=False)
            p = d / "POSCAR"
            p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
            r["states"][tag] = geometry(at, net.bonds(at))
        lengths_aff = np.array([b["length"] for b in bond_list])
        lengths_by_state = {t: np.array([b["length"]
                                         for b in net.bonds(read(state_dir(name, t) / "POSCAR"))])
                            for t in ("10", "01", "11")}
        r["max_bond_change_10"] = float(np.abs(lengths_by_state["10"] - lengths_aff).max())
        r["max_bond_change_11"] = float(np.abs(lengths_by_state["11"] - lengths_aff).max())
        r["max_mismatch_01_vs_Cu"] = float(np.abs((lengths_by_state["01"] - lengths_aff) - C @ uN).max())
        res[name] = r
        print(f"{name}: |u| {r['norm_u']:.4f}  |u_b| {r['norm_ub']:.4f}  |u_r| {r['norm_ur']:.4f}  "
              f"|u_La| {r['norm_uLa']:.4f}  |x_ip| {r['norm_x_ip']:.4f}  |x_ap| {r['norm_x_ap']:.4f}  "
              f"split {r['check_split']:.1e}"
              f"  dL10 {r['max_bond_change_10']*1e3:.2f} mA  dL11 {r['max_bond_change_11']*1e3:.1f} mA"
              f"  mismatch01 {r['max_mismatch_01_vs_Cu']*1e3:.2f} mA")
        for t in ("00", "10", "01", "11"):
            g = r["states"][t]
            print(f"   {t:4s} Ni-O1-Ni {g['angle_interlayer']:.2f}  in-plane {g['angle_inplane']:.2f}  "
                  f"Ni-O ip {g['len_inplane']:.4f} inner {g['len_apical_inner']:.4f} "
                  f"outer {g['len_apical_outer']:.4f}")
    (FOUR_STATE_DIR / "fourstate.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
