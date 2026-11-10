"""Why corner-sharing networks do not amplify bond-preserving motion (existing data only, no new DFT).
For the six systems of 15_Enrichment_AllSystems, split the bond-preserving part (kernel of the B-X bond
Jacobian, same P as enrichment_all.py) of the drive f and of the actual relaxation u into
  - local rigid rotations of each BX_n polyhedron (least-squares omega per centre: db ~ omega x b), and
  - the residual, i.e. internal angular deformation (bending),
measured by the squared relative-bond-vector norm, as in the main text (81% bending for the tight spinel).
Also reports cos(f,u) on the network (motion vs force direction) and the bound 1/drive_share.
usage: python mechanism.py"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "15_Enrichment_AllSystems"))
R6 = HERE.parent
FB = R6 / "10_Force_Balance_Prediction"; TT = R6 / "11_Tight_Relaxation" / "fb_tight"
CORNER = Path(r"$ARCHIVE\Corner_Sharing_Control_20261013\work")
MECH = R6 / "15_Enrichment_AllSystems" / "server_inputs" / "work"


def kernel(f_start, L, bonds, net):
    col = {int(a): c for c, a in enumerate(net)}
    vec = np.array([(f_start[j] - f_start[i] + np.array(img)) @ L for i, j, img in bonds])
    J = np.zeros((len(bonds), 3 * len(net)))
    for r, (i, j, _) in enumerate(bonds):
        e = vec[r] / np.linalg.norm(vec[r]); J[r, 3 * col[i]:3 * col[i] + 3] = -e; J[r, 3 * col[j]:3 * col[j] + 3] = e
    _, s, Vt = np.linalg.svd(J, full_matrices=False)
    return Vt[: int(np.sum(s > s[0] * 1e-10))], vec


def net_part(x, net):
    v = x.reshape(-1, 3)[net]; return (v - v.mean(axis=0)).ravel()


def rigid_split(p, bonds, vec, net):
    """p: network displacement (3*n_net). Returns (rigid, bending) squared relative-bond-vector norms."""
    col = {int(a): c for c, a in enumerate(net)}; P = p.reshape(-1, 3)
    by_centre = {}
    for r, (i, j, _) in enumerate(bonds):
        by_centre.setdefault(int(i), []).append((r, int(j)))
    rig = ben = 0.0
    for i, lst in by_centre.items():
        b = np.array([vec[r] for r, _ in lst]); db = np.array([P[col[j]] - P[col[i]] for _, j in lst])
        # db_k ~ omega x b_k  ->  A omega = db with A_k = -[b_k]_x
        A = np.vstack([np.array([[0, b_[2], -b_[1]], [-b_[2], 0, b_[0]], [b_[1], -b_[0], 0]]) for b_ in b])
        om, *_ = np.linalg.lstsq(A, db.ravel(), rcond=None); fit = A @ om
        rig += float(fit @ fit); ben += float((db.ravel() - fit) @ (db.ravel() - fit))
    return rig, ben


def analyse(name, topo, L, f1, f2, f0, bonds, net, species_of):
    centres = {species_of[int(b[0])] for b in bonds}; ligands = {species_of[int(b[1])] for b in bonds}
    df = f2 - f1; df -= np.round(df); u = (df @ L).ravel()
    B, vec = kernel(f1, L, bonds, net)
    out = {"system": name, "topology": topo, "centres": sorted(centres), "ligands": sorted(ligands)}
    for tag, x in (("drive", f0), ("actual", u)):
        v = net_part(x, net); pk = v - B.T @ (B @ v)
        r, b = rigid_split(pk, bonds, vec, net)
        out[f"{tag}_share"] = float(pk @ pk / (v @ v))
        out[f"{tag}_bp_rigid_frac"] = r / (r + b)
    vf, vu = net_part(f0, net), net_part(u, net)
    out["cos_f_u_network"] = float(vf @ vu / np.linalg.norm(vf) / np.linalg.norm(vu))
    out["enrichment_actual"] = out["actual_share"] / out["drive_share"]
    out["enrichment_bound_1_over_drive"] = 1 / out["drive_share"]
    return out


rows = []
meta = json.loads((TT / "META.json").read_text()); sp = list(np.repeat(meta["species"], meta["counts"]))
bonds = json.loads((FB / "inputs" / "PRESSURE_BOND_ACCOMMODATION.json").read_text())["bonds"]
net = np.flatnonzero(np.isin(sp, ["Sc", "Sb", "Se"]))
rows.append(analyse("MgSc0.5Sb1.5Se4 (k12, tight)", "edge", np.array(meta["lattice_5GPa"]), np.array(meta["frac_start"]),
                    np.array(meta["frac_5GPa"]), np.load(TT / "forces" / "ref.npy").ravel(), bonds, net, sp))
for m in ("CaTiO3", "CaSnO3"):
    W = CORNER / m / "P0_P5"; meta = json.loads((W / "META.json").read_text()); sp = list(np.repeat(meta["species"], meta["counts"]))
    bonds = meta["bonds"]; net = np.array(sorted({b[0] for b in bonds} | {b[1] for b in bonds}))
    rows.append(analyse(m, "corner", np.array(meta["lattice_P2"]), np.array(meta["frac_start"]), np.array(meta["frac_P2"]),
                        np.load(W / "hessian" / "ref" / "forces.npy").ravel(), bonds, net, sp))
for m, label, topo in (("Spinel_k08", "MgScSbSe4 (k08)", "edge"), ("Ca2SnTiO6_rs", "Ca2SnTiO6", "corner"),
                       ("Ca8Sn6Ti2O24", "Ca8Sn6Ti2O24", "corner")):
    W = MECH / m / "P0_P5"; meta = json.loads((W / "META.json").read_text()); sp = list(np.repeat(meta["species"], meta["counts"]))
    rows.append(analyse(label, topo, np.array(meta["lattice_P2"]), np.array(meta["frac_start"]), np.array(meta["frac_P2"]),
                        np.load(W / "hessian" / "ref" / "forces.npy").ravel(), meta["bonds"], np.array(meta["network"]), sp))

(HERE / "MECHANISM.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"{'system':28s} topo   centre->lig   drive%  act%   enr   bound  rigid(drive) rigid(act)  cos(f,u)")
for r in rows:
    print(f"{r['system']:28s} {r['topology']:6s} {'/'.join(r['centres'])}->{'/'.join(r['ligands']):3s}  {100*r['drive_share']:5.2f} {100*r['actual_share']:6.2f} "
          f"{r['enrichment_actual']:6.2f} {r['enrichment_bound_1_over_drive']:6.1f}   {100*r['drive_bp_rigid_frac']:5.1f}%      {100*r['actual_bp_rigid_frac']:5.1f}%     {r['cos_f_u_network']:.3f}")
