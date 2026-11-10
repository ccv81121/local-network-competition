"""Geometry metrics, four-state structures and symmetric-sector Hessian inputs for one interval P1 -> P2.

Definitions follow the spinel and corner-sharing runs:
  * bonds: the six nearest X of every B (P1 structure), bond vectors mapped into the P2 lattice;
  * u: centered shortest-periodic P1 -> P2 displacement of the B+X network in the P2 lattice;
  * u_perp: kernel of the first-order bond Jacobian (bond preserving); u_par = row space (bond changing);
  * four states in the P2 lattice, A cations at P2 positions, network centroid fixed;
  * bridge angles: every B-X-B pair sharing an X; share of the linearized angle change carried by u_perp;
  * Hessian: symmetric (identity-irrep) sector at r_aff = P1 fractional coordinates in the P2 lattice,
    +-DELTA along an orthonormal symmetric basis S (translations removed).
Reads ROOT/MATERIAL/p<P1>/relax/CONTCAR, ROOT/MATERIAL/p<P2>/relax/CONTCAR (e.g. ../electronic_levels/relaxations) and
ROOT/MATERIAL/POTCAR (--write only; not deposited). Prints the geometry metrics; with --write also builds
ROOT/work/MATERIAL/p<P1>_p<P2>/ (four-state and Hessian VASP inputs, S.npy,
meta.json, geometry.json, job lists).
usage: python sector_force_balance_prepare.py ROOT MATERIAL P1 P2 B1,B2 X A FORCE_MESH EDGE_MESH [--write]
  B1,B2 = B-site species, X = anion species, A = A-site species; FORCE_MESH, EDGE_MESH = Gamma-centred k meshes of
  the force runs and of the four-state runs, e.g. 2,3,4 and 4,6,8.
"""
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import spglib

ROOT, MATERIAL, PRESSURE_1, PRESSURE_2 = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
B_SPECIES = sys.argv[5].split(",")
X_SPECIES = sys.argv[6]
A_SPECIES = sys.argv[7]
FORCE_MESH = [int(x) for x in sys.argv[8].split(",")]
EDGE_MESH = [int(x) for x in sys.argv[9].split(",")]
WRITE = "--write" in sys.argv
DELTA = 0.03


def read(path):
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1])
    lattice = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)]) * scale
    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    header = 7 + (lines[7].strip()[0] in "sS")
    frac = np.array([[float(x) for x in lines[header + 1 + i].split()[:3]] for i in range(sum(counts))])
    if lines[header].strip()[0] not in "dD":
        frac = frac @ np.linalg.inv(lattice)
    return lattice, elements, counts, frac % 1.0


def write(path, title, lattice, elements, counts, frac):
    out = [title, "1.0"] + ["  ".join(f"{x:.16f}" for x in vector) for vector in lattice]
    out += (["  ".join(elements), "  ".join(map(str, counts)), "Direct"]
            + ["  ".join(f"{x:.16f}" for x in row % 1.0) for row in frac])
    Path(path).write_bytes(("\n".join(out) + "\n").encode())


material_dir = ROOT / MATERIAL
lattice_1, elements, counts, frac_1 = read(material_dir / f"p{PRESSURE_1}" / "relax" / "CONTCAR")
lattice_2, elements_2, counts_2, frac_2 = read(material_dir / f"p{PRESSURE_2}" / "relax" / "CONTCAR")
assert elements == elements_2 and counts == counts_2
species = np.repeat(elements, counts)
N = len(species)
b_atoms = np.flatnonzero(np.isin(species, B_SPECIES))
x_atoms = np.flatnonzero(species == X_SPECIES)
net = np.sort(np.concatenate([b_atoms, x_atoms]))
net_column = {int(a): c for c, a in enumerate(net)}

bonds = []
for i in b_atoms:
    cand = []
    for j in x_atoms:
        for img in itertools.product((-1, 0, 1), repeat=3):
            cand.append((float(np.linalg.norm((frac_1[j] - frac_1[i] + img) @ lattice_1)), int(j), list(img)))
    cand.sort()
    assert cand[6][0] - cand[5][0] > 0.25, (i, [round(c[0], 3) for c in cand[:8]])
    bonds += [(int(i), j, img) for _, j, img in cand[:6]]
n_bonds = len(bonds)
bond_vectors = np.array([(frac_1[j] - frac_1[i] + np.array(img)) @ lattice_2 for i, j, img in bonds])
J = np.zeros((n_bonds, 3 * len(net)))
for r, (i, j, _) in enumerate(bonds):
    e = bond_vectors[r] / np.linalg.norm(bond_vectors[r])
    J[r, 3 * net_column[i]:3 * net_column[i] + 3] = -e
    J[r, 3 * net_column[j]:3 * net_column[j] + 3] = e
_, s, Vt = np.linalg.svd(J, full_matrices=False)
rank = int(np.sum(s > s[0] * 1e-10))
bond_basis = Vt[:rank]

df = frac_2 - frac_1
df -= np.round(df)
raw = df[net] @ lattice_2
u = (raw - raw.mean(axis=0)).ravel()
u_par = bond_basis.T @ (bond_basis @ u)
u_perp = u - u_par
act2 = (frac_1[net] + df[net]) @ lattice_2
base = act2 - u.reshape(-1, 3)

# bridges: pairs of bonds sharing the same X
bonds_by_x = {}
for r, (i, j, img) in enumerate(bonds):
    bonds_by_x.setdefault(j, []).append(r)
bridges = [(a, b) for rs in bonds_by_x.values() for a, b in itertools.combinations(rs, 2)]


def angles(cart):
    out = []
    for a, b in bridges:
        (i1, j, im1), (i2, _, im2) = bonds[a], bonds[b]
        xa = cart[net_column[j]] + np.array(im1) @ lattice_2
        xb = cart[net_column[j]] + np.array(im2) @ lattice_2
        v1 = cart[net_column[i1]] - xa
        v2 = cart[net_column[i2]] - xb
        out.append(np.degrees(np.arccos(np.clip(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))))
    return np.array(out)


th0 = angles(base)
dth_tot = angles(act2) - th0
dth_perp = angles(base + u_perp.reshape(-1, 3)) - th0
dth_par = angles(base + u_par.reshape(-1, 3)) - th0
nt = float(dth_tot @ dth_tot)
geo = {"material": MATERIAL, "P1": float(PRESSURE_1), "P2": float(PRESSURE_2), "n_atoms": N, "n_network": int(len(net)),
       "n_bonds": n_bonds,
       "bond_rank": rank, "n_bridges": len(bridges),
       "bond_preserving_share_network": float(u_perp @ u_perp / (u @ u)),
       "u_rms_A": float(np.sqrt(u @ u / len(net))),
       "A_cation_rms_A": float(np.sqrt(np.mean(np.sum(((df[species == A_SPECIES]) @ lattice_2) ** 2, axis=1)))),
       "bridge_angle_change_rms_deg": float(np.sqrt(nt / len(bridges))),
       "bridge_share_perp_projection": float(dth_perp @ dth_tot / nt) if nt > 0 else None,
       "bridge_share_par_projection": float(dth_par @ dth_tot / nt) if nt > 0 else None,
       "bridge_rms_perp_deg": float(np.sqrt(dth_perp @ dth_perp / len(bridges))),
       "bridge_rms_par_deg": float(np.sqrt(dth_par @ dth_par / len(bridges)))}

# symmetric sector at r_aff (all atoms)
raff = frac_1.copy()
nums = {x: k for k, x in enumerate(elements)}
symmetry_dataset = spglib.get_symmetry_dataset((lattice_2, raff, [nums[x] for x in species]), symprec=1e-3)
symmetry = spglib.get_symmetry((lattice_2, raff, [nums[x] for x in species]), symprec=1e-3)
Linv = np.linalg.inv(lattice_2)
P = np.zeros((3 * N, 3 * N))
for rotation_frac, translation_frac in zip(symmetry["rotations"], symmetry["translations"]):
    rotation_cart = lattice_2.T @ rotation_frac @ Linv.T
    x = raff @ rotation_frac.T + translation_frac
    for i in range(N):
        dd = raff - x[i]
        dd -= np.round(dd)
        j = int(np.argmin(np.linalg.norm(dd @ lattice_2, axis=1)))
        P[3 * j:3 * j + 3, 3 * i:3 * i + 3] += rotation_cart
P /= len(symmetry["rotations"])
P = 0.5 * (P + P.T)
w, V = np.linalg.eigh(P)
S = V[:, w > 0.5]
T = np.zeros((3 * N, 3))
for a in range(3):
    T[a::3, a] = 1 / np.sqrt(N)
S = S - T @ (T.T @ S)
U_, s_, _ = np.linalg.svd(S, full_matrices=False)
S = U_[:, s_ > 1e-6]
uall = (df @ lattice_2).ravel()
uall = uall - T @ (T.T @ uall)
geo["space_group_raff"] = (symmetry_dataset.international if hasattr(symmetry_dataset, "international")
                           else symmetry_dataset["international"])
geo["n_symmetric_dof"] = int(S.shape[1])
geo["actual_displacement_symmetric_fraction"] = float(np.sum((S.T @ uall) ** 2) / (uall @ uall))
print(json.dumps(geo, indent=1))
if not WRITE:
    sys.exit(0)

work_dir = ROOT / "work" / MATERIAL / f"p{PRESSURE_1}_p{PRESSURE_2}"
work_dir.mkdir(parents=True, exist_ok=True)
inv_lattice_2 = np.linalg.inv(lattice_2)
states = {"state00": base, "state10": base + u_perp.reshape(-1, 3), "state01": base + u_par.reshape(-1, 3),
          "state11": act2}
EDGE_INCAR = """ENCUT = 520
PREC = Accurate
EDIFF = 1E-8
NELM = 200
NELMIN = 5
ALGO = Normal
ISMEAR = 0
SIGMA = 0.02
IBRION = -1
NSW = 0
ISYM = 0
LREAL = .FALSE.
LASPH = .TRUE.
ADDGRID = .TRUE.
LWAVE = .FALSE.
LCHARG = .FALSE.
"""
for name, cart in states.items():
    frac = frac_2.copy()
    frac[net] = cart @ inv_lattice_2
    state_dir = work_dir / "fourstate" / name
    state_dir.mkdir(parents=True, exist_ok=True)
    write(state_dir / "POSCAR", f"{MATERIAL} four-state {name} in P{PRESSURE_2} lattice", lattice_2, elements,
          counts, frac)
    (state_dir / "INCAR").write_text(f"SYSTEM = {MATERIAL} four-state {name}\n" + EDGE_INCAR)
    (state_dir / "KPOINTS").write_text("Gamma edge mesh\n0\nGamma\n{} {} {}\n0 0 0\n".format(*EDGE_MESH))
    shutil.copy(material_dir / "POTCAR", state_dir / "POTCAR")
H_INCAR = """SYSTEM = symmetric-sector finite difference
ENCUT = 520
PREC = Accurate
EDIFF = 1E-8
NELM = 200
NELMIN = 5
ALGO = Normal
ISMEAR = 0
SIGMA = 0.02
IBRION = -1
NSW = 0
ISYM = 0
LREAL = .FALSE.
LASPH = .TRUE.
ADDGRID = .TRUE.
ISTART = {istart}
ICHARG = {icharg}
LWAVE = {lwave}
LCHARG = .FALSE.
"""
jobs = []


def case(name, frac, is_reference):
    case_dir = work_dir / "hessian" / name
    case_dir.mkdir(parents=True, exist_ok=True)
    write(case_dir / "POSCAR", name, lattice_2, elements, counts, frac)
    (case_dir / "INCAR").write_text(H_INCAR.format(istart=0 if is_reference else 1, icharg=2 if is_reference else 0,
                                                   lwave=".TRUE." if is_reference else ".FALSE."))
    (case_dir / "KPOINTS").write_text("Gamma force mesh\n0\nGamma\n{} {} {}\n0 0 0\n".format(*FORCE_MESH))
    shutil.copy(material_dir / "POTCAR", case_dir / "POTCAR")
    jobs.append(name)


case("ref", raff, True)
c0 = raff @ lattice_2
for k in range(S.shape[1]):
    for sgn, tag in ((1, "p"), (-1, "m")):
        c = c0 + sgn * DELTA * S[:, k].reshape(-1, 3)
        case(f"s{k:03d}{tag}", c @ inv_lattice_2, False)
case("r2", frac_2, False)
np.save(work_dir / "S.npy", S)
meta = {"delta_A": DELTA, "lattice_P2": lattice_2.tolist(), "frac_start": frac_1.tolist(), "frac_P2": frac_2.tolist(),
        "species": elements, "counts": counts,
        "bonds": bonds, "network": net.tolist(), "B": B_SPECIES, "X": X_SPECIES, "A": A_SPECIES, "jobs": jobs}
(work_dir / "meta.json").write_text(json.dumps(meta, indent=1))
(work_dir / "geometry.json").write_text(json.dumps(geo, indent=1))
(work_dir / "joblist.txt").write_text("\n".join(jobs) + "\n")
(work_dir / "joblist_rev.txt").write_text("\n".join(["ref"] + jobs[1:][::-1]) + "\n")
print(len(jobs), "symmetric-sector Hessian cases")
