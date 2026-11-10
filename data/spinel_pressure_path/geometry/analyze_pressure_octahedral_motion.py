"""Local rigid-octahedron kinematics of the pressure-path displacements (no force constants, no energies).
For each pressure interval the bond-preserving part of the network displacement is split into the best-fit infinitesimal
rotation of every B-Se octahedron and the angular remainder; ranks, conditioning and reference sensitivity are reported.
Reads pressure_geometry_input_rows.csv and pressure_bond_accommodation.json (next to the script); writes
pressure_octahedral_motion.json and pressure_octahedral_motion_adjacent.csv.
usage: python analyze_pressure_octahedral_motion.py"""
from pathlib import Path
import csv, json, hashlib
import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = Path(__file__).resolve().parent / "pressure_geometry_input_rows.csv"
BOND_SOURCE = Path(__file__).resolve().parent / "pressure_bond_accommodation.json"


def parse_poscar(text):
    t = text.strip().splitlines()
    scale = float(t[1])
    cell = scale * np.array([list(map(float, x.split()[:3])) for x in t[2:5]])
    species = np.repeat(t[5].split(), list(map(int, t[6].split())))
    h = 7
    if t[h].lower().startswith("s"):
        h += 1
    f = np.array([list(map(float, x.split()[:3])) for x in t[h + 1:h + 1 + len(species)]])
    if not t[h].lower().startswith("d"):
        f = scale * f @ np.linalg.inv(cell)
    return cell, f, species


rows = sorted(csv.DictReader(SOURCE.open(encoding="utf-8-sig", newline="")), key=lambda x: float(x["pressure_GPa"]))
structures = [parse_poscar(x["poscar_text"]) for x in rows]
species = structures[0][2]
network = np.flatnonzero(np.isin(species, ["Sc", "Sb", "Se"]))
columns = {int(a): k for k, a in enumerate(network)}
bonds = json.loads(BOND_SOURCE.read_text(encoding="utf-8"))["bonds"]
centers = sorted({b[0] for b in bonds})
assert len(network) == 48 and len(bonds) == 96 and len(centers) == 16
D = np.zeros((288, 144))
for k, (i, j, _) in enumerate(bonds):
    D[3 * k:3 * k + 3, 3 * columns[i]:3 * columns[i] + 3] = -np.eye(3)
    D[3 * k:3 * k + 3, 3 * columns[j]:3 * columns[j] + 3] = np.eye(3)


def rot(r):
    """Columns generate omega cross r."""
    x, y, z = r
    return np.array([[0, z, -y], [-z, 0, x], [y, -x, 0]])


def local_operators(vectors):
    J = np.zeros((96, 144))
    P = np.zeros((288, 288))
    for k, (i, j, _) in enumerate(bonds):
        J[k] = vectors[k] / np.linalg.norm(vectors[k]) @ D[3 * k:3 * k + 3]
    for center in centers:
        ks = [k for k, b in enumerate(bonds) if b[0] == center]
        assert len(ks) == 6
        inds = np.concatenate([np.arange(3 * k, 3 * k + 3) for k in ks])
        A = np.vstack([rot(vectors[k]) for k in ks])
        Q = np.linalg.qr(A, mode="reduced")[0]
        P[np.ix_(inds, inds)] = Q @ Q.T
    return J, P


def analyse(vectors, u):
    J, P = local_operators(vectors)
    _, s, V = np.linalg.svd(J, full_matrices=False)
    n = int(np.count_nonzero(s > s[0] * 1e-10))
    assert n == 96
    up = u - V[:n].T @ (V[:n] @ u)
    vk, v = D @ up, D @ u
    rotation = P @ vk
    angular = vk - rotation
    M = (np.eye(288) - P) @ D
    ss = np.linalg.svd(M, compute_uv=False)
    ranks = {str(t): int(np.count_nonzero(ss > t * ss[0])) for t in [1e-12, 1e-10, 1e-8, 1e-6]}
    nr = ranks["1e-10"]
    translations = np.tile(np.eye(3), (48, 1))
    assert np.linalg.norm(M @ translations) < 1e-12
    assert abs(rotation @ angular) < 1e-12
    assert np.max(abs(J @ up)) < 1e-12
    residual = (np.eye(288) - P) @ v
    return {
        "bond_rank": n, "rigid_octahedra_rank": nr, "rigid_nullity": 144 - nr,
        "rank_tolerance_checks": ranks,
        "rigid_operator_smallest_nonzero": float(ss[nr - 1]),
        "rigid_operator_largest": float(ss[0]),
        "rigid_operator_condition": float(ss[0] / ss[nr - 1]),
        "bond_preserving_atom_squared_fraction": float(up @ up / (u @ u)),
        "kernel_local_rotation_bond_vector_squared_fraction": float(rotation @ rotation / (vk @ vk)),
        "kernel_local_angular_residual_bond_vector_squared_fraction": float(angular @ angular / (vk @ vk)),
        "kernel_bond_vector_rms_A": float(np.sqrt(vk @ vk / 96)),
        "kernel_rotation_bond_vector_rms_A": float(np.sqrt(rotation @ rotation / 96)),
        "kernel_angular_residual_bond_vector_rms_A": float(np.sqrt(angular @ angular / 96)),
        "total_local_rigid_fit_residual_bond_vector_squared_fraction": float(residual @ residual / (v @ v)),
    }


results = []
for k in range(1, len(structures)):
    cell, f, species = structures[k]
    f0 = structures[k - 1][1]
    df = f - f0
    df -= np.round(df)
    u = df[network] @ cell
    u -= u.mean(axis=0)
    u = u.ravel()
    b = np.array([(f0[j] - f0[i] + image) @ cell for i, j, image in bonds])
    dv = (D @ u).reshape(-1, 3)
    base = analyse(b, u)
    sensitivity = {label: analyse(b + alpha * dv, u) for label, alpha in [("midpoint", 0.5), ("endpoint", 1.0)]}
    results.append({"initial_GPa": float(rows[k - 1]["pressure_GPa"]), "final_GPa": float(rows[k]["pressure_GPa"]),
                    **base, "reference_sensitivity": sensitivity})

report = {
    "method": ("Local six-bond least-squares infinitesimal rotation, with exact global periodic shared-ligand "
               "compatibility; fixed mapped lattice."),
    "metric": ("Kernel atom fraction uses centered 48-atom Euclidean norm; rotation/angular fractions use sum of 96 "
               "B-Se relative-vector squared norms. These metrics differ."),
    "scope": ("Kinematics only, not stiffness or energy. Local best-fit rotations need not be jointly realizable as a "
              "globally rigid-octahedral displacement."),
    "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE, BOND_SOURCE]},
    "intervals": results,
}
(OUT / "pressure_octahedral_motion.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
for x in results:
    print(json.dumps({k: x[k]
                      for k in ["initial_GPa", "final_GPa", "rigid_octahedra_rank", "rigid_nullity",
                                "rigid_operator_condition", "bond_preserving_atom_squared_fraction",
                                "kernel_local_rotation_bond_vector_squared_fraction",
                                "kernel_local_angular_residual_bond_vector_squared_fraction"]}))

with (OUT / "pressure_octahedral_motion_adjacent.csv").open("w", encoding="utf-8", newline="") as fh:
    fields = ["initial_GPa", "final_GPa", "bond_preserving_atom_squared_fraction",
              "kernel_local_rotation_bond_vector_squared_fraction",
              "kernel_local_angular_residual_bond_vector_squared_fraction", "rigid_octahedra_rank", "rigid_nullity"]
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in results)
