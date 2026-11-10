"""Project existing pressure displacements onto continued bond constraints.

Purely geometric: no force constants and no electronic calculations.
The source POSCARs and the 96 B-X bonds are read unchanged.
Reads pressure_geometry_input_rows.csv and pressure_bond_accommodation.json (next to the script); writes
pressure_network_kinematics.json and pressure_network_kinematics_adjacent.csv and prints one line per interval.
usage: python analyze_pressure_network_kinematics.py
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE = Path(__file__).resolve().parent / "pressure_geometry_input_rows.csv"
BOND_SOURCE = Path(__file__).resolve().parent / "pressure_bond_accommodation.json"


def parse_poscar(text):
    lines = text.strip().splitlines()
    scale = float(lines[1])
    cell = scale * np.array([list(map(float, s.split()[:3])) for s in lines[2:5]])
    species = np.repeat(lines[5].split(), list(map(int, lines[6].split())))
    header = 7
    if lines[header].lower().startswith("s"):
        header += 1
    positions = np.array([
        list(map(float, s.split()[:3]))
        for s in lines[header + 1:header + 1 + len(species)]
    ])
    if not lines[header].lower().startswith("d"):
        positions = scale * positions @ np.linalg.inv(cell)
    return cell, positions, species


rows = sorted(csv.DictReader(SOURCE.open(encoding="utf-8-sig", newline="")),
              key=lambda r: float(r["pressure_GPa"]))
structures = [parse_poscar(row["poscar_text"]) for row in rows]
species = structures[0][2]
assert all(np.array_equal(s[2], species) for s in structures)
network = np.flatnonzero(np.isin(species, ["Sc", "Sb", "Se"]))
assert len(network) == 48
atom_column = {int(atom): column for column, atom in enumerate(network)}
bonds = json.loads(BOND_SOURCE.read_text(encoding="utf-8"))["bonds"]
assert len(bonds) == 96
periodic_images = np.array([(i, j, k) for i in [-1, 0, 1]
                           for j in [-1, 0, 1] for k in [-1, 0, 1]])


def jacobian(vectors):
    lengths = np.linalg.norm(vectors, axis=1)
    matrix = np.zeros((len(bonds), 3 * len(network)))
    for row, (i, j, _) in enumerate(bonds):
        direction = vectors[row] / lengths[row]
        matrix[row, 3 * atom_column[i]:3 * atom_column[i] + 3] = -direction
        matrix[row, 3 * atom_column[j]:3 * atom_column[j] + 3] = direction
    return matrix


def rms(v):
    return float(np.sqrt(np.mean(np.sum(np.asarray(v).reshape(-1, 3) ** 2, axis=1))))


def project(f_reference, structure, p_reference, p_final):
    cell, f_final, _ = structure
    df = f_final - f_reference
    df -= np.round(df)
    displacements = df[network] @ cell
    translation = displacements.mean(axis=0)
    displacements -= translation
    u = displacements.ravel()
    reference_vectors = np.array([(f_reference[j] - f_reference[i] + image) @ cell
                                  for i, j, image in bonds])
    changes = np.array([(df[j] - df[i]) @ cell for i, j, _ in bonds])
    final_vectors = reference_vectors + changes
    final_lengths = np.linalg.norm(final_vectors, axis=1)
    # Continued bond images agree with independent minimal-image distances.
    minimum = np.array([np.linalg.norm((f_final[j] - f_final[i] + periodic_images) @ cell,
                                      axis=1).min() for i, j, _ in bonds])
    assert np.allclose(final_lengths, minimum, rtol=0, atol=1e-10)
    J = jacobian(reference_vectors)
    left, singular, right = np.linalg.svd(J, full_matrices=False)
    threshold = singular[0] * 1e-10
    rank = int(np.count_nonzero(singular > threshold))
    assert rank == 96
    basis = right[:rank]
    stretching = basis.T @ (basis @ u)
    preserving = u - stretching
    total_squared = float(u @ u)
    fraction = float(preserving @ preserving / total_squared) if total_squared else None
    linear = J @ u
    exact = final_lengths - np.linalg.norm(reference_vectors, axis=1)
    relative_error = (float(np.linalg.norm(exact - linear) / np.linalg.norm(exact))
                      if np.linalg.norm(exact) else None)
    endpoint_fractions = {}
    for name, vectors in [("affine", reference_vectors),
                          ("midpoint", reference_vectors + 0.5 * changes),
                          ("relaxed", final_vectors)]:
        _, _, basis_end = np.linalg.svd(jacobian(vectors), full_matrices=False)
        residual = u - basis_end.T @ (basis_end @ u)
        endpoint_fractions[name] = (float(residual @ residual / total_squared)
                                   if total_squared else None)
    tolerance_checks = []
    for relative_tolerance in [1e-12, 1e-10, 1e-8, 1e-6]:
        r = int(np.count_nonzero(singular > relative_tolerance * singular[0]))
        tolerance_checks.append({"relative_tolerance": relative_tolerance, "rank": r})
        assert r == rank
    per_species = {}
    for element in ["Sc", "Sb", "Se"]:
        select = species[network] == element
        s = stretching.reshape(-1, 3)[select]
        n = preserving.reshape(-1, 3)[select]
        actual = u.reshape(-1, 3)[select]
        cross = float(2 * np.mean(np.sum(s * n, axis=1)))
        per_species[element] = {
            "atom_count": int(np.count_nonzero(select)), "total_rms_A": rms(actual),
            "stretching_rms_A": rms(s), "preserving_rms_A": rms(n),
            "twice_cross_term_A2_per_atom": cross,
            "norm_identity_residual_A2_per_atom": rms(actual) ** 2 - rms(s) ** 2 - rms(n) ** 2 - cross,
        }
    checks = {
        "reconstruction_max_A": float(np.max(np.abs(u - stretching - preserving))),
        "preserving_bond_change_max_A": float(np.max(np.abs(J @ preserving))),
        "orthogonality_A2": float(stretching @ preserving),
        "norm_identity_residual_A2": total_squared - float(stretching @ stretching + preserving @ preserving),
        "minimal_image_max_difference_A": float(np.max(np.abs(final_lengths - minimum))),
    }
    assert all(abs(value) < 1e-10 for value in checks.values())
    return {
        "reference_pressure_GPa": p_reference, "final_pressure_GPa": p_final,
        "network_atom_count": len(network), "constraint_count": len(bonds),
        "cartesian_dimension": 3 * len(network), "rank": rank,
        "nullity_including_translations": 3 * len(network) - rank,
        "nullity_excluding_translations": 3 * len(network) - rank - 3,
        "smallest_singular_value": float(singular[-1]),
        "largest_singular_value": float(singular[0]),
        "condition_number_nonzero": float(singular[0] / singular[-1]),
        "removed_translation_norm_A": float(np.linalg.norm(translation)),
        "network_total_rms_A": rms(u), "stretching_rms_A": rms(stretching),
        "preserving_rms_A": rms(preserving), "preserving_squared_fraction": fraction,
        "actual_internal_bond_change_rms_A": float(np.sqrt(np.mean(exact ** 2))),
        "linearization_error_rms_A": float(np.sqrt(np.mean((exact - linear) ** 2))),
        "relative_linearization_error": relative_error,
        "reference_direction_sensitivity": endpoint_fractions,
        "tolerance_checks": tolerance_checks, "species": per_species, "checks": checks,
    }


cumulative = []
adjacent = []
for i, (row, structure) in enumerate(zip(rows, structures)):
    pressure = float(row["pressure_GPa"])
    cumulative.append(project(structures[0][1], structure, 0.0, pressure))
    if i:
        adjacent.append(project(structures[i - 1][1], structure,
                                float(rows[i - 1]["pressure_GPa"]), pressure))
report = {
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "bond_source_sha256": hashlib.sha256(BOND_SOURCE.read_bytes()).hexdigest(),
    "method": {
        "reference": ("Earlier fractional coordinates mapped through later lattice; wrapped displacements; continued "
                      "B-Se bond identities and images."),
        "network": "4 Sc, 12 Sb, 32 Se; exclude 8 Mg atoms because the selected constraints contain no Mg.",
        "translation": "Remove mean displacement of the 48-atom network before Euclidean projection.",
        "decomposition": ("u_stretch=J^+Ju; u_preserve=(I-J^+J)u; J has rows (-e_ij,+e_ij) on bonded Cartesian atom "
                          "coordinates."),
        "fraction": ("Squared Euclidean network displacement fraction; not an energy fraction or statistical "
                     "significance."),
        "species": "Species components are not separately orthogonal; cross terms are reported.",
        "scope": ("First-order kinematics of the selected 96 bonds, not force constants, energy soft modes, rigid "
                  "octahedra, or causes of electronic crossings."),
    },
    "new_first_principles_calculation": False,
    "cumulative": cumulative, "adjacent": adjacent,
}
(SCRIPT_DIR / "pressure_network_kinematics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
fields = ["reference_pressure_GPa", "final_pressure_GPa", "rank", "network_total_rms_A",
          "stretching_rms_A", "preserving_rms_A", "preserving_squared_fraction",
          "relative_linearization_error"]
with (SCRIPT_DIR / "pressure_network_kinematics_adjacent.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in adjacent)
for row in adjacent:
    print(json.dumps({key: row[key] for key in fields}))
