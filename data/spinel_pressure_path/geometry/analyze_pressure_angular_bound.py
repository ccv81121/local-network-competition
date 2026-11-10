"""Geometry-only angular Rayleigh bound on same-cell bond-preserving motion.
Reads pressure_geometry_input_rows.csv and pressure_bond_accommodation.json (next to the script); writes
pressure_angular_bound.json and pressure_angular_bound.csv.
usage: python analyze_pressure_angular_bound.py"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
source = ROOT / 'pressure_geometry_input_rows.csv'
bond_source = ROOT / 'pressure_bond_accommodation.json'
rows = sorted(csv.DictReader(source.open(encoding='utf-8-sig', newline='')),
              key=lambda x: float(x['pressure_GPa']))
bonds = json.loads(bond_source.read_text(encoding='utf-8'))['bonds']


def parse_poscar(text):
    lines = text.strip().splitlines()
    scale = float(lines[1])
    cell = scale * np.array([list(map(float, x.split()[:3])) for x in lines[2:5]])
    species = np.repeat(lines[5].split(), list(map(int, lines[6].split())))
    h = 8 if lines[7].lower().startswith('s') else 7
    frac = np.array([list(map(float, x.split()[:3])) for x in lines[h + 1:h + 1 + len(species)]])
    if not lines[h].lower().startswith('d'):
        frac = scale * frac @ np.linalg.inv(cell)
    return cell, frac, species


structures = [parse_poscar(x['poscar_text']) for x in rows]
network = np.flatnonzero(np.isin(structures[0][2], ['Sc', 'Sb', 'Se']))
columns = {int(a): i for i, a in enumerate(network)}
centers = sorted({b[0] for b in bonds})
assert len(network) == 48 and len(bonds) == 96 and len(centers) == 16
D = np.zeros((288, 144))
for k, (i, j, _) in enumerate(bonds):
    D[3 * k:3 * k + 3, 3 * columns[i]:3 * columns[i] + 3] = -np.eye(3)
    D[3 * k:3 * k + 3, 3 * columns[j]:3 * columns[j] + 3] = np.eye(3)
T = np.tile(np.eye(3), (48, 1)) / np.sqrt(48)


def bound(vectors):
    C = np.vstack([vectors[k] / np.linalg.norm(vectors[k]) @ D[3 * k:3 * k + 3]
                   for k in range(96)])
    P = np.zeros((288, 288))
    for center in centers:
        ks = [k for k, b in enumerate(bonds) if b[0] == center]
        indices = np.concatenate([np.arange(3 * k, 3 * k + 3) for k in ks])
        rotation = []
        for k in ks:
            x, y, z = vectors[k]
            rotation.append([[0, z, -y], [-z, 0, x], [y, -x, 0]])
        local_basis = np.linalg.qr(np.vstack(rotation), mode='reduced')[0]
        P[np.ix_(indices, indices)] = local_basis @ local_basis.T
    _, singular, vh = np.linalg.svd(np.vstack((C, T.T)), full_matrices=True)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
    assert rank == 99
    Z = vh[rank:].T
    DZ = D @ Z
    B = DZ.T @ DZ
    A = DZ.T @ (np.eye(288) - P) @ DZ
    values, basis = np.linalg.eigh(B)
    assert values[0] > 0
    whitening = basis / np.sqrt(values)
    eigenvalues, eigenvectors = np.linalg.eigh(whitening.T @ A @ whitening)
    minimum_motion = Z @ whitening @ eigenvectors[:, 0]
    assert abs(np.linalg.norm(D @ minimum_motion) - 1) < 1e-12
    floor = float(eigenvalues[0])
    direct = np.linalg.norm((np.eye(288) - P) @ D @ minimum_motion) ** 2
    # An independent subspace formulation obtains the same spectrum without B whitening.
    _, sc, vc = np.linalg.svd(C, full_matrices=True)
    kernel = vc[np.count_nonzero(sc > sc[0] * 1e-10):].T
    q, sd, _ = np.linalg.svd(D @ kernel, full_matrices=False)
    image_rank = int(np.count_nonzero(sd > sd[0] * 1e-10))
    assert image_rank == 45
    q = q[:, :image_rank]
    second = float(np.linalg.eigvalsh(q.T @ (np.eye(288) - P) @ q)[0])
    assert abs(floor - second) < 2e-14 and abs(floor - direct) < 2e-14
    residual = float(np.linalg.norm(C @ minimum_motion))
    assert residual < 1e-12 and np.linalg.norm(T.T @ minimum_motion) < 1e-12
    dv = (D @ minimum_motion).reshape(96, 3)
    errors = [float(np.max(abs(np.linalg.norm(vectors + eps * dv, axis=1)
                               - np.linalg.norm(vectors, axis=1))) / eps ** 2)
              for eps in [1e-3, 5e-4]]
    assert abs(errors[0] / errors[1] - 1) < 1e-3
    return {'angular_fraction_floor': floor, 'independent_subspace_floor': second,
            'angular_fraction_ceiling': float(eigenvalues[-1]),
            'bond_preserving_internal_dimension': 45, 'denominator_smallest_eigenvalue': float(values[0]),
            'extremizer_C_residual': residual, 'bond_error_over_epsilon_squared_A_inverse': errors,
            'extremizing_atomic_displacement_per_unit_bond_vector_norm': minimum_motion.tolist()}


results = []
for k in range(1, len(structures)):
    cell, frac, _ = structures[k]
    previous_frac = structures[k - 1][1]
    change = frac - previous_frac
    change -= np.round(change)
    u = change[network] @ cell
    u -= u.mean(axis=0)
    vectors = np.array([(previous_frac[j] - previous_frac[i] + image) @ cell for i, j, image in bonds])
    dv = (D @ u.ravel()).reshape(96, 3)
    refs = {name: bound(vectors + alpha * dv) for name, alpha in [('affine', 0), ('midpoint', 0.5), ('endpoint', 1)]}
    results.append({'initial_GPa': float(rows[k - 1]['pressure_GPa']),
                    'final_GPa': float(rows[k]['pressure_GPa']), 'references': refs})

report = {'method': ('Generalized Rayleigh spectrum on ker(C) without translations; independent range(D ker(C)) '
                     'orthogonal-subspace check.'),
          'metric': 'Sum of squared increments of 96 relative B-Se bond vectors; identical to S19b angular fraction.',
          'scope': ('Fixed reference lattice, same-cell periodic 48-atom network, first-order bond preservation; no '
                    'energy, stiffness or electronic attribution.'),
          'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, bond_source]},
          'intervals': results}
(ROOT / 'pressure_angular_bound.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
with (ROOT / 'pressure_angular_bound.csv').open('w', encoding='utf-8', newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(['initial_GPa', 'final_GPa', 'affine_floor', 'midpoint_floor', 'endpoint_floor'])
    for row in results:
        writer.writerow([row['initial_GPa'], row['final_GPa']]
                         + [row['references'][x]['angular_fraction_floor'] for x in ['affine', 'midpoint', 'endpoint']])
print(json.dumps({'status': 'PASS', 'intervals': len(results), 'reference_checks': 18,
                  'affine_floors': [row['references']['affine']['angular_fraction_floor'] for row in results]}))
