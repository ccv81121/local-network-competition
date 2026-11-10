"""Geometric decomposition of the pressure-path structures (no electronic calculation).
Selects the six nearest Se of every Sc/Sb in the zero-pressure structure (96 bonds, continued through the path) and
reports bond-length statistics and internal length changes per species.
Reads pressure_geometry_input_rows.csv (next to the script); writes pressure_bond_accommodation.json.
usage: python analyze_pressure_bond_accommodation.py"""
from pathlib import Path
import csv, hashlib, json
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
source = Path(__file__).resolve().parent / 'pressure_geometry_input_rows.csv'
rows = sorted(csv.DictReader(source.open(encoding='utf-8-sig', newline='')),
              key=lambda r: float(r['pressure_GPa']))
assert len(rows) == 6
images = np.array([(i, j, k) for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]])


def parse(text):
    lines = text.strip().splitlines()
    scale = float(lines[1])
    cell = scale * np.array([list(map(float, line.split()[:3])) for line in lines[2:5]])
    species = np.repeat(lines[5].split(), list(map(int, lines[6].split())))
    pos = 7
    if lines[pos].lower().startswith('s'):
        pos += 1
    direct = lines[pos].lower().startswith('d')
    xyz = np.array([list(map(float, line.split()[:3])) for line in lines[pos + 1:pos + 1 + len(species)]])
    if not direct:
        xyz = scale * xyz @ np.linalg.inv(cell)
    return cell, xyz, species


cell0, f0, el0 = parse(rows[0]['poscar_text'])
se = np.where(el0 == 'Se')[0]
selected = []
for i in np.where(np.isin(el0, ['Sc', 'Sb']))[0]:
    vectors = (f0[se, None, :] - f0[i] + images[None, :, :]) @ cell0
    lengths = np.linalg.norm(vectors, axis=2)
    image = lengths.argmin(axis=1)
    nearest = np.argsort(lengths.min(axis=1))[:6]
    selected.extend((int(i), int(se[j]), images[image[j]].tolist()) for j in nearest)
assert len(selected) == 96
results = []
for row in rows:
    cell, f, els = parse(row['poscar_text'])
    assert np.array_equal(els, el0)
    df = f - f0
    df -= np.round(df)
    unwrapped = f0 + df
    data = {'P_GPa': float(row['pressure_GPa'])}
    for species in ['Sc', 'Sb']:
        pairs = [(i, j, np.array(im)) for i, j, im in selected if els[i] == species]
        affine = np.array([(f0[j] - f0[i] + im) @ cell for i, j, im in pairs])
        relaxed = np.array([(unwrapped[j] - unwrapped[i] + im) @ cell for i, j, im in pairs])
        lengths = np.linalg.norm(relaxed, axis=1)
        alengths = np.linalg.norm(affine, axis=1)
        # Independently confirm the continued periodic images remain minimal.
        minimal = [np.linalg.norm((f[j] - f[i] + images) @ cell, axis=1).min() for i, j, _ in pairs]
        assert np.allclose(lengths, minimal, atol=1e-10)
        delta = lengths - alengths
        data.update({f'{species}_mean_A': float(lengths.mean()),
                     f'{species}_bond_sd_A': float(lengths.std()),
                     f'{species}_bond_cv': float(lengths.std() / lengths.mean()),
                     f'{species}_affine_mean_A': float(alengths.mean()),
                     f'{species}_internal_length_mean_A': float(delta.mean()),
                     f'{species}_internal_length_rms_A': float(np.sqrt(np.mean(delta ** 2))),
                     f'{species}_relative_internal_vector_rms_A': float(np.sqrt(np.mean(np.sum((relaxed - affine) ** 2,
                                                                                               axis=1))))})
    results.append(data)
report = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'definition': ('Affine geometry: zero-pressure fractional coordinates mapped through current lattice. '
                         'Internal length change: relaxed minus affine length for each continued B-Se bond. Population '
                         'SD within each species; not uncertainty. Relative-vector RMS includes longitudinal and '
                         'transverse changes.'),
          'bonds': selected, 'rows': results, 'new_first_principles_calculation': False}
(SCRIPT_DIR / 'pressure_bond_accommodation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
