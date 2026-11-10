"""Finite channel responses chi = d(margin)/d(ln lambda) between lambda = 0.75 and 1.25 from the stored endpoint
margins of hamiltonian_45_channel_responses.csv; writes hamiltonian_response_recomputed.json.
usage: python recompute_hamiltonian_responses.py"""
from pathlib import Path
import csv
import json
import math

DATA_DIR = Path(__file__).resolve().parent
rows = list(csv.DictReader((DATA_DIR / 'hamiltonian_45_channel_responses.csv').open(encoding='utf-8-sig', newline='')))
task_signs = {}
errors = []
for row in rows:
    values = []
    for edge in ('V', 'C'):
        assert row[f'{edge}_identity_changed_lo'] == row[f'{edge}_identity_changed_hi'] == 'False'
        response = ((float(row[f'{edge}_oriented_margin_hi_eV']) - float(row[f'{edge}_oriented_margin_lo_eV']))
                    / (math.log(float(row['lambda_hi'])) - math.log(float(row['lambda_lo']))))
        # chi_*_gamma_favored is a column name only; the reference k points are the fixed a/b columns.
        errors.append(abs(response - float(row[f'chi_{edge}_gamma_favored'])))
        values.append(response)
    opposite = values[0] * values[1] < 0
    assert opposite == (row['opposite_sign'] == 'True')
    task_signs.setdefault(row['task_name'], set()).add(opposite)
result = {'directions': len(rows), 'hamiltonians': len(task_signs),
          'opposite_sign_directions': sum(r['opposite_sign'] == 'True' for r in rows),
          'hamiltonians_with_both_sign_relations': sum(len(v) == 2 for v in task_signs.values()),
          'max_absolute_response_replay_error_eV': max(errors),
          'reference_convention': 'Use explicit fixed a/b columns. Historical gamma_favored field names do not imply '
                                  'every pair contains Gamma.'}
assert (result['directions'], result['hamiltonians'], result['opposite_sign_directions'],
        result['hamiltonians_with_both_sign_relations']) == (45, 13, 17, 8)
assert max(errors) < 1e-12
(DATA_DIR / 'hamiltonian_response_recomputed.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
