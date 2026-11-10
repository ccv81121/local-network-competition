"""9/13 spinel internal-reversal margins (read-only).

For each of the 13 endpoint-matched paths, the strongest opposite-label state margin: Indirect state -> Delta_E_DI_eV
(direct minus indirect gap); Direct state -> edge_stability_margin_eV.
Sensitivity: number of reversal paths whose strongest reversal exceeds a threshold.

Inputs:  ../descriptor_tables/structures/spinel_structures.csv (states of all substitution paths, endpoints from the
         14-atom primitive recalculation), inputs/spinel/endpoint_matched_paths_13.csv (the 13 paths).
Output:  tab-separated table and the threshold counts on stdout (committed as spinel_913_margins.txt).
usage:   python spinel_913_margins.py   (no arguments)
"""
import csv
from pathlib import Path
INPUT_DIR = Path(__file__).resolve().parent / "inputs" / "spinel"
RAW_CSV = Path(__file__).resolve().parents[1] / "descriptor_tables" / "structures" / "spinel_structures.csv"
PATHS_CSV = INPUT_DIR / "endpoint_matched_paths_13.csv"
rows = list(csv.DictReader(open(RAW_CSV, encoding='utf-8-sig')))
paths = list(csv.DictReader(open(PATHS_CSV, encoding='utf-8-sig')))


def get(mp, sub, k):
    for r in rows:
        if r['mp_id'] == mp and int(r['k']) == k and (k == 0 or r['substitute'] == sub):
            return r


def margin(r):
    t = r['Bandgap Type']
    v = r['Delta_E_DI_eV'] if t == 'Indirect' else r['edge_stability_margin_eV']
    return float(v) if v not in ('', None) else float('nan')


out = []
print('path\tendpoint\topp_k\topp_margins_meV\tmax_opp_meV\tparent_margin_meV\tendpoint_margin_meV')
for path_row in paths:
    mp, sub = path_row['path_id'].split('_')
    ks = [int(x) for x in path_row['opposite_label_k'].split(';') if x]
    margins = [margin(get(mp, sub, k)) * 1e3 for k in ks]
    parent_margin = margin(get(mp, sub, 0)) * 1e3
    endpoint_margin = margin(get(mp, sub, 16)) * 1e3
    max_opposite = max(margins) if margins else float('nan')
    out.append((path_row['path_id'], path_row['internal_reversal'] == 'True', max_opposite, parent_margin,
                endpoint_margin))
    print('%s\t%s\t%s\t%s\t%.1f\t%.1f\t%.1f' % (path_row['path_id'], path_row['endpoint_label'],
                                                path_row['opposite_label_k'], ','.join('%.1f' % x for x in margins),
                                                max_opposite, parent_margin, endpoint_margin))
rev = [o for o in out if o[1]]
print('reversal paths:', len(rev), '/', len(out))
for th in (1, 2, 5, 10, 20):
    print('max opposite margin > %g meV: %d/13' % (th, sum(o[2] > th for o in rev)))
