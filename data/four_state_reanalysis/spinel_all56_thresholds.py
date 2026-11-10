"""All-56-atom convention (endpoints use the original 56-atom label where the primitive recalculation changed it):
endpoint-matched paths, internal reversals, and the strongest opposite-label margin per path. Read-only.
Margin of an interior state: Indirect -> Delta_E_DI_eV; Direct -> edge_stability_margin_eV
(as in spinel_913_margins.py).

Inputs:  ../descriptor_tables/structures/spinel_structures.csv (states of all substitution paths, with the original 56-atom
         endpoint label where the primitive recalculation changed it).
Output:  one line per endpoint-matched path and the threshold counts on stdout (committed as
         spinel_all56_thresholds.txt).
usage:   python spinel_all56_thresholds.py   (no arguments)
"""
import csv, collections
from pathlib import Path
ENDPOINT_CSV = Path(__file__).resolve().parents[1] / "descriptor_tables" / "structures" / "spinel_structures.csv"
rows = list(csv.DictReader(open(ENDPOINT_CSV, encoding='utf-8-sig')))


def label_of(r):
    return r['endpoint_old_bandgap_type'] if r['endpoint_label_changed_by_primitive'] == 'True' else r['Bandgap Type']


def margin(r):
    v = r['Delta_E_DI_eV'] if r['Bandgap Type'] == 'Indirect' else r['edge_stability_margin_eV']
    return float(v) * 1e3 if v not in ('', None) else float('nan')


parents = {r['mp_id']: r for r in rows if int(r['k']) == 0}
paths_by_mp = collections.defaultdict(dict)
for r in rows:
    if int(r['k']) > 0:
        paths_by_mp[(r['mp_id'], r['substitute'])][int(r['k'])] = r
out = []
for (mp, sub), states_by_k in sorted(paths_by_mp.items()):
    if 16 not in states_by_k or mp not in parents:
        continue
    parent_label = label_of(parents[mp])
    end_label = label_of(states_by_k[16])
    if parent_label in ('Direct', 'Indirect') and parent_label == end_label:
        opposite = [k for k in range(1, 16) if k in states_by_k and label_of(states_by_k[k]) in ('Direct', 'Indirect')
                    and label_of(states_by_k[k]) != parent_label]
        margins = [margin(states_by_k[k]) for k in opposite]
        out.append((f"{mp}_{sub}", parent_label, opposite, margins, max(margins) if margins else float('nan')))
for o in out:
    print(o[0], o[1], o[2], ['%.1f' % x for x in o[3]], '%.1f' % o[4])
rev = [o for o in out if o[2]]
print('matched', len(out), 'reversal', len(rev))
for th in (1, 2, 5, 10, 20):
    print('strongest opposite margin > %g meV: %d/%d' % (th, sum(o[4] > th for o in rev), len(out)))
