"""Apply the endpoint-matched rule of the spinel substitution paths under two label conventions (read-only):
(a) canonical: endpoints (k=0,16) from 14-atom primitive recalculation, interior 56-atom (as in paper);
(b) all-56: endpoints use the original 56-atom label (endpoint_old_bandgap_type where changed).

Inputs:  ../descriptor_tables/structures/spinel_structures.csv (states of all substitution paths, with the original 56-atom
         endpoint label where the primitive recalculation changed it).
Output:  matched paths and reversal paths per convention on stdout (committed as spinel_913_cellconv.txt).
usage:   python spinel_913_cell_convention.py   (no arguments)
"""
import csv, collections
from pathlib import Path
ENDPOINT_CSV = Path(__file__).resolve().parents[1] / "descriptor_tables" / "structures" / "spinel_structures.csv"
rows = list(csv.DictReader(open(ENDPOINT_CSV, encoding='utf-8-sig')))


def label_of(r, conv):
    if conv == 'all56' and r['endpoint_label_changed_by_primitive'] == 'True':
        return r['endpoint_old_bandgap_type']
    return r['Bandgap Type']


parents = {r['mp_id']: r for r in rows if int(r['k']) == 0}
paths_by_mp = collections.defaultdict(dict)
for r in rows:
    if int(r['k']) > 0:
        paths_by_mp[(r['mp_id'], r['substitute'])][int(r['k'])] = r
for conv in ('canonical', 'all56'):
    matched = []
    rev = []
    for (mp, sub), states_by_k in sorted(paths_by_mp.items()):
        if 16 not in states_by_k or mp not in parents:
            continue
        parent_label = label_of(parents[mp], conv)
        end_label = label_of(states_by_k[16], conv)
        if parent_label in ('Direct', 'Indirect') and parent_label == end_label:
            matched.append(mp + '_' + sub)
            opposite = [k for k in range(1, 16) if k in states_by_k
                        and label_of(states_by_k[k], conv) in ('Direct', 'Indirect')
                        and label_of(states_by_k[k], conv) != parent_label]
            if opposite:
                rev.append(mp + '_' + sub)
    print(conv, ': matched', len(matched), 'reversal', len(rev))
    print('   matched:', ' '.join(matched))
    print('   reversal:', ' '.join(rev))
