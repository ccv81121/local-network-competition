"""CaSnO3: per-TRIM VB energies relative to VBM(U=h0h) for PBE, r2SCAN@PBE geom (F1), r2SCAN full (F2); phi on fixed
pairs.

Reads fourstate_152.jsonl (the JSON lines of parse_fourstate.py, not deposited) from the current directory.
usage: python casno3_trim_levels.py
"""
import json
STATE_NAMES = ['state00', 'state10', 'state01', 'state11']
for line in open('fourstate_152.jsonl'):
    record = json.loads(line)
    if 'CaSnO3' not in record['dir']:
        continue
    states = record['states']
    print('##', record['dir'])
    for state in STATE_NAMES:
        trim_levels = states[state]['trim']
        vbm_u = trim_levels['h0h'][0]
        print('  ', state, ' '.join('%s:%.1f' % (k, (vbm_u - v[0]) * 1e3)
                                    for k, v in sorted(trim_levels.items(), key=lambda x: -x[1][0])))
    for rival in ('00h', '000'):
        splitting = {state: (states[state]['trim']['h0h'][0] - states[state]['trim'][rival][0]) * 1e3
                     for state in STATE_NAMES}
        d_perp = splitting['state10'] - splitting['state00']
        d_par = splitting['state01'] - splitting['state00']
        print('   pair U-%s: dperp=%+.2f dpar=%+.2f phi=%.3f' % (rival, d_perp, d_par,
                                                                abs(d_perp) / (abs(d_perp) + abs(d_par))))
