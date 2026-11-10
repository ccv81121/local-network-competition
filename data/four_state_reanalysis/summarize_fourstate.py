"""Summarize the JSON lines written by parse_fourstate.py: settings consistency, TRIM margins, dLambda, phi, energy
lowering.

JSONL is the output of parse_fourstate.py on the archived runs (not deposited; fourstate_summary.txt is the result).
usage: python summarize_fourstate.py JSONL
"""
import json, sys
STATE_NAMES = ['state00', 'state10', 'state01', 'state11']
for line in open(sys.argv[1]):
    record = json.loads(line)
    states = record['states']
    name = record['dir'].replace('PRL_', '').replace('/fourstate', '')
    if any(states[state] is None for state in STATE_NAMES):
        print('##', name, 'INCOMPLETE', [state for state in STATE_NAMES if states[state] is None])
        continue
    sets = {json.dumps(states[state]['set'], sort_keys=True) + states[state]['kp'] for state in STATE_NAMES}
    print('##', name, '| settings identical across 4 states:', len(sets) == 1, '| kp', states['state00']['kp'], '| ',
          {k: v for k, v in states['state00']['set'].items() if v})
    e0 = states['state00']['toten']
    # spinel-type: CB(hhh)-CB(000)
    for state in STATE_NAMES:
        trim_levels = states[state]['trim']
        vb = sorted(((v[0], k) for k, v in trim_levels.items()), reverse=True)
        cb = sorted((v[1], k) for k, v in trim_levels.items())
        lambda_c = (trim_levels['hhh'][1] - trim_levels['000'][1]) * 1e3 if 'hhh' in trim_levels else float('nan')
        print('  %s E-E00=%7.2f meV | VBM(full)%s CBM(full)%s | TRIM VB: %s %s d=%.2f | CB: %s %s d=%.2f '
              '| CB(hhh)-CB(G)=%.2f' % (state, (states[state]['toten'] - e0) * 1e3,
                                        [round(x, 3) for x in states[state]['VBMk']],
                                        [round(x, 3) for x in states[state]['CBMk']], vb[0][1], vb[1][1],
                                        (vb[0][0] - vb[1][0]) * 1e3, cb[0][1], cb[1][1], (cb[1][0] - cb[0][0]) * 1e3,
                                        lambda_c))
