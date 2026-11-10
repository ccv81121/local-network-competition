#!/usr/bin/env python3
"""Read-only: for each four-state dir (state00/10/01/11), re-parse EIGENVAL and report
per-TRIM VB top / CB bottom, TOTEN, and INCAR/KPOINTS setting consistency.

usage: parse_fourstate.py DIR [DIR...]  -> JSON lines
"""
import os, sys, json
TRACKED_INCAR_KEYS = ['ENCUT', 'PREC', 'EDIFF', 'ISMEAR', 'SIGMA', 'ISYM', 'NBANDS', 'GGA', 'METAGGA', 'LASPH',
                      'LSORBIT', 'LDAU', 'ALGO', 'ADDGRID']


def read_incar(path):
    settings = {}
    for line in open(path, errors='ignore'):
        line = line.split('#')[0]
        for part in line.split(';'):
            if '=' in part:
                k, v = part.split('=', 1)
                settings[k.strip().upper()] = v.strip()
    return settings


def read_eigenval(path):
    lines = open(path).read().split('\n')
    nel, nk, nb = map(int, lines[5].split()[:3])
    i = 7
    ks = []
    energies = []
    for _ in range(nk):
        while not lines[i].strip():
            i += 1
        k = [float(x) for x in lines[i].split()[:3]]
        i += 1
        band_energies = []
        for b in range(nb):
            v = lines[i].split()
            band_energies.append(float(v[1]))
            i += 1
        ks.append(k)
        energies.append(band_energies)
    return nel, ks, energies


def is_trim(k):
    return all(abs(abs(x) - 0.5) < 1e-6 or abs(x) < 1e-6 for x in k)


def trim_label(k):
    return ''.join('h' if abs(abs(x) - 0.5) < 1e-6 else '0' for x in k)


def last_total_energy(run_dir):
    energy = None
    for line in open(os.path.join(run_dir, 'OUTCAR'), errors='ignore'):
        if 'free  energy   TOTEN' in line:
            energy = float(line.split('=')[1].split()[0])
    return energy


for root in sys.argv[1:]:
    record = {'dir': root, 'states': {}}
    for state in ['state00', 'state10', 'state01', 'state11']:
        run_dir = os.path.join(root, state)
        if not os.path.exists(os.path.join(run_dir, 'EIGENVAL')):
            record['states'][state] = None
            continue
        incar = read_incar(os.path.join(run_dir, 'INCAR'))
        kp = open(os.path.join(run_dir, 'KPOINTS')).read().split('\n')[3].strip()
        nel, ks, energies = read_eigenval(os.path.join(run_dir, 'EIGENVAL'))
        nocc = nel // 2
        vb = [e[nocc - 1] for e in energies]
        cb = [e[nocc] for e in energies]
        trim_levels = {}
        for k, v, c in zip(ks, vb, cb):
            if is_trim(k):
                label = trim_label(k)
                if label not in trim_levels:
                    trim_levels[label] = [v, c]
        record['states'][state] = {'set': {k: incar.get(k, '') for k in TRACKED_INCAR_KEYS}, 'kp': kp,
                                   'nk': len(ks), 'nel': nel,
                                   'VBM': max(vb), 'VBMk': ks[vb.index(max(vb))], 'CBM': min(cb),
                                   'CBMk': ks[cb.index(min(cb))],
                                   'trim': trim_levels, 'toten': last_total_energy(run_dir)}
    print(json.dumps(record))
