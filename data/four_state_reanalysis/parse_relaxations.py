#!/usr/bin/env python3
"""Read-only batch parser of VASP run dirs.

For each dir containing INCAR and OUTCAR: INCAR keys, KPOINTS mesh, ionic steps,
NSW cap hit, reached-accuracy flag, final max |F| (TOTAL-FORCE, last block), final pressure.
Skips static/displacement runs (NSW<=1) unless -a given.

usage: parse_relaxations.py [-a] ROOT [ROOT...] > out.tsv
"""
import os, sys, math


def read_incar(path):
    settings = {}
    for line in open(path, errors='ignore'):
        line = line.split('#')[0].split('!')[0]
        for part in line.split(';'):
            if '=' in part:
                k, v = part.split('=', 1)
                settings[k.strip().upper()] = v.strip()
    return settings


def kpoints_mesh(path):
    try:
        lines = open(path, errors='ignore').read().split('\n')
        return (lines[2].strip()[:1] + ':' + 'x'.join(lines[3].split()[:3]))
    except Exception:
        return '?'


def parse_outcar(path):
    steps = 0
    last_force = None
    reached = False
    finished = False
    pressure = None
    nions = None
    with open(path, errors='ignore') as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        if nions is None and 'NIONS' in line:
            nions = int(line.split('NIONS =')[1].split()[0])
        elif 'TOTAL-FORCE' in line:
            steps += 1
            force_norms = []
            for j in range(i + 2, i + 2 + nions):
                v = lines[j].split()
                try:
                    force_norms.append(math.sqrt(sum(float(x) ** 2 for x in v[3:6])))
                except Exception:
                    break
            if len(force_norms) == nions:
                last_force = (max(force_norms), math.sqrt(sum(x * x for x in force_norms) / len(force_norms)))
        elif 'reached required accuracy' in line:
            reached = True
        elif 'General timing' in line:
            finished = True
        elif 'external pressure' in line:
            pressure = line.split('=')[1].split()[0]
    return steps, last_force, reached, finished, pressure, nions


def main():
    show_all = '-a' in sys.argv
    roots = [a for a in sys.argv[1:] if a != '-a']
    print('\t'.join(['dir', 'ISIF', 'IBRION', 'NSW', 'EDIFFG', 'PSTRESS', 'ENCUT', 'ISMEAR', 'SIGMA', 'ISYM', 'NBANDS',
                     'XC', 'KP', 'nions', 'steps', 'capHit', 'reachedAcc', 'finished', 'Fmax', 'Frms', 'extP_kB']))
    for root in roots:
        for dirpath, dirnames, filenames in sorted(os.walk(root)):
            dirnames.sort()
            if 'INCAR' in filenames and 'OUTCAR' in filenames:
                incar = read_incar(os.path.join(dirpath, 'INCAR'))
                nsw = int(float(incar.get('NSW', '0') or 0))
                if nsw <= 1 and not show_all:
                    continue
                try:
                    steps, last_force, reached, finished, pressure, nions = parse_outcar(
                        os.path.join(dirpath, 'OUTCAR'))
                except Exception:
                    steps, last_force, reached, finished, pressure, nions = 0, None, False, False, None, None
                functional = incar.get('METAGGA', '') or incar.get('GGA', 'PE')
                if incar.get('LHFCALC', '').upper().startswith('.T'):
                    functional += '+HF'
                if incar.get('LDAU', '').upper().startswith('.T'):
                    functional += '+U'
                print('\t'.join(map(str, [
                    dirpath, incar.get('ISIF', '-'), incar.get('IBRION', '-'), nsw, incar.get('EDIFFG', '-'),
                    incar.get('PSTRESS', '-'), incar.get('ENCUT', '-'), incar.get('ISMEAR', '-'),
                    incar.get('SIGMA', '-'), incar.get('ISYM', '-'), incar.get('NBANDS', '-'), functional,
                    kpoints_mesh(os.path.join(dirpath, 'KPOINTS')) if 'KPOINTS' in filenames
                    else 'KSPACING=' + incar.get('KSPACING', '?'),
                    nions, steps, int(steps >= nsw), int(reached), int(finished),
                    '%.4f' % last_force[0] if last_force else 'NA', '%.4f' % last_force[1] if last_force else 'NA',
                    pressure])))


main()
