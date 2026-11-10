import sys, re, json, numpy as np
# usage: procar_valley.py RUNDIR ; orbital weights of the lowest conduction band at Gamma and (0,0,1/2), each state
root = sys.argv[1]
out = {}
for s in ['state00', 'state10', 'state01', 'state11']:
    pos = open(f'{root}/{s}/POSCAR').read().split('\n')
    species = pos[5].split(); counts = list(map(int, pos[6].split()))
    lab = sum(([sp] * c for sp, c in zip(species, counts)), [])
    T = open(f'{root}/{s}/PROCAR').read()
    nk, nb, ni = map(int, re.search(r'# of k-points:\s*(\d+)\s+# of bands:\s*(\d+)\s+# of ions:\s*(\d+)', T).groups())
    ne = float(re.search(r'NELECT\s*=\s*([\d.]+)', open(f'{root}/{s}/OUTCAR').read()).group(1))
    nocc = int(round(ne)) // 2
    blocks = re.split(r'\n\s*k-point\s+\d+\s*:', T)[1:]
    res = {}
    for blk in blocks:
        k = np.array([float(x) for x in re.findall(r'-?\d+\.\d+', blk[:60])[:3]])
        name = None
        if np.allclose(k, 0, atol=1e-4): name = 'Gamma'
        elif np.allclose(np.abs(k), [0, 0, 0.5], atol=1e-4): name = '00h'
        if name is None or name in res: continue
        bands = re.split(r'\nband\s+', blk)[1:]
        b = bands[nocc]  # lowest conduction band (0-based index nocc)
        en = float(re.search(r'energy\s+(-?[\d.]+)', b).group(1))
        rows = [l.split() for l in b.split('\n') if re.match(r'\s*\d+\s', l) and '#' not in l][:ni]
        A = np.array([[float(x) for x in r[1:]] for r in rows])  # s py pz px dxy dyz dz2 dxz dx2 tot
        tot = A[:, -1].sum()
        w = {}
        for sp in species:
            idx = [i for i, l in enumerate(lab) if l == sp]
            w[sp + '_s'] = A[idx, 0].sum() / tot
            w[sp + '_p'] = A[idx, 1:4].sum() / tot
            w[sp + '_d'] = A[idx, 4:9].sum() / tot
        res[name] = {'E_eV': en, 'weights': {kk: round(v, 3) for kk, v in w.items() if v >= 0.01}}
    out[s] = res
print(json.dumps(out, indent=1))
json.dump(out, open(f'{root}/valley_weights.json', 'w'), indent=1)
