"""BaSnTi full9: cell size, supercell matrix wrt pseudo-cubic a0, whether parent R=(1/2,1/2,1/2)2pi/a0 folds to
supercell Gamma, and V-side truth / prediction (read-only).

Inputs:  inputs/basnti/poscar_<x> for x = x000 ... x100 (the nine canonical POSCARs), inputs/basnti/edge_truth_full9_k8.csv
         (V-side truth), inputs/basnti/vside_confidence_replay.csv (frozen prediction).
Output:  tab-separated table and the three rule scores on stdout (committed as basnti_folding.txt).
usage:   python basnti_folding.py   (no arguments)
"""
import numpy as np, csv, sys
from pathlib import Path
INPUT_DIR = Path(__file__).resolve().parent / "inputs" / "basnti"
truth = {r['sample_id']: r
         for r in csv.DictReader(open(INPUT_DIR / "edge_truth_full9_k8.csv", encoding='utf-8'))}
pred = {r['sample_id']: r
        for r in csv.DictReader(open(INPUT_DIR / "vside_confidence_replay.csv", encoding='utf-8'))}


rows = []
for x in ['x000', 'x0125', 'x025', 'x0375', 'x050', 'x0625', 'x075', 'x0875', 'x100']:
    lines = open(INPUT_DIR / f"poscar_{x}").read().split('\n')
    scale = float(lines[1])
    A = np.array([[float(v) for v in lines[i].split()] for i in (2, 3, 4)]) * scale
    species = lines[5].split()
    counts = [int(v) for v in lines[6].split()]
    frac = np.array([[float(v) for v in lines[8 + i].split()[:3]] for i in range(sum(counts))])
    nfu = counts[0]
    a0 = (abs(np.linalg.det(A)) / nfu) ** (1 / 3)
    ba = frac[:counts[0]] @ A
    # q_R = (pi/a0)(1,1,1) folds to Gamma iff every lattice vector A_i is an even number of nearest-neighbour Ba-Ba hops
    # along the cubic axes; the axes come from the shortest Ba-Ba vectors (|d| ~ a0) in a 3x3x3 image search.
    imgs = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    vecs = []
    for r1 in ba:
        for r2 in ba:
            for im in imgs:
                d = r2 + im @ A - r1
                length = np.linalg.norm(d)
                if 0.8 * a0 < length < 1.2 * a0:
                    vecs.append(d)
    vecs = np.array(vecs)
    e1 = vecs[0] / np.linalg.norm(vecs[0])
    e2 = [v for v in vecs if abs(v @ e1) / np.linalg.norm(v) < 0.2][0]
    e2 = e2 - (e2 @ e1) * e1
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    C = np.array([e1, e2, e3])
    Mi = (A @ C.T) / a0  # lattice vectors in cubic units
    Mr = np.round(Mi).astype(int)
    err = np.abs(Mi - Mr).max()
    folds = all(int(r.sum()) % 2 == 0 for r in Mr)
    truth_row = truth[f'BaSnTi_{x}']
    pred_row = pred[f'BaSnTi_{x}']
    rows.append([x, sum(counts), round(a0, 3), Mr.tolist(), round(err, 3), folds, truth_row['V_edge_label'],
                 truth_row['VBM_k'], truth_row['truth_status'], pred_row['V_truth'],
                 pred_row['legacy_SP_binary'], pred_row['V_correct']])
writer = csv.writer(sys.stdout, delimiter='\t')
writer.writerow(['x', 'natoms', 'a0', 'M_cubic', 'round_err', 'R_folds_to_Gamma', 'V_label', 'VBM_k', 'truth_status',
                 'V_truth(1=Gamma)', 'pred', 'pred_correct'])
for r in rows:
    writer.writerow(r)
print('cell-size rule (natoms in {5,20} -> qV, {10,40} -> Gamma):',
      sum((r[1] in (10, 40)) == (r[9] == '1') for r in rows), '/ 9')
print('R-folding rule (R folds to supercell Gamma -> Gamma):', sum(r[5] == (r[9] == '1') for r in rows), '/ 9')
print('model:', sum(r[11] == 'True' for r in rows), '/ 9')
