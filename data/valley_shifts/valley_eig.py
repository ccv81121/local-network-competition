import sys, numpy as np
# usage: valley_eig.py DIR kc1 kc2 ...  ; prints for each state the VBM(Gamma), lowest conduction at Gamma and at rival k
def read(fn):
    L = open(fn).read().split('\n')
    ne, nk, nb = map(int, L[5].split())
    i = 7; ks = []; E = []
    for _ in range(nk):
        k = list(map(float, L[i].split()[:3])); i += 1
        e = []; occ = []
        for b in range(nb):
            p = L[i].split(); e.append(float(p[1])); occ.append(float(p[2])); i += 1
        ks.append(k); E.append((e, occ)); i += 1
    return ne, np.array(ks), E
root = sys.argv[1]; rival = list(map(float, sys.argv[2:5]))
for s in ['state00', 'state10', 'state01', 'state11']:
    ne, ks, E = read(f'{root}/{s}/EIGENVAL')
    nocc = ne // 2
    def at(k):
        d = np.abs(((ks - k) + 0.5) % 1 - 0.5).sum(1); j = int(np.argmin(d)); assert d[j] < 1e-4, k
        return np.array(E[j][0])
    g = at([0, 0, 0]); r = at(rival)
    vb = g[nocc - 1]
    print(f'{s}: VBM(G) {vb:.4f}  CB(G) {g[nocc]:.4f}  CB(rival) {r[nocc]:.4f}  Lambda_C {1000*(r[nocc]-g[nocc]):.2f} meV  CB(G)-VBM {1000*(g[nocc]-vb):.1f}  CB(rival)-VBM {1000*(r[nocc]-vb):.1f}')
