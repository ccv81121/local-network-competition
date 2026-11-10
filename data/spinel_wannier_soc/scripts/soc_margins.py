"""SOC screen on the tight structures: band-edge margins from EIGENVAL, on the 4x4x4 mesh.

The definitions are those of the earlier screen (07_HSE_SOC_Archives/PRESSURE_SOC_ARCHIVED_ENDPOINTS.csv):
Lambda_V = E_V(Gamma) - max_{k != Gamma} E_V(k), Lambda_C = E_C(R) - E_C(Gamma). SOC values come from
RUNDIR/soc/P<P>/soc, PBE values from the collinear seeds RUNDIR/soc/P<P>/seed of the same structures.

RUNDIR is the cluster run folder; its EIGENVAL and DONE files are not deposited (results/soc_margins.csv is the result).
usage: python soc_margins.py RUNDIR > soc/soc_margins.csv
"""
import sys
from pathlib import Path


def margins(eig, ncl):
    lines = Path(eig).read_text().split("\n")
    ne, nk, nb = [int(x) for x in lines[5].split()]
    nocc = ne if ncl else ne // 2
    K, E, i = [], [], 7
    for _ in range(nk):
        K.append(tuple(round(float(x), 4) for x in lines[i].split()[:3]))
        E.append([float(lines[i + 1 + j].split()[1]) for j in range(nb)])
        i += nb + 2
    vb = [e[nocc - 1] for e in E]
    cb = [e[nocc] for e in E]
    i_gamma = K.index((0.0, 0.0, 0.0))
    i_R = next(j for j, k in enumerate(K) if all(abs(abs(x) - 0.5) < 1e-3 for x in k))
    lv = vb[i_gamma] - max(vb[j] for j in range(nk) if j != i_gamma)
    lc = cb[i_R] - cb[i_gamma]
    return 1000 * lv, 1000 * lc, K[max(range(nk), key=lambda j: vb[j])], K[min(range(nk), key=lambda j: cb[j])], nk


RUN_DIR = Path(sys.argv[1])
print("P_GPa,Lambda_V_SOC_meV,Lambda_C_SOC_meV,VBM_k,CBM_k,nk_SOC,Lambda_V_PBE_meV,Lambda_C_PBE_meV,nk_PBE")
for P in ("4", "5", "6", "7.5"):
    case_dir = RUN_DIR / "soc" / f"P{P}"
    assert (case_dir / "soc" / "DONE").exists() and (case_dir / "seed" / "DONE").exists(), P
    sv, sc, kv, kc, nks = margins(case_dir / "soc" / "EIGENVAL", True)
    pv, pc, _, _, nkp = margins(case_dir / "seed" / "EIGENVAL", False)
    print(f"{P},{sv:.1f},{sc:.1f},\"{kv}\",\"{kc}\",{nks},{pv:.1f},{pc:.1f},{nkp}")
