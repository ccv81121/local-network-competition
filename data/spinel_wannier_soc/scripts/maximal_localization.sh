#!/usr/bin/env bash
# Maximal localisation (num_iter = 200) with the standalone wannier90.x, continued from the disentangled subspace of the
# NSCF run (restart = wannierise, its wannier90.chk). mp_grid, cell and atoms (from POSCAR) and the k points (in VASP's
# IBZKPT order) are appended to the library-mode wannier90.win. Then the exact split (wannier_margin_split.py) of the MLWF basis.
# usage: setsid nohup bash scripts/maximal_localization.sh > logs/mlwf_v2.log 2>&1 < /dev/null &
R=$WORK/PRL_TIGHT_UNIFY_20261102; PY=$HOME/.conda/envs/prl/bin/python3.11
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $R/logs/RUNLOG.txt; }
N=$R/wannier/nscf; M=$R/wannier/mlwf
mkdir -p $M/_v1_no_mpgrid; mv $M/w90.out $M/wannier90.werr $M/wpart.log $M/exit.txt $M/wannier90.win $M/_v1_no_mpgrid/ 2>/dev/null
cd $M
for f in amn mmn eig; do ln -sf $N/wannier90.$f .; done
cp $N/wannier90.chk $N/POSCAR $N/EIGENVAL .
python3 - $N <<'EOF'
import sys
from pathlib import Path
N = Path(sys.argv[1])
win = [l for l in (N / "wannier90.win").read_text().splitlines() if not l.strip().startswith(("num_iter", "restart", "num_bands"))]
L = (N / "POSCAR").read_text().split("\n"); s = float(L[1].split()[0])
A = [[float(x) * s for x in L[i].split()[:3]] for i in (2, 3, 4)]
sp, n = L[5].split(), [int(x) for x in L[6].split()]; i0 = 8 if L[7].strip()[0] in "Dd" else 9
el = sum(([e] * k for e, k in zip(sp, n)), [])
K = (N / "IBZKPT").read_text().split("\n"); nk = int(K[1].split()[0])
kp = [" ".join(K[3 + j].split()[:3]) for j in range(nk)]
assert nk == 64
out = win + ["num_bands = 286", "num_iter = 200", "restart = wannierise", "mp_grid = 4 4 4", "begin unit_cell_cart", "ang"]
out += [" ".join("%.10f" % x for x in a) for a in A] + ["end unit_cell_cart", "begin atoms_frac"]
out += ["%s %s" % (el[j], " ".join(L[i0 + j].split()[:3])) for j in range(sum(n))] + ["end atoms_frac", "begin kpoints"]
out += kp + ["end kpoints"]
Path("wannier90.win").write_text("\n".join(out) + "\n")
EOF
log "mlwf v2 start (CPU, serial, restart from nscf disentanglement)"
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data/software/wannier90-3.1.0/wannier90.x wannier90 > w90.out 2>&1; echo $? > exit.txt
[ -f wannier90_hr.dat ] && touch DONE
log "mlwf v2 exit $(cat exit.txt) $([ -f DONE ] && echo DONE || echo FAILED)"
if [ -f DONE ]; then $PY $R/scripts/wannier_margin_split.py $R/wannier mlwf > wpart.log 2>&1; log "wpart mlwf exit $?"; fi
