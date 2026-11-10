#!/usr/bin/env bash
# Disentanglement to convergence (dis_conv_tol 1e-10, window 5) and maximal localisation (num_iter 1000) from the
# amn/mmn/eig of the 4x4x4 NSCF run, same windows and projections, on an MPI build of Wannier90 3.1.0 with one k point
# per rank; then the exact split (wannier_margin_split.py), which also counts Wannier levels inside the DFT gap.
# A disentanglement limited to 400 iterations leaves one Mg-s Wannier level 0.3 eV inside the gap at all 64 k points.
# usage: setsid nohup bash scripts/disentangle.sh CPUSET NP > logs/dis_v2.log 2>&1 < /dev/null &
R=$WORK/PRL_TIGHT_UNIFY_20261102; PY=$HOME/.conda/envs/prl/bin/python3.11; CPUS=$1; NP=$2
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $R/logs/RUNLOG.txt; }
N=$R/wannier/nscf; M=$R/wannier/dis_v2
mkdir $M || { log "dis_v2: $M exists"; exit 1; }
cd $M
for f in amn mmn eig; do ln -s $N/wannier90.$f .; done
cp $N/POSCAR $N/EIGENVAL .
python3 - $N $R/wannier/mlwf/wannier90.win <<'PY'
import sys
from pathlib import Path
N, ref = Path(sys.argv[1]), Path(sys.argv[2])
win = [l for l in (N / "wannier90.win").read_text().splitlines() if not l.strip().startswith(("num_iter", "restart", "num_bands", "dis_num_iter"))]
tail = ref.read_text().splitlines(); tail = tail[tail.index("mp_grid = 4 4 4"):]   # mp_grid, cell, atoms, k points as in wannier/mlwf
Path("wannier90.win").write_text("\n".join(win + ["num_bands = 286", "dis_num_iter = 20000", "num_iter = 1000"] + tail) + "\n")
PY
grep -q "^begin kpoints" wannier90.win || { log "dis_v2: win incomplete"; exit 1; }
source $HOME/intel/oneapi/setvars.sh > /dev/null 2>&1; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
log "dis_v2 start (MPI $NP ranks, cpus $CPUS, to convergence)"
taskset -c $CPUS mpirun -np $NP $R/tools/wannier90-3.1.0-mpi/wannier90.x wannier90 > w90.out 2>&1; echo $? > exit.txt
[ -f wannier90_hr.dat ] && touch DONE
log "dis_v2 exit $(cat exit.txt) $([ -f DONE ] && echo DONE || echo FAILED) $(grep -c 'DIS$' wannier90.wout) dis-iter"
if [ -f DONE ]; then $PY $R/scripts/wannier_margin_split.py $R/wannier dis_v2 > wpart.log 2>&1; log "wpart dis_v2 exit $? valid=$(grep -o '"valid": [a-z]*' wpart.json)"; fi
