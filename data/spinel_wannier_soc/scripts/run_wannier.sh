#!/usr/bin/env bash
# Wannier chain on one GPU: SCF (6x6x6) -> windows -> NSCF 4x4x4 with Wannier90 (edge12) -> standalone MLWF -> split.
# usage: nohup bash scripts/run_wannier.sh GPU > logs/wannier.log 2>&1 &
R=$WORK/PRL_TIGHT_UNIFY_20261102; G=$1; PY=$HOME/.conda/envs/prl/bin/python3.11
source $WORK/PRL_TIGHT_SPINEL_20261015/env_gpu.sh; export CUDA_VISIBLE_DEVICES=$G
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $R/logs/RUNLOG.txt; }
run() { "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$1" > vasp.out 2>&1 < /dev/null; echo $? > exit.txt; }

cd $R/wannier/scf; if [ ! -f DONE ]; then log "wannier scf start gpu$G"; run $VASP_GPU; grep -q "reached required accuracy\|aborting loop because EDIFF is reached" OUTCAR && touch DONE; log "wannier scf exit $(cat exit.txt)"; fi
[ -f DONE ] || { log "wannier scf FAILED"; exit 1; }

cd $R/wannier/nscf
if [ ! -f DONE ]; then
  cp ../scf/CHGCAR .; $PY $R/scripts/fill_windows.py ../scf/EIGENVAL ../edge12.win.tmpl wannier90.win | tee -a $R/logs/RUNLOG.txt
  log "wannier nscf start gpu$G"; run /data/vasp/vasp.6.5.1_gpu_w90/bin/vasp_std
  [ -f wannier90_hr.dat ] && touch DONE; log "wannier nscf exit $(cat exit.txt)"; rm -f CHGCAR
fi
[ -f DONE ] || { log "wannier nscf FAILED"; exit 1; }

# MLWF from the same matrices: VASP's completed win (cell, atoms, k points) with num_iter = 200
cd $R/wannier/mlwf
if [ ! -f DONE ]; then
  for f in amn mmn eig; do ln -sf ../nscf/wannier90.$f .; done; cp ../nscf/POSCAR ../nscf/EIGENVAL .
  sed 's/^num_iter *=.*/num_iter = 200/' ../nscf/wannier90.win > wannier90.win
  log "mlwf start (CPU, serial)"
  export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  /data/software/wannier90-3.1.0/wannier90.x wannier90 > w90.out 2>&1; echo $? > exit.txt
  [ -f wannier90_hr.dat ] && touch DONE; log "mlwf exit $(cat exit.txt)"
fi

cd $R/wannier
for v in nscf mlwf; do $PY $R/scripts/wannier_margin_split.py $R/wannier $v > $v/wpart.log 2>&1; log "wpart $v exit $?"; done
touch $R/wannier/ALL_DONE; log "wannier chain done"
