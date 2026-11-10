#!/usr/bin/env bash
# usage: alloy_run.sh GPU CASE1 [CASE2 ...] ; s0/s1 statics, ISIF=2 relax (up to 6 cycles), decomposition, four statics, readout
ROOT=$WORK/PRL_MECHANISM_TEST_20261014; W=$ROOT/Alloy_Local
PY=$HOME/.conda/envs/prl/bin/python3.11
source $ROOT/env_gpu.sh
export CUDA_VISIBLE_DEVICES=$1; shift
run() { nice -n 10 "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$VASP_GPU" > "$1" 2>&1 < /dev/null; }
static() { cd "$1" || return; [ -f DONE ] && return; run vasp.log; echo $? > vasp_exit_code.txt
  grep -q "General timing" OUTCAR && touch DONE; rm -f WAVECAR CHG CHGCAR vasprun.xml; }
for C in "$@"; do
  D=$W/$C; echo "$(date -u) start $C" >> $W/run.log
  static $D/s0; static $D/s1
  cd $D/relax
  if [ ! -f DONE ]; then
    for cyc in 1 2 3 4 5 6; do
      run vasp_c$cyc.log; echo $? > EXIT_c$cyc.txt
      cp -f OUTCAR OUTCAR_c$cyc; cp -f OSZICAR OSZICAR_c$cyc; cp -f CONTCAR POSCAR
      n=$(grep -c "F=" OSZICAR)
      if grep -q "reached required accuracy" OUTCAR && [ "$n" -le 2 ]; then touch DONE; break; fi
    done
    rm -f WAVECAR CHG CHGCAR
  fi
  [ -f DONE ] || { echo "$(date -u) $C relax NOT converged" >> $W/run.log; continue; }
  $PY $W/alloy_decomposition.py $D > $D/decomp.log 2>&1 || { echo "$(date -u) $C decomp failed" >> $W/run.log; continue; }
  for s in state00 state10 state01 state11; do static $D/fourstate/$s; done
  $PY $W/alloy_readout.py $D > $D/readout.log 2>&1
  echo "$(date -u) done $C" >> $W/run.log
done
touch $W/DONE_gpu$CUDA_VISIBLE_DEVICES
