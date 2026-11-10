#!/usr/bin/env bash
# usage: run_k08procar_v1.sh GPU DIR ; fixed-ion SCF of a k08 four state with LORBIT=11 (PROCAR)
R=$WORK/spinel/LEVERAGE_20261023
W=$WORK/spinel/K08_PROCAR_20261108
source $R/env_gpu.sh
export CUDA_VISIBLE_DEVICES=$1; d=$2
cd "$W/$d" || exit 1
echo "$(date -u +%FT%TZ) start $d gpu=$1" >> $W/RUNLOG
nice -n 10 "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$VASP_GPU" > vasp.log 2>&1 < /dev/null
rc=$?
if grep -q "General timing" OUTCAR; then touch DONE; else touch FAILED; fi
rm -f WAVECAR CHG CHGCAR
echo "$(date -u +%FT%TZ) end $d gpu=$1 exit=$rc" >> $W/RUNLOG
