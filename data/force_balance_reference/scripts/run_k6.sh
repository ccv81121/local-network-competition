#!/usr/bin/env bash
# Fixed-ion 6x6x6 band-edge readout of one force-balance structure on one GPU.  usage: run_k6.sh CASE GPU_ID
ROOT=$WORK/PRL_FORCEBALANCE_20261007
source $ROOT/env_gpu.sh
c=$1; export CUDA_VISIBLE_DEVICES=$2
d=$ROOT/k6/$c; cd $d || exit 1
cp $ROOT/POTCAR .
date -u +%FT%TZ > started_utc.txt
nice -n 10 "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$VASP_GPU" > vasp.log 2>&1 < /dev/null
echo $? > vasp_exit_code.txt
$HOME/.conda/envs/prl/bin/python3.11 $ROOT/k6/readout_k6.py $d > readout.log 2>&1 < /dev/null
rm -f WAVECAR CHG CHGCAR
date -u +%FT%TZ > finished_utc.txt
