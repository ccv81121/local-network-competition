#!/usr/bin/env bash
# usage: lev_run_scf.sh GPU DIR ; one fixed-ion SCF exactly like fworker.sh (same binary, env, launch line)
R=$WORK/spinel/LEVERAGE_20261023
source $R/env_gpu.sh
export CUDA_VISIBLE_DEVICES=$1; d=$2
cd "$d" || exit 1
echo "start $(date -u +%FT%TZ) gpu=$1" > runinfo.txt
nice -n 10 "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$VASP_GPU" > vasp.log 2>&1 < /dev/null
echo $? > vasp_exit_code.txt
if grep -q "General timing" OUTCAR; then touch DONE; else touch FAILED; fi
echo "end $(date -u +%FT%TZ)" >> runinfo.txt
rm -f WAVECAR CHG CHGCAR vasprun.xml
rm -f $R/gpu_running_$1
