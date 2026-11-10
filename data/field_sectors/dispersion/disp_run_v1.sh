#!/usr/bin/env bash
# usage: disp_run_v1.sh GPU JOB ; one fixed-ion SCF (same binary/env/launch line as lev_run_scf.sh). Forces stay in OUTCAR.
R=$WORK/spinel/LEVERAGE_20261023
W=$WORK/spinel/DISPERSION_20261104
source $R/env_gpu.sh
export CUDA_VISIBLE_DEVICES=$1; j=$2; d=$W/jobs/$j
cd "$d" || exit 1
echo "$(date -u +%FT%TZ) start $j gpu=$1" >> $W/RUNLOG
echo "start $(date -u +%FT%TZ) gpu=$1" > runinfo.txt
nice -n 10 "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$VASP_GPU" > vasp.log 2>&1 < /dev/null
rc=$?
echo $rc > vasp_exit_code.txt
if grep -q "General timing" OUTCAR; then touch DONE; else touch FAILED; fi
echo "end $(date -u +%FT%TZ)" >> runinfo.txt
rm -f WAVECAR CHG CHGCAR PCDAT XDATCAR DOSCAR
echo "$(date -u +%FT%TZ) end $j gpu=$1 exit=$rc" >> $W/RUNLOG
