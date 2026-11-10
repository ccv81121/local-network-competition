#!/usr/bin/env bash
# usage: run_rvalley_v1.sh GPU DIR ; fixed-ion SCF like lev_run_scf.sh, but keeps vasprun.xml and EIGENVAL
R=$WORK/spinel/LEVERAGE_20261023
W=$WORK/spinel/RVALLEY_20261104
source $R/env_gpu.sh
export CUDA_VISIBLE_DEVICES=$1; d=$2
cd "$d" || exit 1
echo "$(date -u +%FT%TZ) start $d gpu=$1" >> $W/RUNLOG
echo "start $(date -u +%FT%TZ) gpu=$1" > runinfo.txt
nice -n 10 "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$VASP_GPU" > vasp.log 2>&1 < /dev/null
rc=$?
echo $rc > vasp_exit_code.txt
if grep -q "General timing" OUTCAR; then touch DONE; else touch FAILED; fi
echo "end $(date -u +%FT%TZ)" >> runinfo.txt
rm -f WAVECAR CHG CHGCAR
echo "$(date -u +%FT%TZ) end $d gpu=$1 exit=$rc" >> $W/RUNLOG
