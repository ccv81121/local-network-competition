#!/usr/bin/env bash
# usage: disp_run_cpu_v1.sh NP JOB ; one fixed-ion SCF on CPU (Intel MPI + MKL, VASP 6.5.1 CPU build), same INCAR/KPOINTS/POTCAR as the GPU jobs.
W=$WORK/spinel/DISPERSION_20261104
source $W/scripts/env_cpu152_v1.sh
np=$1; j=$2; d=$W/jobs/$j
cd "$d" || exit 1
echo "$(date -u +%FT%TZ) start $j cpu np=$np" >> $W/RUNLOG
echo "start $(date -u +%FT%TZ) cpu np=$np" > runinfo.txt
nice -n 10 mpirun -np "$np" "$VASP_CPU" > vasp.log 2>&1 < /dev/null
rc=$?
echo $rc > vasp_exit_code.txt
if grep -q "General timing" OUTCAR; then touch DONE; else touch FAILED; fi
echo "end $(date -u +%FT%TZ)" >> runinfo.txt
rm -f WAVECAR CHG CHGCAR PCDAT XDATCAR DOSCAR
echo "$(date -u +%FT%TZ) end $j cpu exit=$rc" >> $W/RUNLOG
