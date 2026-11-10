#!/usr/bin/env bash
# usage: disp_worker_cpu_v1.sh NP ; CPU worker. Walks joblist.txt from the END and claims a +/- displacement PAIR only if both members
# are unclaimed (same mkdir claim locks as the GPU workers), so each finite-difference pair runs on one platform. Never runs 'ref'.
W=$WORK/spinel/DISPERSION_20261104
np=$1
for j in $(tac $W/jobs/joblist.txt); do
  case $j in *p) ;; *) continue ;; esac          # visit each pair once, via its 'p' member
  m=${j%p}m
  [ -f $W/jobs/$j/DONE ] && [ -f $W/jobs/$m/DONE ] && continue
  mkdir $W/claims/$j 2>/dev/null || continue
  if ! mkdir $W/claims/$m 2>/dev/null; then rmdir $W/claims/$j; continue; fi
  bash $W/scripts/disp_run_cpu_v1.sh $np $j
  bash $W/scripts/disp_run_cpu_v1.sh $np $m
done
echo "$(date -u +%FT%TZ) worker cpu np=$np finished" >> $W/RUNLOG
