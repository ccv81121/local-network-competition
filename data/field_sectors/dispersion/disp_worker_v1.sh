#!/usr/bin/env bash
# usage: disp_worker_v1.sh GPU ; pulls jobs from joblist.txt in order (mkdir claim lock), one at a time on its GPU
W=$WORK/spinel/DISPERSION_20261104
g=$1
while read -r j; do
  [ -f $W/jobs/$j/DONE ] && continue
  mkdir $W/claims/$j 2>/dev/null || continue
  bash $W/scripts/disp_run_v1.sh $g $j
done < $W/jobs/joblist.txt
echo "$(date -u +%FT%TZ) worker gpu=$g finished" >> $W/RUNLOG
