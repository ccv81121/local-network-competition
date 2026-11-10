#!/usr/bin/env bash
# Waits until the four RVALLEY_20261104 folders carry DONE or FAILED and no run_rvalley process is alive, then starts one worker per GPU 0-3.
W=$WORK/spinel/DISPERSION_20261104
RV=$WORK/spinel/RVALLEY_20261104
echo "$(date -u +%FT%TZ) queue started pid=$$ (waiting for RVALLEY)" >> $W/RUNLOG
while true; do
  ok=1
  for f in a000 a010 a020 a040; do
    [ -f $RV/$f/DONE ] || [ -f $RV/$f/FAILED ] || ok=0
  done
  pgrep -f run_rvalley >/dev/null && ok=0
  [ $ok = 1 ] && break
  sleep 60
done
echo "$(date -u +%FT%TZ) RVALLEY finished, GPUs free: launching workers" >> $W/RUNLOG
for g in 0 1 2 3; do nohup bash $W/scripts/disp_worker_v1.sh $g > $W/worker_$g.out 2>&1 < /dev/null & sleep 2; done
wait
touch $W/ALL_DONE
echo "$(date -u +%FT%TZ) queue complete" >> $W/RUNLOG
