#!/usr/bin/env bash
# Thermal statics on the CPU machine: 32 statics, 2 at a time x 43 ranks (NBANDS 172 = 4 x 43, unchanged by VASP).
# Settings are those of static_k4 (tight path readout); only SYSTEM differs. Writes DONE/FAILED per job, ALL_DONE at the end.
set --
source /usr/local/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
R=$WORK/THERMAL_VALLEY_20261025; VASP=$WORK/apps/vasp.6.5.1_cpu_w90_src/bin/vasp_std; NP=43
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; ulimit -s unlimited
ORDER=$(ls $R/jobs | awk '/T300_P5/{print 0, $0; next} /T300_P4/{print 1, $0; next} {print 2, $0}' | sort | cut -d' ' -f2)
worker() {
  for j in $ORDER; do
    d=$R/jobs/$j; mkdir $d/LOCK 2>/dev/null || continue
    cd $d; echo "$(date -u +%FT%TZ) start $j" >> $R/run.log
    mpirun -np $NP $VASP > vasp.log 2>&1 < /dev/null
    if grep -q "General timing" OUTCAR 2>/dev/null; then touch DONE; s=done; else touch FAILED; s=FAILED; fi
    rm -f WAVECAR CHG CHGCAR vasprun.xml PROCAR; echo "$(date -u +%FT%TZ) $s $j" >> $R/run.log
  done
}
worker & sleep 5; worker & wait
echo "$(date -u +%FT%TZ) ALL_DONE failed=$(ls $R/jobs/*/FAILED 2>/dev/null | wc -l)" >> $R/run.log; touch $R/ALL_DONE
