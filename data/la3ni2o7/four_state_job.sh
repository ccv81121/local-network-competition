#!/usr/bin/env bash
# One stage-1b readout SCF in the current directory. In runs/ and runs_U/, states other than 00 restart from
# ../00/WAVECAR; runs_k19/ runs from scratch. DONE only if VASP exits 0 and the loop reached EDIFF.
T=$(cd "$(dirname "$0")" && pwd)
source "$T/../env_gpu.sh"
cp -f "$T/../POTCAR" POTCAR
state=$(basename "$PWD"); tree=$(basename "$(dirname "$(dirname "$PWD")")")
restart=0; [ "$tree" != runs_k19 ] && [ "$state" != 00 ] && restart=1
[ $restart = 1 ] && { cp -f ../00/WAVECAR WAVECAR || { touch FAILED; exit 1; }; }
nice -n 10 "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$VASP_GPU" > vasp.log 2>&1 < /dev/null
rc=$?; echo $rc > exit.txt
[ $restart = 1 ] && rm -f WAVECAR
if [ $rc = 0 ] && grep -q "aborting loop because EDIFF is reached" OUTCAR && grep -q "General timing" OUTCAR; then
  touch DONE
else
  touch FAILED
fi
