#!/usr/bin/env bash
# One stage-1 SCF in the current directory (runs/<I>/<case>). Displaced cases restart from ../ref/WAVECAR.
# DONE only if VASP exits 0 and the electronic loop reached EDIFF; otherwise FAILED.
T=$(cd "$(dirname "$0")" && pwd)
source "$T/../env_gpu.sh"
cp -f "$T/../POTCAR" POTCAR
case $(basename "$PWD") in ref) ;; *) cp -f ../ref/WAVECAR WAVECAR || { touch FAILED; exit 1; } ;; esac
nice -n 10 "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$VASP_GPU" > vasp.log 2>&1 < /dev/null
rc=$?; echo $rc > exit.txt
[ "$(basename "$PWD")" = ref ] || rm -f WAVECAR
if [ $rc = 0 ] && grep -q "aborting loop because EDIFF is reached" OUTCAR && grep -q "General timing" OUTCAR; then
  touch DONE
else
  touch FAILED
fi
