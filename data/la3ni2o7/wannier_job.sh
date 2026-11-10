#!/usr/bin/env bash
# One stage-1c run in the current directory. In wan/, states other than 00 restart from ../00/WAVECAR; readout/
# runs from scratch. DONE only if VASP exits 0, the loop reached EDIFF and (in wan/) wannier90_hr.dat was written.
T=$(cd "$(dirname "$0")" && pwd)
source "$T/../env_gpu.sh"
cp -f "$T/../POTCAR" POTCAR
state=$(basename "$PWD"); tree=$(basename "$(dirname "$(dirname "$PWD")")")
restart=0; [ "$tree" = wan ] && [ "$state" != 00 ] && restart=1
[ $restart = 1 ] && { cp -f ../00/WAVECAR WAVECAR || { touch FAILED; exit 1; }; }
nice -n 10 "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$VASP_GPU" > vasp.log 2>&1 < /dev/null
rc=$?; echo $rc > exit.txt
[ $restart = 1 ] && rm -f WAVECAR
rm -f wannier90.mmn wannier90.amn                       # large; wannier90_hr.dat and .chk are kept
ok=0; [ $rc = 0 ] && grep -q "aborting loop because EDIFF is reached" OUTCAR && grep -q "General timing" OUTCAR && ok=1
[ "$tree" = wan ] && [ ! -s wannier90_hr.dat ] && ok=0
[ $ok = 1 ] && touch DONE || touch FAILED
