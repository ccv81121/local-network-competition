#!/usr/bin/env bash
# One hull job in the current directory: relax (two ISIF=3 rounds) then static, or static only. Writes DONE or FAILED.
source $WORK/spinel/LEVERAGE_20261023/env_gpu.sh
run() { nice -n 10 "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$VASP_GPU" > vasp_$1.log 2>&1 < /dev/null; grep -q "General timing" OUTCAR || return 1; cp OUTCAR OUTCAR.$1; }
set -e
if [ "$(cat KIND)" = relax ]; then
  cp INCAR.r INCAR; run r1 || { touch FAILED; exit 1; }; cp CONTCAR POSCAR.r1; cp CONTCAR POSCAR
  run r2 || { touch FAILED; exit 1; }; cp CONTCAR POSCAR.r2; cp CONTCAR POSCAR
fi
cp INCAR.s INCAR; run s || { touch FAILED; exit 1; }
rm -f WAVECAR CHG CHGCAR PROCAR DOSCAR; touch DONE
