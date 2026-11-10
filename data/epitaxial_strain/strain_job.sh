#!/usr/bin/env bash
# One strain job in the current directory; a final job (CHAIN) also runs its readout static in ./readout.
source $WORK/spinel/LEVERAGE_20261023/env_gpu.sh
run() { nice -n 10 "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$VASP_GPU" > vasp.log 2>&1 < /dev/null; grep -q "General timing" OUTCAR; }
run || { touch FAILED; exit 1; }
if [ -e CHAIN ]; then
  T=$(cd ../../../..; pwd); mkdir -p readout; cp CONTCAR readout/POSCAR; cp POTCAR readout/
  cp $WORK/PRL_MECHANISM_TEST_20261014/Path_Split/work/comp/S11/end/INCAR readout/INCAR
  cp $WORK/PRL_MECHANISM_TEST_20261014/Path_Split/work/comp/S11/end/KPOINTS readout/KPOINTS
  ( cd readout; run && touch DONE || touch FAILED )
  [ -e readout/DONE ] || { touch FAILED; exit 1; }
fi
rm -f WAVECAR CHG CHGCAR PROCAR DOSCAR readout/WAVECAR readout/CHG readout/CHGCAR; touch DONE
