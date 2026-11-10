#!/usr/bin/env bash
# Continue the -2% biaxial relaxation (strain/eta-2) from its current CONTCAR on several GPUs (one MPI rank per GPU,
# KPAR = number of GPUs). Same INCAR as run_relaxation.sh plus KPAR; same cycle and DONE logic.
# usage: run_eta2_multi.sh GPU_LIST      e.g. run_eta2_multi.sh 1,2,3
set -uo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
source "$ROOT/env_gpu.sh"
export CUDA_VISIBLE_DEVICES=$1
NG=$(echo "$1" | tr ',' '\n' | wc -l)
d="$ROOT/strain/eta-2"; cd "$d" || exit 1
[ -f DONE ] && exit 0
[ -s CONTCAR ] && cp -f CONTCAR POSCAR
grep -q "^KPAR" INCAR || echo "KPAR = $NG" >> INCAR
run() { nice -n 10 "$MPI_ROOT/bin/mpirun" -np "$NG" --bind-to none "$VASP_GPU" > "$1" 2>&1 < /dev/null; echo $? > "EXIT_$1.txt"; }
for cyc in m1 m2 m3 m4 m5 m6; do
  run vasp_$cyc.log
  cp -f OUTCAR OUTCAR_$cyc; cp -f OSZICAR OSZICAR_$cyc; cp -f CONTCAR POSCAR
  n=$(grep -c "F=" OSZICAR)
  if grep -q "reached required accuracy" OUTCAR && [ "$n" -le 2 ]; then touch DONE; break; fi
done
echo "stage0 strain eta-2 (GPUs $1) finished $(date)" >> "$ROOT/STAGE0.log"
