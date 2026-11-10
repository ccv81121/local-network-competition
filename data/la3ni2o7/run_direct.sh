#!/usr/bin/env bash
# Stage-0 helper: relax 8 and 4 GPa directly from the experimental structure in pressure_direct/, in parallel with the
# sequential chain of run_relaxation.sh pressure. A finished point is copied to pressure/P$P and marked DONE only if the
# chain has not started that directory; otherwise it stays in pressure_direct/ as a cross-check.
# usage: run_direct.sh GPU
set -uo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
source "$ROOT/env_gpu.sh"
export CUDA_VISIBLE_DEVICES=$1
run() { nice -n 10 "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$VASP_GPU" > "$1" 2>&1 < /dev/null; echo $? > "EXIT_$1.txt"; }
for P in 8 4; do
  d="$ROOT/pressure_direct/P$P"; mkdir -p "$d"; cd "$d" || exit 1
  if [ ! -f DONE ]; then
    cp -f "$ROOT/poscar_exp" POSCAR; cp -f "$ROOT/POTCAR" POTCAR
    printf "Gamma\n0\nGamma\n8 8 2\n0 0 0\n" > KPOINTS
    cat > INCAR <<EOI
SYSTEM = La3Ni2O7 direct P=$P GPa
PREC = Accurate
ENCUT = 520
EDIFF = 1E-8
ISPIN = 1
ISMEAR = 1
SIGMA = 0.05
LREAL = .FALSE.
LASPH = .TRUE.
LWAVE = .FALSE.
LCHARG = .FALSE.
NELM = 200
EDIFFG = -2E-3
IBRION = 2
ISIF = 3
NSW = 200
PSTRESS = $(awk -v p="$P" 'BEGIN{printf "%.4f", p*10}')
EOI
    for cyc in 1 2 3 4 5 6; do
      run vasp_c$cyc.log
      cp -f OUTCAR OUTCAR_c$cyc; cp -f OSZICAR OSZICAR_c$cyc; cp -f CONTCAR POSCAR
      n=$(grep -c "F=" OSZICAR)
      if grep -q "reached required accuracy" OUTCAR && [ "$n" -le 2 ]; then touch DONE; break; fi
    done
  fi
  t="$ROOT/pressure/P$P"
  if [ -f DONE ] && [ ! -e "$t" ]; then
    mkdir -p "$t" && cp -p "$d"/* "$t"/ && echo "copied from pressure_direct/P$P" > "$t/SOURCE.txt" && touch "$t/DONE"
  fi
done
echo "direct finished $(date)" >> "$ROOT/STAGE0.log"
