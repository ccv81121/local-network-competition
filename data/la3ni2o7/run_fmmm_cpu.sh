#!/usr/bin/env bash
# CPU relaxation of the tilt-free (Fmmm) reference at pressure P, same INCAR/KPOINTS/POTCAR as the Amam endpoints.
# usage: run_fmmm_cpu.sh ENV_FILE NPROC P
set -uo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
source "$ROOT/$1"
NP=$2; P=$3
d="$ROOT/fmmm/P$P"; mkdir -p "$d"; cd "$d" || exit 1
if [ ! -f DONE ]; then
  cp -f "$ROOT/poscar_fmmm_start" POSCAR; cp -f "$ROOT/POTCAR" POTCAR
  printf "Gamma\n0\nGamma\n8 8 2\n0 0 0\n" > KPOINTS
  cat > INCAR <<EOI
SYSTEM = La3Ni2O7 Fmmm reference P=$P GPa
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
NCORE = 4
KPAR = 2
EOI
  for cyc in 1 2 3 4 5 6; do
    nice -n 10 mpirun -np "$NP" "$VASP_CPU" > vasp_c$cyc.log 2>&1 < /dev/null; echo $? > EXIT_c$cyc.txt
    cp -f OUTCAR OUTCAR_c$cyc; cp -f OSZICAR OSZICAR_c$cyc; cp -f CONTCAR POSCAR
    n=$(grep -c "F=" OSZICAR)
    if grep -q "reached required accuracy" OUTCAR && [ "$n" -le 2 ]; then touch DONE; break; fi
  done
fi
echo "fmmm P$P finished $(date)" >> "$ROOT/STAGE0.log"
