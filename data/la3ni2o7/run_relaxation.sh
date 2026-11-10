#!/usr/bin/env bash
# Stage 0 for La3Ni2O7 (non-magnetic PBE, conventional 48-atom Amam cell).
# usage: run_relaxation.sh GPU TASK     TASK = pressure | conv | strain
#   pressure : sequential ISIF=3 relaxations at 0, 4, 8 GPa (each starts from the previous CONTCAR)
#   conv     : static forces on the experimental structure, KSPACING-equivalent meshes 8x8x2 / 10x10x3, SIGMA 0.05 / 0.02
#   strain   : waits for P0 DONE, then biaxial in-plane compression -1% and -2% with a,b fixed (LATTICE_CONSTRAINTS = F F T)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
source "$ROOT/env_gpu.sh"
export CUDA_VISIBLE_DEVICES=$1
TASK=$2
run() { nice -n 10 "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$VASP_GPU" > "$1" 2>&1 < /dev/null; echo $? > "EXIT_$1.txt"; }
common() { cat <<EOI
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
EOI
}
relax() {  # dir startPOSCAR pressure_GPa extra_incar_lines
  local d=$1 start=$2 P=$3 extra=$4
  mkdir -p "$d"; cd "$d" || exit 1
  [ -f DONE ] && return 0
  cp -f "$start" POSCAR; cp -f "$ROOT/POTCAR" POTCAR
  printf "Gamma\n0\nGamma\n8 8 2\n0 0 0\n" > KPOINTS
  { echo "SYSTEM = La3Ni2O7 $d"; common; cat <<EOI
EDIFFG = -2E-3
IBRION = 2
ISIF = 3
NSW = 200
PSTRESS = $(awk -v p="$P" 'BEGIN{printf "%.4f", p*10}')
EOI
    printf "%b" "$extra"; } > INCAR
  for cyc in 1 2 3 4 5 6; do
    run vasp_c$cyc.log
    cp -f OUTCAR OUTCAR_c$cyc; cp -f OSZICAR OSZICAR_c$cyc; cp -f CONTCAR POSCAR
    n=$(grep -c "F=" OSZICAR)
    if grep -q "reached required accuracy" OUTCAR && [ "$n" -le 2 ]; then touch DONE; break; fi
  done
}
case $TASK in
  pressure)
    prev="$ROOT/poscar_exp"
    for P in 0 4 8; do relax "$ROOT/pressure/P$P" "$prev" "$P" ""; prev="$ROOT/pressure/P$P/CONTCAR"; done ;;
  conv)
    for k in "8 8 2" "10 10 3"; do for s in 0.05 0.02; do
      d="$ROOT/conv/k${k// /}_s$s"; mkdir -p "$d"; cd "$d" || exit 1
      [ -f DONE ] && continue
      cp -f "$ROOT/poscar_exp" POSCAR; cp -f "$ROOT/POTCAR" POTCAR
      printf "Gamma\n0\nGamma\n$k\n0 0 0\n" > KPOINTS
      { echo "SYSTEM = La3Ni2O7 conv k=$k sigma=$s"; common | sed "s/^SIGMA = .*/SIGMA = $s/"; } > INCAR
      run vasp.log && touch DONE
    done; done ;;
  strain)
    while [ ! -f "$ROOT/pressure/P0/DONE" ]; do sleep 120; done
    base="$ROOT/pressure/P0/CONTCAR"
    for eta in -1 -2; do
      d="$ROOT/strain/eta$eta"; mkdir -p "$d"
      awk -v f="$(awk -v e="$eta" 'BEGIN{printf "%.6f", 1+e/100}')" 'NR==3||NR==4{printf "%.10f %.10f %.10f\n",$1*f,$2*f,$3*f; next}{print}' "$base" > "$d/POSCAR_start"
      relax "$d" "$d/POSCAR_start" 0 "LATTICE_CONSTRAINTS = .FALSE. .FALSE. .TRUE.\n"
    done ;;
esac
echo "stage0 $TASK finished $(date)" >> "$ROOT/STAGE0.log"
