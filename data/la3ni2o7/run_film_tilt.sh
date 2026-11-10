#!/usr/bin/env bash
# Tilt check: does the octahedral tilt survive biaxial compression when the relaxation starts tilt-free?
#   s2_fmmm   : tilt-free start (poscar_fmmm_start fractions) in the -2% cell of strain/eta-2, a,b fixed, c and ions
#               relaxed (ISIF 3, LATTICE_CONSTRAINTS F F T); compare F with strain/eta-2 (Amam start).
#   film_amam : Amam start (eta-2 CONTCAR fractions) in the film cell of Wang et al. PRX 16, 031008 (a = b = 3.756 A
#               pseudocubic, c = 20.72 A), ions only (ISIF 2).
#   film_fmmm : tilt-free start in the same film cell, ions only.
# Same PBE settings as run_relaxation.sh strain. One GPU per case; restart cycles until the relaxation stops moving.
# usage: run_film_tilt.sh CASE GPU
set -uo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
source "$ROOT/env_gpu.sh"
export CUDA_VISIBLE_DEVICES=$2
C=$1; d="$ROOT/tilt_check/$C"; mkdir -p "$d"; cd "$d" || exit 1
[ -f DONE ] && exit 0
E2="$ROOT/strain/eta-2/CONTCAR"; FM="$ROOT/poscar_fmmm_start"
film_lat() { printf "%s\n1.0\n  5.311740 0 0\n  0 5.311740 0\n  0 0 20.720000\n" "$1"; }
if [ ! -f POSCAR ]; then
  case $C in
    s2_fmmm)   { echo "tilt-free start in eta-2 cell"; echo "1.0"; sed -n 3,5p "$E2"; sed -n '6,$p' "$FM"; } > POSCAR ;;
    film_amam) { film_lat "Amam start in film cell"; sed -n '6,$p' "$E2"; } > POSCAR ;;
    film_fmmm) { film_lat "tilt-free start in film cell"; sed -n '6,$p' "$FM"; } > POSCAR ;;
  esac
fi
cp -f "$ROOT/POTCAR" POTCAR
printf "Gamma\n0\nGamma\n8 8 2\n0 0 0\n" > KPOINTS
ISIF=2; LC=""
[ "$C" = s2_fmmm ] && { ISIF=3; LC="LATTICE_CONSTRAINTS = .FALSE. .FALSE. .TRUE."; }
cat > INCAR <<EOI
SYSTEM = La3Ni2O7 tilt check $C
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
IBRION = ${IBRION:-2}
ISIF = $ISIF
NSW = 200
PSTRESS = 0.0000
$LC
EOI
for cyc in 1 2 3 4 5 6; do
  nice -n 10 "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$VASP_GPU" > vasp_c$cyc.log 2>&1 < /dev/null; echo $? > EXIT_c$cyc.txt
  cp -f OUTCAR OUTCAR_c$cyc; cp -f OSZICAR OSZICAR_c$cyc; cp -f CONTCAR POSCAR
  n=$(grep -c "F=" OSZICAR)
  if grep -q "reached required accuracy" OUTCAR && { [ "$ISIF" = 2 ] || [ "$n" -le 2 ]; }; then touch DONE; break; fi
done
echo "tilt_check $C (GPU $2) finished $(date)" >> "$ROOT/STAGE0.log"
