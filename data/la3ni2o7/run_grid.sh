#!/usr/bin/env bash
# Drive-plane grid: R and t_perp of La3Ni2O7 as functions of the in-plane lattice a_p = sqrt(ab/2) and c.
#   a, b scaled together from the 0 GPa PBE cell (b/a = 1.0182 kept), a_p in {3.8522, 3.804, 3.756} A (index A0..A2),
#   c in {19.926, 20.376, 20.826} A (index C0..C2, 0 GPa c +- 0.45 A). A0C1 is the 0 GPa state itself (not rerun).
#   grid/CASE/relax : Amam start (pressure/P0/CONTCAR fractions), ions only (ISIF 2), PBE settings of run_film_tilt.sh;
#                     first cycle CG (IBRION 2, NSW 80), later cycles quasi-Newton (IBRION 1) to avoid CG creep on the
#                     soft tilt mode. EDIFF 1E-6, EDIFFG -5E-3 for A1C0, A2C2, A2C0, A1C2 (the other points: 1E-8, -2E-3).
#   grid/CASE/pbe, u4, wan : readouts with the settings of run_tilt_readout.sh.
# Each VASP start waits for MemAvailable >= 60 GB (the node is shared).
# MODE: a GPU index (one rank, vasp_gpu) or "cpu" with NP ranks (Intel MPI, KPAR = 8 and NCORE = 1 appended to every
# INCAR, as in wannier_job_cpu.sh; same VASP 6.5.1, parallelization only).
# usage: run_grid.sh CASE MODE [NP]   (CASE = A<i>C<j>[suffix]; MODE = 0..3 or cpu)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
MODE=$2; NP=${3:-48}
if [ "$MODE" = cpu ]; then
  source "$ROOT/env_cpu152.sh"; ulimit -s unlimited
  BIN=$VASP_CPU; BINW=$VASP_CPU
else
  source "$ROOT/env_gpu.sh"; export CUDA_VISIBLE_DEVICES=$MODE
  BIN=$VASP_GPU; BINW=/data/vasp/vasp.6.5.1_gpu_w90/bin/vasp_std
fi
vasp() {  # binary, log; INCAR in the current directory
  if [ "$MODE" = cpu ]; then
    sed -i '/^KPAR = /d; /^NCORE = /d' INCAR; printf 'KPAR = 8\nNCORE = 1\n' >> INCAR
    nice -n 10 mpirun -np "$NP" -genv I_MPI_PIN 0 "$1" > "$2" 2>&1 < /dev/null
  else
    nice -n 10 "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$1" > "$2" 2>&1 < /dev/null
  fi
}
C=$1; G="$ROOT/grid/$C"; mkdir -p "$G/relax"; echo "$MODE $NP" > "$G/MODE"
i=${C:1:1}; j=${C:3:1}
S=$(echo "3.8522 3.804 3.756" | cut -d' ' -f$((i + 1)))
CC=$(echo "19.926 20.376 20.826" | cut -d' ' -f$((j + 1)))
P0="$ROOT/pressure/P0/CONTCAR"
memwait() { until [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge 60 ]; do sleep 60; done; }
run() {  # dir, binary
  cd "$1" || return 1
  [ -f DONE ] && return 0
  cp -f "$ROOT/POTCAR" POTCAR
  memwait
  vasp "$2" vasp.log; rc=$?; echo "$rc $MODE" > exit.txt
  rm -f wannier90.mmn wannier90.amn WAVECAR
  if [ $rc = 0 ] && grep -q "aborting loop because EDIFF is reached" OUTCAR && grep -q "General timing" OUTCAR; then touch DONE; else touch FAILED; fi
}
# ---- relaxation at fixed cell
d="$G/relax"; cd "$d" || exit 1
if [ ! -f DONE ]; then
  if [ ! -f POSCAR ]; then
    $HOME/.conda/envs/prl/bin/python - "$P0" "$S" "$CC" > POSCAR <<'EOP'
import sys, math
L = open(sys.argv[1]).read().splitlines()
ap, c = float(sys.argv[2]), float(sys.argv[3])
a0, b0 = float(L[2].split()[0]), float(L[3].split()[1])
s = ap / math.sqrt(a0 * b0 / 2)
print(f"La3Ni2O7 grid a_p {ap} c {c}, Amam start from 0 GPa fractions"); print("1.0")
print(f"  {a0 * s:.10f} 0 0"); print(f"  0 {b0 * s:.10f} 0"); print(f"  0 0 {c:.10f}")
print("\n".join(L[5:]))
EOP
  fi
  cp -f "$ROOT/POTCAR" POTCAR
  printf "Gamma\n0\nGamma\n8 8 2\n0 0 0\n" > KPOINTS
  for cyc in 1 2 3 4 5 6; do
    [ -f DONE ] && break
    rm -f STOPCAR
    [ $cyc = 1 ] && { IB=${IB1:-2}; NSW=80; } || { IB=1; NSW=200; }
    cat > INCAR <<EOI
SYSTEM = La3Ni2O7 grid $C
PREC = Accurate
ENCUT = 520
EDIFF = 1E-6
ISPIN = 1
ISMEAR = 1
SIGMA = 0.05
LREAL = .FALSE.
LASPH = .TRUE.
LWAVE = .FALSE.
LCHARG = .FALSE.
NELM = 200
EDIFFG = -5E-3
IBRION = $IB
ISIF = 2
NSW = $NSW
PSTRESS = 0.0000
EOI
    memwait
    vasp "$BIN" vasp_c$cyc.log; echo "$? $MODE" > EXIT_c$cyc.txt
    cp -f OUTCAR OUTCAR_c$cyc; cp -f OSZICAR OSZICAR_c$cyc; cp -f CONTCAR POSCAR
    grep -q "reached required accuracy" OUTCAR && { touch DONE; break; }
  done
  [ -f DONE ] || { touch FAILED; echo "grid $C relax not converged $(date)" >> "$ROOT/STAGE0.log"; exit 1; }
fi
# ---- readouts
INC="PREC = Accurate
ENCUT = 520
EDIFF = 1E-7
ISMEAR = -5
LREAL = .FALSE.
LASPH = .TRUE.
NELM = 200
IBRION = -1
NSW = 0
LORBIT = 11
NEDOS = 3001
LCHARG = .FALSE.
ALGO = Normal
LWAVE = .FALSE."
U4="LDAU = .TRUE.
LDAUTYPE = 2
LDAUL = -1 2 -1
LDAUU = 0 4 0
LDAUJ = 0 0 0
LMAXMIX = 4"
for tag in pbe u4; do
  d="$G/$tag"; mkdir -p "$d"
  cp -f "$G/relax/CONTCAR" "$d/POSCAR"
  printf "Gamma\n0\nGamma\n12 12 3\n0 0 0\n" > "$d/KPOINTS"
  { echo "SYSTEM = La3Ni2O7 grid readout $C $tag"; echo "$INC"; [ $tag = u4 ] && echo "$U4"; } > "$d/INCAR"
  run "$d" "$BIN"
done
ef=$(grep E-fermi "$G/pbe/OUTCAR" | tail -1 | awk '{print $3}')
d="$G/wan"; mkdir -p "$d"
if [ ! -f "$d/DONE" ]; then
  cp -f "$G/relax/CONTCAR" "$d/POSCAR"
  (cd "$ROOT/grid" && $HOME/.conda/envs/prl/bin/python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from ase.io import read
import wannier_prepare as p
d = Path('$d'); p.put(d, read(d / 'POSCAR'), $ef, True)")
  sed -i 's/^LWAVE = .TRUE./LWAVE = .FALSE./' "$d/INCAR"
  run "$d" "$BINW"
  [ -s "$d/wannier90_hr.dat" ] || { rm -f "$d/DONE"; touch "$d/FAILED"; }
fi
echo "$MODE $NP" > "$G/MODE"
echo "grid $C ($MODE) finished $(date)" >> "$ROOT/STAGE0.log"
