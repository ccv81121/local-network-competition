#!/usr/bin/env bash
# Readouts of the tilt-check structures: after tilt_check/CASE/DONE, on the same GPU,
#   tilt_read/CASE/pbe, u4 : stage-1b readout settings (12x12x3, ISMEAR -5, LORBIT 11; U = 4 eV on Ni d), from scratch
#   tilt_read/CASE/wan     : stage-1c Wannier settings (6x6x2 full mesh, ISYM -1, NBANDS 288, 124 projected WFs),
#                            window around the PBE readout E_F; win and basis_map from wannier_prepare.put.
# usage: run_tilt_readout.sh CASE GPU
set -uo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
source "$ROOT/env_gpu.sh"
export CUDA_VISIBLE_DEVICES=$2
C=$1; T="$ROOT/tilt_read/$C"; mkdir -p "$T"
until [ -f "$ROOT/tilt_check/$C/DONE" ]; do sleep 120; done
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
run() {  # dir, binary
  cd "$1" || return 1
  [ -f DONE ] && return 0
  cp -f "$ROOT/POTCAR" POTCAR
  nice -n 10 "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$2" > vasp.log 2>&1 < /dev/null; rc=$?; echo $rc > exit.txt
  rm -f wannier90.mmn wannier90.amn WAVECAR
  if [ $rc = 0 ] && grep -q "aborting loop because EDIFF is reached" OUTCAR && grep -q "General timing" OUTCAR; then touch DONE; else touch FAILED; fi
}
for tag in pbe u4; do
  d="$T/$tag"; mkdir -p "$d"
  cp -f "$ROOT/tilt_check/$C/CONTCAR" "$d/POSCAR"
  printf "Gamma\n0\nGamma\n12 12 3\n0 0 0\n" > "$d/KPOINTS"
  { echo "SYSTEM = La3Ni2O7 tilt readout $C $tag"; echo "$INC"; [ $tag = u4 ] && echo "$U4"; } > "$d/INCAR"
  run "$d" "$VASP_GPU"
done
ef=$(grep E-fermi "$T/pbe/OUTCAR" | tail -1 | awk '{print $3}')
d="$T/wan"; mkdir -p "$d"
cp -f "$ROOT/tilt_check/$C/CONTCAR" "$d/POSCAR"
(cd "$ROOT/tilt_read" && $HOME/.conda/envs/prl/bin/python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from ase.io import read
import wannier_prepare as p
d = Path('$d'); p.put(d, read(d / 'POSCAR'), $ef, True)")
sed -i 's/^LWAVE = .TRUE./LWAVE = .FALSE./' "$d/INCAR"
run "$d" /data/vasp/vasp.6.5.1_gpu_w90/bin/vasp_std
[ -s "$d/wannier90_hr.dat" ] || { rm -f "$d/DONE"; touch "$d/FAILED"; }
echo "tilt_read $C (GPU $2) finished $(date)" >> "$ROOT/STAGE0.log"
