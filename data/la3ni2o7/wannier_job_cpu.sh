#!/usr/bin/env bash
# One stage-1c Wannier restart on CPUs in the current directory. Same inputs and DONE criteria as wannier_job.sh.
# KPAR = 8 and NCORE = 1 are appended to INCAR for this run only (the Wannier90/PEAD routines require NCORE = 1) and
# removed afterwards, so the INCAR stays as written by wannier_prepare.py.
T=$(cd "$(dirname "$0")" && pwd)
NP=${1:-48}
source "$T/../env_cpu152.sh"
ulimit -s unlimited            # Intel-Fortran VASP needs an unlimited stack for the SCF loop
cp -f "$T/../POTCAR" POTCAR
sed -i '/^KPAR = /d; /^NCORE = /d' INCAR
printf 'KPAR = 8\nNCORE = 1\n' >> INCAR
cp -f ../00/WAVECAR WAVECAR || { sed -i '/^KPAR = /d; /^NCORE = /d' INCAR; touch FAILED; exit 1; }
nice -n 10 mpirun -np "$NP" -genv I_MPI_PIN 0 "$VASP_CPU" > vasp.log 2>&1 < /dev/null
rc=$?; echo "$rc cpu" > exit.txt
sed -i '/^KPAR = /d; /^NCORE = /d' INCAR
rm -f WAVECAR wannier90.mmn wannier90.amn
ok=0; [ $rc = 0 ] && grep -q "aborting loop because EDIFF is reached" OUTCAR && grep -q "General timing" OUTCAR && ok=1
[ ! -s wannier90_hr.dat ] && ok=0
[ $ok = 1 ] && touch DONE || touch FAILED
