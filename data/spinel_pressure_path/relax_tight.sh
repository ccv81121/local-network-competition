#!/usr/bin/env bash
# usage: relax_tight.sh GPU P POSCAR_START
T=$WORK/PRL_TIGHT_SPINEL_20261015; source $T/env_gpu.sh; export CUDA_VISIBLE_DEVICES=$1; P=$2
d=$T/Spinel/P$P/relax; mkdir -p $d; cd $d; cp -f $3 POSCAR; cp -f $T/Spinel/POTCAR .
printf "Gamma\n0\nGamma\n3 3 3\n0 0 0\n" > KPOINTS
KB=$(awk -v p="$P" "BEGIN{printf \"%.4f\", p*10}"); [ -n "$KB" ] || exit 9
cat > INCAR <<EOI
SYSTEM = spinel tight relax P=$P GPa
ENCUT = 520
PREC = Accurate
EDIFF = 1E-8
NELM = 180
NELMIN = 5
ALGO = Normal
ISMEAR = -5
SIGMA = 0.05
ISPIN = 1
LREAL = .FALSE.
LASPH = .TRUE.
ADDGRID = .TRUE.
LMAXMIX = 4
NBANDS = 172
IBRION = 1
POTIM = 0.2
ISIF = 3
NSW = 150
EDIFFG = -1E-3
PSTRESS = $KB
LWAVE = .FALSE.
LCHARG = .FALSE.
EOI
for cyc in 1 2 3 4 5 6; do
  nice -n 10 "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "$VASP_GPU" > vasp_c$cyc.log 2>&1 < /dev/null; echo $? > EXIT_c$cyc.txt
  cp -f OUTCAR OUTCAR_c$cyc; cp -f OSZICAR OSZICAR_c$cyc
  v0=$(grep -m1 "volume of cell" OUTCAR | awk "{print \$5}"); v1=$(grep "volume of cell" OUTCAR | tail -1 | awk "{print \$5}")
  awk -v a=$v0 -v b=$v1 "BEGIN{exit !((b-a)/a>0.05||(a-b)/a>0.05)}" && { echo "volume jump $v0 -> $v1" > ABORT.txt; exit 8; }
  cp -f CONTCAR POSCAR
  n=$(grep -c "F=" OSZICAR); if grep -q "reached required accuracy" OUTCAR && [ "$n" -le 2 ]; then touch DONE; break; fi
done
