#!/usr/bin/env bash
# Orbital-resolved bands and CBM wavefunctions (tight structures, PBE, settings of the four-state run):
#  fb/P5 : orbital-projected bands along R-Gamma-Y at 5 GPa (state11), CHGCAR from parchg_state11/scf   [GPU 0]
#  wf/P5 : WAVECAR at Gamma and R for wavefunction isosurfaces (5 GPa)                                [GPU 0, after fb/P5]
#  fb/P4 : 4x4x4 SCF at 4 GPa (tight relaxed CONTCAR), then the same band path                        [GPU 1]
set -e
R=$WORK/PRL_TIGHT_SPINEL_20261015
D=$R/fatband_wf
S5=$R/work/Spinel/P4_P5/fourstate/state11
mkdir -p $D/fb/P5 $D/wf/P5 $D/fb/P4/scf $D/fb/P4/band

common() {
cat <<EOF
ENCUT = 520
PREC = Accurate
EDIFF = 1E-7
NELM = 200
ALGO = Normal
ISMEAR = 0
SIGMA = 0.02
IBRION = -1
NSW = 0
ISYM = 0
LREAL = .FALSE.
LASPH = .TRUE.
ADDGRID = .TRUE.
NBANDS = 187
EOF
}
line_kpts() {
printf "R-Gamma-Y\n20\nLine-mode\nReciprocal\n0.5 0.5 0.5 ! R\n0.0 0.0 0.0 ! G\n\n0.0 0.0 0.0 ! G\n0.0 0.5 0.0 ! Y\n"
}

# 5 GPa bands
cp $S5/POSCAR $S5/POTCAR $D/fb/P5/
cp $R/parchg_state11/scf/CHGCAR $D/fb/P5/
{ echo "SYSTEM = 5 GPa fat bands"; common; echo "ICHARG = 11"; echo "LORBIT = 11"; echo "LWAVE = .FALSE."; echo "LCHARG = .FALSE."; } > $D/fb/P5/INCAR
line_kpts > $D/fb/P5/KPOINTS

# 5 GPa wavefunctions at Gamma and R
cp $S5/POSCAR $S5/POTCAR $D/wf/P5/
cp $R/parchg_state11/scf/CHGCAR $D/wf/P5/
{ echo "SYSTEM = 5 GPa WAVECAR Gamma R"; common; echo "ICHARG = 11"; echo "LWAVE = .TRUE."; echo "LCHARG = .FALSE."; } > $D/wf/P5/INCAR
printf "Gamma and R\n2\nReciprocal\n0.0 0.0 0.0 1\n0.5 0.5 0.5 1\n" > $D/wf/P5/KPOINTS

# 4 GPa SCF + bands
cp $R/Spinel/P4/relax/CONTCAR $D/fb/P4/scf/POSCAR
cp $S5/POTCAR $D/fb/P4/scf/
{ echo "SYSTEM = 4 GPa SCF"; common; echo "LWAVE = .FALSE."; echo "LCHARG = .TRUE."; } > $D/fb/P4/scf/INCAR
printf "Gamma 4x4x4\n0\nGamma\n4 4 4\n0 0 0\n" > $D/fb/P4/scf/KPOINTS
cp $D/fb/P4/scf/POSCAR $D/fb/P4/scf/POTCAR $D/fb/P4/band/
{ echo "SYSTEM = 4 GPa fat bands"; common; echo "ICHARG = 11"; echo "LORBIT = 11"; echo "LWAVE = .FALSE."; echo "LCHARG = .FALSE."; } > $D/fb/P4/band/INCAR
line_kpts > $D/fb/P4/band/KPOINTS

cat > $D/run_gpu.sh <<EOF
#!/usr/bin/env bash
# usage: run_gpu.sh GPU DIR [DIR ...]  (each DIR run in order; stops the chain on failure)
source $R/env_gpu.sh
export CUDA_VISIBLE_DEVICES=\$1; shift
for d in "\$@"; do
  cd "\$d" || exit 1
  nice -n 10 "\$MPI_ROOT/bin/mpirun" -np 1 --bind-to none "\$VASP_GPU" > vasp.log 2>&1 < /dev/null
  grep -q "General timing" OUTCAR && touch DONE || { touch FAILED; exit 1; }
  [ "\$d" = "$D/fb/P4/scf" ] && cp CHGCAR $D/fb/P4/band/
  rm -f CHG vasprun.xml
done
EOF
chmod +x $D/run_gpu.sh
cd $D
nohup ./run_gpu.sh 0 $D/fb/P5 $D/wf/P5 > gpu0.log 2>&1 < /dev/null &
nohup ./run_gpu.sh 1 $D/fb/P4/scf $D/fb/P4/band > gpu1.log 2>&1 < /dev/null &
echo "started in $D"
