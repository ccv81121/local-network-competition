#!/usr/bin/env bash
# Inputs for the Wannier split and the SOC screen on the tightly relaxed x = 0.75 spinel (structures of
# spinel_pressure_path). Source directories are only read.
# usage: bash prepare.sh   (run once from this directory)
set -euo pipefail
R=$WORK/PRL_TIGHT_UNIFY_20261102; T=$WORK/PRL_TIGHT_SPINEL_20261015
cd $R; mkdir -p inputs wannier/scf wannier/nscf wannier/mlwf soc logs
cp $T/work/Spinel/P4_P5/fourstate/state11/POSCAR inputs/poscar_state11
cp $T/work/Spinel/P4_P5/fourstate/state11/INCAR  inputs/incar_state11
cp $T/work/Spinel/P4_P5/fourstate/state11/KPOINTS inputs/kpoints_6x6x6
cp $T/work/Spinel/P4_P5/fourstate/state11/POTCAR inputs/POTCAR
for P in 4 5 6 7.5; do cp $T/Spinel/P$P/relax/CONTCAR inputs/POSCAR_P$P; done
printf 'Gamma mesh 4x4x4\n0\nGamma\n4 4 4\n0 0 0\n' > inputs/kpoints_4x4x4

# 1. SCF at state 11 with the four-state settings (6x6x6, Gaussian 0.02 eV), keeping the charge density
d=wannier/scf; cp inputs/poscar_state11 $d/POSCAR; cp inputs/kpoints_6x6x6 $d/KPOINTS; cp inputs/POTCAR $d/
sed -e 's/^SYSTEM.*/SYSTEM = tight x0.75 5GPa state11 SCF for Wannier (four-state settings)/' -e 's/^LCHARG.*/LCHARG = .TRUE./' \
    inputs/incar_state11 > $d/INCAR

# 2. Converged non-self-consistent run on the full 4x4x4 mesh at fixed SCF density, Wannier90 in library mode
d=wannier/nscf; cp inputs/poscar_state11 $d/POSCAR; cp inputs/kpoints_4x4x4 $d/KPOINTS; cp inputs/POTCAR $d/
cat > $d/INCAR <<EOF
SYSTEM = tight x0.75 5GPa state11 NSCF 4x4x4 for Wannier (edge12 projections)
ENCUT = 520
PREC = Accurate
EDIFF = 1E-8
NELM = 120
NELMIN = 5
ALGO = Normal
ISMEAR = 0
SIGMA = 0.02
IBRION = -1
NSW = 0
ISYM = -1
LREAL = .FALSE.
LASPH = .TRUE.
ADDGRID = .TRUE.
LORBIT = 11
NBANDS = 286
ISTART = 0
ICHARG = 11
LWAVE = .FALSE.
LCHARG = .FALSE.
LWANNIER90_RUN = .TRUE.
LWRITE_MMN_AMN = .TRUE.
NUM_WANN = 248
EOF

# 3. SOC screen at 4, 5, 6 and 7.5 GPa: PBE seed (GPU) then fixed-density PBE+SOC (CPU), settings of the earlier screen
for P in 4 5 6 7.5; do
  for s in seed soc; do mkdir -p soc/P$P/$s; cp inputs/POSCAR_P$P soc/P$P/$s/POSCAR; cp inputs/kpoints_4x4x4 soc/P$P/$s/KPOINTS; cp inputs/POTCAR soc/P$P/$s/; done
  cat > soc/P$P/seed/INCAR <<EOF
SYSTEM = tight x0.75 P${P}GPa PBE seed for SOC screen
ENCUT = 520
PREC = Accurate
EDIFF = 1E-6
NELM = 180
NELMIN = 4
IBRION = -1
NSW = 0
ISYM = 2
LREAL = .FALSE.
LASPH = .TRUE.
ADDGRID = .TRUE.
LORBIT = 11
LWAVE = .FALSE.
LCHARG = .TRUE.
ALGO = Normal
ISPIN = 1
ISMEAR = -5
SIGMA = 0.05
ISTART = 0
ICHARG = 2
EOF
  cat > soc/P$P/soc/INCAR <<EOF
SYSTEM = tight x0.75 P${P}GPa PBE+SOC fixed-density screen
ENCUT = 520
PREC = Accurate
EDIFF = 1E-5
NELM = 80
NELMIN = 4
IBRION = -1
NSW = 0
ISYM = 2
LREAL = .FALSE.
LASPH = .TRUE.
ADDGRID = .TRUE.
KPAR = 4
NCORE = 3
ALGO = Normal
ISMEAR = -5
SIGMA = 0.05
LSORBIT = .TRUE.
SAXIS = 0 0 1
MAGMOM = 168*0
GGA_COMPAT = .FALSE.
LMAXMIX = 4
ISTART = 0
ICHARG = 11
LWAVE = .FALSE.
LCHARG = .FALSE.
EOF
done

# manifest: every input with its source and sha256
{ echo "# MANIFEST $(date -u +%FT%TZ)"; echo "source_tight=$T"
  echo "source_wannier_template=$WORK/PRL_MECHANISM_TEST_20261014/Wannier_Competition/basis_test"
  sha256sum inputs/* scripts/* wannier/*.tmpl wannier/basis_map.json; } > manifest.txt
echo prepared
