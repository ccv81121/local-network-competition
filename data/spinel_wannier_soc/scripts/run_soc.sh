#!/usr/bin/env bash
# SOC screen at pressure P: PBE seed on GPU SEEDGPU writing its CHGCAR with LMAXMIX = 4 (with LMAXMIX = 2 the density
# lacks the l = 4 one-centre terms of Sc d, which shifts the Gamma conduction valley by about 0.6 meV), then the
# fixed-density PBE+SOC step on the same GPU (MODE gpu, vasp_ncl_gpu, VASP 6.5.1) or on 24 CPU cores (MODE cpu, CPUSET).
# usage: setsid nohup bash scripts/run_soc.sh P SEEDGPU gpu > logs/soc3_P.log 2>&1 < /dev/null &
#        setsid nohup bash scripts/run_soc.sh P SEEDGPU cpu 0-23 > logs/soc3_P.log 2>&1 < /dev/null &
R=$WORK/PRL_TIGHT_UNIFY_20261102; P=$1; G=$2; MODE=$3; CPUS=$4
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $R/logs/RUNLOG.txt; }
d=$R/soc/P$P
( source $WORK/PRL_TIGHT_SPINEL_20261015/env_gpu.sh; export CUDA_VISIBLE_DEVICES=$G; cd $d/seed
  log "soc3 P$P seed start gpu$G (LMAXMIX 4)"
  "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none $VASP_GPU > vasp.out 2>&1 < /dev/null; echo $? > exit.txt
  grep -q "aborting loop because EDIFF is reached" OUTCAR && touch DONE; log "soc3 P$P seed exit $(cat exit.txt)" )
[ -f $d/seed/DONE ] || { log "soc3 P$P seed FAILED"; exit 1; }
cd $d/soc; cp ../seed/CHGCAR .
if [ "$MODE" = gpu ]; then
  grep -v -E "^(KPAR|NCORE)" INCAR.cpu > INCAR
  ( source $WORK/PRL_TIGHT_SPINEL_20261015/env_gpu.sh; export CUDA_VISIBLE_DEVICES=$G; ulimit -s unlimited
    log "soc3 P$P ncl start gpu$G"
    "$MPI_ROOT/bin/mpirun" -np 1 --bind-to none /data/vasp/vasp.6.5.1.bak/bin/vasp_ncl_gpu > vasp.out 2>&1 < /dev/null; echo $? > exit.txt )
else
  cp INCAR.cpu INCAR
  ( source $WORK/PRL_TIGHT_SPINEL_20261015/env_cpu.sh; ulimit -s unlimited; export OMP_STACKSIZE=512m
    log "soc3 P$P ncl start cpu $CPUS 24 ranks"
    taskset -c $CPUS $MPIRUN -np 24 $VASP_NCL > vasp.out 2>&1 < /dev/null; echo $? > exit.txt )
fi
grep -q "General timing" OUTCAR && touch DONE
log "soc3 P$P ncl exit $(cat exit.txt) $([ -f DONE ] && echo DONE || echo FAILED)"
rm -f CHGCAR WAVECAR CHG
