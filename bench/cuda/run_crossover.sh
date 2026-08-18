#!/bin/sh
#---------------------------------------------------------------------------
# run_crossover.sh -- matmul GPU-vs-CPU(SVE2 OMP / serial) across a dimension
# range, with the CPU baseline ENABLED at every size, to locate the crossover
# dimension where the GPU overtakes the CPU.  Per-precision size ranges are
# tailored because GPU cost (heavy multi-double) and CPU cost (serial complex-
# single) blow up very differently with N.
#
# Output: bench/cuda/out/crossover.csv  (same schema as cuda_results.csv)
#---------------------------------------------------------------------------
cd "$(dirname "$0")/../.."
OUT=bench/cuda/out; mkdir -p $OUT
CSV=$OUT/crossover.csv
: "${OMP_NUM_THREADS:=$(nproc)}"; export OMP_NUM_THREADS
echo "CPU baseline threads (SVE2 OMP): $OMP_NUM_THREADS"
echo "op,prec,dim,nnz,cpu_time,gpu_time,cpu_kind,relerr" > $CSV

REPS_LIGHT=${REPS_LIGHT:-3}
REPS_HEAVY=${REPS_HEAVY:-2}
B=128; T=128

emit(){ "$@" 2>/dev/null | sed -n 's/^RESULT,//p' | grep '^matmul' >> $CSV; }

echo "## native real d/f (GPU + CPU fast) -> up to 2048"
emit bench/cuda/cuda_dense_nr  --reps $REPS_LIGHT --blocks $B --threads $T --sizes 128,256,512,1024,2048 --axpy-sizes 4096 --max-cpu 2048

echo "## native complex cd/cf (serial CPU) -> up to 2048"
emit bench/cuda/cuda_dense_ncx --reps $REPS_LIGHT --blocks $B --threads $T --sizes 128,256,512,1024,2048 --max-cpu 2048

echo "## multi-component real dd/td/qd/ds/ts/qs (heavy GPU at large N) -> up to 1024"
emit bench/cuda/cuda_matmul_mr --reps $REPS_HEAVY --blocks $B --threads $T --sizes 128,256,512,1024 --max-cpu 1024

echo "## complex multi-component cdd/ctd/cqd (SVE2 OMP) -> up to 1024"
for p in cdd ctd cqd; do
    emit bench/cuda/cuda_dense_$p --reps $REPS_HEAVY --blocks $B --threads $T --sizes 128,256,512,1024 --max-cpu 1024
done

echo "## complex multi-component cds/cts/cqs (serial CPU; cqs/cts serial blows up) "
emit bench/cuda/cuda_dense_cds --reps $REPS_HEAVY --blocks $B --threads $T --sizes 128,256,512,1024 --max-cpu 1024
emit bench/cuda/cuda_dense_cts --reps $REPS_HEAVY --blocks $B --threads $T --sizes 128,256,512 --max-cpu 512
emit bench/cuda/cuda_dense_cqs --reps $REPS_HEAVY --blocks $B --threads $T --sizes 128,256,512 --max-cpu 512

echo "Wrote $CSV ($(($(wc -l < $CSV) - 1)) matmul rows)"
