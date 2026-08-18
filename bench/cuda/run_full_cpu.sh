#!/bin/sh
#---------------------------------------------------------------------------
# run_full_cpu.sh -- dense GPU-vs-CPU benchmark with the CPU baseline ENABLED
# at every tested dimension (not just small N).  CPU = OpenMP+SVE2 (20 threads)
# where available, serial otherwise.  matmul/matvec swept to N=1024, axpy to 1M.
#
# Exception: the serial complex-single matmul (cts/cqs) is O(N^3) single-thread
# and explodes (cqs N=1024 ~190 s/rep), so its CPU baseline is capped at N=512.
# Complex axpy has no CPU cmul routine in the library -> GPU-only.
#
# Output: bench/cuda/out/cuda_results.csv
#---------------------------------------------------------------------------
cd "$(dirname "$0")/../.."
OUT=bench/cuda/out; mkdir -p $OUT
CSV=$OUT/cuda_results.csv
: "${OMP_NUM_THREADS:=$(nproc)}"; export OMP_NUM_THREADS
echo "CPU baseline threads (SVE2 OMP): $OMP_NUM_THREADS"

REPS=${REPS:-2}; B=128; T=128
DIMS=${DIMS:-128,256,512,1024}
AXPY=${AXPY:-4096,16384,65536,262144,1048576}

echo "op,prec,dim,nnz,cpu_time,gpu_time,cpu_kind,relerr" > $CSV
emit(){ "$@" 2>/dev/null | sed -n 's/^RESULT,//p' >> $CSV; }

echo "## native real d/f (matmul/matvec/axpy, CPU all sizes)"
emit bench/cuda/cuda_dense_nr  --reps $REPS --blocks $B --threads $T --sizes $DIMS --axpy-sizes $AXPY --max-cpu 1024
echo "## native complex cd/cf (matmul/matvec, CPU all sizes)"
emit bench/cuda/cuda_dense_ncx --reps $REPS --blocks $B --threads $T --sizes $DIMS --max-cpu 1024
echo "## multi-component real dd/td/qd/ds/ts/qs"
emit bench/cuda/cuda_matmul_mr --reps $REPS --blocks $B --threads $T --sizes $DIMS --max-cpu 1024
emit bench/cuda/cuda_matvec_mr --reps $REPS --blocks $B --threads $T --sizes $DIMS
emit bench/cuda/cuda_axpy_mr   --reps $REPS --blocks $B --threads $T --sizes $AXPY
echo "## complex multi-component cdd/ctd/cqd (SVE2 OMP, CPU all sizes)"
for p in cdd ctd cqd; do
    emit bench/cuda/cuda_dense_$p --reps $REPS --blocks $B --threads $T --sizes $DIMS --max-cpu 1024
done
echo "## complex multi-component cds/cts/cqs (serial CPU; cts/cqs capped at 512)"
emit bench/cuda/cuda_dense_cds --reps $REPS --blocks $B --threads $T --sizes $DIMS --max-cpu 1024
emit bench/cuda/cuda_dense_cts --reps $REPS --blocks $B --threads $T --sizes $DIMS --max-cpu 512
emit bench/cuda/cuda_dense_cqs --reps $REPS --blocks $B --threads $T --sizes $DIMS --max-cpu 512
echo "## complex axpy (GPU-only; no CPU cmul routine in library)"
for p in cd cf cdd ctd cqd cds cts cqs; do
    emit bench/cuda/cuda_caxpy_$p --reps $REPS --blocks $B --threads $T --sizes $AXPY
done

echo "Wrote $CSV ($(($(wc -l < $CSV) - 1)) rows)"
