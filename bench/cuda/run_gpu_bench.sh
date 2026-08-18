#!/bin/sh
#---------------------------------------------------------------------------
# run_gpu_bench.sh -- AXPY / GEMV / GEMM on the GPU, for BOTH multiple-precision
#                     libraries, in a form that merges across machines.
#
#   GDTQ      : g{dd,td,qd,ds,ts,qs}linear (+ native d/f) GPU kernels
#               CPU baseline = libbncmatmul OpenMP (_bncomp_*)
#   MPC_CUDA  : mpc_cuda cu_freal<PREC> GPU kernels
#               CPU baseline = GMP mpf_t at the same precision + OpenMP
#
# This measures the tree AS IT IS, i.e. BEFORE the branch-free DW/TW/QW FMA is
# switched on (BNC_USE_NEW_FMA is undefined; the prebuilt libraries predate it).
#
# Outputs
#   bench/cuda/out/gdtq_results.csv        raw, GDTQ drivers
#   bench/cuda/out/mpf_results.csv         raw, MPC_CUDA drivers
#   bench/cuda/out/gpu_bench_merged.csv    merge-ready (platform metadata added)
#---------------------------------------------------------------------------
cd "$(dirname "$0")/../.."          # repo root
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

OUT=bench/cuda/out
mkdir -p $OUT

PLATFORM=${PLATFORM:-arm-gb10}
DIMS=${DIMS:-"128 256 512 1024"}
AXPY_DIMS=${AXPY_DIMS:-"4096 16384 65536 262144 1048576"}
REPS=${REPS:-3}
BLOCKS=${BLOCKS:-128}
THREADS=${THREADS:-128}
MAXCPU=${MAXCPU:-1024}
MAXCPU_MPF=${MAXCPU_MPF:-512}
PRECS=${PRECS:-"128 256 512 1024"}
NATIVE=${NATIVE:-1}

if [ -z "$OMP_NUM_THREADS" ]; then OMP_NUM_THREADS=$(nproc); fi
export OMP_NUM_THREADS
echo "CPU OpenMP baseline threads: $OMP_NUM_THREADS"

csv_dims() { echo "$1" | tr ' ' ','; }
HDR="op,prec,dim,nnz,cpu_time,gpu_time,cpu_kind,relerr"

GDTQ_CSV=$OUT/gdtq_results.csv
MPF_CSV=$OUT/mpf_results.csv
echo "$HDR" > $GDTQ_CSV
echo "$HDR" > $MPF_CSV

run_driver() { # csv binary extra-args...
    csv="$1"; bin="$2"; shift 2
    [ -x "$bin" ] || { echo "  (skip, not built: $bin)"; return; }
    echo ">>> $bin $*"
    "$bin" --reps $REPS --blocks $BLOCKS --threads $THREADS "$@" 2>/dev/null \
        | sed -n 's/^RESULT,//p' >> "$csv"
}

echo "######## GDTQ: AXPY / GEMV / GEMM (dd td qd ds ts qs) ########"
run_driver $GDTQ_CSV bench/cuda/cuda_axpy_mr   --sizes "$(csv_dims "$AXPY_DIMS")"
run_driver $GDTQ_CSV bench/cuda/cuda_matvec_mr --sizes "$(csv_dims "$DIMS")"
run_driver $GDTQ_CSV bench/cuda/cuda_matmul_mr --sizes "$(csv_dims "$DIMS")" --max-cpu $MAXCPU

if [ "$NATIVE" = "1" ]; then
    echo "######## GDTQ: native d / f reference points ########"
    run_driver $GDTQ_CSV bench/cuda/cuda_dense_nr \
        --sizes "$(csv_dims "$DIMS")" --axpy-sizes "$(csv_dims "$AXPY_DIMS")" \
        --max-cpu ${MAXCPU_NATIVE:-512}
fi

echo "######## MPC_CUDA: AXPY / GEMV / GEMM (mpf, $PRECS bits) ########"
for P in $PRECS; do
    run_driver $MPF_CSV bench/cuda/cuda_mpf_mr_$P \
        --sizes "$(csv_dims "$DIMS")" --axpy-sizes "$(csv_dims "$AXPY_DIMS")" \
        --max-cpu $MAXCPU_MPF
done

echo
echo "######## merge-ready table ########"
python3 bench/cuda/merge_prep.py annotate \
    --raw $GDTQ_CSV --lib gdtq \
    --raw $MPF_CSV  --lib mpc_cuda \
    --platform "$PLATFORM" --fma pre-FMA \
    --out $OUT/gpu_bench_merged.csv

echo
echo "raw   : $GDTQ_CSV, $MPF_CSV"
echo "merged: $OUT/gpu_bench_merged.csv"
