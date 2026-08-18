#!/bin/sh
#---------------------------------------------------------------------------
# run_cuda.sh -- run the CUDA GPU-vs-CPU(OpenMP) benchmarks, collect CSV.
#
# CPU baseline runs with OpenMP at the machine's max thread count (or the
# value of OMP_NUM_THREADS if already set in the environment).
#
# Output: bench/cuda/out/cuda_results.csv
#   op,prec,dim,nnz,cpu_time,gpu_time,cpu_kind,relerr
#---------------------------------------------------------------------------
cd "$(dirname "$0")/../.."          # repo root
OUT=bench/cuda/out
mkdir -p $OUT
CSV=$OUT/cuda_results.csv

# Max threads for the CPU OpenMP baseline.
if [ -z "$OMP_NUM_THREADS" ]; then
    OMP_NUM_THREADS=$(nproc)
fi
export OMP_NUM_THREADS
echo "CPU OpenMP baseline threads: $OMP_NUM_THREADS"

DIMS=${DIMS:-"128 256 512 1024 2048"}
AXPY_DIMS=${AXPY_DIMS:-"4096 16384 65536 262144 1048576"}
REPS=${REPS:-3}
BLOCKS=${BLOCKS:-128}
THREADS=${THREADS:-128}
# CPU multiprecision matmul gets very slow; cap the CPU baseline size (GPU still runs all).
MAXCPU_MATMUL=${MAXCPU_MATMUL:-1024}

csv_dims() { echo "$1" | tr ' ' ','; }

echo "op,prec,dim,nnz,cpu_time,gpu_time,cpu_kind,relerr" > $CSV

run_driver() { # binary  extra-args...
    bin="$1"; shift
    [ -x "$bin" ] || { echo "  (skip, not built: $bin)"; return; }
    echo ">>> $bin $*"
    "$bin" --reps $REPS --blocks $BLOCKS --threads $THREADS "$@" 2>/dev/null \
        | sed -n 's/^RESULT,//p' >> $CSV
}

# native double/float OMP matmul is AVX-gated -> scalar fallback on ARM (slow); cap harder.
MAXCPU_NATIVE=${MAXCPU_NATIVE:-512}

echo "######## DENSE: native real (d/f) ########"
run_driver bench/cuda/cuda_dense_nr  --sizes "$(csv_dims "$DIMS")" --axpy-sizes "$(csv_dims "$AXPY_DIMS")" --max-cpu $MAXCPU_NATIVE

echo "######## DENSE: native complex (cd/cf) ########"
run_driver bench/cuda/cuda_dense_ncx --sizes "$(csv_dims "$DIMS")" --max-cpu $MAXCPU_NATIVE

echo "######## DENSE: matmul / matvec / axpy (multi-component real) ########"
run_driver bench/cuda/cuda_matmul_mr --sizes "$(csv_dims "$DIMS")" --max-cpu $MAXCPU_MATMUL
run_driver bench/cuda/cuda_matvec_mr --sizes "$(csv_dims "$DIMS")"
run_driver bench/cuda/cuda_axpy_mr   --sizes "$(csv_dims "$AXPY_DIMS")"

echo "######## DENSE: complex multi-component (cdd/ctd/cqd OMP; cds/cts/cqs serial) ########"
for p in cdd ctd cqd cds cts cqs; do
    run_driver bench/cuda/cuda_dense_$p --sizes "$(csv_dims "$DIMS")" --max-cpu $MAXCPU_MATMUL
done

echo "######## SpMV (sparse matrix-vector) ########"
SPMV_DIMS=${SPMV_DIMS:-"16384 65536 262144 1048576"}
run_driver bench/cuda/cuda_spmv_d --sizes "$(csv_dims "$SPMV_DIMS")" --reps ${SPMV_REPS:-20}

echo "######## (remaining SpMV precisions append here as added) ########"

echo "Wrote $CSV ($(($(wc -l < $CSV) - 1)) rows)"
