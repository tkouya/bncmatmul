#!/bin/bash
# Run the EFT-vs-MPFR(real, GPU) comparison for AXPY / GEMV / GEMM.
#  - EFT AXPY (double,float,dd,td,qd,ds,ts,qs):  cuda_axpy_bench      -> mpvs_axpy_eft.csv
#  - EFT GEMV/GEMM (dd,td,qd,ds,ts,qs):          gpu_omp_bench        -> mpvs_dense_eft.csv
#  - native double/float GEMV/GEMM + MPFR-CUDA (24..212 bit) all ops: mpfr_gpu_bench_* -> mpvs_mpfr.csv
# Sizes are matched across EFT and MPFR:  axpy N=4096,16384,65536 ; matvec N=128,256,512 ; matmul N=64,128,256
cd "$(dirname "$0")/../.."
. bench/cuda/env.sh
OUT=bench/cuda
REPS="${REPS:-5}"

echo "==== EFT AXPY (8 precisions) ===="
# --cap32 0 : skip the 32-thread launch config (pathologically slow at large N);
#             we only want the full-occupancy "GPU max threads" numbers.
./bench/cuda/cuda_axpy_bench --sizes 4096,16384,65536 --cap32 0 --reps "$REPS" --out $OUT/mpvs_axpy_eft.csv

echo "==== EFT GEMV/GEMM (6 multi-component) ===="
./bench/cuda/gpu_omp_bench --ops matvec,matmul --mv-sizes 128,256,512 --mm-sizes 64,128,256 \
    --reps "$REPS" --verify-cap 256 --maxcpu-mm 256 --out $OUT/mpvs_dense_eft.csv

echo "==== MPFR-CUDA (real) 8 bit-widths + native double/float ===="
first=1
for P in 24 48 53 72 96 106 159 212; do
  bin=$OUT/mpfr_gpu_bench_$P
  [ -x "$bin" ] || { echo "  (missing $bin)"; continue; }
  if [ $first -eq 1 ]; then
    echo "  PREC=$P (+ native double/float)"
    "$bin" --native --reps "$REPS" --out $OUT/mpvs_mpfr.csv
    first=0
  else
    echo "  PREC=$P"
    "$bin" --reps "$REPS" --append --out $OUT/mpvs_mpfr.csv
  fi
done

echo "==== done ===="
wc -l $OUT/mpvs_axpy_eft.csv $OUT/mpvs_dense_eft.csv $OUT/mpvs_mpfr.csv
