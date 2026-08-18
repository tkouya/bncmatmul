#!/bin/bash
# Build the REAL MPFR-CUDA benchmark for a set of precisions:
#   bench/cuda/mpfr_gpu_bench_<PREC>
# Default precisions match the 8 EFT bit-widths: 24 48 53 72 96 106 159 212.
# Config/paths from env.sh (override PREFIX/CUDA_HOME/ARCH).
set -e
cd "$(dirname "$0")/../.."
. bench/cuda/env.sh
mkdir -p cuda_obj
PRECS="$@"; [ -z "$PRECS" ] && PRECS="24 48 53 72 96 106 159 212"
for P in $PRECS; do
  echo ">> mpfr_gpu_bench PREC=$P"
  nvcc -O3 -arch=$ARCH -std=c++17 -fmad=false -DPREC=$P \
    -I$PREFIX/include/mpc_cuda -I$PREFIX/include -I$CUDA_HOME/include \
    -diag-suppress 20011 -diag-suppress 177 -diag-suppress 550 \
    -rdc=true -dc bench/cuda/mpfr_gpu_bench.cu -o cuda_obj/mpfr_gpu_bench_$P.o
  nvcc -arch=$ARCH -rdc=true cuda_obj/mpfr_gpu_bench_$P.o "$MPCLIB" \
    -L$CUDA_HOME/lib64 -lcudart -lcudadevrt \
    -o bench/cuda/mpfr_gpu_bench_$P
done
echo ">> built: $PRECS"
