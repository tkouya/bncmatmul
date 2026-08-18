#!/bin/bash
# Build the complex mpc_cuda benchmark for a set of precisions:
#   bench/cuda/cmpf_bench_<PREC>   (default precisions 106 212 424 if none given)
# Config/paths from env.sh (override PREFIX/CUDA_HOME/ARCH).
set -e
cd "$(dirname "$0")/../.."
. bench/cuda/env.sh
mkdir -p cuda_obj
PRECS="$@"; [ -z "$PRECS" ] && PRECS="106 212 424"
for P in $PRECS; do
  echo ">> cmpf_bench PREC=$P"
  nvcc -O3 -arch=$ARCH -std=c++17 -fmad=false -DPREC=$P \
    -I$PREFIX/include/mpc_cuda -I$PREFIX/include -I$CUDA_HOME/include \
    -diag-suppress 20011 -diag-suppress 177 -diag-suppress 550 -Xcompiler -fopenmp \
    -rdc=true -dc bench/cuda/cmpf_bench.cu -o cuda_obj/cmpf_bench_$P.o
  nvcc -arch=$ARCH -rdc=true cuda_obj/cmpf_bench_$P.o "$MPCLIB" \
    -Xcompiler -fopenmp -lgomp -L$CUDA_HOME/lib64 -lcudart -lcudadevrt \
    -o bench/cuda/cmpf_bench_$P
done
echo ">> built: $PRECS"
