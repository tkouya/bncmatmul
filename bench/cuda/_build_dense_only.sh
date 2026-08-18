#!/bin/bash
# Build GPU(gdtq) vs CPU-OMP benchmarks:
#   bench/cuda/gpu_omp_bench        -- dense matvec+matmul (dd/td/qd/ds/ts/qs)
#   bench/cuda/gpu_spmv_<type>      -- SpMV, one binary per EFT type
# Config/paths come from env.sh (override PREFIX/CUDA_HOME/ARCH/OMPLIB/SERLIB).
set -e
cd "$(dirname "$0")/../.."                 # repo root
. bench/cuda/env.sh
mkdir -p cuda_obj

compile_k(){ # $1 = kernel source stem in src/
  local k=$1
  if [ src/$k.cu -nt cuda_obj/$k.o ] || [ ! -f cuda_obj/$k.o ]; then
    echo "   nvcc -dc $k.cu"
    $NVCC $INC $DEFS $SUP -Xcompiler -fPIC -rdc=true -dc src/$k.cu -o cuda_obj/$k.o
  fi
}

echo ">> kernels for dense (gdd/gds hold shared arithmetic; gqd/gqs host wrappers separate) ..."
for k in gddlinear gtdlinear gqdlinear gdslinear gtslinear gqslinear; do compile_k $k; done
DOBJ="cuda_obj/gddlinear.o cuda_obj/gtdlinear.o cuda_obj/gqdlinear.o cuda_obj/gdslinear.o cuda_obj/gtslinear.o cuda_obj/gqslinear.o"

echo ">> dense driver ..."
$NVCC $INC $DEFS $SUP -Xcompiler -fopenmp -rdc=true -dc bench/cuda/gpu_omp_bench.cu -o cuda_obj/gpu_omp_bench.o
$NVCC -arch=$ARCH -dlink cuda_obj/gpu_omp_bench.o $DOBJ -o cuda_obj/_gpu_omp_bench_dl.o
$NVCC -arch=$ARCH cuda_obj/gpu_omp_bench.o $DOBJ cuda_obj/_gpu_omp_bench_dl.o $CPULK $SYSLK -o bench/cuda/gpu_omp_bench
echo ">> built bench/cuda/gpu_omp_bench"

