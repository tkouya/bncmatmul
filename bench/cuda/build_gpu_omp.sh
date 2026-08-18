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

# ---- SpMV: one binary per type (each links a single self-contained sparse obj) ----
build_spmv(){ # $1=TAG(DD..) $2=sparse-stem
  local TAG=$1 SPR=$2 low
  low=$(echo "$TAG" | tr 'A-Z' 'a-z')
  compile_k "$SPR"
  echo ">> spmv driver ($low) ..."
  $NVCC $INC $DEFS $SUP -DBENCH_$TAG -Xcompiler -fopenmp -rdc=true -dc bench/cuda/gpu_spmv_bench.cu -o cuda_obj/gpu_spmv_$low.o
  $NVCC -arch=$ARCH -dlink cuda_obj/gpu_spmv_$low.o cuda_obj/$SPR.o -o cuda_obj/_gpu_spmv_${low}_dl.o
  $NVCC -arch=$ARCH cuda_obj/gpu_spmv_$low.o cuda_obj/$SPR.o cuda_obj/_gpu_spmv_${low}_dl.o $CPULK $SYSLK -o bench/cuda/gpu_spmv_$low
  echo ">> built bench/cuda/gpu_spmv_$low"
}
build_spmv DD gddsparse
build_spmv TD gtdsparse
build_spmv QD gqdsparse
build_spmv DS gdssparse
build_spmv TS gtssparse
build_spmv QS gqssparse
echo ">> ALL built."
