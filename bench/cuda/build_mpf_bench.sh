#!/bin/sh
#---------------------------------------------------------------------------
# build_mpf_bench.sh -- build the MPC_CUDA (mpc_cuda / cu_freal<PREC>) GPU-vs-CPU
#                       AXPY / GEMV / GEMM benchmark, one binary per precision.
#
#   GPU : src/mpflinear_cu.cu  (register-resident cu_freal<PREC>, mpc_cuda)
#   CPU : GMP mpf_t at the same precision, OpenMP (_bncomp_mul_mpf*)
#
# PREC is a compile-time constant, so one binary is produced per precision:
#   bench/cuda/cuda_mpf_mr_<PREC>
#
# Target: NVIDIA GB10, sm_121, CUDA 13.  Run from the repository root:
#   sh bench/cuda/build_mpf_bench.sh
#---------------------------------------------------------------------------
set -e
cd "$(dirname "$0")/../.."          # repo root

ARCH=${ARCH:-sm_121}
PRECS=${PRECS:-"128 256 512 1024"}
NVCC="nvcc -O3 -arch=$ARCH -std=c++17 -fmad=false -D__NV_NO_VECTOR_DEPRECATION_DIAG"
INC="-Iinclude -I/usr/local/include -I/usr/local/cuda/include"
DEFS="-DUSE_GMP -DUSE_MPFR"
SUP="-diag-suppress 20011 -diag-suppress 177 -diag-suppress 550 -diag-suppress 20208 -diag-suppress 20044 -Wno-deprecated-declarations"
LK="-L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lmpc -lmpfr -lgmp -lpthread -lqd -lstdc++ -lm"
CPU_TAG="${CPU_TAG:-_sve2}"
OMPLK="-L. -lbncmatmul-0.24-omp${CPU_TAG} -lbncmatmul-0.24${CPU_TAG} -Xcompiler -fopenmp -lgomp"

mkdir -p cuda_obj bench/cuda/out
OK=0; FAIL=0; FL=""

for P in $PRECS; do
    out="bench/cuda/cuda_mpf_mr_$P"
    if $NVCC $INC $DEFS -DPREC=$P $SUP -dc src/mpflinear_cu.cu \
             -o cuda_obj/mpflinear_cu_$P.o 2>/tmp/mpfbe \
       && $NVCC $INC $DEFS -DPREC=$P $SUP -dc bench/cuda/cuda_mpf_mr.cu \
             -o cuda_obj/cuda_mpf_mr_$P.o 2>>/tmp/mpfbe \
       && $NVCC -arch=$ARCH -dlink cuda_obj/cuda_mpf_mr_$P.o cuda_obj/mpflinear_cu_$P.o \
             -o cuda_obj/_cuda_mpf_mr_${P}_dl.o 2>>/tmp/mpfbe \
       && $NVCC -arch=$ARCH cuda_obj/cuda_mpf_mr_$P.o cuda_obj/mpflinear_cu_$P.o \
             cuda_obj/_cuda_mpf_mr_${P}_dl.o $OMPLK $LK -o "$out" 2>>/tmp/mpfbe
    then
        OK=$((OK+1)); echo "  [ok]   $out  (PREC=$P bits)"
    else
        FAIL=$((FAIL+1)); FL="$FL $P"; echo "  [FAIL] $out  (PREC=$P bits)"
        grep -m4 -iE 'error|undefined' /tmp/mpfbe | sed 's/^/         /'
    fi
done

echo "================ MPC_CUDA drivers: $OK built, $FAIL failed ================"
[ -n "$FL" ] && { echo "failed PREC:$FL"; exit 1; }
exit 0
