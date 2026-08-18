#!/bin/sh
#---------------------------------------------------------------------------
# build_cuda_bench.sh -- build the CUDA GPU-vs-CPU(OpenMP) benchmark drivers.
#
#   GPU side : libbncmatmul-0.24_cuda.a kernels (built by build_cuda_full.sh;
#              per-precision objects live in cuda_obj/).
#   CPU side : libbncmatmul-0.24-omp.a  (_bncomp_* OpenMP routines, 20 threads)
#              + libbncmatmul-0.24.a    (serial primitives the OMP code calls).
#
# Each driver prints one CSV line per measurement:
#   RESULT,op,prec,dim,nnz,cpu_time,gpu_time,cpu_kind,relerr
#
# Target: NVIDIA GB10, sm_121, CUDA 13.  Run from the repository root:
#   sh bench/cuda/build_cuda_bench.sh
#---------------------------------------------------------------------------
set -e
cd "$(dirname "$0")/../.."          # repo root

ARCH=sm_121
NVCC="nvcc -O3 -arch=$ARCH -std=c++17 -fmad=false -D__NV_NO_VECTOR_DEPRECATION_DIAG"
INC="-Iinclude -I/usr/local/include/gdtq -I/usr/local/cuda/include"
NAT_INC="-Iinclude -I/usr/local/cuda/include"
MR_DEFS="-DUSE_GMP -DUSE_MPFR"
SUP="-diag-suppress 20011 -diag-suppress 177 -diag-suppress 550 -diag-suppress 20208 -diag-suppress 20044 -Wno-deprecated-declarations"
LK="-L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lmpc -lmpfr -lgmp -lpthread -lqd -lstdc++ -lm"
# CPU baseline library tag.  Default = SVE2 (strongest dense CPU baseline on this
# Neoverse-V2 / GB10 host: OpenMP + SVE2 SIMD).  Override with CPU_TAG=_neon etc.
# (Dense ops only here, so the SVE2 SpMV regression does not apply.)
CPU_TAG="${CPU_TAG:-_sve2}"
OMPLK="-L. -lbncmatmul-0.24-omp${CPU_TAG} -lbncmatmul-0.24${CPU_TAG} -Xcompiler -fopenmp -lgomp"

MR_OBJ="cuda_obj/gddlinear.o cuda_obj/gtdlinear.o cuda_obj/gqdlinear.o cuda_obj/gdslinear.o cuda_obj/gtslinear.o cuda_obj/gqslinear.o"

if [ ! -f libbncmatmul-0.24_cuda.a ]; then
    echo "*** libbncmatmul-0.24_cuda.a missing -- run 'sh bench/build_cuda_full.sh' first." >&2
    exit 1
fi
if [ ! -f libbncmatmul-0.24-omp${CPU_TAG}.a ]; then
    echo "*** libbncmatmul-0.24-omp${CPU_TAG}.a missing (CPU OpenMP baseline)." >&2
    exit 1
fi
mkdir -p cuda_obj bench/cuda/out
OK=0; FAIL=0; FL=""

# build_driver  <out-binary> <src.cu> <compile-defs> <object-list> [include-flags]
build_driver() {
    out="$1"; src="$2"; defs="$3"; obj="$4"; inc="${5:-$INC}"
    base=$(basename "$out")
    if $NVCC $inc $defs $SUP -dc "$src" -o cuda_obj/${base}.o 2>/tmp/be \
       && $NVCC -arch=$ARCH -dlink cuda_obj/${base}.o $obj -o cuda_obj/_${base}_dl.o 2>>/tmp/be \
       && $NVCC -arch=$ARCH cuda_obj/${base}.o $obj cuda_obj/_${base}_dl.o $OMPLK $LK -o "$out" 2>>/tmp/be
    then
        OK=$((OK+1)); echo "  [ok]   $out"
    else
        FAIL=$((FAIL+1)); FL="$FL $base"; echo "  [FAIL] $out"
        grep -m3 -iE 'error|undefined' /tmp/be | sed 's/^/         /'
    fi
}

echo "==== native real (d/f): matmul / matvec / axpy ===="
build_driver bench/cuda/cuda_dense_nr  bench/cuda/cuda_dense_nr.cu  ""        "cuda_obj/gdlinear.o cuda_obj/gflinear.o" "$NAT_INC"

echo "==== native complex (cd/cf): matmul / matvec ===="
build_driver bench/cuda/cuda_dense_ncx bench/cuda/cuda_dense_ncx.cu ""        "cuda_obj/gcdlinear.o cuda_obj/gcflinear.o" "$NAT_INC"

echo "==== multi-component real (dd/td/qd/ds/ts/qs): matmul / matvec / axpy ===="
build_driver bench/cuda/cuda_matmul_mr bench/cuda/cuda_matmul_mr.cu "$MR_DEFS" "$MR_OBJ"
build_driver bench/cuda/cuda_matvec_mr bench/cuda/cuda_matvec_mr.cu "$MR_DEFS" "$MR_OBJ"
build_driver bench/cuda/cuda_axpy_mr   bench/cuda/cuda_axpy_mr.cu   "$MR_DEFS" "$MR_OBJ"

echo "==== complex multi-component (cdd/ctd/cqd/cds/cts/cqs): matmul / matvec ===="
bash bench/cuda/build_complex.sh

# SpMV intentionally skipped: the dense axpy/matvec/matmul benchmark does not
# need it (would require cuda_obj/gdsparse.o).  Re-enable for SpMV numbers.
#echo "==== SpMV: native double ===="
#build_driver bench/cuda/cuda_spmv_d bench/cuda/cuda_spmv_d.cu "" "cuda_obj/gdsparse.o cuda_obj/gdlinear.o" "$NAT_INC"

echo "================ real drivers: $OK built, $FAIL failed ================"
[ -n "$FL" ] && { echo "failed:$FL"; exit 1; }
exit 0
