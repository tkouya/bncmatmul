#!/bin/sh
#---------------------------------------------------------------------------
# build_dense_obj.sh -- compile ONLY the dense linear GPU objects needed by the
# axpy / matvec / matmul benchmark drivers (no LU, no sparse).  Subset of
# build_cuda_full.sh, kept in the current release tree.  Objects -> cuda_obj/.
#   real native    : gd gf
#   real multicomp : gdd gtd gqd gds gts gqs
#   complex native : gcd gcf
#   complex multic.: cgdd cgtd cgqd cgds cgts cgqs
# Target: NVIDIA GB10, sm_121, CUDA 13.  Run from the repo root.
#---------------------------------------------------------------------------
set -e
cd "$(dirname "$0")/../.."          # repo root

ARCH=sm_121
NVCC="nvcc -O3 -arch=$ARCH -std=c++17 -DCUDA_FMA"
GDTQ=/usr/local/include/gdtq
INC="-Iinclude -I$GDTQ -I/usr/local/cuda/include"
NAT_INC="-Iinclude -I/usr/local/cuda/include"
SUP="-diag-suppress 20011 -diag-suppress 177 -diag-suppress 550 -diag-suppress 20208 -Wno-deprecated-declarations"
mkdir -p cuda_obj

defs_dd="-DUSE_DDLINEAR";
defs_td="-DUSE_DDLINEAR -DUSE_TDLINEAR";
defs_qd="-DUSE_DDLINEAR -DUSE_TDLINEAR -DUSE_QDLINEAR";
defs_ds="-DUSE_DSLINEAR";
defs_ts="-DUSE_DSLINEAR -DUSE_TSLINEAR";
defs_qs="-DUSE_DSLINEAR -DUSE_TSLINEAR -DUSE_QSLINEAR";

echo "== multi-component real linear (dd td qd ds ts qs) =="
for t in dd td qd ds ts qs; do
	eval d=\$defs_$t
	$NVCC $INC $d -DUSE_GMP -DUSE_MPFR $SUP -dc src/g${t}linear.cu -o cuda_obj/g${t}linear.o
	echo "  [ok] g${t}linear.o"
done

echo "== native real + complex linear (d f cd cf) =="
for t in d f cd cf; do
	$NVCC $NAT_INC $SUP -dc src/g${t}linear.cu -o cuda_obj/g${t}linear.o
	echo "  [ok] g${t}linear.o (native)"
done

echo "== complex multi-component linear (cgdd cgtd cgqd cgds cgts cgqs) =="
for t in dd td qd ds ts qs; do
	eval d=\$defs_$t
	$NVCC $INC $d -DUSE_GMP -DUSE_MPFR $SUP -dc src/cg${t}linear.cu -o cuda_obj/cg${t}linear.o
	echo "  [ok] cg${t}linear.o"
done

echo "== minimal libbncmatmul-0.24_cuda.a (dense linear only, presence guard) =="
rm -f libbncmatmul-0.24_cuda.a
ar rc libbncmatmul-0.24_cuda.a cuda_obj/g*linear.o cuda_obj/cg*linear.o
echo "DONE: 16 dense linear objects in cuda_obj/"
