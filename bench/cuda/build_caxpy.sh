#!/bin/bash
#---------------------------------------------------------------------------
# build_caxpy.sh -- build the complex axpy (cmul) GPU-only benchmark drivers,
# one per complex precision.  Complements build_complex.sh (matmul/matvec) so
# the complex side covers all three ops axpy/matvec/matmul.
#   native : cd cf            (g<cd|cf>linear.o, scalar = double/float)
#   multic.: cdd ctd cqd cds cts cqs  (cg*linear.o, scalar = gXX_real limb array)
# Run from repo root:  bash bench/cuda/build_caxpy.sh
#---------------------------------------------------------------------------
cd "$(dirname "$0")/../.."
ARCH=sm_121
NVCC=(nvcc -O3 -arch=$ARCH -std=c++17 -fmad=false -D__NV_NO_VECTOR_DEPRECATION_DIAG)
INC=(-Iinclude -I/usr/local/include/gdtq -I/usr/local/cuda/include)
SUP=(-diag-suppress 20011 -diag-suppress 177 -diag-suppress 550 -diag-suppress 20208 -diag-suppress 20044 -Wno-deprecated-declarations)
LK=(-L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lqd -lstdc++ -lm)
src=bench/cuda/cuda_caxpy.cu
OK=0; FAIL=0; FL=""

# build_native <p=cd|cf> <BASE>
build_native() {
    p=$1; BASE=$2; U=$(echo "$p" | tr a-z A-Z)
    out=bench/cuda/cuda_caxpy_${p}; obj=cuda_obj/cuda_caxpy_${p}.o
    D=( -DPREC_NAME="\"${p}\"" -DCGHDR="\"g${p}linear.h\"" -DGVT=G${U}Vector
        -DGINIT_V=init_g${p}vector_dev -DGFREE_V=free_g${p}vector_dev
        -DGCMUL=cmul_g${p}vector_dev -DBASE=$BASE )
    build_link "$p" "$out" "$obj" cuda_obj/g${p}linear.o "${D[@]}"
}
# build_mc <p> <BASE> <SZ> <USE_defs...>
build_mc() {
    p=$1; BASE=$2; SZ=$3; shift 3; defs=("$@"); U=$(echo "$p" | tr a-z A-Z)
    out=bench/cuda/cuda_caxpy_c${p}; obj=cuda_obj/cuda_caxpy_c${p}.o
    D=( -DPREC_NAME="\"c${p}\"" -DCGHDR="\"cg${p}linear.h\"" -DMULTICOMP
        -DGVT=CG${U}Vector -DGINIT_V=init_cg${p}vector_dev -DGFREE_V=free_cg${p}vector_dev
        -DGCMUL=cmul_cg${p}vector_dev -DGR=g${p}_real -DBASE=$BASE -DSZ=$SZ "${defs[@]}" )
    build_link "c${p}" "$out" "$obj" cuda_obj/cg${p}linear.o "${D[@]}"
}
build_link() {
    tag=$1; out=$2; obj=$3; lobj=$4; shift 4; D=("$@")
    if "${NVCC[@]}" "${INC[@]}" "${D[@]}" "${SUP[@]}" -dc "$src" -o "$obj" 2>/tmp/be \
       && "${NVCC[@]:0:3}" -arch=$ARCH -dlink "$obj" "$lobj" -o cuda_obj/_${tag}ax_dl.o 2>>/tmp/be \
       && "${NVCC[@]:0:3}" -arch=$ARCH "$obj" "$lobj" cuda_obj/_${tag}ax_dl.o \
            -L. -lbncmatmul-0.24 "${LK[@]}" -o "$out" 2>>/tmp/be
    then OK=$((OK+1)); echo "  [ok]   $out"
    else FAIL=$((FAIL+1)); FL="$FL $tag"; echo "  [FAIL] $out"; grep -m4 -iE 'error|undefined|Multiple' /tmp/be | sed 's/^/         /'
    fi
}

echo "==== complex axpy(cmul): native cd/cf ===="
build_native cd double
build_native cf float
echo "==== complex axpy(cmul): multi-component cdd/ctd/cqd ===="
build_mc dd double DDSIZE -DUSE_DDLINEAR -DUSE_GMP -DUSE_MPFR
build_mc td double TDSIZE -DUSE_DDLINEAR -DUSE_TDLINEAR -DUSE_GMP -DUSE_MPFR
build_mc qd double QDSIZE -DUSE_DDLINEAR -DUSE_TDLINEAR -DUSE_QDLINEAR -DUSE_GMP -DUSE_MPFR
echo "==== complex axpy(cmul): multi-component cds/cts/cqs ===="
build_mc ds float DSSIZE -DUSE_DSLINEAR
build_mc ts float TSSIZE -DUSE_DSLINEAR -DUSE_TSLINEAR
build_mc qs float QSSIZE -DUSE_DSLINEAR -DUSE_TSLINEAR -DUSE_QSLINEAR

echo "================ complex axpy: $OK built, $FAIL failed ================"
[ -n "$FL" ] && echo "failed:$FL"
exit 0
