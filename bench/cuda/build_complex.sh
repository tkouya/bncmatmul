#!/bin/bash
#---------------------------------------------------------------------------
# build_complex.sh -- build the complex multi-component CUDA benchmark drivers.
#   cuda_dense_cm.cu compiled once PER PRECISION (cg*linear.o embeds the device
#   aggregator -> one precision per binary).  cdd/ctd/cqd use CPU OpenMP;
#   cds/cts/cqs have no OpenMP routine -> CPU serial (_4m).
# Run from repo root:  bash bench/cuda/build_complex.sh
#---------------------------------------------------------------------------
cd "$(dirname "$0")/../.."
ARCH=sm_121
NVCC=(nvcc -O3 -arch=$ARCH -std=c++17 -fmad=false -D__NV_NO_VECTOR_DEPRECATION_DIAG)
INC=(-Iinclude -I/usr/local/include/gdtq -I/usr/local/cuda/include)
SUP=(-diag-suppress 20011 -diag-suppress 177 -diag-suppress 550 -diag-suppress 20208 -diag-suppress 20044 -Wno-deprecated-declarations)
LK=(-L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lmpc -lmpfr -lgmp -lpthread -lqd -lstdc++ -lm)
# CPU baseline tag (default SVE2 = OpenMP+SVE2, strongest dense baseline here).
CPU_TAG="${CPU_TAG:-_sve2}"
OK=0; FAIL=0; FL=""

# build_cm  <p> <BASE> <SZ> <family d|s> <USE_defs...>
build_cm() {
    p=$1; BASE=$2; SZ=$3; fam=$4; shift 4; defs=("$@")
    U=$(echo "$p" | tr a-z A-Z)              # DD / DS ...
    CM="C${U}Matrix"; CV="C${U}Vector"; GCM="CG${U}Matrix"; GCV="CG${U}Vector"
    src=bench/cuda/cuda_dense_cm.cu; out=bench/cuda/cuda_dense_c${p}; obj=cuda_obj/cuda_dense_c${p}.o
    D=( -DPREC_NAME="\"c${p}\"" -DCGHDR="\"cg${p}linear.h\""
        -DMT=$CM -DVT=$CV -DGMT=$GCM -DGVT=$GCV -DBASE=$BASE -DSZ=$SZ
        -DINIT_M=init_c${p}matrix -DINIT_V=init_c${p}vector -DFREE_M=free_c${p}matrix -DFREE_V=free_c${p}vector
        -DSET_M=set_c${p}matrix_ij_d -DSET_V=set_c${p}vector_i_d
        -DGINIT_M=init_cg${p}matrix_dev -DGINIT_V=init_cg${p}vector_dev -DGFREE_M=free_cg${p}matrix_dev -DGFREE_V=free_cg${p}vector_dev
        -DH2D_M=subst_cg${p}matrix_dev_c${p}mat -DH2D_V=subst_cg${p}vector_dev_c${p}vec
        -DD2H_M=subst_c${p}matrix_cg${p}mat_dev -DD2H_V=subst_c${p}vector_cg${p}vec_dev
        -DGMM=mul_cg${p}matrix_dev -DGMV=mul_cg${p}matrix_cg${p}vec
        -DNORM2V=norm2_c${p}vector -DNORMFM=normf_c${p}matrix -DSUBV=sub_c${p}vector -DSUBM=sub_c${p}matrix
        -DCMV_FN=mul_c${p}matrix_c${p}vec_4m )
    # d-family (cdd/ctd/cqd) has an OpenMP routine -> SVE2 OMP baseline.
    # s-family (cds/cts/cqs) is serial-only AND the SVE2 libs do not build the
    # complex-single types at all -> fall back to the plain serial lib (tag="").
    tag="$CPU_TAG"
    if [ "$fam" = d ]; then
        D+=( -DUSE_OMP_CPU -DCPUKIND="\"omp\"" -DCMM_FN=_bncomp_mul_c${p}matrix_strassen_4m
             -DCMV_FN=_bncomp_mul_c${p}matrix_c${p}vec_4m )
    else
        tag=""
        D+=( -DSINGLE_BASED -DCPUKIND="\"serial\"" -DCMM_FN=mul_c${p}matrix_4m )
    fi
    if "${NVCC[@]}" "${INC[@]}" "${defs[@]}" "${D[@]}" "${SUP[@]}" -dc "$src" -o "$obj" 2>/tmp/be \
       && "${NVCC[@]:0:3}" -arch=$ARCH -dlink "$obj" cuda_obj/cg${p}linear.o -o cuda_obj/_c${p}_dl.o 2>>/tmp/be \
       && "${NVCC[@]:0:3}" -arch=$ARCH "$obj" cuda_obj/cg${p}linear.o cuda_obj/_c${p}_dl.o \
            -L. -lbncmatmul-0.24-omp${tag} -lbncmatmul-0.24${tag} -Xcompiler -fopenmp -lgomp "${LK[@]}" -o "$out" 2>>/tmp/be
    then OK=$((OK+1)); echo "  [ok]   $out"
    else FAIL=$((FAIL+1)); FL="$FL c${p}"; echo "  [FAIL] $out"; grep -m3 -iE 'error|undefined|Multiple' /tmp/be | sed 's/^/         /'
    fi
}

echo "==== complex multi-component: cdd/ctd/cqd (OpenMP) ===="
build_cm dd double DDSIZE d -DUSE_DDLINEAR -DUSE_GMP -DUSE_MPFR
build_cm td double TDSIZE d -DUSE_DDLINEAR -DUSE_TDLINEAR -DUSE_GMP -DUSE_MPFR
build_cm qd double QDSIZE d -DUSE_DDLINEAR -DUSE_TDLINEAR -DUSE_QDLINEAR -DUSE_GMP -DUSE_MPFR
echo "==== complex multi-component: cds/cts/cqs (serial; no OpenMP routine) ===="
build_cm ds float DSSIZE s -DUSE_DSLINEAR
build_cm ts float TSSIZE s -DUSE_DSLINEAR -DUSE_TSLINEAR
build_cm qs float QSSIZE s -DUSE_DSLINEAR -DUSE_TSLINEAR -DUSE_QSLINEAR

echo "================ complex: $OK built, $FAIL failed ================"
[ -n "$FL" ] && echo "failed:$FL"
exit 0
