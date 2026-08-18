#!/bin/sh
# Build all arm_bench dense + complex-spmv binaries for serial/neon/sve2.
cd "$(dirname "$0")/../.."   # repo root
CF="-O3 -ffp-contract=off -Iinclude"
LK="-lmpc -lmpfr -lgmp -lpthread -lqd -lstdc++ -lm"
SPDEF="-DUSE_DD -DUSE_QD -DUSE_GMP -DUSE_MPFR -DUSE_DDLINEAR -DUSE_TDLINEAR -DUSE_QDLINEAR"
OUT=bench/arm
OK=0; FAIL=0; FL=""

build() { # out src extradefs
  out="$1"; src="$2"; shift 2; extra="$*"
  for bk in serial neon sve2; do
    case "$bk" in
      serial) bflag=""; blib="-lbncmatmul-0.24";;
      neon)   bflag="-mcpu=cortex-a76 -DBNC_ENABLE_NEON"; blib="-lbncmatmul-0.24_neon";;
      sve2)   bflag="-mcpu=neoverse-v2 -DBNC_ENABLE_SVE2 -DBNC_ENABLE_NEON"; blib="-lbncmatmul-0.24_sve2";;
    esac
    o="${OUT}/${out}_${bk}"
    if gcc $CF $bflag $extra -DBACKEND_NAME="\"$bk\"" "$src" -L. $blib $LK -o "$o" 2>/tmp/be; then
      OK=$((OK+1)); echo "  [ok]   $o"
    else
      FAIL=$((FAIL+1)); FL="$FL $o"; echo "  [FAIL] $o"; grep -m1 error: /tmp/be|sed 's/^/        /'
    fi
  done
}

echo "==== dense native real ===="
build dense_d  $OUT/dense_nr.c -DP=d -DMT=DMatrix -DVT=DVector -DBASE=double -DMVFN=mul_dmatrix_dvec -DPREC_NAME='"double"'
build dense_f  $OUT/dense_nr.c -DP=f -DMT=FMatrix -DVT=FVector -DBASE=float  -DMVFN=mul_fmatrix_dvec -DPREC_NAME='"float"'

echo "==== dense multicomp real ===="
build dense_dd $OUT/dense_mr.c -DP=dd -DMT=DDMatrix -DVT=DDVector -DBASE=double -DPSIZE=DDSIZE -DPREC_NAME='"dd"'
build dense_td $OUT/dense_mr.c -DP=td -DMT=TDMatrix -DVT=TDVector -DBASE=double -DPSIZE=TDSIZE -DPREC_NAME='"td"'
build dense_qd $OUT/dense_mr.c -DP=qd -DMT=QDMatrix -DVT=QDVector -DBASE=double -DPSIZE=QDSIZE -DPREC_NAME='"qd"'
build dense_ds $OUT/dense_mr.c -DP=ds -DMT=DSMatrix -DVT=DSVector -DBASE=float  -DPSIZE=DSSIZE -DPREC_NAME='"ds"'
build dense_ts $OUT/dense_mr.c -DP=ts -DMT=TSMatrix -DVT=TSVector -DBASE=float  -DPSIZE=TSSIZE -DPREC_NAME='"ts"'
build dense_qs $OUT/dense_mr.c -DP=qs -DMT=QSMatrix -DVT=QSVector -DBASE=float  -DPSIZE=QSSIZE -DPREC_NAME='"qs"'

echo "==== dense native complex ===="
build dense_cd $OUT/dense_nc.c -DPREC_NAME='"cd"'
# cf = native 'float _Complex' (self-contained, no lib symbols; serial/neon/sve2 = auto-vec only)
build dense_cf $OUT/dense_ncf.c -DPREC_NAME='"cf"'

echo "==== dense multicomp complex ===="
build dense_cdd $OUT/dense_mc.c -DP=cdd -DMT=CDDMatrix -DVT=CDDVector -DRCSET=rcdd_set_d_d -DPREC_NAME='"cdd"'
build dense_ctd $OUT/dense_mc.c -DP=ctd -DMT=CTDMatrix -DVT=CTDVector -DRCSET=rctd_set_d_d -DPREC_NAME='"ctd"'
build dense_cqd $OUT/dense_mc.c -DP=cqd -DMT=CQDMatrix -DVT=CQDVector -DRCSET=rcqd_set_d_d -DPREC_NAME='"cqd"'
build dense_cds $OUT/dense_mc.c -DSINGLE_BASED -DP=cds -DMT=CDSMatrix -DVT=CDSVector -DRCSET=rcds_set_d_d -DPREC_NAME='"cds"'
build dense_cts $OUT/dense_mc.c -DSINGLE_BASED -DP=cts -DMT=CTSMatrix -DVT=CTSVector -DRCSET=rcts_set_d_d -DPREC_NAME='"cts"'
build dense_cqs $OUT/dense_mc.c -DSINGLE_BASED -DP=cqs -DMT=CQSMatrix -DVT=CQSVector -DRCSET=rcqs_set_d_d -DPREC_NAME='"cqs"'

echo "==== complex SpMV ===="
build spmv_cd  $OUT/spmv_complex.c $SPDEF -DP=cd  -DCT=CDRSMatrix  -DVT=CDVector  -DNATIVE_BASE -DSETVEC=set_cdvector_i     -DPSIZE=1 -DPREC_NAME='"cd"'
build spmv_cdd $OUT/spmv_complex.c $SPDEF -DP=cdd -DCT=CDDRSMatrix -DVT=CDDVector -DSETVEC=set_cddvector_i_cd -DPSIZE=2 -DPREC_NAME='"cdd"'
build spmv_ctd $OUT/spmv_complex.c $SPDEF -DP=ctd -DCT=CTDRSMatrix -DVT=CTDVector -DSETVEC=set_ctdvector_i_cd -DPSIZE=3 -DPREC_NAME='"ctd"'
build spmv_cqd $OUT/spmv_complex.c $SPDEF -DP=cqd -DCT=CQDRSMatrix -DVT=CQDVector -DSETVEC=set_cqdvector_i_cd -DPSIZE=4 -DPREC_NAME='"cqd"'
# cf = native 'float _Complex' SpMV (self-contained MatrixMarket reader, no lib symbols)
build spmv_cf  $OUT/spmv_cf.c -DPREC_NAME='"cf"'

# ---- cf hand-SIMD x86 variants (AVX2 / AVX-512) -- only meaningful on x86_64 ----
# cf is self-contained (bench/arm/cf_simd.h); NEON/SVE2 built above via build().
if [ "$(uname -m)" = "x86_64" ]; then
  echo "==== cf x86 SIMD (AVX2 / AVX-512) ===="
  for v in "avx2:-mavx2 -mfma -DBNC_ENABLE_AVX2" "avx512:-mavx512f -DBNC_ENABLE_AVX512"; do
    bk=${v%%:*}; fl=${v#*:}
    for kind in dense_ncf:dense_cf spmv_cf:spmv_cf; do
      src=${kind%%:*}; out=${kind#*:}
      if gcc -O3 -ffp-contract=off $fl -DPREC_NAME='"cf"' -DBACKEND_NAME="\"$bk\"" \
             "$OUT/${src}.c" -lm -o "$OUT/${out}_${bk}" 2>/tmp/be; then
        OK=$((OK+1)); echo "  [ok]   $OUT/${out}_${bk}"
      else
        FAIL=$((FAIL+1)); FL="$FL $OUT/${out}_${bk}"; echo "  [FAIL] $OUT/${out}_${bk}"; grep -m1 error: /tmp/be|sed 's/^/        /'
      fi
    done
  done
else
  echo "(skipping cf AVX2/AVX-512: host is $(uname -m), not x86_64)"
fi

echo "================ $OK built, $FAIL failed ================"
[ -n "$FL" ] && echo "failed:$FL"
