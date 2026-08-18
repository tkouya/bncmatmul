#!/bin/sh
# Build matmul-algorithm benchmark: 8 precisions x 3 SIMD backends = 24 binaries.
#   precisions: double dd td qd  float ds ts qs
#   algorithms (inside each binary): triple / block / strassen / winograd
#   backends:   serial(no SIMD) / neon / sve2  (chosen by library + -mcpu flags)
cd "$(dirname "$0")/../.."   # repo root
CF="-O3 -ffp-contract=off -Iinclude"
LK="-lmpc -lmpfr -lgmp -lpthread -lqd -lstdc++ -lm"
OUT=bench/mm
mkdir -p $OUT/out
OK=0; FAIL=0; FL=""

# build_one <out> <src> <extra-defs...>   (loops the 3 backends)
build_one() {
  out="$1"; src="$2"; shift 2; extra="$*"
  for bk in serial neon sve2; do
    case "$bk" in
      serial) bflag="";                                        blib="-lbncmatmul-0.24";;
      neon)   bflag="-mcpu=cortex-a76 -DBNC_ENABLE_NEON";      blib="-lbncmatmul-0.24_neon";;
      sve2)   bflag="-mcpu=neoverse-v2 -DBNC_ENABLE_SVE2 -DBNC_ENABLE_NEON"; blib="-lbncmatmul-0.24_sve2";;
    esac
    o="${OUT}/${out}_${bk}"
    if gcc $CF $bflag $extra -DBACKEND_NAME="\"$bk\"" "$src" -L. $blib $LK -o "$o" 2>/tmp/mmbe; then
      OK=$((OK+1)); echo "  [ok]   $o"
    else
      FAIL=$((FAIL+1)); FL="$FL $o"; echo "  [FAIL] $o"; grep -m1 error: /tmp/mmbe|sed 's/^/        /'
    fi
  done
}
# float is self-contained (no library symbols) -> same builder, harmless lib link
build_free() {
  out="$1"; src="$2"; shift 2; extra="$*"
  for bk in serial neon sve2; do
    case "$bk" in
      serial) bflag="";;
      neon)   bflag="-mcpu=cortex-a76 -DBNC_ENABLE_NEON";;
      sve2)   bflag="-mcpu=neoverse-v2 -DBNC_ENABLE_SVE2 -DBNC_ENABLE_NEON";;
    esac
    o="${OUT}/${out}_${bk}"
    if gcc $CF $bflag $extra -DBACKEND_NAME="\"$bk\"" "$src" -lmpfr -lgmp -lm -o "$o" 2>/tmp/mmbe; then
      OK=$((OK+1)); echo "  [ok]   $o"
    else
      FAIL=$((FAIL+1)); FL="$FL $o"; echo "  [FAIL] $o"; grep -m1 error: /tmp/mmbe|sed 's/^/        /'
    fi
  done
}

echo "==== native double ===="
build_one mm_double $OUT/mm_double.c -DPREC_NAME='"double"'

echo "==== multicomponent (double-based: dd td qd) ===="
build_one mm_dd $OUT/mm_multi.c -DP=dd -DMT=DDMatrix -DBASE=double -DPSIZE=DDSIZE -DPREC_NAME='"dd"'
build_one mm_td $OUT/mm_multi.c -DP=td -DMT=TDMatrix -DBASE=double -DPSIZE=TDSIZE -DPREC_NAME='"td"'
build_one mm_qd $OUT/mm_multi.c -DP=qd -DMT=QDMatrix -DBASE=double -DPSIZE=QDSIZE -DPREC_NAME='"qd"'

echo "==== native float (self-contained algorithms) ===="
build_free mm_float $OUT/mm_float.c -DPREC_NAME='"float"'

echo "==== multicomponent (single-based: ds ts qs) ===="
build_one mm_ds $OUT/mm_multi.c -DP=ds -DMT=DSMatrix -DBASE=float -DPSIZE=DSSIZE -DPREC_NAME='"ds"'
build_one mm_ts $OUT/mm_multi.c -DP=ts -DMT=TSMatrix -DBASE=float -DPSIZE=TSSIZE -DPREC_NAME='"ts"'
build_one mm_qs $OUT/mm_multi.c -DP=qs -DMT=QSMatrix -DBASE=float -DPSIZE=QSSIZE -DPREC_NAME='"qs"'

echo "================ $OK built, $FAIL failed ================"
[ -n "$FL" ] && echo "failed:$FL"
