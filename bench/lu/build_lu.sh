#!/bin/sh
# Build LU FMA benchmark binaries: 6 precisions x {serial,neon,sve2} x {nofma,fma}.
# The precision's LU translation unit is recompiled per config and linked BEFORE
# the library, so the fma binaries run the BNC_USE_NEW_FMA kernels while
# everything else (matvec, init, ...) comes from the fixed library.
cd "$(dirname "$0")/../.."   # repo root
CF="-O3 -ffp-contract=off -Iinclude"
LK="-lmpc -lmpfr -lgmp -lpthread -lqd -lstdc++ -lm"
DEFS="-DUSE_GMP -DUSE_MPFR -DUSE_TD_BF -DUSE_QD_BF -DUSE_TS_BF -DUSE_QS_BF"
LINDEFS=""
OUT=bench/lu
OK=0; FAIL=0; FL=""

# prec PU MT VT BASE PSIZE lusrc
TAB="dd DD DDMatrix DDVector double DDSIZE src/ddlu.c
td TD TDMatrix TDVector double TDSIZE src/tdlu.c
qd QD QDMatrix QDVector double QDSIZE src/qdlu.c
ds DS DSMatrix DSVector float DSSIZE src/dslu.c
ts TS TSMatrix TSVector float TSSIZE src/tslinear.c
qs QS QSMatrix QSVector float QSSIZE src/qslinear.c"

echo "$TAB" | while read p PU MT VT BASE PSIZE lusrc; do
  for bk in serial neon sve2; do
    case "$bk" in
      serial) bflag=""; blib="-lbncmatmul-0.24";;
      neon)   bflag="-mcpu=cortex-a76 -DBNC_ENABLE_NEON"; blib="-lbncmatmul-0.24_neon";;
      sve2)   bflag="-mcpu=neoverse-v2 -DBNC_ENABLE_SVE2 -DBNC_ENABLE_NEON"; blib="-lbncmatmul-0.24_sve2";;
    esac
    for fm in nofma fma; do
      [ "$fm" = "fma" ] && fmflag="-DBNC_USE_NEW_FMA" || fmflag=""
      o="${OUT}/lu_${p}_${bk}_${fm}"
      luobj="${OUT}/obj_${p}_${bk}_${fm}.o"
      if gcc $CF $bflag $fmflag $DEFS $LINDEFS -c "$lusrc" -o "$luobj" 2>/tmp/be &&
         gcc $CF $bflag $fmflag $DEFS $LINDEFS \
             -DP=$p -DPU=$PU -DMT=$MT -DVT=$VT -DBASE=$BASE -DPSIZE=$PSIZE \
             -DPREC_NAME="\"$p\"" -DBACKEND_NAME="\"$bk\"" \
             $OUT/lu_mr.c "$luobj" -L. $blib $LK -o "$o" 2>>/tmp/be; then
        OK=$((OK+1)); echo "  [ok]   $o"
      else
        FAIL=$((FAIL+1)); FL="$FL $o"; echo "  [FAIL] $o"; grep -m2 -E "error|Error" /tmp/be | sed 's/^/         /'
      fi
    done
  done
  echo "build so far: ok=$OK fail=$FAIL$FL"
done
