#!/bin/sh
# run_regress.sh - build + run the numerical regression suite and emit a
# timing-free CSV for diffing against a stored baseline.
#
#   sh bench/regress/run_regress.sh out.csv
#
# Coverage (serial / neon / sve2 backends):
#   - matmul accuracy vs 512-bit MPFR: double dd td qd float ds ts qs
#     (bench/mm mm_multi.c, mode acc)
#   - LU decomp+solve relerr, nofma+fma: dd td qd ds ts qs (bench/lu lu_mr.c)
#   - complex matmul/matvec/LU + bit digest: cdd ctd cqd cds cts cqs
#     (bench/regress/regress_mc.c)
#   - sparse SpMV verify: float ds ts qs cds cts cqs (bench/arm test_sparse_*)
#
# All inputs are deterministic; every emitted field is timing-free, so two
# runs on the same sources must produce identical files.
cd "$(dirname "$0")/../.."   # repo root
OUTCSV="${1:-bench/regress/out/regress.csv}"
mkdir -p "$(dirname "$OUTCSV")" bench/regress/out
: > "$OUTCSV"

CF="-O3 -ffp-contract=off -Iinclude"
LK="-lmpc -lmpfr -lgmp -lpthread -lqd -lstdc++ -lm"
DEFS="-DUSE_DD -DUSE_QD -DUSE_GMP -DUSE_MPFR -DUSE_MPC"
MMDIM=128
LUDIM=128
MCDIM=96
SPN=137
OK=0; FAIL=0; FL=""

bflags() { # backend -> compile flags
  case "$1" in
    serial) echo "";;
    neon)   echo "-mcpu=cortex-a76 -DBNC_ENABLE_NEON";;
    sve2)   echo "-mcpu=neoverse-v2 -DBNC_ENABLE_SVE2 -DBNC_ENABLE_NEON";;
  esac
}
blib() {
  case "$1" in
    serial) echo "-lbncmatmul-0.24";;
    neon)   echo "-lbncmatmul-0.24_neon";;
    sve2)   echo "-lbncmatmul-0.24_sve2";;
  esac
}
note() { OK=$((OK+1)); }
bad()  { FAIL=$((FAIL+1)); FL="$FL $1"; echo "  [FAIL] $1"; grep -m2 -E "error|Error" /tmp/rgerr 2>/dev/null | sed 's/^/         /'; }

# ---- 1. matmul accuracy (bench/mm binaries; always rebuild: stale
#         binaries from other machines/architectures must never survive) ----
echo "==== mm acc (dim=$MMDIM) ===="
sh bench/mm/build_mm.sh >/dev/null 2>&1
for p in double dd td qd float ds ts qs; do
  for bk in serial neon sve2; do
    bin=bench/mm/mm_${p}_${bk}
    if [ -x "$bin" ] && "$bin" $MMDIM acc > /tmp/rgout 2>/dev/null; then
      grep '^ACC,' /tmp/rgout >> "$OUTCSV" && note
    else
      bad "$bin"
    fi
  done
done

# ---- 2. LU relerr (bench/lu binaries; always rebuild) ----
echo "==== lu (dim=$LUDIM) ===="
sh bench/lu/build_lu.sh >/dev/null 2>&1
for p in dd td qd ds ts qs; do
  for bk in serial neon sve2; do
    for fm in nofma fma; do
      bin=bench/lu/lu_${p}_${bk}_${fm}
      if [ -x "$bin" ] && "$bin" $LUDIM > /tmp/rgout 2>/dev/null; then
        # RESULT,prec,backend,fma,dim,t_lu,t_solve,relerr -> drop the timings
        grep '^RESULT,' /tmp/rgout | \
          awk -F, -v OFS=, '{print "LU",$2,$3,$4,$5,$8}' >> "$OUTCSV" && note
      else
        bad "$bin"
      fi
    done
  done
done

# ---- 3. complex regression (regress_mc) ----
echo "==== complex mc (dim=$MCDIM) ===="
mc_build_run() { # prec PU MT VT PSIZE single extra_srcs
  p="$1"; PU="$2"; MTN="$3"; VTN="$4"; PS="$5"; sb="$6"; shift 6; xsrc="$*"
  for bk in serial neon sve2; do
    bin=bench/regress/out/regress_${p}_${bk}
    if gcc $CF $(bflags $bk) $DEFS $sb \
         -DP=$p -DPU=$PU -DMT=$MTN -DVT=$VTN -DPSIZE=$PS \
         -DPREC_NAME="\"$p\"" -DBACKEND_NAME="\"$bk\"" \
         bench/regress/regress_mc.c $xsrc -L. $(blib $bk) $LK -o "$bin" 2>/tmp/rgerr \
       && "$bin" $MCDIM > /tmp/rgout 2>/dev/null; then
      grep '^REG,' /tmp/rgout >> "$OUTCSV" && note
    else
      bad "$bin"
    fi
  done
}
mc_build_run cdd CDD CDDMatrix CDDVector DDSIZE ""
mc_build_run ctd CTD CTDMatrix CTDVector TDSIZE ""
mc_build_run cqd CQD CQDMatrix CQDVector QDSIZE ""
# complex-single family is absent from the legacy archives -> compile sources in
mc_build_run cds CDS CDSMatrix CDSVector DSSIZE "-DSINGLE_BASED" src/cdslinear.c src/cdslu.c
mc_build_run cts CTS CTSMatrix CTSVector TSSIZE "-DSINGLE_BASED" src/ctslinear.c src/ctslu.c
mc_build_run cqs CQS CQSMatrix CQSVector QSSIZE "-DSINGLE_BASED" src/cqslinear.c src/cqslu.c

# ---- 4. sparse SpMV verify (bench/arm test_sparse_*) ----
echo "==== sparse verify (n=$SPN) ===="
sp_build_run() { # prec extra_srcs
  p="$1"; shift; xsrc="$*"
  for bk in serial neon sve2; do
    bin=bench/regress/out/tsparse_${p}_${bk}
    if gcc $CF $(bflags $bk) $DEFS \
         -DBACKEND_NAME="\"$bk\"" \
         bench/arm/test_sparse_${p}.c $xsrc -L. $(blib $bk) $LK -o "$bin" 2>/tmp/rgerr \
       && "$bin" $SPN > /tmp/rgout 2>/dev/null; then
      grep '^VERIFY,' /tmp/rgout >> "$OUTCSV" && note
    else
      bad "$bin"
    fi
  done
}
sp_build_run float src/sparse_float.c
sp_build_run ds  src/sparse_ds.c
sp_build_run ts  src/sparse_ts.c
sp_build_run qs  src/sparse_qs.c
sp_build_run cds src/sparse_cds.c src/cdslinear.c
sp_build_run cts src/sparse_cts.c src/ctslinear.c
sp_build_run cqs src/sparse_cqs.c src/cqslinear.c

sort -o "$OUTCSV" "$OUTCSV"
echo "================ ok=$OK fail=$FAIL -> $OUTCSV ($(wc -l < "$OUTCSV") lines) ================"
[ -n "$FL" ] && echo "failed:$FL"
[ "$FAIL" -eq 0 ]
