#!/bin/sh
# Build the branch-free DW/TW/QW FMA tests and the AXPY/GEMV/GEMM benchmark
# for serial / NEON / SVE2.
cd "$(dirname "$0")/../.."   # repo root

CF="-O3 -ffp-contract=off -funroll-loops -Iinclude -Ibench/fma -I/usr/local/include"
LK="-L/usr/local/lib -lmpfr -lgmp -lm"
OUT=bench/fma
OK=0; FAIL=0

# backend flags
NEON_FLAGS="-mcpu=native -DUSE_NEON -DBNC_ENABLE_NEON"
SVE2_FLAGS="-mcpu=neoverse-v2 -msve-vector-bits=128 -DUSE_SVE2 -DBNC_ENABLE_SVE2 -DBNC_ENABLE_NEON"

build() { # name flags... ; reads $SRC
	name="$1"; shift
	if gcc $CF "$@" "$SRC" $LK -o "$OUT/$name" 2>/tmp/fmabe; then
		OK=$((OK+1)); echo "  [ok]   $OUT/$name"
	else
		FAIL=$((FAIL+1)); echo "  [FAIL] $OUT/$name"; sed -n '1,12p' /tmp/fmabe | sed 's/^/         /'
	fi
}

echo "==== reference / certified-bound test (scalar) ===="
SRC=$OUT/test_fma_ref.c
build test_fma_ref

echo "==== backend bitwise-agreement test (NEON + SVE2 in one binary) ===="
SRC=$OUT/test_fma_simd.c
build test_fma_simd -mcpu=neoverse-v2 -DBNC_ENABLE_NEON -DBNC_ENABLE_SVE2

echo "==== SVE2 vs NEON kernel audit ===="
SRC=$OUT/test_sve2_vs_neon.c
build test_sve2_vs_neon -mcpu=neoverse-v2 -DBNC_ENABLE_NEON -DBNC_ENABLE_SVE2

echo "==== OpenMP thread-safety test ===="
SRC=$OUT/test_fma_omp.c
build test_fma_omp -fopenmp -mcpu=neoverse-v2 -DBNC_ENABLE_NEON -DBNC_ENABLE_SVE2

echo "==== AXPY/GEMV/GEMM benchmark ===="
SRC=$OUT/fma_blas_bench.c
build fma_bench_serial -fopenmp
build fma_bench_neon   -fopenmp $NEON_FLAGS
build fma_bench_sve2   -fopenmp $SVE2_FLAGS

echo "==== AXPY/GEMV/GEMM benchmark (single-word base: DS/TS/QS) ===="
build fma_bench_serial_f -fopenmp -DUSE_FLOAT_BASE
build fma_bench_neon_f   -fopenmp -DUSE_FLOAT_BASE $NEON_FLAGS
build fma_bench_sve2_f   -fopenmp -DUSE_FLOAT_BASE $SVE2_FLAGS

echo "==== polynomial benchmark (Horner / Estrin / eval_diff / poly-mul) ===="
# these link the library's own poly sources, built twice: baseline vs BNC_USE_NEW_FMA
POLYSRC="src/dd_poly.c src/td_poly.c src/qd_poly.c"
POLYLK="-L. -lbncmatmul-0.24 -L/usr/local/lib -lmpc -lmpfr -lgmp -lqd -lstdc++ -lm"
pbuild() { # name flags...
	name="$1"; shift
	if gcc $CF -Isrc -DUSE_GMP -DUSE_MPFR "$@" $OUT/poly_bench.c $POLYSRC $POLYLK \
	       -o "$OUT/$name" 2>/tmp/fmabe; then
		OK=$((OK+1)); echo "  [ok]   $OUT/$name"
	else
		FAIL=$((FAIL+1)); echo "  [FAIL] $OUT/$name"; sed -n '1,8p' /tmp/fmabe | sed 's/^/         /'
	fi
}
for bk in serial neon sve2; do
	case "$bk" in
		serial) bf="";;
		neon)   bf="-mcpu=native -DBNC_ENABLE_NEON";;
		sve2)   bf="-mcpu=neoverse-v2 -msve-vector-bits=128 -DBNC_ENABLE_SVE2 -DBNC_ENABLE_NEON";;
	esac
	pbuild poly_bench_$bk       $bf
	pbuild poly_bench_${bk}_bf  $bf -DUSE_DD_BF -DUSE_TD_BF -DUSE_QD_BF
	pbuild poly_bench_${bk}_fma $bf -DBNC_USE_NEW_FMA
done

echo "================ $OK built, $FAIL failed ================"
[ "$FAIL" -eq 0 ]
