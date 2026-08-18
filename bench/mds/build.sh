#!/bin/bash
# Build the mds_bench harness for serial / avx2 / avx512 backends.
set -e
cd "$(dirname "$0")/../.."   # repo root
ROOT=$(pwd)

# -Iinclude MUST precede the GMP/MPFR/QD prefix: a previous `make install`
# leaves BNCmatmul headers under that same prefix, and those stale copies
# would otherwise shadow the in-tree include/ that the .a files were built
# from (mismatched struct layouts, missing new declarations).
INC="-Iinclude -I/home/tkouya/local/include -DUSE_DD -DUSE_QD -DUSE_TD_BF -DUSE_QD_BF -DUSE_TS_BF -DUSE_QS_BF -DUSE_GMP -DUSE_MPFR"
LK="-lmpc -lmpfr -lgmp -lpthread -L/home/tkouya/local/lib -lqd -lstdc++ -lm"
OPT="-std=c++17 -O3 -ffp-contract=off"
SRC=bench/mds/mds_bench.cc

echo "[serial] ..."
g++ $OPT $INC $SRC -L. -lbncmatmul-0.24 $LK -o bench/mds/mds_bench_serial

echo "[avx2] ..."
g++ $OPT -mavx2 -mfma $INC $SRC -L. -lbncmatmul-0.24_avx2 $LK -o bench/mds/mds_bench_avx2

echo "[avx512] ... (full: ds/ts/qs + sparse-mp AVX-512 now supported)"
g++ $OPT -mavx512f -mfma $INC $SRC -L. -lbncmatmul-0.24_avx512 $LK -o bench/mds/mds_bench_avx512

echo "[complex serial/avx2/avx512] ..."
SRCC=bench/mds/mds_bench_cplx.cc
g++ $OPT             $INC $SRCC -L. -lbncmatmul-0.24        $LK -o bench/mds/mds_bench_cplx_serial
g++ $OPT -mavx2 -mfma   $INC $SRCC -L. -lbncmatmul-0.24_avx2   $LK -o bench/mds/mds_bench_cplx_avx2
g++ $OPT -mavx512f -mfma $INC $SRCC -L. -lbncmatmul-0.24_avx512 $LK -o bench/mds/mds_bench_cplx_avx512

echo "done."
ls -la bench/mds/mds_bench_serial bench/mds/mds_bench_avx2 bench/mds/mds_bench_avx512 \
       bench/mds/mds_bench_cplx_serial bench/mds/mds_bench_cplx_avx2 bench/mds/mds_bench_cplx_avx512
