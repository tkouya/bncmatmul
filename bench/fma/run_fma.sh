#!/bin/sh
# Run the branch-free FMA verification + AXPY/GEMV/GEMM benchmark.
#   AXPY n=10^6, GEMV n=2048, GEMM n=512  (same sizes as arXiv:2607.11391)
cd "$(dirname "$0")/../.."   # repo root
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

NAXPY=${NAXPY:-1000000}
NGEMV=${NGEMV:-2048}
NGEMM=${NGEMM:-512}
PDEG=${PDEG:-1000}
PNPTS=${PNPTS:-2000}
PREP=${PREP:-20}
OUT=bench/fma/results
mkdir -p "$OUT"

echo "######## 1. certified error bound (scalar reference vs MPFR 600-bit) ########"
./bench/fma/test_fma_ref 200000 | tee "$OUT/test_fma_ref.txt"

echo
echo "######## 2. backend bitwise agreement (scalar vs NEON vs SVE2) ########"
./bench/fma/test_fma_simd 200000 | tee "$OUT/test_fma_simd.txt"

echo
echo "######## 2b. SVE2 vs NEON kernel audit (bitwise) ########"
./bench/fma/test_sve2_vs_neon 200000 | tee "$OUT/test_sve2_vs_neon.txt"

echo
echo "######## 3. OpenMP thread-safety (serial vs OMP, bitwise) ########"
./bench/fma/test_fma_omp | tee "$OUT/test_fma_omp.txt"

echo
for bk in serial neon sve2 serial_f neon_f sve2_f; do
	:
	echo "######## 4.$bk  AXPY / GEMV / GEMM ########"
	./bench/fma/fma_bench_$bk -naxpy "$NAXPY" -ngemv "$NGEMV" -ngemm "$NGEMM" \
		| tee "$OUT/fma_bench_$bk.txt"
	echo
done
echo "######## 5. polynomial: Horner / Estrin / eval_diff / poly-mul ########"
for bk in serial neon sve2; do
	for v in "" "_bf" "_fma"; do
		echo "---- poly_bench_${bk}${v} ----"
		./bench/fma/poly_bench_${bk}${v} -deg "$PDEG" -npts "$PNPTS" -rep "$PREP" \
			| tee "$OUT/poly_bench_${bk}${v}.txt"
		echo
	done
done

echo "results in $OUT/"
