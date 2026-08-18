#!/bin/bash
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=/home/tkouya/local/lib:$LD_LIBRARY_PATH
SIZES="128 256 512 1024 2048"
for n in $SIZES; do
  echo "### re-avx512 real N=$n $(date +%H:%M:%S)" >&2
  timeout 1800 bench/mds/mds_bench_avx512 "$n" avx512 >> bench/mds/results.csv
done
for n in $SIZES; do for t in cd cdd ctd cqd; do
  echo "### re-avx512 cplx N=$n $t $(date +%H:%M:%S)" >&2
  timeout 1800 bench/mds/mds_bench_cplx_avx512 "$n" avx512 "$t" >> bench/mds/results_cplx.csv
done; done
echo "RE-AVX512 DONE $(date +%H:%M:%S)" >&2
