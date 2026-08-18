#!/bin/bash
# Run the mds benchmark sweep: backends x sizes -> results.csv
cd "$(dirname "$0")/../.."   # repo root
export LD_LIBRARY_PATH=/home/tkouya/local/lib:$LD_LIBRARY_PATH

OUT=bench/mds/results.csv
SIZES="128 256 512 1024 2048"
BACKENDS="serial avx2 avx512"

echo "backend,type,op,N,seconds,iters,mflops" > "$OUT"
for b in $BACKENDS; do
  bin=bench/mds/mds_bench_$b
  for n in $SIZES; do
    echo "### $b N=$n  $(date +%H:%M:%S)" >&2
    timeout 1800 "$bin" "$n" "$b" >> "$OUT"
  done
done
echo "ALL DONE $(date +%H:%M:%S)" >&2
wc -l "$OUT" >&2
