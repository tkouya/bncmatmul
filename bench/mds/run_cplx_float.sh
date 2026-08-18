#!/bin/bash
# Append ONLY the float-based complex types (cf/cds/cts/cqs) to results_cplx.csv.
# Native cd/cdd/ctd/cqd already present (lib/code unchanged) -> not recomputed.
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=/home/tkouya/local/lib:$LD_LIBRARY_PATH
OUT=bench/mds/results_cplx.csv
SIZES="128 256 512 1024 2048"
BACKENDS="avx512 avx2 serial"
TYPES="cf cds cts cqs"
for b in $BACKENDS; do
  bin=bench/mds/mds_bench_cplx_$b
  for n in $SIZES; do
    for t in $TYPES; do
      echo "### cplx-float $b N=$n $t  $(date +%H:%M:%S)" >&2
      timeout 5400 "$bin" "$n" "$b" "$t" >> "$OUT"
    done
  done
done
echo "CPLX-FLOAT DONE $(date +%H:%M:%S)" >&2
wc -l "$OUT" >&2
