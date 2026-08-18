#!/bin/bash
# Run the complex (副疎演算版) benchmark sweep -> results_cplx.csv
cd "$(dirname "$0")/../.."   # repo root
export LD_LIBRARY_PATH=/home/tkouya/local/lib:$LD_LIBRARY_PATH

OUT=bench/mds/results_cplx.csv
SIZES="128 256 512 1024 2048"
BACKENDS="serial avx2 avx512"
# cd/cdd/ctd/cqd = native complex types; cf/cds/cts/cqs = float-based complex,
# timed via the re/im split over the real float / float-EFT routines (no native type).
TYPES="cd cdd ctd cqd cf cds cts cqs"

echo "backend,type,op,N,seconds,iters,mflops" > "$OUT"
for b in $BACKENDS; do
  bin=bench/mds/mds_bench_cplx_$b
  for n in $SIZES; do
    for t in $TYPES; do
      echo "### cplx $b N=$n $t  $(date +%H:%M:%S)" >&2
      timeout 5400 "$bin" "$n" "$b" "$t" >> "$OUT"
    done
  done
done
echo "CPLX ALL DONE $(date +%H:%M:%S)" >&2
wc -l "$OUT" >&2
