#!/bin/sh
# Run the matmul-algorithm benchmark and collect CSV.
#  - timing:   dims 256 512 1024, all 8 precisions x 3 backends x 4 algorithms
#  - accuracy: dim 256, all 8 precisions x 4 algorithms (serial backend;
#              per-element max relative error vs 512-bit MPFR reference).
cd "$(dirname "$0")"
OUT=out
DIMS="${DIMS:-256 512 1024}"
ACC_DIM="${ACC_DIM:-256}"
PRECS="double dd td qd float ds ts qs"
BACKENDS="serial neon sve2"
CSV=$OUT/mm_results.csv

echo "kind,prec,backend,algo,dim,value" > $CSV

echo "==== timing ===="
for p in $PRECS; do
  for bk in $BACKENDS; do
    bin=mm_${p}_${bk}
    [ -x "$bin" ] || { echo "  (missing $bin)"; continue; }
    for d in $DIMS; do
      echo "  time $p/$bk/$d"
      ./$bin $d time | sed 's/^TIME/time/' >> $CSV
    done
  done
done

echo "==== accuracy (serial, dim=$ACC_DIM) ===="
for p in $PRECS; do
  bin=mm_${p}_serial
  [ -x "$bin" ] || { echo "  (missing $bin)"; continue; }
  echo "  acc  $p/$ACC_DIM"
  ./$bin $ACC_DIM acc | sed 's/^ACC/acc/' >> $CSV
done

echo "==== done -> $CSV ===="
wc -l $CSV
