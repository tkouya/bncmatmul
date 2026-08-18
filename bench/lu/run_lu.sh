#!/bin/sh
# Run LU FMA benchmark sweep -> bench/lu/results/lu_fma_bench.csv
cd "$(dirname "$0")"
mkdir -p results
CSV=results/lu_fma_bench.csv
echo "prec,backend,fma,dim,lu_time,solve_time,maxrelerr" > $CSV
for p in dd td qd ds ts qs; do
  case "$p" in
    dd|ds) dims="256 512 1024";;
    *)     dims="256 512";;
  esac
  for bk in serial neon sve2; do
    for fm in nofma fma; do
      for d in $dims; do
        ./lu_${p}_${bk}_${fm} $d 2>/dev/null | grep ^RESULT | sed 's/^RESULT,//' >> $CSV
        echo "done: $p $bk $fm $d"
      done
    done
  done
done
echo "CSV: $CSV"
