#!/bin/sh
# Run ill-conditioned LU/solve accuracy sweep -> bench/lu/results/lu_illcond.csv
cd "$(dirname "$0")"
mkdir -p results
CSV=results/lu_illcond.csv
echo "prec,backend,fma,family,dim,maxrelerr" > $CSV
for fam in hilbert frank; do
  for p in dd td qd ds ts qs; do
    for bk in serial neon sve2; do
      for fm in nofma fma; do
        for d in 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40; do
          ./lu_ill_${p}_${bk}_${fm} $fam $d 2>/dev/null | grep ^ILL | sed 's/^ILL,//' >> $CSV
        done
      done
    done
  done
  echo "family $fam done"
done
echo "CSV: $CSV"
