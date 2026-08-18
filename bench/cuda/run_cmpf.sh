#!/bin/bash
# Complex mpc_cuda GPU vs CPU-OMP(same MPC arithmetic, OpenMP) precision sweep
#   -> bench/cuda/gpu_omp_cmpf.csv   (PRECS default 106 212 424)
cd "$(dirname "$0")/../.."
. bench/cuda/env.sh
PRECS="${PRECS:-106 212 424}"
OUT=bench/cuda/gpu_omp_cmpf.csv
echo "op,family,type,bits,N,nnz,gpu_sec,cpu_sec,cpu_backend,speedup,relerr,gpu_mflops,cpu_mflops" > "$OUT"
: > "$OUT.log"
for P in $PRECS; do
  bench/cuda/cmpf_bench_$P --reps 2 --out /tmp/_cx_$P.csv 2>>"$OUT.log"
  tail -n +2 /tmp/_cx_$P.csv >> "$OUT"
  echo "PREC=$P done" >>"$OUT.log"
done
echo "DONE rows=$(($(wc -l < "$OUT")-1))" | tee -a "$OUT.log"
