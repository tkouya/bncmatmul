#!/bin/bash
# Full GPU(gdtq) vs CPU-OMP real-EFT sweep -> bench/cuda/gpu_omp_real.csv
# (dense matvec/matmul + banded SpMV). Config from env.sh.
cd "$(dirname "$0")/../.."
. bench/cuda/env.sh
BAND=${BAND:-15}
OUT=bench/cuda/gpu_omp_real.csv
HDR="op,family,type,bits,N,nnz,gpu_sec,cpu_sec,cpu_backend,speedup,relerr,gpu_mflops,cpu_mflops"
echo "$HDR" > "$OUT"
echo "# OMP_NUM_THREADS=$OMP_NUM_THREADS band=$BAND $(date)" > "$OUT.log"

# dense matvec (reps 3), verify N<=512
bench/cuda/gpu_omp_bench --reps 3 --ops matvec \
    --mv-sizes 512,1024,2048,4096 --verify-cap 512 \
    --out /tmp/_mv.csv 2>>"$OUT.log"
tail -n +2 /tmp/_mv.csv >> "$OUT"

# dense matmul (reps 2), CPU-OMP up to N=2048
bench/cuda/gpu_omp_bench --reps 2 --ops matmul \
    --mm-sizes 256,512,1024,2048 --maxcpu-mm 2048 --verify-cap 512 \
    --out /tmp/_mm.csv 2>>"$OUT.log"
tail -n +2 /tmp/_mm.csv >> "$OUT"

# SpMV per EFT type (reps 2), banded HBW=BAND.  N capped at 262144 because the
# library's OpenMP EFT SpMV recomputes the row prefix-sum per row (O(N^2)).
SP_SIZES=${SP_SIZES:-4096,16384,65536,262144}
for t in dd td qd ds ts qs; do
    bench/cuda/gpu_spmv_$t --reps 2 --band "$BAND" --sizes "$SP_SIZES" \
        --out /tmp/_sp_$t.csv 2>>"$OUT.log"
    tail -n +2 /tmp/_sp_$t.csv >> "$OUT"
    echo "$t done" >> "$OUT.log"
done
echo "DONE rows=$(($(wc -l < "$OUT")-1)) $(date)" | tee -a "$OUT.log"
