#!/bin/bash
# AXPY CPU sweep: serial (1 thread) and CPU-max (nproc threads) -> CSV
cd "$(dirname "$0")/../.."          # repo root
export LD_LIBRARY_PATH=/home/tkouya/local/lib:$LD_LIBRARY_PATH

BIN=bench/mds/axpy_bench_avx512
OUT=bench/mds/axpy_cpu.csv
SIZES=${SIZES:-"4096 16384 65536 262144 1048576 4194304"}
MAXT=${MAXT:-$(nproc)}

echo "backend,nthreads,family,type,bits,N,seconds,iters,mflops" > "$OUT"
for nt in 1 "$MAXT"; do
  for n in $SIZES; do
    echo "### nt=$nt N=$n $(date +%H:%M:%S)" >&2
    # keep only the CSV data lines (start with the backend label), drop library banners
    "$BIN" "$n" "$nt" avx512 2>/dev/null | grep -E '^avx512,' >> "$OUT"
  done
done
echo "CPU AXPY DONE $(date +%H:%M:%S)  rows=$(($(wc -l < "$OUT")-1))" >&2
