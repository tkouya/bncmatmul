#!/bin/bash
# GPU AXPY sweep: 32 total threads (N<=cap32) and GPU-max threads -> CSV
# Writes the CSV directly (per-row flush) so partial results survive a timeout.
cd "$(dirname "$0")/../.."          # repo root
export LD_LIBRARY_PATH=/home/tkouya/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

BIN=bench/cuda/cuda_axpy_bench
OUT=bench/cuda/axpy_gpu.csv
SIZES=${SIZES:-"4096,16384,65536,262144,1048576,4194304"}
REPS=${REPS:-7}
CAP32=${CAP32:-65536}

# library GPU-init banners go to stdout (discarded); progress -> .log ; CSV -> OUT
"$BIN" --reps "$REPS" --sizes "$SIZES" --cap32 "$CAP32" --out "$OUT" \
    >/dev/null 2>"$OUT.log"
echo "GPU AXPY exit=$?  rows=$(($(wc -l < "$OUT")-1))" >&2
