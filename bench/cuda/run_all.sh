#!/bin/bash
# run_all.sh -- one-shot: build + run + report for the GPU vs CPU-OMP benchmark.
# Usage (from anywhere):   bench/cuda/run_all.sh
# Override paths/arch via env, e.g.:  PREFIX=/opt/local ARCH=sm_80 bench/cuda/run_all.sh
set -e
cd "$(dirname "$0")/../.."
. bench/cuda/env.sh
echo "=== config: PREFIX=$PREFIX CUDA_HOME=$CUDA_HOME ARCH=$ARCH OMP_NUM_THREADS=$OMP_NUM_THREADS ==="
[ -f "$OMPLIB" ] || { echo "ERROR: $OMPLIB not found. Build it first: make avx512 (+ OMP target)."; exit 1; }
[ -f "$SERLIB" ] || { echo "ERROR: $SERLIB not found."; exit 1; }

echo "=== [1/5] build real-EFT binaries ==="
bash bench/cuda/build_gpu_omp.sh
echo "=== [2/5] build complex mpc_cuda binaries ==="
if [ -f "$MPCLIB" ]; then bash bench/cuda/build_cmpf.sh 106 212 424; else
  echo "WARN: $MPCLIB missing -> skipping complex (mpc_cuda) part."; fi
echo "=== [3/5] run real-EFT sweep (matvec/matmul/SpMV) ==="
bash bench/cuda/run_gpu_omp.sh
echo "=== [4/5] run complex sweep ==="
[ -x bench/cuda/cmpf_bench_106 ] && bash bench/cuda/run_cmpf.sh || echo "skip complex run"
echo "=== [5/5] build report (xlsx + pdf) ==="
# If xlsxwriter/numpy/matplotlib live in a non-standard dir, set PYLIBS=/path.
[ -n "${PYLIBS:-}" ] && export PYTHONPATH="$PYLIBS:${PYTHONPATH:-}"
SYS="device: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) ($ARCH) / OpenMP ${OMP_NUM_THREADS} threads / $(date +%Y-%m)"
python3 bench/cuda/make_report.py "$SYS" || {
  echo "report needs python3 + xlsxwriter,numpy,matplotlib (pip install --user xlsxwriter numpy matplotlib)"; exit 1; }
echo "=== DONE: bench/cuda/gpu_report.{xlsx,pdf} ==="
