#!/bin/bash
# run_x86.sh : serial / AVX2 / AVX-512 sweep on one x86-64 host.
#
# Unlike run.sh + run_avx512.sh (which append to the shared results.csv), this
# script writes one self-contained CSV per host so the three backends in it are
# always from the same machine, the same libraries and the same session.
#
#   usage: bench/mds/run_x86.sh [outfile] [sizes...]
#   default outfile: bench/mds/results_x86.csv
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=/home/tkouya/local/lib:$LD_LIBRARY_PATH

OUT=${1:-bench/mds/results_x86.csv}
shift 2>/dev/null
SIZES=${*:-"128 256 512 1024"}

# pin to one core so the three backends see the same frequency/cache conditions
CPU=${BNC_BENCH_CPU:-4}
RUN="taskset -c $CPU"

echo "backend,type,op,N,seconds,iters,mflops" > "$OUT"
for b in serial avx2 avx512; do
  for n in $SIZES; do
    echo "### real $b N=$n $(date +%H:%M:%S)" >&2
    $RUN bench/mds/mds_bench_$b "$n" "$b" >> "$OUT"
  done
done

OUTC=${OUT%.csv}_cplx.csv
echo "backend,type,op,N,seconds,iters,mflops" > "$OUTC"
for b in serial avx2 avx512; do
  for n in $SIZES; do
    for t in cd cdd ctd cqd; do
      echo "### cplx $b N=$n $t $(date +%H:%M:%S)" >&2
      $RUN bench/mds/mds_bench_cplx_$b "$n" "$b" "$t" >> "$OUTC"
    done
  done
done
echo "DONE $(date +%H:%M:%S)" >&2
