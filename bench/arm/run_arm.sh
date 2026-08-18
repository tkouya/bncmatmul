#!/bin/sh
# Run all arm_bench benchmarks, collect CSV results.
cd "$(dirname "$0")/../.."   # repo root
OUT=bench/arm/out
mkdir -p $OUT
DENSE_CSV=$OUT/dense_results.csv
SPMV_CSV=$OUT/spmv_results.csv
: > $DENSE_CSV
: > $SPMV_CSV
echo "kind,prec,backend,dim,axpy,matvec,matmul" >> $DENSE_CSV
echo "kind,prec,backend,matrix,nnz,time" >> $SPMV_CSV

DIMS="128 256 512 1024 2048"
PRECS="d f dd td qd ds ts qs cd cf cdd ctd cqd cds cts cqs"
MTX=bench/memplus/memplus.mtx

echo "######## DENSE ########"
for p in $PRECS; do
  for bk in serial neon sve2; do
    bin=bench/arm/dense_${p}_${bk}
    [ -x "$bin" ] || { echo "missing $bin"; continue; }
    for n in $DIMS; do
      line=$("$bin" $n 2>/dev/null | grep '^RESULT,')
      if [ -n "$line" ]; then
        echo "dense,$(echo "$line" | cut -d, -f2-)" >> $DENSE_CSV
        echo "  $line"
      else
        echo "  [no output] $bin $n"
      fi
    done
  done
done

echo "######## REAL SpMV (spmvbench: d dd td qd) ########"
for bk in scalar neon sve2; do
  bin=bench/spmvbench_${bk}
  [ -x "$bin" ] || { echo "missing $bin"; continue; }
  csv=$("$bin" $MTX 30 2>/dev/null | grep '^CSV,')
  # CSV,backend,threads,nrow,nnz,t_d,t_dd,t_td,t_qd
  bkn=$bk; [ "$bk" = scalar ] && bkn=serial
  nnz=$(echo "$csv" | cut -d, -f5)
  td=$(echo "$csv" | cut -d, -f6); tdd=$(echo "$csv" | cut -d, -f7)
  ttd=$(echo "$csv" | cut -d, -f8); tqd=$(echo "$csv" | cut -d, -f9)
  echo "spmv,d,$bkn,memplus.mtx,$nnz,$td"   >> $SPMV_CSV
  echo "spmv,dd,$bkn,memplus.mtx,$nnz,$tdd" >> $SPMV_CSV
  echo "spmv,td,$bkn,memplus.mtx,$nnz,$ttd" >> $SPMV_CSV
  echo "spmv,qd,$bkn,memplus.mtx,$nnz,$tqd" >> $SPMV_CSV
  echo "  $csv"
done

echo "######## FLOAT-BASED REAL SpMV (spmvbench_float: f ds ts qs) ########"
for bk in scalar neon sve2; do
  bin=bench/spmvbench_float_${bk}
  [ -x "$bin" ] || { echo "missing $bin"; continue; }
  csv=$("$bin" $MTX 30 2>/dev/null | grep '^CSV,')
  # CSV,backend,threads,nrow,nnz,t_f,t_ds,t_ts,t_qs
  bkn=$bk; [ "$bk" = scalar ] && bkn=serial
  nnz=$(echo "$csv" | cut -d, -f5)
  tf=$(echo "$csv" | cut -d, -f6);  tds=$(echo "$csv" | cut -d, -f7)
  tts=$(echo "$csv" | cut -d, -f8); tqs=$(echo "$csv" | cut -d, -f9)
  echo "spmv,f,$bkn,memplus.mtx,$nnz,$tf"   >> $SPMV_CSV
  echo "spmv,ds,$bkn,memplus.mtx,$nnz,$tds" >> $SPMV_CSV
  echo "spmv,ts,$bkn,memplus.mtx,$nnz,$tts" >> $SPMV_CSV
  echo "spmv,qs,$bkn,memplus.mtx,$nnz,$tqs" >> $SPMV_CSV
  echo "  $csv"
done

echo "######## COMPLEX FLOAT-BASED SpMV (spmvbench_cfloat: cds cts cqs) ########"
for bk in scalar neon sve2; do
  bin=bench/spmvbench_cfloat_${bk}
  [ -x "$bin" ] || { echo "missing $bin"; continue; }
  csv=$("$bin" $MTX 30 2>/dev/null | grep '^CSV,')
  # CSV,backend,threads,nrow,nnz,t_cds,t_cts,t_cqs
  bkn=$bk; [ "$bk" = scalar ] && bkn=serial
  nnz=$(echo "$csv" | cut -d, -f5)
  tcds=$(echo "$csv" | cut -d, -f6); tcts=$(echo "$csv" | cut -d, -f7); tcqs=$(echo "$csv" | cut -d, -f8)
  echo "spmv,cds,$bkn,memplus.mtx,$nnz,$tcds" >> $SPMV_CSV
  echo "spmv,cts,$bkn,memplus.mtx,$nnz,$tcts" >> $SPMV_CSV
  echo "spmv,cqs,$bkn,memplus.mtx,$nnz,$tcqs" >> $SPMV_CSV
  echo "  $csv"
done

echo "######## COMPLEX SpMV (cd cf cdd ctd cqd) ########"
for p in cd cf cdd ctd cqd; do
  for bk in serial neon sve2; do
    bin=bench/arm/spmv_${p}_${bk}
    [ -x "$bin" ] || { echo "missing $bin"; continue; }
    line=$("$bin" $MTX 2>/dev/null | grep '^RESULT_SPMV,')
    # RESULT_SPMV,prec,backend,matrix,nnz,time,mflop
    if [ -n "$line" ]; then
      pr=$(echo "$line"|cut -d, -f2); b=$(echo "$line"|cut -d, -f3)
      m=$(echo "$line"|cut -d, -f4); nz=$(echo "$line"|cut -d, -f5); t=$(echo "$line"|cut -d, -f6)
      echo "spmv,$pr,$b,$m,$nz,$t" >> $SPMV_CSV
      echo "  $line"
    fi
  done
done

echo "######## DONE ########"
wc -l $DENSE_CSV $SPMV_CSV
