# mds — multi-precision / multi-backend micro-benchmark harness

Micro-benchmarks for the BNCmatmul dense and sparse kernels across the
real and complex precision types and the three x86 SIMD backends
(serial / AVX2 / AVX-512). The backend is selected at **link time** by the
library the harness is linked against (`libbncmatmul-0.24[_avx2|_avx512].a`).

## Operations
- `axpy`   : `add_cmul` (c = a + α·b)
- `matvec` : A·x
- `matmul` : A·B
- `spmv`   : sparse matrix × vector (banded, half-bandwidth = 3)

## Types
- **Real** : `double, dd, td, qd` (double-based EFT) and `float, ds, ts, qs` (float-based EFT)
- **Complex (native)** : `cd, cdd, ctd, cqd`
- **Complex (float-based)** : `cf, cds, cts, cqs`

  The float-based complex types have **no native implementation** in the
  library. They are timed by decomposing each complex operation over the real
  / imaginary parts using the existing real `float`/`ds`/`ts`/`qs` routines —
  the same re/im split the native `cdd` type uses internally
  (`cddvector = {DDVector re; DDVector im;}`), with the 4-multiply rule
  `(a+bi)(c+di) = (ac − bd) + (ad + bc) i`. FLOP counts (`2N`, `2N²`, `2N³`,
  `2·nnz`) match the native complex types so the MFLOPS columns are comparable.

## Files
| file | purpose |
|------|---------|
| `mds_bench.cc`      | real-type dense+sparse benchmark |
| `mds_bench_cplx.cc` | complex-type dense+sparse benchmark (native + float-based) |
| `mds_verify.cc`     | bit-identity check serial vs AVX2 vs AVX-512 (small structured values) |
| `mds_verify_rand.cc`| same check on pseudo-random operands, with a magnitude column |
| `sp_perf.cc`        | standalone sparse-perf probe (`sp_perf <n> <half-bandwidth>`) |
| `build.sh`          | build all serial/avx2/avx512 binaries |
| `run_x86.sh`        | self-contained serial+avx2+avx512 sweep on one host → `results_x86.csv`, `results_x86_cplx.csv` |
| `run.sh`, `run_avx512.sh` | real-type sweep → `results.csv` (appends; mixes hosts) |
| `run_cplx.sh`       | full complex sweep (all 8 types) → `results_cplx.csv` |
| `run_cplx_float.sh` | append-only sweep of the float-based complex types |
| `make_xlsx.py`      | build `../../mds_gpu_01.xlsx` from the CSV results |
| `results.csv`, `results_cplx.csv` | recorded measurements |

## Usage
```sh
./build.sh                      # build the harness for all three backends
./run_x86.sh                    # all 3 backends, one host -> results_x86*.csv
./run.sh ; ./run_avx512.sh      # real-type sweep   -> results.csv
./run_cplx.sh                   # complex-type sweep -> results_cplx.csv
python3 make_xlsx.py            # -> ../../mds_gpu_01.xlsx
```

`run_x86.sh` writes a fresh CSV per invocation and pins every run to one core
(`BNC_BENCH_CPU`, default 4), so the three backends in a file always come from
the same machine and session; `run.sh`/`run_avx512.sh` append to the shared
`results.csv`, which therefore mixes hosts.

Both `build.sh` and any hand-written compile line must put `-Iinclude` **before**
the GMP/MPFR/QD prefix: a previous `make install` leaves BNCmatmul headers under
that prefix and the stale copies would otherwise shadow the in-tree ones the
libraries were built from.
`make_xlsx.py` expects `xlsxwriter` (install via pip, e.g. into `/tmp/pylibs`).
