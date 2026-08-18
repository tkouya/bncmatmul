# LU decomposition + solve FMA+SIMD benchmark (DS/TS/QS/DD/TD/QD)

`*LUdecompPM` (partial pivoting with row swaps) and the scalar `*LUdecomp/P/C`
variants gained a fused-update path: the trailing-submatrix update
`a_jk := a_jk - a_ji * a_ik` is computed as ONE branch-free FMA
`a_jk := fma(-a_ji, a_ik, a_jk)` (arXiv:2607.11391) instead of mul+sub.

Forward/backward substitution (`SolveXXLS/P/PM`) is restructured into
row-oriented dot products computed by a per-precision fused-FMA SIMD kernel
`_bnc_xxsolve_dot` (lane-wise FMA accumulation + horizontal reduction +
scalar tail); `SolveXXLSC` (complete pivoting, permuted access) is scalar
fused-FMA only.

* Switch: compile the LU translation units with **`-DBNC_USE_NEW_FMA`**.
  Without the flag every build is byte-identical to the previous library.
* SIMD: the PM update runs vectorized on AVX2 / AVX-512 / NEON / SVE2 for all
  six precisions. DS/TS/QS previously had NO SIMD LU at all; DS previously had
  no LU at all (`src/dslu.c` is new: decomp/P/C/PM + Solve*).

## Files
* `lu_mr.c`      — bench/verify template (token-pasting, like `bench/arm/dense_mr.c`)
* `build_lu.sh`  — builds 6 precisions x {serial,neon,sve2} x {nofma,fma} = 36 binaries
* `run_lu.sh`    — sweep -> `results/lu_fma_bench.csv`
* `lu_illcond_mr.c` — ill-conditioned accuracy tests (Hilbert / Frank matrices);
  entries generated flag-independently (QD `c_qd_div`), rhs + forward error
  evaluated in QD so errors down to 1e-60 are measurable
* `build_illcond.sh` / `run_illcond.sh` — 36 binaries, n=2..40 sweep ->
  `results/lu_illcond.csv`; error grows as cond_inf(A)*u_prec, breakdown dims
  match nofma/fma within <=2; QS/Frank shows the 812u^4 vs few-u^4
  error-constant gap (up to 2-3 digits) — order and breakdown unchanged
* `gen_excel_lu.py` — CSVs -> `../../lu_fma_bench.xlsx` (data / speedup / chart / illcond sheets)

## Measured speedup (dim=512, Cortex-X925/A725 "GB10", 2026-08-16)
LU decomposition / forward+backward solve (nofma/fma ratio):
| prec | serial LU | NEON LU | SVE2 LU | serial solve | NEON solve | SVE2 solve |
|------|----------:|--------:|--------:|-------------:|-----------:|-----------:|
| DD   | 1.38x | 1.84x | 1.42x | 1.11x | 2.25x | 2.21x |
| TD   | 1.90x | 1.61x | 1.71x | 1.36x | 2.61x | 2.64x |
| QD   | 2.69x | 1.43x | 1.48x | 1.82x | 3.99x | 3.94x |
| DS   | 1.40x | 4.56x | 5.20x | 1.08x | 3.97x | 3.99x |
| TS   | 2.54x | 9.06x | 8.75x | 1.70x | 5.83x | 5.99x |
| QS   | 3.80x |13.53x |12.26x | 2.83x |10.89x |11.24x |

Accuracy is preserved (the fused FMA is slightly MORE accurate than mul+sub;
e.g. DS random-pivoting test: 1.6e-12 vs 8.1e-12 max rel. error).
