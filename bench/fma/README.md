# Branch-free DW/TW/QW FMA in BNCmatmul

Implementation of the branch-free fused multiply-add (FMA) of

> T. Kouya, *Performance evaluation of branch-free fused multiply-add
> algorithms for multi-component-type multiple-precision floating-point
> arithmetic*, arXiv:2607.11391v1 (2026).

`z := x*y + c` is computed in **one fused stage** instead of `mul` + `add`:

| K | type            | flops (proposed) | flops (BF mul+add) | certified error bound          |
|---|-----------------|------------------|--------------------|--------------------------------|
| 2 | DW = DD / DS    | **17**           | 29                 | `34 u^2 (\|xy\|+\|c\|)`        |
| 3 | TW = TD / TS    | **66**           | 96                 | `184 u^3 (\|xy\|+\|c\|)`       |
| 4 | QW = QD / QS    | **146**          | 209                | `812 u^4 (\|xy\|+\|c\|)`       |

## Files

| file | contents |
|------|----------|
| `include/bncfma_d.h` | scalar `bnc_dwfma` / `bnc_twfma` / `bnc_qwfma` (double base: DD/TD/QD) |
| `include/bncfma_f.h` | scalar `bnc_dwfmaf` / `bnc_twfmaf` / `bnc_qwfmaf` (single base: DS/TS/QS) |
| `include/bncfma.h` | umbrella header including both |
| `include/neon/_bncneon_fma.h` | NEON `_bncneon_{dw,tw,qw}fma[f]` (`float64x2_t` / `float32x4_t`) |
| `include/sve2/_bncsve2_fma.h` | SVE2 `_bncsve2_{dw,tw,qw}fma[f]` (sizeless per-limb convention) |

The routines are operation-by-operation ports of the reference implementation
`fma_ref.c` in the paper's appendix, so they match the FPANVerifier-certified
netlists `dwfma_f2s.fpan` / `twfma_fix2.fpan` / `qwfma_fix3.fpan`.

**`-ffp-contract=off` is mandatory** (as everywhere in BNCmatmul): if the
compiler contracts `a*b+c` into an fma on its own, `two_sum` / `two_prod` stop
being error-free transformations.

## Using it inside the library

Define **`BNC_USE_NEW_FMA`** at library build time. It switches

* `rdd_fma` / `rtd_fma` / `rqd_fma` (`include/rdd.h`) and
  `rds_fma` / `rts_fma` / `rqs_fma` (`include/rds.h`) to the proposed FMA, and
* 62 `mul`+`add` call sites in `src/{dd,td,qd}linear.c` — AXPY
  (`add_cmul_*vector`), dot product (`ip_*vector`), GEMV (`mul_*matrix_*vec`,
  `mul_*matrixt_*vec`), GEMM (`mul_*matrix`) and the Frobenius norm — for the
  serial, NEON and SVE2 paths, and
* 32 sites in `src/{dd,td,qd}_poly.c` — Horner (`eval_*poly_horner`), Estrin
  (`eval_*poly_estrin`, `_bncavx2_eval_*poly_estrin`), the derivative
  (`eval_diff_*poly`) and polynomial multiplication (`mul_*poly`).
  AVX2 / AVX-512 sites are deliberately left untouched (see Notes).

Without the macro the library is bit-for-bit unchanged.

## Tests and benchmark

```sh
./bench/fma/build_fma.sh     # builds 19 binaries (serial / NEON / SVE2, double + single base)
./bench/fma/run_fma.sh       # runs everything, results land in bench/fma/results/
```

| binary | what it checks |
|--------|----------------|
| `test_fma_ref` | certified error bounds against MPFR 600-bit, plus commutativity `fma(x,y,c) == fma(y,x,c)` bitwise |
| `test_fma_simd` | scalar vs NEON vs SVE2 must agree **bitwise** (vectorization must not change the rounding) |
| `test_sve2_vs_neon` | audit: every SVE2 add/mul kernel must be **bitwise** identical to its NEON twin (they implement the same netlist) |
| `test_fma_omp` | OpenMP-parallel evaluation must equal the serial one **bitwise** (the routines are pure — no static temporaries) |
| `fma_bench_{serial,neon,sve2}` | AXPY (n=10^6), GEMV (n=2048), GEMM (n=512) for **DD/TD/QD** x {Q, BF, FMA}: relative error vs MPFR 600-bit, time per call at 1 thread and at `OMP_NUM_THREADS` threads, and the speedups FMA/Q and FMA/BF |
| `fma_bench_{serial,neon,sve2}_f` | the same for the single-word base **DS/TS/QS** (`-DUSE_FLOAT_BASE`) |
| `poly_bench_{serial,neon,sve2}[_bf\|_fma]` | polynomial kernels (Horner scalar+batch-SIMD / Estrin scalar+SIMD / eval_diff / poly-mul) for DD/TD/QD, built three times — **Q** (default), **BF** (`-DUSE_DD_BF -DUSE_TD_BF -DUSE_QD_BF`) and **FMA** (`-DBNC_USE_NEW_FMA`) |

Variants compared by the benchmark:

* **Q** — the library default `mul` + `add` (DD sloppy, TD sloppy, QD Bailey)
* **BF** — the branch-free `mul_bf` + `add_bf` (Zhang–Aiken); **the fair baseline**
* **FMA** — the proposed fused FMA

For the polynomial benchmark the BF variant is selected with the library's own
`USE_DD_BF` / `USE_TD_BF` / `USE_QD_BF` switches, which redirect `r*_mul` /
`r*_add` (scalar, NEON and SVE2) to the `_bf` routines. `r{dd,td,qd}_fma` —
used by `mul_*poly` — now honours the same switches, so the BF column is
coherent across all kernels. The default build is unaffected.

`sweep_deg.sh` runs the polynomial benchmark over degrees 2…1000 for Q/BF/FMA
and writes `results/poly_sweep.csv` (the `PolyDegree` sheet of the workbook).
`poly_bench` accepts `-accpts N` / `-noacc` (limit or skip the MPFR reference)
and `-ntimed N` (timing is the best of N runs after a warm-up).

`HORNER-V` is a batch Horner written inside `poly_bench.c`: it evaluates `LANES`
points at once (SIMD across evaluation points).  The library itself only ships
the AVX2 form `_bncavx2_eval_*poly_horner`, so the NEON/SVE2 counterparts live
in the benchmark.  Note it is only applicable when several evaluation points are
available at the same time.

`gen_excel_fma.py` turns `bench/fma/results/` into **`fma_bench.xlsx`** at the
repository root (sheets: ReadMe / Verification / Accuracy / Timing / Speedup / Raw).

`bench/fma/test_lib_fma.c` is a separate smoke test of the in-library wiring:
build it twice (with and without `-DBNC_USE_NEW_FMA`) and compare — every
type x kernel x backend combination must change by `O(u^K)` and no more.

## Notes

* `src/prefma_backup/` holds the pre-wiring copies of the three `*linear.c`.
* While wiring this up, function-local `static` temporaries were removed from
  `rds_fma` / `rts_fma` / `rqs_fma` and `r{ds,ts,qs}_{sqrt_f,pow}` in
  `include/rds.h`; they were a latent data race under OpenMP (pre-existing,
  unrelated to the FMA itself). `include/rsd.h` still has the same pattern but
  is dead code — it shares the include guard `__BNC_RDS_H_` with `rds.h`.
* **Bug found and fixed while benchmarking (2026-07-28)**: `_bncsve2_rqs_add_sloppy`
  in `include/sve2/_bncsve2_qs.h` added `t2` a second time in the final
  `t0 = t0 + t1 + t3` step of Bailey's sloppy QS addition — `t2` had already been
  consumed by the preceding `fthree_sum2` gate. SVE2 QS additions lost about ten
  digits (max rel. err 1.3e-18 instead of 1.6e-30; QS GEMV 8.1e-17 → 6.2e-25).
  The double-word twin `_bncsve2_rqd_add_sloppy`, the NEON version and the scalar
  `c_qs_add` were all correct — only the float SVE2 transcription was wrong.
  `test_sve2_vs_neon` now reports `qs add_sloppy` as bitwise identical to NEON.
* The `*_mul_sloppy` rows of `test_sve2_vs_neon` still differ between the backends:
  SVE2 uses a fused `svmla` for the cross terms where NEON uses a separate
  `vmulq` + `vaddq`. This is benign — SVE2 is equally or slightly more accurate
  (DD: max rel. err 4.4e-32 vs 5.0e-32).
* `c_ds_qs.h` has **no scalar `_bf` routines**, so the BF column is `n/a` for
  DS/TS/QS on the serial backend (NEON and SVE2 do have float `_bf`).
* AVX2 / AVX-512 versions of the proposed FMA are **not** included; the paper
  covers them, but they could not be tested on this AArch64 machine.
