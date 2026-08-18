# BNCmatmul — SIMD backend benchmark (dd/td/qd and ds/ts/qs precisions)

This directory builds benchmark programs that time the multi-precision matrix
multiply and compare the runtime / verify the result against an MPFR reference,
across SIMD backends:

- **double-based**: `mul_ddmatrix` / `mul_tdmatrix` / `mul_qdmatrix`
  (double-double ≈ 106 bit / triple-double ≈ 159 bit / quad-double ≈ 212 bit) —
  programs `bench_ddlinear` / `bench_tdlinear` / `bench_qdlinear`.
- **float-based**: `mul_dsmatrix` / `mul_tsmatrix` / `mul_qsmatrix`
  (double-single ≈ 48 bit / triple-single ≈ 72 bit / quad-single ≈ 96 bit) —
  programs `bench_dslinear` / `bench_tslinear` / `bench_qslinear`.

Each program loops dims 128..8192, prints the Frobenius norm of C computed both
with the multi-precision SIMD path and with an MPFR reference (so you can eyeball
the agreement), and prints `mpfr(prec): <secs>` and `c_xx: <secs>` timings.

## SIMD backends

| backend  | flag macros                  | typical CPU                         |
|----------|------------------------------|-------------------------------------|
| generic  | (none)                       | any                                 |
| NEON     | `-DBNC_ENABLE_NEON`          | Arm AArch64 (Cortex-A76, …)         |
| SVE2     | `-DBNC_ENABLE_SVE2` (+NEON)  | Armv9-A Grace / Neoverse V2 (DGX Spark) |
| AVX2     | (x86 `-mavx2 -mfma`)         | x86-64 Haswell+                     |
| AVX-512  | (x86 `-mavx512f` …)          | x86-64 Skylake-X / Sapphire Rapids  |

The SVE2 build also defines `BNC_ENABLE_NEON` so any routine that does not yet
have a hand-written SVE2 path falls back to the NEON path (the source uses
`#if … #elif __ARM_SVE2 … #elif __ARM_NEON … #else` ordering, so the
SVE2 block wins whenever the compiler defines `__ARM_SVE2`).

## Build

From the repository root:

`bench_mp` builds **all six** bench programs (dd/td/qd + ds/ts/qs) for one
flavor; `bench_dtq` builds just the double-based three, `bench_dtqs` just the
float-based three.  The `_avx2` / `_avx512` / `_neon` / `_sve2` suffixes select
the library; `bench_mp_all` builds every variant that has a matching library.

### x86 (AVX2 / AVX-512)
```sh
make -C src   -f Makefile.legacy avx2 avx512        # builds libbncmatmul-0.24_avx2.a, _avx512.a
make -C bench -f Makefile.legacy bench_mp bench_mp_avx2 bench_mp_avx512
```

### Arm — NVIDIA DGX Spark / Grace (NEON + SVE2)
```sh
make -C src   -f Makefile.legacy neon sve2          # builds libbncmatmul-0.24_neon.a, _sve2.a
make -C bench -f Makefile.legacy bench_mp bench_mp_neon bench_mp_sve2
```

> Note: `bncmatmul.inc` currently sets `CC_SVE2 = gcc … -mcpu=neoverse-v2 -DBNC_ENABLE_NEON`
> (the `_sve2` library is a Neoverse-V2-tuned NEON build; the hand-written SVE2
> intrinsic headers in `include/sve2/` are not yet usable — see the note in
> `bncmatmul.inc`).  On a different SVE2 part, change `-mcpu=neoverse-v2` to
> e.g. `-mcpu=neoverse-n2` etc.

### everything (where applicable)
```sh
make -C src   -f Makefile.legacy generic            # scalar lib
make -C bench -f Makefile.legacy bench_mp_all       # all SIMD bench variants that have libs
```

## Run the comparison

```sh
cd bench
./bench_simd_compare.sh 1024      # default DIM = 1024
```

It detects which `bench_{dd,td,qd,ds,ts,qs}linear[_neon|_sve2|_avx2|_avx512]`
binaries exist, runs them, and prints a table of elapsed seconds for the
`<DIM> x <DIM>` matmul.  Binaries that were not built show `-`.

You can also run a single bench directly:
```sh
./bench_ddlinear     128          # loops dims 128..8192, prints MPF vs DD ||C||_F and timings
./bench_dslinear_sve2 128         # SVE2-target (Grace-tuned) build of the DS bench
```

## What the numbers mean

Each `bench_<prec>linear` program, for every matrix size, prints:
- `MPF: ||C||_F = …`  — reference Frobenius norm computed at GMP/MPFR precision
- ` DD:|TD:|QD:|DS:|TS:|QS: ||C||_F = …` — same norm computed with the multi-precision SIMD path
- `mpfr( … ): <secs> s` — time for the MPFR-precision matmul
- `c_dd:|c_td:|c_qd:|c_ds:|c_ts:|c_qs: <secs> s` — time for the multi-precision SIMD matmul (this is the value `bench_simd_compare.sh` tabulates)

Agreement between `MPF` and the multi-precision norm should be roughly:
DD ~31 digits, TD ~47, QD ~62 (double-based); DS ~14, TS ~21, QS ~28 (float-based).
