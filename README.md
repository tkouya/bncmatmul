# BNCmatmul

Basic Numerical Computation based on optimized multiple-precision MATrix
MULtiplication and other basic linear computation subprograms.

BNCmatmul provides dense/sparse linear computation (matrix multiplication
with triple-loop / block / Strassen / Winograd algorithms, LU decomposition,
Krylov subspace solvers, polynomial evaluation and root finding) over a wide
range of precisions:

- native `double`, `float` and complex counterparts
- multi-component families based on `double`: DD, TD, QD (2-, 3-, 4-fold)
- multi-component families based on `float`: DS, TS, QS
- complex versions of all of the above (CDD ... CQS)
- arbitrary precision via GMP `mpf_t` / MPFR / MPC

All multi-component arithmetic is unified on the `rxx_` layer: `rdd.h`
(DD/TD/QD) and `rds.h` (DS/TS/QS) are the single arithmetic headers, with
SIMD kernels (AVX2, AVX-512, Arm NEON, Armv9-A SVE2) under
`include/avx2/`, `include/neon/` and `include/sve2/`, and optional CUDA
implementations under `src/cuda/`.  Retired historical sources are kept
out of the build in `attic/`.

Source code: <https://github.com/tkouya/bncmatmul> — release tarballs are
provided on the [Releases](https://github.com/tkouya/bncmatmul/releases)
page.

BNCmatmul is licensed under the GNU LGPL v3 (or later).
See the LICENSE file for details.

## Building with Autotools

Requirements: Autoconf 2.69+, Automake 1.11+, Libtool 2.2+, GMP, MPFR,
MPC, Bailey's QD, a C/C++ compiler, GNU make.

```sh
./autogen.sh                # bootstrap (aclocal + automake + autoconf)
./configure                 # detect compilers and libraries
make                        # build libbncmatmul-X.Y.a (generic, recursive)
make check                  # build test programs under test/
sudo make install           # install headers and libraries to $PREFIX
sudo make uninstall         # remove what install installed
```

The Automake/Libtool build produces both static and shared variants of
the generic library:

```
$libdir/libbncmatmul-X.Y.a            # static archive
$libdir/libbncmatmul-X.Y.so.0.0.0     # shared object
$libdir/libbncmatmul-X.Y.so.0         # symlink (soname)
$libdir/libbncmatmul-X.Y.so           # symlink (development)
$libdir/libbncmatmul.la               # libtool archive
```

To build only one variant, pass `--disable-shared` or `--disable-static`
to `./configure`.

## SIMD / OpenMP variant libraries

The per-ISA variant libraries are built by the preserved handwritten
makefiles (`Makefile.legacy`), reachable through top-level aliases:

```sh
make serial                 # generic scalar + OpenMP + Winograd libraries
make neon                   # Arm NEON variants
make sve2                   # Armv9-A SVE2 variants
make avx2                   # AVX2 variants (+ Winograd)
make avx512                 # AVX-512 variants (+ Winograd)
make omp                    # OpenMP-parallel Strassen library only
make winograd               # Winograd libraries only
```

Each target produces the static archives in the top directory
(`libbncmatmul-X.Y[_simd].a`, `libbncmatmul-X.Y-omp[_simd].a`) and the
shared objects in `python/` (`libbncmatmul-X.Y[_simd].so`, used by the
Python bindings).  All variants export the same function names, so the
SIMD backend is chosen purely at link time.

No precision-family compile flags are required: every family is always
compiled and declared (the former `USE_*LINEAR` gates were removed in
0.24).  Only external-dependency and kernel-selection macros remain
(`USE_GMP`, `USE_MPFR`, `BNC_USE_NEW_FMA`, `USE_*_BF`, ...).

## Distribution and documentation

Standard Automake targets (`make dist`, `make distcheck`, `make uninstall`,
`make clean`, `make distclean`) are supported.

```sh
make dist                   # produce bncmatmul-X.Y.tar.gz
make dist-xz                # produce bncmatmul-X.Y.tar.xz
make dist-bzip2             # produce bncmatmul-X.Y.tar.bz2
make distcheck              # verify the tarball builds cleanly
```

The User's Guide (English and Japanese, generated from a single bilingual
LaTeX source) is bundled as `doc/manual.pdf` and `doc/manual_ja.pdf`;
rebuild with `make pdf` inside `doc/`.

A timing-independent numerical regression suite lives in `bench/regress/`:

```sh
sh bench/regress/run_regress.sh out.csv
diff bench/regress/baseline.csv out.csv    # must be identical
```

## Useful `configure` options

- `--prefix=DIR` — install prefix (default: `/usr/local`)
- `--with-gmp=DIR` / `--with-mpfr=DIR` / `--with-mpc=DIR` / `--with-qd=DIR` — non-standard library prefixes
- `--enable-omp` — build the OpenMP-parallel Strassen / `_bncomp_*` routines
- `--enable-avx2` / `--enable-neon` — force a SIMD variant on/off (default: auto from host CPU)
- `--enable-sve2` — build Armv9-A SVE2 variants (Grace / Neoverse V2; default: no)
- `--enable-cuda` — build the CUDA/GPU library (`libbncmm_cuda.a`)
- `--with-mplapack=DIR` — MPLAPACK install prefix (optional, used by some tests)

`configure` generates `bncmatmul.inc` from `bncmatmul.inc.in` and the
top-level `Makefile` from `Makefile.am`; the per-directory handwritten
`src/Makefile`, `test/Makefile`, and `bench/Makefile` include
`bncmatmul.inc` unchanged, so all existing legacy make targets continue
to work via Automake's delegation hooks.

--
Copyright (c) 2023-2026 Tomonori Kouya
