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

To install the variant libraries without any compilation, use the
copy-only target after building them:

```sh
make serial neon sve2            # build the variants you need
sudo make install-libs           # copy all libbncmatmul-X.Y*.a/.so + headers
sudo make uninstall-libs         # remove exactly what install-libs copied
```

`install-libs` copies every variant archive from the top directory and
every shared object from `python/` into `$libdir`, and the full header
set into `$includedir` (same layout as the Automake install); nothing is
compiled or relinked.  The regular `make install` installs the generic
Automake-built library only, and since 0.24 it no longer builds the
benchmark programs (`bench/` builds on `make check`, like `test/`).

`sudo make install` never compiles.  Automake's stock rule is
`install-am: all-am`, so an install run under `sudo` would rebuild
whatever happens to be out of date -- as root -- and leave root-owned
objects, `.libs/` entries and relinked libraries scattered through the
build tree; the next ordinary `make` then dies with `Permission denied`
until the whole tree is chowned back.  `src/Makefile.am` therefore
overrides the rule: it still builds automatically for whoever owns the
build tree (plain `./configure && make install` is unchanged), but when
a rebuild would be performed by a *different* user it stops first and
says so, without compiling anything.  The fix for that message is always
the same:

```sh
make                 # as the owner of the build tree
sudo make install
```

## Ozaki-scheme routines with an optimized BLAS

The Ozaki-scheme routines (`mul_{dd,td,qd,mpf}matrix_oz()` and their
matrix-vector counterparts, in `src/*_oz_scheme.c`) split their
multi-precision operands into error-free `double` slices and multiply the
slices in plain double precision.  That inner product is an ordinary
DGEMM, so it can be delegated to an optimized BLAS.  Intel MKL
(`-DUSE_IMKL`) has always been wired up; OpenBLAS is now supported as
well:

```sh
./configure --with-openblas                 # look for cblas.h / -lopenblas
./configure --with-openblas=/opt/OpenBLAS   # ... under a given prefix
```

`configure` probes the usual `cblas.h` locations (`$PREFIX/include`,
`$PREFIX/include/openblas`, `/usr/include/openblas`, the multiarch
directories) and link-tests `cblas_dgemm`, then compiles the library with
`-DUSE_OPENBLAS`.  Both backends define the internal `BNC_USE_CBLAS`
switch that the Ozaki kernels test, and MKL wins when both are given, so
existing `-DUSE_IMKL` builds are unaffected.

The results are unchanged: the splitting makes each slice product
error-free, so the summation order chosen by the BLAS cannot alter the
answer — only the speed of the inner DGEMM.

This is opt-in: a plain `./configure` keeps the built-in `mul_dmatrix()`
kernel, so the `bench/` Ozaki timings continue to measure BNCmatmul's own
matrix multiplication unless you ask for OpenBLAS.

When OpenBLAS lives outside the loader's search path (`/opt/OpenBLAS`,
say), `configure` also records it as an `-Wl,-rpath`, so the resulting
programs run without `LD_LIBRARY_PATH`.

Two long-standing defects in these routines were fixed while wiring this
up, and both are independent of the backend:

* `mul_ddmatrix_ddvec_oz()` and friends (including the sparse
  `mul_*rsmatrix_*vec_oz()` variants) sized the split copies of the input
  vector from `ret->dim` instead of `vb->dim`.  For a non-square `A` with
  more columns than rows this overran the heap and aborted the process.
* `mul_dmatrix()` (and the `f`/`mpf`/`cd`/`cmpf` counterparts) checked
  `c->col_dim` against `b->row_dim` rather than `b->col_dim`, so every
  non-square product was rejected with `ERROR: mul_dmatrix`.

## Parallel Ozaki-scheme kernels (OpenMP)

`mul_{d,dd,td,qd,mpf}matrix_oz()`, their matrix-vector counterparts and
the complex 3M/4M wrappers built on top of them are parallelized with
OpenMP.  Only the DGEMM used to be threaded (by whatever the BLAS did on
its own), which left the other two thirds of the work serial:

1. **splitting** the operands into error-free `double` slices --
   `O(num_div * n^2)` multi-component operations,
2. the **slice products** -- one DGEMM per slice pair,
3. **accumulating** every slice product back in DD/TD/QD/`mpf_t` --
   `O(num_div^2 * n^2)` multi-component additions.

All three are parallel now.  The rows of the result are cut into blocks;
a thread takes one block at a time, runs every slice product for it with
a *single-threaded* BLAS call and accumulates on the spot, so the block
of the result stays hot in cache across all slice pairs.  The block loop
is dynamically scheduled, which also keeps heterogeneous (big.LITTLE
style) cores busy where an equal split would stall on the slow ones.
The splitting was rewritten around the same idea: two row-parallel passes
per split instead of six sweeps over the matrix, with the threshold `s`
kept as one value per row (or column) instead of a full matrix.

The results are unchanged -- bit for bit.  The row blocks are disjoint
and every element of the result still sums its slice products in the
original order, so the parallel, serial, CBLAS and non-CBLAS paths all
produce identical output.

### Tuning knobs

Everything is read from the environment at run time, so trying another
setting never needs a rebuild:

| variable | default | meaning |
| --- | --- | --- |
| `BNC_OZ_NUM_THREADS` | OpenMP maximum | threads used by the kernels; `1` disables the kernels' own parallelism and hands the DGEMM back to the BLAS |
| `BNC_OZ_BLAS_THREADS` | `1` | threads left to the CBLAS backend while a kernel runs its own parallel loop |
| `BNC_OZ_BLOCK_ROWS` | automatic | rows of the result per block |
| `BNC_OZ_BLOCKS_PER_THREAD` | `2` | blocks per thread when `BNC_OZ_BLOCK_ROWS` is automatic |
| `BNC_OZ_GEMM_MODE` | `own` | `own` = parallel block loop with a serial BLAS, `blas` = serial block loop with a threaded BLAS |

The same settings are reachable from C as `bnc_oz_set_num_threads()`,
`bnc_oz_set_blas_threads()`, `bnc_oz_set_block_rows()` and
`bnc_oz_set_gemm_mode()` (declared in `oz_scheme.h`).

Two settings outside the library matter as much as those above:

* **Bind the threads.**  `OMP_PROC_BIND=close OMP_PLACES=cores` is worth
  10-20% here, and it is *required* if you ever fall back to
  `BNC_OZ_GEMM_MODE=blas`: OpenBLAS's pthread pool inherits the affinity
  mask of the thread that creates it, so with binding on and no OpenMP
  parallelism of our own its workers all end up on a single core.  That
  is what makes a BLAS-threaded Ozaki scheme look catastrophically slow
  (7x slower in the measurement below) rather than merely limited.
* **Do not let the BLAS thread underneath us.**  The kernels take care of
  this themselves -- they call `openblas_set_num_threads()` /
  `mkl_set_num_threads()` down to `BNC_OZ_BLAS_THREADS` for the duration
  of the parallel region and restore the previous value afterwards.

### Measured

20-core Arm (10x Cortex-X925 + 10x Cortex-A725), OpenBLAS 0.3.34,
`max_num_div = 4`, best of three, `A*B` with square operands.  "before"
is the previous code with OpenBLAS threading, "after" is 20 threads with
`OMP_PROC_BIND=close OMP_PLACES=cores`:

| kernel | n | before | after | speedup |
| --- | --- | --- | --- | --- |
| `mul_ddmatrix_oz` | 1000 | 0.152 s | 0.107 s | 1.4x |
| `mul_ddmatrix_oz` | 2000 | 1.098 s | 0.640 s | 1.7x |
| `mul_ddmatrix_oz` | 4000 | 5.197 s | 2.194 s | 2.4x |
| `mul_tdmatrix_oz` | 4000 | 5.705 s | 2.393 s | 2.4x |
| `mul_qdmatrix_oz` | 4000 | 5.115 s | 1.851 s | 2.8x |
| `mul_mpfmatrix_oz` (256 bit) | 800 | 0.355 s | 0.163 s | 2.2x |

At `n = 4000` the remaining time is essentially the DGEMMs themselves,
which run at roughly 670 GFlop/s -- close to what this machine can do --
so there is little left to win without changing the algorithm.

### Exponent range

The slices are plain `double` matrices, but the operands need not fit in the
`double` exponent range: `mpf_t` reaches 2^±2^30, and even a DD matrix may
legitimately hold entries near `DBL_MAX`.  Splitting them where they stand
failed in three ways, none of them announced:

* `mpf_get_d()` returns ±Inf above `DBL_MAX` and 0 below `DBL_MIN`, so
  everything outside the `double` range was destroyed before the split began;
* the threshold `2^(ceil(log2(mu)) + s)` overflowed to Inf once `mu` passed
  about 2^995, and `(x + Inf) - Inf` turns every entry into NaN;
* below about 2^-1024 the threshold underflowed instead, so `(x + s) - s`
  stopped truncating the mantissa and the slice products were quietly no
  longer error-free — wrong answers with no NaN and no error.

Measured on a 32x32 `mpf_t` product at 256 bits with `max_num_div = 8`, before
the fix: 2^990 gave 7.5e-19 (correct), **2^1000 gave all NaN**, 2^-1020 gave
7.9e-12, 2^-1040 gave 4.2e-6 and 2^-2000 gave a zero matrix.  The usable window
was some 2000 binades out of the 2^31 that MPFR offers.

So each row of A and each column of B is now scaled by a power of two before
it is split, and the exponent is carried along with the slice:

```
A[i][k] = 2^sa[p][i] * sum_p slice_a[p][i][k]
B[k][j] = 2^sb[q][j] * sum_q slice_b[q][k][j]
C[i][j] = sum_{p,q} 2^(sa[p][i] + sb[q][j]) * (slice_a[p] slice_b[q])[i][j]
```

The exponent is constant along `k`, so it factors straight out of the inner
product and the error-free property is untouched; scaling by a power of two is
exact in both `double` and `mpf_t`.  A fresh exponent is taken for every slice,
so the dynamic range a single row may span is bounded by `max_num_div` alone.
After the fix the same measurement gives 7.5e-19 at every scale tried, from
2^-10^8 to 2^10^8, and a row spanning 2^±100000 comes out at 3.0e-19.  For the
`double`-based families the NaN at ~1e300 is gone as well.

The scaled split reproduces the old results **bit for bit** wherever the old
one worked (2.25M values across DD/TD/QD/`mpf_t`, square and rectangular,
padded and unpadded), and costs nothing measurable: the row maximum has to be
scanned anyway, and `mpf_get_d_2exp()` hands back the mantissa and the exponent
in one call, so the conversions are cached rather than repeated.

The split functions carry the exponents through new `_ex` entry points --
`split_{d,dd,td,qd,mpf}matrix_dmat_ex()`, `..._t_dmat_ex()` and
`split_{dd,td,qd,mpf}vector_dvec_ex()` -- each taking a `long int` array of
`num_div * dim` exponents.  The old names remain and forward with a NULL array,
which asks for the unscaled split: callers that cannot apply a scale factor
still get exactly what they got before, including its exponent restriction.
The `double`-based families skip the scaling below 2^-1000, where a slice could
no longer be shifted back exactly; `mpf_t` has no such limit.

### Sparse Ozaki-scheme routines

`mul_{d,dd,td,qd}rsmatrix_*vec_oz()` and their transposed counterparts, with
`split_*rsmatrix_drsmat()` / `..._t_drsmat()` underneath, got the same three
treatments as the dense ones, plus one algorithmic fix that dwarfs the rest.

**The splits were quadratic in the dimension.**  A `DRSMatrix` stores its
values row after row with no row-offset array, so `get_drsmatrix_ij()` walks
every row above the one it wants; `absmax_row_drsmatrix()` calls it once per
*column* of the matrix, and the transposed split then called
`set_drsmatrix_ij()` for all `row_dim x col_dim` positions.  A split that
should cost O(nnz) cost O(row_dim^2 x col_dim) instead.  Materializing the row
offsets once (`bnc_oz_sp_row_start()`) makes it one pass over the non-zeros --
and is what lets the rows go to different threads at all.  At
`n = 2000`, 20 non-zeros per row, `max_num_div = 4`, single-threaded:

| | before | after (1 thread) |
| --- | --- | --- |
| `mul_ddrsmatrix_ddvec_oz` | 3.84 s | 0.0043 s |
| `mul_ddrsmatrixt_ddvec_oz` | 7.74 s | 0.0043 s |

**Parallel.**  `A * v` is blocked over the rows of the result, as in the dense
case.  `A^T * v` scatters into the result, so instead each thread takes a range
of rows and accumulates into its own column vector, which is then reduced; the
partial sums are exact -- the split is what makes every slice product exact in
`double`, whatever the summation order -- so the reduction costs nothing in
accuracy and the result stays independent of the thread count.  At
`n = 10^6` with 20 non-zeros per row: `A * v` 2.27 s → 0.93 s, `A^T * v`
2.31 s → 1.34 s at 20 threads (both are memory bound, and `A^T * v` also pays
for the per-thread accumulators).  Small problems are better off serial:
`BNC_OZ_NUM_THREADS=1` wins below roughly 10^5 non-zeros.

**Exponent range.**  Rows (for `A`) and columns (for `A^T`) are scaled by a
power of two exactly as in the dense kernels, so the operands may use the whole
exponent range.  Before, entries around 2^1000 produced a result vector of
NaN; now they come out exact.

**OpenBLAS.**  There is nothing to hand a sparse product to: OpenBLAS ships
dense BLAS and LAPACK only, no sparse BLAS, so the products use the library's
own CSR kernels.  What the OpenBLAS builds do get is the split: it used to take
a `cblas_daxpy` path only under `-DUSE_IMKL` and the slower
`add_drsmatrix()`/`sub_drsmatrix()` path otherwise, and now takes the same
single fused pass either way, which beats both.

The MKL inspector-executor path that used to sit inside these four routines was
dropped rather than carried over.  It could not have worked:
`convert_indeces_ddrsmatrix_mkl_csrmat()` assigns to its `MKL_INT *` parameters
*by value*, so the index arrays never reached the caller, which then passed the
uninitialized pointers to `mkl_sparse_d_mv()` and `free()`.  (The older
`mkl_cspblas_dcsrgemv` path behind `-DUSE_IMKL_OLD` was consistent, and
`mul_drsmatrix{,t}_dvec()` still call MKL when built with it.)

The rewritten routines reproduce the old results **bit for bit** wherever the
old ones worked, at every thread count.

#### mpf_t sparse operands

`sparse_mpf.c` had no Ozaki-scheme routines at all, so the family is now
complete: `split_mpfrsmatrix_drsmat{,_ex}()`, `split_mpfrsmatrix_t_drsmat{,_ex}()`,
`mul_mpfrsmatrix_mpfvec_oz()` and `mul_mpfrsmatrixt_mpfvec_oz()`, with the same
blocking, the same tuning knobs and the same exponent handling as the other
four.  Two details differ because `mpf_t` is not built on `double`: an
`MPFRSMatrix` carries no per-row SIMD padding, so its values are indexed by the
running sum of `nzero_col_dim[]` while the `double` slices use
`real_nzero_col_dim[]` (the two row-offset arrays are not the same), and nothing
is converted before it has been scaled.  Measured against an exact 256-bit
product of the same matrix in dense form, `A * v` and `A^T * v` come out
**exact** at every scale from 2^-10^8 to 2^10^8.

#### Sparse matrix times dense matrix

`mul_{d,dd,td,qd,mpf}rsmatrix_*mat_oz()` are new: `C = A * B` with `A` sparse
and `B`, `C` dense.  `A` is split by rows and `B` by columns, exactly as in
`mul_*matrix_oz()`, so the row exponent of `A` and the column exponent of `B`
factor out of the inner product and the slice products stay error free.  The
rows of `C` are cut into blocks and one thread runs every slice pair for its
block, so nothing is shared between threads and the result does not depend on
the thread count.  At `n = 4000`, 20 non-zeros per row, a dense 4000x4000 `B`
and `max_num_div = 4`: 3.86 s → 0.43 s on 20 threads (9.0x).

Like the dense matrix multiplication -- and unlike the sparse matrix-vector
routines -- these skip slice pairs with `p + q >= real_num_div_b`, whose
contribution is below the accuracy the split was asked for.  That is worth
knowing when comparing them: on operands that are exactly representable in
`double`, `mul_mpfrsmatrix_mpfvec_oz()` reproduces the exact product bit for
bit, while `mul_mpfrsmatrix_mpfmat_oz()` stops at about 4e-18 relative for the
same reason `mul_mpfmatrix_oz()` stops at about 3e-16.  Raising `max_num_div`
does not remove it; dropping the pruning would (at twice the products).

### The `_bncomp_*` wrappers

`_bncomp_mul_{d,dd,td,qd,mpf}matrix_oz()` and
`_bncomp_mul_mpfmatrix_mpfvec_oz()` in `src/bncomp_linear_*.c` now all
delegate to the kernels above, so every precision runs the same
algorithm and answers to the same tuning knobs.  They used to
parallelize the outer slice loop only and push every accumulation
through an `omp critical`, which both serialized the expensive half and
made the sum order depend on the thread interleaving; delegating makes
them reproducible and, at `n = 1000` in DD, about 70x faster (4.73 s ->
0.068 s -- they also never used the CBLAS backend, testing `USE_IMKL`
directly instead of `BNC_USE_CBLAS`).

Two things had to change to get the whole family lined up:

* **`double`.**  There was no `mul_dmatrix_oz()` at all; the Ozaki
  product for plain `double` existed only as `_bncomp_mul_dmatrix_oz()`,
  which ignored `max_num_div_a` / `max_num_div_b` and always split its
  operands in two.  `mul_dmatrix_oz()` now sits in `src/oz_scheme.c`
  next to the other kernels, and the wrapper delegates to it, so the
  argument means what it says.  That does change the numbers: against a
  DD reference product of a 200x200 random matrix the old body gave
  5.5e-9 whatever it was passed, while the new one gives 1.1e-8 at
  `max_num_div = 2` and 1.6e-15 from 3 slices on.  It is also 125x
  faster at `n = 2000` (12.68 s -> 0.101 s).  Pass at least 3 to get the
  accuracy the scheme is meant to deliver; `max_num_div = 1` keeps a
  single slice and is as inaccurate as it sounds, exactly as in the
  other precisions.
* **`ds` / `ts` / `qs`.**  `bncomp.h` declared
  `_bncomp_mul_{ds,ts,qs}matrix_oz()`, but the definitions have long
  been `#if 0`'d out in `bncomp_linear_{ds,ts,qs}.c`: the Ozaki scheme
  was never implemented for the `float`-based layer (there is no
  `ds_oz_scheme.c`, hence no `split_dsmatrix_dmat()` and friends).
  Calling one produced a link error rather than a compile error, so the
  declarations are now guarded the same way, with the reason.

One more long-standing defect turned up while rewriting the splits, in the
conversion the old split path went through:

* `subst_dmatrix_tdmat()`, `subst_dmatrix_qdmat()` (the copy in
  `qdlinear.c`), `subst_fmatrix_dsmat()` and `subst_fmatrix_tsmat()`
  computed the destination index as `i * col_dim + j` instead of
  `i * real_col_dim + j`.  Every accessor in the library uses the padded
  stride, so whenever `col_dim` was not a multiple of the padding unit
  these functions scattered the values across the wrong positions -- for
  a 5x5 TD matrix in a NEON build (`_BNC_D_WIDTH == 2`, so
  `real_col_dim == 6`), 20 of the 25 entries landed in the wrong place.
  The DD counterpart had been fixed in 2022; these four were missed.  The
  generic scalar build has `_BNC_D_WIDTH == 1` and was never affected,
  which is why it stayed hidden; the SIMD variant libraries need a
  rebuild (`make neon`, `make sve2`, `make avx2`, `make avx512`) to pick
  the fix up.

Note that the parallel path is compiled in only when the library is built
with OpenMP, i.e. `./configure --enable-omp`.  Without it the same
sources compile to exactly the code described above minus the `#pragma
omp` lines, and behave as before.

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
- `--with-openblas[=DIR]` — use OpenBLAS `cblas_dgemm`/`cblas_dgemv` in the Ozaki-scheme routines (default: no)
- `--with-mplapack=DIR` — MPLAPACK install prefix (optional, used by some tests)

`configure` generates `bncmatmul.inc` from `bncmatmul.inc.in` and the
top-level `Makefile` from `Makefile.am`; the per-directory handwritten
`src/Makefile`, `test/Makefile`, and `bench/Makefile` include
`bncmatmul.inc` unchanged, so all existing legacy make targets continue
to work via Automake's delegation hooks.

--
Copyright (c) 2023-2026 Tomonori Kouya
