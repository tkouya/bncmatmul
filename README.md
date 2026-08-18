# BNCmatmul
Basic Numerical Computation based on optimized multiple-precision MATrix MULtiplication and other basic linear computation subprograms

All sources are included in bncmatmul-0.xx.tar.bz2. Please download it to your home directory on Linux box, and compile it.

BNCmatmul library is licensed under the GNU LGPL v2.1.
See LICENSE file for details.

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

For the OpenMP, AVX2, AVX-512, NEON and Winograd variants, the original
handwritten makefiles are preserved as `Makefile.legacy` in the top
level and in `src/`/`test/`/`bench/`:

```sh
make legacy-omp legacy-avx2 legacy-avx512 legacy-neon legacy-winograd
# or directly:
make -C src -f Makefile.legacy avx2
```

Standard Automake targets (`make dist`, `make distcheck`, `make uninstall`,
`make clean`, `make distclean`) are supported.

```sh
make dist                   # produce bncmatmul-X.Y.tar.gz
make dist-xz                # produce bncmatmul-X.Y.tar.xz
make dist-bzip2             # produce bncmatmul-X.Y.tar.bz2
make distcheck              # verify the tarball builds cleanly
```

Useful `configure` options:

- `--prefix=DIR` — install prefix (default: `/usr/local`)
- `--with-gmp=DIR` / `--with-mpfr=DIR` / `--with-mpc=DIR` / `--with-qd=DIR` — non-standard library prefixes
- `--enable-avx2` / `--enable-avx512` / `--enable-neon` — force a SIMD variant on/off (default: auto from host CPU)
- `--with-mplapack=DIR` — MPLAPACK install prefix (optional, used by some tests)

`configure` generates `bncmatmul.inc` from `bncmatmul.inc.in` and the
top-level `Makefile` from `Makefile.am`; the per-directory handwritten
`src/Makefile`, `test/Makefile`, and `bench/Makefile` include
`bncmatmul.inc` unchanged, so all existing legacy make targets continue
to work via Automake's delegation hooks.

--
Copyright (c) 2023-2026 Tomonori Kouya
