#---------------------------------------
# configure.ac for BNCmatmul
# Copyright (c) 2014-2026 Tomonori Kouya
#---------------------------------------

AC_PREREQ([2.69])
AC_INIT([bncmatmul], [0.24], [tkouya@gmail.com])
AC_CONFIG_SRCDIR([include/bncmatmul.h])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])

# Automake (use 'foreign' so we don't need NEWS/AUTHORS/ChangeLog/INSTALL)
AM_INIT_AUTOMAKE([1.11 foreign tar-pax no-dependencies subdir-objects])
AM_MAINTAINER_MODE([enable])

AC_CANONICAL_HOST

#---------------------------------------
# Programs
#---------------------------------------
AC_PROG_CC
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_MKDIR_P
AC_PROG_LN_S
AM_PROG_AR

#---------------------------------------
# Libtool: build both static (.a) and shared (.so) libraries by default.
# The user can opt out with --disable-shared or --disable-static.
#---------------------------------------
LT_PREREQ([2.2])
LT_INIT([shared static pic-only])
LT_LANG([C])
LT_LANG([C++])
AC_SUBST([LIBTOOL_DEPS])

#---------------------------------------
# C++ standard
#---------------------------------------
: ${CXXFLAGS_CXX_STD:=-std=c++17}
AC_SUBST([CXXFLAGS_CXX_STD])

#---------------------------------------
# Optimization defaults
#---------------------------------------
: ${BNCMM_OPT_CFLAGS:=-O3 -ffp-contract=off}
: ${BNCMM_OPT_CXXFLAGS:=-O3 -ffp-contract=off}
AC_SUBST([BNCMM_OPT_CFLAGS])
AC_SUBST([BNCMM_OPT_CXXFLAGS])

#---------------------------------------
# OpenMP
#---------------------------------------
AC_OPENMP
AS_IF([test "x$OPENMP_CFLAGS" = "x"],
      [OPENMP_LIBS=""],
      [OPENMP_LIBS="$OPENMP_CFLAGS"])
AC_SUBST([OPENMP_CFLAGS])
AC_SUBST([OPENMP_LIBS])

#---------------------------------------
# --enable-omp : also build the OpenMP-parallel Strassen / _bncomp_*
# routines (matmul_strassen_general_*_omp.c + bncomp_linear*.c) into
# libbncmatmul, and compile everything with $(OPENMP_CFLAGS) -DUSE_OMP.
# The parallel API (_bncomp_mul_<prec>matrix_strassen, ...) is provided
# IN ADDITION to the plain serial one (different symbol names), so this
# only adds, never replaces.
#---------------------------------------
AC_ARG_ENABLE([omp],
  [AS_HELP_STRING([--enable-omp],
    [build the OpenMP-parallel Strassen / _bncomp_* routines into libbncmatmul (default: no)])],
  [enable_omp=$enableval], [enable_omp=no])
AS_IF([test "x$enable_omp" = "xyes" && test "x$OPENMP_CFLAGS" = "x"],
      [AC_MSG_ERROR([--enable-omp requested but the C compiler does not support OpenMP])])
AC_SUBST([ENABLE_OMP], [$enable_omp])

#---------------------------------------
# Required libraries: GMP, MPFR, MPC, QD
#---------------------------------------
AC_ARG_WITH([gmp],
  [AS_HELP_STRING([--with-gmp=DIR], [GMP install prefix])],
  [GMP_PREFIX="$withval"], [GMP_PREFIX=""])
AC_ARG_WITH([mpfr],
  [AS_HELP_STRING([--with-mpfr=DIR], [MPFR install prefix])],
  [MPFR_PREFIX="$withval"], [MPFR_PREFIX=""])
AC_ARG_WITH([mpc],
  [AS_HELP_STRING([--with-mpc=DIR], [MPC install prefix])],
  [MPC_PREFIX="$withval"], [MPC_PREFIX=""])
AC_ARG_WITH([qd],
  [AS_HELP_STRING([--with-qd=DIR], [Bailey's QD install prefix])],
  [QD_PREFIX="$withval"], [QD_PREFIX=""])

GMP_CPPFLAGS=""; GMP_LDFLAGS=""
MPFR_CPPFLAGS=""; MPFR_LDFLAGS=""
MPC_CPPFLAGS=""; MPC_LDFLAGS=""
QD_CPPFLAGS=""; QD_LDFLAGS=""

AS_IF([test "x$GMP_PREFIX" != "x"],
      [GMP_CPPFLAGS="-I$GMP_PREFIX/include"
       GMP_LDFLAGS="-L$GMP_PREFIX/lib"])
AS_IF([test "x$MPFR_PREFIX" != "x"],
      [MPFR_CPPFLAGS="-I$MPFR_PREFIX/include"
       MPFR_LDFLAGS="-L$MPFR_PREFIX/lib"])
AS_IF([test "x$MPC_PREFIX" != "x"],
      [MPC_CPPFLAGS="-I$MPC_PREFIX/include"
       MPC_LDFLAGS="-L$MPC_PREFIX/lib"])
AS_IF([test "x$QD_PREFIX" != "x"],
      [QD_CPPFLAGS="-I$QD_PREFIX/include"
       QD_LDFLAGS="-L$QD_PREFIX/lib"])

# Save and adjust flags so AC_CHECK_HEADER / AC_CHECK_LIB look in --with-* dirs
saved_CPPFLAGS="$CPPFLAGS"
saved_LDFLAGS="$LDFLAGS"
saved_LIBS="$LIBS"

CPPFLAGS="$saved_CPPFLAGS $GMP_CPPFLAGS"
LDFLAGS="$saved_LDFLAGS $GMP_LDFLAGS"
AC_CHECK_HEADER([gmp.h], [],
  [AC_MSG_ERROR([gmp.h not found; install GMP or use --with-gmp=PREFIX])])
AC_CHECK_LIB([gmp], [__gmpz_init],
  [GMP_LIBS="-lgmp"],
  [AC_MSG_ERROR([libgmp not found; install GMP or use --with-gmp=PREFIX])])

CPPFLAGS="$saved_CPPFLAGS $GMP_CPPFLAGS $MPFR_CPPFLAGS"
LDFLAGS="$saved_LDFLAGS $GMP_LDFLAGS $MPFR_LDFLAGS"
LIBS="$GMP_LIBS $saved_LIBS"
AC_CHECK_HEADER([mpfr.h], [],
  [AC_MSG_ERROR([mpfr.h not found; install MPFR or use --with-mpfr=PREFIX])])
AC_CHECK_LIB([mpfr], [mpfr_init],
  [MPFR_LIBS="-lmpfr"],
  [AC_MSG_ERROR([libmpfr not found; install MPFR or use --with-mpfr=PREFIX])],
  [$GMP_LIBS])

CPPFLAGS="$saved_CPPFLAGS $GMP_CPPFLAGS $MPFR_CPPFLAGS $MPC_CPPFLAGS"
LDFLAGS="$saved_LDFLAGS $GMP_LDFLAGS $MPFR_LDFLAGS $MPC_LDFLAGS"
LIBS="$MPFR_LIBS $GMP_LIBS $saved_LIBS"
AC_CHECK_HEADER([mpc.h], [],
  [AC_MSG_ERROR([mpc.h not found; install MPC or use --with-mpc=PREFIX])])
AC_CHECK_LIB([mpc], [mpc_init2],
  [MPC_LIBS="-lmpc"],
  [AC_MSG_ERROR([libmpc not found; install MPC or use --with-mpc=PREFIX])],
  [$MPFR_LIBS $GMP_LIBS])

# QD library (Bailey's quad-double / double-double) — header is C++-friendly
AC_LANG_PUSH([C++])
CPPFLAGS="$saved_CPPFLAGS $QD_CPPFLAGS"
LDFLAGS="$saved_LDFLAGS $QD_LDFLAGS"
LIBS="$saved_LIBS"
AC_CHECK_HEADER([qd/dd_real.h], [],
  [AC_MSG_ERROR([qd/dd_real.h not found; install QD or use --with-qd=PREFIX])])
AC_CHECK_LIB([qd], [c_dd_add],
  [QD_LIBS="-lqd"],
  [AC_MSG_ERROR([libqd not found; install QD or use --with-qd=PREFIX])])
AC_LANG_POP([C++])

# Restore flags
CPPFLAGS="$saved_CPPFLAGS"
LDFLAGS="$saved_LDFLAGS"
LIBS="$saved_LIBS"

AC_SUBST([GMP_CPPFLAGS])
AC_SUBST([GMP_LDFLAGS])
AC_SUBST([GMP_LIBS])
AC_SUBST([MPFR_CPPFLAGS])
AC_SUBST([MPFR_LDFLAGS])
AC_SUBST([MPFR_LIBS])
AC_SUBST([MPC_CPPFLAGS])
AC_SUBST([MPC_LDFLAGS])
AC_SUBST([MPC_LIBS])
AC_SUBST([QD_CPPFLAGS])
AC_SUBST([QD_LDFLAGS])
AC_SUBST([QD_LIBS])

#---------------------------------------
# SIMD / NEON variants (enable based on host CPU; user can override)
#---------------------------------------
AC_ARG_ENABLE([avx2],
  [AS_HELP_STRING([--enable-avx2], [build AVX2 variants (default: auto on x86_64)])],
  [enable_avx2=$enableval], [enable_avx2=auto])
AC_ARG_ENABLE([avx512],
  [AS_HELP_STRING([--enable-avx512], [build AVX-512 variants (default: auto on x86_64)])],
  [enable_avx512=$enableval], [enable_avx512=auto])
AC_ARG_ENABLE([neon],
  [AS_HELP_STRING([--enable-neon], [build NEON variants (default: auto on aarch64)])],
  [enable_neon=$enableval], [enable_neon=auto])
AC_ARG_ENABLE([sve2],
  [AS_HELP_STRING([--enable-sve2], [build Armv9-A SVE2 variants — Grace / Neoverse V2 (default: no)])],
  [enable_sve2=$enableval], [enable_sve2=no])

case "$host_cpu" in
  x86_64|amd64|i?86)
    AS_IF([test "x$enable_avx2" = "xauto"], [enable_avx2=yes])
    AS_IF([test "x$enable_avx512" = "xauto"], [enable_avx512=yes])
    AS_IF([test "x$enable_neon" = "xauto"], [enable_neon=no])
    ;;
  aarch64|arm64|arm*)
    AS_IF([test "x$enable_avx2" = "xauto"], [enable_avx2=no])
    AS_IF([test "x$enable_avx512" = "xauto"], [enable_avx512=no])
    AS_IF([test "x$enable_neon" = "xauto"], [enable_neon=yes])
    ;;
  *)
    AS_IF([test "x$enable_avx2" = "xauto"], [enable_avx2=no])
    AS_IF([test "x$enable_avx512" = "xauto"], [enable_avx512=no])
    AS_IF([test "x$enable_neon" = "xauto"], [enable_neon=no])
    ;;
esac

AVX2_CFLAGS=""
AVX512_CFLAGS=""
NEON_CFLAGS=""
SVE2_CFLAGS=""
AS_IF([test "x$enable_avx2" = "xyes"], [AVX2_CFLAGS="-mavx2 -mfma"])
AS_IF([test "x$enable_avx512" = "xyes"], [AVX512_CFLAGS="-mavx512f -mfma"])
AS_IF([test "x$enable_neon" = "xyes"], [NEON_CFLAGS="-mcpu=cortex-a76 -DBNC_ENABLE_NEON"])
dnl Arm SVE2 (Armv9-A) target — VL-agnostic intrinsics.  -DBNC_ENABLE_SVE2
dnl selects the SVE2 branches in src/*linear.c / bncomp_linear_*.c.  We keep
dnl -DBNC_ENABLE_NEON as a fallback because some helper layers still only
dnl have a NEON branch (the #if/#elif ordering puts SVE2 first where both
dnl exist).  No -msve-vector-bits — the binary runs on any SVE2 VL.
AS_IF([test "x$enable_sve2" = "xyes"], [SVE2_CFLAGS="-mcpu=neoverse-v2 -DBNC_ENABLE_SVE2 -DBNC_ENABLE_NEON"])
AC_SUBST([AVX2_CFLAGS])
AC_SUBST([AVX512_CFLAGS])
AC_SUBST([NEON_CFLAGS])
AC_SUBST([SVE2_CFLAGS])
AC_SUBST([ENABLE_AVX2], [$enable_avx2])
AC_SUBST([ENABLE_AVX512], [$enable_avx512])
AC_SUBST([ENABLE_NEON], [$enable_neon])
AC_SUBST([ENABLE_SVE2], [$enable_sve2])

#---------------------------------------
# CUDA + gdtq (optional; --enable-cuda to opt in)
#
# When enabled, builds a separate convenience library libbncmm_cuda.la
# from the GPU Strassen sources (gddlinear.cu, gqdlinear.cu,
# matmul_strassen_general_g{dd,qd}.{cu,cc}, bncuda_linear*.c) and links
# it into the final libbncmatmul.la.  Requires nvcc and the gdtq header
# package.
#---------------------------------------
AC_ARG_ENABLE([cuda],
  [AS_HELP_STRING([--enable-cuda],
    [build the CUDA / gdtq GPU Strassen variants (default: no)])],
  [enable_cuda=$enableval], [enable_cuda=no])

AC_ARG_WITH([gdtq],
  [AS_HELP_STRING([--with-gdtq=DIR],
    [gdtq install prefix; headers expected under DIR/include/gdtq @<:@default=/usr/local@:>@])],
  [GDTQ_PREFIX="$withval"], [GDTQ_PREFIX="/usr/local"])

GDTQ_CPPFLAGS=""
AS_IF([test "x$enable_cuda" = "xyes"], [
  AX_CUDA

  AC_MSG_CHECKING([for gdtq headers])
  AS_IF([test -f "$GDTQ_PREFIX/include/gdtq/gqd_type.h"], [
    GDTQ_CPPFLAGS="-I$GDTQ_PREFIX/include/gdtq"
    AC_MSG_RESULT([$GDTQ_PREFIX/include/gdtq])
  ], [test -f "$GDTQ_PREFIX/inc/gqd_type.h"], [
    dnl In-tree gdtq source layout (gdtq-X.Y.Z/inc/...)
    GDTQ_CPPFLAGS="-I$GDTQ_PREFIX/inc"
    AC_MSG_RESULT([$GDTQ_PREFIX/inc (in-tree layout)])
  ], [
    AC_MSG_RESULT([not found])
    AC_MSG_ERROR([gqd_type.h not found under $GDTQ_PREFIX/include/gdtq or $GDTQ_PREFIX/inc; pass --with-gdtq=PREFIX])
  ])
])

AC_SUBST([GDTQ_PREFIX])
AC_SUBST([GDTQ_CPPFLAGS])

#---------------------------------------
# mpc_cuda (optional; only when --enable-cuda).  Provides the header-only
# register-resident cu_freal<>/cu_fcomplex<> multiple-precision types used by
# the GPU MPFR/MPC linear kernels (mpflinear_cu.cu, cmpflinear_cu.cu).  Only the
# headers are needed -- those translation units link WITHOUT libmpc_cuda.a, so
# they device-link cleanly alongside the g[dtq][ds]linear_cu objects.
#---------------------------------------
AC_ARG_WITH([mpc-cuda],
  [AS_HELP_STRING([--with-mpc-cuda=DIR],
    [mpc_cuda prefix; headers expected under DIR/include/mpc_cuda @<:@default: auto-detect@:>@])],
  [MPC_CUDA_PREFIX="$withval"], [MPC_CUDA_PREFIX=""])

MPC_CUDA_CPPFLAGS=""
AS_IF([test "x$enable_cuda" = "xyes"], [
  AC_MSG_CHECKING([for mpc_cuda headers (cu_freal.cuh)])
  dnl explicit --with-mpc-cuda first, then common locations
  mpc_cuda_candidates="$MPC_CUDA_PREFIX /usr/local /home/tkouya/na/cuda/mpc_cuda /home/tkouya/na/cuda/mpc_cuda-0.0.1"
  mpc_cuda_found=no
  for d in $mpc_cuda_candidates; do
    test -z "$d" && continue
    AS_IF([test -f "$d/include/mpc_cuda/cu_freal.cuh"], [
      MPC_CUDA_CPPFLAGS="-I$d/include"
      MPC_CUDA_PREFIX="$d"
      mpc_cuda_found=yes
      break
    ])
  done
  AS_IF([test "x$mpc_cuda_found" = "xyes"], [
    AC_MSG_RESULT([$MPC_CUDA_PREFIX/include])
  ], [
    AC_MSG_RESULT([not found])
    AC_MSG_ERROR([cu_freal.cuh not found; pass --with-mpc-cuda=PREFIX (expects PREFIX/include/mpc_cuda/cu_freal.cuh)])
  ])
])
AC_SUBST([MPC_CUDA_PREFIX])
AC_SUBST([MPC_CUDA_CPPFLAGS])

AC_SUBST([ENABLE_CUDA], [$enable_cuda])

#---------------------------------------
# Build tool tag (kept for compatibility with existing bncmatmul.inc logic)
#---------------------------------------
case "$CC" in
  *icc*|*icx*) BUILD_TOOL=icc ;;
  *emcc*)      BUILD_TOOL=emcc ;;
  *)           BUILD_TOOL=gcc ;;
esac
AC_SUBST([BUILD_TOOL])

#---------------------------------------
# MPLAPACK (optional; paths only — not required to build the core library)
#---------------------------------------
AC_ARG_WITH([mplapack],
  [AS_HELP_STRING([--with-mplapack=DIR], [MPLAPACK install prefix (optional)])],
  [MPLAPACK_PREFIX="$withval"], [MPLAPACK_PREFIX="/usr/local"])
MPLAPACK_INC_PATH="$MPLAPACK_PREFIX/include/mplapack"
MPLAPACK_LIB_PATH="$MPLAPACK_PREFIX/lib"
AC_SUBST([MPLAPACK_INC_PATH])
AC_SUBST([MPLAPACK_LIB_PATH])

#---------------------------------------
# Convenience: package directory name (used by dist tarball)
#---------------------------------------
BNCMATMUL_PREFIX="bncmatmul-${PACKAGE_VERSION}"
AC_SUBST([BNCMATMUL_PREFIX])

#---------------------------------------
# Conditionals (used by Automake AM_CONDITIONAL in subdirs)
#---------------------------------------
AM_CONDITIONAL([HAVE_OPENMP], [test "x$OPENMP_CFLAGS" != "x"])
AM_CONDITIONAL([ENABLE_OMP],    [test "x$enable_omp"    = "xyes"])
AM_CONDITIONAL([ENABLE_AVX2],   [test "x$enable_avx2"   = "xyes"])
AM_CONDITIONAL([ENABLE_AVX512], [test "x$enable_avx512" = "xyes"])
AM_CONDITIONAL([ENABLE_NEON],   [test "x$enable_neon"   = "xyes"])
AM_CONDITIONAL([ENABLE_SVE2],   [test "x$enable_sve2"   = "xyes"])
AM_CONDITIONAL([ENABLE_CUDA],   [test "x$enable_cuda"   = "xyes"])

#---------------------------------------
# Output
#---------------------------------------
AC_CONFIG_FILES([
    Makefile
    bncmatmul.inc
    include/Makefile
    src/Makefile
    src/cuda/Makefile
    test/Makefile
    bench/Makefile
    python/Makefile
    doc/Makefile
])
AC_OUTPUT

echo ""
echo "BNCmatmul ${PACKAGE_VERSION} configuration:"
echo "  host CPU            : $host_cpu"
echo "  CC                  : $CC"
echo "  CXX                 : $CXX"
echo "  AR                  : $AR"
echo "  prefix              : $prefix"
echo "  OpenMP CFLAGS       : $OPENMP_CFLAGS"
echo "  OpenMP Strassen     : $enable_omp"
echo "  GMP                 : $GMP_CPPFLAGS $GMP_LDFLAGS $GMP_LIBS"
echo "  MPFR                : $MPFR_CPPFLAGS $MPFR_LDFLAGS $MPFR_LIBS"
echo "  MPC                 : $MPC_CPPFLAGS $MPC_LDFLAGS $MPC_LIBS"
echo "  QD                  : $QD_CPPFLAGS $QD_LDFLAGS $QD_LIBS"
echo "  AVX2 variants       : $enable_avx2 ($AVX2_CFLAGS)"
echo "  AVX-512 variants    : $enable_avx512 ($AVX512_CFLAGS)"
echo "  NEON variants       : $enable_neon ($NEON_CFLAGS)"
echo "  SVE2 variants       : $enable_sve2 ($SVE2_CFLAGS)"
echo "  CUDA / gdtq         : $enable_cuda"
AS_IF([test "x$enable_cuda" = "xyes"], [
echo "    CUDA toolkit      : $CUDA_DIR"
echo "    CUDA arch         : $CUDA_ARCH"
echo "    NVCC              : $NVCC"
echo "    gdtq headers      : $GDTQ_CPPFLAGS"
echo "    mpc_cuda headers  : $MPC_CUDA_CPPFLAGS"
])
echo ""
echo "Now run 'make' to build, optionally 'make avx2' / 'make avx512' / 'make neon'."
echo ""
