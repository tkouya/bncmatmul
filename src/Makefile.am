#---------------------------------------
# src/Makefile.am for BNCmatmul (Libtool version)
# Copyright (c) 2014-2026 Tomonori Kouya
#---------------------------------------
#
# Builds both static (libbncmatmul-X.Y.a) and shared
# (libbncmatmul-X.Y.so.*) variants via Libtool.  The .so version is
# tagged via -release; ABI versioning is via -version-info CURRENT:REV:AGE.
#---------------------------------------

AUTOMAKE_OPTIONS = foreign subdir-objects no-dependencies

AM_CPPFLAGS = \
	-I$(top_srcdir)/include \
	-DUSE_TD_BF -DUSE_QD_BF \
	-DUSE_TS_BF -DUSE_QS_BF \
	-DUSE_GMP -DUSE_MPFR \
	@GMP_CPPFLAGS@ @MPFR_CPPFLAGS@ @MPC_CPPFLAGS@ @QD_CPPFLAGS@ \
	@OPENBLAS_CPPFLAGS@

AM_CFLAGS   = $(BNCMM_OPT_CFLAGS)
AM_CXXFLAGS = $(BNCMM_OPT_CXXFLAGS)

# --enable-omp: compile everything with OpenMP + USE_OMP so the parallel
# Strassen / _bncomp_* routines (added to BNCMM_BASE_SOURCES below) are
# built, and the existing #pragma omp paths are activated.
if ENABLE_OMP
AM_CFLAGS   += $(OPENMP_CFLAGS) -DUSE_OMP
AM_CXXFLAGS += $(OPENMP_CFLAGS) -DUSE_OMP
endif

#---------------------------------------
# Sources of the BNCmatmul library
#---------------------------------------
BNCMM_BASE_SOURCES = \
	bnc_common_func.c complex.c \
	poly.c mpf_poly.c cmpf_poly.c \
	dd_poly.c cdd_poly.c td_poly.c ctd_poly.c qd_poly.c cqd_poly.c \
	dka.c mpf_dka.c mpc_dka.c dd_dka.c td_dka.c qd_dka.c \
	hirano.c dd_hirano.c td_hirano.c qd_hirano.c \
	array.c mpfarray.c elem_vector.c rdtq_func.c \
	cdlinear.c cdlu.c cmpflinear.c cmpflu.c \
	matmul_strassen_general.c \
	matmul_strassen_general_dd.c matmul_strassen_general_cdd.c \
	matmul_strassen_general_td.c matmul_strassen_general_ctd.c \
	matmul_strassen_general_qd.c matmul_strassen_general_cqd.c \
	matmul_strassen_general_ds.c matmul_strassen_general_ts.c matmul_strassen_general_qs.c \
	matmul_strassen_general_mpf.c matmul_strassen_general_mpc.c \
	get_sec.c bmatrix.c \
	mpflinear.c flinear.c dlinear.c \
	ddlinear.c cddlinear.c tdlinear.c ctdlinear.c \
	qdlinear.c cqdlinear.c \
	dslinear.c tslinear.c qslinear.c \
	gtestmat.c gtestmat_dd.c gtestmat_td.c gtestmat_qd.c \
	ddlu.c cddlu.c tdlu.c ctdlu.c qdlu.c cqdlu.c dslu.c mpflu.c \
	ddlu_strassen.c cddlu_strassen.c tdlu_strassen.c ctdlu_strassen.c \
	qdlu_strassen.c cqdlu_strassen.c \
	mpflu_strassen.c cmpflu_strassen.c \
	ddlu_oz.c cddlu_oz.c tdlu_oz.c ctdlu_oz.c qdlu_oz.c cqdlu_oz.c \
	mpflu_oz.c cmpflu_oz.c \
	benchmark_tools.c benchmark_tools_dd.c \
	mpfr_dtq_sd.c \
	fread_write.c fread_write_complex.c \
	oz_scheme.c dd_oz_scheme.c td_oz_scheme.c qd_oz_scheme.c mpf_oz_scheme.c \
	mmio.c readmatrix.c writematrix.c readwrite_clinear.c \
	sparse_double.c sparse_cd.c \
	sparse_mpf.c sparse_cmpf.c \
	sparse_dd.c sparse_cdd.c \
	sparse_td.c sparse_ctd.c \
	sparse_qd.c sparse_cqd.c \
	cdslinear.c ctslinear.c cqslinear.c \
	cdslu.c ctslu.c cqslu.c \
	sparse_float.c \
	sparse_ds.c sparse_ts.c sparse_qs.c \
	sparse_cds.c sparse_cts.c sparse_cqs.c \
	tridiagonal.c \
	cg.c cocg.c \
	krylov.c krylov_c.c \
	krylov_dd.c krylov_cdd.c \
	krylov_td.c krylov_ctd.c \
	krylov_qd.c krylov_cqd.c \
	lanczos.c \
	algebraic_eq.c hessenberg.c

# Multi-compiled sources: same .c built with different -D flags into
# separate convenience libtool libraries that are then linked into the
# final lib via _LIBADD.
SP_SOURCES = \
	cg.c cocg.c \
	krylov.c krylov_c.c \
	krylov_dd.c krylov_cdd.c \
	krylov_td.c krylov_ctd.c \
	krylov_qd.c krylov_cqd.c \
	lanczos.c

SP_D_SOURCES = \
	cg.c cocg.c \
	krylov.c krylov_c.c \
	krylov_dd.c krylov_cdd.c \
	krylov_td.c krylov_ctd.c \
	krylov_qd.c krylov_cqd.c

SP_PRECOND_SOURCES = \
	krylov.c krylov_c.c \
	krylov_dd.c krylov_cdd.c \
	krylov_td.c krylov_ctd.c \
	krylov_qd.c krylov_cqd.c

#---------------------------------------
# Convenience libtool libraries (not installed; merged into the final lib)
#---------------------------------------
noinst_LTLIBRARIES = \
	libbncmm_base.la \
	libbncmm_sp.la \
	libbncmm_sp_d.la \
	libbncmm_sp_precond.la \
	libbncmm_sp_d_precond.la

# OpenMP-parallel Strassen / bncomp sources — added to libbncmm_base only
# when --enable-omp.  These define the _bncomp_* parallel API (and the
# *_omp Strassen counters); they do NOT redefine the plain serial
# routines, so they coexist with the rest of BNCMM_BASE_SOURCES.
OMP_STRASSEN_SOURCES = \
	matmul_strassen_general_mpf_omp.c matmul_strassen_general_mpc_omp.c \
	matmul_strassen_general_dd_omp.c matmul_strassen_general_td_omp.c matmul_strassen_general_qd_omp.c \
	matmul_strassen_general_ds_omp.c matmul_strassen_general_ts_omp.c matmul_strassen_general_qs_omp.c \
	matmul_strassen_general_cdd_omp.c matmul_strassen_general_ctd_omp.c matmul_strassen_general_cqd_omp.c \
	bncomp_linear.c bncomp_linear_double.c bncomp_linear_mpf.c bncomp_linear_mpc.c \
	bncomp_linear_dd.c bncomp_linear_td.c bncomp_linear_qd.c \
	bncomp_linear_ds.c bncomp_linear_ts.c bncomp_linear_qs.c \
	bncomp_linear_cdd.c bncomp_linear_ctd.c bncomp_linear_cqd.c \
	bncomp_sparse.c

libbncmm_base_la_SOURCES = $(BNCMM_BASE_SOURCES)
if ENABLE_OMP
libbncmm_base_la_SOURCES += $(OMP_STRASSEN_SOURCES)
endif
libbncmm_base_la_CPPFLAGS = $(AM_CPPFLAGS)

# NOTE: the single-based (ds/ts/qs, complex-single, sparse_float) family
# used to live in separate convenience libs (libbncmm_single/libbncmm_fsparse)
# compiled WITHOUT the DD-family macros: cdslinear.h etc. accidentally
# redefined GET_CDDMATRIX_IJ & co. (copy-paste vestiges, since renamed to
# their own GET_CDSMATRIX_IJ names) and the USE_GMP path referenced the
# then-missing r{ds,ts,qs}_sqrt_mpfr (since implemented in mpfr_dtq_sd.c).
# Both root causes are fixed, so everything is one library now and the
# USE_*LINEAR compile-time gates are gone entirely.

libbncmm_sp_la_SOURCES  = $(SP_SOURCES)
libbncmm_sp_la_CPPFLAGS = $(AM_CPPFLAGS) -DUSE_SPARSE_VERSION

libbncmm_sp_d_la_SOURCES  = $(SP_D_SOURCES)
libbncmm_sp_d_la_CPPFLAGS = $(AM_CPPFLAGS) -DUSE_SPARSE_D_VERSION

libbncmm_sp_precond_la_SOURCES  = $(SP_PRECOND_SOURCES)
libbncmm_sp_precond_la_CPPFLAGS = $(AM_CPPFLAGS) -DUSE_SPARSE_VERSION -DUSE_PRECOND

libbncmm_sp_d_precond_la_SOURCES  = $(SP_PRECOND_SOURCES)
libbncmm_sp_d_precond_la_CPPFLAGS = $(AM_CPPFLAGS) -DUSE_SPARSE_D_VERSION -DUSE_PRECOND

#---------------------------------------
# Final installable library: libbncmatmul.la
#   -> libbncmatmul-X.Y.a               (static archive, via -release)
#   -> libbncmatmul-X.Y.so.0.0.0        (shared object)
#   -> libbncmatmul-X.Y.so.0  (symlink)
#   -> libbncmatmul-X.Y.so    (symlink)
# ABI versioning: -version-info CURRENT:REVISION:AGE = 0:0:0 initially.
# Bump when the public ABI changes (see Libtool manual "Updating version
# information").
#---------------------------------------
lib_LTLIBRARIES = libbncmatmul.la

libbncmatmul_la_SOURCES =
libbncmatmul_la_LIBADD = \
	libbncmm_base.la \
	libbncmm_sp.la \
	libbncmm_sp_d.la \
	libbncmm_sp_precond.la \
	libbncmm_sp_d_precond.la \
	@MPC_LDFLAGS@ @MPC_LIBS@ \
	@MPFR_LDFLAGS@ @MPFR_LIBS@ \
	@GMP_LDFLAGS@ @GMP_LIBS@ \
	@QD_LDFLAGS@ @QD_LIBS@ \
	@OPENBLAS_LDFLAGS@ @OPENBLAS_LIBS@ \
	-lpthread -lm -lstdc++

if ENABLE_OMP
libbncmatmul_la_LIBADD += $(OPENMP_LIBS)
endif

libbncmatmul_la_LDFLAGS = \
	-release $(PACKAGE_VERSION) \
	-version-info 0:0:0 \
	-no-undefined

#---------------------------------------
# CUDA / gdtq GPU Strassen (only when --enable-cuda)
#
# Built by a hand-written Makefile in src/cuda/ (NOT processed by
# automake) which produces libbncmm_cuda.a out of nvcc-compiled objects.
# The .a is then linked into libbncmatmul.la as a regular static
# archive — libtool handles "embed plain .a into .la" cleanly.
#
# Source files (gddlinear.cu, matmul_strassen_general_g{dd,qd}.{cu,cc},
# bncuda_linear*.c) physically stay in src/; src/cuda/Makefile finds
# them via VPATH.
#---------------------------------------
if ENABLE_CUDA
SUBDIRS = cuda
libbncmatmul_la_LIBADD += $(top_builddir)/src/cuda/libbncmm_cuda.a \
                         @CUDA_LDFLAGS@ @CUDA_LIBS@
EXTRA_libbncmatmul_la_DEPENDENCIES = $(top_builddir)/src/cuda/libbncmm_cuda.a
endif

#---------------------------------------
# Extra files distributed but not built by Automake.
# Files referenced here MUST exist or `make dist` fails — so this list
# only includes always-present files (cuda Makefile + bncuda).
# Optional sources (legacy backups, .cu/.cc CUDA sources) are added
# below via `wildcard` only if they actually exist in the source tree.
#---------------------------------------
EXTRA_DIST = \
	Makefile.legacy \
	Makefile.txt \
	bncpack \
	mm \
	mpi \
	bncuda_linear.c \
	bncuda_linear_cublas.c \
	cuda/Makefile.in

#---------------------------------------
# Optional CUDA / legacy source files — added to EXTRA_DIST only if
# they exist in the source tree.  This avoids `make dist` aborting
# when a referenced file is missing.
#---------------------------------------
OPTIONAL_SRC_FILES = \
	gddlinear.cu gddlinear.cc gddlinear_original.cu gddlinear_test.cc \
	gqdlinear.cu gqdmatmul.cc gqd_test.cu gdd_test.cc \
	gtdlinear.cu \
	gdslinear.cu gqslinear.cu gtslinear.cu \
	matmul_strassen_general_gdd.cu matmul_strassen_general_gdd.cc \
	matmul_strassen_general_gqd.cu matmul_strassen_general_gqd.cc \
	matmul_strassen_general_gtd.cu matmul_strassen_general_gtd.cc \
	matmul_strassen_general_gds.cu matmul_strassen_general_gds.cc \
	matmul_strassen_general_gqs.cu matmul_strassen_general_gqs.cc \
	matmul_strassen_general_gts.cu matmul_strassen_general_gts.cc \
	\
	cuda/bench_gdtq_strassen.cu \
	cuda/bench_gdtqs_strassen.cu \
	cuda/bench_gdtqs_cmul_vector.cu \
	cuda/bench_gdtqs_matvec.cu \
	cuda/test_gdd_strassen.cu \
	cuda/test_gqd_strassen.cu \
	cuda/test_gtd_strassen.cu \
	\
	small_ddmatrix.c \
	bncold_dka_omp.c cg_omp.c clu.c cocg_omp.c dka_omp.c \
	matmul_strassen_general_omp.c

EXTRA_DIST += $(foreach src,$(OPTIONAL_SRC_FILES),$(wildcard $(srcdir)/$(src)))

#---------------------------------------
# Ship EVERY CUDA source physically present in src/ (and src/cuda/).
# The hand-maintained OPTIONAL_SRC_FILES list above covered only a
# handful of .cu; this wildcard guarantees the full GPU source set
# (gcd*/gcf*/gd*/gq*/gt* linear/lu/sparse, the cg* multi-component
# complex kernels, and all test_*.cu) ends up in `make dist` so the
# CUDA build can be reconstructed.  $(wildcard) keeps it dist-safe.
#---------------------------------------
EXTRA_DIST += $(wildcard $(srcdir)/*.cu) $(wildcard $(srcdir)/cuda/*.cu)

#---------------------------------------
# Never compile on behalf of a user who does not own the build tree.
#
# Automake's stock rule is
#
#     install-am: all-am
#
# so `sudo make install` happily rebuilds whatever is out of date -- as
# root.  Every object, every .libs/ entry and every relinked library it
# writes then belongs to root inside the developer's build tree, and the
# next ordinary `make` stops with "Permission denied" until someone
# chowns the whole tree back.  src/ is the only directory that compiles
# anything at `all` time, so the guard belongs here.
#
# The replacement below keeps `./configure && make install` working
# exactly as before for whoever owns the tree, and refuses only in the
# case that does the damage: a rebuild that would be carried out by a
# different user.  `make -q` answers "would anything be rebuilt?"
# without building it, and the -O test is skipped entirely for the
# owner, so the common path costs nothing.
#---------------------------------------
install-am:
	@if test ! -O . && ! $(MAKE) $(AM_MAKEFLAGS) -q all-am >/dev/null 2>&1; then \
	  echo "" >&2; \
	  echo "ERROR: 'make install' would have to compile, but it is running as" >&2; \
	  echo "       `id -un`, who does not own $(abs_builddir)." >&2; \
	  echo "       Compiling now would leave objects owned by `id -un` behind and" >&2; \
	  echo "       break the next plain 'make' with 'Permission denied'." >&2; \
	  echo "" >&2; \
	  echo "       Build first, then install:" >&2; \
	  echo "" >&2; \
	  echo "           make                 # as the owner of the build tree" >&2; \
	  echo "           sudo make install" >&2; \
	  echo "" >&2; \
	  echo "       ('make install-libs' only copies already-built libraries and" >&2; \
	  echo "        headers; it never compiles.)" >&2; \
	  echo "" >&2; \
	  exit 1; \
	fi
	@$(MAKE) $(AM_MAKEFLAGS) all-am
	@$(MAKE) $(AM_MAKEFLAGS) install-exec-am install-data-am
