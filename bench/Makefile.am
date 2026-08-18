#---------------------------------------
# bench/Makefile.am for BNCmatmul
# Copyright (c) 2014-2026 Tomonori Kouya
#---------------------------------------

AUTOMAKE_OPTIONS = foreign subdir-objects no-dependencies

AM_CPPFLAGS = \
	-I$(top_srcdir)/include \
	\
	-DUSE_TD_BF -DUSE_QD_BF \
	-DUSE_GMP -DUSE_MPFR \
	@GMP_CPPFLAGS@ @MPFR_CPPFLAGS@ @MPC_CPPFLAGS@ @QD_CPPFLAGS@

AM_CFLAGS   = $(BNCMM_OPT_CFLAGS)
AM_CXXFLAGS = $(BNCMM_OPT_CXXFLAGS)

BNCMATMUL_LA = $(top_builddir)/src/libbncmatmul.la

LDADD = \
	$(BNCMATMUL_LA) \
	@MPC_LDFLAGS@ @MPC_LIBS@ \
	@MPFR_LDFLAGS@ @MPFR_LIBS@ \
	@GMP_LDFLAGS@ @GMP_LIBS@ \
	@QD_LDFLAGS@ @QD_LIBS@ \
	-lpthread -lm -lstdc++

#---------------------------------------
# Core benchmark programs (LU and CLU; the legacy Makefile defines
# hundreds of variants — only the canonical entry points are
# Automake-managed.  Use `make legacy-bench` to drive the rest.
# Variants whose source files are not present in the dist tree
# (lu_bench_dd.c etc.) are built only via Makefile.legacy.)
#---------------------------------------
noinst_PROGRAMS = \
	lu_bench \
	clu_bench

lu_bench_SOURCES     = lu_bench.c
clu_bench_SOURCES    = clu_bench.c

$(noinst_PROGRAMS): $(BNCMATMUL_LA)

#---------------------------------------
# Distribute every other benchmark source for reference
# (the legacy handwritten Makefile builds 100+ variants of these)
#---------------------------------------
#---------------------------------------
# EXTRA_DIST: only files that are CONFIRMED to exist on the source tree.
# Files referenced here MUST exist or `make dist` fails — so this list
# stays conservative.  The legacy Makefile builds many more variants;
# add them here only after confirming they exist in your tree.
#---------------------------------------
EXTRA_DIST = \
	Makefile.legacy \
	README_simd_bench.md \
	clu_mpfcmplx.c \
	lu_new.c \
	matvecbench_cdd.cc matvecbench_cmpf.cc \
	matvecbench_cqd.cc matvecbench_ctd.cc \
	matvecbench_dd.cc matvecbench_mpfr.cc \
	matvecbench_qd.cc matvecbench_td.cc \
	vecbench.c

#---------------------------------------
# Optional benchmark sources — automatically added to EXTRA_DIST
# only if they exist in the source tree (avoids `make dist` failure
# when files are absent).  Re-run autoreconf after dropping new files
# in here.
#---------------------------------------
OPTIONAL_BENCH_FILES = \
	bench_simd_compare.sh \
	bench_ddlinear.cc bench_tdlinear.cc bench_qdlinear.cc \
	bench_dslinear.cc bench_tslinear.cc bench_qslinear.cc \
	bench_strassen_omp.cc bench_block_omp.cc \
	bench_cmul_vector_omp.cc bench_matvec_omp.cc \
	clu_bench_dd.c clu_bench_td.c clu_bench_qd.c \
	lu_bench_dd.c lu_bench_td.c lu_bench_qd.c \
	lu_bench_dd_new.c lu_bench_dd_org.c \
	matmulbench_d.c \
	matmulbench_dd.c matmulbench_dd.cc \
	matmulbench_qd.c matmulbench_qd.cc \
	matmulbench_td.cc \
	matmulbench_cdd.cc matmulbench_cqd.cc matmulbench_ctd.cc \
	matmulbench_mpc.cc matmulbench_mpf.c matmulbench_mpfcmplx.cc \
	matmulbench_mpfr.cc matmulbench_mpfr.cpp \
	matmulbench_rec_mpf.c

# Append optional sources to EXTRA_DIST only if they exist
EXTRA_DIST += $(foreach src,$(OPTIONAL_BENCH_FILES),$(wildcard $(srcdir)/$(src)))

#---------------------------------------
# mds/ : multi-precision / multi-backend micro-benchmark harness.
# Ship only the sources, scripts, tools and recorded results — NOT the
# compiled benchmark binaries (mds_bench_*), backups (*.bak/*.pre_float)
# or run logs (*.log).  Guarded by $(wildcard) so missing files never
# break `make dist`.
#---------------------------------------
MDS_DIST_FILES = \
	mds/README.md \
	mds/build.sh \
	mds/run.sh mds/run_avx512.sh mds/run_cplx.sh mds/run_cplx_float.sh \
	mds/run_x86.sh \
	mds/make_xlsx.py mds/make_axpy_xlsx.py \
	mds/run_axpy_cpu.sh \
	mds/mds_bench.cc mds/mds_bench_cplx.cc mds/sp_perf.cc \
	mds/axpy_bench.cc \
	mds/mds_verify.cc mds/mds_verify_rand.cc \
	mds/results.csv mds/results_cplx.csv \
	mds/results_x86.csv mds/results_x86_cplx.csv

EXTRA_DIST += $(foreach src,$(MDS_DIST_FILES),$(wildcard $(srcdir)/$(src)))

#---------------------------------------
# bench/cuda/ : GPU benchmark harness.  Ship the .cu sources plus the
# build/run/Excel scripts so the GPU benchmarks can be reconstructed;
# the compiled binaries (no extension) and out/ logs are NOT shipped.
# $(wildcard) keeps `make dist` safe when the dir is absent.
#---------------------------------------
EXTRA_DIST += \
	$(wildcard $(srcdir)/cuda/*.cu) \
	$(wildcard $(srcdir)/cuda/*.sh) \
	$(wildcard $(srcdir)/cuda/*.py) \
	$(wildcard $(srcdir)/cuda/*.md)

#---------------------------------------
# bench/arm/ : ARM (NEON/SVE2) benchmark harness + cf hand-SIMD.  Ship
# the .c/.h sources and build/run/Excel scripts only — not the compiled
# binaries (extensionless) or generated out/*.csv results.
#---------------------------------------
EXTRA_DIST += \
	$(wildcard $(srcdir)/arm/*.c) \
	$(wildcard $(srcdir)/arm/*.h) \
	$(wildcard $(srcdir)/arm/*.sh) \
	$(wildcard $(srcdir)/arm/*.py)

#---------------------------------------
# bench/fma/ : branch-free DW/TW/QW FMA benchmark harness.  Ship the
# sources, kernel include, and build/run/Excel scripts — not the
# compiled binaries or generated results.
#---------------------------------------
EXTRA_DIST += \
	$(wildcard $(srcdir)/fma/*.c) \
	$(wildcard $(srcdir)/fma/*.h) \
	$(wildcard $(srcdir)/fma/*.inc) \
	$(wildcard $(srcdir)/fma/*.sh) \
	$(wildcard $(srcdir)/fma/*.py) \
	$(wildcard $(srcdir)/fma/*.md)

#---------------------------------------
# bench/mm/ : matmul algorithm benchmark harness (triple/block/
# Strassen/Winograd x serial/SIMD).  Sources and scripts only.
#---------------------------------------
EXTRA_DIST += \
	$(wildcard $(srcdir)/mm/*.c) \
	$(wildcard $(srcdir)/mm/*.h) \
	$(wildcard $(srcdir)/mm/*.sh) \
	$(wildcard $(srcdir)/mm/*.py) \
	$(wildcard $(srcdir)/mm/*.md)

#---------------------------------------
# bench/elem/ : elementary vector-operation micro-benchmarks.
#---------------------------------------
EXTRA_DIST += \
	$(wildcard $(srcdir)/elem/*.c) \
	$(wildcard $(srcdir)/elem/*.h) \
	$(wildcard $(srcdir)/elem/*.sh) \
	$(wildcard $(srcdir)/elem/*.py) \
	$(wildcard $(srcdir)/elem/*.md)
