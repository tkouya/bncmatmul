#---------------------------------------
# test/Makefile.am for BNCmatmul
# Copyright (c) 2014-2026 Tomonori Kouya
#---------------------------------------

AUTOMAKE_OPTIONS = foreign subdir-objects no-dependencies

AM_CPPFLAGS = \
	-I$(top_srcdir)/include \
	\
	-DUSE_TD_BF -DUSE_QD_BF \
	-DUSE_GMP -DUSE_MPFR -DUSE_DD -DUSE_QD \
	@GMP_CPPFLAGS@ @MPFR_CPPFLAGS@ @MPC_CPPFLAGS@ @QD_CPPFLAGS@

AM_CFLAGS   = $(BNCMM_OPT_CFLAGS)
AM_CXXFLAGS = $(BNCMM_OPT_CXXFLAGS)

# Link against the Libtool library built by src/.  Libtool resolves
# both static (-release VERSION .a) and shared (.so) variants
# automatically.
BNCMATMUL_LA = $(top_builddir)/src/libbncmatmul.la

LDADD = \
	$(BNCMATMUL_LA) \
	@MPC_LDFLAGS@ @MPC_LIBS@ \
	@MPFR_LDFLAGS@ @MPFR_LIBS@ \
	@GMP_LDFLAGS@ @GMP_LIBS@ \
	@QD_LDFLAGS@ @QD_LIBS@ \
	-lpthread -lm -lstdc++

#---------------------------------------
# Test programs (built by 'make check', not installed)
#---------------------------------------
check_PROGRAMS = \
	mptest_matrix \
	mptest_cmatrix \
	test_clinear \
	test_clu \
	test_cddlinear \
	test_lanczos \
	test_krylov_spmm_cavity

mptest_matrix_SOURCES         = mptest_matrix.c
mptest_cmatrix_SOURCES        = mptest_cmatrix.c
test_clinear_SOURCES          = test_clinear.c
test_clu_SOURCES              = test_clu.c
test_cddlinear_SOURCES        = test_cddlinear.c
test_lanczos_SOURCES          = test_lanczos.c
test_krylov_spmm_cavity_SOURCES = test_krylov_spmm_cavity.c

# Dependency on the library being built first
$(check_PROGRAMS): $(BNCMATMUL_LA)

#---------------------------------------
# Extra files (additional tests and helpers, kept in distribution)
#---------------------------------------
EXTRA_DIST = \
	Makefile.legacy \
	example_read.c example_read_ex.c example_readmatrix.c \
	example_write.c example_write_ex.c \
	test_iterative_ref.c \
	test_iterative_ref_dd.cpp \
	test_iterative_ref_mpreal.cpp \
	test_iterative_ref_qd.cpp \
	test_mm_cmat.c \
	test_mpf_oz_scheme.c \
	test_mpfr.c \
	test_qd.cc \
	test_rdd.c \
	test_readfile.c \
	test_rsd.c \
	test_sparse_oneapi.c \
	test_copilot_sparse.c \
	test_spmm.c \
	test_urilink.c \
	webrank.c \
	webrank_mpf.c
