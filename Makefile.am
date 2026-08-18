#---------------------------------------
# Makefile.am for BNCmatmul (top-level)
# Copyright (c) 2014-2026 Tomonori Kouya
#---------------------------------------

ACLOCAL_AMFLAGS  = -I m4
AUTOMAKE_OPTIONS = foreign

# The library, headers, tests, benchmarks, python bindings and docs each
# live in their own subdirectory with a Makefile.am.  Order matters:
# include/ must come first (the others #include from it), then src/
# (which the test/bench programs link against).
SUBDIRS = include src test bench python doc

#---------------------------------------
# Top-level files for "make dist"
#---------------------------------------
EXTRA_DIST = \
	LICENSE \
	autogen.sh \
	bncmatmul.inc.in \
	Makefile.legacy \
	mmbench.sh \
	mmbench_dd.sh \
	mmbench_qd.sh \
	README.md \
	mds_gpu_01.xlsx

# Strip generated artifacts from the dist tarball
dist-hook:
	@echo "Cleaning generated files from $(distdir)"
	-find $(distdir) -name '*.o'   -delete
	-find $(distdir) -name '*.lo'  -delete
	-find $(distdir) -name '*.la'  -delete
	-find $(distdir) -name '*.a'   -delete
	-find $(distdir) -name '*.so'  -delete
	-find $(distdir) -name '*.pyc' -delete
	-find $(distdir) -name 'autom4te.cache' -prune -exec rm -rf {} +
	-find $(distdir) -name '.libs'  -type d -prune -exec rm -rf {} +
	-find $(distdir) -name '.deps'  -type d -prune -exec rm -rf {} +
	-find $(distdir) -name '.dirstamp' -delete
	-rm -f $(distdir)/bncmatmul.inc
	-rm -f $(distdir)/config.log $(distdir)/config.status

#---------------------------------------
# Convenience aliases for the legacy CPU-variant builds.
# These delegate to src/Makefile.legacy (kept around as a reference of
# the original handwritten build).  The Automake/Libtool build itself
# produces only the generic libbncmatmul.la — these aliases provide the
# OpenMP / AVX2 / AVX-512 / NEON / Winograd variants that are not yet
# Automake-managed.
#
#   make omp       == make -C src -f Makefile.legacy matmul_strassen_lib_omp
#   make avx2      == make -C src -f Makefile.legacy avx2
#   make avx512    == make -C src -f Makefile.legacy avx512
#   make neon      == make -C src -f Makefile.legacy neon
#   make sve2      == make -C src -f Makefile.legacy sve2
#   make winograd  == make -C src -f Makefile.legacy matmul_winograd_lib matmul_winograd_lib_omp
#
# The longer 'legacy-*' names are kept for explicit invocation.
#---------------------------------------

LEGACY_MAKE = $(MAKE) -C $(top_srcdir)/src -f Makefile.legacy

legacy-check:
	@if test ! -f $(top_srcdir)/src/Makefile.legacy ; then \
	    echo "$(top_srcdir)/src/Makefile.legacy not found — legacy variants unavailable." ; \
	    echo "(The Automake/Libtool build produces only libbncmatmul-$(PACKAGE_VERSION).{a,so}.)" ; \
	    exit 1 ; \
	fi

omp:        legacy-check ; $(LEGACY_MAKE) matmul_strassen_lib_omp
avx2:       legacy-check ; $(LEGACY_MAKE) avx2
avx512:     legacy-check ; $(LEGACY_MAKE) avx512
neon:       legacy-check ; $(LEGACY_MAKE) neon
sve2:       legacy-check ; $(LEGACY_MAKE) sve2
winograd:   legacy-check ; $(LEGACY_MAKE) matmul_winograd_lib matmul_winograd_lib_omp

legacy-omp:      omp
legacy-avx2:     avx2
legacy-avx512:   avx512
legacy-neon:     neon
legacy-sve2:     sve2
legacy-winograd: winograd

distclean-local:
	-rm -f bncmatmul.inc

.PHONY: legacy-check omp avx2 avx512 neon sve2 winograd \
        legacy-omp legacy-avx2 legacy-avx512 legacy-neon legacy-sve2 legacy-winograd
