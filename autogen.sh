#!/bin/sh
#---------------------------------------
# autogen.sh — bootstrap Autotools-generated files
# Copyright (c) 2014-2026 Tomonori Kouya
#---------------------------------------
set -e

mkdir -p build-aux m4

if command -v autoreconf >/dev/null 2>&1; then
    # autoreconf -fi runs aclocal, libtoolize, autoheader (if needed),
    # automake --add-missing, and autoconf, in the right order.
    autoreconf -fi
else
    if command -v libtoolize >/dev/null 2>&1; then
        libtoolize --force --copy
    elif command -v glibtoolize >/dev/null 2>&1; then
        glibtoolize --force --copy
    fi
    aclocal -I m4
    automake --add-missing --copy --foreign
    autoconf
fi

echo ""
echo "Bootstrap complete.  Next steps:"
echo "  ./configure [--with-gmp=PREFIX --with-mpfr=PREFIX --with-mpc=PREFIX --with-qd=PREFIX]"
echo "  make"
echo ""
