# test_rdtq.py : ctypes driver for librdtq.so
#
# librdtq.so is the flat, DLL-ready export of the multi-component arithmetic
# (src/rdtq_func.c + src/elem_vector.c, see include/rdtq_func.h and
# include/bncelem_vector.h).  Build it with
#
#     make -C src -f Makefile.legacy librdtq         # -> python/librdtq.so
#     make -C src -f Makefile.legacy librdtq_neon    # NEON flavour
#
# then run  python3 python/test_rdtq.py .
#
# All six precision families are covered (rdd/rtd/rqd on double limbs,
# rds/rts/rqs on float limbs) plus the limb-planar array functions.  The
# reference is Python's decimal module at 120 significant digits; sin/cos are
# summed there from their Taylor series since decimal has no trigonometry.
#
# Copyright (C) 2026 Tomonori Kouya
# This file is part of BNCmatmul and distributed under the GNU LGPL v3.

import ctypes as ct
import os
import sys
from decimal import Decimal, getcontext

getcontext().prec = 120

HERE = os.path.dirname(os.path.abspath(__file__))
LIBNAME = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'librdtq.so')
lib = ct.cdll.LoadLibrary(LIBNAME)

# name, #limbs, ctypes limb type, unit roundoff of the type
FAMILIES = [
    ('rdd', 2, ct.c_double, Decimal(2) ** -106),
    ('rtd', 3, ct.c_double, Decimal(2) ** -159),
    ('rqd', 4, ct.c_double, Decimal(2) ** -212),
    ('rds', 2, ct.c_float,  Decimal(2) ** -48),
    ('rts', 3, ct.c_float,  Decimal(2) ** -72),
    ('rqs', 4, ct.c_float,  Decimal(2) ** -96),
]

n_fail = 0


def buf(ctype, size, limbs=None):
    """A ctypes limb array, optionally initialised from a list."""
    a = (ctype * size)()
    if limbs:
        for i, v in enumerate(limbs):
            a[i] = v
    return a


def from_float(ctype, size, x):
    """Represent the Python float x exactly: limb 0 = x, the rest 0."""
    return buf(ctype, size, [ctype(x).value] + [0.0] * (size - 1))


def to_dec(a):
    """Exact value of a multi-component number: the limbs are non-overlapping."""
    return sum((Decimal(a[i]) for i in range(len(a))), Decimal(0))


def check(name, got, want, tol):
    """got/want as Decimal; tol is the allowed relative error."""
    global n_fail
    if want == 0:
        err = abs(got)
    else:
        err = abs((got - want) / want)
    ok = err <= tol
    if not ok:
        n_fail += 1
    print('  %-22s rel.err = %.3e  %s' % (name, float(err), 'ok' if ok else 'FAILED'))


def dec_sin(x):
    """sin(x) by Taylor series; |x| is small in this driver."""
    s, term, k = Decimal(0), Decimal(x), 1
    while abs(term) > Decimal(10) ** -(getcontext().prec):
        s += term
        term = -term * x * x / ((k + 1) * (k + 2))
        k += 2
    return s


def dec_cos(x):
    s, term, k = Decimal(0), Decimal(1), 0
    while abs(term) > Decimal(10) ** -(getcontext().prec):
        s += term
        term = -term * x * x / ((k + 1) * (k + 2))
        k += 2
    return s


# --------------------------------------------------------------------------
# arithmetic, FMA and elementary functions, family by family
# --------------------------------------------------------------------------
for fam, size, ctype, eps in FAMILIES:
    print('%s (%d limbs, u = 2^-%d)' % (fam, size, -eps.log10() / Decimal(2).log10()))
    tol = eps * 64          # a few ulps of slack for the composite operations

    a = from_float(ctype, size, 3.25)
    b = from_float(ctype, size, 7.5)
    c = from_float(ctype, size, 0.125)
    r = buf(ctype, size)
    da, db, dc = to_dec(a), to_dec(b), to_dec(c)

    getattr(lib, fam + '_add')(r, a, b);  check('add', to_dec(r), da + db, tol)
    getattr(lib, fam + '_sub')(r, a, b);  check('sub', to_dec(r), da - db, tol)
    getattr(lib, fam + '_mul')(r, a, b);  check('mul', to_dec(r), da * db, tol)
    getattr(lib, fam + '_div')(r, a, b);  check('div', to_dec(r), da / db, tol)
    getattr(lib, fam + '_sqrt')(r, b);    check('sqrt', to_dec(r), db.sqrt(), tol)

    getattr(lib, fam + '_fma')(r, a, b, c)
    check('fma', to_dec(r), da * db + dc, tol)
    getattr(lib, fam + '_div_fma')(r, a, b)
    check('div_fma', to_dec(r), da / db, tol)

    # fma_safe takes a scalar second operand (double for rdd/rtd/rqd, float
    # for rds/rts/rqs -- ctypes promotes both, so declare it explicitly)
    fs = getattr(lib, fam + '_fma_safe')
    fs.argtypes = [ct.POINTER(ctype), ct.POINTER(ctype),
                   ct.c_double if ctype is ct.c_double else ct.c_float,
                   ct.POINTER(ctype)]
    fs(r, a, 7.5, c)
    check('fma_safe', to_dec(r), da * Decimal('7.5') + dc, tol)

    x = from_float(ctype, size, 0.75)
    dx = to_dec(x)
    getattr(lib, fam + '_exp')(r, x);    check('exp', to_dec(r), dx.exp(), tol)
    getattr(lib, fam + '_expm1')(r, x);  check('expm1', to_dec(r), dx.exp() - 1, tol)
    getattr(lib, fam + '_log')(r, b);    check('log', to_dec(r), db.ln(), tol)
    getattr(lib, fam + '_log10')(r, b);  check('log10', to_dec(r), db.log10(), tol)
    getattr(lib, fam + '_sin')(r, x);    check('sin', to_dec(r), dec_sin(dx), tol)
    getattr(lib, fam + '_cos')(r, x);    check('cos', to_dec(r), dec_cos(dx), tol)
    getattr(lib, fam + '_tan')(r, x);    check('tan', to_dec(r), dec_sin(dx) / dec_cos(dx), tol)

    s, co = buf(ctype, size), buf(ctype, size)
    getattr(lib, fam + '_sincos')(s, co, x)
    check('sincos(sin)', to_dec(s), dec_sin(dx), tol)
    check('sincos(cos)', to_dec(co), dec_cos(dx), tol)

    if ctype is ct.c_double:        # nint exists for the double-based types only
        y = from_float(ctype, size, 2.4)
        getattr(lib, fam + '_nint')(r, y)
        check('nint', to_dec(r), Decimal(2), Decimal(0))
    print()

# --------------------------------------------------------------------------
# limb-planar array functions (bncelem_vector.h) vs the scalar functions
# --------------------------------------------------------------------------
print('bnc_XX_yy_array vs scalar (must be bit-identical)')
N = 1000
for fam, tok, size in (('rdd', 'dd', 2), ('rtd', 'td', 3), ('rqd', 'qd', 4)):
    planes = [(ct.c_double * N)() for _ in range(size)]
    for i in range(N):
        planes[0][i] = 0.001 + 3.0 * i / N
    plane_ptrs = (ct.POINTER(ct.c_double) * size)(*[ct.cast(p, ct.POINTER(ct.c_double))
                                                    for p in planes])
    out = [(ct.c_double * N)() for _ in range(size)]
    out_ptrs = (ct.POINTER(ct.c_double) * size)(*[ct.cast(p, ct.POINTER(ct.c_double))
                                                  for p in out])

    for op in ('exp', 'log', 'sin', 'cos'):
        getattr(lib, 'bnc_%s_%s_array' % (tok, op))(ct.c_long(N), out_ptrs, plane_ptrs)
        bad = 0
        scal_in, scal_out = (ct.c_double * size)(), (ct.c_double * size)()
        for i in range(N):
            for k in range(size):
                scal_in[k] = planes[k][i]
            getattr(lib, '%s_%s' % (fam, op))(scal_out, scal_in)
            for k in range(size):
                if scal_out[k] != out[k][i]:
                    bad += 1
        if bad:
            n_fail += 1
        print('  bnc_%s_%s_array      %d/%d limbs differ  %s'
              % (tok, op, bad, N * size, 'ok' if bad == 0 else 'FAILED'))

print()
print('FAILURES: %d' % n_fail)
sys.exit(1 if n_fail else 0)
