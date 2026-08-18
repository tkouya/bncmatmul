# example_rdtq.py : calling BNCmatmul's DD/TD/QD arithmetic from Python
#
# This is the runnable source of the code fragments in Chapter 9 of the User's
# Guide ("Using BNCmatmul from Python").  It needs only the standard library
# plus python/librdtq.so; the last section additionally uses NumPy if it is
# installed and falls back to plain ctypes buffers if it is not.
#
# Build the shared library first:
#
#     cd bncmatmul-0.24/src
#     make -f Makefile.legacy librdtq          # -> ../python/librdtq.so
#
# then run:  python3 python/example_rdtq.py
#
# Copyright (C) 2026 Tomonori Kouya
# This file is part of BNCmatmul and distributed under the GNU LGPL v3.

import ctypes as ct
import math
import os
from decimal import Decimal, getcontext

# ---------------------------------------------------------------------------
# Loading the library and printing full precision
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
lib = ct.CDLL(os.path.join(HERE, "librdtq.so"))

DD = ct.c_double * 2    # double-double  (~32 decimal digits)
TD = ct.c_double * 3    # triple-double  (~48 decimal digits)
QD = ct.c_double * 4    # quad-double    (~63 decimal digits)

getcontext().prec = 70


def to_decimal(limbs):
    return sum((Decimal(x) for x in limbs), Decimal(0))


x = DD(1.5, 0.0)        # x = 1.5 (high limb, low limb)
r = DD()
lib.rdd_exp(r, x)       # result first, as in rdd.h
print("dd exp(1.5) =", to_decimal(r))
# dd exp(1.5) = 4.48168907033806482260205546011928...

# ---------------------------------------------------------------------------
# Arithmetic, FMA and elementary functions
# ---------------------------------------------------------------------------
# quad-double: sqrt(2) to ~63 digits
two = QD(2.0, 0.0, 0.0, 0.0)
s = QD()
lib.rqd_sqrt(s, two)
print("qd sqrt(2) =", to_decimal(s))
# 1.41421356237309504880168872420969807856967187537694807317667973...

# residual sqrt(2)^2 - 2 via the certified FMA: fma(s, s, -2)
m2 = QD(-2.0, 0.0, 0.0, 0.0)
resid = QD()
lib.rqd_fma(resid, s, s, m2)
print("qd sqrt(2)^2 - 2 =", to_decimal(resid))   # ~0 (qd eps level)

# FMA-driven division: 1/3 in triple-double
a = TD(1.0, 0.0, 0.0);  b = TD(3.0, 0.0, 0.0);  q = TD()
lib.rtd_div_fma(q, a, b)
print("td 1/3 =", to_decimal(q))
# 0.33333333333333333333333333333333333333333333333327...

# simultaneous sin/cos (two results, then the input)
sin_x = DD();  cos_x = DD()
lib.rdd_sincos(sin_x, cos_x, x)
print("dd sin(1.5) =", to_decimal(sin_x))
print("dd cos(1.5) =", to_decimal(cos_x))

# div/sqrt-safe FMA: the scalar multiplier is a plain c_double
res = DD()
lib.rdd_fma_safe(res, DD(3.0, 0.0), ct.c_double(-0.5), DD(1.5, 0.0))
print("1.5 - 0.5*3 =", to_decimal(res))          # exactly 0

# ---------------------------------------------------------------------------
# Elementwise array functions (limb-planar layout), with NumPy if available
# ---------------------------------------------------------------------------
def planar(*arrays):
    """double*[k] view of k limb arrays (limb-planar layout)"""
    ptr = ct.POINTER(ct.c_double) * len(arrays)
    return ptr(*[a.ctypes.data_as(ct.POINTER(ct.c_double))
                 if hasattr(a, "ctypes")
                 else ct.cast(a, ct.POINTER(ct.c_double))
                 for a in arrays])


try:
    import numpy as np

    n = 100000
    x_hi = np.linspace(-10.0, 10.0, n)     # high limbs
    x_lo = np.zeros(n)                     # low limbs
    r_hi = np.empty(n);  r_lo = np.empty(n)

    lib.bnc_dd_exp_array(ct.c_long(n),
                         planar(r_hi, r_lo), planar(x_hi, x_lo))

    err = np.max(np.abs((r_hi + r_lo) / np.exp(x_hi) - 1.0))
    print("max rel.err vs numpy =", err)   # ~2.2e-16 (binary64 comparison floor)

except ImportError:
    # same call without NumPy: the limb planes are plain ctypes arrays
    n = 100000
    x_hi = (ct.c_double * n)();  x_lo = (ct.c_double * n)()
    r_hi = (ct.c_double * n)();  r_lo = (ct.c_double * n)()
    for i in range(n):
        x_hi[i] = -10.0 + 20.0 * i / (n - 1)

    lib.bnc_dd_exp_array(ct.c_long(n),
                         planar(r_hi, r_lo), planar(x_hi, x_lo))

    err = max(abs((r_hi[i] + r_lo[i]) / math.exp(x_hi[i]) - 1.0)
              for i in range(n))
    print("max rel.err vs math.exp =", err, "(numpy not installed)")
