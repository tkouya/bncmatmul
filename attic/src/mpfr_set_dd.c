/* mpfr_set_dd -- convert a Bailey's double-double precision float to
                 a multiple precision floating-point number

Copyright 1999, 2000, 2001, 2002, 2003, 2004, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2015 Free Software Foundation, Inc.
Contributed by the AriC and Caramel projects, INRIA.
Mofified by T.Kouya < http://na-inet.jp/ >

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#include <float.h>  /* For DOUBLE_ISINF and DOUBLE_ISNAN */

#define MPFR_NEED_LONGLONG_H
//#include "mpfr-impl.h"
#include "mpfr.h"

// necessary macros of mpfr-impl.h and mpfr-gmp.h 
#include "mpfr_dd.h"

int mpfr_set_dd (mpfr_ptr r, double d, mpfr_rnd_t rnd_mode)
{
  int signd, inexact;
  unsigned int cnt;
  mp_size_t i, k;
  mpfr_t tmp;
  mp_limb_t tmpmant[MPFR_LIMBS_PER_DOUBLE];
  MPFR_SAVE_EXPO_DECL (expo);

  if (MPFR_UNLIKELY(DOUBLE_ISNAN(d)))
    {
      MPFR_SET_NAN(r);
      MPFR_RET_NAN;
    }
  else if (MPFR_UNLIKELY(d == 0))
    {
#if _GMP_IEEE_FLOATS
      union ieee_double_extract x;

      MPFR_SET_ZERO(r);
      /* set correct sign */
      x.d = d;
      if (x.s.sig == 1)
        MPFR_SET_NEG(r);
      else
        MPFR_SET_POS(r);
#else /* _GMP_IEEE_FLOATS */
      MPFR_SET_ZERO(r);
      {
        /* This is to get the sign of zero on non-IEEE hardware
           Some systems support +0.0, -0.0 and unsigned zero.
           We can't use d==+0.0 since it should be always true,
           so we check that the memory representation of d is the
           same than +0.0. etc */
        /* FIXME: consider the case where +0.0 or -0.0 may have several
           representations. */
        double poszero = +0.0, negzero = DBL_NEG_ZERO;
        if (memcmp(&d, &poszero, sizeof(double)) == 0)
          MPFR_SET_POS(r);
        else if (memcmp(&d, &negzero, sizeof(double)) == 0)
          MPFR_SET_NEG(r);
        else
          MPFR_SET_POS(r);
      }
#endif
      return 0; /* 0 is exact */
    }
  else if (MPFR_UNLIKELY(DOUBLE_ISINF(d)))
    {
      MPFR_SET_INF(r);
      if (d > 0)
        MPFR_SET_POS(r);
      else
        MPFR_SET_NEG(r);
      return 0; /* infinity is exact */
    }

  /* now d is neither 0, nor NaN nor Inf */

  MPFR_SAVE_EXPO_MARK (expo);

  /* warning: don't use tmp=r here, even if SIZE(r) >= MPFR_LIMBS_PER_DOUBLE,
     since PREC(r) may be different from PREC(tmp), and then both variables
     would have same precision in the mpfr_set4 call below. */
  MPFR_MANT(tmp) = tmpmant;
  MPFR_PREC(tmp) = IEEE_DBL_MANT_DIG;

  signd = (d < 0) ? MPFR_SIGN_NEG : MPFR_SIGN_POS;
  d = ABS (d);

  /* don't use MPFR_SET_EXP here since the exponent may be out of range */
  MPFR_EXP(tmp) = __gmpfr_extract_double (tmpmant, d);

#ifdef WANT_ASSERT
  /* Failed assertion if the stored value is 0 (e.g., if the exponent range
     has been reduced at the wrong moment and an underflow to 0 occurred).
     Probably a bug in the C implementation if this happens. */
  i = 0;
  while (tmpmant[i] == 0)
    {
      i++;
      MPFR_ASSERTN(i < MPFR_LIMBS_PER_DOUBLE);
    }
#endif

  /* determine the index i-1 of the most significant non-zero limb
     and the number k of zero high limbs */
  i = MPFR_LIMBS_PER_DOUBLE;
  MPN_NORMALIZE_NOT_ZERO(tmpmant, i);
  k = MPFR_LIMBS_PER_DOUBLE - i;

  count_leading_zeros (cnt, tmpmant[i - 1]);

  if (MPFR_LIKELY(cnt != 0))
    mpn_lshift (tmpmant + k, tmpmant, i, cnt);
  else if (k != 0)
    MPN_COPY (tmpmant + k, tmpmant, i);

  if (MPFR_UNLIKELY(k != 0))
    MPN_ZERO (tmpmant, k);

  /* don't use MPFR_SET_EXP here since the exponent may be out of range */
  MPFR_EXP(tmp) -= (mpfr_exp_t) (cnt + k * GMP_NUMB_BITS);

  /* tmp is exact since PREC(tmp)=53 */
  inexact = mpfr_set4 (r, tmp, rnd_mode, signd);

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (r, inexact, rnd_mode);
}



