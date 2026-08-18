// ---------------------------------------------------------------------------------
// mpfr_dtq_sd.c
// These following codes are originally written by members of MPFR development team.
// Modified by Tomonori Kouya
// All codes can be used under MPFR's license.
// ---------------------------------------------------------------------------------
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "mpfr.h"
#include "mpc.h" // 2025-06-26(Thu)
#include "mpfr_dtq_sd.h"

#ifdef USE_GMP

// FALSE(=0), TRUE(=1)
//#define MPFR_IS_SINGULAR(x) (MPFR_EXP(x) <= MPFR_EXP_INF)
int mpfr_is_singular(mpfr_srcptr x)
{
	return (mpfr_get_exp(x) <= _IN_H_MPFR_EXP_INF);
}

/* generic code */
// ret_dd[2] = ret_dd[high == 0], ret_dd[low == 1]
//#define mpf_get_dd(ret_dd, x) mpfr_get_dd(ret_dd, x, MPFR_RNDN)
void mpfr_get_dd(double ret_dd[2], mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
	//if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
	if (mpfr_is_singular(x))
	{
		//return (long double) mpfr_get_d (x, rnd_mode);
		ret_dd[0] = mpfr_get_d(x, rnd_mode);
		ret_dd[1] = (double)0.0;
	}
	else /* now x is a normal non-zero number */
	{
		// long double r; /* result */
		double s; /* part of result */

		//MPFR_SAVE_EXPO_DECL (expo);
		//MPFR_SAVE_EXPO_MARK (expo);

		//#if defined(HAVE_LDOUBLE_MAYBE_DOUBLE_DOUBLE)
		//if (MPFR_LDBL_MANT_DIG == 106)
		//{
		/* Assume double-double format (as found with the PowerPC ABI).
		   The generic code below isn't used because numbers with
		   precision > 106 would not be supported. */
		s = mpfr_get_d (x, MPFR_RNDN); /* high part of x */
		/* Let's first consider special cases separately. The test for
		   infinity is really needed to avoid a NaN result. The test
		   for NaN is mainly for optimization. The test for 0 is useful
		   to get the correct sign (assuming mpfr_get_d supports signed
		   zeros on the implementation). */
		//if (s == 0 || DOUBLE_ISNAN (s) || DOUBLE_ISINF (s))
		if (s == 0 || isnan(s) || isinf(s))
		{
			// r = (long double) s;
			ret_dd[0] = s;
			ret_dd[1] = 0;
		}
		else
		{
			mpfr_t y, z;
			
			mpfr_init2 (y, mpfr_get_prec (x));
			mpfr_init2 (z, 53); /* keep the precision small */
			mpfr_set_d (z, s, MPFR_RNDN);  /* exact */

			// y := x - z = x - s
			mpfr_sub (y, x, z, MPFR_RNDN); /* exact */

			/* Add the second part of y (in the correct rounding mode). */
			//r = (long double) s + (long double) mpfr_get_d (y, rnd_mode);
			ret_dd[0] = s;
			ret_dd[1] = mpfr_get_d(y, rnd_mode);

			mpfr_clear (z);
			mpfr_clear (y);
		}
		//}
		//MPFR_SAVE_EXPO_FREE (expo);
		//return r;
	}
}

/* generic code */
// ret_ds[2] = ret_ds[high == 0], ret_ds[low == 1]
void mpfr_get_ds(float ret_ds[2], mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
	//if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
	if (mpfr_is_singular(x))
	{
		//return (long double) mpfr_get_d (x, rnd_mode);
		ret_ds[0] = mpfr_get_flt(x, rnd_mode);
		ret_ds[1] = (float)0.0;
	}
	else /* now x is a normal non-zero number */
	{
		// long double r; /* result */
		float s; /* part of result */

		//MPFR_SAVE_EXPO_DECL (expo);
		//MPFR_SAVE_EXPO_MARK (expo);

		//#if defined(HAVE_LDOUBLE_MAYBE_DOUBLE_DOUBLE)
		//if (MPFR_LDBL_MANT_DIG == 106)
		//{
		/* Assume double-double format (as found with the PowerPC ABI).
		   The generic code below isn't used because numbers with
		   precision > 106 would not be supported. */
		s = mpfr_get_flt(x, MPFR_RNDN); /* high part of x */
		/* Let's first consider special cases separately. The test for
		   infinity is really needed to avoid a NaN result. The test
		   for NaN is mainly for optimization. The test for 0 is useful
		   to get the correct sign (assuming mpfr_get_d supports signed
		   zeros on the implementation). */
		//if (s == 0 || DOUBLE_ISNAN (s) || DOUBLE_ISINF (s))
		if (s == 0 || isnanf(s) || isinff(s))
		{
			// r = (long double) s;
			ret_ds[0] = s;
			ret_ds[1] = 0;
		}
		else
		{
			mpfr_t y, z;
			
			mpfr_init2(y, mpfr_get_prec (x));
			mpfr_init2(z, 24); /* keep the precision small */
			mpfr_set_flt(z, s, MPFR_RNDN);  /* exact */

			// y := x - z = x - s
			mpfr_sub(y, x, z, MPFR_RNDN); /* exact */

			/* Add the second part of y (in the correct rounding mode). */
			//r = (long double) s + (long double) mpfr_get_d (y, rnd_mode);
			ret_ds[0] = s;
			ret_ds[1] = mpfr_get_flt(y, rnd_mode);

			mpfr_clear(z);
			mpfr_clear(y);
		}
		//}
		//MPFR_SAVE_EXPO_FREE (expo);
		//return r;
	}
}

#ifdef __cplusplus
void mpf_get_dd(dd_real *ret, mpf_t val)
{
	mpfr_get_dd(ret->x, val, MPFR_RNDN);
}
#endif //__cpluplus


/* double-double code */
//#define mpf_set_dd(r, d) mpfr_set_dd(r, d, MPFR_RNDN)
int mpfr_set_dd(mpfr_ptr r, const double d[2], mpfr_rnd_t rnd_mode)
{
	mpfr_t t, u;
	int inexact, shift_exp;
	double h, l;
	//  MPFR_SAVE_EXPO_DECL (expo);
	
	/* Check for NAN */
	//ONGDOUBLE_NAN_ACTION (d, goto nan);
	if(isnan(d[0])) goto nan;

	/* Check for INF */
	//  if (d > MPFR_LDBL_MAX)
	if (d[0] > DBL_MAX)
	{
		mpfr_set_inf (r, 1);
		return 0;
	}
	else if (d[0] < -DBL_MAX)   //else if (d < -MPFR_LDBL_MAX)
	{
		mpfr_set_inf (r, -1);
	    return 0;
	}
	/* Check for ZERO */
	else if (d[0] == 0.0)
		return mpfr_set_d (r, (double) d[0], rnd_mode);

	// if (d >= (long double) MPFR_LDBL_MAX || d <= (long double) -MPFR_LDBL_MAX)
	//   h = (d >= (long double) MPFR_LDBL_MAX) ? MPFR_LDBL_MAX : -MPFR_LDBL_MAX;
	if (d[0] >= DBL_MAX || d[0] <= -DBL_MAX)
		h = (d[0] >= DBL_MAX) ? DBL_MAX : -DBL_MAX;
	else
		h = (double) d[0]; /* should not overflow */

	l = d[1]; //l = (double) (d - (long double) h);

	//MPFR_SAVE_EXPO_MARK (expo);

	mpfr_init2 (t, 53);
	mpfr_init2 (u, 53);

	inexact = mpfr_set_d (t, h, MPFR_RNDN);
	//MPFR_ASSERTN(inexact == 0);
	inexact = mpfr_set_d (u, l, MPFR_RNDN);
	//MPFR_ASSERTN(inexact == 0);
	inexact = mpfr_add (r, t, u, rnd_mode);

	mpfr_clear (t);
	mpfr_clear (u);

	//  MPFR_SAVE_EXPO_FREE (expo);
	inexact = mpfr_check_range (r, inexact, rnd_mode);
	return inexact;
	
nan:
	mpfr_set_nan(r); //MPFR_SET_NAN(r);
	//return MPFR_RET_NAN;
	return 0;

}

/* double-single code */
int mpfr_set_ds(mpfr_ptr r, const float d[2], mpfr_rnd_t rnd_mode)
{
	mpfr_t t, u;
	int inexact, shift_exp;
	float h, l;
	//  MPFR_SAVE_EXPO_DECL (expo);
	
	/* Check for NAN */
	//ONGDOUBLE_NAN_ACTION (d, goto nan);
	if(isnanf(d[0])) goto nan;

	/* Check for INF */
	//  if (d > MPFR_LDBL_MAX)
	if (d[0] > FLT_MAX)
	{
		mpfr_set_inf(r, 1);
		return 0;
	}
	else if (d[0] < -FLT_MAX)   //else if (d < -MPFR_LDBL_MAX)
	{
		mpfr_set_inf (r, -1);
	    return 0;
	}
	/* Check for ZERO */
	else if (d[0] == 0.0)
		return mpfr_set_flt(r, (float)d[0], rnd_mode);

	// if (d >= (long double) MPFR_LDBL_MAX || d <= (long double) -MPFR_LDBL_MAX)
	//   h = (d >= (long double) MPFR_LDBL_MAX) ? MPFR_LDBL_MAX : -MPFR_LDBL_MAX;
	if (d[0] >= FLT_MAX || d[0] <= -FLT_MAX)
		h = (d[0] >= FLT_MAX) ? FLT_MAX : -FLT_MAX;
	else
		h = (float)d[0]; /* should not overflow */

	l = d[1]; //l = (double) (d - (long double) h);

	//MPFR_SAVE_EXPO_MARK (expo);

	mpfr_init2 (t, 24);
	mpfr_init2 (u, 24);

	inexact = mpfr_set_flt(t, h, MPFR_RNDN);
	//MPFR_ASSERTN(inexact == 0);
	inexact = mpfr_set_flt(u, l, MPFR_RNDN);
	//MPFR_ASSERTN(inexact == 0);
	inexact = mpfr_add(r, t, u, rnd_mode);

	mpfr_clear (t);
	mpfr_clear (u);

	//  MPFR_SAVE_EXPO_FREE (expo);
	inexact = mpfr_check_range(r, inexact, rnd_mode);
	return inexact;
	
nan:
	mpfr_set_nan(r); //MPFR_SET_NAN(r);
	//return MPFR_RET_NAN;
	return 0;

}
#ifdef __cplusplus
int mpf_set_dd(mpf_t ret, dd_real val)
{
	mpfr_set_dd(ret, val.x, MPFR_RNDN);
}
#endif //__cpluplus

/* generic code */
// Triple-double precision
// ret_qd[3] = ret_qd[high == 0], ret_qd[1], ret_qd[2]
//#define mpf_get_td(ret_td, x) mpfr_get_td(ret_td, x, MPFR_RNDN)
void mpfr_get_td(double ret_td[3], mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
	int i;
	double s;

	//if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
	if (mpfr_is_singular(x))
	{
		//return (long double) mpfr_get_d (x, rnd_mode);
		ret_td[0] = mpfr_get_d(x, rnd_mode);
		ret_td[1] = (double)0.0;
		ret_td[2] = (double)0.0;
	}
	else /* now x is a normal non-zero number */
	{
		//#if defined(HAVE_LDOUBLE_MAYBE_DOUBLE_DOUBLE)
		//if (MPFR_LDBL_MANT_DIG == 106)
		//{
		/* Assume double-double format (as found with the PowerPC ABI).
		   The generic code below isn't used because numbers with
		   precision > 106 would not be supported. */

		s = mpfr_get_d (x, MPFR_RNDN); /* high part of x */

		/* Let's first consider special cases separately. The test for
		   infinity is really needed to avoid a NaN result. The test
		   for NaN is mainly for optimization. The test for 0 is useful
		   to get the correct sign (assuming mpfr_get_d supports signed
		   zeros on the implementation). */
		//if (s == 0 || DOUBLE_ISNAN (s) || DOUBLE_ISINF (s))

		if (s == 0 || isnan(s) || isinf(s))
		{
			// r = (long double) s;
			ret_td[0] = s;
			ret_td[1] = 0;
			ret_td[1] = 0;
		}
		else
		{
			mpfr_t y, z[2];
			
			mpfr_init2 (y, mpfr_get_prec (x));
			mpfr_init2 (z[0], 53); /* keep the precision small */
			mpfr_init2 (z[1], 53); /* keep the precision small */
			
			// y[0] := x - z = x - s
			mpfr_set_d (z[0], s, MPFR_RNDN);  /* exact */
			mpfr_sub (y, x, z[0], MPFR_RNDN); /* exact */
			ret_td[0] = s;
			
			// y[1] := y[0] - z = x - s
			s = mpfr_get_d (y, MPFR_RNDN); /* high part of y */
			mpfr_set_d (z[1], s, MPFR_RNDN);  /* exact */
			mpfr_sub (y, x, z[0], MPFR_RNDN); /* exact */
			mpfr_sub (y, y, z[1], MPFR_RNDN); /* exact */
			ret_td[1] = s;
			
			// y[2] := x - z = x - s
			ret_td[2] = mpfr_get_d(y, rnd_mode);

			mpfr_clear (z[0]);
			mpfr_clear (z[1]);
			mpfr_clear (y);
		}
		//}
		//MPFR_SAVE_EXPO_FREE (expo);
		//return r;
	}
}

/* generic code */
// Triple-single precision
// ret_ts[3] = ret_ts[high == 0], ret_ts[1], ret_ts[2]
//#define mpf_get_ts(ret_ts, x) mpfr_get_ts(ret_ts, x, MPFR_RNDN)
void mpfr_get_ts(float ret_ts[TSSIZE], mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
	int i;
	float s;

	//if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
	if (mpfr_is_singular(x))
	{
		//return (long double) mpfr_get_d (x, rnd_mode);
		ret_ts[0] = mpfr_get_flt(x, rnd_mode);
		ret_ts[1] = (float)0.0;
		ret_ts[2] = (float)0.0;
	}
	else /* now x is a normal non-zero number */
	{
		//#if defined(HAVE_LDOUBLE_MAYBE_DOUBLE_DOUBLE)
		//if (MPFR_LDBL_MANT_DIG == 106)
		//{
		/* Assume double-double format (as found with the PowerPC ABI).
		   The generic code below isn't used because numbers with
		   precision > 106 would not be supported. */

		s = mpfr_get_flt(x, MPFR_RNDN); /* high part of x */

		/* Let's first consider special cases separately. The test for
		   infinity is really needed to avoid a NaN result. The test
		   for NaN is mainly for optimization. The test for 0 is useful
		   to get the correct sign (assuming mpfr_get_d supports signed
		   zeros on the implementation). */
		//if (s == 0 || DOUBLE_ISNAN (s) || DOUBLE_ISINF (s))

		if (s == 0 || isnanf(s) || isinff(s))
		{
			// r = (long double) s;
			ret_ts[0] = s;
			ret_ts[1] = 0;
			ret_ts[1] = 0;
		}
		else
		{
			mpfr_t y, z[2];
			
			mpfr_init2(y, mpfr_get_prec (x));
			mpfr_init2(z[0], 24); /* keep the precision small */
			mpfr_init2(z[1], 24); /* keep the precision small */
			
			// y[0] := x - z = x - s
			mpfr_set_flt(z[0], s, MPFR_RNDN);  /* exact */
			mpfr_sub(y, x, z[0], MPFR_RNDN); /* exact */
			ret_ts[0] = s;
			
			// y[1] := y[0] - z = x - s
			s = mpfr_get_flt(y, MPFR_RNDN); /* high part of y */
			mpfr_set_flt(z[1], s, MPFR_RNDN);  /* exact */
			mpfr_sub(y, x, z[0], MPFR_RNDN); /* exact */
			mpfr_sub(y, y, z[1], MPFR_RNDN); /* exact */
			ret_ts[1] = s;
			
			// y[2] := x - z = x - s
			ret_ts[2] = mpfr_get_flt(y, rnd_mode);

			mpfr_clear(z[0]);
			mpfr_clear(z[1]);
			mpfr_clear(y);
		}
		//}
		//MPFR_SAVE_EXPO_FREE (expo);
		//return r;
	}
}

/* Triple-double code */
//#define mpf_set_td(r, d) mpfr_set_td(r, d, MPFR_RNDN)
int mpfr_set_td(mpfr_ptr r, const double d[3], mpfr_rnd_t rnd_mode)
{
	mpfr_t t, u, v, w;
	int inexact, shift_exp;
	double h, l[3];
	//  MPFR_SAVE_EXPO_DECL (expo);
	
	/* Check for NAN */
	//ONGDOUBLE_NAN_ACTION (d, goto nan);
	if(isnan(d[0])) goto nan;

	/* Check for INF */
	//  if (d > MPFR_LDBL_MAX)
	if (d[0] > DBL_MAX)
	{
		mpfr_set_inf (r, 1);
		return 0;
	}
	else if (d[0] < -DBL_MAX)	//else if (d < -MPFR_LDBL_MAX)
	{
		mpfr_set_inf (r, -1);
	    return 0;
	}
	/* Check for ZERO */
	else if (d[0] == 0.0)
		return mpfr_set_d (r, (double) d[0], rnd_mode);

	// if (d >= (long double) MPFR_LDBL_MAX || d <= (long double) -MPFR_LDBL_MAX)
	//   h = (d >= (long double) MPFR_LDBL_MAX) ? MPFR_LDBL_MAX : -MPFR_LDBL_MAX;
	if (d[0] >= DBL_MAX || d[0] <= -DBL_MAX)
		h = (d[0] >= DBL_MAX) ? DBL_MAX : -DBL_MAX;
	else
		h = (double) d[0]; /* should not overflow */

	//l = (double) (d - (long double) h);
	l[0] = d[1];
	l[1] = d[2];
	
	//MPFR_SAVE_EXPO_MARK (expo);
	
	mpfr_init2 (t, 53);
	mpfr_init2 (u, 53);
	mpfr_init2 (v, 53);
	
	inexact = mpfr_set_d (t, h, MPFR_RNDN);
	inexact = mpfr_set_d (u, l[0], MPFR_RNDN);
	inexact = mpfr_set_d (v, l[1], MPFR_RNDN);
	
	inexact = mpfr_add (r, t, u, rnd_mode);
	inexact = mpfr_add (r, r, v, rnd_mode);
	
	mpfr_clear (t);
	mpfr_clear (u);
	mpfr_clear (v);
	
	//  MPFR_SAVE_EXPO_FREE (expo);
	inexact = mpfr_check_range (r, inexact, rnd_mode);
	return inexact;

nan:
	mpfr_set_nan(r); //MPFR_SET_NAN(r);
	//return MPFR_RET_NAN;

	return 0;
}

/* Triple-single code */
//#define mpf_set_ts(r, d) mpfr_set_ts(r, d, MPFR_RNDN)
int mpfr_set_ts(mpfr_ptr r, const float s[3], mpfr_rnd_t rnd_mode)
{
	mpfr_t t, u, v, w;
	int inexact, shift_exp;
	float h, l[3];
	//  MPFR_SAVE_EXPO_DECL (expo);
	
	/* Check for NAN */
	//ONGDOUBLE_NAN_ACTION (d, goto nan);
	if(isnanf(s[0])) goto nan;

	/* Check for INF */
	//  if (d > MPFR_LDBL_MAX)
	if (s[0] > FLT_MAX)
	{
		mpfr_set_inf(r, 1);
		return 0;
	}
	else if (s[0] < -FLT_MAX)	//else if (d < -MPFR_LDBL_MAX)
	{
		mpfr_set_inf(r, -1);
	    return 0;
	}
	/* Check for ZERO */
	else if (s[0] == 0.0)
		return mpfr_set_flt(r, (float)s[0], rnd_mode);

	// if (d >= (long double) MPFR_LDBL_MAX || d <= (long double) -MPFR_LDBL_MAX)
	//   h = (d >= (long double) MPFR_LDBL_MAX) ? MPFR_LDBL_MAX : -MPFR_LDBL_MAX;
	if (s[0] >= FLT_MAX || s[0] <= -FLT_MAX)
		h = (s[0] >= FLT_MAX) ? FLT_MAX : -FLT_MAX;
	else
		h = (float)s[0]; /* should not overflow */

	//l = (double) (d - (long double) h);
	l[0] = s[1];
	l[1] = s[2];
	
	//MPFR_SAVE_EXPO_MARK (expo);
	
	mpfr_init2(t, 24);
	mpfr_init2(u, 24);
	mpfr_init2(v, 24);
	
	inexact = mpfr_set_flt(t, h, MPFR_RNDN);
	inexact = mpfr_set_flt(u, l[0], MPFR_RNDN);
	inexact = mpfr_set_flt(v, l[1], MPFR_RNDN);
	
	inexact = mpfr_add(r, t, u, rnd_mode);
	inexact = mpfr_add(r, r, v, rnd_mode);
	
	mpfr_clear(t);
	mpfr_clear(u);
	mpfr_clear(v);
	
	//  MPFR_SAVE_EXPO_FREE (expo);
	inexact = mpfr_check_range(r, inexact, rnd_mode);
	return inexact;

nan:
	mpfr_set_nan(r); //MPFR_SET_NAN(r);
	//return MPFR_RET_NAN;

	return 0;
}


/* generic code */
// ret_qd[4] = ret_qd[high == 0], ret_qd[1], ret_qd[2], ret_qd[3]
//#define mpf_get_qd(ret_qd, x) mpfr_get_qd(ret_qd, x, MPFR_RNDN)
void mpfr_get_qd(double ret_qd[4], mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
	int i;
	double s;

	//if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
	if (mpfr_is_singular(x))
	{
		//return (long double) mpfr_get_d (x, rnd_mode);
		ret_qd[0] = mpfr_get_d(x, rnd_mode);
		ret_qd[1] = (double)0.0;
		ret_qd[2] = (double)0.0;
		ret_qd[3] = (double)0.0;
	}
	else /* now x is a normal non-zero number */
	{
		//#if defined(HAVE_LDOUBLE_MAYBE_DOUBLE_DOUBLE)
		//if (MPFR_LDBL_MANT_DIG == 106)
		//{
		/* Assume double-double format (as found with the PowerPC ABI).
		   The generic code below isn't used because numbers with
		   precision > 106 would not be supported. */

		s = mpfr_get_d (x, MPFR_RNDN); /* high part of x */

		/* Let's first consider special cases separately. The test for
		   infinity is really needed to avoid a NaN result. The test
		   for NaN is mainly for optimization. The test for 0 is useful
		   to get the correct sign (assuming mpfr_get_d supports signed
		   zeros on the implementation). */
		//if (s == 0 || DOUBLE_ISNAN (s) || DOUBLE_ISINF (s))

		if (s == 0 || isnan(s) || isinf(s))
		{
			// r = (long double) s;
			ret_qd[0] = s;
			ret_qd[1] = 0;
			ret_qd[1] = 0;
			ret_qd[1] = 0;
		}
		else
		{
			mpfr_t y, z[3];
			
			mpfr_init2 (y, mpfr_get_prec (x));
			mpfr_init2 (z[0], 53); /* keep the precision small */
			mpfr_init2 (z[1], 53); /* keep the precision small */
			mpfr_init2 (z[2], 53); /* keep the precision small */
			
			// y[0] := x - z = x - s
			mpfr_set_d (z[0], s, MPFR_RNDN);  /* exact */
			mpfr_sub (y, x, z[0], MPFR_RNDN); /* exact */
			ret_qd[0] = s;
			
			// y[1] := y[0] - z = x - s
			s = mpfr_get_d (y, MPFR_RNDN); /* high part of y */
			mpfr_set_d (z[1], s, MPFR_RNDN);  /* exact */
			mpfr_sub (y, x, z[0], MPFR_RNDN); /* exact */
			mpfr_sub (y, y, z[1], MPFR_RNDN); /* exact */
			ret_qd[1] = s;
			
			// y[2] := y[1] - z = x - s
			s = mpfr_get_d (y, MPFR_RNDN); /* high part of y */
			mpfr_set_d (z[2], s, MPFR_RNDN);  /* exact */
			mpfr_sub (y, x, z[0], MPFR_RNDN); /* exact */
			mpfr_sub (y, y, z[1], MPFR_RNDN); /* exact */
			mpfr_sub (y, y, z[2], MPFR_RNDN); /* exact */
			ret_qd[2] = s;

			// y[3] := x - z = x - s
			ret_qd[3] = mpfr_get_d(y, rnd_mode);

			mpfr_clear (z[0]);
			mpfr_clear (z[1]);
			mpfr_clear (z[2]);
			mpfr_clear (y);
		}
		//}
		//MPFR_SAVE_EXPO_FREE (expo);
		//return r;
	}
}
#ifdef __cplusplus
void mpf_get_qd(qd_real *ret, mpf_t val)
{
	mpfr_get_qd(ret->x, val, MPFR_RNDN);
}
#endif //__cpluplus

/* generic code */
// ret_qs[4] = ret_qs[high == 0], ret_qs[1], ret_qs[2], ret_qs[3]
//#define mpf_get_qs(ret_qs, x) mpfr_get_qs(ret_qs, x, MPFR_RNDN)
void mpfr_get_qs(float ret_qs[QSSIZE], mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
	int i;
	float s;

	//if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
	if (mpfr_is_singular(x))
	{
		//return (long double) mpfr_get_d (x, rnd_mode);
		ret_qs[0] = mpfr_get_flt(x, rnd_mode);
		ret_qs[1] = (float)0.0;
		ret_qs[2] = (float)0.0;
		ret_qs[3] = (float)0.0;
	}
	else /* now x is a normal non-zero number */
	{
		//#if defined(HAVE_LDOUBLE_MAYBE_DOUBLE_DOUBLE)
		//if (MPFR_LDBL_MANT_DIG == 106)
		//{
		/* Assume double-double format (as found with the PowerPC ABI).
		   The generic code below isn't used because numbers with
		   precision > 106 would not be supported. */

		s = mpfr_get_flt(x, MPFR_RNDN); /* high part of x */

		/* Let's first consider special cases separately. The test for
		   infinity is really needed to avoid a NaN result. The test
		   for NaN is mainly for optimization. The test for 0 is useful
		   to get the correct sign (assuming mpfr_get_d supports signed
		   zeros on the implementation). */
		//if (s == 0 || DOUBLE_ISNAN (s) || DOUBLE_ISINF (s))

		if (s == 0 || isnanf(s) || isinff(s))
		{
			// r = (long double) s;
			ret_qs[0] = s;
			ret_qs[1] = 0;
			ret_qs[1] = 0;
			ret_qs[1] = 0;
		}
		else
		{
			mpfr_t y, z[3];
			
			mpfr_init2(y, mpfr_get_prec (x));
			mpfr_init2(z[0], 24); /* keep the precision small */
			mpfr_init2(z[1], 24); /* keep the precision small */
			mpfr_init2(z[2], 24); /* keep the precision small */
			
			// y[0] := x - z = x - s
			mpfr_set_flt(z[0], s, MPFR_RNDN);  /* exact */
			mpfr_sub(y, x, z[0], MPFR_RNDN); /* exact */
			ret_qs[0] = s;
			
			// y[1] := y[0] - z = x - s
			s = mpfr_get_flt(y, MPFR_RNDN); /* high part of y */
			mpfr_set_flt(z[1], s, MPFR_RNDN);  /* exact */
			mpfr_sub(y, x, z[0], MPFR_RNDN); /* exact */
			mpfr_sub(y, y, z[1], MPFR_RNDN); /* exact */
			ret_qs[1] = s;
			
			// y[2] := y[1] - z = x - s
			s = mpfr_get_flt(y, MPFR_RNDN); /* high part of y */
			mpfr_set_flt(z[2], s, MPFR_RNDN);  /* exact */
			mpfr_sub(y, x, z[0], MPFR_RNDN); /* exact */
			mpfr_sub(y, y, z[1], MPFR_RNDN); /* exact */
			mpfr_sub(y, y, z[2], MPFR_RNDN); /* exact */
			ret_qs[2] = s;

			// y[3] := x - z = x - s
			ret_qs[3] = mpfr_get_flt(y, rnd_mode);

			mpfr_clear(z[0]);
			mpfr_clear(z[1]);
			mpfr_clear(z[2]);
			mpfr_clear(y);
		}
		//}
		//MPFR_SAVE_EXPO_FREE (expo);
		//return r;
	}
}

/* quad-double code */
//#define mpf_set_qd(r, d) mpfr_set_qd(r, d, MPFR_RNDN)
int mpfr_set_qd(mpfr_ptr r, const double d[4], mpfr_rnd_t rnd_mode)
{
	mpfr_t t, u, v, w;
	int inexact, shift_exp;
	double h, l[3];
	//  MPFR_SAVE_EXPO_DECL (expo);
	
	/* Check for NAN */
	//ONGDOUBLE_NAN_ACTION (d, goto nan);
	if(isnan(d[0])) goto nan;

	/* Check for INF */
	//  if (d > MPFR_LDBL_MAX)
	if (d[0] > DBL_MAX)
	{
		mpfr_set_inf (r, 1);
		return 0;
	}
	else if (d[0] < -DBL_MAX)	//else if (d < -MPFR_LDBL_MAX)
	{
		mpfr_set_inf (r, -1);
	    return 0;
	}
	/* Check for ZERO */
	else if (d[0] == 0.0)
		return mpfr_set_d (r, (double) d[0], rnd_mode);

	// if (d >= (long double) MPFR_LDBL_MAX || d <= (long double) -MPFR_LDBL_MAX)
	//   h = (d >= (long double) MPFR_LDBL_MAX) ? MPFR_LDBL_MAX : -MPFR_LDBL_MAX;
	if (d[0] >= DBL_MAX || d[0] <= -DBL_MAX)
		h = (d[0] >= DBL_MAX) ? DBL_MAX : -DBL_MAX;
	else
		h = (double) d[0]; /* should not overflow */

	//l = (double) (d - (long double) h);
	l[0] = d[1];
	l[1] = d[2];
	l[2] = d[3];
	
	//MPFR_SAVE_EXPO_MARK (expo);
	
	mpfr_init2 (t, 53);
	mpfr_init2 (u, 53);
	mpfr_init2 (v, 53);
	mpfr_init2 (w, 53);
	
	inexact = mpfr_set_d (t, h, MPFR_RNDN);
	inexact = mpfr_set_d (u, l[0], MPFR_RNDN);
	inexact = mpfr_set_d (v, l[1], MPFR_RNDN);
	inexact = mpfr_set_d (w, l[2], MPFR_RNDN);
	
	inexact = mpfr_add (r, t, u, rnd_mode);
	inexact = mpfr_add (r, r, v, rnd_mode);
	inexact = mpfr_add (r, r, w, rnd_mode);
	
	mpfr_clear (t);
	mpfr_clear (u);
	mpfr_clear (v);
	mpfr_clear (w);
	
	//  MPFR_SAVE_EXPO_FREE (expo);
	inexact = mpfr_check_range (r, inexact, rnd_mode);
	return inexact;

nan:
	mpfr_set_nan(r); //MPFR_SET_NAN(r);
	//return MPFR_RET_NAN;

	return 0;
}
#ifdef __cplusplus
int mpf_set_qd(mpf_t ret, qd_real val)
{
	mpfr_set_qd(ret, val.x, MPFR_RNDN);
}
#endif //__cpluplus

/* quad-single code */
//#define mpf_set_qs(r, d) mpfr_set_qs(r, d, MPFR_RNDN)
int mpfr_set_qs(mpfr_ptr r, const float d[QSSIZE], mpfr_rnd_t rnd_mode)
{
	mpfr_t t, u, v, w;
	int inexact, shift_exp;
	double h, l[3];
	//  MPFR_SAVE_EXPO_DECL (expo);
	
	/* Check for NAN */
	//ONGDOUBLE_NAN_ACTION (d, goto nan);
	if(isnanf(d[0])) goto nan;

	/* Check for INF */
	//  if (d > MPFR_LDBL_MAX)
	if (d[0] > FLT_MAX)
	{
		mpfr_set_inf(r, 1);
		return 0;
	}
	else if (d[0] < -FLT_MAX)	//else if (d < -MPFR_LDBL_MAX)
	{
		mpfr_set_inf(r, -1);
	    return 0;
	}
	/* Check for ZERO */
	else if (d[0] == 0.0)
		return mpfr_set_flt(r, (float)d[0], rnd_mode);

	// if (d >= (long double) MPFR_LDBL_MAX || d <= (long double) -MPFR_LDBL_MAX)
	//   h = (d >= (long double) MPFR_LDBL_MAX) ? MPFR_LDBL_MAX : -MPFR_LDBL_MAX;
	if (d[0] >= FLT_MAX || d[0] <= -FLT_MAX)
		h = (d[0] >= FLT_MAX) ? FLT_MAX : -FLT_MAX;
	else
		h = (float)d[0]; /* should not overflow */

	//l = (double) (d - (long double) h);
	l[0] = d[1];
	l[1] = d[2];
	l[2] = d[3];
	
	//MPFR_SAVE_EXPO_MARK (expo);
	
	mpfr_init2(t, 24);
	mpfr_init2(u, 24);
	mpfr_init2(v, 24);
	mpfr_init2(w, 24);
	
	inexact = mpfr_set_flt(t, h, MPFR_RNDN);
	inexact = mpfr_set_flt(u, l[0], MPFR_RNDN);
	inexact = mpfr_set_flt(v, l[1], MPFR_RNDN);
	inexact = mpfr_set_flt(w, l[2], MPFR_RNDN);
	
	inexact = mpfr_add(r, t, u, rnd_mode);
	inexact = mpfr_add(r, r, v, rnd_mode);
	inexact = mpfr_add(r, r, w, rnd_mode);
	
	mpfr_clear(t);
	mpfr_clear(u);
	mpfr_clear(v);
	mpfr_clear(w);
	
	//  MPFR_SAVE_EXPO_FREE (expo);
	inexact = mpfr_check_range(r, inexact, rnd_mode);
	return inexact;

nan:
	mpfr_set_nan(r); //MPFR_SET_NAN(r);
	//return MPFR_RET_NAN;

	return 0;
}

// 2025-06-26(Thu)
// -------------------------
// mpc to cdd, ctd, cqdfloat
// -------------------------
// MPC -> cddfloat
void mpc_get_cdd(cddfloat *ret, mpc_t org)
{
	mpfr_get_dd(ret->val_re, mpc_realref(org), MPFR_DEFAULT_RND);
	mpfr_get_dd(ret->val_im, mpc_imagref(org), MPFR_DEFAULT_RND);
}
// cddfloat -> MPC
void mpc_set_cdd(mpc_t ret, cddfloat *org)
{
	mpfr_set_dd(mpc_realref(ret), org->val_re, MPFR_DEFAULT_RND);
	mpfr_set_dd(mpc_imagref(ret), org->val_im, MPFR_DEFAULT_RND);
}
// MPC -> ctdfloat
void mpc_get_ctd(ctdfloat *ret, mpc_t org)
{
	mpfr_get_td(ret->val_re, mpc_realref(org), MPFR_DEFAULT_RND);
	mpfr_get_td(ret->val_im, mpc_imagref(org), MPFR_DEFAULT_RND);
}
// ctdfloat -> MPC
void mpc_set_ctd(mpc_t ret, ctdfloat *org)
{
	mpfr_set_td(mpc_realref(ret), org->val_re, MPFR_DEFAULT_RND);
	mpfr_set_td(mpc_imagref(ret), org->val_im, MPFR_DEFAULT_RND);
}
// MPC -> cqdfloat
void mpc_get_cqd(cqdfloat *ret, mpc_t org)
{
	mpfr_get_qd(ret->val_re, mpc_realref(org), MPFR_DEFAULT_RND);
	mpfr_get_qd(ret->val_im, mpc_imagref(org), MPFR_DEFAULT_RND);
}
// cqdfloat -> MPC
void mpc_set_cqd(mpc_t ret, cqdfloat *org)
{
	mpfr_set_qd(mpc_realref(ret), org->val_re, MPFR_DEFAULT_RND);
	mpfr_set_qd(mpc_imagref(ret), org->val_im, MPFR_DEFAULT_RND);
}

// rdd_out_str(dd_a)
void rdd_out_str(double dd[DDSIZE])
{
	mpfr_t tmp;
	mpfr_init2(tmp, 53 * DDSIZE);

	mpfr_set_dd(tmp, dd, MPFR_RNDN);
	mpfr_out_str(stdout, 10, (size_t)(16 * DDSIZE), tmp, MPFR_RNDN);

	mpfr_clear(tmp);
}

// rds_out_str(ds_a)
void rds_out_str(float ds[DSSIZE])
{
	mpfr_t tmp;
	mpfr_init2(tmp, 24 * DSSIZE);

	mpfr_set_ds(tmp, ds, MPFR_RNDN);
	mpfr_out_str(stdout, 10, (size_t)(7 * DSSIZE), tmp, MPFR_RNDN);

	mpfr_clear(tmp);
}

// rtd_out_str(td_a)
void rtd_out_str(double td[TDSIZE])
{
	mpfr_t tmp;
	mpfr_init2(tmp, 53 * TDSIZE);

	mpfr_set_td(tmp, td, MPFR_RNDN);
	mpfr_out_str(stdout, 10, (size_t)(16 * TDSIZE), tmp, MPFR_RNDN);

	mpfr_clear(tmp);
}

// rts_out_str(td_a)
void rts_out_str(float ts[TSSIZE])
{
	mpfr_t tmp;
	mpfr_init2(tmp, 24 * TSSIZE);

	mpfr_set_ts(tmp, ts, MPFR_RNDN);
	mpfr_out_str(stdout, 10, (size_t)(7 * TSSIZE), tmp, MPFR_RNDN);

	mpfr_clear(tmp);
}

// rqd_out_str(qd_a)
void rqd_out_str(double qd[QDSIZE])
{
	mpfr_t tmp;
	mpfr_init2(tmp, 53 * QDSIZE);

	mpfr_set_qd(tmp, qd, MPFR_RNDN);
	mpfr_out_str(stdout, 10, (size_t)(16 * QDSIZE), tmp, MPFR_RNDN);

	mpfr_clear(tmp);
}

// rqs_out_str(qs_a)
void rqs_out_str(float qs[QSSIZE])
{
	mpfr_t tmp;
	mpfr_init2(tmp, 24 * QSSIZE);

	mpfr_set_qs(tmp, qs, MPFR_RNDN);
	mpfr_out_str(stdout, 10, (size_t)(7 * QSSIZE), tmp, MPFR_RNDN);

	mpfr_clear(tmp);
}

// 2021-07-15(Thu) Tomonori Kouya
// rdd_set_str
void rdd_set_str(double ret[DDSIZE], const char *str)
{
	mpfr_t tmp;
	mpfr_init2(tmp, 53 * DDSIZE);

	mpfr_set_str(tmp, str, 10, MPFR_RNDN);
	mpfr_get_dd(ret, tmp, MPFR_RNDN);

	mpfr_clear(tmp);
}

// rdd_get_str
void rdd_get_str(char *str, const double val[DDSIZE])
{
	mpfr_t tmp;
	mpfr_init2(tmp, 53 * DDSIZE);

	mpfr_set_dd(tmp, (const double *)val, MPFR_RNDN);
	mpfr_sprintf(str, "%40.32RNg", tmp);

	mpfr_clear(tmp);
}

// rtd_set_str
void rtd_set_str(double ret[TDSIZE], const char *str)
{
	mpfr_t tmp;
	mpfr_init2(tmp, 53 * TDSIZE);

	mpfr_set_str(tmp, str, 10, MPFR_RNDN);
	mpfr_get_td(ret, tmp, MPFR_RNDN);

	mpfr_clear(tmp);
}

// rtd_get_str
void rtd_get_str(char *str, const double val[TDSIZE])
{
	mpfr_t tmp;
	mpfr_init2(tmp, 53 * TDSIZE);

	mpfr_set_td(tmp, (const double *)val, MPFR_RNDN);
	mpfr_sprintf(str, "%56.48RNg", tmp);

	mpfr_clear(tmp);
}

// rqd_set_str
void rqd_set_str(double ret[QDSIZE], const char *str)
{
	mpfr_t tmp;
	mpfr_init2(tmp, 53 * QDSIZE);

	mpfr_set_str(tmp, str, 10, MPFR_RNDN);
	mpfr_get_qd(ret, tmp, MPFR_RNDN);

	mpfr_clear(tmp);
}

// rtd_get_str
void rqd_get_str(char *str, const double val[QDSIZE])
{
	mpfr_t tmp;
	mpfr_init2(tmp, 53 * QDSIZE);

	mpfr_set_qd(tmp, (const double *)val, MPFR_RNDN);
	mpfr_sprintf(str, "%72.64RNg", tmp);

	mpfr_clear(tmp);
}

// 2022-11-17
// ret := a^b
void rdd_pow_mpfr(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE])
{
    mpfr_t in_a, in_b, in_ret;

    mpfr_init2(in_a, 53 * DDSIZE); mpfr_set_dd(in_a, a, MPFR_RNDN);
    mpfr_init2(in_b, 53 * DDSIZE); mpfr_set_dd(in_b, b, MPFR_RNDN);
	mpfr_init2(in_ret, 53 * DDSIZE);

    mpfr_pow(in_ret, in_a, in_b, MPFR_RNDN);
    mpfr_get_dd(ret, in_ret, MPFR_RNDN);

    mpfr_clear(in_a);
    mpfr_clear(in_b);
    mpfr_clear(in_ret);
}

void rtd_pow_mpfr(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE])
{
    mpfr_t in_a, in_b, in_ret;

    mpfr_init2(in_a, 53 * TDSIZE); mpfr_set_td(in_a, a, MPFR_RNDN);
    mpfr_init2(in_b, 53 * TDSIZE); mpfr_set_td(in_b, b, MPFR_RNDN);
	mpfr_init2(in_ret, 53 * TDSIZE);

    mpfr_pow(in_ret, in_a, in_b, MPFR_RNDN);
    mpfr_get_td(ret, in_ret, MPFR_RNDN);

    mpfr_clear(in_a);
    mpfr_clear(in_b);
    mpfr_clear(in_ret);
}

void rqd_pow_mpfr(double ret[QDSIZE], double a[QDSIZE], double b[QDSIZE])
{
    mpfr_t in_a, in_b, in_ret;

    mpfr_init2(in_a, 53 * QDSIZE); mpfr_set_qd(in_a, a, MPFR_RNDN);
    mpfr_init2(in_b, 53 * QDSIZE); mpfr_set_qd(in_b, b, MPFR_RNDN);
	mpfr_init2(in_ret, 53 * QDSIZE);

    mpfr_pow(in_ret, in_a, in_b, MPFR_RNDN);
    mpfr_get_qd(ret, in_ret, MPFR_RNDN);

    mpfr_clear(in_a);
    mpfr_clear(in_b);
    mpfr_clear(in_ret);
}

// 2023-06-02 (Fri)
// ret := exp(x)
void rdd_exp_mpfr(double ret[DDSIZE], double x[DDSIZE])
{
    mpfr_t in_x, in_ret;

    mpfr_init2(in_x, 53 * DDSIZE); mpfr_set_dd(in_x, x, MPFR_RNDN);
	mpfr_init2(in_ret, 53 * DDSIZE);

    mpfr_exp(in_ret, in_x, MPFR_RNDN);
    mpfr_get_dd(ret, in_ret, MPFR_RNDN);

    mpfr_clear(in_x);
    mpfr_clear(in_ret);
}

void rtd_exp_mpfr(double ret[TDSIZE], double x[TDSIZE])
{
    mpfr_t in_x, in_ret;

    mpfr_init2(in_x, 53 * TDSIZE); mpfr_set_td(in_x, x, MPFR_RNDN);
	mpfr_init2(in_ret, 53 * TDSIZE);

    mpfr_exp(in_ret, in_x, MPFR_RNDN);
    mpfr_get_td(ret, in_ret, MPFR_RNDN);

    mpfr_clear(in_x);
    mpfr_clear(in_ret);
}

void rqd_exp_mpfr(double ret[QDSIZE], double x[QDSIZE])
{
    mpfr_t in_x, in_ret;

    mpfr_init2(in_x, 53 * QDSIZE); mpfr_set_qd(in_x, x, MPFR_RNDN);
	mpfr_init2(in_ret, 53 * QDSIZE);

    mpfr_exp(in_ret, in_x, MPFR_RNDN);
    mpfr_get_qd(ret, in_ret, MPFR_RNDN);

    mpfr_clear(in_x);
    mpfr_clear(in_ret);
}

// 2024-04-23 (Tue)
// ret := sqrt(x)
void rdd_sqrt_mpfr(double ret[DDSIZE], double x[DDSIZE])
{
    mpfr_t in_x, in_ret;

    mpfr_init2(in_x, 53 * DDSIZE); mpfr_set_dd(in_x, x, MPFR_RNDN);
	mpfr_init2(in_ret, 53 * DDSIZE);

    mpfr_sqrt(in_ret, in_x, MPFR_RNDN);
    mpfr_get_dd(ret, in_ret, MPFR_RNDN);

    mpfr_clear(in_x);
    mpfr_clear(in_ret);
}
void rdd_sqrt_d_mpfr(double ret[DDSIZE], double x)
{
    mpfr_t in_x, in_ret;

    mpfr_init2(in_x, 53 * DDSIZE); mpfr_set_d(in_x, x, MPFR_RNDN);
	mpfr_init2(in_ret, 53 * DDSIZE);

    mpfr_sqrt(in_ret, in_x, MPFR_RNDN);
    mpfr_get_dd(ret, in_ret, MPFR_RNDN);

    mpfr_clear(in_x);
    mpfr_clear(in_ret);
}

// TD
void rtd_sqrt_mpfr(double ret[TDSIZE], double x[TDSIZE])
{
    mpfr_t in_x, in_ret;

    mpfr_init2(in_x, 53 * TDSIZE); mpfr_set_td(in_x, x, MPFR_RNDN);
	mpfr_init2(in_ret, 53 * TDSIZE);

    mpfr_sqrt(in_ret, in_x, MPFR_RNDN);
    mpfr_get_td(ret, in_ret, MPFR_RNDN);

    mpfr_clear(in_x);
    mpfr_clear(in_ret);
}
void rtd_sqrt_d_mpfr(double ret[TDSIZE], double x)
{
    mpfr_t in_x, in_ret;

    mpfr_init2(in_x, 53 * TDSIZE); mpfr_set_d(in_x, x, MPFR_RNDN);
	mpfr_init2(in_ret, 53 * TDSIZE);

    mpfr_sqrt(in_ret, in_x, MPFR_RNDN);
    mpfr_get_td(ret, in_ret, MPFR_RNDN);

    mpfr_clear(in_x);
    mpfr_clear(in_ret);
}

// QD
void rqd_sqrt_mpfr(double ret[QDSIZE], double x[QDSIZE])
{
    mpfr_t in_x, in_ret;

    mpfr_init2(in_x, 53 * QDSIZE); mpfr_set_qd(in_x, x, MPFR_RNDN);
	mpfr_init2(in_ret, 53 * QDSIZE);

    mpfr_sqrt(in_ret, in_x, MPFR_RNDN);
    mpfr_get_qd(ret, in_ret, MPFR_RNDN);

    mpfr_clear(in_x);
    mpfr_clear(in_ret);
}
void rqd_sqrt_d_mpfr(double ret[QDSIZE], double x)
{
    mpfr_t in_x, in_ret;

    mpfr_init2(in_x, 53 * QDSIZE); mpfr_set_d(in_x, x, MPFR_RNDN);
	mpfr_init2(in_ret, 53 * QDSIZE);

    mpfr_sqrt(in_ret, in_x, MPFR_RNDN);
    mpfr_get_qd(ret, in_ret, MPFR_RNDN);

    mpfr_clear(in_x);
    mpfr_clear(in_ret);
}

// 2025-02-03 (Mon)
// ret := mpfr_func(x)
void rdd_func_mpfr(double ret[DDSIZE], int (* mpfr_func)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t), double x[DDSIZE])
{
    mpfr_t in_x, in_ret;

    mpfr_init2(in_x, 53 * DDSIZE); mpfr_set_dd(in_x, x, MPFR_RNDN);
	mpfr_init2(in_ret, 53 * DDSIZE);

    //mpfr_sqrt(in_ret, in_x, MPFR_RNDN);
	mpfr_func(in_ret, in_x, MPFR_RNDN);
    mpfr_get_dd(ret, in_ret, MPFR_RNDN);

    mpfr_clear(in_x);
    mpfr_clear(in_ret);
}

// ret := mpfr_func(x)
void rtd_func_mpfr(double ret[TDSIZE], int (* mpfr_func)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t), double x[TDSIZE])
{
    mpfr_t in_x, in_ret;

    mpfr_init2(in_x, 53 * TDSIZE); mpfr_set_td(in_x, x, MPFR_RNDN);
	mpfr_init2(in_ret, 53 * TDSIZE);

    mpfr_func(in_ret, in_x, MPFR_RNDN);
    mpfr_get_td(ret, in_ret, MPFR_RNDN);

    mpfr_clear(in_x);
    mpfr_clear(in_ret);
}

// ret := mpfr_func(x)
void rqd_func_mpfr(double ret[QDSIZE], int (* mpfr_func)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t), double x[QDSIZE])

{
    mpfr_t in_x, in_ret;

    mpfr_init2(in_x, 53 * QDSIZE); mpfr_set_qd(in_x, x, MPFR_RNDN);
	mpfr_init2(in_ret, 53 * QDSIZE);

    mpfr_func(in_ret, in_x, MPFR_RNDN);
    mpfr_get_qd(ret, in_ret, MPFR_RNDN);

    mpfr_clear(in_x);
    mpfr_clear(in_ret);
}

// 2025-06-26 (Thu)
// ret := mpc_func(x)
void rcdd_func_mpc(cddfloat *ret, int (* mpc_func)(mpc_ptr, mpc_srcptr, mpc_rnd_t), cddfloat *x)
{
    mpc_t in_x, in_ret;

    mpc_init2(in_x, 53 * DDSIZE); mpc_set_cdd(in_x, x);
	mpc_init2(in_ret, 53 * DDSIZE);

    //mpfr_sqrt(in_ret, in_x, MPFR_RNDN);
	mpc_func(in_ret, in_x, MPC_DEFAULT_RND);
    mpc_get_cdd(ret, in_ret);

    mpc_clear(in_x);
    mpc_clear(in_ret);
}

// ret := mpc_func(x)
void rctd_func_mpc(ctdfloat *ret, int (* mpc_func)(mpc_ptr, mpc_srcptr, mpc_rnd_t), ctdfloat *x)
{
    mpc_t in_x, in_ret;

    mpc_init2(in_x, 53 * TDSIZE); mpc_set_ctd(in_x, x);
	mpc_init2(in_ret, 53 * TDSIZE);

    mpc_func(in_ret, in_x, MPC_DEFAULT_RND);
    mpc_get_ctd(ret, in_ret);

    mpc_clear(in_x);
    mpc_clear(in_ret);
}

// ret := mpc_func(x)
void rcqd_func_mpc(cqdfloat *ret, int (* mpc_func)(mpc_ptr, mpc_srcptr, mpc_rnd_t), cqdfloat *x)
{
    mpc_t in_x, in_ret;

    mpc_init2(in_x, 53 * QDSIZE); mpc_set_cqd(in_x, x);
	mpc_init2(in_ret, 53 * QDSIZE);

    mpc_func(in_ret, in_x, MPC_DEFAULT_RND);
    mpc_get_cqd(ret, in_ret);

    mpc_clear(in_x);
    mpc_clear(in_ret);
}

// mpc_arg(x)
void rcdd_arg(double ret[DDSIZE], cddfloat *x)
{
	mpfr_t in_ret;
	mpc_t in_x;

	mpfr_init2(in_ret, 53 * DDSIZE);
	mpc_init2(in_x, 53 * DDSIZE);
	mpc_set_cdd(in_x, x);
	mpc_arg(in_ret, in_x, MPC_DEFAULT_RND);
	mpfr_get_dd(ret, in_ret, MPFR_DEFAULT_RND);
	mpfr_clear(in_ret);
	mpc_clear(in_x);
}
// mpc_arg(x)
void rctd_arg(double ret[TDSIZE], ctdfloat *x)
{
	mpfr_t in_ret;
	mpc_t in_x;

	mpfr_init2(in_ret, 53 * TDSIZE);
	mpc_init2(in_x, 53 * TDSIZE);
	mpc_set_ctd(in_x, x);
	mpc_arg(in_ret, in_x, MPC_DEFAULT_RND);
	mpfr_get_td(ret, in_ret, MPFR_DEFAULT_RND);
	mpfr_clear(in_ret);
	mpc_clear(in_x);
}

// mpc_arg(x)
void rcqd_arg(double ret[QDSIZE], cqdfloat *x)
{
	mpfr_t in_ret;
	mpc_t in_x;

	mpfr_init2(in_ret, 53 * QDSIZE);
	mpc_init2(in_x, 53 * QDSIZE);
	mpc_set_cqd(in_x, x);
	mpc_arg(in_ret, in_x, MPC_DEFAULT_RND);
	mpfr_get_qd(ret, in_ret, MPFR_DEFAULT_RND);
	mpfr_clear(in_ret);
	mpc_clear(in_x);
}

// 2025-06-26(Thu)
// rdd_const_pi
void rdd_const_pi(double ret[DDSIZE])
{
    mpf_t in_pi;

    mpf_init2(in_pi, 53 * DDSIZE);
    mpfr_const_pi(in_pi, get_bnc_default_rounding_mode());
	mpf_get_dd(ret, in_pi);
	mpf_clear(in_pi);
}

// rtd_const_pi
void rtd_const_pi(double ret[TDSIZE])
{
    mpf_t in_pi;

    mpf_init2(in_pi, 53 * TDSIZE);
    mpfr_const_pi(in_pi, get_bnc_default_rounding_mode());
	mpf_get_td(ret, in_pi);
	mpf_clear(in_pi);
}

// rqd_const_pi
void rqd_const_pi(double ret[QDSIZE])
{
    mpf_t in_pi;

    mpf_init2(in_pi, 53 * QDSIZE);
    mpfr_const_pi(in_pi, get_bnc_default_rounding_mode());
	mpf_get_qd(ret, in_pi);
	mpf_clear(in_pi);
}
#endif // USE_GMP