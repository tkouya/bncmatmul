/********************************************************************************/
/* cmpc_poly.c: Algebraic Equations and Polynomials                             */
/* copyright (c) 2025 Tomonori Kouya                                            */
/*                                                                              */
/* Ver. 0.1 2025-03-10: Generate from mpc_poly.c                                */
/*                                                                              */
/* This program is free software: you can redistribute it and/or modify it      */
/* under the terms of the GNU Lesser General Public License as published by the */
/* Free Software Foundation, either version 3 of the License or any later       */
/* version.                                                                     */
/*                                                                              */
/* This program is distributed in the hope that it will be useful, but WITHOUT  */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License */
/* for more details.                                                            */
/*                                                                              */
/* You should have received a copy of the GNU Lesser General Public License     */
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                              */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//#include "bnc.h"
#include "poly.h"

#ifdef __cpluplus
extern "C" {
#endif // __cplusplus

/*************************************************/
/* Functions for Polynomial Types                */
/*                                               */
/* Initialize:                                   */
/*   CMPFPoly init_cmpfpoly(long int max_length)   */
/*   CMPFPoly init2_cmpfpoly(long int max_length, unsigned long prec)*/
/* Free:                                         */
/*   void free_cmpfpoly(CMPFPoly pol)              */
/* Get & Set Values:                             */
/*   mpc_t *get_cmpfpoly_i(CMPFPoly pol, long int index) */
/*   long int setdegree_cmpfpoly(CMPFPoly)         */
/*   void set_cmpfpoly_i(CMPFPoly pol, long int index, mpc_t val) */
/*   void set_cmpfpoly_i_d(CMPFPoly pol, long int index, double val) */
/* Output:                                       */
/*   void print_cmpfpoly(CMPFPoly pol)             */
/*   void print_fdmpfpoly(FPoly fv, DPoly dv, CMPFPoly mpfv) */
/*************************************************/
#ifdef USE_GMP

CMPFPoly init_cmpfpoly(long int max_length)
{
	CMPFPoly ret = NULL;
	long int i;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	if(max_length <= 0)
	{
		fprintf(stderr, "ERROR: init_cmpfpoly\n");
		return ret;
	}

	ret = (CMPFPoly)malloc(sizeof(cmpfpoly));
	if(ret == NULL)
		return ret;

	ret->coef = (mpc_t *)calloc(max_length, sizeof(mpc_t));
	//ret->coef = (mpc_t *)calloc(sizeof(mpc_t), max_length);
//	ret->coef = (mpc_t *)malloc(sizeof(mpc_t) * max_length);
	if(ret->coef == NULL)
	{
		free(ret);
		return NULL;
	}

	/* All 0 */
	for(i = 0; i < max_length; i++)
	{
		mpc_init((mpc_ptr)(ret->coef + i));
		if((ret->coef + i) == NULL)
		{
			free(ret);
			return NULL;
		}
		mpc_set_ui((mpc_ptr)(ret->coef + i), 0UL, rndc);
	}

    mpc_init(ret->zero);
	mpc_set_ui_ui(ret->zero, 0UL, 0UL, rndc);

	ret->deg = 0;
	ret->max_len = max_length;

	ret->prec = get_bnc_default_prec();

	return ret;
}

/* prec ... A number of at least bits of mantissa */
CMPFPoly init2_cmpfpoly(long int max_length, unsigned long int prec)
{
	CMPFPoly ret = NULL;
	long int i;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	if(max_length <= 0)
	{
		fprintf(stderr, "ERROR: init2_cmpfpoly\n");
		return ret;
	}

	ret = (CMPFPoly)malloc(sizeof(cmpfpoly));
	if(ret == NULL)
		return ret;

	//ret->coef = (mpc_t *)calloc(sizeof(mpc_t), max_length);
	ret->coef = (mpc_t *)calloc(max_length, sizeof(mpc_t));
//	ret->coef = (mpc_t *)malloc(sizeof(mpc_t) * max_length);
	if(ret->coef == NULL)
	{
		free(ret);
		return NULL;
	}

	/* All 0 */
	for(i = 0; i < max_length; i++)
	{
		mpc_init2((mpc_ptr)(ret->coef + i), prec);
		if((ret->coef + i) == NULL)
		{
			free(ret);
			return NULL;
		}
		mpc_set_ui((mpc_ptr)(ret->coef + i), 0UL, rndc);
	}

	mpc_init2(ret->zero, prec);
	mpc_set_ui_ui(ret->zero, 0UL, 0UL, rndc);

	ret->deg = 0;
	ret->max_len = max_length;

	ret->prec = prec;

	return ret;
}

void free_cmpfpoly(CMPFPoly pol)
{
	long int i;

	if(pol == NULL)
		return;

	if(pol->coef != NULL)
	{
//		for(i = 0; i < pol->deg; i++)
		for(i = 0; i < pol->max_len; i++) // Fix! 2012-07-18 by T.Kouya
			mpc_clear((mpc_ptr)(pol->coef + i));

		free(pol->coef); // Fix! 2012-06-03 by T.Kouya
	}

	mpc_clear(pol->zero);

//	free(&(pol->deg));
//	free(&(pol->prec));
	free(pol);

}
#endif // USE_GMP

#ifdef USE_GMP
mpc_ptr get_cmpfpoly_i(CMPFPoly pol, long int index)
{
	if(index > pol->deg)
		return pol->zero;
	else
		return *(pol->coef + index);
}
#endif // USE_GMP

#ifdef USE_GMP
/* find maximum degree at pol->coef[deg] != 0 */
long int setdegree_cmpfpoly(CMPFPoly pol)
{
	long int i;

	for(i = pol->max_len - 1; i > 0; i--)
	{
		//if(mpc_cmp(get_cmpfpoly_i(pol, i), pol->zero) != 0)
		if(mpc_cmp(get_cmpfpoly_i(pol, i), pol->zero) != 0)
        {
			pol->deg = i;
			return i;
		}
	}

	pol->deg = 0;
	return 0;

}
#endif // USE_GMP

#ifdef USE_GMP
void set_cmpfpoly_i(CMPFPoly pol, long int index, mpc_t val)
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	mpc_set(*(pol->coef + index), val, get_bnc_default_rounding_mode_c());
	if((pol->deg < index) && mpc_cmp(val, pol->zero))
		pol->deg = index;
}

void set_cmpfpoly_i_si_si(CMPFPoly pol, long int index, long val_re, long val_im)
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	mpc_set_si_si(*(pol->coef + index), val_re, val_im, get_bnc_default_rounding_mode_c());
	if((pol->deg < index) && ((val_re != 0) || (val_im != 0)))
		pol->deg = index;
}

void set_cmpfpoly_i_si(CMPFPoly pol, long int index, long val)
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	mpc_set_si(*(pol->coef + index), val, get_bnc_default_rounding_mode_c());
	if((pol->deg < index) && (val != 0))
		pol->deg = index;
}

void set_cmpfpoly_i_ui_ui(CMPFPoly pol, long int index, unsigned long val_re, unsigned long val_im)
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	mpc_set_ui_ui(*(pol->coef + index), val_re, val_im, get_bnc_default_rounding_mode_c());
	if((pol->deg < index) && ((val_re != 0) || (val_im != 0)))
		pol->deg = index;
}

void set_cmpfpoly_i_ui(CMPFPoly pol, long int index, unsigned long val)
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	mpc_set_ui(*(pol->coef + index), val, get_bnc_default_rounding_mode_c());
	if((pol->deg < index) && (val != 0))
		pol->deg = index;
}

void set_cmpfpoly_i_cd(CMPFPoly pol, long int index, double _Complex val)
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	//mpc_set_cd(*(pol->coef + index), val, get_bnc_default_rounding_mode_c());
    mpf_set_d(mpc_realref(pol->coef[index]), creal(val));
    mpf_set_d(mpc_imagref(pol->coef[index]), cimag(val));

    if((pol->deg < index) && (cabs(val) != 0.0))
		pol->deg = index;
}

void set_cmpfpoly_i_d(CMPFPoly pol, long int index, double val)
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	mpc_set_d(*(pol->coef + index), val, get_bnc_default_rounding_mode_c());
	if((pol->deg < index) && (val != 0))
		pol->deg = index;
}

void set_cmpfpoly_i_str(CMPFPoly pol, long int index, const char *str, int base)
{
	mpc_t tmp;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	mpc_init2(tmp, pol->prec);
	mpc_set_str(tmp, str, base, get_bnc_default_rounding_mode_c());
	mpc_set(*(pol->coef + index), tmp, get_bnc_default_rounding_mode_c());
	if((pol->deg < index) && (mpc_cmp(pol->zero, tmp) != 0))
		pol->deg = index;
	mpc_clear(tmp);

}

/* get precision of CMPFPoly */
unsigned long int prec_cmpfpoly(CMPFPoly pol)
{
	return pol->prec;
}

/* search minimam precision in CMPFPoly */
unsigned long int minprec_cmpfpoly(CMPFPoly pol)
{
	unsigned long int prec, tmp;
	long int i;

	prec = mpc_get_prec(get_cmpfpoly_i(pol, 0));
	for(i = 1; i < pol->max_len; i++)
	{
		tmp = mpc_get_prec(get_cmpfpoly_i(pol, i));
		if(prec > tmp)
			prec = tmp;
	}

	return prec;
}

/* search maximam precision in CMPFPoly */
unsigned long int maxprec_cmpfpoly(CMPFPoly pol)
{
	unsigned long int prec, tmp;
	long int i;

	prec = mpc_get_prec(get_cmpfpoly_i(pol, 0));
	for(i = 1; i < pol->max_len; i++)
	{
		tmp = mpc_get_prec(get_cmpfpoly_i(pol, i));
		if(prec < tmp)
			prec = tmp;
	}

	return prec;
}


/* get maximum |coef * x^n| */
// return index of max |coef_n * x^n| 
long int max_abs_coefn_xn_cmpfpoly(mpf_t ret, CMPFPoly c, mpc_t x)
{
	long int i, ret_i;
	mpc_t xn, ctmp;
	mpf_t abs_coefn_xn;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();
    mpfr_rnd_t rnd = get_bnc_default_rounding_mode();

	mpf_init2(abs_coefn_xn, c->prec);
	mpc_init2(xn, c->prec);
    mpc_init2(ctmp, c->prec);

	// xn := 1
	mpc_set_ui(xn, 1UL, rndc);

	// ret := |coef_0|
	mpc_abs(ret, get_cmpfpoly_i(c, 0), rnd);
	ret_i = 0;

	for(i = 1; i <= c->deg; i++)
	{
		// xn := xn * x
		mpc_mul(xn, xn, x, rndc);

		// tmp := |coef_n * xn|
		mpc_mul(ctmp, get_cmpfpoly_i(c, i), xn, rndc);
		mpc_abs(abs_coefn_xn, ctmp, rnd);

		// ret < |coef_n * xn|
		if(mpf_cmp(ret, abs_coefn_xn) < 0)
		{
			mpf_set(ret, abs_coefn_xn);
			ret_i = i;
		}
	}

	mpf_clear(abs_coefn_xn);
	mpc_clear(xn);
    mpc_clear(ctmp);

	return ret_i;
}
#endif // USE_GMP

#ifdef USE_GMP
void print_cmpfpoly(CMPFPoly pol)
{
	long int i;

	for(i = 0; i <= pol->deg; i++)
	{
		printf("%5ld ", i);
		//mpc_out_str(stdout, 10, 0, get_cmpfpoly_i(pol, i));
        mpfr_printf("%RNe + %RNe * I", mpc_realref(get_cmpfpoly_i(pol, i)), mpc_imagref(get_cmpfpoly_i(pol, i)));
		printf("\n");
	}
}

// CMPFPoly := MPFPoly
void subst_cmpfpoly_mpfpoly(CMPFPoly ret, MPFPoly pol)
{
    long int i;
    mpc_t ctmp;

    mpc_init2(ctmp, ret->prec);
    mpc_set_ui_ui(ctmp, 0UL, 0UL, get_bnc_default_rounding_mode_c());

    for(i = 0; i <= pol->deg; i++)
    {
        mpc_set_fr(ctmp, get_mpfpoly_i(pol, i), get_bnc_default_rounding_mode_c());
        set_cmpfpoly_i(ret, i, ctmp);
    }

    //setdegree_cmpfpoly(ret);

    mpc_clear(ctmp);
}

// ret := org
CMPFPoly init_set_cmpfpoly_mpfpoly(MPFPoly org)
{
    CMPFPoly ret = NULL;

    ret = init2_cmpfpoly(org->max_len, org->prec);
    subst_cmpfpoly_mpfpoly(ret, org);

    return ret;
}

// ret := org
CMPFPoly init_set_cmpfpoly(CMPFPoly org)
{
    CMPFPoly ret = NULL;

    ret = init2_cmpfpoly(org->max_len, org->prec);
    subst_cmpfpoly(ret, org);

    return ret;
}

#endif // USE_GMP

/*************************************************/
/* Poly Calculations for CMPFPoly             */
/*
void add_cmpfpoly(CMPFPoly c, CMPFPoly a, CMPFPoly b)
void sub_cmpfpoly(CMPFPoly c, CMPFPoly a, CMPFPoly b)
void cmul_cmpfpoly(CMPFPoly c, mpc_t val, CMPFPoly a)
void subst_cmpfpoly(CMPFPoly c, CMPFPoly a)

void diff_cmpfpoly(CMPFPoly a)
void eval_cmpfpoly(mpc_t ret, CMPFPoly a, mpc_t x)
void eval_diff_cmpfpoly(mpc_t ret, CMPFPoly a, mpc_t x)
*/
/*************************************************/
#ifdef USE_GMP
/* c = a + b */
void add_cmpfpoly(CMPFPoly c, CMPFPoly a, CMPFPoly b)
{
	long int i;
	mpc_t tmp;
	unsigned long int prec;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	if((c->max_len < a->deg) || (c->max_len < b->deg))
	{
		fprintf(stderr, "ERROR: add_cmpfpoly\n");
		return;
	}

	prec = prec_cmpfpoly(c);

	mpc_init2(tmp, prec);

	c->deg = (a->deg > b->deg) ? a->deg : b->deg;

	for(i = 0; i <= c->deg; i++)
	{
		mpc_add(tmp, get_cmpfpoly_i(a, i), get_cmpfpoly_i(b, i), rndc);
		set_cmpfpoly_i(c, i, tmp);
	}

	mpc_clear(tmp);

}

/* c += a */
void add2_cmpfpoly(CMPFPoly c, CMPFPoly a)
{
	long int i;
	mpc_t tmp;
	unsigned long int prec;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: add2_cmpfpoly\n");
		return;
	}

	prec = prec_cmpfpoly(c);

	mpc_init2(tmp, prec);

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
	{
		mpc_add(tmp, get_cmpfpoly_i(c, i), get_cmpfpoly_i(a, i), rndc);
		set_cmpfpoly_i(c, i, tmp);
	}

	mpc_clear(tmp);
}

/* c = a - b */
void sub_cmpfpoly(CMPFPoly c, CMPFPoly a, CMPFPoly b)
{
	long int i;
	mpc_t tmp;
	unsigned long int prec;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	if((c->max_len < a->deg) || (c->max_len < b->deg))
	{
		fprintf(stderr, "ERROR: sub_cmpfpoly\n");
		return;
	}

	prec = prec_cmpfpoly(c);

	mpc_init2(tmp, prec);

	c->deg = (a->deg > b->deg) ? a->deg : b->deg;

	for(i = 0; i <= c->deg; i++)
	{
		mpc_sub(tmp, get_cmpfpoly_i(a, i), get_cmpfpoly_i(b, i), rndc);
		set_cmpfpoly_i(c, i, tmp);
	}

	mpc_clear(tmp);
}

/* c -= a */
void sub2_cmpfpoly(CMPFPoly c, CMPFPoly a)
{
	long int i, min_deg;
	mpc_t tmp;
	unsigned long int prec;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	prec = prec_cmpfpoly(c);

	mpc_init2(tmp, prec);

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: sub2_cmpfpoly\n");
		return;
	}

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
	{
		mpc_sub(tmp, get_cmpfpoly_i(c, i), get_cmpfpoly_i(a, i), rndc);
		set_cmpfpoly_i(c, i, tmp);
	}

	mpc_clear(tmp);

}

/* c = a * b */
void mul_cmpfpoly(CMPFPoly c, CMPFPoly a, CMPFPoly b)
{
	long int i, j;
	mpc_t tmp;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	if(c->max_len < (a->deg + b->deg))
	{
		fprintf(stderr, "ERROR: mul_cmpfpoly\n");
		return;
	}

	mpc_init2(tmp, c->prec);

	/* set c = 0 */
	set0_cmpfpoly(c);

	for(i = 0; i <= a->deg; i++)
	{
		for(j = 0; j <= b->deg; j++)
		{
			mpc_set(tmp, get_cmpfpoly_i(c, i + j), rndc);
			mpc_fma(tmp, get_cmpfpoly_i(a, i), get_cmpfpoly_i(b, j), tmp, rndc);
			set_cmpfpoly_i(c, i + j, tmp);
		}
	}

	c->deg = setdegree_cmpfpoly(c);
//	c->deg = a->deg + b->deg;

	mpc_clear(tmp);
}

/* c = val * a */
void cmul_cmpfpoly(CMPFPoly c, mpc_t val, CMPFPoly a)
{
	long int i;
	mpc_t tmp;
	unsigned long int prec;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: cmul_cmpfpoly\n");
		return;
	}

	prec = prec_cmpfpoly(c);

	mpc_init2(tmp, prec);

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
	{
		mpc_mul(tmp, val, get_cmpfpoly_i(a, i), rndc);
		set_cmpfpoly_i(c, i, tmp);
	}

	mpc_clear(tmp);
}

/* c *= val */
void cmul2_cmpfpoly(CMPFPoly c, mpc_t val)
{
	long int i;
	mpc_t tmp;
	unsigned long int prec;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	prec = prec_cmpfpoly(c);

	mpc_init2(tmp, prec);

	for(i = 0; i <= c->deg; i++)
	{
		mpc_mul(tmp, val, get_cmpfpoly_i(c, i), rndc);
		set_cmpfpoly_i(c, i, tmp);
	}

	mpc_clear(tmp);

}

/* c := a */
void subst_cmpfpoly(CMPFPoly c, CMPFPoly a)
{
	long int i;

	if(c->deg > a->deg)
	{
		for(i = c->deg; i >= a->deg; i--)
			set_cmpfpoly_i_ui_ui(c, i, 0UL, 0UL);
	}

	c->deg = a->deg;
	for(i = 0; i <= c->deg; i++)
		set_cmpfpoly_i(c, i, get_cmpfpoly_i(a, i));
}

/* c := 0 */
void set0_cmpfpoly(CMPFPoly c)
{
	unsigned long int prec;
	long int i;
	mpc_t tmp;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	prec = prec_cmpfpoly(c);

	mpc_init2(tmp, prec);
	mpc_set_ui(tmp, 0UL, rndc);

	for(i = 0; i <= c->deg; i++)
		set_cmpfpoly_i(c, i, tmp);

	mpc_clear(tmp);
}

/* number of nonzero coef */
long int num_nonzero_cmpfpoly(CMPFPoly c)
{
	long int i, ret;

	ret = 0;
	for(i = 0; i <= c->deg; i++)
	{
		if( (mpf_cmp_ui(mpc_realref(get_cmpfpoly_i(c, i)), 0UL) != 0) || (mpf_cmp_ui(mpc_realref(get_cmpfpoly_i(c, i)), 0UL) != 0) )
			ret++;
	}

	return ret;
}

/* get maximum |coef| */
long int max_abscoef_cmpfpoly(CMPFPoly c)
{
	long int i, ret;
	mpf_t tmp1, tmp;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();
    mpfr_rnd_t rnd = get_bnc_default_rounding_mode();

	mpf_init2(tmp1, c->prec);
	mpf_init2(tmp, c->prec);

	mpc_abs(tmp1, get_cmpfpoly_i(c, 0), rnd);
	ret = 0;
	for(i = 1; i <= c->deg; i++)
	{
		mpc_abs(tmp, get_cmpfpoly_i(c, i), rnd);
		if(mpf_cmp(tmp1, tmp) < 0)
		{
			mpf_set(tmp1, tmp);
			ret = i;
		}
	}

	mpf_clear(tmp1);
	mpf_clear(tmp);

	return ret;
}

/* a := a'(x) */
void diff_cmpfpoly(CMPFPoly a)
{
	long int diff_deg, i;
	mpc_t tmp_coef;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	diff_deg = a->deg - 1;

	if(diff_deg < 0)
	{
		set0_cmpfpoly(a);
		a->deg = 0;
		return;
	}

	mpc_init2(tmp_coef, a->prec);
	//for(i = 1; i <= diff_deg; i++)
	for(i = 1; i <= a->deg; i++) // Fix!! : 2007-01-11
	{
		mpc_mul_ui(tmp_coef, get_cmpfpoly_i(a, i), (unsigned long)i, rndc);
		set_cmpfpoly_i(a, i - 1, tmp_coef);
	}
	a->deg = diff_deg;
	mpc_clear(tmp_coef);
}

/* value of a(x) */
// Based on Horner method
void eval_cmpfpoly_horner(mpc_t ret, CMPFPoly a, mpc_t x)
{
	long int i;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	mpc_set(ret, get_cmpfpoly_i(a, a->deg), rndc);
	for(i = a->deg - 1; i >= 0; i--)
	{
		mpc_mul(ret, ret, x, rndc);
		mpc_add(ret, ret, get_cmpfpoly_i(a, i), rndc);
	}
}

/* value of a(x) */
// Based on Estrin method
void eval_cmpfpoly_estrin(mpc_t ret, CMPFPoly a, mpc_t x)
{
// Evaluation of polynomial using Estrin method
// k := degree
// coef[] = a_0, a_1, ..., a_{k-1}, a_k
// poly(x) = a_k * x^k + a_{k-1} * x^{k-1} + ... + a_1 * x + a_0
// ex) k = 7
// poly(x) = a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0
// q0(x) 
// = (a_0 + a_1 * x) + (a_2 + a_3 * x) * x^2 + (a_4 + a_5 * x) * x^4 + (a_6 + a_7 * x) * x^6
// =  b_0            +  b_1            * y   +  b_2            * y^2 +  b_3            * y^3
// = (b_0 + b_1 * y) + (b_2 + b_3 * y) * y^2
// =  c_0            +  c_1            * z
	unsigned long prec = mpc_get_prec(ret);
    mpc_t in_x, tmp;
	CMPFArray in_coef_old, in_coef_new;
    long int in_degree, num_in_coef, i;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

    if(a->deg == 0) {
        //return coef[0];
		mpc_set(ret, get_cmpfpoly_i(a, 0), rndc);
		return;
    }
    if(a->deg == 1) {
        //return coef[0] + coef[1] * x;
		mpc_mul(ret, get_cmpfpoly_i(a, 1), x, rndc);
		mpc_add(ret, ret, get_cmpfpoly_i(a, 0), rndc);
		return;
    }

	// Initialize
	mpc_init2(in_x, prec);
	mpc_init2(tmp, prec);

    num_in_coef = a->deg + 1;
    if((num_in_coef % 2) == 1) num_in_coef++;

    in_degree = num_in_coef - 1;
    //in_coef_old = (double *)calloc(num_in_coef, sizeof(double));
    //in_coef_new = (double *)calloc(num_in_coef, sizeof(double));
	in_coef_old = init2_cmpfarray(num_in_coef, prec);
	in_coef_new = init2_cmpfarray(num_in_coef, prec);

    for(i = 0; i <= a->deg; i++) set_cmpfarray_i(in_coef_old, i, get_cmpfpoly_i(a, i));
    for(i = a->deg + 1; i <= in_degree; i++) set_cmpfarray_i_ui(in_coef_old, i, 0UL);

    mpc_set(in_x, x, rndc);

    //printf("degree -> in_degree: %d -> %d\n", degree, in_degree);

    while(in_degree > 1)
    {
		num_in_coef /= 2;
        for(i = 0; i < num_in_coef; i++)
        {
            //in_coef_new[i] = in_coef_old[i * 2] + in_coef_old[i * 2 + 1] * in_x;
			mpc_mul(tmp, get_cmpfarray_i(in_coef_old, i * 2 + 1), in_x, rndc);
			mpc_add(tmp, tmp, get_cmpfarray_i(in_coef_old, i * 2), rndc);
			set_cmpfarray_i(in_coef_new, i, tmp);
            //printf("coef[%d] = in_coef[%d] + in_coef[%d] * %f = %f\n", i, i * 2, i * 2 + 1, in_x, in_coef_new[i]);
        }

        //return estrin(x * x, coef, num_in_coef / 2 - 1);
        //in_x = in_x * in_x;
		mpc_mul(in_x, in_x, in_x, rndc);
        for(i = 0; i < num_in_coef; i++)
			set_cmpfarray_i(in_coef_old, i, get_cmpfarray_i(in_coef_new, i));
           	//in_coef_old[i] = in_coef_new[i];

        if((num_in_coef % 2) == 1)
		{
			//in_coef_old[num_in_coef / 2 + 1] = 0; 
			num_in_coef += 1;
			set_cmpfarray_i_ui(in_coef_old, num_in_coef - 1, 0UL);
		}

        in_degree = num_in_coef - 1;

    }
    //ret = in_coef_new[0] + in_coef_new[1] * in_x;
	mpc_mul(ret, get_cmpfarray_i(in_coef_new, 1), in_x, rndc);
	mpc_add(ret, ret, get_cmpfarray_i(in_coef_new, 0), rndc);

	mpc_clear(tmp);
	mpc_clear(in_x);
    free_cmpfarray(in_coef_old);
    free_cmpfarray(in_coef_new);

    //return ret;
	return;
}

/* value of a'(x) */
// Based on Horner method
void eval_diff_cmpfpoly(mpc_t ret, CMPFPoly a, mpc_t x)
{
	long int i;
	mpc_t tmp;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	mpc_set(ret, get_cmpfpoly_i(a, a->deg), rndc);
	mpc_mul_ui(ret, ret, (unsigned long)a->deg, rndc);

	mpc_init2(tmp, a->prec);
	for(i = a->deg - 1; i >= 1; i--)
	{
		mpc_mul(ret, ret, x, rndc);
		mpc_mul_ui(tmp, get_cmpfpoly_i(a, i), (unsigned long)i, rndc);
		mpc_add(ret, ret, tmp, rndc);
	}
	mpc_clear(tmp);

}

// ------------------------------------
// New implementation
// 2025-01-15 (Wed) T.Kouya
// ------------------------------------
/* complex value of a(x) */
//void ceval_cmpfpoly(MPFCmplx ret, CMPFPoly a, MPFCmplx x)
// Based on Horner method
void ceval_cmpfpoly_horner(mpc_t ret, CMPFPoly a, mpc_t x)
{
	long int i;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	//set0_mpfcmplx(ret);
	mpc_set_ui(ret, 0UL, get_bnc_default_rounding_mode_c());

	//set_real_mpfcmplx(ret, get_cmpfpoly_i(a, a->deg));
	mpc_set(ret, get_cmpfpoly_i(a, a->deg), get_bnc_default_rounding_mode_c());
	for(i = a->deg - 1; i >= 0; i--)
	{
		//mul2_mpfcmplx(ret, x);
		mpc_mul(ret, ret, x, get_bnc_default_rounding_mode_c());
		//add_mpfcmplx_mpf(ret, ret, get_cmpfpoly_i(a, i));
		mpc_add(ret, ret, get_cmpfpoly_i(a, i), get_bnc_default_rounding_mode_c());
	}
}

/* value of a(x) */
// Based on Estrin method
void ceval_cmpfpoly_estrin(mpc_t ret, CMPFPoly a, mpc_t x)
{
// Evaluation of polynomial using Estrin method
// k := degree
// coef[] = a_0, a_1, ..., a_{k-1}, a_k
// poly(x) = a_k * x^k + a_{k-1} * x^{k-1} + ... + a_1 * x + a_0
// ex) k = 7
// poly(x) = a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0
// q0(x) 
// = (a_0 + a_1 * x) + (a_2 + a_3 * x) * x^2 + (a_4 + a_5 * x) * x^4 + (a_6 + a_7 * x) * x^6
// =  b_0            +  b_1            * y   +  b_2            * y^2 +  b_3            * y^3
// = (b_0 + b_1 * y) + (b_2 + b_3 * y) * y^2
// =  c_0            +  c_1            * z
	unsigned long prec = mpc_get_prec(ret);
    mpc_t in_x, tmp;
	CMPFArray in_coef_old, in_coef_new;
	mpc_rnd_t rmode = get_bnc_default_rounding_mode_c();
    long int in_degree, num_in_coef, i;

    if(a->deg == 0) {
        //return coef[0];
		mpc_set(ret, get_cmpfpoly_i(a, 0), rmode);
		return;
    }
    if(a->deg == 1) {
        //return coef[0] + coef[1] * x;
		mpc_mul(ret, x, get_cmpfpoly_i(a, 1), rmode);
		mpc_add(ret, ret, get_cmpfpoly_i(a, 0), rmode);
		return;
    }

	// Initialize
	mpc_init2(in_x, prec);
	mpc_init2(tmp, prec);

    num_in_coef = a->deg + 1;
    if((num_in_coef % 2) == 1) num_in_coef++;

    in_degree = num_in_coef - 1;
    //in_coef_old = (double *)calloc(num_in_coef, sizeof(double));
    //in_coef_new = (double *)calloc(num_in_coef, sizeof(double));
	in_coef_old = init2_cmpfarray(num_in_coef, prec);
	in_coef_new = init2_cmpfarray(num_in_coef, prec);

    for(i = 0; i <= a->deg; i++) set_cmpfarray_i(in_coef_old, i, get_cmpfpoly_i(a, i));
    for(i = a->deg + 1; i <= in_degree; i++) set_cmpfarray_i_ui(in_coef_old, i, 0UL);

    mpc_set(in_x, x, rmode);

    //printf("degree -> in_degree: %d -> %d\n", degree, in_degree);

    while(in_degree > 1)
    {
		num_in_coef /= 2;
        for(i = 0; i < num_in_coef; i++)
        {
            //in_coef_new[i] = in_coef_old[i * 2] + in_coef_old[i * 2 + 1] * in_x;
			mpc_mul(tmp, get_cmpfarray_i(in_coef_old, i * 2 + 1), in_x, rmode);
			mpc_add(tmp, tmp, get_cmpfarray_i(in_coef_old, i * 2), rmode);
			set_cmpfarray_i(in_coef_new, i, tmp);
            //printf("coef[%d] = in_coef[%d] + in_coef[%d] * %f = %f\n", i, i * 2, i * 2 + 1, in_x, in_coef_new[i]);
        }

        //return estrin(x * x, coef, num_in_coef / 2 - 1);
        //in_x = in_x * in_x;
		mpc_mul(in_x, in_x, in_x, rmode);
        for(i = 0; i < num_in_coef; i++)
			set_cmpfarray_i(in_coef_old, i, get_cmpfarray_i(in_coef_new, i));
           	//in_coef_old[i] = in_coef_new[i];

        if((num_in_coef % 2) == 1)
		{
			//in_coef_old[num_in_coef / 2 + 1] = 0; 
			num_in_coef += 1;
			set_cmpfarray_i_ui(in_coef_old, num_in_coef - 1, 0UL);
		}

        in_degree = num_in_coef - 1;

    }
    //ret = in_coef_new[0] + in_coef_new[1] * in_x;
	mpc_mul(ret, get_cmpfarray_i(in_coef_new, 1), in_x, rmode);
	mpc_add(ret, ret, get_cmpfarray_i(in_coef_new, 0), rmode);

	mpc_clear(tmp);
	mpc_clear(in_x);
    free_cmpfarray(in_coef_old);
    free_cmpfarray(in_coef_new);

    //return ret;
	return;
}

/* complex value of a'(x) */
//void ceval_diff_cmpfpoly(MPFCmplx ret, CMPFPoly a, MPFCmplx x)
// Based on Horner method
void ceval_diff_cmpfpoly(mpc_t ret, CMPFPoly a, mpc_t x)
{
	long int i;
	mpc_t tmp;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	//set0_mpfcmplx(ret);
	mpc_set_ui(ret, 0UL, rndc);
	//set_real_mpfcmplx(ret, get_cmpfpoly_i(a, a->deg));
	mpc_set(ret, get_cmpfpoly_i(a, a->deg), rndc); //get_bnc_default_rounding_mode_c());
	//mul_mpfcmplx_ui(ret, ret, (unsigned long)a->deg);
	mpc_mul_ui(ret, ret, (unsigned long)a->deg, rndc); //get_bnc_default_rounding_mode_c());

	mpc_init2(tmp, a->prec);
	for(i = a->deg - 1; i >= 1; i--)
	{
		//mul_mpfcmplx(ret, ret, x);
		mpc_mul(ret, ret, x, rndc); //get_bnc_default_rounding_mode_c());
		mpc_mul_ui(tmp, get_cmpfpoly_i(a, i), (unsigned long)i, rndc);
		//add_mpfcmplx_mpf(ret, ret, tmp);
		mpc_add(ret, ret, tmp,  rndc); //get_bnc_default_rounding_mode_c());
	}
	mpc_clear(tmp);
}

// eval based on Hessenberg matrix
// mphes: Upper Hessenberg matrix
// return |hes - x * I|
void ceval_hes_cmpfmatrix(mpc_t ret, CMPFMatrix hes, mpc_t x)
{
	unsigned prec = mpc_get_prec(ret);
	long int i, j;
	CMPFArray x_array;
	mpc_t axsum, tmp;
	mpc_rnd_t rnd = get_bnc_default_rounding_mode_c();
    mpc_t coef;

	x_array = init2_cmpfarray(hes->row_dim, prec);
	mpc_init2(axsum, prec);
	mpc_init2(tmp, prec);
    mpc_init2(coef, prec);

    //x_carray[degree - 1] = mpcomplex(mpreal(1UL), mpreal(0UL)); // mpreal(1UL); // one;
    mpc_set_ui_ui(get_cmpfarray_i(x_array, hes->row_dim - 1), 1UL, 0UL, rnd);
    for(i = hes->row_dim - 2; i >= 0; i--)
    {
        mpc_set_ui(axsum, 0UL, rnd); //= mpcomplex(mpreal(0UL), mpreal(0UL));
        for(j = i + 1; j < hes->col_dim; j++)
		{
            //axsum += hes[RMIJ(i + 1, j, degree, degree)] * x_carray[j];
			mpc_mul(tmp, get_cmpfarray_i(x_array, j), get_cmpfmatrix_ij(hes, i + 1, j), rnd);
			mpc_add(axsum, axsum, tmp, rnd);
        }
       	// x_carray[i] = (cx * x_carray[i + 1] - caxsum) / hes[RMIJ(i + 1, i, degree, degree)];
		mpc_mul(tmp, x, get_cmpfarray_i(x_array, i + 1), rnd);
		mpc_sub(tmp, tmp, axsum, rnd);
		mpc_div(tmp, tmp, get_cmpfmatrix_ij(hes, i + 1, i), rnd);
		set_cmpfarray_i(x_array, i, tmp);
        //mpfr_printf("x_carray[%d] = %25.17RNe\n", i, mpfr_ptr(x_carray[i].real()));
    }
    // axsum == p(x)
    //caxsum = mpcomplex(mpreal(0UL), mpreal(0UL)); // zero; // x * x_array[0];
	mpc_set_ui(ret, 0UL, rnd);
    for(i = 0; i < hes->row_dim; i++)
	{
        //caxsum += hes[RMIJ(0, i, degree, degree)] * x_carray[i];
		mpc_mul(tmp, get_cmpfarray_i(x_array, i), get_cmpfmatrix_ij(hes, 0, i), rnd);
		mpc_add(ret, ret, tmp, rnd);
	}

    //caxsum = cx * x_carray[0] - caxsum;
	mpc_mul(tmp, x, get_cmpfarray_i(x_array, 0), rnd);
	mpc_sub(ret, tmp, ret, rnd);

    // coef := (-1)^n * a_12 * a_23 * ... * a_n-1,n
    mpc_set_ui(coef, 1UL, rnd);
    for(i = 1; i < hes->row_dim; i++)
        mpc_mul(coef, coef, get_cmpfmatrix_ij(hes, i, i - 1), rnd);

    if(hes->row_dim % 2 == 1) mpc_neg(coef, coef, rnd);

    // ret := coef * ret
    mpc_mul(ret, ret, coef, rnd);

	free_cmpfarray(x_array);
	mpc_clear(axsum);
	mpc_clear(tmp);
    mpc_clear(coef);
}
#endif // USE_GMP

#ifdef __cpluplus
} // extern "C" {
#endif // __cplusplus
