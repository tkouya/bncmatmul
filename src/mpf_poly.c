/********************************************************************************/
/* mpf_poly.c: Algebraic Equations and Polynomials                              */
/* copyright (c) 2025 Tomonori Kouya                                            */
/*                                                                              */
/* Ver. 0.2 2025-01-10: modify with MPC                                         */
/* Ver. 0.1 2012-07-18: Fix free_mpfpoly                                        */
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
/*   MPFPoly init_mpfpoly(long int max_length)   */
/*   MPFPoly init2_mpfpoly(long int max_length, unsigned long prec)*/
/* Free:                                         */
/*   void free_mpfpoly(MPFPoly pol)              */
/* Get & Set Values:                             */
/*   mpf_t *get_mpfpoly_i(MPFPoly pol, long int index) */
/*   long int setdegree_mpfpoly(MPFPoly)         */
/*   void set_mpfpoly_i(MPFPoly pol, long int index, mpf_t val) */
/*   void set_mpfpoly_i_d(MPFPoly pol, long int index, double val) */
/* Output:                                       */
/*   void print_mpfpoly(MPFPoly pol)             */
/*   void print_fdmpfpoly(FPoly fv, DPoly dv, MPFPoly mpfv) */
/*************************************************/
#ifdef USE_GMP

MPFPoly init_mpfpoly(long int max_length)
{
	MPFPoly ret = NULL;
	long int i;

	if(max_length <= 0)
	{
		fprintf(stderr, "ERROR: init_mpfpoly\n");
		return ret;
	}

	ret = (MPFPoly)malloc(sizeof(mpfpoly));
	if(ret == NULL)
		return ret;

	ret->coef = (mpf_t *)calloc(max_length, sizeof(mpf_t));
//	ret->coef = (mpf_t *)calloc(sizeof(mpf_t), max_length);
//	ret->coef = (mpf_t *)malloc(sizeof(mpf_t) * max_length);
	if(ret->coef == NULL)
	{
		free(ret);
		return NULL;
	}

	/* All 0 */
	for(i = 0; i < max_length; i++)
	{
		mpf_init((mpf_ptr)(ret->coef + i));
		if((ret->coef + i) == NULL)
		{
			free(ret);
			return NULL;
		}
		mpf_set_ui((mpf_ptr)(ret->coef + i), 0UL);
	}

	mpf_init_set_ui(ret->zero, 0UL);

	ret->deg = 0;
	ret->max_len = max_length;

	ret->prec = get_bnc_default_prec();

	return ret;
}

/* prec ... A number of at least bits of mantissa */
MPFPoly init2_mpfpoly(long int max_length, unsigned long int prec)
{
	MPFPoly ret = NULL;
	long int i;

	if(max_length <= 0)
	{
		fprintf(stderr, "ERROR: init2_mpfpoly\n");
		return ret;
	}

	ret = (MPFPoly)malloc(sizeof(mpfpoly));
	if(ret == NULL)
		return ret;

	ret->coef = (mpf_t *)calloc(max_length, sizeof(mpf_t));
//	ret->coef = (mpf_t *)calloc(sizeof(mpf_t), max_length);
//	ret->coef = (mpf_t *)malloc(sizeof(mpf_t) * max_length);
	if(ret->coef == NULL)
	{
		free(ret);
		return NULL;
	}

	/* All 0 */
	for(i = 0; i < max_length; i++)
	{
		mpf_init2((mpf_ptr)(ret->coef + i), prec);
		if((ret->coef + i) == NULL)
		{
			free(ret);
			return NULL;
		}
		mpf_set_ui((mpf_ptr)(ret->coef + i), 0UL);
	}

	mpf_init2(ret->zero, prec);
	mpf_set_ui(ret->zero, 0UL);

	ret->deg = 0;
	ret->max_len = max_length;

	ret->prec = prec;

	return ret;
}

// init_set_mpfpoly
MPFPoly init_set_mpfpoly(MPFPoly org_pol)
{
    MPFPoly ret;
    long int i;

    ret = init_mpfpoly(org_pol->max_len);
    if(ret == NULL) return NULL;

    for(i = 0; i <= org_pol->deg; i++)
        set_mpfpoly_i(ret, i, get_mpfpoly_i(org_pol, i));

    return ret;
}

void free_mpfpoly(MPFPoly pol)
{
	long int i;

	if(pol == NULL)
		return;

	if(pol->coef != NULL)
	{
//		for(i = 0; i < pol->deg; i++)
		for(i = 0; i < pol->max_len; i++) // Fix! 2012-07-18 by T.Kouya
			mpf_clear((mpf_ptr)(pol->coef + i));

		free(pol->coef); // Fix! 2012-06-03 by T.Kouya
	}

	mpf_clear(pol->zero);

//	free(&(pol->deg));
//	free(&(pol->prec));
	free(pol);
}
#endif // USE_GMP

#ifdef USE_GMP
mpf_ptr get_mpfpoly_i(MPFPoly pol, long int index)
{
	if(index > pol->deg)
		return pol->zero;
	else
		return *(pol->coef + index);
}
#endif // USE_GMP

#ifdef USE_GMP
/* find maximum degree at pol->coef[deg] != 0 */
long int setdegree_mpfpoly(MPFPoly pol)
{
	long int i;

	for(i = pol->max_len - 1; i > 0; i--)
	{
		if(mpf_cmp(get_mpfpoly_i(pol, i), pol->zero) != 0)
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
void set_mpfpoly_i(MPFPoly pol, long int index, mpf_t val)
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	mpf_set(*(pol->coef + index), val);
	if((pol->deg < index) && (mpf_cmp(val, pol->zero) != 0))
		pol->deg = index;
}

void set_mpfpoly_i_si(MPFPoly pol, long int index, long val)
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	mpf_set_si(*(pol->coef + index), val);
	if((pol->deg < index) && (val != 0))
		pol->deg = index;
}

void set_mpfpoly_i_ui(MPFPoly pol, long int index, unsigned long val)
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	mpf_set_ui(*(pol->coef + index), val);
	if((pol->deg < index) && (val != 0))
		pol->deg = index;
}

void set_mpfpoly_i_d(MPFPoly pol, long int index, double val)
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	mpf_set_d(*(pol->coef + index), val);
	if((pol->deg < index) && (val != 0))
		pol->deg = index;
}

void set_mpfpoly_i_str(MPFPoly pol, long int index, const char *str, int base)
{
	mpf_t tmp;

	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	mpf_init2(tmp, pol->prec);
	mpf_set_str(tmp, str, base);
	mpf_set(*(pol->coef + index), tmp);
	if((pol->deg < index) && (mpf_cmp(pol->zero, tmp) != 0))
		pol->deg = index;
	mpf_clear(tmp);

}

/* get precision of MPFPoly */
unsigned long int prec_mpfpoly(MPFPoly pol)
{
	return pol->prec;
}

/* search minimam precision in MPFPoly */
unsigned long int minprec_mpfpoly(MPFPoly pol)
{
	unsigned long int prec, tmp;
	long int i;

	prec = mpf_get_prec(get_mpfpoly_i(pol, 0));
	for(i = 1; i < pol->max_len; i++)
	{
		tmp = mpf_get_prec(get_mpfpoly_i(pol, i));
		if(prec > tmp)
			prec = tmp;
	}

	return prec;
}

/* search maximam precision in MPFPoly */
unsigned long int maxprec_mpfpoly(MPFPoly pol)
{
	unsigned long int prec, tmp;
	long int i;

	prec = mpf_get_prec(get_mpfpoly_i(pol, 0));
	for(i = 1; i < pol->max_len; i++)
	{
		tmp = mpf_get_prec(get_mpfpoly_i(pol, i));
		if(prec < tmp)
			prec = tmp;
	}

	return prec;
}


/* get maximum |coef * x^n| */
// return index of max |coef_n * x^n| 
long int max_abs_coefn_xn_mpfpoly(mpf_t ret, MPFPoly c, mpf_t x)
{
	long int i, ret_i;
	mpf_t abs_coefn_xn, xn;

	mpf_init2(abs_coefn_xn, c->prec);
	mpf_init2(xn, c->prec);

	// xn := 1
	mpf_set_ui(xn, 1UL);

	// ret := |coef_0|
	mpf_abs(ret, get_mpfpoly_i(c, 0));
	ret_i = 0;

	for(i = 1; i <= c->deg; i++)
	{
		// xn := xn * x
		mpf_mul(xn, xn, x);

		// tmp := |coef_n * xn|
		mpf_mul(abs_coefn_xn, get_mpfpoly_i(c, i), xn);
		mpf_abs(abs_coefn_xn, abs_coefn_xn);

		// ret < |coef_n * xn|
		if(mpf_cmp(ret, abs_coefn_xn) < 0)
		{
			mpf_set(ret, abs_coefn_xn);
			ret_i = i;
		}
	}

	mpf_clear(abs_coefn_xn);
	mpf_clear(xn);

	return ret_i;
}

/* get maximum |coef * x^n| */
// return index of max |coef_n * x^n| 
long int max_abs_coefn_cxn_mpfpoly(mpf_t ret, MPFPoly c, mpc_t x)
{
	long int i, ret_i;
	mpf_t abs_coefn_xn;
	mpc_t xn, ctmp;
	mpfr_rnd_t rnd = get_bnc_default_rounding_mode();
	mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	mpf_init2(abs_coefn_xn, c->prec);
	mpc_init2(xn, c->prec);
	mpc_init2(ctmp, c->prec);

	// xn := 1
	mpc_set_ui(xn, 1UL, rndc);

	// ret := |coef_0|
	mpf_abs(ret, get_mpfpoly_i(c, 0));
	ret_i = 0;

	for(i = 1; i <= c->deg; i++)
	{
		// xn := xn * x
		mpc_mul(xn, xn, x, rndc);

		// tmp := |coef_n * xn|
		mpc_mul_fr(ctmp, xn, get_mpfpoly_i(c, i), rndc);
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
void print_mpfpoly(MPFPoly pol)
{
	long int i;

	for(i = 0; i <= pol->deg; i++)
	{
		printf("%5ld ", i);
		mpf_out_str(stdout, 10, 0, get_mpfpoly_i(pol, i));
		printf("\n");
	}
}

void print_fdmpfpoly(FPoly fv, DPoly dv, MPFPoly mpfv)
{
	long int i;

	for(i = 0; i <= fv->deg; i++)
	{
		printf("%5ld %15.7e %25.17e ", i, get_fpoly_i(fv, i), get_dpoly_i(dv, i));
		mpf_out_str(stdout, 10, 0, get_mpfpoly_i(mpfv, i));
		printf("\n");
	}
}
#endif // USE_GMP

/*************************************************/
/* Poly Calculations for MPFPoly             */
/*
void add_mpfpoly(MPFPoly c, MPFPoly a, MPFPoly b)
void sub_mpfpoly(MPFPoly c, MPFPoly a, MPFPoly b)
void cmul_mpfpoly(MPFPoly c, mpf_t val, MPFPoly a)
void subst_mpfpoly(MPFPoly c, MPFPoly a)

void diff_mpfpoly(MPFPoly a)
void eval_mpfpoly(mpf_t ret, MPFPoly a, mpf_t x)
void eval_diff_mpfpoly(mpf_t ret, MPFPoly a, mpf_t x)
*/
/*************************************************/
#ifdef USE_GMP
/* c = a + b */
void add_mpfpoly(MPFPoly c, MPFPoly a, MPFPoly b)
{
	long int i;
	mpf_t tmp;
	unsigned long int prec;

	if((c->max_len < a->deg) || (c->max_len < b->deg))
	{
		fprintf(stderr, "ERROR: add_mpfpoly\n");
		return;
	}

	prec = prec_mpfpoly(c);

	mpf_init2(tmp, prec);

	c->deg = (a->deg > b->deg) ? a->deg : b->deg;

	for(i = 0; i <= c->deg; i++)
	{
		mpf_add(tmp, get_mpfpoly_i(a, i), get_mpfpoly_i(b, i));
		set_mpfpoly_i(c, i, tmp);
	}

	mpf_clear(tmp);

}

/* c += a */
void add2_mpfpoly(MPFPoly c, MPFPoly a)
{
	long int i;
	mpf_t tmp;
	unsigned long int prec;

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: add2_mpfpoly\n");
		return;
	}

	prec = prec_mpfpoly(c);

	mpf_init2(tmp, prec);

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
	{
		mpf_add(tmp, get_mpfpoly_i(c, i), get_mpfpoly_i(a, i));
		set_mpfpoly_i(c, i, tmp);
	}

	mpf_clear(tmp);
}

/* c = a - b */
void sub_mpfpoly(MPFPoly c, MPFPoly a, MPFPoly b)
{
	long int i;
	mpf_t tmp;
	unsigned long int prec;

	if((c->max_len < a->deg) || (c->max_len < b->deg))
	{
		fprintf(stderr, "ERROR: sub_mpfpoly\n");
		return;
	}

	prec = prec_mpfpoly(c);

	mpf_init2(tmp, prec);

	c->deg = (a->deg > b->deg) ? a->deg : b->deg;

	for(i = 0; i <= c->deg; i++)
	{
		mpf_sub(tmp, get_mpfpoly_i(a, i), get_mpfpoly_i(b, i));
		set_mpfpoly_i(c, i, tmp);
	}

	mpf_clear(tmp);
}

/* c -= a */
void sub2_mpfpoly(MPFPoly c, MPFPoly a)
{
	long int i, min_deg;
	mpf_t tmp;
	unsigned long int prec;

	prec = prec_mpfpoly(c);

	mpf_init2(tmp, prec);

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: sub2_mpfpoly\n");
		return;
	}

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
	{
		mpf_sub(tmp, get_mpfpoly_i(c, i), get_mpfpoly_i(a, i));
		set_mpfpoly_i(c, i, tmp);
	}

	mpf_clear(tmp);

}

/* c = a * b */
void mul_mpfpoly(MPFPoly c, MPFPoly a, MPFPoly b)
{
	long int i, j;
	mpf_t tmp;

	if(c->max_len < (a->deg + b->deg))
	{
		fprintf(stderr, "ERROR: mul_mpfpoly\n");
		return;
	}

	mpf_init2(tmp, c->prec);

	/* set c = 0 */
	set0_mpfpoly(c);

	for(i = 0; i <= a->deg; i++)
	{
		for(j = 0; j <= b->deg; j++)
		{
			mpf_set(tmp, get_mpfpoly_i(c, i + j));
			mpf_fma(tmp, get_mpfpoly_i(a, i), get_mpfpoly_i(b, j), tmp);
			set_mpfpoly_i(c, i + j, tmp);
		}
	}

	c->deg = setdegree_mpfpoly(c);
//	c->deg = a->deg + b->deg;

	mpf_clear(tmp);
}

/* c = val * a */
void cmul_mpfpoly(MPFPoly c, mpf_t val, MPFPoly a)
{
	long int i;
	mpf_t tmp;
	unsigned long int prec;

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: cmul_mpfpoly\n");
		return;
	}

	prec = prec_mpfpoly(c);

	mpf_init2(tmp, prec);

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
	{
		mpf_mul(tmp, val, get_mpfpoly_i(a, i));
		set_mpfpoly_i(c, i, tmp);
	}

	mpf_clear(tmp);
}

/* c *= val */
void cmul2_mpfpoly(MPFPoly c, mpf_t val)
{
	long int i;
	mpf_t tmp;
	unsigned long int prec;

	prec = prec_mpfpoly(c);

	mpf_init2(tmp, prec);

	for(i = 0; i <= c->deg; i++)
	{
		mpf_mul(tmp, val, get_mpfpoly_i(c, i));
		set_mpfpoly_i(c, i, tmp);
	}

	mpf_clear(tmp);

}

/* c := a */
void subst_mpfpoly(MPFPoly c, MPFPoly a)
{
	long int i;

	if(c->deg > a->deg)
	{
		for(i = c->deg; i >= a->deg; i--)
			set_mpfpoly_i_ui(c, i, 0UL);
	}

	c->deg = a->deg;
	for(i = 0; i <= c->deg; i++)
		set_mpfpoly_i(c, i, get_mpfpoly_i(a, i));
}

/* c := 0 */
void set0_mpfpoly(MPFPoly c)
{
	unsigned long int prec;
	long int i;
	mpf_t tmp;

	prec = prec_mpfpoly(c);

	mpf_init2(tmp, prec);
	mpf_set_ui(tmp, 0UL);

	for(i = 0; i <= c->deg; i++)
		set_mpfpoly_i(c, i, tmp);

	mpf_clear(tmp);
}

/* number of nonzero coef */
long int num_nonzero_mpfpoly(MPFPoly c)
{
	long int i, ret;

	ret = 0;
	for(i = 0; i <= c->deg; i++)
	{
		if(mpf_sgn(get_mpfpoly_i(c, i)) != 0)
			ret++;
	}

	return ret;
}

/* get maximum |coef| */
long int max_abscoef_mpfpoly(MPFPoly c)
{
	long int i, ret;
	mpf_t tmp1, tmp;

	mpf_init2(tmp1, c->prec);
	mpf_init2(tmp, c->prec);

	mpf_abs(tmp1, get_mpfpoly_i(c, 0));
	ret = 0;
	for(i = 1; i <= c->deg; i++)
	{
		mpf_abs(tmp, get_mpfpoly_i(c, i));
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
void diff_mpfpoly(MPFPoly a)
{
	long int diff_deg, i;
	mpf_t tmp_coef;

	diff_deg = a->deg - 1;

	if(diff_deg < 0)
	{
		set0_mpfpoly(a);
		a->deg = 0;
		return;
	}

	mpf_init2(tmp_coef, a->prec);
	//for(i = 1; i <= diff_deg; i++)
	for(i = 1; i <= a->deg; i++) // Fix!! : 2007-01-11
	{
		mpf_mul_ui(tmp_coef, get_mpfpoly_i(a, i), (unsigned long)i);
		set_mpfpoly_i(a, i - 1, tmp_coef);
	}
	a->deg = diff_deg;
	mpf_clear(tmp_coef);
}

/* value of a(x) */
// Based on Horner method
void eval_mpfpoly_horner(mpf_t ret, MPFPoly a, mpf_t x)
{
	long int i;

	mpf_set(ret, get_mpfpoly_i(a, a->deg));
	for(i = a->deg - 1; i >= 0; i--)
	{
		mpf_mul(ret, ret, x);
		mpf_add(ret, ret, get_mpfpoly_i(a, i));
	}
}

/* value of a(x) */
// Based on Estrin method
void eval_mpfpoly_estrin(mpf_t ret, MPFPoly a, mpf_t x)
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
	unsigned long prec = mpf_get_prec(ret);
    mpf_t in_x, tmp;
	MPFArray in_coef_old, in_coef_new;
    long int in_degree, num_in_coef, i;

    if(a->deg == 0) {
        //return coef[0];
		mpf_set(ret, get_mpfpoly_i(a, 0));
		return;
    }
    if(a->deg == 1) {
        //return coef[0] + coef[1] * x;
		mpf_mul(ret, get_mpfpoly_i(a, 1), x);
		mpf_add(ret, ret, get_mpfpoly_i(a, 0));
		return;
    }

	// Initialize
	mpf_init2(in_x, prec);
	mpf_init2(tmp, prec);

    num_in_coef = a->deg + 1;
    if((num_in_coef % 2) == 1) num_in_coef++;

    in_degree = num_in_coef - 1;
    //in_coef_old = (double *)calloc(num_in_coef, sizeof(double));
    //in_coef_new = (double *)calloc(num_in_coef, sizeof(double));
	in_coef_old = init2_mpfarray(num_in_coef, prec);
	in_coef_new = init2_mpfarray(num_in_coef, prec);

    for(i = 0; i <= a->deg; i++) set_mpfarray_i(in_coef_old, i, get_mpfpoly_i(a, i));
    for(i = a->deg + 1; i <= in_degree; i++) set_mpfarray_i_ui(in_coef_old, i, 0UL);

    mpf_set(in_x, x);

    //printf("degree -> in_degree: %d -> %d\n", degree, in_degree);

    while(in_degree > 1)
    {
		num_in_coef /= 2;
        for(i = 0; i < num_in_coef; i++)
        {
            //in_coef_new[i] = in_coef_old[i * 2] + in_coef_old[i * 2 + 1] * in_x;
			mpf_mul(tmp, get_mpfarray_i(in_coef_old, i * 2 + 1), in_x);
			mpf_add(tmp, tmp, get_mpfarray_i(in_coef_old, i * 2));
			set_mpfarray_i(in_coef_new, i, tmp);
            //printf("coef[%d] = in_coef[%d] + in_coef[%d] * %f = %f\n", i, i * 2, i * 2 + 1, in_x, in_coef_new[i]);
        }

        //return estrin(x * x, coef, num_in_coef / 2 - 1);
        //in_x = in_x * in_x;
		mpf_mul(in_x, in_x, in_x);
        for(i = 0; i < num_in_coef; i++)
			set_mpfarray_i(in_coef_old, i, get_mpfarray_i(in_coef_new, i));
           	//in_coef_old[i] = in_coef_new[i];

        if((num_in_coef % 2) == 1)
		{
			//in_coef_old[num_in_coef / 2 + 1] = 0; 
			num_in_coef += 1;
			set_mpfarray_i_ui(in_coef_old, num_in_coef - 1, 0UL);
		}

        in_degree = num_in_coef - 1;

    }
    //ret = in_coef_new[0] + in_coef_new[1] * in_x;
	mpf_mul(ret, get_mpfarray_i(in_coef_new, 1), in_x);
	mpf_add(ret, ret, get_mpfarray_i(in_coef_new, 0));

	mpf_clear(tmp);
	mpf_clear(in_x);
    free_mpfarray(in_coef_old);
    free_mpfarray(in_coef_new);

    //return ret;
	return;
}

/* value of a'(x) */
// Based on Horner method
void eval_diff_mpfpoly(mpf_t ret, MPFPoly a, mpf_t x)
{
	long int i;
	mpf_t tmp;

	mpf_set(ret, get_mpfpoly_i(a, a->deg));
	mpf_mul_ui(ret, ret, (unsigned long)a->deg);

	mpf_init2(tmp, a->prec);
	for(i = a->deg - 1; i >= 1; i--)
	{
		mpf_mul(ret, ret, x);
		mpf_mul_ui(tmp, get_mpfpoly_i(a, i), (unsigned long)i);
		mpf_add(ret, ret, tmp);
	}
	mpf_clear(tmp);

}

// ------------------------------------
// New implementation
// 2025-01-15 (Wed) T.Kouya
// ------------------------------------
/* complex value of a(x) */
//void ceval_mpfpoly(MPFCmplx ret, MPFPoly a, MPFCmplx x)
// Based on Horner method
void ceval_mpfpoly_horner(mpc_t ret, MPFPoly a, mpc_t x)
{
	long int i;

	//set0_mpfcmplx(ret);
	mpc_set_ui(ret, 0UL, get_bnc_default_rounding_mode_c());

	//set_real_mpfcmplx(ret, get_mpfpoly_i(a, a->deg));
	mpc_set_fr(ret, get_mpfpoly_i(a, a->deg), get_bnc_default_rounding_mode_c());
	for(i = a->deg - 1; i >= 0; i--)
	{
		//mul2_mpfcmplx(ret, x);
		mpc_mul(ret, ret, x, get_bnc_default_rounding_mode_c());
		//add_mpfcmplx_mpf(ret, ret, get_mpfpoly_i(a, i));
		mpc_add_fr(ret, ret, get_mpfpoly_i(a, i), get_bnc_default_rounding_mode_c());
	}
}

/* value of a(x) */
// Based on Estrin method
void ceval_mpfpoly_estrin(mpc_t ret, MPFPoly a, mpc_t x)
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
		mpc_set_fr(ret, get_mpfpoly_i(a, 0), rmode);
		return;
    }
    if(a->deg == 1) {
        //return coef[0] + coef[1] * x;
		mpc_mul_fr(ret, x, get_mpfpoly_i(a, 1), rmode);
		mpc_add_fr(ret, ret, get_mpfpoly_i(a, 0), rmode);
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

    for(i = 0; i <= a->deg; i++) set_cmpfarray_i_real(in_coef_old, i, get_mpfpoly_i(a, i));
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
//void ceval_diff_mpfpoly(MPFCmplx ret, MPFPoly a, MPFCmplx x)
// Based on Horner method
void ceval_diff_mpfpoly(mpc_t ret, MPFPoly a, mpc_t x)
{
	long int i;
	mpf_t tmp;

	//set0_mpfcmplx(ret);
	mpc_set_ui(ret, 0UL, get_bnc_default_rounding_mode_c());
	//set_real_mpfcmplx(ret, get_mpfpoly_i(a, a->deg));
	mpc_set_fr(ret, get_mpfpoly_i(a, a->deg), get_bnc_default_rounding_mode_c());
	//mul_mpfcmplx_ui(ret, ret, (unsigned long)a->deg);
	mpc_mul_ui(ret, ret, (unsigned long)a->deg, get_bnc_default_rounding_mode_c());

	mpf_init2(tmp, a->prec);
	for(i = a->deg - 1; i >= 1; i--)
	{
		//mul_mpfcmplx(ret, ret, x);
		mpc_mul(ret, ret, x, get_bnc_default_rounding_mode_c());
		mpf_mul_ui(tmp, get_mpfpoly_i(a, i), (unsigned long)i);
		//add_mpfcmplx_mpf(ret, ret, tmp);
		mpc_add_fr(ret, ret, tmp, get_bnc_default_rounding_mode_c());
	}
	mpf_clear(tmp);
}

// eval based on Hessenberg matrix
// mphes: Upper Hessenberg matrix
// return |hes - x * I|
void ceval_hes_mpfmatrix(mpc_t ret, MPFMatrix hes, mpc_t x)
{
	unsigned prec = mpc_get_prec(ret);
	long int i, j;
	CMPFArray x_array;
	mpc_t axsum, tmp;
	mpc_rnd_t rnd = get_bnc_default_rounding_mode_c();
    mpf_t coef;

	x_array = init2_cmpfarray(hes->row_dim, prec);
	mpc_init2(axsum, prec);
	mpc_init2(tmp, prec);
    mpf_init2(coef, prec);

    //x_carray[degree - 1] = mpcomplex(mpreal(1UL), mpreal(0UL)); // mpreal(1UL); // one;
    mpc_set_ui_ui(get_cmpfarray_i(x_array, hes->row_dim - 1), 1UL, 0UL, rnd);
    for(i = hes->row_dim - 2; i >= 0; i--)
    {
        mpc_set_ui(axsum, 0UL, rnd); //= mpcomplex(mpreal(0UL), mpreal(0UL));
        for(j = i + 1; j < hes->col_dim; j++)
		{
            //axsum += hes[RMIJ(i + 1, j, degree, degree)] * x_carray[j];
			mpc_mul_fr(tmp, get_cmpfarray_i(x_array, j), get_mpfmatrix_ij(hes, i + 1, j), rnd);
			mpc_add(axsum, axsum, tmp, rnd);
        }
       	// x_carray[i] = (cx * x_carray[i + 1] - caxsum) / hes[RMIJ(i + 1, i, degree, degree)];
		mpc_mul(tmp, x, get_cmpfarray_i(x_array, i + 1), rnd);
		mpc_sub(tmp, tmp, axsum, rnd);
		mpc_div_fr(tmp, tmp, get_mpfmatrix_ij(hes, i + 1, i), rnd);
		set_cmpfarray_i(x_array, i, tmp);
        //mpfr_printf("x_carray[%d] = %25.17RNe\n", i, mpfr_ptr(x_carray[i].real()));
    }
    // axsum == p(x)
    //caxsum = mpcomplex(mpreal(0UL), mpreal(0UL)); // zero; // x * x_array[0];
	mpc_set_ui(ret, 0UL, rnd);
    for(i = 0; i < hes->row_dim; i++)
	{
        //caxsum += hes[RMIJ(0, i, degree, degree)] * x_carray[i];
		mpc_mul_fr(tmp, get_cmpfarray_i(x_array, i), get_mpfmatrix_ij(hes, 0, i), rnd);
		mpc_add(ret, ret, tmp, rnd);
	}

    //caxsum = cx * x_carray[0] - caxsum;
	mpc_mul(tmp, x, get_cmpfarray_i(x_array, 0), rnd);
	mpc_sub(ret, tmp, ret, rnd);

    // coef := (-1)^n * a_12 * a_23 * ... * a_n-1,n
    mpf_set_ui(coef, 1UL);
    for(i = 1; i < hes->row_dim; i++)
        mpf_mul(coef, coef, get_mpfmatrix_ij(hes, i, i - 1));

    if(hes->row_dim % 2 == 1) mpf_neg(coef, coef);

    // ret := coef * ret
    mpc_mul_fr(ret, ret, coef, rnd);

	free_cmpfarray(x_array);
	mpc_clear(axsum);
	mpc_clear(tmp);
    mpf_clear(coef);
}
// ------------------------------------
// Old implementation
// ------------------------------------

/* complex value of a(x) */
// Based on Horner method
void _bncold_ceval_mpfpoly(MPFCmplx ret, MPFPoly a, MPFCmplx x)
{
	long int i;

	set0_mpfcmplx(ret);
	set_real_mpfcmplx(ret, get_mpfpoly_i(a, a->deg));
	for(i = a->deg - 1; i >= 0; i--)
	{
		mul2_mpfcmplx(ret, x);
		add_mpfcmplx_mpf(ret, ret, get_mpfpoly_i(a, i));
	}
}

/* complex value of a'(x) */
// Based on Horner method
void _bncold_ceval_diff_mpfpoly(MPFCmplx ret, MPFPoly a, MPFCmplx x)
{
	long int i;
	mpf_t tmp;

	set0_mpfcmplx(ret);
	set_real_mpfcmplx(ret, get_mpfpoly_i(a, a->deg));
	mul_mpfcmplx_ui(ret, ret, (unsigned long)a->deg);

	mpf_init2(tmp, a->prec);
	for(i = a->deg - 1; i >= 1; i--)
	{
		mul_mpfcmplx(ret, ret, x);
		mpf_mul_ui(tmp, get_mpfpoly_i(a, i), (unsigned long)i);
		add_mpfcmplx_mpf(ret, ret, tmp);
	}
	mpf_clear(tmp);

}
#endif // USE_GMP

#ifdef __cpluplus
} // extern "C" {
#endif // __cplusplus
