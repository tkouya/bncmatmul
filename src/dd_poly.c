/********************************************************************************/
/* dd_poly.c: Algebraic Equations and Polynomials (Double-Double Precision)     */
/* copyright (c) 2025 Tomonori Kouya                                            */
/*                                                                              */
/* Ver. 1.1 2025-01-22: Added AVX-512 and ARM Neon SIMD optimizations          */
/* Ver. 1.0 2025-01-01: Initial version for double-double polynomials          */
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

// SIMD intrinsics
/*
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON)
#include <arm_neon.h>
#endif
*/

//#include "bnc.h"
#include "poly.h"

#ifdef __cpluplus
extern "C" {
#endif // __cplusplus

/*************************************************/
/* Functions for Polynomial Types                */
/*                                               */
/* Initialize:                                   */
/*   DDPoly init_ddpoly(long int max_length)   */
/*   DDPoly init2_ddpoly(long int max_length, unsigned long prec)*/
/* Free:                                         */
/*   void free_ddpoly(DDPoly pol)              */
/* Get & Set Values:                             */
/*   ddfloat *get_ddpoly_i(DDPoly pol, long int index) */
/*   long int setdegree_ddpoly(DDPoly)         */
/*   void set_ddpoly_i(DDPoly pol, long int index, double val[DDSIZE]) */
/*   void set_ddpoly_i_d(DDPoly pol, long int index, double val[DDSIZE]) */
/* Output:                                       */
/*   void print_ddpoly(DDPoly pol)             */
/*************************************************/
DDPoly init_ddpoly(long int max_length)
{
	DDPoly ret = NULL;
	long int i;

	if(max_length <= 0)
	{
		fprintf(stderr, "ERROR: init_ddpoly\n");
		return ret;
	}

	ret = (DDPoly)malloc(sizeof(ddpoly));
	if(ret == NULL)
		return ret;

	ret->coef = (ddfloat *)calloc(sizeof(ddfloat), max_length);
	if(ret->coef == NULL)
		return NULL;

	/* All 0 */
	for(i = 0; i < max_length; i++)
		rdd_set_ui(ret->coef[i].val, 0UL);

    // zero := 0
    rdd_set0(ret->zero.val);

	ret->deg = 0;
	ret->max_len = max_length;

	return ret;
}

void free_ddpoly(DDPoly pol)
{
	long int i;

	if(pol == NULL)
		return;

	if(pol->coef != NULL)
		free(pol->coef); // Fix! 2012-06-03 by T.Kouya

	free(pol);
}

ddfloat get_ddpoly_i_float(DDPoly pol, long int index)
{
	if(index > pol->deg)
		return pol->zero;
	else
		return pol->coef[index];
}

double *get_ddpoly_i(DDPoly pol, long int index)
{
	if(index > pol->deg)
		return pol->zero.val;
	else
		return pol->coef[index].val;
}

/* find maximum degree at pol->coef[deg] != 0 */
long int setdegree_ddpoly(DDPoly pol)
{
	long int i;

	for(i = pol->max_len - 1; i > 0; i--)
	{
		if(rdd_cmp(get_ddpoly_i(pol, i), pol->zero.val) != 0)
		{
			pol->deg = i;
			return i;
		}
	}

	pol->deg = 0;
	return 0;

}

void set_ddpoly_i(DDPoly pol, long int index, double val[DDSIZE])
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	rdd_set(pol->coef[index].val, val);
	if((pol->deg < index) && (rdd_cmp(val, pol->zero.val) != 0))
		pol->deg = index;
}

void set_ddpoly_i_si(DDPoly pol, long int index, long val)
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	rdd_set_d(pol->coef[index].val, (double)val);
	if((pol->deg < index) && (val != 0))
		pol->deg = index;
}

void set_ddpoly_i_ui(DDPoly pol, long int index, unsigned long val)
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	rdd_set_ui(pol->coef[index].val, val);
	if((pol->deg < index) && (val != 0))
		pol->deg = index;
}

void set_ddpoly_i_d(DDPoly pol, long int index, double val)
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	rdd_set_d(pol->coef[index].val, val);
	if((pol->deg < index) && (val != 0))
		pol->deg = index;
}

void set_ddpoly_i_str(DDPoly pol, long int index, const char *str, int base)
{
	double tmp[DDSIZE];

	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	rdd_set_str(tmp, str);
	rdd_set(pol->coef[index].val, tmp);
	if((pol->deg < index) && (rdd_cmp(pol->zero.val, tmp) != 0))
		pol->deg = index;

}

#ifdef USE_GMP
void set_ddpoly_i_mpf(DDPoly pol, long int index, mpf_t val)
{
	double tmp[DDSIZE];

	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	mpf_get_dd(tmp, val);
	rdd_set(pol->coef[index].val, tmp);
	if((pol->deg < index) && (rdd_cmp(pol->zero.val, tmp) != 0))
		pol->deg = index;

}

// Initialize and substitute polynomial from org_pol
DDPoly init_set_ddpoly_mpfpoly(MPFPoly org_pol)
{
	DDPoly ret = NULL;
	long int i;

	ret = init_ddpoly(org_pol->max_len);
	if(ret == NULL)
		return ret;

	for(i = 0; i <= org_pol->deg; i++)
		set_ddpoly_i_mpf(ret, i, get_mpfpoly_i(org_pol, i));

	return ret;
}

#endif // USE_GMP

void print_ddpoly(DDPoly pol)
{
	long int i;

	for(i = 0; i <= pol->deg; i++)
	{
		printf("%5ld ", i);
		rdd_out_str(get_ddpoly_i(pol, i));
		printf("\n");
	}
}

/*************************************************/
/* Poly Calculations for DDPoly             */
/*
void add_ddpoly(DDPoly c, DDPoly a, DDPoly b)
void sub_ddpoly(DDPoly c, DDPoly a, DDPoly b)
void cmul_ddpoly(DDPoly c, double val[DDSIZE], DDPoly a)
void subst_ddpoly(DDPoly c, DDPoly a)

void diff_ddpoly(DDPoly a)
void eval_ddpoly(double ret, DDPoly a, double x[DDSIZE])
void eval_diff_ddpoly(double ret, DDPoly a, double x[DDSIZE])
*/
/*************************************************/

/* c = a + b */
void add_ddpoly(DDPoly c, DDPoly a, DDPoly b)
{
	long int i;
	double tmp[DDSIZE];

	if((c->max_len < a->deg) || (c->max_len < b->deg))
	{
		fprintf(stderr, "ERROR: add_ddpoly\n");
		return;
	}

	c->deg = (a->deg > b->deg) ? a->deg : b->deg;

	for(i = 0; i <= c->deg; i++)
	{
		rdd_add(tmp, get_ddpoly_i(a, i), get_ddpoly_i(b, i));
		set_ddpoly_i(c, i, tmp);
	}
}

/* c += a */
void add2_ddpoly(DDPoly c, DDPoly a)
{
	long int i;
	double tmp[DDSIZE];

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: add2_ddpoly\n");
		return;
	}

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
	{
		rdd_add(tmp, get_ddpoly_i(c, i), get_ddpoly_i(a, i));
		set_ddpoly_i(c, i, tmp);
	}
}

/* c = a - b */
void sub_ddpoly(DDPoly c, DDPoly a, DDPoly b)
{
	long int i;
	double tmp[DDSIZE];

	if((c->max_len < a->deg) || (c->max_len < b->deg))
	{
		fprintf(stderr, "ERROR: sub_ddpoly\n");
		return;
	}

	c->deg = (a->deg > b->deg) ? a->deg : b->deg;

	for(i = 0; i <= c->deg; i++)
	{
		rdd_sub(tmp, get_ddpoly_i(a, i), get_ddpoly_i(b, i));
		set_ddpoly_i(c, i, tmp);
	}
}

/* c -= a */
void sub2_ddpoly(DDPoly c, DDPoly a)
{
	long int i, min_deg;
	double tmp[DDSIZE];

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: sub2_ddpoly\n");
		return;
	}

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
	{
		rdd_sub(tmp, get_ddpoly_i(c, i), get_ddpoly_i(a, i));
		set_ddpoly_i(c, i, tmp);
	}
}

/* c = a * b */
void mul_ddpoly(DDPoly c, DDPoly a, DDPoly b)
{
	long int i, j;
	double tmp[DDSIZE];

	if(c->max_len < (a->deg + b->deg))
	{
		fprintf(stderr, "ERROR: mul_ddpoly\n");
		return;
	}

	/* set c = 0 */
	set0_ddpoly(c);

	for(i = 0; i <= a->deg; i++)
	{
		for(j = 0; j <= b->deg; j++)
		{
			rdd_set(tmp, get_ddpoly_i(c, i + j));
			rdd_fma(tmp, get_ddpoly_i(a, i), get_ddpoly_i(b, j), tmp);
			set_ddpoly_i(c, i + j, tmp);
		}
	}

	c->deg = setdegree_ddpoly(c);
//	c->deg = a->deg + b->deg;
}

/* c = val * a */
void cmul_ddpoly(DDPoly c, double val[DDSIZE], DDPoly a)
{
	long int i;
	double tmp[DDSIZE];

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: cmul_ddpoly\n");
		return;
	}

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
	{
		rdd_mul(tmp, val, get_ddpoly_i(a, i));
		set_ddpoly_i(c, i, tmp);
	}
}

/* c *= val */
void cmul2_ddpoly(DDPoly c, double val[DDSIZE])
{
	long int i;
	double tmp[DDSIZE];

	for(i = 0; i <= c->deg; i++)
	{
		rdd_mul(tmp, val, get_ddpoly_i(c, i));
		set_ddpoly_i(c, i, tmp);
	}
}

/* c := a */
void subst_ddpoly(DDPoly c, DDPoly a)
{
	long int i;

	if(c->deg > a->deg)
	{
		for(i = c->deg; i >= a->deg; i--)
			set_ddpoly_i_ui(c, i, 0UL);
	}

	c->deg = a->deg;
	for(i = 0; i <= c->deg; i++)
		set_ddpoly_i(c, i, get_ddpoly_i(a, i));
}

/* c := 0 */
void set0_ddpoly(DDPoly c)
{
	long int i;
	double tmp[DDSIZE];

	rdd_set_ui(tmp, 0UL);

	for(i = 0; i <= c->deg; i++)
		set_ddpoly_i(c, i, tmp);
}

/* number of nonzero coef */
long int num_nonzero_ddpoly(DDPoly c)
{
	long int i, ret;

	ret = 0;
	for(i = 0; i <= c->deg; i++)
	{
		if(rdd_cmp_ui(get_ddpoly_i(c, i), 0UL) != 0)
			ret++;
	}

	return ret;
}

/* get maximum |coef| */
long int max_abscoef_ddpoly(DDPoly c)
{
	long int i, ret;
	double tmp1[DDSIZE], tmp[DDSIZE];

	rdd_abs(tmp1, get_ddpoly_i(c, 0));
	ret = 0;
	for(i = 1; i <= c->deg; i++)
	{
		rdd_abs(tmp, get_ddpoly_i(c, i));
		if(rdd_cmp(tmp1, tmp) < 0)
		{
			rdd_set(tmp1, tmp);
			ret = i;
		}
	}

	return ret;
}

/* a := a'(x) */
void diff_ddpoly(DDPoly a)
{
	long int diff_deg, i;
	double tmp_coef[DDSIZE];

	diff_deg = a->deg - 1;

	if(diff_deg < 0)
	{
		set0_ddpoly(a);
		a->deg = 0;
		return;
	}

	//for(i = 1; i <= diff_deg; i++)
	for(i = 1; i <= a->deg; i++) // Fix!! : 2007-01-11
	{
		rdd_mul_ui(tmp_coef, get_ddpoly_i(a, i), (unsigned long)i);
		set_ddpoly_i(a, i - 1, tmp_coef);
	}
	a->deg = diff_deg;
}

/* value of a(x) */
// Based on Horner method
void eval_ddpoly_horner(double ret[DDSIZE], DDPoly a, double x[DDSIZE])
{
	long int i;

	rdd_set(ret, get_ddpoly_i(a, a->deg));
	for(i = a->deg - 1; i >= 0; i--)
	{
		rdd_mul(ret, ret, x);
		rdd_add(ret, ret, get_ddpoly_i(a, i));
	}
}

/* value of a(x) */
// Based on Estrin method
void eval_ddpoly_estrin(double ret[DDSIZE], DDPoly a, double x[DDSIZE])
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
    double in_x[DDSIZE], tmp[DDSIZE];
	ddfloat *in_coef_old, *in_coef_new;
    long int in_degree, num_in_coef, i;

    if(a->deg == 0) {
        //return coef[0];
		rdd_set(ret, get_ddpoly_i(a, 0));
		return;
    }
    if(a->deg == 1) {
        //return coef[0] + coef[1] * x;
		rdd_mul(ret, get_ddpoly_i(a, 1), x);
		rdd_add(ret, ret, get_ddpoly_i(a, 0));
		return;
    }

    num_in_coef = a->deg + 1;
    if((num_in_coef % 2) == 1) num_in_coef++;

    in_degree = num_in_coef - 1;
    in_coef_old = (ddfloat *)calloc(num_in_coef, sizeof(ddfloat));
    in_coef_new = (ddfloat *)calloc(num_in_coef, sizeof(ddfloat));

    for(i = 0; i <= a->deg; i++) rdd_set(in_coef_old[i].val, get_ddpoly_i(a, i));
    for(i = a->deg + 1; i <= in_degree; i++) rdd_set_ui(in_coef_old[i].val, 0UL);

    rdd_set(in_x, x);

    //printf("degree -> in_degree: %d -> %d\n", degree, in_degree);

    while(in_degree > 1)
    {
		num_in_coef /= 2;
        for(i = 0; i < num_in_coef; i++)
        {
            //in_coef_new[i] = in_coef_old[i * 2] + in_coef_old[i * 2 + 1] * in_x;
			rdd_mul(tmp, in_coef_old[i * 2 + 1].val, in_x);
			rdd_add(tmp, tmp, in_coef_old[i * 2].val);
			rdd_set(in_coef_new[i].val, tmp);
            //printf("coef[%d] = in_coef[%d] + in_coef[%d] * %f = %f\n", i, i * 2, i * 2 + 1, in_x, in_coef_new[i]);
        }

        //return estrin(x * x, coef, num_in_coef / 2 - 1);
        //in_x = in_x * in_x;
		rdd_mul(in_x, in_x, in_x);
        for(i = 0; i < num_in_coef; i++)
			rdd_set(in_coef_old[i].val, in_coef_new[i].val);
           	//in_coef_old[i] = in_coef_new[i];

        if((num_in_coef % 2) == 1)
		{
			//in_coef_old[num_in_coef / 2 + 1] = 0; 
			num_in_coef += 1;
			rdd_set_ui(in_coef_old[num_in_coef - 1].val, 0UL);
		}

        in_degree = num_in_coef - 1;

    }
    //ret = in_coef_new[0] + in_coef_new[1] * in_x;
	rdd_mul(ret, in_coef_new[1].val, in_x);
	rdd_add(ret, ret, in_coef_new[0].val);

    free(in_coef_old);
    free(in_coef_new);

    //return ret;
	return;
}

/* value of a'(x) */
// Based on Horner method
void eval_diff_ddpoly(double ret[DDSIZE], DDPoly a, double x[DDSIZE])
{
	long int i;
	double tmp[DDSIZE];

	rdd_set(ret, get_ddpoly_i(a, a->deg));
	rdd_mul_ui(ret, ret, (unsigned long)a->deg);

	for(i = a->deg - 1; i >= 1; i--)
	{
		rdd_mul(ret, ret, x);
		rdd_mul_ui(tmp, get_ddpoly_i(a, i), (unsigned long)i);
		rdd_add(ret, ret, tmp);
	}

}

// ------------------------------------
// New implementation
// 2025-01-15 (Wed) T.Kouya
// ------------------------------------
/* complex value of a(x) */
//void ceval_ddpoly(MPFCmplx ret, DDPoly a, MPFCmplx x)
// Based on Horner method
void ceval_ddpoly_horner(cddfloat *ret, DDPoly a, cddfloat *x)
{
	long int i;

	//set0_mpfcmplx(ret);
	rcdd_set_ui(ret, 0UL);

	//set_real_mpfcmplx(ret, get_ddpoly_i(a, a->deg));
	rcdd_set_dd(ret, get_ddpoly_i(a, a->deg));
	for(i = a->deg - 1; i >= 0; i--)
	{
		//mul2_mpfcmplx(ret, x);
		rcdd_mul(ret, ret, x);
		//add_mpfcmplx_mpf(ret, ret, get_ddpoly_i(a, i));
		rcdd_add_dd(ret, ret, get_ddpoly_i(a, i));
	}
}

/* value of a(x) */
// Based on Estrin method
void ceval_ddpoly_estrin(cddfloat *ret, DDPoly a, cddfloat *x)
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
    cddfloat in_x, tmp;
	cddfloat *in_coef_old, *in_coef_new;
    long int in_degree, num_in_coef, i;

    if(a->deg == 0) {
        //return coef[0];
		rcdd_set_dd(ret, get_ddpoly_i(a, 0));
		return;
    }
    if(a->deg == 1) {
        //return coef[0] + coef[1] * x;
		rcdd_mul_dd(ret, x, get_ddpoly_i(a, 1));
		rcdd_add_dd(ret, ret, get_ddpoly_i(a, 0));
		return;
    }

    num_in_coef = a->deg + 1;
    if((num_in_coef % 2) == 1) num_in_coef++;

    in_degree = num_in_coef - 1;
    in_coef_old = (cddfloat *)calloc(num_in_coef, sizeof(cddfloat));
    in_coef_new = (cddfloat *)calloc(num_in_coef, sizeof(cddfloat));

    for(i = 0; i <= a->deg; i++) rcdd_set_dd(&in_coef_old[i], get_ddpoly_i(a, i));
    for(i = a->deg + 1; i <= in_degree; i++) rcdd_set_ui(&in_coef_old[i], 0UL);

    rcdd_set(&in_x, x);

    //printf("degree -> in_degree: %d -> %d\n", degree, in_degree);

    while(in_degree > 1)
    {
		num_in_coef /= 2;
        for(i = 0; i < num_in_coef; i++)
        {
            //in_coef_new[i] = in_coef_old[i * 2] + in_coef_old[i * 2 + 1] * in_x;
			rcdd_mul(&tmp, &in_coef_old[i * 2 + 1], &in_x);
			rcdd_add(&tmp, &tmp, &in_coef_old[i * 2]);
			rcdd_set(&in_coef_new[i], &tmp);
            //printf("coef[%d] = in_coef[%d] + in_coef[%d] * %f = %f\n", i, i * 2, i * 2 + 1, in_x, in_coef_new[i]);
        }

        //return estrin(x * x, coef, num_in_coef / 2 - 1);
        //in_x = in_x * in_x;
		rcdd_mul(&in_x, &in_x, &in_x);
        for(i = 0; i < num_in_coef; i++)
			rcdd_set(&in_coef_old[i], &in_coef_new[i]);
           	//in_coef_old[i] = in_coef_new[i];

        if((num_in_coef % 2) == 1)
		{
 			num_in_coef += 1;
			rcdd_set_ui(&in_coef_old[num_in_coef - 1], 0UL);
		}

        in_degree = num_in_coef - 1;

    }
    //ret = in_coef_new[0] + in_coef_new[1] * in_x;
	rcdd_mul(ret, &in_coef_new[1], &in_x);
	rcdd_add(ret, ret, &in_coef_new[0]);

    free(in_coef_old);
    free(in_coef_new);

    //return ret;
	return;
}

/* complex value of a'(x) */
//void ceval_diff_ddpoly(MPFCmplx ret, DDPoly a, MPFCmplx x)
// Based on Horner method
void ceval_diff_ddpoly(cddfloat *ret, DDPoly a, cddfloat *x)
{
	long int i;
	double tmp[DDSIZE];

	//set0_mpfcmplx(ret);
	rcdd_set_ui(ret, 0UL);
	//set_real_mpfcmplx(ret, get_ddpoly_i(a, a->deg));
	rcdd_set_dd(ret, get_ddpoly_i(a, a->deg));
	//mul_mpfcmplx_ui(ret, ret, (unsigned long)a->deg);
	rcdd_mul_ui(ret, ret, (unsigned long)a->deg);

	for(i = a->deg - 1; i >= 1; i--)
	{
		//mul_mpfcmplx(ret, ret, x);
		rcdd_mul(ret, ret, x);
		rdd_mul_ui(tmp, get_ddpoly_i(a, i), (unsigned long)i);
		//add_mpfcmplx_mpf(ret, ret, tmp);
		rcdd_add_dd(ret, ret, tmp);
	}
}

//------------
// AVX2
//------------
/* value of a(x) */
// Based on Horner method
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
void _bncavx2_eval_ddpoly_horner(__m256d ret[DDSIZE], DDPoly a, __m256d x[DDSIZE])
{
	long int i;
	//double ret;
	__m256d a_i[DDSIZE], tmp[DDSIZE];

	_bncavx2_rdd_set1_dd(ret, get_ddpoly_i(a, a->deg));
	for(i = a->deg - 1; i >= 0; i--)
	{
		_bncavx2_rdd_set1_dd(a_i, get_ddpoly_i(a, i));
		//ret = ret * x + get_dpoly_i(a, i);
		_bncavx2_rdd_mul(tmp, ret, x);
		_bncavx2_rdd_add(ret, ret, tmp);
	}

	return;
}

/* complex value of a(x) */
// Based on Horner method
void _bncavx2_ceval_ddpoly_horner(__m256d ret_re[DDSIZE], __m256d ret_im[DDSIZE], DDPoly a, __m256d x_re[DDSIZE], __m256d x_im[DDSIZE])
{
	long int i;
	__m256d tmp_re[DDSIZE], tmp_im[DDSIZE], a_i[DDSIZE];

	//set0_dcmplx(ret);
	_bncavx2_rcdd_set0(ret_re, ret_im); // = _mm256_setzero_pd();

	//set_real_dcmplx(ret, get_dpoly_i(a, a->deg));
	_bncavx2_rdd_set1_dd(ret_re, get_ddpoly_i(a, a->deg));

	for(i = a->deg - 1; i >= 0; i--)
	{
		/* ret = ret * x + get_dpoly_i(a, i) */
		//a_i = _mm256_set1_pd(get_dpoly_i(a, i));
		_bncavx2_rdd_set1_dd(a_i, get_ddpoly_i(a, i));

		//mul2_dcmplx(ret, x);
		_bncavx2_rcdd_mul(tmp_re, tmp_im, ret_re, ret_im, x_re, x_im);

		//add_dcmplx_d(ret, ret, get_dpoly_i(a, i));
		_bncavx2_rcdd_add_dd(ret_re, ret_im, tmp_re, tmp_im, a_i);
	}
}
#endif // __AVX2__ 

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
void _bncavx2_eval_ddpoly_estrin(double ret[DDSIZE], DDPoly a, double x[DDSIZE])
{
    //double *in_coef_old, *in_coef_new;
    DDVector in_coef_old, in_coef_new;
    double in_x[DDSIZE], tmp[DDSIZE];
    long int in_degree, num_in_coef, i, num_loop_unit;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d old_coef4[2][DDSIZE], new_coef4[DDSIZE], a04[DDSIZE], a14[DDSIZE], x4[DDSIZE], zero4[DDSIZE];
    __m256d tmp4[DDSIZE];

    if((a->deg + 1) <= _BNC_D_WIDTH)
    {
        eval_ddpoly_horner(ret, a, x);
        return;
    }

    // zero4 := 0
    _bncavx2_set0_dd(zero4); //  = _mm256_setzero_pd();
    num_loop_unit = 2 * _BNC_D_WIDTH;

#elif defined(__AVX512F__) // __AVX512F__
	__m512d old_coef8[2][DDSIZE], new_coef8[DDSIZE], a08[DDSIZE], a18[DDSIZE], x8[DDSIZE], zero8[DDSIZE];
    __m512d tmp8[DDSIZE];

    if((a->deg + 1) <= _BNC_D_WIDTH)
    {
        eval_ddpoly_horner(ret, a, x);
        return;
    }

    // zero8 := 0
    _bncavx512_set0_dd(zero8); // = _mm512_setzero_pd();
    num_loop_unit = 2 * _BNC_D_WIDTH;

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
    {
        long _vl0 = (long)svcntd();
        if((a->deg + 1) <= _vl0)
        {
            eval_ddpoly_horner(ret, a, x);
            return;
        }
        num_loop_unit = 2 * _vl0;
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // ARM Neon
    float64x2_t old_coef2[2][DDSIZE], new_coef2[DDSIZE], a02[DDSIZE], a12[DDSIZE], x2[DDSIZE], zero2[DDSIZE];
    float64x2_t tmp2[DDSIZE];
    float64x2x2_t unzip[DDSIZE];

    if((a->deg + 1) <= _BNC_D_WIDTH)
    {
        eval_ddpoly_horner(ret, a, x);
        return;
    }

    // zero2 := 0
    _bncneon_set0_dd(zero2); // = vdupq_n_f64(0.0)
    num_loop_unit = 2 * _BNC_D_WIDTH;

#else // __AVX2__
    eval_ddpoly_estrin(ret, a, x);
    return;
#endif // __AVX2__

    num_in_coef = a->deg + 1;
    //if((num_in_coef % 2) == 1) num_in_coef++;
    if((num_in_coef % num_loop_unit) != 0) 
        num_in_coef = (long int)ceil(((double)num_in_coef / (double)num_loop_unit)) * num_loop_unit; ////num_in_coef += num_in_coef % num_loop_unit;

    in_degree = num_in_coef - 1;
    //in_coef_old = (double *)calloc(num_in_coef, sizeof(double));
    //in_coef_new = (double *)calloc(num_in_coef, sizeof(double));
    in_coef_old = init_ddvector(num_in_coef);
    //in_coef_new = init_dvector(num_in_coef);

    //for(i = 0; i <= a->deg; i++) in_coef_old[i] = get_dpoly_i(a, i);
    //for(i = a->deg + 1; i <= in_degree; i++) in_coef_old[i] = 0.0;
    for(i = 0; i <= a->deg; i++) set_ddvector_i(in_coef_old, i, get_ddpoly_i(a, i));

    rdd_set(in_x, x); //  = x;

    //printf("num_in_coef = %ld\n", num_in_coef);
    //printf("degree -> in_degree: %d -> %d\n", degree, in_degree);

    while(1)
    //while(num_in_coef > num_loop_unit)
    {
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
        // x2_4 := x^2
        //x4 = _mm256_set1_pd(in_x);
        _bncavx2_rdd_set1_dd(x4, in_x);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            old_coef4[0][0] = _mm256_load_pd(&(in_coef_old->element[0][i * num_loop_unit]));
            old_coef4[1][0] = _mm256_load_pd(&(in_coef_old->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef4[0][1] = _mm256_load_pd(&(in_coef_old->element[1][i * num_loop_unit]));
            old_coef4[1][1] = _mm256_load_pd(&(in_coef_old->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            //printf("i = %ld, num_in_coef = %ld\n", i, num_in_coef);
            //printf("old_coef4[0] = "); PRINT_M256D_SL(old_coef4[0]);
            //printf("old_coef4[1] = "); PRINT_M256D_SL(old_coef4[1]);
            //printf("          x4 = "); PRINT_M256D_SL(x4);
            // old[0] = (a0 + a1 * x)       + (a2 + a3 * x) * x^2
            // old[1] = (a4 + a5 * x) * x^4 + (a6 + a7 * x) * x^6
            // a04    = a0 a2 a4 a6
            // a14    = a1 a3 a5 a7
            // x4     =  x  x  x  x
            // b0 b1 b2 b3 := a04 + a14 * x4
            //        =  b0 + b1 * y + b2 * y^2 + b3 * y^3
            a04[0] = _mm256_unpacklo_pd(old_coef4[0][0], old_coef4[1][0]);
            a14[0] = _mm256_unpackhi_pd(old_coef4[0][0], old_coef4[1][0]);
            a04[1] = _mm256_unpacklo_pd(old_coef4[0][1], old_coef4[1][1]);
            a14[1] = _mm256_unpackhi_pd(old_coef4[0][1], old_coef4[1][1]);
            //printf("         a04 = "); PRINT_M256D_SL(a04);  
            //printf("         a14 = "); PRINT_M256D_SL(a14);
            //new_coef4 = _mm256_fmadd_pd(a14, x4, a04);
            _bncavx2_rdd_mul(new_coef4, a14, x4);
            _bncavx2_rdd_add(new_coef4, new_coef4, a04);
            new_coef4[0] = _mm256_permute4x64_pd(new_coef4[0], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4[1] = _mm256_permute4x64_pd(new_coef4[1], (int)(3*64 + 1*16 + 2*4 + 0));
            // b0, b2, b1, b3 -> b0, b1, b2, b3
            //_mm256_permute4x64_pd(new_coef4, (int)(3*64 + 1*16 + 2*4 + 0));
            //printf("   new_coef4 = "); PRINT_M256D_SL(new_coef4);
            // embed 0s
            _mm256_store_pd(&(in_coef_old->element[0][i * num_loop_unit]), zero4[0]);
            _mm256_store_pd(&(in_coef_old->element[1][i * num_loop_unit]), zero4[1]);
            _mm256_store_pd(&(in_coef_old->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero4[0]);
            _mm256_store_pd(&(in_coef_old->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero4[1]);

            _mm256_store_pd(&(in_coef_old->element[0][i * _BNC_D_WIDTH]), new_coef4[0]);
            _mm256_store_pd(&(in_coef_old->element[1][i * _BNC_D_WIDTH]), new_coef4[1]);
            //printf("in_coef_old  = %f %f %f %f\n", in_coef_old->element[i * _BNC_D_WIDTH], in_coef_old->element[i * _BNC_D_WIDTH + 1], in_coef_old->element[i * _BNC_D_WIDTH + 2], in_coef_old->element[i * _BNC_D_WIDTH + 3]);
        }
#elif defined(__AVX512F__) // __AVX512F__
        // AVX-512: 8-wide SIMD processing for double-double
        _bncavx512_rdd_set1_dd(x8, in_x);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            old_coef8[0][0] = _mm512_load_pd(&(in_coef_old->element[0][i * num_loop_unit]));
            old_coef8[1][0] = _mm512_load_pd(&(in_coef_old->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef8[0][1] = _mm512_load_pd(&(in_coef_old->element[1][i * num_loop_unit]));
            old_coef8[1][1] = _mm512_load_pd(&(in_coef_old->element[1][i * num_loop_unit + _BNC_D_WIDTH]));

            // Unpack: separate even and odd indices
            a08[0] = _mm512_unpacklo_pd(old_coef8[0][0], old_coef8[1][0]);
            a18[0] = _mm512_unpackhi_pd(old_coef8[0][0], old_coef8[1][0]);
            a08[1] = _mm512_unpacklo_pd(old_coef8[0][1], old_coef8[1][1]);
            a18[1] = _mm512_unpackhi_pd(old_coef8[0][1], old_coef8[1][1]);

            // new = a18 * x8 + a08
            _bncavx512_rdd_mul(new_coef8, a18, x8);
            _bncavx512_rdd_add(new_coef8, new_coef8, a08);
            
            // Permute: b0 b2 b4 b6 b1 b3 b5 b7 -> b0 b1 b2 b3 b4 b5 b6 b7
            new_coef8[0] = _mm512_permutex_pd(new_coef8[0], 0xD8); // 0b11011000
            new_coef8[1] = _mm512_permutex_pd(new_coef8[1], 0xD8);
            
            // Embed zeros and store result
            _mm512_store_pd(&(in_coef_old->element[0][i * num_loop_unit]), zero8[0]);
            _mm512_store_pd(&(in_coef_old->element[1][i * num_loop_unit]), zero8[1]);
            _mm512_store_pd(&(in_coef_old->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero8[0]);
            _mm512_store_pd(&(in_coef_old->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero8[1]);

            _mm512_store_pd(&(in_coef_old->element[0][i * _BNC_D_WIDTH]), new_coef8[0]);
            _mm512_store_pd(&(in_coef_old->element[1][i * _BNC_D_WIDTH]), new_coef8[1]);
        }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
        {
            long _vl = (long)svcntd();
            svbool_t _pg = svptrue_b64();
            svfloat64_t _x0 = svdup_n_f64(in_x[0]), _x1 = svdup_n_f64(in_x[1]);
            svfloat64_t _z = svdup_n_f64(0.0);
            for(i = 0; i < (num_in_coef / num_loop_unit); i++)
            {
                /* even/odd deinterleave of the 2*vl coefficients via svuzp1/2 */
                svfloat64_t _v00 = svld1_f64(_pg, &(in_coef_old->element[0][i*num_loop_unit]));
                svfloat64_t _v10 = svld1_f64(_pg, &(in_coef_old->element[0][i*num_loop_unit + _vl]));
                svfloat64_t _v01 = svld1_f64(_pg, &(in_coef_old->element[1][i*num_loop_unit]));
                svfloat64_t _v11 = svld1_f64(_pg, &(in_coef_old->element[1][i*num_loop_unit + _vl]));
                svfloat64_t _e0 = svuzp1_f64(_v00, _v10), _o0 = svuzp2_f64(_v00, _v10);
                svfloat64_t _e1 = svuzp1_f64(_v01, _v11), _o1 = svuzp2_f64(_v01, _v11);
                /* new = even + odd * x   (double-double) */
                svfloat64_t _m0, _m1, _n0, _n1;
                _bncsve2_rdd_mul(_pg, &_m0, &_m1, _o0, _o1, _x0, _x1);
                _bncsve2_rdd_add(_pg, &_n0, &_n1, _e0, _e1, _m0, _m1);
                svst1_f64(_pg, &(in_coef_old->element[0][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old->element[1][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old->element[0][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old->element[1][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old->element[0][i*_vl]), _n0);
                svst1_f64(_pg, &(in_coef_old->element[1][i*_vl]), _n1);
            }
        }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // ARM Neon
        // ARM Neon: 2-wide SIMD processing for double-double
        _bncneon_rdd_set1_dd(x2, in_x);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            old_coef2[0][0] = vld1q_f64(&(in_coef_old->element[0][i * num_loop_unit]));
            old_coef2[1][0] = vld1q_f64(&(in_coef_old->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2[0][1] = vld1q_f64(&(in_coef_old->element[1][i * num_loop_unit]));
            old_coef2[1][1] = vld1q_f64(&(in_coef_old->element[1][i * num_loop_unit + _BNC_D_WIDTH]));

            // 修正版: AVX2と同じロジック
            // old_coef2[0] = [a0, a1], old_coef2[1] = [a2, a3]
            // a02 = [a0, a2], a12 = [a1, a3] を作る
            a02[0] = vzip1q_f64(old_coef2[0][0], old_coef2[1][0]);  // [a0, a2]
            a12[0] = vzip2q_f64(old_coef2[0][0], old_coef2[1][0]);  // [a1, a3]
            a02[1] = vzip1q_f64(old_coef2[0][1], old_coef2[1][1]);
            a12[1] = vzip2q_f64(old_coef2[0][1], old_coef2[1][1]);

            // new = a12 * x2 + a02  →  [b0, b2]
            _bncneon_rdd_mul(new_coef2, a12, x2);
            _bncneon_rdd_add(new_coef2, new_coef2, a02);
            
            // [b0, b2] のまま格納（AVX2のpermute後と同じ順序）
            // 格納は変更なし

            // Embed zeros and store result
            vst1q_f64(&(in_coef_old->element[0][i * num_loop_unit]), zero2[0]);
            vst1q_f64(&(in_coef_old->element[1][i * num_loop_unit]), zero2[1]);
            vst1q_f64(&(in_coef_old->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero2[0]);
            vst1q_f64(&(in_coef_old->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero2[1]);

            vst1q_f64(&(in_coef_old->element[0][i * _BNC_D_WIDTH]), new_coef2[0]);
            vst1q_f64(&(in_coef_old->element[1][i * _BNC_D_WIDTH]), new_coef2[1]);
        }
#endif // __AVX2__

        //return estrin(x * x, coef, num_in_coef / 2 - 1);
        //in_x = in_x * in_x;
        rdd_mul(tmp, in_x, in_x);
        rdd_set(in_x, tmp);
        if(num_in_coef == num_loop_unit) break;
        num_in_coef /= 2;
        if((num_in_coef % num_loop_unit) != 0) // num_in_coef += num_in_coef % num_loop_unit;
            num_in_coef = (long int)ceil(((double)num_in_coef / (double)num_loop_unit)) * num_loop_unit; ////num_in_coef += num_in_coef % num_loop_unit;
        //set0_dvector(in_coef_old);

        //printf("%ld ", num_in_coef);
        in_degree = num_in_coef - 1;

        //for(i = 0; i <= in_degree; i++)
        //    set_dvector_i(in_coef_old, i, get_dvector_i(in_coef_new, i));
            //in_coef_old[i] = in_coef_new[i];
        //set0_dvector(in_coef_new);
    }

    //ret = in_coef_new[0] + in_coef_new[1] * in_x;
    // horner method
    //ret = get_dvector_i(in_coef_new, num_loop_unit - 1);
#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2: final fold over vl coeffs
    {
        long _vl = (long)svcntd();
        rdd_set(ret, get_ddvector_i(in_coef_old, _vl - 1));
        for(i = _vl - 2; i >= 0; i--)
        {
            rdd_mul(tmp, ret, in_x);
            rdd_add(ret, tmp, get_ddvector_i(in_coef_old, i));
        }
    }
#elif defined(__AVX2__) || defined(__AVX512F__) || (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // SIMD
    rdd_set(ret, get_ddvector_i(in_coef_old, _BNC_D_WIDTH - 1));
    for(i = _BNC_D_WIDTH - 2; i >= 0; i--)
    {
        //ret = ret * in_x + get_dvector_i(in_coef_old, i);
        rdd_mul(tmp, ret, in_x);
        rdd_add(ret, tmp, get_ddvector_i(in_coef_old, i));
    }
#endif // SIMD

    //free(in_coef_old);
    //free(in_coef_new);
    free_ddvector(in_coef_old);
    //free_dvector(in_coef_new);

    return; // ret;
}

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
void _bncavx2_ceval_ddpoly_estrin(cddfloat *ret, DDPoly a, cddfloat *x)
{
    //double *in_coef_old, *in_coef_new;
    DDVector in_coef_old_real, in_coef_old_imag;
    cddfloat in_x; // , ret;
    cddfloat in_ret, ctmp;
    long int in_degree, num_in_coef, i, num_loop_unit;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d old_coef4_real[2][DDSIZE], old_coef4_imag[2][DDSIZE];
    __m256d new_coef4_real[DDSIZE], new_coef4_imag[DDSIZE];
    __m256d a04_real[DDSIZE], a04_imag[DDSIZE], a14_real[DDSIZE], a14_imag[DDSIZE], x4_real[DDSIZE], x4_imag[DDSIZE], zero4[DDSIZE];
    __m256d ctmp4_real[DDSIZE], ctmp4_imag[DDSIZE];

    if((a->deg + 1) <= _BNC_D_WIDTH)
    {
        ceval_ddpoly_horner(ret, a, x);
        return;
    }

    // zero4 := 0
    _bncavx2_rdd_set0(zero4); //  = _mm256_setzero_pd();
    num_loop_unit = 2 * _BNC_D_WIDTH;

#elif defined(__AVX512F__) // __AVX512F__
	__m512d old_coef8_real[2][DDSIZE], old_coef8_imag[2][DDSIZE];
    __m512d new_coef8_real[DDSIZE], new_coef8_imag[DDSIZE];
    __m512d a08_real[DDSIZE], a08_imag[DDSIZE], a18_real[DDSIZE], a18_imag[DDSIZE], x8_real[DDSIZE], x8_imag[DDSIZE], zero8[DDSIZE];
    __m512d ctmp8_real[DDSIZE], ctmp8_imag[DDSIZE];

    if((a->deg + 1) <= _BNC_D_WIDTH)
    {
        ceval_ddpoly_horner(ret, a, x);
        return;
    }

    // zero8 := 0
    _bncavx512_rdd_set0(zero8); // = _mm512_setzero_pd();
    num_loop_unit = 2 * _BNC_D_WIDTH;

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
    {
        long _vl0 = (long)svcntd();
        if((a->deg + 1) <= _vl0)
        {
            ceval_ddpoly_horner(ret, a, x);
            return;
        }
        num_loop_unit = 2 * _vl0;
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // ARM Neon
    float64x2_t old_coef2_real[2][DDSIZE], old_coef2_imag[2][DDSIZE];
    float64x2_t new_coef2_real[DDSIZE], new_coef2_imag[DDSIZE];
    float64x2_t a02_real[DDSIZE], a02_imag[DDSIZE], a12_real[DDSIZE], a12_imag[DDSIZE], x2_real[DDSIZE], x2_imag[DDSIZE], zero2[DDSIZE];
    float64x2_t ctmp2_real[DDSIZE], ctmp2_imag[DDSIZE];
    float64x2x2_t unzip_real[DDSIZE];
    float64x2x2_t unzip_imag[DDSIZE];

    if((a->deg + 1) <= _BNC_D_WIDTH)
    {
        ceval_ddpoly_horner(ret, a, x);
        return;
    }

    // zero2 := 0
    _bncneon_rdd_set0(zero2); // = vdupq_n_f64(0.0)
    num_loop_unit = 2 * _BNC_D_WIDTH;

#else // __AVX2__
    ceval_ddpoly_estrin(ret, a, x);
    return;
#endif // __AVX2__

    num_in_coef = a->deg + 1;
    //if((num_in_coef % 2) == 1) num_in_coef++;
    if((num_in_coef % num_loop_unit) != 0) 
        num_in_coef = (long int)ceil(((double)num_in_coef / (double)num_loop_unit)) * num_loop_unit; ////num_in_coef += num_in_coef % num_loop_unit;

    in_degree = num_in_coef - 1;
    //in_coef_old = (double *)calloc(num_in_coef, sizeof(double));
    //in_coef_new = (double *)calloc(num_in_coef, sizeof(double));
    in_coef_old_real = init_ddvector(num_in_coef);
    in_coef_old_imag = init_ddvector(num_in_coef);
    //in_coef_new = init_dvector(num_in_coef);

    //for(i = 0; i <= a->deg; i++) in_coef_old[i] = get_dpoly_i(a, i);
    //for(i = a->deg + 1; i <= in_degree; i++) in_coef_old[i] = 0.0;
    for(i = 0; i <= a->deg; i++) set_ddvector_i(in_coef_old_real, i, get_ddpoly_i(a, i));

    //in_x = x->re + x->im * I;
    rcdd_set(&in_x, x);

    //printf("num_in_coef = %ld\n", num_in_coef);
    //printf("degree -> in_degree: %d -> %d\n", degree, in_degree);

    while(1)
    //while(num_in_coef > num_loop_unit)
    {
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
        // x2_4 := x^2
        //x4 = _mm256_set1_pd(in_x);
        _bncavx2_rdd_set1_dd(x4_real, in_x.val_re);
        _bncavx2_rdd_set1_dd(x4_imag, in_x.val_im);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            old_coef4_real[0][0] = _mm256_load_pd(&(in_coef_old_real->element[0][i * num_loop_unit]));
            old_coef4_real[1][0] = _mm256_load_pd(&(in_coef_old_real->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef4_real[0][1] = _mm256_load_pd(&(in_coef_old_real->element[1][i * num_loop_unit]));
            old_coef4_real[1][1] = _mm256_load_pd(&(in_coef_old_real->element[1][i * num_loop_unit + _BNC_D_WIDTH]));

            old_coef4_imag[0][0] = _mm256_load_pd(&(in_coef_old_imag->element[0][i * num_loop_unit]));
            old_coef4_imag[1][0] = _mm256_load_pd(&(in_coef_old_imag->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef4_imag[0][1] = _mm256_load_pd(&(in_coef_old_imag->element[1][i * num_loop_unit]));
            old_coef4_imag[1][1] = _mm256_load_pd(&(in_coef_old_imag->element[1][i * num_loop_unit + _BNC_D_WIDTH]));

            //printf("i = %ld, num_in_coef = %ld\n", i, num_in_coef);
            //printf("old_coef4[0] = "); PRINT_M256D_SL(old_coef4[0]);
            //printf("old_coef4[1] = "); PRINT_M256D_SL(old_coef4[1]);
            //printf("          x4 = "); PRINT_M256D_SL(x4);
            // old[0] = (a0 + a1 * x)       + (a2 + a3 * x) * x^2
            // old[1] = (a4 + a5 * x) * x^4 + (a6 + a7 * x) * x^6
            // a04    = a0 a2 a4 a6
            // a14    = a1 a3 a5 a7
            // x4     =  x  x  x  x
            // b0 b1 b2 b3 := a04 + a14 * x4
            //        =  b0 + b1 * y + b2 * y^2 + b3 * y^3
            a04_real[0] = _mm256_unpacklo_pd(old_coef4_real[0][0], old_coef4_real[1][0]);
            a14_real[0] = _mm256_unpackhi_pd(old_coef4_real[0][0], old_coef4_real[1][0]);
            a04_real[1] = _mm256_unpacklo_pd(old_coef4_real[0][1], old_coef4_real[1][1]);
            a14_real[1] = _mm256_unpackhi_pd(old_coef4_real[0][1], old_coef4_real[1][1]);

            a04_imag[0] = _mm256_unpacklo_pd(old_coef4_imag[0][0], old_coef4_imag[1][0]);
            a14_imag[0] = _mm256_unpackhi_pd(old_coef4_imag[0][0], old_coef4_imag[1][0]);
            a04_imag[1] = _mm256_unpacklo_pd(old_coef4_imag[0][1], old_coef4_imag[1][1]);
            a14_imag[1] = _mm256_unpackhi_pd(old_coef4_imag[0][1], old_coef4_imag[1][1]);
             //printf("         a04 = "); PRINT_M256D_SL(a04);  
            //printf("         a14 = "); PRINT_M256D_SL(a14);
            //new_coef4 = _mm256_fmadd_pd(a14, x4, a04);

            // a14 * x4 -> (a14_re * x4_re - a14_im * x4_im) + (a14_re * x4_im + a14_im * x4_re) * I
            //new_coef4_real = _mm256_sub_pd(_mm256_mul_pd(a14_real, x4_real), _mm256_mul_pd(a14_imag, x4_imag));
            //new_coef4_imag = _mm256_add_pd(_mm256_mul_pd(a14_real, x4_imag), _mm256_mul_pd(a14_imag, x4_real));
            _bncavx2_rcdd_mul(ctmp4_real, ctmp4_imag, a14_real, a14_imag, x4_real, x4_imag);

            // a14_x4 + a04 -> a14_x4_re + a_04_re + (a_14_x4_im + a_04_im) * I
            //new_coef4_real = _mm256_add_pd(new_coef4_real, a04_real);
            //new_coef4_imag = _mm256_add_pd(new_coef4_imag, a04_imag);
            _bncavx2_rcdd_add(new_coef4_real, new_coef4_imag, ctmp4_real, ctmp4_imag, a04_real, a04_imag);

            new_coef4_real[0] = _mm256_permute4x64_pd(new_coef4_real[0], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_imag[0] = _mm256_permute4x64_pd(new_coef4_imag[0], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_real[1] = _mm256_permute4x64_pd(new_coef4_real[1], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_imag[1] = _mm256_permute4x64_pd(new_coef4_imag[1], (int)(3*64 + 1*16 + 2*4 + 0));
            // b0, b2, b1, b3 -> b0, b1, b2, b3
            //_mm256_permute4x64_pd(new_coef4, (int)(3*64 + 1*16 + 2*4 + 0));
            //printf("   new_coef4 = "); PRINT_M256D_SL(new_coef4);
            // embed 0s
            _mm256_store_pd(&(in_coef_old_real->element[0][i * num_loop_unit]), zero4[0]);
            _mm256_store_pd(&(in_coef_old_imag->element[0][i * num_loop_unit]), zero4[0]);
            _mm256_store_pd(&(in_coef_old_real->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero4[0]);
            _mm256_store_pd(&(in_coef_old_imag->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero4[0]);
            _mm256_store_pd(&(in_coef_old_real->element[1][i * num_loop_unit]), zero4[1]);
            _mm256_store_pd(&(in_coef_old_imag->element[1][i * num_loop_unit]), zero4[1]);
            _mm256_store_pd(&(in_coef_old_real->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero4[1]);
            _mm256_store_pd(&(in_coef_old_imag->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero4[1]);

            _mm256_store_pd(&(in_coef_old_real->element[0][i * _BNC_D_WIDTH]), new_coef4_real[0]);
            _mm256_store_pd(&(in_coef_old_imag->element[0][i * _BNC_D_WIDTH]), new_coef4_imag[0]);
            _mm256_store_pd(&(in_coef_old_real->element[1][i * _BNC_D_WIDTH]), new_coef4_real[1]);
            _mm256_store_pd(&(in_coef_old_imag->element[1][i * _BNC_D_WIDTH]), new_coef4_imag[1]);
            //printf("in_coef_old_real  = %f %f %f %f\n", in_coef_old_real->element[0][i * _BNC_D_WIDTH], in_coef_old_real->element[0][i * _BNC_D_WIDTH + 1], in_coef_old_real->element[0][i * _BNC_D_WIDTH + 2], in_coef_old_real->element[0][i * _BNC_D_WIDTH + 3]);
        }
#elif defined(__AVX512F__) // __AVX512F__
        // AVX-512: 8-wide SIMD processing for complex double-double
        _bncavx512_rdd_set1_dd(x8_real, in_x.val_re);
        _bncavx512_rdd_set1_dd(x8_imag, in_x.val_im);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            // Load real parts
            old_coef8_real[0][0] = _mm512_load_pd(&(in_coef_old_real->element[0][i * num_loop_unit]));
            old_coef8_real[1][0] = _mm512_load_pd(&(in_coef_old_real->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef8_real[0][1] = _mm512_load_pd(&(in_coef_old_real->element[1][i * num_loop_unit]));
            old_coef8_real[1][1] = _mm512_load_pd(&(in_coef_old_real->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            
            // Load imaginary parts
            old_coef8_imag[0][0] = _mm512_load_pd(&(in_coef_old_imag->element[0][i * num_loop_unit]));
            old_coef8_imag[1][0] = _mm512_load_pd(&(in_coef_old_imag->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef8_imag[0][1] = _mm512_load_pd(&(in_coef_old_imag->element[1][i * num_loop_unit]));
            old_coef8_imag[1][1] = _mm512_load_pd(&(in_coef_old_imag->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            
            // Unpack: separate even and odd indices
            a08_real[0] = _mm512_unpacklo_pd(old_coef8_real[0][0], old_coef8_real[1][0]);
            a18_real[0] = _mm512_unpackhi_pd(old_coef8_real[0][0], old_coef8_real[1][0]);
            a08_real[1] = _mm512_unpacklo_pd(old_coef8_real[0][1], old_coef8_real[1][1]);
            a18_real[1] = _mm512_unpackhi_pd(old_coef8_real[0][1], old_coef8_real[1][1]);
            
            a08_imag[0] = _mm512_unpacklo_pd(old_coef8_imag[0][0], old_coef8_imag[1][0]);
            a18_imag[0] = _mm512_unpackhi_pd(old_coef8_imag[0][0], old_coef8_imag[1][0]);
            a08_imag[1] = _mm512_unpacklo_pd(old_coef8_imag[0][1], old_coef8_imag[1][1]);
            a18_imag[1] = _mm512_unpackhi_pd(old_coef8_imag[0][1], old_coef8_imag[1][1]);

            // Complex multiplication: a18 * x8
            _bncavx512_rcdd_mul(ctmp8_real, ctmp8_imag, a18_real, a18_imag, x8_real, x8_imag);

            // Add a08: new = a18*x8 + a08
            _bncavx512_rcdd_add(new_coef8_real, new_coef8_imag, ctmp8_real, ctmp8_imag, a08_real, a08_imag);
            
            // Permute: b0 b2 b4 b6 b1 b3 b5 b7 -> b0 b1 b2 b3 b4 b5 b6 b7
            new_coef8_real[0] = _mm512_permutex_pd(new_coef8_real[0], 0xD8);
            new_coef8_imag[0] = _mm512_permutex_pd(new_coef8_imag[0], 0xD8);
            new_coef8_real[1] = _mm512_permutex_pd(new_coef8_real[1], 0xD8);
            new_coef8_imag[1] = _mm512_permutex_pd(new_coef8_imag[1], 0xD8);
            
            // Embed zeros and store result
            _mm512_store_pd(&(in_coef_old_real->element[0][i * num_loop_unit]), zero8[0]);
            _mm512_store_pd(&(in_coef_old_imag->element[0][i * num_loop_unit]), zero8[0]);
            _mm512_store_pd(&(in_coef_old_real->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero8[0]);
            _mm512_store_pd(&(in_coef_old_imag->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero8[0]);
            _mm512_store_pd(&(in_coef_old_real->element[1][i * num_loop_unit]), zero8[1]);
            _mm512_store_pd(&(in_coef_old_imag->element[1][i * num_loop_unit]), zero8[1]);
            _mm512_store_pd(&(in_coef_old_real->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero8[1]);
            _mm512_store_pd(&(in_coef_old_imag->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero8[1]);

            _mm512_store_pd(&(in_coef_old_real->element[0][i * _BNC_D_WIDTH]), new_coef8_real[0]);
            _mm512_store_pd(&(in_coef_old_imag->element[0][i * _BNC_D_WIDTH]), new_coef8_imag[0]);
            _mm512_store_pd(&(in_coef_old_real->element[1][i * _BNC_D_WIDTH]), new_coef8_real[1]);
            _mm512_store_pd(&(in_coef_old_imag->element[1][i * _BNC_D_WIDTH]), new_coef8_imag[1]);
        }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
        {
            long _vl = (long)svcntd();
            svbool_t _pg = svptrue_b64();
            svfloat64_t _xr0 = svdup_n_f64(in_x.val_re[0]), _xr1 = svdup_n_f64(in_x.val_re[1]);
            svfloat64_t _xi0 = svdup_n_f64(in_x.val_im[0]), _xi1 = svdup_n_f64(in_x.val_im[1]);
            svfloat64_t _z = svdup_n_f64(0.0);
            for(i = 0; i < (num_in_coef / num_loop_unit); i++)
            {
                svfloat64_t _r00 = svld1_f64(_pg, &(in_coef_old_real->element[0][i*num_loop_unit]));
                svfloat64_t _r10 = svld1_f64(_pg, &(in_coef_old_real->element[0][i*num_loop_unit + _vl]));
                svfloat64_t _r01 = svld1_f64(_pg, &(in_coef_old_real->element[1][i*num_loop_unit]));
                svfloat64_t _r11 = svld1_f64(_pg, &(in_coef_old_real->element[1][i*num_loop_unit + _vl]));
                svfloat64_t _i00 = svld1_f64(_pg, &(in_coef_old_imag->element[0][i*num_loop_unit]));
                svfloat64_t _i10 = svld1_f64(_pg, &(in_coef_old_imag->element[0][i*num_loop_unit + _vl]));
                svfloat64_t _i01 = svld1_f64(_pg, &(in_coef_old_imag->element[1][i*num_loop_unit]));
                svfloat64_t _i11 = svld1_f64(_pg, &(in_coef_old_imag->element[1][i*num_loop_unit + _vl]));
                /* even/odd deinterleave (real & imag, both DD limbs) */
                svfloat64_t _er0 = svuzp1_f64(_r00,_r10), _or0 = svuzp2_f64(_r00,_r10);
                svfloat64_t _er1 = svuzp1_f64(_r01,_r11), _or1 = svuzp2_f64(_r01,_r11);
                svfloat64_t _ei0 = svuzp1_f64(_i00,_i10), _oi0 = svuzp2_f64(_i00,_i10);
                svfloat64_t _ei1 = svuzp1_f64(_i01,_i11), _oi1 = svuzp2_f64(_i01,_i11);
                /* odd * x (complex DD): pr = or*xr - oi*xi,  pi = or*xi + oi*xr */
                svfloat64_t _a0,_a1,_b0,_b1,_pr0,_pr1,_pi0,_pi1;
                _bncsve2_rdd_mul(_pg, &_a0,&_a1, _or0,_or1, _xr0,_xr1);
                _bncsve2_rdd_mul(_pg, &_b0,&_b1, _oi0,_oi1, _xi0,_xi1);
                _b0 = svneg_f64_x(_pg,_b0); _b1 = svneg_f64_x(_pg,_b1);
                _bncsve2_rdd_add(_pg, &_pr0,&_pr1, _a0,_a1, _b0,_b1);
                _bncsve2_rdd_mul(_pg, &_a0,&_a1, _or0,_or1, _xi0,_xi1);
                _bncsve2_rdd_mul(_pg, &_b0,&_b1, _oi0,_oi1, _xr0,_xr1);
                _bncsve2_rdd_add(_pg, &_pi0,&_pi1, _a0,_a1, _b0,_b1);
                /* new = even + odd*x  (complex) */
                svfloat64_t _nr0,_nr1,_ni0,_ni1;
                _bncsve2_rdd_add(_pg, &_nr0,&_nr1, _er0,_er1, _pr0,_pr1);
                _bncsve2_rdd_add(_pg, &_ni0,&_ni1, _ei0,_ei1, _pi0,_pi1);
                svst1_f64(_pg, &(in_coef_old_real->element[0][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old_real->element[1][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old_real->element[0][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old_real->element[1][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old_imag->element[0][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old_imag->element[1][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old_imag->element[0][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old_imag->element[1][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old_real->element[0][i*_vl]), _nr0);
                svst1_f64(_pg, &(in_coef_old_real->element[1][i*_vl]), _nr1);
                svst1_f64(_pg, &(in_coef_old_imag->element[0][i*_vl]), _ni0);
                svst1_f64(_pg, &(in_coef_old_imag->element[1][i*_vl]), _ni1);
            }
        }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // ARM Neon
        // ARM Neon: 2-wide SIMD processing for complex double-double
        _bncneon_rdd_set1_dd(x2_real, in_x.val_re);
        _bncneon_rdd_set1_dd(x2_imag, in_x.val_im);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            // Load real parts
            old_coef2_real[0][0] = vld1q_f64(&(in_coef_old_real->element[0][i * num_loop_unit]));
            old_coef2_real[1][0] = vld1q_f64(&(in_coef_old_real->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_real[0][1] = vld1q_f64(&(in_coef_old_real->element[1][i * num_loop_unit]));
            old_coef2_real[1][1] = vld1q_f64(&(in_coef_old_real->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            
            // Load imaginary parts
            old_coef2_imag[0][0] = vld1q_f64(&(in_coef_old_imag->element[0][i * num_loop_unit]));
            old_coef2_imag[1][0] = vld1q_f64(&(in_coef_old_imag->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_imag[0][1] = vld1q_f64(&(in_coef_old_imag->element[1][i * num_loop_unit]));
            old_coef2_imag[1][1] = vld1q_f64(&(in_coef_old_imag->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            
           // Separate even/odd coefficients - Real part
            a02_real[0] = vuzp1q_f64(old_coef2_real[0][0], old_coef2_real[1][0]);
            a12_real[0] = vuzp2q_f64(old_coef2_real[0][0], old_coef2_real[1][0]);
            a02_real[1] = vuzp1q_f64(old_coef2_real[0][1], old_coef2_real[1][1]);
            a12_real[1] = vuzp2q_f64(old_coef2_real[0][1], old_coef2_real[1][1]);

            // Separate even/odd coefficients - Imaginary part
            a02_imag[0] = vuzp1q_f64(old_coef2_imag[0][0], old_coef2_imag[1][0]);
            a12_imag[0] = vuzp2q_f64(old_coef2_imag[0][0], old_coef2_imag[1][0]);
            a02_imag[1] = vuzp1q_f64(old_coef2_imag[0][1], old_coef2_imag[1][1]);
            a12_imag[1] = vuzp2q_f64(old_coef2_imag[0][1], old_coef2_imag[1][1]);

            // Complex multiplication: a12 * x2
            _bncneon_rcdd_mul(ctmp2_real, ctmp2_imag, a12_real, a12_imag, x2_real, x2_imag);

            // Add a02: new = a12*x2 + a02
            _bncneon_rcdd_add(new_coef2_real, new_coef2_imag, ctmp2_real, ctmp2_imag, a02_real, a02_imag);
            
            // Embed zeros and store result
            vst1q_f64(&(in_coef_old_real->element[0][i * num_loop_unit]), zero2[0]);
            vst1q_f64(&(in_coef_old_imag->element[0][i * num_loop_unit]), zero2[0]);
            vst1q_f64(&(in_coef_old_real->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero2[0]);
            vst1q_f64(&(in_coef_old_imag->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero2[0]);
            vst1q_f64(&(in_coef_old_real->element[1][i * num_loop_unit]), zero2[1]);
            vst1q_f64(&(in_coef_old_imag->element[1][i * num_loop_unit]), zero2[1]);
            vst1q_f64(&(in_coef_old_real->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero2[1]);
            vst1q_f64(&(in_coef_old_imag->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero2[1]);

            vst1q_f64(&(in_coef_old_real->element[0][i * _BNC_D_WIDTH]), new_coef2_real[0]);
            vst1q_f64(&(in_coef_old_imag->element[0][i * _BNC_D_WIDTH]), new_coef2_imag[0]);
            vst1q_f64(&(in_coef_old_real->element[1][i * _BNC_D_WIDTH]), new_coef2_real[1]);
            vst1q_f64(&(in_coef_old_imag->element[1][i * _BNC_D_WIDTH]), new_coef2_imag[1]);
        }
#endif // __AVX2__

        //return estrin(x * x, coef, num_in_coef / 2 - 1);
        //in_x = in_x * in_x;
        rcdd_mul(&ctmp, &in_x, &in_x);
        rcdd_set(&in_x, &ctmp);
        if(num_in_coef == num_loop_unit) break;
        num_in_coef /= 2;
        if((num_in_coef % num_loop_unit) != 0) // num_in_coef += num_in_coef % num_loop_unit;
            num_in_coef = (long int)ceil(((double)num_in_coef / (double)num_loop_unit)) * num_loop_unit; ////num_in_coef += num_in_coef % num_loop_unit;
        //set0_dvector(in_coef_old);

        //printf("%ld ", num_in_coef);
        in_degree = num_in_coef - 1;

        //for(i = 0; i <= in_degree; i++)
        //    set_dvector_i(in_coef_old, i, get_dvector_i(in_coef_new, i));
            //in_coef_old[i] = in_coef_new[i];
        //set0_dvector(in_coef_new);
    }

    //ret = in_coef_new[0] + in_coef_new[1] * in_x;
    // horner method
    //ret = get_dvector_i(in_coef_new, num_loop_unit - 1);
#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2: final complex fold over vl coeffs
    {
        long _vl = (long)svcntd();
        rdd_set(in_ret.val_re, get_ddvector_i(in_coef_old_real, _vl - 1));
        rdd_set(in_ret.val_im, get_ddvector_i(in_coef_old_imag, _vl - 1));
        for(i = _vl - 2; i >= 0; i--)
        {
            rcdd_mul(&ctmp, &in_ret, &in_x);
            rdd_set(in_ret.val_re, get_ddvector_i(in_coef_old_real, i));
            rdd_set(in_ret.val_im, get_ddvector_i(in_coef_old_imag, i));
            rcdd_add(&in_ret, &in_ret, &ctmp);
        }
    }
#elif defined(__AVX2__) || defined(__AVX512F__) || (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // SIMD
    //in_ret = get_dvector_i(in_coef_old_real, _BNC_D_WIDTH - 1) + get_dvector_i(in_coef_old_imag, _BNC_D_WIDTH - 1) * I;
    rdd_set(in_ret.val_re, get_ddvector_i(in_coef_old_real, _BNC_D_WIDTH - 1));
    rdd_set(in_ret.val_im, get_ddvector_i(in_coef_old_imag, _BNC_D_WIDTH - 1));
    for(i = _BNC_D_WIDTH - 2; i >= 0; i--)
    {
        //in_ret = in_ret * in_x + (get_dvector_i(in_coef_old_real, i) + get_dvector_i(in_coef_old_imag, i) * I);
        rcdd_mul(&ctmp, &in_ret, &in_x);
        rdd_set(in_ret.val_re, get_ddvector_i(in_coef_old_real, i));
        rdd_set(in_ret.val_im, get_ddvector_i(in_coef_old_imag, i));
        rcdd_add(&in_ret, &in_ret, &ctmp);
    }
#endif // SIMD

    //ret->re = creal(in_ret);
    //ret->im = cimag(in_ret);
    rcdd_set(ret, &in_ret);

    //free(in_coef_old);
    //free(in_coef_new);
    free_ddvector(in_coef_old_real);
    free_ddvector(in_coef_old_imag);
    //free_dvector(in_coef_new);

    //return ret;
    return;
}
#ifdef __cpluplus
} // extern "C" {
#endif // __cplusplus
