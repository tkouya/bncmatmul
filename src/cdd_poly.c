/********************************************************************************/
/* cdd_poly.c: Algebraic Equations and Complex Polynomials                      */
/* copyright (c) 2025 Tomonori Kouya                                            */
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
/*   CDDPoly init_cddpoly(long int max_length)   */
/*   CDDPoly init2_cddpoly(long int max_length, unsigned long prec)*/
/* Free:                                         */
/*   void free_cddpoly(CDDPoly pol)              */
/* Get & Set Values:                             */
/*   cddfloat *get_cddpoly_i(CDDPoly pol, long int index) */
/*   long int setdegree_cddpoly(CDDPoly)         */
/*   void set_cddpoly_i(CDDPoly pol, long int index, cddfloat val) */
/* Output:                                       */
/*   void print_cddpoly(CDDPoly pol)             */
/*************************************************/
CDDPoly init_cddpoly(long int max_length)
{
	CDDPoly ret = NULL;
	long int i;

	if(max_length <= 0)
	{
		fprintf(stderr, "ERROR: init_cddpoly\n");
		return ret;
	}

	ret = (CDDPoly)malloc(sizeof(cddpoly));
	if(ret == NULL)
		return ret;

	ret->coef = (cddfloat *)calloc(sizeof(cddfloat), max_length);
	if(ret->coef == NULL)
    {
        free(ret);
		return NULL;
    }

	/* All 0 */
	for(i = 0; i < max_length; i++)
		rcdd_set_ui(&(ret->coef[i]), 0UL);

    // zero := 0
    rcdd_set0(&(ret->zero));

	ret->deg = 0;
	ret->max_len = max_length;

	return ret;
}

void free_cddpoly(CDDPoly pol)
{
	long int i;

	if(pol == NULL)
		return;

	if(pol->coef != NULL)
		free(pol->coef); // Fix! 2012-06-03 by T.Kouya

	free(pol);
}

cddfloat *get_cddpoly_i(CDDPoly pol, long int index)
{
	if(index > pol->deg)
		return &(pol->zero);
	else
		return &(pol->coef[index]);
}

/* find maximum degree at pol->coef[deg] != 0 */
long int setdegree_cddpoly(CDDPoly pol)
{
	long int i;
    ddfloat tmp;

	for(i = pol->max_len - 1; i > 0; i--)
	{
        rcdd_abs(&tmp, get_cddpoly_i(pol, i));
		if(rdd_cmp_ui(tmp.val, 0UL) != 0)
		{
			pol->deg = i;
			return i;
		}
	}

	pol->deg = 0;
	return 0;

}

void set_cddpoly_i(CDDPoly pol, long int index, cddfloat *val)
{
    ddfloat tmp;

    if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	rcdd_set(&(pol->coef[index]), val);
    rcdd_abs(&tmp, val);
	if((pol->deg < index) && (rdd_cmp_ui(tmp.val, 0UL) != 0))
		pol->deg = index;
}

#ifdef USE_GMP
void set_cddpoly_i_mpc(CDDPoly pol, long int index, mpc_t val)
{
	cddfloat ctmp;
    ddfloat tmp;

	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	mpf_get_dd(ctmp.val_re, mpc_realref(val));
    mpf_get_dd(ctmp.val_im, mpc_imagref(val));
    set_cddpoly_i(pol, index, &ctmp);
	//rcdd_set(pol->coef[index].val, tmp);
    rcdd_abs(&tmp, &ctmp);

    if((pol->deg < index) && (rdd_cmp_ui(tmp.val, 0UL) != 0))
		pol->deg = index;

}

// Initialize and substitute polynomial from org_pol
CDDPoly init_set_cddpoly_cmpfpoly(CMPFPoly org_pol)
{
	CDDPoly ret = NULL;
	long int i;

	ret = init_cddpoly(org_pol->max_len);
	if(ret == NULL)
		return ret;

	for(i = 0; i <= org_pol->deg; i++)
		set_cddpoly_i_mpc(ret, i, get_cmpfpoly_i(org_pol, i));

	return ret;
}

#endif // USE_GMP

// Initialize and substitute polynomial from org_pol
CDDPoly init_set_cddpoly(CDDPoly org_pol)
{
	CDDPoly ret = NULL;
	long int i;

	ret = init_cddpoly(org_pol->max_len);
	if(ret == NULL)
		return ret;

	subst_cddpoly(ret, org_pol);

	return ret;
}

// Initialize and substitute polynomial from org_pol
CDDPoly init_set_cddpoly_ddpoly(DDPoly org_pol)
{
	CDDPoly ret = NULL;
	long int i;
	cddfloat ctmp;

	ret = init_cddpoly(org_pol->max_len);
	if(ret == NULL)
		return ret;

	rdd_set0(ctmp.val_im);
	for(i = 0; i <= org_pol->deg; i++)
	{
		rdd_set(ctmp.val_re, get_ddpoly_i(org_pol, i));
		set_cddpoly_i(ret, i, &ctmp);
	}

	return ret;
}

void print_cddpoly(CDDPoly pol)
{
	long int i;

	for(i = 0; i <= pol->deg; i++)
	{
		printf("%5ld ", i);
		rcdd_out_str(get_cddpoly_i(pol, i));
		printf("\n");
	}
}

/*************************************************/
/* Poly Calculations for CDDPoly             */
/*
void add_cddpoly(CDDPoly c, CDDPoly a, CDDPoly b)
void sub_cddpoly(CDDPoly c, CDDPoly a, CDDPoly b)
void cmul_cddpoly(CDDPoly c, double val[DDSIZE], CDDPoly a)
void subst_cddpoly(CDDPoly c, CDDPoly a)

void diff_cddpoly(CDDPoly a)
void eval_cddpoly(double ret, CDDPoly a, double x[DDSIZE])
void eval_diff_cddpoly(double ret, CDDPoly a, double x[DDSIZE])
*/
/*************************************************/

/* c = a + b */
void add_cddpoly(CDDPoly c, CDDPoly a, CDDPoly b)
{
	long int i;
	cddfloat tmp;

	if((c->max_len < a->deg) || (c->max_len < b->deg))
	{
		fprintf(stderr, "ERROR: add_cddpoly\n");
		return;
	}

	c->deg = (a->deg > b->deg) ? a->deg : b->deg;

	for(i = 0; i <= c->deg; i++)
	{
		rcdd_add(&tmp, get_cddpoly_i(a, i), get_cddpoly_i(b, i));
		set_cddpoly_i(c, i, &tmp);
	}
}

/* c += a */
void add2_cddpoly(CDDPoly c, CDDPoly a)
{
	long int i;
	cddfloat tmp;

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: add2_cddpoly\n");
		return;
	}

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
	{
		rcdd_add(&tmp, get_cddpoly_i(c, i), get_cddpoly_i(a, i));
		set_cddpoly_i(c, i, &tmp);
	}
}

/* c = a - b */
void sub_cddpoly(CDDPoly c, CDDPoly a, CDDPoly b)
{
	long int i;
	cddfloat tmp;

	if((c->max_len < a->deg) || (c->max_len < b->deg))
	{
		fprintf(stderr, "ERROR: sub_cddpoly\n");
		return;
	}

	c->deg = (a->deg > b->deg) ? a->deg : b->deg;

	for(i = 0; i <= c->deg; i++)
	{
		rcdd_sub(&tmp, get_cddpoly_i(a, i), get_cddpoly_i(b, i));
		set_cddpoly_i(c, i, &tmp);
	}
}

/* c -= a */
void sub2_cddpoly(CDDPoly c, CDDPoly a)
{
	long int i, min_deg;
	cddfloat tmp;

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: sub2_cddpoly\n");
		return;
	}

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
	{
		rcdd_sub(&tmp, get_cddpoly_i(c, i), get_cddpoly_i(a, i));
		set_cddpoly_i(c, i, &tmp);
	}
}

/* c = a * b */
void mul_cddpoly(CDDPoly c, CDDPoly a, CDDPoly b)
{
	long int i, j;
	cddfloat tmp;

	if(c->max_len < (a->deg + b->deg))
	{
		fprintf(stderr, "ERROR: mul_cddpoly\n");
		return;
	}

	/* set c = 0 */
	set0_cddpoly(c);

	for(i = 0; i <= a->deg; i++)
	{
		for(j = 0; j <= b->deg; j++)
		{
			rcdd_set(&tmp, get_cddpoly_i(c, i + j));
			rcdd_fma(&tmp, get_cddpoly_i(a, i), get_cddpoly_i(b, j), &tmp);
			set_cddpoly_i(c, i + j, &tmp);
		}
	}

	c->deg = setdegree_cddpoly(c);
//	c->deg = a->deg + b->deg;
}

/* c = val * a */
void cmul_cddpoly(CDDPoly c, cddfloat *val, CDDPoly a)
{
	long int i;
	cddfloat tmp;

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: cmul_cddpoly\n");
		return;
	}

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
	{
		rcdd_mul(&tmp, val, get_cddpoly_i(a, i));
		set_cddpoly_i(c, i, &tmp);
	}
}

/* c *= val */
void cmul2_cddpoly(CDDPoly c, cddfloat *val)
{
	long int i;
	cddfloat tmp;

	for(i = 0; i <= c->deg; i++)
	{
		rcdd_mul(&tmp, val, get_cddpoly_i(c, i));
		set_cddpoly_i(c, i, &tmp);
	}
}

/* c := a */
void subst_cddpoly(CDDPoly c, CDDPoly a)
{
	long int i;

	if(c->deg > a->deg)
	{
		for(i = c->deg; i >= a->deg; i--)
			set_cddpoly_i(c, i, &(c->zero));
	}

	c->deg = a->deg;
	for(i = 0; i <= c->deg; i++)
		set_cddpoly_i(c, i, get_cddpoly_i(a, i));
}

/* c := 0 */
void set0_cddpoly(CDDPoly c)
{
	long int i;
	cddfloat tmp;

	rcdd_set0(&tmp);

	for(i = 0; i <= c->deg; i++)
		set_cddpoly_i(c, i, &tmp);
}

/* number of nonzero coef */
long int num_nonzero_cddpoly(CDDPoly c)
{
	long int i, ret;
    ddfloat tmp;

	ret = 0;
	for(i = 0; i <= c->deg; i++)
	{
        rcdd_abs(&tmp, get_cddpoly_i(c, i));
		if(rdd_cmp_ui(tmp.val, 0UL) != 0)
			ret++;
	}

	return ret;
}

/* get maximum |coef| */
long int max_abscoef_cddpoly(CDDPoly c)
{
	long int i, ret;
	ddfloat tmp1, tmp;

	rcdd_abs(&tmp1, get_cddpoly_i(c, 0));
	ret = 0;
	for(i = 1; i <= c->deg; i++)
	{
		rcdd_abs(&tmp, get_cddpoly_i(c, i));
		if(rdd_cmp(tmp1.val, tmp.val) < 0)
		{
			rdd_set(tmp1.val, tmp.val);
			ret = i;
		}
	}

	return ret;
}

/* a := a'(x) */
void diff_cddpoly(CDDPoly a)
{
	long int diff_deg, i;
	cddfloat tmp_coef;

	diff_deg = a->deg - 1;

	if(diff_deg < 0)
	{
		set0_cddpoly(a);
		a->deg = 0;
		return;
	}

	//for(i = 1; i <= diff_deg; i++)
	for(i = 1; i <= a->deg; i++) // Fix!! : 2007-01-11
	{
		rcdd_mul_ui(&tmp_coef, get_cddpoly_i(a, i), (unsigned long)i);
		set_cddpoly_i(a, i - 1, &tmp_coef);
	}
	a->deg = diff_deg;
}

// ------------------------------------
// New implementation
// 2025-01-15 (Wed) T.Kouya
// ------------------------------------
/* complex value of a(x) */
//void ceval_cddpoly(MPFCmplx ret, CDDPoly a, MPFCmplx x)
// Based on Horner method
void eval_cddpoly_horner(cddfloat *ret, CDDPoly a, cddfloat *x)
{
	long int i;

	//set0_mpfcmplx(ret);
	rcdd_set_ui(ret, 0UL);

	//set_real_mpfcmplx(ret, get_cddpoly_i(a, a->deg));
	rcdd_set(ret, get_cddpoly_i(a, a->deg));
	for(i = a->deg - 1; i >= 0; i--)
	{
		//mul2_mpfcmplx(ret, x);
		rcdd_mul(ret, ret, x);
		//add_mpfcmplx_mpf(ret, ret, get_cddpoly_i(a, i));
		rcdd_add(ret, ret, get_cddpoly_i(a, i));
	}
}

/* value of a(x) */
// Based on Estrin method
void eval_cddpoly_estrin(cddfloat *ret, CDDPoly a, cddfloat *x)
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
		rcdd_set(ret, get_cddpoly_i(a, 0));
		return;
    }
    if(a->deg == 1) {
        //return coef[0] + coef[1] * x;
		rcdd_mul(ret, x, get_cddpoly_i(a, 1));
		rcdd_add(ret, ret, get_cddpoly_i(a, 0));
		return;
    }

    num_in_coef = a->deg + 1;
    if((num_in_coef % 2) == 1) num_in_coef++;

    in_degree = num_in_coef - 1;
    in_coef_old = (cddfloat *)calloc(num_in_coef, sizeof(cddfloat));
    in_coef_new = (cddfloat *)calloc(num_in_coef, sizeof(cddfloat));

    for(i = 0; i <= a->deg; i++) rcdd_set(&in_coef_old[i], get_cddpoly_i(a, i));
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
//void ceval_diff_cddpoly(MPFCmplx ret, CDDPoly a, MPFCmplx x)
// Based on Horner method
void eval_diff_cddpoly(cddfloat *ret, CDDPoly a, cddfloat *x)
{
	long int i;
	cddfloat tmp;

	//set0_mpfcmplx(ret);
	rcdd_set_ui(ret, 0UL);
	//set_real_mpfcmplx(ret, get_cddpoly_i(a, a->deg));
	rcdd_set(ret, get_cddpoly_i(a, a->deg));
	//mul_mpfcmplx_ui(ret, ret, (unsigned long)a->deg);
	rcdd_mul_ui(ret, ret, (unsigned long)a->deg);

	for(i = a->deg - 1; i >= 1; i--)
	{
		//mul_mpfcmplx(ret, ret, x);
		rcdd_mul(ret, ret, x);
		rcdd_mul_ui(&tmp, get_cddpoly_i(a, i), (unsigned long)i);
		//add_mpfcmplx_mpf(ret, ret, tmp);
		rcdd_add(ret, ret, &tmp);
	}
}

//------------
// AVX2
//------------
/* value of a(x) */
// Based on Horner method
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
/* complex value of a(x) */
// Based on Horner method
void _bncavx2_eval_cddpoly_horner(__m256d ret_re[DDSIZE], __m256d ret_im[DDSIZE], CDDPoly a, __m256d x_re[DDSIZE], __m256d x_im[DDSIZE])
{
	long int i;
	__m256d tmp_re[DDSIZE], tmp_im[DDSIZE], a_i_re[DDSIZE], a_i_im[DDSIZE];

	//set0_dcmplx(ret);
	_bncavx2_rcdd_set0(ret_re, ret_im); // = _mm256_setzero_pd();

	//set_real_dcmplx(ret, get_dpoly_i(a, a->deg));
	_bncavx2_rdd_set1_dd(ret_re, get_cddpoly_i(a, a->deg)->val_re);
	_bncavx2_rdd_set1_dd(ret_im, get_cddpoly_i(a, a->deg)->val_im);

	for(i = a->deg - 1; i >= 0; i--)
	{
		/* ret = ret * x + get_dpoly_i(a, i) */
		//a_i = _mm256_set1_pd(get_dpoly_i(a, i));
		_bncavx2_rdd_set1_dd(a_i_re, get_cddpoly_i(a, i)->val_re);
		_bncavx2_rdd_set1_dd(a_i_im, get_cddpoly_i(a, i)->val_im);

		//mul2_dcmplx(ret, x);
		_bncavx2_rcdd_mul(tmp_re, tmp_im, ret_re, ret_im, x_re, x_im);

		//add_dcmplx_d(ret, ret, get_dpoly_i(a, i));
		//_bncavx2_rcdd_add_dd(ret_re, ret_im, tmp_re, tmp_im, a_i);
		_bncavx2_rcdd_add(ret_re, ret_im, tmp_re, tmp_im, a_i_re, a_i_im);
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
void _bncavx2_eval_cddpoly_estrin(cddfloat *ret, CDDPoly a, cddfloat *x)
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
        eval_cddpoly_horner(ret, a, x);
        return;
    }

    // zero4 := 0
    _bncavx2_rdd_set0(zero4); //  = _mm256_setzero_pd();
    num_loop_unit = 2 * _BNC_D_WIDTH;

#elif defined(__AVX512F__) // __AVX512F__
	__m512d old_coef8_real[2][DDSIZE], old_coef8_imag[2][DDSIZE];
    __m512d new_coef8_real[DDSIZE], new_coef8_imag[DDSIZE];
    __m512d a08_real[DDSIZE], a08_imag[DDSIZE], a18_real[DDSIZE], a18_imag[DDSIZE];
    __m512d x8_real[DDSIZE], x8_imag[DDSIZE], zero8[DDSIZE];
    __m512d ctmp8_real[DDSIZE], ctmp8_imag[DDSIZE];

    if((a->deg + 1) <= (2 * _BNC_D_WIDTH))
    {
        eval_cddpoly_horner(ret, a, x);
        return;
    }

    // zero8 := 0
    _bncavx512_rdd_set0(zero8);
    num_loop_unit = 2 * (2 * _BNC_D_WIDTH);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, 2-lane, derived from NEON)
	
	svfloat64_t old_coef2_real_0_0, old_coef2_real_0_1, old_coef2_real_1_0, old_coef2_real_1_1;
	svfloat64_t old_coef2_imag_0_0, old_coef2_imag_0_1, old_coef2_imag_1_0, old_coef2_imag_1_1;
    
	svfloat64_t new_coef2_real_0, new_coef2_real_1;
	svfloat64_t new_coef2_imag_0, new_coef2_imag_1;
    
	svfloat64_t a02_real_0, a02_real_1;
	svfloat64_t a02_imag_0, a02_imag_1;
	svfloat64_t a12_real_0, a12_real_1;
	svfloat64_t a12_imag_0, a12_imag_1;
    
	svfloat64_t x2_real_0, x2_real_1;
	svfloat64_t x2_imag_0, x2_imag_1;
	svfloat64_t zero2_0, zero2_1;
    
	svfloat64_t ctmp2_real_0, ctmp2_real_1;
	svfloat64_t ctmp2_imag_0, ctmp2_imag_1;

    if((a->deg + 1) <= 2)
    {
        eval_cddpoly_horner(ret, a, x);
        return;
    }

    // zero2 := 0
    _bncsve2_rdd_set0(&zero2_0, &zero2_1);
    num_loop_unit = 2 * 2;

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM NEON
	float64x2_t old_coef2_real[2][DDSIZE], old_coef2_imag[2][DDSIZE];
    float64x2_t new_coef2_real[DDSIZE], new_coef2_imag[DDSIZE];
    float64x2_t a02_real[DDSIZE], a02_imag[DDSIZE], a12_real[DDSIZE], a12_imag[DDSIZE];
    float64x2_t x2_real[DDSIZE], x2_imag[DDSIZE], zero2[DDSIZE];
    float64x2_t ctmp2_real[DDSIZE], ctmp2_imag[DDSIZE];

    if((a->deg + 1) <= 2)
    {
        eval_cddpoly_horner(ret, a, x);
        return;
    }

    // zero2 := 0
    _bncneon_rdd_set0(zero2);
    num_loop_unit = 2 * 2;

#else // __AVX2__
    eval_cddpoly_estrin(ret, a, x);
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
    //for(i = 0; i <= a->deg; i++) set_ddvector_i(in_coef_old_real, i, get_cddpoly_i(a, i));
    for(i = 0; i <= a->deg; i++)
    {
        set_ddvector_i(in_coef_old_real, i, get_cddpoly_i(a, i)->val_re);
        set_ddvector_i(in_coef_old_imag, i, get_cddpoly_i(a, i)->val_im);
    }

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
        // Broadcast complex x to vectors
        _bncavx512_rdd_set1_dd(x8_real, in_x.val_re);
        _bncavx512_rdd_set1_dd(x8_imag, in_x.val_im);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            // Load real coefficients for DDSIZE=2 components
            old_coef8_real[0][0] = _mm512_load_pd(&(in_coef_old_real->element[0][i * num_loop_unit]));
            old_coef8_real[1][0] = _mm512_load_pd(&(in_coef_old_real->element[0][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_real[0][1] = _mm512_load_pd(&(in_coef_old_real->element[1][i * num_loop_unit]));
            old_coef8_real[1][1] = _mm512_load_pd(&(in_coef_old_real->element[1][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));

            // Load imaginary coefficients for DDSIZE=2 components
            old_coef8_imag[0][0] = _mm512_load_pd(&(in_coef_old_imag->element[0][i * num_loop_unit]));
            old_coef8_imag[1][0] = _mm512_load_pd(&(in_coef_old_imag->element[0][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_imag[0][1] = _mm512_load_pd(&(in_coef_old_imag->element[1][i * num_loop_unit]));
            old_coef8_imag[1][1] = _mm512_load_pd(&(in_coef_old_imag->element[1][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            
            // Separate even/odd coefficients - Real part
            a08_real[0] = _mm512_unpacklo_pd(old_coef8_real[0][0], old_coef8_real[1][0]);
            a18_real[0] = _mm512_unpackhi_pd(old_coef8_real[0][0], old_coef8_real[1][0]);
            a08_real[1] = _mm512_unpacklo_pd(old_coef8_real[0][1], old_coef8_real[1][1]);
            a18_real[1] = _mm512_unpackhi_pd(old_coef8_real[0][1], old_coef8_real[1][1]);

            // Separate even/odd coefficients - Imaginary part
            a08_imag[0] = _mm512_unpacklo_pd(old_coef8_imag[0][0], old_coef8_imag[1][0]);
            a18_imag[0] = _mm512_unpackhi_pd(old_coef8_imag[0][0], old_coef8_imag[1][0]);
            a08_imag[1] = _mm512_unpacklo_pd(old_coef8_imag[0][1], old_coef8_imag[1][1]);
            a18_imag[1] = _mm512_unpackhi_pd(old_coef8_imag[0][1], old_coef8_imag[1][1]);

            // Complex DD multiplication: a1 * x
            _bncavx512_rcdd_mul(ctmp8_real, ctmp8_imag, a18_real, a18_imag, x8_real, x8_imag);

            // Complex DD addition: (a1 * x) + a0
            _bncavx512_rcdd_add(new_coef8_real, new_coef8_imag, ctmp8_real, ctmp8_imag, a08_real, a08_imag);
            
            // Permute to restore order
            new_coef8_real[0] = _mm512_permutex_pd(new_coef8_real[0], 0xD8);
            new_coef8_imag[0] = _mm512_permutex_pd(new_coef8_imag[0], 0xD8);
            new_coef8_real[1] = _mm512_permutex_pd(new_coef8_real[1], 0xD8);
            new_coef8_imag[1] = _mm512_permutex_pd(new_coef8_imag[1], 0xD8);
            
            // Store zeros to clear array
            _mm512_store_pd(&(in_coef_old_real->element[0][i * num_loop_unit]), zero8[0]);
            _mm512_store_pd(&(in_coef_old_imag->element[0][i * num_loop_unit]), zero8[0]);
            _mm512_store_pd(&(in_coef_old_real->element[0][i * num_loop_unit + (2 * _BNC_D_WIDTH)]), zero8[0]);
            _mm512_store_pd(&(in_coef_old_imag->element[0][i * num_loop_unit + (2 * _BNC_D_WIDTH)]), zero8[0]);
            _mm512_store_pd(&(in_coef_old_real->element[1][i * num_loop_unit]), zero8[1]);
            _mm512_store_pd(&(in_coef_old_imag->element[1][i * num_loop_unit]), zero8[1]);
            _mm512_store_pd(&(in_coef_old_real->element[1][i * num_loop_unit + (2 * _BNC_D_WIDTH)]), zero8[1]);
            _mm512_store_pd(&(in_coef_old_imag->element[1][i * num_loop_unit + (2 * _BNC_D_WIDTH)]), zero8[1]);

            // Store results
            _mm512_store_pd(&(in_coef_old_real->element[0][i * (2 * _BNC_D_WIDTH)]), new_coef8_real[0]);
            _mm512_store_pd(&(in_coef_old_imag->element[0][i * (2 * _BNC_D_WIDTH)]), new_coef8_imag[0]);
            _mm512_store_pd(&(in_coef_old_real->element[1][i * (2 * _BNC_D_WIDTH)]), new_coef8_real[1]);
            _mm512_store_pd(&(in_coef_old_imag->element[1][i * (2 * _BNC_D_WIDTH)]), new_coef8_imag[1]);
        }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, 2-lane, derived from NEON)

	svbool_t pg = svwhilelt_b64_s64((int64_t)0, (int64_t)2);        // Broadcast complex x to vectors
        x2_real_0 = svdup_f64((in_x.val_re)[0]); x2_real_1 = svdup_f64((in_x.val_re)[1]);
        x2_imag_0 = svdup_f64((in_x.val_im)[0]); x2_imag_1 = svdup_f64((in_x.val_im)[1]);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            // Load real coefficients for DDSIZE=2 components
            old_coef2_real_0_0 = svld1_f64(pg, &(in_coef_old_real->element[0][i * num_loop_unit]));
            old_coef2_real_1_0 = svld1_f64(pg, &(in_coef_old_real->element[0][i * num_loop_unit + 2]));
            old_coef2_real_0_1 = svld1_f64(pg, &(in_coef_old_real->element[1][i * num_loop_unit]));
            old_coef2_real_1_1 = svld1_f64(pg, &(in_coef_old_real->element[1][i * num_loop_unit + 2]));

            // Load imaginary coefficients for DDSIZE=2 components
            old_coef2_imag_0_0 = svld1_f64(pg, &(in_coef_old_imag->element[0][i * num_loop_unit]));
            old_coef2_imag_1_0 = svld1_f64(pg, &(in_coef_old_imag->element[0][i * num_loop_unit + 2]));
            old_coef2_imag_0_1 = svld1_f64(pg, &(in_coef_old_imag->element[1][i * num_loop_unit]));
            old_coef2_imag_1_1 = svld1_f64(pg, &(in_coef_old_imag->element[1][i * num_loop_unit + 2]));
            
            // Separate even/odd coefficients - Real part
            a02_real_0 = svtrn1_f64(old_coef2_real_0_0, old_coef2_real_1_0);
            a12_real_0 = svtrn2_f64(old_coef2_real_0_0, old_coef2_real_1_0);
            a02_real_1 = svtrn1_f64(old_coef2_real_0_1, old_coef2_real_1_1);
            a12_real_1 = svtrn2_f64(old_coef2_real_0_1, old_coef2_real_1_1);

            // Separate even/odd coefficients - Imaginary part
            a02_imag_0 = svtrn1_f64(old_coef2_imag_0_0, old_coef2_imag_1_0);
            a12_imag_0 = svtrn2_f64(old_coef2_imag_0_0, old_coef2_imag_1_0);
            a02_imag_1 = svtrn1_f64(old_coef2_imag_0_1, old_coef2_imag_1_1);
            a12_imag_1 = svtrn2_f64(old_coef2_imag_0_1, old_coef2_imag_1_1);

            // Complex DD multiplication: a1 * x
            _bncsve2_rcdd_mul(pg, &ctmp2_real_0, &ctmp2_real_1, &ctmp2_imag_0, &ctmp2_imag_1, a12_real_0, a12_real_1, a12_imag_0, a12_imag_1, x2_real_0, x2_real_1, x2_imag_0, x2_imag_1);

            // Complex DD addition: (a1 * x) + a0
            _bncsve2_rcdd_add(pg, &new_coef2_real_0, &new_coef2_real_1, &new_coef2_imag_0, &new_coef2_imag_1, ctmp2_real_0, ctmp2_real_1, ctmp2_imag_0, ctmp2_imag_1, a02_real_0, a02_real_1, a02_imag_0, a02_imag_1);
                        
            // Store zeros to clear array
            svst1_f64(pg, &(in_coef_old_real->element[0][i * num_loop_unit]), zero2_0);
            svst1_f64(pg, &(in_coef_old_imag->element[0][i * num_loop_unit]), zero2_0);
            svst1_f64(pg, &(in_coef_old_real->element[0][i * num_loop_unit + 2]), zero2_0);
            svst1_f64(pg, &(in_coef_old_imag->element[0][i * num_loop_unit + 2]), zero2_0);
            svst1_f64(pg, &(in_coef_old_real->element[1][i * num_loop_unit]), zero2_1);
            svst1_f64(pg, &(in_coef_old_imag->element[1][i * num_loop_unit]), zero2_1);
            svst1_f64(pg, &(in_coef_old_real->element[1][i * num_loop_unit + 2]), zero2_1);
            svst1_f64(pg, &(in_coef_old_imag->element[1][i * num_loop_unit + 2]), zero2_1);

            // Store results
            svst1_f64(pg, &(in_coef_old_real->element[0][i * 2]), new_coef2_real_0);
            svst1_f64(pg, &(in_coef_old_imag->element[0][i * 2]), new_coef2_imag_0);
            svst1_f64(pg, &(in_coef_old_real->element[1][i * 2]), new_coef2_real_1);
            svst1_f64(pg, &(in_coef_old_imag->element[1][i * 2]), new_coef2_imag_1);
        }
#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM NEON
        // Broadcast complex x to vectors
        _bncneon_rdd_set1_dd(x2_real, in_x.val_re);
        _bncneon_rdd_set1_dd(x2_imag, in_x.val_im);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            // Load real coefficients for DDSIZE=2 components
            old_coef2_real[0][0] = vld1q_f64(&(in_coef_old_real->element[0][i * num_loop_unit]));
            old_coef2_real[1][0] = vld1q_f64(&(in_coef_old_real->element[0][i * num_loop_unit + 2]));
            old_coef2_real[0][1] = vld1q_f64(&(in_coef_old_real->element[1][i * num_loop_unit]));
            old_coef2_real[1][1] = vld1q_f64(&(in_coef_old_real->element[1][i * num_loop_unit + 2]));

            // Load imaginary coefficients for DDSIZE=2 components
            old_coef2_imag[0][0] = vld1q_f64(&(in_coef_old_imag->element[0][i * num_loop_unit]));
            old_coef2_imag[1][0] = vld1q_f64(&(in_coef_old_imag->element[0][i * num_loop_unit + 2]));
            old_coef2_imag[0][1] = vld1q_f64(&(in_coef_old_imag->element[1][i * num_loop_unit]));
            old_coef2_imag[1][1] = vld1q_f64(&(in_coef_old_imag->element[1][i * num_loop_unit + 2]));
            
            // Separate even/odd coefficients - Real part
            a02_real[0] = vzip1q_f64(old_coef2_real[0][0], old_coef2_real[1][0]);
            a12_real[0] = vzip2q_f64(old_coef2_real[0][0], old_coef2_real[1][0]);
            a02_real[1] = vzip1q_f64(old_coef2_real[0][1], old_coef2_real[1][1]);
            a12_real[1] = vzip2q_f64(old_coef2_real[0][1], old_coef2_real[1][1]);

            // Separate even/odd coefficients - Imaginary part
            a02_imag[0] = vzip1q_f64(old_coef2_imag[0][0], old_coef2_imag[1][0]);
            a12_imag[0] = vzip2q_f64(old_coef2_imag[0][0], old_coef2_imag[1][0]);
            a02_imag[1] = vzip1q_f64(old_coef2_imag[0][1], old_coef2_imag[1][1]);
            a12_imag[1] = vzip2q_f64(old_coef2_imag[0][1], old_coef2_imag[1][1]);

            // Complex DD multiplication: a1 * x
            _bncneon_rcdd_mul(ctmp2_real, ctmp2_imag, a12_real, a12_imag, x2_real, x2_imag);

            // Complex DD addition: (a1 * x) + a0
            _bncneon_rcdd_add(new_coef2_real, new_coef2_imag, ctmp2_real, ctmp2_imag, a02_real, a02_imag);
                        
            // Store zeros to clear array
            vst1q_f64(&(in_coef_old_real->element[0][i * num_loop_unit]), zero2[0]);
            vst1q_f64(&(in_coef_old_imag->element[0][i * num_loop_unit]), zero2[0]);
            vst1q_f64(&(in_coef_old_real->element[0][i * num_loop_unit + 2]), zero2[0]);
            vst1q_f64(&(in_coef_old_imag->element[0][i * num_loop_unit + 2]), zero2[0]);
            vst1q_f64(&(in_coef_old_real->element[1][i * num_loop_unit]), zero2[1]);
            vst1q_f64(&(in_coef_old_imag->element[1][i * num_loop_unit]), zero2[1]);
            vst1q_f64(&(in_coef_old_real->element[1][i * num_loop_unit + 2]), zero2[1]);
            vst1q_f64(&(in_coef_old_imag->element[1][i * num_loop_unit + 2]), zero2[1]);

            // Store results
            vst1q_f64(&(in_coef_old_real->element[0][i * 2]), new_coef2_real[0]);
            vst1q_f64(&(in_coef_old_imag->element[0][i * 2]), new_coef2_imag[0]);
            vst1q_f64(&(in_coef_old_real->element[1][i * 2]), new_coef2_real[1]);
            vst1q_f64(&(in_coef_old_imag->element[1][i * 2]), new_coef2_imag[1]);
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
#if defined(__AVX2__) || defined(__AVX512F__) // __AVX2__
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
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    rdd_set(in_ret.val_re, get_ddvector_i(in_coef_old_real, 1));
    rdd_set(in_ret.val_im, get_ddvector_i(in_coef_old_imag, 1));
    for(i = 0; i >= 0; i--)
    {
        rcdd_mul(&ctmp, &in_ret, &in_x);
        rdd_set(in_ret.val_re, get_ddvector_i(in_coef_old_real, i));
        rdd_set(in_ret.val_im, get_ddvector_i(in_coef_old_imag, i));
        rcdd_add(&in_ret, &in_ret, &ctmp);
    }
#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM NEON
    rdd_set(in_ret.val_re, get_ddvector_i(in_coef_old_real, 1));
    rdd_set(in_ret.val_im, get_ddvector_i(in_coef_old_imag, 1));
    for(i = 0; i >= 0; i--)
    {
        rcdd_mul(&ctmp, &in_ret, &in_x);
        rdd_set(in_ret.val_re, get_ddvector_i(in_coef_old_real, i));
        rdd_set(in_ret.val_im, get_ddvector_i(in_coef_old_imag, i));
        rcdd_add(&in_ret, &in_ret, &ctmp);
    }
#endif // __AVX2__

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
