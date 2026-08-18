/********************************************************************************/
/* cqd_poly.c: Algebraic Equations and Complex Polynomials                      */
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
/*   CQDPoly init_cqdpoly(long int max_length)   */
/*   CQDPoly init2_cqdpoly(long int max_length, unsigned long prec)*/
/* Free:                                         */
/*   void free_cqdpoly(CQDPoly pol)              */
/* Get & Set Values:                             */
/*   cqdfloat *get_cqdpoly_i(CQDPoly pol, long int index) */
/*   long int setdegree_cqdpoly(CQDPoly)         */
/*   void set_cqdpoly_i(CQDPoly pol, long int index, cqdfloat val) */
/* Output:                                       */
/*   void print_cqdpoly(CQDPoly pol)             */
/*************************************************/
CQDPoly init_cqdpoly(long int max_length)
{
	CQDPoly ret = NULL;
	long int i;

	if(max_length <= 0)
	{
		fprintf(stderr, "ERROR: init_cqdpoly\n");
		return ret;
	}

	ret = (CQDPoly)malloc(sizeof(cqdpoly));
	if(ret == NULL)
		return ret;

	ret->coef = (cqdfloat *)calloc(sizeof(cqdfloat), max_length);
	if(ret->coef == NULL)
    {
        free(ret);
		return NULL;
    }

	/* All 0 */
	for(i = 0; i < max_length; i++)
		rcqd_set_ui(&(ret->coef[i]), 0UL);

    // zero := 0
    rcqd_set0(&(ret->zero));

	ret->deg = 0;
	ret->max_len = max_length;

	return ret;
}

void free_cqdpoly(CQDPoly pol)
{
	long int i;

	if(pol == NULL)
		return;

	if(pol->coef != NULL)
		free(pol->coef); // Fix! 2012-06-03 by T.Kouya

	free(pol);
}

cqdfloat *get_cqdpoly_i(CQDPoly pol, long int index)
{
	if(index > pol->deg)
		return &(pol->zero);
	else
		return &(pol->coef[index]);
}

/* find maximum degree at pol->coef[deg] != 0 */
long int setdegree_cqdpoly(CQDPoly pol)
{
	long int i;
    qdfloat tmp;

	for(i = pol->max_len - 1; i > 0; i--)
	{
        rcqd_abs(&tmp, get_cqdpoly_i(pol, i));
		if(rqd_cmp_ui(tmp.val, 0UL) != 0)
		{
			pol->deg = i;
			return i;
		}
	}

	pol->deg = 0;
	return 0;

}

void set_cqdpoly_i(CQDPoly pol, long int index, cqdfloat *val)
{
    qdfloat tmp;

    if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	rcqd_set(&(pol->coef[index]), val);
    rcqd_abs(&tmp, val);
	if((pol->deg < index) && (rqd_cmp_ui(tmp.val, 0UL) != 0))
		pol->deg = index;
}

#ifdef USE_GMP
void set_cqdpoly_i_mpc(CQDPoly pol, long int index, mpc_t val)
{
	cqdfloat ctmp;
    qdfloat tmp;

	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	mpf_get_qd(ctmp.val_re, mpc_realref(val));
    mpf_get_qd(ctmp.val_im, mpc_imagref(val));
    set_cqdpoly_i(pol, index, &ctmp);
	//rcqd_set(pol->coef[index].val, tmp);
    rcqd_abs(&tmp, &ctmp);

    if((pol->deg < index) && (rqd_cmp_ui(tmp.val, 0UL) != 0))
		pol->deg = index;

}

// Initialize and substitute polynomial from org_pol
CQDPoly init_set_cqdpoly_cmpfpoly(CMPFPoly org_pol)
{
	CQDPoly ret = NULL;
	long int i;

	ret = init_cqdpoly(org_pol->max_len);
	if(ret == NULL)
		return ret;

	for(i = 0; i <= org_pol->deg; i++)
		set_cqdpoly_i_mpc(ret, i, get_cmpfpoly_i(org_pol, i));

	return ret;
}

#endif // USE_GMP

// Initialize and substitute polynomial from org_pol
CQDPoly init_set_cqdpoly(CQDPoly org_pol)
{
	CQDPoly ret = NULL;
	long int i;

	ret = init_cqdpoly(org_pol->max_len);
	if(ret == NULL)
		return ret;

	subst_cqdpoly(ret, org_pol);

	return ret;
}

// substitute the values to pol
void subst_cqdpoly(CQDPoly pol, CQDPoly org_pol)
{
	long int i;

	if(pol == NULL || org_pol == NULL)
		return;

	if(pol->deg > org_pol->deg)
	{
		for(i = pol->deg; i >= org_pol->deg; i--)
			set_cqdpoly_i(pol, i, &(pol->zero));
	}

	pol->deg = org_pol->deg;

    for(i = 0; i <= org_pol->deg; i++)
		set_cqdpoly_i(pol, i, get_cqdpoly_i(org_pol, i));

	setdegree_cqdpoly(pol);
}

// set all coefficients to zero
void set0_cqdpoly(CQDPoly pol)
{
	long int i;

	for(i = 0; i < pol->max_len; i++)
		rcqd_set_ui(&(pol->coef[i]), 0UL);

	pol->deg = 0;
}

// printf
void print_cqdpoly(CQDPoly pol)
{
	long int i;

	if(pol == NULL)
	{
		printf("NULL\n");
		return;
	}

	printf("deg = %ld\n", pol->deg);
	for(i = 0; i <= pol->deg; i++)
	{
		printf("coef[%5ld] = ", i);
		rcqd_out_str(get_cqdpoly_i(pol, i));
		printf("\n");
	}
}

// fprintf
/*
void fprint_cqdpoly(FILE *fp, CQDPoly pol)
{
	long int i;

	if(pol == NULL)
	{
		fprintf(fp, "NULL\n");
		return;
	}

	fprintf(fp, "deg = %ld\n", pol->deg);
	for(i = 0; i <= pol->deg; i++)
	{
		fprintf(fp, "coef[%5ld] = ", i);
		rcqd_fprint(fp, get_cqdpoly_i(pol, i));
		fprintf(fp, "\n");
	}
}
*/

// Evaluation
void eval_cqdpoly(cqdfloat *ret, CQDPoly pol, cqdfloat *x)
{
	long int i;
	cqdfloat ctmp;

	// Horner method
	rcqd_set(ret, get_cqdpoly_i(pol, pol->deg));
	for(i = pol->deg - 1; i >= 0; i--)
	{
		rcqd_mul(&ctmp, ret, x);
		rcqd_add(ret, &ctmp, get_cqdpoly_i(pol, i));
	}
}

// Addition: ret = pol1 + pol2
void add_cqdpoly(CQDPoly ret, CQDPoly pol1, CQDPoly pol2)
{
	long int i, max_deg;
	cqdfloat ctmp;

	max_deg = (pol1->deg > pol2->deg) ? pol1->deg : pol2->deg;

	for(i = 0; i <= max_deg; i++)
	{
		rcqd_add(&ctmp, get_cqdpoly_i(pol1, i), get_cqdpoly_i(pol2, i));
		set_cqdpoly_i(ret, i, &ctmp);
	}

	setdegree_cqdpoly(ret);
}

// Subtraction: ret = pol1 - pol2
void sub_cqdpoly(CQDPoly ret, CQDPoly pol1, CQDPoly pol2)
{
	long int i, max_deg;
	cqdfloat ctmp;

	max_deg = (pol1->deg > pol2->deg) ? pol1->deg : pol2->deg;

	for(i = 0; i <= max_deg; i++)
	{
		rcqd_sub(&ctmp, get_cqdpoly_i(pol1, i), get_cqdpoly_i(pol2, i));
		set_cqdpoly_i(ret, i, &ctmp);
	}

	setdegree_cqdpoly(ret);
}

// Scalar multiplication: ret = alpha * pol
void scal_cqdpoly(CQDPoly ret, cqdfloat *alpha, CQDPoly pol)
{
	long int i;
	cqdfloat ctmp;

	for(i = 0; i <= pol->deg; i++)
	{
		rcqd_mul(&ctmp, alpha, get_cqdpoly_i(pol, i));
		set_cqdpoly_i(ret, i, &ctmp);
	}

	setdegree_cqdpoly(ret);
}

// Multiplication: ret = pol1 * pol2
void mul_cqdpoly(CQDPoly ret, CQDPoly pol1, CQDPoly pol2)
{
	long int i, j;
	cqdfloat ctmp, ctmp2;
	CQDPoly work;

	work = init_cqdpoly(pol1->deg + pol2->deg + 1);
	set0_cqdpoly(work);

	for(i = 0; i <= pol1->deg; i++)
	{
		for(j = 0; j <= pol2->deg; j++)
		{
			rcqd_mul(&ctmp, get_cqdpoly_i(pol1, i), get_cqdpoly_i(pol2, j));
			rcqd_add(&ctmp2, get_cqdpoly_i(work, i + j), &ctmp);
			set_cqdpoly_i(work, i + j, &ctmp2);
		}
	}

	subst_cqdpoly(ret, work);
	free_cqdpoly(work);
}

// Division: quot = pol1 / pol2, rem = pol1 % pol2
void div_cqdpoly(CQDPoly quot, CQDPoly rem, CQDPoly pol1, CQDPoly pol2)
{
	long int i;
	cqdfloat ctmp, ctmp2;
	CQDPoly work_quot, work_rem, work_pol;

	if(pol1->deg < pol2->deg)
	{
		set0_cqdpoly(quot);
		subst_cqdpoly(rem, pol1);
		return;
	}

	work_quot = init_cqdpoly(pol1->deg - pol2->deg + 1);
	work_rem = init_set_cqdpoly(pol1);
	work_pol = init_cqdpoly(pol1->deg + 1);

	set0_cqdpoly(work_quot);

	for(i = pol1->deg - pol2->deg; i >= 0; i--)
	{
		rcqd_div(&ctmp, get_cqdpoly_i(work_rem, i + pol2->deg), get_cqdpoly_i(pol2, pol2->deg));
		set_cqdpoly_i(work_quot, i, &ctmp);

		// work_pol = ctmp * pol2 * x^i
		set0_cqdpoly(work_pol);
		scal_cqdpoly(work_pol, &ctmp, pol2);
		
		// Shift work_pol by i positions
		long int j;
		for(j = work_pol->deg; j >= 0; j--)
		{
			if(j + i <= work_pol->max_len - 1)
			{
				rcqd_set(&ctmp2, get_cqdpoly_i(work_pol, j));
				set_cqdpoly_i(work_pol, j + i, &ctmp2);
			}
		}
		for(j = 0; j < i; j++)
		{
			rcqd_set_ui(&ctmp2, 0UL);
			set_cqdpoly_i(work_pol, j, &ctmp2);
		}
		setdegree_cqdpoly(work_pol);

		sub_cqdpoly(work_rem, work_rem, work_pol);
	}

	subst_cqdpoly(quot, work_quot);
	subst_cqdpoly(rem, work_rem);

	free_cqdpoly(work_quot);
	free_cqdpoly(work_rem);
	free_cqdpoly(work_pol);
}

// Derivative: ret = d(pol)/dx
void diff_cqdpoly(CQDPoly ret, CQDPoly pol)
{
	long int i;
	cqdfloat ctmp;
	qdfloat rtmp;

	if(pol->deg == 0)
	{
		set0_cqdpoly(ret);
		return;
	}

	for(i = 0; i < pol->deg; i++)
	{
		rqd_set_ui(rtmp.val, (unsigned long)(i + 1));
		rcqd_mul_qd(&ctmp, get_cqdpoly_i(pol, i + 1), rtmp.val);
		set_cqdpoly_i(ret, i, &ctmp);
	}

	setdegree_cqdpoly(ret);
}

// ------------------------------------
// Complex quad-double precision polynomial evaluation
// ------------------------------------
/* complex value of a(x) */
// Based on Horner method
void eval_cqdpoly_horner(cqdfloat *ret, CQDPoly a, cqdfloat *x)
{
	long int i;
	cqdfloat tmp;

	// ret = 0
	rcqd_set_ui(ret, 0UL);

	// ret = a[deg]
	rcqd_set(ret, get_cqdpoly_i(a, a->deg));

	for(i = a->deg - 1; i >= 0; i--)
	{
		// ret = ret * x
		rcqd_mul(&tmp, ret, x);
		
		// ret = tmp + a[i]
		rcqd_add(ret, &tmp, get_cqdpoly_i(a, i));
	}
}

/* value of a(x) */
// Based on Estrin method
void eval_cqdpoly_estrin(cqdfloat *ret, CQDPoly a, cqdfloat *x)
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
    cqdfloat in_x, tmp;
	cqdfloat *in_coef_old, *in_coef_new;
    long int in_degree, num_in_coef, i;

    if(a->deg == 0) {
        rcqd_set(ret, get_cqdpoly_i(a, 0));
		return;
    }
    if(a->deg == 1) {
        // ret = a[1] * x + a[0]
		rcqd_mul(&tmp, get_cqdpoly_i(a, 1), x);
		rcqd_add(ret, &tmp, get_cqdpoly_i(a, 0));
		return;
    }

    num_in_coef = a->deg + 1;
    if((num_in_coef % 2) == 1) num_in_coef++;

    in_degree = num_in_coef - 1;
    in_coef_old = (cqdfloat *)calloc(num_in_coef, sizeof(cqdfloat));
    in_coef_new = (cqdfloat *)calloc(num_in_coef, sizeof(cqdfloat));

    for(i = 0; i <= a->deg; i++) 
        rcqd_set(&in_coef_old[i], get_cqdpoly_i(a, i));
    for(i = a->deg + 1; i <= in_degree; i++) 
        rcqd_set_ui(&in_coef_old[i], 0UL);

    rcqd_set(&in_x, x);

    while(in_degree > 1)
    {
		num_in_coef /= 2;
        for(i = 0; i < num_in_coef; i++)
        {
            // tmp = in_coef_old[i * 2 + 1] * in_x
			rcqd_mul(&tmp, &in_coef_old[i * 2 + 1], &in_x);
            
            // in_coef_new[i] = tmp + in_coef_old[i * 2]
			rcqd_add(&in_coef_new[i], &tmp, &in_coef_old[i * 2]);
        }

        // in_x = in_x * in_x
		rcqd_mul(&in_x, &in_x, &in_x);

        for(i = 0; i < num_in_coef; i++)
			rcqd_set(&in_coef_old[i], &in_coef_new[i]);

        if((num_in_coef % 2) == 1)
		{
 			num_in_coef += 1;
			rcqd_set_ui(&in_coef_old[num_in_coef - 1], 0UL);
		}

        in_degree = num_in_coef - 1;
    }
    
    // ret = in_coef_new[1] * in_x + in_coef_new[0]
	rcqd_mul(&tmp, &in_coef_new[1], &in_x);
	rcqd_add(ret, &tmp, &in_coef_new[0]);

    free(in_coef_old);
    free(in_coef_new);

	return;
}

/* complex value of a'(x) (derivative) */
// Based on Horner method
void eval_diff_cqdpoly(cqdfloat *ret, CQDPoly a, cqdfloat *x)
{
	long int i;
	cqdfloat tmp;

	// ret = 0
	rcqd_set_ui(ret, 0UL);

	// ret = a[deg] * deg
	rcqd_set(ret, get_cqdpoly_i(a, a->deg));
	rcqd_mul_ui(ret, ret, (unsigned long)a->deg);

	for(i = a->deg - 1; i >= 1; i--)
	{
		// ret = ret * x
		rcqd_mul(ret, ret, x);
		
		// tmp = a[i] * i
		rcqd_mul_ui(&tmp, get_cqdpoly_i(a, i), (unsigned long)i);
		
		// ret = ret + tmp
		rcqd_add(ret, ret, &tmp);
	}
}
//------------
// AVX2
//------------
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
void _bncavx2_eval_cqdpoly_horner(__m256d ret_re[QDSIZE], __m256d ret_im[QDSIZE], CQDPoly a, __m256d x_re[QDSIZE], __m256d x_im[QDSIZE])
{
	long int i;
	__m256d tmp_re[QDSIZE], tmp_im[QDSIZE], a_i_re[QDSIZE], a_i_im[QDSIZE];

	_bncavx2_rcqd_set0(ret_re, ret_im);

	_bncavx2_rqd_set1_qd(ret_re, get_cqdpoly_i(a, a->deg)->val_re);
	_bncavx2_rqd_set1_qd(ret_im, get_cqdpoly_i(a, a->deg)->val_im);

	for(i = a->deg - 1; i >= 0; i--)
	{
		_bncavx2_rqd_set1_qd(a_i_re, get_cqdpoly_i(a, i)->val_re);
		_bncavx2_rqd_set1_qd(a_i_im, get_cqdpoly_i(a, i)->val_im);

		_bncavx2_rcqd_mul(tmp_re, tmp_im, ret_re, ret_im, x_re, x_im);

		_bncavx2_rcqd_add(ret_re, ret_im, tmp_re, tmp_im, a_i_re, a_i_im);
	}
}
#elif defined(__AVX512F__) // __AVX512F__
void _bncavx512_eval_cqdpoly_horner(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], CQDPoly a, __m512d x_re[QDSIZE], __m512d x_im[QDSIZE])
{
	long int i;
	__m512d tmp_re[QDSIZE], tmp_im[QDSIZE], a_i_re[QDSIZE], a_i_im[QDSIZE];

	_bncavx512_rcqd_set0(ret_re, ret_im);

	_bncavx512_rqd_set1_qd(ret_re, get_cqdpoly_i(a, a->deg)->val_re);
	_bncavx512_rqd_set1_qd(ret_im, get_cqdpoly_i(a, a->deg)->val_im);

	for(i = a->deg - 1; i >= 0; i--)
	{
		_bncavx512_rqd_set1_qd(a_i_re, get_cqdpoly_i(a, i)->val_re);
		_bncavx512_rqd_set1_qd(a_i_im, get_cqdpoly_i(a, i)->val_im);

		_bncavx512_rcqd_mul(tmp_re, tmp_im, ret_re, ret_im, x_re, x_im);

		_bncavx512_rcqd_add(ret_re, ret_im, tmp_re, tmp_im, a_i_re, a_i_im);
	}
}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // ARM NEON
void _bncneon_eval_cqdpoly_horner(float64x2_t ret_re[QDSIZE], float64x2_t ret_im[QDSIZE], CQDPoly a, float64x2_t x_re[QDSIZE], float64x2_t x_im[QDSIZE])
{
	long int i;
	float64x2_t tmp_re[QDSIZE], tmp_im[QDSIZE], a_i_re[QDSIZE], a_i_im[QDSIZE];

	_bncneon_rcqd_set0(ret_re, ret_im);

	_bncneon_rqd_set1_qd(ret_re, get_cqdpoly_i(a, a->deg)->val_re);
	_bncneon_rqd_set1_qd(ret_im, get_cqdpoly_i(a, a->deg)->val_im);

	for(i = a->deg - 1; i >= 0; i--)
	{
		_bncneon_rqd_set1_qd(a_i_re, get_cqdpoly_i(a, i)->val_re);
		_bncneon_rqd_set1_qd(a_i_im, get_cqdpoly_i(a, i)->val_im);

		_bncneon_rcqd_mul(tmp_re, tmp_im, ret_re, ret_im, x_re, x_im);

		_bncneon_rcqd_add(ret_re, ret_im, tmp_re, tmp_im, a_i_re, a_i_im);
	}
}
#endif // __AVX2__

void _bncavx2_eval_cqdpoly_estrin(cqdfloat *ret, CQDPoly a, cqdfloat *x)
{
    QDVector in_coef_old_real, in_coef_old_imag;
    cqdfloat in_x;
    cqdfloat in_ret, ctmp;
    long int in_degree, num_in_coef, i, num_loop_unit;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d old_coef4_real[2][QDSIZE], old_coef4_imag[2][QDSIZE];
    __m256d new_coef4_real[QDSIZE], new_coef4_imag[QDSIZE];
    __m256d a04_real[QDSIZE], a04_imag[QDSIZE], a14_real[QDSIZE], a14_imag[QDSIZE], x4_real[QDSIZE], x4_imag[QDSIZE], zero4[QDSIZE];
    __m256d ctmp4_real[QDSIZE], ctmp4_imag[QDSIZE];

    if((a->deg + 1) <= _BNC_D_WIDTH)
    {
        eval_cqdpoly_horner(ret, a, x);
        return;
    }

    _bncavx2_rqd_set0(zero4);
    num_loop_unit = 2 * _BNC_D_WIDTH;

#elif defined(__AVX512F__) // __AVX512F__
	__m512d old_coef8_real[2][QDSIZE], old_coef8_imag[2][QDSIZE];
    __m512d new_coef8_real[QDSIZE], new_coef8_imag[QDSIZE];
    __m512d a08_real[QDSIZE], a08_imag[QDSIZE], a18_real[QDSIZE], a18_imag[QDSIZE];
    __m512d x8_real[QDSIZE], x8_imag[QDSIZE], zero8[QDSIZE];
    __m512d ctmp8_real[QDSIZE], ctmp8_imag[QDSIZE];

    if((a->deg + 1) <= (2 * _BNC_D_WIDTH))
    {
        eval_cqdpoly_horner(ret, a, x);
        return;
    }

    _bncavx512_rqd_set0(zero8);
    num_loop_unit = 2 * (2 * _BNC_D_WIDTH);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, 2-lane, derived from NEON)
	
	svfloat64_t old_coef2_real_0_0, old_coef2_real_0_1, old_coef2_real_0_2, old_coef2_real_0_3, old_coef2_real_1_0, old_coef2_real_1_1, old_coef2_real_1_2, old_coef2_real_1_3;
	svfloat64_t old_coef2_imag_0_0, old_coef2_imag_0_1, old_coef2_imag_0_2, old_coef2_imag_0_3, old_coef2_imag_1_0, old_coef2_imag_1_1, old_coef2_imag_1_2, old_coef2_imag_1_3;
    
	svfloat64_t new_coef2_real_0, new_coef2_real_1, new_coef2_real_2, new_coef2_real_3;
	svfloat64_t new_coef2_imag_0, new_coef2_imag_1, new_coef2_imag_2, new_coef2_imag_3;
    
	svfloat64_t a02_real_0, a02_real_1, a02_real_2, a02_real_3;
	svfloat64_t a02_imag_0, a02_imag_1, a02_imag_2, a02_imag_3;
	svfloat64_t a12_real_0, a12_real_1, a12_real_2, a12_real_3;
	svfloat64_t a12_imag_0, a12_imag_1, a12_imag_2, a12_imag_3;
    
	svfloat64_t x2_real_0, x2_real_1, x2_real_2, x2_real_3;
	svfloat64_t x2_imag_0, x2_imag_1, x2_imag_2, x2_imag_3;
	svfloat64_t zero2_0, zero2_1, zero2_2, zero2_3;
    
	svfloat64_t ctmp2_real_0, ctmp2_real_1, ctmp2_real_2, ctmp2_real_3;
	svfloat64_t ctmp2_imag_0, ctmp2_imag_1, ctmp2_imag_2, ctmp2_imag_3;

    if((a->deg + 1) <= 2)
    {
        eval_cqdpoly_horner(ret, a, x);
        return;
    }

    _bncsve2_rqd_set0(&zero2_0, &zero2_1, &zero2_2, &zero2_3);
    num_loop_unit = 2 * 2;

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // ARM NEON
	float64x2_t old_coef2_real[2][QDSIZE], old_coef2_imag[2][QDSIZE];
    float64x2_t new_coef2_real[QDSIZE], new_coef2_imag[QDSIZE];
    float64x2_t a02_real[QDSIZE], a02_imag[QDSIZE], a12_real[QDSIZE], a12_imag[QDSIZE];
    float64x2_t x2_real[QDSIZE], x2_imag[QDSIZE], zero2[QDSIZE];
    float64x2_t ctmp2_real[QDSIZE], ctmp2_imag[QDSIZE];

    if((a->deg + 1) <= 2)
    {
        eval_cqdpoly_horner(ret, a, x);
        return;
    }

    _bncneon_rqd_set0(zero2);
    num_loop_unit = 2 * 2;

#else // __AVX2__
    eval_cqdpoly_estrin(ret, a, x);
    return;
#endif // __AVX2__

    num_in_coef = a->deg + 1;
    if((num_in_coef % num_loop_unit) != 0) 
        num_in_coef = (long int)ceil(((double)num_in_coef / (double)num_loop_unit)) * num_loop_unit;

    in_degree = num_in_coef - 1;
    in_coef_old_real = init_qdvector(num_in_coef);
    in_coef_old_imag = init_qdvector(num_in_coef);

    for(i = 0; i <= a->deg; i++)
    {
        set_qdvector_i(in_coef_old_real, i, get_cqdpoly_i(a, i)->val_re);
        set_qdvector_i(in_coef_old_imag, i, get_cqdpoly_i(a, i)->val_im);
    }

    rcqd_set(&in_x, x);

    while(1)
    {
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
        _bncavx2_rqd_set1_qd(x4_real, in_x.val_re);
        _bncavx2_rqd_set1_qd(x4_imag, in_x.val_im);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            old_coef4_real[0][0] = _mm256_load_pd(&(in_coef_old_real->element[0][i * num_loop_unit]));
            old_coef4_real[1][0] = _mm256_load_pd(&(in_coef_old_real->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef4_real[0][1] = _mm256_load_pd(&(in_coef_old_real->element[1][i * num_loop_unit]));
            old_coef4_real[1][1] = _mm256_load_pd(&(in_coef_old_real->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef4_real[0][2] = _mm256_load_pd(&(in_coef_old_real->element[2][i * num_loop_unit]));
            old_coef4_real[1][2] = _mm256_load_pd(&(in_coef_old_real->element[2][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef4_real[0][3] = _mm256_load_pd(&(in_coef_old_real->element[3][i * num_loop_unit]));
            old_coef4_real[1][3] = _mm256_load_pd(&(in_coef_old_real->element[3][i * num_loop_unit + _BNC_D_WIDTH]));

            old_coef4_imag[0][0] = _mm256_load_pd(&(in_coef_old_imag->element[0][i * num_loop_unit]));
            old_coef4_imag[1][0] = _mm256_load_pd(&(in_coef_old_imag->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef4_imag[0][1] = _mm256_load_pd(&(in_coef_old_imag->element[1][i * num_loop_unit]));
            old_coef4_imag[1][1] = _mm256_load_pd(&(in_coef_old_imag->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef4_imag[0][2] = _mm256_load_pd(&(in_coef_old_imag->element[2][i * num_loop_unit]));
            old_coef4_imag[1][2] = _mm256_load_pd(&(in_coef_old_imag->element[2][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef4_imag[0][3] = _mm256_load_pd(&(in_coef_old_imag->element[3][i * num_loop_unit]));
            old_coef4_imag[1][3] = _mm256_load_pd(&(in_coef_old_imag->element[3][i * num_loop_unit + _BNC_D_WIDTH]));

            a04_real[0] = _mm256_unpacklo_pd(old_coef4_real[0][0], old_coef4_real[1][0]);
            a14_real[0] = _mm256_unpackhi_pd(old_coef4_real[0][0], old_coef4_real[1][0]);
            a04_real[1] = _mm256_unpacklo_pd(old_coef4_real[0][1], old_coef4_real[1][1]);
            a14_real[1] = _mm256_unpackhi_pd(old_coef4_real[0][1], old_coef4_real[1][1]);
            a04_real[2] = _mm256_unpacklo_pd(old_coef4_real[0][2], old_coef4_real[1][2]);
            a14_real[2] = _mm256_unpackhi_pd(old_coef4_real[0][2], old_coef4_real[1][2]);
            a04_real[3] = _mm256_unpacklo_pd(old_coef4_real[0][3], old_coef4_real[1][3]);
            a14_real[3] = _mm256_unpackhi_pd(old_coef4_real[0][3], old_coef4_real[1][3]);

            a04_imag[0] = _mm256_unpacklo_pd(old_coef4_imag[0][0], old_coef4_imag[1][0]);
            a14_imag[0] = _mm256_unpackhi_pd(old_coef4_imag[0][0], old_coef4_imag[1][0]);
            a04_imag[1] = _mm256_unpacklo_pd(old_coef4_imag[0][1], old_coef4_imag[1][1]);
            a14_imag[1] = _mm256_unpackhi_pd(old_coef4_imag[0][1], old_coef4_imag[1][1]);
            a04_imag[2] = _mm256_unpacklo_pd(old_coef4_imag[0][2], old_coef4_imag[1][2]);
            a14_imag[2] = _mm256_unpackhi_pd(old_coef4_imag[0][2], old_coef4_imag[1][2]);
            a04_imag[3] = _mm256_unpacklo_pd(old_coef4_imag[0][3], old_coef4_imag[1][3]);
            a14_imag[3] = _mm256_unpackhi_pd(old_coef4_imag[0][3], old_coef4_imag[1][3]);

            _bncavx2_rcqd_mul(ctmp4_real, ctmp4_imag, a14_real, a14_imag, x4_real, x4_imag);
            _bncavx2_rcqd_add(new_coef4_real, new_coef4_imag, ctmp4_real, ctmp4_imag, a04_real, a04_imag);

            new_coef4_real[0] = _mm256_permute4x64_pd(new_coef4_real[0], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_imag[0] = _mm256_permute4x64_pd(new_coef4_imag[0], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_real[1] = _mm256_permute4x64_pd(new_coef4_real[1], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_imag[1] = _mm256_permute4x64_pd(new_coef4_imag[1], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_real[2] = _mm256_permute4x64_pd(new_coef4_real[2], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_imag[2] = _mm256_permute4x64_pd(new_coef4_imag[2], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_real[3] = _mm256_permute4x64_pd(new_coef4_real[3], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_imag[3] = _mm256_permute4x64_pd(new_coef4_imag[3], (int)(3*64 + 1*16 + 2*4 + 0));

            _mm256_store_pd(&(in_coef_old_real->element[0][i * num_loop_unit]), zero4[0]);
            _mm256_store_pd(&(in_coef_old_imag->element[0][i * num_loop_unit]), zero4[0]);
            _mm256_store_pd(&(in_coef_old_real->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero4[0]);
            _mm256_store_pd(&(in_coef_old_imag->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero4[0]);
            _mm256_store_pd(&(in_coef_old_real->element[1][i * num_loop_unit]), zero4[1]);
            _mm256_store_pd(&(in_coef_old_imag->element[1][i * num_loop_unit]), zero4[1]);
            _mm256_store_pd(&(in_coef_old_real->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero4[1]);
            _mm256_store_pd(&(in_coef_old_imag->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero4[1]);
            _mm256_store_pd(&(in_coef_old_real->element[2][i * num_loop_unit]), zero4[2]);
            _mm256_store_pd(&(in_coef_old_imag->element[2][i * num_loop_unit]), zero4[2]);
            _mm256_store_pd(&(in_coef_old_real->element[2][i * num_loop_unit + _BNC_D_WIDTH]), zero4[2]);
            _mm256_store_pd(&(in_coef_old_imag->element[2][i * num_loop_unit + _BNC_D_WIDTH]), zero4[2]);
            _mm256_store_pd(&(in_coef_old_real->element[3][i * num_loop_unit]), zero4[3]);
            _mm256_store_pd(&(in_coef_old_imag->element[3][i * num_loop_unit]), zero4[3]);
            _mm256_store_pd(&(in_coef_old_real->element[3][i * num_loop_unit + _BNC_D_WIDTH]), zero4[3]);
            _mm256_store_pd(&(in_coef_old_imag->element[3][i * num_loop_unit + _BNC_D_WIDTH]), zero4[3]);

            _mm256_store_pd(&(in_coef_old_real->element[0][i * _BNC_D_WIDTH]), new_coef4_real[0]);
            _mm256_store_pd(&(in_coef_old_imag->element[0][i * _BNC_D_WIDTH]), new_coef4_imag[0]);
            _mm256_store_pd(&(in_coef_old_real->element[1][i * _BNC_D_WIDTH]), new_coef4_real[1]);
            _mm256_store_pd(&(in_coef_old_imag->element[1][i * _BNC_D_WIDTH]), new_coef4_imag[1]);
            _mm256_store_pd(&(in_coef_old_real->element[2][i * _BNC_D_WIDTH]), new_coef4_real[2]);
            _mm256_store_pd(&(in_coef_old_imag->element[2][i * _BNC_D_WIDTH]), new_coef4_imag[2]);
            _mm256_store_pd(&(in_coef_old_real->element[3][i * _BNC_D_WIDTH]), new_coef4_real[3]);
            _mm256_store_pd(&(in_coef_old_imag->element[3][i * _BNC_D_WIDTH]), new_coef4_imag[3]);
        }
#elif defined(__AVX512F__) // __AVX512F__
        _bncavx512_rqd_set1_qd(x8_real, in_x.val_re);
        _bncavx512_rqd_set1_qd(x8_imag, in_x.val_im);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            old_coef8_real[0][0] = _mm512_load_pd(&(in_coef_old_real->element[0][i * num_loop_unit]));
            old_coef8_real[1][0] = _mm512_load_pd(&(in_coef_old_real->element[0][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_real[0][1] = _mm512_load_pd(&(in_coef_old_real->element[1][i * num_loop_unit]));
            old_coef8_real[1][1] = _mm512_load_pd(&(in_coef_old_real->element[1][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_real[0][2] = _mm512_load_pd(&(in_coef_old_real->element[2][i * num_loop_unit]));
            old_coef8_real[1][2] = _mm512_load_pd(&(in_coef_old_real->element[2][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_real[0][3] = _mm512_load_pd(&(in_coef_old_real->element[3][i * num_loop_unit]));
            old_coef8_real[1][3] = _mm512_load_pd(&(in_coef_old_real->element[3][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));

            old_coef8_imag[0][0] = _mm512_load_pd(&(in_coef_old_imag->element[0][i * num_loop_unit]));
            old_coef8_imag[1][0] = _mm512_load_pd(&(in_coef_old_imag->element[0][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_imag[0][1] = _mm512_load_pd(&(in_coef_old_imag->element[1][i * num_loop_unit]));
            old_coef8_imag[1][1] = _mm512_load_pd(&(in_coef_old_imag->element[1][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_imag[0][2] = _mm512_load_pd(&(in_coef_old_imag->element[2][i * num_loop_unit]));
            old_coef8_imag[1][2] = _mm512_load_pd(&(in_coef_old_imag->element[2][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_imag[0][3] = _mm512_load_pd(&(in_coef_old_imag->element[3][i * num_loop_unit]));
            old_coef8_imag[1][3] = _mm512_load_pd(&(in_coef_old_imag->element[3][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            
            a08_real[0] = _mm512_unpacklo_pd(old_coef8_real[0][0], old_coef8_real[1][0]);
            a18_real[0] = _mm512_unpackhi_pd(old_coef8_real[0][0], old_coef8_real[1][0]);
            a08_real[1] = _mm512_unpacklo_pd(old_coef8_real[0][1], old_coef8_real[1][1]);
            a18_real[1] = _mm512_unpackhi_pd(old_coef8_real[0][1], old_coef8_real[1][1]);
            a08_real[2] = _mm512_unpacklo_pd(old_coef8_real[0][2], old_coef8_real[1][2]);
            a18_real[2] = _mm512_unpackhi_pd(old_coef8_real[0][2], old_coef8_real[1][2]);
            a08_real[3] = _mm512_unpacklo_pd(old_coef8_real[0][3], old_coef8_real[1][3]);
            a18_real[3] = _mm512_unpackhi_pd(old_coef8_real[0][3], old_coef8_real[1][3]);

            a08_imag[0] = _mm512_unpacklo_pd(old_coef8_imag[0][0], old_coef8_imag[1][0]);
            a18_imag[0] = _mm512_unpackhi_pd(old_coef8_imag[0][0], old_coef8_imag[1][0]);
            a08_imag[1] = _mm512_unpacklo_pd(old_coef8_imag[0][1], old_coef8_imag[1][1]);
            a18_imag[1] = _mm512_unpackhi_pd(old_coef8_imag[0][1], old_coef8_imag[1][1]);
            a08_imag[2] = _mm512_unpacklo_pd(old_coef8_imag[0][2], old_coef8_imag[1][2]);
            a18_imag[2] = _mm512_unpackhi_pd(old_coef8_imag[0][2], old_coef8_imag[1][2]);
            a08_imag[3] = _mm512_unpacklo_pd(old_coef8_imag[0][3], old_coef8_imag[1][3]);
            a18_imag[3] = _mm512_unpackhi_pd(old_coef8_imag[0][3], old_coef8_imag[1][3]);

            _bncavx512_rcqd_mul(ctmp8_real, ctmp8_imag, a18_real, a18_imag, x8_real, x8_imag);
            _bncavx512_rcqd_add(new_coef8_real, new_coef8_imag, ctmp8_real, ctmp8_imag, a08_real, a08_imag);
            
            new_coef8_real[0] = _mm512_permutex_pd(new_coef8_real[0], 0xD8);
            new_coef8_imag[0] = _mm512_permutex_pd(new_coef8_imag[0], 0xD8);
            new_coef8_real[1] = _mm512_permutex_pd(new_coef8_real[1], 0xD8);
            new_coef8_imag[1] = _mm512_permutex_pd(new_coef8_imag[1], 0xD8);
            new_coef8_real[2] = _mm512_permutex_pd(new_coef8_real[2], 0xD8);
            new_coef8_imag[2] = _mm512_permutex_pd(new_coef8_imag[2], 0xD8);
            new_coef8_real[3] = _mm512_permutex_pd(new_coef8_real[3], 0xD8);
            new_coef8_imag[3] = _mm512_permutex_pd(new_coef8_imag[3], 0xD8);
            
            _mm512_store_pd(&(in_coef_old_real->element[0][i * num_loop_unit]), zero8[0]);
            _mm512_store_pd(&(in_coef_old_imag->element[0][i * num_loop_unit]), zero8[0]);
            _mm512_store_pd(&(in_coef_old_real->element[0][i * num_loop_unit + (2 * _BNC_D_WIDTH)]), zero8[0]);
            _mm512_store_pd(&(in_coef_old_imag->element[0][i * num_loop_unit + (2 * _BNC_D_WIDTH)]), zero8[0]);
            _mm512_store_pd(&(in_coef_old_real->element[1][i * num_loop_unit]), zero8[1]);
            _mm512_store_pd(&(in_coef_old_imag->element[1][i * num_loop_unit]), zero8[1]);
            _mm512_store_pd(&(in_coef_old_real->element[1][i * num_loop_unit + (2 * _BNC_D_WIDTH)]), zero8[1]);
            _mm512_store_pd(&(in_coef_old_imag->element[1][i * num_loop_unit + (2 * _BNC_D_WIDTH)]), zero8[1]);
            _mm512_store_pd(&(in_coef_old_real->element[2][i * num_loop_unit]), zero8[2]);
            _mm512_store_pd(&(in_coef_old_imag->element[2][i * num_loop_unit]), zero8[2]);
            _mm512_store_pd(&(in_coef_old_real->element[2][i * num_loop_unit + (2 * _BNC_D_WIDTH)]), zero8[2]);
            _mm512_store_pd(&(in_coef_old_imag->element[2][i * num_loop_unit + (2 * _BNC_D_WIDTH)]), zero8[2]);
            _mm512_store_pd(&(in_coef_old_real->element[3][i * num_loop_unit]), zero8[3]);
            _mm512_store_pd(&(in_coef_old_imag->element[3][i * num_loop_unit]), zero8[3]);
            _mm512_store_pd(&(in_coef_old_real->element[3][i * num_loop_unit + (2 * _BNC_D_WIDTH)]), zero8[3]);
            _mm512_store_pd(&(in_coef_old_imag->element[3][i * num_loop_unit + (2 * _BNC_D_WIDTH)]), zero8[3]);

            _mm512_store_pd(&(in_coef_old_real->element[0][i * (2 * _BNC_D_WIDTH)]), new_coef8_real[0]);
            _mm512_store_pd(&(in_coef_old_imag->element[0][i * (2 * _BNC_D_WIDTH)]), new_coef8_imag[0]);
            _mm512_store_pd(&(in_coef_old_real->element[1][i * (2 * _BNC_D_WIDTH)]), new_coef8_real[1]);
            _mm512_store_pd(&(in_coef_old_imag->element[1][i * (2 * _BNC_D_WIDTH)]), new_coef8_imag[1]);
            _mm512_store_pd(&(in_coef_old_real->element[2][i * (2 * _BNC_D_WIDTH)]), new_coef8_real[2]);
            _mm512_store_pd(&(in_coef_old_imag->element[2][i * (2 * _BNC_D_WIDTH)]), new_coef8_imag[2]);
            _mm512_store_pd(&(in_coef_old_real->element[3][i * (2 * _BNC_D_WIDTH)]), new_coef8_real[3]);
            _mm512_store_pd(&(in_coef_old_imag->element[3][i * (2 * _BNC_D_WIDTH)]), new_coef8_imag[3]);
        }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, 2-lane, derived from NEON)

	svbool_t pg = svwhilelt_b64_s64((int64_t)0, (int64_t)2);        x2_real_0 = svdup_f64((in_x.val_re)[0]); x2_real_1 = svdup_f64((in_x.val_re)[1]); x2_real_2 = svdup_f64((in_x.val_re)[2]); x2_real_3 = svdup_f64((in_x.val_re)[3]);
        x2_imag_0 = svdup_f64((in_x.val_im)[0]); x2_imag_1 = svdup_f64((in_x.val_im)[1]); x2_imag_2 = svdup_f64((in_x.val_im)[2]); x2_imag_3 = svdup_f64((in_x.val_im)[3]);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            old_coef2_real_0_0 = svld1_f64(pg, &(in_coef_old_real->element[0][i * num_loop_unit]));
            old_coef2_real_1_0 = svld1_f64(pg, &(in_coef_old_real->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_real_0_1 = svld1_f64(pg, &(in_coef_old_real->element[1][i * num_loop_unit]));
            old_coef2_real_1_1 = svld1_f64(pg, &(in_coef_old_real->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_real_0_2 = svld1_f64(pg, &(in_coef_old_real->element[2][i * num_loop_unit]));
            old_coef2_real_1_2 = svld1_f64(pg, &(in_coef_old_real->element[2][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_real_0_3 = svld1_f64(pg, &(in_coef_old_real->element[3][i * num_loop_unit]));
            old_coef2_real_1_3 = svld1_f64(pg, &(in_coef_old_real->element[3][i * num_loop_unit + _BNC_D_WIDTH]));

            old_coef2_imag_0_0 = svld1_f64(pg, &(in_coef_old_imag->element[0][i * num_loop_unit]));
            old_coef2_imag_1_0 = svld1_f64(pg, &(in_coef_old_imag->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_imag_0_1 = svld1_f64(pg, &(in_coef_old_imag->element[1][i * num_loop_unit]));
            old_coef2_imag_1_1 = svld1_f64(pg, &(in_coef_old_imag->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_imag_0_2 = svld1_f64(pg, &(in_coef_old_imag->element[2][i * num_loop_unit]));
            old_coef2_imag_1_2 = svld1_f64(pg, &(in_coef_old_imag->element[2][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_imag_0_3 = svld1_f64(pg, &(in_coef_old_imag->element[3][i * num_loop_unit]));
            old_coef2_imag_1_3 = svld1_f64(pg, &(in_coef_old_imag->element[3][i * num_loop_unit + _BNC_D_WIDTH]));
            
            a02_real_0 = svtrn1_f64(old_coef2_real_0_0, old_coef2_real_1_0);
            a12_real_0 = svtrn2_f64(old_coef2_real_0_0, old_coef2_real_1_0);
            a02_real_1 = svtrn1_f64(old_coef2_real_0_1, old_coef2_real_1_1);
            a12_real_1 = svtrn2_f64(old_coef2_real_0_1, old_coef2_real_1_1);
            a02_real_2 = svtrn1_f64(old_coef2_real_0_2, old_coef2_real_1_2);
            a12_real_2 = svtrn2_f64(old_coef2_real_0_2, old_coef2_real_1_2);
            a02_real_3 = svtrn1_f64(old_coef2_real_0_3, old_coef2_real_1_3);
            a12_real_3 = svtrn2_f64(old_coef2_real_0_3, old_coef2_real_1_3);

            a02_imag_0 = svtrn1_f64(old_coef2_imag_0_0, old_coef2_imag_1_0);
            a12_imag_0 = svtrn2_f64(old_coef2_imag_0_0, old_coef2_imag_1_0);
            a02_imag_1 = svtrn1_f64(old_coef2_imag_0_1, old_coef2_imag_1_1);
            a12_imag_1 = svtrn2_f64(old_coef2_imag_0_1, old_coef2_imag_1_1);
            a02_imag_2 = svtrn1_f64(old_coef2_imag_0_2, old_coef2_imag_1_2);
            a12_imag_2 = svtrn2_f64(old_coef2_imag_0_2, old_coef2_imag_1_2);
            a02_imag_3 = svtrn1_f64(old_coef2_imag_0_3, old_coef2_imag_1_3);
            a12_imag_3 = svtrn2_f64(old_coef2_imag_0_3, old_coef2_imag_1_3);

            _bncsve2_rcqd_mul(pg, &ctmp2_real_0, &ctmp2_real_1, &ctmp2_real_2, &ctmp2_real_3, &ctmp2_imag_0, &ctmp2_imag_1, &ctmp2_imag_2, &ctmp2_imag_3, a12_real_0, a12_real_1, a12_real_2, a12_real_3, a12_imag_0, a12_imag_1, a12_imag_2, a12_imag_3, x2_real_0, x2_real_1, x2_real_2, x2_real_3, x2_imag_0, x2_imag_1, x2_imag_2, x2_imag_3);
            _bncsve2_rcqd_add(pg, &new_coef2_real_0, &new_coef2_real_1, &new_coef2_real_2, &new_coef2_real_3, &new_coef2_imag_0, &new_coef2_imag_1, &new_coef2_imag_2, &new_coef2_imag_3, ctmp2_real_0, ctmp2_real_1, ctmp2_real_2, ctmp2_real_3, ctmp2_imag_0, ctmp2_imag_1, ctmp2_imag_2, ctmp2_imag_3, a02_real_0, a02_real_1, a02_real_2, a02_real_3, a02_imag_0, a02_imag_1, a02_imag_2, a02_imag_3);
            
            svst1_f64(pg, &(in_coef_old_real->element[0][i * num_loop_unit]), zero2_0);
            svst1_f64(pg, &(in_coef_old_imag->element[0][i * num_loop_unit]), zero2_0);
            svst1_f64(pg, &(in_coef_old_real->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero2_0);
            svst1_f64(pg, &(in_coef_old_imag->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero2_0);
            svst1_f64(pg, &(in_coef_old_real->element[1][i * num_loop_unit]), zero2_1);
            svst1_f64(pg, &(in_coef_old_imag->element[1][i * num_loop_unit]), zero2_1);
            svst1_f64(pg, &(in_coef_old_real->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero2_1);
            svst1_f64(pg, &(in_coef_old_imag->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero2_1);
            svst1_f64(pg, &(in_coef_old_real->element[2][i * num_loop_unit]), zero2_2);
            svst1_f64(pg, &(in_coef_old_imag->element[2][i * num_loop_unit]), zero2_2);
            svst1_f64(pg, &(in_coef_old_real->element[2][i * num_loop_unit + _BNC_D_WIDTH]), zero2_2);
            svst1_f64(pg, &(in_coef_old_imag->element[2][i * num_loop_unit + _BNC_D_WIDTH]), zero2_2);
            svst1_f64(pg, &(in_coef_old_real->element[3][i * num_loop_unit]), zero2_3);
            svst1_f64(pg, &(in_coef_old_imag->element[3][i * num_loop_unit]), zero2_3);
            svst1_f64(pg, &(in_coef_old_real->element[3][i * num_loop_unit + _BNC_D_WIDTH]), zero2_3);
            svst1_f64(pg, &(in_coef_old_imag->element[3][i * num_loop_unit + _BNC_D_WIDTH]), zero2_3);

            svst1_f64(pg, &(in_coef_old_real->element[0][i * _BNC_D_WIDTH]), new_coef2_real_0);
            svst1_f64(pg, &(in_coef_old_imag->element[0][i * _BNC_D_WIDTH]), new_coef2_imag_0);
            svst1_f64(pg, &(in_coef_old_real->element[1][i * _BNC_D_WIDTH]), new_coef2_real_1);
            svst1_f64(pg, &(in_coef_old_imag->element[1][i * _BNC_D_WIDTH]), new_coef2_imag_1);
            svst1_f64(pg, &(in_coef_old_real->element[2][i * _BNC_D_WIDTH]), new_coef2_real_2);
            svst1_f64(pg, &(in_coef_old_imag->element[2][i * _BNC_D_WIDTH]), new_coef2_imag_2);
            svst1_f64(pg, &(in_coef_old_real->element[3][i * _BNC_D_WIDTH]), new_coef2_real_3);
            svst1_f64(pg, &(in_coef_old_imag->element[3][i * _BNC_D_WIDTH]), new_coef2_imag_3);
        }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // ARM NEON
        _bncneon_rqd_set1_qd(x2_real, in_x.val_re);
        _bncneon_rqd_set1_qd(x2_imag, in_x.val_im);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            old_coef2_real[0][0] = vld1q_f64(&(in_coef_old_real->element[0][i * num_loop_unit]));
            old_coef2_real[1][0] = vld1q_f64(&(in_coef_old_real->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_real[0][1] = vld1q_f64(&(in_coef_old_real->element[1][i * num_loop_unit]));
            old_coef2_real[1][1] = vld1q_f64(&(in_coef_old_real->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_real[0][2] = vld1q_f64(&(in_coef_old_real->element[2][i * num_loop_unit]));
            old_coef2_real[1][2] = vld1q_f64(&(in_coef_old_real->element[2][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_real[0][3] = vld1q_f64(&(in_coef_old_real->element[3][i * num_loop_unit]));
            old_coef2_real[1][3] = vld1q_f64(&(in_coef_old_real->element[3][i * num_loop_unit + _BNC_D_WIDTH]));

            old_coef2_imag[0][0] = vld1q_f64(&(in_coef_old_imag->element[0][i * num_loop_unit]));
            old_coef2_imag[1][0] = vld1q_f64(&(in_coef_old_imag->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_imag[0][1] = vld1q_f64(&(in_coef_old_imag->element[1][i * num_loop_unit]));
            old_coef2_imag[1][1] = vld1q_f64(&(in_coef_old_imag->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_imag[0][2] = vld1q_f64(&(in_coef_old_imag->element[2][i * num_loop_unit]));
            old_coef2_imag[1][2] = vld1q_f64(&(in_coef_old_imag->element[2][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_imag[0][3] = vld1q_f64(&(in_coef_old_imag->element[3][i * num_loop_unit]));
            old_coef2_imag[1][3] = vld1q_f64(&(in_coef_old_imag->element[3][i * num_loop_unit + _BNC_D_WIDTH]));
            
            a02_real[0] = vzip1q_f64(old_coef2_real[0][0], old_coef2_real[1][0]);
            a12_real[0] = vzip2q_f64(old_coef2_real[0][0], old_coef2_real[1][0]);
            a02_real[1] = vzip1q_f64(old_coef2_real[0][1], old_coef2_real[1][1]);
            a12_real[1] = vzip2q_f64(old_coef2_real[0][1], old_coef2_real[1][1]);
            a02_real[2] = vzip1q_f64(old_coef2_real[0][2], old_coef2_real[1][2]);
            a12_real[2] = vzip2q_f64(old_coef2_real[0][2], old_coef2_real[1][2]);
            a02_real[3] = vzip1q_f64(old_coef2_real[0][3], old_coef2_real[1][3]);
            a12_real[3] = vzip2q_f64(old_coef2_real[0][3], old_coef2_real[1][3]);

            a02_imag[0] = vzip1q_f64(old_coef2_imag[0][0], old_coef2_imag[1][0]);
            a12_imag[0] = vzip2q_f64(old_coef2_imag[0][0], old_coef2_imag[1][0]);
            a02_imag[1] = vzip1q_f64(old_coef2_imag[0][1], old_coef2_imag[1][1]);
            a12_imag[1] = vzip2q_f64(old_coef2_imag[0][1], old_coef2_imag[1][1]);
            a02_imag[2] = vzip1q_f64(old_coef2_imag[0][2], old_coef2_imag[1][2]);
            a12_imag[2] = vzip2q_f64(old_coef2_imag[0][2], old_coef2_imag[1][2]);
            a02_imag[3] = vzip1q_f64(old_coef2_imag[0][3], old_coef2_imag[1][3]);
            a12_imag[3] = vzip2q_f64(old_coef2_imag[0][3], old_coef2_imag[1][3]);

            _bncneon_rcqd_mul(ctmp2_real, ctmp2_imag, a12_real, a12_imag, x2_real, x2_imag);
            _bncneon_rcqd_add(new_coef2_real, new_coef2_imag, ctmp2_real, ctmp2_imag, a02_real, a02_imag);
            
            vst1q_f64(&(in_coef_old_real->element[0][i * num_loop_unit]), zero2[0]);
            vst1q_f64(&(in_coef_old_imag->element[0][i * num_loop_unit]), zero2[0]);
            vst1q_f64(&(in_coef_old_real->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero2[0]);
            vst1q_f64(&(in_coef_old_imag->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero2[0]);
            vst1q_f64(&(in_coef_old_real->element[1][i * num_loop_unit]), zero2[1]);
            vst1q_f64(&(in_coef_old_imag->element[1][i * num_loop_unit]), zero2[1]);
            vst1q_f64(&(in_coef_old_real->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero2[1]);
            vst1q_f64(&(in_coef_old_imag->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero2[1]);
            vst1q_f64(&(in_coef_old_real->element[2][i * num_loop_unit]), zero2[2]);
            vst1q_f64(&(in_coef_old_imag->element[2][i * num_loop_unit]), zero2[2]);
            vst1q_f64(&(in_coef_old_real->element[2][i * num_loop_unit + _BNC_D_WIDTH]), zero2[2]);
            vst1q_f64(&(in_coef_old_imag->element[2][i * num_loop_unit + _BNC_D_WIDTH]), zero2[2]);
            vst1q_f64(&(in_coef_old_real->element[3][i * num_loop_unit]), zero2[3]);
            vst1q_f64(&(in_coef_old_imag->element[3][i * num_loop_unit]), zero2[3]);
            vst1q_f64(&(in_coef_old_real->element[3][i * num_loop_unit + _BNC_D_WIDTH]), zero2[3]);
            vst1q_f64(&(in_coef_old_imag->element[3][i * num_loop_unit + _BNC_D_WIDTH]), zero2[3]);

            vst1q_f64(&(in_coef_old_real->element[0][i * _BNC_D_WIDTH]), new_coef2_real[0]);
            vst1q_f64(&(in_coef_old_imag->element[0][i * _BNC_D_WIDTH]), new_coef2_imag[0]);
            vst1q_f64(&(in_coef_old_real->element[1][i * _BNC_D_WIDTH]), new_coef2_real[1]);
            vst1q_f64(&(in_coef_old_imag->element[1][i * _BNC_D_WIDTH]), new_coef2_imag[1]);
            vst1q_f64(&(in_coef_old_real->element[2][i * _BNC_D_WIDTH]), new_coef2_real[2]);
            vst1q_f64(&(in_coef_old_imag->element[2][i * _BNC_D_WIDTH]), new_coef2_imag[2]);
            vst1q_f64(&(in_coef_old_real->element[3][i * _BNC_D_WIDTH]), new_coef2_real[3]);
            vst1q_f64(&(in_coef_old_imag->element[3][i * _BNC_D_WIDTH]), new_coef2_imag[3]);
        }
#endif // __AVX2__

        rcqd_mul(&ctmp, &in_x, &in_x);
        rcqd_set(&in_x, &ctmp);
        if(num_in_coef == num_loop_unit) break;
        num_in_coef /= 2;
        if((num_in_coef % num_loop_unit) != 0)
            num_in_coef = (long int)ceil(((double)num_in_coef / (double)num_loop_unit)) * num_loop_unit;

        in_degree = num_in_coef - 1;
    }

#if defined(__AVX2__) || defined(__AVX512F__) // __AVX2__
    rqd_set(in_ret.val_re, get_qdvector_i(in_coef_old_real, _BNC_D_WIDTH - 1));
    rqd_set(in_ret.val_im, get_qdvector_i(in_coef_old_imag, _BNC_D_WIDTH - 1));
    for(i = _BNC_D_WIDTH - 2; i >= 0; i--)
    {
        rcqd_mul(&ctmp, &in_ret, &in_x);
        rqd_set(in_ret.val_re, get_qdvector_i(in_coef_old_real, i));
        rqd_set(in_ret.val_im, get_qdvector_i(in_coef_old_imag, i));
        rcqd_add(&in_ret, &in_ret, &ctmp);
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    rqd_set(in_ret.val_re, get_qdvector_i(in_coef_old_real, _BNC_D_WIDTH - 1));
    rqd_set(in_ret.val_im, get_qdvector_i(in_coef_old_imag, _BNC_D_WIDTH - 1));
    for(i = _BNC_D_WIDTH - 2; i >= 0; i--)
    {
        rcqd_mul(&ctmp, &in_ret, &in_x);
        rqd_set(in_ret.val_re, get_qdvector_i(in_coef_old_real, i));
        rqd_set(in_ret.val_im, get_qdvector_i(in_coef_old_imag, i));
        rcqd_add(&in_ret, &in_ret, &ctmp);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // ARM NEON
    rqd_set(in_ret.val_re, get_qdvector_i(in_coef_old_real, _BNC_D_WIDTH - 1));
    rqd_set(in_ret.val_im, get_qdvector_i(in_coef_old_imag, _BNC_D_WIDTH - 1));
    for(i = _BNC_D_WIDTH - 2; i >= 0; i--)
    {
        rcqd_mul(&ctmp, &in_ret, &in_x);
        rqd_set(in_ret.val_re, get_qdvector_i(in_coef_old_real, i));
        rqd_set(in_ret.val_im, get_qdvector_i(in_coef_old_imag, i));
        rcqd_add(&in_ret, &in_ret, &ctmp);
    }
#endif // __AVX2__

    rcqd_set(ret, &in_ret);

    free_qdvector(in_coef_old_real);
    free_qdvector(in_coef_old_imag);

    return;
}

#ifdef __cpluplus
} // extern "C" {
#endif // __cplusplus
