/********************************************************************************/
/* ctd_poly.c: Algebraic Equations and Complex Polynomials                      */
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
/*   CTDPoly init_ctdpoly(long int max_length)   */
/*   CTDPoly init2_ctdpoly(long int max_length, unsigned long prec)*/
/* Free:                                         */
/*   void free_ctdpoly(CTDPoly pol)              */
/* Get & Set Values:                             */
/*   ctdfloat *get_ctdpoly_i(CTDPoly pol, long int index) */
/*   long int setdegree_ctdpoly(CTDPoly)         */
/*   void set_ctdpoly_i(CTDPoly pol, long int index, ctdfloat val) */
/* Output:                                       */
/*   void print_ctdpoly(CTDPoly pol)             */
/*************************************************/
CTDPoly init_ctdpoly(long int max_length)
{
	CTDPoly ret = NULL;
	long int i;

	if(max_length <= 0)
	{
		fprintf(stderr, "ERROR: init_ctdpoly\n");
		return ret;
	}

	ret = (CTDPoly)malloc(sizeof(ctdpoly));
	if(ret == NULL)
		return ret;

	ret->coef = (ctdfloat *)calloc(sizeof(ctdfloat), max_length);
	if(ret->coef == NULL)
    {
        free(ret);
		return NULL;
    }

	/* All 0 */
	for(i = 0; i < max_length; i++)
		rctd_set_ui(&(ret->coef[i]), 0UL);

    // zero := 0
    rctd_set0(&(ret->zero));

	ret->deg = 0;
	ret->max_len = max_length;

	return ret;
}

void free_ctdpoly(CTDPoly pol)
{
	long int i;

	if(pol == NULL)
		return;

	if(pol->coef != NULL)
		free(pol->coef); // Fix! 2012-06-03 by T.Kouya

	free(pol);
}

ctdfloat *get_ctdpoly_i(CTDPoly pol, long int index)
{
	if(index > pol->deg)
		return &(pol->zero);
	else
		return &(pol->coef[index]);
}

/* find maximum degree at pol->coef[deg] != 0 */
long int setdegree_ctdpoly(CTDPoly pol)
{
	long int i;
    tdfloat tmp;

	for(i = pol->max_len - 1; i > 0; i--)
	{
        rctd_abs(&tmp, get_ctdpoly_i(pol, i));
		if(rtd_cmp_ui(tmp.val, 0UL) != 0)
		{
			pol->deg = i;
			return i;
		}
	}

	pol->deg = 0;
	return 0;

}

void set_ctdpoly_i(CTDPoly pol, long int index, ctdfloat *val)
{
    tdfloat tmp;

    if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	rctd_set(&(pol->coef[index]), val);
    rctd_abs(&tmp, val);
	if((pol->deg < index) && (rtd_cmp_ui(tmp.val, 0UL) != 0))
		pol->deg = index;
}

#ifdef USE_GMP
void set_ctdpoly_i_mpc(CTDPoly pol, long int index, mpc_t val)
{
	ctdfloat ctmp;
    tdfloat tmp;

	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	mpf_get_td(ctmp.val_re, mpc_realref(val));
    mpf_get_td(ctmp.val_im, mpc_imagref(val));
    set_ctdpoly_i(pol, index, &ctmp);
	//rctd_set(pol->coef[index].val, tmp);
    rctd_abs(&tmp, &ctmp);

    if((pol->deg < index) && (rtd_cmp_ui(tmp.val, 0UL) != 0))
		pol->deg = index;

}

// Initialize and substitute polynomial from org_pol
CTDPoly init_set_ctdpoly_cmpfpoly(CMPFPoly org_pol)
{
	CTDPoly ret = NULL;
	long int i;

	ret = init_ctdpoly(org_pol->max_len);
	if(ret == NULL)
		return ret;

	for(i = 0; i <= org_pol->deg; i++)
		set_ctdpoly_i_mpc(ret, i, get_cmpfpoly_i(org_pol, i));

	return ret;
}

#endif // USE_GMP

// Initialize and substitute polynomial from org_pol
CTDPoly init_set_ctdpoly_tdpoly(TDPoly org_pol)
{
	CTDPoly ret = NULL;
	long int i;
	ctdfloat ctmp;

	ret = init_ctdpoly(org_pol->max_len);
	if(ret == NULL)
		return ret;

	rtd_set0(ctmp.val_im);
	for(i = 0; i <= org_pol->deg; i++)
	{
		rtd_set(ctmp.val_re, get_tdpoly_i(org_pol, i));
		set_ctdpoly_i(ret, i, &ctmp);
	}

	return ret;
}


// Initialize and substitute polynomial from org_pol
CTDPoly init_set_ctdpoly(CTDPoly org_pol)
{
	CTDPoly ret = NULL;
	long int i;

	ret = init_ctdpoly(org_pol->max_len);
	if(ret == NULL)
		return ret;

	subst_ctdpoly(ret, org_pol);

	return ret;
}

// substitute the values to pol
void subst_ctdpoly(CTDPoly pol, CTDPoly org_pol)
{
	long int i;

	if(pol == NULL || org_pol == NULL)
		return;

	if(pol->deg > org_pol->deg)
	{
		for(i = pol->deg; i >= org_pol->deg; i--)
			set_ctdpoly_i(pol, i, &(pol->zero));
	}

	pol->deg = org_pol->deg;

	for(i = 0; i <= org_pol->deg; i++)
		set_ctdpoly_i(pol, i, get_ctdpoly_i(org_pol, i));

	setdegree_ctdpoly(pol);
}

// set all coefficients to zero
void set0_ctdpoly(CTDPoly pol)
{
	long int i;

	for(i = 0; i < pol->max_len; i++)
		rctd_set_ui(&(pol->coef[i]), 0UL);

	pol->deg = 0;
}

// printf
void print_ctdpoly(CTDPoly pol)
{
	long int i;
	ctdfloat *pol_i;

	if(pol == NULL)
	{
		printf("NULL\n");
		return;
	}

	printf("deg = %ld\n", pol->deg);
	for(i = 0; i <= pol->deg; i++)
	{
		printf("coef[%5ld] = ", i);
		//rctd_print(get_ctdpoly_i(pol, i));
		//rctd_out_str(get_ctdpoly_i(pol, i));
		pol_i = get_ctdpoly_i(pol, i);
		rtd_out_str(pol_i->val_re);
		printf(" + ");
		rtd_out_str(pol_i->val_im);
		printf(" * I ");
        printf("\n");
	}
}

// fprintf
/*
void fprint_ctdpoly(FILE *fp, CTDPoly pol)
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
		rctd_out_str(fp, get_ctdpoly_i(pol, i));
		fprintf(fp, "\n");
	}
}
*/

// Evaluation
void eval_ctdpoly(ctdfloat *ret, CTDPoly pol, ctdfloat *x)
{
	long int i;
	ctdfloat ctmp;

	// Horner method
	rctd_set(ret, get_ctdpoly_i(pol, pol->deg));
	for(i = pol->deg - 1; i >= 0; i--)
	{
		rctd_mul(&ctmp, ret, x);
		rctd_add(ret, &ctmp, get_ctdpoly_i(pol, i));
	}
}

// Addition: ret = pol1 + pol2
void add_ctdpoly(CTDPoly ret, CTDPoly pol1, CTDPoly pol2)
{
	long int i, max_deg;
	ctdfloat ctmp;

	max_deg = (pol1->deg > pol2->deg) ? pol1->deg : pol2->deg;

	for(i = 0; i <= max_deg; i++)
	{
		rctd_add(&ctmp, get_ctdpoly_i(pol1, i), get_ctdpoly_i(pol2, i));
		set_ctdpoly_i(ret, i, &ctmp);
	}

	setdegree_ctdpoly(ret);
}

// Subtraction: ret = pol1 - pol2
void sub_ctdpoly(CTDPoly ret, CTDPoly pol1, CTDPoly pol2)
{
	long int i, max_deg;
	ctdfloat ctmp;

	max_deg = (pol1->deg > pol2->deg) ? pol1->deg : pol2->deg;

	for(i = 0; i <= max_deg; i++)
	{
		rctd_sub(&ctmp, get_ctdpoly_i(pol1, i), get_ctdpoly_i(pol2, i));
		set_ctdpoly_i(ret, i, &ctmp);
	}

	setdegree_ctdpoly(ret);
}

// Scalar multiplication: ret = alpha * pol
void scal_ctdpoly(CTDPoly ret, ctdfloat *alpha, CTDPoly pol)
{
	long int i;
	ctdfloat ctmp;

	for(i = 0; i <= pol->deg; i++)
	{
		rctd_mul(&ctmp, alpha, get_ctdpoly_i(pol, i));
		set_ctdpoly_i(ret, i, &ctmp);
	}

	setdegree_ctdpoly(ret);
}

// Multiplication: ret = pol1 * pol2
void mul_ctdpoly(CTDPoly ret, CTDPoly pol1, CTDPoly pol2)
{
	long int i, j;
	ctdfloat ctmp, ctmp2;
	CTDPoly work;

	work = init_ctdpoly(pol1->deg + pol2->deg + 1);
	set0_ctdpoly(work);

	for(i = 0; i <= pol1->deg; i++)
	{
		for(j = 0; j <= pol2->deg; j++)
		{
			rctd_mul(&ctmp, get_ctdpoly_i(pol1, i), get_ctdpoly_i(pol2, j));
			rctd_add(&ctmp2, get_ctdpoly_i(work, i + j), &ctmp);
			set_ctdpoly_i(work, i + j, &ctmp2);
		}
	}

	subst_ctdpoly(ret, work);
	free_ctdpoly(work);
}

// Division: quot = pol1 / pol2, rem = pol1 % pol2
void div_ctdpoly(CTDPoly quot, CTDPoly rem, CTDPoly pol1, CTDPoly pol2)
{
	long int i;
	ctdfloat ctmp, ctmp2;
	CTDPoly work_quot, work_rem, work_pol;

	if(pol1->deg < pol2->deg)
	{
		set0_ctdpoly(quot);
		subst_ctdpoly(rem, pol1);
		return;
	}

	work_quot = init_ctdpoly(pol1->deg - pol2->deg + 1);
	work_rem = init_set_ctdpoly(pol1);
	work_pol = init_ctdpoly(pol1->deg + 1);

	set0_ctdpoly(work_quot);

	for(i = pol1->deg - pol2->deg; i >= 0; i--)
	{
		rctd_div(&ctmp, get_ctdpoly_i(work_rem, i + pol2->deg), get_ctdpoly_i(pol2, pol2->deg));
		set_ctdpoly_i(work_quot, i, &ctmp);

		// work_pol = ctmp * pol2 * x^i
		set0_ctdpoly(work_pol);
		scal_ctdpoly(work_pol, &ctmp, pol2);
		
		// Shift work_pol by i positions
		long int j;
		for(j = work_pol->deg; j >= 0; j--)
		{
			if(j + i <= work_pol->max_len - 1)
			{
				rctd_set(&ctmp2, get_ctdpoly_i(work_pol, j));
				set_ctdpoly_i(work_pol, j + i, &ctmp2);
			}
		}
		for(j = 0; j < i; j++)
		{
			rctd_set_ui(&ctmp2, 0UL);
			set_ctdpoly_i(work_pol, j, &ctmp2);
		}
		setdegree_ctdpoly(work_pol);

		sub_ctdpoly(work_rem, work_rem, work_pol);
	}

	subst_ctdpoly(quot, work_quot);
	subst_ctdpoly(rem, work_rem);

	free_ctdpoly(work_quot);
	free_ctdpoly(work_rem);
	free_ctdpoly(work_pol);
}

// Derivative: ret = d(pol)/dx
//void derivative_ctdpoly(CTDPoly ret, CTDPoly pol)
void diff_ctdpoly(CTDPoly ret, CTDPoly pol)
{
	long int i;
	ctdfloat ctmp;
	tdfloat rtmp;

	if(pol->deg == 0)
	{
		set0_ctdpoly(ret);
		return;
	}

	for(i = 0; i < pol->deg; i++)
	{
		rtd_set_ui(rtmp.val, (unsigned long)(i + 1));
		rctd_mul_td(&ctmp, get_ctdpoly_i(pol, i + 1), rtmp.val);
		set_ctdpoly_i(ret, i, &ctmp);
	}

	setdegree_ctdpoly(ret);
}

// ------------------------------------
// Complex triple-double precision polynomial evaluation
// ------------------------------------
/* complex value of a(x) */
// Based on Horner method
void eval_ctdpoly_horner(ctdfloat *ret, CTDPoly a, ctdfloat *x)
{
	long int i;
	ctdfloat tmp;

	// ret = 0
	rctd_set_ui(ret, 0UL);

	// ret = a[deg]
	rctd_set(ret, get_ctdpoly_i(a, a->deg));

	for(i = a->deg - 1; i >= 0; i--)
	{
		// ret = ret * x
		rctd_mul(&tmp, ret, x);
		
		// ret = tmp + a[i]
		rctd_add(ret, &tmp, get_ctdpoly_i(a, i));
	}
}

/* value of a(x) */
// Based on Estrin method
void eval_ctdpoly_estrin(ctdfloat *ret, CTDPoly a, ctdfloat *x)
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
    ctdfloat in_x, tmp;
	ctdfloat *in_coef_old, *in_coef_new;
    long int in_degree, num_in_coef, i;

    if(a->deg == 0) {
        rctd_set(ret, get_ctdpoly_i(a, 0));
		return;
    }
    if(a->deg == 1) {
        // ret = a[1] * x + a[0]
		rctd_mul(&tmp, get_ctdpoly_i(a, 1), x);
		rctd_add(ret, &tmp, get_ctdpoly_i(a, 0));
		return;
    }

    num_in_coef = a->deg + 1;
    if((num_in_coef % 2) == 1) num_in_coef++;

    in_degree = num_in_coef - 1;
    in_coef_old = (ctdfloat *)calloc(num_in_coef, sizeof(ctdfloat));
    in_coef_new = (ctdfloat *)calloc(num_in_coef, sizeof(ctdfloat));

    for(i = 0; i <= a->deg; i++) 
        rctd_set(&in_coef_old[i], get_ctdpoly_i(a, i));
    for(i = a->deg + 1; i <= in_degree; i++) 
        rctd_set_ui(&in_coef_old[i], 0UL);

    rctd_set(&in_x, x);

    while(in_degree > 1)
    {
		num_in_coef /= 2;
        for(i = 0; i < num_in_coef; i++)
        {
            // tmp = in_coef_old[i * 2 + 1] * in_x
			rctd_mul(&tmp, &in_coef_old[i * 2 + 1], &in_x);
            
            // in_coef_new[i] = tmp + in_coef_old[i * 2]
			rctd_add(&in_coef_new[i], &tmp, &in_coef_old[i * 2]);
        }

        // in_x = in_x * in_x
		rctd_mul(&in_x, &in_x, &in_x);

        for(i = 0; i < num_in_coef; i++)
			rctd_set(&in_coef_old[i], &in_coef_new[i]);

        if((num_in_coef % 2) == 1)
		{
 			num_in_coef += 1;
			rctd_set_ui(&in_coef_old[num_in_coef - 1], 0UL);
		}

        in_degree = num_in_coef - 1;
    }
    
    // ret = in_coef_new[1] * in_x + in_coef_new[0]
	rctd_mul(&tmp, &in_coef_new[1], &in_x);
	rctd_add(ret, &tmp, &in_coef_new[0]);

    free(in_coef_old);
    free(in_coef_new);

	return;
}

/* complex value of a'(x) (derivative) */
// Based on Horner method
void eval_diff_ctdpoly(ctdfloat *ret, CTDPoly a, ctdfloat *x)
{
	long int i;
	ctdfloat tmp;

	// ret = 0
	rctd_set_ui(ret, 0UL);

	// ret = a[deg] * deg
	rctd_set(ret, get_ctdpoly_i(a, a->deg));
	rctd_mul_ui(ret, ret, (unsigned long)a->deg);

	for(i = a->deg - 1; i >= 1; i--)
	{
		// ret = ret * x
		rctd_mul(ret, ret, x);
		
		// tmp = a[i] * i
		rctd_mul_ui(&tmp, get_ctdpoly_i(a, i), (unsigned long)i);
		
		// ret = ret + tmp
		rctd_add(ret, ret, &tmp);
	}
}

//------------
// AVX2
//------------
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
void _bncavx2_eval_ctdpoly_horner(__m256d ret_re[TDSIZE], __m256d ret_im[TDSIZE], CTDPoly a, __m256d x_re[TDSIZE], __m256d x_im[TDSIZE])
{
	long int i;
	__m256d tmp_re[TDSIZE], tmp_im[TDSIZE], a_i_re[TDSIZE], a_i_im[TDSIZE];

	_bncavx2_rctd_set0(ret_re, ret_im);

	_bncavx2_rtd_set1_td(ret_re, get_ctdpoly_i(a, a->deg)->val_re);
	_bncavx2_rtd_set1_td(ret_im, get_ctdpoly_i(a, a->deg)->val_im);

	for(i = a->deg - 1; i >= 0; i--)
	{
		_bncavx2_rtd_set1_td(a_i_re, get_ctdpoly_i(a, i)->val_re);
		_bncavx2_rtd_set1_td(a_i_im, get_ctdpoly_i(a, i)->val_im);

		_bncavx2_rctd_mul(tmp_re, tmp_im, ret_re, ret_im, x_re, x_im);

		_bncavx2_rctd_add(ret_re, ret_im, tmp_re, tmp_im, a_i_re, a_i_im);
	}
}
#elif defined(__AVX512F__) // __AVX512F__
void _bncavx512_eval_ctdpoly_horner(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE], CTDPoly a, __m512d x_re[TDSIZE], __m512d x_im[TDSIZE])
{
	long int i;
	__m512d tmp_re[TDSIZE], tmp_im[TDSIZE], a_i_re[TDSIZE], a_i_im[TDSIZE];

	_bncavx512_rctd_set0(ret_re, ret_im);

	_bncavx512_rtd_set1_td(ret_re, get_ctdpoly_i(a, a->deg)->val_re);
	_bncavx512_rtd_set1_td(ret_im, get_ctdpoly_i(a, a->deg)->val_im);

	for(i = a->deg - 1; i >= 0; i--)
	{
		_bncavx512_rtd_set1_td(a_i_re, get_ctdpoly_i(a, i)->val_re);
		_bncavx512_rtd_set1_td(a_i_im, get_ctdpoly_i(a, i)->val_im);

		_bncavx512_rctd_mul(tmp_re, tmp_im, ret_re, ret_im, x_re, x_im);

		_bncavx512_rctd_add(ret_re, ret_im, tmp_re, tmp_im, a_i_re, a_i_im);
	}
}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // ARM NEON
void _bncneon_eval_ctdpoly_horner(float64x2_t ret_re[TDSIZE], float64x2_t ret_im[TDSIZE], CTDPoly a, float64x2_t x_re[TDSIZE], float64x2_t x_im[TDSIZE])
{
	long int i;
	float64x2_t tmp_re[TDSIZE], tmp_im[TDSIZE], a_i_re[TDSIZE], a_i_im[TDSIZE];

	_bncneon_rctd_set0(ret_re, ret_im);

	_bncneon_rtd_set1_td(ret_re, get_ctdpoly_i(a, a->deg)->val_re);
	_bncneon_rtd_set1_td(ret_im, get_ctdpoly_i(a, a->deg)->val_im);

	for(i = a->deg - 1; i >= 0; i--)
	{
		_bncneon_rtd_set1_td(a_i_re, get_ctdpoly_i(a, i)->val_re);
		_bncneon_rtd_set1_td(a_i_im, get_ctdpoly_i(a, i)->val_im);

		_bncneon_rctd_mul(tmp_re, tmp_im, ret_re, ret_im, x_re, x_im);

		_bncneon_rctd_add(ret_re, ret_im, tmp_re, tmp_im, a_i_re, a_i_im);
	}
}
#endif // __AVX2__
void _bncavx2_eval_ctdpoly_estrin(ctdfloat *ret, CTDPoly a, ctdfloat *x)
{
    TDVector in_coef_old_real, in_coef_old_imag;
    ctdfloat in_x;
    ctdfloat in_ret, ctmp;
    long int in_degree, num_in_coef, i, num_loop_unit;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d old_coef4_real[2][TDSIZE], old_coef4_imag[2][TDSIZE];
    __m256d new_coef4_real[TDSIZE], new_coef4_imag[TDSIZE];
    __m256d a04_real[TDSIZE], a04_imag[TDSIZE], a14_real[TDSIZE], a14_imag[TDSIZE], x4_real[TDSIZE], x4_imag[TDSIZE], zero4[TDSIZE];
    __m256d ctmp4_real[TDSIZE], ctmp4_imag[TDSIZE];

    if((a->deg + 1) <= _BNC_D_WIDTH)
    {
        eval_ctdpoly_horner(ret, a, x);
        return;
    }

    _bncavx2_rtd_set0(zero4);
    num_loop_unit = 2 * _BNC_D_WIDTH;

#elif defined(__AVX512F__) // __AVX512F__
	__m512d old_coef8_real[2][TDSIZE], old_coef8_imag[2][TDSIZE];
    __m512d new_coef8_real[TDSIZE], new_coef8_imag[TDSIZE];
    __m512d a08_real[TDSIZE], a08_imag[TDSIZE], a18_real[TDSIZE], a18_imag[TDSIZE];
    __m512d x8_real[TDSIZE], x8_imag[TDSIZE], zero8[TDSIZE];
    __m512d ctmp8_real[TDSIZE], ctmp8_imag[TDSIZE];

    if((a->deg + 1) <= (2 * _BNC_D_WIDTH))
    {
        eval_ctdpoly_horner(ret, a, x);
        return;
    }

    _bncavx512_rtd_set0(zero8);
    num_loop_unit = 2 * (2 * _BNC_D_WIDTH);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, 2-lane, derived from NEON)
	
	svfloat64_t old_coef2_real_0_0, old_coef2_real_0_1, old_coef2_real_0_2, old_coef2_real_1_0, old_coef2_real_1_1, old_coef2_real_1_2;
	svfloat64_t old_coef2_imag_0_0, old_coef2_imag_0_1, old_coef2_imag_0_2, old_coef2_imag_1_0, old_coef2_imag_1_1, old_coef2_imag_1_2;
    
	svfloat64_t new_coef2_real_0, new_coef2_real_1, new_coef2_real_2;
	svfloat64_t new_coef2_imag_0, new_coef2_imag_1, new_coef2_imag_2;
    
	svfloat64_t a02_real_0, a02_real_1, a02_real_2;
	svfloat64_t a02_imag_0, a02_imag_1, a02_imag_2;
	svfloat64_t a12_real_0, a12_real_1, a12_real_2;
	svfloat64_t a12_imag_0, a12_imag_1, a12_imag_2;
    
	svfloat64_t x2_real_0, x2_real_1, x2_real_2;
	svfloat64_t x2_imag_0, x2_imag_1, x2_imag_2;
	svfloat64_t zero2_0, zero2_1, zero2_2;
    
	svfloat64_t ctmp2_real_0, ctmp2_real_1, ctmp2_real_2;
	svfloat64_t ctmp2_imag_0, ctmp2_imag_1, ctmp2_imag_2;

    if((a->deg + 1) <= 2)
    {
        eval_ctdpoly_horner(ret, a, x);
        return;
    }

    _bncsve2_rtd_set0(&zero2_0, &zero2_1, &zero2_2);
    num_loop_unit = 2 * 2;

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // ARM NEON
	float64x2_t old_coef2_real[2][TDSIZE], old_coef2_imag[2][TDSIZE];
    float64x2_t new_coef2_real[TDSIZE], new_coef2_imag[TDSIZE];
    float64x2_t a02_real[TDSIZE], a02_imag[TDSIZE], a12_real[TDSIZE], a12_imag[TDSIZE];
    float64x2_t x2_real[TDSIZE], x2_imag[TDSIZE], zero2[TDSIZE];
    float64x2_t ctmp2_real[TDSIZE], ctmp2_imag[TDSIZE];

    if((a->deg + 1) <= 2)
    {
        eval_ctdpoly_horner(ret, a, x);
        return;
    }

    _bncneon_rtd_set0(zero2);
    num_loop_unit = 2 * 2;

#else // __AVX2__
    eval_ctdpoly_estrin(ret, a, x);
    return;
#endif // __AVX2__

    num_in_coef = a->deg + 1;
    if((num_in_coef % num_loop_unit) != 0) 
        num_in_coef = (long int)ceil(((double)num_in_coef / (double)num_loop_unit)) * num_loop_unit;

    in_degree = num_in_coef - 1;
    in_coef_old_real = init_tdvector(num_in_coef);
    in_coef_old_imag = init_tdvector(num_in_coef);

    for(i = 0; i <= a->deg; i++)
    {
        set_tdvector_i(in_coef_old_real, i, get_ctdpoly_i(a, i)->val_re);
        set_tdvector_i(in_coef_old_imag, i, get_ctdpoly_i(a, i)->val_im);
    }

    rctd_set(&in_x, x);

    while(1)
    {
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
        _bncavx2_rtd_set1_td(x4_real, in_x.val_re);
        _bncavx2_rtd_set1_td(x4_imag, in_x.val_im);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            old_coef4_real[0][0] = _mm256_load_pd(&(in_coef_old_real->element[0][i * num_loop_unit]));
            old_coef4_real[1][0] = _mm256_load_pd(&(in_coef_old_real->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef4_real[0][1] = _mm256_load_pd(&(in_coef_old_real->element[1][i * num_loop_unit]));
            old_coef4_real[1][1] = _mm256_load_pd(&(in_coef_old_real->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef4_real[0][2] = _mm256_load_pd(&(in_coef_old_real->element[2][i * num_loop_unit]));
            old_coef4_real[1][2] = _mm256_load_pd(&(in_coef_old_real->element[2][i * num_loop_unit + _BNC_D_WIDTH]));

            old_coef4_imag[0][0] = _mm256_load_pd(&(in_coef_old_imag->element[0][i * num_loop_unit]));
            old_coef4_imag[1][0] = _mm256_load_pd(&(in_coef_old_imag->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef4_imag[0][1] = _mm256_load_pd(&(in_coef_old_imag->element[1][i * num_loop_unit]));
            old_coef4_imag[1][1] = _mm256_load_pd(&(in_coef_old_imag->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef4_imag[0][2] = _mm256_load_pd(&(in_coef_old_imag->element[2][i * num_loop_unit]));
            old_coef4_imag[1][2] = _mm256_load_pd(&(in_coef_old_imag->element[2][i * num_loop_unit + _BNC_D_WIDTH]));

            a04_real[0] = _mm256_unpacklo_pd(old_coef4_real[0][0], old_coef4_real[1][0]);
            a14_real[0] = _mm256_unpackhi_pd(old_coef4_real[0][0], old_coef4_real[1][0]);
            a04_real[1] = _mm256_unpacklo_pd(old_coef4_real[0][1], old_coef4_real[1][1]);
            a14_real[1] = _mm256_unpackhi_pd(old_coef4_real[0][1], old_coef4_real[1][1]);
            a04_real[2] = _mm256_unpacklo_pd(old_coef4_real[0][2], old_coef4_real[1][2]);
            a14_real[2] = _mm256_unpackhi_pd(old_coef4_real[0][2], old_coef4_real[1][2]);

            a04_imag[0] = _mm256_unpacklo_pd(old_coef4_imag[0][0], old_coef4_imag[1][0]);
            a14_imag[0] = _mm256_unpackhi_pd(old_coef4_imag[0][0], old_coef4_imag[1][0]);
            a04_imag[1] = _mm256_unpacklo_pd(old_coef4_imag[0][1], old_coef4_imag[1][1]);
            a14_imag[1] = _mm256_unpackhi_pd(old_coef4_imag[0][1], old_coef4_imag[1][1]);
            a04_imag[2] = _mm256_unpacklo_pd(old_coef4_imag[0][2], old_coef4_imag[1][2]);
            a14_imag[2] = _mm256_unpackhi_pd(old_coef4_imag[0][2], old_coef4_imag[1][2]);

            _bncavx2_rctd_mul(ctmp4_real, ctmp4_imag, a14_real, a14_imag, x4_real, x4_imag);
            _bncavx2_rctd_add(new_coef4_real, new_coef4_imag, ctmp4_real, ctmp4_imag, a04_real, a04_imag);

            new_coef4_real[0] = _mm256_permute4x64_pd(new_coef4_real[0], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_imag[0] = _mm256_permute4x64_pd(new_coef4_imag[0], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_real[1] = _mm256_permute4x64_pd(new_coef4_real[1], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_imag[1] = _mm256_permute4x64_pd(new_coef4_imag[1], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_real[2] = _mm256_permute4x64_pd(new_coef4_real[2], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_imag[2] = _mm256_permute4x64_pd(new_coef4_imag[2], (int)(3*64 + 1*16 + 2*4 + 0));

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

            _mm256_store_pd(&(in_coef_old_real->element[0][i * _BNC_D_WIDTH]), new_coef4_real[0]);
            _mm256_store_pd(&(in_coef_old_imag->element[0][i * _BNC_D_WIDTH]), new_coef4_imag[0]);
            _mm256_store_pd(&(in_coef_old_real->element[1][i * _BNC_D_WIDTH]), new_coef4_real[1]);
            _mm256_store_pd(&(in_coef_old_imag->element[1][i * _BNC_D_WIDTH]), new_coef4_imag[1]);
            _mm256_store_pd(&(in_coef_old_real->element[2][i * _BNC_D_WIDTH]), new_coef4_real[2]);
            _mm256_store_pd(&(in_coef_old_imag->element[2][i * _BNC_D_WIDTH]), new_coef4_imag[2]);
        }
#elif defined(__AVX512F__) // __AVX512F__
        _bncavx512_rtd_set1_td(x8_real, in_x.val_re);
        _bncavx512_rtd_set1_td(x8_imag, in_x.val_im);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            old_coef8_real[0][0] = _mm512_load_pd(&(in_coef_old_real->element[0][i * num_loop_unit]));
            old_coef8_real[1][0] = _mm512_load_pd(&(in_coef_old_real->element[0][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_real[0][1] = _mm512_load_pd(&(in_coef_old_real->element[1][i * num_loop_unit]));
            old_coef8_real[1][1] = _mm512_load_pd(&(in_coef_old_real->element[1][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_real[0][2] = _mm512_load_pd(&(in_coef_old_real->element[2][i * num_loop_unit]));
            old_coef8_real[1][2] = _mm512_load_pd(&(in_coef_old_real->element[2][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));

            old_coef8_imag[0][0] = _mm512_load_pd(&(in_coef_old_imag->element[0][i * num_loop_unit]));
            old_coef8_imag[1][0] = _mm512_load_pd(&(in_coef_old_imag->element[0][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_imag[0][1] = _mm512_load_pd(&(in_coef_old_imag->element[1][i * num_loop_unit]));
            old_coef8_imag[1][1] = _mm512_load_pd(&(in_coef_old_imag->element[1][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_imag[0][2] = _mm512_load_pd(&(in_coef_old_imag->element[2][i * num_loop_unit]));
            old_coef8_imag[1][2] = _mm512_load_pd(&(in_coef_old_imag->element[2][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            
            a08_real[0] = _mm512_unpacklo_pd(old_coef8_real[0][0], old_coef8_real[1][0]);
            a18_real[0] = _mm512_unpackhi_pd(old_coef8_real[0][0], old_coef8_real[1][0]);
            a08_real[1] = _mm512_unpacklo_pd(old_coef8_real[0][1], old_coef8_real[1][1]);
            a18_real[1] = _mm512_unpackhi_pd(old_coef8_real[0][1], old_coef8_real[1][1]);
            a08_real[2] = _mm512_unpacklo_pd(old_coef8_real[0][2], old_coef8_real[1][2]);
            a18_real[2] = _mm512_unpackhi_pd(old_coef8_real[0][2], old_coef8_real[1][2]);

            a08_imag[0] = _mm512_unpacklo_pd(old_coef8_imag[0][0], old_coef8_imag[1][0]);
            a18_imag[0] = _mm512_unpackhi_pd(old_coef8_imag[0][0], old_coef8_imag[1][0]);
            a08_imag[1] = _mm512_unpacklo_pd(old_coef8_imag[0][1], old_coef8_imag[1][1]);
            a18_imag[1] = _mm512_unpackhi_pd(old_coef8_imag[0][1], old_coef8_imag[1][1]);
            a08_imag[2] = _mm512_unpacklo_pd(old_coef8_imag[0][2], old_coef8_imag[1][2]);
            a18_imag[2] = _mm512_unpackhi_pd(old_coef8_imag[0][2], old_coef8_imag[1][2]);

            _bncavx512_rctd_mul(ctmp8_real, ctmp8_imag, a18_real, a18_imag, x8_real, x8_imag);
            _bncavx512_rctd_add(new_coef8_real, new_coef8_imag, ctmp8_real, ctmp8_imag, a08_real, a08_imag);
            
            new_coef8_real[0] = _mm512_permutex_pd(new_coef8_real[0], 0xD8);
            new_coef8_imag[0] = _mm512_permutex_pd(new_coef8_imag[0], 0xD8);
            new_coef8_real[1] = _mm512_permutex_pd(new_coef8_real[1], 0xD8);
            new_coef8_imag[1] = _mm512_permutex_pd(new_coef8_imag[1], 0xD8);
            new_coef8_real[2] = _mm512_permutex_pd(new_coef8_real[2], 0xD8);
            new_coef8_imag[2] = _mm512_permutex_pd(new_coef8_imag[2], 0xD8);
            
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

            _mm512_store_pd(&(in_coef_old_real->element[0][i * (2 * _BNC_D_WIDTH)]), new_coef8_real[0]);
            _mm512_store_pd(&(in_coef_old_imag->element[0][i * (2 * _BNC_D_WIDTH)]), new_coef8_imag[0]);
            _mm512_store_pd(&(in_coef_old_real->element[1][i * (2 * _BNC_D_WIDTH)]), new_coef8_real[1]);
            _mm512_store_pd(&(in_coef_old_imag->element[1][i * (2 * _BNC_D_WIDTH)]), new_coef8_imag[1]);
            _mm512_store_pd(&(in_coef_old_real->element[2][i * (2 * _BNC_D_WIDTH)]), new_coef8_real[2]);
            _mm512_store_pd(&(in_coef_old_imag->element[2][i * (2 * _BNC_D_WIDTH)]), new_coef8_imag[2]);
        }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, 2-lane, derived from NEON)

	svbool_t pg = svwhilelt_b64_s64((int64_t)0, (int64_t)2);        x2_real_0 = svdup_f64((in_x.val_re)[0]); x2_real_1 = svdup_f64((in_x.val_re)[1]); x2_real_2 = svdup_f64((in_x.val_re)[2]);
        x2_imag_0 = svdup_f64((in_x.val_im)[0]); x2_imag_1 = svdup_f64((in_x.val_im)[1]); x2_imag_2 = svdup_f64((in_x.val_im)[2]);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            old_coef2_real_0_0 = svld1_f64(pg, &(in_coef_old_real->element[0][i * num_loop_unit]));
            old_coef2_real_1_0 = svld1_f64(pg, &(in_coef_old_real->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_real_0_1 = svld1_f64(pg, &(in_coef_old_real->element[1][i * num_loop_unit]));
            old_coef2_real_1_1 = svld1_f64(pg, &(in_coef_old_real->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_real_0_2 = svld1_f64(pg, &(in_coef_old_real->element[2][i * num_loop_unit]));
            old_coef2_real_1_2 = svld1_f64(pg, &(in_coef_old_real->element[2][i * num_loop_unit + _BNC_D_WIDTH]));

            old_coef2_imag_0_0 = svld1_f64(pg, &(in_coef_old_imag->element[0][i * num_loop_unit]));
            old_coef2_imag_1_0 = svld1_f64(pg, &(in_coef_old_imag->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_imag_0_1 = svld1_f64(pg, &(in_coef_old_imag->element[1][i * num_loop_unit]));
            old_coef2_imag_1_1 = svld1_f64(pg, &(in_coef_old_imag->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_imag_0_2 = svld1_f64(pg, &(in_coef_old_imag->element[2][i * num_loop_unit]));
            old_coef2_imag_1_2 = svld1_f64(pg, &(in_coef_old_imag->element[2][i * num_loop_unit + _BNC_D_WIDTH]));
            
            a02_real_0 = svtrn1_f64(old_coef2_real_0_0, old_coef2_real_1_0);
            a12_real_0 = svtrn2_f64(old_coef2_real_0_0, old_coef2_real_1_0);
            a02_real_1 = svtrn1_f64(old_coef2_real_0_1, old_coef2_real_1_1);
            a12_real_1 = svtrn2_f64(old_coef2_real_0_1, old_coef2_real_1_1);
            a02_real_2 = svtrn1_f64(old_coef2_real_0_2, old_coef2_real_1_2);
            a12_real_2 = svtrn2_f64(old_coef2_real_0_2, old_coef2_real_1_2);

            a02_imag_0 = svtrn1_f64(old_coef2_imag_0_0, old_coef2_imag_1_0);
            a12_imag_0 = svtrn2_f64(old_coef2_imag_0_0, old_coef2_imag_1_0);
            a02_imag_1 = svtrn1_f64(old_coef2_imag_0_1, old_coef2_imag_1_1);
            a12_imag_1 = svtrn2_f64(old_coef2_imag_0_1, old_coef2_imag_1_1);
            a02_imag_2 = svtrn1_f64(old_coef2_imag_0_2, old_coef2_imag_1_2);
            a12_imag_2 = svtrn2_f64(old_coef2_imag_0_2, old_coef2_imag_1_2);

            _bncsve2_rctd_mul(pg, &ctmp2_real_0, &ctmp2_real_1, &ctmp2_real_2, &ctmp2_imag_0, &ctmp2_imag_1, &ctmp2_imag_2, a12_real_0, a12_real_1, a12_real_2, a12_imag_0, a12_imag_1, a12_imag_2, x2_real_0, x2_real_1, x2_real_2, x2_imag_0, x2_imag_1, x2_imag_2);
            _bncsve2_rctd_add(pg, &new_coef2_real_0, &new_coef2_real_1, &new_coef2_real_2, &new_coef2_imag_0, &new_coef2_imag_1, &new_coef2_imag_2, ctmp2_real_0, ctmp2_real_1, ctmp2_real_2, ctmp2_imag_0, ctmp2_imag_1, ctmp2_imag_2, a02_real_0, a02_real_1, a02_real_2, a02_imag_0, a02_imag_1, a02_imag_2);
                        
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

            svst1_f64(pg, &(in_coef_old_real->element[0][i * _BNC_D_WIDTH]), new_coef2_real_0);
            svst1_f64(pg, &(in_coef_old_imag->element[0][i * _BNC_D_WIDTH]), new_coef2_imag_0);
            svst1_f64(pg, &(in_coef_old_real->element[1][i * _BNC_D_WIDTH]), new_coef2_real_1);
            svst1_f64(pg, &(in_coef_old_imag->element[1][i * _BNC_D_WIDTH]), new_coef2_imag_1);
            svst1_f64(pg, &(in_coef_old_real->element[2][i * _BNC_D_WIDTH]), new_coef2_real_2);
            svst1_f64(pg, &(in_coef_old_imag->element[2][i * _BNC_D_WIDTH]), new_coef2_imag_2);
        }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // ARM NEON
        _bncneon_rtd_set1_td(x2_real, in_x.val_re);
        _bncneon_rtd_set1_td(x2_imag, in_x.val_im);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            old_coef2_real[0][0] = vld1q_f64(&(in_coef_old_real->element[0][i * num_loop_unit]));
            old_coef2_real[1][0] = vld1q_f64(&(in_coef_old_real->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_real[0][1] = vld1q_f64(&(in_coef_old_real->element[1][i * num_loop_unit]));
            old_coef2_real[1][1] = vld1q_f64(&(in_coef_old_real->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_real[0][2] = vld1q_f64(&(in_coef_old_real->element[2][i * num_loop_unit]));
            old_coef2_real[1][2] = vld1q_f64(&(in_coef_old_real->element[2][i * num_loop_unit + _BNC_D_WIDTH]));

            old_coef2_imag[0][0] = vld1q_f64(&(in_coef_old_imag->element[0][i * num_loop_unit]));
            old_coef2_imag[1][0] = vld1q_f64(&(in_coef_old_imag->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_imag[0][1] = vld1q_f64(&(in_coef_old_imag->element[1][i * num_loop_unit]));
            old_coef2_imag[1][1] = vld1q_f64(&(in_coef_old_imag->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_imag[0][2] = vld1q_f64(&(in_coef_old_imag->element[2][i * num_loop_unit]));
            old_coef2_imag[1][2] = vld1q_f64(&(in_coef_old_imag->element[2][i * num_loop_unit + _BNC_D_WIDTH]));
            
            a02_real[0] = vzip1q_f64(old_coef2_real[0][0], old_coef2_real[1][0]);
            a12_real[0] = vzip2q_f64(old_coef2_real[0][0], old_coef2_real[1][0]);
            a02_real[1] = vzip1q_f64(old_coef2_real[0][1], old_coef2_real[1][1]);
            a12_real[1] = vzip2q_f64(old_coef2_real[0][1], old_coef2_real[1][1]);
            a02_real[2] = vzip1q_f64(old_coef2_real[0][2], old_coef2_real[1][2]);
            a12_real[2] = vzip2q_f64(old_coef2_real[0][2], old_coef2_real[1][2]);

            a02_imag[0] = vzip1q_f64(old_coef2_imag[0][0], old_coef2_imag[1][0]);
            a12_imag[0] = vzip2q_f64(old_coef2_imag[0][0], old_coef2_imag[1][0]);
            a02_imag[1] = vzip1q_f64(old_coef2_imag[0][1], old_coef2_imag[1][1]);
            a12_imag[1] = vzip2q_f64(old_coef2_imag[0][1], old_coef2_imag[1][1]);
            a02_imag[2] = vzip1q_f64(old_coef2_imag[0][2], old_coef2_imag[1][2]);
            a12_imag[2] = vzip2q_f64(old_coef2_imag[0][2], old_coef2_imag[1][2]);

            _bncneon_rctd_mul(ctmp2_real, ctmp2_imag, a12_real, a12_imag, x2_real, x2_imag);
            _bncneon_rctd_add(new_coef2_real, new_coef2_imag, ctmp2_real, ctmp2_imag, a02_real, a02_imag);
                        
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

            vst1q_f64(&(in_coef_old_real->element[0][i * _BNC_D_WIDTH]), new_coef2_real[0]);
            vst1q_f64(&(in_coef_old_imag->element[0][i * _BNC_D_WIDTH]), new_coef2_imag[0]);
            vst1q_f64(&(in_coef_old_real->element[1][i * _BNC_D_WIDTH]), new_coef2_real[1]);
            vst1q_f64(&(in_coef_old_imag->element[1][i * _BNC_D_WIDTH]), new_coef2_imag[1]);
            vst1q_f64(&(in_coef_old_real->element[2][i * _BNC_D_WIDTH]), new_coef2_real[2]);
            vst1q_f64(&(in_coef_old_imag->element[2][i * _BNC_D_WIDTH]), new_coef2_imag[2]);
        }
#endif // __AVX2__

        rctd_mul(&ctmp, &in_x, &in_x);
        rctd_set(&in_x, &ctmp);
        if(num_in_coef == num_loop_unit) break;
        num_in_coef /= 2;
        if((num_in_coef % num_loop_unit) != 0)
            num_in_coef = (long int)ceil(((double)num_in_coef / (double)num_loop_unit)) * num_loop_unit;

        in_degree = num_in_coef - 1;
    }

#if defined(__AVX2__) || defined(__AVX512F__) // __AVX2__
    rtd_set(in_ret.val_re, get_tdvector_i(in_coef_old_real, _BNC_D_WIDTH - 1));
    rtd_set(in_ret.val_im, get_tdvector_i(in_coef_old_imag, _BNC_D_WIDTH - 1));
    for(i = _BNC_D_WIDTH - 2; i >= 0; i--)
    {
        rctd_mul(&ctmp, &in_ret, &in_x);
        rtd_set(in_ret.val_re, get_tdvector_i(in_coef_old_real, i));
        rtd_set(in_ret.val_im, get_tdvector_i(in_coef_old_imag, i));
        rctd_add(&in_ret, &in_ret, &ctmp);
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    rtd_set(in_ret.val_re, get_tdvector_i(in_coef_old_real, _BNC_D_WIDTH - 1));
    rtd_set(in_ret.val_im, get_tdvector_i(in_coef_old_imag, _BNC_D_WIDTH - 1));
    for(i = _BNC_D_WIDTH - 2; i >= 0; i--)
    {
        rctd_mul(&ctmp, &in_ret, &in_x);
        rtd_set(in_ret.val_re, get_tdvector_i(in_coef_old_real, i));
        rtd_set(in_ret.val_im, get_tdvector_i(in_coef_old_imag, i));
        rctd_add(&in_ret, &in_ret, &ctmp);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // ARM NEON
    rtd_set(in_ret.val_re, get_tdvector_i(in_coef_old_real, _BNC_D_WIDTH - 1));
    rtd_set(in_ret.val_im, get_tdvector_i(in_coef_old_imag, _BNC_D_WIDTH - 1));
    for(i = _BNC_D_WIDTH - 2; i >= 0; i--)
    {
        rctd_mul(&ctmp, &in_ret, &in_x);
        rtd_set(in_ret.val_re, get_tdvector_i(in_coef_old_real, i));
        rtd_set(in_ret.val_im, get_tdvector_i(in_coef_old_imag, i));
        rctd_add(&in_ret, &in_ret, &ctmp);
    }
#endif // __AVX2__

    rctd_set(ret, &in_ret);

    free_tdvector(in_coef_old_real);
    free_tdvector(in_coef_old_imag);

    return;
}
#ifdef __cpluplus
} // extern "C" {
#endif // __cplusplus
