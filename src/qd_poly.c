/********************************************************************************/
/* qd_poly.c: Algebraic Equations and Polynomials                               */
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
/*   QDPoly init_qdpoly(long int max_length)   */
/*   QDPoly init2_qdpoly(long int max_length, unsigned long prec)*/
/* Free:                                         */
/*   void free_qdpoly(QDPoly pol)              */
/* Get & Set Values:                             */
/*   qdfloat *get_qdpoly_i(QDPoly pol, long int index) */
/*   long int setdegree_qdpoly(QDPoly)         */
/*   void set_qdpoly_i(QDPoly pol, long int index, double val[QDSIZE]) */
/*   void set_qdpoly_i_d(QDPoly pol, long int index, double val[QDSIZE]) */
/* Output:                                       */
/*   void print_qdpoly(QDPoly pol)             */
/*************************************************/
QDPoly init_qdpoly(long int max_length)
{
	QDPoly ret = NULL;
	long int i;

	if(max_length <= 0)
	{
		fprintf(stderr, "ERROR: init_qdpoly\n");
		return ret;
	}

	ret = (QDPoly)malloc(sizeof(qdpoly));
	if(ret == NULL)
		return ret;

	ret->coef = (qdfloat *)calloc(sizeof(qdfloat), max_length);
	if(ret->coef == NULL)
		return NULL;

	/* All 0 */
	for(i = 0; i < max_length; i++)
		rqd_set_ui(ret->coef[i].val, 0UL);

    // zero := 0
    rqd_set0(ret->zero.val);

	ret->deg = 0;
	ret->max_len = max_length;

	return ret;
}

void free_qdpoly(QDPoly pol)
{
	long int i;

	if(pol == NULL)
		return;

	if(pol->coef != NULL)
		free(pol->coef); // Fix! 2012-06-03 by T.Kouya

	free(pol);
}

qdfloat get_qdpoly_i_float(QDPoly pol, long int index)
{
	if(index > pol->deg)
		return pol->zero;
	else
		return pol->coef[index];
}

double *get_qdpoly_i(QDPoly pol, long int index)
{
	if(index > pol->deg)
		return pol->zero.val;
	else
		return pol->coef[index].val;
}

/* find maximum degree at pol->coef[deg] != 0 */
long int setdegree_qdpoly(QDPoly pol)
{
	long int i;

	for(i = pol->max_len - 1; i > 0; i--)
	{
		if(rqd_cmp(get_qdpoly_i(pol, i), pol->zero.val) != 0)
		{
			pol->deg = i;
			return i;
		}
	}

	pol->deg = 0;
	return 0;
}

void set_qdpoly_i(QDPoly pol, long int index, double val[QDSIZE])
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	rqd_set(pol->coef[index].val, val);
	if((pol->deg < index) && (rqd_cmp(val, pol->zero.val) != 0))
		pol->deg = index;
}

void set_qdpoly_i_si(QDPoly pol, long int index, long val)
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	rqd_set_d(pol->coef[index].val, (double)val);
	if((pol->deg < index) && (val != 0))
		pol->deg = index;
}

void set_qdpoly_i_ui(QDPoly pol, long int index, unsigned long val)
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	rqd_set_ui(pol->coef[index].val, val);
	if((pol->deg < index) && (val != 0))
		pol->deg = index;
}

void set_qdpoly_i_d(QDPoly pol, long int index, double val)
{
	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	rqd_set_d(pol->coef[index].val, val);
	if((pol->deg < index) && (val != 0))
		pol->deg = index;
}

void set_qdpoly_i_str(QDPoly pol, long int index, const char *str, int base)
{
	double tmp[QDSIZE];

	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	rqd_set_str(tmp, str);
	rqd_set(pol->coef[index].val, tmp);
	if((pol->deg < index) && (rqd_cmp(pol->zero.val, tmp) != 0))
		pol->deg = index;

}

#ifdef USE_GMP
void set_qdpoly_i_mpf(QDPoly pol, long int index, mpf_t val)
{
	double tmp[QDSIZE];

	if(index > pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	mpf_get_qd(tmp, val);
	rqd_set(pol->coef[index].val, tmp);
	if((pol->deg < index) && (rqd_cmp(pol->zero.val, tmp) != 0))
		pol->deg = index;

}

// Initialize and substitute polynomial from org_pol
QDPoly init_set_qdpoly_mpfpoly(MPFPoly org_pol)
{
	QDPoly ret = NULL;
	long int i;

	ret = init_qdpoly(org_pol->max_len);
	if(ret == NULL)
		return ret;

	for(i = 0; i <= org_pol->deg; i++)
		set_qdpoly_i_mpf(ret, i, get_mpfpoly_i(org_pol, i));

	return ret;
}
#endif // USE_GMP

void print_qdpoly(QDPoly pol)
{
	long int i;

	for(i = 0; i <= pol->deg; i++)
	{
		printf("%5ld ", i);
		rqd_out_str(get_qdpoly_i(pol, i));
		printf("\n");
	}
}

/*************************************************/
/* Poly Calculations for QDPoly             */
/*
void add_qdpoly(QDPoly c, QDPoly a, QDPoly b)
void sub_qdpoly(QDPoly c, QDPoly a, QDPoly b)
void cmul_qdpoly(QDPoly c, double val[QDSIZE], QDPoly a)
void subst_qdpoly(QDPoly c, QDPoly a)

void diff_qdpoly(QDPoly a)
void eval_qdpoly(double ret, QDPoly a, double x[QDSIZE])
void eval_diff_qdpoly(double ret, QDPoly a, double x[QDSIZE])
*/
/*************************************************/

/* c = a + b */
void add_qdpoly(QDPoly c, QDPoly a, QDPoly b)
{
	long int i;
	double tmp[QDSIZE];

	if((c->max_len < a->deg) || (c->max_len < b->deg))
	{
		fprintf(stderr, "ERROR: add_qdpoly\n");
		return;
	}

	c->deg = (a->deg > b->deg) ? a->deg : b->deg;

	for(i = 0; i <= c->deg; i++)
	{
		rqd_add(tmp, get_qdpoly_i(a, i), get_qdpoly_i(b, i));
		set_qdpoly_i(c, i, tmp);
	}
}

/* c += a */
void add2_qdpoly(QDPoly c, QDPoly a)
{
	long int i;
	double tmp[QDSIZE];

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: add2_qdpoly\n");
		return;
	}

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
	{
		rqd_add(tmp, get_qdpoly_i(c, i), get_qdpoly_i(a, i));
		set_qdpoly_i(c, i, tmp);
	}
}

/* c = a - b */
void sub_qdpoly(QDPoly c, QDPoly a, QDPoly b)
{
	long int i;
	double tmp[QDSIZE];

	if((c->max_len < a->deg) || (c->max_len < b->deg))
	{
		fprintf(stderr, "ERROR: sub_qdpoly\n");
		return;
	}

	c->deg = (a->deg > b->deg) ? a->deg : b->deg;

	for(i = 0; i <= c->deg; i++)
	{
		rqd_sub(tmp, get_qdpoly_i(a, i), get_qdpoly_i(b, i));
		set_qdpoly_i(c, i, tmp);
	}
}

/* c -= a */
void sub2_qdpoly(QDPoly c, QDPoly a)
{
	long int i, min_deg;
	double tmp[QDSIZE];

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: sub2_qdpoly\n");
		return;
	}

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
	{
		rqd_sub(tmp, get_qdpoly_i(c, i), get_qdpoly_i(a, i));
		set_qdpoly_i(c, i, tmp);
	}
}

/* c = a * b */
void mul_qdpoly(QDPoly c, QDPoly a, QDPoly b)
{
	long int i, j;
	double tmp[QDSIZE];

	if(c->max_len < (a->deg + b->deg))
	{
		fprintf(stderr, "ERROR: mul_qdpoly\n");
		return;
	}

	/* set c = 0 */
	set0_qdpoly(c);

	for(i = 0; i <= a->deg; i++)
	{
		for(j = 0; j <= b->deg; j++)
		{
			rqd_set(tmp, get_qdpoly_i(c, i + j));
			rqd_fma(tmp, get_qdpoly_i(a, i), get_qdpoly_i(b, j), tmp);
			set_qdpoly_i(c, i + j, tmp);
		}
	}

	c->deg = setdegree_qdpoly(c);
//	c->deg = a->deg + b->deg;
}

/* c = val * a */
void cmul_qdpoly(QDPoly c, double val[QDSIZE], QDPoly a)
{
	long int i;
	double tmp[QDSIZE];

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: cmul_qdpoly\n");
		return;
	}

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
	{
		rqd_mul(tmp, val, get_qdpoly_i(a, i));
		set_qdpoly_i(c, i, tmp);
	}
}

/* c *= val */
void cmul2_qdpoly(QDPoly c, double val[QDSIZE])
{
	long int i;
	double tmp[QDSIZE];

	for(i = 0; i <= c->deg; i++)
	{
		rqd_mul(tmp, val, get_qdpoly_i(c, i));
		set_qdpoly_i(c, i, tmp);
	}
}

/* c := a */
void subst_qdpoly(QDPoly c, QDPoly a)
{
	long int i;

	if(c->deg > a->deg)
	{
		for(i = c->deg; i >= a->deg; i--)
			set_qdpoly_i_ui(c, i, 0UL);
	}

	c->deg = a->deg;
	for(i = 0; i <= c->deg; i++)
		set_qdpoly_i(c, i, get_qdpoly_i(a, i));
}

/* c := 0 */
void set0_qdpoly(QDPoly c)
{
	long int i;
	double tmp[QDSIZE];

	rqd_set_ui(tmp, 0UL);

	for(i = 0; i <= c->deg; i++)
		set_qdpoly_i(c, i, tmp);
}

/* number of nonzero coef */
long int num_nonzero_qdpoly(QDPoly c)
{
	long int i, ret;

	ret = 0;
	for(i = 0; i <= c->deg; i++)
	{
		if(rqd_cmp_ui(get_qdpoly_i(c, i), 0UL) != 0)
			ret++;
	}

	return ret;
}

/* get maximum |coef| */
long int max_abscoef_qdpoly(QDPoly c)
{
	long int i, ret;
	double tmp1[QDSIZE], tmp[QDSIZE];

	rqd_abs(tmp1, get_qdpoly_i(c, 0));
	ret = 0;
	for(i = 1; i <= c->deg; i++)
	{
		rqd_abs(tmp, get_qdpoly_i(c, i));
		if(rqd_cmp(tmp1, tmp) < 0)
		{
			rqd_set(tmp1, tmp);
			ret = i;
		}
	}

	return ret;
}

/* a := a'(x) */
void diff_qdpoly(QDPoly a)
{
	long int diff_deg, i;
	double tmp_coef[QDSIZE];

	diff_deg = a->deg - 1;

	if(diff_deg < 0)
	{
		set0_qdpoly(a);
		a->deg = 0;
		return;
	}

	//for(i = 1; i <= diff_deg; i++)
	for(i = 1; i <= a->deg; i++) // Fix!! : 2007-01-11
	{
		rqd_mul_ui(tmp_coef, get_qdpoly_i(a, i), (unsigned long)i);
		set_qdpoly_i(a, i - 1, tmp_coef);
	}
	a->deg = diff_deg;
}

/* value of a(x) */
// Based on Horner method
void eval_qdpoly_horner(double ret[QDSIZE], QDPoly a, double x[QDSIZE])
{
	long int i;

	rqd_set(ret, get_qdpoly_i(a, a->deg));
	for(i = a->deg - 1; i >= 0; i--)
	{
		rqd_mul(ret, ret, x);
		rqd_add(ret, ret, get_qdpoly_i(a, i));
	}
}

/* value of a(x) */
// Based on Estrin method
void eval_qdpoly_estrin(double ret[QDSIZE], QDPoly a, double x[QDSIZE])
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
    double in_x[QDSIZE], tmp[QDSIZE];
	qdfloat *in_coef_old, *in_coef_new;
    long int in_degree, num_in_coef, i;

    if(a->deg == 0) {
        //return coef[0];
		rqd_set(ret, get_qdpoly_i(a, 0));
		return;
    }
    if(a->deg == 1) {
        //return coef[0] + coef[1] * x;
		rqd_mul(ret, get_qdpoly_i(a, 1), x);
		rqd_add(ret, ret, get_qdpoly_i(a, 0));
		return;
    }

    num_in_coef = a->deg + 1;
    if((num_in_coef % 2) == 1) num_in_coef++;

    in_degree = num_in_coef - 1;
    in_coef_old = (qdfloat *)calloc(num_in_coef, sizeof(qdfloat));
    in_coef_new = (qdfloat *)calloc(num_in_coef, sizeof(qdfloat));

    for(i = 0; i <= a->deg; i++) rqd_set(in_coef_old[i].val, get_qdpoly_i(a, i));
    for(i = a->deg + 1; i <= in_degree; i++) rqd_set_ui(in_coef_old[i].val, 0UL);

    rqd_set(in_x, x);

    //printf("degree -> in_degree: %d -> %d\n", degree, in_degree);

    while(in_degree > 1)
    {
		num_in_coef /= 2;
        for(i = 0; i < num_in_coef; i++)
        {
            //in_coef_new[i] = in_coef_old[i * 2] + in_coef_old[i * 2 + 1] * in_x;
			rqd_mul(tmp, in_coef_old[i * 2 + 1].val, in_x);
			rqd_add(tmp, tmp, in_coef_old[i * 2].val);
			rqd_set(in_coef_new[i].val, tmp);
            //printf("coef[%d] = in_coef[%d] + in_coef[%d] * %f = %f\n", i, i * 2, i * 2 + 1, in_x, in_coef_new[i]);
        }

        //return estrin(x * x, coef, num_in_coef / 2 - 1);
        //in_x = in_x * in_x;
		rqd_mul(in_x, in_x, in_x);
        for(i = 0; i < num_in_coef; i++)
			rqd_set(in_coef_old[i].val, in_coef_new[i].val);
           	//in_coef_old[i] = in_coef_new[i];

        if((num_in_coef % 2) == 1)
		{
			//in_coef_old[num_in_coef / 2 + 1] = 0; 
			num_in_coef += 1;
			rqd_set_ui(in_coef_old[num_in_coef - 1].val, 0UL);
		}

        in_degree = num_in_coef - 1;

    }
    //ret = in_coef_new[0] + in_coef_new[1] * in_x;
	rqd_mul(ret, in_coef_new[1].val, in_x);
	rqd_add(ret, ret, in_coef_new[0].val);

    free(in_coef_old);
    free(in_coef_new);

    //return ret;
	return;
}

/* value of a'(x) */
// Based on Horner method
void eval_diff_qdpoly(double ret[QDSIZE], QDPoly a, double x[QDSIZE])
{
	long int i;
	double tmp[QDSIZE];

	rqd_set(ret, get_qdpoly_i(a, a->deg));
	rqd_mul_ui(ret, ret, (unsigned long)a->deg);

	for(i = a->deg - 1; i >= 1; i--)
	{
		rqd_mul(ret, ret, x);
		rqd_mul_ui(tmp, get_qdpoly_i(a, i), (unsigned long)i);
		rqd_add(ret, ret, tmp);
	}

}

// ------------------------------------
// New implementation
// 2025-01-15 (Wed) T.Kouya
// ------------------------------------
/* complex value of a(x) */
//void ceval_qdpoly(MPFCmplx ret, QDPoly a, MPFCmplx x)
// Based on Horner method
void ceval_qdpoly_horner(cqdfloat *ret, QDPoly a, cqdfloat *x)
{
	long int i;

	//set0_mpfcmplx(ret);
	rcqd_set_ui(ret, 0UL);

	//set_real_mpfcmplx(ret, get_qdpoly_i(a, a->deg));
	rcqd_set_qd(ret, get_qdpoly_i(a, a->deg));
	for(i = a->deg - 1; i >= 0; i--)
	{
		//mul2_mpfcmplx(ret, x);
		rcqd_mul(ret, ret, x);
		//add_mpfcmplx_mpf(ret, ret, get_qdpoly_i(a, i));
		rcqd_add_qd(ret, ret, get_qdpoly_i(a, i));
	}
}

/* value of a(x) */
// Based on Estrin method
void ceval_qdpoly_estrin(cqdfloat *ret, QDPoly a, cqdfloat *x)
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
        //return coef[0];
		rcqd_set_qd(ret, get_qdpoly_i(a, 0));
		return;
    }
    if(a->deg == 1) {
        //return coef[0] + coef[1] * x;
		rcqd_mul_qd(ret, x, get_qdpoly_i(a, 1));
		rcqd_add_qd(ret, ret, get_qdpoly_i(a, 0));
		return;
    }

    num_in_coef = a->deg + 1;
    if((num_in_coef % 2) == 1) num_in_coef++;

    in_degree = num_in_coef - 1;
    in_coef_old = (cqdfloat *)calloc(num_in_coef, sizeof(cqdfloat));
    in_coef_new = (cqdfloat *)calloc(num_in_coef, sizeof(cqdfloat));

    for(i = 0; i <= a->deg; i++) rcqd_set_qd(&in_coef_old[i], get_qdpoly_i(a, i));
    for(i = a->deg + 1; i <= in_degree; i++) rcqd_set_ui(&in_coef_old[i], 0UL);

    rcqd_set(&in_x, x);

    //printf("degree -> in_degree: %d -> %d\n", degree, in_degree);

    while(in_degree > 1)
    {
		num_in_coef /= 2;
        for(i = 0; i < num_in_coef; i++)
        {
            //in_coef_new[i] = in_coef_old[i * 2] + in_coef_old[i * 2 + 1] * in_x;
			rcqd_mul(&tmp, &in_coef_old[i * 2 + 1], &in_x);
			rcqd_add(&tmp, &tmp, &in_coef_old[i * 2]);
			rcqd_set(&in_coef_new[i], &tmp);
            //printf("coef[%d] = in_coef[%d] + in_coef[%d] * %f = %f\n", i, i * 2, i * 2 + 1, in_x, in_coef_new[i]);
        }

        //return estrin(x * x, coef, num_in_coef / 2 - 1);
        //in_x = in_x * in_x;
		rcqd_mul(&in_x, &in_x, &in_x);
        for(i = 0; i < num_in_coef; i++)
			rcqd_set(&in_coef_old[i], &in_coef_new[i]);
           	//in_coef_old[i] = in_coef_new[i];

        if((num_in_coef % 2) == 1)
		{
 			num_in_coef += 1;
			rcqd_set_ui(&in_coef_old[num_in_coef - 1], 0UL);
		}

        in_degree = num_in_coef - 1;

    }
    //ret = in_coef_new[0] + in_coef_new[1] * in_x;
	rcqd_mul(ret, &in_coef_new[1], &in_x);
	rcqd_add(ret, ret, &in_coef_new[0]);

    free(in_coef_old);
    free(in_coef_new);

    //return ret;
	return;
}

/* value of a'(x) */
// Based on Horner method
void ceval_diff_qdpoly(cqdfloat *ret, QDPoly a, cqdfloat *x)
{
	long int i;
	double tmp[QDSIZE];

	//set0_mpfcmplx(ret);
	rcqd_set_ui(ret, 0UL);
	//set_real_mpfcmplx(ret, get_qdpoly_i(a, a->deg));
	rcqd_set_qd(ret, get_qdpoly_i(a, a->deg));
	//mul_mpfcmplx_ui(ret, ret, (unsigned long)a->deg);
	rcqd_mul_ui(ret, ret, (unsigned long)a->deg);

	for(i = a->deg - 1; i >= 1; i--)
	{
		//mul_mpfcmplx(ret, ret, x);
		rcqd_mul(ret, ret, x);
		rqd_mul_ui(tmp, get_qdpoly_i(a, i), (unsigned long)i);
		//add_mpfcmplx_mpf(ret, ret, tmp);
		rcqd_add_qd(ret, ret, tmp);
	}
}

//------------
// AVX2
//------------

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
void _bncavx2_eval_qdpoly_estrin(double ret[QDSIZE], QDPoly a, double x[QDSIZE])
{
    //double *in_coef_old, *in_coef_new;
    QDVector in_coef_old, in_coef_new;
    double in_x[QDSIZE], tmp[QDSIZE];
    long int in_degree, num_in_coef, i, num_loop_unit;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d old_coef4[2][QDSIZE], new_coef4[QDSIZE], a04[QDSIZE], a14[QDSIZE], x4[QDSIZE], zero4[QDSIZE];
    __m256d tmp4[QDSIZE];

    if((a->deg + 1) <= _BNC_D_WIDTH)
    {
        eval_qdpoly_horner(ret, a, x);
        return;
    }

    // zero4 := 0
    _bncavx2_set0_qd(zero4); //  = _mm256_setzero_pd();
    num_loop_unit = 2 * _BNC_D_WIDTH;

#elif defined(__AVX512F__) // __AVX512F__
	__m512d old_coef8[2][QDSIZE], new_coef8[QDSIZE], a08[QDSIZE], a18[QDSIZE], x8[QDSIZE], zero8[QDSIZE];
    __m512d tmp8[QDSIZE];

    if((a->deg + 1) <= (2 * _BNC_D_WIDTH))
    {
        eval_qdpoly_horner(ret, a, x);
        return;
    }

    // zero8 := 0
    _bncavx512_set0_qd(zero8); // = _mm512_setzero_pd();
    num_loop_unit = 2 * (2 * _BNC_D_WIDTH);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
    {
        long _vl0 = (long)svcntd();
        if((a->deg + 1) <= _vl0)
        {
            eval_qdpoly_horner(ret, a, x);
            return;
        }
        num_loop_unit = 2 * _vl0;
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // ARM NEON
	float64x2_t old_coef2[2][QDSIZE], new_coef2[QDSIZE], a02[QDSIZE], a12[QDSIZE], x2[QDSIZE], zero2[QDSIZE];
    float64x2_t tmp2[QDSIZE];
    
    if((a->deg + 1) <= 2)
    {
        eval_qdpoly_horner(ret, a, x);
        return;
    }

    // zero2 := 0
    _bncneon_set0_qd(zero2);
    num_loop_unit = 2 * 2; // Process 2 QD values at a time

#else // __AVX2__
    eval_qdpoly_estrin(ret, a, x);
    return;
#endif // __AVX2__

    num_in_coef = a->deg + 1;
    //if((num_in_coef % 2) == 1) num_in_coef++;
    if((num_in_coef % num_loop_unit) != 0) 
        num_in_coef = (long int)ceil(((double)num_in_coef / (double)num_loop_unit)) * num_loop_unit; ////num_in_coef += num_in_coef % num_loop_unit;

    in_degree = num_in_coef - 1;
    //in_coef_old = (double *)calloc(num_in_coef, sizeof(double));
    //in_coef_new = (double *)calloc(num_in_coef, sizeof(double));
    in_coef_old = init_qdvector(num_in_coef);
    //in_coef_new = init_dvector(num_in_coef);

    //for(i = 0; i <= a->deg; i++) in_coef_old[i] = get_dpoly_i(a, i);
    //for(i = a->deg + 1; i <= in_degree; i++) in_coef_old[i] = 0.0;
    for(i = 0; i <= a->deg; i++) set_qdvector_i(in_coef_old, i, get_qdpoly_i(a, i));

    rqd_set(in_x, x); //  = x;

    //printf("num_in_coef = %ld\n", num_in_coef);
    //printf("degree -> in_degree: %d -> %d\n", degree, in_degree);

    while(1)
    //while(num_in_coef > num_loop_unit)
    {
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
        // x2_4 := x^2
        //x4 = _mm256_set1_pd(in_x);
        _bncavx2_rqd_set1_qd(x4, in_x);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            old_coef4[0][0] = _mm256_load_pd(&(in_coef_old->element[0][i * num_loop_unit]));
            old_coef4[1][0] = _mm256_load_pd(&(in_coef_old->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef4[0][1] = _mm256_load_pd(&(in_coef_old->element[1][i * num_loop_unit]));
            old_coef4[1][1] = _mm256_load_pd(&(in_coef_old->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef4[0][2] = _mm256_load_pd(&(in_coef_old->element[2][i * num_loop_unit]));
            old_coef4[1][2] = _mm256_load_pd(&(in_coef_old->element[2][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef4[0][3] = _mm256_load_pd(&(in_coef_old->element[3][i * num_loop_unit]));
            old_coef4[1][3] = _mm256_load_pd(&(in_coef_old->element[3][i * num_loop_unit + _BNC_D_WIDTH]));
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
            a04[2] = _mm256_unpacklo_pd(old_coef4[0][2], old_coef4[1][2]);
            a14[2] = _mm256_unpackhi_pd(old_coef4[0][2], old_coef4[1][2]);
            a04[3] = _mm256_unpacklo_pd(old_coef4[0][3], old_coef4[1][3]);
            a14[3] = _mm256_unpackhi_pd(old_coef4[0][3], old_coef4[1][3]);
            //printf("         a04 = "); PRINT_M256D_SL(a04);  
            //printf("         a14 = "); PRINT_M256D_SL(a14);
            //new_coef4 = _mm256_fmadd_pd(a14, x4, a04);
            _bncavx2_rqd_mul(new_coef4, a14, x4);
            _bncavx2_rqd_add(new_coef4, new_coef4, a04);
            new_coef4[0] = _mm256_permute4x64_pd(new_coef4[0], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4[1] = _mm256_permute4x64_pd(new_coef4[1], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4[2] = _mm256_permute4x64_pd(new_coef4[2], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4[3] = _mm256_permute4x64_pd(new_coef4[3], (int)(3*64 + 1*16 + 2*4 + 0));
            // b0, b2, b1, b3 -> b0, b1, b2, b3
            //_mm256_permute4x64_pd(new_coef4, (int)(3*64 + 1*16 + 2*4 + 0));
            //printf("   new_coef4 = "); PRINT_M256D_SL(new_coef4);
            // embed 0s
            _mm256_store_pd(&(in_coef_old->element[0][i * num_loop_unit]), zero4[0]);
            _mm256_store_pd(&(in_coef_old->element[1][i * num_loop_unit]), zero4[1]);
            _mm256_store_pd(&(in_coef_old->element[2][i * num_loop_unit]), zero4[2]);
            _mm256_store_pd(&(in_coef_old->element[3][i * num_loop_unit]), zero4[3]);
            _mm256_store_pd(&(in_coef_old->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero4[0]);
            _mm256_store_pd(&(in_coef_old->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero4[1]);
            _mm256_store_pd(&(in_coef_old->element[2][i * num_loop_unit + _BNC_D_WIDTH]), zero4[2]);
            _mm256_store_pd(&(in_coef_old->element[3][i * num_loop_unit + _BNC_D_WIDTH]), zero4[3]);
            _mm256_store_pd(&(in_coef_old->element[0][i * num_loop_unit]), zero4[0]);
            _mm256_store_pd(&(in_coef_old->element[1][i * num_loop_unit]), zero4[1]);
            _mm256_store_pd(&(in_coef_old->element[2][i * num_loop_unit]), zero4[2]);
            _mm256_store_pd(&(in_coef_old->element[3][i * num_loop_unit]), zero4[3]);
            _mm256_store_pd(&(in_coef_old->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero4[0]);
            _mm256_store_pd(&(in_coef_old->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero4[1]);
            _mm256_store_pd(&(in_coef_old->element[2][i * num_loop_unit + _BNC_D_WIDTH]), zero4[2]);
            _mm256_store_pd(&(in_coef_old->element[3][i * num_loop_unit + _BNC_D_WIDTH]), zero4[3]);

            _mm256_store_pd(&(in_coef_old->element[0][i * _BNC_D_WIDTH]), new_coef4[0]);
            _mm256_store_pd(&(in_coef_old->element[1][i * _BNC_D_WIDTH]), new_coef4[1]);
            _mm256_store_pd(&(in_coef_old->element[2][i * _BNC_D_WIDTH]), new_coef4[2]);
            _mm256_store_pd(&(in_coef_old->element[3][i * _BNC_D_WIDTH]), new_coef4[3]);
            //printf("in_coef_old  = %f %f %f %f\n", in_coef_old->element[i * _BNC_D_WIDTH], in_coef_old->element[i * _BNC_D_WIDTH + 1], in_coef_old->element[i * _BNC_D_WIDTH + 2], in_coef_old->element[i * _BNC_D_WIDTH + 3]);
        }
#elif defined(__AVX512F__) // __AVX512F__
        // x8 := x (broadcast to 8 elements)
        _bncavx512_rqd_set1_qd(x8, in_x);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            // Load coefficients for QDSIZE=4 components
            old_coef8[0][0] = _mm512_load_pd(&(in_coef_old->element[0][i * num_loop_unit]));
            old_coef8[1][0] = _mm512_load_pd(&(in_coef_old->element[0][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8[0][1] = _mm512_load_pd(&(in_coef_old->element[1][i * num_loop_unit]));
            old_coef8[1][1] = _mm512_load_pd(&(in_coef_old->element[1][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8[0][2] = _mm512_load_pd(&(in_coef_old->element[2][i * num_loop_unit]));
            old_coef8[1][2] = _mm512_load_pd(&(in_coef_old->element[2][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8[0][3] = _mm512_load_pd(&(in_coef_old->element[3][i * num_loop_unit]));
            old_coef8[1][3] = _mm512_load_pd(&(in_coef_old->element[3][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            
            // Separate even/odd coefficients using unpack
            a08[0] = _mm512_unpacklo_pd(old_coef8[0][0], old_coef8[1][0]);
            a18[0] = _mm512_unpackhi_pd(old_coef8[0][0], old_coef8[1][0]);
            a08[1] = _mm512_unpacklo_pd(old_coef8[0][1], old_coef8[1][1]);
            a18[1] = _mm512_unpackhi_pd(old_coef8[0][1], old_coef8[1][1]);
            a08[2] = _mm512_unpacklo_pd(old_coef8[0][2], old_coef8[1][2]);
            a18[2] = _mm512_unpackhi_pd(old_coef8[0][2], old_coef8[1][2]);
            a08[3] = _mm512_unpacklo_pd(old_coef8[0][3], old_coef8[1][3]);
            a18[3] = _mm512_unpackhi_pd(old_coef8[0][3], old_coef8[1][3]);
            
            // Compute: new_coef = a0 + a1 * x using QD arithmetic
            _bncavx512_rqd_mul(new_coef8, a18, x8);
            _bncavx512_rqd_add(new_coef8, new_coef8, a08);
            
            // Permute to restore order
            new_coef8[0] = _mm512_permutex_pd(new_coef8[0], 0xD8); // 0xD8 = 11011000
            new_coef8[1] = _mm512_permutex_pd(new_coef8[1], 0xD8);
            new_coef8[2] = _mm512_permutex_pd(new_coef8[2], 0xD8);
            new_coef8[3] = _mm512_permutex_pd(new_coef8[3], 0xD8);
            
            // Store zeros to clear array
            _mm512_store_pd(&(in_coef_old->element[0][i * num_loop_unit]), zero8[0]);
            _mm512_store_pd(&(in_coef_old->element[1][i * num_loop_unit]), zero8[1]);
            _mm512_store_pd(&(in_coef_old->element[2][i * num_loop_unit]), zero8[2]);
            _mm512_store_pd(&(in_coef_old->element[3][i * num_loop_unit]), zero8[3]);
            _mm512_store_pd(&(in_coef_old->element[0][i * num_loop_unit + (2 * _BNC_D_WIDTH)]), zero8[0]);
            _mm512_store_pd(&(in_coef_old->element[1][i * num_loop_unit + (2 * _BNC_D_WIDTH)]), zero8[1]);
            _mm512_store_pd(&(in_coef_old->element[2][i * num_loop_unit + (2 * _BNC_D_WIDTH)]), zero8[2]);
            _mm512_store_pd(&(in_coef_old->element[3][i * num_loop_unit + (2 * _BNC_D_WIDTH)]), zero8[3]);

            // Store result
            _mm512_store_pd(&(in_coef_old->element[0][i * (2 * _BNC_D_WIDTH)]), new_coef8[0]);
            _mm512_store_pd(&(in_coef_old->element[1][i * (2 * _BNC_D_WIDTH)]), new_coef8[1]);
            _mm512_store_pd(&(in_coef_old->element[2][i * (2 * _BNC_D_WIDTH)]), new_coef8[2]);
            _mm512_store_pd(&(in_coef_old->element[3][i * (2 * _BNC_D_WIDTH)]), new_coef8[3]);
        }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
        {
            long _vl = (long)svcntd();
            svbool_t _pg = svptrue_b64();
            svfloat64_t _z = svdup_n_f64(0.0);
            svfloat64_t _x0 = svdup_n_f64(in_x[0]);
            svfloat64_t _x1 = svdup_n_f64(in_x[1]);
            svfloat64_t _x2 = svdup_n_f64(in_x[2]);
            svfloat64_t _x3 = svdup_n_f64(in_x[3]);
            for(i = 0; i < (num_in_coef / num_loop_unit); i++)
            {
                svfloat64_t _v0a = svld1_f64(_pg, &(in_coef_old->element[0][i*num_loop_unit]));
                svfloat64_t _v0b = svld1_f64(_pg, &(in_coef_old->element[0][i*num_loop_unit + _vl]));
                svfloat64_t _v1a = svld1_f64(_pg, &(in_coef_old->element[1][i*num_loop_unit]));
                svfloat64_t _v1b = svld1_f64(_pg, &(in_coef_old->element[1][i*num_loop_unit + _vl]));
                svfloat64_t _v2a = svld1_f64(_pg, &(in_coef_old->element[2][i*num_loop_unit]));
                svfloat64_t _v2b = svld1_f64(_pg, &(in_coef_old->element[2][i*num_loop_unit + _vl]));
                svfloat64_t _v3a = svld1_f64(_pg, &(in_coef_old->element[3][i*num_loop_unit]));
                svfloat64_t _v3b = svld1_f64(_pg, &(in_coef_old->element[3][i*num_loop_unit + _vl]));
                svfloat64_t _e0 = svuzp1_f64(_v0a,_v0b), _o0 = svuzp2_f64(_v0a,_v0b);
                svfloat64_t _e1 = svuzp1_f64(_v1a,_v1b), _o1 = svuzp2_f64(_v1a,_v1b);
                svfloat64_t _e2 = svuzp1_f64(_v2a,_v2b), _o2 = svuzp2_f64(_v2a,_v2b);
                svfloat64_t _e3 = svuzp1_f64(_v3a,_v3b), _o3 = svuzp2_f64(_v3a,_v3b);
                svfloat64_t _m0, _m1, _m2, _m3;
                svfloat64_t _n0, _n1, _n2, _n3;
                _bncsve2_rqd_mul(_pg, &_m0, &_m1, &_m2, &_m3, _o0, _o1, _o2, _o3, _x0, _x1, _x2, _x3);
                _bncsve2_rqd_add(_pg, &_n0, &_n1, &_n2, &_n3, _e0, _e1, _e2, _e3, _m0, _m1, _m2, _m3);
                svst1_f64(_pg, &(in_coef_old->element[0][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old->element[0][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old->element[1][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old->element[1][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old->element[2][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old->element[2][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old->element[3][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old->element[3][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old->element[0][i*_vl]), _n0);
                svst1_f64(_pg, &(in_coef_old->element[1][i*_vl]), _n1);
                svst1_f64(_pg, &(in_coef_old->element[2][i*_vl]), _n2);
                svst1_f64(_pg, &(in_coef_old->element[3][i*_vl]), _n3);
            }
        }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // ARM NEON
        // x2 := x (broadcast to 2 elements)
        _bncneon_rqd_set1_qd(x2, in_x);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            // Load coefficients for QDSIZE=4 components
            old_coef2[0][0] = vld1q_f64(&(in_coef_old->element[0][i * num_loop_unit]));
            old_coef2[1][0] = vld1q_f64(&(in_coef_old->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2[0][1] = vld1q_f64(&(in_coef_old->element[1][i * num_loop_unit]));
            old_coef2[1][1] = vld1q_f64(&(in_coef_old->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2[0][2] = vld1q_f64(&(in_coef_old->element[2][i * num_loop_unit]));
            old_coef2[1][2] = vld1q_f64(&(in_coef_old->element[2][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2[0][3] = vld1q_f64(&(in_coef_old->element[3][i * num_loop_unit]));
            old_coef2[1][3] = vld1q_f64(&(in_coef_old->element[3][i * num_loop_unit + _BNC_D_WIDTH]));
            
            // Separate even/odd coefficients
            a02[0] = vzip1q_f64(old_coef2[0][0], old_coef2[1][0]);
            a12[0] = vzip2q_f64(old_coef2[0][0], old_coef2[1][0]);
            a02[1] = vzip1q_f64(old_coef2[0][1], old_coef2[1][1]);
            a12[1] = vzip2q_f64(old_coef2[0][1], old_coef2[1][1]);
            a02[2] = vzip1q_f64(old_coef2[0][2], old_coef2[1][2]);
            a12[2] = vzip2q_f64(old_coef2[0][2], old_coef2[1][2]);
            a02[3] = vzip1q_f64(old_coef2[0][3], old_coef2[1][3]);
            a12[3] = vzip2q_f64(old_coef2[0][3], old_coef2[1][3]);
            
            // Compute: new_coef = a0 + a1 * x using QD arithmetic
            _bncneon_rqd_mul(new_coef2, a12, x2);
            _bncneon_rqd_add(new_coef2, new_coef2, a02);
           
            // Store zeros to clear array
            vst1q_f64(&(in_coef_old->element[0][i * num_loop_unit]), zero2[0]);
            vst1q_f64(&(in_coef_old->element[1][i * num_loop_unit]), zero2[1]);
            vst1q_f64(&(in_coef_old->element[2][i * num_loop_unit]), zero2[2]);
            vst1q_f64(&(in_coef_old->element[3][i * num_loop_unit]), zero2[3]);
            vst1q_f64(&(in_coef_old->element[0][i * num_loop_unit + _BNC_D_WIDTH]), zero2[0]);
            vst1q_f64(&(in_coef_old->element[1][i * num_loop_unit + _BNC_D_WIDTH]), zero2[1]);
            vst1q_f64(&(in_coef_old->element[2][i * num_loop_unit + _BNC_D_WIDTH]), zero2[2]);
            vst1q_f64(&(in_coef_old->element[3][i * num_loop_unit + _BNC_D_WIDTH]), zero2[3]);

            // Store result
            vst1q_f64(&(in_coef_old->element[0][i * _BNC_D_WIDTH]), new_coef2[0]);
            vst1q_f64(&(in_coef_old->element[1][i * _BNC_D_WIDTH]), new_coef2[1]);
            vst1q_f64(&(in_coef_old->element[2][i * _BNC_D_WIDTH]), new_coef2[2]);
            vst1q_f64(&(in_coef_old->element[3][i * _BNC_D_WIDTH]), new_coef2[3]);
        }
#endif // __AVX2__

        //return estrin(x * x, coef, num_in_coef / 2 - 1);
        //in_x = in_x * in_x;
        rqd_mul(tmp, in_x, in_x);
        rqd_set(in_x, tmp);
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
    rqd_set(ret, get_qdvector_i(in_coef_old, _BNC_D_WIDTH - 1));
    for(i = _BNC_D_WIDTH - 2; i >= 0; i--)
    {
        //ret = ret * in_x + get_dvector_i(in_coef_old, i);
        rqd_mul(tmp, ret, in_x);
        rqd_add(ret, tmp, get_qdvector_i(in_coef_old, i));
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2: final fold over vl coeffs
    {
        long _vl = (long)svcntd();
        rqd_set(ret, get_qdvector_i(in_coef_old, _vl - 1));
        for(i = _vl - 2; i >= 0; i--)
        {
            rqd_mul(tmp, ret, in_x);
            rqd_add(ret, tmp, get_qdvector_i(in_coef_old, i));
        }
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // ARM NEON
    rqd_set(ret, get_qdvector_i(in_coef_old, 1));
    for(i = 0; i >= 0; i--)
    {
        rqd_mul(tmp, ret, in_x);
        rqd_add(ret, tmp, get_qdvector_i(in_coef_old, i));
    }
#endif // __AVX2__

    //free(in_coef_old);
    //free(in_coef_new);
    free_qdvector(in_coef_old);
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
void _bncavx2_ceval_qdpoly_estrin(cqdfloat *ret, QDPoly a, cqdfloat *x)
{
    //double *in_coef_old, *in_coef_new;
    QDVector in_coef_old_real, in_coef_old_imag;
    cqdfloat in_x; // , ret;
    cqdfloat in_ret, ctmp;
    long int in_degree, num_in_coef, i, num_loop_unit;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d old_coef4_real[2][QDSIZE], old_coef4_imag[2][QDSIZE];
    __m256d new_coef4_real[QDSIZE], new_coef4_imag[QDSIZE];
    __m256d a04_real[QDSIZE], a04_imag[QDSIZE], a14_real[QDSIZE], a14_imag[QDSIZE], x4_real[QDSIZE], x4_imag[QDSIZE], zero4[QDSIZE];
    __m256d ctmp4_real[QDSIZE], ctmp4_imag[QDSIZE];

    if((a->deg + 1) <= _BNC_D_WIDTH)
    {
        ceval_qdpoly_horner(ret, a, x);
        return;
    }

    // zero4 := 0
    _bncavx2_rqd_set0(zero4); //  = _mm256_setzero_pd();
    num_loop_unit = 2 * _BNC_D_WIDTH;

#elif defined(__AVX512F__) // __AVX512F__
	__m512d old_coef8_real[2][QDSIZE], old_coef8_imag[2][QDSIZE];
    __m512d new_coef8_real[QDSIZE], new_coef8_imag[QDSIZE];
    __m512d a08_real[QDSIZE], a08_imag[QDSIZE], a18_real[QDSIZE], a18_imag[QDSIZE];
    __m512d x8_real[QDSIZE], x8_imag[QDSIZE], zero8[QDSIZE];
    __m512d ctmp8_real[QDSIZE], ctmp8_imag[QDSIZE];

    if((a->deg + 1) <= (2 * _BNC_D_WIDTH))
    {
        ceval_qdpoly_horner(ret, a, x);
        return;
    }

    // zero8 := 0
    _bncavx512_rqd_set0(zero8);
    num_loop_unit = 2 * (2 * _BNC_D_WIDTH);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
    {
        long _vl0 = (long)svcntd();
        if((a->deg + 1) <= _vl0)
        {
            ceval_qdpoly_horner(ret, a, x);
            return;
        }
        num_loop_unit = 2 * _vl0;
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // ARM NEON
	float64x2_t old_coef2_real[2][QDSIZE], old_coef2_imag[2][QDSIZE];
    float64x2_t new_coef2_real[QDSIZE], new_coef2_imag[QDSIZE];
    float64x2_t a02_real[QDSIZE], a02_imag[QDSIZE], a12_real[QDSIZE], a12_imag[QDSIZE];
    float64x2_t x2_real[QDSIZE], x2_imag[QDSIZE], zero2[QDSIZE];
    float64x2_t ctmp2_real[QDSIZE], ctmp2_imag[QDSIZE];

    if((a->deg + 1) <= 2)
    {
        ceval_qdpoly_horner(ret, a, x);
        return;
    }

    // zero2 := 0
    _bncneon_rqd_set0(zero2);
    num_loop_unit = 2 * 2;

#else // __AVX2__
    ceval_qdpoly_estrin(ret, a, x);
    return;
#endif // __AVX2__

    num_in_coef = a->deg + 1;
    //if((num_in_coef % 2) == 1) num_in_coef++;
    if((num_in_coef % num_loop_unit) != 0) 
        num_in_coef = (long int)ceil(((double)num_in_coef / (double)num_loop_unit)) * num_loop_unit; ////num_in_coef += num_in_coef % num_loop_unit;

    in_degree = num_in_coef - 1;
    //in_coef_old = (double *)calloc(num_in_coef, sizeof(double));
    //in_coef_new = (double *)calloc(num_in_coef, sizeof(double));
    in_coef_old_real = init_qdvector(num_in_coef);
    in_coef_old_imag = init_qdvector(num_in_coef);
    //in_coef_new = init_dvector(num_in_coef);

    //for(i = 0; i <= a->deg; i++) in_coef_old[i] = get_dpoly_i(a, i);
    //for(i = a->deg + 1; i <= in_degree; i++) in_coef_old[i] = 0.0;
    for(i = 0; i <= a->deg; i++) set_qdvector_i(in_coef_old_real, i, get_qdpoly_i(a, i));

    //in_x = x->re + x->im * I;
    rcqd_set(&in_x, x);

    //printf("num_in_coef = %ld\n", num_in_coef);
    //printf("degree -> in_degree: %d -> %d\n", degree, in_degree);

    while(1)
    //while(num_in_coef > num_loop_unit)
    {
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
        // x2_4 := x^2
        //x4 = _mm256_set1_pd(in_x);
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
             //printf("         a04 = "); PRINT_M256D_SL(a04);  
            //printf("         a14 = "); PRINT_M256D_SL(a14);
            //new_coef4 = _mm256_fmadd_pd(a14, x4, a04);

            // a14 * x4 -> (a14_re * x4_re - a14_im * x4_im) + (a14_re * x4_im + a14_im * x4_re) * I
            //new_coef4_real = _mm256_sub_pd(_mm256_mul_pd(a14_real, x4_real), _mm256_mul_pd(a14_imag, x4_imag));
            //new_coef4_imag = _mm256_add_pd(_mm256_mul_pd(a14_real, x4_imag), _mm256_mul_pd(a14_imag, x4_real));
            _bncavx2_rcqd_mul(ctmp4_real, ctmp4_imag, a14_real, a14_imag, x4_real, x4_imag);

            // a14_x4 + a04 -> a14_x4_re + a_04_re + (a_14_x4_im + a_04_im) * I
            //new_coef4_real = _mm256_add_pd(new_coef4_real, a04_real);
            //new_coef4_imag = _mm256_add_pd(new_coef4_imag, a04_imag);
            _bncavx2_rcqd_add(new_coef4_real, new_coef4_imag, ctmp4_real, ctmp4_imag, a04_real, a04_imag);

            new_coef4_real[0] = _mm256_permute4x64_pd(new_coef4_real[0], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_imag[0] = _mm256_permute4x64_pd(new_coef4_imag[0], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_real[1] = _mm256_permute4x64_pd(new_coef4_real[1], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_imag[1] = _mm256_permute4x64_pd(new_coef4_imag[1], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_real[2] = _mm256_permute4x64_pd(new_coef4_real[2], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_imag[2] = _mm256_permute4x64_pd(new_coef4_imag[2], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_real[3] = _mm256_permute4x64_pd(new_coef4_real[3], (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_imag[3] = _mm256_permute4x64_pd(new_coef4_imag[3], (int)(3*64 + 1*16 + 2*4 + 0));
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
            //printf("in_coef_old_real  = %f %f %f %f\n", in_coef_old_real->element[0][i * _BNC_D_WIDTH], in_coef_old_real->element[0][i * _BNC_D_WIDTH + 1], in_coef_old_real->element[0][i * _BNC_D_WIDTH + 2], in_coef_old_real->element[0][i * _BNC_D_WIDTH + 3]);
        }
#elif defined(__AVX512F__) // __AVX512F__
        // Broadcast complex x to vectors
        _bncavx512_rqd_set1_qd(x8_real, in_x.val_re);
        _bncavx512_rqd_set1_qd(x8_imag, in_x.val_im);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            // Load real coefficients for QDSIZE=4 components
            old_coef8_real[0][0] = _mm512_load_pd(&(in_coef_old_real->element[0][i * num_loop_unit]));
            old_coef8_real[1][0] = _mm512_load_pd(&(in_coef_old_real->element[0][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_real[0][1] = _mm512_load_pd(&(in_coef_old_real->element[1][i * num_loop_unit]));
            old_coef8_real[1][1] = _mm512_load_pd(&(in_coef_old_real->element[1][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_real[0][2] = _mm512_load_pd(&(in_coef_old_real->element[2][i * num_loop_unit]));
            old_coef8_real[1][2] = _mm512_load_pd(&(in_coef_old_real->element[2][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_real[0][3] = _mm512_load_pd(&(in_coef_old_real->element[3][i * num_loop_unit]));
            old_coef8_real[1][3] = _mm512_load_pd(&(in_coef_old_real->element[3][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));

            // Load imaginary coefficients for QDSIZE=4 components
            old_coef8_imag[0][0] = _mm512_load_pd(&(in_coef_old_imag->element[0][i * num_loop_unit]));
            old_coef8_imag[1][0] = _mm512_load_pd(&(in_coef_old_imag->element[0][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_imag[0][1] = _mm512_load_pd(&(in_coef_old_imag->element[1][i * num_loop_unit]));
            old_coef8_imag[1][1] = _mm512_load_pd(&(in_coef_old_imag->element[1][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_imag[0][2] = _mm512_load_pd(&(in_coef_old_imag->element[2][i * num_loop_unit]));
            old_coef8_imag[1][2] = _mm512_load_pd(&(in_coef_old_imag->element[2][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            old_coef8_imag[0][3] = _mm512_load_pd(&(in_coef_old_imag->element[3][i * num_loop_unit]));
            old_coef8_imag[1][3] = _mm512_load_pd(&(in_coef_old_imag->element[3][i * num_loop_unit + (2 * _BNC_D_WIDTH)]));
            
            // Separate even/odd coefficients using unpack - Real part
            a08_real[0] = _mm512_unpacklo_pd(old_coef8_real[0][0], old_coef8_real[1][0]);
            a18_real[0] = _mm512_unpackhi_pd(old_coef8_real[0][0], old_coef8_real[1][0]);
            a08_real[1] = _mm512_unpacklo_pd(old_coef8_real[0][1], old_coef8_real[1][1]);
            a18_real[1] = _mm512_unpackhi_pd(old_coef8_real[0][1], old_coef8_real[1][1]);
            a08_real[2] = _mm512_unpacklo_pd(old_coef8_real[0][2], old_coef8_real[1][2]);
            a18_real[2] = _mm512_unpackhi_pd(old_coef8_real[0][2], old_coef8_real[1][2]);
            a08_real[3] = _mm512_unpacklo_pd(old_coef8_real[0][3], old_coef8_real[1][3]);
            a18_real[3] = _mm512_unpackhi_pd(old_coef8_real[0][3], old_coef8_real[1][3]);

            // Separate even/odd coefficients using unpack - Imaginary part
            a08_imag[0] = _mm512_unpacklo_pd(old_coef8_imag[0][0], old_coef8_imag[1][0]);
            a18_imag[0] = _mm512_unpackhi_pd(old_coef8_imag[0][0], old_coef8_imag[1][0]);
            a08_imag[1] = _mm512_unpacklo_pd(old_coef8_imag[0][1], old_coef8_imag[1][1]);
            a18_imag[1] = _mm512_unpackhi_pd(old_coef8_imag[0][1], old_coef8_imag[1][1]);
            a08_imag[2] = _mm512_unpacklo_pd(old_coef8_imag[0][2], old_coef8_imag[1][2]);
            a18_imag[2] = _mm512_unpackhi_pd(old_coef8_imag[0][2], old_coef8_imag[1][2]);
            a08_imag[3] = _mm512_unpacklo_pd(old_coef8_imag[0][3], old_coef8_imag[1][3]);
            a18_imag[3] = _mm512_unpackhi_pd(old_coef8_imag[0][3], old_coef8_imag[1][3]);

            // Complex QD multiplication: a1 * x using complex QD arithmetic
            _bncavx512_rcqd_mul(ctmp8_real, ctmp8_imag, a18_real, a18_imag, x8_real, x8_imag);

            // Complex QD addition: (a1 * x) + a0
            _bncavx512_rcqd_add(new_coef8_real, new_coef8_imag, ctmp8_real, ctmp8_imag, a08_real, a08_imag);
            
            // Permute to restore order
            new_coef8_real[0] = _mm512_permutex_pd(new_coef8_real[0], 0xD8);
            new_coef8_imag[0] = _mm512_permutex_pd(new_coef8_imag[0], 0xD8);
            new_coef8_real[1] = _mm512_permutex_pd(new_coef8_real[1], 0xD8);
            new_coef8_imag[1] = _mm512_permutex_pd(new_coef8_imag[1], 0xD8);
            new_coef8_real[2] = _mm512_permutex_pd(new_coef8_real[2], 0xD8);
            new_coef8_imag[2] = _mm512_permutex_pd(new_coef8_imag[2], 0xD8);
            new_coef8_real[3] = _mm512_permutex_pd(new_coef8_real[3], 0xD8);
            new_coef8_imag[3] = _mm512_permutex_pd(new_coef8_imag[3], 0xD8);
            
            // Store zeros to clear array
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

            // Store results
            _mm512_store_pd(&(in_coef_old_real->element[0][i * (2 * _BNC_D_WIDTH)]), new_coef8_real[0]);
            _mm512_store_pd(&(in_coef_old_imag->element[0][i * (2 * _BNC_D_WIDTH)]), new_coef8_imag[0]);
            _mm512_store_pd(&(in_coef_old_real->element[1][i * (2 * _BNC_D_WIDTH)]), new_coef8_real[1]);
            _mm512_store_pd(&(in_coef_old_imag->element[1][i * (2 * _BNC_D_WIDTH)]), new_coef8_imag[1]);
            _mm512_store_pd(&(in_coef_old_real->element[2][i * (2 * _BNC_D_WIDTH)]), new_coef8_real[2]);
            _mm512_store_pd(&(in_coef_old_imag->element[2][i * (2 * _BNC_D_WIDTH)]), new_coef8_imag[2]);
            _mm512_store_pd(&(in_coef_old_real->element[3][i * (2 * _BNC_D_WIDTH)]), new_coef8_real[3]);
            _mm512_store_pd(&(in_coef_old_imag->element[3][i * (2 * _BNC_D_WIDTH)]), new_coef8_imag[3]);
        }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
        {
            long _vl = (long)svcntd();
            svbool_t _pg = svptrue_b64();
            svfloat64_t _z = svdup_n_f64(0.0);
            svfloat64_t _xr0 = svdup_n_f64(in_x.val_re[0]);
            svfloat64_t _xr1 = svdup_n_f64(in_x.val_re[1]);
            svfloat64_t _xr2 = svdup_n_f64(in_x.val_re[2]);
            svfloat64_t _xr3 = svdup_n_f64(in_x.val_re[3]);
            svfloat64_t _xi0 = svdup_n_f64(in_x.val_im[0]);
            svfloat64_t _xi1 = svdup_n_f64(in_x.val_im[1]);
            svfloat64_t _xi2 = svdup_n_f64(in_x.val_im[2]);
            svfloat64_t _xi3 = svdup_n_f64(in_x.val_im[3]);
            for(i = 0; i < (num_in_coef / num_loop_unit); i++)
            {
                svfloat64_t _r0a = svld1_f64(_pg, &(in_coef_old_real->element[0][i*num_loop_unit]));
                svfloat64_t _r0b = svld1_f64(_pg, &(in_coef_old_real->element[0][i*num_loop_unit + _vl]));
                svfloat64_t _i0a = svld1_f64(_pg, &(in_coef_old_imag->element[0][i*num_loop_unit]));
                svfloat64_t _i0b = svld1_f64(_pg, &(in_coef_old_imag->element[0][i*num_loop_unit + _vl]));
                svfloat64_t _r1a = svld1_f64(_pg, &(in_coef_old_real->element[1][i*num_loop_unit]));
                svfloat64_t _r1b = svld1_f64(_pg, &(in_coef_old_real->element[1][i*num_loop_unit + _vl]));
                svfloat64_t _i1a = svld1_f64(_pg, &(in_coef_old_imag->element[1][i*num_loop_unit]));
                svfloat64_t _i1b = svld1_f64(_pg, &(in_coef_old_imag->element[1][i*num_loop_unit + _vl]));
                svfloat64_t _r2a = svld1_f64(_pg, &(in_coef_old_real->element[2][i*num_loop_unit]));
                svfloat64_t _r2b = svld1_f64(_pg, &(in_coef_old_real->element[2][i*num_loop_unit + _vl]));
                svfloat64_t _i2a = svld1_f64(_pg, &(in_coef_old_imag->element[2][i*num_loop_unit]));
                svfloat64_t _i2b = svld1_f64(_pg, &(in_coef_old_imag->element[2][i*num_loop_unit + _vl]));
                svfloat64_t _r3a = svld1_f64(_pg, &(in_coef_old_real->element[3][i*num_loop_unit]));
                svfloat64_t _r3b = svld1_f64(_pg, &(in_coef_old_real->element[3][i*num_loop_unit + _vl]));
                svfloat64_t _i3a = svld1_f64(_pg, &(in_coef_old_imag->element[3][i*num_loop_unit]));
                svfloat64_t _i3b = svld1_f64(_pg, &(in_coef_old_imag->element[3][i*num_loop_unit + _vl]));
                svfloat64_t _er0 = svuzp1_f64(_r0a,_r0b), _or0 = svuzp2_f64(_r0a,_r0b);
                svfloat64_t _ei0 = svuzp1_f64(_i0a,_i0b), _oi0 = svuzp2_f64(_i0a,_i0b);
                svfloat64_t _er1 = svuzp1_f64(_r1a,_r1b), _or1 = svuzp2_f64(_r1a,_r1b);
                svfloat64_t _ei1 = svuzp1_f64(_i1a,_i1b), _oi1 = svuzp2_f64(_i1a,_i1b);
                svfloat64_t _er2 = svuzp1_f64(_r2a,_r2b), _or2 = svuzp2_f64(_r2a,_r2b);
                svfloat64_t _ei2 = svuzp1_f64(_i2a,_i2b), _oi2 = svuzp2_f64(_i2a,_i2b);
                svfloat64_t _er3 = svuzp1_f64(_r3a,_r3b), _or3 = svuzp2_f64(_r3a,_r3b);
                svfloat64_t _ei3 = svuzp1_f64(_i3a,_i3b), _oi3 = svuzp2_f64(_i3a,_i3b);
                svfloat64_t _a0, _a1, _a2, _a3;
                svfloat64_t _b0, _b1, _b2, _b3;
                svfloat64_t _pr0, _pr1, _pr2, _pr3;
                svfloat64_t _pi0, _pi1, _pi2, _pi3;
                _bncsve2_rqd_mul(_pg, &_a0, &_a1, &_a2, &_a3, _or0, _or1, _or2, _or3, _xr0, _xr1, _xr2, _xr3);
                _bncsve2_rqd_mul(_pg, &_b0, &_b1, &_b2, &_b3, _oi0, _oi1, _oi2, _oi3, _xi0, _xi1, _xi2, _xi3);
                _b0 = svneg_f64_x(_pg, _b0);
                _b1 = svneg_f64_x(_pg, _b1);
                _b2 = svneg_f64_x(_pg, _b2);
                _b3 = svneg_f64_x(_pg, _b3);
                _bncsve2_rqd_add(_pg, &_pr0, &_pr1, &_pr2, &_pr3, _a0, _a1, _a2, _a3, _b0, _b1, _b2, _b3);
                _bncsve2_rqd_mul(_pg, &_a0, &_a1, &_a2, &_a3, _or0, _or1, _or2, _or3, _xi0, _xi1, _xi2, _xi3);
                _bncsve2_rqd_mul(_pg, &_b0, &_b1, &_b2, &_b3, _oi0, _oi1, _oi2, _oi3, _xr0, _xr1, _xr2, _xr3);
                _bncsve2_rqd_add(_pg, &_pi0, &_pi1, &_pi2, &_pi3, _a0, _a1, _a2, _a3, _b0, _b1, _b2, _b3);
                svfloat64_t _nr0, _nr1, _nr2, _nr3;
                svfloat64_t _ni0, _ni1, _ni2, _ni3;
                _bncsve2_rqd_add(_pg, &_nr0, &_nr1, &_nr2, &_nr3, _er0, _er1, _er2, _er3, _pr0, _pr1, _pr2, _pr3);
                _bncsve2_rqd_add(_pg, &_ni0, &_ni1, &_ni2, &_ni3, _ei0, _ei1, _ei2, _ei3, _pi0, _pi1, _pi2, _pi3);
                svst1_f64(_pg, &(in_coef_old_real->element[0][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old_real->element[0][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old_imag->element[0][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old_imag->element[0][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old_real->element[1][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old_real->element[1][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old_imag->element[1][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old_imag->element[1][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old_real->element[2][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old_real->element[2][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old_imag->element[2][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old_imag->element[2][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old_real->element[3][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old_real->element[3][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old_imag->element[3][i*num_loop_unit]), _z);
                svst1_f64(_pg, &(in_coef_old_imag->element[3][i*num_loop_unit + _vl]), _z);
                svst1_f64(_pg, &(in_coef_old_real->element[0][i*_vl]), _nr0);
                svst1_f64(_pg, &(in_coef_old_imag->element[0][i*_vl]), _ni0);
                svst1_f64(_pg, &(in_coef_old_real->element[1][i*_vl]), _nr1);
                svst1_f64(_pg, &(in_coef_old_imag->element[1][i*_vl]), _ni1);
                svst1_f64(_pg, &(in_coef_old_real->element[2][i*_vl]), _nr2);
                svst1_f64(_pg, &(in_coef_old_imag->element[2][i*_vl]), _ni2);
                svst1_f64(_pg, &(in_coef_old_real->element[3][i*_vl]), _nr3);
                svst1_f64(_pg, &(in_coef_old_imag->element[3][i*_vl]), _ni3);
            }
        }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // ARM NEON
        // Broadcast complex x to vectors
        _bncneon_rqd_set1_qd(x2_real, in_x.val_re);
        _bncneon_rqd_set1_qd(x2_imag, in_x.val_im);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            // Load real coefficients for QDSIZE=4 components
            old_coef2_real[0][0] = vld1q_f64(&(in_coef_old_real->element[0][i * num_loop_unit]));
            old_coef2_real[1][0] = vld1q_f64(&(in_coef_old_real->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_real[0][1] = vld1q_f64(&(in_coef_old_real->element[1][i * num_loop_unit]));
            old_coef2_real[1][1] = vld1q_f64(&(in_coef_old_real->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_real[0][2] = vld1q_f64(&(in_coef_old_real->element[2][i * num_loop_unit]));
            old_coef2_real[1][2] = vld1q_f64(&(in_coef_old_real->element[2][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_real[0][3] = vld1q_f64(&(in_coef_old_real->element[3][i * num_loop_unit]));
            old_coef2_real[1][3] = vld1q_f64(&(in_coef_old_real->element[3][i * num_loop_unit + _BNC_D_WIDTH]));

            // Load imaginary coefficients for QDSIZE=4 components
            old_coef2_imag[0][0] = vld1q_f64(&(in_coef_old_imag->element[0][i * num_loop_unit]));
            old_coef2_imag[1][0] = vld1q_f64(&(in_coef_old_imag->element[0][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_imag[0][1] = vld1q_f64(&(in_coef_old_imag->element[1][i * num_loop_unit]));
            old_coef2_imag[1][1] = vld1q_f64(&(in_coef_old_imag->element[1][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_imag[0][2] = vld1q_f64(&(in_coef_old_imag->element[2][i * num_loop_unit]));
            old_coef2_imag[1][2] = vld1q_f64(&(in_coef_old_imag->element[2][i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef2_imag[0][3] = vld1q_f64(&(in_coef_old_imag->element[3][i * num_loop_unit]));
            old_coef2_imag[1][3] = vld1q_f64(&(in_coef_old_imag->element[3][i * num_loop_unit + _BNC_D_WIDTH]));
            
            // Separate even/odd coefficients - Real part
            a02_real[0] = vzip1q_f64(old_coef2_real[0][0], old_coef2_real[1][0]);
            a12_real[0] = vzip2q_f64(old_coef2_real[0][0], old_coef2_real[1][0]);
            a02_real[1] = vzip1q_f64(old_coef2_real[0][1], old_coef2_real[1][1]);
            a12_real[1] = vzip2q_f64(old_coef2_real[0][1], old_coef2_real[1][1]);
            a02_real[2] = vzip1q_f64(old_coef2_real[0][2], old_coef2_real[1][2]);
            a12_real[2] = vzip2q_f64(old_coef2_real[0][2], old_coef2_real[1][2]);
            a02_real[3] = vzip1q_f64(old_coef2_real[0][3], old_coef2_real[1][3]);
            a12_real[3] = vzip2q_f64(old_coef2_real[0][3], old_coef2_real[1][3]);

            // Separate even/odd coefficients - Imaginary part
            a02_imag[0] = vzip1q_f64(old_coef2_imag[0][0], old_coef2_imag[1][0]);
            a12_imag[0] = vzip2q_f64(old_coef2_imag[0][0], old_coef2_imag[1][0]);
            a02_imag[1] = vzip1q_f64(old_coef2_imag[0][1], old_coef2_imag[1][1]);
            a12_imag[1] = vzip2q_f64(old_coef2_imag[0][1], old_coef2_imag[1][1]);
            a02_imag[2] = vzip1q_f64(old_coef2_imag[0][2], old_coef2_imag[1][2]);
            a12_imag[2] = vzip2q_f64(old_coef2_imag[0][2], old_coef2_imag[1][2]);
            a02_imag[3] = vzip1q_f64(old_coef2_imag[0][3], old_coef2_imag[1][3]);
            a12_imag[3] = vzip2q_f64(old_coef2_imag[0][3], old_coef2_imag[1][3]);

            // Complex QD multiplication: a1 * x
            _bncneon_rcqd_mul(ctmp2_real, ctmp2_imag, a12_real, a12_imag, x2_real, x2_imag);

            // Complex QD addition: (a1 * x) + a0
            _bncneon_rcqd_add(new_coef2_real, new_coef2_imag, ctmp2_real, ctmp2_imag, a02_real, a02_imag);
            
            // Store zeros to clear array
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

            // Store results
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

        //return estrin(x * x, coef, num_in_coef / 2 - 1);
        //in_x = in_x * in_x;
        rcqd_mul(&ctmp, &in_x, &in_x);
        rcqd_set(&in_x, &ctmp);
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
    rqd_set(in_ret.val_re, get_qdvector_i(in_coef_old_real, _BNC_D_WIDTH - 1));
    rqd_set(in_ret.val_im, get_qdvector_i(in_coef_old_imag, _BNC_D_WIDTH - 1));
    for(i = _BNC_D_WIDTH - 2; i >= 0; i--)
    {
        //in_ret = in_ret * in_x + (get_dvector_i(in_coef_old_real, i) + get_dvector_i(in_coef_old_imag, i) * I);
        rcqd_mul(&ctmp, &in_ret, &in_x);
        rqd_set(in_ret.val_re, get_qdvector_i(in_coef_old_real, i));
        rqd_set(in_ret.val_im, get_qdvector_i(in_coef_old_imag, i));
        rcqd_add(&in_ret, &in_ret, &ctmp);
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2: final complex fold over vl coeffs
    {
        long _vl = (long)svcntd();
        rqd_set(in_ret.val_re, get_qdvector_i(in_coef_old_real, _vl - 1));
        rqd_set(in_ret.val_im, get_qdvector_i(in_coef_old_imag, _vl - 1));
        for(i = _vl - 2; i >= 0; i--)
        {
            rcqd_mul(&ctmp, &in_ret, &in_x);
            rqd_set(in_ret.val_re, get_qdvector_i(in_coef_old_real, i));
            rqd_set(in_ret.val_im, get_qdvector_i(in_coef_old_imag, i));
            rcqd_add(&in_ret, &in_ret, &ctmp);
        }
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

    //ret->re = creal(in_ret);
    //ret->im = cimag(in_ret);
    rcqd_set(ret, &in_ret);

    //free(in_coef_old);
    //free(in_coef_new);
    free_qdvector(in_coef_old_real);
    free_qdvector(in_coef_old_imag);
    //free_dvector(in_coef_new);

    //return ret;
    return;
}

// eval based on Hessenberg matrix
// mphes: Upper Hessenberg matrix
// return |hes - x * I|
void test_ceval_hes_qdmatrix(cqdfloat *ret, QDMatrix hes, cqdfloat *x)
{
	long int i, j;
	cqdfloat *x_array;
	cqdfloat axsum, tmp;
    double coef[QDSIZE], rtmp[QDSIZE];

	x_array = (cqdfloat *)calloc(hes->row_dim, sizeof(cqdfloat));

    //x_carray[degree - 1] = mpcomplex(mpreal(1UL), mpreal(0UL)); // mpreal(1UL); // one;
    rcqd_set_ui_ui(&(x_array[hes->row_dim - 1]), 1UL, 0UL);
    for(i = hes->row_dim - 2; i >= 0; i--)
    {
        rcqd_set_ui_ui(&axsum, 0UL, 0UL); //= mpcomplex(mpreal(0UL), mpreal(0UL));
        for(j = i + 1; j < hes->col_dim; j++)
		{
            //axsum += hes[RMIJ(i + 1, j, degree, degree)] * x_carray[j];
			rcqd_mul_qd(&tmp, &(x_array[j]), get_qdmatrix_ij(hes, i + 1, j));
			rcqd_add(&axsum, &axsum, &tmp);
        }

       	// x_carray[i] = (cx * x_carray[i + 1] - caxsum) / hes[RMIJ(i + 1, i, degree, degree)];
		rcqd_mul(&tmp, x, &(x_array[i + 1]));
		rcqd_sub(&tmp, &tmp, &axsum);
		rcqd_div_qd(&tmp, &tmp, get_qdmatrix_ij(hes, i + 1, i));
		rcqd_set(&(x_array[i]), &tmp);
        //mpfr_printf("x_carray[%d] = %25.17RNe\n", i, mpfr_ptr(x_carray[i].real()));
    }
    // axsum == p(x)
    //caxsum = mpcomplex(mpreal(0UL), mpreal(0UL)); // zero; // x * x_array[0];
	rcqd_set_ui_ui(ret, 0UL, 0UL);
    for(i = 0; i < hes->row_dim; i++)
	{
        //caxsum += hes[RMIJ(0, i, degree, degree)] * x_carray[i];
		rcqd_mul_qd(&tmp, &(x_array[i]), get_qdmatrix_ij(hes, 0, i));
		rcqd_add(ret, ret, &tmp);
    }

    //caxsum = cx * x_carray[0] - caxsum;
	rcqd_mul(&tmp, x, &(x_array[0]));
	rcqd_sub(ret, &tmp, ret);

    // coef := (-1)^n * a_12 * a_23 * ... * a_n-1,n
    rqd_set_ui(coef, 1UL);
    for(i = 1; i < hes->row_dim; i++)
    {
        rqd_mul(rtmp, coef, get_qdmatrix_ij(hes, i, i - 1));
        rqd_set(coef, rtmp);
	}
    if(hes->row_dim % 2 == 1) rqd_neg(coef, coef);

    // ret := coef * ret
    rcqd_mul_qd(&tmp, ret, coef);
    rcqd_set(ret, &tmp);

	free(x_array);
}

#ifdef __cpluplus
} // extern "C" {
#endif // __cplusplus
