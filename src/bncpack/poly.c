/********************************************************************************/
/* poly.c: Algebraic Equations and Polynomials                                  */
/* copyright (c) 2002-2012 Tomonori Kouya                                       */
/*                                                                              */
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

#include "bnc.h"

/*************************************************/
/* Functions for Polynomial Types                */
/*                                               */
/* Initialize:                                   */
/*   FPoly init_fpoly(long int max_length)       */
/*   DPoly init_dpoly(long int max_length)       */
/*   MPFPoly init_mpfpoly(long int max_length)   */
/*   MPFPoly init2_mpfpoly(long int max_length, unsigned long prec)*/
/* Free:                                         */
/*   void free_fpoly(FPoly pol)                  */
/*   void free_dpoly(DPoly pol)                  */
/*   void free_mpfpoly(MPFPoly pol)              */
/* Get & Set Values:                             */
/*   float get_fpoly_i(FPoly pol, long int index) */
/*   double get_dpoly_i(DPoly pol, long int index) */
/*   mpf_t *get_mpfpoly_i(MPFPoly pol, long int index) */
/*   long int setdegree_fpoly(FPoly)             */
/*   long int setdegree_dpoly(DPoly)             */
/*   long int setdegree_mpfpoly(MPFPoly)         */
/*   void set_fpoly_i(FPoly pol, long int index, float val) */
/*   void set_dpoly_i(DPoly pol, long int index, double val) */
/*   void set_mpfpoly_i(MPFPoly pol, long int index, mpf_t val) */
/*   void set_mpfpoly_i_d(MPFPoly pol, long int index, double val) */
/* Output:                                       */
/*   void print_fpoly(FPoly pol)                 */
/*   void print_dpoly(DPoly pol)                 */
/*   void print_mpfpoly(MPFPoly pol)             */
/*   void print_fdmpfpoly(FPoly fv, DPoly dv, MPFPoly mpfv) */
/*************************************************/
FPoly init_fpoly(long int max_length)
{
	FPoly ret = NULL;
	long int i;

	if(max_length <= 0)
	{
		fprintf(stderr, "ERROR: init_fpoly\n");
		return ret;
	}

	ret = (FPoly)malloc(sizeof(fpoly));
	if(ret == NULL)
		return ret;

	ret->coef = (float *)calloc(sizeof(float), max_length);
	if(ret->coef == NULL)
		return ret;

	/* All 0 */
	for(i = 0; i < max_length; i++)
		*(ret->coef + i) = 0.0;

	ret->deg = 0;
	ret->max_len = max_length;

	return ret;
}

DPoly init_dpoly(long int max_length)
{
	DPoly ret = NULL;
	long int i;

	if(max_length <= 0)
	{
		fprintf(stderr, "ERROR: init_dpoly\n");
		return ret;
	}

	ret = (DPoly)malloc(sizeof(dpoly));
	if(ret == NULL)
		return ret;

	ret->coef = (double *)calloc(sizeof(double), max_length);
	if(ret->coef == NULL)
		return ret;

	/* All 0 */
	for(i = 0; i < max_length; i++)
		*(ret->coef + i) = 0.0;

	ret->deg = 0;
	ret->max_len = max_length;

	return ret;
}

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

	ret->coef = (mpf_t *)calloc(sizeof(mpf_t), max_length);
//	ret->coef = (mpf_t *)malloc(sizeof(mpf_t) * max_length);
	if(ret->coef == NULL)
		return NULL;

	/* All 0 */
	for(i = 0; i < max_length; i++)
	{
		mpf_init((mpf_ptr)(ret->coef + i));
		mpf_set_ui((mpf_ptr)(ret->coef + i), 0UL);
		if((ret->coef+i) == NULL)
			return NULL;
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

	ret->coef = (mpf_t *)calloc(sizeof(mpf_t), max_length);
//	ret->coef = (mpf_t *)malloc(sizeof(mpf_t) * max_length);
	if(ret->coef == NULL)
		return NULL;

	/* All 0 */
	for(i = 0; i < max_length; i++)
	{
		mpf_init2((mpf_ptr)(ret->coef + i), prec);
		mpf_set_ui((mpf_ptr)(ret->coef + i), 0UL);
		if((ret->coef+i) == NULL)
			return NULL;
	}

	mpf_init2(ret->zero, prec);
	mpf_set_ui(ret->zero, 0UL);

	ret->deg = 0;
	ret->max_len = max_length;

	ret->prec = prec;

	return ret;
}
#endif

void free_fpoly(FPoly pol)
{
	if(pol == NULL)
		return;

	if(pol->coef != NULL)
		free(pol->coef);

	free(pol);
}
void free_dpoly(DPoly pol)
{
	if(pol == NULL)
		return;

	if(pol->coef != NULL)
		free(pol->coef);

	free(pol);
}

#ifdef USE_GMP
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
		mpf_clear(pol->zero);
		free(pol->coef); // Fix! 2012-06-03 by T.Kouya
	}

//	free(&(pol->deg));
//	free(&(pol->prec));
	free(pol);

}
#endif

float get_fpoly_i(FPoly pol, long int index)
{
	if(index > pol->deg)
		return 0;
	else
		return *(pol->coef + index);
}

double get_dpoly_i(DPoly pol, long int index)
{
	if(index > pol->deg)
		return 0;
	else
		return *(pol->coef + index);
}

#ifdef USE_GMP
mpf_ptr get_mpfpoly_i(MPFPoly pol, long int index)
{
	if(index > pol->deg)
		return pol->zero;
	else
		return *(pol->coef + index);
}
#endif

/* find maximum degree at pol->coef[deg] != 0 */
long int setdegree_fpoly(FPoly pol)
{
	long int i;

	for(i = pol->max_len - 1; i > 0; i--)
	{
		if(get_fpoly_i(pol, i) != 0)
		{
			pol->deg = i;
			return i;
		}
	}

	pol->deg = 0;
	return 0;
}

/* find maximum degree at pol->coef[deg] != 0 */
long int setdegree_dpoly(DPoly pol)
{
	long int i;

	for(i = pol->max_len - 1; i > 0; i--)
	{
		if(get_dpoly_i(pol, i) != 0)
		{
			pol->deg = i;
			return i;
		}
	}

	pol->deg = 0;
	return 0;
}

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
#endif

void set_fpoly_i(FPoly pol, long int index, float val)
{
	if(index >= pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	*(pol->coef + index) = val;
	if((pol->deg < index) && (val != 0))
		pol->deg = index;
}

void set_dpoly_i(DPoly pol, long int index, double val)
{
	if(index >= pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	*(pol->coef + index) = val;
	if((pol->deg < index) && (val != 0))
		pol->deg = index;
}

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
#endif

void print_fpoly(FPoly fv)
{
	long int i;

	for(i = 0; i <= fv->deg; i++)
		printf("%5ld %15.7e\n", i, get_fpoly_i(fv, i));
}

void print_dpoly(DPoly dv)
{
	long int i, deg;

	for(i = 0; i <= dv->deg; i++)
		printf("%5ld %25.17e\n", i, get_dpoly_i(dv, i));
}

void print_fdpoly(FPoly fv, DPoly dv)
{
	long int i, deg;

	deg = fv->deg;
	if(deg > dv->deg)
		deg = dv->deg;

	for(i = 0; i <= deg; i++)
		printf("%5ld %15.7e %25.17e\n", i, get_fpoly_i(fv, i), get_dpoly_i(dv, i));
}

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
#endif

/*************************************************/
/* Poly Calculations for FPoly               */
/*
void add_fpoly(FPoly c, FPoly a, FPoly b)
void sub_fpoly(FPoly c, FPoly a, FPoly b)
void cmul_fpoly(FPoly c, float val, FPoly a)
void subst_fpoly(FPoly c, FPoly a)

void diff_fpoly(FPoly a)
float eval_fpoly(FPoly a, float x)
float eval_diff_fpoly(FPoly a, float x)
*/
/*************************************************/
/* c = a + b */
void add_fpoly(FPoly c, FPoly a, FPoly b)
{
	long int i;

	if((c->max_len < a->deg) || (c->max_len < b->deg))
	{
		fprintf(stderr, "ERROR: add_fpoly\n");
		return;
	}

	c->deg = (a->deg > b->deg) ? a->deg : b->deg;

	for(i = 0; i <= c->deg; i++)
		set_fpoly_i(c, i, get_fpoly_i(a, i) + get_fpoly_i(b, i));

}

/* c += a */
void add2_fpoly(FPoly c, FPoly a)
{
	long int i;

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: add2_fpoly\n");
		return;
	}

	c->deg = (c->deg > a->deg) ? c->deg : a->deg;

	for(i = 0; i <= c->deg; i++)
		set_fpoly_i(c, i, get_fpoly_i(c, i) + get_fpoly_i(a, i));

}

/* c = a - b */
void sub_fpoly(FPoly c, FPoly a, FPoly b)
{
	long int i;

	if((c->max_len < a->deg) || (c->max_len < b->deg))
	{
		fprintf(stderr, "ERROR: sub_fpoly\n");
		return;
	}

	c->deg = (a->deg > b->deg) ? a->deg : b->deg;

	for(i = 0; i <= c->deg; i++)
		set_fpoly_i(c, i, get_fpoly_i(a, i) - get_fpoly_i(b, i));

}

/* c -= a */
void sub2_fpoly(FPoly c, FPoly a)
{
	long int i;

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: sub2_fpoly\n");
		return;
	}

	c->deg = (c->deg > a->deg) ? c->deg : a->deg;

	for(i = 0; i <= c->deg; i++)
		set_fpoly_i(c, i, get_fpoly_i(c, i) - get_fpoly_i(a, i));

}

/* c = val * a */
void cmul_fpoly(FPoly c, float val, FPoly a)
{
	long int i;

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: cmul_fpoly\n");
		return;
	}

	// fix!!: 2006-12-20
	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
		set_fpoly_i(c, i, val * get_fpoly_i(a, i));

}

/* c *= val */
void cmul2_fpoly(FPoly c, float val)
{
	long int i;

	for(i = 0; i <= c->deg; i++)
		set_fpoly_i(c, i, val * get_fpoly_i(c, i));

}

/* c := a */
void subst_fpoly(FPoly c, FPoly a)
{
	long int i;

	if(c->deg > a->deg)
	{
		for(i = c->deg; i >= a->deg; i--)
			set_fpoly_i(c, i, 0);
	}

	c->deg = a->deg;
	for(i = 0; i <= c->deg; i++)
		set_fpoly_i(c, i, get_fpoly_i(a, i));
}

/* c := 0 */
void set0_fpoly(FPoly c)
{
	long int i;

	for(i = 0; i <= c->deg; i++)
		set_fpoly_i(c, i, (float)0);
}

/* number of nonzero coef */
long int num_nonzero_fpoly(FPoly c)
{
	long int i, ret;
	float tmp;

	ret = 0;
	for(i = 0; i <= c->deg; i++)
	{
		tmp = get_fpoly_i(c, i);
		if(tmp != 0.0)
			ret++;
	}

	return ret;
}

/* get maximum |coef| */
long int max_abscoef_fpoly(FPoly c)
{
	long int i, ret;
	float tmp1, tmp;

	tmp1 = (float)fabs((double)get_fpoly_i(c, 0));
	ret = 0;
	for(i = 1; i <= c->deg; i++)
	{
		tmp = (float)fabs((double)get_fpoly_i(c, i));
		if(tmp1 < tmp)
		{
			tmp1 = tmp;
			ret = i;
		}
	}

	return ret;
}

/* a := a'(x) */
void diff_fpoly(FPoly a)
{
	long int diff_deg, i;
	float tmp_coef;

	diff_deg = a->deg - 1;

	if(diff_deg < 0)
	{
		set0_fpoly(a);
		a->deg = 0;
		return;
	}

	for(i = 1; i <= diff_deg; i++)
	{
		tmp_coef = (float)i * get_fpoly_i(a, i);
		set_fpoly_i(a, i - 1, tmp_coef);
	}
	a->deg = diff_deg;
}

/* value of a(x) */
float eval_fpoly(FPoly a, float x)
{
	long int i;
	float ret;

	ret = get_fpoly_i(a, a->deg);
	for(i = a->deg - 1; i >= 0; i--)
	{
		ret = ret * x + get_fpoly_i(a, i);
	}

	return ret;
}

/* value of a'(x) */
float eval_diff_fpoly(FPoly a, float x)
{
	long int i;
	float ret;

	ret = (float)a->deg * get_fpoly_i(a, a->deg);
	for(i = a->deg - 1; i >= 1; i--)
	{
		ret = ret * x + (float)i * get_fpoly_i(a, i);
	}

	return ret;
}

/* complex value of a(x) */
void ceval_fpoly(FCmplx ret, FPoly a, FCmplx x)
{
	long int i;

	set0_fcmplx(ret);
	set_real_fcmplx(ret, get_fpoly_i(a, a->deg));

	for(i = a->deg - 1; i >= 0; i--)
	{
		/* ret = ret * x + get_fpoly_i(a, i) */
		mul2_fcmplx(ret, x);
		add_fcmplx_f(ret, ret, get_fpoly_i(a, i));
	}

}

/* complex value of a'(x) */
void ceval_diff_fpoly(FCmplx ret, FPoly a, FCmplx x)
{
	long int i;
	float tmp;

	set0_fcmplx(ret);
	tmp = get_fpoly_i(a, a->deg) * (float)a->deg;
	set_real_fcmplx(ret, tmp);
	for(i = a->deg - 1; i >= 1; i--)
	{
		/* ret = ret * x + (float)i * get_fpoly_i(a, i) */
		mul2_fcmplx(ret, x);
		tmp = (float)i * get_fpoly_i(a, i);
		add_fcmplx_f(ret, ret, tmp);
	}
}


/*************************************************/
/* Poly Calculations for DPoly               */
/*
void add_dpoly(DPoly c, DPoly a, DPoly b)
void sub_dpoly(DPoly c, DPoly a, DPoly b)
void cmul_dpoly(DPoly c, double val, DPoly a)
void subst_dpoly(DPoly c, DPoly a)

void diff_dpoly(DPoly a)
double eval_dpoly(DPoly a, double x)
double eval_diff_dpoly(dPoly a, double x)
*/
/*************************************************/
/* c = a + b */
void add_dpoly(DPoly c, DPoly a, DPoly b)
{
	long int i;

	if((c->max_len < a->deg) || (c->max_len < b->deg))
	{
		fprintf(stderr, "ERROR: add_dpoly\n");
		return;
	}

	c->deg = (a->deg > b->deg) ? a->deg : b->deg;

	for(i = 0; i <= c->deg; i++)
		set_dpoly_i(c, i, get_dpoly_i(a, i) + get_dpoly_i(b, i));

}

/* c += a */
void add2_dpoly(DPoly c, DPoly a)
{
	long int i;

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: add2_dpoly\n");
		return;
	}

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
		set_dpoly_i(c, i, get_dpoly_i(c, i) + get_dpoly_i(a, i));

}

/* c = a - b */
void sub_dpoly(DPoly c, DPoly a, DPoly b)
{
	long int i;

	if((c->max_len < a->deg) || (c->max_len < b->deg))
	{
		fprintf(stderr, "ERROR: sub_dpoly\n");
		return;
	}

	c->deg = (a->deg > b->deg) ? a->deg : b->deg;

	for(i = 0; i <= c->deg; i++)
		set_dpoly_i(c, i, get_dpoly_i(a, i) - get_dpoly_i(b, i));

}

/* c -= a */
void sub2_dpoly(DPoly c, DPoly a)
{
	long int i;

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: sub2_dpoly\n");
		return;
	}

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
		set_dpoly_i(c, i, get_dpoly_i(c, i) - get_dpoly_i(a, i));

}

/* c = a * b */
void mul_dpoly(DPoly c, DPoly a, DPoly b)
{
	long int i, j;
	double tmp;

	if(c->max_len < (a->deg + b->deg))
	{
		fprintf(stderr, "ERROR: mul_dpoly\n");
		return;
	}

	/* set c = 0 */
	set0_dpoly(c);

	for(i = 0; i <= a->deg; i++)
	{
		for(j = 0; j <= b->deg; j++)
		{
			tmp = get_dpoly_i(c, i + j);
			tmp += get_dpoly_i(a, i) * get_dpoly_i(b, j);
			set_dpoly_i(c, i + j, tmp);
		}
	}

	c->deg = setdegree_dpoly(c);

}

/* c = val * a */
void cmul_dpoly(DPoly c, double val, DPoly a)
{
	long int i;

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: cmul_dpoly\n");
		return;
	}

	// fix!!: 2006-12-20
	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
		set_dpoly_i(c, i, val * get_dpoly_i(a, i));

}

/* c *= val */
void cmul2_dpoly(DPoly c, double val)
{
	long int i;

	for(i = 0; i <= c->deg; i++)
		set_dpoly_i(c, i, val * get_dpoly_i(c, i));

}

/* c := a */
void subst_dpoly(DPoly c, DPoly a)
{
	long int i;

	if(c->deg > a->deg)
	{
		for(i = c->deg; i >= a->deg; i--)
			set_dpoly_i(c, i, 0);
	}

	c->deg = a->deg;
	for(i = 0; i <= c->deg; i++)
		set_dpoly_i(c, i, get_dpoly_i(a, i));
}

/* c := 0 */
void set0_dpoly(DPoly c)
{
	long int i;

	for(i = 0; i <= c->deg; i++)
		set_dpoly_i(c, i, (double)0);
}

/* number of nonzero coef */
long int num_nonzero_dpoly(DPoly c)
{
	long int i, ret;
	double tmp;

	ret = 0;
	for(i = 0; i <= c->deg; i++)
	{
		tmp = get_dpoly_i(c, i);
		if(tmp != 0.0)
			ret++;
	}

	return ret;
}

/* get maximum |coef| */
long int max_abscoef_dpoly(DPoly c)
{
	long int i, ret;
	double tmp1, tmp;

	tmp1 = fabs(get_dpoly_i(c, 0));
	ret = 0;
	for(i = 1; i <= c->deg; i++)
	{
		tmp = fabs(get_dpoly_i(c, i));
		if(tmp1 < tmp)
		{
			tmp1 = tmp;
			ret = i;
		}
	}

	return ret;
}

/* a := a'(x) */
void diff_dpoly(DPoly a)
{
	long int diff_deg, i;
	double tmp_coef;

	diff_deg = a->deg - 1;

	if(diff_deg < 0)
	{
		set0_dpoly(a);
		a->deg = 0;
		return;
	}

	for(i = 1; i <= diff_deg; i++)
	{
		tmp_coef = (double)i * get_dpoly_i(a, i);
		set_dpoly_i(a, i - 1, tmp_coef);
	}
	a->deg = diff_deg;
}

/* value of a(x) */
double eval_dpoly(DPoly a, double x)
{
	long int i;
	double ret;

	ret = get_dpoly_i(a, a->deg);
	for(i = a->deg - 1; i >= 0; i--)
	{
		ret = ret * x + get_dpoly_i(a, i);
	}

	return ret;
}

/* value of a'(x) */
double eval_diff_dpoly(DPoly a, double x)
{
	long int i;
	double ret;

	ret = (double)a->deg * get_dpoly_i(a, a->deg);
	for(i = a->deg - 1; i >= 1; i--)
	{
		ret = ret * x + (double)i * get_dpoly_i(a, i);
	}

	return ret;
}

/* complex value of a(x) */
void ceval_dpoly(DCmplx ret, DPoly a, DCmplx x)
{
	long int i;

	set0_dcmplx(ret);
	set_real_dcmplx(ret, get_dpoly_i(a, a->deg));

	for(i = a->deg - 1; i >= 0; i--)
	{
		/* ret = ret * x + get_dpoly_i(a, i) */
		mul2_dcmplx(ret, x);
		add_dcmplx_d(ret, ret, get_dpoly_i(a, i));
	}

}

/* complex value of a'(x) */
void ceval_diff_dpoly(DCmplx ret, DPoly a, DCmplx x)
{
	long int i;
	double tmp;

	set0_dcmplx(ret);
	tmp = get_dpoly_i(a, a->deg) * (double)a->deg;
	set_real_dcmplx(ret, tmp);
	for(i = a->deg - 1; i >= 1; i--)
	{
		/* ret = ret * x + (double)i * get_dpoly_i(a, i) */
		mul2_dcmplx(ret, x);
		tmp = (double)i * get_dpoly_i(a, i);
		add_dcmplx_d(ret, ret, tmp);
	}
}

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
void eval_mpfpoly(mpf_t ret, MPFPoly a, mpf_t x)
{
	long int i;

	mpf_set(ret, get_mpfpoly_i(a, a->deg));
	for(i = a->deg - 1; i >= 0; i--)
	{
		mpf_mul(ret, ret, x);
		mpf_add(ret, ret, get_mpfpoly_i(a, i));
	}
}

/* value of a'(x) */
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

/* complex value of a(x) */
void ceval_mpfpoly(MPFCmplx ret, MPFPoly a, MPFCmplx x)
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
void ceval_diff_mpfpoly(MPFCmplx ret, MPFPoly a, MPFCmplx x)
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
#endif
