/********************************************************************************/
/* poly.c: Algebraic Equations and Polynomials                                  */
/* copyright (c) 2002-2025 Tomonori Kouya                                       */
/*                                                                              */
/* Ver. 0.4 2025-06-12: Appended polynomial manip. with complex coefficients    */
/* Ver. 0.3 2025-01-22: Separated MPFPoly to mpf_poly.c                         */
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

/****************************************************************/
/* Functions for Polynomial Types                               */
/*                                                              */
/* Initialize:                                                  */
/*   FPoly init_fpoly(long int max_length)                      */
/*   DPoly init_dpoly(long int max_length)                      */
/*   CDPoly init_cdpoly(long int max_length)                    */
/* Free:                                                        */
/*   void free_fpoly(FPoly pol)                                 */
/*   void free_dpoly(DPoly pol)                                 */
/*   void free_cdpoly(CDPoly pol)                               */
/* Get & Set Values:                                            */
/*   float get_fpoly_i(FPoly pol, long int index)               */
/*   double get_dpoly_i(DPoly pol, long int index)              */
/*   double _Complex get_cdpoly_i(CDPoly pol, long int index)   */
/*   long int setdegree_fpoly(FPoly)                            */
/*   long int setdegree_dpoly(DPoly)                            */
/*   long int setdegree_cdpoly(CDPoly)                          */
/*   void set_fpoly_i(FPoly pol, long int index, float val)     */
/*   void set_dpoly_i(DPoly pol, long int index, double val)    */
/*   void set_cdpoly_i(CDPoly pol, long int index,              */
/*   double _Complex val)                                       */
/* Output:                                                      */
/*   void print_fpoly(FPoly pol)                                */
/*   void print_dpoly(DPoly pol)                                */
/*   void print_cdpoly(CDPoly pol)                              */
/****************************************************************/
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
	{
		free(ret);
		return ret;
	}

	/* All 0 */
	for(i = 0; i < max_length; i++)
		*(ret->coef + i) = 0.0;

	ret->deg = 0;
	ret->max_len = max_length;

	return ret;
}

// Initialize a complex polynomial
CDPoly init_cdpoly(long int max_length)
{
	CDPoly ret = NULL;
	long int i;

	if(max_length <= 0)
	{
		fprintf(stderr, "ERROR: init_cdpoly\n");
		return ret;
	}

	ret = (CDPoly)malloc(sizeof(cdpoly));
	if(ret == NULL)
		return ret;

	ret->coef = (double _Complex *)calloc(max_length, sizeof(double _Complex));
	if(ret->coef == NULL)
	{
		free(ret);
		return ret;
	}

	/* All 0 */
	for(i = 0; i < max_length; i++)
		ret->coef[i] = 0.0 + 0.0 * I;

	ret->deg = 0;
	ret->max_len = max_length;

	return ret;
}

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

// free CDPoly
void free_cdpoly(CDPoly pol)
{
	if(pol == NULL)
		return;

	if(pol->coef != NULL)
		free(pol->coef);

	free(pol);
}

// initialize and set polynomial based on org
DPoly init_set_dpoly(DPoly org)
{
    DPoly ret = NULL;

    ret = init_dpoly(org->max_len);
    if(ret == NULL)
        return NULL;

    // ret := org
    subst_dpoly(ret, org);    

    return ret;
}


// initialize and set polynomial based on org
CDPoly init_set_cdpoly(CDPoly org)
{
    CDPoly ret = NULL;

    ret = init_cdpoly(org->max_len);
    if(ret == NULL)
        return NULL;

    // ret := org
    subst_cdpoly(ret, org);    

    return ret;
}

// initialize and set polynomial based on org
CDPoly init_set_cdpoly_dpoly(DPoly org)
{
    CDPoly ret = NULL;

    ret = init_cdpoly(org->max_len);
    if(ret == NULL)
        return NULL;

    // ret := org
    set_cdpoly_dpoly(ret, org);    

    return ret;
}

// cdpoly := dpoly
void set_cdpoly_dpoly(CDPoly c, DPoly a)
{
	long int i;

	if(c->deg > a->deg)
	{
		for(i = c->deg; i >= a->deg; i--)
			set_cdpoly_i(c, i, 0.0 + 0.0 * I);
	}

	c->deg = a->deg;
	for(i = 0; i <= c->deg; i++)
		set_cdpoly_i_d(c, i, get_dpoly_i(a, i));
}


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

// return pol->coef[i]
double _Complex get_cdpoly_i(CDPoly pol, long int index)
{
	if(index > pol->deg)
		return 0.0 + 0.0 * I;
	else
		return pol->coef[index];
}

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

/* find maximum degree at pol->coef[deg] != 0 */
long int setdegree_cdpoly(CDPoly pol)
{
	long int i;
	double tmp;

	for(i = pol->max_len - 1; i > 0; i--)
	{
		tmp = cabs(get_cdpoly_i(pol, i));
		if(tmp != 0.0)
		{
			pol->deg = i;
			return i;
		}
	}

	pol->deg = 0;
	return 0;
}


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

// coef[i] = val
void set_cdpoly_i(CDPoly pol, long int index, double _Complex val)
{
	double absval;
	if(index >= pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	//*(pol->coef + index) = val;
	pol->coef[index] = val;
	//subst_dcmplx(pol->coef[index], val);
	//set_real_dcmplx(pol->coef[index], creal(val));
	//set_image_dcmplx(pol->coef[index], cimag(val));

	absval = cabs(val);

	if((pol->deg < index) && (absval != 0.0))
		pol->deg = index;
}

// coef[i] = val
void set_cdpoly_i_d(CDPoly pol, long int index, double val)
{
	double absval;
	if(index >= pol->max_len)
	{
		fprintf(stderr, "ERROR: Too large index!\n");
		return;
	}

	//*(pol->coef + index) = val;
	pol->coef[index] = val + 0.0 * I;

	if((pol->deg < index) && (val != 0.0))
		pol->deg = index;
}

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

// print cdpoly
void print_cdpoly(CDPoly dv)
{
	long int i, deg;
	double _Complex ctmp;

	for(i = 0; i <= dv->deg; i++)
	{
		ctmp = get_cdpoly_i(dv, i);
		printf("%5ld %25.17e + %25.17e * I\n", i, creal(ctmp), cimag(ctmp));
	}
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
// Based on Horner method
double eval_dpoly_horner(DPoly a, double x)
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

// Division of polynomial
// a(x) / b(x) = quo(x) * b(x) + rem(x)
void div_dpoly(DPoly quo, DPoly rem, DPoly a, DPoly b)
{
    long int i, diff_deg;
    DPoly dividend, divisor;
    double new_coef_div, new_coef;

    // Check degs
    // quo := 0, rem := a
    if(a->deg < b->deg)
    {
        set0_dpoly(quo);
        subst_dpoly(rem, a);

        return;
    }

    // in case of a->deg >= b->deg
    set0_dpoly(quo);
    set0_dpoly(rem);

    // a->deg > b->deg
    dividend = init_set_dpoly(a);
    divisor  = init_set_dpoly(b);
    do {
        diff_deg = dividend->deg - divisor->deg;
        //printf("dividend, divisor->deg, diff: %5ld, %5ld, %5ld\n", dividend->deg, divisor->deg, diff_deg);

        // set new coefs
        new_coef_div = get_dpoly_i(dividend, dividend->deg) / get_dpoly_i(divisor, divisor->deg);
        set_dpoly_i(quo, diff_deg, new_coef_div);
        set_dpoly_i(dividend, dividend->deg, 0.0); // a(deg) := 0
        for(i = 0; i < divisor->deg; i++)
        {
            new_coef = get_dpoly_i(dividend, i + diff_deg) - new_coef_div * get_dpoly_i(divisor, i);
            set_dpoly_i(dividend, i + diff_deg, new_coef);
        }

        // set degrees
        setdegree_dpoly(quo);
        setdegree_dpoly(dividend); // deg--

    } while(diff_deg > 0);

    // set quotient and remainder
    subst_dpoly(rem, dividend);

    // clear
    free_dpoly(dividend);
    free_dpoly(divisor);
}

/* c = x^n * a */
// c[0] = ... = c[n-1] = 0
// c[n] = a[0], ..., c[deg + n] = a[deg]
void xpow_mul_dpoly(DPoly c, long int npow, DPoly a)
{
	long int i;

	if(c->max_len < (a->deg + npow))
	{
		fprintf(stderr, "ERROR: xpow_mul_dpoly\n");
		return;
	}

	// fix!!: 2006-12-20
	c->deg = a->deg + npow; //((a->deg + npow) > c->deg) ? (a->deg + npow) : c->deg;

    // c[0] = c[1] = ... c[npow - 1] = 0
    for(i = 0; i < npow; i++)
        c->coef[i] = 0.0;

	for(i = 0; i <= a->deg; i++)
		set_dpoly_i(c, i + npow, get_dpoly_i(a, i));

    //setdegree_dpoly(c);

}
/****************************************************************/
/* Manipulate CDPoly                                            */
/*                                                              */
/* void add_cdpoly(CDPoly c, CDPoly a, CDPoly b)                */
/* void sub_cdpoly(CDPoly c, CDPoly a, CDPoly b)                */
/* void cmul_cdpoly(CDPoly c, double _Complex val, CDPoly a)    */
/* void subst_cdpoly(CDPoly c, CDPoly a)                        */
/*                                                              */
/* void diff_cdpoly(CDPoly a)                                   */
/* double _Complex eval_cdpoly(CDPoly a, double _Complex x)     */
/* double _Complex eval_diff_cdpoly(dPoly a, double _Complex x) */
/*                                                              */
/****************************************************************/
/* c = a + b */
void add_cdpoly(CDPoly c, CDPoly a, CDPoly b)
{
	long int i;

	if((c->max_len < a->deg) || (c->max_len < b->deg))
	{
		fprintf(stderr, "ERROR: add_cdpoly\n");
		return;
	}

	c->deg = (a->deg > b->deg) ? a->deg : b->deg;

	for(i = 0; i <= c->deg; i++)
		set_cdpoly_i(c, i, get_cdpoly_i(a, i) + get_cdpoly_i(b, i));

}

/* c += a */
void add2_cdpoly(CDPoly c, CDPoly a)
{
	long int i;

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: add2_cdpoly\n");
		return;
	}

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
		set_cdpoly_i(c, i, get_cdpoly_i(c, i) + get_cdpoly_i(a, i));

}

/* c = a - b */
void sub_cdpoly(CDPoly c, CDPoly a, CDPoly b)
{
	long int i;

	if((c->max_len < a->deg) || (c->max_len < b->deg))
	{
		fprintf(stderr, "ERROR: sub_cdpoly\n");
		return;
	}

	c->deg = (a->deg > b->deg) ? a->deg : b->deg;

	for(i = 0; i <= c->deg; i++)
		set_cdpoly_i(c, i, get_cdpoly_i(a, i) - get_cdpoly_i(b, i));

}

/* c -= a */
void sub2_cdpoly(CDPoly c, CDPoly a)
{
	long int i;

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: sub2_cdpoly\n");
		return;
	}

	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
		set_cdpoly_i(c, i, get_cdpoly_i(c, i) - get_cdpoly_i(a, i));

}

/* c = a * b */
void mul_cdpoly(CDPoly c, CDPoly a, CDPoly b)
{
	long int i, j;
	double _Complex tmp;

	if(c->max_len < (a->deg + b->deg))
	{
		fprintf(stderr, "ERROR: mul_cdpoly\n");
		return;
	}

	/* set c = 0 */
	set0_cdpoly(c);

	for(i = 0; i <= a->deg; i++)
	{
		for(j = 0; j <= b->deg; j++)
		{
			tmp = get_cdpoly_i(c, i + j);
			tmp += get_cdpoly_i(a, i) * get_cdpoly_i(b, j);
			set_cdpoly_i(c, i + j, tmp);
		}
	}

	c->deg = setdegree_cdpoly(c);
}

// ret := ret * a
void mul2_cdpoly(CDPoly ret, CDPoly a)
{
	CDPoly tmp;

	tmp = init_set_cdpoly(ret);
	mul_cdpoly(ret, tmp, a);
	free_cdpoly(tmp);
}

/* c = val * a */
void cmul_cdpoly(CDPoly c, double _Complex val, CDPoly a)
{
	long int i;

	if(c->max_len < a->deg)
	{
		fprintf(stderr, "ERROR: cmul_cdpoly\n");
		return;
	}

	// fix!!: 2006-12-20
	c->deg = (a->deg > c->deg) ? a->deg : c->deg;

	for(i = 0; i <= c->deg; i++)
		set_cdpoly_i(c, i, val * get_cdpoly_i(a, i));

}

/* c *= val */
void cmul2_cdpoly(CDPoly c, double _Complex val)
{
	long int i;

	for(i = 0; i <= c->deg; i++)
		set_cdpoly_i(c, i, val * get_cdpoly_i(c, i));

}

/* c := a */
void subst_cdpoly(CDPoly c, CDPoly a)
{
	long int i;

	if(c->deg > a->deg)
	{
		for(i = c->deg; i >= a->deg; i--)
			set_cdpoly_i(c, i, 0);
	}

	c->deg = a->deg;
	for(i = 0; i <= c->deg; i++)
		set_cdpoly_i(c, i, get_cdpoly_i(a, i));
}

/* c := 0 */
void set0_cdpoly(CDPoly c)
{
	long int i;

	for(i = 0; i <= c->deg; i++)
		set_cdpoly_i(c, i, 0.0 + 0.0 * I);
}

/* number of nonzero coef */
long int num_nonzero_cdpoly(CDPoly c)
{
	long int i, ret;
	double tmp;

	ret = 0;
	for(i = 0; i <= c->deg; i++)
	{
		tmp = cabs(get_cdpoly_i(c, i));
		if(tmp != 0.0)
			ret++;
	}

	return ret;
}

/* get maximum |coef| */
long int max_abscoef_cdpoly(CDPoly c)
{
	long int i, ret;
	double tmp1, tmp;

	tmp1 = cabs(get_cdpoly_i(c, 0));
	ret = 0;
	for(i = 1; i <= c->deg; i++)
	{
		tmp = cabs(get_cdpoly_i(c, i));
		if(tmp1 < tmp)
		{
			tmp1 = tmp;
			ret = i;
		}
	}

	return ret;
}

/* a := a'(x) */
void diff_cdpoly(CDPoly a)
{
	long int diff_deg, i;
	double _Complex tmp_coef;

	diff_deg = a->deg - 1;

	if(diff_deg < 0)
	{
		set0_cdpoly(a);
		a->deg = 0;
		return;
	}

	for(i = 1; i <= diff_deg; i++)
	{
		tmp_coef = (double)i * get_cdpoly_i(a, i);
		set_cdpoly_i(a, i - 1, tmp_coef);
	}
	a->deg = diff_deg;
}

/* value of a(x) */
// Based on Horner method
double _Complex eval_cdpoly_horner(CDPoly a, double _Complex x)
{
	long int i;
	double _Complex ret;

	ret = get_cdpoly_i(a, a->deg);
	for(i = a->deg - 1; i >= 0; i--)
	{
		ret = ret * x + get_cdpoly_i(a, i);
	}

	return ret;
}

// Evaluation of polynomial using Estrin method
double _Complex ceval_cdpoly_estrin(CDPoly a, double _Complex x)
{
    double _Complex *in_coef_old, *in_coef_new, in_x, ctmp[2], ret;
    long int in_degree, num_in_coef, i;

	//set0_dcmplx(ret);
	ret = 0.0 + 0.0 * I;
	//in_x = x->re + x->im * I;
	in_x = x;

    if(a->deg == 0) {
        return get_cdpoly_i(a, 0); // return constant;
		//set_real_dcmplx(ret, get_dpoly_i(a, 0));
		//return;
    }
    if(a->deg == 1) {
        //return get_dpoly_i(a, 0) + get_dpoly_i(a, 1) * x; // return a0 + a1 * x
		ctmp[0] = get_cdpoly_i(a, 0); // + I * 0.0;
		ctmp[1] = get_cdpoly_i(a, 1); // + I * 0.0;
		ret = ctmp[0] + ctmp[1] * in_x;
		//ret->re = creal(in_ret);
		//ret->im = cimag(in_ret);
    }

    num_in_coef = a->deg + 1;
    if((num_in_coef % 2) == 1) num_in_coef++;

    in_degree = num_in_coef - 1;
    in_coef_old = (double _Complex *)calloc(num_in_coef, sizeof(double _Complex));
    in_coef_new = (double _Complex *)calloc(num_in_coef, sizeof(double _Complex));

    for(i = 0; i <= a->deg; i++) in_coef_old[i] = get_cdpoly_i(a, i); //  + I * 0.0;
    for(i = a->deg + 1; i <= in_degree; i++) in_coef_old[i] = 0.0 + 0.0 * I;

    //in_x = x;

    //printf("degree -> in_degree: %d -> %d\n", degree, in_degree);

    while(in_degree > 1)
    {
        num_in_coef /= 2;
        for(i = 0; i < num_in_coef; i++)
        {
            in_coef_new[i] = in_coef_old[i * 2] + in_coef_old[i * 2 + 1] * in_x;
            //printf("coef[%d] = in_coef[%d] + in_coef[%d] * %f = %f\n", i, i * 2, i * 2 + 1, in_x, in_coef_new[i]);
        }

        //return estrin(x * x, coef, num_in_coef / 2 - 1);
        in_x = in_x * in_x;
        for(i = 0; i < num_in_coef; i++)
           	in_coef_old[i] = in_coef_new[i];

        if((num_in_coef % 2) == 1)
		{
            num_in_coef++;
			in_coef_old[num_in_coef - 1] = 0.0 + 0.0 * I;
		} 

        in_degree = num_in_coef - 1;

        //for(i = 0; i <= in_degree; i++)
        //    in_coef_old[i] = in_coef_new[i];

    }
    ret = in_coef_new[0] + in_coef_new[1] * in_x;
	//ret->re = creal(in_ret);
	//ret->im = cimag(in_ret);

    free(in_coef_old);
    free(in_coef_new);

    return ret;
	//return;
}

/* value of a'(x) */
// Based on Horner method
double _Complex eval_diff_cdpoly(CDPoly a, double _Complex x)
{
	long int i;
	double _Complex ret;

	ret = (double)a->deg * get_cdpoly_i(a, a->deg);
	for(i = a->deg - 1; i >= 1; i--)
	{
		ret = ret * x + (double)i * get_cdpoly_i(a, i);
	}

	return ret;
}

// Division of polynomial
// a(x) / b(x) = quo(x) * b(x) + rem(x)
void div_cdpoly(CDPoly quo, CDPoly rem, CDPoly a, CDPoly b)
{
    long int i, diff_deg;
    CDPoly dividend, divisor;
    double _Complex new_coef_div, new_coef;

    // Check degs
    // quo := 0, rem := a
    if(a->deg < b->deg)
    {
        set0_cdpoly(quo);
        subst_cdpoly(rem, a);

        return;
    }

    // in case of a->deg >= b->deg
    set0_cdpoly(quo);
    set0_cdpoly(rem);

    // a->deg > b->deg
    dividend = init_set_cdpoly(a);
    divisor  = init_set_cdpoly(b);
    do {
        diff_deg = dividend->deg - divisor->deg;
        //printf("dividend, divisor->deg, diff: %5ld, %5ld, %5ld\n", dividend->deg, divisor->deg, diff_deg);

        // set new coefs
        new_coef_div = get_cdpoly_i(dividend, dividend->deg) / get_cdpoly_i(divisor, divisor->deg);
        set_cdpoly_i(quo, diff_deg, new_coef_div);
        set_cdpoly_i(dividend, dividend->deg, 0.0); // a(deg) := 0
        for(i = 0; i < divisor->deg; i++)
        {
            new_coef = get_cdpoly_i(dividend, i + diff_deg) - new_coef_div * get_cdpoly_i(divisor, i);
            set_cdpoly_i(dividend, i + diff_deg, new_coef);
        }

        // set degrees
        setdegree_cdpoly(quo);
        setdegree_cdpoly(dividend); // deg--

    } while(diff_deg > 0);

    // set quotient and remainder
    subst_cdpoly(rem, dividend);

    // clear
    free_cdpoly(dividend);
    free_cdpoly(divisor);
}

/* c = x^n * a */
// c[0] = ... = c[n-1] = 0
// c[n] = a[0], ..., c[deg + n] = a[deg]
void xpow_mul_cdpoly(CDPoly c, long int npow, CDPoly a)
{
	long int i;

	if(c->max_len < (a->deg + npow))
	{
		fprintf(stderr, "ERROR: xpow_mul_cdpoly\n");
		return;
	}

	// fix!!: 2006-12-20
	c->deg = a->deg + npow; //((a->deg + npow) > c->deg) ? (a->deg + npow) : c->deg;

    // c[0] = c[1] = ... c[npow - 1] = 0
    for(i = 0; i < npow; i++)
        set_cdpoly_i(c, i, 0.0 + 0.0 * I);

	for(i = 0; i <= a->deg; i++)
		set_cdpoly_i(c, i + npow, get_cdpoly_i(a, i));

    //setdegree_dpoly(c);

}

/////////////////////////////////
// Evaluate polynomial
/////////////////////////////////

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
double eval_dpoly_estrin(DPoly a, double x)
{
    double *in_coef_old, *in_coef_new, in_x, ret;
    long int in_degree, num_in_coef, i;

    if(a->deg == 0) {
        return get_dpoly_i(a, 0); // return constant;
    }
    if(a->deg == 1) {
        return get_dpoly_i(a, 0) + get_dpoly_i(a, 1) * x; // return a0 + a1 * x
    }

    num_in_coef = a->deg + 1;
    if((num_in_coef % 2) == 1) num_in_coef++;

    in_degree = num_in_coef - 1;
    in_coef_old = (double *)calloc(num_in_coef, sizeof(double));
    in_coef_new = (double *)calloc(num_in_coef, sizeof(double));

    for(i = 0; i <= a->deg; i++) in_coef_old[i] = get_dpoly_i(a, i);
    for(i = a->deg + 1; i <= in_degree; i++) in_coef_old[i] = 0.0;

    in_x = x;

    //printf("degree -> in_degree: %d -> %d\n", degree, in_degree);

    while(in_degree > 1)
    {
		num_in_coef /= 2;
        for(i = 0; i < num_in_coef; i++)
        {
            in_coef_new[i] = in_coef_old[i * 2] + in_coef_old[i * 2 + 1] * in_x;
            //printf("coef[%d] = in_coef[%d] + in_coef[%d] * %f = %f\n", i, i * 2, i * 2 + 1, in_x, in_coef_new[i]);
        }

        //return estrin(x * x, coef, num_in_coef / 2 - 1);
        in_x = in_x * in_x;
        for(i = 0; i < num_in_coef; i++)
            in_coef_old[i] = in_coef_new[i];

        if((num_in_coef % 2) == 1)
		{
            num_in_coef++;
			in_coef_old[num_in_coef - 1] = 0.0;
		}

        in_degree = num_in_coef - 1;

    }
    ret = in_coef_new[0] + in_coef_new[1] * in_x;

    free(in_coef_old);
    free(in_coef_new);

    return ret;
}

/* value of a'(x) */
// Based on Horner method
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
// Based on Horner method
void ceval_dpoly_horner(DCmplx ret, DPoly a, DCmplx x)
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
void ceval_dpoly_estrin(DCmplx ret, DPoly a, DCmplx x)
{
    double _Complex *in_coef_old, *in_coef_new, in_x, in_ret, ctmp[2];
    long int in_degree, num_in_coef, i;

	set0_dcmplx(ret);
	in_x = x->re + x->im * I;

    if(a->deg == 0) {
        ///return get_dpoly_i(a, 0); // return constant;
		set_real_dcmplx(ret, get_dpoly_i(a, 0));
		return;
    }
    if(a->deg == 1) {
        //return get_dpoly_i(a, 0) + get_dpoly_i(a, 1) * x; // return a0 + a1 * x
		ctmp[0] = get_dpoly_i(a, 0) + I * 0.0;
		ctmp[1] = get_dpoly_i(a, 1) + I * 0.0;
		in_ret = ctmp[0] + ctmp[1] * in_x;
		ret->re = creal(in_ret);
		ret->im = cimag(in_ret);
    }

    num_in_coef = a->deg + 1;
    if((num_in_coef % 2) == 1) num_in_coef++;

    in_degree = num_in_coef - 1;
    in_coef_old = (double _Complex *)calloc(num_in_coef, sizeof(double _Complex));
    in_coef_new = (double _Complex *)calloc(num_in_coef, sizeof(double _Complex));

    for(i = 0; i <= a->deg; i++) in_coef_old[i] = get_dpoly_i(a, i) + I * 0.0;
    for(i = a->deg + 1; i <= in_degree; i++) in_coef_old[i] = 0.0 + 0.0 * I;

    //in_x = x;

    //printf("degree -> in_degree: %d -> %d\n", degree, in_degree);

    while(in_degree > 1)
    {
        num_in_coef /= 2;
        for(i = 0; i < num_in_coef; i++)
        {
            in_coef_new[i] = in_coef_old[i * 2] + in_coef_old[i * 2 + 1] * in_x;
            //printf("coef[%d] = in_coef[%d] + in_coef[%d] * %f = %f\n", i, i * 2, i * 2 + 1, in_x, in_coef_new[i]);
        }

        //return estrin(x * x, coef, num_in_coef / 2 - 1);
        in_x = in_x * in_x;
        for(i = 0; i < num_in_coef; i++)
           	in_coef_old[i] = in_coef_new[i];

        if((num_in_coef % 2) == 1)
		{
            num_in_coef++;
			in_coef_old[num_in_coef - 1] = 0.0 + 0.0 * I;
		} 

        in_degree = num_in_coef - 1;

        //for(i = 0; i <= in_degree; i++)
        //    in_coef_old[i] = in_coef_new[i];

    }
    in_ret = in_coef_new[0] + in_coef_new[1] * in_x;
	ret->re = creal(in_ret);
	ret->im = cimag(in_ret);

    free(in_coef_old);
    free(in_coef_new);

    //return ret;
	return;
}

/* complex value of a'(x) */
// Based on Horner method
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

//------------
// AVX2
//------------

/* value of a(x) */
// Based on Horner method
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
__m256d _bncavx2_eval_dpoly_horner(DPoly a, __m256d x)
{
	long int i;
	//double ret;
	__m256d ret, a_i;

	ret = _mm256_set1_pd(get_dpoly_i(a, a->deg));
	for(i = a->deg - 1; i >= 0; i--)
	{
		a_i = _mm256_set1_pd(get_dpoly_i(a, i));
		//ret = ret * x + get_dpoly_i(a, i);
		ret = _mm256_fmadd_pd(ret, x, a_i);
	}

	return ret;
}

/* complex value of a(x) */
// Based on Horner method
void _bncavx2_ceval_dpoly_horner(__m256d *ret_re, __m256d *ret_im, DPoly a, __m256d x_re, __m256d x_im)
{
	long int i;
	__m256d tmp_re, tmp_im, a_i;

	//set0_dcmplx(ret);
	*ret_re = _mm256_setzero_pd();
	*ret_im = _mm256_setzero_pd();

	//set_real_dcmplx(ret, get_dpoly_i(a, a->deg));
	*ret_re = _mm256_set1_pd(get_dpoly_i(a, a->deg));

	for(i = a->deg - 1; i >= 0; i--)
	{
		/* ret = ret * x + get_dpoly_i(a, i) */
		a_i = _mm256_set1_pd(get_dpoly_i(a, i));

		//mul2_dcmplx(ret, x);
		tmp_re = _mm256_sub_pd(
			_mm256_mul_pd(*ret_re, x_re),
			_mm256_mul_pd(*ret_im, x_im)
		);
		tmp_im = _mm256_add_pd(
			_mm256_mul_pd(*ret_re, x_im),
			_mm256_mul_pd(*ret_im, x_re)
		);

		//add_dcmplx_d(ret, ret, get_dpoly_i(a, i));
		*ret_re = _mm256_add_pd(tmp_re, a_i);
		*ret_im = tmp_im;
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
double _bncavx2_eval_dpoly_estrin(DPoly a, double x)
{
    //double *in_coef_old, *in_coef_new;
    DVector in_coef_old, in_coef_new;
    double in_x, ret;
    long int in_degree, num_in_coef, i, num_loop_unit;

    if((a->deg + 1) <= 8) // _BNC_D_WIDTH)
        return eval_dpoly_horner(a, x);

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d old_coef4[2], new_coef4, a04, a14, x4, zero4;

    // zero4 := 0
    zero4 = _mm256_setzero_pd();
    num_loop_unit = 2 * _BNC_D_WIDTH;

#elif defined(__AVX512F__) // __AVX512F__
	__m512d old_coef8[2], new_coef8, a08, a18, x8, zero8;

    // zero8 := 0
    zero8 = _mm512_setzero_pd();
    num_loop_unit = 2 * _BNC_D_WIDTH;

#else // __AVX2__
    return eval_dpoly_estrin(a, x);

#endif // __AVX2__

    num_in_coef = a->deg + 1;
    //if((num_in_coef % 2) == 1) num_in_coef++;
    if((num_in_coef % num_loop_unit) != 0) 
        num_in_coef = (long int)ceil(((double)num_in_coef / (double)num_loop_unit)) * num_loop_unit; ////num_in_coef += num_in_coef % num_loop_unit;

    in_degree = num_in_coef - 1;
    //in_coef_old = (double *)calloc(num_in_coef, sizeof(double));
    //in_coef_new = (double *)calloc(num_in_coef, sizeof(double));
    in_coef_old = init_dvector(num_in_coef);
    //in_coef_new = init_dvector(num_in_coef);

    //for(i = 0; i <= a->deg; i++) in_coef_old[i] = get_dpoly_i(a, i);
    //for(i = a->deg + 1; i <= in_degree; i++) in_coef_old[i] = 0.0;
    for(i = 0; i <= a->deg; i++) set_dvector_i(in_coef_old, i, get_dpoly_i(a, i));

    in_x = x;

    //printf("num_in_coef = %ld\n", num_in_coef);
    //printf("degree -> in_degree: %d -> %d\n", degree, in_degree);

    while(1)
    //while(num_in_coef > num_loop_unit)
    {
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
        // x2_4 := x^2
        x4 = _mm256_set1_pd(in_x);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            old_coef4[0] = _mm256_load_pd(&(in_coef_old->element[i * num_loop_unit]));
            old_coef4[1] = _mm256_load_pd(&(in_coef_old->element[i * num_loop_unit + _BNC_D_WIDTH]));
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
            a04 = _mm256_unpacklo_pd(old_coef4[0], old_coef4[1]);
            a14 = _mm256_unpackhi_pd(old_coef4[0], old_coef4[1]);
            //printf("         a04 = "); PRINT_M256D_SL(a04);  
            //printf("         a14 = "); PRINT_M256D_SL(a14);
            new_coef4 = _mm256_fmadd_pd(a14, x4, a04);
            new_coef4 = _mm256_permute4x64_pd(new_coef4, (int)(3*64 + 1*16 + 2*4 + 0));
            // b0, b2, b1, b3 -> b0, b1, b2, b3
            //_mm256_permute4x64_pd(new_coef4, (int)(3*64 + 1*16 + 2*4 + 0));
            //printf("   new_coef4 = "); PRINT_M256D_SL(new_coef4);
			// embed 0s
            _mm256_store_pd(&in_coef_old->element[i * num_loop_unit], zero4); // embed 0s            
            _mm256_store_pd(&in_coef_old->element[i * num_loop_unit + _BNC_D_WIDTH], zero4);

            //_mm256_store_pd(&in_coef_old->element[i * _BNC_D_WIDTH], zero4); // embed 0s            
            _mm256_store_pd(&in_coef_old->element[i * _BNC_D_WIDTH], new_coef4);
            //printf("in_coef_old  = %f %f %f %f\n", in_coef_old->element[i * _BNC_D_WIDTH], in_coef_old->element[i * _BNC_D_WIDTH + 1], in_coef_old->element[i * _BNC_D_WIDTH + 2], in_coef_old->element[i * _BNC_D_WIDTH + 3]);
        }
#elif defined(__AVX512F__) // __AVX512F__
        // x2_4 := x^2
        x8 = _mm512_set1_pd(in_x);

        for(i = 0; i < (num_in_coef / num_loop_unit); i++) {
            old_coef8[0] = _mm512_load_pd(&in_coef_old[i * num_loop_unit]);
            old_coef8[1] = _mm512_load_pd(&in_coef_old[i * num_loop_unit + BNC_D_WIDTH]);
            
            a08 = _mm512_unpacklo_pd(old_coef8[0], old_coef8[1]);
            a18 = _mm512_unpackhi_pd(old_coef8[0], old_coef8[1]);
            
            // FMA: a18 * x8 + a08
            new_coef8 = _mm512_fmadd_pd(a18, x8, a08);
            
            new_coef8 = _mm512_permutex_pd(new_coef8, 0xD8);
            
            _mm512_store_pd(&in_coef_old[i * num_loop_unit], zero8);
            _mm512_store_pd(&in_coef_old[i * num_loop_unit + BNC_D_WIDTH], zero8);
            _mm512_store_pd(&in_coef_old[i * BNC_D_WIDTH], new_coef8);
        }
#endif // __AVX2__

        //return estrin(x * x, coef, num_in_coef / 2 - 1);
        in_x = in_x * in_x;
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
    ret = get_dvector_i(in_coef_old, _BNC_D_WIDTH - 1);
    for(i = _BNC_D_WIDTH - 2; i >= 0; i--)
        ret = ret * in_x + get_dvector_i(in_coef_old, i);
#endif // __AVX2__

    //free(in_coef_old);
    //free(in_coef_new);
    free_dvector(in_coef_old);
    //free_dvector(in_coef_new);

    return ret;
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
void _bncavx2_ceval_dpoly_estrin(DCmplx ret, DPoly a, DCmplx x)
{
    //double *in_coef_old, *in_coef_new;
    DVector in_coef_old_real, in_coef_old_imag;
    double _Complex in_x; // , ret;
    double _Complex in_ret;
    long int in_degree, num_in_coef, i, num_loop_unit;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d old_coef4_real[2], old_coef4_imag[2];
    __m256d new_coef4_real, new_coef4_imag;
    __m256d a04_real, a04_imag, a14_real, a14_imag, x4_real, x4_imag, zero4;

    if((a->deg + 1) <= _BNC_D_WIDTH)
    {
        ceval_dpoly_horner(ret, a, x);
        return;
    }

    // zero4 := 0
    zero4 = _mm256_setzero_pd();
    num_loop_unit = 2 * _BNC_D_WIDTH;

#elif defined(__AVX512F__) // __AVX512F__
	__m512d old_coef8_real[2], old_coef8_imag[2];
    __m512d new_coef8_real, new_coef8_imag;
    __m512d a08_real, a08_imag, a18_real, a18_imag, x8_real, x8_imag, zero8;

    if((a->deg + 1) <= _BNC_D_WIDTH)
    {
        ceval_dpoly_horner(ret, a, x);
        return;
    }

    // zero8 := 0
    zero8 = _mm512_setzero_pd();
    num_loop_unit = 2 * _BNC_D_WIDTH;

#else // __AVX2__
    ceval_dpoly_estrin(ret, a, x);
    return;
#endif // __AVX2__

    num_in_coef = a->deg + 1;
    //if((num_in_coef % 2) == 1) num_in_coef++;
    if((num_in_coef % num_loop_unit) != 0) 
        num_in_coef = (long int)ceil(((double)num_in_coef / (double)num_loop_unit)) * num_loop_unit; ////num_in_coef += num_in_coef % num_loop_unit;

    in_degree = num_in_coef - 1;
    //in_coef_old = (double *)calloc(num_in_coef, sizeof(double));
    //in_coef_new = (double *)calloc(num_in_coef, sizeof(double));
    in_coef_old_real = init_dvector(num_in_coef);
    in_coef_old_imag = init_dvector(num_in_coef);
    //in_coef_new = init_dvector(num_in_coef);

    //for(i = 0; i <= a->deg; i++) in_coef_old[i] = get_dpoly_i(a, i);
    //for(i = a->deg + 1; i <= in_degree; i++) in_coef_old[i] = 0.0;
    for(i = 0; i <= a->deg; i++) set_dvector_i(in_coef_old_real, i, get_dpoly_i(a, i));

    in_x = x->re + x->im * I;

    //printf("num_in_coef = %ld\n", num_in_coef);
    //printf("degree -> in_degree: %d -> %d\n", degree, in_degree);

    while(1)
    //while(num_in_coef > num_loop_unit)
    {
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
        // x2_4 := x^2
        //x4 = _mm256_set1_pd(in_x);
        x4_real = _mm256_set1_pd(creal(in_x));
        x4_imag = _mm256_set1_pd(cimag(in_x));

        for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            old_coef4_real[0] = _mm256_load_pd(&(in_coef_old_real->element[i * num_loop_unit]));
            old_coef4_real[1] = _mm256_load_pd(&(in_coef_old_real->element[i * num_loop_unit + _BNC_D_WIDTH]));
            old_coef4_imag[0] = _mm256_load_pd(&(in_coef_old_imag->element[i * num_loop_unit]));
            old_coef4_imag[1] = _mm256_load_pd(&(in_coef_old_imag->element[i * num_loop_unit + _BNC_D_WIDTH]));

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
            a04_real = _mm256_unpacklo_pd(old_coef4_real[0], old_coef4_real[1]);
            a14_real = _mm256_unpackhi_pd(old_coef4_real[0], old_coef4_real[1]);
            a04_imag = _mm256_unpacklo_pd(old_coef4_imag[0], old_coef4_imag[1]);
            a14_imag = _mm256_unpackhi_pd(old_coef4_imag[0], old_coef4_imag[1]);
             //printf("         a04 = "); PRINT_M256D_SL(a04);  
            //printf("         a14 = "); PRINT_M256D_SL(a14);
            //new_coef4 = _mm256_fmadd_pd(a14, x4, a04);

            // a14 * x4 -> (a14_re * x4_re - a14_im * x4_im) + (a14_re * x4_im + a14_im * x4_re) * I
            new_coef4_real = _mm256_sub_pd(_mm256_mul_pd(a14_real, x4_real), _mm256_mul_pd(a14_imag, x4_imag));
            new_coef4_imag = _mm256_add_pd(_mm256_mul_pd(a14_real, x4_imag), _mm256_mul_pd(a14_imag, x4_real));

            // a14_x4 + a04 -> a14_x4_re + a_04_re + (a_14_x4_im + a_04_im) * I
            new_coef4_real = _mm256_add_pd(new_coef4_real, a04_real);
            new_coef4_imag = _mm256_add_pd(new_coef4_imag, a04_imag);

            new_coef4_real = _mm256_permute4x64_pd(new_coef4_real, (int)(3*64 + 1*16 + 2*4 + 0));
            new_coef4_imag = _mm256_permute4x64_pd(new_coef4_imag, (int)(3*64 + 1*16 + 2*4 + 0));
            // b0, b2, b1, b3 -> b0, b1, b2, b3
            //_mm256_permute4x64_pd(new_coef4, (int)(3*64 + 1*16 + 2*4 + 0));
            //printf("   new_coef4 = "); PRINT_M256D_SL(new_coef4);
            // embed 0s
            _mm256_store_pd(&in_coef_old_real->element[i * num_loop_unit], zero4);
            _mm256_store_pd(&in_coef_old_imag->element[i * num_loop_unit], zero4);
            _mm256_store_pd(&in_coef_old_real->element[i * num_loop_unit + _BNC_D_WIDTH], zero4);
            _mm256_store_pd(&in_coef_old_imag->element[i * num_loop_unit + _BNC_D_WIDTH], zero4);

            _mm256_store_pd(&in_coef_old_real->element[i * _BNC_D_WIDTH], new_coef4_real);
            _mm256_store_pd(&in_coef_old_imag->element[i * _BNC_D_WIDTH], new_coef4_imag);
            //printf("in_coef_old  = %f %f %f %f\n", in_coef_old_real->element[i * _BNC_D_WIDTH], in_coef_old_real->element[i * _BNC_D_WIDTH + 1], in_coef_old_real->element[i * _BNC_D_WIDTH + 2], in_coef_old_real->element[i * _BNC_D_WIDTH + 3]);
        }
#elif defined(__AVX512F__) // __AVX512F__
        // x2_4 := x^2
        //x8 = _mm512_set1_pd(in_x);
        x8_real = _mm512_set1_pd(creal(in_x));
        x8_imag = _mm512_set1_pd(cimag(in_x));

		for(i = 0; i < (num_in_coef / num_loop_unit); i++)
        {
            old_coef8_real[0] = _mm512_load_pd(&get_dvector_i(in_coef_old_real,  i * num_loop_unit));
            old_coef8_real[1] = _mm512_load_pd(&get_dvector_i(in_coef_old_real, (i + 1) * num_loop_unit + _BNC_D_WIDTH));
            old_coef8_imag[0] = _mm512_load_pd(&get_dvector_i(in_coef_old_imag,  i * num_loop_unit));
            old_coef8_imag[1] = _mm512_load_pd(&get_dvector_i(in_coef_old_imag, (i + 1) * num_loop_unit + _BNC_D_WIDTH));
            
            // old[0] = (a0 + a1 * x)       + (a2  + a3  * x) * x^2  + (a4 + a5 * x)   * x^4  + (a6  +  a7 * x) * x^6
            // old[1] = (a8 + a9 * x) * x^8 + (a10 + a11 * x) * x^10 + (a12 + a13 * x) * x^12 + (a14 + a15 * x) * x^14
            // a08    = a0 a2 a4 a6 a8 a10 a12 a14
            // a18    = a1 a3 a5 a7 a9 a11 a13 a15
			// x8     =  x  x  x  x  x   x  x   x
            // b0 b1 b2 b3 b4 b5 b6 b7 := a08 + a18 * x8
            // =  b0 + b1 * y + b2 * y^2 + b3 * y^3 + b4 * y^4 + b5 * y^5 + b6 * y^6 + b7 * y^7
            a08_real = _mm512_unpacklo_pd(old_coef8_real[0], old_coef8_real[1]);
            a18_real = _mm512_unpackhi_pd(old_coef8_real[0], old_coef8_real[1]);
            a08_imag = _mm512_unpacklo_pd(old_coef8_imag[0], old_coef8_imag[1]);
            a18_imag = _mm512_unpackhi_pd(old_coef8_imag[0], old_coef8_imag[1]);

           // a14 * x4 -> (a14_re * x4_re - a14_im * x4_im) + (a14_re * x4_im + a14_im * x4_re) * I
            new_coef8_real = _mm512_sub_pd(_mm512_mul_pd(a18_real, x8_real), _mm512_mul_pd(a18_imag, x8_imag));
            new_coef8_imag = _mm512_add_pd(_mm512_mul_pd(a18_real, x8_imag), _mm512_mul_pd(a18_imag, x8_real));

            // a14_x4 + a04 -> a14_x4_re + a_04_re + (a_14_x4_im + a_04_im) * I
            new_coef8_real = _mm512_add_pd(new_coef8_real, a08_real);
            new_coef8_imag = _mm512_add_pd(new_coef8_imag, a08_imag);

            new_coef8_real = _mm512_permute8x64_pd(new_coef8_real, (int)(0 + 2*2 + 4*4 + 6*8 + 1*16 + 3*32 + 5*64 * 7*128)); //(int)(3*64 + 1*16 + 2*4 + 0));
            new_coef8_imag = _mm512_permute8x64_pd(new_coef8_imag, (int)(0 + 2*2 + 4*4 + 6*8 + 1*16 + 3*32 + 5*64 * 7*128)); //(int)(3*64 + 1*16 + 2*4 + 0));
            // b0, b2, b4, b6, b1, b3, b5, b7 -> b0, b1, b2, b3, b4, b5, b6, b7
            //_mm256_permute4x64_pd(new_coef4, (int)(3*64 + 1*16 + 2*4 + 0));
            //printf("   new_coef4 = "); PRINT_M256D_SL(new_coef4);
            // embed 0s
            _mm512_store_pd(&in_coef_old_real->element[i * num_loop_unit], zero8);
            _mm512_store_pd(&in_coef_old_imag->element[i * num_loop_unit], zero8);
            _mm512_store_pd(&in_coef_old_real->element[i * num_loop_unit + _BNC_D_WIDTH], zero8);
            _mm512_store_pd(&in_coef_old_imag->element[i * num_loop_unit + _BNC_D_WIDTH], zero8);

            _mm512_store_pd(&in_coef_old_real->element[i * _BNC_D_WIDTH], new_coef8_real);
            _mm512_store_pd(&in_coef_old_imag->element[i * _BNC_D_WIDTH], new_coef8_imag);
          }
#endif // __AVX2__

        //return estrin(x * x, coef, num_in_coef / 2 - 1);
        in_x = in_x * in_x;
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
    in_ret = get_dvector_i(in_coef_old_real, _BNC_D_WIDTH - 1) + get_dvector_i(in_coef_old_imag, _BNC_D_WIDTH - 1) * I;
    for(i = _BNC_D_WIDTH - 2; i >= 0; i--)
        in_ret = in_ret * in_x + (get_dvector_i(in_coef_old_real, i) + get_dvector_i(in_coef_old_imag, i) * I);
#endif // __AVX2__

    ret->re = creal(in_ret);
    ret->im = cimag(in_ret);

    //free(in_coef_old);
    //free(in_coef_new);
    free_dvector(in_coef_old_real);
    free_dvector(in_coef_old_imag);
    //free_dvector(in_coef_new);

    //return ret;
    return;
}

#ifdef __cpluplus
} // extern "C" {
#endif // __cplusplus
