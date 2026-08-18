/********************************************************************************/
/* efunc.c: Elemantary functions                                                */
/* Copyright (c) 2000-2013 Tomonori Kouya                                       */
/*                                                                              */
/* Version 0.1, 2013-06-07: Bug fix in mpf_min                                  */
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
#include <math.h>
#include <stdlib.h>
// #include <alloc.h>
#include <string.h>

#include "bnc.h"

#ifdef USE_GMP
/* set bnc_default_prec */
void set_bnc_default_prec(unsigned long precision)
{
	mpf_set_default_prec(precision);
	bnc_default_prec = precision;
	printf("-------------------------------------------------------------------------------\n");
	printf("BNC Default Precision    : %ld bits(%.1f decimal digits)\n", precision, (double)precision * log10(2.0));
#ifdef USE_MPFR
	bnc_default_rounding_mode = GMP_RNDN; /* round to nearest */
	mpfr_set_default_rounding_mode(bnc_default_rounding_mode);
	printf("BNC Default Rounding Mode: ");
	switch((int)bnc_default_rounding_mode)
	{
		case GMP_RNDN: printf("Round to Nearest\n"); break;
		case GMP_RNDU: printf("Round to Infinity\n"); break;
		case GMP_RNDD: printf("Round to -Infinity\n"); break;
		case GMP_RNDZ: printf("Round to Zero\n"); break;
		default: printf("unknown mode\n");
	}
#endif
	printf("-------------------------------------------------------------------------------\n");
}
void _set_bnc_default_prec(unsigned long precision)
{
	mpf_set_default_prec(precision);
	bnc_default_prec = precision;
#ifdef USE_MPFR
	bnc_default_rounding_mode = GMP_RNDN; /* round to nearest */
	mpfr_set_default_rounding_mode(bnc_default_rounding_mode);
#endif
}
void set_bnc_default_prec_decimal(unsigned long precision)
{
	unsigned long binary_prec;

	binary_prec = ceil(precision / log10(2.0));
	set_bnc_default_prec(binary_prec);
}

void _set_bnc_default_prec_decimal(unsigned long precision)
{
	unsigned long binary_prec;

	binary_prec = ceil(precision / log10(2.0));
	_set_bnc_default_prec(binary_prec);
}


#ifdef USE_MPFR
mp_rnd_t get_bnc_default_rounding_mode(void)
{
	return bnc_default_rounding_mode;
}

void set_bnc_rounding_mode(mp_rnd_t rounding_mode)
{
	mpfr_set_default_rounding_mode(rounding_mode);
	bnc_default_rounding_mode = rounding_mode;
	printf("-------------------------------------------------------------------------------\n");
	printf("BNC Default Rounding Mode: ");
	switch((int)bnc_default_rounding_mode)
	{
		case GMP_RNDN: printf("Round to Nearest\n"); break;
		case GMP_RNDU: printf("Round to Infinity\n"); break;
		case GMP_RNDD: printf("Round to -Infinity\n"); break;
		case GMP_RNDZ: printf("Round to Zero\n"); break;
		default: printf("unknown mode\n");
	}
	printf("-------------------------------------------------------------------------------\n");
}
void _set_bnc_rounding_mode(mp_rnd_t rounding_mode)
{
	mpfr_set_default_rounding_mode(rounding_mode);
	bnc_default_rounding_mode = rounding_mode;
}
#endif

/* get bnc_default_prec */
unsigned long int get_bnc_default_prec(void)
{
	return bnc_default_prec;
}
#endif

/* lmax */
long int lmax(long int a, long int b)
{
	if(a > b)
		return a;
	else
		return b;
}

/* lmin */
long int lmin(long int a, long int b)
{
	if(a < b)
		return a;
	else
		return b;
}

#if __GNUC__  < 3
/* fmax */
float fmax(float a, float b)
{
	if(a > b)
		return a;
	else
		return b;
}

/* fmin */
float fmin(float a, float b)
{
	if(a < b)
		return a;
	else
		return b;
}
#endif

/* dmax */
double dmax(double a, double b)
{
	if(a > b)
		return a;
	else
		return b;
}

/* dmin */
double dmin(double a, double b)
{
	if(a < b)
		return a;
	else
		return b;
}

#ifdef USE_GMP
/* mpf_max */
mpf_ptr mpf_max(mpf_t a, mpf_t b)
{
	if(mpf_cmp(a, b) >= 0) /* a > b */
		return (mpf_ptr)a;
	else
		return (mpf_ptr)b;
}

/* mpf_min */
mpf_ptr mpf_min(mpf_t a, mpf_t b)
{
	if(mpf_cmp(a, b) <= 0) /* a <= b */
		return (mpf_ptr)a; // fix! 2013-06-07 T.Kouya
	else
		return (mpf_ptr)b; // fix! 2013-06-07 T.Kouya
}
#endif

/* power             */
/* base^n (if n > 0) */
/* 1 (if n = 0)      */
/* 0 (if n < 0)      */
long int lpower(long int base, long int n)
{
	long int ans;

	if(n == 0)
		return 1L;
	else if(n < 0)
		return 0L;

	ans = 1L;
	while(--n >= 0);
		ans *= base;

	return ans;
}

float fpower(float base, long int n)
{
	float ans;

	if(n < 0)
		base = 1.0 / base;

	ans = 1.0;
	while(n-- > 0)
		ans *= base;

	return ans;
}

double dpower(double base, long int n)
{
	double ans;

	if(n < 0)
		base = 1.0 / base;

	ans = 1.0;
	while(n-- > 0)
		ans *= base;

	return ans;
}

#ifdef USE_GMP
void mpf_power(mpf_t ans, mpf_t base, long int n)
{
	mpf_t tmp;

	mpf_init2(tmp, mpf_get_prec(base));
	if(n < 0)
		mpf_ui_div(tmp, 1UL, base);
	else
		mpf_set(tmp, base);

	mpf_set_ui(ans, 1UL);
	while(n-- > 0)
		mpf_mul(ans, ans, tmp);

	mpf_clear(tmp);

	return;
}
#endif

#ifdef USE_GMP
/* add 2001 4/24 */
void bnc_mpf_set_d(mpf_t ret, double dval)
{
	char tmp_str[64];

	sprintf(tmp_str, "%25.17e", dval);
	mpf_set_str(ret, tmp_str, 10);
}

void bnc_mpf_init_set_d(mpf_t ret, double dval)
{
	char tmp_str[64];

	mpf_init(ret);

	sprintf(tmp_str, "%25.17e", dval);
	mpf_set_str(ret, tmp_str, 10);
}

#ifndef USE_MPFR

#if (__GNU_MP_VERSION < 3)
/* mpf_t -> IEEE float */
float mpf2float(mpf_t x)
{
	static char str[64], tmp_str[20];
	mp_exp_t exponent;
	long int i;

	for(i = 0; i < 64; i++)
		str[i] = '\0';
	for(i = 0; i < 20; i++)
		tmp_str[i] = '\0';

	mpf_get_str(tmp_str, &exponent, 10, 20, x);
	if(tmp_str[0] == '-')
	{
		strcpy(str, "-0.");
		for(i = 0; i < 19; i++)
			tmp_str[i] = tmp_str[i + 1];
	}
	else
		strcpy(str, "0.");
	strcat(str, tmp_str);
	strcat(str, "e");
	sprintf(str, "%s%+03ld", str, (long)exponent);

	return (float)atof(str);
}

/* floor(x) */
void mpf_floor(mpf_t ans, mpf_t x)
{
	char *str, str_exp[64] = "";
	long sign, i;
	mp_exp_t exponent, tmp;

	sign = mpf_sgn(x); /* sing < 0 : minus */
	mpf_abs(ans, x);
	str = mpf_get_str(NULL, &exponent, 2, 0, ans);
	if((long)exponent > 0) /* x_int > 1 */
	{
		if(strlen(str) > exponent)
			*(str + exponent) = '.';
		else
			return;
		for(i = exponent + 1; *(str + i) != '\0'; i++)
			*(str + i) = '0';
		sprintf(str_exp, "@0");
		strcat(str, str_exp); /* ex) 110101111....1@0 */
		//printf("mpf_floor -> %s\n", str);
		mpf_set_str(ans, str, 2);
	}
	else
		mpf_set_ui(ans, 0UL);

	/* x < 0 : -floor(|x|) - 1 */
	if(sign < 0)
	{
		mpf_add_ui(ans, ans, 1UL);
		mpf_neg(ans, ans);
	}
}

/* mpf_t -> IEEE double */
double mpf2double(mpf_t x)
{
	static char str[64], tmp_str[20];
	mp_exp_t exponent;
	long int i;

	for(i = 0; i < 64; i++)
		str[i] = '\0';
	for(i = 0; i < 20; i++)
		tmp_str[i] = '\0';

	mpf_get_str(tmp_str, &exponent, 10, 25, x);
	if(tmp_str[0] == '-')
	{
		strcpy(str, "-0.");
		for(i = 0; i < 19; i++)
			tmp_str[i] = tmp_str[i + 1];
	}
	else
		strcpy(str, "0.");
	strcat(str, tmp_str);
	strcat(str, "e");
	sprintf(str, "%s%+03ld", str, (long)exponent);

	return atof(str);
}
#else
double mpf2double(mpf_t x)
{
	return mpf_get_d(x);
}
float mpf2float(mpf_t x)
{
	return (float)mpf_get_d(x);
}
#endif

/* PI for GMP */
/* arctan(1/2) + arctan(1/3) = pi/4 */
void mpf_pi(mpf_t ans)
{
	mpf_t at12, at13, old_at12, old_at13, tmp_at12, tmp_at13, tmp;
	unsigned long times;
	unsigned long prec;

	/* get prec */
	prec = mpf_get_prec(ans);

	/* initialize temporary variables */
	mpf_init2(at12, prec);
	mpf_init2(at13, prec);
	mpf_init2(old_at12, prec);
	mpf_init2(old_at13, prec);
	mpf_init2(tmp_at12, prec);
	mpf_init2(tmp_at13, prec);
	mpf_init2(tmp, prec);
	
	mpf_set_ui(at12, 1UL);
	mpf_set_ui(at13, 1UL);
	mpf_div_ui(at12, at12, 2UL); /* at12 = 1/2 */
	mpf_div_ui(at13, at13, 3UL); /* at13 = 1/3 */
	
	times = 3;
	mpf_set(tmp_at12, at12);
	mpf_set(old_at12, at12);
	mpf_set(tmp_at13, at13);
	mpf_set(old_at13, at13);

	do {
		mpf_neg(tmp_at12, tmp_at12);
		mpf_div_ui(tmp_at12, tmp_at12, 4UL);
		mpf_div_ui(tmp, tmp_at12, times);
		mpf_add(at12, at12, tmp);
		
		mpf_neg(tmp_at13, tmp_at13);
		mpf_div_ui(tmp_at13, tmp_at13, 9UL);
		mpf_div_ui(tmp, tmp_at13, times);
		mpf_add(at13, at13, tmp);

		if((mpf_cmp(at12, old_at12) == 0) && (mpf_cmp(at13, old_at13) == 0))
			break;
		
		mpf_set(old_at12, at12);
		mpf_set(old_at13, at13);
		times += 2;
	}while(1);

	mpf_add(ans, at12, at13);
	mpf_mul_ui(ans, ans, 4UL);

	/* clear */
	mpf_clear(at12);
	mpf_clear(at13);
	mpf_clear(old_at12);
	mpf_clear(old_at13);
	mpf_clear(tmp_at12);
	mpf_clear(tmp_at13);
	mpf_clear(tmp);
}

/* exp(1) for GMP */
void mpf_e(mpf_t ans)
{
        mpf_t old_ans, xn;
        unsigned long times;
        unsigned long prec;

	/* get prec */
	prec = mpf_get_prec(ans);

	/* initialize temporary variables */
	mpf_init2(old_ans, prec);
	mpf_init2(xn, prec);

        /* ans := 1 */
        /* old_ans := ans */
        /* xn  := 1   */
        /* times := 0 */
        mpf_set_ui(ans, 1UL);
        mpf_set(old_ans, ans);
        mpf_set_ui(xn, 1);

	times = 0;

        do
        {
                /* /n! */
                mpf_div_ui(xn, xn, ++times);

                /* ans += ans + x^n/n! */
                mpf_add(ans, ans, xn);

                /* ond_ans == ans */
                if(mpf_cmp(old_ans, ans) == 0)
                        break;

                /* old_ans := ans */
                mpf_set(old_ans, ans);
                //printf(" %ld ", times/2);
        }while(1);

	/* clear */
	mpf_clear(old_ans);
	mpf_clear(xn);
}


/* sin x for GMP */
void mpf_sin(mpf_t ans, mpf_t x)
{
	mpf_t old_ans, xn, x2, pi2, x_int;
	mp_exp_t exponent;
	char *str;
	long times, sign = 0; /* 1:plus, -1:minus */
	unsigned long int prec;

	/* get prec */
	prec = mpf_get_prec(ans);

	/* initialize temporary variables */
	mpf_init2(old_ans, prec);
	mpf_init2(xn, prec);
	mpf_init2(x2, prec);
	mpf_init2(pi2, prec);
	mpf_init2(x_int, prec);

	/* change argument */
	mpf_set(xn, x);
	if(mpf_sgn(xn) < 0)
	{
		sign = -1; /* sin(-x) = -sin(x) */
		mpf_neg(xn, xn);
	}
	else
		sign = 1; /* sin(x) */
	mpf_pi(pi2);
	mpf_mul_ui(pi2, pi2, 2L); /* pi2 := 2 * PI */
	if(mpf_cmp(xn, pi2) >= 0)
	{
		mpf_div(x2, xn, pi2); /* x2 := x - [x/pi2]*pi2 */
		mpf_floor(x2, x2);
		mpf_mul(x2, x2, pi2);
		mpf_sub(xn, xn, x2);
	}
	mpf_div_ui(pi2, pi2, 2L); /* pi2 := PI */
	if(mpf_cmp(xn, pi2) >= 0)
	{
		mpf_sub(xn, xn, pi2); /* x := x - PI */
		sign *= -1; /* sin(x + PI) = -sin(x) */
	}

	/* ans := changed x   */
	/* old_ans := ans */
	/* xn  := x   */
	/* x2  := x^2 */
	/* times := 1 */
	mpf_set(ans, xn);
	mpf_set(old_ans, ans);
	mpf_mul(x2, xn, xn);

	times = 1;

	do
	{
		/* x^(2n + 1) */
		mpf_mul(xn, xn, x2);
		/* /(2n + 1)! */
		mpf_div_ui(xn, xn, ++times);
		mpf_div_ui(xn, xn, ++times);
		/* (-1)^(n+1) */
		mpf_neg(xn, xn);

		/* ans += ans + (-1)^(n+1)*x^n/(2*n+1) */
		mpf_add(ans, ans, xn);

		/* ond_ans == ans */
		if(mpf_cmp(old_ans, ans) == 0)
			break;

		/* old_ans := ans */
		mpf_set(old_ans, ans);
		//printf(" %ld ", times/2);
	}while(1);

	/* check sign */
	if(sign == -1)
		mpf_neg(ans, ans);

	/* clear */
	mpf_clear(old_ans);
	mpf_clear(xn);
	mpf_clear(x2);
	mpf_clear(pi2);
	mpf_clear(x_int);

}

/* cos x for GMP */
void mpf_cos(mpf_t ans, mpf_t x)
{
        mpf_t old_ans, xn, x2, pi2;
        unsigned long times;
	unsigned long int prec;
	long int sign;
	
	/* get prec */
	prec = mpf_get_prec(ans);

	/* initialize temporary variables */
	mpf_init2(old_ans, prec);
	mpf_init2(xn, prec);
	mpf_init2(x2, prec);
	mpf_init2(pi2, prec);

	/* change argument */
	mpf_set(xn, x);
	sign = 1;
	if(mpf_sgn(xn) < 0)
	{
		sign = 1; /* cos(-x) = cos(x) */
		mpf_neg(xn, xn);
	}

	mpf_pi(pi2);
	mpf_mul_ui(pi2, pi2, 2L); /* pi2 := 2 * PI */
	if(mpf_cmp(xn, pi2) >= 0)
	{
		mpf_div(x2, xn, pi2); /* x2 := x - [x/pi2]*pi2 */
		mpf_floor(x2, x2);
		mpf_mul(x2, x2, pi2);
		mpf_sub(xn, xn, x2);
	}
	mpf_div_ui(pi2, pi2, 2L); /* pi2 := PI */
	if(mpf_cmp(xn, pi2) >= 0)
	{
		mpf_sub(xn, xn, pi2); /* x := x - PI */
		sign *= -1; /* cos(x + PI) = -cos(x) */
	}

        /* ans := 1   */
        /* old_ans := ans */
        /* xn  := 1   */
        /* x2  := x^2 */
        /* times := 0 */
        mpf_set_ui(ans, 1UL);
        mpf_set(old_ans, ans);
        mpf_set(x2, xn);
        mpf_mul(x2, x2, xn);
        mpf_set_si(xn, 1L);

        times = 0;

        do
        {
                /* x^(2n) */
                mpf_mul(xn, xn, x2);
                /* /(2n)! */
                mpf_div_ui(xn, xn, ++times);
                mpf_div_ui(xn, xn, ++times);
                /* (-1)^(n+1) */
                mpf_neg(xn, xn);

                /* ans += ans + (-1)^(n+1)*x^n/(2*n) */
                mpf_add(ans, ans, xn);

                /* ond_ans == ans */
                if(mpf_cmp(old_ans, ans) == 0)
                        break;

                /* old_ans := ans */
                mpf_set(old_ans, ans);
                //printf(" %ld ", times/2);
        }while(1);

	/* check sign */
	if(sign == -1)
		mpf_neg(ans, ans);

	/* clear */
	mpf_clear(old_ans);
	mpf_clear(xn);
	mpf_clear(x2);
}

/* exp(x) for GMP */
void mpf_exp(mpf_t ans, mpf_t x)
{
        mpf_t old_ans, xn, new_x;
	long int sign;
        unsigned long times;
	unsigned long int prec;
	
	/* get prec */
	prec = mpf_get_prec(ans);

	/* initialize temporary variables */
	mpf_init2(old_ans, prec);
	mpf_init2(xn, prec);
	mpf_init2(new_x, prec);

	/* x < 0 */
	
	if(mpf_sgn(x) < 0)
	{
		sign = 0; /* minus */
		mpf_neg(new_x, x);
	}
	else
	{
		sign = 1; /* plus */
		mpf_set(new_x, x);
	}
	
        /* ans := 1 */
        /* old_ans := ans */
        /* xn  := x   */
        /* times := 0 */
        mpf_set_ui(ans, 1UL);
        mpf_set(old_ans, ans);
        mpf_set_ui(xn, 1UL);

	times = 0;

        do
        {
                /* x^n */
                mpf_mul(xn, xn, new_x);
                /* /n! */
                mpf_div_ui(xn, xn, ++times);

                /* ans += ans + x^n/n! */
                mpf_add(ans, ans, xn);

                /* ond_ans == ans */
                if(mpf_cmp(old_ans, ans) == 0)
                        break;

		/* old_ans := ans */
                mpf_set(old_ans, ans);
                //printf(" %ld ", times/2);
	}while(1);

	if(sign == 0) /* minus */
		mpf_ui_div(ans, 1UL, ans);

	/* clear */
	mpf_clear(old_ans);
	mpf_clear(xn);
	mpf_clear(new_x);

}

/* log_e 2                           */
/* original code by Haruhiko Okumura */
/* adapted for GMP by Tomonori Kouya */
void mpf_ln_2(mpf_t ans)
{
	long int i;
	unsigned long int prec;
	mpf_t x, x2, s, last;

	/* get prec of ans */
	prec = mpf_get_prec(ans);

	mpf_init2(x, prec);
	mpf_init2(x2, prec);
	mpf_init2(s, prec);
	mpf_init2(last, prec);

	mpf_set_ui(x, 2UL);

	/* x = (2 - 1) / (2 + 1) */
	mpf_set_ui(x, 1UL); mpf_div_ui(x, x, 3UL);

	mpf_mul(x2, x, x);
	i = 1;
	mpf_set(s, x);

	do {
		mpf_mul(x, x, x2);
		i += 2;
		mpf_set(last, s);
		mpf_div_ui(s, x, i);
		mpf_add(s, last, s);
	} while(mpf_cmp(s, last) != 0);

	/* s *= 2; */
	mpf_mul_2exp(ans, s, 1UL);

	mpf_clear(s);
	mpf_clear(x);
	mpf_clear(x2);
	mpf_clear(last);

}

/* log_e x                           */
/* original code by Haruhiko Okumura */
/* adapted for GMP by Tomonori Kouya */
void mpf_ln(mpf_t ans, mpf_t x)
{
	long int i, k;
	unsigned long int prec;
	mpf_t new_x, x2, s, last, ln2;

	if(mpf_sgn(x) <= 0)
	{
		fprintf(stderr, "mpf_ln(x): illegal argument\n");
		return;
	}

	/* get prec of ans */
	prec = mpf_get_prec(ans);

	mpf_init2(x2, prec);
	mpf_init2(new_x, prec);
	mpf_init2(s, prec);
	mpf_init2(last, prec);
	mpf_init2(ln2, prec);

	/* get exponent */
	mpf_set(new_x, x);
	k = new_x->_mp_exp;

	if(k < 0)
		mpf_mul_2exp(new_x, new_x, (unsigned long)(-k));
	else
		mpf_div_2exp(new_x, new_x, (unsigned long)k);
	
	/* x = (x - 1) / (x + 1) */
	mpf_sub_ui(new_x, new_x, 1UL); 
	mpf_add_ui(x2, new_x, 2UL);
	mpf_div(new_x, new_x, x2);

	mpf_mul(x2, new_x, new_x);
	i = 1;
	mpf_set(s, new_x);

	do {
		mpf_mul(new_x, new_x, x2);
		i += 2;
		mpf_set(last, s);
		mpf_div_ui(s, new_x, i);
		mpf_add(s, last, s);
	} while(mpf_cmp(s, last) != 0);

	/* s *= 2; */
	mpf_mul_2exp(ans, s, 1UL);
	mpf_ln_2(ln2);
	if(k < 0)
	{
		k = -k;
		mpf_mul_ui(x2, ln2, k);
		mpf_neg(x2, x2);
	}
	else
		mpf_mul_ui(x2, ln2, k);
	mpf_add(ans, ans, x2);

	mpf_clear(s);
	mpf_clear(new_x);
	mpf_clear(x2);
	mpf_clear(last);
	mpf_clear(ln2);

}

/* log_10 x                          */
void mpf_log10(mpf_t ans, mpf_t x)
{
	unsigned long int prec;
	mpf_t ln10;

	if(mpf_sgn(x) <= 0)
	{
		fprintf(stderr, "mpf_log10(x): illegal argument\n");
		return;
	}

	/* init */
	prec = mpf_get_prec(ans);
	mpf_init2(ln10, prec);

	/* log10 x := ln x / ln 10 */
	mpf_ln(ans, x);
	mpf_set_ui(ln10, 10UL);
	mpf_ln(ln10, ln10);
	mpf_div(ans, ans, ln10);

	/* clear */
	mpf_clear(ln10);

	return;
}

#else /* defined USE_MPFR */
void mpf_e(mpfr_t ans)
{
	mpfr_t one; /* one = 1UL */

	mpfr_init2(one, mpfr_get_prec(ans));
	mpfr_set_ui(one, 1UL, bnc_default_rounding_mode);
	mpfr_exp(ans, one, bnc_default_rounding_mode);
	mpfr_clear(one);
}

/* Why doesn't this function exit in mpfr ? */
void mpfr_get_f_bnc(_bnc_mpf_t rop, mpfr_t op, mp_rnd_t rnd)
{
	static char str_tmp[10000], str_tmp0[8192], str_exp_tmp[128];
	static mp_exp_t exp_tmp;
	static int base = 10;

	mpfr_get_str(str_tmp0, &exp_tmp, base, 0, op, rnd);
	sprintf(str_exp_tmp, "@%d", (int)exp_tmp);
	strcat(str_tmp0, str_exp_tmp);
	sprintf(str_tmp, "0.%s", str_tmp0);
	_bnc_mpf_set_str(rop, str_tmp, base);
}
#endif /* if undefined USE_MPFR */

/* fix!: 2012-07-04 */
/* Jan.26, 2006 */
/* rop := op1 * op2 + op3 */
void mpf_fma(mpf_t rop, mpf_t op1, mpf_t op2, mpf_t op3)
{
#ifdef USE_MPFR
  #if ( _BNC_MPFR_VER > 201 )
	mpfr_fma(rop, op1, op2, op3, bnc_default_rounding_mode);
  #else
	mpfr_t tmp; mpfr_init2(tmp, mpfr_get_prec(rop));
	mpfr_mul(tmp, op1, op2, bnc_default_rounding_mode);
	mpfr_add(rop, tmp, op3, bnc_default_rounding_mode);
	mpfr_clear(tmp);
  #endif
#else
	mpf_t tmp; mpf_init2(tmp, mpf_get_prec(rop));
	mpf_mul(tmp, op1, op2);
	mpf_add(rop, tmp, op3);
	mpf_clear(tmp);
#endif
}

#endif /* USE_GMP */
