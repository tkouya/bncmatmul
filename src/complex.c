/********************************************************************************/
/* complex.c:                                                                   */
/* Copyright (C) 2002-2023 Tomonori Kouya                                       */
/*                                                                              */
/* Version 0.3: 2023-02-22 Adopt in BNCmatmul                                   */
/* Version 0.2: 2011-12-08 Fix setting of precision                             */
/* Version 0.1: 2008-06-04 Fix sign_*cmplx                                      */
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
#include <complex.h> // Standard C complex number
//#include "bnc.h"
#include "bnc_common.h"

#ifdef __cpluplus
extern "C" {
#endif // __cplusplus


/* float:
	FCmplx init_fcmplx(void)
	void free_fcmplx(FCmplx a)
	float get_real_fcmplx(FCmplx a)
	float get_image_fcmplx(FCmplx a)
	void set_real_fcmplx(FCmplx c, float val)
	void set_image_fcmplx(FCmplx c, float val)
	void subst_fcmplx(FCmplx c, FCmplx a)
	void set0_fcmplx(FCmplx c)
	void add_fcmplx(FCmplx c, FCmplx a, FCmplx b)
	void add2_fcmplx(FCmplx c, FCmplx a)
	void sub_fcmplx(FCmplx c, FCmplx a, FCmplx b)
	void conj_fcmplx(FCmplx c, FCmplx a)
	void sign_fcmplx(FCmplx c, FCmplx a)
	void sign2_fcmplx(FCmplx c)
	float abs_fcmplx(FCmplx a)
	float mul_fcmplx(FCmplx c, FCmplx a, FCmplx b)
	float mul2_fcmplx(FCmplx c, FCmplx a)
	float div_fcmplx(FCmplx c, FCmplx a, FCmplx b)
	void print_fcmplx(FCmplx a)
*/

/* init */
FCmplx init_fcmplx(void)
{
	FCmplx ret;

	ret = (FCmplx)malloc(sizeof(fcmplx));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: init_fcmplx\n");
		return NULL;
	}

	ret->re = 0;
	ret->im = 0;

	return ret;
}

/* clear */
void free_fcmplx(FCmplx a)
{
	if(a != NULL)
		free(a);
}

/* return real part */
float get_real_fcmplx(FCmplx a)
{
	return a->re;
}

/* set real part */
void set_real_fcmplx(FCmplx c, float val)
{
	c->re = val;
}

/* return imaginary part */
float get_image_fcmplx(FCmplx a)
{
	return a->im;
}

/* set imaginary part */
void set_image_fcmplx(FCmplx c, float val)
{
	c->im = val;
}

/* c := a */
void subst_fcmplx(FCmplx c, FCmplx a)
{
	float re_a, im_a;

	re_a = get_real_fcmplx(a);
	im_a = get_image_fcmplx(a);

	set_real_fcmplx(c, re_a);
	set_image_fcmplx(c, im_a);
}

/* c := 0 */
void set0_fcmplx(FCmplx c)
{
	set_real_fcmplx(c, 0);
	set_image_fcmplx(c, 0);
}

/* c := a + b */
void add_fcmplx(FCmplx c, FCmplx a, FCmplx b)
{
	float re_a, im_a, re_b, im_b;

	re_a = get_real_fcmplx(a);
	im_a = get_image_fcmplx(a);
	re_b = get_real_fcmplx(b);
	im_b = get_image_fcmplx(b);

	set_real_fcmplx(c, re_a + re_b);
	set_image_fcmplx(c, im_a + im_b);
}

/* c := a + re_b */
void add_fcmplx_f(FCmplx c, FCmplx a, float b)
{
	float re_a, im_a;

	re_a = get_real_fcmplx(a);
	im_a = get_image_fcmplx(a);

	set_real_fcmplx(c, re_a + b);
	set_image_fcmplx(c, im_a);
}

/* c += a */
void add2_fcmplx(FCmplx c, FCmplx a)
{
	float re_a, im_a, re_c, im_c;

	re_a = get_real_fcmplx(a);
	im_a = get_image_fcmplx(a);
	re_c = get_real_fcmplx(c);
	im_c = get_image_fcmplx(c);

	set_real_fcmplx(c, re_c + re_a);
	set_image_fcmplx(c, im_c + im_a);
}

/* c := a - b */
void sub_fcmplx(FCmplx c, FCmplx a, FCmplx b)
{
	float re_a, im_a, re_b, im_b;

	re_a = get_real_fcmplx(a);
	im_a = get_image_fcmplx(a);
	re_b = get_real_fcmplx(b);
	im_b = get_image_fcmplx(b);

	set_real_fcmplx(c, re_a - re_b);
	set_image_fcmplx(c, im_a - im_b);
}

/* c := re(a) - i * im(a) */
void conj_fcmplx(FCmplx c, FCmplx a)
{
	float re_a, im_a;

	re_a = get_real_fcmplx(a);
	im_a = get_image_fcmplx(a);

	set_real_fcmplx(c, re_a);
	set_image_fcmplx(c, -im_a);

}

/* c := -a */
void sign_fcmplx(FCmplx c, FCmplx a)
{
	float re_a, im_a;

	re_a = get_real_fcmplx(a);
	im_a = get_image_fcmplx(a);

	set_real_fcmplx(c, -re_a);
	set_image_fcmplx(c, -im_a); // Fix!!: 2008-06-04

}

/* c := -c */
void sign2_fcmplx(FCmplx c)
{
	float re_c, im_c;

	re_c = get_real_fcmplx(c);
	im_c = get_image_fcmplx(c);

	set_real_fcmplx(c, -re_c);
	set_image_fcmplx(c, -im_c); // Fix!!: 2008-06-04

}

/* |a| */
float abs_fcmplx(FCmplx a)
{
	float ret, re_a, im_a;

	re_a = get_real_fcmplx(a);
	im_a = get_image_fcmplx(a);

	ret = (float)sqrt(re_a * re_a + im_a * im_a);

	return ret;
}

/* c := a * b */
void mul_fcmplx(FCmplx c, FCmplx a, FCmplx b)
{
	float re_a, im_a, re_b, im_b;

	re_a = get_real_fcmplx(a);
	im_a = get_image_fcmplx(a);
	re_b = get_real_fcmplx(b);
	im_b = get_image_fcmplx(b);

	set_real_fcmplx(c, re_a * re_b - im_a * im_b);
	set_image_fcmplx(c, re_a * im_b + im_a * re_b);
}

/* c := a * re_b */
void mul_fcmplx_f(FCmplx c, FCmplx a, float b)
{
	float re_a, im_a;

	re_a = get_real_fcmplx(a);
	im_a = get_image_fcmplx(a);

	set_real_fcmplx(c, re_a * b);
	set_image_fcmplx(c, im_a * b);
}

/* c *= a */
void mul2_fcmplx(FCmplx c, FCmplx a)
{
	float re_a, im_a, re_c, im_c;

	re_a = get_real_fcmplx(a);
	im_a = get_image_fcmplx(a);
	re_c = get_real_fcmplx(c);
	im_c = get_image_fcmplx(c);

	set_real_fcmplx(c, re_c * re_a - im_c * im_a);
	set_image_fcmplx(c, re_c * im_a + im_c * re_a);
}

/* c := a / b */
void div_fcmplx(FCmplx c, FCmplx a, FCmplx b)
{
	float re_a, im_a, re_b, im_b, tmp, reim_b;

	re_a = get_real_fcmplx(a);
	im_a = get_image_fcmplx(a);
	re_b = get_real_fcmplx(b);
	im_b = get_image_fcmplx(b);

	if((re_b == 0) && (im_b == 0))
	{
		fprintf(stderr, "ERROR: Impossible division!\n");
		set0_fcmplx(c);
		return;
	}
	if((re_a == 0) && (im_a == 0))
	{
		set0_fcmplx(c);
		return;
	}

	if(fabs((double)re_b) >= fabs((double)im_b))
	{
		reim_b = im_b / re_b;
		tmp = re_b + reim_b * im_b;
		set_real_fcmplx(c, (re_a + reim_b * im_a) / tmp );
		set_image_fcmplx(c, (-re_a * reim_b + im_a) / tmp);
	}
	else
	{
		reim_b = re_b / im_b;
		tmp = re_b * reim_b + im_b;
		set_real_fcmplx(c, (re_a * reim_b + im_a) / tmp );
		set_image_fcmplx(c, (-re_a + im_a * reim_b) / tmp);
	}
}

/* exp(i*x) = cos(x) + i * sin(x) */
void iexp_fcmplx(FCmplx ret, float x)
{
	float re_ret, im_ret;

	set0_fcmplx(ret);
	re_ret = get_real_fcmplx(ret);
	im_ret = get_image_fcmplx(ret);

	re_ret = (float)cos((double)x);
	im_ret = (float)sin((double)x);

	set_real_fcmplx(ret, re_ret);
	set_image_fcmplx(ret, im_ret);
}

/* print complex */
void print_fcmplx(FCmplx a)
{
	float re_a, im_a;

	re_a = get_real_fcmplx(a);
	im_a = get_image_fcmplx(a);

	printf("(%15.7e, %15.7e)\n", re_a, im_a);
}

/* double:
	DCmplx init_dcmplx(void)
	void free_dcmplx(DCmplx a)
	double get_real_dcmplx(DCmplx a)
	double get_image_dcmplx(DCmplx a)
	void set_real_dcmplx(DCmplx c, double val)
	void set_image_dcmplx(DCmplx c, double val)
	void subst_dcmplx(DCmplx c, DCmplx a)
	void set0_dcmplx(DCmplx c)
	void add_dcmplx(DCmplx c, DCmplx a, DCmplx b)
	void add2_dcmplx(DCmplx c, DCmplx a)
	void sub_dcmplx(DCmplx c, DCmplx a, DCmplx b)
	void conj_dcmplx(DCmplx c, DCmplx a)
	void sign_dcmplx(DCmplx c, DCmplx a)
	void sign2_dcmplx(DCmplx c)
	double abs_dcmplx(DCmplx a)
	double mul_dcmplx(DCmplx c, DCmplx a, DCmplx b)
	double mul2_dcmplx(DCmplx c, DCmplx a)
	double div_dcmplx(DCmplx c, DCmplx a, DCmplx b)
	void print_dcmplx(DCmplx a)
*/

/* init */
DCmplx init_dcmplx(void)
{
	DCmplx ret;

	ret = (DCmplx)malloc(sizeof(dcmplx));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: init_dcmplx\n");
		return NULL;
	}

	ret->re = 0;
	ret->im = 0;

	return ret;
}

/* clear */
void free_dcmplx(DCmplx a)
{
	if(a != NULL)
		free(a);
}


/* return real part */
double get_real_dcmplx(DCmplx a)
{
	return a->re;
}

/* set real part */
void set_real_dcmplx(DCmplx c, double val)
{
	c->re = val;
}

/* return imaginary part */
double get_image_dcmplx(DCmplx a)
{
	return a->im;
}

/* set imaginary part */
void set_image_dcmplx(DCmplx c, double val)
{
	c->im = val;
}

/* c := a */
void subst_dcmplx(DCmplx c, DCmplx a)
{
	double re_a, im_a;

	re_a = get_real_dcmplx(a);
	im_a = get_image_dcmplx(a);

	set_real_dcmplx(c, re_a);
	set_image_dcmplx(c, im_a);
}

/* c := 0 */
void set0_dcmplx(DCmplx c)
{
	set_real_dcmplx(c, 0);
	set_image_dcmplx(c, 0);
}

/* c := 1 */
void set1_dcmplx(DCmplx c)
{
	set_real_dcmplx(c, 1);
	set_image_dcmplx(c, 0);
}

/* c := a + b */
void add_dcmplx(DCmplx c, DCmplx a, DCmplx b)
{
	double re_a, im_a, re_b, im_b;

	re_a = get_real_dcmplx(a);
	im_a = get_image_dcmplx(a);
	re_b = get_real_dcmplx(b);
	im_b = get_image_dcmplx(b);

	set_real_dcmplx(c, re_a + re_b);
	set_image_dcmplx(c, im_a + im_b);
}

/* c := a + re_b */
void add_dcmplx_d(DCmplx c, DCmplx a, double b)
{
	double re_a, im_a;

	re_a = get_real_dcmplx(a);
	im_a = get_image_dcmplx(a);

	set_real_dcmplx(c, re_a + b);
	set_image_dcmplx(c, im_a);
}

/* c += a */
void add2_dcmplx(DCmplx c, DCmplx a)
{
	double re_a, im_a, re_c, im_c;

	re_a = get_real_dcmplx(a);
	im_a = get_image_dcmplx(a);
	re_c = get_real_dcmplx(c);
	im_c = get_image_dcmplx(c);

	set_real_dcmplx(c, re_c + re_a);
	set_image_dcmplx(c, im_c + im_a);
}

/* c := a - b */
void sub_dcmplx(DCmplx c, DCmplx a, DCmplx b)
{
	double re_a, im_a, re_b, im_b;

	re_a = get_real_dcmplx(a);
	im_a = get_image_dcmplx(a);
	re_b = get_real_dcmplx(b);
	im_b = get_image_dcmplx(b);

	set_real_dcmplx(c, re_a - re_b);
	set_image_dcmplx(c, im_a - im_b);
}

/* c := re(a) - i * im(a) */
void conj_dcmplx(DCmplx c, DCmplx a)
{
	double re_a, im_a;

	re_a = get_real_dcmplx(a);
	im_a = get_image_dcmplx(a);

	set_real_dcmplx(c, re_a);
	set_image_dcmplx(c, -im_a);

}

/* c := -a */
void sign_dcmplx(DCmplx c, DCmplx a)
{
	double re_a, im_a;

	re_a = get_real_dcmplx(a);
	im_a = get_image_dcmplx(a);

	set_real_dcmplx(c, -re_a);
	set_image_dcmplx(c, -im_a); // Fix!! : 2008-06-04

}

/* c := -c */
void sign2_dcmplx(DCmplx c)
{
	double re_c, im_c;

	re_c = get_real_dcmplx(c);
	im_c = get_image_dcmplx(c);

	set_real_dcmplx(c, -re_c);
	set_image_dcmplx(c, -im_c); // Fix!! : 2008-06-04

}

/* |a| */
double abs_dcmplx(DCmplx a)
{
	double ret, re_a, im_a;

	re_a = get_real_dcmplx(a);
	im_a = get_image_dcmplx(a);

	ret = sqrt(re_a * re_a + im_a * im_a);

	return ret;
}

/* c := a * b */
void mul_dcmplx(DCmplx c, DCmplx a, DCmplx b)
{
	double re_a, im_a, re_b, im_b;

	re_a = get_real_dcmplx(a);
	im_a = get_image_dcmplx(a);
	re_b = get_real_dcmplx(b);
	im_b = get_image_dcmplx(b);

	set_real_dcmplx(c, re_a * re_b - im_a * im_b);
	set_image_dcmplx(c, re_a * im_b + im_a * re_b);
}

/* c := a * re_b */
void mul_dcmplx_d(DCmplx c, DCmplx a, double b)
{
	double re_a, im_a;

	re_a = get_real_dcmplx(a);
	im_a = get_image_dcmplx(a);

	set_real_dcmplx(c, re_a * b);
	set_image_dcmplx(c, im_a * b);
}

/* c *= a */
void mul2_dcmplx(DCmplx c, DCmplx a)
{
	double re_a, im_a, re_c, im_c;

	re_a = get_real_dcmplx(a);
	im_a = get_image_dcmplx(a);
	re_c = get_real_dcmplx(c);
	im_c = get_image_dcmplx(c);

	set_real_dcmplx(c, re_c * re_a - im_c * im_a);
	set_image_dcmplx(c, re_c * im_a + im_c * re_a);
}

/* c := a / b */
void div_dcmplx(DCmplx c, DCmplx a, DCmplx b)
{
	double re_a, im_a, re_b, im_b, tmp, reim_b;

	re_a = get_real_dcmplx(a);
	im_a = get_image_dcmplx(a);
	re_b = get_real_dcmplx(b);
	im_b = get_image_dcmplx(b);

	if((re_b == 0) && (im_b == 0))
	{
		fprintf(stderr, "ERROR: Impossible division!\n");
		set0_dcmplx(c);
		return;
	}
	if((re_a == 0) && (im_a == 0))
	{
		set0_dcmplx(c);
		return;
	}

	if(fabs(re_b) >= fabs(im_b))
	{
		reim_b = im_b / re_b;
		tmp = re_b + reim_b * im_b;
		set_real_dcmplx(c, (re_a + reim_b * im_a) / tmp );
		set_image_dcmplx(c, (-re_a * reim_b + im_a) / tmp);
	}
	else
	{
		reim_b = re_b / im_b;
		tmp = re_b * reim_b + im_b;
		set_real_dcmplx(c, (re_a * reim_b + im_a) / tmp );
		set_image_dcmplx(c, (-re_a + im_a * reim_b) / tmp);
	}
}

/* exp(i*x) = cos(x) + i * sin(x) */
void iexp_dcmplx(DCmplx ret, double x)
{
	double re_ret, im_ret;

	set0_dcmplx(ret);
	re_ret = get_real_dcmplx(ret);
	im_ret = get_image_dcmplx(ret);

	re_ret = cos(x);
	im_ret = sin(x);

	set_real_dcmplx(ret, re_ret);
	set_image_dcmplx(ret, im_ret);
}

/* print complex */
void print_dcmplx(DCmplx a)
{
	double re_a, im_a;

	re_a = get_real_dcmplx(a);
	im_a = get_image_dcmplx(a);

	printf("(%25.17e, %25.17e)\n", re_a, im_a);
}

#ifdef USE_GMP
/* mpf_t :
	MPFCmplx init_mpfcmplx(void)
	void free_mpfcmplx(MPFCmplx a)
	void get_real_mpfcmplx(mpf_t ret, MPFCmplx a)
	void get_image_mpfcmplx(mpf_t ret, MPFCmplx a)
	void set_real_mpfcmplx(MPFCmplx c, mpf_t val)
	void set_image_mpfcmplx(MPFCmplx c, mpf_t val)
	void set_real_mpfcmplx_ui(MPFCmplx c, unsigned long val)
	void set_image_mpfcmplx_ui(MPFCmplx c, unsigned long val)
	void subst_mpfcmplx(MPFCmplx c, MPFCmplx a)
	void set0_mpfcmplx(MPFCmplx c)
	void add_mpfcmplx(MPFCmplx c, MPFCmplx a, MPFCmplx b)
	void add2_mpfcmplx(MPFCmplx c, MPFCmplx a)
	void sub_mpfcmplx(MPFCmplx c, MPFCmplx a, MPFCmplx b)
	void conj_mpfcmplx(MPFCmplx c, MPFCmplx a)
	void sign_mpfcmplx(MPFCmplx c, MPFCmplx a)
	void sign2_mpfcmplx(MPFCmplx c)
	void abs_mpfcmplx(mpf_t ret, MPFCmplx a)
	void mul_mpfcmplx(MPFCmplx c, MPFCmplx a, MPFCmplx b)
	void mul2_mpfcmplx(MPFCmplx c, MPFCmplx a)
	void div_mpfcmplx(MPFCmplx c, MPFCmplx a, MPFCmplx b)
	void print_mpfcmplx(MPFCmplx a)
*/

/* init */
MPFCmplx init_mpfcmplx(void)
{
	MPFCmplx ret;

	ret = (MPFCmplx)malloc(sizeof(mpfcmplx));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: init_mpfcmplx\n");
		return NULL;
	}

	mpf_init_set_ui(ret->re, 0UL);
	mpf_init_set_ui(ret->im, 0UL);
	ret->prec = mpf_get_prec(ret->re);

	return ret;
}

/* init2 */
MPFCmplx init2_mpfcmplx(unsigned long int prec)
{
	MPFCmplx ret;

	ret = (MPFCmplx)malloc(sizeof(mpfcmplx));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: init2_mpfcmplx\n");
		return NULL;
	}

	mpf_init2(ret->re, prec);
	mpf_init2(ret->im, prec);
	mpf_set_ui(ret->re, 0UL);
	mpf_set_ui(ret->im, 0UL);
	ret->prec = prec;

	return ret;
}

/* clear */
void free_mpfcmplx(MPFCmplx a)
{
	if(a != NULL)
	{
		mpf_clear(a->re);
		mpf_clear(a->im);
		free(a);
	}
}


/* return real part */
void get_real_mpfcmplx(mpf_t ret, MPFCmplx a)
{
	mpf_set(ret, a->re);
}

/* set real part */
void set_real_mpfcmplx(MPFCmplx c, mpf_t val)
{
	mpf_set(c->re, val);
}

/* set unsigned long real part*/
void set_real_mpfcmplx_ui(MPFCmplx c, unsigned long val)
{
	mpf_set_ui(c->re, val);
}

/* return imaginary part */
void get_image_mpfcmplx(mpf_t ret, MPFCmplx a)
{
	mpf_set(ret, a->im);
}

/* return imaginary part */
void set_image_mpfcmplx(MPFCmplx c, mpf_t val)
{
	mpf_set(c->im, val);
}

/* return unsigned long imaginary part */
void set_image_mpfcmplx_ui(MPFCmplx c, unsigned long val)
{
	mpf_set_ui(c->im, val);
}

/* c := a */
void subst_mpfcmplx(MPFCmplx c, MPFCmplx a)
{
	mpf_t re_a, im_a;

	mpf_init2(re_a, c->prec); // Fix! 2011-12-08
	mpf_init2(im_a, c->prec); // Fix! 2011-12-08

	get_real_mpfcmplx(re_a, a);
	get_image_mpfcmplx(im_a, a);

	set_real_mpfcmplx(c, re_a);
	set_image_mpfcmplx(c, im_a);

	mpf_clear(re_a);
	mpf_clear(im_a);

}

/* c := 0 */
void set0_mpfcmplx(MPFCmplx c)
{
	set_real_mpfcmplx_ui(c, 0UL);
	set_image_mpfcmplx_ui(c, 0UL);
}

/* c := a + b */
void add_mpfcmplx(MPFCmplx c, MPFCmplx a, MPFCmplx b)
{
	unsigned long prec;
	mpf_t tmp, re_a, im_a, re_b, im_b;

	// Fix! 2011-12-08
	prec = c->prec;

	mpf_init2(re_a, prec);
	mpf_init2(im_a, prec);
	mpf_init2(re_b, prec);
	mpf_init2(im_b, prec);

	get_real_mpfcmplx(re_a, a);
	get_image_mpfcmplx(im_a, a);
	get_real_mpfcmplx(re_b, b);
	get_image_mpfcmplx(im_b, b);

	mpf_init2(tmp, prec);

	mpf_add(tmp, re_a, re_b);
	set_real_mpfcmplx(c, tmp);
	mpf_add(tmp, im_a, im_b);
	set_image_mpfcmplx(c, tmp);

	mpf_clear(re_a);
	mpf_clear(im_a);
	mpf_clear(re_b);
	mpf_clear(im_b);

	mpf_clear(tmp);
}

/* c := a + real_b */
void add_mpfcmplx_mpf(MPFCmplx c, MPFCmplx a, mpf_t b)
{
	unsigned long prec;
	mpf_t tmp, re_a, im_a;

	// Fix! 2011-12-08
	prec = c->prec;

	mpf_init2(re_a, prec); // Fix! 2011-12-08
	mpf_init2(im_a, prec); // Fix! 2011-12-08

	get_real_mpfcmplx(re_a, a);
	get_image_mpfcmplx(im_a, a);

	mpf_init2(tmp, prec);

	mpf_add(tmp, re_a, b);
	set_real_mpfcmplx(c, tmp);
	set_image_mpfcmplx(c, im_a);

	mpf_clear(re_a);
	mpf_clear(im_a);

	mpf_clear(tmp);
}

/* c += a */
void add2_mpfcmplx(MPFCmplx c, MPFCmplx a)
{
	mpf_t re_a, im_a, re_c, im_c;

	// Fix! 2011-12-08
	mpf_init2(re_a, c->prec);
	mpf_init2(im_a, c->prec);
	mpf_init2(re_c, c->prec);
	mpf_init2(im_c, c->prec);

	get_real_mpfcmplx(re_a, a);
	get_image_mpfcmplx(im_a, a);
	get_real_mpfcmplx(re_c, c);
	get_image_mpfcmplx(im_c, c);

	mpf_add(re_c, re_c, re_a);
	set_real_mpfcmplx(c, re_c);
	mpf_add(im_c, im_c, im_a);
	set_image_mpfcmplx(c, im_c);

	mpf_clear(re_a);
	mpf_clear(im_a);
	mpf_clear(re_c);
	mpf_clear(im_c);
}

/* c := a - b */
void sub_mpfcmplx(MPFCmplx c, MPFCmplx a, MPFCmplx b)
{
	unsigned long prec;
	mpf_t tmp, re_a, im_a, re_b, im_b;

	// Fix! 2011-12-08
	prec = c->prec;

	mpf_init2(re_a, prec);
	mpf_init2(im_a, prec);
	mpf_init2(re_b, prec);
	mpf_init2(im_b, prec);

	get_real_mpfcmplx(re_a, a);
	get_image_mpfcmplx(im_a, a);
	get_real_mpfcmplx(re_b, b);
	get_image_mpfcmplx(im_b, b);

	mpf_init2(tmp, prec);

	mpf_sub(tmp, re_a, re_b);
	set_real_mpfcmplx(c, tmp);
	mpf_sub(tmp, im_a, im_b);
	set_image_mpfcmplx(c, tmp);

	mpf_clear(re_a);
	mpf_clear(im_a);
	mpf_clear(re_b);
	mpf_clear(im_b);

	mpf_clear(tmp);
}

/* c := re(a) - i * im(a) */
void conj_mpfcmplx(MPFCmplx c, MPFCmplx a)
{
	mpf_t re_a, im_a;

	// Fix! 2011-12-08
	mpf_init2(re_a, c->prec);
	mpf_init2(im_a, c->prec);

	get_real_mpfcmplx(re_a, a);
	get_image_mpfcmplx(im_a, a);

	set_real_mpfcmplx(c, re_a);
	mpf_neg(im_a, im_a);
	set_image_mpfcmplx(c, im_a);

	mpf_clear(re_a);
	mpf_clear(im_a);

}

/* c := -a */
void sign_mpfcmplx(MPFCmplx c, MPFCmplx a)
{
	mpf_t re_a, im_a;

	mpf_init2(re_a, c->prec); // Fix! 2011-12-08
	mpf_init2(im_a, c->prec); // Fix! 2011-12-08

	get_real_mpfcmplx(re_a, a);
	get_image_mpfcmplx(im_a, a);

	mpf_neg(re_a, re_a);
	mpf_neg(im_a, im_a);
	set_real_mpfcmplx(c, re_a);
	set_image_mpfcmplx(c, im_a); // Fix!!: 2008-06-04

	mpf_clear(re_a);
	mpf_clear(im_a);

}

/* c := -c */
void sign2_mpfcmplx(MPFCmplx c)
{
	mpf_t re_c, im_c;

	mpf_init2(re_c, c->prec);
	mpf_init2(im_c, c->prec);

	get_real_mpfcmplx(re_c, c);
	get_image_mpfcmplx(im_c, c);

	mpf_neg(re_c, re_c);
	set_real_mpfcmplx(c, re_c);
	mpf_neg(im_c, im_c);
	set_image_mpfcmplx(c, im_c); // Fix!!: 2008-06-04

	mpf_clear(re_c);
	mpf_clear(im_c);

}

/* |a| */
void abs_mpfcmplx(mpf_t ret, MPFCmplx a)
{
	unsigned long prec;
	mpf_t tmp, re_a, im_a;

	prec = mpf_get_prec(ret); // Fix! 2011-12-08

	mpf_init2(re_a, prec); // Fix! 2011-12-08
	mpf_init2(im_a, prec); // Fix! 2011-12-08
	mpf_init2(tmp, prec); // Fix! 2011-12-08

	get_real_mpfcmplx(re_a, a);
	get_image_mpfcmplx(im_a, a);

	mpf_mul(ret, re_a, re_a);
	mpf_mul(tmp, im_a, im_a);
	mpf_add(tmp, tmp, ret);
	mpf_sqrt(ret, tmp);

	mpf_clear(re_a);
	mpf_clear(im_a);
	mpf_clear(tmp);
}

/* c := a * b */
void mul_mpfcmplx(MPFCmplx c, MPFCmplx a, MPFCmplx b)
{
	unsigned long prec;
	mpf_t tmp, tmp1, tmp2, re_a, im_a, re_b, im_b;

	// Fix! 2011-12-08
	prec = c->prec;

	mpf_init2(re_a, prec);
	mpf_init2(im_a, prec);
	mpf_init2(re_b, prec);
	mpf_init2(im_b, prec);

	get_real_mpfcmplx(re_a, a);
	get_image_mpfcmplx(im_a, a);
	get_real_mpfcmplx(re_b, b);
	get_image_mpfcmplx(im_b, b);

	//prec = (a->prec > b->prec) ? a->prec : b->prec;
	mpf_init2(tmp , prec);
	mpf_init2(tmp1, prec);
	mpf_init2(tmp2, prec);

	mpf_mul(tmp1, re_a, re_b);
	mpf_mul(tmp2, im_a, im_b);
	mpf_sub(tmp, tmp1, tmp2);
	set_real_mpfcmplx(c, tmp);

	mpf_mul(tmp1, re_a, im_b);
	mpf_mul(tmp2, im_a, re_b);
	mpf_add(tmp, tmp1, tmp2);
	set_image_mpfcmplx(c, tmp);

	mpf_clear(re_a);
	mpf_clear(im_a);
	mpf_clear(re_b);
	mpf_clear(im_b);
	mpf_clear(tmp );
	mpf_clear(tmp1);
	mpf_clear(tmp2);

}

/* c := a * re_b */
void mul_mpfcmplx_mpf(MPFCmplx c, MPFCmplx a, mpf_t b)
{
	unsigned long prec;
	mpf_t tmp, re_a, im_a;

	// Fix! 2011-12-08
	prec = c->prec;

	mpf_init2(re_a, prec);
	mpf_init2(im_a, prec);

	get_real_mpfcmplx(re_a, a);
	get_image_mpfcmplx(im_a, a);

//	prec = mpf_get_prec(b);
//	if(prec < a->prec)
//		prec = a->prec;
	mpf_init2(tmp , prec);

	mpf_mul(tmp, re_a, b);
	set_real_mpfcmplx(c, tmp);

	mpf_mul(tmp, im_a, b);
	set_image_mpfcmplx(c, tmp);

	mpf_clear(re_a);
	mpf_clear(im_a);
	mpf_clear(tmp );

}

/* c := a * unsiged long int b */
void mul_mpfcmplx_ui(MPFCmplx c, MPFCmplx a, unsigned long int b)
{
	mpf_t tmp, re_a, im_a;

	// Fix! 2011-12-08
	mpf_init2(re_a, c->prec);
	mpf_init2(im_a, c->prec);

	get_real_mpfcmplx(re_a, a);
	get_image_mpfcmplx(im_a, a);

	mpf_init2(tmp, c->prec);

	mpf_mul_ui(tmp, re_a, b);
	set_real_mpfcmplx(c, tmp);

	mpf_mul_ui(tmp, im_a, b);
	set_image_mpfcmplx(c, tmp);

	mpf_clear(re_a);
	mpf_clear(im_a);
	mpf_clear(tmp );

}

/* c *= a */
void mul2_mpfcmplx(MPFCmplx c, MPFCmplx a)
{
	unsigned long prec;
	mpf_t tmp, tmp1, tmp2, re_a, im_a, re_c, im_c;

	// Fix! 2011-12-08
	prec = c->prec;

	mpf_init2(re_a, prec);
	mpf_init2(im_a, prec);
	mpf_init2(re_c, prec);
	mpf_init2(im_c, prec);

	//prec = (c->prec > a->prec) ? c->prec : a->prec;
	mpf_init2(tmp, prec);
	mpf_init2(tmp1, prec);
	mpf_init2(tmp2, prec);

	get_real_mpfcmplx(re_a, a);
	get_image_mpfcmplx(im_a, a);
	get_real_mpfcmplx(re_c, c);
	get_image_mpfcmplx(im_c, c);

	/* re_c * re_a - im_c * im_a */
	mpf_mul(tmp1, re_c, re_a);
	mpf_mul(tmp2, im_c, im_a);
	mpf_sub(tmp, tmp1, tmp2);
	set_real_mpfcmplx(c, tmp);

	/* re_c * im_a + im_c * re_a */
	mpf_mul(tmp1, re_c, im_a);
	mpf_mul(tmp2, im_c, re_a);
	mpf_add(tmp, tmp1, tmp2);
	set_image_mpfcmplx(c, tmp);

	mpf_clear(re_a);
	mpf_clear(im_a);
	mpf_clear(re_c);
	mpf_clear(im_c);
	mpf_clear(tmp );
	mpf_clear(tmp1);
	mpf_clear(tmp2);
}

/* c := a / b */
void div_mpfcmplx(MPFCmplx c, MPFCmplx a, MPFCmplx b)
{
	unsigned long prec;
	mpf_t tmp, tmp1, tmp2, reim_b, re_a, im_a, re_b, im_b;

	// Fix! 2011-12-08
	prec = c->prec;

	mpf_init2(re_a, prec);
	mpf_init2(im_a, prec);
	mpf_init2(re_b, prec);
	mpf_init2(im_b, prec);
	mpf_init2(reim_b, prec);

	get_real_mpfcmplx(re_a, a);
	get_image_mpfcmplx(im_a, a);
	get_real_mpfcmplx(re_b, b);
	get_image_mpfcmplx(im_b, b);

//	prec = (a->prec > b->prec) ? a->prec : b->prec;
	mpf_init2(tmp , prec);
	mpf_init2(tmp1, prec);
	mpf_init2(tmp2, prec);

//#ifndef USE_MPFR
	if((mpf_sgn(re_b) == 0) && (mpf_sgn(im_b) == 0))
	{
		fprintf(stderr, "ERROR: Impossible division!\n");
		set0_mpfcmplx(c);
		return;
	}
	if((mpf_sgn(re_a) == 0) && (mpf_sgn(im_a) == 0))
	{
		set0_mpfcmplx(c);
		return;
	}
//#endif


	mpf_abs(tmp, re_b); mpf_abs(tmp1, im_b);
	if(mpf_cmp(tmp, tmp1) >= 0)
	{
		/* reim_b = im_b / re_b */
		mpf_div(reim_b, im_b, re_b);
	
		/* tmp = re_b + reim_b * im_b */
		mpf_mul(tmp, im_b, reim_b);
		mpf_add(tmp, tmp, re_b);
	
		/* (re_a + reim_b * im_a) / tmp */
		mpf_mul(tmp2, im_a, reim_b);
		mpf_add(tmp2, tmp2, re_a);
		mpf_div(tmp1, tmp2, tmp);
		set_real_mpfcmplx(c, tmp1);
	
		/* (-re_a * reim_b + im_a) / tmp */
		mpf_mul(tmp2, re_a, reim_b);
		mpf_sub(tmp2, im_a, tmp2);
		mpf_div(tmp1, tmp2, tmp);
		set_image_mpfcmplx(c, tmp1);
	}
	else
	{
		/* reim_b = re_b / im_b */
		mpf_div(reim_b, re_b, im_b);
	
		/* tmp = re_b * reim_b + im_b */
		mpf_mul(tmp, re_b, reim_b);
		mpf_add(tmp, tmp, im_b);
	
		/* (re_a * reim_b + im_a) / tmp */
		mpf_mul(tmp2, re_a, reim_b);
		mpf_add(tmp2, tmp2, im_a);
		mpf_div(tmp1, tmp2, tmp);
		set_real_mpfcmplx(c, tmp1);
	
		/* (-re_a + im_a * reim_b) / tmp */
		mpf_mul(tmp2, im_a, reim_b);
		mpf_sub(tmp2, tmp2, re_a);
		mpf_div(tmp1, tmp2, tmp);
		set_image_mpfcmplx(c, tmp1);
	}

	mpf_clear(re_a);
	mpf_clear(im_a);
	mpf_clear(re_b);
	mpf_clear(im_b);
	mpf_clear(reim_b);
	mpf_clear(tmp );
	mpf_clear(tmp1);
	mpf_clear(tmp2);
}

/* exp(i*x) = cos(x) + i * sin(x) */
void iexp_mpfcmplx(MPFCmplx ret, mpf_t x)
{
	mpf_t re_ret, im_ret;

	set0_mpfcmplx(ret);

	mpf_init2(re_ret, ret->prec);
	mpf_init2(im_ret, ret->prec);

	get_real_mpfcmplx(re_ret, ret);
	get_image_mpfcmplx(im_ret, ret);

	mpf_cos(re_ret, x);
	mpf_sin(im_ret, x);

	set_real_mpfcmplx(ret, re_ret);
	set_image_mpfcmplx(ret, im_ret);

	mpf_clear(re_ret);
	mpf_clear(im_ret);
}

/* print complex */
void print_mpfcmplx(MPFCmplx a)
{
	mpf_t re_a, im_a;

	mpf_init2(re_a, a->prec);
	mpf_init2(im_a, a->prec);

	get_real_mpfcmplx(re_a, a);
	get_image_mpfcmplx(im_a, a);

	printf("(") ; mpf_out_str(stdout, 10, 0, re_a);
	printf(", "); mpf_out_str(stdout, 10, 0, im_a); 
	printf(")\n");

	mpf_clear(re_a);
	mpf_clear(im_a);
}

/******************************************************/
// 2011-11-27 -> will be added in complex.c of BNCpack

/* set unsigned long */
void set_mpfcmplx_ui_ui(MPFCmplx c, unsigned long re, unsigned long im)
{
	mpf_set_ui(c->re, re);
	mpf_set_ui(c->im, im);
}

/* set strings as complex number */
void set_mpfcmplx_str_str(MPFCmplx c, const char *re_str, long re_base, const char *im_str, long im_base)
{
	mpf_set_str(c->re, re_str, re_base);
	mpf_set_str(c->im, im_str, im_base);
}

/* set double precision real part */
void set_real_mpfcmplx_d(MPFCmplx c, double val)
{
	mpf_set_d(c->re, val);
}
/* set double precision imaginary part */
void set_image_mpfcmplx_d(MPFCmplx c, double val)
{
	mpf_set_d(c->im, val);
}
void set_mpfcmplx_d(MPFCmplx c, double _Complex val)
//void set_mpfcmplx_d(MPFCmplx c, DCmplx val)
{
	mpf_set_d(c->re, creal(val));
	mpf_set_d(c->im, cimag(val));
//	mpf_set_d(c->re, get_real_dcmplx(val));
//	mpf_set_d(c->im, get_image_dcmplx(val));
}

unsigned long int get_prec_mpfcmplx(MPFCmplx val)
{
	return val->prec;
}

mpf_ptr getp_real_mpfcmplx(MPFCmplx a)
{
	return (mpf_ptr)(a->re);
}
mpf_ptr getp_image_mpfcmplx(MPFCmplx a)
{
	return (mpf_ptr)(a->im);
}

/* c := a / (real)b */
void div_mpfcmplx_real(MPFCmplx c, MPFCmplx a, mpf_t b)
{
    mpf_div(getp_real_mpfcmplx(c), getp_real_mpfcmplx(a), b);
    mpf_div(getp_image_mpfcmplx(c), getp_image_mpfcmplx(a), b);
}

/* c := a / (double)b */
void div_mpfcmplx_d(MPFCmplx c, MPFCmplx a, double _Complex b)
//void div_mpfcmplx_d(MPFCmplx c, MPFCmplx a, DCmplx b)
{
	MPFCmplx tmp;

	tmp = init2_mpfcmplx(get_prec_mpfcmplx(c));

	set_mpfcmplx_d(tmp, b);
	div_mpfcmplx(c, a, tmp);

	free_mpfcmplx(tmp);
}

/* c := (double)a / b */
void div_d_mpfcmplx(MPFCmplx c, double _Complex a, MPFCmplx b)
//void div_mpfcmplx_d(MPFCmplx c, MPFCmplx a, DCmplx b)
{
	MPFCmplx tmp;

	tmp = init2_mpfcmplx(get_prec_mpfcmplx(c));

	set_mpfcmplx_d(tmp, a);
	div_mpfcmplx(c, tmp, b);

	free_mpfcmplx(tmp);
}

/* c := 1 / a */
void inv_mpfcmplx(MPFCmplx ret, MPFCmplx a)
{
    dcmplx one;
	// ret := 1
	set_mpfcmplx_d(ret, 1.0 + 0.0 * I);
    //set1_dcmplx(&one);
	//set_mpfcmplx_d(ret, &one);

	div_mpfcmplx(ret, ret, a);
}

/* c := -a */
void neg_mpfcmplx(MPFCmplx ret, MPFCmplx a)
{
	mpf_set(ret->re, a->re);
	mpf_set(ret->im, a->im);
	mpf_neg(ret->re, ret->re);
	mpf_neg(ret->im, ret->im);
}
#endif // USE_GMP

#ifdef __cpluplus
} // extern "C" {
#endif // __cplusplus
