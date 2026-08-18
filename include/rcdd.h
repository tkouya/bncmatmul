/********************************************************************************/
/* rcdd.h: Reverse definition for double-double and quadruple-double            */
/*                                                          complex arithmetic  */
/* Copyright (C) 2023 Tomonori Kouya                                            */
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
#ifndef __BNC_RCDD_H_
#define __BNC_RCDD_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h> // double complex

// Common defs
#include "bnc_common.h"

#include "rdd.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// cddfloat, ctdfloat, cqdfloat
#ifndef CDDFLOAT
#define CDDFLOAT
typedef struct { double val_re[DDSIZE]; double val_im[DDSIZE]; } cddfloat; // 53 * 2 = 106
typedef struct { double val_re[TDSIZE]; double val_im[TDSIZE]; } ctdfloat; // 53 * 3 = 159
typedef struct { double val_re[QDSIZE]; double val_im[QDSIZE]; } cqdfloat; // 53 * 4 = 212

// 2026-02-06(Fri) T.Kouya
static inline double *rcdd_realref(cddfloat *val) { return val->val_re; }
static inline double *rcdd_imagref(cddfloat *val) { return val->val_im; }
static inline double *rctd_realref(ctdfloat *val) { return val->val_re; }
static inline double *rctd_imagref(ctdfloat *val) { return val->val_im; }
static inline double *rcqd_realref(cqdfloat *val) { return val->val_re; }
static inline double *rcqd_imagref(cqdfloat *val) { return val->val_im; }
#endif // CDDFLOAT

//-------------------------------------
// DD complex arithmetic
//-------------------------------------
// ret := 0
static inline void rcdd_set0(cddfloat *ret)
{
    ret->val_re[0] = 0.0; ret->val_re[1] = 0.0;
    ret->val_im[0] = 0.0; ret->val_im[1] = 0.0;
}
static inline void set0_cdd(cddfloat *ret)
{
    ret->val_re[0] = 0.0; ret->val_re[1] = 0.0;
    ret->val_im[0] = 0.0; ret->val_im[1] = 0.0;
}

// ret := a
static inline void rcdd_set(cddfloat *ret, cddfloat *a)
{
    rdd_set(ret->val_re, a->val_re);
    rdd_set(ret->val_im, a->val_im);
}

// ret := a
static inline void rcdd_set_dd(cddfloat *ret, double a[DDSIZE])
{
    rdd_set(ret->val_re, a);
    rdd_set0(ret->val_im);
}
static inline void rcdd_set_ddfloat(cddfloat *ret, ddfloat *a)
{
    rdd_set(ret->val_re, a->val);
    rdd_set0(ret->val_im);
}

// ret := a
static inline void rcdd_set_dd_dd(cddfloat *ret, double a_real[DDSIZE], double a_imag[DDSIZE])
{
    rdd_set(ret->val_re, a_real);
    rdd_set(ret->val_im, a_imag);
}

// ret := a
static inline void rcdd_set_d(cddfloat *ret, double a)
{
    ret->val_re[0] = a;
    ret->val_re[1] = 0.0;
    rdd_set0(ret->val_im);
}

// 2024-11-23(Sat) T.Kouya
// 2026-02-06(Fri) T.Kouya
#define rcdd_set_si(ret, val) rcdd_set_d((ret), (double)(val))
#define rcdd_set_ui(ret, val) rcdd_set_d((ret), (double)(val))
#define rcdd_set_f(ret, val) rcdd_set_d((ret), (double)(val))

// ret := a
static inline void rcdd_set_cd(cddfloat *ret, double _Complex a)
{
    ret->val_re[0] = __real__ a; //creal(a);
    ret->val_re[1] = 0.0;
    ret->val_im[0] = __imag__ a; //cimag(a);
    ret->val_im[1] = 0.0;
}

// ret := a_real + a_imag * I
static inline void rcdd_set_d_d(cddfloat *ret, double a_real, double a_imag)
{
    ret->val_re[0] = a_real;
    ret->val_re[1] = 0.0;
    ret->val_im[0] = a_imag;
    ret->val_im[1] = 0.0;
}
#define rcdd_set_si_si(ret, a_real, a_imag) rcdd_set_d_d((ret), (double)(a_real), (double)(a_imag))
#define rcdd_set_ui_ui(ret, a_real, a_imag) rcdd_set_d_d((ret), (double)(a_real), (double)(a_imag))
#define rcdd_set_f_f(ret, val_re, val_im) rcdd_set_d_d((ret), (double)(val_re), (double)(val_im))

// ret := conj(a)
static inline void rcdd_conj(cddfloat *ret, cddfloat *a)
{
    rdd_set(ret->val_re, a->val_re);
    rdd_neg(ret->val_im, a->val_im);
}

// ret := -a
static inline void rcdd_neg(cddfloat *ret, cddfloat *a)
{
    rdd_neg(ret->val_re, a->val_re);
    rdd_neg(ret->val_im, a->val_im);
}

// ret := a + b
static inline void rcdd_add(cddfloat *ret, cddfloat *a, cddfloat *b)
{
    // ret := a + b
    rdd_add(ret->val_re, a->val_re, b->val_re);
    rdd_add(ret->val_im, a->val_im, b->val_im);
}

// ret := a + (double)b
static inline void rcdd_add_dd(cddfloat *ret, cddfloat *a, double b[DDSIZE])
{
    // ret := a + b
    rdd_add(ret->val_re, a->val_re, b); //b->val_re);
    rdd_set(ret->val_im, a->val_im); // , b->val_im);
}

// ret := a - b
static inline void rcdd_sub(cddfloat *ret, cddfloat *a, cddfloat *b)
{
    // ret := a - b
    rdd_sub(ret->val_re, a->val_re, b->val_re);
    rdd_sub(ret->val_im, a->val_im, b->val_im);
}

// ret := a - (double)b
static inline void rcdd_sub_dd(cddfloat *ret, cddfloat *a, double b[DDSIZE])
{
    // ret := a - b
    rdd_sub(ret->val_re, a->val_re, b);
    rdd_set(ret->val_im, a->val_im); // , b->val_im);
}

// ret := a * b : 4M
static inline void rcdd_mul_4m(cddfloat *ret, cddfloat *a, cddfloat *b)
{
    ddfloat t1, t2, t3, t4;

    // ret := a * b
    //rdd_mul(ret->val_re, a->val_re, b->val_re);
    rdd_mul(t1.val, a->val_re, b->val_re);
    rdd_mul(t2.val, a->val_im, b->val_im);
    rdd_mul(t3.val, a->val_re, b->val_im);
    rdd_mul(t4.val, a->val_im, b->val_re);

    rdd_sub(ret->val_re, t1.val, t2.val);
    rdd_add(ret->val_im, t3.val, t4.val);
}

// ret := a * b : 3M
static inline void rcdd_mul_3m(cddfloat *ret, cddfloat *a, cddfloat *b)
{
    ddfloat t1, t2, tmp, a_re_im, b_re_im;

    // ret := a * b

    // t1      := re_a * re_b
    // t2      := im_a * im_b
    // -> ret_re  := t1 - t2 
    rdd_mul(t1.val, a->val_re, b->val_re);
    rdd_mul(t2.val, a->val_im, b->val_im);
    rdd_sub(ret->val_re, t1.val, t2.val);
    /*
    printf("t1->"); rdd_out_str(t1.val); printf("\n");
    printf("t2->"); rdd_out_str(t2.val); printf("\n");
    printf("ret_re->"); rdd_out_str(ret->val_re); printf("\n"); 
    */

    // a_re_im := re_a + im_a
    // b_re_im := re_b + im_b
    // -> ret_im  := a_re_im * b_re_im - t1 - t2
    rdd_add(a_re_im.val, a->val_re, a->val_im);
    rdd_add(b_re_im.val, b->val_re, b->val_im);
    rdd_mul(tmp.val, a_re_im.val, b_re_im.val);
    //printf("tmp->"); rdd_out_str(tmp.val); printf("\n");
    rdd_sub(tmp.val, tmp.val, t1.val);
    //printf("tmp-t1->"); rdd_out_str(tmp.val); printf("\n");
    rdd_sub(ret->val_im, tmp.val, t2.val);
    //printf("tmp-t2->"); rdd_out_str(ret->val_im); printf("\n");

    //printf("rcdd_mul_3m_ret->"); rcdd_out_str(ret); printf("\n");
}

// ret := |a|^2
static inline void rcdd_nrm2(ddfloat *ret, cddfloat *a)
{
    ddfloat tmp_re2, tmp_im2;

    rdd_mul(tmp_re2.val, a->val_re, a->val_re);
    rdd_mul(tmp_im2.val, a->val_im, a->val_im);
    rdd_add(ret->val, tmp_re2.val, tmp_im2.val);
}
static inline void rcdd_nrm2_dd(double ret[DDSIZE], cddfloat *a)
{
    double tmp_re2[DDSIZE], tmp_im2[DDSIZE];

    rdd_mul(tmp_re2, a->val_re, a->val_re);
    rdd_mul(tmp_im2, a->val_im, a->val_im);
    rdd_add(ret, tmp_re2, tmp_im2);
}


// ret := |a|
static inline void rcdd_abs(ddfloat *ret, cddfloat *a)
{
    ddfloat tmp_re2, tmp_im2;

    rdd_mul(tmp_re2.val, a->val_re, a->val_re);
    rdd_mul(tmp_im2.val, a->val_im, a->val_im);
    rdd_add(ret->val, tmp_re2.val, tmp_im2.val);
    rdd_sqrt(ret->val, ret->val);
}
static inline void rcdd_abs_dd(double ret[DDSIZE], cddfloat *a)
{
    double tmp_re2[DDSIZE], tmp_im2[DDSIZE];

    rdd_mul(tmp_re2, a->val_re, a->val_re);
    rdd_mul(tmp_im2, a->val_im, a->val_im);
    rdd_add(ret, tmp_re2, tmp_im2);
    rdd_sqrt(ret, ret);
}

// ret := 1 / a
static inline void rcdd_inv_old(cddfloat *ret, cddfloat *a)
{
    ddfloat a_nrm2, tmp;
    cddfloat in_ret;

    //rcdd_conj(ret, a);
    rdd_set(in_ret.val_re, a->val_re);
    rdd_neg(in_ret.val_im, a->val_im);
    //rcdd_nrm2(&a_nrm2, a);
    rdd_mul(   tmp.val, a->val_re, a->val_re);
    rdd_mul(a_nrm2.val, a->val_im, a->val_im);
    rdd_add(a_nrm2.val, a_nrm2.val, tmp.val);

    rdd_div(ret->val_re, in_ret.val_re, a_nrm2.val);
    rdd_div(ret->val_im, in_ret.val_im, a_nrm2.val);
}
static inline void rcdd_inv(cddfloat *ret, cddfloat *a)
{
    double a_nrm2[DDSIZE];
    double tmp[DDSIZE];
    double neg_im[DDSIZE];
    cddfloat in_a;
    
    // エイリアシング対策
    rcdd_set(&in_a, a);
    
    // |a|² = a_re² + a_im²
    rdd_mul(tmp, in_a.val_re, in_a.val_re);
    rdd_mul(a_nrm2, in_a.val_im, in_a.val_im);
    rdd_add(a_nrm2, a_nrm2, tmp);
    
    // 実部 = a_re / |a|²
    rdd_div(ret->val_re, in_a.val_re, a_nrm2);
    
    // 虚部 = -a_im / |a|²（明示的に符号反転）
    rdd_neg(neg_im, in_a.val_im);
    rdd_div(ret->val_im, neg_im, a_nrm2);
}
// ret := a / b : 3M
static inline void rcdd_div_3m(cddfloat *ret, cddfloat *a, cddfloat *b)
{
    cddfloat inv_b, in_a;
    ddfloat t1, t2, tmp, a_re_im, b_re_im, b_nrm2;

    rcdd_inv(&inv_b, b);
    rcdd_set(&in_a, a);
    //printf("inv_b-> "); rcdd_out_str(&inv_b); printf("\n");
    rcdd_mul_3m(ret, &in_a, &inv_b);
    //printf("ret-> "); rcdd_out_str(ret); printf("\n");
}

// ret := a / b : 4M
static inline void rcdd_div_4m(cddfloat *ret, cddfloat *a, cddfloat *b)
{
    cddfloat in_a, inv_b;

    rcdd_set(&in_a, a);
    //printf("in_a = %25.17e + i* %25.17e\n", in_a.val_re[0], in_a.val_im[0]);
    rcdd_inv(&inv_b, b);
    //printf("inv_b = %25.17e + i* %25.17e\n", inv_b.val_re[0], inv_b.val_im[0]);
    rcdd_mul_4m(ret, &in_a, &inv_b); // fix! 2026-02-08(SUN) T.Kouya
    //printf("in_a *inv_b = %25.17e + i* %25.17e\n", ret->val_re[0], ret->val_im[0]);
#if 0
    double b_nrm2[DDSIZE];
    double tmp[DDSIZE];
    double neg_im[DDSIZE];
    cddfloat in_a, in_b, inv_b;
    
    // エイリアシング対策
    //rcdd_set(&in_a, a);
    rdd_set(in_b.val_re, b->val_re);
    rdd_set(in_b.val_im, b->val_im);

    // |a|² = a_re² + a_im²
    rdd_mul(tmp, in_b.val_re, in_b.val_re);
    rdd_mul(b_nrm2, in_b.val_im, in_b.val_im);
    rdd_add(b_nrm2, b_nrm2, tmp);
    
    // 実部 = a_re / |a|²
    rdd_div(inv_b.val_re, in_b.val_re, b_nrm2);
    
    // 虚部 = -a_im / |a|²（明示的に符号反転）
    rdd_neg(neg_im, in_b.val_im);
    rdd_div(inv_b.val_im, neg_im, b_nrm2);

    //rcdd_set(&in_a, a);
    rdd_set(in_a.val_re, a->val_re);
    rdd_set(in_a.val_im, a->val_im); 
    
    //rcdd_mul_4m(ret, &in_a, &inv_b); // fix! 2026-02-08(SUN) T.Kouya

    // ret := in_a * inv_b

    rdd_mul(ret->val_re, in_a.val_re, inv_b.val_re);
    rdd_mul(tmp, in_a.val_im, inv_b.val_im);
    rdd_sub(ret->val_re, ret->val_re, tmp);

    rdd_mul(ret->val_im, in_a.val_re, inv_b.val_im);
    rdd_mul(tmp, in_a.val_im, inv_b.val_re);
    rdd_add(ret->val_im, ret->val_im, tmp);
#endif // 0
}

// ret := a * (real)b = real(a) * b + imag(a) * b
static inline void rcdd_mul_dd(cddfloat *ret, cddfloat *a, double b[DDSIZE])
{
    rdd_mul(ret->val_re, a->val_re, b);
    rdd_mul(ret->val_im, a->val_im, b);
}

// ret := a * (real)b = real(a) * b + imag(a) * b
static inline void rcdd_mul_d(cddfloat *ret, cddfloat *a, double b)
{
    rdd_mul_d(ret->val_re, a->val_re, b);
    rdd_mul_d(ret->val_im, a->val_im, b);
}
#define rcdd_mul_ui(ret, a, b) rcdd_mul_d((ret), (a), (double)(b))
#define rcdd_mul_si(ret, a, b) rcdd_mul_d((ret), (a), (double)(b))

// ret := a * (real)b = real(a) / b + imag(a) / b
static inline void rcdd_div_dd(cddfloat *ret, cddfloat *a, double b[DDSIZE])
{
    rdd_div(ret->val_re, a->val_re, b);
    rdd_div(ret->val_im, a->val_im, b);
}

// ret := a * (real)b = real(a) / b + imag(a) / b
static inline void rcdd_div_d(cddfloat *ret, cddfloat *a, double b)
{
    rdd_div_d(ret->val_re, a->val_re, b);
    rdd_div_d(ret->val_im, a->val_im, b);
}
#define rcdd_div_ui(ret, a, b) rcdd_div_d((ret), (a), (double)(b))
#define rcdd_div_si(ret, a, b) rcdd_div_d((ret), (a), (double)(b))

// 2024-02-18(SUN) Tomonori Kouya
// Default: 4M method
#ifdef USE_RCDD_MUL_4M // 4M
    #define rcdd_mul rcdd_mul_4m
    #define rcdd_div rcdd_div_4m
#elif defined(USE_RCDD_MUL_3M) // 3M
    #define rcdd_mul rcdd_mul_3m
    #define rcdd_div rcdd_div_3m
#else // USE_RCDD_MUL_4M
    #define rcdd_mul rcdd_mul_4m
    #define rcdd_div rcdd_div_4m
    //#define rcdd_mul rcdd_mul_3m
    //#define rcdd_div rcdd_div_3m
#endif // USE_RCDD_MUL_4M

// ret := a * b : 4M
static inline void rcdd_mul_cd(cddfloat *ret, cddfloat *a, double _Complex b)
{
    ddfloat t1, t2, t3, t4;

    // ret := a * b
    //rdd_mul(ret->val_re, a->val_re, b->val_re);
    rdd_mul_d(t1.val, a->val_re, __real__ b); //creal(b));
    rdd_mul_d(t2.val, a->val_im, __imag__ b); //cimag(b));
    rdd_mul_d(t3.val, a->val_re, __imag__ b); //cimag(b));
    rdd_mul_d(t4.val, a->val_im, __real__ b); //creal(b));

    rdd_sub(ret->val_re, t1.val, t2.val);
    rdd_add(ret->val_im, t3.val, t4.val);
}

// ret := a / b : 4M
static inline void rcdd_div_cd(cddfloat *ret, cddfloat *a, double _Complex b)
{
    ddfloat t1, t2, t3, t4, abs_b2;

    // ret := a * conj(b) / |b|^2
    //rdd_mul(ret->val_re, a->val_re, b->val_re);
    rdd_mul_d(t1.val, a->val_re, __real__ b); // creal(b));
    rdd_mul_d(t2.val, a->val_im, __imag__ b); // cimag(b));
    rdd_mul_d(t3.val, a->val_re, __imag__ b); // cimag(b));
    rdd_mul_d(t4.val, a->val_im, __real__ b); // creal(b));

    // (a_re * b_re + a_im * b_im) + (a_im * b_re - a_re * b_im) * I
    rdd_add(ret->val_re, t1.val, t2.val);
    rdd_sub(ret->val_im, t4.val, t3.val);

    rdd_set_d(abs_b2.val, __real__ b);
    rdd_mul_d(abs_b2.val, abs_b2.val, __real__ b);
    rdd_set_d(t1.val, __imag__ b);
    rdd_mul_d(t1.val, t1.val, __imag__ b);
    rdd_add(abs_b2.val, abs_b2.val, t1.val);

    rdd_div(ret->val_re, ret->val_re, abs_b2.val);
    rdd_div(ret->val_im, ret->val_im, abs_b2.val);
}

// print cddfloat
static inline void rcdd_out_str(cddfloat *ret)
{
    rdd_out_str(ret->val_re);
    //rdd_out_str_base(stdout, 10, 32, ret->val_re);
    printf(" + ");
    rdd_out_str(ret->val_im);
    //rdd_out_str_base(stdout, 10, 32, ret->val_im);
    printf(" * I");
}

// 2024-12-03 (Tue) T.Kouya
// |a| >  |b| -> +1
// |a| == |b| ->  0
// |a| <  |b| -> -1
static inline int rcdd_cmp_abs(cddfloat *a, cddfloat *b)
{
    double abs_a[DDSIZE], abs_b[DDSIZE];

    rcdd_abs_dd(abs_a, a);
    rcdd_abs_dd(abs_b, b);

    return rdd_cmp(abs_a, abs_b);
}
static inline int rcdd_cmp_abs_dd(cddfloat *a, double b[DDSIZE])
{
    double abs_a[DDSIZE], abs_b[DDSIZE];

    rcdd_abs_dd(abs_a, a);
    rdd_abs(abs_b, b);

    return rdd_cmp(abs_a, abs_b);
}
static inline int rcdd_cmp_abs_d(cddfloat *a, double b)
{
    double abs_a[DDSIZE], abs_b;

    rcdd_abs_dd(abs_a, a);
    abs_b = fabs(b);

    return rdd_cmp_d(abs_a, abs_b);
}
#define rcdd_cmp_abs_ui(a, b) rcdd_cmp_abs_d((a), ((double)(b)))

// CDD fma
// ret = a * b + c
static inline void rcdd_fma(cddfloat *ret, cddfloat *a, cddfloat *b, cddfloat *c)
{
    cddfloat tmp;

	rcdd_mul(&tmp, a, b);
	rcdd_add(ret, &tmp, c);

	return;
}


//-------------------------------------
// TD complex arithmetic
//-------------------------------------
// ret := 0
static inline void rctd_set0(ctdfloat *ret)
{
    ret->val_re[0] = 0.0;  ret->val_re[1] = 0.0; ret->val_re[2] = 0.0;
    ret->val_im[0] = 0.0;  ret->val_im[1] = 0.0; ret->val_im[2] = 0.0;
}
static inline void set0_ctd(ctdfloat *ret)
{
    ret->val_re[0] = 0.0;  ret->val_re[1] = 0.0; ret->val_re[2] = 0.0;
    ret->val_im[0] = 0.0;  ret->val_im[1] = 0.0; ret->val_im[2] = 0.0;
}


// ret := a
static inline void rctd_set(ctdfloat *ret, ctdfloat *a)
{
    rtd_set(ret->val_re, a->val_re);
    rtd_set(ret->val_im, a->val_im);
}

// ret := a
static inline void rctd_set_tdfloat(ctdfloat *ret, tdfloat *a)
{
    rtd_set(ret->val_re, a->val);
    rtd_set0(ret->val_im);
}
static inline void rctd_set_td(ctdfloat *ret, double a[TDSIZE])
{
    rtd_set(ret->val_re, a);
    rtd_set0(ret->val_im);
}

// ret := a
static inline void rctd_set_td_td(ctdfloat *ret, double a_real[TDSIZE], double a_imag[TDSIZE])
{
    rtd_set(ret->val_re, a_real);
    rtd_set(ret->val_im, a_imag);
}

// ret := a
static inline void rctd_set_ddfloat(ctdfloat *ret, ddfloat *a)
{
    ret->val_re[0] = a->val[0];
    ret->val_re[1] = a->val[1];
    ret->val_re[2] = 0.0;
    rtd_set0(ret->val_im);
}
static inline void rctd_set_dd(ctdfloat *ret, double a[DDSIZE])
{
    ret->val_re[0] = a[0];
    ret->val_re[1] = a[1];
    ret->val_re[2] = 0.0;
    rtd_set0(ret->val_im);
}

// ret := a
static inline void rctd_set_d(ctdfloat *ret, double a)
{
    ret->val_re[0] = a;
    ret->val_re[1] = 0.0;
    ret->val_re[2] = 0.0;
    rtd_set0(ret->val_im);
}

// 2024-11-23(Sat) T.Kouya
#define rctd_set_ui(ret, val) rctd_set_d((ret), (double)(val))
#define rctd_set_si(ret, val) rctd_set_d((ret), (double)(val))
#define rctd_set_f(ret, val) rctd_set_d((ret), (double)(val))

// ret := a
static inline void rctd_set_cd(ctdfloat *ret, double _Complex a)
{
    ret->val_re[0] = __real__ a; //creal(a);
    ret->val_re[1] = 0.0;
    ret->val_re[2] = 0.0;
    ret->val_im[0] = __imag__ a; //cimag(a);
    ret->val_im[1] = 0.0;
    ret->val_im[2] = 0.0;
}

// ret := a_real + a_imag * I
static inline void rctd_set_d_d(ctdfloat *ret, double a_real, double a_imag)
{
    ret->val_re[0] = a_real;
    ret->val_re[1] = 0.0;
    ret->val_re[2] = 0.0;
    ret->val_im[0] = a_imag;
    ret->val_im[1] = 0.0;
    ret->val_im[2] = 0.0;
}
#define rctd_set_si_si(ret, a_real, a_imag) rctd_set_d_d((ret), (double)(a_real), (double)(a_imag))
#define rctd_set_ui_ui(ret, a_real, a_imag) rctd_set_d_d((ret), (double)(a_real), (double)(a_imag))
#define rctd_set_f_f(ret, a_real, a_imag) rctd_set_d_d((ret), (double)(a_real), (double)(a_imag))

// ret := a
static inline void rctd_set_cddfloat(ctdfloat *ret, cddfloat *a)
{
    ret->val_re[0] = a->val_re[0]; //creal(a);
    ret->val_re[1] = a->val_re[1];
    ret->val_re[2] = 0.0;
    ret->val_im[0] = a->val_im[0]; //cimag(a);
    ret->val_im[1] = a->val_im[1];
    ret->val_im[2] = 0.0;
}

// ret := conj(a)
static inline void rctd_conj(ctdfloat *ret, ctdfloat *a)
{
    rtd_set(ret->val_re, a->val_re);
    rtd_neg(ret->val_im, a->val_im);
}

// ret := -a
static inline void rctd_neg(ctdfloat *ret, ctdfloat *a)
{
    rtd_neg(ret->val_re, a->val_re);
    rtd_neg(ret->val_im, a->val_im);
}

// ret := a + b
static inline void rctd_add(ctdfloat *ret, ctdfloat *a, ctdfloat *b)
{
    // ret := a + b
    rtd_add(ret->val_re, a->val_re, b->val_re);
    rtd_add(ret->val_im, a->val_im, b->val_im);
}

// ret := a + (TD)b
static inline void rctd_add_td(ctdfloat *ret, ctdfloat *a, double b[TDSIZE])
{
    // ret := a + b
    rtd_add(ret->val_re, a->val_re, b); //b->val_re);
    rdd_set(ret->val_im, a->val_im);
}

// ret := a - b
static inline void rctd_sub(ctdfloat *ret, ctdfloat *a, ctdfloat *b)
{
    // ret := a - b
    rtd_sub(ret->val_re, a->val_re, b->val_re);
    rtd_sub(ret->val_im, a->val_im, b->val_im);
}

// ret := a - (TD)b
static inline void rctd_sub_td(ctdfloat *ret, ctdfloat *a, double b[TDSIZE])
{
    // ret := a - b
    rtd_sub(ret->val_re, a->val_re, b); //b->val_re);
    rdd_set(ret->val_im, a->val_im);
}

// ret := a * b : 4M
static inline void rctd_mul_4m(ctdfloat *ret, ctdfloat *a, ctdfloat *b)
{
    tdfloat t1, t2, t3, t4;

    // ret := a * b
    rtd_mul(t1.val, a->val_re, b->val_re);
    rtd_mul(t2.val, a->val_im, b->val_im);
    rtd_mul(t3.val, a->val_re, b->val_im);
    rtd_mul(t4.val, a->val_im, b->val_re);
    rtd_sub(ret->val_re, t1.val, t2.val);
    rtd_add(ret->val_im, t3.val, t4.val);
}

// ret := a * b : 3M
static inline void rctd_mul_3m(ctdfloat *ret, ctdfloat *a, ctdfloat *b)
{
    tdfloat t1, t2, a_re_im, b_re_im;

    // ret := a * b
    rtd_mul(t1.val, a->val_re, b->val_re);
    rtd_mul(t2.val, a->val_im, b->val_im);
    rtd_sub(ret->val_re, t1.val, t2.val);

    rtd_add(a_re_im.val, a->val_re, a->val_im);
    rtd_add(b_re_im.val, b->val_re, b->val_im);
    rtd_mul(ret->val_im, a_re_im.val, b_re_im.val);
    rtd_sub(ret->val_im, ret->val_im, t1.val);
    rtd_sub(ret->val_im, ret->val_im, t2.val);
}

// ret := |a|^2
static inline void rctd_nrm2(tdfloat *ret, ctdfloat *a)
{
    tdfloat tmp_re2, tmp_im2;

    rtd_mul(tmp_re2.val, a->val_re, a->val_re);
    rtd_mul(tmp_im2.val, a->val_im, a->val_im);
    rtd_add(ret->val, tmp_re2.val, tmp_im2.val);
}
static inline void rctd_nrm2_td(double ret[TDSIZE], ctdfloat *a)
{
    double tmp_re2[TDSIZE], tmp_im2[TDSIZE];

    rtd_mul(tmp_re2, a->val_re, a->val_re);
    rtd_mul(tmp_im2, a->val_im, a->val_im);
    rtd_add(ret, tmp_re2, tmp_im2);
}

// ret := |a|
static inline void rctd_abs(tdfloat *ret, ctdfloat *a)
{
    tdfloat tmp_re2, tmp_im2;

    rtd_mul(tmp_re2.val, a->val_re, a->val_re);
    rtd_mul(tmp_im2.val, a->val_im, a->val_im);
    rtd_add(ret->val, tmp_re2.val, tmp_im2.val);
    rtd_sqrt(ret->val, ret->val);
}
static inline void rctd_abs_td(double ret[TDSIZE], ctdfloat *a)
{
    double tmp_re2[TDSIZE], tmp_im2[TDSIZE];

    rtd_mul(tmp_re2, a->val_re, a->val_re);
    rtd_mul(tmp_im2, a->val_im, a->val_im);
    rtd_add(ret, tmp_re2, tmp_im2);
    rtd_sqrt(ret, ret);
}

// ret := 1 / a
static inline void rctd_inv(ctdfloat *ret, ctdfloat *a)
{
    tdfloat a_nrm2, tmp;
    ctdfloat in_ret;

    //rctd_conj(ret, a);
    rtd_set(in_ret.val_re, a->val_re);
    rtd_neg(in_ret.val_im, a->val_im);

    //rctd_nrm2(&a_nrm2, a);
    rtd_mul(a_nrm2.val, a->val_re, a->val_re);
    rtd_mul(tmp.val, a->val_im, a->val_im);
    rtd_add(a_nrm2.val, a_nrm2.val, tmp.val);

    rtd_div(ret->val_re, in_ret.val_re, a_nrm2.val);
    rtd_div(ret->val_im, in_ret.val_im, a_nrm2.val);
}

// ret := a / b : 3M
static inline void rctd_div_3m(ctdfloat *ret, ctdfloat *a, ctdfloat *b)
{
    ctdfloat inv_b;

    rctd_inv(&inv_b, b);
    rctd_mul_3m(ret, a, &inv_b);
}

// ret := a / b : 4M
static inline void rctd_div_4m(ctdfloat *ret, ctdfloat *a, ctdfloat *b)
{
    ctdfloat in_a, inv_b;

    rctd_inv(&inv_b, b);
    rctd_set(&in_a, a);
    rctd_mul_4m(ret, &in_a, &inv_b);
}

// ret := a * (real)b = real(a) * b + imag(a) * b
static inline void rctd_mul_td(ctdfloat *ret, ctdfloat *a, double b[TDSIZE])
{
    rtd_mul(ret->val_re, a->val_re, b);
    rtd_mul(ret->val_im, a->val_im, b);
}

// ret := a * (real)b = real(a) * b + imag(a) * b
static inline void rctd_mul_d(ctdfloat *ret, ctdfloat *a, double b)
{
    rtd_mul_d(ret->val_re, a->val_re, b);
    rtd_mul_d(ret->val_im, a->val_im, b);
}
#define rctd_mul_ui(ret, a, b) rctd_mul_d((ret), (a), (double)(b))
#define rctd_mul_si(ret, a, b) rctd_mul_d((ret), (a), (double)(b))

// ret := a * (real)b = real(a) / b + imag(a) / b
static inline void rctd_div_td(ctdfloat *ret, ctdfloat *a, double b[TDSIZE])
{
    rtd_div(ret->val_re, a->val_re, b);
    rtd_div(ret->val_im, a->val_im, b);
}

// ret := a * (real)b = real(a) / b + imag(a) / b
static inline void rctd_div_d(ctdfloat *ret, ctdfloat *a, double b)
{
    rtd_div_d(ret->val_re, a->val_re, b);
    rtd_div_d(ret->val_im, a->val_im, b);
}
#define rctd_div_ui(ret, a, b) rctd_div_d((ret), (a), (double)(b))
#define rctd_div_si(ret, a, b) rctd_div_d((ret), (a), (double)(b))


// 2024-02-18(SUN) Tomonori Kouya
// Default: 4M method
#ifdef USE_RCTD_MUL_4M // 4M
    #define rctd_mul rctd_mul_4m
    #define rctd_div rctd_div_4m
#elif defined(USE_RCTD_MUL_3M) // 3M
    #define rctd_mul rctd_mul_3m
    #define rctd_div rctd_div_3m
#else // USE_RCTD_MUL_4M
    #define rctd_mul rctd_mul_4m
    #define rctd_div rctd_div_4m
    //#define rctd_mul rctd_mul_3m
    //#define rctd_div rctd_div_3m
#endif // USE_RCDD_MUL_4M

// ret := a * b : 4M
static inline void rctd_mul_cd(ctdfloat *ret, ctdfloat *a, double _Complex b)
{
    tdfloat t1, t2, t3, t4;

    // ret := a * b
    //rdd_mul(ret->val_re, a->val_re, b->val_re);
    rtd_mul_d(t1.val, a->val_re, __real__ b); //creal(b));
    rtd_mul_d(t2.val, a->val_im, __imag__ b); //cimag(b));
    rtd_mul_d(t3.val, a->val_re, __imag__ b); //cimag(b));
    rtd_mul_d(t4.val, a->val_im, __real__ b); //creal(b));

    rtd_sub(ret->val_re, t1.val, t2.val);
    rtd_add(ret->val_im, t3.val, t4.val);
}

// ret := a / b : 4M
static inline void rctd_div_cd(ctdfloat *ret, ctdfloat *a, double _Complex b)
{
    tdfloat t1, t2, t3, t4, abs_b2;

    // ret := a * conj(b) / |b|^2
    //rdd_mul(ret->val_re, a->val_re, b->val_re);
    rtd_mul_d(t1.val, a->val_re, __real__ b); // creal(b));
    rtd_mul_d(t2.val, a->val_im, __imag__ b); // cimag(b));
    rtd_mul_d(t3.val, a->val_re, __imag__ b); // cimag(b));
    rtd_mul_d(t4.val, a->val_im, __real__ b); // creal(b));

    rtd_add(ret->val_re, t1.val, t2.val);
    rtd_sub(ret->val_im, t4.val, t3.val);

    rtd_set_d(abs_b2.val, __real__ b);
    rtd_mul_d(abs_b2.val, abs_b2.val, __real__ b);
    rtd_set_d(t1.val, __imag__ b);
    rtd_mul_d(t1.val, t1.val, __imag__ b);
    rtd_add(abs_b2.val, abs_b2.val, t1.val);

    rtd_div(ret->val_re, ret->val_re, abs_b2.val);
    rtd_div(ret->val_im, ret->val_im, abs_b2.val);
}

// print ctdfloat
static inline void rctd_out_str(ctdfloat *ret)
{
    //rtd_out_str(ret->val_re);
    rtd_out_str_base(stdout, 10, 48, ret->val_re);
    printf(" + ");
    //rtd_out_str(ret->val_im);
    rtd_out_str_base(stdout, 10, 48, ret->val_im);
    printf(" * I");
}

// 2024-12-03 (Tue) T.Kouya
// |a| >  |b| -> +1
// |a| == |b| ->  0
// |a| <  |b| -> -1
static inline int rctd_cmp_abs(ctdfloat *a, ctdfloat *b)
{
    double abs_a[TDSIZE], abs_b[TDSIZE];

    rctd_abs_td(abs_a, a);
    rctd_abs_td(abs_b, b);

    return rtd_cmp(abs_a, abs_b);
}
static inline int rctd_cmp_abs_td(ctdfloat *a, double b[TDSIZE])
{
    double abs_a[TDSIZE], abs_b[TDSIZE];

    rctd_abs_td(abs_a, a);
    rtd_abs(abs_b, b);

    return rtd_cmp(abs_a, abs_b);
}
static inline int rctd_cmp_abs_d(ctdfloat *a, double b)
{
    double abs_a[TDSIZE], abs_b;

    rctd_abs_td(abs_a, a);
    abs_b = fabs(b);

    return rtd_cmp_d(abs_a, abs_b);
}
#define rctd_cmp_abs_ui(a, b) rctd_cmp_abs_d((a), ((double)(b)))

// CTD fma
// ret = a * b + c
static inline void rctd_fma(ctdfloat *ret, ctdfloat *a, ctdfloat *b, ctdfloat *c)
{
    ctdfloat tmp;

	rctd_mul(&tmp, a, b);
	rctd_add(ret, &tmp, c);

	return;
}

//-------------------------------------
// QD complex arithmetic
//-------------------------------------
// ret := 0
static inline void rcqd_set0(cqdfloat *ret)
{
    ret->val_re[0] = 0.0;  ret->val_re[1] = 0.0; ret->val_re[2] = 0.0; ret->val_re[3] = 0.0;
    ret->val_im[0] = 0.0;  ret->val_im[1] = 0.0; ret->val_im[2] = 0.0; ret->val_im[3] = 0.0;
}
static inline void set0_cqd(cqdfloat *ret)
{
    ret->val_re[0] = 0.0;  ret->val_re[1] = 0.0; ret->val_re[2] = 0.0; ret->val_re[3] = 0.0;
    ret->val_im[0] = 0.0;  ret->val_im[1] = 0.0; ret->val_im[2] = 0.0; ret->val_im[3] = 0.0;
}

// ret := a
static inline void rcqd_set(cqdfloat *ret, cqdfloat *a)
{
    rqd_set(ret->val_re, a->val_re);
    rqd_set(ret->val_im, a->val_im);
}

// ret := a
static inline void rcqd_set_qdfloat(cqdfloat *ret, qdfloat *a)
{
    rqd_set(ret->val_re, a->val);
    rqd_set0(ret->val_im);
}
static inline void rcqd_set_qd(cqdfloat *ret, double a[QDSIZE])
{
    rqd_set(ret->val_re, a);
    rqd_set0(ret->val_im);
}

// ret := a
static inline void rcqd_set_qd_qd(cqdfloat *ret, double a_real[QDSIZE], double a_imag[QDSIZE])
{
    rqd_set(ret->val_re, a_real);
    rqd_set(ret->val_im, a_imag);
}

// ret := a
static inline void rcqd_set_ddfloat(cqdfloat *ret, ddfloat *a)
{
    ret->val_re[0] = a->val[0];
    ret->val_re[1] = a->val[1];
    ret->val_re[2] = 0.0;
    ret->val_re[3] = 0.0;
    rqd_set0(ret->val_im);
}

// ret := a
static inline void rcqd_set_tdfloat(cqdfloat *ret, tdfloat *a)
{
    ret->val_re[0] = a->val[0];
    ret->val_re[1] = a->val[1];
    ret->val_re[2] = a->val[2];
    ret->val_re[3] = 0.0;
    rqd_set0(ret->val_im);
}

// ret := a
static inline void rcqd_set_d(cqdfloat *ret, double a)
{
    ret->val_re[0] = a;
    ret->val_re[1] = 0.0;
    ret->val_re[2] = 0.0;
    ret->val_re[3] = 0.0;
    rqd_set0(ret->val_im);
}

// 2024-11-23(Sat) T.Kouya
#define rcqd_set_ui(ret, val) rcqd_set_d((ret), (double)(val))
#define rcqd_set_si(ret, val) rcqd_set_d((ret), (double)(val))
#define rcqd_set_f (ret, val) rcqd_set_d((ret), (double)(val))

// ret := a
static inline void rcqd_set_cd(cqdfloat *ret, double _Complex a)
{
    ret->val_re[0] = __real__ a; //creal(a);
    ret->val_re[1] = 0.0;
    ret->val_re[2] = 0.0;
    ret->val_re[3] = 0.0;
    ret->val_im[0] = __imag__ a; //cimag(a);
    ret->val_im[1] = 0.0;
    ret->val_im[2] = 0.0;
    ret->val_im[3] = 0.0;
}

// ret := a_real + a_imag * I
static inline void rcqd_set_d_d(cqdfloat *ret, double a_real, double a_imag)
{
    ret->val_re[0] = a_real;
    ret->val_re[1] = 0.0;
    ret->val_re[2] = 0.0;
    ret->val_re[3] = 0.0;
    ret->val_im[0] = a_imag;
    ret->val_im[1] = 0.0;
    ret->val_im[2] = 0.0;
    ret->val_im[3] = 0.0;
}
#define rcqd_set_si_si(ret, a_real, a_imag) rcqd_set_d_d((ret), (double)(a_real), (double)(a_imag))
#define rcqd_set_ui_ui(ret, a_real, a_imag) rcqd_set_d_d((ret), (double)(a_real), (double)(a_imag))
#define rcqd_set_f_f  (ret, a_real, a_imag) rcqd_set_d_d((ret), (double)(a_real), (double)(a_imag))

// ret := a
static inline void rcqd_set_cdd(cqdfloat *ret, cddfloat *a)
{
    ret->val_re[0] = a->val_re[0]; //creal(a);
    ret->val_re[1] = a->val_re[1];
    ret->val_re[2] = 0.0;
    ret->val_re[3] = 0.0;
    ret->val_im[0] = a->val_im[0]; //cimag(a);
    ret->val_im[1] = a->val_im[1];
    ret->val_im[2] = 0.0;
    ret->val_im[3] = 0.0;
}

// ret := a
static inline void rcqd_set_ctd(cqdfloat *ret, ctdfloat *a)
{
    ret->val_re[0] = a->val_re[0]; //creal(a);
    ret->val_re[1] = a->val_re[1];
    ret->val_re[2] = a->val_re[2];
    ret->val_re[3] = 0.0;
    ret->val_im[0] = a->val_im[0]; //cimag(a);
    ret->val_im[1] = a->val_im[1];
    ret->val_im[2] = a->val_im[2];
    ret->val_im[3] = 0.0;
}

// ret := conj(a)
static inline void rcqd_conj(cqdfloat *ret, cqdfloat *a)
{
    rqd_set(ret->val_re, a->val_re);
    rqd_neg(ret->val_im, a->val_im);
}

// ret := -a
static inline void rcqd_neg(cqdfloat *ret, cqdfloat *a)
{
    rqd_neg(ret->val_re, a->val_re);
    rqd_neg(ret->val_im, a->val_im);
}

// ret := a + b
static inline void rcqd_add(cqdfloat *ret, cqdfloat *a, cqdfloat *b)
{
    // ret := a + b
    rqd_add(ret->val_re, a->val_re, b->val_re);
    rqd_add(ret->val_im, a->val_im, b->val_im);
}

// ret := a + (QD)b
static inline void rcqd_add_qd(cqdfloat *ret, cqdfloat *a, double b[QDSIZE])
{
    // ret := a + b
    rqd_add(ret->val_re, a->val_re, b); // ->val_re);
    rqd_set(ret->val_im, a->val_im); // , b->val_im);
}

// ret := a - b
static inline void rcqd_sub(cqdfloat *ret, cqdfloat *a, cqdfloat *b)
{
    // ret := a - b
    rqd_sub(ret->val_re, a->val_re, b->val_re);
    rqd_sub(ret->val_im, a->val_im, b->val_im);
}

// ret := a - (QD)b
static inline void rcqd_sub_qd(cqdfloat *ret, cqdfloat *a, double b[QDSIZE])
{
    // ret := a - b
    rqd_sub(ret->val_re, a->val_re, b); // ->val_re);
    rqd_set(ret->val_im, a->val_im); // , b->val_im);
}

// ret := a * b : 4M
static inline void rcqd_mul_4m(cqdfloat *ret, cqdfloat *a, cqdfloat *b)
{
    qdfloat t1, t2, t3, t4;

    // ret := a * b
    rqd_mul(t1.val, a->val_re, b->val_re);
    rqd_mul(t2.val, a->val_im, b->val_im);
    rqd_mul(t3.val, a->val_re, b->val_im);
    rqd_mul(t4.val, a->val_im, b->val_re);
    rqd_sub(ret->val_re, t1.val, t2.val);
    rqd_add(ret->val_im, t3.val, t4.val);
}

// ret := a * b : 3M
static inline void rcqd_mul_3m(cqdfloat *ret, cqdfloat *a, cqdfloat *b)
{
    qdfloat t1, t2, a_re_im, b_re_im;

    // ret := a * b
    // t1      := re_a * re_b
    // t2      := im_a * im_b
    // -> ret_re  := t1 - t2 
    // a_re_im := re_a + im_a
    // b_re_im := re_b + im_b
    // -> ret_im  := a_re_im * b_re_im - t1 - t2
    rqd_mul(t1.val, a->val_re, b->val_re);
    rqd_mul(t2.val, a->val_im, b->val_im);
    rqd_sub(ret->val_re, t1.val, t2.val);

    rqd_add(a_re_im.val, a->val_re, a->val_im);
    rqd_add(b_re_im.val, b->val_re, b->val_im);
    rqd_mul(ret->val_im, a_re_im.val, b_re_im.val);
    rqd_sub(ret->val_im, ret->val_im, t1.val);
    rqd_sub(ret->val_im, ret->val_im, t2.val);
}

// ret := |a|^2
static inline void rcqd_nrm2(qdfloat *ret, cqdfloat *a)
{
    qdfloat tmp_re2, tmp_im2;

    rqd_mul(tmp_re2.val, a->val_re, a->val_re);
    rqd_mul(tmp_im2.val, a->val_im, a->val_im);
    rqd_add(ret->val, tmp_re2.val, tmp_im2.val);
}
static inline void rcqd_nrm2_qd(double ret[QDSIZE], cqdfloat *a)
{
    double tmp_re2[QDSIZE], tmp_im2[QDSIZE];

    rqd_mul(tmp_re2, a->val_re, a->val_re);
    rqd_mul(tmp_im2, a->val_im, a->val_im);
    rqd_add(ret, tmp_re2, tmp_im2);
}

// ret := |a|
static inline void rcqd_abs(qdfloat *ret, cqdfloat *a)
{
    qdfloat tmp_re2, tmp_im2;

    rqd_mul(tmp_re2.val, a->val_re, a->val_re);
    rqd_mul(tmp_im2.val, a->val_im, a->val_im);
    rqd_add(ret->val, tmp_re2.val, tmp_im2.val);
#ifdef USE_GMP
#else // USE_GMP
    rqd_sqrt(ret->val, ret->val);
#endif // USE_GMP
}
static inline void rcqd_abs_qd(double ret[QDSIZE], cqdfloat *a)
{
    double tmp_re2[QDSIZE], tmp_im2[QDSIZE];

    rqd_mul(tmp_re2, a->val_re, a->val_re);
    rqd_mul(tmp_im2, a->val_im, a->val_im);
    rqd_add(ret, tmp_re2, tmp_im2);
    rqd_sqrt(ret, ret);
}

// ret := 1 / a
static inline void rcqd_inv(cqdfloat *ret, cqdfloat *a)
{
    qdfloat a_nrm2, tmp;
    cqdfloat in_a;

    rcqd_set(&in_a, a);

    //rcqd_conj(ret, a);
    rqd_set(ret->val_re, in_a.val_re);
    rqd_neg(ret->val_im, in_a.val_im);

    //rcqd_nrm2(&a_nrm2, a);
    rqd_mul(a_nrm2.val, in_a.val_re, in_a.val_re);
    rqd_mul(tmp.val, in_a.val_im, in_a.val_im);
    rqd_add(a_nrm2.val, a_nrm2.val, tmp.val);

    rqd_div(ret->val_re, ret->val_re, a_nrm2.val);
    rqd_div(ret->val_im, ret->val_im, a_nrm2.val);
}

// ret := a / b : 3M
static inline void rcqd_div_3m(cqdfloat *ret, cqdfloat *a, cqdfloat *b)
{
    cqdfloat in_a, inv_b;

    rcqd_inv(&inv_b, b);
    rcqd_set(&in_a, a);
    rcqd_mul_3m(ret, &in_a, &inv_b);
}

// ret := a / b : 4M
static inline void rcqd_div_4m(cqdfloat *ret, cqdfloat *a, cqdfloat *b)
{
    cqdfloat in_a, inv_b;

    rcqd_inv(&inv_b, b);
    rcqd_set(&in_a, a); // fix! 2026-02-06(Sun) T.Kouya
    rcqd_mul_4m(ret, &in_a, &inv_b);
}

// ret := a * (real)b = real(a) * b + imag(a) * b
static inline void rcqd_mul_qd(cqdfloat *ret, cqdfloat *a, double b[QDSIZE])
{
    rqd_mul(ret->val_re, a->val_re, b);
    rqd_mul(ret->val_im, a->val_im, b);
}

// ret := a * (real)b = real(a) * b + imag(a) * b
static inline void rcqd_mul_d(cqdfloat *ret, cqdfloat *a, double b)
{
    rqd_mul_d(ret->val_re, a->val_re, b);
    rqd_mul_d(ret->val_im, a->val_im, b);
}
#define rcqd_mul_ui(ret, a, b) rcqd_mul_d((ret), (a), (double)(b))
#define rcqd_mul_si(ret, a, b) rcqd_mul_d((ret), (a), (double)(b))

// ret := a * (real)b = real(a) / b + imag(a) / b
static inline void rcqd_div_qd(cqdfloat *ret, cqdfloat *a, double b[QDSIZE])
{
    rqd_div(ret->val_re, a->val_re, b);
    rqd_div(ret->val_im, a->val_im, b);
}

// ret := a * (real)b = real(a) / b + imag(a) / b
static inline void rcqd_div_d(cqdfloat *ret, cqdfloat *a, double b)
{
    rqd_div_d(ret->val_re, a->val_re, b);
    rqd_div_d(ret->val_im, a->val_im, b);
}
#define rcqd_div_ui(ret, a, b) rcqd_div_d((ret), (a), (double)(b))
#define rcqd_div_si(ret, a, b) rcqd_div_d((ret), (a), (double)(b))

// 2024-02-18(SUN) Tomonori Kouya
// Default: 4M method
#ifdef USE_RCQD_MUL_4M // 4M
    #define rcqd_mul rcqd_mul_4m
    #define rcqd_div rcqd_div_4m
#elif defined(USE_RCQD_MUL_3M) // 3M
    #define rcqd_mul rcqd_mul_3m
    #define rcqd_div rcqd_div_3m
#else // USE_RCDD_MUL_4M or 3M
    #define rcqd_mul rcqd_mul_4m
    #define rcqd_div rcqd_div_4m
    //#define rcqd_mul rcqd_mul_3m
    //#define rcqd_div rcqd_div_3m
#endif // USE_RCDD_MUL_4M

// ret := a * b : 4M
static inline void rcqd_mul_cd(cqdfloat *ret, cqdfloat *a, double _Complex b)
{
    qdfloat t1, t2, t3, t4;

    // ret := a * b
    //rdd_mul(ret->val_re, a->val_re, b->val_re);
    rqd_mul_d(t1.val, a->val_re, __real__ b); // creal(b));
    rqd_mul_d(t2.val, a->val_im, __imag__ b); // cimag(b));
    rqd_mul_d(t3.val, a->val_re, __imag__ b); // cimag(b));
    rqd_mul_d(t4.val, a->val_im, __real__ b); // creal(b));

    rqd_sub(ret->val_re, t1.val, t2.val);
    rqd_add(ret->val_im, t3.val, t4.val);
}

// ret := a / b : 4M
static inline void rcqd_div_cd(cqdfloat *ret, cqdfloat *a, double _Complex b)
{
    qdfloat t1, t2, t3, t4, abs_b2;

    // ret := a * conj(b) / |b|^2
    //rdd_mul(ret->val_re, a->val_re, b->val_re);
    rqd_mul_d(t1.val, a->val_re, __real__ b); // creal(b));
    rqd_mul_d(t2.val, a->val_im, __imag__ b); // cimag(b));
    rqd_mul_d(t3.val, a->val_re, __imag__ b); // cimag(b));
    rqd_mul_d(t4.val, a->val_im, __real__ b); // creal(b));

    rqd_add(ret->val_re, t1.val, t2.val);
    rqd_sub(ret->val_im, t4.val, t3.val);

    rqd_set_d(abs_b2.val, __real__ b);
    rqd_mul_d(abs_b2.val, abs_b2.val, __real__ b);
    rqd_set_d(t1.val, __imag__ b);
    rqd_mul_d(t1.val, t1.val, __imag__ b);
    rqd_add(abs_b2.val, abs_b2.val, t1.val);

    rqd_div(ret->val_re, ret->val_re, abs_b2.val);
    rqd_div(ret->val_im, ret->val_im, abs_b2.val);
}

// print cqdfloat
static inline void rcqd_out_str(cqdfloat *ret)
{
    rqd_out_str(ret->val_re);
    //rqd_out_str_base(stdout, 10, 64, ret->val_re);
    printf(" + ");
    rqd_out_str(ret->val_im);
    //rqd_out_str_base(stdout, 10, 64, ret->val_im);
    printf(" * I");
}

// 2024-12-03 (Tue) T.Kouya
// |a| >  |b| -> +1
// |a| == |b| ->  0
// |a| <  |b| -> -1
static inline int rcqd_cmp_abs(cqdfloat *a, cqdfloat *b)
{
    double abs_a[QDSIZE], abs_b[QDSIZE];

    rcqd_abs_qd(abs_a, a);
    rcqd_abs_qd(abs_b, b);

    return rqd_cmp(abs_a, abs_b);
}
static inline int rcqd_cmp_abs_qd(cqdfloat *a, double b[QDSIZE])
{
    double abs_a[QDSIZE], abs_b[QDSIZE];

    rcqd_abs_qd(abs_a, a);
    rqd_abs(abs_b, b);

    return rqd_cmp(abs_a, abs_b);
}
static inline int rcqd_cmp_abs_d(cqdfloat *a, double b)
{
    double abs_a[QDSIZE], abs_b;

    rcqd_abs_qd(abs_a, a);
    abs_b = fabs(b);

    return rdd_cmp_d(abs_a, abs_b);
}
#define rcqd_cmp_abs_ui(a, b) rcqd_cmp_abs_d((a), ((double)(b)))

// CQD fma
// ret = a * b + c
static inline void rcqd_fma(cqdfloat *ret, cqdfloat *a, cqdfloat *b, cqdfloat *c)
{
    cqdfloat tmp;

	rcqd_mul(&tmp, a, b);
	rcqd_add(ret, &tmp, c);

	return;
}

#ifdef __cplusplus
} //extern "C" {
#endif // __cplusplus

#endif //ifndef __BNC_RCDD_H_
