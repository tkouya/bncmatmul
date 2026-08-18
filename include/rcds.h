/********************************************************************************/
/* rcds.h: Reverse definition for float-float and quadruple-float            */
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
#ifndef __BNC_RCDS_H_
#define __BNC_RCDS_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h> // float complex

// Common defs
#include "bnc_common.h"

#include "rds.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// cdsfloat, ctsfloat, cqsfloat
#ifndef CDSFLOAT
#define CDSFLOAT
typedef struct { float val_re[DSSIZE]; float val_im[DSSIZE]; } cdsfloat; // 53 * 2 = 106
typedef struct { float val_re[TSSIZE]; float val_im[TSSIZE]; } ctsfloat; // 53 * 3 = 159
typedef struct { float val_re[QSSIZE]; float val_im[QSSIZE]; } cqsfloat; // 53 * 4 = 212

// 2026-02-06(Fri) T.Kouya
static inline float *rcds_realref(cdsfloat *val) { return val->val_re; }
static inline float *rcds_imagref(cdsfloat *val) { return val->val_im; }
static inline float *rcts_realref(ctsfloat *val) { return val->val_re; }
static inline float *rcts_imagref(ctsfloat *val) { return val->val_im; }
static inline float *rcqs_realref(cqsfloat *val) { return val->val_re; }
static inline float *rcqs_imagref(cqsfloat *val) { return val->val_im; }
#endif // CDSFLOAT

//-------------------------------------
// DD complex arithmetic
//-------------------------------------
// ret := 0
static inline void rcds_set0(cdsfloat *ret)
{
    ret->val_re[0] = 0.0; ret->val_re[1] = 0.0;
    ret->val_im[0] = 0.0; ret->val_im[1] = 0.0;
}
static inline void set0_cds(cdsfloat *ret)
{
    ret->val_re[0] = 0.0; ret->val_re[1] = 0.0;
    ret->val_im[0] = 0.0; ret->val_im[1] = 0.0;
}

// ret := a
static inline void rcds_set(cdsfloat *ret, cdsfloat *a)
{
    rds_set(ret->val_re, a->val_re);
    rds_set(ret->val_im, a->val_im);
}

// ret := a
static inline void rcds_set_ds(cdsfloat *ret, float a[DSSIZE])
{
    rds_set(ret->val_re, a);
    rds_set0(ret->val_im);
}
static inline void rcds_set_dsfloat(cdsfloat *ret, dsfloat *a)
{
    rds_set(ret->val_re, a->val);
    rds_set0(ret->val_im);
}

// ret := a
static inline void rcds_set_ds_ds(cdsfloat *ret, float a_real[DSSIZE], float a_imag[DSSIZE])
{
    rds_set(ret->val_re, a_real);
    rds_set(ret->val_im, a_imag);
}

// ret := a
static inline void rcds_set_d(cdsfloat *ret, float a)
{
    ret->val_re[0] = a;
    ret->val_re[1] = 0.0;
    rds_set0(ret->val_im);
}

// 2024-11-23(Sat) T.Kouya
// 2026-02-06(Fri) T.Kouya
#define rcds_set_si(ret, val) rcds_set_d((ret), (float)(val))
#define rcds_set_ui(ret, val) rcds_set_d((ret), (float)(val))
#define rcds_set_f(ret, val) rcds_set_d((ret), (float)(val))

// ret := a
static inline void rcds_set_cd(cdsfloat *ret, float _Complex a)
{
    ret->val_re[0] = __real__ a; //creal(a);
    ret->val_re[1] = 0.0;
    ret->val_im[0] = __imag__ a; //cimag(a);
    ret->val_im[1] = 0.0;
}

// ret := a_real + a_imag * I
static inline void rcds_set_d_d(cdsfloat *ret, float a_real, float a_imag)
{
    ret->val_re[0] = a_real;
    ret->val_re[1] = 0.0;
    ret->val_im[0] = a_imag;
    ret->val_im[1] = 0.0;
}
#define rcds_set_si_si(ret, a_real, a_imag) rcds_set_d_d((ret), (float)(a_real), (float)(a_imag))
#define rcds_set_ui_ui(ret, a_real, a_imag) rcds_set_d_d((ret), (float)(a_real), (float)(a_imag))
#define rcds_set_f_f(ret, val_re, val_im) rcds_set_d_d((ret), (float)(val_re), (float)(val_im))

// ret := conj(a)
static inline void rcds_conj(cdsfloat *ret, cdsfloat *a)
{
    rds_set(ret->val_re, a->val_re);
    rds_neg(ret->val_im, a->val_im);
}

// ret := -a
static inline void rcds_neg(cdsfloat *ret, cdsfloat *a)
{
    rds_neg(ret->val_re, a->val_re);
    rds_neg(ret->val_im, a->val_im);
}

// ret := a + b
static inline void rcds_add(cdsfloat *ret, cdsfloat *a, cdsfloat *b)
{
    // ret := a + b
    rds_add(ret->val_re, a->val_re, b->val_re);
    rds_add(ret->val_im, a->val_im, b->val_im);
}

// ret := a + (float)b
static inline void rcds_add_ds(cdsfloat *ret, cdsfloat *a, float b[DSSIZE])
{
    // ret := a + b
    rds_add(ret->val_re, a->val_re, b); //b->val_re);
    rds_set(ret->val_im, a->val_im); // , b->val_im);
}

// ret := a - b
static inline void rcds_sub(cdsfloat *ret, cdsfloat *a, cdsfloat *b)
{
    // ret := a - b
    rds_sub(ret->val_re, a->val_re, b->val_re);
    rds_sub(ret->val_im, a->val_im, b->val_im);
}

// ret := a - (float)b
static inline void rcds_sub_ds(cdsfloat *ret, cdsfloat *a, float b[DSSIZE])
{
    // ret := a - b
    rds_sub(ret->val_re, a->val_re, b);
    rds_set(ret->val_im, a->val_im); // , b->val_im);
}

// ret := a * b : 4M
static inline void rcds_mul_4m(cdsfloat *ret, cdsfloat *a, cdsfloat *b)
{
    dsfloat t1, t2, t3, t4;

    // ret := a * b
    //rds_mul(ret->val_re, a->val_re, b->val_re);
    rds_mul(t1.val, a->val_re, b->val_re);
    rds_mul(t2.val, a->val_im, b->val_im);
    rds_mul(t3.val, a->val_re, b->val_im);
    rds_mul(t4.val, a->val_im, b->val_re);

    rds_sub(ret->val_re, t1.val, t2.val);
    rds_add(ret->val_im, t3.val, t4.val);
}

// ret := a * b : 3M
static inline void rcds_mul_3m(cdsfloat *ret, cdsfloat *a, cdsfloat *b)
{
    dsfloat t1, t2, tmp, a_re_im, b_re_im;

    // ret := a * b

    // t1      := re_a * re_b
    // t2      := im_a * im_b
    // -> ret_re  := t1 - t2 
    rds_mul(t1.val, a->val_re, b->val_re);
    rds_mul(t2.val, a->val_im, b->val_im);
    rds_sub(ret->val_re, t1.val, t2.val);
    /*
    printf("t1->"); rds_out_str_base(stdout, 10, 33, t1.val); printf("\n");
    printf("t2->"); rds_out_str_base(stdout, 10, 33, t2.val); printf("\n");
    printf("ret_re->"); rds_out_str_base(stdout, 10, 33, ret->val_re); printf("\n"); 
    */

    // a_re_im := re_a + im_a
    // b_re_im := re_b + im_b
    // -> ret_im  := a_re_im * b_re_im - t1 - t2
    rds_add(a_re_im.val, a->val_re, a->val_im);
    rds_add(b_re_im.val, b->val_re, b->val_im);
    rds_mul(tmp.val, a_re_im.val, b_re_im.val);
    //printf("tmp->"); rds_out_str_base(stdout, 10, 33, tmp.val); printf("\n");
    rds_sub(tmp.val, tmp.val, t1.val);
    //printf("tmp-t1->"); rds_out_str_base(stdout, 10, 33, tmp.val); printf("\n");
    rds_sub(ret->val_im, tmp.val, t2.val);
    //printf("tmp-t2->"); rds_out_str_base(stdout, 10, 33, ret->val_im); printf("\n");

    //printf("rcds_mul_3m_ret->"); rcds_out_str(ret); printf("\n");
}

// ret := |a|^2
static inline void rcds_nrm2(dsfloat *ret, cdsfloat *a)
{
    dsfloat tmp_re2, tmp_im2;

    rds_mul(tmp_re2.val, a->val_re, a->val_re);
    rds_mul(tmp_im2.val, a->val_im, a->val_im);
    rds_add(ret->val, tmp_re2.val, tmp_im2.val);
}
static inline void rcds_nrm2_ds(float ret[DSSIZE], cdsfloat *a)
{
    float tmp_re2[DSSIZE], tmp_im2[DSSIZE];

    rds_mul(tmp_re2, a->val_re, a->val_re);
    rds_mul(tmp_im2, a->val_im, a->val_im);
    rds_add(ret, tmp_re2, tmp_im2);
}


// ret := |a|
static inline void rcds_abs(dsfloat *ret, cdsfloat *a)
{
    dsfloat tmp_re2, tmp_im2;

    rds_mul(tmp_re2.val, a->val_re, a->val_re);
    rds_mul(tmp_im2.val, a->val_im, a->val_im);
    rds_add(ret->val, tmp_re2.val, tmp_im2.val);
    rds_sqrt(ret->val, ret->val);
}
static inline void rcds_abs_ds(float ret[DSSIZE], cdsfloat *a)
{
    float tmp_re2[DSSIZE], tmp_im2[DSSIZE];

    rds_mul(tmp_re2, a->val_re, a->val_re);
    rds_mul(tmp_im2, a->val_im, a->val_im);
    rds_add(ret, tmp_re2, tmp_im2);
    rds_sqrt(ret, ret);
}

// ret := 1 / a
static inline void rcds_inv_old(cdsfloat *ret, cdsfloat *a)
{
    dsfloat a_nrm2, tmp;
    cdsfloat in_ret;

    //rcds_conj(ret, a);
    rds_set(in_ret.val_re, a->val_re);
    rds_neg(in_ret.val_im, a->val_im);
    //rcds_nrm2(&a_nrm2, a);
    rds_mul(   tmp.val, a->val_re, a->val_re);
    rds_mul(a_nrm2.val, a->val_im, a->val_im);
    rds_add(a_nrm2.val, a_nrm2.val, tmp.val);

    rds_div(ret->val_re, in_ret.val_re, a_nrm2.val);
    rds_div(ret->val_im, in_ret.val_im, a_nrm2.val);
}
static inline void rcds_inv(cdsfloat *ret, cdsfloat *a)
{
    float a_nrm2[DSSIZE];
    float tmp[DSSIZE];
    float neg_im[DSSIZE];
    cdsfloat in_a;
    
    // エイリアシング対策
    rcds_set(&in_a, a);
    
    // |a|² = a_re² + a_im²
    rds_mul(tmp, in_a.val_re, in_a.val_re);
    rds_mul(a_nrm2, in_a.val_im, in_a.val_im);
    rds_add(a_nrm2, a_nrm2, tmp);
    
    // 実部 = a_re / |a|²
    rds_div(ret->val_re, in_a.val_re, a_nrm2);
    
    // 虚部 = -a_im / |a|²（明示的に符号反転）
    rds_neg(neg_im, in_a.val_im);
    rds_div(ret->val_im, neg_im, a_nrm2);
}
// ret := a / b : 3M
static inline void rcds_div_3m(cdsfloat *ret, cdsfloat *a, cdsfloat *b)
{
    cdsfloat inv_b, in_a;
    dsfloat t1, t2, tmp, a_re_im, b_re_im, b_nrm2;

    rcds_inv(&inv_b, b);
    rcds_set(&in_a, a);
    //printf("inv_b-> "); rcds_out_str(&inv_b); printf("\n");
    rcds_mul_3m(ret, &in_a, &inv_b);
    //printf("ret-> "); rcds_out_str(ret); printf("\n");
}

// ret := a / b : 4M
static inline void rcds_div_4m(cdsfloat *ret, cdsfloat *a, cdsfloat *b)
{
    cdsfloat in_a, inv_b;

    rcds_set(&in_a, a);
    //printf("in_a = %25.17e + i* %25.17e\n", in_a.val_re[0], in_a.val_im[0]);
    rcds_inv(&inv_b, b);
    //printf("inv_b = %25.17e + i* %25.17e\n", inv_b.val_re[0], inv_b.val_im[0]);
    rcds_mul_4m(ret, &in_a, &inv_b); // fix! 2026-02-08(SUN) T.Kouya
    //printf("in_a *inv_b = %25.17e + i* %25.17e\n", ret->val_re[0], ret->val_im[0]);
#if 0
    float b_nrm2[DSSIZE];
    float tmp[DSSIZE];
    float neg_im[DSSIZE];
    cdsfloat in_a, in_b, inv_b;
    
    // エイリアシング対策
    //rcds_set(&in_a, a);
    rds_set(in_b.val_re, b->val_re);
    rds_set(in_b.val_im, b->val_im);

    // |a|² = a_re² + a_im²
    rds_mul(tmp, in_b.val_re, in_b.val_re);
    rds_mul(b_nrm2, in_b.val_im, in_b.val_im);
    rds_add(b_nrm2, b_nrm2, tmp);
    
    // 実部 = a_re / |a|²
    rds_div(inv_b.val_re, in_b.val_re, b_nrm2);
    
    // 虚部 = -a_im / |a|²（明示的に符号反転）
    rds_neg(neg_im, in_b.val_im);
    rds_div(inv_b.val_im, neg_im, b_nrm2);

    //rcds_set(&in_a, a);
    rds_set(in_a.val_re, a->val_re);
    rds_set(in_a.val_im, a->val_im); 
    
    //rcds_mul_4m(ret, &in_a, &inv_b); // fix! 2026-02-08(SUN) T.Kouya

    // ret := in_a * inv_b

    rds_mul(ret->val_re, in_a.val_re, inv_b.val_re);
    rds_mul(tmp, in_a.val_im, inv_b.val_im);
    rds_sub(ret->val_re, ret->val_re, tmp);

    rds_mul(ret->val_im, in_a.val_re, inv_b.val_im);
    rds_mul(tmp, in_a.val_im, inv_b.val_re);
    rds_add(ret->val_im, ret->val_im, tmp);
#endif // 0
}

// ret := a * (real)b = real(a) * b + imag(a) * b
static inline void rcds_mul_ds(cdsfloat *ret, cdsfloat *a, float b[DSSIZE])
{
    rds_mul(ret->val_re, a->val_re, b);
    rds_mul(ret->val_im, a->val_im, b);
}

// ret := a * (real)b = real(a) * b + imag(a) * b
static inline void rcds_mul_d(cdsfloat *ret, cdsfloat *a, float b)
{
    rds_mul_f(ret->val_re, a->val_re, b);
    rds_mul_f(ret->val_im, a->val_im, b);
}
#define rcds_mul_ui(ret, a, b) rcds_mul_d((ret), (a), (float)(b))
#define rcds_mul_si(ret, a, b) rcds_mul_d((ret), (a), (float)(b))

// ret := a * (real)b = real(a) / b + imag(a) / b
static inline void rcds_div_ds(cdsfloat *ret, cdsfloat *a, float b[DSSIZE])
{
    rds_div(ret->val_re, a->val_re, b);
    rds_div(ret->val_im, a->val_im, b);
}

// ret := a * (real)b = real(a) / b + imag(a) / b
static inline void rcds_div_d(cdsfloat *ret, cdsfloat *a, float b)
{
    rds_div_f(ret->val_re, a->val_re, b);
    rds_div_f(ret->val_im, a->val_im, b);
}
#define rcds_div_ui(ret, a, b) rcds_div_d((ret), (a), (float)(b))
#define rcds_div_si(ret, a, b) rcds_div_d((ret), (a), (float)(b))

// 2024-02-18(SUN) Tomonori Kouya
// Default: 4M method
#ifdef USE_RCDS_MUL_4M // 4M
    #define rcds_mul rcds_mul_4m
    #define rcds_div rcds_div_4m
#elif defined(USE_RCDS_MUL_3M) // 3M
    #define rcds_mul rcds_mul_3m
    #define rcds_div rcds_div_3m
#else // USE_RCDS_MUL_4M
    #define rcds_mul rcds_mul_4m
    #define rcds_div rcds_div_4m
    //#define rcds_mul rcds_mul_3m
    //#define rcds_div rcds_div_3m
#endif // USE_RCDS_MUL_4M

// ret := a * b : 4M
static inline void rcds_mul_cd(cdsfloat *ret, cdsfloat *a, float _Complex b)
{
    dsfloat t1, t2, t3, t4;

    // ret := a * b
    //rds_mul(ret->val_re, a->val_re, b->val_re);
    rds_mul_f(t1.val, a->val_re, __real__ b); //creal(b));
    rds_mul_f(t2.val, a->val_im, __imag__ b); //cimag(b));
    rds_mul_f(t3.val, a->val_re, __imag__ b); //cimag(b));
    rds_mul_f(t4.val, a->val_im, __real__ b); //creal(b));

    rds_sub(ret->val_re, t1.val, t2.val);
    rds_add(ret->val_im, t3.val, t4.val);
}

// ret := a / b : 4M
static inline void rcds_div_cd(cdsfloat *ret, cdsfloat *a, float _Complex b)
{
    dsfloat t1, t2, t3, t4, abs_b2;

    // ret := a * conj(b) / |b|^2
    //rds_mul(ret->val_re, a->val_re, b->val_re);
    rds_mul_f(t1.val, a->val_re, __real__ b); // creal(b));
    rds_mul_f(t2.val, a->val_im, __imag__ b); // cimag(b));
    rds_mul_f(t3.val, a->val_re, __imag__ b); // cimag(b));
    rds_mul_f(t4.val, a->val_im, __real__ b); // creal(b));

    // (a_re * b_re + a_im * b_im) + (a_im * b_re - a_re * b_im) * I
    rds_add(ret->val_re, t1.val, t2.val);
    rds_sub(ret->val_im, t4.val, t3.val);

    rds_set_d(abs_b2.val, __real__ b);
    rds_mul_f(abs_b2.val, abs_b2.val, __real__ b);
    rds_set_d(t1.val, __imag__ b);
    rds_mul_f(t1.val, t1.val, __imag__ b);
    rds_add(abs_b2.val, abs_b2.val, t1.val);

    rds_div(ret->val_re, ret->val_re, abs_b2.val);
    rds_div(ret->val_im, ret->val_im, abs_b2.val);
}

// print cdsfloat
static inline void rcds_out_str(cdsfloat *ret)
{
    rds_out_str_base(stdout, 10, 33, ret->val_re);
    //rds_out_str_base(stdout, 10, 32, ret->val_re);
    printf(" + ");
    rds_out_str_base(stdout, 10, 33, ret->val_im);
    //rds_out_str_base(stdout, 10, 32, ret->val_im);
    printf(" * I");
}

// 2024-12-03 (Tue) T.Kouya
// |a| >  |b| -> +1
// |a| == |b| ->  0
// |a| <  |b| -> -1
static inline int rcds_cmp_abs(cdsfloat *a, cdsfloat *b)
{
    float abs_a[DSSIZE], abs_b[DSSIZE];

    rcds_abs_ds(abs_a, a);
    rcds_abs_ds(abs_b, b);

    return rds_cmp(abs_a, abs_b);
}
static inline int rcds_cmp_abs_ds(cdsfloat *a, float b[DSSIZE])
{
    float abs_a[DSSIZE], abs_b[DSSIZE];

    rcds_abs_ds(abs_a, a);
    rds_abs(abs_b, b);

    return rds_cmp(abs_a, abs_b);
}
static inline int rcds_cmp_abs_d(cdsfloat *a, float b)
{
    float abs_a[DSSIZE], abs_b;

    rcds_abs_ds(abs_a, a);
    abs_b = fabs(b);

    return rds_cmp_f(abs_a, abs_b);
}
#define rcds_cmp_abs_ui(a, b) rcds_cmp_abs_d((a), ((float)(b)))

// CDD fma
// ret = a * b + c
static inline void rcds_fma(cdsfloat *ret, cdsfloat *a, cdsfloat *b, cdsfloat *c)
{
    cdsfloat tmp;

	rcds_mul(&tmp, a, b);
	rcds_add(ret, &tmp, c);

	return;
}


//-------------------------------------
// TD complex arithmetic
//-------------------------------------
// ret := 0
static inline void rcts_set0(ctsfloat *ret)
{
    ret->val_re[0] = 0.0;  ret->val_re[1] = 0.0; ret->val_re[2] = 0.0;
    ret->val_im[0] = 0.0;  ret->val_im[1] = 0.0; ret->val_im[2] = 0.0;
}
static inline void set0_cts(ctsfloat *ret)
{
    ret->val_re[0] = 0.0;  ret->val_re[1] = 0.0; ret->val_re[2] = 0.0;
    ret->val_im[0] = 0.0;  ret->val_im[1] = 0.0; ret->val_im[2] = 0.0;
}


// ret := a
static inline void rcts_set(ctsfloat *ret, ctsfloat *a)
{
    rts_set(ret->val_re, a->val_re);
    rts_set(ret->val_im, a->val_im);
}

// ret := a
static inline void rcts_set_tsfloat(ctsfloat *ret, tsfloat *a)
{
    rts_set(ret->val_re, a->val);
    rts_set0(ret->val_im);
}
static inline void rcts_set_ts(ctsfloat *ret, float a[TSSIZE])
{
    rts_set(ret->val_re, a);
    rts_set0(ret->val_im);
}

// ret := a
static inline void rcts_set_ts_ts(ctsfloat *ret, float a_real[TSSIZE], float a_imag[TSSIZE])
{
    rts_set(ret->val_re, a_real);
    rts_set(ret->val_im, a_imag);
}

// ret := a
static inline void rcts_set_dsfloat(ctsfloat *ret, dsfloat *a)
{
    ret->val_re[0] = a->val[0];
    ret->val_re[1] = a->val[1];
    ret->val_re[2] = 0.0;
    rts_set0(ret->val_im);
}
static inline void rcts_set_ds(ctsfloat *ret, float a[DSSIZE])
{
    ret->val_re[0] = a[0];
    ret->val_re[1] = a[1];
    ret->val_re[2] = 0.0;
    rts_set0(ret->val_im);
}

// ret := a
static inline void rcts_set_d(ctsfloat *ret, float a)
{
    ret->val_re[0] = a;
    ret->val_re[1] = 0.0;
    ret->val_re[2] = 0.0;
    rts_set0(ret->val_im);
}

// 2024-11-23(Sat) T.Kouya
#define rcts_set_ui(ret, val) rcts_set_d((ret), (float)(val))
#define rcts_set_si(ret, val) rcts_set_d((ret), (float)(val))
#define rcts_set_f(ret, val) rcts_set_d((ret), (float)(val))

// ret := a
static inline void rcts_set_cd(ctsfloat *ret, float _Complex a)
{
    ret->val_re[0] = __real__ a; //creal(a);
    ret->val_re[1] = 0.0;
    ret->val_re[2] = 0.0;
    ret->val_im[0] = __imag__ a; //cimag(a);
    ret->val_im[1] = 0.0;
    ret->val_im[2] = 0.0;
}

// ret := a_real + a_imag * I
static inline void rcts_set_d_d(ctsfloat *ret, float a_real, float a_imag)
{
    ret->val_re[0] = a_real;
    ret->val_re[1] = 0.0;
    ret->val_re[2] = 0.0;
    ret->val_im[0] = a_imag;
    ret->val_im[1] = 0.0;
    ret->val_im[2] = 0.0;
}
#define rcts_set_si_si(ret, a_real, a_imag) rcts_set_d_d((ret), (float)(a_real), (float)(a_imag))
#define rcts_set_ui_ui(ret, a_real, a_imag) rcts_set_d_d((ret), (float)(a_real), (float)(a_imag))
#define rcts_set_f_f(ret, a_real, a_imag) rcts_set_d_d((ret), (float)(a_real), (float)(a_imag))

// ret := a
static inline void rcts_set_cdsfloat(ctsfloat *ret, cdsfloat *a)
{
    ret->val_re[0] = a->val_re[0]; //creal(a);
    ret->val_re[1] = a->val_re[1];
    ret->val_re[2] = 0.0;
    ret->val_im[0] = a->val_im[0]; //cimag(a);
    ret->val_im[1] = a->val_im[1];
    ret->val_im[2] = 0.0;
}

// ret := conj(a)
static inline void rcts_conj(ctsfloat *ret, ctsfloat *a)
{
    rts_set(ret->val_re, a->val_re);
    rts_neg(ret->val_im, a->val_im);
}

// ret := -a
static inline void rcts_neg(ctsfloat *ret, ctsfloat *a)
{
    rts_neg(ret->val_re, a->val_re);
    rts_neg(ret->val_im, a->val_im);
}

// ret := a + b
static inline void rcts_add(ctsfloat *ret, ctsfloat *a, ctsfloat *b)
{
    // ret := a + b
    rts_add(ret->val_re, a->val_re, b->val_re);
    rts_add(ret->val_im, a->val_im, b->val_im);
}

// ret := a + (TD)b
static inline void rcts_add_ts(ctsfloat *ret, ctsfloat *a, float b[TSSIZE])
{
    // ret := a + b
    rts_add(ret->val_re, a->val_re, b); //b->val_re);
    rds_set(ret->val_im, a->val_im);
}

// ret := a - b
static inline void rcts_sub(ctsfloat *ret, ctsfloat *a, ctsfloat *b)
{
    // ret := a - b
    rts_sub(ret->val_re, a->val_re, b->val_re);
    rts_sub(ret->val_im, a->val_im, b->val_im);
}

// ret := a - (TD)b
static inline void rcts_sub_ts(ctsfloat *ret, ctsfloat *a, float b[TSSIZE])
{
    // ret := a - b
    rts_sub(ret->val_re, a->val_re, b); //b->val_re);
    rds_set(ret->val_im, a->val_im);
}

// ret := a * b : 4M
static inline void rcts_mul_4m(ctsfloat *ret, ctsfloat *a, ctsfloat *b)
{
    tsfloat t1, t2, t3, t4;

    // ret := a * b
    rts_mul(t1.val, a->val_re, b->val_re);
    rts_mul(t2.val, a->val_im, b->val_im);
    rts_mul(t3.val, a->val_re, b->val_im);
    rts_mul(t4.val, a->val_im, b->val_re);
    rts_sub(ret->val_re, t1.val, t2.val);
    rts_add(ret->val_im, t3.val, t4.val);
}

// ret := a * b : 3M
static inline void rcts_mul_3m(ctsfloat *ret, ctsfloat *a, ctsfloat *b)
{
    tsfloat t1, t2, a_re_im, b_re_im;

    // ret := a * b
    rts_mul(t1.val, a->val_re, b->val_re);
    rts_mul(t2.val, a->val_im, b->val_im);
    rts_sub(ret->val_re, t1.val, t2.val);

    rts_add(a_re_im.val, a->val_re, a->val_im);
    rts_add(b_re_im.val, b->val_re, b->val_im);
    rts_mul(ret->val_im, a_re_im.val, b_re_im.val);
    rts_sub(ret->val_im, ret->val_im, t1.val);
    rts_sub(ret->val_im, ret->val_im, t2.val);
}

// ret := |a|^2
static inline void rcts_nrm2(tsfloat *ret, ctsfloat *a)
{
    tsfloat tmp_re2, tmp_im2;

    rts_mul(tmp_re2.val, a->val_re, a->val_re);
    rts_mul(tmp_im2.val, a->val_im, a->val_im);
    rts_add(ret->val, tmp_re2.val, tmp_im2.val);
}
static inline void rcts_nrm2_ts(float ret[TSSIZE], ctsfloat *a)
{
    float tmp_re2[TSSIZE], tmp_im2[TSSIZE];

    rts_mul(tmp_re2, a->val_re, a->val_re);
    rts_mul(tmp_im2, a->val_im, a->val_im);
    rts_add(ret, tmp_re2, tmp_im2);
}

// ret := |a|
static inline void rcts_abs(tsfloat *ret, ctsfloat *a)
{
    tsfloat tmp_re2, tmp_im2;

    rts_mul(tmp_re2.val, a->val_re, a->val_re);
    rts_mul(tmp_im2.val, a->val_im, a->val_im);
    rts_add(ret->val, tmp_re2.val, tmp_im2.val);
    rts_sqrt(ret->val, ret->val);
}
static inline void rcts_abs_ts(float ret[TSSIZE], ctsfloat *a)
{
    float tmp_re2[TSSIZE], tmp_im2[TSSIZE];

    rts_mul(tmp_re2, a->val_re, a->val_re);
    rts_mul(tmp_im2, a->val_im, a->val_im);
    rts_add(ret, tmp_re2, tmp_im2);
    rts_sqrt(ret, ret);
}

// ret := 1 / a
static inline void rcts_inv(ctsfloat *ret, ctsfloat *a)
{
    tsfloat a_nrm2, tmp;
    ctsfloat in_ret;

    //rcts_conj(ret, a);
    rts_set(in_ret.val_re, a->val_re);
    rts_neg(in_ret.val_im, a->val_im);

    //rcts_nrm2(&a_nrm2, a);
    rts_mul(a_nrm2.val, a->val_re, a->val_re);
    rts_mul(tmp.val, a->val_im, a->val_im);
    rts_add(a_nrm2.val, a_nrm2.val, tmp.val);

    rts_div(ret->val_re, in_ret.val_re, a_nrm2.val);
    rts_div(ret->val_im, in_ret.val_im, a_nrm2.val);
}

// ret := a / b : 3M
static inline void rcts_div_3m(ctsfloat *ret, ctsfloat *a, ctsfloat *b)
{
    ctsfloat inv_b;

    rcts_inv(&inv_b, b);
    rcts_mul_3m(ret, a, &inv_b);
}

// ret := a / b : 4M
static inline void rcts_div_4m(ctsfloat *ret, ctsfloat *a, ctsfloat *b)
{
    ctsfloat in_a, inv_b;

    rcts_inv(&inv_b, b);
    rcts_set(&in_a, a);
    rcts_mul_4m(ret, &in_a, &inv_b);
}

// ret := a * (real)b = real(a) * b + imag(a) * b
static inline void rcts_mul_ts(ctsfloat *ret, ctsfloat *a, float b[TSSIZE])
{
    rts_mul(ret->val_re, a->val_re, b);
    rts_mul(ret->val_im, a->val_im, b);
}

// ret := a * (real)b = real(a) * b + imag(a) * b
static inline void rcts_mul_d(ctsfloat *ret, ctsfloat *a, float b)
{
    rts_mul_f(ret->val_re, a->val_re, b);
    rts_mul_f(ret->val_im, a->val_im, b);
}
#define rcts_mul_ui(ret, a, b) rcts_mul_d((ret), (a), (float)(b))
#define rcts_mul_si(ret, a, b) rcts_mul_d((ret), (a), (float)(b))

// ret := a * (real)b = real(a) / b + imag(a) / b
static inline void rcts_div_ts(ctsfloat *ret, ctsfloat *a, float b[TSSIZE])
{
    rts_div(ret->val_re, a->val_re, b);
    rts_div(ret->val_im, a->val_im, b);
}

// ret := a * (real)b = real(a) / b + imag(a) / b
static inline void rcts_div_d(ctsfloat *ret, ctsfloat *a, float b)
{
    rts_div_f(ret->val_re, a->val_re, b);
    rts_div_f(ret->val_im, a->val_im, b);
}
#define rcts_div_ui(ret, a, b) rcts_div_d((ret), (a), (float)(b))
#define rcts_div_si(ret, a, b) rcts_div_d((ret), (a), (float)(b))


// 2024-02-18(SUN) Tomonori Kouya
// Default: 4M method
#ifdef USE_RCTS_MUL_4M // 4M
    #define rcts_mul rcts_mul_4m
    #define rcts_div rcts_div_4m
#elif defined(USE_RCTS_MUL_3M) // 3M
    #define rcts_mul rcts_mul_3m
    #define rcts_div rcts_div_3m
#else // USE_RCTS_MUL_4M
    #define rcts_mul rcts_mul_4m
    #define rcts_div rcts_div_4m
    //#define rcts_mul rcts_mul_3m
    //#define rcts_div rcts_div_3m
#endif // USE_RCDS_MUL_4M

// ret := a * b : 4M
static inline void rcts_mul_cd(ctsfloat *ret, ctsfloat *a, float _Complex b)
{
    tsfloat t1, t2, t3, t4;

    // ret := a * b
    //rds_mul(ret->val_re, a->val_re, b->val_re);
    rts_mul_f(t1.val, a->val_re, __real__ b); //creal(b));
    rts_mul_f(t2.val, a->val_im, __imag__ b); //cimag(b));
    rts_mul_f(t3.val, a->val_re, __imag__ b); //cimag(b));
    rts_mul_f(t4.val, a->val_im, __real__ b); //creal(b));

    rts_sub(ret->val_re, t1.val, t2.val);
    rts_add(ret->val_im, t3.val, t4.val);
}

// ret := a / b : 4M
static inline void rcts_div_cd(ctsfloat *ret, ctsfloat *a, float _Complex b)
{
    tsfloat t1, t2, t3, t4, abs_b2;

    // ret := a * conj(b) / |b|^2
    //rds_mul(ret->val_re, a->val_re, b->val_re);
    rts_mul_f(t1.val, a->val_re, __real__ b); // creal(b));
    rts_mul_f(t2.val, a->val_im, __imag__ b); // cimag(b));
    rts_mul_f(t3.val, a->val_re, __imag__ b); // cimag(b));
    rts_mul_f(t4.val, a->val_im, __real__ b); // creal(b));

    rts_add(ret->val_re, t1.val, t2.val);
    rts_sub(ret->val_im, t4.val, t3.val);

    rts_set_d(abs_b2.val, __real__ b);
    rts_mul_f(abs_b2.val, abs_b2.val, __real__ b);
    rts_set_d(t1.val, __imag__ b);
    rts_mul_f(t1.val, t1.val, __imag__ b);
    rts_add(abs_b2.val, abs_b2.val, t1.val);

    rts_div(ret->val_re, ret->val_re, abs_b2.val);
    rts_div(ret->val_im, ret->val_im, abs_b2.val);
}

// print ctsfloat
static inline void rcts_out_str(ctsfloat *ret)
{
    //rts_out_str(ret->val_re);
    rts_out_str_base(stdout, 10, 48, ret->val_re);
    printf(" + ");
    //rts_out_str(ret->val_im);
    rts_out_str_base(stdout, 10, 48, ret->val_im);
    printf(" * I");
}

// 2024-12-03 (Tue) T.Kouya
// |a| >  |b| -> +1
// |a| == |b| ->  0
// |a| <  |b| -> -1
static inline int rcts_cmp_abs(ctsfloat *a, ctsfloat *b)
{
    float abs_a[TSSIZE], abs_b[TSSIZE];

    rcts_abs_ts(abs_a, a);
    rcts_abs_ts(abs_b, b);

    return rts_cmp(abs_a, abs_b);
}
static inline int rcts_cmp_abs_ts(ctsfloat *a, float b[TSSIZE])
{
    float abs_a[TSSIZE], abs_b[TSSIZE];

    rcts_abs_ts(abs_a, a);
    rts_abs(abs_b, b);

    return rts_cmp(abs_a, abs_b);
}
static inline int rcts_cmp_abs_d(ctsfloat *a, float b)
{
    float abs_a[TSSIZE], abs_b;

    rcts_abs_ts(abs_a, a);
    abs_b = fabs(b);

    return rts_cmp_f(abs_a, abs_b);
}
#define rcts_cmp_abs_ui(a, b) rcts_cmp_abs_d((a), ((float)(b)))

// CTD fma
// ret = a * b + c
static inline void rcts_fma(ctsfloat *ret, ctsfloat *a, ctsfloat *b, ctsfloat *c)
{
    ctsfloat tmp;

	rcts_mul(&tmp, a, b);
	rcts_add(ret, &tmp, c);

	return;
}

//-------------------------------------
// QD complex arithmetic
//-------------------------------------
// ret := 0
static inline void rcqs_set0(cqsfloat *ret)
{
    ret->val_re[0] = 0.0;  ret->val_re[1] = 0.0; ret->val_re[2] = 0.0; ret->val_re[3] = 0.0;
    ret->val_im[0] = 0.0;  ret->val_im[1] = 0.0; ret->val_im[2] = 0.0; ret->val_im[3] = 0.0;
}
static inline void set0_cqs(cqsfloat *ret)
{
    ret->val_re[0] = 0.0;  ret->val_re[1] = 0.0; ret->val_re[2] = 0.0; ret->val_re[3] = 0.0;
    ret->val_im[0] = 0.0;  ret->val_im[1] = 0.0; ret->val_im[2] = 0.0; ret->val_im[3] = 0.0;
}

// ret := a
static inline void rcqs_set(cqsfloat *ret, cqsfloat *a)
{
    rqs_set(ret->val_re, a->val_re);
    rqs_set(ret->val_im, a->val_im);
}

// ret := a
static inline void rcqs_set_qsfloat(cqsfloat *ret, qsfloat *a)
{
    rqs_set(ret->val_re, a->val);
    rqs_set0(ret->val_im);
}
static inline void rcqs_set_qs(cqsfloat *ret, float a[QSSIZE])
{
    rqs_set(ret->val_re, a);
    rqs_set0(ret->val_im);
}

// ret := a
static inline void rcqs_set_qs_qs(cqsfloat *ret, float a_real[QSSIZE], float a_imag[QSSIZE])
{
    rqs_set(ret->val_re, a_real);
    rqs_set(ret->val_im, a_imag);
}

// ret := a
static inline void rcqs_set_dsfloat(cqsfloat *ret, dsfloat *a)
{
    ret->val_re[0] = a->val[0];
    ret->val_re[1] = a->val[1];
    ret->val_re[2] = 0.0;
    ret->val_re[3] = 0.0;
    rqs_set0(ret->val_im);
}

// ret := a
static inline void rcqs_set_tsfloat(cqsfloat *ret, tsfloat *a)
{
    ret->val_re[0] = a->val[0];
    ret->val_re[1] = a->val[1];
    ret->val_re[2] = a->val[2];
    ret->val_re[3] = 0.0;
    rqs_set0(ret->val_im);
}

// ret := a
static inline void rcqs_set_d(cqsfloat *ret, float a)
{
    ret->val_re[0] = a;
    ret->val_re[1] = 0.0;
    ret->val_re[2] = 0.0;
    ret->val_re[3] = 0.0;
    rqs_set0(ret->val_im);
}

// 2024-11-23(Sat) T.Kouya
#define rcqs_set_ui(ret, val) rcqs_set_d((ret), (float)(val))
#define rcqs_set_si(ret, val) rcqs_set_d((ret), (float)(val))
#define rcqs_set_f (ret, val) rcqs_set_d((ret), (float)(val))

// ret := a
static inline void rcqs_set_cd(cqsfloat *ret, float _Complex a)
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
static inline void rcqs_set_d_d(cqsfloat *ret, float a_real, float a_imag)
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
#define rcqs_set_si_si(ret, a_real, a_imag) rcqs_set_d_d((ret), (float)(a_real), (float)(a_imag))
#define rcqs_set_ui_ui(ret, a_real, a_imag) rcqs_set_d_d((ret), (float)(a_real), (float)(a_imag))
#define rcqs_set_f_f  (ret, a_real, a_imag) rcqs_set_d_d((ret), (float)(a_real), (float)(a_imag))

// ret := a
static inline void rcqs_set_cdd(cqsfloat *ret, cdsfloat *a)
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
static inline void rcqs_set_ctd(cqsfloat *ret, ctsfloat *a)
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
static inline void rcqs_conj(cqsfloat *ret, cqsfloat *a)
{
    rqs_set(ret->val_re, a->val_re);
    rqs_neg(ret->val_im, a->val_im);
}

// ret := -a
static inline void rcqs_neg(cqsfloat *ret, cqsfloat *a)
{
    rqs_neg(ret->val_re, a->val_re);
    rqs_neg(ret->val_im, a->val_im);
}

// ret := a + b
static inline void rcqs_add(cqsfloat *ret, cqsfloat *a, cqsfloat *b)
{
    // ret := a + b
    rqs_add(ret->val_re, a->val_re, b->val_re);
    rqs_add(ret->val_im, a->val_im, b->val_im);
}

// ret := a + (QD)b
static inline void rcqs_add_qs(cqsfloat *ret, cqsfloat *a, float b[QSSIZE])
{
    // ret := a + b
    rqs_add(ret->val_re, a->val_re, b); // ->val_re);
    rqs_set(ret->val_im, a->val_im); // , b->val_im);
}

// ret := a - b
static inline void rcqs_sub(cqsfloat *ret, cqsfloat *a, cqsfloat *b)
{
    // ret := a - b
    rqs_sub(ret->val_re, a->val_re, b->val_re);
    rqs_sub(ret->val_im, a->val_im, b->val_im);
}

// ret := a - (QD)b
static inline void rcqs_sub_qs(cqsfloat *ret, cqsfloat *a, float b[QSSIZE])
{
    // ret := a - b
    rqs_sub(ret->val_re, a->val_re, b); // ->val_re);
    rqs_set(ret->val_im, a->val_im); // , b->val_im);
}

// ret := a * b : 4M
static inline void rcqs_mul_4m(cqsfloat *ret, cqsfloat *a, cqsfloat *b)
{
    qsfloat t1, t2, t3, t4;

    // ret := a * b
    rqs_mul(t1.val, a->val_re, b->val_re);
    rqs_mul(t2.val, a->val_im, b->val_im);
    rqs_mul(t3.val, a->val_re, b->val_im);
    rqs_mul(t4.val, a->val_im, b->val_re);
    rqs_sub(ret->val_re, t1.val, t2.val);
    rqs_add(ret->val_im, t3.val, t4.val);
}

// ret := a * b : 3M
static inline void rcqs_mul_3m(cqsfloat *ret, cqsfloat *a, cqsfloat *b)
{
    qsfloat t1, t2, a_re_im, b_re_im;

    // ret := a * b
    // t1      := re_a * re_b
    // t2      := im_a * im_b
    // -> ret_re  := t1 - t2 
    // a_re_im := re_a + im_a
    // b_re_im := re_b + im_b
    // -> ret_im  := a_re_im * b_re_im - t1 - t2
    rqs_mul(t1.val, a->val_re, b->val_re);
    rqs_mul(t2.val, a->val_im, b->val_im);
    rqs_sub(ret->val_re, t1.val, t2.val);

    rqs_add(a_re_im.val, a->val_re, a->val_im);
    rqs_add(b_re_im.val, b->val_re, b->val_im);
    rqs_mul(ret->val_im, a_re_im.val, b_re_im.val);
    rqs_sub(ret->val_im, ret->val_im, t1.val);
    rqs_sub(ret->val_im, ret->val_im, t2.val);
}

// ret := |a|^2
static inline void rcqs_nrm2(qsfloat *ret, cqsfloat *a)
{
    qsfloat tmp_re2, tmp_im2;

    rqs_mul(tmp_re2.val, a->val_re, a->val_re);
    rqs_mul(tmp_im2.val, a->val_im, a->val_im);
    rqs_add(ret->val, tmp_re2.val, tmp_im2.val);
}
static inline void rcqs_nrm2_qs(float ret[QSSIZE], cqsfloat *a)
{
    float tmp_re2[QSSIZE], tmp_im2[QSSIZE];

    rqs_mul(tmp_re2, a->val_re, a->val_re);
    rqs_mul(tmp_im2, a->val_im, a->val_im);
    rqs_add(ret, tmp_re2, tmp_im2);
}

// ret := |a|
static inline void rcqs_abs(qsfloat *ret, cqsfloat *a)
{
    qsfloat tmp_re2, tmp_im2;

    rqs_mul(tmp_re2.val, a->val_re, a->val_re);
    rqs_mul(tmp_im2.val, a->val_im, a->val_im);
    rqs_add(ret->val, tmp_re2.val, tmp_im2.val);
#ifdef USE_GMP
#else // USE_GMP
    rqs_sqrt(ret->val, ret->val);
#endif // USE_GMP
}
static inline void rcqs_abs_qs(float ret[QSSIZE], cqsfloat *a)
{
    float tmp_re2[QSSIZE], tmp_im2[QSSIZE];

    rqs_mul(tmp_re2, a->val_re, a->val_re);
    rqs_mul(tmp_im2, a->val_im, a->val_im);
    rqs_add(ret, tmp_re2, tmp_im2);
    rqs_sqrt(ret, ret);
}

// ret := 1 / a
static inline void rcqs_inv(cqsfloat *ret, cqsfloat *a)
{
    qsfloat a_nrm2, tmp;
    cqsfloat in_a;

    rcqs_set(&in_a, a);

    //rcqs_conj(ret, a);
    rqs_set(ret->val_re, in_a.val_re);
    rqs_neg(ret->val_im, in_a.val_im);

    //rcqs_nrm2(&a_nrm2, a);
    rqs_mul(a_nrm2.val, in_a.val_re, in_a.val_re);
    rqs_mul(tmp.val, in_a.val_im, in_a.val_im);
    rqs_add(a_nrm2.val, a_nrm2.val, tmp.val);

    rqs_div(ret->val_re, ret->val_re, a_nrm2.val);
    rqs_div(ret->val_im, ret->val_im, a_nrm2.val);
}

// ret := a / b : 3M
static inline void rcqs_div_3m(cqsfloat *ret, cqsfloat *a, cqsfloat *b)
{
    cqsfloat in_a, inv_b;

    rcqs_inv(&inv_b, b);
    rcqs_set(&in_a, a);
    rcqs_mul_3m(ret, &in_a, &inv_b);
}

// ret := a / b : 4M
static inline void rcqs_div_4m(cqsfloat *ret, cqsfloat *a, cqsfloat *b)
{
    cqsfloat in_a, inv_b;

    rcqs_inv(&inv_b, b);
    rcqs_set(&in_a, a); // fix! 2026-02-06(Sun) T.Kouya
    rcqs_mul_4m(ret, &in_a, &inv_b);
}

// ret := a * (real)b = real(a) * b + imag(a) * b
static inline void rcqs_mul_qs(cqsfloat *ret, cqsfloat *a, float b[QSSIZE])
{
    rqs_mul(ret->val_re, a->val_re, b);
    rqs_mul(ret->val_im, a->val_im, b);
}

// ret := a * (real)b = real(a) * b + imag(a) * b
static inline void rcqs_mul_d(cqsfloat *ret, cqsfloat *a, float b)
{
    rqs_mul_f(ret->val_re, a->val_re, b);
    rqs_mul_f(ret->val_im, a->val_im, b);
}
#define rcqs_mul_ui(ret, a, b) rcqs_mul_d((ret), (a), (float)(b))
#define rcqs_mul_si(ret, a, b) rcqs_mul_d((ret), (a), (float)(b))

// ret := a * (real)b = real(a) / b + imag(a) / b
static inline void rcqs_div_qs(cqsfloat *ret, cqsfloat *a, float b[QSSIZE])
{
    rqs_div(ret->val_re, a->val_re, b);
    rqs_div(ret->val_im, a->val_im, b);
}

// ret := a * (real)b = real(a) / b + imag(a) / b
static inline void rcqs_div_d(cqsfloat *ret, cqsfloat *a, float b)
{
    rqs_div_f(ret->val_re, a->val_re, b);
    rqs_div_f(ret->val_im, a->val_im, b);
}
#define rcqs_div_ui(ret, a, b) rcqs_div_d((ret), (a), (float)(b))
#define rcqs_div_si(ret, a, b) rcqs_div_d((ret), (a), (float)(b))

// 2024-02-18(SUN) Tomonori Kouya
// Default: 4M method
#ifdef USE_RCQS_MUL_4M // 4M
    #define rcqs_mul rcqs_mul_4m
    #define rcqs_div rcqs_div_4m
#elif defined(USE_RCQS_MUL_3M) // 3M
    #define rcqs_mul rcqs_mul_3m
    #define rcqs_div rcqs_div_3m
#else // USE_RCDS_MUL_4M or 3M
    #define rcqs_mul rcqs_mul_4m
    #define rcqs_div rcqs_div_4m
    //#define rcqs_mul rcqs_mul_3m
    //#define rcqs_div rcqs_div_3m
#endif // USE_RCDS_MUL_4M

// ret := a * b : 4M
static inline void rcqs_mul_cd(cqsfloat *ret, cqsfloat *a, float _Complex b)
{
    qsfloat t1, t2, t3, t4;

    // ret := a * b
    //rds_mul(ret->val_re, a->val_re, b->val_re);
    rqs_mul_f(t1.val, a->val_re, __real__ b); // creal(b));
    rqs_mul_f(t2.val, a->val_im, __imag__ b); // cimag(b));
    rqs_mul_f(t3.val, a->val_re, __imag__ b); // cimag(b));
    rqs_mul_f(t4.val, a->val_im, __real__ b); // creal(b));

    rqs_sub(ret->val_re, t1.val, t2.val);
    rqs_add(ret->val_im, t3.val, t4.val);
}

// ret := a / b : 4M
static inline void rcqs_div_cd(cqsfloat *ret, cqsfloat *a, float _Complex b)
{
    qsfloat t1, t2, t3, t4, abs_b2;

    // ret := a * conj(b) / |b|^2
    //rds_mul(ret->val_re, a->val_re, b->val_re);
    rqs_mul_f(t1.val, a->val_re, __real__ b); // creal(b));
    rqs_mul_f(t2.val, a->val_im, __imag__ b); // cimag(b));
    rqs_mul_f(t3.val, a->val_re, __imag__ b); // cimag(b));
    rqs_mul_f(t4.val, a->val_im, __real__ b); // creal(b));

    rqs_add(ret->val_re, t1.val, t2.val);
    rqs_sub(ret->val_im, t4.val, t3.val);

    rqs_set_d(abs_b2.val, __real__ b);
    rqs_mul_f(abs_b2.val, abs_b2.val, __real__ b);
    rqs_set_d(t1.val, __imag__ b);
    rqs_mul_f(t1.val, t1.val, __imag__ b);
    rqs_add(abs_b2.val, abs_b2.val, t1.val);

    rqs_div(ret->val_re, ret->val_re, abs_b2.val);
    rqs_div(ret->val_im, ret->val_im, abs_b2.val);
}

// print cqsfloat
static inline void rcqs_out_str(cqsfloat *ret)
{
    rqs_out_str_base(stdout, 10, 64, ret->val_re);
    //rqs_out_str_base(stdout, 10, 64, ret->val_re);
    printf(" + ");
    rqs_out_str_base(stdout, 10, 64, ret->val_im);
    //rqs_out_str_base(stdout, 10, 64, ret->val_im);
    printf(" * I");
}

// 2024-12-03 (Tue) T.Kouya
// |a| >  |b| -> +1
// |a| == |b| ->  0
// |a| <  |b| -> -1
static inline int rcqs_cmp_abs(cqsfloat *a, cqsfloat *b)
{
    float abs_a[QSSIZE], abs_b[QSSIZE];

    rcqs_abs_qs(abs_a, a);
    rcqs_abs_qs(abs_b, b);

    return rqs_cmp(abs_a, abs_b);
}
static inline int rcqs_cmp_abs_qs(cqsfloat *a, float b[QSSIZE])
{
    float abs_a[QSSIZE], abs_b[QSSIZE];

    rcqs_abs_qs(abs_a, a);
    rqs_abs(abs_b, b);

    return rqs_cmp(abs_a, abs_b);
}
static inline int rcqs_cmp_abs_d(cqsfloat *a, float b)
{
    float abs_a[QSSIZE], abs_b;

    rcqs_abs_qs(abs_a, a);
    abs_b = fabs(b);

    return rds_cmp_f(abs_a, abs_b);
}
#define rcqs_cmp_abs_ui(a, b) rcqs_cmp_abs_d((a), ((float)(b)))

// CQD fma
// ret = a * b + c
static inline void rcqs_fma(cqsfloat *ret, cqsfloat *a, cqsfloat *b, cqsfloat *c)
{
    cqsfloat tmp;

	rcqs_mul(&tmp, a, b);
	rcqs_add(ret, &tmp, c);

	return;
}

#ifdef __cplusplus
} //extern "C" {
#endif // __cplusplus

#endif //ifndef __BNC_RCDS_H_
