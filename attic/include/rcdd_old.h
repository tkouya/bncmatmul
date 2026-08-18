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
typedef struct { double val_re[DDSIZE]; double val_im[DDSIZE]; } cddfloat; // 53 * 2 = 106
typedef struct { double val_re[TDSIZE]; double val_im[TDSIZE]; } ctdfloat; // 53 * 3 = 159
typedef struct { double val_re[QDSIZE]; double val_im[QDSIZE]; } cqdfloat; // 53 * 4 = 212

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

// ret := a
static inline void rcdd_set_cd(cddfloat *ret, double _Complex a)
{
    ret->val_re[0] = __real__ a; //creal(a);
    ret->val_re[1] = 0.0;
    ret->val_im[0] = __imag__ a; //cimag(a);
    ret->val_im[1] = 0.0;
}

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

// ret := a - b
static inline void rcdd_sub(cddfloat *ret, cddfloat *a, cddfloat *b)
{
    // ret := a - b
    rdd_sub(ret->val_re, a->val_re, b->val_re);
    rdd_sub(ret->val_im, a->val_im, b->val_im);
}

// ret := a * b : 4M
static inline void rcdd_mul_4m(cddfloat *ret, cddfloat *a, cddfloat *b)
{
    ddfloat tmp;

    // ret := a * b
    rdd_mul(ret->val_re, a->val_re, b->val_re);
    rdd_mul(tmp.val, a->val_im, b->val_im);
    rdd_sub(ret->val_re, ret->val_re, tmp.val);

    rdd_mul(ret->val_im, a->val_re, b->val_im);
    rdd_mul(tmp.val, a->val_im, b->val_re);
    rdd_add(ret->val_im, ret->val_im, tmp.val);
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

    // a_re_im := re_a + im_a
    // b_re_im := re_b + im_b
    // -> ret_im  := a_re_im * b_re_im - t1 - t2
    rdd_add(a_re_im.val, a->val_re, a->val_im);
    rdd_add(b_re_im.val, b->val_re, b->val_im);
    rdd_mul(tmp.val, a_re_im.val, b_re_im.val);
    rdd_sub(tmp.val, tmp.val, t1.val);
    rdd_sub(ret->val_im, tmp.val, t2.val);
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
static inline void rcdd_inv(cddfloat *ret, cddfloat *a)
{
    ddfloat a_nrm2;

    rcdd_conj(ret, a);
    rcdd_nrm2(&a_nrm2, a);
    rdd_div(ret->val_re, ret->val_re, a_nrm2.val);
    rdd_div(ret->val_im, ret->val_im, a_nrm2.val);
}

// ret := a / b : 3M
static inline void rcdd_div_3m(cddfloat *ret, cddfloat *a, cddfloat *b)
{
    cddfloat inv_b;
    ddfloat t1, t2, tmp, a_re_im, b_re_im, b_nrm2;

    //rcdd_inv(&inv_b, b);

        //rcdd_conj(&inv_b, b);
        rdd_set(inv_b.val_re, b->val_re);
        rdd_neg(inv_b.val_im, b->val_im);
        //rcdd_nrm2(&b_nrm2, b);
        rdd_mul(t1.val, inv_b.val_re, inv_b.val_re);
        rdd_mul(t2.val, inv_b.val_im, inv_b.val_im);
        rdd_add(b_nrm2.val, t1.val, t2.val);

    rdd_div(inv_b.val_re, inv_b.val_re, b_nrm2.val);
    rdd_div(inv_b.val_im, inv_b.val_im, b_nrm2.val);

    //rcdd_mul_3m(ret, a, &inv_b);

    // t1      := re_a * re_b
    // t2      := im_a * im_b
    // -> ret_re  := t1 - t2 
    rdd_mul(t1.val, a->val_re, inv_b.val_re);
    rdd_mul(t2.val, a->val_im, inv_b.val_im);
    rdd_sub(ret->val_re, t1.val, t2.val);

    // a_re_im := re_a + im_a
    // b_re_im := re_b + im_b
    // -> ret_im  := a_re_im * b_re_im - t1 - t2
    rdd_add(a_re_im.val, a->val_re, a->val_im);
    rdd_add(b_re_im.val, inv_b.val_re, inv_b.val_im);
    rdd_mul(tmp.val, a_re_im.val, b_re_im.val);
    rdd_sub(tmp.val, tmp.val, t1.val);
    rdd_sub(ret->val_im, tmp.val, t2.val);
}

// ret := a / b : 4M
static inline void rcdd_div_4m(cddfloat *ret, cddfloat *a, cddfloat *b)
{
    cddfloat inv_b;
    ddfloat tmp;

    //rcdd_inv(&inv_b, b);
    
    //rcdd_mul_4m(ret, a, &inv_b);

    // ret := a * b
    rdd_mul(ret->val_re, a->val_re, inv_b.val_re);
    rdd_mul(tmp.val, a->val_im, inv_b.val_im);
    rdd_sub(ret->val_re, ret->val_re, tmp.val);

    rdd_mul(ret->val_im, a->val_re, inv_b.val_im);
    rdd_mul(tmp.val, a->val_im, inv_b.val_re);
    rdd_add(ret->val_im, ret->val_im, tmp.val);
}

#ifndef USE_RCDD_MUL_4M
    #define rcdd_mul rcdd_mul_3m
    #define rcdd_div rcdd_div_3m
#else // USE_RCDD_MUL_4M
    #define rcdd_mul rcdd_mul_4m
    #define rcdd_div rcdd_div_4m
#endif // USE_RCDD_MUL_4M

// print cddfloat
static inline void rcdd_out_str(cddfloat *ret)
{
    rdd_out_str(ret->val_re);
    printf(" + ");
    rdd_out_str(ret->val_im);
    printf(" * I");
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
    rdd_set0(ret->val_im);
}
static inline void rctd_set_dd(ctdfloat *ret, double a[DDSIZE])
{
    ret->val_re[0] = a[0];
    ret->val_re[1] = a[1];
    ret->val_re[2] = 0.0;
    rdd_set0(ret->val_im);
}

// ret := a
static inline void rctd_set_d(ctdfloat *ret, double a)
{
    ret->val_re[0] = a;
    ret->val_re[1] = 0.0;
    ret->val_re[2] = 0.0;
    rdd_set0(ret->val_im);
}

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

// ret := a - b
static inline void rctd_sub(ctdfloat *ret, ctdfloat *a, ctdfloat *b)
{
    // ret := a - b
    rtd_sub(ret->val_re, a->val_re, b->val_re);
    rtd_sub(ret->val_im, a->val_im, b->val_im);
}

// ret := a * b : 4M
static inline void rctd_mul_4m(ctdfloat *ret, ctdfloat *a, ctdfloat *b)
{
    tdfloat tmp;

    // ret := a * b
    rtd_mul(ret->val_re, a->val_re, b->val_re);
    rtd_mul(tmp.val, a->val_im, b->val_im);
    rtd_sub(ret->val_re, ret->val_re, tmp.val);

    rtd_mul(ret->val_im, a->val_re, b->val_im);
    rtd_mul(tmp.val, a->val_im, b->val_re);
    rtd_add(ret->val_im, ret->val_im, tmp.val);
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
    tdfloat a_nrm2;

    rctd_conj(ret, a);
    rctd_nrm2(&a_nrm2, a);
    rtd_div(ret->val_re, ret->val_re, a_nrm2.val);
    rtd_div(ret->val_im, ret->val_im, a_nrm2.val);
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
    ctdfloat inv_b;

    rctd_inv(&inv_b, b);
    rctd_mul_4m(ret, a, &inv_b);
}

#ifndef USE_4M
    #define rctd_mul rctd_mul_3m
    #define rctd_div rctd_div_3m
#else // USE_4M
    #define rctd_mul rctd_mul_4m
    #define rctd_div rctd_div_4m
#endif // USE_4M

// print ctdfloat
static inline void rctd_out_str(ctdfloat *ret)
{
    rtd_out_str(ret->val_re);
    printf(" + ");
    rtd_out_str(ret->val_im);
    printf(" * I");
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

// ret := a
static inline void rcqd_set_cdd(ctdfloat *ret, cddfloat *a)
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
static inline void rcqd_set_ctd(ctdfloat *ret, ctdfloat *a)
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

// ret := a - b
static inline void rcqd_sub(cqdfloat *ret, cqdfloat *a, cqdfloat *b)
{
    // ret := a - b
    rqd_sub(ret->val_re, a->val_re, b->val_re);
    rqd_sub(ret->val_im, a->val_im, b->val_im);
}

// ret := a * b : 4M
static inline void rcqd_mul_4m(cqdfloat *ret, cqdfloat *a, cqdfloat *b)
{
    qdfloat tmp;

    // ret := a * b
    rqd_mul(ret->val_re, a->val_re, b->val_re);
    rqd_mul(tmp.val, a->val_im, b->val_im);
    rqd_sub(ret->val_re, ret->val_re, tmp.val);

    rqd_mul(ret->val_im, a->val_re, b->val_im);
    rqd_mul(tmp.val, a->val_im, b->val_re);
    rqd_add(ret->val_im, ret->val_im, tmp.val);
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
    rqd_sqrt(ret->val, ret->val);
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
    qdfloat a_nrm2;

    rcqd_conj(ret, a);
    rcqd_nrm2(&a_nrm2, a);
    rqd_div(ret->val_re, ret->val_re, a_nrm2.val);
    rqd_div(ret->val_im, ret->val_im, a_nrm2.val);
}

// ret := a / b : 3M
static inline void rcqd_div_3m(cqdfloat *ret, cqdfloat *a, cqdfloat *b)
{
    cqdfloat inv_b;

    rcqd_inv(&inv_b, b);
    rcqd_mul_3m(ret, a, &inv_b);
}

// ret := a / b : 4M
static inline void rcqd_div_4m(cqdfloat *ret, cqdfloat *a, cqdfloat *b)
{
    cqdfloat inv_b;

    rcqd_inv(&inv_b, b);
    rcqd_mul_4m(ret, a, &inv_b);
}

#ifndef USE_4M
    #define rcqd_mul rcqd_mul_3m
    #define rcqd_div rcqd_div_3m
#else // USE_4M
    #define rcqd_mul rcqd_mul_4m
    #define rcqd_div rcqd_div_4m
#endif // USE_4M

// print cqdfloat
static inline void rcqd_out_str(cqdfloat *ret)
{
    rqd_out_str(ret->val_re);
    printf(" + ");
    rqd_out_str(ret->val_im);
    printf(" * I");
}

#ifdef __cplusplus
} //extern "C" {
#endif // __cplusplus

#endif //ifndef __BNC_RCDD_H_
