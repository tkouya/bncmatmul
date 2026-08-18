// ------------------------
// ------ Complex TD ------
// ------------------------
#ifndef __BNCNEON_CTD_H
#define __BNCNEON_CTD_H

#include "_bncneon_td.h"

#ifndef TDSIZE
    #define TDSIZE 3
#endif // TDSIZE

// ret := 0
static inline void _bncneon_rctd_set0(float64x2_t ret_re[TDSIZE], float64x2_t ret_im[TDSIZE])
{
    _bncneon_rtd_set0(ret_re);
    _bncneon_rtd_set0(ret_im);
}

// ret := a
static inline void _bncneon_rctd_set(float64x2_t ret_re[TDSIZE], float64x2_t ret_im[TDSIZE], float64x2_t a_re[TDSIZE], float64x2_t a_im[TDSIZE])
{
    _bncneon_rtd_set(ret_re, a_re);
    _bncneon_rtd_set(ret_im, a_im);
}

// ret := conj(a)
static inline void _bncneon_rctd_conj(float64x2_t ret_re[TDSIZE], float64x2_t ret_im[TDSIZE], float64x2_t a_re[TDSIZE], float64x2_t a_im[TDSIZE])
{
    _bncneon_rtd_set(ret_re, a_re);
    _bncneon_rtd_neg(ret_im, a_im);
}

// ret := a + b
static inline void _bncneon_rctd_add(float64x2_t ret_re[TDSIZE], float64x2_t ret_im[TDSIZE], float64x2_t a_re[TDSIZE], float64x2_t a_im[TDSIZE], float64x2_t b_re[TDSIZE], float64x2_t b_im[TDSIZE])
{
    _bncneon_rtd_add(ret_re, a_re, b_re);
    _bncneon_rtd_add(ret_im, a_im, b_im);
}

// ret := a - b
static inline void _bncneon_rctd_sub(float64x2_t ret_re[TDSIZE], float64x2_t ret_im[TDSIZE], float64x2_t a_re[TDSIZE], float64x2_t a_im[TDSIZE], float64x2_t b_re[TDSIZE], float64x2_t b_im[TDSIZE])
{
    _bncneon_rtd_sub(ret_re, a_re, b_re);
    _bncneon_rtd_sub(ret_im, a_im, b_im);
}

// ret := a * b
static inline void _bncneon_rctd_mul(float64x2_t ret_re[TDSIZE], float64x2_t ret_im[TDSIZE], float64x2_t a_re[TDSIZE], float64x2_t a_im[TDSIZE], float64x2_t b_re[TDSIZE], float64x2_t b_im[TDSIZE])
{
    float64x2_t t1[TDSIZE], t2[TDSIZE];
    _bncneon_rtd_mul(t1, a_re, b_re);
    _bncneon_rtd_mul(t2, a_im, b_im);
    _bncneon_rtd_sub(ret_re, t1, t2);

    _bncneon_rtd_mul(t1, a_re, b_im);
    _bncneon_rtd_mul(t2, a_im, b_re);
    _bncneon_rtd_add(ret_im, t1, t2);
}

#endif // __BNCNEON_CTD_H
