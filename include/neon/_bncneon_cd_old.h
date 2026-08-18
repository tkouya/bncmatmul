// ------------------------
// ---- Complex Double ----
// ------------------------
#ifndef __BNCNEON_CD_H
#define __BNCNEON_CD_H

#include <arm_neon.h>
#include "_bncneon_d.h"  // For _bncneon_dneg
#include "_bncneon_dd.h" // For double-double functions

#ifndef DDSIZE
    #define DDSIZE 2
#endif // DDSIZE

// ret := 0
static inline void _bncneon_rcdd_set0(float64x2_t ret_re[DDSIZE], float64x2_t ret_im[DDSIZE])
{
    _bncneon_set0_dd(ret_re);
    _bncneon_set0_dd(ret_im);
}

// ret := a
static inline void _bncneon_rcdd_set(float64x2_t ret_re[DDSIZE], float64x2_t ret_im[DDSIZE], float64x2_t a_re[DDSIZE], float64x2_t a_im[DDSIZE])
{
    ret_re[0] = a_re[0]; ret_re[1] = a_re[1];
    ret_im[0] = a_im[0]; ret_im[1] = a_im[1];
}

// ret := conj(a)
static inline void _bncneon_rcdd_conj(float64x2_t ret_re[DDSIZE], float64x2_t ret_im[DDSIZE], float64x2_t a_re[DDSIZE], float64x2_t a_im[DDSIZE])
{
    _bncneon_rcdd_set(ret_re, ret_im, a_re, a_im); // Real part is the same
    ret_im[0] = vnegq_f64(a_im[0]);
    ret_im[1] = vnegq_f64(a_im[1]);
}

// ret := a + b
static inline void _bncneon_rcdd_add(float64x2_t ret_re[DDSIZE], float64x2_t ret_im[DDSIZE], float64x2_t a_re[DDSIZE], float64x2_t a_im[DDSIZE], float64x2_t b_re[DDSIZE], float64x2_t b_im[DDSIZE])
{
    _bncneon_rdd_add(ret_re, a_re, b_re);
    _bncneon_rdd_add(ret_im, a_im, b_im);
}

// ret := a - b
static inline void _bncneon_rcdd_sub(float64x2_t ret_re[DDSIZE], float64x2_t ret_im[DDSIZE], float64x2_t a_re[DDSIZE], float64x2_t a_im[DDSIZE], float64x2_t b_re[DDSIZE], float64x2_t b_im[DDSIZE])
{
    _bncneon_rdd_sub(ret_re, a_re, b_re);
    _bncneon_rdd_sub(ret_im, a_im, b_im);
}

// ret := a * b
static inline void _bncneon_rcdd_mul(float64x2_t ret_re[DDSIZE], float64x2_t ret_im[DDSIZE], float64x2_t a_re[DDSIZE], float64x2_t a_im[DDSIZE], float64x2_t b_re[DDSIZE], float64x2_t b_im[DDSIZE])
{
    float64x2_t term1[DDSIZE], term2[DDSIZE];
    _bncneon_rdd_mul(term1, a_re, b_re);
    _bncneon_rdd_mul(term2, a_im, b_im);
    _bncneon_rdd_sub(ret_re, term1, term2);

    _bncneon_rdd_mul(term1, a_re, b_im);
    _bncneon_rdd_mul(term2, a_im, b_re);
    _bncneon_rdd_add(ret_im, term1, term2);
}

#endif // __BNCNEON_CD_H
