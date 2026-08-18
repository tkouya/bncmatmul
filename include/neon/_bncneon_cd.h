// ------------------------
// ---- Complex Double ----
// ------------------------
#ifndef __BNCNEON_CD_H
#define __BNCNEON_CD_H

#include <arm_neon.h>
#include "_bncneon_d.h"  // For _bncneon_dneg
//#include "_bncneon_dd.h" // For double-double functions

//#ifndef DDSIZE
//    #define DDSIZE 2
//#endif // DDSIZE

// ret := 0
static inline void _bncneon_rcd_set0(float64x2_t *ret_re, float64x2_t *ret_im)
{
    //_bncneon_set0_dd(ret_re);
    //_bncneon_set0_dd(ret_im);
    *ret_re = vdupq_n_f64(0.0); // 初期化
    *ret_im = vdupq_n_f64(0.0);
}

// ret := a
static inline void _bncneon_rcd_set(float64x2_t *ret_re, float64x2_t *ret_im, float64x2_t a_re, float64x2_t a_im)
{
    //ret_re[0] = a_re[0]; ret_re[1] = a_re[1];
    //ret_im[0] = a_im[0]; ret_im[1] = a_im[1];
    *ret_re = a_re;
    *ret_im = a_im;
}//

// ret := conj(a)
static inline void _bncneon_rcd_conj(float64x2_t *ret_re, float64x2_t *ret_im, float64x2_t a_re, float64x2_t a_im)
{
    _bncneon_rcd_set(*ret_re, *ret_im, a_re, a_im); // Real part is the same
    *ret_im = vnegq_f64(a_im);
}

// ret := a + b
static inline void _bncneon_rcd_add(float64x2_t *ret_re, float64x2_t *ret_im, float64x2_t a_re, float64x2_t a_im, float64x2_t b_re, float64x2_t b_im)
{
    //_bncneon_rdd_add(ret_re, a_re, b_re);
    //_bncneon_rdd_add(ret_im, a_im, b_im);
    *ret_re = vaddq_f64(a_re, b_re);
    *ret_im = vaddq_f64(a_im, b_im);
}

// ret := a - b
static inline void _bncneon_rcd_sub(float64x2_t *ret_re, float64x2_t *ret_im, float64x2_t a_re, float64x2_t a_im, float64x2_t b_re, float64x2_t b_im)
{
    //_bncneon_rdd_sub(ret_re, a_re, b_re);
    //_bncneon_rdd_sub(ret_im, a_im, b_im);
    *ret_re = vsubq_f64(a_re, b_re);
    *ret_im = vsubq_f64(a_im, b_im);
}

// ret := a * b
static inline void _bncneon_rcd_mul(float64x2_t *ret_re, float64x2_t *ret_im, float64x2_t a_re, float64x2_t a_im, float64x2_t b_re, float64x2_t b_im)
{
    float64x2_t term1, term2;
    //_bncneon_rdd_mul(term1, a_re, b_re);
    //_bncneon_rdd_mul(term2, a_im, b_im);
    //_bncneon_rdd_sub(ret_re, term1, term2);
    term1 = vmulq_f64(a_re, b_re);
    term2 = vmulq_f64(a_im, b_im);
    *ret_re = vsubq_f64(term1, term2);

    //_bncneon_rdd_mul(term1, a_re, b_im);
    //_bncneon_rdd_mul(term2, a_im, b_re);
    //_bncneon_rdd_add(ret_im, term1, term2);
    term1 = vmulq_f64(a_re, b_im);
    term2 = vmulq_f64(a_im, b_re);
    *ret_im = vaddq_f64(term1, term2);
}

#endif // __BNCNEON_CD_H
