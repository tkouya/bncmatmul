// ------------------------
// ------ Complex QD ------
// ------------------------
#ifndef __BNCNEON_CQD_H
#define __BNCNEON_CQD_H

#include "_bncneon_qd.h"

#ifndef QDSIZE
    #define QDSIZE 4
#endif // QDSIZE

// ret := 0
static inline void _bncneon_rcqd_set0(float64x2_t ret_re[QDSIZE], float64x2_t ret_im[QDSIZE])
{
    _bncneon_rqd_set0(ret_re);
    _bncneon_rqd_set0(ret_im);
}

// ret := a
static inline void _bncneon_rcqd_set(float64x2_t ret_re[QDSIZE], float64x2_t ret_im[QDSIZE], float64x2_t a_re[QDSIZE], float64x2_t a_im[QDSIZE])
{
    _bncneon_rqd_set(ret_re, a_re);
    _bncneon_rqd_set(ret_im, a_im);
}

// ret := conj(a)
static inline void _bncneon_rcqd_conj(float64x2_t ret_re[QDSIZE], float64x2_t ret_im[QDSIZE], float64x2_t a_re[QDSIZE], float64x2_t a_im[QDSIZE])
{
    _bncneon_rqd_set(ret_re, a_re);
    _bncneon_rqd_neg(ret_im, a_im);
}

// ret := a + b
static inline void _bncneon_rcqd_add(float64x2_t ret_re[QDSIZE], float64x2_t ret_im[QDSIZE], float64x2_t a_re[QDSIZE], float64x2_t a_im[QDSIZE], float64x2_t b_re[QDSIZE], float64x2_t b_im[QDSIZE])
{
    _bncneon_rqd_add(ret_re, a_re, b_re);
    _bncneon_rqd_add(ret_im, a_im, b_im);
}

// ret := a - b
static inline void _bncneon_rcqd_sub(float64x2_t ret_re[QDSIZE], float64x2_t ret_im[QDSIZE], float64x2_t a_re[QDSIZE], float64x2_t a_im[QDSIZE], float64x2_t b_re[QDSIZE], float64x2_t b_im[QDSIZE])
{
    _bncneon_rqd_sub(ret_re, a_re, b_re);
    _bncneon_rqd_sub(ret_im, a_im, b_im);
}

// ret := a * b
static inline void _bncneon_rcqd_mul(float64x2_t ret_re[QDSIZE], float64x2_t ret_im[QDSIZE], float64x2_t a_re[QDSIZE], float64x2_t a_im[QDSIZE], float64x2_t b_re[QDSIZE], float64x2_t b_im[QDSIZE])
{
    float64x2_t t1[QDSIZE], t2[QDSIZE];
    _bncneon_rqd_mul(t1, a_re, b_re);
    _bncneon_rqd_mul(t2, a_im, b_im);
    _bncneon_rqd_sub(ret_re, t1, t2);

    _bncneon_rqd_mul(t1, a_re, b_im);
    _bncneon_rqd_mul(t2, a_im, b_re);
    _bncneon_rqd_add(ret_im, t1, t2);
}

#endif // __BNCNEON_CQD_H
