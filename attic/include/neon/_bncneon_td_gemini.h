// ------------------------
// ---------- TD ----------
// ------------------------
#ifndef __BNCNEON_TD_H
#define __BNCNEON_TD_H

#include "_bncneon_dd.h"

#ifndef TDSIZE
    #define TDSIZE 3
#endif // TDSIZE

// NOTE: Implementation depends on scalar helper functions like renorm, vec_sum, merge.

// ret := 0
static inline void _bncneon_rtd_set0(float64x2_t ret[TDSIZE])
{
    ret[0] = vdupq_n_f64(0.0);
    ret[1] = vdupq_n_f64(0.0);
    ret[2] = vdupq_n_f64(0.0);
}

// ret := val
static inline void _bncneon_rtd_set(float64x2_t ret[TDSIZE], float64x2_t val[TDSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = val[2];
}

// ret := -a
static inline void _bncneon_rtd_neg(float64x2_t c[TDSIZE], float64x2_t a[TDSIZE])
{
    c[0] = vnegq_f64(a[0]);
    c[1] = vnegq_f64(a[1]);
    c[2] = vnegq_f64(a[2]);
}

// Placeholder for add, sub, mul, div which require scalar helpers
static inline void _bncneon_rtd_add(float64x2_t ret[TDSIZE], float64x2_t a[TDSIZE], float64x2_t b[TDSIZE]) { /* Requires scalar renorm etc. */ }
static inline void _bncneon_rtd_sub(float64x2_t ret[TDSIZE], float64x2_t a[TDSIZE], float64x2_t b[TDSIZE]) { _bncneon_rtd_neg(b, b); _bncneon_rtd_add(ret, a, b); }
static inline void _bncneon_rtd_mul(float64x2_t ret[TDSIZE], float64x2_t a[TDSIZE], float64x2_t b[TDSIZE]) { /* Requires scalar renorm etc. */ }
static inline void _bncneon_rtd_div(float64x2_t ret[TDSIZE], float64x2_t a[TDSIZE], float64x2_t b[TDSIZE]) { /* Requires scalar renorm etc. */ }


#endif // __BNCNEON_TD_H
