#ifndef __BNCNEON_DD_H
#define __BNCNEON_DD_H

#include <arm_neon.h>

// ----------------------------------------
// Double-Double Arithmetic with NEON
// ----------------------------------------

#ifndef DDSIZE
    #define DDSIZE 2
#endif // DDSIZE

// ret := 0
static inline void _bncneon_set0_dd(float64x2_t ret[DDSIZE])
{
    ret[0] = vdupq_n_f64(0.0);
    ret[1] = vdupq_n_f64(0.0);
}
#define _bncneon_rdd_set0(ret) _bncneon_set0_dd((ret))


// ret := a + b
static inline void _bncneon_rdd_add(float64x2_t ret[DDSIZE], float64x2_t a[DDSIZE], float64x2_t b[DDSIZE])
{
    float64x2_t s, e;
    s = _bncneon_dtwo_sum(a[0], b[0], &e);
    e = vaddq_f64(e, vaddq_f64(a[1], b[1]));
    ret[0] = _bncneon_dquick_two_sum(s, e, &e);
    ret[1] = e;
}

// ret := a - b
static inline void _bncneon_rdd_sub(float64x2_t ret[DDSIZE], float64x2_t a[DDSIZE], float64x2_t b[DDSIZE])
{
    float64x2_t s1, s2, t1, t2;
    s1 = _bncneon_dtwo_diff(a[0], b[0], &s2);
    t1 = _bncneon_dtwo_diff(a[1], b[1], &t2);
    s2 = vaddq_f64(s2, t1);
    s1 = _bncneon_dquick_two_sum(s1, s2, &s2);
    s2 = vaddq_f64(s2, t2);
    ret[0] = _bncneon_dquick_two_sum(s1, s2, &ret[1]);
}

// ret := a * b
static inline void _bncneon_rdd_mul(float64x2_t ret[DDSIZE], float64x2_t a[DDSIZE], float64x2_t b[DDSIZE])
{
    float64x2_t p1, p2;
    p1 = _bncneon_dtwo_prod(a[0], b[0], &p2);
    p2 = vaddq_f64(p2, vmulq_f64(a[0], b[1]));
    p2 = vaddq_f64(p2, vmulq_f64(a[1], b[0]));
    ret[0] = _bncneon_dquick_two_sum(p1, p2, &ret[1]);
}

// ret := |a|
static inline void _bncneon_rdd_abs(float64x2_t ret[DDSIZE], float64x2_t a[DDSIZE])
{
    // Check the sign of the high part (a[0])
    // If a[0] is negative, negate both a[0] and a[1]
    uint64x2_t sign_mask = vcltq_f64(a[0], vdupq_n_f64(0.0));
    ret[0] = vbslq_f64(sign_mask, vnegq_f64(a[0]), a[0]);
    ret[1] = vbslq_f64(sign_mask, vnegq_f64(a[1]), a[1]);
}

// ----------------------------------------
// Reduction/Utility Functions
// ----------------------------------------

// ret := ret2[0] + ret2[1]
static inline void _bncneon_rdd_sum128d(double ret[DDSIZE], float64x2_t ret2[DDSIZE])
{
    double val0[DDSIZE], val1[DDSIZE];
    val0[0] = vgetq_lane_f64(ret2[0], 0);
    val0[1] = vgetq_lane_f64(ret2[1], 0);
    val1[0] = vgetq_lane_f64(ret2[0], 1);
    val1[1] = vgetq_lane_f64(ret2[1], 1);
    
    rdd_add(ret, val0, val1);
}

// ret := |ret2[0]| + |ret2[1]|
static inline void _bncneon_rdd_abssum128d(double ret[DDSIZE], float64x2_t ret2[DDSIZE])
{
    double val0[DDSIZE], val1[DDSIZE];
    double tmp0[DDSIZE], tmp1[DDSIZE];

    val0[0] = vgetq_lane_f64(ret2[0], 0);
    val0[1] = vgetq_lane_f64(ret2[1], 0);
    val1[0] = vgetq_lane_f64(ret2[0], 1);
    val1[1] = vgetq_lane_f64(ret2[1], 1);
    
    rdd_abs(tmp0, val0);
    rdd_abs(tmp1, val1);
    
    rdd_add(ret, tmp0, tmp1);
}

#endif // __BNCNEON_DD_H
