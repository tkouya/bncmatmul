// ------------------------
// ---------- DS ----------
// ------------------------
#ifndef __BNCNEON_DS_H
#define __BNCNEON_DS_H

#include "_bncneon_feft.h"

#ifndef DSSIZE
    #define DSSIZE 2
#endif // DSSIZE

// ret := 0
static inline void _bncneon_set0_ds(float32x4_t ret[DSSIZE])
{
    ret[0] = vdupq_n_f32(0.0f);
    ret[1] = vdupq_n_f32(0.0f);
}
#define _bncneon_rds_set0(ret) _bncneon_set0_ds((ret))

// ret := a + b
static inline void _bncneon_rds_add(float32x4_t ret[DSSIZE], float32x4_t a[DSSIZE], float32x4_t b[DSSIZE])
{
    float32x4_t s, e;
    s = _bncneon_ftwo_sum(a[0], b[0], &e);
    e = vaddq_f32(e, vaddq_f32(a[1], b[1]));
    ret[0] = _bncneon_fquick_two_sum(s, e, &e);
    ret[1] = e;
}

// ret := a - b
static inline void _bncneon_rds_sub(float32x4_t ret[DSSIZE], float32x4_t a[DSSIZE], float32x4_t b[DSSIZE])
{
    float32x4_t s1, s2, t1, t2;
    s1 = _bncneon_ftwo_diff(a[0], b[0], &s2);
    t1 = _bncneon_ftwo_diff(a[1], b[1], &t2);
    s2 = vaddq_f32(s2, t1);
    s1 = _bncneon_fquick_two_sum(s1, s2, &s2);
    s2 = vaddq_f32(s2, t2);
    ret[0] = _bncneon_fquick_two_sum(s1, s2, &ret[1]);
}

// ret := a * b
static inline void _bncneon_rds_mul(float32x4_t ret[DSSIZE], float32x4_t a[DSSIZE], float32x4_t b[DSSIZE])
{
    float32x4_t p1, p2;
    p1 = _bncneon_ftwo_prod(a[0], b[0], &p2);
    p2 = vfmaq_f32(p2, a[0], b[1]);
    p2 = vfmaq_f32(p2, a[1], b[0]);
    ret[0] = _bncneon_fquick_two_sum(p1, p2, &ret[1]);
}

// ret := a * (float)b
static inline void _bncneon_rds_mul_f(float32x4_t ret[DSSIZE], float32x4_t a[DSSIZE], float32x4_t b)
{
    float32x4_t p1, p2;
    p1 = _bncneon_ftwo_prod(a[0], b, &p2);
    p2 = vfmaq_f32(p2, a[1], b);
    ret[0] = _bncneon_fquick_two_sum(p1, p2, &ret[1]);
}

// ret := a / b
static inline void _bncneon_rds_div(float32x4_t ret[DSSIZE], float32x4_t a[DSSIZE], float32x4_t b[DSSIZE])
{
    float32x4_t q1, s1, s2;
    float32x4_t r[DSSIZE], tmp[DSSIZE];

    q1 = vdivq_f32(a[0], b[0]);
    _bncneon_rds_mul_f(tmp, b, q1);
    _bncneon_rds_sub(r, a, tmp);
    
    s1 = vdivq_f32(r[0], b[0]);
    
    ret[0] = _bncneon_fquick_two_sum(q1, s1, &ret[1]);
}

#endif // __BNCNEON_DS_H
