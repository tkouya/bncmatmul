// ------------------------
// -------- NEON DD -------
// ------------------------
#ifndef __BNCNEON_DD_H
#define __BNCNEON_DD_H

// defined in rdd.h
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

// ret := val
static inline void _bncneon_rdd_set(float64x2_t ret[DDSIZE], float64x2_t val[DDSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
}

// ret := (double)val
static inline void _bncneon_rdd_set_d(float64x2_t ret[DDSIZE], float64x2_t val)
{
    ret[0] = val;
    ret[1] = vdupq_n_f64(0.0);
}

// ret := [val val]
static inline void _bncneon_rdd_set1_dd(float64x2_t ret[DDSIZE], double val[DDSIZE])
{
    ret[0] = vdupq_n_f64(val[0]);
    ret[1] = vdupq_n_f64(val[1]);
}

// ret := -val
static inline void _bncneon_rdd_neg(float64x2_t ret[DDSIZE], float64x2_t val[DDSIZE])
{
    ret[0] = vnegq_f64(val[0]);
    ret[1] = vnegq_f64(val[1]);
}

#if 0
// ret := ret2[neon_index] (extract single element)
// dd ret[0] := (ret2[0][0], ret2[1][0])
// dd ret[1] := (ret2[0][1], ret2[1][1])
static inline void _bncneon_get_dd_f64x2_i(ddfloat *ret, float64x2_t ret2[DDSIZE], const int neon_index)
{
    // ret->val[0] = vgetq_lane_f64(ret2[0], neon_index);
    // ret->val[1] = vgetq_lane_f64(ret2[1], neon_index);
    ret->val[0] = vdupd_laneq_f64(ret2[0], neon_index);
    ret->val[1] = vdupd_laneq_f64(ret2[1], neon_index);
    return;
}
#endif // 0

// ret := ret2[neon_index] (extract single element)
// dd ret[0] := (ret2[0][0], ret2[1][0])
// dd ret[1] := (ret2[0][1], ret2[1][1])
static inline void _bncneon_get_dd_f64x2_i_0(ddfloat *ret, float64x2_t ret2[DDSIZE]) // , const int neon_index)
{
    ret->val[0] = vgetq_lane_f64(ret2[0], 0); // neon_index);
    ret->val[1] = vgetq_lane_f64(ret2[1], 0); // neon_index);
    //ret->val[0] = vdupd_laneq_f64(ret2[0], neon_index);
    //ret->val[1] = vdupd_laneq_f64(ret2[1], neon_index);
    return;
}

// ret := ret2[neon_index] (extract single element)
// dd ret[0] := (ret2[0][0], ret2[1][0])
// dd ret[1] := (ret2[0][1], ret2[1][1])
static inline void _bncneon_get_dd_f64x2_i_1(ddfloat *ret, float64x2_t ret2[DDSIZE]) // , const int neon_index)
{
    ret->val[0] = vgetq_lane_f64(ret2[0], 1); // neon_index);
    ret->val[1] = vgetq_lane_f64(ret2[1], 1); // neon_index);
    //ret->val[0] = vdupd_laneq_f64(ret2[0], neon_index);
    //ret->val[1] = vdupd_laneq_f64(ret2[1], neon_index);
    return;
}

// ret := ret2[0] + ret2[1] (sum of 2-element vector)
static void _bncneon_rdd_sum128d(double ret[DDSIZE], float64x2_t ret2[DDSIZE])
{
    ddfloat ret2_i[2];

    // ret2_i := ret2
    //_bncneon_get_dd_f64x2_i(&ret2_i[0], ret2, 0);
    _bncneon_get_dd_f64x2_i_0(&ret2_i[0], ret2); //, 0);
    //_bncneon_get_dd_f64x2_i(&ret2_i[1], ret2, 1);
    _bncneon_get_dd_f64x2_i_1(&ret2_i[1], ret2); //, 1);

    rdd_set(ret, ret2_i[0].val);
    rdd_add(ret, ret, ret2_i[1].val);
}

// ret := |ret2[0]| + |ret2[1]| (absolute sum of 2-element vector)
static void _bncneon_rdd_abssum128d(double ret[DDSIZE], float64x2_t ret2[DDSIZE])
{
    ddfloat ret2_i[2];
    double tmp[DDSIZE];

    // ret2_i := ret2
    //_bncneon_get_dd_f64x2_i(&ret2_i[0], ret2, 0);
    //_bncneon_get_dd_f64x2_i(&ret2_i[1], ret2, 1);
    _bncneon_get_dd_f64x2_i_0(&ret2_i[0], ret2); //, 0);
    _bncneon_get_dd_f64x2_i_1(&ret2_i[1], ret2); //, 1);

    rdd_abs(tmp, ret2_i[0].val);
    rdd_set(ret, tmp);

    rdd_abs(tmp, ret2_i[1].val);
    rdd_add(ret, ret, tmp);
}

// ret := max(|ret2[0]|, |ret2[1]|) (absolute maximum of 2-element vector)
static void _bncneon_rdd_absmax128d(double ret[DDSIZE], float64x2_t ret2[DDSIZE])
{
    ddfloat ret2_i[2];
    double tmp[DDSIZE];

    // ret2_i := ret2
    //_bncneon_get_dd_f64x2_i(&ret2_i[0], ret2, 0);
    //_bncneon_get_dd_f64x2_i(&ret2_i[1], ret2, 1);
    _bncneon_get_dd_f64x2_i_0(&ret2_i[0], ret2); //, 0);
    _bncneon_get_dd_f64x2_i_1(&ret2_i[1], ret2); //, 1);

    rdd_abs(tmp, ret2_i[0].val);
    rdd_set(ret, tmp); // ret := |ret2_i[0]|

    rdd_abs(tmp, ret2_i[1].val);
    if(rdd_cmp(ret, tmp) < 0) // if(ret < |ret2_i[1]|)
        rdd_set(ret, tmp);    //   ret := |ret2_i[1]|
}

// ret := || ret2[0]^2 + ret2[1]^2 ||_2 (L2 norm of 2-element vector)
static void _bncneon_rdd_norm128d(double ret[DDSIZE], float64x2_t ret2[DDSIZE])
{
    ddfloat ret2_i[2];
    double tmp[DDSIZE];

    // ret2_i := ret2
    //_bncneon_get_dd_f64x2_i(&ret2_i[0], ret2, 0);
    //_bncneon_get_dd_f64x2_i(&ret2_i[1], ret2, 1);
    _bncneon_get_dd_f64x2_i_0(&ret2_i[0], ret2); //, 0);
    _bncneon_get_dd_f64x2_i_1(&ret2_i[1], ret2); //, 1);

    rdd_mul(tmp, ret2_i[0].val, ret2_i[0].val);
    rdd_set(ret, tmp);

    rdd_mul(tmp, ret2_i[1].val, ret2_i[1].val);
    rdd_add(ret, ret, tmp);

    rdd_sqrt(tmp, ret);
    rdd_set(ret, tmp);
}

// Helper functions for two_sum, two_diff, two_prod operations
// These would need to be implemented based on the original algorithms

#if 0
// Placeholder for two_sum - needs actual implementation
static inline float64x2_t _bncneon_dtwo_sum(float64x2_t a, float64x2_t b, float64x2_t *err)
{
    float64x2_t s = vaddq_f64(a, b);
    float64x2_t v = vsubq_f64(s, a);
    *err = vaddq_f64(vsubq_f64(a, vsubq_f64(s, v)), vsubq_f64(b, v));
    return s;
}

// Placeholder for two_diff - needs actual implementation
static inline float64x2_t _bncneon_dtwo_diff(float64x2_t a, float64x2_t b, float64x2_t *err)
{
    float64x2_t s = vsubq_f64(a, b);
    float64x2_t v = vsubq_f64(s, a);
    *err = vsubq_f64(vaddq_f64(a, vsubq_f64(s, v)), vaddq_f64(b, v));
    return s;
}

// Placeholder for two_prod - needs actual implementation
static inline float64x2_t _bncneon_dtwo_prod(float64x2_t a, float64x2_t b, float64x2_t *err)
{
    float64x2_t p = vmulq_f64(a, b);
    // This is a simplified version - actual implementation would need FMA
    *err = vsubq_f64(vmulq_f64(a, b), p);
    return p;
}

// Placeholder for quick_two_sum - needs actual implementation
static inline float64x2_t _bncneon_dquick_two_sum(float64x2_t a, float64x2_t b, float64x2_t *err)
{
    float64x2_t s = vaddq_f64(a, b);
    *err = vsubq_f64(b, vsubq_f64(s, a));
    return s;
}
#endif // 0

// Double-double arithmetic operations with NEON

#ifdef USE_DD_BF
    #define _bncneon_rdd_add    _bncneon_rdd_add_bf
    #define _bncneon_rdd_mul    _bncneon_rdd_mul_bf
#else // USE_DD_BF
    #define _bncneon_rdd_add    _bncneon_rdd_add_sloppy
    #define _bncneon_rdd_mul    _bncneon_rdd_mul_sloppy
#endif // USE_DD_BF

// ret := a + b (double-double addition)
static inline void _bncneon_rdd_add_sloppy(float64x2_t ret[DDSIZE], float64x2_t a[DDSIZE], float64x2_t b[DDSIZE])
{
    float64x2_t s, e;

    s = _bncneon_dtwo_sum(a[0], b[0], &e);
    e = vaddq_f64(e, vaddq_f64(a[1], b[1]));

    ret[0] = _bncneon_dquick_two_sum(s, e, &e);
    ret[1] = e;
}


// DD add (branch-free gates), NEON版
static inline void _bncneon_rdd_add_bf(float64x2_t ret[DDSIZE], float64x2_t a[DDSIZE], float64x2_t b[DDSIZE])
{
    float64x2_t g1, g2, g3, g4, g5, g6;
    float64x2_t g1e, g2e, g3e, g6e;

    g1 = _bncneon_dtwo_sum(a[0], b[0], &g1e);          // ゲート1
    g2 = _bncneon_dtwo_sum(a[1], b[1], &g2e);          // ゲート2
    g3 = _bncneon_dquick_two_sum(g1, g2, &g3e);
    g4 = vaddq_f64(g1e, g2e);
    g5 = vaddq_f64(g4, g3e);
    g6 = _bncneon_dquick_two_sum(g3, g5, &g6e);

    ret[0] = g6;
    ret[1] = g6e;
}

// ret := a + b (double-double + double addition)
static inline void _bncneon_rdd_add_d(float64x2_t ret[DDSIZE], float64x2_t a[DDSIZE], float64x2_t b)
{
    float64x2_t s1, s2;

    s1 = _bncneon_dtwo_sum(a[0], b, &s2);
    s2 = vaddq_f64(s2, a[1]);

    s1 = _bncneon_dquick_two_sum(s1, s2, &s2);
    ret[0] = s1;
    ret[1] = s2;
}

// ret := a - b (double-double subtraction)
static inline void _bncneon_rdd_sub(float64x2_t ret[DDSIZE], float64x2_t a[DDSIZE], float64x2_t b[DDSIZE])
{
    float64x2_t s1, s2, t1, t2;

    s1 = _bncneon_dtwo_diff(a[0], b[0], &s2);
    t1 = _bncneon_dtwo_diff(a[1], b[1], &t2);

    s2 = vaddq_f64(s2, t1);
    s1 = _bncneon_dquick_two_sum(s1, s2, &s2);
    s2 = vaddq_f64(s2, t2);
    s1 = _bncneon_dquick_two_sum(s1, s2, &s2);
    ret[0] = s1;
    ret[1] = s2;
}

// ret := a - b (double-double - double subtraction)
static inline void _bncneon_rdd_sub_d(float64x2_t ret[DDSIZE], float64x2_t a[DDSIZE], float64x2_t b)
{
    float64x2_t s1, s2;
    s1 = _bncneon_dtwo_diff(a[0], b, &s2);
    s2 = vaddq_f64(s2, a[1]);
    s1 = _bncneon_dquick_two_sum(s1, s2, &s2);
    ret[0] = s1;
    ret[1] = s2;
}

// ret := a * b (double-double multiplication)
static inline void _bncneon_rdd_mul_sloppy(float64x2_t ret[DDSIZE], float64x2_t a[DDSIZE], float64x2_t b[DDSIZE])
{
    float64x2_t p1, p2;

    p1 = _bncneon_dtwo_prod(a[0], b[0], &p2);

    p2 = vaddq_f64(p2, 
        vaddq_f64(
            vmulq_f64(a[0], b[1]),
            vmulq_f64(a[1], b[0])
        )
    );

    ret[0] = _bncneon_dquick_two_sum(p1, p2, &p2);
    ret[1] = p2;
}

// --- DD mul (branch-free gates), NEON版 --------------------------------
static inline void _bncneon_rdd_mul_bf(float64x2_t ret[DDSIZE],
                                      const float64x2_t a[DDSIZE],
                                      const float64x2_t b[DDSIZE])
{
    float64x2_t p00, p01, p10;
    float64x2_t pe00;
    float64x2_t g1, g2, g3, ge3;

    p00 = _bncneon_dtwo_prod(a[0], b[0], &pe00);   // [cite: 250, 307]
    p01 = vmulq_f64(a[0], b[1]);                  // [cite: 250, 307]
    p10 = vmulq_f64(a[1], b[0]);                  // [cite: 250, 307]

    g1 = vaddq_f64(p01, p10);
    g2 = vaddq_f64(pe00, g1);
    g3 = _bncneon_dquick_two_sum(p00, g2, &ge3);

    ret[0] = g3;   // [cite: 307]
    ret[1] = ge3;  // 全誤差の統合 [cite: 307]
}

// ret := a * b (double-double * double multiplication)
static inline void _bncneon_rdd_mul_d(float64x2_t ret[DDSIZE], float64x2_t a[DDSIZE], float64x2_t b)
{
    float64x2_t p1, p2;

    p1 = _bncneon_dtwo_prod(a[0], b, &p2);

    p2 = vaddq_f64(p2, 
        vmulq_f64(a[1], b)
    );

    ret[0] = _bncneon_dquick_two_sum(p1, p2, &p2);
    ret[1] = p2;
}

// ret := a / b (double-double division, sloppy version)
static inline void _bncneon_rdd_div(float64x2_t ret[DDSIZE], float64x2_t a[DDSIZE], float64x2_t b[DDSIZE])
{
    float64x2_t s1, s2;
    float64x2_t q1, q2;
    float64x2_t r[DDSIZE];

    q1 = vdivq_f64(a[0], b[0]);  /* approximate quotient */

    /* compute  this - q1 * dd */
    _bncneon_rdd_mul_d(r, b, q1);

    s1 = _bncneon_dtwo_diff(a[0], r[0], &s2);
    s2 = vsubq_f64(s2, r[1]);
    s2 = vaddq_f64(s2, a[1]);

    /* get next approximation */
    q2 = vdivq_f64(
        vaddq_f64(s1, s2),
        b[0]
    );

    /* renormalize */
    ret[0] = _bncneon_dquick_two_sum(q1, q2, &ret[1]);
}

// ret := a / b (double-double / double division)
static inline void _bncneon_rdd_div_d(float64x2_t ret[DDSIZE], float64x2_t a[DDSIZE], float64x2_t b)
{
    float64x2_t q1, q2;
    float64x2_t p1, p2;
    float64x2_t s, e;
    float64x2_t r[DDSIZE];

    q1 = vdivq_f64(a[0], b);   /* approximate quotient. */

    /* Compute  this - q1 * d */
    p1 = _bncneon_dtwo_prod(q1, b, &p2);
    s = _bncneon_dtwo_diff(a[0], p1, &e);
    e = vaddq_f64(e, a[1]);
    e = vsubq_f64(e, p2);
    /* get next approximation. */
    q2 = vdivq_f64(vaddq_f64(s, e), b);

    /* renormalize */
    r[0] = _bncneon_dquick_two_sum(q1, q2, &r[1]);

    ret[0] = r[0];
    ret[1] = r[1];
}

// ret := |a| (double-double absolute value)
static inline void _bncneon_rdd_abs(float64x2_t ret[DDSIZE], float64x2_t a[DDSIZE])
{
    // Use vector comparison to create mask for negative values
    uint64x2_t mask = vcltq_f64(a[0], vdupq_n_f64(0.0));
    
    // Apply conditional negation based on sign of a[0]
    ret[0] = vbslq_f64(mask, vnegq_f64(a[0]), a[0]);
    ret[1] = vbslq_f64(mask, vnegq_f64(a[1]), a[1]);
}

/* ================================================================
 *  ARM NEON implementation  (_bncneon_ prefix)
 *  Processes 2 DD operations per call
 *  FMA: vfmaq_f64(c, a, b) = a*b + c  (accumulator is 1st arg)
 * ================================================================ */

/* ----------------------------------------------------------------
 * _bncneon_rdd_fma_d_d
 *   ret[DDSIZE] := a * b + c   (scalar x scalar -> DD) x 2
 *   a, b, c : scalar, passed by value
 *
 *   vfmaq_f64(c, a, b) = a*b + c
 * ---------------------------------------------------------------- */
static inline void _bncneon_rdd_fma_d_d(
        float64x2_t ret[DDSIZE],
        float64x2_t a,
        float64x2_t b,
        float64x2_t c)
{
    ret[0] = vfmaq_f64(c, a, b);
    ret[1] = vfmaq_f64(vsubq_f64(c, ret[0]), a, b);
}

/* ----------------------------------------------------------------
 * _bncneon_rdd_fma_d_dd
 *   ret[DDSIZE] := a * b + c[DDSIZE]  (scalar x scalar + DD -> DD) x 2
 *   a, b : scalar, passed by value
 * ---------------------------------------------------------------- */
static inline void _bncneon_rdd_fma_d_dd(
        float64x2_t       ret[DDSIZE],
        float64x2_t       a,
        float64x2_t       b,
        const float64x2_t c[DDSIZE])
{
    float64x2_t hi = vfmaq_f64(c[0], a, b);
    float64x2_t t  = vsubq_f64(c[0], hi);
    float64x2_t e  = vfmaq_f64(t, a, b);
    float64x2_t f  = vaddq_f64(e, c[1]);
    ret[0] = _bncneon_dquick_two_sum(hi, f, &(ret[1]));
}

/* ----------------------------------------------------------------
 * _bncneon_rdd_fma_dm_dd
 *   ret[DDSIZE] := a * b[DDSIZE] + c[DDSIZE]  (scalar x DD + DD -> DD) x 2
 *   a : scalar, passed by value
 * ---------------------------------------------------------------- */
static inline void _bncneon_rdd_fma_dm_dd(
        float64x2_t       ret[DDSIZE],
        float64x2_t       a,
        const float64x2_t b[DDSIZE],
        const float64x2_t c[DDSIZE])
{
    float64x2_t hi = vfmaq_f64(c[0], a, b[0]);
    float64x2_t t  = vsubq_f64(c[0], hi);
    float64x2_t e  = vfmaq_f64(t, a, b[0]);
    float64x2_t f  = vaddq_f64(e, c[1]);
    float64x2_t lo = vfmaq_f64(f, a, b[1]);
    ret[0] = _bncneon_dquick_two_sum(hi, lo, &(ret[1]));
}

/* ----------------------------------------------------------------
 * _bncneon_rdd_fma
 *   ret[DDSIZE] := a[DDSIZE] * b[DDSIZE] + c[DDSIZE]  (DD x DD + DD -> DD) x 2
 * ---------------------------------------------------------------- */
static inline void _bncneon_rdd_fma(
        float64x2_t       ret[DDSIZE],
        const float64x2_t a[DDSIZE],
        const float64x2_t b[DDSIZE],
        const float64x2_t c[DDSIZE])
{
    float64x2_t d0 = vfmaq_f64(c[0], a[0], b[0]);
    float64x2_t t  = vsubq_f64(c[0], d0);
    float64x2_t e  = vfmaq_f64(t,    a[0], b[0]);
    float64x2_t f  = vaddq_f64(e, c[1]);
    float64x2_t g  = vfmaq_f64(f,    a[0], b[1]);
    float64x2_t d1 = vfmaq_f64(g,    a[1], b[0]);
    ret[0] = _bncneon_dquick_two_sum(d0, d1, &(ret[1]));
}

#ifdef __BNC_DDLINEAR_H__
//#include "ddv_addmul.c"
#endif //__BNC_DDLINEAR_H__

#endif // ifndef __BNCNEON_DD_H