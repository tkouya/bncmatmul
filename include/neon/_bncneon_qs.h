// ------------------------
// -------- QS NEON -------
// ------------------------
// Auto-generated from _bncneon_qd.h (quadruple-double) by mechanical
// translation: float64x2_t -> float32x4_t, double -> float, f64 -> f32,
// rqd_* (scalar) -> rqs_*, qdfloat -> qsfloat, QDSIZE -> QSSIZE,
// rtd_* -> rts_*, TDSIZE -> TSSIZE,
// renorm -> frenorm, renorm4 -> frenorm4,
// _bncneon_dtwo_* -> _bncneon_ftwo_*.
// Operates on float32x4_t (4 single-precision lanes per call).
#ifndef __BNCNEON_QS_H
#define __BNCNEON_QS_H

#include "_bncneon_ts.h"

#ifndef QSSIZE
    #define QSSIZE 4
#endif // QSSIZE

// ret := 0
static inline void _bncneon_set0_qs(float32x4_t ret[QSSIZE])
{
    ret[0] = vdupq_n_f32(0.0f);
    ret[1] = vdupq_n_f32(0.0f);
    ret[2] = vdupq_n_f32(0.0f);
    ret[3] = vdupq_n_f32(0.0f);
}
#define _bncneon_rqs_set0(ret) _bncneon_set0_qs((ret))

// ret := val
static inline void _bncneon_rqs_set(float32x4_t ret[QSSIZE], float32x4_t val[QSSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = val[2];
    ret[3] = val[3];
}

// ret := (float)val
static inline void _bncneon_rqs_set_f(float32x4_t ret[QSSIZE], float32x4_t val)
{
    ret[0] = val;
    ret[1] = vdupq_n_f32(0.0f);
    ret[2] = vdupq_n_f32(0.0f);
    ret[3] = vdupq_n_f32(0.0f);
}

// ret := (DS)val
static inline void _bncneon_rqs_set_ds(float32x4_t ret[QSSIZE], float32x4_t val[DSSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = vdupq_n_f32(0.0f);
    ret[3] = vdupq_n_f32(0.0f);
}

// ret := (TS)val
static inline void _bncneon_rqs_set_ts(float32x4_t ret[QSSIZE], float32x4_t val[TSSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = val[2];
    ret[3] = vdupq_n_f32(0.0f);
}

// ret := [(QS)val ...]
static inline void _bncneon_rqs_set1_qs(float32x4_t ret[QSSIZE], float val[QSSIZE])
{
    ret[0] = vdupq_n_f32(val[0]);
    ret[1] = vdupq_n_f32(val[1]);
    ret[2] = vdupq_n_f32(val[2]);
    ret[3] = vdupq_n_f32(val[3]);
}

// Lane extraction helpers (4 lanes per float32x4_t)
static inline void _bncneon_get_qs_f32x4_i_0(qsfloat *ret, float32x4_t ret4[QSSIZE])
{
    ret->val[0] = vgetq_lane_f32(ret4[0], 0);
    ret->val[1] = vgetq_lane_f32(ret4[1], 0);
    ret->val[2] = vgetq_lane_f32(ret4[2], 0);
    ret->val[3] = vgetq_lane_f32(ret4[3], 0);
}

static inline void _bncneon_get_qs_f32x4_i_1(qsfloat *ret, float32x4_t ret4[QSSIZE])
{
    ret->val[0] = vgetq_lane_f32(ret4[0], 1);
    ret->val[1] = vgetq_lane_f32(ret4[1], 1);
    ret->val[2] = vgetq_lane_f32(ret4[2], 1);
    ret->val[3] = vgetq_lane_f32(ret4[3], 1);
}

static inline void _bncneon_get_qs_f32x4_i_2(qsfloat *ret, float32x4_t ret4[QSSIZE])
{
    ret->val[0] = vgetq_lane_f32(ret4[0], 2);
    ret->val[1] = vgetq_lane_f32(ret4[1], 2);
    ret->val[2] = vgetq_lane_f32(ret4[2], 2);
    ret->val[3] = vgetq_lane_f32(ret4[3], 2);
}

static inline void _bncneon_get_qs_f32x4_i_3(qsfloat *ret, float32x4_t ret4[QSSIZE])
{
    ret->val[0] = vgetq_lane_f32(ret4[0], 3);
    ret->val[1] = vgetq_lane_f32(ret4[1], 3);
    ret->val[2] = vgetq_lane_f32(ret4[2], 3);
    ret->val[3] = vgetq_lane_f32(ret4[3], 3);
}

// ret := sum_{i=0..3} ret4[i] (horizontal sum across 4 lanes)
// Macro form (avoids the GCC OpenMP outline + static-symbol pruning issue).
#define _bncneon_rqs_sum128f(_ret, _ret4) do {                          \
    float _bncn_l0_[QSSIZE] = {                                         \
        vgetq_lane_f32((_ret4)[0], 0),                                  \
        vgetq_lane_f32((_ret4)[1], 0),                                  \
        vgetq_lane_f32((_ret4)[2], 0),                                  \
        vgetq_lane_f32((_ret4)[3], 0)                                   \
    };                                                                  \
    float _bncn_l1_[QSSIZE] = {                                         \
        vgetq_lane_f32((_ret4)[0], 1),                                  \
        vgetq_lane_f32((_ret4)[1], 1),                                  \
        vgetq_lane_f32((_ret4)[2], 1),                                  \
        vgetq_lane_f32((_ret4)[3], 1)                                   \
    };                                                                  \
    float _bncn_l2_[QSSIZE] = {                                         \
        vgetq_lane_f32((_ret4)[0], 2),                                  \
        vgetq_lane_f32((_ret4)[1], 2),                                  \
        vgetq_lane_f32((_ret4)[2], 2),                                  \
        vgetq_lane_f32((_ret4)[3], 2)                                   \
    };                                                                  \
    float _bncn_l3_[QSSIZE] = {                                         \
        vgetq_lane_f32((_ret4)[0], 3),                                  \
        vgetq_lane_f32((_ret4)[1], 3),                                  \
        vgetq_lane_f32((_ret4)[2], 3),                                  \
        vgetq_lane_f32((_ret4)[3], 3)                                   \
    };                                                                  \
    rqs_set((_ret), _bncn_l0_);                                         \
    rqs_add((_ret), (_ret), _bncn_l1_);                                 \
    rqs_add((_ret), (_ret), _bncn_l2_);                                 \
    rqs_add((_ret), (_ret), _bncn_l3_);                                 \
} while (0)

// abs
static inline void _bncneon_rqs_abs(float32x4_t ret[QSSIZE], float32x4_t a[QSSIZE])
{
    ret[0] = vabsq_f32(a[0]);
    ret[1] = vabsq_f32(a[1]);
    ret[2] = vabsq_f32(a[2]);
    ret[3] = vabsq_f32(a[3]);
}

// ret := sum_{i=0..3} |ret4[i]|
static __attribute__((used)) void _bncneon_rqs_abssum128f(float ret[QSSIZE], float32x4_t ret4[QSSIZE])
{
    qsfloat lane[4];
    float tmp[QSSIZE];

    _bncneon_get_qs_f32x4_i_0(&lane[0], ret4);
    _bncneon_get_qs_f32x4_i_1(&lane[1], ret4);
    _bncneon_get_qs_f32x4_i_2(&lane[2], ret4);
    _bncneon_get_qs_f32x4_i_3(&lane[3], ret4);

    rqs_abs(tmp, lane[0].val); rqs_set(ret, tmp);
    rqs_abs(tmp, lane[1].val); rqs_add(ret, ret, tmp);
    rqs_abs(tmp, lane[2].val); rqs_add(ret, ret, tmp);
    rqs_abs(tmp, lane[3].val); rqs_add(ret, ret, tmp);
}

// ret := max_{i=0..3} |ret4[i]|
static __attribute__((used)) void _bncneon_rqs_absmax128f(float ret[QSSIZE], float32x4_t ret4[QSSIZE])
{
    qsfloat lane[4];
    float tmp[QSSIZE];
    int i;

    _bncneon_get_qs_f32x4_i_0(&lane[0], ret4);
    _bncneon_get_qs_f32x4_i_1(&lane[1], ret4);
    _bncneon_get_qs_f32x4_i_2(&lane[2], ret4);
    _bncneon_get_qs_f32x4_i_3(&lane[3], ret4);

    rqs_abs(tmp, lane[0].val); rqs_set(ret, tmp);
    for(i = 1; i < 4; i++)
    {
        rqs_abs(tmp, lane[i].val);
        if(rqs_cmp(ret, tmp) < 0)
            rqs_set(ret, tmp);
    }
}

// ret := sqrt(sum_{i=0..3} ret4[i]^2)
static __attribute__((used)) void _bncneon_rqs_norm128f(float ret[QSSIZE], float32x4_t ret4[QSSIZE])
{
    qsfloat lane[4];
    float tmp[QSSIZE];
    int i;

    _bncneon_get_qs_f32x4_i_0(&lane[0], ret4);
    _bncneon_get_qs_f32x4_i_1(&lane[1], ret4);
    _bncneon_get_qs_f32x4_i_2(&lane[2], ret4);
    _bncneon_get_qs_f32x4_i_3(&lane[3], ret4);

    rqs_mul(tmp, lane[0].val, lane[0].val);
    rqs_set(ret, tmp);
    for(i = 1; i < 4; i++)
    {
        rqs_mul(tmp, lane[i].val, lane[i].val);
        rqs_add(ret, ret, tmp);
    }
    rqs_sqrt(tmp, ret);
    rqs_set(ret, tmp);
}

// frenorm function for NEON (4 lanes)
static inline void _bncneon_frenorm(float32x4_t *c0, float32x4_t *c1, float32x4_t *c2, float32x4_t *c3)
{
    float __attribute ((aligned(16))) q[4][4];

    q[0][0] = vgetq_lane_f32(*c0, 0);
    q[0][1] = vgetq_lane_f32(*c1, 0);
    q[0][2] = vgetq_lane_f32(*c2, 0);
    q[0][3] = vgetq_lane_f32(*c3, 0);
    q[1][0] = vgetq_lane_f32(*c0, 1);
    q[1][1] = vgetq_lane_f32(*c1, 1);
    q[1][2] = vgetq_lane_f32(*c2, 1);
    q[1][3] = vgetq_lane_f32(*c3, 1);
    q[2][0] = vgetq_lane_f32(*c0, 2);
    q[2][1] = vgetq_lane_f32(*c1, 2);
    q[2][2] = vgetq_lane_f32(*c2, 2);
    q[2][3] = vgetq_lane_f32(*c3, 2);
    q[3][0] = vgetq_lane_f32(*c0, 3);
    q[3][1] = vgetq_lane_f32(*c1, 3);
    q[3][2] = vgetq_lane_f32(*c2, 3);
    q[3][3] = vgetq_lane_f32(*c3, 3);

    frenorm(&q[0][0], &q[0][1], &q[0][2], &q[0][3]);
    frenorm(&q[1][0], &q[1][1], &q[1][2], &q[1][3]);
    frenorm(&q[2][0], &q[2][1], &q[2][2], &q[2][3]);
    frenorm(&q[3][0], &q[3][1], &q[3][2], &q[3][3]);

    *c0 = vsetq_lane_f32(q[0][0], *c0, 0);
    *c1 = vsetq_lane_f32(q[0][1], *c1, 0);
    *c2 = vsetq_lane_f32(q[0][2], *c2, 0);
    *c3 = vsetq_lane_f32(q[0][3], *c3, 0);
    *c0 = vsetq_lane_f32(q[1][0], *c0, 1);
    *c1 = vsetq_lane_f32(q[1][1], *c1, 1);
    *c2 = vsetq_lane_f32(q[1][2], *c2, 1);
    *c3 = vsetq_lane_f32(q[1][3], *c3, 1);
    *c0 = vsetq_lane_f32(q[2][0], *c0, 2);
    *c1 = vsetq_lane_f32(q[2][1], *c1, 2);
    *c2 = vsetq_lane_f32(q[2][2], *c2, 2);
    *c3 = vsetq_lane_f32(q[2][3], *c3, 2);
    *c0 = vsetq_lane_f32(q[3][0], *c0, 3);
    *c1 = vsetq_lane_f32(q[3][1], *c1, 3);
    *c2 = vsetq_lane_f32(q[3][2], *c2, 3);
    *c3 = vsetq_lane_f32(q[3][3], *c3, 3);
}

// frenorm4 function for NEON (4 lanes, 5 limbs in)
static inline void _bncneon_frenorm4(float32x4_t *c0, float32x4_t *c1, float32x4_t *c2, float32x4_t *c3, float32x4_t *c4)
{
    float __attribute ((aligned(16))) q[4][5];

    q[0][0] = vgetq_lane_f32(*c0, 0);
    q[0][1] = vgetq_lane_f32(*c1, 0);
    q[0][2] = vgetq_lane_f32(*c2, 0);
    q[0][3] = vgetq_lane_f32(*c3, 0);
    q[0][4] = vgetq_lane_f32(*c4, 0);
    q[1][0] = vgetq_lane_f32(*c0, 1);
    q[1][1] = vgetq_lane_f32(*c1, 1);
    q[1][2] = vgetq_lane_f32(*c2, 1);
    q[1][3] = vgetq_lane_f32(*c3, 1);
    q[1][4] = vgetq_lane_f32(*c4, 1);
    q[2][0] = vgetq_lane_f32(*c0, 2);
    q[2][1] = vgetq_lane_f32(*c1, 2);
    q[2][2] = vgetq_lane_f32(*c2, 2);
    q[2][3] = vgetq_lane_f32(*c3, 2);
    q[2][4] = vgetq_lane_f32(*c4, 2);
    q[3][0] = vgetq_lane_f32(*c0, 3);
    q[3][1] = vgetq_lane_f32(*c1, 3);
    q[3][2] = vgetq_lane_f32(*c2, 3);
    q[3][3] = vgetq_lane_f32(*c3, 3);
    q[3][4] = vgetq_lane_f32(*c4, 3);

    frenorm4(&q[0][0], &q[0][1], &q[0][2], &q[0][3], &q[0][4]);
    frenorm4(&q[1][0], &q[1][1], &q[1][2], &q[1][3], &q[1][4]);
    frenorm4(&q[2][0], &q[2][1], &q[2][2], &q[2][3], &q[2][4]);
    frenorm4(&q[3][0], &q[3][1], &q[3][2], &q[3][3], &q[3][4]);

    *c0 = vsetq_lane_f32(q[0][0], *c0, 0);
    *c1 = vsetq_lane_f32(q[0][1], *c1, 0);
    *c2 = vsetq_lane_f32(q[0][2], *c2, 0);
    *c3 = vsetq_lane_f32(q[0][3], *c3, 0);
    *c4 = vsetq_lane_f32(q[0][4], *c4, 0);
    *c0 = vsetq_lane_f32(q[1][0], *c0, 1);
    *c1 = vsetq_lane_f32(q[1][1], *c1, 1);
    *c2 = vsetq_lane_f32(q[1][2], *c2, 1);
    *c3 = vsetq_lane_f32(q[1][3], *c3, 1);
    *c4 = vsetq_lane_f32(q[1][4], *c4, 1);
    *c0 = vsetq_lane_f32(q[2][0], *c0, 2);
    *c1 = vsetq_lane_f32(q[2][1], *c1, 2);
    *c2 = vsetq_lane_f32(q[2][2], *c2, 2);
    *c3 = vsetq_lane_f32(q[2][3], *c3, 2);
    *c4 = vsetq_lane_f32(q[2][4], *c4, 2);
    *c0 = vsetq_lane_f32(q[3][0], *c0, 3);
    *c1 = vsetq_lane_f32(q[3][1], *c1, 3);
    *c2 = vsetq_lane_f32(q[3][2], *c2, 3);
    *c3 = vsetq_lane_f32(q[3][3], *c3, 3);
    *c4 = vsetq_lane_f32(q[3][4], *c4, 3);
}

/********** Additions ************/
// three_sum for NEON (float)
static inline void _bncneon_fthree_sum(float32x4_t *a, float32x4_t *b, float32x4_t *c)
{
    float32x4_t t1, t2, t3;

    t1 = _bncneon_ftwo_sum(*a, *b, &t2);
    *a = _bncneon_ftwo_sum(*c, t1, &t3);
    *b = _bncneon_ftwo_sum(t2, t3, c);
}

// three_sum2 for NEON (float)
static inline void _bncneon_fthree_sum2(float32x4_t *a, float32x4_t *b, float32x4_t *c)
{
    float32x4_t t1, t2, t3;

    t1 = _bncneon_ftwo_sum(*a, *b, &t2);
    *a = _bncneon_ftwo_sum(*c, t1, &t3);
    *b = vaddq_f32(t2, t3);
}

#ifdef USE_QS_BF
    #define _bncneon_rqs_add _bncneon_rqs_add_bf
    #define _bncneon_rqs_mul _bncneon_rqs_mul_bf
#else // USE_QS_BF
    #define _bncneon_rqs_add _bncneon_rqs_add_sloppy
    #define _bncneon_rqs_mul _bncneon_rqs_mul_sloppy
#endif // USE_QS_BF

// QS addition for NEON
static inline void _bncneon_rqs_add_sloppy(float32x4_t ret[QSSIZE], float32x4_t a[QSSIZE], float32x4_t b[QSSIZE])
{
    float32x4_t s0, s1, s2, s3;
    float32x4_t t0, t1, t2, t3;
    float32x4_t v0, v1, v2, v3;
    float32x4_t u0, u1, u2, u3;
    float32x4_t w0, w1, w2, w3;

    s0 = vaddq_f32(a[0], b[0]);
    s1 = vaddq_f32(a[1], b[1]);
    s2 = vaddq_f32(a[2], b[2]);
    s3 = vaddq_f32(a[3], b[3]);

    v0 = vsubq_f32(s0, a[0]);
    v1 = vsubq_f32(s1, a[1]);
    v2 = vsubq_f32(s2, a[2]);
    v3 = vsubq_f32(s3, a[3]);

    u0 = vsubq_f32(s0, v0);
    u1 = vsubq_f32(s1, v1);
    u2 = vsubq_f32(s2, v2);
    u3 = vsubq_f32(s3, v3);

    w0 = vsubq_f32(a[0], u0);
    w1 = vsubq_f32(a[1], u1);
    w2 = vsubq_f32(a[2], u2);
    w3 = vsubq_f32(a[3], u3);

    u0 = vsubq_f32(b[0], v0);
    u1 = vsubq_f32(b[1], v1);
    u2 = vsubq_f32(b[2], v2);
    u3 = vsubq_f32(b[3], v3);

    t0 = vaddq_f32(w0, u0);
    t1 = vaddq_f32(w1, u1);
    t2 = vaddq_f32(w2, u2);
    t3 = vaddq_f32(w3, u3);

    s1 = _bncneon_ftwo_sum(s1, t0, &t0);
    _bncneon_fthree_sum(&s2, &t0, &t1);
    _bncneon_fthree_sum2(&s3, &t0, &t2);
    t0 = vaddq_f32(vaddq_f32(t0, t1), t3);

    /* renormalize */
    _bncneon_frenorm4(&s0, &s1, &s2, &s3, &t0);

    ret[0] = s0;
    ret[1] = s1;
    ret[2] = s2;
    ret[3] = s3;
}

// Branch-free algorithm
static inline void _bncneon_rqs_add_bf(float32x4_t ret[QSSIZE],
                                      const float32x4_t a[QSSIZE],
                                      const float32x4_t b[QSSIZE])
{
    float32x4_t a0 , b0 , c0 , d0 , e0, f0, g0, h0;
    float32x4_t a1 , b1 , c1 , d1 , e1, f1, g1, h1;
    float32x4_t a2 , b2 , c2 , d2 , e2, f2, g2;
    float32x4_t b3 , c3 , d3 , e3, f3, g3, h3;
    float32x4_t a3;
    float32x4_t a4 , c4 , d4 , e4, f4;
    float32x4_t b5 , d5 , e5;
    float32x4_t b6 , c6 , d6 , e6;
    float32x4_t a7 , b7 , c7 , d7;
    float32x4_t b8 , c8 , e8;
    float32x4_t d9;
    float32x4_t a10, b10, c10, d10;
    float32x4_t b11, c11;
    float32x4_t c12, d12;

    a0 = a[0];
    b0 = b[0];
    c0 = a[1];
    d0 = b[1];
    e0 = a[2];
    f0 = b[2];
    g0 = a[3];
    h0 = b[3];

    a1 = _bncneon_ftwo_sum(a0, b0, &b1);
    c1 = _bncneon_ftwo_sum(c0, d0, &d1);
    e1 = _bncneon_ftwo_sum(e0, f0, &f1);
    g1 = _bncneon_ftwo_sum(g0, h0, &h1);

    a2 = _bncneon_fquick_two_sum(a1, c1, &c2);
    b2 = vaddq_f32(b1, h1);
    d2 = _bncneon_ftwo_sum(d1, e1, &e2);
    f2 = _bncneon_ftwo_sum(f1, g1, &g2);

    b3 = _bncneon_ftwo_sum(b2, g2, &g3);
    c3 = _bncneon_fquick_two_sum(c2, d2, &d3);
    e3 = _bncneon_ftwo_sum(e2, f2, &f3);

    a4 = _bncneon_fquick_two_sum(a2, c3, &c4);
    d4 = _bncneon_fquick_two_sum(d3, e3, &e4);

    b5 = _bncneon_ftwo_sum(b3, d4, &d5);
    e5 = vaddq_f32(e4, f3);

    b6 = _bncneon_ftwo_sum(b5, c4, &c6);
    d6 = _bncneon_ftwo_sum(d5, e5, &e6);

    a7 = _bncneon_fquick_two_sum(a4, b6, &b7);
    c7 = _bncneon_fquick_two_sum(c6, d6, &d7);

    e8 = vaddq_f32(e6, g3);
    b8 = _bncneon_fquick_two_sum(b7, c7, &c8);

    d9 = vaddq_f32(d7, e8);
    a10 = _bncneon_fquick_two_sum(a7, b8, &b10);
    c10 = _bncneon_fquick_two_sum(c8, d9, &d10);

    b11 = _bncneon_fquick_two_sum(b10, c10, &c11);
    c12 = _bncneon_fquick_two_sum(c11, d10, &d12);

    ret[0] = a10;
    ret[1] = b11;
    ret[2] = c12;
    ret[3] = d12;

    (void)a3; (void)h3;
}

// TS Q addition — real implementation forward-declared in _bncneon_ts.h.
// Uses _bncneon_fthree_sum and _bncneon_frenorm defined in this file.
// Mirrors _bncneon_rtd_addq in _bncneon_qd.h (the renorm-based TD add).
static inline void _bncneon_rts_addq(float32x4_t ret[TSSIZE], float32x4_t a[TSSIZE], float32x4_t b[TSSIZE])
{
    float32x4_t s0, s1, s2;
    float32x4_t t0, t1, t2;
    float32x4_t v0, v1, v2;
    float32x4_t u0, u1, u2;
    float32x4_t w0, w1, w2;

    s0 = vaddq_f32(a[0], b[0]);
    s1 = vaddq_f32(a[1], b[1]);
    s2 = vaddq_f32(a[2], b[2]);

    v0 = vsubq_f32(s0, a[0]);
    v1 = vsubq_f32(s1, a[1]);
    v2 = vsubq_f32(s2, a[2]);

    u0 = vsubq_f32(s0, v0);
    u1 = vsubq_f32(s1, v1);
    u2 = vsubq_f32(s2, v2);

    w0 = vsubq_f32(a[0], u0);
    w1 = vsubq_f32(a[1], u1);
    w2 = vsubq_f32(a[2], u2);

    u0 = vsubq_f32(b[0], v0);
    u1 = vsubq_f32(b[1], v1);
    u2 = vsubq_f32(b[2], v2);

    t0 = vaddq_f32(w0, u0);
    t1 = vaddq_f32(w1, u1);
    t2 = vaddq_f32(w2, u2);

    s1 = _bncneon_ftwo_sum(s1, t0, &t0);
    _bncneon_fthree_sum(&s2, &t0, &t1);
    t0 = vaddq_f32(vaddq_f32(t0, t1), t2);

    _bncneon_frenorm(&s0, &s1, &s2, &t0);

    ret[0] = s0;
    ret[1] = s1;
    ret[2] = s2;
}

// TS Q multiplication — real implementation forward-declared in _bncneon_ts.h.
// Uses _bncneon_fthree_sum/_bncneon_fthree_sum2/_bncneon_frenorm defined in this file.
// Mirrors _bncneon_rtd_mulq in _bncneon_qd.h (the renorm-based TD mul).
static inline void _bncneon_rts_mulq(float32x4_t ret[TSSIZE], float32x4_t a[TSSIZE], float32x4_t b[TSSIZE])
{
    float32x4_t p0, p1, p2, p3, p4, p5;
    float32x4_t q0, q1, q2, q3, q4, q5;
    float32x4_t t0, t1;
    float32x4_t s0, s1;

    p0 = _bncneon_ftwo_prod(a[0], b[0], &q0);

    p1 = _bncneon_ftwo_prod(a[0], b[1], &q1);
    p2 = _bncneon_ftwo_prod(a[1], b[0], &q2);

    p3 = _bncneon_ftwo_prod(a[0], b[2], &q3);
    p4 = _bncneon_ftwo_prod(a[1], b[1], &q4);
    p5 = _bncneon_ftwo_prod(a[2], b[0], &q5);

    _bncneon_fthree_sum(&p1, &p2, &q0);

    _bncneon_fthree_sum2(&p2, &q1, &q2);
    _bncneon_fthree_sum2(&p3, &p4, &p5);

    s0 = _bncneon_ftwo_sum(p2, p3, &t0);
    s1 = _bncneon_ftwo_sum(q1, p4, &t1);
    s1 = _bncneon_ftwo_sum(s1, t0, &t0);

    s1 = vaddq_f32(s1, vmulq_f32(a[1], b[2]));
    s1 = vaddq_f32(s1, vmulq_f32(a[2], b[1]));
    s1 = vaddq_f32(s1, q0);
    s1 = vaddq_f32(s1, q3);
    s1 = vaddq_f32(s1, q4);

    _bncneon_frenorm(&p0, &p1, &s0, &s1);

    ret[0] = p0;
    ret[1] = p1;
    ret[2] = s0;
}

// QS multiplication
static inline void _bncneon_rqs_mul_sloppy(float32x4_t ret[QSSIZE], float32x4_t a[QSSIZE], float32x4_t b[QSSIZE])
{
    float32x4_t p0, p1, p2, p3, p4, p5;
    float32x4_t q0, q1, q2, q3, q4, q5;
    float32x4_t t0, t1;
    float32x4_t s0, s1, s2;

    p0 = _bncneon_ftwo_prod(a[0], b[0], &q0);

    p1 = _bncneon_ftwo_prod(a[0], b[1], &q1);
    p2 = _bncneon_ftwo_prod(a[1], b[0], &q2);

    p3 = _bncneon_ftwo_prod(a[0], b[2], &q3);
    p4 = _bncneon_ftwo_prod(a[1], b[1], &q4);
    p5 = _bncneon_ftwo_prod(a[2], b[0], &q5);

    /* Start Accumulation */
    _bncneon_fthree_sum(&p1, &p2, &q0);

    /* Six-Three Sum of p2, q1, q2, p3, p4, p5. */
    _bncneon_fthree_sum(&p2, &q1, &q2);
    _bncneon_fthree_sum(&p3, &p4, &p5);

    /* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
    s0 = _bncneon_ftwo_sum(p2, p3, &t0);
    s1 = _bncneon_ftwo_sum(q1, p4, &t1);
    s2 = vaddq_f32(q2, p5);
    s1 = _bncneon_ftwo_sum(s1, t0, &t0);
    s2 = vaddq_f32(s2, vaddq_f32(t0, t1));

    /* O(eps^3) order terms */
    s1 = vaddq_f32(s1, vmulq_f32(a[0], b[3]));
    s1 = vaddq_f32(s1, vmulq_f32(a[1], b[2]));
    s1 = vaddq_f32(s1, vmulq_f32(a[2], b[1]));
    s1 = vaddq_f32(s1, vmulq_f32(a[3], b[0]));
    s1 = vaddq_f32(s1, q0);
    s1 = vaddq_f32(s1, q3);
    s1 = vaddq_f32(s1, q4);
    s1 = vaddq_f32(s1, q5);

    _bncneon_frenorm4(&p0, &p1, &s0, &s1, &s2);

    ret[0] = p0;
    ret[1] = p1;
    ret[2] = s0;
    ret[3] = s1;
}

// Branch-free algorithm for QS mul
static inline void _bncneon_rqs_mul_bf(float32x4_t ret[QSSIZE],
                                      const float32x4_t a[QSSIZE],
                                      const float32x4_t b[QSSIZE])
{
    float32x4_t a0, b0, c0, d0, e0, f0, g0, h0, i0, j0, k0, l0, m0, n0, o0, p0;
    float32x4_t c1, d1, e1, f1, g1, i1, j1, m1, n1;
    float32x4_t b2, c2, e2, f2, i2, h2, m2;
    float32x4_t a3, b3, c3, d3, e3, f3, g3, h3;
    float32x4_t c4, d4, e4, f4;
    float32x4_t d5;
    float32x4_t c6, d6;
    float32x4_t b7, c7, d7;
    float32x4_t a8, b8, c8, d8;
    float32x4_t b9, c9;
    float32x4_t c10, d10;

    a0 = _bncneon_ftwo_prod(a[0], b[0], &b0);
    c0 = _bncneon_ftwo_prod(a[0], b[1], &e0);
    d0 = _bncneon_ftwo_prod(a[1], b[0], &f0);
    g0 = _bncneon_ftwo_prod(a[0], b[2], &j0);
    h0 = _bncneon_ftwo_prod(a[1], b[1], &k0);
    i0 = _bncneon_ftwo_prod(a[2], b[0], &l0);

    m0 = vmulq_f32(a[0], b[3]);
    n0 = vmulq_f32(a[1], b[2]);
    o0 = vmulq_f32(a[2], b[1]);
    p0 = vmulq_f32(a[3], b[0]);

    c1 = _bncneon_ftwo_sum(c0, d0, &d1);
    e1 = _bncneon_ftwo_sum(e0, f0, &f1);
    g1 = _bncneon_ftwo_sum(g0, i0, &i1);

    j1 = vaddq_f32(j0, l0);
    m1 = vaddq_f32(m0, p0);
    n1 = vaddq_f32(n0, o0);

    b2 = _bncneon_ftwo_sum(b0, c1, &c2);
    e2 = _bncneon_ftwo_sum(e1, h0, &h2);

    f2 = vaddq_f32(f1, j1);
    i2 = vaddq_f32(i1, k0);
    m2 = vaddq_f32(m1, n1);

    a3 = _bncneon_fquick_two_sum(a0, b2, &b3);
    c3 = _bncneon_fquick_two_sum(c2, d1, &d3);
    e3 = _bncneon_ftwo_sum(e2, g1, &g3);

    f3 = vaddq_f32(f2, m2);
    h3 = vaddq_f32(h2, i2);

    c4 = _bncneon_ftwo_sum(c3, e3, &e4);
    d4 = vaddq_f32(d3, h3);
    f4 = vaddq_f32(f3, g3);

    d5 = vaddq_f32(d4, e4);
    c6 = _bncneon_ftwo_sum(c4, d5, &d6);

    b7 = _bncneon_ftwo_sum(b3, c6, &c7);
    d7 = vaddq_f32(d6, f4);

    a8 = _bncneon_fquick_two_sum(a3, b7, &b8);
    c8 = _bncneon_ftwo_sum(c7, d7, &d8);

    b9  = _bncneon_ftwo_sum(b8, c8, &c9);
    c10 = _bncneon_fquick_two_sum(c9, d8, &d10);

    ret[0] = a8;
    ret[1] = b9;
    ret[2] = c10;
    ret[3] = d10;
}


// QS negation
static inline void _bncneon_rqs_neg(float32x4_t c[QSSIZE], float32x4_t a[QSSIZE])
{
    c[0] = vnegq_f32(a[0]);
    c[1] = vnegq_f32(a[1]);
    c[2] = vnegq_f32(a[2]);
    c[3] = vnegq_f32(a[3]);
}

// QS subtraction
static inline void _bncneon_rqs_sub(float32x4_t c[QSSIZE], float32x4_t a[QSSIZE], float32x4_t b[QSSIZE])
{
    float32x4_t mb[QSSIZE];

    _bncneon_rqs_neg(mb, b);
    _bncneon_rqs_add(c, a, mb);
}

// QS + float addition
static inline void _bncneon_rqs_add_f(float32x4_t c[QSSIZE], const float32x4_t a[QSSIZE], float32x4_t b)
{
    float32x4_t e0, e1, s0, s1, s2, s3;

    s0 = _bncneon_ftwo_sum(a[0], b,  &e0);
    s1 = _bncneon_ftwo_sum(a[1], e0, &e1);
    s2 = _bncneon_ftwo_sum(a[2], e1, &e0);
    s3 = _bncneon_ftwo_sum(a[3], e0, &e1);

    _bncneon_frenorm4(&s0, &s1, &s2, &s3, &e1);

    c[0] = s0;
    c[1] = s1;
    c[2] = s2;
    c[3] = s3;
}

// TS + float addition (re-exposed)
static inline void _bncneon_rts_addq_f(float32x4_t c[TSSIZE], const float32x4_t a[TSSIZE], float32x4_t b)
{
    float32x4_t e0, e1, s0, s1, s2;

    s0 = _bncneon_ftwo_sum(a[0], b,  &e0);
    s1 = _bncneon_ftwo_sum(a[1], e0, &e1);
    s2 = _bncneon_ftwo_sum(a[2], e1, &e0);

    _bncneon_frenorm(&s0, &s1, &s2, &e0);

    c[0] = s0;
    c[1] = s1;
    c[2] = s2;
}

// QS * float multiplication
static inline void _bncneon_rqs_mul_f(float32x4_t c[QSSIZE], const float32x4_t a[QSSIZE], float32x4_t b)
{
    float32x4_t p0, p1, p2, p3;
    float32x4_t q0, q1, q2;
    float32x4_t s0, s1, s2, s3, s4;

    p0 = _bncneon_ftwo_prod(a[0], b, &q0);
    p1 = _bncneon_ftwo_prod(a[1], b, &q1);
    p2 = _bncneon_ftwo_prod(a[2], b, &q2);
    p3 = vmulq_f32(a[3], b);

    s0 = p0;

    s1 = _bncneon_ftwo_sum(q0, p1, &s2);

    _bncneon_fthree_sum(&s2, &q1, &p2);

    _bncneon_fthree_sum2(&q1, &q2, &p3);
    s3 = q1;

    s4 = vaddq_f32(q2, p2);

    _bncneon_frenorm4(&s0, &s1, &s2, &s3, &s4);

    c[0] = s0;
    c[1] = s1;
    c[2] = s2;
    c[3] = s3;
}

// TS * float multiplication (re-exposed)
static inline void _bncneon_rts_mulq_f(float32x4_t c[TSSIZE], const float32x4_t a[TSSIZE], float32x4_t b)
{
    float32x4_t p0, p1, p2;
    float32x4_t q0, q1, q2;
    float32x4_t s0, s1, s2, s4;

    p0 = _bncneon_ftwo_prod(a[0], b, &q0);
    p1 = _bncneon_ftwo_prod(a[1], b, &q1);
    p2 = _bncneon_ftwo_prod(a[2], b, &q2);

    s0 = p0;

    s1 = _bncneon_ftwo_sum(q0, p1, &s2);

    _bncneon_fthree_sum(&s2, &q1, &p2);

    s4 = vaddq_f32(q2, p2);

    _bncneon_frenorm(&s0, &s1, &s2, &s4);

    c[0] = s0;
    c[1] = s1;
    c[2] = s2;
}

// TS division (renorm-based)
static inline void _bncneon_rts_divq_renorm(float32x4_t c[TSSIZE], float32x4_t a[TSSIZE], float32x4_t b[TSSIZE])
{
    float32x4_t q0, q1, q2, q3;
    float32x4_t r[TSSIZE], tmp[TSSIZE];

    q0 = vdivq_f32(a[0], b[0]);

    _bncneon_rts_mul_f(tmp, q0, b);
    _bncneon_rts_subq(r, a, tmp);

    q1 = vdivq_f32(r[0], b[0]);

    _bncneon_rts_mulq_f(tmp, b, q1);
    _bncneon_rts_addq(r, r, tmp);

    q2 = vdivq_f32(r[0], b[0]);
    _bncneon_rts_mulq_f(tmp, b, q2);
    _bncneon_rts_addq(r, r, tmp);

    q3 = vdivq_f32(r[0], b[0]);

    _bncneon_frenorm(&q0, &q1, &q2, &q3);

    c[0] = q0;
    c[1] = q1;
    c[2] = q2;
}

// QS division
static inline void _bncneon_rqs_div(float32x4_t c[QSSIZE], float32x4_t a[QSSIZE], float32x4_t b[QSSIZE])
{
    float32x4_t q0, q1, q2, q3;
    float32x4_t r[QSSIZE], tmp[QSSIZE];

    q0 = vdivq_f32(a[0], b[0]);

    _bncneon_rqs_mul_f(tmp, b, q0);
    _bncneon_rqs_sub(r, a, tmp);

    q1 = vdivq_f32(r[0], b[0]);

    _bncneon_rqs_mul_f(tmp, b, q1);
    _bncneon_rqs_sub(r, r, tmp);

    q2 = vdivq_f32(r[0], b[0]);
    _bncneon_rqs_mul_f(tmp, b, q2);
    _bncneon_rqs_sub(r, r, tmp);

    q3 = vdivq_f32(r[0], b[0]);

    _bncneon_frenorm4(&q0, &q1, &q2, &q3, &tmp[0]);

    c[0] = q0;
    c[1] = q1;
    c[2] = q2;
    c[3] = q3;
}

#endif // ifndef __BNCNEON_QS_H
