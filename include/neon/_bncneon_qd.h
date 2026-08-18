// ------------------------
// -------- QD NEON -------
// ------------------------
#ifndef __BNCNEON_QD_H
#define __BNCNEON_QD_H

#ifndef QDSIZE
    #define QDSIZE 4
#endif // QDSIZE

//#if defined(__aarch64__) || defined(__arm64__)
//#include <arm_neon.h>

// ret := 0
static inline void _bncneon_set0_qd(float64x2_t ret[QDSIZE])
{
    ret[0] = vdupq_n_f64(0.0);
    ret[1] = vdupq_n_f64(0.0);
    ret[2] = vdupq_n_f64(0.0);
    ret[3] = vdupq_n_f64(0.0);
}
#define _bncneon_rqd_set0(ret) _bncneon_set0_qd((ret))

// ret := val
static inline void _bncneon_rqd_set(float64x2_t ret[QDSIZE], float64x2_t val[QDSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = val[2];
    ret[3] = val[3];
}

// ret := (double)val
static inline void _bncneon_rqd_set_d(float64x2_t ret[QDSIZE], float64x2_t val)
{
    ret[0] = val;
    ret[1] = vdupq_n_f64(0.0);
    ret[2] = vdupq_n_f64(0.0);
    ret[3] = vdupq_n_f64(0.0);
}

// ret := (DD)val
static inline void _bncneon_rqd_set_dd(float64x2_t ret[QDSIZE], float64x2_t val[DDSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = vdupq_n_f64(0.0);
    ret[3] = vdupq_n_f64(0.0);
}

// ret := (TD)val
static inline void _bncneon_rqd_set_td(float64x2_t ret[QDSIZE], float64x2_t val[TDSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = val[2];
    ret[3] = vdupq_n_f64(0.0);
}

// ret := [(QD)val (QD)val]
static inline void _bncneon_rqd_set1_qd(float64x2_t ret[QDSIZE], double val[QDSIZE])
{
    ret[0] = vdupq_n_f64(val[0]);
    ret[1] = vdupq_n_f64(val[1]);
    ret[2] = vdupq_n_f64(val[2]);
    ret[3] = vdupq_n_f64(val[3]);
}

#if 0
// ret := ret2[neon_index] (0 or 1)
static inline void _bncneon_get_qd_f64x2_i(qdfloat *ret, float64x2_t ret2[QDSIZE], int neon_index)
{
    ret->val[0] = vgetq_lane_f64(ret2[0], neon_index);
    ret->val[1] = vgetq_lane_f64(ret2[1], neon_index);
    ret->val[2] = vgetq_lane_f64(ret2[2], neon_index);
    ret->val[3] = vgetq_lane_f64(ret2[3], neon_index);

    return;
}
#endif // 0

// ret := ret2[neon_index] (0 or 1)
static inline void _bncneon_get_qd_f64x2_i_0(qdfloat *ret, float64x2_t ret2[QDSIZE]) // , int neon_index)
{
    ret->val[0] = vgetq_lane_f64(ret2[0], 0); // neon_index);
    ret->val[1] = vgetq_lane_f64(ret2[1], 0); // neon_index);
    ret->val[2] = vgetq_lane_f64(ret2[2], 0); // neon_index);
    ret->val[3] = vgetq_lane_f64(ret2[3], 0); // neon_index);

    return;
}
// ret := ret2[neon_index] (0 or 1)
static inline void _bncneon_get_qd_f64x2_i_1(qdfloat *ret, float64x2_t ret2[QDSIZE]) // , int neon_index)
{
    ret->val[0] = vgetq_lane_f64(ret2[0], 1); // neon_index);
    ret->val[1] = vgetq_lane_f64(ret2[1], 1); // neon_index);
    ret->val[2] = vgetq_lane_f64(ret2[2], 1); // neon_index);
    ret->val[3] = vgetq_lane_f64(ret2[3], 1); // neon_index);

    return;
}

// ret := neonval[0] + neonval[1]
static void _bncneon_rqd_sum128d(double ret[QDSIZE], float64x2_t ret2[QDSIZE])
{
    qdfloat ret2_i[2];

    // ret2_i := ret2
    //_bncneon_get_qd_f64x2_i(&ret2_i[0], ret2, 0);
    //_bncneon_get_qd_f64x2_i(&ret2_i[1], ret2, 1);
    _bncneon_get_qd_f64x2_i_0(&ret2_i[0], ret2); //, 0);
    _bncneon_get_qd_f64x2_i_1(&ret2_i[1], ret2); //, 1);

    rqd_set(ret, ret2_i[0].val);
    rqd_add(ret, ret, ret2_i[1].val);
}

// abs
static inline void _bncneon_rqd_abs(float64x2_t ret[QDSIZE], float64x2_t a[QDSIZE])
{
    ret[0] = vabsq_f64(a[0]);
    ret[1] = vabsq_f64(a[1]);
    ret[2] = vabsq_f64(a[2]);
    ret[3] = vabsq_f64(a[3]);
}

// ret := |ret2[0]| + |ret2[1]|
static void _bncneon_rqd_abssum128d(double ret[QDSIZE], float64x2_t ret2[QDSIZE])
{
    qdfloat ret2_i[2];
    double tmp[QDSIZE];

    // ret2_i := ret2
    //_bncneon_get_qd_f64x2_i(&ret2_i[0], ret2, 0);
    //_bncneon_get_qd_f64x2_i(&ret2_i[1], ret2, 1);
    _bncneon_get_qd_f64x2_i_0(&ret2_i[0], ret2); //, 0);
    _bncneon_get_qd_f64x2_i_1(&ret2_i[1], ret2); //, 1);

    rqd_abs(tmp, ret2_i[0].val);
    rqd_set(ret, tmp);

    rqd_abs(tmp, ret2_i[1].val);
    rqd_add(ret, ret, tmp);
}

// ret := max(|ret2[0]|, |ret2[1]|)
static void _bncneon_rqd_absmax128d(double ret[QDSIZE], float64x2_t ret2[QDSIZE])
{
    qdfloat ret2_i[2];
    double tmp[QDSIZE];

    // ret2_i := ret2
    //_bncneon_get_qd_f64x2_i(&ret2_i[0], ret2, 0);
    //_bncneon_get_qd_f64x2_i(&ret2_i[1], ret2, 1);
    _bncneon_get_qd_f64x2_i_0(&ret2_i[0], ret2); // , 0);
    _bncneon_get_qd_f64x2_i_1(&ret2_i[1], ret2); // , 1);

    rqd_abs(tmp, ret2_i[0].val);
    rqd_set(ret, tmp); // ret := |ret2_i[0]|

    rqd_abs(tmp, ret2_i[1].val);
    if(rqd_cmp(ret, tmp) < 0) // if(ret < |ret2_i[1]|)
        rqd_set(ret, tmp);    //   ret := |ret2_i[1]|
}

// ret := || ret2[0]^2 + ret2[1]^2 ||_2
static void _bncneon_rqd_norm128d(double ret[QDSIZE], float64x2_t ret2[QDSIZE])
{
    qdfloat ret2_i[2];
    double tmp[QDSIZE];

    // ret2_i := ret2
    //_bncneon_get_qd_f64x2_i(&ret2_i[0], ret2, 0);
    //_bncneon_get_qd_f64x2_i(&ret2_i[1], ret2, 1);
    _bncneon_get_qd_f64x2_i_0(&ret2_i[0], ret2); //, 0);
    _bncneon_get_qd_f64x2_i_1(&ret2_i[1], ret2); //, 1);

    rqd_mul(tmp, ret2_i[0].val, ret2_i[0].val);
    rqd_set(ret, tmp);

    rqd_mul(tmp, ret2_i[1].val, ret2_i[1].val);
    rqd_add(ret, ret, tmp);

    rqd_sqrt(tmp, ret);
    rqd_set(ret, tmp);
}

// renorm function for NEON
static inline void _bncneon_renorm(float64x2_t *c0, float64x2_t *c1, float64x2_t *c2, float64x2_t *c3)
{
    // New codes adapted for NEON
    double __attribute ((aligned(16))) q0[4], q1[4]; //, q2[2], q3[2];

    //vst1q_f64(q0, *c0);
    //vst1q_f64(q1, *c1);
    //vst1q_f64(q2, *c2);
    //vst1q_f64(q3, *c3);
    //q0[0] = (* c0)[0]; q0[1] = (* c1)[0]; q0[2] = (* c2)[0]; q0[3] = (* c3)[0];
    //q1[0] = (* c0)[1]; q1[1] = (* c1)[1]; q1[2] = (* c2)[1]; q1[3] = (* c3)[1];
    q0[0] = vgetq_lane_f64(*c0, 0);
    q0[1] = vgetq_lane_f64(*c1, 0);
    q0[2] = vgetq_lane_f64(*c2, 0);
    q0[3] = vgetq_lane_f64(*c3, 0);
    q1[0] = vgetq_lane_f64(*c0, 1);
    q1[1] = vgetq_lane_f64(*c1, 1);
    q1[2] = vgetq_lane_f64(*c2, 1);
    q1[3] = vgetq_lane_f64(*c3, 1);

    renorm(&q0[0], &q0[1], &q0[2], &q0[3]);
    renorm(&q1[0], &q1[1], &q1[2], &q1[3]);

    *c0 = vsetq_lane_f64(q0[0], *c0, 0);
    *c1 = vsetq_lane_f64(q0[1], *c1, 0);
    *c2 = vsetq_lane_f64(q0[2], *c2, 0);
    *c3 = vsetq_lane_f64(q0[3], *c3, 0);
    *c0 = vsetq_lane_f64(q1[0], *c0, 1);
    *c1 = vsetq_lane_f64(q1[1], *c1, 1);
    *c2 = vsetq_lane_f64(q1[2], *c2, 1);
    *c3 = vsetq_lane_f64(q1[3], *c3, 1);

    //(* c0)[0] = q0[0]; (* c1)[0] = q0[1]; (* c2)[0] = q0[2]; (* c3)[0] = q0[3];
    //(* c0)[1] = q1[0]; (* c1)[1] = q1[1]; (* c2)[1] = q1[2]; (* c3)[1] = q1[3];

}

// renorm4 function for NEON
static inline void _bncneon_renorm4(float64x2_t *c0, float64x2_t *c1, float64x2_t *c2, float64x2_t *c3, float64x2_t *c4)
{
    // New codes adapted for NEON
    double __attribute ((aligned(16))) q0[5], q1[5];

    // vst1q_f64(&q0[0], *c0);
    // vst1q_f64(&q0[2], *c2);
    // q0[4] = vgetq_lane_f64(*c4, 0);
    // 
    // vst1q_f64(&q1[0], *c1);
    // vst1q_f64(&q1[2], *c3);
    // q1[4] = vgetq_lane_f64(*c4, 1);

    //q0[0] = (* c0)[0]; q0[1] = (* c1)[0]; q0[2] = (* c2)[0]; q0[3] = (* c3)[0]; q0[4] = (* c4)[0];
    //q1[0] = (* c0)[1]; q1[1] = (* c1)[1]; q1[2] = (* c2)[1]; q1[3] = (* c3)[1]; q1[4] = (* c4)[1];
    q0[0] = vgetq_lane_f64(*c0, 0);
    q0[1] = vgetq_lane_f64(*c1, 0);
    q0[2] = vgetq_lane_f64(*c2, 0);
    q0[3] = vgetq_lane_f64(*c3, 0);
    q0[4] = vgetq_lane_f64(*c4, 0);
    q1[0] = vgetq_lane_f64(*c0, 1);
    q1[1] = vgetq_lane_f64(*c1, 1);
    q1[2] = vgetq_lane_f64(*c2, 1);
    q1[3] = vgetq_lane_f64(*c3, 1);
    q1[4] = vgetq_lane_f64(*c4, 1);

    renorm4(&q0[0], &q0[1], &q0[2], &q0[3], &q0[4]);
    renorm4(&q1[0], &q1[1], &q1[2], &q1[3], &q1[4]);

    //*c0 = vld1q_f64(&q0[0]);
    //*c1 = vld1q_f64(&q1[0]);
    //*c2 = vld1q_f64(&q0[2]);
    //*c3 = vld1q_f64(&q1[2]);
    //*c4 = vsetq_lane_f64(q0[4], vsetq_lane_f64(q1[4], vdupq_n_f64(0.0), 1), 0);
    //(* c0)[0] = q0[0]; (* c1)[0] = q0[1]; (* c2)[0] = q0[2]; (* c3)[0] = q0[3]; (* c4)[0] = q0[4];
    //(* c0)[1] = q1[0]; (* c1)[1] = q1[1]; (* c2)[1] = q1[2]; (* c3)[1] = q1[3]; (* c4)[1] = q1[4];

    *c0 = vsetq_lane_f64(q0[0], *c0, 0);
    *c1 = vsetq_lane_f64(q0[1], *c1, 0);
    *c2 = vsetq_lane_f64(q0[2], *c2, 0);
    *c3 = vsetq_lane_f64(q0[3], *c3, 0);
    *c4 = vsetq_lane_f64(q0[4], *c4, 0);
    *c0 = vsetq_lane_f64(q1[0], *c0, 1);
    *c1 = vsetq_lane_f64(q1[1], *c1, 1);
    *c2 = vsetq_lane_f64(q1[2], *c2, 1);
    *c3 = vsetq_lane_f64(q1[3], *c3, 1);
    *c4 = vsetq_lane_f64(q1[4], *c4, 1);
}

/********** Additions ************/
// three_sum for NEON
static inline void _bncneon_three_sum(float64x2_t *a, float64x2_t *b, float64x2_t *c)
{
    float64x2_t t1, t2, t3;

    t1 = _bncneon_dtwo_sum(*a, *b, &t2);
    *a = _bncneon_dtwo_sum(*c, t1, &t3);
    *b = _bncneon_dtwo_sum(t2, t3, c);
}

// three_sum2 for NEON
static inline void _bncneon_three_sum2(float64x2_t *a, float64x2_t *b, float64x2_t *c)
{
    float64x2_t t1, t2, t3;

    t1 = _bncneon_dtwo_sum(*a, *b, &t2);
    *a = _bncneon_dtwo_sum(*c, t1, &t3);
    *b = vaddq_f64(t2, t3);
}

#ifdef USE_QD_BF
    #define _bncneon_rqd_add _bncneon_rqd_add_bf
    #define _bncneon_rqd_mul _bncneon_rqd_mul_bf
#else // USE_QD_BF
    #define _bncneon_rqd_add _bncneon_rqd_add_sloppy
    #define _bncneon_rqd_mul _bncneon_rqd_mul_sloppy
#endif // USE_QD_BF

// QD addition for NEON
static inline void _bncneon_rqd_add_sloppy(float64x2_t ret[QDSIZE], float64x2_t a[QDSIZE], float64x2_t b[QDSIZE])
{
    float64x2_t s0, s1, s2, s3;
    float64x2_t t0, t1, t2, t3;
    float64x2_t v0, v1, v2, v3;
    float64x2_t u0, u1, u2, u3;
    float64x2_t w0, w1, w2, w3;

    s0 = vaddq_f64(a[0], b[0]);
    s1 = vaddq_f64(a[1], b[1]);
    s2 = vaddq_f64(a[2], b[2]);
    s3 = vaddq_f64(a[3], b[3]);

    v0 = vsubq_f64(s0, a[0]);
    v1 = vsubq_f64(s1, a[1]);
    v2 = vsubq_f64(s2, a[2]);
    v3 = vsubq_f64(s3, a[3]);

    u0 = vsubq_f64(s0, v0);
    u1 = vsubq_f64(s1, v1);
    u2 = vsubq_f64(s2, v2);
    u3 = vsubq_f64(s3, v3);

    w0 = vsubq_f64(a[0], u0);
    w1 = vsubq_f64(a[1], u1);
    w2 = vsubq_f64(a[2], u2);
    w3 = vsubq_f64(a[3], u3);

    u0 = vsubq_f64(b[0], v0);
    u1 = vsubq_f64(b[1], v1);
    u2 = vsubq_f64(b[2], v2);
    u3 = vsubq_f64(b[3], v3);

    t0 = vaddq_f64(w0, u0);
    t1 = vaddq_f64(w1, u1);
    t2 = vaddq_f64(w2, u2);
    t3 = vaddq_f64(w3, u3);

    s1 = _bncneon_dtwo_sum(s1, t0, &t0);
    _bncneon_three_sum(&s2, &t0, &t1);
    _bncneon_three_sum2(&s3, &t0, &t2);
    t0 = vaddq_f64(vaddq_f64(t0, t1), t3);

    /* renormalize */
    _bncneon_renorm4(&s0, &s1, &s2, &s3, &t0);

    ret[0] = s0;
    ret[1] = s1;
    ret[2] = s2;
    ret[3] = s3;
}

// Branch free algorithm
static inline void _bncneon_rqd_add_bf(float64x2_t ret[QDSIZE],
                                      const float64x2_t a[QDSIZE],
                                      const float64x2_t b[QDSIZE])
{
    float64x2_t a0 , b0 , c0 , d0 , e0, f0, g0, h0;
    float64x2_t a1 , b1 , c1 , d1 , e1, f1, g1, h1;
    float64x2_t a2 , b2 , c2 , d2 , e2, f2, g2;
    float64x2_t a3 , b3 , c3 , d3 , e3, f3, g3;
    float64x2_t a4 , c4 , d4 , e4;
    float64x2_t b5 , d5 , e5;
    float64x2_t b6 , c6 , d6 , e6;
    float64x2_t a7 , b7 , c7 , d7;
    float64x2_t b8 , c8 , e8;
    float64x2_t d9;
    float64x2_t a10, b10, c10, d10;
    float64x2_t b11, c11;
    float64x2_t c12, d12;

    a0 = a[0];
    b0 = b[0];
    c0 = a[1];
    d0 = b[1];
    e0 = a[2];
    f0 = b[2];
    g0 = a[3];
    h0 = b[3];

    a1 = _bncneon_dtwo_sum(a0, b0, &b1);
    c1 = _bncneon_dtwo_sum(c0, d0, &d1);
    e1 = _bncneon_dtwo_sum(e0, f0, &f1);
    g1 = _bncneon_dtwo_sum(g0, h0, &h1);

    a2 = _bncneon_dquick_two_sum(a1, c1, &c2);
    b2 = vaddq_f64(b1, h1);
    d2 = _bncneon_dtwo_sum(d1, e1, &e2);
    f2 = _bncneon_dtwo_sum(f1, g1, &g2);

    b3 = _bncneon_dtwo_sum(b2, g2, &g3);
    c3 = _bncneon_dquick_two_sum(c2, d2, &d3);
    e3 = _bncneon_dtwo_sum(e2, f2, &f3);

    a4 = _bncneon_dquick_two_sum(a2, c3, &c4);
    d4 = _bncneon_dquick_two_sum(d3, e3, &e4);

    b5 = _bncneon_dtwo_sum(b3, d4, &d5);
    e5 = vaddq_f64(e4, f3);

    b6 = _bncneon_dtwo_sum(b5, c4, &c6);
    d6 = _bncneon_dtwo_sum(d5, e5, &e6);

    a7 = _bncneon_dquick_two_sum(a4, b6, &b7);
    c7 = _bncneon_dquick_two_sum(c6, d6, &d7);

    e8 = vaddq_f64(e6, g3);
    b8 = _bncneon_dquick_two_sum(b7, c7, &c8);

    d9 = vaddq_f64(d7, e8);
    a10 = _bncneon_dquick_two_sum(a7, b8, &b10);
    c10 = _bncneon_dquick_two_sum(c8, d9, &d10);

    b11 = _bncneon_dquick_two_sum(b10, c10, &c11);
    c12 = _bncneon_dquick_two_sum(c11, d10, &d12);

    // return MultiFloat<T, 4>{a10, b11, c12, d12};
    ret[0] = a10;
    ret[1] = b11;
    ret[2] = c12;
    ret[3] = d12;
}


// TD addition (triple-double)
static inline void _bncneon_rtd_addq(float64x2_t ret[TDSIZE], float64x2_t a[TDSIZE], float64x2_t b[TDSIZE])
{
    float64x2_t s0, s1, s2;
    float64x2_t t0, t1, t2;
    float64x2_t v0, v1, v2;
    float64x2_t u0, u1, u2;
    float64x2_t w0, w1, w2;

    s0 = vaddq_f64(a[0], b[0]);
    s1 = vaddq_f64(a[1], b[1]);
    s2 = vaddq_f64(a[2], b[2]);

    v0 = vsubq_f64(s0, a[0]);
    v1 = vsubq_f64(s1, a[1]);
    v2 = vsubq_f64(s2, a[2]);

    u0 = vsubq_f64(s0, v0);
    u1 = vsubq_f64(s1, v1);
    u2 = vsubq_f64(s2, v2);

    w0 = vsubq_f64(a[0], u0);
    w1 = vsubq_f64(a[1], u1);
    w2 = vsubq_f64(a[2], u2);

    u0 = vsubq_f64(b[0], v0);
    u1 = vsubq_f64(b[1], v1);
    u2 = vsubq_f64(b[2], v2);

    t0 = vaddq_f64(w0, u0);
    t1 = vaddq_f64(w1, u1);
    t2 = vaddq_f64(w2, u2);

    s1 = _bncneon_dtwo_sum(s1, t0, &t0);
    _bncneon_three_sum(&s2, &t0, &t1);
    t0 = vaddq_f64(vaddq_f64(t0, t1), t2);

    /* renormalize */
    _bncneon_renorm(&s0, &s1, &s2, &t0);

    ret[0] = s0;
    ret[1] = s1;
    ret[2] = s2;
}

// TD multiplication
static inline void _bncneon_rtd_mulq(float64x2_t ret[TDSIZE], float64x2_t a[TDSIZE], float64x2_t b[TDSIZE])
{
    float64x2_t p0, p1, p2, p3, p4, p5;
    float64x2_t q0, q1, q2, q3, q4, q5;
    float64x2_t t0, t1;
    float64x2_t s0, s1, s2;

    p0 = _bncneon_dtwo_prod(a[0], b[0], &q0);

    p1 = _bncneon_dtwo_prod(a[0], b[1], &q1);
    p2 = _bncneon_dtwo_prod(a[1], b[0], &q2);

    p3 = _bncneon_dtwo_prod(a[0], b[2], &q3);
    p4 = _bncneon_dtwo_prod(a[1], b[1], &q4);
    p5 = _bncneon_dtwo_prod(a[2], b[0], &q5);

    /* Start Accumulation */
    _bncneon_three_sum(&p1, &p2, &q0);

    /* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
    _bncneon_three_sum2(&p2, &q1, &q2);
    _bncneon_three_sum2(&p3, &p4, &p5);

    /* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
    s0 = _bncneon_dtwo_sum(p2, p3, &t0);
    s1 = _bncneon_dtwo_sum(q1, p4, &t1);
    s1 = _bncneon_dtwo_sum(s1, t0, &t0);

    /* O(eps^3) order terms */
    s1 = vaddq_f64(s1, vmulq_f64(a[1], b[2]));
    s1 = vaddq_f64(s1, vmulq_f64(a[2], b[1]));
    s1 = vaddq_f64(s1, q0);
    s1 = vaddq_f64(s1, q3);
    s1 = vaddq_f64(s1, q4);

    _bncneon_renorm(&p0, &p1, &s0, &s1);

    ret[0] = p0;
    ret[1] = p1;
    ret[2] = s0;
}

// QD multiplication
static inline void _bncneon_rqd_mul_sloppy(float64x2_t ret[QDSIZE], float64x2_t a[QDSIZE], float64x2_t b[QDSIZE])
{
    float64x2_t p0, p1, p2, p3, p4, p5;
    float64x2_t q0, q1, q2, q3, q4, q5;
    float64x2_t t0, t1;
    float64x2_t s0, s1, s2;

    p0 = _bncneon_dtwo_prod(a[0], b[0], &q0);

    p1 = _bncneon_dtwo_prod(a[0], b[1], &q1);
    p2 = _bncneon_dtwo_prod(a[1], b[0], &q2);

    p3 = _bncneon_dtwo_prod(a[0], b[2], &q3);
    p4 = _bncneon_dtwo_prod(a[1], b[1], &q4);
    p5 = _bncneon_dtwo_prod(a[2], b[0], &q5);

    /* Start Accumulation */
    _bncneon_three_sum(&p1, &p2, &q0);

    /* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
    _bncneon_three_sum(&p2, &q1, &q2);
    _bncneon_three_sum(&p3, &p4, &p5);

    /* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
    s0 = _bncneon_dtwo_sum(p2, p3, &t0);
    s1 = _bncneon_dtwo_sum(q1, p4, &t1);
    s2 = vaddq_f64(q2, p5);
    s1 = _bncneon_dtwo_sum(s1, t0, &t0);
    s2 = vaddq_f64(s2, vaddq_f64(t0, t1));

    /* O(eps^3) order terms */
    s1 = vaddq_f64(s1, vmulq_f64(a[0], b[3]));
    s1 = vaddq_f64(s1, vmulq_f64(a[1], b[2]));
    s1 = vaddq_f64(s1, vmulq_f64(a[2], b[1]));
    s1 = vaddq_f64(s1, vmulq_f64(a[3], b[0]));
    s1 = vaddq_f64(s1, q0);
    s1 = vaddq_f64(s1, q3);
    s1 = vaddq_f64(s1, q4);
    s1 = vaddq_f64(s1, q5);

    _bncneon_renorm4(&p0, &p1, &s0, &s1, &s2);

    ret[0] = p0;
    ret[1] = p1;
    ret[2] = s0;
    ret[3] = s1;
}

// Branch free algorithm
static inline void _bncneon_rqd_mul_bf(float64x2_t ret[QDSIZE],
                                      const float64x2_t a[QDSIZE],
                                      const float64x2_t b[QDSIZE])
{
    float64x2_t a0, b0, c0, d0, e0, f0, g0, h0, i0, j0, k0, l0, m0, n0, o0, p0;
    float64x2_t a1, b1, c1, d1, e1, f1, g1, h1, i1, j1, m1, n1;
    float64x2_t b2, c2, e2, f2, i2, h2, m2;
    float64x2_t a3, b3, c3, d3, e3, f3, g3, h3;
    float64x2_t c4, d4, e4, f4;
    float64x2_t d5;
    float64x2_t c6, d6;
    float64x2_t b7, c7, d7;
    float64x2_t a8, b8, c8, d8;
    float64x2_t b9, c9;
    float64x2_t c10, d10;

    a0 = _bncneon_dtwo_prod(a[0], b[0], &b0);
    c0 = _bncneon_dtwo_prod(a[0], b[1], &e0);
    d0 = _bncneon_dtwo_prod(a[1], b[0], &f0);
    g0 = _bncneon_dtwo_prod(a[0], b[2], &j0);
    h0 = _bncneon_dtwo_prod(a[1], b[1], &k0);
    i0 = _bncneon_dtwo_prod(a[2], b[0], &l0);

    m0 = vmulq_f64(a[0], b[3]);
    n0 = vmulq_f64(a[1], b[2]);
    o0 = vmulq_f64(a[2], b[1]);
    p0 = vmulq_f64(a[3], b[0]);

    c1 = _bncneon_dtwo_sum(c0, d0, &d1);
    e1 = _bncneon_dtwo_sum(e0, f0, &f1);
    g1 = _bncneon_dtwo_sum(g0, i0, &i1);

    j1 = vaddq_f64(j0, l0);
    m1 = vaddq_f64(m0, p0);
    n1 = vaddq_f64(n0, o0);

    b2 = _bncneon_dtwo_sum(b0, c1, &c2);
    e2 = _bncneon_dtwo_sum(e1, h0, &h2);

    f2 = vaddq_f64(f1, j1);
    i2 = vaddq_f64(i1, k0);
    m2 = vaddq_f64(m1, n1);

    a3 = _bncneon_dquick_two_sum(a0, b2, &b3);
    c3 = _bncneon_dquick_two_sum(c2, d1, &d3);
    e3 = _bncneon_dtwo_sum(e2, g1, &g3);

    f3 = vaddq_f64(f2, m2);
    h3 = vaddq_f64(h2, i2);

    c4 = _bncneon_dtwo_sum(c3, e3, &e4);
    d4 = vaddq_f64(d3, h3);
    f4 = vaddq_f64(f3, g3);

    d5 = vaddq_f64(d4, e4);
    c6 = _bncneon_dtwo_sum(c4, d5, &d6);

    b7 = _bncneon_dtwo_sum(b3, c6, &c7);
    d7 = vaddq_f64(d6, f4);

    a8 = _bncneon_dquick_two_sum(a3, b7, &b8);
    c8 = _bncneon_dtwo_sum(c7, d7, &d8);

    b9  = _bncneon_dtwo_sum(b8, c8, &c9);
    c10 = _bncneon_dquick_two_sum(c9, d8, &d10);

    // return MultiFloat<T, 4>{a8, b9, c10, d10};
    ret[0] = a8;
    ret[1] = b9;
    ret[2] = c10;
    ret[3] = d10;
}


// QD negation
static inline void _bncneon_rqd_neg(float64x2_t c[QDSIZE], float64x2_t a[QDSIZE])
{
    c[0] = vnegq_f64(a[0]);
    c[1] = vnegq_f64(a[1]);
    c[2] = vnegq_f64(a[2]);
    c[3] = vnegq_f64(a[3]);
}

// QD subtraction
static inline void _bncneon_rqd_sub(float64x2_t c[QDSIZE], float64x2_t a[QDSIZE], float64x2_t b[QDSIZE])
{
    float64x2_t mb[QDSIZE];

    // a + (-b)
    _bncneon_rqd_neg(mb, b);
    _bncneon_rqd_add(c, a, mb);
}

// QD + double addition
static inline void _bncneon_rqd_add_d(float64x2_t c[QDSIZE], const float64x2_t a[QDSIZE], float64x2_t b)
{
    float64x2_t e0, e1, s0, s1, s2, s3;

    s0 = _bncneon_dtwo_sum(a[0], b,  &e0);
    s1 = _bncneon_dtwo_sum(a[1], e0, &e1);
    s2 = _bncneon_dtwo_sum(a[2], e1, &e0);
    s3 = _bncneon_dtwo_sum(a[3], e0, &e1);

    _bncneon_renorm4(&s0, &s1, &s2, &s3, &e1);

    c[0] = s0;
    c[1] = s1;
    c[2] = s2;
    c[3] = s3;

    return;
}

// TD + double addition
static inline void _bncneon_rtd_addq_d(float64x2_t c[TDSIZE], const float64x2_t a[TDSIZE], float64x2_t b)
{
    float64x2_t e0, e1, s0, s1, s2;

    s0 = _bncneon_dtwo_sum(a[0], b,  &e0);
    s1 = _bncneon_dtwo_sum(a[1], e0, &e1);
    s2 = _bncneon_dtwo_sum(a[2], e1, &e0);

    _bncneon_renorm(&s0, &s1, &s2, &e0);

    c[0] = s0;
    c[1] = s1;
    c[2] = s2;

    return;
}

// QD * double multiplication
static inline void _bncneon_rqd_mul_d(float64x2_t c[QDSIZE], const float64x2_t a[QDSIZE], float64x2_t b)
{
    float64x2_t p0, p1, p2, p3;
    float64x2_t q0, q1, q2;
    float64x2_t s0, s1, s2, s3, s4;

    p0 = _bncneon_dtwo_prod(a[0], b, &q0);
    p1 = _bncneon_dtwo_prod(a[1], b, &q1);
    p2 = _bncneon_dtwo_prod(a[2], b, &q2);
    p3 = vmulq_f64(a[3], b);

    s0 = p0;

    s1 = _bncneon_dtwo_sum(q0, p1, &s2);

    _bncneon_three_sum(&s2, &q1, &p2);

    _bncneon_three_sum2(&q1, &q2, &p3);
    s3 = q1;

    s4 = vaddq_f64(q2, p2);

    _bncneon_renorm4(&s0, &s1, &s2, &s3, &s4);

    c[0] = s0;
    c[1] = s1;
    c[2] = s2;
    c[3] = s3;
}

// TD * double multiplication
static inline void _bncneon_rtd_mulq_d(float64x2_t c[TDSIZE], const float64x2_t a[TDSIZE], float64x2_t b)
{
    float64x2_t p0, p1, p2;
    float64x2_t q0, q1, q2;
    float64x2_t s0, s1, s2, s4;

    p0 = _bncneon_dtwo_prod(a[0], b, &q0);
    p1 = _bncneon_dtwo_prod(a[1], b, &q1);
    p2 = _bncneon_dtwo_prod(a[2], b, &q2);

    s0 = p0;

    s1 = _bncneon_dtwo_sum(q0, p1, &s2);

    _bncneon_three_sum(&s2, &q1, &p2);

    s4 = vaddq_f64(q2, p2);

    _bncneon_renorm(&s0, &s1, &s2, &s4);

    c[0] = s0;
    c[1] = s1;
    c[2] = s2;
}

// TD division
static inline void _bncneon_rtd_divq(float64x2_t c[TDSIZE], float64x2_t a[TDSIZE], float64x2_t b[TDSIZE])
{
    float64x2_t q0, q1, q2, q3;
    float64x2_t r[TDSIZE], tmp[TDSIZE];

    q0 = vdivq_f64(a[0], b[0]);

    _bncneon_rtd_mul_d(tmp, q0, b);
    _bncneon_rtd_subq(r, a, tmp);

    q1 = vdivq_f64(r[0], b[0]);

    _bncneon_rtd_mulq_d(tmp, b, q1);
    _bncneon_rtd_addq(r, r, tmp);

    q2 = vdivq_f64(r[0], b[0]);
    _bncneon_rtd_mulq_d(tmp, b, q2);
    _bncneon_rtd_addq(r, r, tmp);

    q3 = vdivq_f64(r[0], b[0]);

    _bncneon_renorm(&q0, &q1, &q2, &q3);

    c[0] = q0;
    c[1] = q1;
    c[2] = q2;
}

// QD division
static inline void _bncneon_rqd_div(float64x2_t c[QDSIZE], float64x2_t a[QDSIZE], float64x2_t b[QDSIZE])
{
    float64x2_t q0, q1, q2, q3;
    float64x2_t r[QDSIZE], tmp[QDSIZE];

    q0 = vdivq_f64(a[0], b[0]);

    _bncneon_rqd_mul_d(tmp, b, q0);
    _bncneon_rqd_sub(r, a, tmp);

    q1 = vdivq_f64(r[0], b[0]);

    _bncneon_rqd_mul_d(tmp, b, q1);
    _bncneon_rqd_sub(r, r, tmp);

    q2 = vdivq_f64(r[0], b[0]);
    _bncneon_rqd_mul_d(tmp, b, q2);
    _bncneon_rqd_sub(r, r, tmp);

    q3 = vdivq_f64(r[0], b[0]);

    _bncneon_renorm4(&q0, &q1, &q2, &q3, &tmp[0]);

    c[0] = q0;
    c[1] = q1;
    c[2] = q2;
    c[3] = q3;
}

#endif // defined(__aarch64__) || defined(__arm64__)

#if 0
// Helper functions that need to be implemented elsewhere
// These are placeholders for the basic two_sum and two_prod operations
static inline float64x2_t _bncneon_dtwo_sum(float64x2_t a, float64x2_t b, float64x2_t *err)
{
    float64x2_t s = vaddq_f64(a, b);
    float64x2_t v = vsubq_f64(s, a);
    *err = vaddq_f64(vsubq_f64(a, vsubq_f64(s, v)), vsubq_f64(b, v));
    return s;
}

static inline float64x2_t _bncneon_dtwo_prod(float64x2_t a, float64x2_t b, float64x2_t *err)
{
    float64x2_t p = vmulq_f64(a, b);
    // For exact error computation, we would need FMA operations
    // This is a simplified version - actual implementation would need
    // proper error-free transformation
    *err = vsubq_f64(vmulq_f64(a, b), p);
    return p;
}
#endif // 0

#ifdef __BNC_TDLINEAR_H__
#if 0
// tdrel_diff
static inline tdfloat tdrel_diff(tdfloat a, tdfloat b)
{
    tdfloat rel_diff, abs_a;

    rtd_sub(rel_diff.val, a.val, b.val);
    rtd_abs(rel_diff.val, rel_diff.val);

    if(rtd_cmp_ui(a.val, 0UL) != 0)
    {
        rtd_abs(abs_a.val, a.val);
        rtd_div(rel_diff.val, rel_diff.val, a.val);
    }

    return rel_diff;
}

// qdrel_diff
static inline qdfloat qdrel_diff(qdfloat a, qdfloat b)
{
    qdfloat rel_diff, abs_a;

    rqd_sub(rel_diff.val, a.val, b.val);
    rqd_abs(rel_diff.val, rel_diff.val);

    if(rqd_cmp_ui(a.val, 0UL) != 0)
    {
        rqd_abs(abs_a.val, a.val);
        rqd_div(rel_diff.val, rel_diff.val, a.val);
    }

    return rel_diff;
}
#endif //  0
#endif //def __BNC_TDLINEAR_H__

#if defined(__BNC_QDLINEAR_H__) && defined(_DEF_BNC_QDVECTOR)
// qdmatmul_qdvec
static void qdmatmul_qdvec(QDVector ret, QDVector mat_a, QDVector mat_b, int row_dim, int mid_dim, int col_dim)
{
    int i, j, k;
    qdfloat tmp_mul, aik, bkj, cij;

    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            rqd_set_ui(cij.val, 0UL);
            for(k = 0; k < mid_dim; k++)
            {
                aik.val[0] = mat_a->element[0][i * mid_dim + k];
                aik.val[1] = mat_a->element[1][i * mid_dim + k];
                aik.val[2] = mat_a->element[2][i * mid_dim + k];
                aik.val[3] = mat_a->element[3][i * mid_dim + k];

                bkj.val[0] = mat_b->element[0][k * col_dim + j];
                bkj.val[1] = mat_b->element[1][k * col_dim + j];
                bkj.val[2] = mat_b->element[2][k * col_dim + j];
                bkj.val[3] = mat_b->element[3][k * col_dim + j];

                rqd_mul(tmp_mul.val, aik.val, bkj.val);
                rqd_add(cij.val, cij.val, tmp_mul.val);
            }
            ret->element[0][i * col_dim + j] = cij.val[0];
            ret->element[1][i * col_dim + j] = cij.val[1];
            ret->element[2][i * col_dim + j] = cij.val[2];
            ret->element[3][i * col_dim + j] = cij.val[3];
        }
    }
}

// qdmatmul_qdvec_ur2 (unrolled by 2 for NEON)
static void qdmatmul_qdvec_ur2(QDVector ret, QDVector mat_a, QDVector mat_b, int row_dim, int mid_dim, int col_dim)
{
    int i, j, k;
    qdfloat tmp_mul[2], aik[2], bkj[2], cij;

    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            rqd_set_ui(cij.val, 0UL);
            for(k = 0; k < mid_dim; k += 2)
            {
                // Load A elements
                aik[0].val[0] = mat_a->element[0][i * mid_dim + k];
                aik[1].val[0] = mat_a->element[0][i * mid_dim + k + 1];
                aik[0].val[1] = mat_a->element[1][i * mid_dim + k];
                aik[1].val[1] = mat_a->element[1][i * mid_dim + k + 1];
                aik[0].val[2] = mat_a->element[2][i * mid_dim + k];
                aik[1].val[2] = mat_a->element[2][i * mid_dim + k + 1];
                aik[0].val[3] = mat_a->element[3][i * mid_dim + k];
                aik[1].val[3] = mat_a->element[3][i * mid_dim + k + 1];

                // Load B elements
                bkj[0].val[0] = mat_b->element[0][k * col_dim + j];
                bkj[1].val[0] = mat_b->element[0][(k + 1) * col_dim + j];
                bkj[0].val[1] = mat_b->element[1][k * col_dim + j];
                bkj[1].val[1] = mat_b->element[1][(k + 1) * col_dim + j];
                bkj[0].val[2] = mat_b->element[2][k * col_dim + j];
                bkj[1].val[2] = mat_b->element[2][(k + 1) * col_dim + j];
                bkj[0].val[3] = mat_b->element[3][k * col_dim + j];
                bkj[1].val[3] = mat_b->element[3][(k + 1) * col_dim + j];

                // Compute products
                rqd_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rqd_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);

                // Accumulate
                rqd_add(cij.val, cij.val, tmp_mul[0].val);
                rqd_add(cij.val, cij.val, tmp_mul[1].val);
            }
            ret->element[0][i * col_dim + j] = cij.val[0];
            ret->element[1][i * col_dim + j] = cij.val[1];
            ret->element[2][i * col_dim + j] = cij.val[2];
            ret->element[3][i * col_dim + j] = cij.val[3];
        }
    }
}

#if defined(__aarch64__) || defined(__arm64__)
// qdmatmul_qdvec_neon
static void qdmatmul_qdvec_neon(QDVector ret, QDVector mat_a, QDVector mat_b, int row_dim, int mid_dim, int col_dim)
{
    int i, j, k;
    double cijval[2][QDSIZE];
    float64x2_t cij[QDSIZE], aik[QDSIZE], bkj[QDSIZE], tmp_mul[QDSIZE];

    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            cij[0] = vdupq_n_f64(0.0);
            cij[1] = vdupq_n_f64(0.0);
            cij[2] = vdupq_n_f64(0.0);
            cij[3] = vdupq_n_f64(0.0);

            for(k = 0; k < mid_dim; k += 2)
            {
                // Load A matrix elements (2 elements at a time)
                aik[0] = vsetq_lane_f64(mat_a->element[0][i * mid_dim + k], 
                                        vsetq_lane_f64(mat_a->element[0][i * mid_dim + k + 1], 
                                                       vdupq_n_f64(0.0), 1), 0);
                aik[1] = vsetq_lane_f64(mat_a->element[1][i * mid_dim + k], 
                                        vsetq_lane_f64(mat_a->element[1][i * mid_dim + k + 1], 
                                                       vdupq_n_f64(0.0), 1), 0);
                aik[2] = vsetq_lane_f64(mat_a->element[2][i * mid_dim + k], 
                                        vsetq_lane_f64(mat_a->element[2][i * mid_dim + k + 1], 
                                                       vdupq_n_f64(0.0), 1), 0);
                aik[3] = vsetq_lane_f64(mat_a->element[3][i * mid_dim + k], 
                                        vsetq_lane_f64(mat_a->element[3][i * mid_dim + k + 1], 
                                                       vdupq_n_f64(0.0), 1), 0);

                // Load B matrix elements (2 elements at a time)
                bkj[0] = vsetq_lane_f64(mat_b->element[0][k * col_dim + j], 
                                        vsetq_lane_f64(mat_b->element[0][(k + 1) * col_dim + j], 
                                                       vdupq_n_f64(0.0), 1), 0);
                bkj[1] = vsetq_lane_f64(mat_b->element[1][k * col_dim + j], 
                                        vsetq_lane_f64(mat_b->element[1][(k + 1) * col_dim + j], 
                                                       vdupq_n_f64(0.0), 1), 0);
                bkj[2] = vsetq_lane_f64(mat_b->element[2][k * col_dim + j], 
                                        vsetq_lane_f64(mat_b->element[2][(k + 1) * col_dim + j], 
                                                       vdupq_n_f64(0.0), 1), 0);
                bkj[3] = vsetq_lane_f64(mat_b->element[3][k * col_dim + j], 
                                        vsetq_lane_f64(mat_b->element[3][(k + 1) * col_dim + j], 
                                                       vdupq_n_f64(0.0), 1), 0);

                // Perform NEON QD multiplication
                _bncneon_rqd_mul(tmp_mul, aik, bkj);

                // Accumulate results
                _bncneon_rqd_add(cij, cij, tmp_mul);
            }
            
            // Extract results and sum the 2 elements
            vst1q_f64(&cijval[0][0], cij[0]);
            vst1q_f64(&cijval[0][2], cij[2]);
            cijval[1][0] = vgetq_lane_f64(cij[1], 0); cijval[1][1] = vgetq_lane_f64(cij[1], 1);
            cijval[1][2] = vgetq_lane_f64(cij[3], 0); cijval[1][3] = vgetq_lane_f64(cij[3], 1);
            
            rqd_add(cijval[0], cijval[0], cijval[1]);

            ret->element[0][i * col_dim + j] = cijval[0][0];
            ret->element[1][i * col_dim + j] = cijval[0][1];
            ret->element[2][i * col_dim + j] = cijval[0][2];
            ret->element[3][i * col_dim + j] = cijval[0][3];
        }
    }
}
#endif // #if defined(__BNC_QDLINEAR_H__) && defined(_DEF_BNC_QDVECTOR)

#endif // ifndef __BNCNEON_QD_H
