// ------------------------
// -------- NEON DS -------
// ------------------------
// Auto-generated from _bncneon_dd.h (double-double) by mechanical
// translation: float64x2_t -> float32x4_t, double -> float, f64 -> f32,
// rdd_* (scalar) -> rds_*, ddfloat -> dsfloat, DDSIZE -> DSSIZE.
// Operates on float32x4_t (4 single-precision lanes per call).
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

// ret := val
static inline void _bncneon_rds_set(float32x4_t ret[DSSIZE], float32x4_t val[DSSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
}

// ret := (float)val
static inline void _bncneon_rds_set_f(float32x4_t ret[DSSIZE], float32x4_t val)
{
    ret[0] = val;
    ret[1] = vdupq_n_f32(0.0f);
}

// ret := [val val val val]
static inline void _bncneon_rds_set1_ds(float32x4_t ret[DSSIZE], float val[DSSIZE])
{
    ret[0] = vdupq_n_f32(val[0]);
    ret[1] = vdupq_n_f32(val[1]);
}

// ret := -val
static inline void _bncneon_rds_neg(float32x4_t ret[DSSIZE], float32x4_t val[DSSIZE])
{
    ret[0] = vnegq_f32(val[0]);
    ret[1] = vnegq_f32(val[1]);
}

// Lane extraction helpers: ret := ret4[lane]
// ds ret := (ret4[0][lane], ret4[1][lane])
static inline void _bncneon_get_ds_f32x4_i_0(dsfloat *ret, float32x4_t ret4[DSSIZE])
{
    ret->val[0] = vgetq_lane_f32(ret4[0], 0);
    ret->val[1] = vgetq_lane_f32(ret4[1], 0);
}

static inline void _bncneon_get_ds_f32x4_i_1(dsfloat *ret, float32x4_t ret4[DSSIZE])
{
    ret->val[0] = vgetq_lane_f32(ret4[0], 1);
    ret->val[1] = vgetq_lane_f32(ret4[1], 1);
}

static inline void _bncneon_get_ds_f32x4_i_2(dsfloat *ret, float32x4_t ret4[DSSIZE])
{
    ret->val[0] = vgetq_lane_f32(ret4[0], 2);
    ret->val[1] = vgetq_lane_f32(ret4[1], 2);
}

static inline void _bncneon_get_ds_f32x4_i_3(dsfloat *ret, float32x4_t ret4[DSSIZE])
{
    ret->val[0] = vgetq_lane_f32(ret4[0], 3);
    ret->val[1] = vgetq_lane_f32(ret4[1], 3);
}

// ret := sum_{i=0..3} ret4[i] (horizontal sum across 4 lanes)
// Macro form (avoids the GCC OpenMP outline + static-symbol pruning issue
// where _bncneon_rds_sum128f is referenced from inside _omp_fn.0 but never
// emitted into the .o, even with __attribute__((used))).  Macros have no
// linker symbol so this can never produce an undefined reference.
#define _bncneon_rds_sum128f(_ret, _ret4) do {                          \
    float _bncn_l0_[DSSIZE] = {                                         \
        vgetq_lane_f32((_ret4)[0], 0),                                  \
        vgetq_lane_f32((_ret4)[1], 0)                                   \
    };                                                                  \
    float _bncn_l1_[DSSIZE] = {                                         \
        vgetq_lane_f32((_ret4)[0], 1),                                  \
        vgetq_lane_f32((_ret4)[1], 1)                                   \
    };                                                                  \
    float _bncn_l2_[DSSIZE] = {                                         \
        vgetq_lane_f32((_ret4)[0], 2),                                  \
        vgetq_lane_f32((_ret4)[1], 2)                                   \
    };                                                                  \
    float _bncn_l3_[DSSIZE] = {                                         \
        vgetq_lane_f32((_ret4)[0], 3),                                  \
        vgetq_lane_f32((_ret4)[1], 3)                                   \
    };                                                                  \
    rds_set((_ret), _bncn_l0_);                                         \
    rds_add((_ret), (_ret), _bncn_l1_);                                 \
    rds_add((_ret), (_ret), _bncn_l2_);                                 \
    rds_add((_ret), (_ret), _bncn_l3_);                                 \
} while (0)

// ret := sum_{i=0..3} |ret4[i]|
static __attribute__((used)) void _bncneon_rds_abssum128f(float ret[DSSIZE], float32x4_t ret4[DSSIZE])
{
    dsfloat lane[4];
    float tmp[DSSIZE];

    _bncneon_get_ds_f32x4_i_0(&lane[0], ret4);
    _bncneon_get_ds_f32x4_i_1(&lane[1], ret4);
    _bncneon_get_ds_f32x4_i_2(&lane[2], ret4);
    _bncneon_get_ds_f32x4_i_3(&lane[3], ret4);

    rds_abs(tmp, lane[0].val); rds_set(ret, tmp);
    rds_abs(tmp, lane[1].val); rds_add(ret, ret, tmp);
    rds_abs(tmp, lane[2].val); rds_add(ret, ret, tmp);
    rds_abs(tmp, lane[3].val); rds_add(ret, ret, tmp);
}

// ret := max_{i=0..3} |ret4[i]|
static __attribute__((used)) void _bncneon_rds_absmax128f(float ret[DSSIZE], float32x4_t ret4[DSSIZE])
{
    dsfloat lane[4];
    float tmp[DSSIZE];
    int i;

    _bncneon_get_ds_f32x4_i_0(&lane[0], ret4);
    _bncneon_get_ds_f32x4_i_1(&lane[1], ret4);
    _bncneon_get_ds_f32x4_i_2(&lane[2], ret4);
    _bncneon_get_ds_f32x4_i_3(&lane[3], ret4);

    rds_abs(tmp, lane[0].val); rds_set(ret, tmp);
    for(i = 1; i < 4; i++)
    {
        rds_abs(tmp, lane[i].val);
        if(rds_cmp(ret, tmp) < 0)
            rds_set(ret, tmp);
    }
}

// ret := sqrt(sum_{i=0..3} ret4[i]^2)
static __attribute__((used)) void _bncneon_rds_norm128f(float ret[DSSIZE], float32x4_t ret4[DSSIZE])
{
    dsfloat lane[4];
    float tmp[DSSIZE];
    int i;

    _bncneon_get_ds_f32x4_i_0(&lane[0], ret4);
    _bncneon_get_ds_f32x4_i_1(&lane[1], ret4);
    _bncneon_get_ds_f32x4_i_2(&lane[2], ret4);
    _bncneon_get_ds_f32x4_i_3(&lane[3], ret4);

    rds_mul(tmp, lane[0].val, lane[0].val);
    rds_set(ret, tmp);
    for(i = 1; i < 4; i++)
    {
        rds_mul(tmp, lane[i].val, lane[i].val);
        rds_add(ret, ret, tmp);
    }
    rds_sqrt(tmp, ret);
    rds_set(ret, tmp);
}

// Double-single arithmetic with NEON

#ifdef USE_DS_BF
    #define _bncneon_rds_add    _bncneon_rds_add_bf
    #define _bncneon_rds_mul    _bncneon_rds_mul_bf
#else // USE_DS_BF
    #define _bncneon_rds_add    _bncneon_rds_add_sloppy
    #define _bncneon_rds_mul    _bncneon_rds_mul_sloppy
#endif // USE_DS_BF

// ret := a + b (double-single addition, sloppy)
static inline void _bncneon_rds_add_sloppy(float32x4_t ret[DSSIZE], float32x4_t a[DSSIZE], float32x4_t b[DSSIZE])
{
    float32x4_t s, e;
    s = _bncneon_ftwo_sum(a[0], b[0], &e);
    e = vaddq_f32(e, vaddq_f32(a[1], b[1]));
    ret[0] = _bncneon_fquick_two_sum(s, e, &e);
    ret[1] = e;
}

// DS add (branch-free gates), NEON
static inline void _bncneon_rds_add_bf(float32x4_t ret[DSSIZE], float32x4_t a[DSSIZE], float32x4_t b[DSSIZE])
{
    float32x4_t g1, g2, g3, g4, g5, g6;
    float32x4_t g1e, g2e, g3e, g6e;

    g1 = _bncneon_ftwo_sum(a[0], b[0], &g1e);
    g2 = _bncneon_ftwo_sum(a[1], b[1], &g2e);
    g3 = _bncneon_fquick_two_sum(g1, g2, &g3e);
    g4 = vaddq_f32(g1e, g2e);
    g5 = vaddq_f32(g4, g3e);
    g6 = _bncneon_fquick_two_sum(g3, g5, &g6e);

    ret[0] = g6;
    ret[1] = g6e;
}

// ret := a + b (double-single + float)
static inline void _bncneon_rds_add_f(float32x4_t ret[DSSIZE], float32x4_t a[DSSIZE], float32x4_t b)
{
    float32x4_t s1, s2;

    s1 = _bncneon_ftwo_sum(a[0], b, &s2);
    s2 = vaddq_f32(s2, a[1]);
    s1 = _bncneon_fquick_two_sum(s1, s2, &s2);
    ret[0] = s1;
    ret[1] = s2;
}

// ret := a - b (double-single subtraction)
static inline void _bncneon_rds_sub(float32x4_t ret[DSSIZE], float32x4_t a[DSSIZE], float32x4_t b[DSSIZE])
{
    float32x4_t s1, s2, t1, t2;
    s1 = _bncneon_ftwo_diff(a[0], b[0], &s2);
    t1 = _bncneon_ftwo_diff(a[1], b[1], &t2);
    s2 = vaddq_f32(s2, t1);
    s1 = _bncneon_fquick_two_sum(s1, s2, &s2);
    s2 = vaddq_f32(s2, t2);
    s1 = _bncneon_fquick_two_sum(s1, s2, &s2);
    ret[0] = s1;
    ret[1] = s2;
}

// ret := a - b (double-single - float)
static inline void _bncneon_rds_sub_f(float32x4_t ret[DSSIZE], float32x4_t a[DSSIZE], float32x4_t b)
{
    float32x4_t s1, s2;
    s1 = _bncneon_ftwo_diff(a[0], b, &s2);
    s2 = vaddq_f32(s2, a[1]);
    s1 = _bncneon_fquick_two_sum(s1, s2, &s2);
    ret[0] = s1;
    ret[1] = s2;
}

// ret := a * b (double-single multiplication, sloppy)
static inline void _bncneon_rds_mul_sloppy(float32x4_t ret[DSSIZE], float32x4_t a[DSSIZE], float32x4_t b[DSSIZE])
{
    float32x4_t p1, p2;

    p1 = _bncneon_ftwo_prod(a[0], b[0], &p2);
    p2 = vaddq_f32(p2,
        vaddq_f32(
            vmulq_f32(a[0], b[1]),
            vmulq_f32(a[1], b[0])
        )
    );

    ret[0] = _bncneon_fquick_two_sum(p1, p2, &p2);
    ret[1] = p2;
}

// DS mul (branch-free gates), NEON
static inline void _bncneon_rds_mul_bf(float32x4_t ret[DSSIZE],
                                      const float32x4_t a[DSSIZE],
                                      const float32x4_t b[DSSIZE])
{
    float32x4_t p00, p01, p10;
    float32x4_t pe00;
    float32x4_t g1, g2, g3, ge3;

    p00 = _bncneon_ftwo_prod(a[0], b[0], &pe00);
    p01 = vmulq_f32(a[0], b[1]);
    p10 = vmulq_f32(a[1], b[0]);

    g1 = vaddq_f32(p01, p10);
    g2 = vaddq_f32(pe00, g1);
    g3 = _bncneon_fquick_two_sum(p00, g2, &ge3);

    ret[0] = g3;
    ret[1] = ge3;
}

// ret := a * b (double-single * float)
static inline void _bncneon_rds_mul_f(float32x4_t ret[DSSIZE], float32x4_t a[DSSIZE], float32x4_t b)
{
    float32x4_t p1, p2;
    p1 = _bncneon_ftwo_prod(a[0], b, &p2);
    p2 = vfmaq_f32(p2, a[1], b);
    ret[0] = _bncneon_fquick_two_sum(p1, p2, &ret[1]);
}

// ret := a / b (double-single division)
static inline void _bncneon_rds_div(float32x4_t ret[DSSIZE], float32x4_t a[DSSIZE], float32x4_t b[DSSIZE])
{
    float32x4_t s1, s2;
    float32x4_t q1, q2;
    float32x4_t r[DSSIZE];

    q1 = vdivq_f32(a[0], b[0]);  /* approximate quotient */

    /* compute  this - q1 * ds */
    _bncneon_rds_mul_f(r, b, q1);

    s1 = _bncneon_ftwo_diff(a[0], r[0], &s2);
    s2 = vsubq_f32(s2, r[1]);
    s2 = vaddq_f32(s2, a[1]);

    /* get next approximation */
    q2 = vdivq_f32(vaddq_f32(s1, s2), b[0]);

    /* renormalize */
    ret[0] = _bncneon_fquick_two_sum(q1, q2, &ret[1]);
}

// ret := a / b (double-single / float)
static inline void _bncneon_rds_div_f(float32x4_t ret[DSSIZE], float32x4_t a[DSSIZE], float32x4_t b)
{
    float32x4_t q1, q2;
    float32x4_t p1, p2;
    float32x4_t s, e;
    float32x4_t r[DSSIZE];

    q1 = vdivq_f32(a[0], b);

    p1 = _bncneon_ftwo_prod(q1, b, &p2);
    s = _bncneon_ftwo_diff(a[0], p1, &e);
    e = vaddq_f32(e, a[1]);
    e = vsubq_f32(e, p2);
    q2 = vdivq_f32(vaddq_f32(s, e), b);

    r[0] = _bncneon_fquick_two_sum(q1, q2, &r[1]);
    ret[0] = r[0];
    ret[1] = r[1];
}

// ret := |a| (double-single absolute value)
static inline void _bncneon_rds_abs(float32x4_t ret[DSSIZE], float32x4_t a[DSSIZE])
{
    uint32x4_t mask = vcltq_f32(a[0], vdupq_n_f32(0.0f));
    ret[0] = vbslq_f32(mask, vnegq_f32(a[0]), a[0]);
    ret[1] = vbslq_f32(mask, vnegq_f32(a[1]), a[1]);
}

/* ================================================================
 *  ARM NEON FMA helpers for DS  (_bncneon_ prefix)
 *  Processes 4 DS operations per call
 *  FMA: vfmaq_f32(c, a, b) = a*b + c  (accumulator is 1st arg)
 * ================================================================ */

// ret[DSSIZE] := a * b + c  (scalar x scalar -> DS) x 4
static inline void _bncneon_rds_fma_f_f(
        float32x4_t ret[DSSIZE],
        float32x4_t a,
        float32x4_t b,
        float32x4_t c)
{
    ret[0] = vfmaq_f32(c, a, b);
    ret[1] = vfmaq_f32(vsubq_f32(c, ret[0]), a, b);
}

// ret[DSSIZE] := a * b + c[DSSIZE]  (scalar x scalar + DS -> DS) x 4
static inline void _bncneon_rds_fma_f_ds(
        float32x4_t       ret[DSSIZE],
        float32x4_t       a,
        float32x4_t       b,
        const float32x4_t c[DSSIZE])
{
    float32x4_t hi = vfmaq_f32(c[0], a, b);
    float32x4_t t  = vsubq_f32(c[0], hi);
    float32x4_t e  = vfmaq_f32(t, a, b);
    float32x4_t f  = vaddq_f32(e, c[1]);
    ret[0] = _bncneon_fquick_two_sum(hi, f, &(ret[1]));
}

// ret[DSSIZE] := a * b[DSSIZE] + c[DSSIZE]  (scalar x DS + DS -> DS) x 4
static inline void _bncneon_rds_fma_fm_ds(
        float32x4_t       ret[DSSIZE],
        float32x4_t       a,
        const float32x4_t b[DSSIZE],
        const float32x4_t c[DSSIZE])
{
    float32x4_t hi = vfmaq_f32(c[0], a, b[0]);
    float32x4_t t  = vsubq_f32(c[0], hi);
    float32x4_t e  = vfmaq_f32(t, a, b[0]);
    float32x4_t f  = vaddq_f32(e, c[1]);
    float32x4_t lo = vfmaq_f32(f, a, b[1]);
    ret[0] = _bncneon_fquick_two_sum(hi, lo, &(ret[1]));
}

// ret[DSSIZE] := a[DSSIZE] * b[DSSIZE] + c[DSSIZE]
static inline void _bncneon_rds_fma(
        float32x4_t       ret[DSSIZE],
        const float32x4_t a[DSSIZE],
        const float32x4_t b[DSSIZE],
        const float32x4_t c[DSSIZE])
{
    float32x4_t d0 = vfmaq_f32(c[0], a[0], b[0]);
    float32x4_t t  = vsubq_f32(c[0], d0);
    float32x4_t e  = vfmaq_f32(t,    a[0], b[0]);
    float32x4_t f  = vaddq_f32(e, c[1]);
    float32x4_t g  = vfmaq_f32(f,    a[0], b[1]);
    float32x4_t d1 = vfmaq_f32(g,    a[1], b[0]);
    ret[0] = _bncneon_fquick_two_sum(d0, d1, &(ret[1]));
}

#endif // ifndef __BNCNEON_DS_H
