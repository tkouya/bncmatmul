// ------------------------
// -------- NEON TS -------
// ------------------------
// Auto-generated from _bncneon_td.h (triple-double) by mechanical
// translation: float64x2_t -> float32x4_t, double -> float, f64 -> f32,
// rtd_* (scalar) -> rts_*, tdfloat -> tsfloat, TDSIZE -> TSSIZE,
// two_sum -> ftwo_sum, _bncneon_dtwo_* -> _bncneon_ftwo_*.
// Operates on float32x4_t (4 single-precision lanes per call).
#ifndef __BNCNEON_TS_H
#define __BNCNEON_TS_H

#include "_bncneon_ds.h"

#ifndef TSSIZE
    #define TSSIZE 3
#endif // TSSIZE

// Constants for division algorithms (single-precision counterparts)
// FLT_EPSILON = 2^-23 ~= 1.1920929e-7
#ifndef ONE_P_2FLT_EPS
#define ONE_P_2FLT_EPS 1.0000002384185791f
#endif
#ifndef ONE_M_2FLT_EPS
#define ONE_M_2FLT_EPS 0.9999997615814209f
#endif

// ret := 0
static inline void _bncneon_set0_ts(float32x4_t ret[TSSIZE])
{
    ret[0] = vdupq_n_f32(0.0f);
    ret[1] = vdupq_n_f32(0.0f);
    ret[2] = vdupq_n_f32(0.0f);
}
#define _bncneon_rts_set0(ret) _bncneon_set0_ts((ret))

// ret := val
static inline void _bncneon_rts_set(float32x4_t ret[TSSIZE], float32x4_t val[TSSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = val[2];
}

// ret := (float)val
static inline void _bncneon_rts_set_f(float32x4_t ret[TSSIZE], float32x4_t val)
{
    ret[0] = val;
    ret[1] = vdupq_n_f32(0.0f);
    ret[2] = vdupq_n_f32(0.0f);
}

// ret := (DS)val
static inline void _bncneon_rts_set_ds(float32x4_t ret[TSSIZE], float32x4_t val[DSSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = vdupq_n_f32(0.0f);
}

// ret := [val val val val]
static inline void _bncneon_rts_set1_ts(float32x4_t ret[TSSIZE], float val[TSSIZE])
{
    ret[0] = vdupq_n_f32(val[0]);
    ret[1] = vdupq_n_f32(val[1]);
    ret[2] = vdupq_n_f32(val[2]);
}

// Lane extraction helpers (4 lanes)
static inline void _bncneon_get_ts_f32x4_i_0(tsfloat *ret, float32x4_t ret4[TSSIZE])
{
    ret->val[0] = vgetq_lane_f32(ret4[0], 0);
    ret->val[1] = vgetq_lane_f32(ret4[1], 0);
    ret->val[2] = vgetq_lane_f32(ret4[2], 0);
}

static inline void _bncneon_get_ts_f32x4_i_1(tsfloat *ret, float32x4_t ret4[TSSIZE])
{
    ret->val[0] = vgetq_lane_f32(ret4[0], 1);
    ret->val[1] = vgetq_lane_f32(ret4[1], 1);
    ret->val[2] = vgetq_lane_f32(ret4[2], 1);
}

static inline void _bncneon_get_ts_f32x4_i_2(tsfloat *ret, float32x4_t ret4[TSSIZE])
{
    ret->val[0] = vgetq_lane_f32(ret4[0], 2);
    ret->val[1] = vgetq_lane_f32(ret4[1], 2);
    ret->val[2] = vgetq_lane_f32(ret4[2], 2);
}

static inline void _bncneon_get_ts_f32x4_i_3(tsfloat *ret, float32x4_t ret4[TSSIZE])
{
    ret->val[0] = vgetq_lane_f32(ret4[0], 3);
    ret->val[1] = vgetq_lane_f32(ret4[1], 3);
    ret->val[2] = vgetq_lane_f32(ret4[2], 3);
}

// ret := sum_{i=0..3} ret4[i] (horizontal sum across 4 lanes)
// Macro form (avoids the GCC OpenMP outline + static-symbol pruning issue).
#define _bncneon_rts_sum128f(_ret, _ret4) do {                          \
    float _bncn_l0_[TSSIZE] = {                                         \
        vgetq_lane_f32((_ret4)[0], 0),                                  \
        vgetq_lane_f32((_ret4)[1], 0),                                  \
        vgetq_lane_f32((_ret4)[2], 0)                                   \
    };                                                                  \
    float _bncn_l1_[TSSIZE] = {                                         \
        vgetq_lane_f32((_ret4)[0], 1),                                  \
        vgetq_lane_f32((_ret4)[1], 1),                                  \
        vgetq_lane_f32((_ret4)[2], 1)                                   \
    };                                                                  \
    float _bncn_l2_[TSSIZE] = {                                         \
        vgetq_lane_f32((_ret4)[0], 2),                                  \
        vgetq_lane_f32((_ret4)[1], 2),                                  \
        vgetq_lane_f32((_ret4)[2], 2)                                   \
    };                                                                  \
    float _bncn_l3_[TSSIZE] = {                                         \
        vgetq_lane_f32((_ret4)[0], 3),                                  \
        vgetq_lane_f32((_ret4)[1], 3),                                  \
        vgetq_lane_f32((_ret4)[2], 3)                                   \
    };                                                                  \
    rts_set((_ret), _bncn_l0_);                                         \
    rts_add((_ret), (_ret), _bncn_l1_);                                 \
    rts_add((_ret), (_ret), _bncn_l2_);                                 \
    rts_add((_ret), (_ret), _bncn_l3_);                                 \
} while (0)

// ret := |a| (triple-single absolute value)
static inline void _bncneon_rts_abs(float32x4_t ret[TSSIZE], float32x4_t a[TSSIZE])
{
    uint32x4_t mask = vcltq_f32(a[0], vdupq_n_f32(0.0f));
    ret[0] = vbslq_f32(mask, vnegq_f32(a[0]), a[0]);
    ret[1] = vbslq_f32(mask, vnegq_f32(a[1]), a[1]);
    ret[2] = vbslq_f32(mask, vnegq_f32(a[2]), a[2]);
}

// ret := sum_{i=0..3} |ret4[i]|
static __attribute__((used)) void _bncneon_rts_abssum128f(float ret[TSSIZE], float32x4_t ret4[TSSIZE])
{
    tsfloat lane[4];
    static float tmp[TSSIZE];

    _bncneon_get_ts_f32x4_i_0(&lane[0], ret4);
    _bncneon_get_ts_f32x4_i_1(&lane[1], ret4);
    _bncneon_get_ts_f32x4_i_2(&lane[2], ret4);
    _bncneon_get_ts_f32x4_i_3(&lane[3], ret4);

    rts_abs(tmp, lane[0].val); rts_set(ret, tmp);
    rts_abs(tmp, lane[1].val); rts_add(ret, ret, tmp);
    rts_abs(tmp, lane[2].val); rts_add(ret, ret, tmp);
    rts_abs(tmp, lane[3].val); rts_add(ret, ret, tmp);
}

// ret := max_{i=0..3} |ret4[i]|
static __attribute__((used)) void _bncneon_rts_absmax128f(float ret[TSSIZE], float32x4_t ret4[TSSIZE])
{
    tsfloat lane[4];
    static float tmp[TSSIZE];
    int i;

    _bncneon_get_ts_f32x4_i_0(&lane[0], ret4);
    _bncneon_get_ts_f32x4_i_1(&lane[1], ret4);
    _bncneon_get_ts_f32x4_i_2(&lane[2], ret4);
    _bncneon_get_ts_f32x4_i_3(&lane[3], ret4);

    rts_abs(tmp, lane[0].val); rts_set(ret, tmp);
    for(i = 1; i < 4; i++)
    {
        rts_abs(tmp, lane[i].val);
        if(rts_cmp(ret, tmp) < 0)
            rts_set(ret, tmp);
    }
}

// ret := sqrt(sum_{i=0..3} ret4[i]^2)
static __attribute__((used)) void _bncneon_rts_norm128f(float ret[TSSIZE], float32x4_t ret4[TSSIZE])
{
    tsfloat lane[4];
    static float tmp[TSSIZE];
    int i;

    _bncneon_get_ts_f32x4_i_0(&lane[0], ret4);
    _bncneon_get_ts_f32x4_i_1(&lane[1], ret4);
    _bncneon_get_ts_f32x4_i_2(&lane[2], ret4);
    _bncneon_get_ts_f32x4_i_3(&lane[3], ret4);

    rts_mul(tmp, lane[0].val, lane[0].val);
    rts_set(ret, tmp);
    for(i = 1; i < 4; i++)
    {
        rts_mul(tmp, lane[i].val, lane[i].val);
        rts_add(ret, ret, tmp);
    }
    rts_sqrt(tmp, ret);
    rts_set(ret, tmp);
}

// e[n] := vec_sum(x[n])  (float version)
static inline void _bncneon_fvec_sum(float32x4_t e[], const float32x4_t x[], int n)
{
    float32x4_t s;

    s = x[--n];
    while(--n >= 0)
    {
        s = _bncneon_ftwo_sum(x[n], s, &e[n + 1]);
    }
    e[0] = s;
}

// y[n] := vec_sum_err_branch(vseb)(k)(e[n])  (float, 4-lane version)
static inline void _bncneon_fvseb(float32x4_t y[], int ny, const float32x4_t e[], int ne)
{
    int i, j[4], neon_index;
    float dtmp;
    float32x4_t r, eps, temp, in_y[16];

    j[0] = j[1] = j[2] = j[3] = 0;
    eps = e[0];
    for(i = 0; i < (ne - 2); i++)
    {
        r = _bncneon_ftwo_sum(eps, e[i + 1], &temp);

        for(neon_index = 0; neon_index < 4; neon_index++)
        {
            if(vgetq_lane_f32(temp, neon_index) != 0.0f)
            {
                in_y[j[neon_index]] = vsetq_lane_f32(vgetq_lane_f32(r, neon_index), in_y[j[neon_index]], neon_index);
                eps = vsetq_lane_f32(vgetq_lane_f32(temp, neon_index), eps, neon_index);
                j[neon_index]++;
            }
            else
                eps = vsetq_lane_f32(vgetq_lane_f32(r, neon_index), eps, neon_index);
        }
    }

    for(neon_index = 0; neon_index < 4; neon_index++)
    {
        float eps_lane = vgetq_lane_f32(eps, neon_index);
        float e_lane = vgetq_lane_f32(e[ne - 1], neon_index);
        float sum = ftwo_sum(eps_lane, e_lane, &dtmp);

        in_y[j[neon_index]] = vsetq_lane_f32(sum, in_y[j[neon_index]], neon_index);
        in_y[j[neon_index] + 1] = vsetq_lane_f32(dtmp, in_y[j[neon_index] + 1], neon_index);

        for(i = j[neon_index] + 2; i < ne; i++)
            in_y[i] = vsetq_lane_f32(0.0f, in_y[i], neon_index);
    }

    for(i = 0; i < ny; i++)
        y[i] = in_y[i];
}

// Merge a[na] & b[nb] into c[na + nb]  (float, 4-lane version)
static inline void _bncneon_fmerge(float32x4_t c[], float32x4_t a[], int na, float32x4_t b[], int nb)
{
    int i, j, k;
    int neon_index;

    for(neon_index = 0; neon_index < 4; neon_index++)
    {
        i = j = k = 0;
        while((i < na) && (j < nb))
        {
            float a_val = vgetq_lane_f32(a[i], neon_index);
            float b_val = vgetq_lane_f32(b[j], neon_index);

            if(fabsf(a_val) >= fabsf(b_val))
            {
                c[k] = vsetq_lane_f32(a_val, c[k], neon_index);
                i++;
            }
            else
            {
                c[k] = vsetq_lane_f32(b_val, c[k], neon_index);
                j++;
            }
            k++;
        }
        while(i < na)
        {
            c[k] = vsetq_lane_f32(vgetq_lane_f32(a[i], neon_index), c[k], neon_index);
            i++; k++;
        }
        while(j < nb)
        {
            c[k] = vsetq_lane_f32(vgetq_lane_f32(b[j], neon_index), c[k], neon_index);
            j++; k++;
        }
    }
}

// Triple-single arithmetic operations

#ifdef USE_TS_BF
    #define _bncneon_rts_add _bncneon_rts_add_bf
    #define _bncneon_rts_mul _bncneon_rts_mul_bf
#else // USE_TS_BF
    #define _bncneon_rts_add _bncneon_rts_addq
    #define _bncneon_rts_mul _bncneon_rts_mulq
#endif // USE_TS_BF

// Forward declarations
static inline void _bncneon_rts_addq(float32x4_t ret[TSSIZE], float32x4_t a[TSSIZE], float32x4_t b[TSSIZE]);
static inline void _bncneon_rts_mulq(float32x4_t ret[TSSIZE], float32x4_t a[TSSIZE], float32x4_t b[TSSIZE]);

// Triple-single addition (alternative implementation)
static inline void _bncneon_rts_addt(float32x4_t ret[TSSIZE], float32x4_t a[TSSIZE], float32x4_t b[TSSIZE])
{
    float32x4_t z[6], e[6];

    _bncneon_fmerge(z, a, 3, b, 3);
    _bncneon_fvec_sum(e, z, 6);
    _bncneon_fvseb(ret, 3, e, 6);
}

// Branch-free algorithm for TS add
static inline void _bncneon_rts_add_bf(float32x4_t ret[TSSIZE],
                                      const float32x4_t a[TSSIZE],
                                      const float32x4_t b[TSSIZE])
{
    float32x4_t a0, b0, c0, d0, e0, f0;
    float32x4_t a1, b1, c1, d1, e1, f1;
    float32x4_t a2, b2, c2, d2, e2;
    float32x4_t a3, b3, c3, d3;
    float32x4_t c4, c5, d5;
    float32x4_t b6, c6;
    float32x4_t a7, b7, c7;
    float32x4_t b8, c8;

    a0 = a[0];
    b0 = b[0];
    c0 = a[1];
    d0 = b[1];
    e0 = a[2];
    f0 = b[2];

    a1 = _bncneon_ftwo_sum(a0, b0, &b1);
    c1 = _bncneon_ftwo_sum(c0, d0, &d1);
    e1 = _bncneon_ftwo_sum(e0, f0, &f1);

    a2 = _bncneon_fquick_two_sum(a1, c1, &c2);
    b2 = vaddq_f32(b1, f1);
    d2 = _bncneon_ftwo_sum(d1, e1, &e2);

    a3 = _bncneon_fquick_two_sum(a2, d2, &d3);
    b3 = _bncneon_ftwo_sum(b2, c2, &c3);

    c4 = vaddq_f32(c3, e2);
    c5 = _bncneon_ftwo_sum(c4, d3, &d5);

    b6 = _bncneon_ftwo_sum(b3, c5, &c6);
    a7 = _bncneon_fquick_two_sum(a3, b6, &b7);

    c7 = vaddq_f32(c6, d5);
    b8 = _bncneon_fquick_two_sum(b7, c7, &c8);

    ret[0] = a7;
    ret[1] = b8;
    ret[2] = c8;
}

// Q version of TS add: real implementation provided in _bncneon_qs.h
// (uses _bncneon_frenorm / _bncneon_fthree_sum which are defined there).
// We only forward-declare here; the body is defined after qs.h is included.

// Triple-single multiplication (alternative implementation)
static inline void _bncneon_rts_mult(float32x4_t ret[TSSIZE], float32x4_t a[TSSIZE], float32x4_t b[TSSIZE])
{
    float32x4_t z00[2], z01[2], z10[2];
    float32x4_t in_b[3], in_c, z[3], e[4], temp[4];

    z00[0] = _bncneon_ftwo_prod(a[0], b[0], &z00[1]);
    z01[0] = _bncneon_ftwo_prod(a[0], b[1], &z01[1]);
    z10[0] = _bncneon_ftwo_prod(a[1], b[0], &z10[1]);

    z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
    _bncneon_fvec_sum(in_b, z, 3);
    in_c = vfmaq_f32(in_b[2], a[1], b[1]);

    z[0] = vfmaq_f32(z10[1], a[0], b[2]);
    z[1] = vfmaq_f32(z01[1], a[2], b[0]);
    z[2] = vaddq_f32(z[0], z[1]);

    temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1];
    temp[3] = vaddq_f32(in_c, z[2]);
    _bncneon_fvec_sum(e, temp, 4);
    ret[0] = e[0];
    _bncneon_fvseb(&ret[1], 2, &e[1], 3);
}

// Branch-free algorithm for TS mul
static inline void _bncneon_rts_mul_bf(float32x4_t ret[TSSIZE],
                                      const float32x4_t a[TSSIZE],
                                      const float32x4_t b[TSSIZE])
{
    float32x4_t a0, b0, c0, d0, e0, f0, g0, h0, i0;
    float32x4_t c1, d1, e1, g1;
    float32x4_t b2, c2, g2;
    float32x4_t a3, b3, c3, e3;
    float32x4_t c4;
    float32x4_t b5, c5;
    float32x4_t a6, b6;
    float32x4_t b7, c7;

    a0 = _bncneon_ftwo_prod(a[0], b[0], &b0);
    c0 = _bncneon_ftwo_prod(a[0], b[1], &e0);
    d0 = _bncneon_ftwo_prod(a[1], b[0], &f0);

    g0 = vmulq_f32(a[0], b[2]);
    h0 = vmulq_f32(a[1], b[1]);
    i0 = vmulq_f32(a[2], b[0]);

    c1 = _bncneon_ftwo_sum(c0, d0, &d1);
    e1 = vaddq_f32(e0, f0);
    g1 = vaddq_f32(g0, i0);

    b2 = _bncneon_ftwo_sum(b0, c1, &c2);
    g2 = vaddq_f32(g1, h0);

    a3 = _bncneon_fquick_two_sum(a0, b2, &b3);
    c3 = vaddq_f32(c2, d1);
    e3 = vaddq_f32(e1, g2);

    c4 = vaddq_f32(c3, e3);
    b5 = _bncneon_fquick_two_sum(b3, c4, &c5);

    a6 = _bncneon_fquick_two_sum(a3, b5, &b6);
    b7 = _bncneon_fquick_two_sum(b6, c5, &c7);

    ret[0] = a6;
    ret[1] = b7;
    ret[2] = c7;
}

// Q version of TS mul: real implementation provided in _bncneon_qs.h
// (uses _bncneon_frenorm / _bncneon_fthree_sum / _bncneon_fthree_sum2
// which are defined there). Forward declaration only here.

// ret := -a (triple-single negation)
static inline void _bncneon_rts_neg(float32x4_t c[TSSIZE], float32x4_t a[TSSIZE])
{
    c[0] = vnegq_f32(a[0]);
    c[1] = vnegq_f32(a[1]);
    c[2] = vnegq_f32(a[2]);
}

// ret := a - b (triple-single subtraction)
static inline void _bncneon_rts_sub(float32x4_t c[TSSIZE], float32x4_t a[TSSIZE], float32x4_t b[TSSIZE])
{
    float32x4_t mb[TSSIZE];

    _bncneon_rts_neg(mb, b);
    _bncneon_rts_add(c, a, mb);
}

// ret := a - b (Q version)
static inline void _bncneon_rts_subq(float32x4_t c[TSSIZE], float32x4_t a[TSSIZE], float32x4_t b[TSSIZE])
{
    float32x4_t mb[TSSIZE];

    _bncneon_rts_neg(mb, b);
    _bncneon_rts_addq(c, a, mb);
}

// Forward declaration
static inline void _bncneon_rts_divq(float32x4_t ret[TSSIZE], float32x4_t a[TSSIZE], float32x4_t b[TSSIZE]);

// r[3] := to_ts(a, b, c)
static inline void _bncneon_to_ts(float32x4_t r[TSSIZE], float32x4_t a, float32x4_t b, float32x4_t c)
{
    float32x4_t d[3], e[3];

    d[0] = _bncneon_ftwo_sum(a, b, &d[1]);
    d[2] = c;
    _bncneon_fvec_sum(e, d, 3);
    _bncneon_fvseb(r, 3, e, 3);
}

// c[3] := a * b[3]  (float * triple-single)
static inline void _bncneon_rts_mul_f(float32x4_t c[TSSIZE], float32x4_t a, float32x4_t b[TSSIZE])
{
    float32x4_t z00[2], z01[2];
    float32x4_t in_b[3], z[3], e[4], temp[4];

    z00[0] = _bncneon_ftwo_prod(a, b[0], &z00[1]);
    z01[0] = _bncneon_ftwo_prod(a, b[1], &z01[1]);

    z[0] = z00[1]; z[1] = z01[0];
    _bncneon_fvec_sum(in_b, z, 2);

    z[0] = vmulq_f32(a, b[2]);
    z[1] = vaddq_f32(z[0], z01[1]);
    temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; temp[3] = z[1];
    _bncneon_fvec_sum(e, temp, 4);
    c[0] = e[0];
    _bncneon_fvseb(&c[1], 2, &e[1], 3);
}

// c[3] := a[2] * b[3]  (double-single * triple-single)
static inline void _bncneon_rts_mul_ds(float32x4_t c[TSSIZE], float32x4_t a[DSSIZE], float32x4_t b[TSSIZE])
{
    float32x4_t z00[2], z01[2], z10[2];
    float32x4_t in_b[3], in_c, z[3], e[4], temp[4];

    z00[0] = _bncneon_ftwo_prod(a[0], b[0], &z00[1]);
    z01[0] = _bncneon_ftwo_prod(a[0], b[1], &z01[1]);
    z10[0] = _bncneon_ftwo_prod(a[1], b[0], &z10[1]);
    z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
    _bncneon_fvec_sum(in_b, z, 3);
    in_c = vfmaq_f32(in_b[2], a[1], b[1]);

    z[0] = vfmaq_f32(z10[1], a[0], b[2]);
    z[1] = vaddq_f32(z[0], z01[1]);
    temp[0] = z00[0];
    temp[1] = in_b[0];
    temp[2] = in_b[1];
    temp[3] = vaddq_f32(in_c, z[1]);
    _bncneon_fvec_sum(e, temp, 4);
    c[0] = e[0];
    _bncneon_fvseb(&c[1], 2, &e[1], 3);
}

#define _bncneon_rts_div _bncneon_rts_divtq

// Triple-single division (implementation 1)
static inline void _bncneon_rts_divt(float32x4_t ret[TSSIZE], float32x4_t a[TSSIZE], float32x4_t b[TSSIZE])
{
    float32x4_t alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];
    float32x4_t zero4, two4, one_p_2flt_eps4, one_m_2flt_eps4;

    zero4 = vdupq_n_f32(0.0f);
    two4 = vdupq_n_f32(2.0f);
    one_p_2flt_eps4 = vdupq_n_f32(ONE_P_2FLT_EPS);
    one_m_2flt_eps4 = vdupq_n_f32(ONE_M_2FLT_EPS);

    _bncneon_to_ts(d2, two4, zero4, zero4);

    alpha = vdivq_f32(one_p_2flt_eps4, b[0]);

    h1 = vfmsq_f32(alpha, b[0], one_p_2flt_eps4);
    h1 = vnegq_f32(vfmaq_f32(h1, alpha, b[1]));

    in_b[0] = _bncneon_ftwo_prod(alpha, one_m_2flt_eps4, &in_b[1]);

    in_b12 = vfmaq_f32(in_b[1], alpha, h1);

    in_b[0] = _bncneon_fquick_two_sum(in_b[0], in_b12, &in_b[1]);

    _bncneon_rts_mul_ds(temp, in_b, b);
    _bncneon_rts_sub(temp, d2, temp);
    _bncneon_rts_mul_ds(in_c, in_b, a);
    _bncneon_rts_mul(ret, in_c, temp);
}

// Triple-single division (implementation 2)
static inline void _bncneon_rts_divtq(float32x4_t ret[TSSIZE], float32x4_t a[TSSIZE], float32x4_t b[TSSIZE])
{
    float32x4_t alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];
    float32x4_t zero4, two4, one_p_2flt_eps4, one_m_2flt_eps4;

    zero4 = vdupq_n_f32(0.0f);
    two4 = vdupq_n_f32(2.0f);
    one_p_2flt_eps4 = vdupq_n_f32(ONE_P_2FLT_EPS);
    one_m_2flt_eps4 = vdupq_n_f32(ONE_M_2FLT_EPS);

    _bncneon_to_ts(d2, two4, zero4, zero4);

    alpha = vdivq_f32(one_p_2flt_eps4, b[0]);

    h1 = vfmsq_f32(alpha, b[0], one_p_2flt_eps4);
    h1 = vnegq_f32(vfmaq_f32(h1, alpha, b[1]));

    in_b[0] = _bncneon_ftwo_prod(alpha, one_m_2flt_eps4, &in_b[1]);

    in_b12 = vfmaq_f32(in_b[1], alpha, h1);

    in_b[0] = _bncneon_fquick_two_sum(in_b[0], in_b12, &in_b[1]);

    _bncneon_rts_mul_ds(temp, in_b, b);
    _bncneon_rts_subq(temp, d2, temp);
    _bncneon_rts_mul_ds(in_c, in_b, a);
    _bncneon_rts_mul(ret, in_c, temp);
}

// Q version of TS div
static inline void _bncneon_rts_divq(float32x4_t ret[TSSIZE], float32x4_t a[TSSIZE], float32x4_t b[TSSIZE])
{
    _bncneon_rts_divtq(ret, a, b);
}

#endif // ifndef __BNCNEON_TS_H
