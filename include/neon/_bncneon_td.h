// ------------------------
// -------- NEON TD -------
// ------------------------
#ifndef __BNCNEON_TD_H
#define __BNCNEON_TD_H

#ifndef TDSIZE
    #define TDSIZE 3
#endif // TDSIZE

// Constants for division algorithms
#ifndef ONE_P_2DBL_EPS
#define ONE_P_2DBL_EPS 1.0000000000000002
#endif
#ifndef ONE_M_2DBL_EPS  
#define ONE_M_2DBL_EPS 0.9999999999999998
#endif

// ret := 0
static inline void _bncneon_set0_td(float64x2_t ret[TDSIZE])
{
    ret[0] = vdupq_n_f64(0.0);
    ret[1] = vdupq_n_f64(0.0);
    ret[2] = vdupq_n_f64(0.0);
}
#define _bncneon_rtd_set0(ret) _bncneon_set0_td((ret))

// ret := val
static inline void _bncneon_rtd_set(float64x2_t ret[TDSIZE], float64x2_t val[TDSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = val[2];
}

// ret := (double)val
static inline void _bncneon_rtd_set_d(float64x2_t ret[TDSIZE], float64x2_t val)
{
    ret[0] = val;
    ret[1] = vdupq_n_f64(0.0);
    ret[2] = vdupq_n_f64(0.0);
}

// ret := (DD)val
static inline void _bncneon_rtd_set_dd(float64x2_t ret[TDSIZE], float64x2_t val[DDSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = vdupq_n_f64(0.0);
}

// ret := [val val]
static inline void _bncneon_rtd_set1_td(float64x2_t ret[TDSIZE], double val[TDSIZE])
{
    ret[0] = vdupq_n_f64(val[0]);
    ret[1] = vdupq_n_f64(val[1]);
    ret[2] = vdupq_n_f64(val[2]);
}

#if 0
// ret := ret2[][neon_index]
static inline void _bncneon_get_td_f64x2_i(tdfloat *ret, float64x2_t ret2[TDSIZE], int neon_index)
{
    ret->val[0] = vgetq_lane_f64(ret2[0], neon_index);
    ret->val[1] = vgetq_lane_f64(ret2[1], neon_index);
    ret->val[2] = vgetq_lane_f64(ret2[2], neon_index);
    return;
}
#endif // 0

// ret := ret2[][neon_index]
static inline void _bncneon_get_td_f64x2_i_0(tdfloat *ret, float64x2_t ret2[TDSIZE]) // , int neon_index)
{
    ret->val[0] = vgetq_lane_f64(ret2[0], 0); // neon_index);
    ret->val[1] = vgetq_lane_f64(ret2[1], 0); // neon_index);
    ret->val[2] = vgetq_lane_f64(ret2[2], 0); // neon_index);
    return;
}
// ret := ret2[][neon_index]
static inline void _bncneon_get_td_f64x2_i_1(tdfloat *ret, float64x2_t ret2[TDSIZE]) // , int neon_index)
{
    ret->val[0] = vgetq_lane_f64(ret2[0], 1); // neon_index);
    ret->val[1] = vgetq_lane_f64(ret2[1], 1); // neon_index);
    ret->val[2] = vgetq_lane_f64(ret2[2], 1); // neon_index);
    return;
}


// ret := ret2[0] + ret2[1] (sum of 2-element vector)
static void _bncneon_rtd_sum128d(double ret[TDSIZE], float64x2_t ret2[TDSIZE])
{
    tdfloat ret2_i[2];

    // ret2_i := ret2
    //_bncneon_get_td_f64x2_i(&ret2_i[0], ret2, 0);
    //_bncneon_get_td_f64x2_i(&ret2_i[1], ret2, 1);
    _bncneon_get_td_f64x2_i_0(&ret2_i[0], ret2); //, 0);
    _bncneon_get_td_f64x2_i_1(&ret2_i[1], ret2); //, 1);


    rtd_set(ret, ret2_i[0].val);
    rtd_add(ret, ret, ret2_i[1].val);
}

// ret := |a| (triple-double absolute value)
static inline void _bncneon_rtd_abs(float64x2_t ret[TDSIZE], float64x2_t a[TDSIZE])
{
    // Use vector comparison to create mask for negative values
    uint64x2_t mask = vcltq_f64(a[0], vdupq_n_f64(0.0));
    
    // Apply conditional negation based on sign of a[0]
    ret[0] = vbslq_f64(mask, vnegq_f64(a[0]), a[0]);
    ret[1] = vbslq_f64(mask, vnegq_f64(a[1]), a[1]);
    ret[2] = vbslq_f64(mask, vnegq_f64(a[2]), a[2]);
}

// ret := |ret2[0]| + |ret2[1]| (absolute sum of 2-element vector)
static void _bncneon_rtd_abssum128d(double ret[TDSIZE], float64x2_t ret2[TDSIZE])
{
    tdfloat ret2_i[2];
    static double tmp[TDSIZE];

    // ret2_i := ret2
    _bncneon_get_td_f64x2_i_0(&ret2_i[0], ret2); //, 0);
    _bncneon_get_td_f64x2_i_1(&ret2_i[1], ret2); //, 1);

    rtd_abs(tmp, ret2_i[0].val);
    rtd_set(ret, tmp);

    rtd_abs(tmp, ret2_i[1].val);
    rtd_add(ret, ret, tmp);
}

// ret := max(|ret2[0]|, |ret2[1]|) (absolute maximum of 2-element vector)
static void _bncneon_rtd_absmax128d(double ret[TDSIZE], float64x2_t ret2[TDSIZE])
{
    tdfloat ret2_i[2];
    static double tmp[TDSIZE];

    // ret2_i := ret2
    // _bncneon_get_td_f64x2_i(&ret2_i[0], ret2, 0);
    // _bncneon_get_td_f64x2_i(&ret2_i[1], ret2, 1);
    _bncneon_get_td_f64x2_i_0(&ret2_i[0], ret2); //, 0);
    _bncneon_get_td_f64x2_i_1(&ret2_i[1], ret2); //, 1);

    rtd_abs(tmp, ret2_i[0].val); 
    rtd_set(ret, tmp); // ret:= |ret2_i[0]|

    rtd_abs(tmp, ret2_i[1].val);
    if(rtd_cmp(ret, tmp) < 0) // if(ret < |ret2_i[1]|)
        rtd_set(ret, tmp);    //   ret := |ret2_i[1]|
}

// ret := || ret2[0]^2 + ret2[1]^2 ||_2 (L2 norm of 2-element vector)
static void _bncneon_rtd_norm128d(double ret[TDSIZE], float64x2_t ret2[TDSIZE])
{
    tdfloat ret2_i[2];
    static double tmp[TDSIZE];

    // ret2_i := ret2
    //_bncneon_get_td_f64x2_i(&ret2_i[0], ret2, 0);
    //_bncneon_get_td_f64x2_i(&ret2_i[1], ret2, 1);
    _bncneon_get_td_f64x2_i_0(&ret2_i[0], ret2); //, 0);
    _bncneon_get_td_f64x2_i_1(&ret2_i[1], ret2); //, 1);

    rtd_mul(tmp, ret2_i[0].val, ret2_i[0].val);
    rtd_set(ret, tmp);

    rtd_mul(tmp, ret2_i[1].val, ret2_i[1].val);
    rtd_add(ret, ret, tmp);

    rtd_sqrt(tmp, ret);
    rtd_set(ret, tmp);
}

// Helper functions for triple-double arithmetic with NEON
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

// e[n] := vec_sum(x[n])
static inline void _bncneon_vec_sum(float64x2_t e[], const float64x2_t x[], int n)
{
    float64x2_t s;

    s = x[--n];
    while(--n >= 0)
    {
        s = _bncneon_dtwo_sum(x[n], s, &e[n + 1]);
    }
    e[0] = s;
}

// y[n] := vec_sum_err_branch(vseb)(k)(e[n])
static inline void _bncneon_vseb(float64x2_t y[], int ny, const float64x2_t e[], int ne)
{
    int i, j[2], neon_index;
    double dtmp;
    float64x2_t r, eps, temp, in_y[16];

    j[0] = 0; j[1] = 0;
    eps = e[0];
    for(i = 0; i < (ne - 2); i++)
    {
        r = _bncneon_dtwo_sum(eps, e[i + 1], &temp);

        // if(temp != 0.0) - process each lane
        for(neon_index = 0; neon_index < 2; neon_index++)
        {
            if(vgetq_lane_f64(temp, neon_index) != 0.0)
            {
                in_y[j[neon_index]] = vsetq_lane_f64(vgetq_lane_f64(r, neon_index), in_y[j[neon_index]], neon_index);
                eps = vsetq_lane_f64(vgetq_lane_f64(temp, neon_index), eps, neon_index);
                j[neon_index]++;
            }
            else
                eps = vsetq_lane_f64(vgetq_lane_f64(r, neon_index), eps, neon_index);
        }
    }

    for(neon_index = 0; neon_index < 2; neon_index++)
    {
        double eps_lane = vgetq_lane_f64(eps, neon_index);
        double e_lane = vgetq_lane_f64(e[ne - 1], neon_index);
        double sum = two_sum(eps_lane, e_lane, &dtmp);
        
        in_y[j[neon_index]] = vsetq_lane_f64(sum, in_y[j[neon_index]], neon_index);
        in_y[j[neon_index] + 1] = vsetq_lane_f64(dtmp, in_y[j[neon_index] + 1], neon_index);

        for(i = j[neon_index] + 2; i < ne; i++)
            in_y[i] = vsetq_lane_f64(0.0, in_y[i], neon_index);
    }

    for(i = 0; i < ny; i++)
        y[i] = in_y[i];
}

// Merge a[na] & b[nb] into c[na + nb]
static inline void _bncneon_merge(float64x2_t c[], float64x2_t a[], int na, float64x2_t b[], int nb)
{
    int i, j, k;
    int neon_index;

    for(neon_index = 0; neon_index < 2; neon_index++)
    {
        i = j = k = 0;
        while((i < na) && (j < nb))
        {
            double a_val = vgetq_lane_f64(a[i], neon_index);
            double b_val = vgetq_lane_f64(b[j], neon_index);
            
            if(fabs(a_val) >= fabs(b_val))
            {
                c[k] = vsetq_lane_f64(a_val, c[k], neon_index);
                i++;
            }
            else
            {
                c[k] = vsetq_lane_f64(b_val, c[k], neon_index);
                j++;
            }
            k++;
        }
        while(i < na)
        {
            c[k] = vsetq_lane_f64(vgetq_lane_f64(a[i], neon_index), c[k], neon_index);
            i++; k++;
        }
        while(j < nb)
        {
            c[k] = vsetq_lane_f64(vgetq_lane_f64(b[j], neon_index), c[k], neon_index);
            j++; k++;
        }
    }
}

// Triple-double arithmetic operations

#ifdef USE_TD_BF
    #define _bncneon_rtd_add _bncneon_rtd_add_bf
    #define _bncneon_rtd_mul _bncneon_rtd_mul_bf
#else // USE_TD_BF
    #define _bncneon_rtd_add _bncneon_rtd_addq
    #define _bncneon_rtd_mul _bncneon_rtd_mulq
#endif // USE_TD_BF
// Forward declarations
static inline void _bncneon_rtd_addq(float64x2_t ret[TDSIZE], float64x2_t a[TDSIZE], float64x2_t b[TDSIZE]);

// Triple-double addition (alternative implementation)
static inline void _bncneon_rtd_addt(float64x2_t ret[TDSIZE], float64x2_t a[TDSIZE], float64x2_t b[TDSIZE])
{
    float64x2_t z[6], e[6];

    _bncneon_merge(z, a, 3, b, 3);
    _bncneon_vec_sum(e, z, 6);
    _bncneon_vseb(ret, 3, e, 6);
}

// Branch free algorithm
static inline void _bncneon_rtd_add_bf(float64x2_t ret[TDSIZE],
                                      const float64x2_t a[TDSIZE],
                                      const float64x2_t b[TDSIZE])
{
    float64x2_t a0, b0, c0, d0, e0, f0;
    float64x2_t a1, b1, c1, d1, e1, f1;
    float64x2_t a2, b2, c2, d2, e2;
    float64x2_t a3, b3, c3, d3;
    float64x2_t c4, c5, d5;
    float64x2_t b6, c6;
    float64x2_t a7, b7, c7;
    float64x2_t b8, c8;

    a0 = a[0];
    b0 = b[0];
    c0 = a[1];
    d0 = b[1];
    e0 = a[2];
    f0 = b[2];

    a1 = _bncneon_dtwo_sum(a0, b0, &b1);
    c1 = _bncneon_dtwo_sum(c0, d0, &d1);
    e1 = _bncneon_dtwo_sum(e0, f0, &f1);

    a2 = _bncneon_dquick_two_sum(a1, c1, &c2);
    b2 = vaddq_f64(b1, f1);
    d2 = _bncneon_dtwo_sum(d1, e1, &e2);

    a3 = _bncneon_dquick_two_sum(a2, d2, &d3);
    b3 = _bncneon_dtwo_sum(b2, c2, &c3);

    c4 = vaddq_f64(c3, e2);
    c5 = _bncneon_dtwo_sum(c4, d3, &d5);

    b6 = _bncneon_dtwo_sum(b3, c5, &c6);
    a7 = _bncneon_dquick_two_sum(a3, b6, &b7);

    c7 = vaddq_f64(c6, d5);
    b8 = _bncneon_dquick_two_sum(b7, c7, &c8);

    // return MultiFloat<T, 3>{a7, b8, c8};
    ret[0] = a7;
    ret[1] = b8;
    ret[2] = c8;
}

//#define _bncneon_rtd_mul _bncneon_rtd_mulq

// Forward declarations
static inline void _bncneon_rtd_mulq(float64x2_t ret[TDSIZE], float64x2_t a[TDSIZE], float64x2_t b[TDSIZE]);

// Triple-double multiplication (alternative implementation)
static inline void _bncneon_rtd_mult(float64x2_t ret[TDSIZE], float64x2_t a[TDSIZE], float64x2_t b[TDSIZE])
{
    float64x2_t z00[2], z01[2], z10[2];
    float64x2_t in_b[3], in_c, z[3], e[4], temp[4];

    z00[0] = _bncneon_dtwo_prod(a[0], b[0], &z00[1]);
    z01[0] = _bncneon_dtwo_prod(a[0], b[1], &z01[1]);
    z10[0] = _bncneon_dtwo_prod(a[1], b[0], &z10[1]);

    z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
    _bncneon_vec_sum(in_b, z, 3);
    in_c = vfmaq_f64(in_b[2], a[1], b[1]);  // in_c = a[1] * b[1] + in_b[2]

    z[0] = vfmaq_f64(z10[1], a[0], b[2]);   // z[0] = a[0] * b[2] + z10[1]
    z[1] = vfmaq_f64(z01[1], a[2], b[0]);   // z[1] = a[2] * b[0] + z01[1]
    z[2] = vaddq_f64(z[0], z[1]);

    temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; 
    temp[3] = vaddq_f64(in_c, z[2]);
    _bncneon_vec_sum(e, temp, 4);
    ret[0] = e[0];
    _bncneon_vseb(&ret[1], 2, &e[1], 3);
}

// Branch free algorithm
static inline void _bncneon_rtd_mul_bf(float64x2_t ret[TDSIZE],
                                      const float64x2_t a[TDSIZE],
                                      const float64x2_t b[TDSIZE])
{
    float64x2_t a0, b0, c0, d0, e0, f0, g0, h0, i0;
    float64x2_t c1, d1, e1, g1;
    float64x2_t b2, c2, g2;
    float64x2_t a3, b3, c3, e3;
    float64x2_t c4;
    float64x2_t b5, c5;
    float64x2_t a6, b6;
    float64x2_t b7, c7;

    a0 = _bncneon_dtwo_prod(a[0], b[0], &b0);
    c0 = _bncneon_dtwo_prod(a[0], b[1], &e0);
    d0 = _bncneon_dtwo_prod(a[1], b[0], &f0);

    g0 = vmulq_f64(a[0], b[2]);
    h0 = vmulq_f64(a[1], b[1]);
    i0 = vmulq_f64(a[2], b[0]);

    c1 = _bncneon_dtwo_sum(c0, d0, &d1);
    e1 = vaddq_f64(e0, f0);
    g1 = vaddq_f64(g0, i0);

    b2 = _bncneon_dtwo_sum(b0, c1, &c2);
    g2 = vaddq_f64(g1, h0);

    a3 = _bncneon_dquick_two_sum(a0, b2, &b3);
    c3 = vaddq_f64(c2, d1);
    e3 = vaddq_f64(e1, g2);

    c4 = vaddq_f64(c3, e3);
    b5 = _bncneon_dquick_two_sum(b3, c4, &c5);

    a6 = _bncneon_dquick_two_sum(a3, b5, &b6);
    b7 = _bncneon_dquick_two_sum(b6, c5, &c7);

    // return MultiFloat<T, 3>{a6, b7, c7};
    ret[0] = a6;
    ret[1] = b7;
    ret[2] = c7;
}

// ret := -a (triple-double negation)
static inline void _bncneon_rtd_neg(float64x2_t c[TDSIZE], float64x2_t a[TDSIZE])
{
    c[0] = vnegq_f64(a[0]);
    c[1] = vnegq_f64(a[1]);
    c[2] = vnegq_f64(a[2]);
}

// ret := a - b (triple-double subtraction)
static inline void _bncneon_rtd_sub(float64x2_t c[TDSIZE], float64x2_t a[TDSIZE], float64x2_t b[TDSIZE])
{
    float64x2_t mb[TDSIZE];

    _bncneon_rtd_neg(mb, b);
    _bncneon_rtd_add(c, a, mb);
}

// ret := a - b (triple-double subtraction, Q version)
static inline void _bncneon_rtd_subq(float64x2_t c[TDSIZE], float64x2_t a[TDSIZE], float64x2_t b[TDSIZE])
{
    float64x2_t mb[TDSIZE];

    _bncneon_rtd_neg(mb, b);
    _bncneon_rtd_addq(c, a, mb);
}

// Forward declaration
static inline void _bncneon_rtd_divq(float64x2_t ret[TDSIZE], float64x2_t a[TDSIZE], float64x2_t b[TDSIZE]);

// r[3] := to_td(a, b, c)
static inline void _bncneon_to_td(float64x2_t r[TDSIZE], float64x2_t a, float64x2_t b, float64x2_t c)
{
    float64x2_t d[3], e[3];

    d[0] = _bncneon_dtwo_sum(a, b, &d[1]);
    d[2] = c;
    _bncneon_vec_sum(e, d, 3);
    _bncneon_vseb(r, 3, e, 3);
}

// c[3] := a * b[3] (double * triple-double multiplication)
static inline void _bncneon_rtd_mul_d(float64x2_t c[TDSIZE], float64x2_t a, float64x2_t b[TDSIZE])
{
    float64x2_t z00[2], z01[2];
    float64x2_t in_b[3], z[3], e[4], temp[4];

    z00[0] = _bncneon_dtwo_prod(a, b[0], &z00[1]);
    z01[0] = _bncneon_dtwo_prod(a, b[1], &z01[1]);

    z[0] = z00[1]; z[1] = z01[0];
    _bncneon_vec_sum(in_b, z, 2);

    z[0] = vmulq_f64(a, b[2]);  // Simplified - should use FMA if available
    z[1] = vaddq_f64(z[0], z01[1]);
    temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; temp[3] = z[1];
    _bncneon_vec_sum(e, temp, 4);
    c[0] = e[0];
    _bncneon_vseb(&c[1], 2, &e[1], 3);
}

// c[3] := a[2] * b[3] (double-double * triple-double multiplication)
static inline void _bncneon_rtd_mul_dd(float64x2_t c[TDSIZE], float64x2_t a[DDSIZE], float64x2_t b[TDSIZE])
{
    float64x2_t z00[2], z01[2], z10[2];
    float64x2_t in_b[3], in_c, z[3], e[4], temp[4];

    z00[0] = _bncneon_dtwo_prod(a[0], b[0], &z00[1]);
    z01[0] = _bncneon_dtwo_prod(a[0], b[1], &z01[1]);
    z10[0] = _bncneon_dtwo_prod(a[1], b[0], &z10[1]);
    z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
    _bncneon_vec_sum(in_b, z, 3);
    in_c = vfmaq_f64(in_b[2], a[1], b[1]);

    z[0] = vfmaq_f64(z10[1], a[0], b[2]);
    z[1] = vaddq_f64(z[0], z01[1]);
    temp[0] = z00[0];
    temp[1] = in_b[0];
    temp[2] = in_b[1];
    temp[3] = vaddq_f64(in_c, z[1]);
    _bncneon_vec_sum(e, temp, 4);
    c[0] = e[0];
    _bncneon_vseb(&c[1], 2, &e[1], 3);
}

#define _bncneon_rtd_div _bncneon_rtd_divtq

// Triple-double division (implementation 1)
static inline void _bncneon_rtd_divt(float64x2_t ret[TDSIZE], float64x2_t a[TDSIZE], float64x2_t b[TDSIZE])
{
    float64x2_t alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];
    float64x2_t zero2, two2, one_p_2dbl_eps2, one_m_2dbl_eps2;

    zero2 = vdupq_n_f64(0.0);
    two2 = vdupq_n_f64(2.0);
    one_p_2dbl_eps2 = vdupq_n_f64(ONE_P_2DBL_EPS);
    one_m_2dbl_eps2 = vdupq_n_f64(ONE_M_2DBL_EPS);

    _bncneon_to_td(d2, two2, zero2, zero2);

    alpha = vdivq_f64(one_p_2dbl_eps2, b[0]);

    h1 = vfmsq_f64(alpha, b[0], one_p_2dbl_eps2);  // h1 = alpha * b[0] - one_p_2dbl_eps2
    h1 = vnegq_f64(vfmaq_f64(h1, alpha, b[1]));    // h1 = -(alpha * b[1] + h1)

    in_b[0] = _bncneon_dtwo_prod(alpha, one_m_2dbl_eps2, &in_b[1]);

    in_b12 = vfmaq_f64(in_b[1], alpha, h1);

    in_b[0] = _bncneon_dquick_two_sum(in_b[0], in_b12, &in_b[1]);

    _bncneon_rtd_mul_dd(temp, in_b, b);
    _bncneon_rtd_sub(temp, d2, temp);
    _bncneon_rtd_mul_dd(in_c, in_b, a);
    _bncneon_rtd_mul(ret, in_c, temp);
}

// Triple-double division (implementation 2)
static inline void _bncneon_rtd_divtq(float64x2_t ret[TDSIZE], float64x2_t a[TDSIZE], float64x2_t b[TDSIZE])
{
    float64x2_t alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];
    float64x2_t zero2, two2, one_p_2dbl_eps2, one_m_2dbl_eps2;

    zero2 = vdupq_n_f64(0.0);
    two2 = vdupq_n_f64(2.0);
    one_p_2dbl_eps2 = vdupq_n_f64(ONE_P_2DBL_EPS);
    one_m_2dbl_eps2 = vdupq_n_f64(ONE_M_2DBL_EPS);

    _bncneon_to_td(d2, two2, zero2, zero2);

    alpha = vdivq_f64(one_p_2dbl_eps2, b[0]);

    h1 = vfmsq_f64(alpha, b[0], one_p_2dbl_eps2);
    h1 = vnegq_f64(vfmaq_f64(h1, alpha, b[1]));

    in_b[0] = _bncneon_dtwo_prod(alpha, one_m_2dbl_eps2, &in_b[1]);

    in_b12 = vfmaq_f64(in_b[1], alpha, h1);

    in_b[0] = _bncneon_dquick_two_sum(in_b[0], in_b12, &in_b[1]);

    _bncneon_rtd_mul_dd(temp, in_b, b);
    _bncneon_rtd_subq(temp, d2, temp); // Use Q version
    _bncneon_rtd_mul_dd(in_c, in_b, a);
    _bncneon_rtd_mul(ret, in_c, temp);
}

#include "sort.h" // _bncneon_bubble_sort (if needed)

#endif // ifndef __BNCNEON_TD_H