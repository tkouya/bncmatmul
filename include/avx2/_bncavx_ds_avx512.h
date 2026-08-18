// ------------------------
// ---------- DS AVX-512 --
// ------------------------
#ifndef __BNCAVX_DS_AVX512_H
#define __BNCAVX_DS_AVX512_H

// defined in rds.h
#ifndef DSSIZE
    #define DSSIZE 2
#endif // DSSIZE

#if defined(__AVX512F__)

// ret := 0
// NOTE: _bncavx512_set0_ds is already defined in _bncavx_ds.h
#define _bncavx512_rds_set0(ret) _bncavx512_set0_ds((ret))
// ret := ret16[][avx_index]

// ret := val
static inline void _bncavx512_rds_set(__m512 ret[DSSIZE], __m512 val[DSSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
}

// ret := (float)val
static inline void _bncavx512_rds_set_d(__m512 ret[DSSIZE], __m512 val)
{
    ret[0] = val;
    ret[1] = _mm512_setzero_ps(); // val[1];
}

// ret := [val val val val]
static inline void _bncavx512_rds_set1_ds(__m512 ret[DSSIZE], float val[DSSIZE])
{
    ret[0] = _mm512_set1_ps(val[0]);
    ret[1] = _mm512_set1_ps(val[1]);
}

static inline void _bncavx512_get_ds_m512_i(dsfloat *ret, __m512 ret8[DSSIZE], int avx_index)
{
    ret->val[0] = ret8[0][avx_index];
    ret->val[1] = ret8[1][avx_index];

    return;
}

// ret := ret8[0] + ret8[1] + ... + ret8[15]
static void _bncavx512_rds_sum512d(float ret[DSSIZE], __m512 ret8[DSSIZE])
{
    dsfloat ret16_i[16];

    // ret16_i := ret8 (16 lanes)
    _bncavx512_get_ds_m512_i(&ret16_i[0], ret8, 0);
    _bncavx512_get_ds_m512_i(&ret16_i[1], ret8, 1);
    _bncavx512_get_ds_m512_i(&ret16_i[2], ret8, 2);
    _bncavx512_get_ds_m512_i(&ret16_i[3], ret8, 3);
    _bncavx512_get_ds_m512_i(&ret16_i[4], ret8, 4);
    _bncavx512_get_ds_m512_i(&ret16_i[5], ret8, 5);
    _bncavx512_get_ds_m512_i(&ret16_i[6], ret8, 6);
    _bncavx512_get_ds_m512_i(&ret16_i[7], ret8, 7);
    _bncavx512_get_ds_m512_i(&ret16_i[8], ret8, 8);
    _bncavx512_get_ds_m512_i(&ret16_i[9], ret8, 9);
    _bncavx512_get_ds_m512_i(&ret16_i[10], ret8, 10);
    _bncavx512_get_ds_m512_i(&ret16_i[11], ret8, 11);
    _bncavx512_get_ds_m512_i(&ret16_i[12], ret8, 12);
    _bncavx512_get_ds_m512_i(&ret16_i[13], ret8, 13);
    _bncavx512_get_ds_m512_i(&ret16_i[14], ret8, 14);
    _bncavx512_get_ds_m512_i(&ret16_i[15], ret8, 15);

    rds_set(ret, ret16_i[0].val);
    rds_add(ret, ret, ret16_i[1].val);
    rds_add(ret, ret, ret16_i[2].val);
    rds_add(ret, ret, ret16_i[3].val);
    rds_add(ret, ret, ret16_i[4].val);
    rds_add(ret, ret, ret16_i[5].val);
    rds_add(ret, ret, ret16_i[6].val);
    rds_add(ret, ret, ret16_i[7].val);
    rds_add(ret, ret, ret16_i[8].val);
    rds_add(ret, ret, ret16_i[9].val);
    rds_add(ret, ret, ret16_i[10].val);
    rds_add(ret, ret, ret16_i[11].val);
    rds_add(ret, ret, ret16_i[12].val);
    rds_add(ret, ret, ret16_i[13].val);
    rds_add(ret, ret, ret16_i[14].val);
    rds_add(ret, ret, ret16_i[15].val);
}

// ret := |ret8[0]| + |ret8[1]| + ... + |ret8[15]|
static void _bncavx512_rds_abssum512d(float ret[DSSIZE], __m512 ret8[DSSIZE])
{
    dsfloat ret16_i[16];
    float tmp[DSSIZE];

    // ret16_i := ret8 (16 lanes)
    _bncavx512_get_ds_m512_i(&ret16_i[0], ret8, 0);
    _bncavx512_get_ds_m512_i(&ret16_i[1], ret8, 1);
    _bncavx512_get_ds_m512_i(&ret16_i[2], ret8, 2);
    _bncavx512_get_ds_m512_i(&ret16_i[3], ret8, 3);
    _bncavx512_get_ds_m512_i(&ret16_i[4], ret8, 4);
    _bncavx512_get_ds_m512_i(&ret16_i[5], ret8, 5);
    _bncavx512_get_ds_m512_i(&ret16_i[6], ret8, 6);
    _bncavx512_get_ds_m512_i(&ret16_i[7], ret8, 7);
    _bncavx512_get_ds_m512_i(&ret16_i[8], ret8, 8);
    _bncavx512_get_ds_m512_i(&ret16_i[9], ret8, 9);
    _bncavx512_get_ds_m512_i(&ret16_i[10], ret8, 10);
    _bncavx512_get_ds_m512_i(&ret16_i[11], ret8, 11);
    _bncavx512_get_ds_m512_i(&ret16_i[12], ret8, 12);
    _bncavx512_get_ds_m512_i(&ret16_i[13], ret8, 13);
    _bncavx512_get_ds_m512_i(&ret16_i[14], ret8, 14);
    _bncavx512_get_ds_m512_i(&ret16_i[15], ret8, 15);

    rds_abs(tmp, ret16_i[0].val);  rds_set(ret, tmp);
    rds_abs(tmp, ret16_i[1].val);  rds_add(ret, ret, tmp);
    rds_abs(tmp, ret16_i[2].val);  rds_add(ret, ret, tmp);
    rds_abs(tmp, ret16_i[3].val);  rds_add(ret, ret, tmp);
    rds_abs(tmp, ret16_i[4].val);  rds_add(ret, ret, tmp);
    rds_abs(tmp, ret16_i[5].val);  rds_add(ret, ret, tmp);
    rds_abs(tmp, ret16_i[6].val);  rds_add(ret, ret, tmp);
    rds_abs(tmp, ret16_i[7].val);  rds_add(ret, ret, tmp);
    rds_abs(tmp, ret16_i[8].val);  rds_add(ret, ret, tmp);
    rds_abs(tmp, ret16_i[9].val);  rds_add(ret, ret, tmp);
    rds_abs(tmp, ret16_i[10].val); rds_add(ret, ret, tmp);
    rds_abs(tmp, ret16_i[11].val); rds_add(ret, ret, tmp);
    rds_abs(tmp, ret16_i[12].val); rds_add(ret, ret, tmp);
    rds_abs(tmp, ret16_i[13].val); rds_add(ret, ret, tmp);
    rds_abs(tmp, ret16_i[14].val); rds_add(ret, ret, tmp);
    rds_abs(tmp, ret16_i[15].val); rds_add(ret, ret, tmp);
}

// ret := max(|ret8[0]|, |ret8[1]|, ..., |ret8[15]|)
static void _bncavx512_rds_absmax512d(float ret[DSSIZE], __m512 ret8[DSSIZE])
{
    dsfloat ret16_i[16];
    float tmp[DSSIZE];

    // ret16_i := ret8 (16 lanes)
    _bncavx512_get_ds_m512_i(&ret16_i[0], ret8, 0);
    _bncavx512_get_ds_m512_i(&ret16_i[1], ret8, 1);
    _bncavx512_get_ds_m512_i(&ret16_i[2], ret8, 2);
    _bncavx512_get_ds_m512_i(&ret16_i[3], ret8, 3);
    _bncavx512_get_ds_m512_i(&ret16_i[4], ret8, 4);
    _bncavx512_get_ds_m512_i(&ret16_i[5], ret8, 5);
    _bncavx512_get_ds_m512_i(&ret16_i[6], ret8, 6);
    _bncavx512_get_ds_m512_i(&ret16_i[7], ret8, 7);
    _bncavx512_get_ds_m512_i(&ret16_i[8], ret8, 8);
    _bncavx512_get_ds_m512_i(&ret16_i[9], ret8, 9);
    _bncavx512_get_ds_m512_i(&ret16_i[10], ret8, 10);
    _bncavx512_get_ds_m512_i(&ret16_i[11], ret8, 11);
    _bncavx512_get_ds_m512_i(&ret16_i[12], ret8, 12);
    _bncavx512_get_ds_m512_i(&ret16_i[13], ret8, 13);
    _bncavx512_get_ds_m512_i(&ret16_i[14], ret8, 14);
    _bncavx512_get_ds_m512_i(&ret16_i[15], ret8, 15);

    rds_abs(tmp, ret16_i[0].val);
    rds_set(ret, tmp); // ret := |ret16_i[0]|

    rds_abs(tmp, ret16_i[1].val);  if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp);
    rds_abs(tmp, ret16_i[2].val);  if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp);
    rds_abs(tmp, ret16_i[3].val);  if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp);
    rds_abs(tmp, ret16_i[4].val);  if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp);
    rds_abs(tmp, ret16_i[5].val);  if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp);
    rds_abs(tmp, ret16_i[6].val);  if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp);
    rds_abs(tmp, ret16_i[7].val);  if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp);
    rds_abs(tmp, ret16_i[8].val);  if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp);
    rds_abs(tmp, ret16_i[9].val);  if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp);
    rds_abs(tmp, ret16_i[10].val); if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp);
    rds_abs(tmp, ret16_i[11].val); if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp);
    rds_abs(tmp, ret16_i[12].val); if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp);
    rds_abs(tmp, ret16_i[13].val); if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp);
    rds_abs(tmp, ret16_i[14].val); if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp);
    rds_abs(tmp, ret16_i[15].val); if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp);
}

// ret := || ret8[0]^2 + ret8[1]^2 + ... + ret8[15]^2 ||_2
static void _bncavx512_rds_norm512d(float ret[DSSIZE], __m512 ret8[DSSIZE])
{
    dsfloat ret16_i[16];
    float tmp[DSSIZE];

    // ret16_i := ret8 (16 lanes)
    _bncavx512_get_ds_m512_i(&ret16_i[0], ret8, 0);
    _bncavx512_get_ds_m512_i(&ret16_i[1], ret8, 1);
    _bncavx512_get_ds_m512_i(&ret16_i[2], ret8, 2);
    _bncavx512_get_ds_m512_i(&ret16_i[3], ret8, 3);
    _bncavx512_get_ds_m512_i(&ret16_i[4], ret8, 4);
    _bncavx512_get_ds_m512_i(&ret16_i[5], ret8, 5);
    _bncavx512_get_ds_m512_i(&ret16_i[6], ret8, 6);
    _bncavx512_get_ds_m512_i(&ret16_i[7], ret8, 7);
    _bncavx512_get_ds_m512_i(&ret16_i[8], ret8, 8);
    _bncavx512_get_ds_m512_i(&ret16_i[9], ret8, 9);
    _bncavx512_get_ds_m512_i(&ret16_i[10], ret8, 10);
    _bncavx512_get_ds_m512_i(&ret16_i[11], ret8, 11);
    _bncavx512_get_ds_m512_i(&ret16_i[12], ret8, 12);
    _bncavx512_get_ds_m512_i(&ret16_i[13], ret8, 13);
    _bncavx512_get_ds_m512_i(&ret16_i[14], ret8, 14);
    _bncavx512_get_ds_m512_i(&ret16_i[15], ret8, 15);

    rds_mul(tmp, ret16_i[0].val, ret16_i[0].val);   rds_set(ret, tmp);
    rds_mul(tmp, ret16_i[1].val, ret16_i[1].val);   rds_add(ret, ret, tmp);
    rds_mul(tmp, ret16_i[2].val, ret16_i[2].val);   rds_add(ret, ret, tmp);
    rds_mul(tmp, ret16_i[3].val, ret16_i[3].val);   rds_add(ret, ret, tmp);
    rds_mul(tmp, ret16_i[4].val, ret16_i[4].val);   rds_add(ret, ret, tmp);
    rds_mul(tmp, ret16_i[5].val, ret16_i[5].val);   rds_add(ret, ret, tmp);
    rds_mul(tmp, ret16_i[6].val, ret16_i[6].val);   rds_add(ret, ret, tmp);
    rds_mul(tmp, ret16_i[7].val, ret16_i[7].val);   rds_add(ret, ret, tmp);
    rds_mul(tmp, ret16_i[8].val, ret16_i[8].val);   rds_add(ret, ret, tmp);
    rds_mul(tmp, ret16_i[9].val, ret16_i[9].val);   rds_add(ret, ret, tmp);
    rds_mul(tmp, ret16_i[10].val, ret16_i[10].val); rds_add(ret, ret, tmp);
    rds_mul(tmp, ret16_i[11].val, ret16_i[11].val); rds_add(ret, ret, tmp);
    rds_mul(tmp, ret16_i[12].val, ret16_i[12].val); rds_add(ret, ret, tmp);
    rds_mul(tmp, ret16_i[13].val, ret16_i[13].val); rds_add(ret, ret, tmp);
    rds_mul(tmp, ret16_i[14].val, ret16_i[14].val); rds_add(ret, ret, tmp);
    rds_mul(tmp, ret16_i[15].val, ret16_i[15].val); rds_add(ret, ret, tmp);

    rds_sqrt(tmp, ret);
    rds_set(ret, tmp);
}

#ifndef USE_DS_BF

// ret := a + b
static inline void _bncavx512_rds_add(__m512 ret[DSSIZE], __m512 a[DSSIZE], __m512 b[DSSIZE])
{
    // c_ds_add_sloppy
    __m512 s, e;

    s = _bncavx512_ftwo_sum(a[0], b[0], &e);
    e = _mm512_add_ps(e, _mm512_add_ps(a[1], b[1]));

    ret[0] = _bncavx512_fquick_two_sum(s, e, &e);
    ret[1] = e;
}

#else // USE_DS_BF
// Branch free algorithm
static inline void _bncavx512_rds_add(__m512 ret[DSSIZE], __m512 a[DSSIZE], __m512 b[DSSIZE])
{
    __m512 g1, g2, g3, g4, g5, g6;
    __m512 g1e, g2e, g3e, g6e;

    g1 = _bncavx512_ftwo_sum(a[0], b[0], &g1e);
    g2 = _bncavx512_ftwo_sum(a[1], b[1], &g2e);
    g3 = _bncavx512_fquick_two_sum(g1, g2, &g3e);
    g4 = _mm512_add_ps(g1e, g2e);
    g5 = _mm512_add_ps(g4, g3e);
    g6 = _bncavx512_fquick_two_sum(g3, g5, &g6e);
    ret[0] = g6;
    ret[1] = g6e;
}
#endif // USE_DS_BF

// c[2] := -a[2]
static inline void _bncavx512_rds_neg(__m512 c[DSSIZE], __m512 a[DSSIZE])
{
    __m512 zero8;

    zero8 = _mm512_setzero_ps();

    c[0] = _mm512_sub_ps(zero8, a[0]);
    c[1] = _mm512_sub_ps(zero8, a[1]);
}

// ret := a - b
static inline void _bncavx512_rds_sub(__m512 ret[DSSIZE], __m512 a[DSSIZE], __m512 b[DSSIZE])
{
    // c_ds_sub_sloppy
    __m512 s1, s2, t1, t2;

    s1 = _bncavx512_ftwo_diff(a[0], b[0], &s2);
    t1 = _bncavx512_ftwo_diff(a[1], b[1], &t2);

    s2 = _mm512_add_ps(s2, t1);
    s1 = _bncavx512_fquick_two_sum(s1, s2, &s2);
    s2 = _mm512_add_ps(s2, t2);
    s1 = _bncavx512_fquick_two_sum(s1, s2, &s2);
    ret[0] = s1;
    ret[1] = s2;
}

#ifndef USE_DS_BF

// mul
static inline void _bncavx512_rds_mul(__m512 ret[DSSIZE], __m512 a[DSSIZE], __m512 b[DSSIZE])
{
    __m512 p1, p2;

    p1 = _bncavx512_ftwo_prod(a[0], b[0], &p2);

    p2 = _mm512_add_ps(p2,
        _mm512_add_ps(
            _mm512_mul_ps(a[0], b[1]),
            _mm512_mul_ps(a[1], b[0])
        )
    );

    ret[0] = _bncavx512_fquick_two_sum(p1, p2, &p2);
    ret[1] = p2;
}

#else // USE_DS_BF
// Branch free algorithm
static inline void _bncavx512_rds_mul(__m512 ret[DSSIZE], __m512 a[DSSIZE], __m512 b[DSSIZE])
{
    __m512 p00, p01, p10;
    __m512 pe00;
    __m512 g1, g2, g3, ge3;

    p00 = _bncavx512_ftwo_prod(a[0], b[0], &pe00);
    p01 = _mm512_mul_ps(a[0], b[1]);
    p10 = _mm512_mul_ps(a[1], b[0]);
    g1 = _mm512_add_ps(p01, p10);
    g2 = _mm512_add_ps(pe00, g1);
    g3 = _bncavx512_fquick_two_sum(p00, g2, &ge3);

    ret[0] = g3;
    ret[1] = ge3;
}
#endif // USE_DS_BF

// ds * d
static inline void _bncavx512_rds_mul_d(__m512 ret[DSSIZE], __m512 a[DSSIZE], __m512 b)
{
    __m512 p1, p2;

    p1 = _bncavx512_ftwo_prod(a[0], b, &p2);

    p2 = _mm512_add_ps(p2,
        _mm512_mul_ps(a[1], b)
    );

    ret[0] = _bncavx512_fquick_two_sum(p1, p2, &p2);
    ret[1] = p2;
}

// sloppy_div
static inline void _bncavx512_rds_div(__m512 ret[DSSIZE], __m512 a[DSSIZE], __m512 b[DSSIZE])
{
    // c_ds_sloppy_div(float *a, float *b, float *c)
    __m512 s1, s2;
    __m512 q1, q2;
    __m512 r[DSSIZE];

    q1 = _mm512_div_ps(a[0], b[0]);

    _bncavx512_rds_mul_d(r, b, q1);

    s1 = _bncavx512_ftwo_diff(a[0], r[0], &s2);
    s2 = _mm512_sub_ps(s2, r[1]);
    s2 = _mm512_add_ps(s2, a[1]);

    q2 = _mm512_div_ps(
        _mm512_add_ps(s1, s2),
        b[0]
    );

    ret[0] = _bncavx512_fquick_two_sum(q1, q2, &ret[1]);
}

// abs
static inline void _bncavx512_rds_abs(__m512 ret[DSSIZE], __m512 a[DSSIZE])
{
    int avx_index;

    for(avx_index = 0; avx_index < 16; avx_index++)
    {
        if(a[0][avx_index] < 0.0)
        {
            ret[0][avx_index] = -a[0][avx_index];
            ret[1][avx_index] = -a[1][avx_index];
        }
        else
        {
            ret[0][avx_index] = a[0][avx_index];
            ret[1][avx_index] = a[1][avx_index];
        }
    }
}

/* ================================================================
 *  AVX-512 implementation  (_bncavx512_ prefix)
 *  Processes 16 DS operations per call
 * ================================================================ */

/* ----------------------------------------------------------------
 * _bncavx512_rds_fma_d_d
 *   ret[DSSIZE] := a * b + c   (scalar x scalar -> DS) x 16
 * ---------------------------------------------------------------- */
static inline void _bncavx512_rds_fma_d_d(
        __m512 ret[DSSIZE],
        __m512 a,
        __m512 b,
        __m512 c)
{
    ret[0] = _mm512_fmadd_ps(a, b, c);
    ret[1] = _mm512_fmadd_ps(a, b, _mm512_sub_ps(c, ret[0]));
}

/* ----------------------------------------------------------------
 * _bncavx512_rds_fma_d_ds
 *   ret[DSSIZE] := a * b + c[DSSIZE]  (scalar x scalar + DS -> DS) x 16
 * ---------------------------------------------------------------- */
static inline void _bncavx512_rds_fma_d_ds(
        __m512       ret[DSSIZE],
        __m512       a,
        __m512       b,
        const __m512 c[DSSIZE])
{
    __m512 hi = _mm512_fmadd_ps(a, b, c[0]);
    __m512 t  = _mm512_sub_ps(c[0], hi);
    __m512 e  = _mm512_fmadd_ps(a, b, t);
    __m512 f  = _mm512_add_ps(e, c[1]);
    ret[0] = _bncavx512_fquick_two_sum(hi, f, &(ret[1]));
}

/* ----------------------------------------------------------------
 * _bncavx512_rds_fma_dm_ds
 *   ret[DSSIZE] := a * b[DSSIZE] + c[DSSIZE]  (scalar x DS + DS -> DS) x 16
 * ---------------------------------------------------------------- */
static inline void _bncavx512_rds_fma_dm_ds(
        __m512       ret[DSSIZE],
        __m512       a,
        const __m512 b[DSSIZE],
        const __m512 c[DSSIZE])
{
    __m512 hi = _mm512_fmadd_ps(a, b[0], c[0]);
    __m512 t  = _mm512_sub_ps(c[0], hi);
    __m512 e  = _mm512_fmadd_ps(a, b[0], t);
    __m512 f  = _mm512_add_ps(e, c[1]);
    __m512 lo = _mm512_fmadd_ps(a, b[1], f);
    ret[0] = _bncavx512_fquick_two_sum(hi, lo, &(ret[1]));
}

/* ----------------------------------------------------------------
 * _bncavx512_rds_fma
 *   ret[DSSIZE] := a[DSSIZE] * b[DSSIZE] + c[DSSIZE]  (DS x DS + DS -> DS) x 16
 * ---------------------------------------------------------------- */
static inline void _bncavx512_rds_fma(
        __m512       ret[DSSIZE],
        const __m512 a[DSSIZE],
        const __m512 b[DSSIZE],
        const __m512 c[DSSIZE])
{
    __m512 d0 = _mm512_fmadd_ps(a[0], b[0], c[0]);
    __m512 t  = _mm512_sub_ps(c[0], d0);
    __m512 e  = _mm512_fmadd_ps(a[0], b[0], t);
    __m512 f  = _mm512_add_ps(e, c[1]);
    __m512 g  = _mm512_fmadd_ps(a[0], b[1], f);
    __m512 d1 = _mm512_fmadd_ps(a[1], b[0], g);
    ret[0] = _bncavx512_fquick_two_sum(d0, d1, &(ret[1]));
}

// Aliases for naming variants used in *linear.c
#define _bncavx2_rds_sum512d   _bncavx512_rds_sum512d
#define _bncavx512_rds_sum512  _bncavx512_rds_sum512d
#define _bncavx512_rds_abssum512  _bncavx512_rds_abssum512d
#define _bncavx512_rds_norm512    _bncavx512_rds_norm512d

#endif // defined(__AVX512F__)

#ifdef __BNC_DSLINEAR_H__
//#include "dsv_addmul.c"
#endif //__BNC_DSLINEAR_H__

#endif // ifndef __BNCAVX_DS_AVX512_H
