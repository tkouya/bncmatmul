// ------------------------
// ------ QS AVX-512 ------
// ------------------------
#ifndef __BNCAVX_QS_AVX512_H
#define __BNCAVX_QS_AVX512_H

#ifndef QSSIZE
    #define QSSIZE 4
#endif // QSSIZE

#if defined(__AVX512F__)

// ret := 0
// NOTE: _bncavx512_set0_qs is already defined in _bncavx_qs.h
#define _bncavx512_rqs_set0(ret) _bncavx512_set0_qs((ret))

// ret := val
static inline void _bncavx512_rqs_set(__m512 ret[QSSIZE], __m512 val[QSSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = val[2];
    ret[3] = val[3];
}

// ret := (float)val
static inline void _bncavx512_rqs_set_d(__m512 ret[QSSIZE], __m512 val)
{
    ret[0] = val;
    ret[1] = _mm512_setzero_ps(); // val[1];
    ret[2] = _mm512_setzero_ps(); // val[2];
    ret[3] = _mm512_setzero_ps(); // val[3];
}

// ret := (DS)val
static inline void _bncavx512_rqs_set_ds(__m512 ret[QSSIZE], __m512 val[DSSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = _mm512_setzero_ps(); // val[2];
    ret[3] = _mm512_setzero_ps(); // val[3];
}

// ret := (TS)val
static inline void _bncavx512_rqs_set_ts(__m512 ret[QSSIZE], __m512 val[TSSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = val[2];
    ret[3] = _mm512_setzero_ps(); // val[3];
}

// ret := [val val val val]
static inline void _bncavx512_rqs_set1_qs(__m512 ret[QSSIZE], float val[QSSIZE])
{
    ret[0] = _mm512_set1_ps(val[0]);
    ret[1] = _mm512_set1_ps(val[1]);
    ret[2] = _mm512_set1_ps(val[2]);
    ret[3] = _mm512_set1_ps(val[3]);
}

// ret := ret8[][avx_index]
static inline void _bncavx512_get_qs_m512_i(qsfloat *ret, __m512 ret8[QSSIZE], int avx_index)
{
    ret->val[0] = ret8[0][avx_index];
    ret->val[1] = ret8[1][avx_index];
    ret->val[2] = ret8[2][avx_index];
    ret->val[3] = ret8[3][avx_index];

    return;
}

// ret := mmval[0] + ... + mmval[15]
static void _bncavx512_rqs_sum512d(float ret[QSSIZE], __m512 ret8[QSSIZE])
{
    qsfloat ret16_i[16];

    // ret16_i := ret8 (16 lanes)
    _bncavx512_get_qs_m512_i(&(ret16_i[0]), ret8, 0);
    _bncavx512_get_qs_m512_i(&(ret16_i[1]), ret8, 1);
    _bncavx512_get_qs_m512_i(&(ret16_i[2]), ret8, 2);
    _bncavx512_get_qs_m512_i(&(ret16_i[3]), ret8, 3);
    _bncavx512_get_qs_m512_i(&(ret16_i[4]), ret8, 4);
    _bncavx512_get_qs_m512_i(&(ret16_i[5]), ret8, 5);
    _bncavx512_get_qs_m512_i(&(ret16_i[6]), ret8, 6);
    _bncavx512_get_qs_m512_i(&(ret16_i[7]), ret8, 7);
    _bncavx512_get_qs_m512_i(&(ret16_i[8]), ret8, 8);
    _bncavx512_get_qs_m512_i(&(ret16_i[9]), ret8, 9);
    _bncavx512_get_qs_m512_i(&(ret16_i[10]), ret8, 10);
    _bncavx512_get_qs_m512_i(&(ret16_i[11]), ret8, 11);
    _bncavx512_get_qs_m512_i(&(ret16_i[12]), ret8, 12);
    _bncavx512_get_qs_m512_i(&(ret16_i[13]), ret8, 13);
    _bncavx512_get_qs_m512_i(&(ret16_i[14]), ret8, 14);
    _bncavx512_get_qs_m512_i(&(ret16_i[15]), ret8, 15);

    rqs_set(ret, ret16_i[0].val);
    rqs_add(ret, ret, ret16_i[1].val);
    rqs_add(ret, ret, ret16_i[2].val);
    rqs_add(ret, ret, ret16_i[3].val);
    rqs_add(ret, ret, ret16_i[4].val);
    rqs_add(ret, ret, ret16_i[5].val);
    rqs_add(ret, ret, ret16_i[6].val);
    rqs_add(ret, ret, ret16_i[7].val);
    rqs_add(ret, ret, ret16_i[8].val);
    rqs_add(ret, ret, ret16_i[9].val);
    rqs_add(ret, ret, ret16_i[10].val);
    rqs_add(ret, ret, ret16_i[11].val);
    rqs_add(ret, ret, ret16_i[12].val);
    rqs_add(ret, ret, ret16_i[13].val);
    rqs_add(ret, ret, ret16_i[14].val);
    rqs_add(ret, ret, ret16_i[15].val);
}

// abs
static inline void _bncavx512_rqs_abs(__m512 ret[QSSIZE], __m512 a[QSSIZE])
{
    int avx_index;

    for(avx_index = 0; avx_index < 16; avx_index++)
    {
        if(a[0][avx_index] < 0.0)
        {
            ret[0][avx_index] = -a[0][avx_index];
            ret[1][avx_index] = -a[1][avx_index];
            ret[2][avx_index] = -a[2][avx_index];
            ret[3][avx_index] = -a[3][avx_index];
        }
        else
        {
            ret[0][avx_index] = a[0][avx_index];
            ret[1][avx_index] = a[1][avx_index];
            ret[2][avx_index] = a[2][avx_index];
            ret[3][avx_index] = a[3][avx_index];
        }
    }
}
// ret := |ret8[0]| + |ret8[1]| + ... + |ret8[15]|
static void _bncavx512_rqs_abssum512d(float ret[QSSIZE], __m512 ret8[QSSIZE])
{
    qsfloat ret16_i[16];
    float tmp[QSSIZE];

    // ret16_i := ret8 (16 lanes)
    _bncavx512_get_qs_m512_i(&ret16_i[0], ret8, 0);
    _bncavx512_get_qs_m512_i(&ret16_i[1], ret8, 1);
    _bncavx512_get_qs_m512_i(&ret16_i[2], ret8, 2);
    _bncavx512_get_qs_m512_i(&ret16_i[3], ret8, 3);
    _bncavx512_get_qs_m512_i(&ret16_i[4], ret8, 4);
    _bncavx512_get_qs_m512_i(&ret16_i[5], ret8, 5);
    _bncavx512_get_qs_m512_i(&ret16_i[6], ret8, 6);
    _bncavx512_get_qs_m512_i(&ret16_i[7], ret8, 7);
    _bncavx512_get_qs_m512_i(&ret16_i[8], ret8, 8);
    _bncavx512_get_qs_m512_i(&ret16_i[9], ret8, 9);
    _bncavx512_get_qs_m512_i(&ret16_i[10], ret8, 10);
    _bncavx512_get_qs_m512_i(&ret16_i[11], ret8, 11);
    _bncavx512_get_qs_m512_i(&ret16_i[12], ret8, 12);
    _bncavx512_get_qs_m512_i(&ret16_i[13], ret8, 13);
    _bncavx512_get_qs_m512_i(&ret16_i[14], ret8, 14);
    _bncavx512_get_qs_m512_i(&ret16_i[15], ret8, 15);

    rqs_abs(tmp, ret16_i[0].val);  rqs_set(ret, tmp);
    rqs_abs(tmp, ret16_i[1].val);  rqs_add(ret, ret, tmp);
    rqs_abs(tmp, ret16_i[2].val);  rqs_add(ret, ret, tmp);
    rqs_abs(tmp, ret16_i[3].val);  rqs_add(ret, ret, tmp);
    rqs_abs(tmp, ret16_i[4].val);  rqs_add(ret, ret, tmp);
    rqs_abs(tmp, ret16_i[5].val);  rqs_add(ret, ret, tmp);
    rqs_abs(tmp, ret16_i[6].val);  rqs_add(ret, ret, tmp);
    rqs_abs(tmp, ret16_i[7].val);  rqs_add(ret, ret, tmp);
    rqs_abs(tmp, ret16_i[8].val);  rqs_add(ret, ret, tmp);
    rqs_abs(tmp, ret16_i[9].val);  rqs_add(ret, ret, tmp);
    rqs_abs(tmp, ret16_i[10].val); rqs_add(ret, ret, tmp);
    rqs_abs(tmp, ret16_i[11].val); rqs_add(ret, ret, tmp);
    rqs_abs(tmp, ret16_i[12].val); rqs_add(ret, ret, tmp);
    rqs_abs(tmp, ret16_i[13].val); rqs_add(ret, ret, tmp);
    rqs_abs(tmp, ret16_i[14].val); rqs_add(ret, ret, tmp);
    rqs_abs(tmp, ret16_i[15].val); rqs_add(ret, ret, tmp);
}

// ret := max(|ret8[0]|, |ret8[1]|, ..., |ret8[15]|)
static void _bncavx512_rqs_absmax512d(float ret[QSSIZE], __m512 ret8[QSSIZE])
{
    qsfloat ret16_i[16];
    float tmp[QSSIZE];

    // ret16_i := ret8 (16 lanes)
    _bncavx512_get_qs_m512_i(&ret16_i[0], ret8, 0);
    _bncavx512_get_qs_m512_i(&ret16_i[1], ret8, 1);
    _bncavx512_get_qs_m512_i(&ret16_i[2], ret8, 2);
    _bncavx512_get_qs_m512_i(&ret16_i[3], ret8, 3);
    _bncavx512_get_qs_m512_i(&ret16_i[4], ret8, 4);
    _bncavx512_get_qs_m512_i(&ret16_i[5], ret8, 5);
    _bncavx512_get_qs_m512_i(&ret16_i[6], ret8, 6);
    _bncavx512_get_qs_m512_i(&ret16_i[7], ret8, 7);
    _bncavx512_get_qs_m512_i(&ret16_i[8], ret8, 8);
    _bncavx512_get_qs_m512_i(&ret16_i[9], ret8, 9);
    _bncavx512_get_qs_m512_i(&ret16_i[10], ret8, 10);
    _bncavx512_get_qs_m512_i(&ret16_i[11], ret8, 11);
    _bncavx512_get_qs_m512_i(&ret16_i[12], ret8, 12);
    _bncavx512_get_qs_m512_i(&ret16_i[13], ret8, 13);
    _bncavx512_get_qs_m512_i(&ret16_i[14], ret8, 14);
    _bncavx512_get_qs_m512_i(&ret16_i[15], ret8, 15);

    rqs_abs(tmp, ret16_i[0].val);
    rqs_set(ret, tmp); // ret := |ret16_i[0]|

    rqs_abs(tmp, ret16_i[1].val);  if(rqs_cmp(ret, tmp) < 0) rqs_set(ret, tmp);
    rqs_abs(tmp, ret16_i[2].val);  if(rqs_cmp(ret, tmp) < 0) rqs_set(ret, tmp);
    rqs_abs(tmp, ret16_i[3].val);  if(rqs_cmp(ret, tmp) < 0) rqs_set(ret, tmp);
    rqs_abs(tmp, ret16_i[4].val);  if(rqs_cmp(ret, tmp) < 0) rqs_set(ret, tmp);
    rqs_abs(tmp, ret16_i[5].val);  if(rqs_cmp(ret, tmp) < 0) rqs_set(ret, tmp);
    rqs_abs(tmp, ret16_i[6].val);  if(rqs_cmp(ret, tmp) < 0) rqs_set(ret, tmp);
    rqs_abs(tmp, ret16_i[7].val);  if(rqs_cmp(ret, tmp) < 0) rqs_set(ret, tmp);
    rqs_abs(tmp, ret16_i[8].val);  if(rqs_cmp(ret, tmp) < 0) rqs_set(ret, tmp);
    rqs_abs(tmp, ret16_i[9].val);  if(rqs_cmp(ret, tmp) < 0) rqs_set(ret, tmp);
    rqs_abs(tmp, ret16_i[10].val); if(rqs_cmp(ret, tmp) < 0) rqs_set(ret, tmp);
    rqs_abs(tmp, ret16_i[11].val); if(rqs_cmp(ret, tmp) < 0) rqs_set(ret, tmp);
    rqs_abs(tmp, ret16_i[12].val); if(rqs_cmp(ret, tmp) < 0) rqs_set(ret, tmp);
    rqs_abs(tmp, ret16_i[13].val); if(rqs_cmp(ret, tmp) < 0) rqs_set(ret, tmp);
    rqs_abs(tmp, ret16_i[14].val); if(rqs_cmp(ret, tmp) < 0) rqs_set(ret, tmp);
    rqs_abs(tmp, ret16_i[15].val); if(rqs_cmp(ret, tmp) < 0) rqs_set(ret, tmp);
}

// ret := || ret8[0]^2 + ret8[1]^2 + ... + ret8[15]^2 ||_2
static void _bncavx512_rqs_norm512d(float ret[QSSIZE], __m512 ret8[QSSIZE])
{
    qsfloat ret16_i[16];
    float tmp[QSSIZE];

    // ret16_i := ret8 (16 lanes)
    _bncavx512_get_qs_m512_i(&ret16_i[0], ret8, 0);
    _bncavx512_get_qs_m512_i(&ret16_i[1], ret8, 1);
    _bncavx512_get_qs_m512_i(&ret16_i[2], ret8, 2);
    _bncavx512_get_qs_m512_i(&ret16_i[3], ret8, 3);
    _bncavx512_get_qs_m512_i(&ret16_i[4], ret8, 4);
    _bncavx512_get_qs_m512_i(&ret16_i[5], ret8, 5);
    _bncavx512_get_qs_m512_i(&ret16_i[6], ret8, 6);
    _bncavx512_get_qs_m512_i(&ret16_i[7], ret8, 7);
    _bncavx512_get_qs_m512_i(&ret16_i[8], ret8, 8);
    _bncavx512_get_qs_m512_i(&ret16_i[9], ret8, 9);
    _bncavx512_get_qs_m512_i(&ret16_i[10], ret8, 10);
    _bncavx512_get_qs_m512_i(&ret16_i[11], ret8, 11);
    _bncavx512_get_qs_m512_i(&ret16_i[12], ret8, 12);
    _bncavx512_get_qs_m512_i(&ret16_i[13], ret8, 13);
    _bncavx512_get_qs_m512_i(&ret16_i[14], ret8, 14);
    _bncavx512_get_qs_m512_i(&ret16_i[15], ret8, 15);

    rqs_mul(tmp, ret16_i[0].val, ret16_i[0].val);   rqs_set(ret, tmp);
    rqs_mul(tmp, ret16_i[1].val, ret16_i[1].val);   rqs_add(ret, ret, tmp);
    rqs_mul(tmp, ret16_i[2].val, ret16_i[2].val);   rqs_add(ret, ret, tmp);
    rqs_mul(tmp, ret16_i[3].val, ret16_i[3].val);   rqs_add(ret, ret, tmp);
    rqs_mul(tmp, ret16_i[4].val, ret16_i[4].val);   rqs_set(ret, tmp);
    rqs_mul(tmp, ret16_i[5].val, ret16_i[5].val);   rqs_add(ret, ret, tmp);
    rqs_mul(tmp, ret16_i[6].val, ret16_i[6].val);   rqs_add(ret, ret, tmp);
    rqs_mul(tmp, ret16_i[7].val, ret16_i[7].val);   rqs_add(ret, ret, tmp);
    rqs_mul(tmp, ret16_i[8].val, ret16_i[8].val);   rqs_add(ret, ret, tmp);
    rqs_mul(tmp, ret16_i[9].val, ret16_i[9].val);   rqs_add(ret, ret, tmp);
    rqs_mul(tmp, ret16_i[10].val, ret16_i[10].val); rqs_add(ret, ret, tmp);
    rqs_mul(tmp, ret16_i[11].val, ret16_i[11].val); rqs_add(ret, ret, tmp);
    rqs_mul(tmp, ret16_i[12].val, ret16_i[12].val); rqs_add(ret, ret, tmp);
    rqs_mul(tmp, ret16_i[13].val, ret16_i[13].val); rqs_add(ret, ret, tmp);
    rqs_mul(tmp, ret16_i[14].val, ret16_i[14].val); rqs_add(ret, ret, tmp);
    rqs_mul(tmp, ret16_i[15].val, ret16_i[15].val); rqs_add(ret, ret, tmp);

    rqs_sqrt(tmp, ret);
    rqs_set(ret, tmp);
}

//inline void renorm(float *c0, float *c1, float *c2, float *c3)  (single-precision variant)
static inline void _bncavx512_frenorm(__m512 *c0, __m512 *c1, __m512 *c2, __m512 *c3)
{
    float q[16][4];
    int k;

    for(k = 0; k < 16; k++)
    {
        q[k][0] = (* c0)[k]; q[k][1] = (* c1)[k]; q[k][2] = (* c2)[k]; q[k][3] = (* c3)[k];
        frenorm(&q[k][0], &q[k][1], &q[k][2], &q[k][3]);
    }
    for(k = 0; k < 16; k++)
    {
        (* c0)[k] = q[k][0]; (* c1)[k] = q[k][1]; (* c2)[k] = q[k][2]; (* c3)[k] = q[k][3];
    }
}

//inline void renorm4(float *c0, float *c1, float *c2, float *c3, float *c4)  (single-precision variant)
static inline void _bncavx512_frenorm4(__m512 *c0, __m512 *c1, __m512 *c2, __m512 *c3, __m512 *c4)
{
    float q[16][5];
    int k;

    for(k = 0; k < 16; k++)
    {
        q[k][0] = (* c0)[k]; q[k][1] = (* c1)[k]; q[k][2] = (* c2)[k]; q[k][3] = (* c3)[k]; q[k][4] = (* c4)[k];
        frenorm4(&q[k][0], &q[k][1], &q[k][2], &q[k][3], &q[k][4]);
    }
    for(k = 0; k < 16; k++)
    {
        (* c0)[k] = q[k][0]; (* c1)[k] = q[k][1]; (* c2)[k] = q[k][2]; (* c3)[k] = q[k][3]; (* c4)[k] = q[k][4];
    }
}


/********** Additions ************/
//inline void three_sum(float *a, float *b, float *c)  (single-precision variant)
static inline void _bncavx512_fthree_sum(__m512 *a, __m512 *b, __m512 *c)
{
    __m512 t1, t2, t3;

    t1 = _bncavx512_ftwo_sum(*a, *b, &t2);
    *a = _bncavx512_ftwo_sum(*c, t1, &t3);
    *b = _bncavx512_ftwo_sum(t2, t3, c);
}

//inline void three_sum2(float *a, float *b, float *c)  (single-precision variant)
static inline void _bncavx512_fthree_sum2(__m512 *a, __m512 *b, __m512 *c)
{
    __m512 t1, t2, t3;

    t1 = _bncavx512_ftwo_sum(*a, *b, &t2);
    *a = _bncavx512_ftwo_sum(*c, t1, &t3);
    *b = _mm512_add_ps(t2, t3);
}

#ifdef USE_QS_BF
    #define _bncavx512_rqs_add _bncavx512_rqs_add_bf
#else // USE_QS_BF
    #define _bncavx512_rqs_add _bncavx512_rqs_add_sloppy
#endif // USE_QS_BF

static inline void _bncavx512_rqs_add_sloppy(__m512 ret[QSSIZE], __m512 a[QSSIZE], __m512 b[QSSIZE])
{
    float in_ret[16][QSSIZE], in_a[16][QSSIZE], in_b[16][QSSIZE];
    int k;

    for(k = 0; k < 16; k++)
    {
        in_a[k][0] = a[0][k]; in_a[k][1] = a[1][k]; in_a[k][2] = a[2][k]; in_a[k][3] = a[3][k];
        in_b[k][0] = b[0][k]; in_b[k][1] = b[1][k]; in_b[k][2] = b[2][k]; in_b[k][3] = b[3][k];
    }
    for(k = 0; k < 16; k++)
        rqs_add(in_ret[k], in_a[k], in_b[k]);
    for(k = 0; k < 16; k++)
    {
        ret[0][k] = in_ret[k][0]; ret[1][k] = in_ret[k][1]; ret[2][k] = in_ret[k][2]; ret[3][k] = in_ret[k][3];
    }
}

// Branch free algorithm by D.K.Zhang and A.Aiken at SC2025
static inline void _bncavx512_rqs_add_bf(__m512 ret[QSSIZE], __m512 a[QSSIZE], __m512 b[QSSIZE])
{
    __m512 a0 , b0 , c0 , d0 , e0, f0, g0, h0;
    __m512 a1 , b1 , c1 , d1 , e1, f1, g1, h1;
    __m512 a2 , b2 , c2 , d2 , e2, f2, g2;
    __m512 a3 , b3 , c3 , d3 , e3, f3, g3;
    __m512 a4 , c4 , d4 , e4;
    __m512 b5 , d5 , e5;
    __m512 b6 , c6 , d6 , e6;
    __m512 a7 , b7 , c7 , d7;
    __m512 b8 , c8 , e8;
    __m512 d9;
    __m512 a10, b10, c10, d10;
    __m512 b11, c11;
    __m512 c12, d12;

    a0 = a[0];
    b0 = b[0];
    c0 = a[1];
    d0 = b[1];
    e0 = a[2];
    f0 = b[2];
    g0 = a[3];
    h0 = b[3];
    a1 = _bncavx512_ftwo_sum(a0, b0, &b1);
    c1 = _bncavx512_ftwo_sum(c0, d0, &d1);
    e1 = _bncavx512_ftwo_sum(e0, f0, &f1);
    g1 = _bncavx512_ftwo_sum(g0, h0, &h1);
    a2 = _bncavx512_fquick_two_sum(a1, c1, &c2);
    b2 = _mm512_add_ps(b1, h1);
    d2 = _bncavx512_ftwo_sum(d1, e1, &e2);
    f2 = _bncavx512_ftwo_sum(f1, g1, &g2);
    b3 = _bncavx512_ftwo_sum(b2, g2, &g3);
    c3 = _bncavx512_fquick_two_sum(c2, d2, &d3);
    e3 = _bncavx512_ftwo_sum(e2, f2, &f3);
    a4 = _bncavx512_fquick_two_sum(a2, c3, &c4);
    d4 = _bncavx512_fquick_two_sum(d3, e3, &e4);
    b5 = _bncavx512_ftwo_sum(b3, d4, &d5);
    e5 = _mm512_add_ps(e4, f3);
    b6 = _bncavx512_ftwo_sum(b5, c4, &c6);
    d6 = _bncavx512_ftwo_sum(d5, e5, &e6);
    a7 = _bncavx512_fquick_two_sum(a4, b6, &b7);
    c7 = _bncavx512_fquick_two_sum(c6, d6, &d7);
    e8 = _mm512_add_ps(e6, g3);
    b8 = _bncavx512_fquick_two_sum(b7, c7, &c8);
    d9 = _mm512_add_ps(d7, e8);
    a10 = _bncavx512_fquick_two_sum(a7, b8, &b10);
    c10 = _bncavx512_fquick_two_sum(c8, d9, &d10);
    b11 = _bncavx512_fquick_two_sum(b10, c10, &c11);
    c12 = _bncavx512_fquick_two_sum(c11, d10, &d12);
    ret[0] = a10;
    ret[1] = b11;
    ret[2] = c12;
    ret[3] = d12;
}


// rqs_add -> rts_add
static inline void _bncavx512_rts_addq(__m512 ret[TSSIZE], __m512 a[TSSIZE], __m512 b[TSSIZE])
{
    // c_qs_add_sloppy (re-organized to minimize data dependency)
    __m512 s0, s1, s2;
    __m512 t0, t1, t2;
    __m512 v0, v1, v2;
    __m512 u0, u1, u2;
    __m512 w0, w1, w2;

    s0 = _mm512_add_ps(a[0], b[0]);
    s1 = _mm512_add_ps(a[1], b[1]);
    s2 = _mm512_add_ps(a[2], b[2]);

    v0 = _mm512_sub_ps(s0, a[0]);
    v1 = _mm512_sub_ps(s1, a[1]);
    v2 = _mm512_sub_ps(s2, a[2]);

    u0 = _mm512_sub_ps(s0, v0);
    u1 = _mm512_sub_ps(s1, v1);
    u2 = _mm512_sub_ps(s2, v2);

    w0 = _mm512_sub_ps(a[0], u0);
    w1 = _mm512_sub_ps(a[1], u1);
    w2 = _mm512_sub_ps(a[2], u2);

    u0 = _mm512_sub_ps(b[0], v0);
    u1 = _mm512_sub_ps(b[1], v1);
    u2 = _mm512_sub_ps(b[2], v2);

    t0 = _mm512_add_ps(w0, u0);
    t1 = _mm512_add_ps(w1, u1);
    t2 = _mm512_add_ps(w2, u2);

    s1 = _bncavx512_ftwo_sum(s1, t0, &t0);
    _bncavx512_fthree_sum(&s2, &t0, &t1);
    t0 = _mm512_add_ps(t0, t1);

    /* renormalize */
    _bncavx512_frenorm(&s0, &s1, &s2, &t0);

    ret[0] = s0;
    ret[1] = s1;
    ret[2] = s2;
}

// mul
static inline void _bncavx512_rts_mulq(__m512 ret[TSSIZE], __m512 a[TSSIZE], __m512 b[TSSIZE])
{
    // c_qs_mul_sloppy
    __m512 p0, p1, p2, p3, p4, p5;
    __m512 q0, q1, q2, q3, q4, q5;
    __m512 t0, t1;
    __m512 s0, s1, s2;

    p0 = _bncavx512_ftwo_prod(a[0], b[0], &q0);

    p1 = _bncavx512_ftwo_prod(a[0], b[1], &q1);
    p2 = _bncavx512_ftwo_prod(a[1], b[0], &q2);

    p3 = _bncavx512_ftwo_prod(a[0], b[2], &q3);
    p4 = _bncavx512_ftwo_prod(a[1], b[1], &q4);
    p5 = _bncavx512_ftwo_prod(a[2], b[0], &q5);

    /* Start Accumulation */
    _bncavx512_fthree_sum(&p1, &p2, &q0);

    /* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
    _bncavx512_fthree_sum2(&p2, &q1, &q2);
    _bncavx512_fthree_sum2(&p3, &p4, &p5);

    /* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
    s0 = _bncavx512_ftwo_sum(p2, p3, &t0);
    s1 = _bncavx512_ftwo_sum(q1, p4, &t1);
    s1 = _bncavx512_ftwo_sum(s1, t0, &t0);

    /* O(eps^3) order terms */
    s1 = _mm512_add_ps(s1, _mm512_mul_ps(a[1], b[2]));
    s1 = _mm512_add_ps(s1, _mm512_mul_ps(a[2], b[1]));
    s1 = _mm512_add_ps(s1, q0);
    s1 = _mm512_add_ps(s1, q3);
    s1 = _mm512_add_ps(s1, q4);

    _bncavx512_frenorm(&p0, &p1, &s0, &s1);

    ret[0] = p0;
    ret[1] = p1;
    ret[2] = s0;
}

#ifdef USE_QS_BF
    #define _bncavx512_rqs_mul _bncavx512_rqs_mul_bf
#else // USE_QS_BF
    #define _bncavx512_rqs_mul _bncavx512_rqs_mul_sloppy
#endif // USE_QS_BF

// mul
static inline void _bncavx512_rqs_mul_sloppy(__m512 ret[QSSIZE], __m512 a[QSSIZE], __m512 b[QSSIZE])
{
    float in_ret[16][QSSIZE], in_a[16][QSSIZE], in_b[16][QSSIZE];
    int k;

    for(k = 0; k < 16; k++)
    {
        in_a[k][0] = a[0][k]; in_a[k][1] = a[1][k]; in_a[k][2] = a[2][k]; in_a[k][3] = a[3][k];
        in_b[k][0] = b[0][k]; in_b[k][1] = b[1][k]; in_b[k][2] = b[2][k]; in_b[k][3] = b[3][k];
    }
    for(k = 0; k < 16; k++)
        rqs_mul(in_ret[k], in_a[k], in_b[k]);
    for(k = 0; k < 16; k++)
    {
        ret[0][k] = in_ret[k][0]; ret[1][k] = in_ret[k][1]; ret[2][k] = in_ret[k][2]; ret[3][k] = in_ret[k][3];
    }
}

// Branch free algorithm by D.K.Zhang and A.Aiken at SC2025
static inline void _bncavx512_rqs_mul_bf(__m512 ret[QSSIZE], __m512 a[QSSIZE], __m512 b[QSSIZE])
{
    __m512 a0, b0, c0, d0, e0, f0, g0, h0, i0, j0, k0, l0, m0, n0, o0, p0;
    __m512 c1, d1, e1, f1, g1, i1, j1, m1, n1;
    __m512 b2, c2, e2, f2, h2, i2, m2;
    __m512 a3, b3, c3, d3, e3, f3, g3, h3;
    __m512 c4, d4, e4, f4;
    __m512 d5;
    __m512 c6, d6;
    __m512 b7, c7, d7;
    __m512 a8, b8, c8, d8;
    __m512 b9, c9;
    __m512 c10, d10;

    a0 = _bncavx512_ftwo_prod(a[0], b[0], &b0);
    c0 = _bncavx512_ftwo_prod(a[0], b[1], &e0);
    d0 = _bncavx512_ftwo_prod(a[1], b[0], &f0);
    g0 = _bncavx512_ftwo_prod(a[0], b[2], &j0);
    h0 = _bncavx512_ftwo_prod(a[1], b[1], &k0);
    i0 = _bncavx512_ftwo_prod(a[2], b[0], &l0);
    m0 = _mm512_mul_ps(a[0], b[3]);
    n0 = _mm512_mul_ps(a[1], b[2]);
    o0 = _mm512_mul_ps(a[2], b[1]);
    p0 = _mm512_mul_ps(a[3], b[0]);
    c1 = _bncavx512_ftwo_sum(c0, d0, &d1);
    e1 = _bncavx512_ftwo_sum(e0, f0, &f1);
    g1 = _bncavx512_ftwo_sum(g0, i0, &i1);
    j1 = _mm512_add_ps(j0, l0);
    m1 = _mm512_add_ps(m0, p0);
    n1 = _mm512_add_ps(n0, o0);
    b2 = _bncavx512_ftwo_sum(b0, c1, &c2);
    e2 = _bncavx512_ftwo_sum(e1, h0, &h2);
    f2 = _mm512_add_ps(f1, j1);
    i2 = _mm512_add_ps(i1, k0);
    m2 = _mm512_add_ps(m1, n1);
    a3 = _bncavx512_fquick_two_sum(a0, b2, &b3);
    c3 = _bncavx512_fquick_two_sum(c2, d1, &d3);
    e3 = _bncavx512_ftwo_sum(e2, g1, &g3);
    f3 = _mm512_add_ps(f2, m2);
    h3 = _mm512_add_ps(h2, i2);
    c4 = _bncavx512_ftwo_sum(c3, e3, &e4);
    d4 = _mm512_add_ps(d3, h3);
    f4 = _mm512_add_ps(f3, g3);
    d5 = _mm512_add_ps(d4, e4);
    c6 = _bncavx512_ftwo_sum(c4, d5, &d6);
    b7 = _bncavx512_ftwo_sum(b3, c6, &c7);
    d7 = _mm512_add_ps(d6, f4);
    a8 = _bncavx512_fquick_two_sum(a3, b7, &b8);
    c8 = _bncavx512_ftwo_sum(c7, d7, &d8);
    b9 = _bncavx512_ftwo_sum(b8, c8, &c9);
    c10 = _bncavx512_fquick_two_sum(c9, d8, &d10);
    ret[0] = a8;
    ret[1] = b9;
    ret[2] = c10;
    ret[3] = d10;
}

// c[4] := -a[4]
static inline void _bncavx512_rqs_neg(__m512 c[QSSIZE], __m512 a[QSSIZE])
{
    __m512 zero8;

    zero8 = _mm512_setzero_ps();

    c[0] = _mm512_sub_ps(zero8, a[0]);
    c[1] = _mm512_sub_ps(zero8, a[1]);
    c[2] = _mm512_sub_ps(zero8, a[2]);
    c[3] = _mm512_sub_ps(zero8, a[3]);
}

/* sub */
// c := a - b
static inline void _bncavx512_rqs_sub(__m512 c[QSSIZE], __m512 a[QSSIZE], __m512 b[QSSIZE])
{
    __m512 mb[QSSIZE];

    // a + (-b)
    _bncavx512_rqs_neg(mb, b);
    _bncavx512_rqs_add(c, a, mb);
}

// c:= a + (float)b
static inline void _bncavx512_rqs_add_d(__m512 c[QSSIZE], const __m512 a[QSSIZE], __m512 b)
{
    __m512 e;

    c[0] = _bncavx512_ftwo_sum(a[0], b, &e);
    c[1] = _bncavx512_ftwo_sum(a[1], e, &e);
    c[2] = _bncavx512_ftwo_sum(a[2], e, &e);
    c[3] = _bncavx512_ftwo_sum(a[3], e, &e);

    _bncavx512_frenorm4(&(c[0]), &(c[1]), &(c[2]), &(c[3]), &e);

    return;
}

// c := a * (float)b
static inline void _bncavx512_rqs_mul_d(__m512 c[QSSIZE], const __m512 a[QSSIZE], __m512 b)
{
    __m512 p0, p1, p2, p3;
    __m512 q0, q1, q2;
    __m512 s0, s1, s2, s3, s4;

    p0 = _bncavx512_ftwo_prod(a[0], b, &q0);
    p1 = _bncavx512_ftwo_prod(a[1], b, &q1);
    p2 = _bncavx512_ftwo_prod(a[2], b, &q2);
    p3 = _mm512_mul_ps(a[3], b);

    s0 = p0;

    s1 = _bncavx512_ftwo_sum(q0, p1, &s2);

    _bncavx512_fthree_sum(&s2, &q1, &p2);

    _bncavx512_fthree_sum2(&q1, &q2, &p3);
    s3 = q1;

    s4 = _mm512_add_ps(q2, p2);

    _bncavx512_frenorm4(&s0, &s1, &s2, &s3, &s4);
    c[0] = s0;
    c[1] = s1;
    c[2] = s2;
    c[3] = s3;
}

// c := a / b
static inline void _bncavx512_rts_divq(__m512 c[TSSIZE], __m512 a[TSSIZE], __m512 b[TSSIZE])
{
    __m512 q0, q1, q2, q3;
    __m512 r[TSSIZE], tmp[TSSIZE];

    q0 = _mm512_div_ps(a[0], b[0]);

    _bncavx512_rts_mul_d(tmp, b, q0);
    _bncavx512_rts_subq(r, a, tmp);

    q1 = _mm512_div_ps(r[0], b[0]);

    _bncavx512_rts_mul_d(tmp, b, q1);
    _bncavx512_rts_subq(r, tmp, r);

    q2 = _mm512_div_ps(r[0], b[0]);
    _bncavx512_rts_mul_d(tmp, b, q2);
    _bncavx512_rts_subq(r, tmp, r);

    q3 = _mm512_div_ps(r[0], b[0]);

    _bncavx512_frenorm(&q0, &q1, &q2, &q3);

    c[0] = q0;
    c[1] = q1;
    c[2] = q2;
    //c[3] = q3;
}

/* div */
// c := a / b
static inline void _bncavx512_rqs_div(__m512 c[QSSIZE], __m512 a[QSSIZE], __m512 b[QSSIZE])
{
    __m512 q0, q1, q2, q3;
    __m512 r[QSSIZE], tmp[QSSIZE];

    q0 = _mm512_div_ps(a[0], b[0]);

    _bncavx512_rqs_mul_d(tmp, b, q0);
    _bncavx512_rqs_sub(r, a, tmp);

    q1 = _mm512_div_ps(r[0], b[0]);

    _bncavx512_rqs_mul_d(tmp, b, q1);
    _bncavx512_rqs_sub(r, tmp, r);

    q2 = _mm512_div_ps(r[0], b[0]);
    _bncavx512_rqs_mul_d(tmp, b, q2);
    _bncavx512_rqs_sub(r, tmp, r);

    q3 = _mm512_div_ps(r[0], b[0]);

    _bncavx512_frenorm(&q0, &q1, &q2, &q3);

    c[0] = q0;
    c[1] = q1;
    c[2] = q2;
    c[3] = q3;
}

// Aliases for naming variants used in *linear.c
#define _bncavx2_rqs_sum512    _bncavx512_rqs_sum512d
#define _bncavx512_rqs_sum512  _bncavx512_rqs_sum512d
#define _bncavx512_rqs_abssum512  _bncavx512_rqs_abssum512d
#define _bncavx512_rqs_norm512    _bncavx512_rqs_norm512d

#endif //defined(__AVX512F__)
#endif // ifndef __BNCAVX_QS_AVX512_H
