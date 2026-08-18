// ------------------------
// ------ TS AVX-512 ------
// ------------------------
#ifndef __BNCAVX_TS_AVX512_H
#define __BNCAVX_TS_AVX512_H

#ifndef TSSIZE
    #define TSSIZE 3
#endif // TSSIZE

#if defined(__AVX512F__)

// ret := 0
// NOTE: _bncavx512_set0_ts is already defined in _bncavx_ts.h
#define _bncavx512_rts_set0(ret) _bncavx512_set0_ts((ret))

// ret := val
static inline void _bncavx512_rts_set(__m512 ret[TSSIZE], __m512 val[TSSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = val[2];
}

// ret := (float)val
static inline void _bncavx512_rts_set_d(__m512 ret[TSSIZE], __m512 val)
{
    ret[0] = val;
    ret[1] = _mm512_setzero_ps(); // val[1];
    ret[2] = _mm512_setzero_ps(); // val[2];
}

// ret := (DS)val
static inline void _bncavx512_rts_set_ds(__m512 ret[TSSIZE], __m512 val[DSSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = _mm512_setzero_ps(); // val[2];
}

// ret := [val val val val]
static inline void _bncavx512_rts_set1_ts(__m512 ret[TSSIZE], float val[TSSIZE])
{
    ret[0] = _mm512_set1_ps(val[0]);
    ret[1] = _mm512_set1_ps(val[1]);
    ret[2] = _mm512_set1_ps(val[2]);
}

// ret := ret8[][avx_index]
static inline void _bncavx512_get_ts_m512_i(tsfloat *ret, __m512 ret8[TSSIZE], int avx_index)
{
    ret->val[0] = ret8[0][avx_index];
    ret->val[1] = ret8[1][avx_index];
    ret->val[2] = ret8[2][avx_index];

    return;
}

// ret := ret8[0] + ret8[1] + ... + ret8[15]
static void _bncavx512_rts_sum512d(float ret[TSSIZE], __m512 ret8[TSSIZE])
{
    tsfloat ret16_i[16];

    // ret16_i := ret8 (16 lanes)
    _bncavx512_get_ts_m512_i(&ret16_i[0], ret8, 0);
    _bncavx512_get_ts_m512_i(&ret16_i[1], ret8, 1);
    _bncavx512_get_ts_m512_i(&ret16_i[2], ret8, 2);
    _bncavx512_get_ts_m512_i(&ret16_i[3], ret8, 3);
    _bncavx512_get_ts_m512_i(&ret16_i[4], ret8, 4);
    _bncavx512_get_ts_m512_i(&ret16_i[5], ret8, 5);
    _bncavx512_get_ts_m512_i(&ret16_i[6], ret8, 6);
    _bncavx512_get_ts_m512_i(&ret16_i[7], ret8, 7);
    _bncavx512_get_ts_m512_i(&ret16_i[8], ret8, 8);
    _bncavx512_get_ts_m512_i(&ret16_i[9], ret8, 9);
    _bncavx512_get_ts_m512_i(&ret16_i[10], ret8, 10);
    _bncavx512_get_ts_m512_i(&ret16_i[11], ret8, 11);
    _bncavx512_get_ts_m512_i(&ret16_i[12], ret8, 12);
    _bncavx512_get_ts_m512_i(&ret16_i[13], ret8, 13);
    _bncavx512_get_ts_m512_i(&ret16_i[14], ret8, 14);
    _bncavx512_get_ts_m512_i(&ret16_i[15], ret8, 15);

    rts_set(ret, ret16_i[0].val);
    rts_add(ret, ret, ret16_i[1].val);
    rts_add(ret, ret, ret16_i[2].val);
    rts_add(ret, ret, ret16_i[3].val);
    rts_add(ret, ret, ret16_i[4].val);
    rts_add(ret, ret, ret16_i[5].val);
    rts_add(ret, ret, ret16_i[6].val);
    rts_add(ret, ret, ret16_i[7].val);
    rts_add(ret, ret, ret16_i[8].val);
    rts_add(ret, ret, ret16_i[9].val);
    rts_add(ret, ret, ret16_i[10].val);
    rts_add(ret, ret, ret16_i[11].val);
    rts_add(ret, ret, ret16_i[12].val);
    rts_add(ret, ret, ret16_i[13].val);
    rts_add(ret, ret, ret16_i[14].val);
    rts_add(ret, ret, ret16_i[15].val);
}

// abs
static inline void _bncavx512_rts_abs(__m512 ret[TSSIZE], __m512 a[TSSIZE])
{
    int avx_index;

    for(avx_index = 0; avx_index < 16; avx_index++)
    {
        if(a[0][avx_index] < 0.0)
        {
            ret[0][avx_index] = -a[0][avx_index];
            ret[1][avx_index] = -a[1][avx_index];
            ret[2][avx_index] = -a[2][avx_index];
        }
        else
        {
            ret[0][avx_index] = a[0][avx_index];
            ret[1][avx_index] = a[1][avx_index];
            ret[2][avx_index] = a[2][avx_index];
        }
    }
}

// ret := |ret8[0]| + |ret8[1]| + ... + |ret8[15]|
static void _bncavx512_rts_abssum512d(float ret[TSSIZE], __m512 ret8[TSSIZE])
{
    tsfloat ret16_i[16];
    static float tmp[TSSIZE];

    // ret16_i := ret8 (16 lanes)
    _bncavx512_get_ts_m512_i(&ret16_i[0], ret8, 0);
    _bncavx512_get_ts_m512_i(&ret16_i[1], ret8, 1);
    _bncavx512_get_ts_m512_i(&ret16_i[2], ret8, 2);
    _bncavx512_get_ts_m512_i(&ret16_i[3], ret8, 3);
    _bncavx512_get_ts_m512_i(&ret16_i[4], ret8, 4);
    _bncavx512_get_ts_m512_i(&ret16_i[5], ret8, 5);
    _bncavx512_get_ts_m512_i(&ret16_i[6], ret8, 6);
    _bncavx512_get_ts_m512_i(&ret16_i[7], ret8, 7);
    _bncavx512_get_ts_m512_i(&ret16_i[8], ret8, 8);
    _bncavx512_get_ts_m512_i(&ret16_i[9], ret8, 9);
    _bncavx512_get_ts_m512_i(&ret16_i[10], ret8, 10);
    _bncavx512_get_ts_m512_i(&ret16_i[11], ret8, 11);
    _bncavx512_get_ts_m512_i(&ret16_i[12], ret8, 12);
    _bncavx512_get_ts_m512_i(&ret16_i[13], ret8, 13);
    _bncavx512_get_ts_m512_i(&ret16_i[14], ret8, 14);
    _bncavx512_get_ts_m512_i(&ret16_i[15], ret8, 15);

    rts_abs(tmp, ret16_i[0].val);
    rts_set(ret, tmp);

    rts_abs(tmp, ret16_i[1].val);  rts_add(ret, ret, tmp);
    rts_abs(tmp, ret16_i[2].val);  rts_add(ret, ret, tmp);
    rts_abs(tmp, ret16_i[3].val);  rts_add(ret, ret, tmp);
    rts_abs(tmp, ret16_i[4].val);  rts_add(ret, ret, tmp);
    rts_abs(tmp, ret16_i[5].val);  rts_add(ret, ret, tmp);
    rts_abs(tmp, ret16_i[6].val);  rts_add(ret, ret, tmp);
    rts_abs(tmp, ret16_i[7].val);  rts_add(ret, ret, tmp);
    rts_abs(tmp, ret16_i[8].val);  rts_add(ret, ret, tmp);
    rts_abs(tmp, ret16_i[9].val);  rts_add(ret, ret, tmp);
    rts_abs(tmp, ret16_i[10].val); rts_add(ret, ret, tmp);
    rts_abs(tmp, ret16_i[11].val); rts_add(ret, ret, tmp);
    rts_abs(tmp, ret16_i[12].val); rts_add(ret, ret, tmp);
    rts_abs(tmp, ret16_i[13].val); rts_add(ret, ret, tmp);
    rts_abs(tmp, ret16_i[14].val); rts_add(ret, ret, tmp);
    rts_abs(tmp, ret16_i[15].val); rts_add(ret, ret, tmp);
}

// ret := max(|ret8[0]|, |ret8[1]|, ..., |ret8[15]|)
static void _bncavx512_rts_absmax512d(float ret[TSSIZE], __m512 ret8[TSSIZE])
{
    tsfloat ret16_i[16];
    static float tmp[TSSIZE];

    // ret16_i := ret8 (16 lanes)
    _bncavx512_get_ts_m512_i(&ret16_i[0], ret8, 0);
    _bncavx512_get_ts_m512_i(&ret16_i[1], ret8, 1);
    _bncavx512_get_ts_m512_i(&ret16_i[2], ret8, 2);
    _bncavx512_get_ts_m512_i(&ret16_i[3], ret8, 3);
    _bncavx512_get_ts_m512_i(&ret16_i[4], ret8, 4);
    _bncavx512_get_ts_m512_i(&ret16_i[5], ret8, 5);
    _bncavx512_get_ts_m512_i(&ret16_i[6], ret8, 6);
    _bncavx512_get_ts_m512_i(&ret16_i[7], ret8, 7);
    _bncavx512_get_ts_m512_i(&ret16_i[8], ret8, 8);
    _bncavx512_get_ts_m512_i(&ret16_i[9], ret8, 9);
    _bncavx512_get_ts_m512_i(&ret16_i[10], ret8, 10);
    _bncavx512_get_ts_m512_i(&ret16_i[11], ret8, 11);
    _bncavx512_get_ts_m512_i(&ret16_i[12], ret8, 12);
    _bncavx512_get_ts_m512_i(&ret16_i[13], ret8, 13);
    _bncavx512_get_ts_m512_i(&ret16_i[14], ret8, 14);
    _bncavx512_get_ts_m512_i(&ret16_i[15], ret8, 15);

    rts_abs(tmp, ret16_i[0].val);
    rts_set(ret, tmp); // ret := |ret16_i[0]|

    rts_abs(tmp, ret16_i[1].val);  if(rts_cmp(ret, tmp) < 0) rts_set(ret, tmp);
    rts_abs(tmp, ret16_i[2].val);  if(rts_cmp(ret, tmp) < 0) rts_set(ret, tmp);
    rts_abs(tmp, ret16_i[3].val);  if(rts_cmp(ret, tmp) < 0) rts_set(ret, tmp);
    rts_abs(tmp, ret16_i[4].val);  if(rts_cmp(ret, tmp) < 0) rts_set(ret, tmp);
    rts_abs(tmp, ret16_i[5].val);  if(rts_cmp(ret, tmp) < 0) rts_set(ret, tmp);
    rts_abs(tmp, ret16_i[6].val);  if(rts_cmp(ret, tmp) < 0) rts_set(ret, tmp);
    rts_abs(tmp, ret16_i[7].val);  if(rts_cmp(ret, tmp) < 0) rts_set(ret, tmp);
    rts_abs(tmp, ret16_i[8].val);  if(rts_cmp(ret, tmp) < 0) rts_set(ret, tmp);
    rts_abs(tmp, ret16_i[9].val);  if(rts_cmp(ret, tmp) < 0) rts_set(ret, tmp);
    rts_abs(tmp, ret16_i[10].val); if(rts_cmp(ret, tmp) < 0) rts_set(ret, tmp);
    rts_abs(tmp, ret16_i[11].val); if(rts_cmp(ret, tmp) < 0) rts_set(ret, tmp);
    rts_abs(tmp, ret16_i[12].val); if(rts_cmp(ret, tmp) < 0) rts_set(ret, tmp);
    rts_abs(tmp, ret16_i[13].val); if(rts_cmp(ret, tmp) < 0) rts_set(ret, tmp);
    rts_abs(tmp, ret16_i[14].val); if(rts_cmp(ret, tmp) < 0) rts_set(ret, tmp);
    rts_abs(tmp, ret16_i[15].val); if(rts_cmp(ret, tmp) < 0) rts_set(ret, tmp);
}

// ret := || ret8[0]^2 + ret8[1]^2 + ... + ret8[15]^2 ||_2
static void _bncavx512_rts_norm512d(float ret[TSSIZE], __m512 ret8[TSSIZE])
{
    tsfloat ret16_i[16];
    static float tmp[TSSIZE];

    // ret16_i := ret8 (16 lanes)
    _bncavx512_get_ts_m512_i(&ret16_i[0], ret8, 0);
    _bncavx512_get_ts_m512_i(&ret16_i[1], ret8, 1);
    _bncavx512_get_ts_m512_i(&ret16_i[2], ret8, 2);
    _bncavx512_get_ts_m512_i(&ret16_i[3], ret8, 3);
    _bncavx512_get_ts_m512_i(&ret16_i[4], ret8, 4);
    _bncavx512_get_ts_m512_i(&ret16_i[5], ret8, 5);
    _bncavx512_get_ts_m512_i(&ret16_i[6], ret8, 6);
    _bncavx512_get_ts_m512_i(&ret16_i[7], ret8, 7);
    _bncavx512_get_ts_m512_i(&ret16_i[8], ret8, 8);
    _bncavx512_get_ts_m512_i(&ret16_i[9], ret8, 9);
    _bncavx512_get_ts_m512_i(&ret16_i[10], ret8, 10);
    _bncavx512_get_ts_m512_i(&ret16_i[11], ret8, 11);
    _bncavx512_get_ts_m512_i(&ret16_i[12], ret8, 12);
    _bncavx512_get_ts_m512_i(&ret16_i[13], ret8, 13);
    _bncavx512_get_ts_m512_i(&ret16_i[14], ret8, 14);
    _bncavx512_get_ts_m512_i(&ret16_i[15], ret8, 15);

    rts_mul(tmp, ret16_i[0].val, ret16_i[0].val);
    rts_set(ret, tmp);

    rts_mul(tmp, ret16_i[1].val, ret16_i[1].val);   rts_add(ret, ret, tmp);
    rts_mul(tmp, ret16_i[2].val, ret16_i[2].val);   rts_add(ret, ret, tmp);
    rts_mul(tmp, ret16_i[3].val, ret16_i[3].val);   rts_add(ret, ret, tmp);
    rts_mul(tmp, ret16_i[4].val, ret16_i[4].val);   rts_add(ret, ret, tmp);
    rts_mul(tmp, ret16_i[5].val, ret16_i[5].val);   rts_add(ret, ret, tmp);
    rts_mul(tmp, ret16_i[6].val, ret16_i[6].val);   rts_add(ret, ret, tmp);
    rts_mul(tmp, ret16_i[7].val, ret16_i[7].val);   rts_add(ret, ret, tmp);
    rts_mul(tmp, ret16_i[8].val, ret16_i[8].val);   rts_add(ret, ret, tmp);
    rts_mul(tmp, ret16_i[9].val, ret16_i[9].val);   rts_add(ret, ret, tmp);
    rts_mul(tmp, ret16_i[10].val, ret16_i[10].val); rts_add(ret, ret, tmp);
    rts_mul(tmp, ret16_i[11].val, ret16_i[11].val); rts_add(ret, ret, tmp);
    rts_mul(tmp, ret16_i[12].val, ret16_i[12].val); rts_add(ret, ret, tmp);
    rts_mul(tmp, ret16_i[13].val, ret16_i[13].val); rts_add(ret, ret, tmp);
    rts_mul(tmp, ret16_i[14].val, ret16_i[14].val); rts_add(ret, ret, tmp);
    rts_mul(tmp, ret16_i[15].val, ret16_i[15].val); rts_add(ret, ret, tmp);

    rts_sqrt(tmp, ret);
    rts_set(ret, tmp);
}

// e[n] := vec_sum(x[n])  (single-precision variant)
static inline void _bncavx512_fvec_sum(__m512 e[], const __m512 x[], int n)
{
    __m512 s;

    s = x[--n];
    while(--n >= 0)
    {
        s = _bncavx512_ftwo_sum(x[n], s, &e[n + 1]);
    }
    e[0] = s;
}

// y[n] := vec_sum_err_branch(vseb)(k)(e[n])  (single-precision variant)
static inline void _bncavx512_fvseb(__m512 y[], int ny, const __m512 e[], int ne)
{
    int i, j[16], avx_index;
    float ftmp;
    __m512 r, eps, temp, in_y[16];

    j[0] = 0; j[1] = 0; j[2] = 0; j[3] = 0;
    j[4] = 0; j[5] = 0; j[6] = 0; j[7] = 0;
    j[8] = 0; j[9] = 0; j[10] = 0; j[11] = 0;
    j[12] = 0; j[13] = 0; j[14] = 0; j[15] = 0;
    eps = e[0];
    for(i = 0; i < (ne - 2); i++)
    {
        r = _bncavx512_ftwo_sum(eps, e[i + 1], &temp);

        for(avx_index = 0; avx_index < 16; avx_index++)
        {
            if(temp[avx_index] != 0.0)
            {
                in_y[j[avx_index]][avx_index] = r[avx_index];
                eps[avx_index] = temp[avx_index];
                j[avx_index]++;
            }
            else
                eps[avx_index] = r[avx_index];
        }
    }
    for(avx_index = 0; avx_index < 16; avx_index++)
    {
        in_y[j[avx_index]][avx_index] = ftwo_sum(eps[avx_index], e[ne - 1][avx_index], &ftmp);
        in_y[j[avx_index] + 1][avx_index] = ftmp;

        for(i = j[avx_index] + 2; i < ne; i++)
            in_y[i][avx_index] = 0.0;
    }

    for(i = 0; i < ny; i++)
        y[i] = in_y[i];
}

// y[n] := vec_sum_err_branch(vseb)(k)(e[n])  (single-precision variant)
static inline void _bncavx512_fvseb_new(__m512 y[], int ny, const __m512 e[], int ne)
{
    int i;
    __m512 eps, temp, zeros;
    __mmask16 mask16;

    zeros = _mm512_setzero_ps();

    // y := 0
    for(i = 0; i < ny; i++)
        y[i] = _mm512_setzero_ps();

    eps = e[0];
    for(i = 0; i < (ne - 2); i++)
    {
        eps = _bncavx512_ftwo_sum(eps, e[i + 1], &temp);

        // temp != 0
        mask16 = _mm512_cmp_ps_mask(temp, zeros, _CMP_NEQ_OQ);
        eps = _mm512_maskz_and_ps(mask16, temp, zeros);
    }
}

// Merge a[na] & b[nb] into c[na + nb]  (single-precision variant)
static inline void _bncavx512_fmerge(__m512 c[], __m512 a[], int na, __m512 b[], int nb)
{
    int i, j, k;
    int avx_index;

    for(avx_index = 0; avx_index < 16; avx_index++)
    {
        i = j = k = 0;
        while((i < na) && (j < nb))
        {
            if(fabsf(a[i][avx_index]) >= fabsf(b[j][avx_index]))
                c[k++][avx_index] = a[i++][avx_index];
            else
                c[k++][avx_index] = b[j++][avx_index];
        }
        while(i < na)
            c[k++][avx_index] = a[i++][avx_index];
        while(j < nb)
            c[k++][avx_index] = b[j++][avx_index];
    }
}

#include "sort.h" // _bncavx512_bubble_sort

#ifdef USE_TS_BF
    #define _bncavx512_rts_add _bncavx512_rts_add_bf
    #define _bncavx512_rts_mul _bncavx512_rts_mul_bf
#else // USE_TS_BF
    #define _bncavx512_rts_add _bncavx512_rts_addq
    #define _bncavx512_rts_mul _bncavx512_rts_mult
#endif // USE_TS_BF

// defined in QS arith (forward declaration only)
static inline void _bncavx512_rts_addq(__m512 ret[TSSIZE], __m512 a[TSSIZE], __m512 b[TSSIZE]);

static inline void _bncavx512_rts_addt(__m512 ret[TSSIZE], __m512 a[TSSIZE], __m512 b[TSSIZE])
{
    __m512 z[6], e[6];

    _bncavx512_fmerge(z, a, 3, b, 3);

    _bncavx512_fvec_sum(e, z, 6);
    _bncavx512_fvseb(ret, 3, e, 6);
}

// Branch free algorithm by D.K.Zhang and A.Aiken at SC2025
static inline void _bncavx512_rts_add_bf(__m512 ret[TSSIZE], __m512 a[TSSIZE], __m512 b[TSSIZE])
{
    __m512 a0, b0, c0, d0, e0, f0;
    __m512 a1, b1, c1, d1, e1, f1;
    __m512 a2, b2, c2, d2, e2;
    __m512 a3, b3, c3, d3;
    __m512 c4;
    __m512 c5, d5;
    __m512 b6, c6;
    __m512 a7, b7, c7;
    __m512 b8, c8;

    a0 = a[0];
    b0 = b[0];
    c0 = a[1];
    d0 = b[1];
    e0 = a[2];
    f0 = b[2];
    a1 = _bncavx512_ftwo_sum(a0, b0, &b1);
    c1 = _bncavx512_ftwo_sum(c0, d0, &d1);
    e1 = _bncavx512_ftwo_sum(e0, f0, &f1);
    a2 = _bncavx512_fquick_two_sum(a1, c1, &c2);
    b2 = _mm512_add_ps(b1, f1);
    d2 = _bncavx512_ftwo_sum(d1, e1, &e2);
    a3 = _bncavx512_fquick_two_sum(a2, d2, &d3);
    b3 = _bncavx512_ftwo_sum(b2, c2, &c3);
    c4 = _mm512_add_ps(c3, e2);
    c5 = _bncavx512_ftwo_sum(c4, d3, &d5);
    b6 = _bncavx512_ftwo_sum(b3, c5, &c6);
    a7 = _bncavx512_fquick_two_sum(a3, b6, &b7);
    c7 = _mm512_add_ps(c6, d5);
    b8 = _bncavx512_fquick_two_sum(b7, c7, &c8);
    ret[0] = a7;
    ret[1] = b8;
    ret[2] = c8;
}


// mul (forward declaration only)
static inline void _bncavx512_rts_mulq(__m512 ret[TSSIZE], __m512 a[TSSIZE], __m512 b[TSSIZE]);

// mul
static inline void _bncavx512_rts_mult(__m512 ret[TSSIZE], __m512 a[TSSIZE], __m512 b[TSSIZE])
{
    __m512 z00[2], z01[2], z10[2];
    __m512 in_b[3], in_c, z[3], e[4], temp[4];

    z00[0] = _bncavx512_ftwo_prod(a[0], b[0], &z00[1]);
    z01[0] = _bncavx512_ftwo_prod(a[0], b[1], &z01[1]);
    z10[0] = _bncavx512_ftwo_prod(a[1], b[0], &z10[1]);

    z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
    _bncavx512_fvec_sum(in_b, z, 3);
    in_c = _mm512_fmadd_ps(a[1], b[1], in_b[2]);

    z[0] = _mm512_fmadd_ps(a[0], b[2], z10[1]);
    z[1] = _mm512_fmadd_ps(a[2], b[0], z01[1]);
    z[2] = _mm512_add_ps(z[0], z[1]);
    temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1];
    temp[3] = _mm512_add_ps(in_c, z[2]);
    _bncavx512_fvec_sum(e, temp, 4);
    ret[0] = e[0];
    _bncavx512_fvseb(&ret[1], 2, &e[1], 3);
}

// Branch free algorithm by D.K.Zhang and A.Aiken at SC2025
static inline void _bncavx512_rts_mul_bf(__m512 ret[TSSIZE], __m512 a[TSSIZE], __m512 b[TSSIZE])
{
    __m512 a0, b0, c0, d0, e0, f0, g0, h0, i0;
    __m512 c1, d1, e1, f1, g1;
    __m512 b2, c2, g2;
    __m512 a3, b3, c3, e3;
    __m512 c4;
    __m512 b5, c5;
    __m512 a6, b6;
    __m512 b7, c7;

    a0 = _bncavx512_ftwo_prod(a[0], b[0], &b0);
    c0 = _bncavx512_ftwo_prod(a[0], b[1], &e0);
    d0 = _bncavx512_ftwo_prod(a[1], b[0], &f0);
    g0 = _mm512_mul_ps(a[0], b[2]);
    h0 = _mm512_mul_ps(a[1], b[1]);
    i0 = _mm512_mul_ps(a[2], b[0]);
    c1 = _bncavx512_ftwo_sum(c0, d0, &d1);
    e1 = _mm512_add_ps(e0, f0);
    g1 = _mm512_add_ps(g0, i0);
    b2 = _bncavx512_ftwo_sum(b0, c1, &c2);
    g2 = _mm512_add_ps(g1, h0);
    a3 = _bncavx512_fquick_two_sum(a0, b2, &b3);
    c3 = _mm512_add_ps(c2, d1);
    e3 = _mm512_add_ps(e1, g2);
    c4 = _mm512_add_ps(c3, e3);
    b5 = _bncavx512_fquick_two_sum(b3, c4, &c5);
    a6 = _bncavx512_fquick_two_sum(a3, b5, &b6);
    b7 = _bncavx512_fquick_two_sum(b6, c5, &c7);
    ret[0] = a6;
    ret[1] = b7;
    ret[2] = c7;
}


// c[3] := -a[3]
static inline void _bncavx512_rts_neg(__m512 c[TSSIZE], __m512 a[TSSIZE])
{
    __m512 zero8;

    zero8 = _mm512_setzero_ps();

    c[0] = _mm512_sub_ps(zero8, a[0]);
    c[1] = _mm512_sub_ps(zero8, a[1]);
    c[2] = _mm512_sub_ps(zero8, a[2]);
}

// c[3] := a[3] - b[3]
static inline void _bncavx512_rts_sub(__m512 c[TSSIZE], __m512 a[TSSIZE], __m512 b[TSSIZE])
{
    __m512 mb[TSSIZE];

    _bncavx512_rts_neg(mb, b);
    _bncavx512_rts_add(c, a, mb);
}

// c[3] := a[3] - b[3]
static inline void _bncavx512_rts_subq(__m512 c[TSSIZE], __m512 a[TSSIZE], __m512 b[TSSIZE])
{
    __m512 mb[TSSIZE];

    _bncavx512_rts_neg(mb, b);
    _bncavx512_rts_addq(c, a, mb);
}

// divq (forward declaration)
static inline void _bncavx512_rts_divq(__m512 ret[TSSIZE], __m512 a[TSSIZE], __m512 b[TSSIZE]);

// r[3] := to_ts(a, b, c)
static inline void _bncavx512_to_ts(__m512 r[TSSIZE], __m512 a, __m512 b, __m512 c)
{
    __m512 d[3], e[3];

    d[0] = _bncavx512_ftwo_sum(a, b, &d[1]);
    d[2] = c;
    _bncavx512_fvec_sum(e, d, 3);
    _bncavx512_fvseb(r, 3, e, 3);
}

// c[3] := a * b[3]
static inline void _bncavx512_rts_mul_d(__m512 c[TSSIZE], __m512 b[TSSIZE], __m512 a)
{
    __m512 z00[2], z01[2], z10[2];
    __m512 in_b[3], z[3], e[4], temp[4];

    z00[0] = _bncavx512_ftwo_prod(a, b[0], &z00[1]);
    z01[0] = _bncavx512_ftwo_prod(a, b[1], &z01[1]);
    z[0] = z00[1]; z[1] = z01[0];
    _bncavx512_fvec_sum(in_b, z, 2);

    z[0] = _mm512_fmadd_ps(a, b[2], z10[1]);
    z[1] = _mm512_add_ps(z[0], z01[1]);
    temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; temp[3] = z[1];
    _bncavx512_fvec_sum(e, temp, 4);
    c[0] = e[0];
    _bncavx512_fvseb(&c[1], 2, &e[1], 3);
}

// c[3] := a[2] * b[3]
static inline void _bncavx512_rts_mul_ds(__m512 c[TSSIZE], __m512 a[DSSIZE], __m512 b[TSSIZE])
{
    __m512 z00[2], z01[2], z10[2];
    __m512 in_b[3], in_c, z[3], e[4], temp[4];

    z00[0] = _bncavx512_ftwo_prod(a[0], b[0], &z00[1]);
    z01[0] = _bncavx512_ftwo_prod(a[0], b[1], &z01[1]);
    z10[0] = _bncavx512_ftwo_prod(a[1], b[0], &z10[1]);
    z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
    _bncavx512_fvec_sum(in_b, z, 3);
    in_c = _mm512_fmadd_ps(a[1], b[1], in_b[2]);

    z[0] = _mm512_fmadd_ps(a[0], b[2], z10[1]);
    z[1] = _mm512_add_ps(z[0], z01[1]);
    temp[0] = z00[0];
    temp[1] = in_b[0];
    temp[2] = in_b[1];
    temp[3] = _mm512_add_ps(in_c, z[1]);
    _bncavx512_fvec_sum(e, temp, 4);
    c[0] = e[0];
    _bncavx512_fvseb(&c[1], 2, &e[1], 3);
}

#define _bncavx512_rts_div _bncavx512_rts_divtq

// div
static inline void _bncavx512_rts_divt(__m512 ret[TSSIZE], __m512 a[TSSIZE], __m512 b[TSSIZE])
{
    __m512 alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];
    __m512 zero4, two4, one_p_2flt_eps4, one_m_2flt_eps4;

    zero4 = _mm512_setzero_ps();
    two4 = _mm512_set1_ps(2.0);
    one_p_2flt_eps4 = _mm512_set1_ps(ONE_P_2FLT_EPS);
    one_m_2flt_eps4 = _mm512_set1_ps(ONE_M_2FLT_EPS);

    _bncavx512_to_ts(d2, two4, zero4, zero4);

    alpha = _mm512_div_ps(one_p_2flt_eps4, b[0]);

    h1 = _mm512_fmsub_ps(alpha, b[0], one_p_2flt_eps4);

    h1 = _mm512_fmadd_ps(alpha, b[1], h1);
    h1 = _mm512_sub_ps(zero4, h1);

    in_b[0] = _bncavx512_ftwo_prod(alpha, one_m_2flt_eps4, &in_b[1]);

    in_b12 = _mm512_fmadd_ps(alpha, h1, in_b[1]);

    in_b[0] = _bncavx512_fquick_two_sum(in_b[0], in_b12, &in_b[1]);
    _bncavx512_rts_mul_ds(temp, in_b, b);
    _bncavx512_rts_sub(temp, d2, temp);
    _bncavx512_rts_mul_ds(in_c, in_b, a);
    _bncavx512_rts_mul(ret, in_c, temp);
}
// div
static inline void _bncavx512_rts_divtq(__m512 ret[TSSIZE], __m512 a[TSSIZE], __m512 b[TSSIZE])
{
    __m512 alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];
    __m512 zero4, two4, one_p_2flt_eps4, one_m_2flt_eps4;

    zero4 = _mm512_setzero_ps();
    two4 = _mm512_set1_ps(2.0);
    one_p_2flt_eps4 = _mm512_set1_ps(ONE_P_2FLT_EPS);
    one_m_2flt_eps4 = _mm512_set1_ps(ONE_M_2FLT_EPS);

    _bncavx512_to_ts(d2, two4, zero4, zero4);

    alpha = _mm512_div_ps(one_p_2flt_eps4, b[0]);

    h1 = _mm512_fmsub_ps(alpha, b[0], one_p_2flt_eps4);

    h1 = _mm512_fmadd_ps(alpha, b[1], h1);
    h1 = _mm512_sub_ps(zero4, h1);

    in_b[0] = _bncavx512_ftwo_prod(alpha, one_m_2flt_eps4, &in_b[1]);

    in_b12 = _mm512_fmadd_ps(alpha, h1, in_b[1]);

    in_b[0] = _bncavx512_fquick_two_sum(in_b[0], in_b12, &in_b[1]);
    _bncavx512_rts_mul_ds(temp, in_b, b);
    _bncavx512_rts_subq(temp, d2, temp);
    _bncavx512_rts_mul_ds(in_c, in_b, a);
    _bncavx512_rts_mul(ret, in_c, temp);
}

// Aliases for naming variants used in *linear.c
#define _bncavx2_rts_sum512    _bncavx512_rts_sum512d
#define _bncavx2_rts_sum512d   _bncavx512_rts_sum512d
#define _bncavx512_rts_sum512  _bncavx512_rts_sum512d
#define _bncavx512_rts_abssum512  _bncavx512_rts_abssum512d
#define _bncavx512_rts_norm512    _bncavx512_rts_norm512d

#endif // __AVX512F__
#endif //ndef __BNCAVX_TS_AVX512_H
