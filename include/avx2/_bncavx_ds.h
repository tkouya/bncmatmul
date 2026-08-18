// ------------------------
// ---------- DS ----------
// ------------------------
#ifndef __BNCAVX_DS_H
#define __BNCAVX_DS_H

// defined in rdd.h
#ifndef DSSIZE
    #define DSSIZE 2
#endif // DSSIZE

// ret := 0
#if defined(__AVX2__)
static inline void _bncavx2_set0_ds(__m256 ret[DSSIZE])
{
    ret[0] = _mm256_setzero_ps();
    ret[1] = _mm256_setzero_ps();
}
#define _bncavx2_rds_set0(ret) _bncavx2_set0_ds((ret))
#endif // __AVX2__

// ret := 0
#if defined(__AVX512F__)
static inline void _bncavx512_set0_ds(__m512 ret[DSSIZE])
{
    ret[0] = _mm512_setzero_ps();
    ret[1] = _mm512_setzero_ps();
}
#endif // __AVX512F__

#if defined(__AVX2__)
// ret := ret4[][avx_index]
static inline void _bncavx2_get_ds_m256_i(dsfloat *ret, __m256 ret4[DSSIZE], int avx_index)
{
    ret->val[0] = ret4[0][avx_index];
    ret->val[1] = ret4[1][avx_index];

    return;
}

// ret := ret4[0] + ret4[1] + ret4[2] + ret4[3]
static void _bncavx2_rds_sum256(float ret[DSSIZE], __m256 ret8[DSSIZE])
{
    dsfloat ret8_i[8];

    // ret4_i := ret4
    _bncavx2_get_ds_m256_i(&ret8_i[0], ret8, 0);
    _bncavx2_get_ds_m256_i(&ret8_i[1], ret8, 1);
    _bncavx2_get_ds_m256_i(&ret8_i[2], ret8, 2);
    _bncavx2_get_ds_m256_i(&ret8_i[3], ret8, 3);
    _bncavx2_get_ds_m256_i(&ret8_i[4], ret8, 4);
    _bncavx2_get_ds_m256_i(&ret8_i[5], ret8, 5);
    _bncavx2_get_ds_m256_i(&ret8_i[6], ret8, 6);
    _bncavx2_get_ds_m256_i(&ret8_i[7], ret8, 7);

    rds_set(ret, ret8_i[0].val);
    rds_add(ret, ret, ret8_i[1].val);
    rds_add(ret, ret, ret8_i[2].val);
    rds_add(ret, ret, ret8_i[3].val);
    rds_add(ret, ret, ret8_i[4].val);
    rds_add(ret, ret, ret8_i[5].val);
    rds_add(ret, ret, ret8_i[6].val);
    rds_add(ret, ret, ret8_i[7].val);
}

// ret := |ret4[0]| + |ret4[1]| + |ret4[2]| + |ret4[3]|
static void _bncavx2_rds_abssum256(float ret[DSSIZE], __m256 ret8[DSSIZE])
{
    dsfloat ret8_i[8];
    float tmp[DSSIZE];

    // ret8_i := ret4
    _bncavx2_get_ds_m256_i(&ret8_i[0], ret8, 0);
    _bncavx2_get_ds_m256_i(&ret8_i[1], ret8, 1);
    _bncavx2_get_ds_m256_i(&ret8_i[2], ret8, 2);
    _bncavx2_get_ds_m256_i(&ret8_i[3], ret8, 3);
    _bncavx2_get_ds_m256_i(&ret8_i[4], ret8, 4);
    _bncavx2_get_ds_m256_i(&ret8_i[5], ret8, 5);
    _bncavx2_get_ds_m256_i(&ret8_i[6], ret8, 6);
    _bncavx2_get_ds_m256_i(&ret8_i[7], ret8, 7);

    rds_abs(tmp, ret8_i[0].val); rds_set(ret, tmp);
    rds_abs(tmp, ret8_i[1].val); rds_add(ret, ret, tmp);
    rds_abs(tmp, ret8_i[2].val); rds_add(ret, ret, tmp);
    rds_abs(tmp, ret8_i[3].val); rds_add(ret, ret, tmp);
    rds_abs(tmp, ret8_i[4].val); rds_add(ret, ret, tmp);
    rds_abs(tmp, ret8_i[5].val); rds_add(ret, ret, tmp);
    rds_abs(tmp, ret8_i[6].val); rds_add(ret, ret, tmp);
    rds_abs(tmp, ret8_i[7].val); rds_add(ret, ret, tmp);
}

// ret := max(|ret4[0]|, |ret4[1]|, |ret4[2]|, |ret4[3]|)
static void _bncavx2_rds_absmax256(float ret[DSSIZE], __m256 ret8[DSSIZE])
{
    dsfloat ret8_i[8];
    float tmp[DSSIZE];

    // ret8_i := ret8
    _bncavx2_get_ds_m256_i(&ret8_i[0], ret8, 0);
    _bncavx2_get_ds_m256_i(&ret8_i[1], ret8, 1);
    _bncavx2_get_ds_m256_i(&ret8_i[2], ret8, 2);
    _bncavx2_get_ds_m256_i(&ret8_i[3], ret8, 3);
    _bncavx2_get_ds_m256_i(&ret8_i[4], ret8, 4);
    _bncavx2_get_ds_m256_i(&ret8_i[5], ret8, 5);
    _bncavx2_get_ds_m256_i(&ret8_i[6], ret8, 6);
    _bncavx2_get_ds_m256_i(&ret8_i[7], ret8, 7);

    rds_abs(tmp, ret8_i[0].val);
    rds_set(ret, tmp); // ret := |ret4_i[0]|

    rds_abs(tmp, ret8_i[1].val);
    if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp); //   ret := |ret8_i[1]|
    rds_abs(tmp, ret8_i[2].val);
    if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp); //   ret := |ret8_i[2]|
    rds_abs(tmp, ret8_i[3].val);
    if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp); //   ret := |ret8_i[3]|
    rds_abs(tmp, ret8_i[4].val);
    if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp); //   ret := |ret8_i[4]|
    rds_abs(tmp, ret8_i[5].val);
    if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp); //   ret := |ret8_i[5]|
    rds_abs(tmp, ret8_i[6].val);
    if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp); //   ret := |ret8_i[6]|
    rds_abs(tmp, ret8_i[7].val);
    if(rds_cmp(ret, tmp) < 0) rds_set(ret, tmp); //   ret := |ret8_i[7]|
}

// ret := || ret4[0]^2 + ret4[1]^2 + ret4[2]^2 + ret4[3]^2 ||_2
static void _bncavx2_rds_norm256(float ret[DSSIZE], __m256 ret8[DSSIZE])
{
    dsfloat ret8_i[8];
    float tmp[DSSIZE];

    // ret8_i := ret8
    _bncavx2_get_ds_m256_i(&ret8_i[0], ret8, 0);
    _bncavx2_get_ds_m256_i(&ret8_i[1], ret8, 1);
    _bncavx2_get_ds_m256_i(&ret8_i[2], ret8, 2);
    _bncavx2_get_ds_m256_i(&ret8_i[3], ret8, 3);
    _bncavx2_get_ds_m256_i(&ret8_i[4], ret8, 4);
    _bncavx2_get_ds_m256_i(&ret8_i[5], ret8, 5);
    _bncavx2_get_ds_m256_i(&ret8_i[6], ret8, 6);
    _bncavx2_get_ds_m256_i(&ret8_i[7], ret8, 7);

    rds_mul(tmp, ret8_i[0].val, ret8_i[0].val); rds_set(ret, tmp);
    rds_mul(tmp, ret8_i[1].val, ret8_i[1].val); rds_add(ret, ret, tmp);
    rds_mul(tmp, ret8_i[2].val, ret8_i[2].val); rds_add(ret, ret, tmp);
    rds_mul(tmp, ret8_i[3].val, ret8_i[3].val); rds_add(ret, ret, tmp);
    rds_mul(tmp, ret8_i[4].val, ret8_i[4].val); rds_add(ret, ret, tmp);
    rds_mul(tmp, ret8_i[5].val, ret8_i[5].val); rds_add(ret, ret, tmp);
    rds_mul(tmp, ret8_i[6].val, ret8_i[6].val); rds_add(ret, ret, tmp);
    rds_mul(tmp, ret8_i[7].val, ret8_i[7].val); rds_add(ret, ret, tmp);

    rds_sqrt(tmp, ret);
    rds_set(ret, tmp);
}
#endif // __AVX2__

#if defined(__AVX512F__)
// ret := ret8[][avx_index]
static inline void _bncavx2_get_ds_m512_i(dsfloat *ret, __m512d ret16[DSSIZE], int avx_index)
{
    ret->val[0] = ret16[0][avx_index];
    ret->val[1] = ret16[1][avx_index];

    return;
}

// ret := ret4[0] + ret4[1] + ret4[2]
static void _bncavx2_rds_sum512(float ret[DSSIZE], __m512d ret16[DSSIZE])
{
    dsfloat ret16_i[16];

    // ret16_i := ret16
    _bncavx2_get_ds_m512_i(&ret16_i[0], ret16, 0);
    _bncavx2_get_ds_m512_i(&ret16_i[1], ret16, 1);
    _bncavx2_get_ds_m512_i(&ret16_i[2], ret16, 2);
    _bncavx2_get_ds_m512_i(&ret16_i[3], ret16, 3);
    _bncavx2_get_ds_m512_i(&ret16_i[4], ret16, 4);
    _bncavx2_get_ds_m512_i(&ret16_i[5], ret16, 5);
    _bncavx2_get_ds_m512_i(&ret16_i[6], ret16, 6);
    _bncavx2_get_ds_m512_i(&ret16_i[7], ret16, 7);
    _bncavx2_get_ds_m512_i(&ret16_i[8], ret16, 8);
    _bncavx2_get_ds_m512_i(&ret16_i[9], ret16, 9);
    _bncavx2_get_ds_m512_i(&ret16_i[10], ret16, 10);
    _bncavx2_get_ds_m512_i(&ret16_i[11], ret16, 11);
    _bncavx2_get_ds_m512_i(&ret16_i[12], ret16, 12);
    _bncavx2_get_ds_m512_i(&ret16_i[13], ret16, 13);
    _bncavx2_get_ds_m512_i(&ret16_i[14], ret16, 14);
    _bncavx2_get_ds_m512_i(&ret16_i[16], ret16, 15);

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
#endif // __AVX512F__

#if defined(__AVX2__)
// element[1] -> { v0.x[1], v1.x[0], ... }
// rdd with avx2
// a0 = (a[0].0, a[1].0), b0 = (b[0].0, b[1].0)
// a1 = (a[0],1, a[1].1), b1 = (b[0],1, b[1].1)
// a2 = (a[0].2, a[1].2), b2 = (b[0].2, b[1].2)
// a3 = (a[0],3, a[1].3), b3 = (b[0],3, b[1].3)
// -> ret[0].0 = a0 + b0
//    ret[0].1 = a1 + b1
//    ret[0].2 = a0 + b2
//    ret[0].3 = a0 + b3
static inline void _bncavx2_rds_add(__m256 ret[DSSIZE], __m256 a[DSSIZE], __m256 b[DSSIZE])
{
    // c_ds_add_sloppy
    //float s, e;
    __m256 s, e;

    //s = two_sum(a[0], b[0], &e);
    s = _bncavx2_ftwo_sum(a[0], b[0], &e);
	//e += (a[1] + b[1]);
    e = _mm256_add_ps(e, _mm256_add_ps(a[1], b[1]));

	//c[0] = quick_two_sum(s, e, &e);
	//c[1] = e;
    ret[0] = _bncavx2_fquick_two_sum(s, e, &e);
    ret[1] = e;
}

// ret := a - b
static inline void _bncavx2_rds_sub(__m256 ret[DSSIZE], __m256 a[DSSIZE], __m256 b[DSSIZE])
{
    // c_ds_sub_sloppy
	//float s1, s2, t1, t2;
    __m256 s1, s2, t1, t2;

//	s1 = two_diff(a[0], b[0], &s2);
//	t1 = two_diff(a[1], b[1], &t2);
    s1 = _bncavx2_ftwo_diff(a[0], b[0], &s2);
    t1 = _bncavx2_ftwo_diff(a[1], b[1], &t2);

//	s2 += t1;
    s2 = _mm256_add_ps(s2, t1);
//	s1 = quick_two_sum(s1, s2, &s2);
    s1 = _bncavx2_fquick_two_sum(s1, s2, &s2);
//	s2 += t2;
    s2 = _mm256_add_ps(s2, t2);
//	s1 = quick_two_sum(s1, s2, &s2);
    s1 = _bncavx2_fquick_two_sum(s1, s2, &s2);
	//return dd_real(s1, s2);
//	c[0] = s1;
//	c[1] = s2;
    ret[0] = s1;
    ret[1] = s2;
}

// mul
static inline void _bncavx2_rds_mul(__m256 ret[DSSIZE], __m256 a[DSSIZE], __m256 b[DSSIZE])
{
	//float p1, p2;
    __m256 p1, p2;

//	p1 = two_prod(a[0], b[0], &p2);
    p1 = _bncavx2_ftwo_prod(a[0], b[0], &p2);

//	p2 += (a[0] * b[1] + a[1] * b[0]);
    p2 = _mm256_add_ps(p2, 
        _mm256_add_ps(
            _mm256_mul_ps(a[0], b[1]),
            _mm256_mul_ps(a[1], b[0])
        )
    );

//	p1 = quick_two_sum(p1, p2, &p2);
//	c[0] = p1;
//	c[1] = p2;

    ret[0] = _bncavx2_fquick_two_sum(p1, p2, &p2);
    ret[1] = p2;
}

// dd * d
static inline void _bncavx2_rds_mul_f(__m256 ret[DSSIZE], __m256 a[DSSIZE], __m256 b)
{
	//float p1, p2;
    __m256 p1, p2;

//	p1 = two_prod(a[0], b, &p2);
    p1 = _bncavx2_ftwo_prod(a[0], b, &p2);

//	p2 += a[1] * b;
    p2 = _mm256_add_ps(p2, 
        _mm256_mul_ps(a[1], b)
    );

//	p1 = quick_two_sum(p1, p2, &p2);
//	c[0] = p1;
//	c[1] = p2;

    ret[0] = _bncavx2_fquick_two_sum(p1, p2, &p2);
    ret[1] = p2;
//#endif // 0
}

// sloppy_div
static inline void _bncavx2_rds_div(__m256 ret[DSSIZE], __m256 a[DSSIZE], __m256 b[DSSIZE])
{
    // c_ds_sloppy_div(float *a, float *b, float *c)

	//float s1, s2;
	//float q1, q2;
	//float r[DSSIZE];
	__m256 s1, s2;
	__m256 q1, q2;
	__m256 r[DSSIZE];

	//q1 = a[0] / b[0];  /* approximate quotient */
    q1 = _mm256_div_ps(a[0], b[0]);

	/* compute  this - q1 * dd */
	//r = b * q1;
	//c_ds_mul_ds_d(b, q1, r);
    _bncavx2_rds_mul_f(r, b, q1);

	//s1 = two_diff(a[0], r[0], &s2);
    s1 = _bncavx2_ftwo_diff(a[0], r[0], &s2);
	//s2 -= r[1];
    s2 = _mm256_sub_ps(s2, r[1]);
	//s2 += a[1];
    s2 = _mm256_add_ps(s2, a[1]);

	/* get next approximation */
	//q2 = (s1 + s2) / b[0];
    q2 = _mm256_div_ps(
        _mm256_add_ps(s1, s2),
        b[0]
    );

	/* renormalize */
	//r[0] = quick_two_sum(q1, q2, &r[1]);
    ret[0] = _bncavx2_fquick_two_sum(q1, q2, &ret[1]);
}

// abs
static inline void _bncavx2_rds_abs(__m256 ret[DSSIZE], __m256 a[DSSIZE])
{
    int avx_index;

    for(avx_index = 0; avx_index < _BNC_S_WIDTH; avx_index++)
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

// ret := val
static inline void _bncavx2_rds_set(__m256 ret[DSSIZE], __m256 val[DSSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
}

// ret := (float)val
static inline void _bncavx2_rds_set_f(__m256 ret[DSSIZE], __m256 val)
{
    ret[0] = val;
    ret[1] = _mm256_setzero_ps();
}

// ret := [val[0] val[0] ... val[1] val[1] ...]
static inline void _bncavx2_rds_set1_ds(__m256 ret[DSSIZE], float val[DSSIZE])
{
    ret[0] = _mm256_set1_ps(val[0]);
    ret[1] = _mm256_set1_ps(val[1]);
}

// ret := -val
static inline void _bncavx2_rds_neg(__m256 ret[DSSIZE], __m256 val[DSSIZE])
{
    ret[0] = _mm256_sub_ps(_mm256_setzero_ps(), val[0]);
    ret[1] = _mm256_sub_ps(_mm256_setzero_ps(), val[1]);
}

// Branch-free DS add (Zhang & Aiken SC2025-style)
static inline void _bncavx2_rds_add_bf(__m256 ret[DSSIZE], __m256 a[DSSIZE], __m256 b[DSSIZE])
{
    __m256 g1, g2, g3, g4, g5, g6;
    __m256 g1e, g2e, g3e, g6e;

    g1 = _bncavx2_ftwo_sum(a[0], b[0], &g1e);
    g2 = _bncavx2_ftwo_sum(a[1], b[1], &g2e);
    g3 = _bncavx2_fquick_two_sum(g1, g2, &g3e);
    g4 = _mm256_add_ps(g1e, g2e);
    g5 = _mm256_add_ps(g4, g3e);
    g6 = _bncavx2_fquick_two_sum(g3, g5, &g6e);
    ret[0] = g6;
    ret[1] = g6e;
}

// Branch-free DS mul
static inline void _bncavx2_rds_mul_bf(__m256 ret[DSSIZE], __m256 a[DSSIZE], __m256 b[DSSIZE])
{
    __m256 p00, p01, p10;
    __m256 pe00;
    __m256 g1, g2, g3, ge3;

    p00 = _bncavx2_ftwo_prod(a[0], b[0], &pe00);
    p01 = _mm256_mul_ps(a[0], b[1]);
    p10 = _mm256_mul_ps(a[1], b[0]);
    g1 = _mm256_add_ps(p01, p10);
    g2 = _mm256_add_ps(pe00, g1);
    g3 = _bncavx2_fquick_two_sum(p00, g2, &ge3);

    ret[0] = g3;
    ret[1] = ge3;
}

// ret := a + b (DS + float)
static inline void _bncavx2_rds_add_f(__m256 ret[DSSIZE], __m256 a[DSSIZE], __m256 b)
{
    __m256 s1, s2;

    s1 = _bncavx2_ftwo_sum(a[0], b, &s2);
    s2 = _mm256_add_ps(s2, a[1]);
    s1 = _bncavx2_fquick_two_sum(s1, s2, &s2);
    ret[0] = s1;
    ret[1] = s2;
}

// ret := a - b (DS - float)
static inline void _bncavx2_rds_sub_f(__m256 ret[DSSIZE], __m256 a[DSSIZE], __m256 b)
{
    __m256 s1, s2;
    s1 = _bncavx2_ftwo_diff(a[0], b, &s2);
    s2 = _mm256_add_ps(s2, a[1]);
    s1 = _bncavx2_fquick_two_sum(s1, s2, &s2);
    ret[0] = s1;
    ret[1] = s2;
}

// ret := a / b (DS / float)
static inline void _bncavx2_rds_div_f(__m256 ret[DSSIZE], __m256 a[DSSIZE], __m256 b)
{
    __m256 q1, q2;
    __m256 p1, p2;
    __m256 s, e;
    __m256 r[DSSIZE];

    q1 = _mm256_div_ps(a[0], b);

    p1 = _bncavx2_ftwo_prod(q1, b, &p2);
    s = _bncavx2_ftwo_diff(a[0], p1, &e);
    e = _mm256_add_ps(e, a[1]);
    e = _mm256_sub_ps(e, p2);
    q2 = _mm256_div_ps(_mm256_add_ps(s, e), b);

    r[0] = _bncavx2_fquick_two_sum(q1, q2, &r[1]);
    ret[0] = r[0];
    ret[1] = r[1];
}

/* ================================================================
 *  AVX2 FMA helpers for DS
 *  Processes 8 DS operations per call
 * ================================================================ */

// ret[DSSIZE] := a * b + c   (scalar x scalar -> DS) x 8
static inline void _bncavx2_rds_fma_f_f(
        __m256 ret[DSSIZE],
        __m256 a,
        __m256 b,
        __m256 c)
{
    ret[0] = _mm256_fmadd_ps(a, b, c);
    ret[1] = _mm256_fmadd_ps(a, b, _mm256_sub_ps(c, ret[0]));
}

// ret[DSSIZE] := a * b + c[DSSIZE]  (scalar x scalar + DS -> DS) x 8
static inline void _bncavx2_rds_fma_f_ds(
        __m256       ret[DSSIZE],
        __m256       a,
        __m256       b,
        const __m256 c[DSSIZE])
{
    __m256 hi = _mm256_fmadd_ps(a, b, c[0]);
    __m256 t  = _mm256_sub_ps(c[0], hi);
    __m256 e  = _mm256_fmadd_ps(a, b, t);
    __m256 f  = _mm256_add_ps(e, c[1]);
    ret[0] = _bncavx2_fquick_two_sum(hi, f, &(ret[1]));
}

// ret[DSSIZE] := a * b[DSSIZE] + c[DSSIZE]  (scalar x DS + DS -> DS) x 8
static inline void _bncavx2_rds_fma_fm_ds(
        __m256       ret[DSSIZE],
        __m256       a,
        const __m256 b[DSSIZE],
        const __m256 c[DSSIZE])
{
    __m256 hi = _mm256_fmadd_ps(a, b[0], c[0]);
    __m256 t  = _mm256_sub_ps(c[0], hi);
    __m256 e  = _mm256_fmadd_ps(a, b[0], t);
    __m256 f  = _mm256_add_ps(e, c[1]);
    __m256 lo = _mm256_fmadd_ps(a, b[1], f);
    ret[0] = _bncavx2_fquick_two_sum(hi, lo, &(ret[1]));
}

// ret[DSSIZE] := a[DSSIZE] * b[DSSIZE] + c[DSSIZE]
static inline void _bncavx2_rds_fma(
        __m256       ret[DSSIZE],
        const __m256 a[DSSIZE],
        const __m256 b[DSSIZE],
        const __m256 c[DSSIZE])
{
    __m256 d0 = _mm256_fmadd_ps(a[0], b[0], c[0]);
    __m256 t  = _mm256_sub_ps(c[0], d0);
    __m256 e  = _mm256_fmadd_ps(a[0], b[0], t);
    __m256 f  = _mm256_add_ps(e, c[1]);
    __m256 g  = _mm256_fmadd_ps(a[0], b[1], f);
    __m256 d1 = _mm256_fmadd_ps(a[1], b[0], g);
    ret[0] = _bncavx2_fquick_two_sum(d0, d1, &(ret[1]));
}
#endif // defined(__AVX2__)

#ifdef __BNC_DSLINEAR_H__
//#include "ddv_addmul.c"
#endif //__BNC_DSLINEAR_H__

#endif // ifndef __BNCAVX_DS_H
