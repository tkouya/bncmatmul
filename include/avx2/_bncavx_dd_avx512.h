// ------------------------
// ---------- DD AVX-512 --
// ------------------------
#ifndef __BNCAVX_DD_AVX512_H
#define __BNCAVX_DD_AVX512_H

// defined in rdd.h
#ifndef DDSIZE
    #define DDSIZE 2
#endif // DDSIZE

#if defined(__AVX512F__)

// ret := 0
static inline void _bncavx512_set0_dd(__m512d ret[DDSIZE])
{
    ret[0] = _mm512_setzero_pd();
    ret[1] = _mm512_setzero_pd();
}
#define _bncavx512_rdd_set0(ret) _bncavx512_set0_dd((ret))
// ret := ret8[][avx_index]

// ret := val
static inline void _bncavx512_rdd_set(__m512d ret[DDSIZE], __m512d val[DDSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
}

// ret := (double)val
static inline void _bncavx512_rdd_set_d(__m512d ret[DDSIZE], __m512d val)
{
    ret[0] = val;
    ret[1] = _mm512_setzero_pd(); // val[1];
}

// ret := [val val val val]
static inline void _bncavx512_rdd_set1_dd(__m512d ret[DDSIZE], double val[DDSIZE])
{
    ret[0] = _mm512_set1_pd(val[0]);
    ret[1] = _mm512_set1_pd(val[1]);
}

static inline void _bncavx512_get_dd_m512d_i(ddfloat *ret, __m512d ret8[DDSIZE], int avx_index)
{
    ret->val[0] = ret8[0][avx_index];
    ret->val[1] = ret8[1][avx_index];

    return;
}

// ret := ret4[0] + ret4[1] + ret4[2]
static void _bncavx512_rdd_sum512d(double ret[DDSIZE], __m512d ret8[DDSIZE])
{
    ddfloat ret8_i[8];

    // ret8_i := ret8
    _bncavx512_get_dd_m512d_i(&ret8_i[0], ret8, 0);
    _bncavx512_get_dd_m512d_i(&ret8_i[1], ret8, 1);
    _bncavx512_get_dd_m512d_i(&ret8_i[2], ret8, 2);
    _bncavx512_get_dd_m512d_i(&ret8_i[3], ret8, 3);
    _bncavx512_get_dd_m512d_i(&ret8_i[4], ret8, 4);
    _bncavx512_get_dd_m512d_i(&ret8_i[5], ret8, 5);
    _bncavx512_get_dd_m512d_i(&ret8_i[6], ret8, 6);
    _bncavx512_get_dd_m512d_i(&ret8_i[7], ret8, 7);

    rdd_set(ret, ret8_i[0].val);
    rdd_add(ret, ret, ret8_i[1].val);
    rdd_add(ret, ret, ret8_i[2].val);
    rdd_add(ret, ret, ret8_i[3].val);
    rdd_add(ret, ret, ret8_i[4].val);
    rdd_add(ret, ret, ret8_i[5].val);
    rdd_add(ret, ret, ret8_i[6].val);
    rdd_add(ret, ret, ret8_i[7].val);
}

// ret := |ret4[0]| + |ret4[1]| + |ret4[2]| + |ret4[3]|
static void _bncavx512_rdd_abssum512d(double ret[DDSIZE], __m512d ret8[DDSIZE])
{
    ddfloat ret8_i[8];
    double tmp[DDSIZE];

    // ret8_i := ret8
    _bncavx512_get_dd_m512d_i(&ret8_i[0], ret8, 0);
    _bncavx512_get_dd_m512d_i(&ret8_i[1], ret8, 1);
    _bncavx512_get_dd_m512d_i(&ret8_i[2], ret8, 2);
    _bncavx512_get_dd_m512d_i(&ret8_i[3], ret8, 3);
    _bncavx512_get_dd_m512d_i(&ret8_i[4], ret8, 4);
    _bncavx512_get_dd_m512d_i(&ret8_i[5], ret8, 5);
    _bncavx512_get_dd_m512d_i(&ret8_i[6], ret8, 6);
    _bncavx512_get_dd_m512d_i(&ret8_i[7], ret8, 7);

    rdd_abs(tmp, ret8_i[0].val);
    rdd_set(ret, tmp);

    rdd_abs(tmp, ret8_i[1].val);
    rdd_add(ret, ret, tmp);

    rdd_abs(tmp, ret8_i[2].val);
    rdd_add(ret, ret, tmp);

    rdd_abs(tmp, ret8_i[3].val);
    rdd_add(ret, ret, tmp);

    rdd_abs(tmp, ret8_i[4].val);
    rdd_add(ret, ret, tmp);

    rdd_abs(tmp, ret8_i[5].val);
    rdd_add(ret, ret, tmp);

    rdd_abs(tmp, ret8_i[6].val);
    rdd_add(ret, ret, tmp);

    rdd_abs(tmp, ret8_i[7].val);
    rdd_add(ret, ret, tmp);
}

// ret := max(|ret8[0]|, |ret8[1]|, ..., |ret8[7]|)
static void _bncavx512_rdd_absmax512d(double ret[DDSIZE], __m512d ret8[DDSIZE])
{
    ddfloat ret8_i[8];
    double tmp[DDSIZE];

    // ret8_i := ret8
    _bncavx512_get_dd_m512d_i(&ret8_i[0], ret8, 0);
    _bncavx512_get_dd_m512d_i(&ret8_i[1], ret8, 1);
    _bncavx512_get_dd_m512d_i(&ret8_i[2], ret8, 2);
    _bncavx512_get_dd_m512d_i(&ret8_i[3], ret8, 3);
    _bncavx512_get_dd_m512d_i(&ret8_i[4], ret8, 4);
    _bncavx512_get_dd_m512d_i(&ret8_i[5], ret8, 5);
    _bncavx512_get_dd_m512d_i(&ret8_i[6], ret8, 6);
    _bncavx512_get_dd_m512d_i(&ret8_i[7], ret8, 7);

    rdd_abs(tmp, ret8_i[0].val);
    rdd_set(ret, tmp); // ret := |ret4_i[0]|

    rdd_abs(tmp, ret8_i[1].val);
    if(rdd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[1]|)
        rdd_set(ret, tmp);    //   ret := |ret8_i[1]|

    rdd_abs(tmp, ret8_i[2].val);
    if(rdd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[2]|)
        rdd_set(ret, tmp);    //   ret := |ret8_i[2]|

    rdd_abs(tmp, ret8_i[3].val);
    if(rdd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[3]|)
        rdd_set(ret, tmp);    //   ret := |ret8_i[3]|

    rdd_abs(tmp, ret8_i[4].val);
    if(rdd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[4]|)
        rdd_set(ret, tmp);    //   ret := |ret8_i[4]|

    rdd_abs(tmp, ret8_i[5].val);
    if(rdd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[5]|)
        rdd_set(ret, tmp);    //   ret := |ret8_i[5]|

    rdd_abs(tmp, ret8_i[6].val);
    if(rdd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[6]|)
        rdd_set(ret, tmp);    //   ret := |ret8_i[6]|

    rdd_abs(tmp, ret8_i[7].val);
    if(rdd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[7]|)
        rdd_set(ret, tmp);    //   ret := |ret8_i[7]|

}

// ret := || ret4[0]^2 + ret4[1]^2 + ret4[2]^2 + ret4[3]^2 ||_2
static void _bncavx512_rdd_norm512d(double ret[DDSIZE], __m512d ret8[DDSIZE])
{
    ddfloat ret8_i[8];
    double tmp[DDSIZE];

    // ret8_i := ret8
    _bncavx512_get_dd_m512d_i(&ret8_i[0], ret8, 0);
    _bncavx512_get_dd_m512d_i(&ret8_i[1], ret8, 1);
    _bncavx512_get_dd_m512d_i(&ret8_i[2], ret8, 2);
    _bncavx512_get_dd_m512d_i(&ret8_i[3], ret8, 3);
    _bncavx512_get_dd_m512d_i(&ret8_i[4], ret8, 4);
    _bncavx512_get_dd_m512d_i(&ret8_i[5], ret8, 5);
    _bncavx512_get_dd_m512d_i(&ret8_i[6], ret8, 6);
    _bncavx512_get_dd_m512d_i(&ret8_i[7], ret8, 7);

    rdd_mul(tmp, ret8_i[0].val, ret8_i[0].val);
    rdd_set(ret, tmp);

    rdd_mul(tmp, ret8_i[1].val, ret8_i[1].val);
    rdd_add(ret, ret, tmp);

    rdd_mul(tmp, ret8_i[2].val, ret8_i[2].val);
    rdd_add(ret, ret, tmp);

    rdd_mul(tmp, ret8_i[3].val, ret8_i[3].val);
    rdd_add(ret, ret, tmp);

    rdd_mul(tmp, ret8_i[4].val, ret8_i[4].val);
    rdd_add(ret, ret, tmp);

    rdd_mul(tmp, ret8_i[5].val, ret8_i[5].val);
    rdd_add(ret, ret, tmp);

    rdd_mul(tmp, ret8_i[6].val, ret8_i[6].val);
    rdd_add(ret, ret, tmp);

    rdd_mul(tmp, ret8_i[7].val, ret8_i[7].val);
    rdd_add(ret, ret, tmp);

    rdd_sqrt(tmp, ret);
    rdd_set(ret, tmp);
}

#ifndef USE_DD_BF

// ret := a + b
static inline void _bncavx512_rdd_add(__m512d ret[DDSIZE], __m512d a[DDSIZE], __m512d b[DDSIZE])
{
    // c_dd_add_sloppy
    //double s, e;
    __m512d s, e;

    //s = two_sum(a[0], b[0], &e);
    s = _bncavx512_dtwo_sum(a[0], b[0], &e);
	//e += (a[1] + b[1]);
    e = _mm512_add_pd(e, _mm512_add_pd(a[1], b[1]));

	//c[0] = quick_two_sum(s, e, &e);
	//c[1] = e;
    ret[0] = _bncavx512_dquick_two_sum(s, e, &e);
    ret[1] = e;
//#endif // 0
}

#else // USE_DD_BF
// 2025-12-24(Wed) T.Kouya
// Branch free algorithm 
//void Add2(const double x[2], const double y[2], double z[2]) {
static inline void _bncavx512_rdd_add(__m512d ret[DDSIZE], __m512d a[DDSIZE], __m512d b[DDSIZE])
{
    __m512d g1, g2, g3, g4, g5, g6;
	__m512d g1e, g2e, g3e, g6e;

	g1 = _bncavx512_dtwo_sum(a[0], b[0], &g1e); // ゲート1 [cite: 220, 241]
    g2 = _bncavx512_dtwo_sum(a[1], b[1], &g2e); // ゲート2 [cite: 220, 241]
	g3 = _bncavx512_dquick_two_sum(g1, g2, &g3e);
	g4 = _mm512_add_pd(g1e, g2e);
	g5 = _mm512_add_pd(g4, g3e);
	g6 = _bncavx512_dquick_two_sum(g3, g5, &g6e);
	ret[0] = g6;
	ret[1] = g6e;
}
#endif // USE_DD_BF

// c[3] := -a[3]
static inline void _bncavx512_rdd_neg(__m512d c[DDSIZE], __m512d a[DDSIZE])
{
    __m512d zero8;

    zero8 = _mm512_setzero_pd();

	//c[0] = -a[0];
    c[0] = _mm512_sub_pd(zero8, a[0]);
	//c[1] = -a[1];
    c[1] = _mm512_sub_pd(zero8, a[1]);

}

// ret := a - b
static inline void _bncavx512_rdd_sub(__m512d ret[DDSIZE], __m512d a[DDSIZE], __m512d b[DDSIZE])
{
    // c_dd_sub_sloppy
	//double s1, s2, t1, t2;
    __m512d s1, s2, t1, t2;

//	s1 = two_diff(a[0], b[0], &s2);
//	t1 = two_diff(a[1], b[1], &t2);
    s1 = _bncavx512_dtwo_diff(a[0], b[0], &s2);
    t1 = _bncavx512_dtwo_diff(a[1], b[1], &t2);

//	s2 += t1;
    s2 = _mm512_add_pd(s2, t1);
//	s1 = quick_two_sum(s1, s2, &s2);
    s1 = _bncavx512_dquick_two_sum(s1, s2, &s2);
//	s2 += t2;
    s2 = _mm512_add_pd(s2, t2);
//	s1 = quick_two_sum(s1, s2, &s2);
    s1 = _bncavx512_dquick_two_sum(s1, s2, &s2);
	//return dd_real(s1, s2);
//	c[0] = s1;
//	c[1] = s2;
    ret[0] = s1;
    ret[1] = s2;
}

#ifndef USE_DD_BF

// mul
static inline void _bncavx512_rdd_mul(__m512d ret[DDSIZE], __m512d a[DDSIZE], __m512d b[DDSIZE])
{
//#if 0
	//double p1, p2;
    __m512d p1, p2;

//	p1 = two_prod(a[0], b[0], &p2);
    p1 = _bncavx512_dtwo_prod(a[0], b[0], &p2);

//	p2 += (a[0] * b[1] + a[1] * b[0]);
    p2 = _mm512_add_pd(p2, 
        _mm512_add_pd(
            _mm512_mul_pd(a[0], b[1]),
            _mm512_mul_pd(a[1], b[0])
        )
    );

//	p1 = quick_two_sum(p1, p2, &p2);
//	c[0] = p1;
//	c[1] = p2;

    ret[0] = _bncavx512_dquick_two_sum(p1, p2, &p2);
    ret[1] = p2;
//#endif // 0
}

#else // USE_DD_BF
// 2025-12-24(Wed) T.Kouya
// Branch free algorithm
// void Mul2(const double x[2], const double y[2], double z[2]) {
static inline void _bncavx512_rdd_mul(__m512d ret[DDSIZE], __m512d a[DDSIZE], __m512d b[DDSIZE])
{
	__m512d p00, p01, p10;
	__m512d pe00;
	__m512d g1, g2, g3, ge3;

    p00 = _bncavx512_dtwo_prod(a[0], b[0], &pe00); // [cite: 250, 307]
    p01 = _mm512_mul_pd(a[0], b[1]); // [cite: 250, 307]
    p10 = _mm512_mul_pd(a[1], b[0]); // [cite: 250, 307]
	g1 = _mm512_add_pd(p01, p10);
	g2 = _mm512_add_pd(pe00, g1);
	g3 = _bncavx512_dquick_two_sum(p00, g2, &ge3);
	
    ret[0] = g3;                       // [cite: 307]
    ret[1] = ge3; // 全誤差の統合 [cite: 307]
}
#endif // USE_DD_BF

// dd * d
static inline void _bncavx512_rdd_mul_d(__m512d ret[DDSIZE], __m512d a[DDSIZE], __m512d b)
{
	//double p1, p2;
    __m512d p1, p2;

//	p1 = two_prod(a[0], b, &p2);
    p1 = _bncavx512_dtwo_prod(a[0], b, &p2);

//	p2 += a[1] * b;
    p2 = _mm512_add_pd(p2, 
        _mm512_mul_pd(a[1], b)
    );

//	p1 = quick_two_sum(p1, p2, &p2);
//	c[0] = p1;
//	c[1] = p2;

    ret[0] = _bncavx512_dquick_two_sum(p1, p2, &p2);
    ret[1] = p2;
//#endif // 0
}

// sloppy_div
static inline void _bncavx512_rdd_div(__m512d ret[DDSIZE], __m512d a[DDSIZE], __m512d b[DDSIZE])
{
    // c_dd_sloppy_div(double *a, double *b, double *c)

	//double s1, s2;
	//double q1, q2;
	//double r[DDSIZE];
	__m512d s1, s2;
	__m512d q1, q2;
	__m512d r[DDSIZE];

	//q1 = a[0] / b[0];  /* approximate quotient */
    q1 = _mm512_div_pd(a[0], b[0]);

	/* compute  this - q1 * dd */
	//r = b * q1;
	//c_dd_mul_dd_d(b, q1, r);
    _bncavx512_rdd_mul_d(r, b, q1);

	//s1 = two_diff(a[0], r[0], &s2);
    s1 = _bncavx512_dtwo_diff(a[0], r[0], &s2);
	//s2 -= r[1];
    s2 = _mm512_sub_pd(s2, r[1]);
	//s2 += a[1];
    s2 = _mm512_add_pd(s2, a[1]);

	/* get next approximation */
	//q2 = (s1 + s2) / b[0];
    q2 = _mm512_div_pd(
        _mm512_add_pd(s1, s2),
        b[0]
    );

	/* renormalize */
	//r[0] = quick_two_sum(q1, q2, &r[1]);
    ret[0] = _bncavx512_dquick_two_sum(q1, q2, &ret[1]);
}

// abs
static inline void _bncavx512_rdd_abs(__m512d ret[DDSIZE], __m512d a[DDSIZE])
{
    int avx_index;

    for(avx_index = 0; avx_index < 8; avx_index++)
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
 *  Processes 8 DD operations per call
 * ================================================================ */

/* ----------------------------------------------------------------
 * _bncavx512_rdd_fma_d_d
 *   ret[DDSIZE] := a * b + c   (scalar x scalar -> DD) x 8
 *   a, b, c : scalar, passed by value
 * ---------------------------------------------------------------- */
static inline void _bncavx512_rdd_fma_d_d(
        __m512d ret[DDSIZE],
        __m512d a,
        __m512d b,
        __m512d c)
{
    ret[0] = _mm512_fmadd_pd(a, b, c);
    ret[1] = _mm512_fmadd_pd(a, b, _mm512_sub_pd(c, ret[0]));
}

/* ----------------------------------------------------------------
 * _bncavx512_rdd_fma_d_dd
 *   ret[DDSIZE] := a * b + c[DDSIZE]  (scalar x scalar + DD -> DD) x 8
 *   a, b : scalar, passed by value
 * ---------------------------------------------------------------- */
static inline void _bncavx512_rdd_fma_d_dd(
        __m512d       ret[DDSIZE],
        __m512d       a,
        __m512d       b,
        const __m512d c[DDSIZE])
{
    __m512d hi = _mm512_fmadd_pd(a, b, c[0]);
    __m512d t  = _mm512_sub_pd(c[0], hi);
    __m512d e  = _mm512_fmadd_pd(a, b, t);
    __m512d f  = _mm512_add_pd(e, c[1]);
    ret[0] = _bncavx512_dquick_two_sum(hi, f, &(ret[1]));
}

/* ----------------------------------------------------------------
 * _bncavx512_rdd_fma_dm_dd
 *   ret[DDSIZE] := a * b[DDSIZE] + c[DDSIZE]  (scalar x DD + DD -> DD) x 8
 *   a : scalar, passed by value
 * ---------------------------------------------------------------- */
static inline void _bncavx512_rdd_fma_dm_dd(
        __m512d       ret[DDSIZE],
        __m512d       a,
        const __m512d b[DDSIZE],
        const __m512d c[DDSIZE])
{
    __m512d hi = _mm512_fmadd_pd(a, b[0], c[0]);
    __m512d t  = _mm512_sub_pd(c[0], hi);
    __m512d e  = _mm512_fmadd_pd(a, b[0], t);
    __m512d f  = _mm512_add_pd(e, c[1]);
    __m512d lo = _mm512_fmadd_pd(a, b[1], f);
    ret[0] = _bncavx512_dquick_two_sum(hi, lo, &(ret[1]));
}

/* ----------------------------------------------------------------
 * _bncavx512_rdd_fma
 *   ret[DDSIZE] := a[DDSIZE] * b[DDSIZE] + c[DDSIZE]  (DD x DD + DD -> DD) x 8
 * ---------------------------------------------------------------- */
static inline void _bncavx512_rdd_fma(
        __m512d       ret[DDSIZE],
        const __m512d a[DDSIZE],
        const __m512d b[DDSIZE],
        const __m512d c[DDSIZE])
{
    __m512d d0 = _mm512_fmadd_pd(a[0], b[0], c[0]);
    __m512d t  = _mm512_sub_pd(c[0], d0);
    __m512d e  = _mm512_fmadd_pd(a[0], b[0], t);
    __m512d f  = _mm512_add_pd(e, c[1]);
    __m512d g  = _mm512_fmadd_pd(a[0], b[1], f);
    __m512d d1 = _mm512_fmadd_pd(a[1], b[0], g);
    ret[0] = _bncavx512_dquick_two_sum(d0, d1, &(ret[1]));
}

#endif // defined(__AVX512F__)

#ifdef __BNC_DDLINEAR_H__
//#include "ddv_addmul.c"
#endif //__BNC_DDLINEAR_H__

#endif // ifndef __BNCAVX_DD_AVX512_H
