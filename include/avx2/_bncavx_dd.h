// ------------------------
// ---------- DD ----------
// ------------------------
#ifndef __BNCAVX_DD_H
#define __BNCAVX_DD_H

// defined in rdd.h
#ifndef DDSIZE
    #define DDSIZE 2
#endif // DDSIZE

// ret := 0
#if defined(__AVX2__)
static inline void _bncavx2_set0_dd(__m256d ret[DDSIZE])
{
    ret[0] = _mm256_setzero_pd();
    ret[1] = _mm256_setzero_pd();
}
#define _bncavx2_rdd_set0(ret) _bncavx2_set0_dd((ret))

// ret := val
static inline void _bncavx2_rdd_set(__m256d ret[DDSIZE], __m256d val[DDSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
}

// ret := (double)val
static inline void _bncavx2_rdd_set_d(__m256d ret[DDSIZE], __m256d val)
{
    ret[0] = val;
    ret[1] = _mm256_setzero_pd(); // val[1];
}

// ret := [val val val val]
static inline void _bncavx2_rdd_set1_dd(__m256d ret[DDSIZE], double val[DDSIZE])
{
    ret[0] = _mm256_set1_pd(val[0]);
    ret[1] = _mm256_set1_pd(val[1]);
}

// ret := -val
static inline void _bncavx2_rdd_neg(__m256d ret[DDSIZE], __m256d val[DDSIZE])
{
    ret[0] = _mm256_setzero_pd(); ret[0] = _mm256_sub_pd(ret[0], val[0]);
    ret[1] = _mm256_setzero_pd(); ret[1] = _mm256_sub_pd(ret[1], val[1]);
}

// ret := ret4[][avx_index]
static inline void _bncavx2_get_dd_m256d_i(ddfloat *ret, __m256d ret4[DDSIZE], int avx_index)
{
    ret->val[0] = ret4[0][avx_index];
    ret->val[1] = ret4[1][avx_index];

    return;
}

// ret := ret4[0] + ret4[1] + ret4[2] + ret4[3]
static void _bncavx2_rdd_sum256d(double ret[DDSIZE], __m256d ret4[DDSIZE])
{
    ddfloat ret4_i[4];

    // ret4_i := ret4
    _bncavx2_get_dd_m256d_i(&ret4_i[0], ret4, 0);
    _bncavx2_get_dd_m256d_i(&ret4_i[1], ret4, 1);
    _bncavx2_get_dd_m256d_i(&ret4_i[2], ret4, 2);
    _bncavx2_get_dd_m256d_i(&ret4_i[3], ret4, 3);

    rdd_set(ret, ret4_i[0].val);
    rdd_add(ret, ret, ret4_i[1].val);
    rdd_add(ret, ret, ret4_i[2].val);
    rdd_add(ret, ret, ret4_i[3].val);
}

// ret := |ret4[0]| + |ret4[1]| + |ret4[2]| + |ret4[3]|
static void _bncavx2_rdd_abssum256d(double ret[DDSIZE], __m256d ret4[DDSIZE])
{
    ddfloat ret4_i[4];
    double tmp[DDSIZE];

    // ret4_i := ret4
    _bncavx2_get_dd_m256d_i(&ret4_i[0], ret4, 0);
    _bncavx2_get_dd_m256d_i(&ret4_i[1], ret4, 1);
    _bncavx2_get_dd_m256d_i(&ret4_i[2], ret4, 2);
    _bncavx2_get_dd_m256d_i(&ret4_i[3], ret4, 3);

    rdd_abs(tmp, ret4_i[0].val);
    rdd_set(ret, tmp);

    rdd_abs(tmp, ret4_i[1].val);
    rdd_add(ret, ret, tmp);

    rdd_abs(tmp, ret4_i[2].val);
    rdd_add(ret, ret, tmp);

    rdd_abs(tmp, ret4_i[3].val);
    rdd_add(ret, ret, tmp);
}

// ret := max(|ret4[0]|, |ret4[1]|, |ret4[2]|, |ret4[3]|)
static void _bncavx2_rdd_absmax256d(double ret[DDSIZE], __m256d ret4[DDSIZE])
{
    ddfloat ret4_i[4];
    double tmp[DDSIZE];

    // ret4_i := ret4
    _bncavx2_get_dd_m256d_i(&ret4_i[0], ret4, 0);
    _bncavx2_get_dd_m256d_i(&ret4_i[1], ret4, 1);
    _bncavx2_get_dd_m256d_i(&ret4_i[2], ret4, 2);
    _bncavx2_get_dd_m256d_i(&ret4_i[3], ret4, 3);

    rdd_abs(tmp, ret4_i[0].val);
    rdd_set(ret, tmp); // ret := |ret4_i[0]|

    rdd_abs(tmp, ret4_i[1].val);
    if(rdd_cmp(ret, tmp) < 0) // if(ret < |ret4_i[1]|)
        rdd_set(ret, tmp);    //   ret := |ret4_i[1]|

    rdd_abs(tmp, ret4_i[2].val);
    if(rdd_cmp(ret, tmp) < 0) // if(ret < |ret4_i[2]|)
        rdd_set(ret, tmp);    //   ret := |ret4_i[2]|

    rdd_abs(tmp, ret4_i[3].val);
    if(rdd_cmp(ret, tmp) < 0) // if(ret < |ret4_i[3]|)
        rdd_set(ret, tmp);    //   ret := |ret4_i[3]|
}

// ret := || ret4[0]^2 + ret4[1]^2 + ret4[2]^2 + ret4[3]^2 ||_2
static void _bncavx2_rdd_norm256d(double ret[DDSIZE], __m256d ret4[DDSIZE])
{
    ddfloat ret4_i[4];
    double tmp[DDSIZE];

    // ret4_i := ret4
    _bncavx2_get_dd_m256d_i(&ret4_i[0], ret4, 0);
    _bncavx2_get_dd_m256d_i(&ret4_i[1], ret4, 1);
    _bncavx2_get_dd_m256d_i(&ret4_i[2], ret4, 2);
    _bncavx2_get_dd_m256d_i(&ret4_i[3], ret4, 3);

    rdd_mul(tmp, ret4_i[0].val, ret4_i[0].val);
    rdd_set(ret, tmp);

    rdd_mul(tmp, ret4_i[1].val, ret4_i[1].val);
    rdd_add(ret, ret, tmp);

    rdd_mul(tmp, ret4_i[2].val, ret4_i[2].val);
    rdd_add(ret, ret, tmp);

    rdd_mul(tmp, ret4_i[3].val, ret4_i[3].val);
    rdd_add(ret, ret, tmp);

    rdd_sqrt(tmp, ret);
    rdd_set(ret, tmp);
}
#endif // __AVX2__

#if defined(__AVX512F__)
// ret := ret8[][avx_index]
static inline void _bncavx2_get_dd_m512d_i(ddfloat *ret, __m512d ret8[DDSIZE], int avx_index)
{
    ret->val[0] = ret8[0][avx_index];
    ret->val[1] = ret8[1][avx_index];

    return;
}

// ret := ret4[0] + ret4[1] + ret4[2]
static void _bncavx2_rdd_sum512d(double ret[DDSIZE], __m512d ret8[DDSIZE])
{
    ddfloat ret8_i[8];

    // ret8_i := ret8
    _bncavx2_get_dd_m512d_i(&ret8_i[0], ret8, 0);
    _bncavx2_get_dd_m512d_i(&ret8_i[1], ret8, 1);
    _bncavx2_get_dd_m512d_i(&ret8_i[2], ret8, 2);
    _bncavx2_get_dd_m512d_i(&ret8_i[3], ret8, 3);
    _bncavx2_get_dd_m512d_i(&ret8_i[4], ret8, 4);
    _bncavx2_get_dd_m512d_i(&ret8_i[5], ret8, 5);
    _bncavx2_get_dd_m512d_i(&ret8_i[6], ret8, 6);
    _bncavx2_get_dd_m512d_i(&ret8_i[7], ret8, 7);

    rdd_set(ret, ret8_i[0].val);
    rdd_add(ret, ret, ret8_i[1].val);
    rdd_add(ret, ret, ret8_i[2].val);
    rdd_add(ret, ret, ret8_i[3].val);
    rdd_add(ret, ret, ret8_i[4].val);
    rdd_add(ret, ret, ret8_i[5].val);
    rdd_add(ret, ret, ret8_i[6].val);
    rdd_add(ret, ret, ret8_i[7].val);
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

static inline void _bncavx2_rdd_add_sloppy(__m256d ret[DDSIZE], __m256d a[DDSIZE], __m256d b[DDSIZE])
{
#if 0
    double in_ret[4][DDSIZE], in_a[4][DDSIZE], in_b[4][DDSIZE];

    in_ret[0][0] = ret[0][0]; in_ret[0][1] = ret[1][0];
    in_ret[1][0] = ret[0][1]; in_ret[1][1] = ret[1][1];
    in_ret[2][0] = ret[0][2]; in_ret[2][1] = ret[1][2];
    in_ret[3][0] = ret[0][3]; in_ret[3][1] = ret[1][3];

    in_a[0][0] = a[0][0]; in_a[0][1] = a[1][0];
    in_a[1][0] = a[0][1]; in_a[1][1] = a[1][1];
    in_a[2][0] = a[0][2]; in_a[2][1] = a[1][2];
    in_a[3][0] = a[0][3]; in_a[3][1] = a[1][3];

    in_b[0][0] = b[0][0]; in_b[0][1] = b[1][0];
    in_b[1][0] = b[0][1]; in_b[1][1] = b[1][1];
    in_b[2][0] = b[0][2]; in_b[2][1] = b[1][2];
    in_b[3][0] = b[0][3]; in_b[3][1] = b[1][3];

    rdd_add(in_ret[0], in_a[0], in_b[0]);
    rdd_add(in_ret[1], in_a[1], in_b[1]);
    rdd_add(in_ret[2], in_a[2], in_b[2]);
    rdd_add(in_ret[3], in_a[3], in_b[3]);

    ret[0][0] = in_ret[0][0]; ret[1][0] = in_ret[0][1];
    ret[0][1] = in_ret[1][0]; ret[1][1] = in_ret[1][1];
    ret[0][2] = in_ret[2][0]; ret[1][2] = in_ret[2][1];
    ret[0][3] = in_ret[3][0]; ret[1][3] = in_ret[3][1];
#endif // 0
//#if 0
    // c_dd_add_sloppy
    //double s, e;
    __m256d s, e;

    //s = two_sum(a[0], b[0], &e);
    s = _bncavx2_dtwo_sum(a[0], b[0], &e);
	//e += (a[1] + b[1]);
    e = _mm256_add_pd(e, _mm256_add_pd(a[1], b[1]));

	//c[0] = quick_two_sum(s, e, &e);
	//c[1] = e;
    ret[0] = _bncavx2_dquick_two_sum(s, e, &e);
    ret[1] = e;
//#endif // 0
}

// 2025-12-24(Wed) T.Kouya
// Branch free algorithm by D.K.Zhang and A.Aiken at SC2025 by D.K.Zhang and A.Aiken at SC2025
//void Add2(const double x[2], const double y[2], double z[2]) {
static inline void _bncavx2_rdd_add_bf(__m256d ret[DDSIZE], __m256d a[DDSIZE], __m256d b[DDSIZE])
{
    __m256d g1, g2, g3, g4, g5, g6;
	__m256d g1e, g2e, g3e, g6e;

	g1 = _bncavx2_dtwo_sum(a[0], b[0], &g1e); // ゲート1 [cite: 220, 241]
    g2 = _bncavx2_dtwo_sum(a[1], b[1], &g2e); // ゲート2 [cite: 220, 241]
	g3 = _bncavx2_dquick_two_sum(g1, g2, &g3e);
	g4 = _mm256_add_pd(g1e, g2e);
	g5 = _mm256_add_pd(g4, g3e);
	g6 = _bncavx2_dquick_two_sum(g3, g5, &g6e);
	ret[0] = g6;
	ret[1] = g6e;
}

// 2024-09-04(Wed)
// (dd_real)c := (dd_real)a + b
//static inline void c_dd_add_dd_d(const double *a, double b, double *c)
static inline void _bncavx2_rdd_add_d(__m256d ret[DDSIZE], __m256d a[DDSIZE], __m256d b)
{
//#ifdef NATIVE_C
	//double cc[DDSIZE];
    //double s1, s2;
	__m256d s1, s2;

	//c[0] = a[0] + b;
	//c[1] = a[1];
	s1 = _bncavx2_dtwo_sum(a[0], b, &s2);
	//t1 = two_sum(a[1], b[1], &t2);
	//s2 += t1;
	//s2 += a[1];
    s2 = _mm256_add_pd(s2, a[1]);

	s1 = _bncavx2_dquick_two_sum(s1, s2, &s2);
	//s2 += t2;
	//s1 = quick_two_sum(s1, s2, &s2);
//  return dd_real(s1, s2);
	ret[0] = s1;
	ret[1] = s2;
}
	
// ret := a - b
static inline void _bncavx2_rdd_sub(__m256d ret[DDSIZE], __m256d a[DDSIZE], __m256d b[DDSIZE])
{
    // c_dd_sub_sloppy
	//double s1, s2, t1, t2;
    __m256d s1, s2, t1, t2;

//	s1 = two_diff(a[0], b[0], &s2);
//	t1 = two_diff(a[1], b[1], &t2);
    s1 = _bncavx2_dtwo_diff(a[0], b[0], &s2);
    t1 = _bncavx2_dtwo_diff(a[1], b[1], &t2);

//	s2 += t1;
    s2 = _mm256_add_pd(s2, t1);
//	s1 = quick_two_sum(s1, s2, &s2);
    s1 = _bncavx2_dquick_two_sum(s1, s2, &s2);
//	s2 += t2;
    s2 = _mm256_add_pd(s2, t2);
//	s1 = quick_two_sum(s1, s2, &s2);
    s1 = _bncavx2_dquick_two_sum(s1, s2, &s2);
	//return dd_real(s1, s2);
//	c[0] = s1;
//	c[1] = s2;
    ret[0] = s1;
    ret[1] = s2;
}

// 2024-09-04(Wed)
/* double-double - double */
//static inline void c_dd_sub_dd_d(const double *a, double b, double *c)
static inline void _bncavx2_rdd_sub_d(__m256d ret[DDSIZE], __m256d a[DDSIZE], __m256d b)
{
	//double s1, s2;
    __m256d s1, s2;
	s1 = _bncavx2_dtwo_diff(a[0], b, &s2);
	//s2 += a[1];
    s2 = _mm256_add_pd(s2, a[1]);
	s1 = _bncavx2_dquick_two_sum(s1, s2, &s2);
	//return dd_real(s1, s2);
	ret[0] = s1;
	ret[1] = s2;
}

// mul
static inline void _bncavx2_rdd_mul_sloppy(__m256d ret[DDSIZE], __m256d a[DDSIZE], __m256d b[DDSIZE])
{
#if 0
    double in_ret[4][DDSIZE], in_a[4][DDSIZE], in_b[4][DDSIZE];

    in_ret[0][0] = ret[0][0]; in_ret[0][1] = ret[1][0];
    in_ret[1][0] = ret[0][1]; in_ret[1][1] = ret[1][1];
    in_ret[2][0] = ret[0][2]; in_ret[2][1] = ret[1][2];
    in_ret[3][0] = ret[0][3]; in_ret[3][1] = ret[1][3];

    in_a[0][0] = a[0][0]; in_a[0][1] = a[1][0];
    in_a[1][0] = a[0][1]; in_a[1][1] = a[1][1];
    in_a[2][0] = a[0][2]; in_a[2][1] = a[1][2];
    in_a[3][0] = a[0][3]; in_a[3][1] = a[1][3];

    in_b[0][0] = b[0][0]; in_b[0][1] = b[1][0];
    in_b[1][0] = b[0][1]; in_b[1][1] = b[1][1];
    in_b[2][0] = b[0][2]; in_b[2][1] = b[1][2];
    in_b[3][0] = b[0][3]; in_b[3][1] = b[1][3];

    rdd_mul(in_ret[0], in_a[0], in_b[0]);
    rdd_mul(in_ret[1], in_a[1], in_b[1]);
    rdd_mul(in_ret[2], in_a[2], in_b[2]);
    rdd_mul(in_ret[3], in_a[3], in_b[3]);

    ret[0][0] = in_ret[0][0]; ret[1][0] = in_ret[0][1];
    ret[0][1] = in_ret[1][0]; ret[1][1] = in_ret[1][1];
    ret[0][2] = in_ret[2][0]; ret[1][2] = in_ret[2][1];
    ret[0][3] = in_ret[3][0]; ret[1][3] = in_ret[3][1];
#endif // 0
//#if 0
	//double p1, p2;
    __m256d p1, p2;

//	p1 = two_prod(a[0], b[0], &p2);
    p1 = _bncavx2_dtwo_prod(a[0], b[0], &p2);

//	p2 += (a[0] * b[1] + a[1] * b[0]);
    p2 = _mm256_add_pd(p2, 
        _mm256_add_pd(
            _mm256_mul_pd(a[0], b[1]),
            _mm256_mul_pd(a[1], b[0])
        )
    );

//	p1 = quick_two_sum(p1, p2, &p2);
//	c[0] = p1;
//	c[1] = p2;

    ret[0] = _bncavx2_dquick_two_sum(p1, p2, &p2);
    ret[1] = p2;
//#endif // 0
}

// 2025-12-24(Wed) T.Kouya
// Branch free algorithm by D.K.Zhang and A.Aiken at SC2025
// void Mul2(const double x[2], const double y[2], double z[2]) {
static inline void _bncavx2_rdd_mul_bf(__m256d ret[DDSIZE], __m256d a[DDSIZE], __m256d b[DDSIZE])
{
	__m256d p00, p01, p10;
	__m256d pe00;
	__m256d g1, g2, g3, ge3;

    p00 = _bncavx2_dtwo_prod(a[0], b[0], &pe00); // [cite: 250, 307]
    p01 = _mm256_mul_pd(a[0], b[1]); // [cite: 250, 307]
    p10 = _mm256_mul_pd(a[1], b[0]); // [cite: 250, 307]
	g1 = _mm256_add_pd(p01, p10);
	g2 = _mm256_add_pd(pe00, g1);
	g3 = _bncavx2_dquick_two_sum(p00, g2, &ge3);
	
    ret[0] = g3;                       // [cite: 307]
    ret[1] = ge3; // 全誤差の統合 [cite: 307]
}

// 2025-12-26(Fri) T.Kouya
#ifdef USE_DD_BF
    #define _bncavx2_rdd_add _bncavx2_rdd_add_bf
    #define _bncavx2_rdd_mul _bncavx2_rdd_mul_bf
#else // USE_DD_BF
    #define _bncavx2_rdd_add _bncavx2_rdd_add_sloppy
    #define _bncavx2_rdd_mul _bncavx2_rdd_mul_sloppy
#endif //  USE_DD_BF

// dd * d
static inline void _bncavx2_rdd_mul_d(__m256d ret[DDSIZE], __m256d a[DDSIZE], __m256d b)
{
	//double p1, p2;
    __m256d p1, p2;

//	p1 = two_prod(a[0], b, &p2);
    p1 = _bncavx2_dtwo_prod(a[0], b, &p2);

//	p2 += a[1] * b;
    p2 = _mm256_add_pd(p2, 
        _mm256_mul_pd(a[1], b)
    );

//	p1 = quick_two_sum(p1, p2, &p2);
//	c[0] = p1;
//	c[1] = p2;

    ret[0] = _bncavx2_dquick_two_sum(p1, p2, &p2);
    ret[1] = p2;
//#endif // 0
}

// sloppy_div
static inline void _bncavx2_rdd_div(__m256d ret[DDSIZE], __m256d a[DDSIZE], __m256d b[DDSIZE])
{
    // c_dd_sloppy_div(double *a, double *b, double *c)

	//double s1, s2;
	//double q1, q2;
	//double r[DDSIZE];
	__m256d s1, s2;
	__m256d q1, q2;
	__m256d r[DDSIZE];

	//q1 = a[0] / b[0];  /* approximate quotient */
    q1 = _mm256_div_pd(a[0], b[0]);

	/* compute  this - q1 * dd */
	//r = b * q1;
	//c_dd_mul_dd_d(b, q1, r);
    _bncavx2_rdd_mul_d(r, b, q1);

	//s1 = two_diff(a[0], r[0], &s2);
    s1 = _bncavx2_dtwo_diff(a[0], r[0], &s2);
	//s2 -= r[1];
    s2 = _mm256_sub_pd(s2, r[1]);
	//s2 += a[1];
    s2 = _mm256_add_pd(s2, a[1]);

	/* get next approximation */
	//q2 = (s1 + s2) / b[0];
    q2 = _mm256_div_pd(
        _mm256_add_pd(s1, s2),
        b[0]
    );

	/* renormalize */
	//r[0] = quick_two_sum(q1, q2, &r[1]);
    ret[0] = _bncavx2_dquick_two_sum(q1, q2, &ret[1]);
}

/* double-double / double */
//static inline void c_dd_div_dd_d(const double *a, double b, double *c)
static inline void _bncavx2_rdd_div_d(__m256d ret[DDSIZE], __m256d a[DDSIZE], __m256d b)
{
	__m256d q1, q2;
	__m256d p1, p2;
	__m256d s, e;
//	dd_real r;
	__m256d r[DDSIZE];

	//q1 = a[0] / b;   /* approximate quotient. */
    q1 = _mm256_div_pd(a[0], b);   /* approximate quotient. */

	/* Compute  this - q1 * d */
	p1 = _bncavx2_dtwo_prod(q1, b, &p2);
	s = _bncavx2_dtwo_diff(a[0], p1, &e);
	//e += a[1];
	//e -= p2;
	e = _mm256_add_pd(e, a[1]);
	e = _mm256_sub_pd(e, p2);
	/* get next approximation. */
	//q2 = (s + e) / b;
    q2 = _mm256_div_pd(_mm256_add_pd(s, e), b);

	/* renormalize */
	r[0] = _bncavx2_dquick_two_sum(q1, q2, &r[1]);

	// return r;
	ret[0] = r[0];
	ret[1] = r[1];
}

// abs
static inline void _bncavx2_rdd_abs(__m256d ret[DDSIZE], __m256d a[DDSIZE])
{
    int avx_index;

    for(avx_index = 0; avx_index < 4; avx_index++)
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

// 2026-02-24(Tue) (ret[0], ret[1]) := (a, b, c)
// a * b + c = d + e
/* ================================================================
 *  AVX2 implementation  (_bncavx2_ prefix)
 *  Processes 4 DD operations per call
 * ================================================================ */

/* ----------------------------------------------------------------
 * _bncavx2_rdd_fma_d_d
 *   ret[DDSIZE] := a * b + c   (scalar x scalar -> DD) x 4
 *   a, b, c : scalar, passed by value
 *
 *   ret[0] = fma(a, b, c)
 *   ret[1] = fma(a, b, c - ret[0])
 * ---------------------------------------------------------------- */
static inline void _bncavx2_rdd_fma_d_d(
        __m256d ret[DDSIZE],
        __m256d a,
        __m256d b,
        __m256d c)
{
    ret[0] = _mm256_fmadd_pd(a, b, c);
    ret[1] = _mm256_fmadd_pd(a, b, _mm256_sub_pd(c, ret[0]));
}

/* ----------------------------------------------------------------
 * _bncavx2_rdd_fma_d_dd
 *   ret[DDSIZE] := a * b + c[DDSIZE]  (scalar x scalar + DD -> DD) x 4
 *   a, b : scalar, passed by value
 *
 *   hi   = fma(a, b, c[0])
 *   t    = c[0] - hi
 *   e    = fma(a, b, t)
 *   f    = e + c[1]
 *   ret[0] = dquick_two_sum(hi, f, &ret[1])
 * ---------------------------------------------------------------- */
static inline void _bncavx2_rdd_fma_d_dd(
        __m256d       ret[DDSIZE],
        __m256d       a,
        __m256d       b,
        const __m256d c[DDSIZE])
{
    __m256d hi = _mm256_fmadd_pd(a, b, c[0]);
    __m256d t  = _mm256_sub_pd(c[0], hi);
    __m256d e  = _mm256_fmadd_pd(a, b, t);
    __m256d f  = _mm256_add_pd(e, c[1]);
    ret[0] = _bncavx2_dquick_two_sum(hi, f, &(ret[1]));
}

/* ----------------------------------------------------------------
 * _bncavx2_rdd_fma_dm_dd
 *   ret[DDSIZE] := a * b[DDSIZE] + c[DDSIZE]  (scalar x DD + DD -> DD) x 4
 *   a : scalar, passed by value
 *
 *   hi   = fma(a, b[0], c[0])
 *   t    = c[0] - hi
 *   e    = fma(a, b[0], t)
 *   f    = e + c[1]
 *   lo   = fma(a, b[1], f)
 *   ret[0] = dquick_two_sum(hi, lo, &ret[1])
 * ---------------------------------------------------------------- */
static inline void _bncavx2_rdd_fma_dm_dd(
        __m256d       ret[DDSIZE],
        __m256d       a,
        const __m256d b[DDSIZE],
        const __m256d c[DDSIZE])
{
    __m256d hi = _mm256_fmadd_pd(a, b[0], c[0]);
    __m256d t  = _mm256_sub_pd(c[0], hi);
    __m256d e  = _mm256_fmadd_pd(a, b[0], t);
    __m256d f  = _mm256_add_pd(e, c[1]);
    __m256d lo = _mm256_fmadd_pd(a, b[1], f);
    ret[0] = _bncavx2_dquick_two_sum(hi, lo, &(ret[1]));
}

/* ----------------------------------------------------------------
 * _bncavx2_rdd_fma
 *   ret[DDSIZE] := a[DDSIZE] * b[DDSIZE] + c[DDSIZE]  (DD x DD + DD -> DD) x 4
 *
 *   d0   = fma(a[0], b[0], c[0])
 *   t    = c[0] - d0
 *   e    = fma(a[0], b[0], t)
 *   f    = e + c[1]
 *   g    = fma(a[0], b[1], f)
 *   d1   = fma(a[1], b[0], g)
 *   ret[0] = dquick_two_sum(d0, d1, &ret[1])
 * ---------------------------------------------------------------- */
static inline void _bncavx2_rdd_fma(
        __m256d       ret[DDSIZE],
        const __m256d a[DDSIZE],
        const __m256d b[DDSIZE],
        const __m256d c[DDSIZE])
{
    __m256d d0 = _mm256_fmadd_pd(a[0], b[0], c[0]);
    __m256d t  = _mm256_sub_pd(c[0], d0);
    __m256d e  = _mm256_fmadd_pd(a[0], b[0], t);
    __m256d f  = _mm256_add_pd(e, c[1]);
    __m256d g  = _mm256_fmadd_pd(a[0], b[1], f);
    __m256d d1 = _mm256_fmadd_pd(a[1], b[0], g);
    ret[0] = _bncavx2_dquick_two_sum(d0, d1, &(ret[1]));
}

#endif // defined(__AVX2__)

#ifdef __BNC_DDLINEAR_H__
//#include "ddv_addmul.c"
#endif //__BNC_DDLINEAR_H__

#endif // ifndef __BNCAVX_DD_H
