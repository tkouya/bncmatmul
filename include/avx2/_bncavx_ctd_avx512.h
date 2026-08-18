// ------------------------
// ------ Complex TD ------
// ------ AVX-512 Version --
// ------------------------
#ifndef __BNCAVX_CTD_AVX512_H
#define __BNCAVX_CTD_AVX512_H

#ifndef TDSIZE
    #define TDSIZE 3
#endif // TDSIZE

#if defined(__AVX512F__)

// ret := 0
static inline void _bncavx512_rctd_set0(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE])
{
    ret_re[0] = _mm512_setzero_pd();  ret_re[1] = _mm512_setzero_pd(); ret_re[2] = _mm512_setzero_pd();
    ret_im[0] = _mm512_setzero_pd();  ret_im[1] = _mm512_setzero_pd(); ret_im[2] = _mm512_setzero_pd();
}
static inline void _bncavx512_set0_ctd(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE])
{
    ret_re[0] = _mm512_setzero_pd();  ret_re[1] = _mm512_setzero_pd(); ret_re[2] = _mm512_setzero_pd();
    ret_im[0] = _mm512_setzero_pd();  ret_im[1] = _mm512_setzero_pd(); ret_im[2] = _mm512_setzero_pd();
}

// ret := a
static inline void _bncavx512_rctd_set(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE], __m512d a_re[TDSIZE], __m512d a_im[TDSIZE])
{
    _bncavx512_rtd_set(ret_re, a_re);
    _bncavx512_rtd_set(ret_im, a_im);
}

// ret := a
static inline void _bncavx512_rctd_set_td(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE], __m512d a[TDSIZE])
{
    _bncavx512_rtd_set(ret_re, a);
    _bncavx512_rtd_set0(ret_im);
}

// ret := a
static inline void _bncavx512_rctd_set_td_td(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE], __m512d a_real[TDSIZE], __m512d a_imag[TDSIZE])
{
    _bncavx512_rtd_set(ret_re, a_real);
    _bncavx512_rtd_set(ret_im, a_imag);
}

// ret := a
static inline void _bncavx512_rctd_set_dd(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE], __m512d a[DDSIZE])
{
    ret_re[0] = a[0];
    ret_re[1] = a[1];
    ret_re[2] = _mm512_setzero_pd();
    _bncavx512_rtd_set0(ret_im);
}

// ret := a
static inline void _bncavx512_rctd_set_d(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE], __m512d a)
{
    ret_re[0] = a;
    ret_re[1] = _mm512_setzero_pd();
    ret_re[2] = _mm512_setzero_pd();
    _bncavx512_rtd_set0(ret_im);
}

// ret := a
static inline void _bncavx512_rctd_set_cd(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE], __m512d a_re, __m512d a_im)
{
    ret_re[0] = a_re; //creal(a);
    ret_re[1] = _mm512_setzero_pd();
    ret_re[2] = _mm512_setzero_pd();
    ret_im[0] = a_im; //cimag(a);
    ret_im[1] = _mm512_setzero_pd();
    ret_im[2] = _mm512_setzero_pd();
}

// ret := a
static inline void _bncavx512_rctd_set_cddfloat(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE], __m512d a_re[DDSIZE], __m512d a_im[DDSIZE])
{
    ret_re[0] = a_re[0]; //creal(a);
    ret_re[1] = a_re[1];
    ret_re[2] = _mm512_setzero_pd();
    ret_im[0] = a_im[0]; //cimag(a);
    ret_im[1] = a_im[1];
    ret_im[2] = _mm512_setzero_pd();
}

// ret := conj(a)
static inline void _bncavx512_rctd_conj(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE], __m512d a_re[TDSIZE], __m512d a_im[TDSIZE])
{
    _bncavx512_rtd_set(ret_re, a_re);
    _bncavx512_rtd_neg(ret_im, a_im);
}

// ret := -a
static inline void _bncavx512_rctd_neg(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE], __m512d a_re[TDSIZE], __m512d a_im[TDSIZE])
{
    _bncavx512_rtd_neg(ret_re, a_re);
    _bncavx512_rtd_neg(ret_im, a_im);
}

// ret := a + b
static inline void _bncavx512_rctd_add(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE], __m512d a_re[TDSIZE], __m512d a_im[TDSIZE], __m512d b_re[TDSIZE], __m512d b_im[TDSIZE])
{
    // ret := a + b
    _bncavx512_rtd_add(ret_re, a_re, b_re);
    _bncavx512_rtd_add(ret_im, a_im, b_im);
}

// ret := a - b
static inline void _bncavx512_rctd_sub(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE], __m512d a_re[TDSIZE], __m512d a_im[TDSIZE], __m512d b_re[TDSIZE], __m512d b_im[TDSIZE])
{
    // ret := a - b
    _bncavx512_rtd_sub(ret_re, a_re, b_re);
    _bncavx512_rtd_sub(ret_im, a_im, b_im);
}

// ret := a * b : 4M
static inline void _bncavx512_rctd_mul_4m(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE], __m512d a_re[TDSIZE], __m512d a_im[TDSIZE], __m512d b_re[TDSIZE], __m512d b_im[TDSIZE])
{
    __m512d tmp[TDSIZE];

    // ret := a * b
    _bncavx512_rtd_mul(ret_re, a_re, b_re);
    _bncavx512_rtd_mul(tmp, a_im, b_im);
    _bncavx512_rtd_sub(ret_re, ret_re, tmp);

    _bncavx512_rtd_mul(ret_im, a_re, b_im);
    _bncavx512_rtd_mul(tmp, a_im, b_re);
    _bncavx512_rtd_add(ret_im, ret_im, tmp);
}

// ret := a * b : 3M
static inline void _bncavx512_rctd_mul_3m(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE], __m512d a_re[TDSIZE], __m512d a_im[TDSIZE], __m512d b_re[TDSIZE], __m512d b_im[TDSIZE])
{
    __m512d t1[TDSIZE], t2[TDSIZE], a_re_im[TDSIZE], b_re_im[TDSIZE];

    // ret := a * b
    _bncavx512_rtd_mul(t1, a_re, b_re);
    _bncavx512_rtd_mul(t2, a_im, b_im);
    _bncavx512_rtd_sub(ret_re, t1, t2);

    _bncavx512_rtd_add(a_re_im, a_re, a_im);
    _bncavx512_rtd_add(b_re_im, b_re, b_im);
    _bncavx512_rtd_mul(ret_im, a_re_im, b_re_im);
    _bncavx512_rtd_sub(ret_im, ret_im, t1);
    _bncavx512_rtd_sub(ret_im, ret_im, t2);
}

// ret := |a|^2
static inline void _bncavx512_rctd_nrm2(__m512d ret[TDSIZE], __m512d a_re[TDSIZE], __m512d a_im[TDSIZE])
{
    __m512d tmp_re2[TDSIZE], tmp_im2[TDSIZE];

    _bncavx512_rtd_mul(tmp_re2, a_re, a_re);
    _bncavx512_rtd_mul(tmp_im2, a_im, a_im);
    _bncavx512_rtd_add(ret, tmp_re2, tmp_im2);
}

// ret := |a|
/*
static inline void _bncavx512_rctd_abs(tdfloat *ret, __m512d a_re[TDSIZE], __m512d a_im[TDSIZE])
{
    tdfloat tmp_re2, tmp_im2;

    _bncavx512_rtd_mul(tmp_re2.val, a_re, a_re);
    _bncavx512_rtd_mul(tmp_im2.val, a_im, a_im);
    _bncavx512_rtd_add(ret->val, tmp_re2.val, tmp_im2.val);
    _bncavx512_rtd_sqrt(ret->val, ret->val);
}
static inline void _bncavx512_rctd_abs_td(double ret[TDSIZE], __m512d a_re[TDSIZE], __m512d a_im[TDSIZE])
{
    double tmp_re2[TDSIZE], tmp_im2[TDSIZE];

    _bncavx512_rtd_mul(tmp_re2, a_re, a_re);
    _bncavx512_rtd_mul(tmp_im2, a_im, a_im);
    _bncavx512_rtd_add(ret, tmp_re2, tmp_im2);
    _bncavx512_rtd_sqrt(ret, ret);
}
*/

// ret := 1 / a
static inline void _bncavx512_rctd_inv(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE], __m512d a_re[TDSIZE], __m512d a_im[TDSIZE])
{
    __m512d a_nrm2[TDSIZE], tmp[TDSIZE];

    //rctd_conj(ret, a);
    _bncavx512_rtd_set(ret_re, a_re);
    _bncavx512_rtd_neg(ret_im, a_im);

    //rctd_nrm2(&a_nrm2, a);
    _bncavx512_rtd_mul(a_nrm2, a_re, a_re);
    _bncavx512_rtd_mul(tmp, a_im, a_im);
    _bncavx512_rtd_add(a_nrm2, a_nrm2, tmp);

    _bncavx512_rtd_div(ret_re, ret_re, a_nrm2);
    _bncavx512_rtd_div(ret_im, ret_im, a_nrm2);
}

// ret := a / b : 3M
static inline void _bncavx512_rctd_div_3m(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE], __m512d a_re[TDSIZE], __m512d a_im[TDSIZE], __m512d b_re[TDSIZE], __m512d b_im[TDSIZE])
{
    __m512d inv_b_re[TDSIZE], inv_b_im[TDSIZE];

    _bncavx512_rctd_inv(inv_b_re, inv_b_im, b_re, b_im);
    _bncavx512_rctd_mul_3m(ret_re, ret_im, a_re, a_im, inv_b_re, inv_b_im);
}

// ret := a / b : 4M
static inline void _bncavx512_rctd_div_4m(__m512d ret_re[TDSIZE], __m512d ret_im[TDSIZE], __m512d a_re[TDSIZE], __m512d a_im[TDSIZE], __m512d b_re[TDSIZE], __m512d b_im[TDSIZE])
{
    __m512d inv_b_re[TDSIZE], inv_b_im[TDSIZE];

    _bncavx512_rctd_inv(inv_b_re, inv_b_im, b_re, b_im);
    _bncavx512_rctd_mul_4m(ret_re, ret_im, a_re, a_im, inv_b_re, inv_b_im);
}

// 2024-02-18 (SUN) Tomonori Kouya
// 2025-01-22: AVX-512 version
// Default: 4M method
#ifdef USE_RCTD_MUL_4M // 4M
    #define _bncavx512_rctd_mul _bncavx512_rctd_mul_4m
    #define _bncavx512_rctd_div _bncavx512_rctd_div_4m
#elif defined(USE_RCTD_MUL_3M) // 3M
    #define _bncavx512_rctd_mul _bncavx512_rctd_mul_3m
    #define _bncavx512_rctd_div _bncavx512_rctd_div_3m
#else // USE_RCTD_MUL_4M
    #define _bncavx512_rctd_mul _bncavx512_rctd_mul_4m
    #define _bncavx512_rctd_div _bncavx512_rctd_div_4m
    //#define rctd_mul rctd_mul_3m
    //#define rctd_div rctd_div_3m
#endif // USE_RCDD_MUL_4M

#endif // defined(__AVX512F__)

#endif // ifndef __BNCAVX_CTD_AVX512_H
