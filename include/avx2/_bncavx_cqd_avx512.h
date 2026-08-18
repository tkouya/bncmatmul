// ------------------------
// ------ Complex QD ------
// ------ AVX-512 Version --
// ------------------------
#ifndef __BNCAVX_CQD_AVX512_H
#define __BNCAVX_CQD_AVX512_H

#ifndef QDSIZE
    #define QDSIZE 4
#endif // QDSIZE

#if defined(__AVX512F__)

// ret := 0
static inline void _bncavx512_rcqd_set0(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE])
{
    ret_re[0] = _mm512_setzero_pd(); ret_re[1] = _mm512_setzero_pd();
    ret_re[2] = _mm512_setzero_pd(); ret_re[3] = _mm512_setzero_pd();
    ret_im[0] = _mm512_setzero_pd(); ret_im[1] = _mm512_setzero_pd();
    ret_im[2] = _mm512_setzero_pd(); ret_im[3] = _mm512_setzero_pd();
}
static inline void _bncavx512_set0_cqd(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE])
{
    ret_re[0] = _mm512_setzero_pd();  ret_re[1] = _mm512_setzero_pd(); ret_re[2] = _mm512_setzero_pd(); ret_re[3] = _mm512_setzero_pd();
    ret_im[0] = _mm512_setzero_pd();  ret_im[1] = _mm512_setzero_pd(); ret_im[2] = _mm512_setzero_pd(); ret_im[3] = _mm512_setzero_pd();
}

// ret := a
static inline void _bncavx512_rcqd_set(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a_re[QDSIZE], __m512d a_im[QDSIZE])
{
    _bncavx512_rqd_set(ret_re, a_re);
    _bncavx512_rqd_set(ret_im, a_im);
}

// ret := a
static inline void _bncavx512_rcqd_set_qd(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a[QDSIZE])
{
    _bncavx512_rqd_set(ret_re, a);
    _bncavx512_rqd_set0(ret_im);
}

// ret := a
static inline void _bncavx512_rcqd_set_qd_qd(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a_real[QDSIZE], __m512d a_imag[QDSIZE])
{
    _bncavx512_rqd_set(ret_re, a_real);
    _bncavx512_rqd_set(ret_im, a_imag);
}

// ret := a
static inline void _bncavx512_rcqd_set_ddfloat(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a[DDSIZE])
{
    ret_re[0] = a[0];
    ret_re[1] = a[1];
    ret_re[2] = _mm512_setzero_pd();
    ret_re[3] = _mm512_setzero_pd();
    _bncavx512_rqd_set0(ret_im);
}

// ret := a
static inline void _bncavx512_rcqd_set_tdfloat(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a[TDSIZE])
{
    ret_re[0] = a[0];
    ret_re[1] = a[1];
    ret_re[2] = a[2];
    ret_re[3] = _mm512_setzero_pd();
    _bncavx512_rqd_set0(ret_im);
}

// ret := a
static inline void _bncavx512_rcqd_set_d(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a)
{
    ret_re[0] = a;
    ret_re[1] = _mm512_setzero_pd();
    ret_re[2] = _mm512_setzero_pd();
    ret_re[3] = _mm512_setzero_pd();
    _bncavx512_rqd_set0(ret_im);
}

// ret := a
static inline void _bncavx512_rcqd_set_cd(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a_re, __m512d a_im)
{
    ret_re[0] = a_re; //creal(a);
    ret_re[1] = _mm512_setzero_pd();
    ret_re[2] = _mm512_setzero_pd();
    ret_re[3] = _mm512_setzero_pd();
    ret_im[0] = a_im; //cimag(a);
    ret_im[1] = _mm512_setzero_pd();
    ret_im[2] = _mm512_setzero_pd();
    ret_im[3] = _mm512_setzero_pd();
}

// ret := a
static inline void _bncavx512_rcqd_set_cdd(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a_re[DDSIZE], __m512d a_im[DDSIZE])
{
    ret_re[0] = a_re[0]; //creal(a);
    ret_re[1] = a_re[1];
    ret_re[2] = _mm512_setzero_pd();
    ret_re[3] = _mm512_setzero_pd();
    ret_im[0] = a_im[0]; //cimag(a);
    ret_im[1] = a_im[1];
    ret_im[2] = _mm512_setzero_pd();
    ret_im[3] = _mm512_setzero_pd();
}

// ret := a
static inline void _bncavx512_rcqd_set_ctd(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a_re[TDSIZE], __m512d a_im[TDSIZE])
{
    ret_re[0] = a_re[0]; //creal(a);
    ret_re[1] = a_re[1];
    ret_re[2] = a_re[2];
    ret_re[3] = _mm512_setzero_pd();
    ret_im[0] = a_im[0]; //cimag(a);
    ret_im[1] = a_im[1];
    ret_im[2] = a_im[2];
    ret_im[3] = _mm512_setzero_pd();
}

// ret := conj(a)
static inline void _bncavx512_rcqd_conj(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a_re[QDSIZE], __m512d a_im[QDSIZE])
{
    _bncavx512_rqd_set(ret_re, a_re);
    _bncavx512_rqd_neg(ret_im, a_im);
}

// ret := -a
static inline void _bncavx512_rcqd_neg(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a_re[QDSIZE], __m512d a_im[QDSIZE])
{
    _bncavx512_rqd_neg(ret_re, a_re);
    _bncavx512_rqd_neg(ret_im, a_im);
}

// ret := a + b
static inline void _bncavx512_rcqd_add(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a_re[QDSIZE], __m512d a_im[QDSIZE], __m512d b_re[QDSIZE], __m512d b_im[QDSIZE])
{
    // ret := a + b
    _bncavx512_rqd_add(ret_re, a_re, b_re);
    _bncavx512_rqd_add(ret_im, a_im, b_im);
}

// ret := a - b
static inline void _bncavx512_rcqd_sub(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a_re[QDSIZE], __m512d a_im[QDSIZE], __m512d b_re[QDSIZE], __m512d b_im[QDSIZE])
{
    // ret := a - b
    _bncavx512_rqd_sub(ret_re, a_re, b_re);
    _bncavx512_rqd_sub(ret_im, a_im, b_im);
}

// ret := a * b : 4M
static inline void _bncavx512_rcqd_mul_4m(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a_re[QDSIZE], __m512d a_im[QDSIZE], __m512d b_re[QDSIZE], __m512d b_im[QDSIZE])
{
    __m512d tmp[QDSIZE];

    // ret := a * b
    _bncavx512_rqd_mul(ret_re, a_re, b_re);
    _bncavx512_rqd_mul(tmp, a_im, b_im);
    _bncavx512_rqd_sub(ret_re, ret_re, tmp);

    _bncavx512_rqd_mul(ret_im, a_re, b_im);
    _bncavx512_rqd_mul(tmp, a_im, b_re);
    _bncavx512_rqd_add(ret_im, ret_im, tmp);
}

// ret := a * b : 3M
static inline void _bncavx512_rcqd_mul_3m(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a_re[QDSIZE], __m512d a_im[QDSIZE], __m512d b_re[QDSIZE], __m512d b_im[QDSIZE])
{
    __m512d t1[QDSIZE], t2[QDSIZE], a_re_im[QDSIZE], b_re_im[QDSIZE];

    // ret := a * b
    // t1      := re_a * re_b
    // t2      := im_a * im_b
    // -> ret_re  := t1 - t2 
    // a_re_im := re_a + im_a
    // b_re_im := re_b + im_b
    // -> ret_im  := a_re_im * b_re_im - t1 - t2
    _bncavx512_rqd_mul(t1, a_re, b_re);
    _bncavx512_rqd_mul(t2, a_im, b_im);
    _bncavx512_rqd_sub(ret_re, t1, t2);

    _bncavx512_rqd_add(a_re_im, a_re, a_im);
    _bncavx512_rqd_add(b_re_im, b_re, b_im);
    _bncavx512_rqd_mul(ret_im, a_re_im, b_re_im);
    _bncavx512_rqd_sub(ret_im, ret_im, t1);
    _bncavx512_rqd_sub(ret_im, ret_im, t2);
}

// ret := |a|^2
static inline void _bncavx512_rcqd_nrm2(__m512d ret[QDSIZE], __m512d a_re[QDSIZE], __m512d a_im[QDSIZE])
{
    __m512d tmp_re2[QDSIZE], tmp_im2[QDSIZE];

    _bncavx512_rqd_mul(tmp_re2, a_re, a_re);
    _bncavx512_rqd_mul(tmp_im2, a_im, a_im);
    _bncavx512_rqd_add(ret, tmp_re2, tmp_im2);
}

// ret := |a|
/* static inline void _bncavx512_rcqd_abs(qdfloat *ret, __m512d a_re[QDSIZE], __m512d a_im[QDSIZE])
{
    qdfloat tmp_re2, tmp_im2;

    _bncavx512_rqd_mul(tmp_re2.val, a_re, a_re);
    _bncavx512_rqd_mul(tmp_im2.val, a_im, a_im);
    _bncavx512_rqd_add(ret->val, tmp_re2.val, tmp_im2.val);
    _bncavx512_rqd_sqrt(ret->val, ret->val);
}

static inline void _bncavx512_rcqd_abs_qd(double ret[QDSIZE], __m512d a_re[QDSIZE], __m512d a_im[QDSIZE])
{
    double tmp_re2[QDSIZE], tmp_im2[QDSIZE];

    _bncavx512_rqd_mul(tmp_re2, a_re, a_re);
    _bncavx512_rqd_mul(tmp_im2, a_im, a_im);
    _bncavx512_rqd_add(ret, tmp_re2, tmp_im2);
    _bncavx512_rqd_sqrt(ret, ret);
}
*/

// ret := 1 / a
static inline void _bncavx512_rcqd_inv(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a_re[QDSIZE], __m512d a_im[QDSIZE])
{
    __m512d a_nrm2[QDSIZE], tmp[QDSIZE];

    //rcqd_conj(ret, a);
    _bncavx512_rqd_set(ret_re, a_re);
    _bncavx512_rqd_neg(ret_im, a_im);

    //rcqd_nrm2(&a_nrm2, a);
    _bncavx512_rqd_mul(a_nrm2, a_re, a_re);
    _bncavx512_rqd_mul(tmp, a_im, a_im);
    _bncavx512_rqd_add(a_nrm2, a_nrm2, tmp);

    _bncavx512_rqd_div(ret_re, ret_re, a_nrm2);
    _bncavx512_rqd_div(ret_im, ret_im, a_nrm2);
}

// ret := a / b : 3M
static inline void _bncavx512_rcqd_div_3m(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a_re[QDSIZE], __m512d a_im[QDSIZE], __m512d b_re[QDSIZE], __m512d b_im[QDSIZE])
{
    __m512d inv_b_re[QDSIZE], inv_b_im[QDSIZE];

    _bncavx512_rcqd_inv(inv_b_re, inv_b_im, b_re, b_im);
    _bncavx512_rcqd_mul_3m(ret_re, ret_im, a_re, a_im, inv_b_re, inv_b_im);
}

// ret := a / b : 4M
static inline void _bncavx512_rcqd_div_4m(__m512d ret_re[QDSIZE], __m512d ret_im[QDSIZE], __m512d a_re[QDSIZE], __m512d a_im[QDSIZE], __m512d b_re[QDSIZE], __m512d b_im[QDSIZE])
{
    __m512d inv_b_re[QDSIZE], inv_b_im[QDSIZE];

    _bncavx512_rcqd_inv(inv_b_re, inv_b_im, b_re, b_im);
    _bncavx512_rcqd_mul_4m(ret_re, ret_im, a_re, a_im, inv_b_re, inv_b_im);
}

// 2024-02-18 (SUN) Tomonori Kouya
// 2025-01-22: AVX-512 version
// Default: 4M method
#ifdef USE_RCQD_MUL_4M // 4M
    #define _bncavx512_rcqd_mul _bncavx512_rcqd_mul_4m
    #define _bncavx512_rcqd_div _bncavx512_rcqd_div_4m
#elif defined(USE_RCQD_MUL_3M) // 3M
    #define _bncavx512_rcqd_mul _bncavx512_rcqd_mul_3m
    #define _bncavx512_rcqd_div _bncavx512_rcqd_div_3m
#else // USE_RCDD_MUL_4M or 3M
    #define _bncavx512_rcqd_mul _bncavx512_rcqd_mul_4m
    #define _bncavx512_rcqd_div _bncavx512_rcqd_div_4m
    //#define _bncavx512_rcqd_mul _bncavx512_rcqd_mul_3m
    //#define _bncavx512_rcqd_div _bncavx512_rcqd_div_3m
#endif // USE_RCDD_MUL_4M

#endif // defined(__AVX512F__)

#endif // ifndef __BNCAVX_CQD_AVX512_H
