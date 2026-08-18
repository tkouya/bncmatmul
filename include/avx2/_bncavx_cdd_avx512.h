// ------------------------
// ------ Complex DD ------
// ------ AVX-512 Version --
// ------------------------
#ifndef __BNCAVX_CDD_AVX512_H
#define __BNCAVX_CDD_AVX512_H

// defined in rdd.h
#ifndef DDSIZE
    #define DDSIZE 2
#endif // DDSIZE

#if defined(__AVX512F__)

// ret := 0
static inline void _bncavx512_rcdd_set0(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE])
{
    ret_re[0] = _mm512_setzero_pd(); ret_re[1] = _mm512_setzero_pd();
    ret_im[0] = _mm512_setzero_pd(); ret_im[1] = _mm512_setzero_pd();
}
#define _bncavx512_set0_cdd(ret_re, ret_im) _bncavx512_rcdd_set0((ret_re), (ret_im))

// ret := a
static inline void _bncavx512_rcdd_set(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE], __m512d a_re[DDSIZE], __m512d a_im[DDSIZE])
{
    _bncavx512_rdd_set(ret_re, a_re);
    _bncavx512_rdd_set(ret_im, a_im);
}

// ret := a
static inline void _bncavx512_rcdd_set_dd(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE], __m512d a[DDSIZE])
{
    _bncavx512_rdd_set(ret_re, a);
    _bncavx512_rdd_set0(ret_im);
}
static inline void _bncavx512_rcdd_set_ddfloat(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE], __m512d a[DDSIZE])
{
    _bncavx512_rdd_set(ret_re, a);
    _bncavx512_rdd_set0(ret_im);
}

// ret := a
static inline void _bncavx512_rcdd_set_dd_dd(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE], __m512d a_real[DDSIZE], __m512d a_imag[DDSIZE])
{
    _bncavx512_rdd_set(ret_re, a_real);
    _bncavx512_rdd_set(ret_im, a_imag);
}

// ret := a
static inline void _bncavx512_rcdd_set_d(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE], __m512d a)
{
    ret_re[0] = a;
    ret_re[1] = _mm512_setzero_pd();
    _bncavx512_rdd_set0(ret_im);
}

// ret := a
static inline void _bncavx512_rcdd_set_cd(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE], __m512d a_re, __m512d a_im)
{
    ret_re[0] = a_re; //creal(a);
    ret_re[1] = _mm512_setzero_pd();
    ret_im[0] = a_im; //cimag(a);
    ret_im[1] = _mm512_setzero_pd();
}

// ret := conj(a)
static inline void _bncavx512_rcdd_conj(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE], __m512d a_re[DDSIZE], __m512d a_im[DDSIZE])
{
    _bncavx512_rdd_set(ret_re, a_re);
    _bncavx512_rdd_neg(ret_im, a_im);
}

// ret := -a
static inline void _bncavx512_rcdd_neg(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE], __m512d a_re[DDSIZE], __m512d a_im[DDSIZE])
{
    _bncavx512_rdd_neg(ret_re, a_re);
    _bncavx512_rdd_neg(ret_im, a_im);
}

// ret := a + b
static inline void _bncavx512_rcdd_add(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE], __m512d a_re[DDSIZE], __m512d a_im[DDSIZE], __m512d b_re[DDSIZE], __m512d b_im[DDSIZE])
{
    // ret := a + b
    _bncavx512_rdd_add(ret_re, a_re, b_re);
    _bncavx512_rdd_add(ret_im, a_im, b_im);
}

// ret := a + (double)b
static inline void _bncavx512_rcdd_add_dd(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE], __m512d a_re[DDSIZE], __m512d a_im[DDSIZE], __m512d b[DDSIZE])
{
    // ret := a + b
    _bncavx512_rdd_add(ret_re, a_re, b);
    _bncavx512_rdd_set(ret_im, a_im); //, b_im);
}

// ret := a - b
static inline void _bncavx512_rcdd_sub(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE], __m512d a_re[DDSIZE], __m512d a_im[DDSIZE], __m512d b_re[DDSIZE], __m512d b_im[DDSIZE])
{
    // ret := a - b
    _bncavx512_rdd_sub(ret_re, a_re, b_re);
    _bncavx512_rdd_sub(ret_im, a_im, b_im);
}

// ret := a - (double)b
static inline void _bncavx512_rcdd_sub_dd(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE], __m512d a_re[DDSIZE], __m512d a_im[DDSIZE], __m512d b[DDSIZE])
{
    // ret := a - b
    _bncavx512_rdd_sub(ret_re, a_re, b);
    _bncavx512_rdd_set(ret_im, a_im); // , b_im);
}

// ret := a * b : 4M
static inline void _bncavx512_rcdd_mul_4m(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE], __m512d a_re[DDSIZE], __m512d a_im[DDSIZE], __m512d b_re[DDSIZE], __m512d b_im[DDSIZE])
{
    //ddfloat tmp;
    __m512d tmp[DDSIZE];

    // ret := a * b
    _bncavx512_rdd_mul(ret_re, a_re, b_re);
    _bncavx512_rdd_mul(tmp, a_im, b_im);
    _bncavx512_rdd_sub(ret_re, ret_re, tmp);

    _bncavx512_rdd_mul(ret_im, a_re, b_im);
    _bncavx512_rdd_mul(tmp, a_im, b_re);
    _bncavx512_rdd_add(ret_im, ret_im, tmp);
}

// ret := a * b : 3M
static inline void _bncavx512_rcdd_mul_3m(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE], __m512d a_re[DDSIZE], __m512d a_im[DDSIZE], __m512d b_re[DDSIZE], __m512d b_im[DDSIZE])
{
    __m512d t1[DDSIZE], t2[DDSIZE], tmp[DDSIZE], a_re_im[DDSIZE], b_re_im[DDSIZE];

    // ret := a * b

    // t1      := re_a * re_b
    // t2      := im_a * im_b
    // -> ret_re  := t1 - t2 
    _bncavx512_rdd_mul(t1, a_re, b_re);
    _bncavx512_rdd_mul(t2, a_im, b_im);
    _bncavx512_rdd_sub(ret_re, t1, t2);
    /*
    printf("t1->"); rdd_out_str(t1.val); printf("\n");
    printf("t2->"); rdd_out_str(t2.val); printf("\n");
    printf("ret_re->"); rdd_out_str(ret_re); printf("\n"); 
    */

    // a_re_im := re_a + im_a
    // b_re_im := re_b + im_b
    // -> ret_im  := a_re_im * b_re_im - t1 - t2
    _bncavx512_rdd_add(a_re_im, a_re, a_im);
    _bncavx512_rdd_add(b_re_im, b_re, b_im);
    _bncavx512_rdd_mul(tmp, a_re_im, b_re_im);
    //printf("tmp->"); rdd_out_str(tmp.val); printf("\n");
    _bncavx512_rdd_sub(tmp, tmp, t1);
    //printf("tmp-t1->"); rdd_out_str(tmp.val); printf("\n");
    _bncavx512_rdd_sub(ret_im, tmp, t2);
    //printf("tmp-t2->"); rdd_out_str(ret_im); printf("\n");

    //printf("rcdd_mul_3m_ret->");_bncavx512_rcdd_out_str(ret); printf("\n");
}

static inline void _bncavx512_rcdd_nrm2(__m512d ret[DDSIZE], __m512d a_re[DDSIZE], __m512d a_im[DDSIZE])
{
    __m512d tmp_re2[DDSIZE], tmp_im2[DDSIZE];

    _bncavx512_rdd_mul(tmp_re2, a_re, a_re);
    _bncavx512_rdd_mul(tmp_im2, a_im, a_im);
    _bncavx512_rdd_add(ret, tmp_re2, tmp_im2);
}

/*
// ret := |a|
static inline void _bncavx512_rcdd_abs(double ret[DDSIZE], __m512d a_re[DDSIZE], __m512d a_im[DDSIZE])
{
    __m512d tmp_re2[DDSIZE], tmp_im2[DDSIZE];

    _bncavx512_rdd_mul(tmp_re2, a_re, a_re);
    _bncavx512_rdd_mul(tmp_im2, a_im, a_im);
    _bncavx512_rdd_add(ret, tmp_re2, tmp_im2);
    _bncavx512_rdd_sqrt(ret, ret);
}
*/

// ret := 1 / a
static inline void _bncavx512_rcdd_inv(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE], __m512d a_re[DDSIZE], __m512d a_im[DDSIZE])
{
    __m512d a_nrm2[DDSIZE], tmp[DDSIZE];

    //rcdd_conj(ret, a);
    _bncavx512_rdd_set(ret_re, a_re);
    _bncavx512_rdd_neg(ret_im, a_im);
    //rcdd_nrm2(&a_nrm2, a);
    _bncavx512_rdd_mul(   tmp, a_re, a_re);
    _bncavx512_rdd_mul(a_nrm2, a_im, a_im);
    _bncavx512_rdd_add(a_nrm2, a_nrm2, tmp);

    _bncavx512_rdd_div(ret_re, ret_re, a_nrm2);
    _bncavx512_rdd_div(ret_im, ret_im, a_nrm2);
}

// ret := a / b : 3M
static inline void _bncavx512_rcdd_div_3m(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE], __m512d a_re[DDSIZE], __m512d a_im[DDSIZE], __m512d b_re[DDSIZE], __m512d b_im[DDSIZE])
{
    __m512d inv_b_re[DDSIZE], inv_b_im[DDSIZE], in_a_re[DDSIZE], in_a_im[DDSIZE];

   _bncavx512_rcdd_inv(inv_b_re, inv_b_im, b_re, b_im);
   _bncavx512_rcdd_set(in_a_re, in_a_im, a_re, a_im);
    //printf("inv_b-> ");_bncavx512_rcdd_out_str(&inv_b); printf("\n");
   _bncavx512_rcdd_mul_3m(ret_re, ret_im, in_a_re, in_a_im, inv_b_re, inv_b_im);
    //printf("ret-> ");_bncavx512_rcdd_out_str(ret); printf("\n");
}

// ret := a / b : 4M
static inline void _bncavx512_rcdd_div_4m(__m512d ret_re[DDSIZE], __m512d ret_im[DDSIZE], __m512d a_re[DDSIZE], __m512d a_im[DDSIZE], __m512d b_re[DDSIZE], __m512d b_im[DDSIZE])
{
    __m512d inv_b_re[DDSIZE], inv_b_im[DDSIZE];

   _bncavx512_rcdd_inv(inv_b_re, inv_b_im, b_re, b_im);    
   _bncavx512_rcdd_mul_4m(ret_re, ret_im, a_re, a_im, inv_b_re, inv_b_im);

    // ret := a * b
/*
    rdd_mul(ret_re, a_re, inv_b.val_re);
    rdd_mul(tmp.val, a_im, inv_b.val_im);
    rdd_sub(ret_re, ret_re, tmp.val);

    rdd_mul(ret_im, a_re, inv_b.val_im);
    rdd_mul(tmp.val, a_im, inv_b.val_re);
    rdd_add(ret_im, ret_im, tmp.val);
*/
}

// 2024-02-18 (SUN) Tomonori Kouya
// 2025-01-22: AVX-512 version
// Default: 4M method
#ifdef USE_RCDD_MUL_4M // 4M
    #define _bncavx512_rcdd_mul _bncavx512_rcdd_mul_4m
    #define _bncavx512_rcdd_div _bncavx512_rcdd_div_4m
#elif defined(USE_RCDD_MUL_3M) // 3M
    #define _bncavx512_rcdd_mul _bncavx512_rcdd_mul_3m
    #define _bncavx512_rcdd_div _bncavx512_rcdd_div_3m
#else // USE_RCDD_MUL_4M
    #define _bncavx512_rcdd_mul _bncavx512_rcdd_mul_4m
    #define _bncavx512_rcdd_div _bncavx512_rcdd_div_4m
    //#define_bncavx512_rcdd_mul _bncavx512_rcdd_mul_3m
    //#define_bncavx512_rcdd_div _bncavx512_rcdd_div_3m
#endif // USE_RCDD_MUL_4M

#endif // defined(__AVX512F__)

#endif // ifndef __BNCAVX_CDD_AVX512_H
