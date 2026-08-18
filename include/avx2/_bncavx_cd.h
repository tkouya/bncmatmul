// ------------------------
// ---- Complex Double ----
// ------------------------
#ifndef __BNCAVX_CD_H
#define __BNCAVX_CD_H

#if defined(__AVX2__)

// ret := 0
static inline void _bncavx2_cd_set0(__m256d ret_re, __m256d ret_im)
{
    ret_re = _mm256_setzero_pd();
    ret_im = _mm256_setzero_pd();
}
#define _bncavx2_cd_set0(ret_re, ret_im) _bncavx2_set0_cd((ret_re), (ret_im))

// ret := a
static inline void _bncavx2_cd_set(__m256d ret_re, __m256d ret_im, __m256d a_re, __m256d a_im)
{
    ret_re = a_re;
    ret_im = a_im;
}

// ret := a
static inline void _bncavx2_cd_set_d(__m256d ret_re, __m256d ret_im, __m256d a)
{
    ret_re = a;
    _bncavx2_rdd_set0(ret_im);
}
static inline void _bncavx2_cd_set_ddfloat(__m256d ret_re, __m256d ret_im, __m256d a)
{
    _bncavx2_rdd_set(ret_re, a);
    _bncavx2_rdd_set0(ret_im);
}

// ret := a
static inline void _bncavx2_cd_set_dd_dd(__m256d ret_re, __m256d ret_im, __m256d a_real, __m256d a_imag)
{
    _bncavx2_rdd_set(ret_re, a_real);
    _bncavx2_rdd_set(ret_im, a_imag);
}

// ret := a
static inline void _bncavx2_cd_set_d(__m256d ret_re, __m256d ret_im, __m256d a)
{
    ret_re[0] = a;
    ret_re[1] = _mm256_setzero_pd();
    _bncavx2_rdd_set0(ret_im);
}

// ret := a
static inline void _bncavx2_cd_set_cd(__m256d ret_re, __m256d ret_im, __m256d a_re, __m256d a_im)
{
    ret_re[0] = a_re; //creal(a);
    ret_re[1] = _mm256_setzero_pd();
    ret_im[0] = a_im; //cimag(a);
    ret_im[1] = _mm256_setzero_pd();
}

// ret := conj(a)
static inline void _bncavx2_cd_conj(__m256d ret_re, __m256d ret_im, __m256d a_re, __m256d a_im)
{
    _bncavx2_rdd_set(ret_re, a_re);
    _bncavx2_rdd_neg(ret_im, a_im);
}

// ret := -a
static inline void _bncavx2_cd_neg(__m256d ret_re, __m256d ret_im, __m256d a_re, __m256d a_im)
{
    _bncavx2_rdd_neg(ret_re, a_re);
    _bncavx2_rdd_neg(ret_im, a_im);
}

// ret := a + b
static inline void _bncavx2_cd_add(__m256d ret_re, __m256d ret_im, __m256d a_re, __m256d a_im, __m256d b_re, __m256d b_im)
{
    // ret := a + b
    _bncavx2_rdd_add(ret_re, a_re, b_re);
    _bncavx2_rdd_add(ret_im, a_im, b_im);
}

// ret := a - b
static inline void _bncavx2_cd_sub(__m256d ret_re, __m256d ret_im, __m256d a_re, __m256d a_im, __m256d b_re, __m256d b_im)
{
    // ret := a - b
    _bncavx2_rdd_sub(ret_re, a_re, b_re);
    _bncavx2_rdd_sub(ret_im, a_im, b_im);
}

// ret := a * b : 4M
static inline void _bncavx2_cd_mul_4m(__m256d ret_re, __m256d ret_im, __m256d a_re, __m256d a_im, __m256d b_re, __m256d b_im)
{
    //ddfloat tmp;
    __m256d tmp;

    // ret := a * b
    _bncavx2_rdd_mul(ret_re, a_re, b_re);
    _bncavx2_rdd_mul(tmp, a_im, b_im);
    _bncavx2_rdd_sub(ret_re, ret_re, tmp);

    _bncavx2_rdd_mul(ret_im, a_re, b_im);
    _bncavx2_rdd_mul(tmp, a_im, b_re);
    _bncavx2_rdd_add(ret_im, ret_im, tmp);
}

// ret := a * b : 3M
static inline void _bncavx2_cd_mul_3m(__m256d ret_re, __m256d ret_im, __m256d a_re, __m256d a_im, __m256d b_re, __m256d b_im)
{
    __m256d t1, t2, tmp, a_re_im, b_re_im;

    // ret := a * b

    // t1      := re_a * re_b
    // t2      := im_a * im_b
    // -> ret_re  := t1 - t2 
    _bncavx2_rdd_mul(t1, a_re, b_re);
    _bncavx2_rdd_mul(t2, a_im, b_im);
    _bncavx2_rdd_sub(ret_re, t1, t2);
    /*
    printf("t1->"); rdd_out_str(t1.val); printf("\n");
    printf("t2->"); rdd_out_str(t2.val); printf("\n");
    printf("ret_re->"); rdd_out_str(ret_re); printf("\n"); 
    */

    // a_re_im := re_a + im_a
    // b_re_im := re_b + im_b
    // -> ret_im  := a_re_im * b_re_im - t1 - t2
    _bncavx2_rdd_add(a_re_im, a_re, a_im);
    _bncavx2_rdd_add(b_re_im, b_re, b_im);
    _bncavx2_rdd_mul(tmp, a_re_im, b_re_im);
    //printf("tmp->"); rdd_out_str(tmp.val); printf("\n");
    _bncavx2_rdd_sub(tmp, tmp, t1);
    //printf("tmp-t1->"); rdd_out_str(tmp.val); printf("\n");
    _bncavx2_rdd_sub(ret_im, tmp, t2);
    //printf("tmp-t2->"); rdd_out_str(ret_im); printf("\n");

    //printf("cd_mul_3m_ret->");_bncavx2_cd_out_str(ret); printf("\n");
}

static inline void _bncavx2_cd_nrm2(__m256d ret, __m256d a_re, __m256d a_im)
{
    __m256d tmp_re2, tmp_im2;

    _bncavx2_rdd_mul(tmp_re2, a_re, a_re);
    _bncavx2_rdd_mul(tmp_im2, a_im, a_im);
    _bncavx2_rdd_add(ret, tmp_re2, tmp_im2);
}

/*
// ret := |a|
static inline void _bncavx2_cd_abs(double ret, __m256d a_re, __m256d a_im)
{
    __m256d tmp_re2, tmp_im2;

    _bncavx2_rdd_mul(tmp_re2, a_re, a_re);
    _bncavx2_rdd_mul(tmp_im2, a_im, a_im);
    _bncavx2_rdd_add(ret, tmp_re2, tmp_im2);
    _bncavx2_rdd_sqrt(ret, ret);
}
*/

// ret := 1 / a
static inline void _bncavx2_cd_inv(__m256d ret_re, __m256d ret_im, __m256d a_re, __m256d a_im)
{
    __m256d a_nrm2, tmp;

    //cd_conj(ret, a);
    _bncavx2_rdd_set(ret_re, a_re);
    _bncavx2_rdd_neg(ret_im, a_im);
    //cd_nrm2(&a_nrm2, a);
    _bncavx2_rdd_mul(   tmp, a_re, a_re);
    _bncavx2_rdd_mul(a_nrm2, a_im, a_im);
    _bncavx2_rdd_add(a_nrm2, a_nrm2, tmp);

    _bncavx2_rdd_div(ret_re, ret_re, a_nrm2);
    _bncavx2_rdd_div(ret_im, ret_im, a_nrm2);
}

// ret := a / b : 3M
static inline void _bncavx2_cd_div_3m(__m256d ret_re, __m256d ret_im, __m256d a_re, __m256d a_im, __m256d b_re, __m256d b_im)
{
    __m256d inv_b_re, inv_b_im, in_a_re, in_a_im;

   _bncavx2_cd_inv(inv_b_re, inv_b_im, b_re, b_im);
   _bncavx2_cd_set(in_a_re, in_a_im, a_re, a_im);
    //printf("inv_b-> ");_bncavx2_cd_out_str(&inv_b); printf("\n");
   _bncavx2_cd_mul_3m(ret_re, ret_im, in_a_re, in_a_im, inv_b_re, inv_b_im);
    //printf("ret-> ");_bncavx2_cd_out_str(ret); printf("\n");
}

// ret := a / b : 4M
static inline void _bncavx2_cd_div_4m(__m256d ret_re, __m256d ret_im, __m256d a_re, __m256d a_im, __m256d b_re, __m256d b_im)
{
    __m256d inv_b_re, inv_b_im;

   _bncavx2_cd_inv(inv_b_re, inv_b_im, b_re, b_im);    
   _bncavx2_cd_mul_4m(ret_re, ret_im, a_re, a_im, inv_b_re, inv_b_im);

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
// Default: 4M method
#ifdef USE_cd_MUL_4M // 4M
    #define _bncavx2_cd_mul _bncavx2_cd_mul_4m
    #define _bncavx2_cd_div _bncavx2_cd_div_4m
#elif defined(USE_cd_MUL_3M) // 3M
    #define _bncavx2_cd_mul _bncavx2_cd_mul_3m
    #define _bncavx2_cd_div _bncavx2_cd_div_3m
#else // USE_cd_MUL_4M
    #define _bncavx2_cd_mul _bncavx2_cd_mul_4m
    #define _bncavx2_cd_div _bncavx2_cd_div_4m
    //#define_bncavx2_cd_mul _bncavx2_cd_mul_3m
    //#define_bncavx2_cd_div _bncavx2_cd_div_3m
#endif // USE_cd_MUL_4M

#endif // defined(__AVX2__)

#endif // ifndef __BNCAVX_CDD_H
