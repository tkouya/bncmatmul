// ------------------------
// ------ Complex DD ------
// ------------------------
#ifndef __BNCNEON_CDD_H
#define __BNCNEON_CDD_H

#include "_bncneon_dd.h" // Assumes the previously created NEON DD header is available

// defined in rdd.h
#ifndef DDSIZE
    #define DDSIZE 2
#endif // DDSIZE

// ret := 0
static inline void _bncneon_rcdd_set0(float64x2_t ret_re[DDSIZE], float64x2_t ret_im[DDSIZE])
{
    _bncneon_set0_dd(ret_re);
    _bncneon_set0_dd(ret_im);
}
#define _bncneon_set0_cdd(ret_re, ret_im) _bncneon_rcdd_set0((ret_re), (ret_im))

// ret := a
static inline void _bncneon_rcdd_set(float64x2_t ret_re[DDSIZE], float64x2_t ret_im[DDSIZE], float64x2_t a_re[DDSIZE], float64x2_t a_im[DDSIZE])
{
    ret_re[0] = a_re[0];
    ret_re[1] = a_re[1];
    ret_im[0] = a_im[0];
    ret_im[1] = a_im[1];
}

// ret := conj(a)
static inline void _bncneon_rcdd_conj(float64x2_t ret_re[DDSIZE], float64x2_t ret_im[DDSIZE], float64x2_t a_re[DDSIZE], float64x2_t a_im[DDSIZE])
{
    ret_re[0] = a_re[0];
    ret_re[1] = a_re[1];
    ret_im[0] = vnegq_f64(a_im[0]);
    ret_im[1] = vnegq_f64(a_im[1]);
}

// ret := -a
static inline void _bncneon_rcdd_neg(float64x2_t ret_re[DDSIZE], float64x2_t ret_im[DDSIZE], float64x2_t a_re[DDSIZE], float64x2_t a_im[DDSIZE])
{
    ret_re[0] = vnegq_f64(a_re[0]);
    ret_re[1] = vnegq_f64(a_re[1]);
    ret_im[0] = vnegq_f64(a_im[0]);
    ret_im[1] = vnegq_f64(a_im[1]);
}

// ret := a + b
static inline void _bncneon_rcdd_add(float64x2_t ret_re[DDSIZE], float64x2_t ret_im[DDSIZE], float64x2_t a_re[DDSIZE], float64x2_t a_im[DDSIZE], float64x2_t b_re[DDSIZE], float64x2_t b_im[DDSIZE])
{
    _bncneon_rdd_add(ret_re, a_re, b_re);
    _bncneon_rdd_add(ret_im, a_im, b_im);
}

// ret := a - b
static inline void _bncneon_rcdd_sub(float64x2_t ret_re[DDSIZE], float64x2_t ret_im[DDSIZE], float64x2_t a_re[DDSIZE], float64x2_t a_im[DDSIZE], float64x2_t b_re[DDSIZE], float64x2_t b_im[DDSIZE])
{
    _bncneon_rdd_sub(ret_re, a_re, b_re);
    _bncneon_rdd_sub(ret_im, a_im, b_im);
}

// ret := a * b : 4M
static inline void _bncneon_rcdd_mul_4m(float64x2_t ret_re[DDSIZE], float64x2_t ret_im[DDSIZE], float64x2_t a_re[DDSIZE], float64x2_t a_im[DDSIZE], float64x2_t b_re[DDSIZE], float64x2_t b_im[DDSIZE])
{
    float64x2_t tmp1[DDSIZE], tmp2[DDSIZE];

    // ret_re = a_re * b_re - a_im * b_im
    _bncneon_rdd_mul(tmp1, a_re, b_re);
    _bncneon_rdd_mul(tmp2, a_im, b_im);
    _bncneon_rdd_sub(ret_re, tmp1, tmp2);

    // ret_im = a_re * b_im + a_im * b_re
    _bncneon_rdd_mul(tmp1, a_re, b_im);
    _bncneon_rdd_mul(tmp2, a_im, b_re);
    _bncneon_rdd_add(ret_im, tmp1, tmp2);
}

// ret := a * b : 3M
static inline void _bncneon_rcdd_mul_3m(float64x2_t ret_re[DDSIZE], float64x2_t ret_im[DDSIZE], float64x2_t a_re[DDSIZE], float64x2_t a_im[DDSIZE], float64x2_t b_re[DDSIZE], float64x2_t b_im[DDSIZE])
{
    float64x2_t t1[DDSIZE], t2[DDSIZE], tmp[DDSIZE], a_re_im[DDSIZE], b_re_im[DDSIZE];

    // t1 = re_a * re_b
    // t2 = im_a * im_b
    _bncneon_rdd_mul(t1, a_re, b_re);
    _bncneon_rdd_mul(t2, a_im, b_im);
    
    // ret_re = t1 - t2
    _bncneon_rdd_sub(ret_re, t1, t2);

    // ret_im = (re_a + im_a) * (re_b + im_b) - t1 - t2
    _bncneon_rdd_add(a_re_im, a_re, a_im);
    _bncneon_rdd_add(b_re_im, b_re, b_im);
    _bncneon_rdd_mul(tmp, a_re_im, b_re_im);
    _bncneon_rdd_sub(tmp, tmp, t1);
    _bncneon_rdd_sub(ret_im, tmp, t2);
}

// ret := |a|^2 = re(a)^2 + im(a)^2
static inline void _bncneon_rcdd_nrm2(float64x2_t ret[DDSIZE], float64x2_t a_re[DDSIZE], float64x2_t a_im[DDSIZE])
{
    float64x2_t tmp_re2[DDSIZE], tmp_im2[DDSIZE];

    _bncneon_rdd_mul(tmp_re2, a_re, a_re);
    _bncneon_rdd_mul(tmp_im2, a_im, a_im);
    _bncneon_rdd_add(ret, tmp_re2, tmp_im2);
}

// ret := 1 / a
static inline void _bncneon_rcdd_inv(float64x2_t ret_re[DDSIZE], float64x2_t ret_im[DDSIZE], float64x2_t a_re[DDSIZE], float64x2_t a_im[DDSIZE])
{
    float64x2_t a_nrm2[DDSIZE], tmp[DDSIZE];

    //rcdd_conj(ret, a);
    _bncneon_rdd_set(ret_re, a_re);
    _bncneon_rdd_neg(ret_im, a_im);
    //rcdd_nrm2(&a_nrm2, a);
    _bncneon_rdd_mul(   tmp, a_re, a_re);
    _bncneon_rdd_mul(a_nrm2, a_im, a_im);
    _bncneon_rdd_add(a_nrm2, a_nrm2, tmp);

    _bncneon_rdd_div(ret_re, ret_re, a_nrm2);
    _bncneon_rdd_div(ret_im, ret_im, a_nrm2);

#if 0
    float64x2_t a_nrm2[DDSIZE];
    float64x2_t conj_a_re[DDSIZE], conj_a_im[DDSIZE];

    // a_nrm2 = |a|^2
    _bncneon_rcdd_nrm2(a_nrm2, a_re, a_im);

    // conj(a)
    _bncneon_rcdd_conj(conj_a_re, conj_a_im, a_re, a_im);

    // ret = conj(a) / |a|^2
    // Since nrm2 is real, division is component-wise.
    // We need a DD / DD division function here. Let's assume one exists based on the AVX version.
    // The AVX version _bncavx2_rdd_div is complex, let's create a real one for this.
    // For now, let's implement a sloppy division for real DD.
    
    // Sloppy DD/DD division: ret = x / y
    // q1 = x.h / y.h
    // r = x - q1 * y
    // q2 = r / y.h
    // ret = q1 + q2
    float64x2_t q1_re, q2_re, r_re[DDSIZE], tmp_re[DDSIZE];
    float64x2_t q1_im, q2_im, r_im[DDSIZE], tmp_im[DDSIZE];

    // Real part
    q1_re = vdivq_f64(conj_a_re[0], a_nrm2[0]);
    _bncneon_rdd_mul(tmp_re, a_nrm2, (float64x2_t[2]){q1_re, vdupq_n_f64(0.0)});
    _bncneon_rdd_sub(r_re, conj_a_re, tmp_re);
    q2_re = vdivq_f64(r_re[0], a_nrm2[0]);
    ret_re[0] = _bncneon_dquick_two_sum(q1_re, q2_re, &ret_re[1]);

    // Imaginary part
    q1_im = vdivq_f64(conj_a_im[0], a_nrm2[0]);
    _bncneon_rdd_mul(tmp_im, a_nrm2, (float64x2_t[2]){q1_im, vdupq_n_f64(0.0)});
    _bncneon_rdd_sub(r_im, conj_a_im, tmp_im);
    q2_im = vdivq_f64(r_im[0], a_nrm2[0]);
    ret_im[0] = _bncneon_dquick_two_sum(q1_im, q2_im, &ret_im[1]);
#endif // 0
}

// ret := a / b : 3M
static inline void _bncneon_rcdd_div_3m(float64x2_t ret_re[DDSIZE], float64x2_t ret_im[DDSIZE], float64x2_t a_re[DDSIZE], float64x2_t a_im[DDSIZE], float64x2_t b_re[DDSIZE], float64x2_t b_im[DDSIZE])
{
    float64x2_t inv_b_re[DDSIZE], inv_b_im[DDSIZE];
   _bncneon_rcdd_inv(inv_b_re, inv_b_im, b_re, b_im);
   _bncneon_rcdd_mul_3m(ret_re, ret_im, a_re, a_im, inv_b_re, inv_b_im);
}

// ret := a / b : 4M
static inline void _bncneon_rcdd_div_4m(float64x2_t ret_re[DDSIZE], float64x2_t ret_im[DDSIZE], float64x2_t a_re[DDSIZE], float64x2_t a_im[DDSIZE], float64x2_t b_re[DDSIZE], float64x2_t b_im[DDSIZE])
{
    float64x2_t inv_b_re[DDSIZE], inv_b_im[DDSIZE];
   _bncneon_rcdd_inv(inv_b_re, inv_b_im, b_re, b_im);    
   _bncneon_rcdd_mul_4m(ret_re, ret_im, a_re, a_im, inv_b_re, inv_b_im);
}

// Default: 4M method
#ifdef USE_RCDD_MUL_4M // 4M
    #define _bncneon_rcdd_mul _bncneon_rcdd_mul_4m
    #define _bncneon_rcdd_div _bncneon_rcdd_div_4m
#elif defined(USE_RCDD_MUL_3M) // 3M
    #define _bncneon_rcdd_mul _bncneon_rcdd_mul_3m
    #define _bncneon_rcdd_div _bncneon_rcdd_div_3m
#else // Default to 4M
    #define _bncneon_rcdd_mul _bncneon_rcdd_mul_4m
    #define _bncneon_rcdd_div _bncneon_rcdd_div_4m
#endif

#endif // __BNCNEON_CDD_H
