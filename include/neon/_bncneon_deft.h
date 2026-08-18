// ------------------------
// Error Freee Trans. with double
// ------------------------
#ifndef __BNCNEON_DEFT_H
#define __BNCNEON_DEFT_H

#if defined(__ARM_NEON)
#include <arm_neon.h>

// ----------------------------------------
// Error Free Transformations with NEON
// ----------------------------------------

/* Computes fl(a+b) and err(a+b). Assumes |a| >= |b|. */
/* ----------------------------------------------------------------
 * _bncneon_dquick_two_sum
 *   return value = a + b  (hi part)
 *   *err         = b - (return - a)  (lo part)
 *   Precondition: |a| >= |b|
 *
 *   Usage: ret[0] = _bncneon_dquick_two_sum(a, b, &(ret[1]));
 * ---------------------------------------------------------------- */
static inline float64x2_t _bncneon_dquick_two_sum(
        float64x2_t  a,
        float64x2_t  b,
        float64x2_t *err)
{
    float64x2_t s = vaddq_f64(a, b);
    *err = vsubq_f64(b, vsubq_f64(s, a));
    return s;
}


/* Computes fl(a+b) and err(a+b). */
static inline float64x2_t _bncneon_dtwo_sum(float64x2_t a, float64x2_t b, float64x2_t *err)
{
	float64x2_t s, bb;
    s = vaddq_f64(a, b);
    bb = vsubq_f64(s, a);
    *err = vaddq_f64(vsubq_f64(a, vsubq_f64(s, bb)), vsubq_f64(b, bb));
	return s;
}

/* Computes fl(a-b) and err(a-b). */
static inline float64x2_t _bncneon_dtwo_diff(float64x2_t a, float64x2_t b, float64x2_t *err)
{
	float64x2_t s, bb;
    s = vsubq_f64(a, b);
    bb = vsubq_f64(s, a);
    *err = vsubq_f64(vsubq_f64(a, vsubq_f64(s, bb)), vaddq_f64(b, bb));
	return s;
}

/* Computes fl(a*b) and err(a*b). */
static inline float64x2_t _bncneon_dtwo_prod(float64x2_t a, float64x2_t b, float64x2_t *err)
{
	float64x2_t p;
    p = vmulq_f64(a, b);
    // err = a * b - p (FMA)
    *err = vfmaq_f64(vnegq_f64(p), a, b);
	return p;
}
#endif //defined(__ARM_NEON)

#endif //ifndef __BNCNEON_DEFT_H
