#ifndef __BNCNEON_FEFT_H
#define __BNCNEON_FEFT_H

#if defined(__ARM_NEON)
#include <arm_neon.h>

/************ Basic Functions for float ************ start **/

/* Computes fl(a+b) and err(a+b).  Assumes |a| >= |b|. */
static inline float32x4_t _bncneon_fquick_two_sum(float32x4_t a, float32x4_t b, float32x4_t *err)
{
	float32x4_t s;
    s = vaddq_f32(a, b);
    *err = vsubq_f32(b, vsubq_f32(s, a));
	return s;
}

/* Computes fl(a-b) and err(a-b).  Assumes |a| >= |b| */
static inline float32x4_t _bncneon_fquick_two_diff(float32x4_t a, float32x4_t b, float32x4_t *err)
{
	float32x4_t s;
    s = vsubq_f32(a, b);
    *err = vsubq_f32(vsubq_f32(a, s), b);
	return s;
}

/* Computes fl(a+b) and err(a+b).  */
static inline float32x4_t _bncneon_ftwo_sum(float32x4_t a, float32x4_t b, float32x4_t *err)
{
	float32x4_t s, bb;
    s = vaddq_f32(a, b);
    bb = vsubq_f32(s, a);
    *err = vaddq_f32(vsubq_f32(a, vsubq_f32(s, bb)), vsubq_f32(b, bb));
	return s;
}

/* Computes fl(a-b) and err(a-b).  */
static inline float32x4_t _bncneon_ftwo_diff(float32x4_t a, float32x4_t b, float32x4_t *err)
{
	float32x4_t s, bb;
    s = vsubq_f32(a, b);
    bb = vsubq_f32(s, a);
    *err = vsubq_f32(vsubq_f32(a, vsubq_f32(s, bb)), vaddq_f32(b, bb));
	return s;
}

/* Computes fl(a*b) and err(a*b). */
static inline float32x4_t _bncneon_ftwo_prod(float32x4_t a, float32x4_t b, float32x4_t *err)
{
	float32x4_t p;
    p = vmulq_f32(a, b);
    // *err = a * b - p (using Fused Multiply-Add)
    // FMA instruction vfmaq_f32(add, mul1, mul2) computes add + (mul1 * mul2)
    // We can compute -p + (a * b)
    *err = vfmaq_f32(vnegq_f32(p), a, b);
	return p;
}

#endif // defined(__ARM_NEON)

#endif // __BNCNEON_FEFT_H
