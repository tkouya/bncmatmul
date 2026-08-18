// ------------------------
// SVE2 EFT primitives — float
// ------------------------
// Rewritten 2026: SVE "sizeless" types (svfloat32_t / svbool_t) cannot be
// array elements, struct members, sized, or have static storage, so this
// header passes each scalar EFT operand by value and the error component by
// pointer (per-limb-variable design).  All ops are predicated (svXXX_f32_x
// = "don't care about inactive lanes" mode, which is the right choice
// here because the caller always re-applies pg on store).
//
// FMA mapping (NEON -> SVE2, mirroring _bncneon_feft.h):
//   vfmaq_f32(acc, a, b) = acc + a*b   -> svmla_f32_x(pg, acc, a, b)
//   vfmsq_f32(acc, a, b) = acc - a*b   -> svmls_f32_x(pg, acc, a, b)
//   vnegq_f32(x)                       -> svneg_f32_x(pg, x)
//
// For ftwo_prod, the NEON version computes err = fma(a, b, -p)
// (i.e. -p + a*b), which is exactly svmla_f32_x(pg, svneg_f32_x(pg, p), a, b).
// We use that form below.
//
#ifndef __BNCSVE2_FEFT_H
#define __BNCSVE2_FEFT_H

#if defined(__ARM_SVE2)
#include <arm_sve.h>

/* s = a + b ; *err = b - (s - a)  (Assumes |a| >= |b|) */
static inline svfloat32_t _bncsve2_fquick_two_sum(
        svbool_t     pg,
        svfloat32_t  a,
        svfloat32_t  b,
        svfloat32_t *err)
{
    svfloat32_t s = svadd_f32_x(pg, a, b);
    *err          = svsub_f32_x(pg, b, svsub_f32_x(pg, s, a));
    return s;
}

/* s = a - b ; *err = (a - s) - b  (Assumes |a| >= |b|) */
static inline svfloat32_t _bncsve2_fquick_two_diff(
        svbool_t     pg,
        svfloat32_t  a,
        svfloat32_t  b,
        svfloat32_t *err)
{
    svfloat32_t s  = svsub_f32_x(pg, a, b);
    *err           = svsub_f32_x(pg, svsub_f32_x(pg, a, s), b);
    return s;
}

/* Standard Knuth/Dekker two-sum: s = a + b ; *err = (a - (s - bb)) + (b - bb) */
static inline svfloat32_t _bncsve2_ftwo_sum(
        svbool_t     pg,
        svfloat32_t  a,
        svfloat32_t  b,
        svfloat32_t *err)
{
    svfloat32_t s  = svadd_f32_x(pg, a, b);
    svfloat32_t bb = svsub_f32_x(pg, s, a);
    *err = svadd_f32_x(pg,
               svsub_f32_x(pg, a, svsub_f32_x(pg, s, bb)),
               svsub_f32_x(pg, b, bb));
    return s;
}

/* Two-diff (mirrors NEON _bncneon_ftwo_diff). */
static inline svfloat32_t _bncsve2_ftwo_diff(
        svbool_t     pg,
        svfloat32_t  a,
        svfloat32_t  b,
        svfloat32_t *err)
{
    svfloat32_t s  = svsub_f32_x(pg, a, b);
    svfloat32_t bb = svsub_f32_x(pg, s, a);
    *err = svsub_f32_x(pg,
               svsub_f32_x(pg, a, svsub_f32_x(pg, s, bb)),
               svadd_f32_x(pg, b, bb));
    return s;
}

/* FMA-based two-prod: p = a*b ; *err = a*b - p = fma(a, b, -p). */
static inline svfloat32_t _bncsve2_ftwo_prod(
        svbool_t     pg,
        svfloat32_t  a,
        svfloat32_t  b,
        svfloat32_t *err)
{
    svfloat32_t p = svmul_f32_x(pg, a, b);
    /* svmla_f32_x(pg, addend, x, y) = addend + x*y  -> err = -p + a*b */
    *err = svmla_f32_x(pg, svneg_f32_x(pg, p), a, b);
    return p;
}

#endif // defined(__ARM_SVE2)
#endif // __BNCSVE2_FEFT_H
