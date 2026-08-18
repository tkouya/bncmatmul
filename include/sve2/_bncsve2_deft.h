// ------------------------
// SVE2 EFT primitives — double
// ------------------------
// Rewritten 2026: sizeless SVE types must NOT be array elements.  Each
// EFT primitive returns the high (or sum/product) result as svfloat64_t
// and writes the error component through an svfloat64_t* (per-limb-variable
// design).  All ops are predicated (svXXX_f64_x = "don't care about inactive
// lanes").  Mirrors include/neon/_bncneon_deft.h exactly.
//
// FMA mapping (NEON -> SVE2, mirroring _bncneon_deft.h):
//   vfmaq_f64(acc, a, b) = acc + a*b  -> svmla_f64_x(pg, acc, a, b)
//   vfmsq_f64(acc, a, b) = acc - a*b  -> svmls_f64_x(pg, acc, a, b)
//
#ifndef __BNCSVE2_DEFT_H
#define __BNCSVE2_DEFT_H

#if defined(__ARM_SVE2)
#include <arm_sve.h>

/* s = a + b ; *err = b - (s - a)  (Assumes |a| >= |b|) */
static inline svfloat64_t _bncsve2_dquick_two_sum(
        svbool_t     pg,
        svfloat64_t  a,
        svfloat64_t  b,
        svfloat64_t *err)
{
    svfloat64_t s = svadd_f64_x(pg, a, b);
    *err          = svsub_f64_x(pg, b, svsub_f64_x(pg, s, a));
    return s;
}

/* s = a - b ; *err = (a - s) - b  (Assumes |a| >= |b|) */
static inline svfloat64_t _bncsve2_dquick_two_diff(
        svbool_t     pg,
        svfloat64_t  a,
        svfloat64_t  b,
        svfloat64_t *err)
{
    svfloat64_t s = svsub_f64_x(pg, a, b);
    *err          = svsub_f64_x(pg, svsub_f64_x(pg, a, s), b);
    return s;
}

/* Knuth two-sum (no |a|>=|b| precondition). */
static inline svfloat64_t _bncsve2_dtwo_sum(
        svbool_t     pg,
        svfloat64_t  a,
        svfloat64_t  b,
        svfloat64_t *err)
{
    svfloat64_t s  = svadd_f64_x(pg, a, b);
    svfloat64_t bb = svsub_f64_x(pg, s, a);
    *err = svadd_f64_x(pg,
               svsub_f64_x(pg, a, svsub_f64_x(pg, s, bb)),
               svsub_f64_x(pg, b, bb));
    return s;
}

/* Knuth two-diff. */
static inline svfloat64_t _bncsve2_dtwo_diff(
        svbool_t     pg,
        svfloat64_t  a,
        svfloat64_t  b,
        svfloat64_t *err)
{
    svfloat64_t s  = svsub_f64_x(pg, a, b);
    svfloat64_t bb = svsub_f64_x(pg, s, a);
    *err = svsub_f64_x(pg,
               svsub_f64_x(pg, a, svsub_f64_x(pg, s, bb)),
               svadd_f64_x(pg, b, bb));
    return s;
}

/* FMA-based two-prod: p = a*b ; *err = a*b - p = fma(a, b, -p). */
static inline svfloat64_t _bncsve2_dtwo_prod(
        svbool_t     pg,
        svfloat64_t  a,
        svfloat64_t  b,
        svfloat64_t *err)
{
    svfloat64_t p = svmul_f64_x(pg, a, b);
    /* svmla_f64_x(pg, addend, x, y) = addend + x*y  -> err = -p + a*b */
    *err = svmla_f64_x(pg, svneg_f64_x(pg, p), a, b);
    return p;
}

#endif // defined(__ARM_SVE2)
#endif // __BNCSVE2_DEFT_H
