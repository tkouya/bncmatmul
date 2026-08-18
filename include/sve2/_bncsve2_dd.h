// -------------------------------------------------------
// _bncsve2_dd.h  --  SVE2 Double-Double (DD) per-limb-variable helpers
// -------------------------------------------------------
// Rewritten 2026: a DD value is now passed as TWO separate svfloat64_t
// variables (hi, lo) instead of an svfloat64_t[2] array, because SVE
// "sizeless" types cannot be array elements.  Output limbs are returned
// through svfloat64_t* pointers.
//
// Mirrors include/neon/_bncneon_dd.h.  Only the operations needed by the
// matmul kernels (set0, set_d, neg, add, mul) plus their variants are
// ported.  Other helpers can be added later as needed.
// -------------------------------------------------------
#ifndef __BNCSVE2_DD_H
#define __BNCSVE2_DD_H

#if defined(__ARM_SVE2)
#include <arm_sve.h>

#ifndef DDSIZE
    #define DDSIZE 2
#endif

/* *r0 = 0, *r1 = 0 */
static inline void _bncsve2_rdd_set0(svfloat64_t *r0, svfloat64_t *r1)
{
    *r0 = svdup_n_f64(0.0);
    *r1 = svdup_n_f64(0.0);
}

/* (*r0,*r1) := (val, 0)  -- promote scalar/double-vec to DD */
static inline void _bncsve2_rdd_set_d(svfloat64_t *r0, svfloat64_t *r1,
                                       svfloat64_t v)
{
    *r0 = v;
    *r1 = svdup_n_f64(0.0);
}

/* (*r0,*r1) := -(a0,a1) */
static inline void _bncsve2_rdd_neg(svbool_t pg,
                                     svfloat64_t *r0, svfloat64_t *r1,
                                     svfloat64_t a0, svfloat64_t a1)
{
    *r0 = svneg_f64_x(pg, a0);
    *r1 = svneg_f64_x(pg, a1);
}

/* (*r0,*r1) := (a0,a1) + (b0,b1)  -- sloppy (NEON: _bncneon_rdd_add_sloppy) */
static inline void _bncsve2_rdd_add_sloppy(
        svbool_t pg,
        svfloat64_t *r0, svfloat64_t *r1,
        svfloat64_t a0, svfloat64_t a1,
        svfloat64_t b0, svfloat64_t b1)
{
    svfloat64_t e;
    svfloat64_t s = _bncsve2_dtwo_sum(pg, a0, b0, &e);
    e  = svadd_f64_x(pg, e, svadd_f64_x(pg, a1, b1));
    *r0 = _bncsve2_dquick_two_sum(pg, s, e, r1);
}

/* (*r0,*r1) := (a0,a1) + (b0,b1)  -- branch-free gates */
static inline void _bncsve2_rdd_add_bf(
        svbool_t pg,
        svfloat64_t *r0, svfloat64_t *r1,
        svfloat64_t a0, svfloat64_t a1,
        svfloat64_t b0, svfloat64_t b1)
{
    svfloat64_t g1e, g2e, g3e, g6e;
    svfloat64_t g1 = _bncsve2_dtwo_sum(pg, a0, b0, &g1e);
    svfloat64_t g2 = _bncsve2_dtwo_sum(pg, a1, b1, &g2e);
    svfloat64_t g3 = _bncsve2_dquick_two_sum(pg, g1, g2, &g3e);
    svfloat64_t g4 = svadd_f64_x(pg, g1e, g2e);
    svfloat64_t g5 = svadd_f64_x(pg, g4, g3e);
    svfloat64_t g6 = _bncsve2_dquick_two_sum(pg, g3, g5, &g6e);
    *r0 = g6;
    *r1 = g6e;
}

/* (*r0,*r1) := (a0,a1) * (b0,b1)  -- sloppy */
static inline void _bncsve2_rdd_mul_sloppy(
        svbool_t pg,
        svfloat64_t *r0, svfloat64_t *r1,
        svfloat64_t a0, svfloat64_t a1,
        svfloat64_t b0, svfloat64_t b1)
{
    svfloat64_t p2;
    svfloat64_t p1 = _bncsve2_dtwo_prod(pg, a0, b0, &p2);
    /* p2 += a0*b1 + a1*b0   (svmla = acc + a*b) */
    p2 = svmla_f64_x(pg, svmla_f64_x(pg, p2, a0, b1), a1, b0);
    *r0 = _bncsve2_dquick_two_sum(pg, p1, p2, r1);
}

/* (*r0,*r1) := (a0,a1) * (b0,b1)  -- branch-free gates */
static inline void _bncsve2_rdd_mul_bf(
        svbool_t pg,
        svfloat64_t *r0, svfloat64_t *r1,
        svfloat64_t a0, svfloat64_t a1,
        svfloat64_t b0, svfloat64_t b1)
{
    svfloat64_t pe00;
    svfloat64_t p00 = _bncsve2_dtwo_prod(pg, a0, b0, &pe00);
    svfloat64_t p01 = svmul_f64_x(pg, a0, b1);
    svfloat64_t p10 = svmul_f64_x(pg, a1, b0);
    svfloat64_t g1  = svadd_f64_x(pg, p01, p10);
    svfloat64_t g2  = svadd_f64_x(pg, pe00, g1);
    svfloat64_t ge3;
    svfloat64_t g3  = _bncsve2_dquick_two_sum(pg, p00, g2, &ge3);
    *r0 = g3;
    *r1 = ge3;
}

/* Default-variant macros (match NEON header) */
#ifdef USE_DD_BF
    #define _bncsve2_rdd_add  _bncsve2_rdd_add_bf
    #define _bncsve2_rdd_mul  _bncsve2_rdd_mul_bf
#else
    #define _bncsve2_rdd_add  _bncsve2_rdd_add_sloppy
    #define _bncsve2_rdd_mul  _bncsve2_rdd_mul_sloppy
#endif

/* (*r0,*r1) := (a0,a1) * b  -- DD * D  (matches _bncneon_rdd_mul_d) */
static inline void _bncsve2_rdd_mul_d(
        svbool_t pg,
        svfloat64_t *r0, svfloat64_t *r1,
        svfloat64_t a0, svfloat64_t a1,
        svfloat64_t b)
{
    svfloat64_t p2;
    svfloat64_t p1 = _bncsve2_dtwo_prod(pg, a0, b, &p2);
    /* p2 += a1 * b */
    p2 = svmla_f64_x(pg, p2, a1, b);
    *r0 = _bncsve2_dquick_two_sum(pg, p1, p2, r1);
}

/* -------- horizontal EFT reductions (mirror _bncneon_rdd_*128d) -------- */
#ifndef _BNC_SVE_MAXLEN_D
#define _BNC_SVE_MAXLEN_D 32   /* SVE max vector = 2048 bit => 32 doubles */
#endif

/* (*r0,*r1) := |(a0,a1)|  (DD abs: negate both limbs when hi < 0) */
static inline void _bncsve2_rdd_abs(svbool_t pg,
                                    svfloat64_t *r0, svfloat64_t *r1,
                                    svfloat64_t a0, svfloat64_t a1)
{
    svbool_t neg = svcmplt_f64(pg, a0, svdup_n_f64(0.0));
    *r0 = svsel_f64(neg, svneg_f64_x(pg, a0), a0);
    *r1 = svsel_f64(neg, svneg_f64_x(pg, a1), a1);
}

/* ret := EFT sum over ALL lanes of the DD accumulator (a0=hi, a1=lo).
 * Inactive (tail) lanes must already be 0 — callers accumulate with an
 * all-true predicate over pg-zeroed loads, so tail lanes are exactly 0. */
static inline void _bncsve2_rdd_sum128d(svbool_t pg, double ret[DDSIZE],
                                        svfloat64_t a0, svfloat64_t a1)
{
    long int n = (long int)svcntd(), k;
    double b0[_BNC_SVE_MAXLEN_D], b1[_BNC_SVE_MAXLEN_D], v[DDSIZE];
    (void)pg;
    svst1_f64(svptrue_b64(), b0, a0);
    svst1_f64(svptrue_b64(), b1, a1);
    v[0] = b0[0]; v[1] = b1[0];
    rdd_set(ret, v);
    for(k = 1; k < n; k++){ v[0] = b0[k]; v[1] = b1[k]; rdd_add(ret, ret, v); }
}

/* ret := sum over lanes; lanes are already |.|-accumulated, so == sum128d. */
static inline void _bncsve2_rdd_abssum128d(svbool_t pg, double ret[DDSIZE],
                                           svfloat64_t a0, svfloat64_t a1)
{
    _bncsve2_rdd_sum128d(pg, ret, a0, a1);
}

#endif // defined(__ARM_SVE2)
#endif // __BNCSVE2_DD_H
