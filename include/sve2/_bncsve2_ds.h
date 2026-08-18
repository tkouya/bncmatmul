// -------------------------------------------------------
// _bncsve2_ds.h  --  SVE2 Double-Single (DS) per-limb-variable helpers
// -------------------------------------------------------
// Rewritten 2026: a DS value is passed as TWO separate svfloat32_t variables
// (hi, lo) instead of an svfloat32_t[2] array.  Output limbs are returned
// through svfloat32_t* pointers.
//
// Mirrors include/neon/_bncneon_ds.h (the f32 counterpart of dd).  Only the
// operations needed by the matmul kernels (set0, set_f, neg, add, mul) plus
// their _sloppy / _bf variants are ported.
// -------------------------------------------------------
#ifndef __BNCSVE2_DS_H
#define __BNCSVE2_DS_H

#if defined(__ARM_SVE2)
#include <arm_sve.h>

#ifndef DSSIZE
    #define DSSIZE 2
#endif

/* *r0 = 0, *r1 = 0 */
static inline void _bncsve2_rds_set0(svfloat32_t *r0, svfloat32_t *r1)
{
    *r0 = svdup_n_f32(0.0f);
    *r1 = svdup_n_f32(0.0f);
}

/* (*r0,*r1) := (val, 0) */
static inline void _bncsve2_rds_set_f(svfloat32_t *r0, svfloat32_t *r1,
                                       svfloat32_t v)
{
    *r0 = v;
    *r1 = svdup_n_f32(0.0f);
}

/* (*r0,*r1) := -(a0,a1) */
static inline void _bncsve2_rds_neg(svbool_t pg,
                                     svfloat32_t *r0, svfloat32_t *r1,
                                     svfloat32_t a0, svfloat32_t a1)
{
    *r0 = svneg_f32_x(pg, a0);
    *r1 = svneg_f32_x(pg, a1);
}

/* (*r0,*r1) := (a0,a1) + (b0,b1) -- sloppy */
static inline void _bncsve2_rds_add_sloppy(
        svbool_t pg,
        svfloat32_t *r0, svfloat32_t *r1,
        svfloat32_t a0, svfloat32_t a1,
        svfloat32_t b0, svfloat32_t b1)
{
    svfloat32_t e;
    svfloat32_t s = _bncsve2_ftwo_sum(pg, a0, b0, &e);
    e  = svadd_f32_x(pg, e, svadd_f32_x(pg, a1, b1));
    *r0 = _bncsve2_fquick_two_sum(pg, s, e, r1);
}

/* (*r0,*r1) := (a0,a1) + (b0,b1) -- branch-free */
static inline void _bncsve2_rds_add_bf(
        svbool_t pg,
        svfloat32_t *r0, svfloat32_t *r1,
        svfloat32_t a0, svfloat32_t a1,
        svfloat32_t b0, svfloat32_t b1)
{
    svfloat32_t g1e, g2e, g3e, g6e;
    svfloat32_t g1 = _bncsve2_ftwo_sum(pg, a0, b0, &g1e);
    svfloat32_t g2 = _bncsve2_ftwo_sum(pg, a1, b1, &g2e);
    svfloat32_t g3 = _bncsve2_fquick_two_sum(pg, g1, g2, &g3e);
    svfloat32_t g4 = svadd_f32_x(pg, g1e, g2e);
    svfloat32_t g5 = svadd_f32_x(pg, g4, g3e);
    svfloat32_t g6 = _bncsve2_fquick_two_sum(pg, g3, g5, &g6e);
    *r0 = g6;
    *r1 = g6e;
}

/* (*r0,*r1) := (a0,a1) * (b0,b1) -- sloppy */
static inline void _bncsve2_rds_mul_sloppy(
        svbool_t pg,
        svfloat32_t *r0, svfloat32_t *r1,
        svfloat32_t a0, svfloat32_t a1,
        svfloat32_t b0, svfloat32_t b1)
{
    svfloat32_t p2;
    svfloat32_t p1 = _bncsve2_ftwo_prod(pg, a0, b0, &p2);
    /* p2 += a0*b1 + a1*b0 */
    p2 = svmla_f32_x(pg, svmla_f32_x(pg, p2, a0, b1), a1, b0);
    *r0 = _bncsve2_fquick_two_sum(pg, p1, p2, r1);
}

/* (*r0,*r1) := (a0,a1) * (b0,b1) -- branch-free */
static inline void _bncsve2_rds_mul_bf(
        svbool_t pg,
        svfloat32_t *r0, svfloat32_t *r1,
        svfloat32_t a0, svfloat32_t a1,
        svfloat32_t b0, svfloat32_t b1)
{
    svfloat32_t pe00;
    svfloat32_t p00 = _bncsve2_ftwo_prod(pg, a0, b0, &pe00);
    svfloat32_t p01 = svmul_f32_x(pg, a0, b1);
    svfloat32_t p10 = svmul_f32_x(pg, a1, b0);
    svfloat32_t g1  = svadd_f32_x(pg, p01, p10);
    svfloat32_t g2  = svadd_f32_x(pg, pe00, g1);
    svfloat32_t ge3;
    svfloat32_t g3  = _bncsve2_fquick_two_sum(pg, p00, g2, &ge3);
    *r0 = g3;
    *r1 = ge3;
}

/* Default-variant macros (match NEON header). */
#ifdef USE_DS_BF
    #define _bncsve2_rds_add  _bncsve2_rds_add_bf
    #define _bncsve2_rds_mul  _bncsve2_rds_mul_bf
#else
    #define _bncsve2_rds_add  _bncsve2_rds_add_sloppy
    #define _bncsve2_rds_mul  _bncsve2_rds_mul_sloppy
#endif

/* -------- horizontal EFT reductions (mirror _bncneon_rds_*128) -------- */
#ifndef _BNC_SVE_MAXLEN_S
#define _BNC_SVE_MAXLEN_S 64   /* SVE max vector = 2048 bit => 64 floats */
#endif

static inline void _bncsve2_rds_abs(svbool_t pg,
                                    svfloat32_t *r0, svfloat32_t *r1,
                                    svfloat32_t a0, svfloat32_t a1)
{
    svbool_t neg = svcmplt_f32(pg, a0, svdup_n_f32(0.0f));
    *r0 = svsel_f32(neg, svneg_f32_x(pg, a0), a0);
    *r1 = svsel_f32(neg, svneg_f32_x(pg, a1), a1);
}

static inline void _bncsve2_rds_sum128(svbool_t pg, float ret[DSSIZE],
                                       svfloat32_t a0, svfloat32_t a1)
{
    long int n = (long int)svcntw(), k;
    float b0[_BNC_SVE_MAXLEN_S], b1[_BNC_SVE_MAXLEN_S], v[DSSIZE];
    (void)pg;
    svst1_f32(svptrue_b32(), b0, a0);
    svst1_f32(svptrue_b32(), b1, a1);
    v[0] = b0[0]; v[1] = b1[0];
    rds_set(ret, v);
    for(k = 1; k < n; k++){ v[0]=b0[k]; v[1]=b1[k]; rds_add(ret, ret, v); }
}

static inline void _bncsve2_rds_abssum128(svbool_t pg, float ret[DSSIZE],
                                          svfloat32_t a0, svfloat32_t a1)
{
    _bncsve2_rds_sum128(pg, ret, a0, a1);
}

/* ret := sqrt( sum_lanes lane^2 )  (mirror _bncavx2_rds_norm256 / _bncneon_rds_norm128f) */
static inline void _bncsve2_rds_norm128(svbool_t pg, float ret[DSSIZE],
                                        svfloat32_t a0, svfloat32_t a1)
{
    long int n = (long int)svcntw(), k;
    float b0[_BNC_SVE_MAXLEN_S], b1[_BNC_SVE_MAXLEN_S], v[DSSIZE], t[DSSIZE];
    (void)pg;
    svst1_f32(svptrue_b32(), b0, a0);
    svst1_f32(svptrue_b32(), b1, a1);
    v[0]=b0[0]; v[1]=b1[0]; rds_mul(t, v, v); rds_set(ret, t);
    for(k = 1; k < n; k++){ v[0]=b0[k]; v[1]=b1[k]; rds_mul(t, v, v); rds_add(ret, ret, t); }
    rds_sqrt(t, ret); rds_set(ret, t);
}

#endif // defined(__ARM_SVE2)
#endif // __BNCSVE2_DS_H
