/*
 * fma_blas_bench.c : AXPY / GEMV / GEMM benchmark of the proposed branch-free
 *                    DW/TW/QW FMA (arXiv:2607.11391) against the existing
 *                    multiplication+addition variants of BNCmatmul.
 *
 * Types    : DD (K=2), TD (K=3), QD (K=4)
 * Variants : Q   = library default mul + add   (DD sloppy / TD sloppy / QD Bailey)
 *            BF  = branch-free mul_bf + add_bf (Zhang-Aiken; the fair baseline)
 *            FMA = proposed branch-free fused FMA (17 / 66 / 146 flops)
 * Backends : scalar (default) | -DUSE_NEON | -DUSE_SVE2
 *
 * Reports, per (type, kernel, variant):
 *   max / mean relative error against MPFR 600-bit,
 *   time [s/call] with 1 thread and with OMP_NUM_THREADS threads,
 *   speedups FMA/Q and FMA/BF,
 *   and a bitwise check that the OpenMP run reproduces the 1-thread run.
 *
 * build: see build_fma.sh
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpfr.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "bnc_common.h"   /* maps __ARM_FEATURE_SVE2 -> __ARM_SVE2 */
#include "rdd.h"          /* ddfloat/tdfloat/qdfloat, DDSIZE..., c_dd_* */
#include "bncfma.h"

#ifdef USE_FLOAT_BASE
typedef float  BASE;          /* single-word base: DS / TS / QS */
# define WORD_BITS 24
#else
typedef double BASE;          /* double-word base: DD / TD / QD */
# define WORD_BITS 53
#endif

/* ==========================================================================
 *  Backend selection
 * ========================================================================== */
#if defined(USE_SVE2)
# if !defined(__ARM_SVE2)
#  error "USE_SVE2 requires an SVE2-capable -mcpu/-march"
# endif
# include <arm_sve.h>
# include "sve2/bncsve2.h"
# ifndef SVE_BITS
#  define SVE_BITS 128
# endif
# ifdef USE_FLOAT_BASE
typedef svfloat32_t VEC __attribute__((arm_sve_vector_bits(SVE_BITS)));
#  define LANES (SVE_BITS / 32)
#  define PG    svptrue_b32()
#  define VLOAD(d, p)   (d) = svld1_f32(PG, (p))
#  define VSTORE(p, s)  svst1_f32(PG, (p), (s))
#  define VSPLAT(d, s)  (d) = svdup_n_f32(s)
#  define VZERO(d)      (d) = svdup_n_f32(0.0f)
# else
typedef svfloat64_t VEC __attribute__((arm_sve_vector_bits(SVE_BITS)));
#  define LANES (SVE_BITS / 64)
#  define PG    svptrue_b64()
#  define VLOAD(d, p)   (d) = svld1_f64(PG, (p))
#  define VSTORE(p, s)  svst1_f64(PG, (p), (s))
#  define VSPLAT(d, s)  (d) = svdup_n_f64(s)
#  define VZERO(d)      (d) = svdup_n_f64(0.0)
# endif
# define BACKEND_NAME  "sve2"


#elif defined(USE_NEON)
# if !defined(__ARM_NEON)
#  error "USE_NEON requires NEON"
# endif
# include <arm_neon.h>
# include "neon/bncneon.h"
# ifdef USE_FLOAT_BASE
typedef float32x4_t VEC;
#  define LANES 4
#  define VLOAD(d, p)   (d) = vld1q_f32(p)
#  define VSTORE(p, s)  vst1q_f32((p), (s))
#  define VSPLAT(d, s)  (d) = vdupq_n_f32(s)
#  define VZERO(d)      (d) = vdupq_n_f32(0.0f)
# else
typedef float64x2_t VEC;
#  define LANES 2
#  define VLOAD(d, p)   (d) = vld1q_f64(p)
#  define VSTORE(p, s)  vst1q_f64((p), (s))
#  define VSPLAT(d, s)  (d) = vdupq_n_f64(s)
#  define VZERO(d)      (d) = vdupq_n_f64(0.0)
# endif
# define BACKEND_NAME  "neon"

#else /* scalar */
typedef BASE VEC;
# define LANES 1
# define VLOAD(d, p)   (d) = *(p)
# define VSTORE(p, s)  *(p) = (s)
# define VSPLAT(d, s)  (d) = (s)
# define VZERO(d)      (d) = 0
# define BACKEND_NAME  "serial"
#endif

/* ==========================================================================
 *  MAC macros: r[K] := a[K] * b[K] + c[K]
 * ========================================================================== */
#if defined(USE_SVE2)

# define MAC_DD_Q(r, a, b, c)  do { svfloat64_t _p0,_p1,_s0,_s1; \
    _bncsve2_rdd_mul_sloppy(PG, &_p0,&_p1, (a)[0],(a)[1], (b)[0],(b)[1]); \
    _bncsve2_rdd_add_sloppy(PG, &_s0,&_s1, _p0,_p1, (c)[0],(c)[1]); \
    (r)[0]=_s0; (r)[1]=_s1; } while (0)
# define MAC_DD_BF(r, a, b, c) do { svfloat64_t _p0,_p1,_s0,_s1; \
    _bncsve2_rdd_mul_bf(PG, &_p0,&_p1, (a)[0],(a)[1], (b)[0],(b)[1]); \
    _bncsve2_rdd_add_bf(PG, &_s0,&_s1, _p0,_p1, (c)[0],(c)[1]); \
    (r)[0]=_s0; (r)[1]=_s1; } while (0)
# define MAC_DD_FMA(r, a, b, c) do { svfloat64_t _z0,_z1; \
    _bncsve2_dwfma(PG, &_z0,&_z1, (a)[0],(a)[1], (b)[0],(b)[1], (c)[0],(c)[1]); \
    (r)[0]=_z0; (r)[1]=_z1; } while (0)

# define MAC_TD_Q(r, a, b, c)  do { svfloat64_t _p0,_p1,_p2,_s0,_s1,_s2; \
    _bncsve2_rtd_mulq(PG, &_p0,&_p1,&_p2, (a)[0],(a)[1],(a)[2], (b)[0],(b)[1],(b)[2]); \
    _bncsve2_rtd_addq(PG, &_s0,&_s1,&_s2, _p0,_p1,_p2, (c)[0],(c)[1],(c)[2]); \
    (r)[0]=_s0; (r)[1]=_s1; (r)[2]=_s2; } while (0)
# define MAC_TD_BF(r, a, b, c) do { svfloat64_t _p0,_p1,_p2,_s0,_s1,_s2; \
    _bncsve2_rtd_mul_bf(PG, &_p0,&_p1,&_p2, (a)[0],(a)[1],(a)[2], (b)[0],(b)[1],(b)[2]); \
    _bncsve2_rtd_add_bf(PG, &_s0,&_s1,&_s2, _p0,_p1,_p2, (c)[0],(c)[1],(c)[2]); \
    (r)[0]=_s0; (r)[1]=_s1; (r)[2]=_s2; } while (0)
# define MAC_TD_FMA(r, a, b, c) do { svfloat64_t _z0,_z1,_z2; \
    _bncsve2_twfma(PG, &_z0,&_z1,&_z2, (a)[0],(a)[1],(a)[2], (b)[0],(b)[1],(b)[2], \
                   (c)[0],(c)[1],(c)[2]); \
    (r)[0]=_z0; (r)[1]=_z1; (r)[2]=_z2; } while (0)

# define MAC_QD_Q(r, a, b, c)  do { svfloat64_t _p0,_p1,_p2,_p3,_s0,_s1,_s2,_s3; \
    _bncsve2_rqd_mul_sloppy(PG, &_p0,&_p1,&_p2,&_p3, (a)[0],(a)[1],(a)[2],(a)[3], \
                            (b)[0],(b)[1],(b)[2],(b)[3]); \
    _bncsve2_rqd_add_sloppy(PG, &_s0,&_s1,&_s2,&_s3, _p0,_p1,_p2,_p3, \
                            (c)[0],(c)[1],(c)[2],(c)[3]); \
    (r)[0]=_s0; (r)[1]=_s1; (r)[2]=_s2; (r)[3]=_s3; } while (0)
# define MAC_QD_BF(r, a, b, c) do { svfloat64_t _p0,_p1,_p2,_p3,_s0,_s1,_s2,_s3; \
    _bncsve2_rqd_mul_bf(PG, &_p0,&_p1,&_p2,&_p3, (a)[0],(a)[1],(a)[2],(a)[3], \
                        (b)[0],(b)[1],(b)[2],(b)[3]); \
    _bncsve2_rqd_add_bf(PG, &_s0,&_s1,&_s2,&_s3, _p0,_p1,_p2,_p3, \
                        (c)[0],(c)[1],(c)[2],(c)[3]); \
    (r)[0]=_s0; (r)[1]=_s1; (r)[2]=_s2; (r)[3]=_s3; } while (0)
# define MAC_QD_FMA(r, a, b, c) do { svfloat64_t _z0,_z1,_z2,_z3; \
    _bncsve2_qwfma(PG, &_z0,&_z1,&_z2,&_z3, (a)[0],(a)[1],(a)[2],(a)[3], \
                   (b)[0],(b)[1],(b)[2],(b)[3], (c)[0],(c)[1],(c)[2],(c)[3]); \
    (r)[0]=_z0; (r)[1]=_z1; (r)[2]=_z2; (r)[3]=_z3; } while (0)

# ifdef USE_FLOAT_BASE
#  undef MAC_DD_Q
#  undef MAC_DD_BF
#  undef MAC_DD_FMA
#  undef MAC_TD_Q
#  undef MAC_TD_BF
#  undef MAC_TD_FMA
#  undef MAC_QD_Q
#  undef MAC_QD_BF
#  undef MAC_QD_FMA
#  define MAC_DD_Q(r, a, b, c)  do { svfloat32_t _p0,_p1,_s0,_s1; \
     _bncsve2_rds_mul_sloppy(PG, &_p0,&_p1, (a)[0],(a)[1], (b)[0],(b)[1]); \
     _bncsve2_rds_add_sloppy(PG, &_s0,&_s1, _p0,_p1, (c)[0],(c)[1]); \
     (r)[0]=_s0; (r)[1]=_s1; } while (0)
#  define MAC_DD_BF(r, a, b, c) do { svfloat32_t _p0,_p1,_s0,_s1; \
     _bncsve2_rds_mul_bf(PG, &_p0,&_p1, (a)[0],(a)[1], (b)[0],(b)[1]); \
     _bncsve2_rds_add_bf(PG, &_s0,&_s1, _p0,_p1, (c)[0],(c)[1]); \
     (r)[0]=_s0; (r)[1]=_s1; } while (0)
#  define MAC_DD_FMA(r, a, b, c) do { svfloat32_t _z0,_z1; \
     _bncsve2_dwfmaf(PG, &_z0,&_z1, (a)[0],(a)[1], (b)[0],(b)[1], (c)[0],(c)[1]); \
     (r)[0]=_z0; (r)[1]=_z1; } while (0)
#  define MAC_TD_Q(r, a, b, c)  do { svfloat32_t _p0,_p1,_p2,_s0,_s1,_s2; \
     _bncsve2_rts_mulq(PG, &_p0,&_p1,&_p2, (a)[0],(a)[1],(a)[2], (b)[0],(b)[1],(b)[2]); \
     _bncsve2_rts_addq(PG, &_s0,&_s1,&_s2, _p0,_p1,_p2, (c)[0],(c)[1],(c)[2]); \
     (r)[0]=_s0; (r)[1]=_s1; (r)[2]=_s2; } while (0)
#  define MAC_TD_BF(r, a, b, c) do { svfloat32_t _p0,_p1,_p2,_s0,_s1,_s2; \
     _bncsve2_rts_mul_bf(PG, &_p0,&_p1,&_p2, (a)[0],(a)[1],(a)[2], (b)[0],(b)[1],(b)[2]); \
     _bncsve2_rts_add_bf(PG, &_s0,&_s1,&_s2, _p0,_p1,_p2, (c)[0],(c)[1],(c)[2]); \
     (r)[0]=_s0; (r)[1]=_s1; (r)[2]=_s2; } while (0)
#  define MAC_TD_FMA(r, a, b, c) do { svfloat32_t _z0,_z1,_z2; \
     _bncsve2_twfmaf(PG, &_z0,&_z1,&_z2, (a)[0],(a)[1],(a)[2], (b)[0],(b)[1],(b)[2], \
                     (c)[0],(c)[1],(c)[2]); \
     (r)[0]=_z0; (r)[1]=_z1; (r)[2]=_z2; } while (0)
#  define MAC_QD_Q(r, a, b, c)  do { svfloat32_t _p0,_p1,_p2,_p3,_s0,_s1,_s2,_s3; \
     _bncsve2_rqs_mul_sloppy(PG, &_p0,&_p1,&_p2,&_p3, (a)[0],(a)[1],(a)[2],(a)[3], \
                             (b)[0],(b)[1],(b)[2],(b)[3]); \
     _bncsve2_rqs_add_sloppy(PG, &_s0,&_s1,&_s2,&_s3, _p0,_p1,_p2,_p3, \
                             (c)[0],(c)[1],(c)[2],(c)[3]); \
     (r)[0]=_s0; (r)[1]=_s1; (r)[2]=_s2; (r)[3]=_s3; } while (0)
#  define MAC_QD_BF(r, a, b, c) do { svfloat32_t _p0,_p1,_p2,_p3,_s0,_s1,_s2,_s3; \
     _bncsve2_rqs_mul_bf(PG, &_p0,&_p1,&_p2,&_p3, (a)[0],(a)[1],(a)[2],(a)[3], \
                         (b)[0],(b)[1],(b)[2],(b)[3]); \
     _bncsve2_rqs_add_bf(PG, &_s0,&_s1,&_s2,&_s3, _p0,_p1,_p2,_p3, \
                         (c)[0],(c)[1],(c)[2],(c)[3]); \
     (r)[0]=_s0; (r)[1]=_s1; (r)[2]=_s2; (r)[3]=_s3; } while (0)
#  define MAC_QD_FMA(r, a, b, c) do { svfloat32_t _z0,_z1,_z2,_z3; \
     _bncsve2_qwfmaf(PG, &_z0,&_z1,&_z2,&_z3, (a)[0],(a)[1],(a)[2],(a)[3], \
                     (b)[0],(b)[1],(b)[2],(b)[3], (c)[0],(c)[1],(c)[2],(c)[3]); \
     (r)[0]=_z0; (r)[1]=_z1; (r)[2]=_z2; (r)[3]=_z3; } while (0)
# endif /* USE_FLOAT_BASE */

#elif defined(USE_NEON)

# define MAC_DD_Q(r, a, b, c)  do { float64x2_t _t[2]; \
    _bncneon_rdd_mul_sloppy(_t, (float64x2_t *)(a), (float64x2_t *)(b)); \
    _bncneon_rdd_add_sloppy((r), _t, (float64x2_t *)(c)); } while (0)
# define MAC_DD_BF(r, a, b, c) do { float64x2_t _t[2]; \
    _bncneon_rdd_mul_bf(_t, (a), (b)); \
    _bncneon_rdd_add_bf((r), _t, (c)); } while (0)
# define MAC_DD_FMA(r, a, b, c) _bncneon_dwfma((r), (a), (b), (c))

# define MAC_TD_Q(r, a, b, c)  do { float64x2_t _t[3]; \
    _bncneon_rtd_mulq(_t, (float64x2_t *)(a), (float64x2_t *)(b)); \
    _bncneon_rtd_addq((r), _t, (float64x2_t *)(c)); } while (0)
# define MAC_TD_BF(r, a, b, c) do { float64x2_t _t[3]; \
    _bncneon_rtd_mul_bf(_t, (a), (b)); \
    _bncneon_rtd_add_bf((r), _t, (c)); } while (0)
# define MAC_TD_FMA(r, a, b, c) _bncneon_twfma((r), (a), (b), (c))

# define MAC_QD_Q(r, a, b, c)  do { float64x2_t _t[4]; \
    _bncneon_rqd_mul_sloppy(_t, (float64x2_t *)(a), (float64x2_t *)(b)); \
    _bncneon_rqd_add_sloppy((r), _t, (float64x2_t *)(c)); } while (0)
# define MAC_QD_BF(r, a, b, c) do { float64x2_t _t[4]; \
    _bncneon_rqd_mul_bf(_t, (a), (b)); \
    _bncneon_rqd_add_bf((r), _t, (c)); } while (0)
# define MAC_QD_FMA(r, a, b, c) _bncneon_qwfma((r), (a), (b), (c))

# ifdef USE_FLOAT_BASE
#  undef MAC_DD_Q
#  undef MAC_DD_BF
#  undef MAC_DD_FMA
#  undef MAC_TD_Q
#  undef MAC_TD_BF
#  undef MAC_TD_FMA
#  undef MAC_QD_Q
#  undef MAC_QD_BF
#  undef MAC_QD_FMA
#  define MAC_DD_Q(r, a, b, c)  do { float32x4_t _t[2]; \
     _bncneon_rds_mul_sloppy(_t, (float32x4_t *)(a), (float32x4_t *)(b)); \
     _bncneon_rds_add_sloppy((r), _t, (float32x4_t *)(c)); } while (0)
#  define MAC_DD_BF(r, a, b, c) do { float32x4_t _t[2]; \
     _bncneon_rds_mul_bf(_t, (a), (b)); \
     _bncneon_rds_add_bf((r), _t, (c)); } while (0)
#  define MAC_DD_FMA(r, a, b, c) _bncneon_dwfmaf((r), (a), (b), (c))
#  define MAC_TD_Q(r, a, b, c)  do { float32x4_t _t[3]; \
     _bncneon_rts_mulq(_t, (float32x4_t *)(a), (float32x4_t *)(b)); \
     _bncneon_rts_addq((r), _t, (float32x4_t *)(c)); } while (0)
#  define MAC_TD_BF(r, a, b, c) do { float32x4_t _t[3]; \
     _bncneon_rts_mul_bf(_t, (a), (b)); \
     _bncneon_rts_add_bf((r), _t, (c)); } while (0)
#  define MAC_TD_FMA(r, a, b, c) _bncneon_twfmaf((r), (a), (b), (c))
#  define MAC_QD_Q(r, a, b, c)  do { float32x4_t _t[4]; \
     _bncneon_rqs_mul_sloppy(_t, (float32x4_t *)(a), (float32x4_t *)(b)); \
     _bncneon_rqs_add_sloppy((r), _t, (float32x4_t *)(c)); } while (0)
#  define MAC_QD_BF(r, a, b, c) do { float32x4_t _t[4]; \
     _bncneon_rqs_mul_bf(_t, (a), (b)); \
     _bncneon_rqs_add_bf((r), _t, (c)); } while (0)
#  define MAC_QD_FMA(r, a, b, c) _bncneon_qwfmaf((r), (a), (b), (c))
# endif /* USE_FLOAT_BASE */

#else /* scalar */

# define MAC_DD_Q(r, a, b, c)  do { double _t[2]; \
    c_dd_mul((a), (b), _t); c_dd_add_sloppy(_t, (c), (r)); } while (0)
# define MAC_DD_BF(r, a, b, c) do { double _t[2]; \
    c_dd_mul_bf((a), (b), _t); c_dd_add_bf(_t, (c), (r)); } while (0)
# define MAC_DD_FMA(r, a, b, c) bnc_dwfma((r), (a), (b), (c))

# define MAC_TD_Q(r, a, b, c)  do { double _t[3]; \
    c_td_mul_sloppy((double *)(a), (double *)(b), _t); \
    c_td_add(_t, (double *)(c), (r)); } while (0)
# define MAC_TD_BF(r, a, b, c) do { double _t[3]; \
    c_td_mul_bf((double *)(a), (double *)(b), _t); \
    c_td_add_bf(_t, (double *)(c), (r)); } while (0)
# define MAC_TD_FMA(r, a, b, c) bnc_twfma((r), (a), (b), (c))

# define MAC_QD_Q(r, a, b, c)  do { double _t[4]; \
    c_qd_mul((a), (b), _t); c_qd_add(_t, (c), (r)); } while (0)
# define MAC_QD_BF(r, a, b, c) do { double _t[4]; \
    c_qd_mul_bf((a), (b), _t); c_qd_add_bf(_t, (c), (r)); } while (0)
# define MAC_QD_FMA(r, a, b, c) bnc_qwfma((r), (a), (b), (c))

# ifdef USE_FLOAT_BASE
#  undef MAC_DD_Q
#  undef MAC_DD_BF
#  undef MAC_DD_FMA
#  undef MAC_TD_Q
#  undef MAC_TD_BF
#  undef MAC_TD_FMA
#  undef MAC_QD_Q
#  undef MAC_QD_BF
#  undef MAC_QD_FMA
/* NOTE: c_ds_qs.h has no scalar branch-free (_bf) variants, so on the scalar
   backend BF falls back to the same code as Q and is reported as "n/a". */
#  define MAC_DD_Q(r, a, b, c)  do { float _t[2]; \
     c_ds_mul((a), (b), _t); c_ds_add(_t, (c), (r)); } while (0)
#  define MAC_DD_BF  MAC_DD_Q
#  define MAC_DD_FMA(r, a, b, c) bnc_dwfmaf((r), (a), (b), (c))
#  define MAC_TD_Q(r, a, b, c)  do { float _t[3]; \
     c_ts_mul_sloppy((float *)(a), (float *)(b), _t); \
     c_ts_add(_t, (float *)(c), (r)); } while (0)
#  define MAC_TD_BF  MAC_TD_Q
#  define MAC_TD_FMA(r, a, b, c) bnc_twfmaf((r), (a), (b), (c))
#  define MAC_QD_Q(r, a, b, c)  do { float _t[4]; \
     c_qs_mul((a), (b), _t); c_qs_add(_t, (c), (r)); } while (0)
#  define MAC_QD_BF  MAC_QD_Q
#  define MAC_QD_FMA(r, a, b, c) bnc_qwfmaf((r), (a), (b), (c))
#  define BF_IS_NA 1
# endif /* USE_FLOAT_BASE */

#endif

/* ==========================================================================
 *  Kernel instantiation: 3 types x 3 variants
 * ========================================================================== */
#define TAG dd_q
#define K   2
#define MAC MAC_DD_Q
#include "fma_kernels.inc"

#define TAG dd_bf
#define K   2
#define MAC MAC_DD_BF
#include "fma_kernels.inc"

#define TAG dd_fma
#define K   2
#define MAC MAC_DD_FMA
#include "fma_kernels.inc"

#define TAG td_q
#define K   3
#define MAC MAC_TD_Q
#include "fma_kernels.inc"

#define TAG td_bf
#define K   3
#define MAC MAC_TD_BF
#include "fma_kernels.inc"

#define TAG td_fma
#define K   3
#define MAC MAC_TD_FMA
#include "fma_kernels.inc"

#define TAG qd_q
#define K   4
#define MAC MAC_QD_Q
#include "fma_kernels.inc"

#define TAG qd_bf
#define K   4
#define MAC MAC_QD_BF
#include "fma_kernels.inc"

#define TAG qd_fma
#define K   4
#define MAC MAC_QD_FMA
#include "fma_kernels.inc"

/* ==========================================================================
 *  Driver
 * ========================================================================== */
typedef void (*axpy_fn)(long, const VEC *, BASE *const *, BASE **);
typedef void (*gemv_fn)(long, BASE *const *, BASE *const *, BASE **);
typedef void (*gemm_fn)(long, BASE *const *, BASE *const *, BASE **);

struct variant {
	const char *name;
	axpy_fn axpy;
	gemv_fn gemv;
	gemm_fn gemm;
};
struct typeinfo {
	const char *name;
	int K;
	struct variant v[3];   /* Q, BF, FMA */
};

#ifdef USE_FLOAT_BASE
# define TN_DD "ds"
# define TN_TD "ts"
# define TN_QD "qs"
#else
# define TN_DD "dd"
# define TN_TD "td"
# define TN_QD "qd"
#endif

static struct typeinfo TYPES[3] = {
	{ TN_DD, 2, { { "Q",   axpy_dd_q,   gemv_dd_q,   gemm_dd_q   },
	             { "BF",  axpy_dd_bf,  gemv_dd_bf,  gemm_dd_bf  },
	             { "FMA", axpy_dd_fma, gemv_dd_fma, gemm_dd_fma } } },
	{ TN_TD, 3, { { "Q",   axpy_td_q,   gemv_td_q,   gemm_td_q   },
	             { "BF",  axpy_td_bf,  gemv_td_bf,  gemm_td_bf  },
	             { "FMA", axpy_td_fma, gemv_td_fma, gemm_td_fma } } },
	{ TN_QD, 4, { { "Q",   axpy_qd_q,   gemv_qd_q,   gemm_qd_q   },
	             { "BF",  axpy_qd_bf,  gemv_qd_bf,  gemm_qd_bf  },
	             { "FMA", axpy_qd_fma, gemv_qd_fma, gemm_qd_fma } } },
};

#define REF_PREC 600
#define MAXK 4

static double wtime(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void *xalloc(size_t n)
{
	void *p = aligned_alloc(64, ((n + 63) / 64) * 64);
	if (p == NULL) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(1); }
	memset(p, 0, ((n + 63) / 64) * 64);
	return p;
}

/* random non-overlapping K-word value */
static unsigned long rstate = 88172645463325252UL;
static double urand(void)
{
	rstate ^= rstate << 13; rstate ^= rstate >> 7; rstate ^= rstate << 17;
	return (double)(rstate >> 11) / 9007199254740992.0;   /* [0,1) */
}
static void rand_mw(BASE *z, int K)
{
	int i;
	z[0] = (BASE)(2.0 * urand() - 1.0);
	for (i = 1; i < K; i++) {
		z[i] = (BASE)ldexp((double)z[i - 1] * (0.5 + 0.5 * urand()), -WORD_BITS);
		if (urand() < 0.5) z[i] = -z[i];
	}
	for (i = 0; i + 1 < K; i++)
#ifdef USE_FLOAT_BASE
		z[i] = fquick_two_sum(z[i], z[i + 1], &z[i + 1]);
#else
		z[i] = quick_two_sum(z[i], z[i + 1], &z[i + 1]);
#endif
}

static void store_mw(BASE *v[MAXK], long idx, const BASE *z, int K)
{
	int k;
	for (k = 0; k < K; k++) v[k][idx] = z[k];
}
static void load_mw(BASE *z, BASE *const v[MAXK], long idx, int K)
{
	int k;
	for (k = 0; k < K; k++) z[k] = v[k][idx];
}
/* exact: every BASE value is exactly representable as a double */
static void mw_to_mpfr(mpfr_t r, const BASE *z, int K)
{
	int i;
	mpfr_set_d(r, z[0], MPFR_RNDN);
	for (i = 1; i < K; i++) mpfr_add_d(r, r, z[i], MPFR_RNDN);
}

/* accumulate relative error statistics of a K-word value against an mpfr ref */
struct acc { double maxrel, sumrel; long n; };
static void acc_init(struct acc *a) { a->maxrel = 0.0; a->sumrel = 0.0; a->n = 0; }
static void acc_add(struct acc *a, const BASE *z, int K, mpfr_t ref, mpfr_t t1, mpfr_t t2)
{
	double rel;
	mw_to_mpfr(t1, z, K);
	mpfr_sub(t1, t1, ref, MPFR_RNDN);
	mpfr_abs(t1, t1, MPFR_RNDN);
	mpfr_abs(t2, ref, MPFR_RNDN);
	if (mpfr_zero_p(t2)) return;
	mpfr_div(t1, t1, t2, MPFR_RNDN);
	rel = mpfr_get_d(t1, MPFR_RNDN);
	if (rel > a->maxrel) a->maxrel = rel;
	a->sumrel += rel; a->n++;
}

/* ---- result buffers for the bitwise OpenMP check --------------------------- */
static int bitcmp_buf(BASE *const a[MAXK], BASE *const b[MAXK], long n, int K)
{
	int k;
	for (k = 0; k < K; k++)
		if (memcmp(a[k], b[k], (size_t)n * sizeof(BASE)) != 0) return 1;
	return 0;
}

/* sampled GEMM entries whose reference is computed in MPFR */
static long *gemm_si, *gemm_sj;

#ifdef _OPENMP
# define SET_THREADS(n) omp_set_num_threads(n)
#else
# define SET_THREADS(n) ((void)(n))
#endif

#define REPORT(ty, op, var, ac, t1, tomp, ok)                                   \
	printf("%-4s %-5s %-4s %12.4e %12.4e %13.4e %13.4e %7.2f %6s\n",            \
	       (ty), (op), (var), (ac)->maxrel,                                     \
	       (ac)->sumrel / (double)((ac)->n ? (ac)->n : 1), (t1), (tomp),        \
	       (tomp) > 0.0 ? (t1) / (tomp) : 0.0, (ok) ? "ok" : "DIFF")

int main(int argc, char *argv[])
{
	long n_axpy = 1000000, n_gemv = 2048, n_gemm = 512;
	int reps_axpy = 10, reps_gemv = 3, reps_gemm = 1;
	long gemm_samples = 2048;
	int t, v, nthreads = 1;
	int i;

	for (i = 1; i < argc; i++) {
		if      (!strcmp(argv[i], "-naxpy") && i + 1 < argc) n_axpy = atol(argv[++i]);
		else if (!strcmp(argv[i], "-ngemv") && i + 1 < argc) n_gemv = atol(argv[++i]);
		else if (!strcmp(argv[i], "-ngemm") && i + 1 < argc) n_gemm = atol(argv[++i]);
		else if (!strcmp(argv[i], "-reps")  && i + 1 < argc) {
			reps_axpy = reps_gemv = reps_gemm = atoi(argv[++i]);
		} else {
			fprintf(stderr, "usage: %s [-naxpy N] [-ngemv N] [-ngemm N] [-reps R]\n", argv[0]);
			return 1;
		}
	}
	n_axpy = ((n_axpy + LANES - 1) / LANES) * LANES;
	n_gemv = ((n_gemv + LANES - 1) / LANES) * LANES;
	n_gemm = ((n_gemm + LANES - 1) / LANES) * LANES;
	if (n_gemm > n_gemv) { fprintf(stderr, "n_gemm must not exceed n_gemv\n"); return 1; }

#ifdef _OPENMP
	nthreads = omp_get_max_threads();
#endif

	printf("# backend=%s lanes=%d threads=%d\n", BACKEND_NAME, LANES, nthreads);
	printf("# AXPY n=%ld  GEMV n=%ld  GEMM n=%ld   reference: MPFR %d bits\n",
	       n_axpy, n_gemv, n_gemm, REF_PREC);
	printf("# t1 = 1 thread, tOMP = %d threads; omp column = bitwise equality of the two runs\n#\n",
	       nthreads);
	printf("%-4s %-5s %-4s %12s %12s %13s %13s %7s %6s\n",
	       "type", "op", "var", "maxrelerr", "meanrelerr", "t1[s/call]",
	       "tOMP[s/call]", "ompspd", "omp");

	gemm_si = (long *)malloc(sizeof(long) * (size_t)gemm_samples);
	gemm_sj = (long *)malloc(sizeof(long) * (size_t)gemm_samples);

	for (t = 0; t < 3; t++) {
		struct typeinfo *T = &TYPES[t];
		int K = T->K;
		BASE *X[MAXK], *Y0[MAXK], *Yw[MAXK], *Yomp[MAXK];
		BASE *A[MAXK], *B[MAXK], *C[MAXK], *Comp[MAXK];
		BASE alpha_s[MAXK];
		VEC alpha_v[MAXK];
		mpfr_t *ref_axpy, *ref_gemv, *ref_gemm;
		mpfr_t mt1, mt2, mprod;
		long ii, jj, kk;
		long nmax = n_axpy;
		double tt[3][3];   /* [variant][kernel] 1-thread time */
		double to[3][3];   /* [variant][kernel] OpenMP time   */
		int k;

		if (n_gemv > nmax) nmax = n_gemv;
		if (n_gemm > nmax) nmax = n_gemm;

		for (k = 0; k < K; k++) {
			X[k]    = xalloc((size_t)nmax * sizeof(BASE));
			Y0[k]   = xalloc((size_t)nmax * sizeof(BASE));
			Yw[k]   = xalloc((size_t)nmax * sizeof(BASE));
			Yomp[k] = xalloc((size_t)nmax * sizeof(BASE));
			A[k]    = xalloc((size_t)n_gemv * n_gemv * sizeof(BASE));
			B[k]    = xalloc((size_t)n_gemm * n_gemm * sizeof(BASE));
			C[k]    = xalloc((size_t)n_gemm * n_gemm * sizeof(BASE));
			Comp[k] = xalloc((size_t)n_gemm * n_gemm * sizeof(BASE));
		}
		mpfr_inits2(REF_PREC, mt1, mt2, mprod, (mpfr_ptr)0);

		/* ---------------- data ---------------- */
		rstate = 88172645463325252UL + (unsigned long)t * 1234567UL;
		rand_mw(alpha_s, K);
		for (k = 0; k < K; k++) VSPLAT(alpha_v[k], alpha_s[k]);
		for (ii = 0; ii < nmax; ii++) {
			BASE z[MAXK];
			rand_mw(z, K); store_mw(X, ii, z, K);
			rand_mw(z, K); store_mw(Y0, ii, z, K);
		}
		for (ii = 0; ii < (long)n_gemv * n_gemv; ii++) {
			BASE z[MAXK]; rand_mw(z, K); store_mw(A, ii, z, K);
		}
		for (ii = 0; ii < (long)n_gemm * n_gemm; ii++) {
			BASE z[MAXK]; rand_mw(z, K); store_mw(B, ii, z, K);
		}
		for (ii = 0; ii < gemm_samples; ii++) {
			gemm_si[ii] = (long)(urand() * (double)n_gemm);
			gemm_sj[ii] = (long)(urand() * (double)n_gemm);
			if (gemm_si[ii] >= n_gemm) gemm_si[ii] = n_gemm - 1;
			if (gemm_sj[ii] >= n_gemm) gemm_sj[ii] = n_gemm - 1;
		}

		/* ---------------- MPFR references ---------------- */
		fprintf(stderr, "[%s] computing MPFR references ...\n", T->name);
		ref_axpy = (mpfr_t *)malloc(sizeof(mpfr_t) * (size_t)n_axpy);
		ref_gemv = (mpfr_t *)malloc(sizeof(mpfr_t) * (size_t)n_gemv);
		ref_gemm = (mpfr_t *)malloc(sizeof(mpfr_t) * (size_t)gemm_samples);
		{
			mpfr_t ma, mx, my;
			BASE z[MAXK];
			mpfr_inits2(REF_PREC, ma, mx, my, (mpfr_ptr)0);
			mw_to_mpfr(ma, alpha_s, K);
			for (ii = 0; ii < n_axpy; ii++) {
				mpfr_init2(ref_axpy[ii], REF_PREC);
				load_mw(z, X, ii, K);  mw_to_mpfr(mx, z, K);
				load_mw(z, Y0, ii, K); mw_to_mpfr(my, z, K);
				mpfr_mul(ref_axpy[ii], ma, mx, MPFR_RNDN);
				mpfr_add(ref_axpy[ii], ref_axpy[ii], my, MPFR_RNDN);
			}
			for (ii = 0; ii < n_gemv; ii++) {
				mpfr_init2(ref_gemv[ii], REF_PREC);
				mpfr_set_zero(ref_gemv[ii], 1);
			}
			for (jj = 0; jj < n_gemv; jj++) {
				load_mw(z, X, jj, K); mw_to_mpfr(mx, z, K);
				for (ii = 0; ii < n_gemv; ii++) {
					load_mw(z, A, jj * n_gemv + ii, K); mw_to_mpfr(my, z, K);
					mpfr_mul(mprod, my, mx, MPFR_RNDN);
					mpfr_add(ref_gemv[ii], ref_gemv[ii], mprod, MPFR_RNDN);
				}
			}
			for (ii = 0; ii < gemm_samples; ii++) {
				mpfr_init2(ref_gemm[ii], REF_PREC);
				mpfr_set_zero(ref_gemm[ii], 1);
				for (kk = 0; kk < n_gemm; kk++) {
					load_mw(z, A, gemm_si[ii] * n_gemm + kk, K); mw_to_mpfr(mx, z, K);
					load_mw(z, B, kk * n_gemm + gemm_sj[ii], K); mw_to_mpfr(my, z, K);
					mpfr_mul(mprod, mx, my, MPFR_RNDN);
					mpfr_add(ref_gemm[ii], ref_gemm[ii], mprod, MPFR_RNDN);
				}
			}
			mpfr_clears(ma, mx, my, (mpfr_ptr)0);
		}

		/* ---------------- measure ---------------- */
		for (v = 0; v < 3; v++) {
			struct variant *V = &T->v[v];
			struct acc ac;
			double t0;
			int omp_ok;
			int r;
			BASE z[MAXK];

			/* ========== AXPY ========== */
			acc_init(&ac); omp_ok = 1;
			SET_THREADS(1);
			t0 = wtime();
			for (r = 0; r < reps_axpy; r++) {
				for (k = 0; k < K; k++) memcpy(Yw[k], Y0[k], (size_t)n_axpy * sizeof(BASE));
				V->axpy(n_axpy, alpha_v, X, Yw);
			}
			tt[v][0] = (wtime() - t0) / reps_axpy;
			for (ii = 0; ii < n_axpy; ii++) {
				load_mw(z, Yw, ii, K);
				acc_add(&ac, z, K, ref_axpy[ii], mt1, mt2);
			}
			SET_THREADS(nthreads);
			t0 = wtime();
			for (r = 0; r < reps_axpy; r++) {
				for (k = 0; k < K; k++) memcpy(Yomp[k], Y0[k], (size_t)n_axpy * sizeof(BASE));
				V->axpy(n_axpy, alpha_v, X, Yomp);
			}
			to[v][0] = (wtime() - t0) / reps_axpy;
			if (bitcmp_buf(Yw, Yomp, n_axpy, K)) omp_ok = 0;
			REPORT(T->name, "AXPY", V->name, &ac, tt[v][0], to[v][0], omp_ok);

			/* ========== GEMV ========== */
			acc_init(&ac); omp_ok = 1;
			SET_THREADS(1);
			t0 = wtime();
			for (r = 0; r < reps_gemv; r++) V->gemv(n_gemv, A, X, Yw);
			tt[v][1] = (wtime() - t0) / reps_gemv;
			for (ii = 0; ii < n_gemv; ii++) {
				load_mw(z, Yw, ii, K);
				acc_add(&ac, z, K, ref_gemv[ii], mt1, mt2);
			}
			SET_THREADS(nthreads);
			t0 = wtime();
			for (r = 0; r < reps_gemv; r++) V->gemv(n_gemv, A, X, Yomp);
			to[v][1] = (wtime() - t0) / reps_gemv;
			if (bitcmp_buf(Yw, Yomp, n_gemv, K)) omp_ok = 0;
			REPORT(T->name, "GEMV", V->name, &ac, tt[v][1], to[v][1], omp_ok);

			/* ========== GEMM ========== */
			acc_init(&ac); omp_ok = 1;
			SET_THREADS(1);
			t0 = wtime();
			for (r = 0; r < reps_gemm; r++) V->gemm(n_gemm, A, B, C);
			tt[v][2] = (wtime() - t0) / reps_gemm;
			for (ii = 0; ii < gemm_samples; ii++) {
				load_mw(z, C, gemm_si[ii] * n_gemm + gemm_sj[ii], K);
				acc_add(&ac, z, K, ref_gemm[ii], mt1, mt2);
			}
			SET_THREADS(nthreads);
			t0 = wtime();
			for (r = 0; r < reps_gemm; r++) V->gemm(n_gemm, A, B, Comp);
			to[v][2] = (wtime() - t0) / reps_gemm;
			if (bitcmp_buf(C, Comp, (long)n_gemm * n_gemm, K)) omp_ok = 0;
			REPORT(T->name, "GEMM", V->name, &ac, tt[v][2], to[v][2], omp_ok);
		}

		printf("# %s speedup of FMA  (1 thread) : AXPY FMA/Q=%.2f FMA/BF=%.2f |"
		       " GEMV FMA/Q=%.2f FMA/BF=%.2f | GEMM FMA/Q=%.2f FMA/BF=%.2f\n",
		       T->name,
		       tt[0][0] / tt[2][0], tt[1][0] / tt[2][0],
		       tt[0][1] / tt[2][1], tt[1][1] / tt[2][1],
		       tt[0][2] / tt[2][2], tt[1][2] / tt[2][2]);
		printf("# %s speedup of FMA  (%d thr)   : AXPY FMA/Q=%.2f FMA/BF=%.2f |"
		       " GEMV FMA/Q=%.2f FMA/BF=%.2f | GEMM FMA/Q=%.2f FMA/BF=%.2f\n",
		       T->name, nthreads,
		       to[0][0] / to[2][0], to[1][0] / to[2][0],
		       to[0][1] / to[2][1], to[1][1] / to[2][1],
		       to[0][2] / to[2][2], to[1][2] / to[2][2]);
		printf("#\n");

		for (ii = 0; ii < n_axpy; ii++) mpfr_clear(ref_axpy[ii]);
		for (ii = 0; ii < n_gemv; ii++) mpfr_clear(ref_gemv[ii]);
		for (ii = 0; ii < gemm_samples; ii++) mpfr_clear(ref_gemm[ii]);
		free(ref_axpy); free(ref_gemv); free(ref_gemm);
		mpfr_clears(mt1, mt2, mprod, (mpfr_ptr)0);
		for (k = 0; k < K; k++) {
			free(X[k]); free(Y0[k]); free(Yw[k]); free(Yomp[k]);
			free(A[k]); free(B[k]); free(C[k]); free(Comp[k]);
		}
	}
	free(gemm_si); free(gemm_sj);
	return 0;
}
