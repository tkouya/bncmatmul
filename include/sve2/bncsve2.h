/************************************************************/
/* bncsve2.h ... Vector arithmetic with ARM SVE2            */
/* 2025-08-05 (Tue) Tomonori Kouya                          */
/* Rewritten 2026 for VL-agnostic, per-limb-variable design.*/
/*                                                          */
/* SVE2 is the Armv9-A SIMD extension (a superset of SVE,   */
/* AArch64 only).  All operations are vector-length-agnostic*/
/* (svcntw() / svcntd() at runtime + predicate masks), so   */
/* the SAME compiled object runs on VL = 128, 256, 512, ... */
/* bits — i.e. you DO NOT pass -msve-vector-bits=N.         */
/*                                                          */
/* Per-limb-variable design: multi-precision N-limb values  */
/* are passed as N SEPARATE svfloatNN_t variables (inputs   */
/* by value, outputs by pointer).  Sizeless SVE types       */
/* cannot be array elements, struct members, sizeof'd, or   */
/* have static storage — this design respects that.         */
/************************************************************/
#ifndef __BNCSVE2_H__
#define __BNCSVE2_H__

#include <stdio.h>
#include <math.h>

#if defined(__ARM_SVE2)

#include <arm_sve.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
    #include <malloc.h>
#else
    #include <stdlib.h>
#endif

// scalar multi-precision types (shared with NEON)
#include "rdd.h"
#include "rds.h"
#if defined(USE_GMP) && defined(USE_MPFR)
#include "mpfr_dtq_sd.h"
#endif

/* ----------------------------------------------------------
 *  Include order matters:
 *    1. EFT primitives (two_sum / two_prod) — float and double
 *    2. Double-Double / Double-Single helpers (depend on EFT)
 *    3. Quad-Double / Quad-Single helpers (define renorm and
 *       host the renorm-based TD/TS helpers as well)
 *    4. Triple-Double / Triple-Single helpers (just size
 *       macros + forward declarations; bodies live in qd/qs)
 *    5. Double vector helpers (independent; pre-existing)
 * ---------------------------------------------------------- */
#include "_bncsve2_feft.h"
#include "_bncsve2_deft.h"

/* branch-free DW/TW/QW FMA (arXiv:2607.11391) */
#include "_bncsve2_fma.h"

#include "_bncsve2_dd.h"
#include "_bncsve2_ds.h"

#include "_bncsve2_qd.h"     /* must come before _bncsve2_td.h */
#include "_bncsve2_qs.h"     /* must come before _bncsve2_ts.h */

#include "_bncsve2_td.h"
#include "_bncsve2_ts.h"

#include "_bncsve2_d.h"
#include "_bncsve2_cdd.h"  // complex (added 2026-06-15)
#include "_bncsve2_ctd.h"
#include "_bncsve2_cqd.h"

#ifdef __cplusplus
} // extern "C"
#endif

#endif // defined(__ARM_SVE2)
#endif // __BNCSVE2_H__
