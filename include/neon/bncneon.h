/************************************************************/
/* BNCneon.h ... Vector arithmetic with NEON                */
/* 2025-08-05 (Tue) Tomonori Kouya                          */
/************************************************************/
#ifndef __BNCNEON_H__
#define __BNCNEON_H__

#include <stdio.h>
#include <math.h>

// Helper functions for double-double arithmetic with NEON
// Note: ARM NEON doesn't have native double arithmetic in older versions
// We'll use float64x2_t for 2-element double vectors

/* nvcc on AArch64 + GCC 13 cannot parse <arm_neon.h> (cudafe++ does
 * not recognize the GCC builtins).  NEON SIMD code is host-only, so
 * skip the entire NEON block under __CUDACC__. */
#if defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) && !defined(__CUDACC__) //|| defined(__aarch64__)

#include <arm_neon.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// get_secv()
//#include "get_secv.h"

#if defined (_MSC_VER) || defined(__MINGW32__)
    #include <malloc.h> // _aligned_malloc, _aligned_free
#else //  defined (_MSC_VER) || defined(__MINGW32__)
    #include <stdlib.h> // aligned_alloc, free
#endif //  defined (_MSC_VER) || defined(__MINGW32__)

// rd[sd].h
#include "rdd.h"
#include "rds.h"
// mpfr_dtq_sd.h
#if defined(USE_GMP) && defined(USE_MPFR)
#include "mpfr_dtq_sd.h" // + mpfr_dd.
#endif //defined(USE_GMP) && defined(USE_MPFR)

//#ifndef __BNC_DDLINEAR_H__
//#include "dlinear.h"
//#include "ddlinear.h"
//#endif // __BNC_DDLINEAR_H__

// ------------------------
// -------- Float ---------
// ------------------------
// NOTE: _bncneon_f.h is a STALE, incomplete duplicate of the DS NEON header
// (it is labelled "DS" internally and shares the guard __BNCNEON_DS_H).  When
// included here it shadowed the complete "_bncneon_ds.h" below, dropping the
// reductions _bncneon_rds_{sum128f,abssum128f,norm128f,abs} that {ds}linear.c
// needs -> undefined references in the NEON build.  Disabled so the complete
// _bncneon_ds.h (included in the DS section below) takes effect.  The default
// DS add/mul (sloppy) are byte-identical to f.h's, so numerics are unchanged.
//#include "_bncneon_f.h"

// ------------------------
// -------- Double --------
// ------------------------
#include "_bncneon_d.h"

// ------------------------
// -- Error Free Trans. ---
// ------------------------
#include "_bncneon_feft.h"
#include "_bncneon_deft.h"

// ------------------------
// Branch-free DW/TW/QW FMA (arXiv:2607.11391)
// ------------------------
#include "_bncneon_fma.h"

// ------------------------
// ---------- DD ----------
// ------------------------
#include "_bncneon_dd.h"
#include "_bncneon_cdd.h" // 2024-01-26(Fri)

// ------------------------
// ---------- TD ----------
// ------------------------
#include "_bncneon_td.h"
#include "_bncneon_ctd.h" // 2024-01-26(Fri)

// ------------------------
// ---------- QD ----------
// ------------------------
#include "_bncneon_qd.h"
#include "_bncneon_cqd.h" // 2024-01-26(Fri)

// ------------------------
// ---------- DS ----------
// ------------------------
#include "_bncneon_ds.h"

// ------------------------
// ---------- TS ----------
// ------------------------
#include "_bncneon_ts.h"

// ------------------------
// ---------- QS ----------
// ------------------------
#include "_bncneon_qs.h"

#ifdef __cplusplus
} // extern "C" {
#endif // __cpluplus

#endif // defined(__ARM_NEON) && !defined(__CUDACC__)

#endif // __BNCNEON_H__