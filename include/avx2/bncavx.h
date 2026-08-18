/************************************************************/
/* BNCavx.h ... Vector arithmetic with AVX, AVX2 or AVX-512 */
/* 2020-02-29 (SAT) Tomonori Kouya                          */
/* gcc -O3 -mavx2 -mfma bncavx.c -L./ -lbnc rdd.o mpfr_dd.o -lmpfr -lgmp -lm */
/************************************************************/
#ifndef __BNCAVX_H__
#define __BNCAVX_H__

#include <stdio.h>
#include <math.h>

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

// for copy & paste
#if defined(__AVX2__) // __AVX2__
#elif defined(__AVX512F__) // __AVX512F__
#endif // __AVX2__

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
#include "_bncavx_f.h"

// ------------------------
// -------- Double --------
// ------------------------
#include "_bncavx_d.h"

// ------------------------
// -- Error Free Trans. ---
// ------------------------
#include "_bncavx_feft.h"
#include "_bncavx_feft_avx512.h"
#include "_bncavx_deft.h"
#include "_bncavx_deft_avx512.h"

// ------------------------
// Branch-free DW/TW/QW FMA (arXiv:2607.11391)
// ------------------------
#include "_bncavx_fma.h"

// ------------------------
// ---------- DD ----------
// ------------------------
#include "_bncavx_dd.h"
#include "_bncavx_dd_avx512.h"
#include "_bncavx_cdd.h" // 2024-01-26(Fri)
#include "_bncavx_cdd_avx512.h" // 2025-01-21(Wed)

// ------------------------
// ---------- TD ----------
// ------------------------
#include "_bncavx_td.h"
#include "_bncavx_td_avx512.h"
#include "_bncavx_ctd.h" // 2024-01-26(Fri)
#include "_bncavx_ctd_avx512.h" // 2025-01-21(Wed)

// ------------------------
// ---------- QD ----------
// ------------------------
#include "_bncavx_qd.h"
#include "_bncavx_qd_avx512.h"
#include "_bncavx_cqd.h" // 2024-01-26(Fri)
#include "_bncavx_cqd_avx512.h" // 2025-01-21(Wed)

// ------------------------
// ---------- DS ----------
// ------------------------
#include "_bncavx_ds.h"
#include "_bncavx_ds_avx512.h"

// ------------------------
// ---------- TS ----------
// ------------------------
#include "_bncavx_ts.h"
#include "_bncavx_ts_avx512.h"

// ------------------------
// ---------- QS ----------
// ------------------------
#include "_bncavx_qs.h"
#include "_bncavx_qs_avx512.h"

#ifdef __cplusplus
} // extern "C" {
#endif // __cpluplus


#endif // __BNCAVX_H__