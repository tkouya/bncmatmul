/********************************************************************************/
/* rdtq_func.h : exported (non-inline) function versions of the DD/TD/QD and    */
/*               DS/TS/QS FMA and elementary-function interfaces                */
/*                                                                              */
/* These are real external symbols compiled into src/rdtq_func.c, intended for  */
/* shared-library (DLL) consumers such as Python/ctypes -- unlike the macro     */
/* and static-inline versions in rdd.h / rds.h, which exist only inside a       */
/* C translation unit.  All of them delegate to the dtq-0.0.3 ports             */
/* (bncfma_d.h / bncfma_f.h / bncelem*.h):                                      */
/*                                                                              */
/*   rXX_fma       certified branch-free FMA        (bnc_{dw,tw,qw}fma[f])      */
/*   rXX_fma_safe  div/sqrt-safe FMA, scalar b      (bnc_{dw,tw,qw}fma[f]_safe) */
/*   rXX_div_fma   FMA-driven long division         (bnc_XX_div_fma)            */
/*   rXX_exp ...   FMA-fused elementary functions   (bnc_XX_exp ...)            */
/*                                                                              */
/* Argument order follows the rdd.h convention: result first.                   */
/* The SIMD-vectorized array functions (bnc_XX_yy_array) are likewise real      */
/* symbols; see bncelem_vector.h.                                               */
/*                                                                              */
/* NOTE: do not include this header together with rdd.h / rds.h unless          */
/* USE_RDD_FUNCTIONS etc. are defined first -- otherwise the lowercase macro    */
/* aliases in those headers would mangle these prototypes.                      */
/*                                                                              */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/* This file is part of BNCmatmul and distributed under the GNU LGPL v3.        */
/********************************************************************************/
#ifndef __BNC_RDTQ_FUNC_H
#define __BNC_RDTQ_FUNC_H

#if (defined(__BNC_RDD_H_) && !defined(USE_RDD_FUNCTIONS)) || \
    (defined(__BNC_RDS_H_) && !defined(USE_RDS_FUNCTIONS))
#error "rdtq_func.h: define USE_RDD_FUNCTIONS / USE_RDS_FUNCTIONS (etc.) before including rdd.h / rds.h, or the macro aliases there will mangle these prototypes."
#endif

/* DLL export/import decoration (no-op on ELF platforms) */
#if defined(_WIN32) && defined(BNC_BUILD_DLL)
	#define BNC_API __declspec(dllexport)
#elif defined(_WIN32) && defined(BNC_USE_DLL)
	#define BNC_API __declspec(dllimport)
#else
	#define BNC_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- DD (double-double, double[2]) ---- */
#ifndef __BNC_RDD_H_	/* rdd.h defines static inline rdd_fma/rtd_fma/rqd_fma */
BNC_API void rdd_fma(double *ret, const double *a, const double *b, const double *c);
#endif
BNC_API void rdd_fma_safe(double *ret, const double *a, double b, const double *c);
BNC_API void rdd_div_fma(double *ret, const double *a, const double *b);
BNC_API void rdd_exp(double *ret, const double *x);
BNC_API void rdd_expm1(double *ret, const double *x);
BNC_API void rdd_log(double *ret, const double *x);
BNC_API void rdd_log10(double *ret, const double *x);
BNC_API void rdd_sin(double *ret, const double *x);
BNC_API void rdd_cos(double *ret, const double *x);
BNC_API void rdd_sincos(double *sin_ret, double *cos_ret, const double *x);
BNC_API void rdd_tan(double *ret, const double *x);
BNC_API void rdd_nint(double *ret, const double *x);

/* ---- TD (triple-double, double[3]) ---- */
#ifndef __BNC_RDD_H_
BNC_API void rtd_fma(double *ret, const double *a, const double *b, const double *c);
#endif
BNC_API void rtd_fma_safe(double *ret, const double *a, double b, const double *c);
BNC_API void rtd_div_fma(double *ret, const double *a, const double *b);
BNC_API void rtd_exp(double *ret, const double *x);
BNC_API void rtd_expm1(double *ret, const double *x);
BNC_API void rtd_log(double *ret, const double *x);
BNC_API void rtd_log10(double *ret, const double *x);
BNC_API void rtd_sin(double *ret, const double *x);
BNC_API void rtd_cos(double *ret, const double *x);
BNC_API void rtd_sincos(double *sin_ret, double *cos_ret, const double *x);
BNC_API void rtd_tan(double *ret, const double *x);
BNC_API void rtd_nint(double *ret, const double *x);

/* ---- QD (quadruple-double, double[4]) ---- */
#ifndef __BNC_RDD_H_
BNC_API void rqd_fma(double *ret, const double *a, const double *b, const double *c);
#endif
BNC_API void rqd_fma_safe(double *ret, const double *a, double b, const double *c);
BNC_API void rqd_div_fma(double *ret, const double *a, const double *b);
BNC_API void rqd_exp(double *ret, const double *x);
BNC_API void rqd_expm1(double *ret, const double *x);
BNC_API void rqd_log(double *ret, const double *x);
BNC_API void rqd_log10(double *ret, const double *x);
BNC_API void rqd_sin(double *ret, const double *x);
BNC_API void rqd_cos(double *ret, const double *x);
BNC_API void rqd_sincos(double *sin_ret, double *cos_ret, const double *x);
BNC_API void rqd_tan(double *ret, const double *x);
BNC_API void rqd_nint(double *ret, const double *x);

/* ---- DS (double-single, float[2]) ---- */
#ifndef __BNC_RDS_H_	/* rds.h defines static inline rds_fma/rts_fma/rqs_fma */
BNC_API void rds_fma(float *ret, const float *a, const float *b, const float *c);
#endif
BNC_API void rds_fma_safe(float *ret, const float *a, float b, const float *c);
BNC_API void rds_div_fma(float *ret, const float *a, const float *b);
BNC_API void rds_exp(float *ret, const float *x);
BNC_API void rds_expm1(float *ret, const float *x);
BNC_API void rds_log(float *ret, const float *x);
BNC_API void rds_log10(float *ret, const float *x);
BNC_API void rds_sin(float *ret, const float *x);
BNC_API void rds_cos(float *ret, const float *x);
BNC_API void rds_sincos(float *sin_ret, float *cos_ret, const float *x);
BNC_API void rds_tan(float *ret, const float *x);

/* ---- TS (triple-single, float[3]) ---- */
#ifndef __BNC_RDS_H_
BNC_API void rts_fma(float *ret, const float *a, const float *b, const float *c);
#endif
BNC_API void rts_fma_safe(float *ret, const float *a, float b, const float *c);
BNC_API void rts_div_fma(float *ret, const float *a, const float *b);
BNC_API void rts_exp(float *ret, const float *x);
BNC_API void rts_expm1(float *ret, const float *x);
BNC_API void rts_log(float *ret, const float *x);
BNC_API void rts_log10(float *ret, const float *x);
BNC_API void rts_sin(float *ret, const float *x);
BNC_API void rts_cos(float *ret, const float *x);
BNC_API void rts_sincos(float *sin_ret, float *cos_ret, const float *x);
BNC_API void rts_tan(float *ret, const float *x);

/* ---- QS (quadruple-single, float[4]) ---- */
#ifndef __BNC_RDS_H_
BNC_API void rqs_fma(float *ret, const float *a, const float *b, const float *c);
#endif
BNC_API void rqs_fma_safe(float *ret, const float *a, float b, const float *c);
BNC_API void rqs_div_fma(float *ret, const float *a, const float *b);
BNC_API void rqs_exp(float *ret, const float *x);
BNC_API void rqs_expm1(float *ret, const float *x);
BNC_API void rqs_log(float *ret, const float *x);
BNC_API void rqs_log10(float *ret, const float *x);
BNC_API void rqs_sin(float *ret, const float *x);
BNC_API void rqs_cos(float *ret, const float *x);
BNC_API void rqs_sincos(float *sin_ret, float *cos_ret, const float *x);
BNC_API void rqs_tan(float *ret, const float *x);

/* ---- Basic arithmetic (classic kernels; rXX_div follows the library's
   default division, i.e. BNC_USE_FMA_DIV at build time redirects it) ---- */
BNC_API void rdd_add(double *ret, const double *a, const double *b);
BNC_API void rdd_sub(double *ret, const double *a, const double *b);
BNC_API void rdd_mul(double *ret, const double *a, const double *b);
BNC_API void rdd_div(double *ret, const double *a, const double *b);
BNC_API void rdd_sqrt(double *ret, const double *a);
BNC_API void rtd_add(double *ret, const double *a, const double *b);
BNC_API void rtd_sub(double *ret, const double *a, const double *b);
BNC_API void rtd_mul(double *ret, const double *a, const double *b);
BNC_API void rtd_div(double *ret, const double *a, const double *b);
BNC_API void rtd_sqrt(double *ret, const double *a);
BNC_API void rqd_add(double *ret, const double *a, const double *b);
BNC_API void rqd_sub(double *ret, const double *a, const double *b);
BNC_API void rqd_mul(double *ret, const double *a, const double *b);
BNC_API void rqd_div(double *ret, const double *a, const double *b);
BNC_API void rqd_sqrt(double *ret, const double *a);
BNC_API void rds_add(float *ret, const float *a, const float *b);
BNC_API void rds_sub(float *ret, const float *a, const float *b);
BNC_API void rds_mul(float *ret, const float *a, const float *b);
BNC_API void rds_div(float *ret, const float *a, const float *b);
BNC_API void rds_sqrt(float *ret, const float *a);
BNC_API void rts_add(float *ret, const float *a, const float *b);
BNC_API void rts_sub(float *ret, const float *a, const float *b);
BNC_API void rts_mul(float *ret, const float *a, const float *b);
BNC_API void rts_div(float *ret, const float *a, const float *b);
BNC_API void rts_sqrt(float *ret, const float *a);
BNC_API void rqs_add(float *ret, const float *a, const float *b);
BNC_API void rqs_sub(float *ret, const float *a, const float *b);
BNC_API void rqs_mul(float *ret, const float *a, const float *b);
BNC_API void rqs_div(float *ret, const float *a, const float *b);
BNC_API void rqs_sqrt(float *ret, const float *a);

#ifdef __cplusplus
}
#endif

#endif /* __BNC_RDTQ_FUNC_H */
