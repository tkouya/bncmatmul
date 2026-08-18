/********************************************************************************/
/* bnc_common.h: BNC common definitions and funtions                            */
/* Copyright (C) 2022 Tomonori Kouya                                            */
/*                                                                              */
/* This program is free software: you can redistribute it and/or modify it      */
/* under the terms of the GNU Lesser General Public License as published by the */
/* Free Software Foundation, either version 3 of the License or any later       */
/* version.                                                                     */
/*                                                                              */
/* This program is distributed in the hope that it will be useful, but WITHOUT  */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License */
/* for more details.                                                            */
/*                                                                              */
/* You should have received a copy of the GNU Lesser General Public License     */
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                              */
/********************************************************************************/
// common definition 
#ifndef __BNC_COMMON_H
#define __BNC_COMMON_H

/* Concise SVE2 feature alias: mirror the compiler's __ARM_NEON style for
 * the predefined __ARM_FEATURE_SVE2 macro, so the rest of the library can
 * test __ARM_SVE2 consistently with __ARM_NEON. */
#if defined(__ARM_FEATURE_SVE2) && !defined(__ARM_SVE2)
#define __ARM_SVE2 1
#endif

#ifdef __cpluplus
extern "C" {
#endif // __cplusplus

#ifdef USE_GMP
#include "gmp.h"
#ifdef USE_MPFR
#include "mpfr.h"
#include "mpf2mpfr.h"
// mpc : 2024-11-28
#include "mpc.h"

#endif // USE_MPFR
#endif // USE_GMP

// SUCCESS or ERROR
#define BNC_SUCCESS (0)
#define BNC_ERROR (-1)

// DD, TD, QD size
#ifndef DDSIZE
#define DDSIZE 2
#endif // DDSIZE
#ifndef TDSIZE
#define TDSIZE 3
#endif // TDSIZE
#ifndef QDSIZE
#define QDSIZE 4
#endif // QDSIZE

// cddfloat, ctdfloat, cqdfloat
#ifndef CDDFLOAT
#define CDDFLOAT
typedef struct { double val_re[DDSIZE]; double val_im[DDSIZE]; } cddfloat; // 53 * 2 = 106
typedef struct { double val_re[TDSIZE]; double val_im[TDSIZE]; } ctdfloat; // 53 * 3 = 159
typedef struct { double val_re[QDSIZE]; double val_im[QDSIZE]; } cqdfloat; // 53 * 4 = 212
#endif // CDDFLOAT

#include "mpfr_dtq_sd.h" // mpf_get_[dtq]d

// ChatGPT suggested on 2024-02-26(Wed)
#include <stdio.h>
#include <errno.h>
#include <string.h>

// The macro showing error messages
#define BNC_PRINT_ERROR_MESSAGE(msg) \
    fprintf(stderr, "At : %s\n In the function: %s\nIn the SRC file: %s\nAt the #line: %d\nSystem error: %s\n", \
            msg, __func__, __FILE__, __LINE__, strerror(errno))

#define BNC_PRINT_ERROR \
			fprintf(stderr, "In the function: %s\nIn the SRC file: %s\nAt the #line: %d\nSystem error: %s\n", \
					__func__, __FILE__, __LINE__, strerror(errno))
		

/* Example of usage
void sampleFunction() {
    FILE *fp = fopen("non_existent_file.txt", "r");
    if (fp == NULL) {
        BNC_PRINT_ERROR("Cannot open the file!");
        return;
    }
    fclose(fp);
}
*/

// Intel Compiler or not
// ref) https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/macros/additional-predefined-macros.html
#if defined(__ICC) || defined(__ICL) // __ICC: Linux or macOS, __ICL: Windows
    #define BNC_USE_ICC
// https://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html#Common-Predefined-Macros
#elif defined(__GNUC__) // GCC
    #define BNC_USE_GCC
#else // unknown compiler
    #undef BNC_USE_ICC
    #undef BNC_USE_GCC
#endif // BNC_USE_ICC or _GCC

// 2022-03-07 (Mon) T.Kouya
void bnc_print_env_all(void);

// log2(x)
//#define mylog2(x) (log((double)(x)) / log((double)2.0)) // <-- fix! 2015-06-16(Tue)
double mylog2(double x);

// GFlops
double matmul_gflops(double comp_sec, int dim);

// GB of Double prec. Square matrix
int byte_double_sqmat(int dim);

#ifdef USE_GMP

static unsigned long int bnc_default_prec = 128; /* Global variable */
#ifdef GMP_RNDN
static unsigned long int bnc_default_rounding_mode = GMP_RNDN; /* Global variable */
//#elif MPFR_RNDN
//static unsigned long int bnc_default_rounding_mode = MPFR_RNDN; /* Global variable */
static mpc_rnd_t bnc_default_rounding_mode_c = MPC_RNDNN; // Global variable
#endif // GMP_RNDN

/* set bnc_default_prec */
void set_bnc_default_prec(unsigned long precision);
void _set_bnc_default_prec(unsigned long precision);
void set_bnc_default_prec_decimal(unsigned long precision);
void _set_bnc_default_prec_decimal(unsigned long precision);

#ifdef USE_MPFR
mp_rnd_t get_bnc_default_rounding_mode(void);

void set_bnc_rounding_mode(mp_rnd_t rounding_mode);
void _set_bnc_rounding_mode(mp_rnd_t rounding_mode);

mpc_rnd_t get_bnc_default_rounding_mode_c(void);
void set_bnc_default_rounding_mode_c(mpc_rnd_t rounding_mode_c);
void _set_bnc_default_rounding_mode_c(mpc_rnd_t rounding_mode_c);
#endif // USE_MPFR

/* get bnc_default_prec */
unsigned long int get_bnc_default_prec(void);

/* rop := op1 * op2 + op3 */
void mpf_fma(mpf_t rop, mpf_t op1, mpf_t op2, mpf_t op3);

// if |a| >  |b|, return +1
//    |a| == |b|, return 0
//    |a| <  |b|, return -1
int mpc_abscmp_ui(mpc_t a, unsigned long b);
int mpc_abscmp(mpc_t a, mpc_t b);
#endif //USE_GMP

// mpf_log10, log, exp, sin, cos, tan, atan
#ifdef USE_GMP
#define mpf_log10(ret, x) mpfr_log10((ret), (x), GMP_RNDN)
#define mpf_exp(ret, x) mpfr_exp((ret), (x), GMP_RNDN)
#define mpf_log(ret, x) mpfr_log((ret), (x), GMP_RNDN)
#define mpf_sin(ret, x) mpfr_sin((ret), (x), GMP_RNDN)
#define mpf_cos(ret, x) mpfr_cos((ret), (x), GMP_RNDN)
#define mpf_tan(ret, x) mpfr_tan((ret), (x), GMP_RNDN)
#define mpf_atan(ret, x) mpfr_atan((ret), (x), GMP_RNDN)
#endif // USE_GMP

// log2(x) := log10(x) / log10(2)
#define DLOG2(x) (log10((x)) / 0.30102999566398119521373889472449)

// 2023-04-06(Thu) Random functions
#ifdef USE_GMP
#ifndef __DEF_BNC_DEFAULT_RANDSTATE
#define __DEF_BNC_DEFAULT_RANDSTATE
//extern gmp_randstate_t bnc_default_randstate; // default randstate
static gmp_randstate_t bnc_default_randstate; // default randstate
#endif // __DEF_BNC_DEFAULT_RANDSTATE

#ifndef __DEF_BNC_DEFAULT_RANDSTATE_DD
#define __DEF_BNC_DEFAULT_RANDSTATE_DD
//extern gmp_randstate_t bnc_default_randstate_dd; // default randstate
static gmp_randstate_t bnc_default_randstate_dd; // default randstate
#endif // __DEF_BNC_DEFAULT_RANDSTATE_DD

#ifndef __DEF_BNC_DEFAULT_RANDSTATE_TD
#define __DEF_BNC_DEFAULT_RANDSTATE_TD
//extern gmp_randstate_t bnc_default_randstate_td; // default randstate
static gmp_randstate_t bnc_default_randstate_td; // default randstate
#endif // __DEF_BNC_DEFAULT_RANDSTATE_TD

#ifndef __DEF_BNC_DEFAULT_RANDSTATE_QD
#define __DEF_BNC_DEFAULT_RANDSTATE_QD
//extern gmp_randstate_t bnc_default_randstate_qd; // default randstate
static gmp_randstate_t bnc_default_randstate_qd; // default randstate
#endif // __DEF_BNC_DEFAULT_RANDSTATE_QD

// 2023-04-06(Thu) Random functions
#ifndef __DEF_BNC_DEFAULT_RANDSTATE
#define __DEF_BNC_DEFAULT_RANDSTATE
//gmp_randstate_t bnc_default_randstate; // default randstate
static gmp_randstate_t bnc_default_randstate; // default randstate
#endif // __DEF_BNC_DEFAULT_RANDSTATE

#ifndef __DEF_BNC_DEFAULT_RANDSTATE_DD
#define __DEF_BNC_DEFAULT_RANDSTATE_DD
//gmp_randstate_t bnc_default_randstate_dd; // default randstate
static gmp_randstate_t bnc_default_randstate_dd; // default randstate
#endif // __DEF_BNC_DEFAULT_RANDSTATE_DD

#ifndef __DEF_BNC_DEFAULT_RANDSTATE_TD
#define __DEF_BNC_DEFAULT_RANDSTATE_TD
static gmp_randstate_t bnc_default_randstate_td; // default randstate
//gmp_randstate_t bnc_default_randstate_td; // default randstate
#endif // __DEF_BNC_DEFAULT_RANDSTATE_TD

#ifndef __DEF_BNC_DEFAULT_RANDSTATE_QD
#define __DEF_BNC_DEFAULT_RANDSTATE_QD
//gmp_randstate_t bnc_default_randstate_qd; // default randstate
static gmp_randstate_t bnc_default_randstate_qd; // default randstate
#endif // __DEF_BNC_DEFAULT_RANDSTATE_QD

// set random seed & initialize randstate
void mpf_srand(unsigned long seed);

// random number on [0, 1]
void mpf_urand(mpf_t ret);

// random number following standard normal distribution [-1, 1]
void mpf_nrand(mpf_t ret);

// mpc_urand on [0, 1]
void mpc_urand(mpc_t ret);
// mpc_nrand on [-1, 1]
void mpc_nrand(mpc_t ret);

// 2023-05-31 (Thu) RDD random function based on MPFR
// rdd_srand, rdd_urand, rdd_nrand
void rdd_srand(unsigned long seed);
void rdd_urand(double ret[DDSIZE]);
void rdd_nrand(double ret[DDSIZE]);

// 2023-06-01 (Fri) RTD random function based on MPFR
// rtd_srand, rtd_urand, rtd_nrand
void rtd_srand(unsigned long seed);
void rtd_urand(double ret[TDSIZE]);
void rtd_nrand(double ret[TDSIZE]);

// 2023-06-01 (Fri) RQD random function based on MPFR
// rqd_srand, rqd_urand, rqd_nrand
void rqd_srand(unsigned long seed);
void rqd_urand(double ret[QDSIZE]);
void rqd_nrand(double ret[QDSIZE]);

#endif // USE_GMP


/*************************************************************/
/* algebraic_eq.c: Algebraic Solvers for Algebraic Equations */
/*************************************************************/
int dquadratic_eq(double[2], double[2], double[3]);
#ifdef USE_GMP
#define mpfquadratic_eq mpf_quadratic_eq // 2026-02-04 (Wed) T.Kouya
int mpf_quadratic_eq(mpf_t[2], mpf_t[2], mpf_t[3]);
int mpc_quadratic_eq(mpc_t ans[2], mpc_t coef[3]);
#endif // USE_GMP

//---------------------------------------------------------
// Complex number and arithmetic
//---------------------------------------------------------
//---------------------------
// fcmplx, FCmplx
//---------------------------
typedef struct{
	float re;
	float im;
} fcmplx;

typedef fcmplx *FCmplx;

//---------------------------
// dcmplx, DCmplx
//---------------------------
typedef struct{
	double re;
	double im;
} dcmplx;

typedef dcmplx *DCmplx;

//---------------------------
// mpfcmplx, MPFCmplx
//---------------------------
#ifdef USE_GMP
typedef struct{
	unsigned long int prec;
	mpf_t re;
	mpf_t im;
} mpfcmplx;

typedef mpfcmplx *MPFCmplx;
#endif // USE_GMP

/*************************************************/
/* complex.c: Complex arithmetic                 */
/*************************************************/
FCmplx init_fcmplx(void);
void free_fcmplx(FCmplx);
float get_real_fcmplx(FCmplx);
float get_image_fcmplx(FCmplx);
void set_real_fcmplx(FCmplx, float);
void set_image_fcmplx(FCmplx, float);
void subst_fcmplx(FCmplx, FCmplx);
void set0_fcmplx(FCmplx);
void add_fcmplx(FCmplx, FCmplx, FCmplx);
void add_fcmplx_f(FCmplx, FCmplx, float);
void add2_fcmplx(FCmplx, FCmplx);
void sub_fcmplx(FCmplx, FCmplx, FCmplx);
void conj_fcmplx(FCmplx, FCmplx);
void sign_fcmplx(FCmplx, FCmplx);
void sign2_fcmplx(FCmplx);
float abs_fcmplx(FCmplx);
void mul_fcmplx(FCmplx, FCmplx, FCmplx);
void mul_fcmplx_f(FCmplx, FCmplx, float);
void mul2_fcmplx(FCmplx, FCmplx);
void div_fcmplx(FCmplx, FCmplx, FCmplx);
void iexp_fcmplx(FCmplx, float);
void print_fcmplx(FCmplx);

DCmplx init_dcmplx(void);
void free_dcmplx(DCmplx);
double get_real_dcmplx(DCmplx);
double get_image_dcmplx(DCmplx);
void set_real_dcmplx(DCmplx, double);
void set_image_dcmplx(DCmplx, double);
void subst_dcmplx(DCmplx, DCmplx);
void set0_dcmplx(DCmplx);
void add_dcmplx(DCmplx, DCmplx, DCmplx);
void add_dcmplx_d(DCmplx, DCmplx, double);
void add2_dcmplx(DCmplx, DCmplx);
void sub_dcmplx(DCmplx, DCmplx, DCmplx);
void conj_dcmplx(DCmplx, DCmplx);
void sign_dcmplx(DCmplx, DCmplx);
void sign2_dcmplx(DCmplx);
double abs_dcmplx(DCmplx);
void mul_dcmplx(DCmplx, DCmplx, DCmplx);
void mul_dcmplx_d(DCmplx, DCmplx, double);
void mul2_dcmplx(DCmplx, DCmplx);
void div_dcmplx(DCmplx, DCmplx, DCmplx);
void iexp_dcmplx(DCmplx, double);
void print_dcmplx(DCmplx);

#ifdef USE_GMP
MPFCmplx init_mpfcmplx(void);
MPFCmplx init2_mpfcmplx(unsigned long int);
void free_mpfcmplx(MPFCmplx);
void get_real_mpfcmplx(mpf_t, MPFCmplx);
void get_image_mpfcmplx(mpf_t, MPFCmplx);
void set_real_mpfcmplx(MPFCmplx, mpf_t);
void set_image_mpfcmplx(MPFCmplx, mpf_t);
void set_real_mpfcmplx_ui(MPFCmplx, unsigned long int);
void set_image_mpfcmplx_ui(MPFCmplx, unsigned long int);
void subst_mpfcmplx(MPFCmplx, MPFCmplx);
void set0_mpfcmplx(MPFCmplx);
void add_mpfcmplx(MPFCmplx, MPFCmplx, MPFCmplx);
void add_mpfcmplx_mpf(MPFCmplx, MPFCmplx, mpf_t);
void add2_mpfcmplx(MPFCmplx, MPFCmplx);
void sub_mpfcmplx(MPFCmplx, MPFCmplx, MPFCmplx);
void conj_mpfcmplx(MPFCmplx, MPFCmplx);
void sign_mpfcmplx(MPFCmplx, MPFCmplx);
void sign2_mpfcmplx(MPFCmplx);
void abs_mpfcmplx(mpf_t, MPFCmplx);
void mul_mpfcmplx(MPFCmplx, MPFCmplx, MPFCmplx);
void mul_mpfcmplx_ui(MPFCmplx, MPFCmplx, unsigned long int);
void mul_mpfcmplx_mpf(MPFCmplx, MPFCmplx, mpf_t);
void mul2_mpfcmplx(MPFCmplx, MPFCmplx);
void div_mpfcmplx(MPFCmplx, MPFCmplx, MPFCmplx);
void div_mpfcmplx_real(MPFCmplx c, MPFCmplx a, mpf_t b);
void iexp_mpfcmplx(MPFCmplx, mpf_t);
void print_mpfcmplx(MPFCmplx);

/* set unsigned long */
void set_mpfcmplx_ui_ui(MPFCmplx, unsigned long, unsigned long);

/* set strings as complex number */
void set_mpfcmplx_str_str(MPFCmplx, const char *, long, const char *, long);

/* set double precision real part */
void set_real_mpfcmplx_d(MPFCmplx, double);

/* set double precision imaginary part */
void set_image_mpfcmplx_d(MPFCmplx, double);

//void set_mpfcmplx_d(MPFCmplx, DCmplx);
void set_mpfcmplx_d(MPFCmplx, double _Complex);


unsigned long int get_prec_mpfcmplx(MPFCmplx);
mpf_ptr getp_real_mpfcmplx(MPFCmplx);
mpf_ptr getp_image_mpfcmplx(MPFCmplx);

/* c := a / b */
//void div_mpfcmplx_d(MPFCmplx, MPFCmplx, DCmplx);
void div_mpfcmplx_d(MPFCmplx, MPFCmplx, double _Complex);

/* c := 1 / a */
void inv_mpfcmplx(MPFCmplx, MPFCmplx);

/* c := -a */
void neg_mpfcmplx(MPFCmplx, MPFCmplx);
#endif // USE_GMP


// AVX, AVX2, FMA
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    #include <immintrin.h>
	#define _BNC_ALIGN_UNIT_SIZE 32 // 32 byte aligntment
	//#define _BNC_ALIGN_UNIT_SIZE 16 // 16 byte aligntment
	#define _BNC_D_WIDTH ((_BNC_ALIGN_UNIT_SIZE) / sizeof(double)) // 4
	#define _BNC_S_WIDTH ((_BNC_ALIGN_UNIT_SIZE) / sizeof(float)) // 8
	#define BNC_MALLOC(size) aligned_alloc((size_t)(_BNC_ALIGN_UNIT_SIZE), (size_t)(size))
	#define BNC_CALLOC(num, unit_size) aligned_alloc((size_t)(_BNC_ALIGN_UNIT_SIZE), (size_t)((unit_size) * (num)))
	#define PRINT_M256D_LS(m256d_val) printf("%f %f %f %f\n", (m256d_val)[3], (m256d_val)[2], (m256d_val)[1], (m256d_val)[0])
	#define PRINT_M256D_SL(m256d_val) printf("%f %f %f %f\n", (m256d_val)[0], (m256d_val)[1], (m256d_val)[2], (m256d_val)[3])
	#define PRINT_M256D PRINT_M256D_LS
//	#define PRINT_M256D(m256d_val) printf("%f %f %f %f\n", (m256d_val)[3], (m256d_val)[2], (m256d_val)[1], (m256d_val)[0])
#elif defined(__AVX512F__) // __AVX512F__
    #include <immintrin.h>
//    #include <zmmintrin.h>
	#define _BNC_ALIGN_UNIT_SIZE 64 // 64 byte aligntment
//	#define _BNC_ALIGN_UNIT_SIZE 32 // 32 byte aligntment
	#define _BNC_D_WIDTH ((_BNC_ALIGN_UNIT_SIZE) / sizeof(double)) // 8
	#define _BNC_S_WIDTH ((_BNC_ALIGN_UNIT_SIZE) / sizeof(float)) // 16
	#define BNC_MALLOC(size) aligned_alloc((size_t)(_BNC_ALIGN_UNIT_SIZE), (size_t)(size))
	#define BNC_CALLOC(num, unit_size) aligned_alloc((size_t)(_BNC_ALIGN_UNIT_SIZE), (size_t)((unit_size) * (num)))
#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) && !defined(__CUDACC__) // NEON (skipped under nvcc)
	/* nvcc on AArch64 + GCC 13 cannot parse <arm_neon.h>: cudafe++ does
	 * not recognize the GCC builtins (__builtin_aarch64_st2_lanev4hf
	 * etc.) and the dependent type names (float16x4x2_t, ...) become
	 * undefined.  The NEON SIMD path is host-only anyway, so for CUDA
	 * translation units we fall through to the scalar `else` branch. */
	#include <arm_neon.h>
	//#define _BNC_ALIGN_UNIT_SIZE 32 // 32 byte aligntment
	#define _BNC_ALIGN_UNIT_SIZE 16 // 16 byte aligntment
	#define _BNC_D_WIDTH ((_BNC_ALIGN_UNIT_SIZE) / sizeof(double)) // 2
	#define _BNC_S_WIDTH ((_BNC_ALIGN_UNIT_SIZE) / sizeof(float)) // 4
	#define BNC_MALLOC(size) aligned_alloc((size_t)(_BNC_ALIGN_UNIT_SIZE), (size_t)(size))
	#define BNC_CALLOC(num, unit_size) aligned_alloc((size_t)(_BNC_ALIGN_UNIT_SIZE), (size_t)((unit_size) * (num)))
#else // others
	#define _BNC_D_WIDTH 1 // 1
	#define _BNC_S_WIDTH 1 // 1
    #define BNC_MALLOC(size) malloc((size_t)(size))
	#define BNC_CALLOC(num, unit_size) calloc((size_t)(num), (size_t)(unit_size))
#endif // __AVX2__

#ifdef __cpluplus
} // extern "C" {
#endif // __cplusplus

#endif // __BNC_COMMON_H
