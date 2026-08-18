/********************************************************************************/
/* rdd.h: Reverse definition for double-double and quadruple-double arithmetic  */
/* Copyright (C) 2016-2023 Tomonori Kouya                                       */
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
#ifndef __BNC_RDD_H_
#define __BNC_RDD_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Common defs
//#include "bnc_common.h"

#include "c_dd_qd.h"
//#include "c_dtqd.h"

// Proposed branch-free DW/TW/QW FMA (arXiv:2607.11391).
// Define BNC_USE_NEW_FMA to make rdd_fma/rtd_fma/rqd_fma use it.
#include "bncfma_d.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// DD & QD size
#define DDSIZE 2 // double * 2
#define TDSIZE 3 // double * 3
#define QDSIZE 4 // double * 4

// dsfloat, tsfloat, qsfloat
//typedef struct { float val[DDSIZE]; } dsfloat; // 24 * 2 = 48
//typedef struct { float val[TDSIZE]; } tsfloat; // 24 * 3 = 72
//typedef struct { float val[QDSIZE]; } qsfloat; // 24 * 4 = 96

// ddfloat, tdfloat, qdfloat
typedef struct { double val[DDSIZE]; } ddfloat; // 53 * 2 = 106
typedef struct { double val[TDSIZE]; } tdfloat; // 53 * 3 = 159
typedef struct { double val[QDSIZE]; } qdfloat; // 53 * 4 = 21

// DD QD Macros
#define SET0_DD(val) { val[0] = (double)0.0; val[1] = (double)0.0; }
#define SET0_TD(val) { val[0] = (double)0.0; val[1] = (double)0.0; val[2] = (double)0.0; } 
#define SET0_QD(val) { val[0] = (double)0.0; val[1] = (double)0.0; val[2] = (double)0.0; val[3] = (double)0.0; }

// DD in C
//#define RDD_ADD(ret, a, b) c_dd_add(a, b, ret)
#ifdef USE_ACCURATE_DD_ADD
	#define RDD_ADD(ret, a, b) c_dd_add(a, b, ret)
#else // USE_ACCURATE_DD_ADD
	#ifdef USE_DD_BF
		#define RDD_ADD(ret, a, b) c_dd_add_bf(a, b, ret)
	#else // USE_DD_BF
		#define RDD_ADD(ret, a, b) c_dd_add_sloppy(a, b, ret)
	#endif // USE_DD_BF
#endif // USE_ACCURATE_DD_ADD
#define RDD_SUB(ret, a, b) c_dd_sub(a, b, ret)
#ifdef USE_DD_BF
	#define RDD_MUL(ret, a, b) c_dd_mul_bf(a, b, ret)
#else // USE_DD_BF
	#define RDD_MUL(ret, a, b) c_dd_mul(a, b, ret)
#endif // USE_DD_BF
#ifdef USE_ACCURATE_DD_DIV
	#define RDD_DIV(ret, a, b) c_dd_div(a, b, ret)
#else // USE_ACCURATE_DD_DIV
	#define RDD_DIV(ret, a, b) c_dd_sloppy_div(a, b, ret)
#endif // USE_ACCURATE_DD_DIV
#define RDD_SQRT(ret, a) c_dd_sqrt(a, ret)
//#define RDD_OUT_STR(a) c_dd_write(a)
#define RDD_OUT_STR(a) rdd_out_str_base(stdout, 10, 33, a)

//#define RDD_SET_STR(str, a) c_dd_swrite(a, 33, str, 48)
//#define RDD_GET_STR(a, str) c_dd_read(str, a)

// rdd_get_str("str", a) -> "str" := (char *)a
// rdd_set_str(a, "str") -> a := (dd_real)"str"
//#define RDD_GET_STR(str, a) c_dd_swrite(a, 33, str, 48)
//#define RDD_SET_STR(a, str) c_dd_read(str, a)
//#define RDD_GET_STR(str, a)
//#define RDD_SET_STR(a, str)

#define RDD_GET_D(a) ((a)[0])
#define RDD_SET_D(ret, d) c_dd_copy_d((double)(d), ret)
#define RDD_SET_UI(ret, org) c_dd_copy_d((double)(org), ret)
#define RDD_SET(ret, org) c_dd_copy(org, ret)
#define RDD_NEG(ret, a) c_dd_neg(a, ret)
#define RDD_ABS(ret, a) c_dd_abs(a, ret)
#define RDD_UI_DIV(ret, a, b) c_dd_div_d_dd((double)(a), b, ret)
#define RDD_UI_SUB(ret, a, b) c_dd_sub_d_dd((double)(a), b, ret)
#define RDD_DIV_D(ret, a, b) c_dd_div_dd_d(a, b, ret)
#define RDD_ADD_D(ret, a, b) c_dd_add_dd_d(a, b, ret)
#define RDD_SUB_D(ret, a, b) c_dd_sub_dd_d(a, b, ret)
#define RDD_MUL_D(ret, a, b) c_dd_mul_dd_d(a, b, ret)
#define RDD_DIV_UI(ret, a, b) c_dd_div_dd_d(a, (double)(b), ret)
#define RDD_ADD_UI(ret, a, b) c_dd_add_dd_d(a, (double)(b), ret)
#define RDD_SUB_UI(ret, a, b) c_dd_sub_dd_d(a, (double)(b), ret)
#define RDD_MUL_UI(ret, a, b) c_dd_mul_dd_d(a, (double)(b), ret)

#define RDD_PI(ret) c_dd_pi(ret)
#define RDD_EXP(ret, x) c_dd_exp(x, ret)
#define RDD_SIN(ret, x) c_dd_sin(x, ret)
#define RDD_COS(ret, x) c_dd_cos(x, ret)
#define RDD_LOG(ret, x) c_dd_log(x, ret)
#define RDD_ASIN(ret, x) c_dd_asin(x, ret)
#define RDD_ACOS(ret, x) c_dd_acos(x, ret)
#define RDD_LOG2(ret, x) c_dd_log2(x, ret)
#define RDD_LOG10(ret, x) c_dd_log10(x, ret)
#define RDD_FLOOR(ret, x) c_dd_floor(x, ret)
#define RDD_CEIL(ret, x) c_dd_ceil(x, ret)

// TD only in C
//#define RTD_ADD(ret, a, b) c_td_add(a, b, ret)
#define RTD_ADDT(ret, a, b) c_td_add(a, b, ret) // original
#ifdef USE_TD_BF
	#define RTD_ADD(ret, a, b) c_td_add_bf(a, b, ret) // branch free
#else // USE_TD_BF
	#define RTD_ADD(ret, a, b) c_td_addq(a, b, ret) // default
#endif // USE_TD_BF
#define RTD_ADDQ(ret, a, b) c_td_addq(a, b, ret)
#define RTD_SUBT(ret, a, b) c_td_sub(a, b, ret) // original
#define RTD_SUB(ret, a, b) c_td_subq(a, b, ret) // default
#define RTD_SUBQ(ret, a, b) c_td_subq(a, b, ret)
#ifdef USE_ACCURATE_TD_MUL
	#define RTD_MUL(ret, a, b) c_td_mul_accurate(a, b, ret)
#else // USE_ACCURATE_TD_MUL
	#ifdef USE_TD_BF
		#define RTD_MUL(ret, a, b) c_td_mul_bf(a, b, ret) // branch free
	#else // USE_TD_BF
		#define RTD_MUL(ret, a, b) c_td_mul_sloppy(a, b, ret) // default
		//#define RTD_MUL(ret, a, b) c_td_mulq_sloppy(a, b, ret)
	#endif // USE_TD_BF
#endif // USE_ACCURATE_TD_MUL
//#define RTD_MUL(ret, a, b) c_td_mul(a, b, ret)
#define RTD_DIVT(ret, a, b) c_td_divt(a, b, ret)
#define RTD_DIVTQ(ret, a, b) c_td_divtq(a, b, ret)
#ifdef BNC_USE_FMA_DIV
	#define RTD_DIV(ret, a, b) bnc_td_div_fma(a, b, ret)
#else
	#define RTD_DIV(ret, a, b) c_td_divtq(a, b, ret) // default
#endif // BNC_USE_FMA_DIV
//#define RTD_DIV(ret, a, b) c_td_divq(a, b, ret) // 2024-01-31
#define RTD_DIVQ(ret, a, b) c_td_divq(a, b, ret)
#define RTD_SQRT(ret, a) c_td_sqrt(a, ret)
//#define RTD_OUT_STR(a) c_td_write(a)
//#define RTD_OUT_STR(a) rtd_out_str_base(stdout, 10, 33, a)

//#define RTD_SET_STR(str, a) c_td_swrite(a, 33, str, 48)
//#define RTD_GET_STR(a, str) c_td_read(str, a)
//#define RTD_SET_STR(str, a) 
//#define RTD_GET_STR(a, str) 

// rtd_get_str("str", a) -> "str" := (char *)a
// rtd_set_str(a, "str") -> a := (td_real)"str"
//#define RTD_GET_STR(str, a) c_td_swrite(a, 33, str, 48)
//#define RTD_SET_STR(a, str) c_td_read(str, a)

#define RTD_GET_D(a) ((a)[0])
#define RTD_SET_D(ret, d) c_td_copy_d((double)(d), ret)
#define RTD_SET_UI(ret, org) c_td_copy_d((double)(org), ret)
#define RTD_SET_DD(ret, org) c_td_copy_dd(org, ret)
#define RTD_SET_QD(ret, org) c_td_copy_qd(org, ret)
#define RTD_SET(ret, org) c_td_copy(org, ret)
#define RTD_NEG(ret, a) c_td_neg(a, ret)
#define RTD_ABS(ret, a) c_td_abs(a, ret)
#define RTD_UI_DIV(ret, a, b) c_td_div_d_td((double)(a), b, ret)
#define RTD_UI_SUB(ret, a, b) c_td_sub_d_td((double)(a), b, ret)
#define RTD_DIV_D(ret, a, b) c_td_div_td_d(a, b, ret)
#define RTD_ADD_D(ret, a, b) c_td_add_td_d(a, b, ret)
#define RTD_SUB_D(ret, a, b) c_td_sub_td_d(a, b, ret)
#define RTD_MUL_D(ret, a, b) c_td_mul_td_d(a, b, ret)
#define RTD_MULQ_D(ret, a, b) c_td_mulq_td_d(a, b, ret) // 2024-10-21 T.Kouya
#define RTD_DIV_UI(ret, a, b) c_td_div_td_d(a, (double)(b), ret)
#define RTD_ADD_UI(ret, a, b) c_td_add_td_d(a, (double)(b), ret)
#define RTD_SUB_UI(ret, a, b) c_td_sub_td_d(a, (double)(b), ret)
#define RTD_MUL_UI(ret, a, b) c_td_mul_td_d(a, (double)(b), ret)

#define RTD_PI(ret) c_td_pi(ret)
#define RTD_EXP(ret, x) c_td_exp(x, ret)
#define RTD_SIN(ret, x) c_td_sin(x, ret)
#define RTD_COS(ret, x) c_td_cos(x, ret)
#define RTD_LOG(ret, x) c_td_log(x, ret)
#define RTD_ASIN(ret, x) c_td_asin(x, ret)
#define RTD_ACOS(ret, x) c_td_acos(x, ret)

// QD in C
#ifdef USE_QD_BF
	#define RQD_ADD(ret, a, b) c_qd_add_bf(a, b, ret)
	#define RQD_MUL(ret, a, b) c_qd_mul_bf(a, b, ret)
	//#define RQD_ADD(ret, a, b) c_qd_add(a, b, ret)
	//#define RQD_MUL(ret, a, b) c_qd_mul(a, b, ret)
#else // USE_TD_BF
	#define RQD_ADD(ret, a, b) c_qd_add(a, b, ret)
	#define RQD_MUL(ret, a, b) c_qd_mul(a, b, ret)
#endif // USE_TD_BF
#define RQD_SUB(ret, a, b) c_qd_sub(a, b, ret)
#define RQD_DIV(ret, a, b) c_qd_div(a, b, ret)
#define RQD_SQRT(ret, a) c_qd_sqrt(a, ret)

//#define RQD_OUT_STR(a) c_qd_write(a)
//#define RQD_OUT_STR(a) rqd_out_str_base(stdout, 10, 64, a)

//#define RQD_SET_STR(str, a) c_qd_swrite(a, 66, str, 84)
//#define RQD_GET_STR(a, str) c_qd_read(str, a)
//#define RQD_SET_STR(str, a)
//#define RQD_GET_STR(a, str)

// rqd_get_str("str", a) -> "str" := (char *)a
// rqd_set_str(a, "str") -> a := (qd_real)"str"
//#define RQD_GET_STR(str, a) c_qd_swrite(a, 33, str, 48)
//#define RQD_SET_STR(a, str) c_qd_read(str, a)

#define RQD_GET_D(a) ((a)[0])
#define RQD_SET_D(ret, d) c_qd_copy_d((double)(d), ret)
#define RQD_SET_UI(ret, org) c_qd_copy_d((double)(org), ret)
#define RQD_SET_DD(ret, org) c_qd_copy_dd(org, ret)
#define RQD_SET_TD(ret, org) c_qd_copy_td(org, ret)
#define RQD_SET(ret, org) c_qd_copy(org, ret)
#define RQD_NEG(ret, a) c_qd_neg(a, ret)
#define RQD_ABS(ret, a) c_qd_abs(a, ret)
#define RQD_UI_DIV(ret, a, b) c_qd_div_d_qd((double)(a), b, ret)
#define RQD_UI_SUB(ret, a, b) c_qd_sub_d_qd((double)(a), b, ret)
#define RQD_DIV_D(ret, a, b) c_qd_div_qd_d(a, b, ret)
#define RQD_ADD_D(ret, a, b) c_qd_add_qd_d(a, b, ret)
#define RQD_SUB_D(ret, a, b) c_qd_sub_qd_d(a, b, ret)
#define RQD_MUL_D(ret, a, b) c_qd_mul_qd_d(a, b, ret)
#define RQD_DIV_UI(ret, a, b) c_qd_div_qd_d(a, (double)(b), ret)
#define RQD_ADD_UI(ret, a, b) c_qd_add_qd_d(a, (double)(b), ret)
#define RQD_SUB_UI(ret, a, b) c_qd_sub_qd_d(a, (double)(b), ret)
#define RQD_MUL_UI(ret, a, b) c_qd_mul_qd_d(a, (double)(b), ret)

#define RQD_PI(ret) c_qd_pi(ret)
#define RQD_EXP(ret, x) c_qd_exp(x, ret)
#define RQD_SIN(ret, x) c_qd_sin(x, ret)
#define RQD_COS(ret, x) c_qd_cos(x, ret)
#define RQD_LOG(ret, x) c_qd_log(x, ret)
#define RQD_ASIN(ret, x) c_qd_asin(x, ret)
#define RQD_ACOS(ret, x) c_qd_acos(x, ret)

// DD print(no appending CR)
static inline void rdd_out_str_base(FILE *fp, int base, int length, double val[DDSIZE])
{
	static char str[64];
	c_dd_swrite(val, (length > 40) ? 40 : length, str, 46);
	fprintf(fp, "%s", str);
}

// DD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
static inline int rdd_cmp(double a[DDSIZE], double b[DDSIZE])
{
	int ret;

	c_dd_comp(a, b, &ret);

	return ret;
}

// DD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
static inline int rdd_cmp_d(double a[DDSIZE], double b)
{
	int ret;

	c_dd_comp_dd_d(a, b, &ret);

	return ret;
}

#ifdef USE_RDD_SQRT
// DD sqrt_d
static inline void rdd_sqrt_d(double ret[DDSIZE], double a)
{
	double tmp[DDSIZE];

	c_dd_copy_d(a, tmp);
	c_dd_sqrt(tmp, ret);

	return;
}
#endif // USE_RDD_SQRT

// ---------------
// Pair arithmetic
// ---------------

// ret[DDSIZE] := a * b + c
static inline void rpd_fma_d_d(double ret[DDSIZE], const double a, const double b, const double c)
{
	//ret[0] = DFMA(a, b, c);
	//ret[1] = DFMA(a, b, (c - ret[0]));
	ret[0] = fast_two_fma(a, b, c, &(ret[1]));
}

// ret[DDSIZE] := a * b + c[DDSIZE]
static inline void rpd_fma_d_dd(double ret[DDSIZE], const double a, const double b, const double c[DDSIZE])
{
	double t, e;

	//ret[0] = DFMA(a, b, c[0]);
	//t = c[0] - ret[0];
	//e = DFMA(a, b, t);
	ret[0] = fast_two_fma(a, b, c[0], &e);
	ret[1] = e + c[1];
	//ret[0] = quick_two_sum(ret[0], ret[1], &(ret[1]));
}

// ret[DDSIZE] := a * b[DDSIZE] + c[DDSIZE]
static inline void rpd_fma_dm_dd(double ret[DDSIZE], const double a, const double b[DDSIZE], const double c[DDSIZE])
{
	double t, e, f;

	//ret[0] = DFMA(a, b[0], c[0]);
	//t = c[0] - ret[0];
	//e = DFMA(a, b[0], t);
	ret[0] = fast_two_fma(a, b[0], c[0], &e);
	f = e + c[1];
	ret[1] = DFMA(a, b[1], f);
	//ret[0] = quick_two_sum(ret[0], ret[1], &(ret[1]));
}

// ret[DDSIZE] := a[DDSIZE] * b[DDSIZE] + c[DDSIZE]
static inline void rpd_fma(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE], double c[DDSIZE])
{
	double t, e, f, g, d[DDSIZE];

	//d[0] = DFMA(a[0], b[0], c[0]);
	//t = c[0] - d[0];
	//e = DFMA(a[0], b[0], t);
	d[0] = fast_two_fma(a[0], b[0], c[0], &e);
	//d[0] = fma_error(a[0], b[0], c[0], &e);
	f = e + c[1];
	g = DFMA(a[0], b[1], f);
	d[1] = DFMA(a[1], b[0], g);
	ret[0] = d[0]; ret[1] = d[1];
	//ret[0] = quick_two_sum(d[0], d[1], &(ret[1]));

	return;
}

// ---------------
// DD fma
// ---------------

// ret[DDSIZE] := a * b + c
static inline void rdd_fma_d_d(double ret[DDSIZE], const double a, const double b, const double c)
{
	//ret[0] = DFMA(a, b, c);
	//ret[1] = DFMA(a, b, (c - ret[0]));
	//ret[0] = fast_two_fma(a, b, c, &(ret[1]));
	rpd_fma_d_d(ret, a, b, c);
	ret[0] = quick_two_sum(ret[0], ret[1], &(ret[1]));
}
// ret[DDSIZE] := a * b + c[DDSIZE]
static inline void rdd_fma_d_dd(double ret[DDSIZE], const double a, const double b, const double c[DDSIZE])
{
	//double t, e;

	//ret[0] = DFMA(a, b, c[0]);
	//t = c[0] - ret[0];
	//e = DFMA(a, b, t);
	//ret[0] = fast_two_fma(a, b, c[0], &e);
	//ret[1] = e + c[1];
	rpd_fma_d_dd(ret, a, b, c);
	ret[0] = quick_two_sum(ret[0], ret[1], &(ret[1]));
}
// ret[DDSIZE] := a * b[DDSIZE] + c[DDSIZE]
static inline void rdd_fma_dm_dd(double ret[DDSIZE], const double a, const double b[DDSIZE], const double c[DDSIZE])
{
	//double t, e, f;

	//ret[0] = DFMA(a, b[0], c[0]);
	//t = c[0] - ret[0];
	//e = DFMA(a, b[0], t);
	//ret[0] = fast_two_fma(a, b, c[0], &e);
	//f = e + c[1];
	//ret[1] = DFMA(a, b[1], f);
	rpd_fma_dm_dd(ret, a, b, c);
	ret[0] = quick_two_sum(ret[0], ret[1], &(ret[1]));
}

// ret[DDSIZE] := a[DDSIZE] * b[DDSIZE] + c[DDSIZE]
static inline void rdd_fma(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE], double c[DDSIZE])
{
#ifdef BNC_USE_NEW_FMA
	// Proposed branch-free DW-FMA (17 flops, arXiv:2607.11391 Alg.1)
	bnc_dwfma(ret, a, b, c);

	return;
#elif defined(USE_DD_BF)
	// branch-free baseline (Zhang-Aiken): mul_bf + add_bf
	{
		double tmp_bf[DDSIZE];
		c_dd_mul_bf(a, b, tmp_bf);
		c_dd_add_bf(tmp_bf, c, ret);
	}

	return;
#else // BNC_USE_NEW_FMA
	double tmp[DDSIZE];

//#ifdef USE_OLD_DDFMA
#ifndef USE_NEW_DDFMA
	c_dd_mul(a, b, tmp);
	//c_dd_add(tmp, c, ret);
	c_dd_add_sloppy(tmp, c, ret);
	//rdd_mul(tmp, a, b);
	//rdd_add(ret, tmp, a);
#else // USE_OLD_DDFMA
	double t, e, f, g;
	//double t, e, f, g, d[DDSIZE];

	//d[0] = DFMA(a[0], b[0], c[0]);
	//t = c[0] - d[0];
	//e = DFMA(a[0], b[0], t);
	//d[0] = fast_two_fma(a[0], b[0], c[0], &e);
	//f = e + c[1];
	//g = DFMA(a[0], b[1], f);
	//d[1] = DFMA(a[1], b[0], g);
	//rpd_fma(tmp, a, b, c);
	tmp[0] = fma_error(a[0], b[0], c[0], &(tmp[1]));
	//t = fma_error(a[1], b[0], c[1], &e);
	t = DFMA(a[1], b[0], c[1]);
	//f = fma_error(a[0], b[1], t, &g);
	f = DFMA(a[0], b[1], t);
	tmp[1] += DFMA(a[1], b[1], f);
	ret[0] = quick_two_sum(tmp[0], tmp[1], &(ret[1]));
#endif // USE_OLD_DDFMA

	return;
#endif // BNC_USE_NEW_FMA
}

// DD pow
// ret = base^power = exp(power * log(base))
static inline void rdd_pow(double ret[DDSIZE], double base[DDSIZE], double power[DDSIZE])
{
	double tmp[DDSIZE], tmp1[DDSIZE];

	c_dd_log(base, tmp);
	c_dd_mul(power, tmp, tmp1);
	c_dd_exp(tmp1, ret);

	return;
}

#define RDD_CMP(a, b) rdd_cmp(a, b)
#define RDD_CMP_D(a, b) rdd_cmp_d(a, b)
#define RDD_CMP_UI(a, b) rdd_cmp_d(a, (double)(b))
#define RDD_SQRT_D(ret, a) rdd_sqrt_d(ret, a)
#define RDD_SQRT_UI(ret, a) rdd_sqrt_d(ret, (double)(a))

// QD print(no appending CR)
static inline void rqd_out_str_base(FILE *fp, int base, int length, double val[QDSIZE])
{
	char str[128];
	// void c_qd_swrite(const double *a, int precision, char *s, int len);
	c_qd_swrite(val, (length > 70) ? 70 : length, str, 80);
	fprintf(fp, "%s", str);
}

// QD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
static inline int rqd_cmp(double a[QDSIZE], double b[QDSIZE])
{
	int ret;

	c_qd_comp(a, b, &ret);

	return ret;
}

// QD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
static inline int rqd_cmp_d(double a[QDSIZE], double b)
{
	int ret;

	c_qd_comp_qd_d(a, b, &ret);

	return ret;
}

#ifdef USE_RQD_SQRT
// QD sqrt_d
static inline void rqd_sqrt_d(double ret[QDSIZE], double a)
{
	double tmp[QDSIZE];

	c_qd_copy_d(a, tmp);
	c_qd_sqrt(tmp, ret);

	return;
}
#endif // USE_RQD_SQRT

// QD fma
// ret = a * b + c
static inline void rqd_fma(double ret[QDSIZE], double a[QDSIZE], double b[QDSIZE], double c[QDSIZE])
{
#ifdef BNC_USE_NEW_FMA
	// Proposed branch-free QW-FMA (146 flops, arXiv:2607.11391 Alg.3)
	bnc_qwfma(ret, a, b, c);
#elif defined(USE_QD_BF)
	// branch-free baseline (Zhang-Aiken): mul_bf + add_bf
	{
		double tmp_bf[QDSIZE];
		c_qd_mul_bf(a, b, tmp_bf);
		c_qd_add_bf(tmp_bf, c, ret);
	}
#else // BNC_USE_NEW_FMA
	double tmp[QDSIZE];

	c_qd_mul(a, b, tmp);
	c_qd_add(tmp, c, ret);
#endif // BNC_USE_NEW_FMA

	return;
}

// QD pow
// ret = base^power = exp(power * log(base))
static inline void rqd_pow(double ret[QDSIZE], double base[QDSIZE], double power[QDSIZE])
{
	double tmp[QDSIZE], tmp1[QDSIZE];

	c_qd_log(base, tmp);
	c_qd_mul(power, tmp, tmp1);
	c_qd_exp(tmp1, ret);

	return;
}

#define RQD_CMP(a, b) rqd_cmp(a, b)
#define RQD_CMP_D(a, b) rqd_cmp_d(a, b)
#define RQD_CMP_UI(a, b) rqd_cmp_d(a, (double)(b))
#define RQD_SQRT_D(ret, a) rqd_sqrt_d(ret, a)
#define RQD_SQRT_UI(ret, a) rqd_sqrt_d(ret, (double)(a))

// TD print(no appending CR)
static inline void rtd_out_str_base(FILE *fp, int base, int length, double val[TDSIZE])
{
	char str[128];
	// void c_qd_swrite(const double *a, int precision, char *s, int len);
	c_td_swrite(val, (length > 70) ? 70 : length, str, 80);
	fprintf(fp, "%s", str);
}

// TD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
static inline int rtd_cmp(double a[TDSIZE], double b[TDSIZE])
{
	int ret;

	c_td_comp(a, b, &ret);

	return ret;
}

// TD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
static inline int rtd_cmp_d(double a[TDSIZE], double b)
{
	int ret;

	c_td_comp_td_d(a, b, &ret);

	return ret;
}

#ifdef USE_RQD_SQRT
// TD sqrt_d
static inline void rtd_sqrt_d(double ret[TDSIZE], double a)
{
	double tmp[TDSIZE];

	c_td_copy_d(a, tmp);
	c_td_sqrt(tmp, ret);

	return;
}
#endif // USE_RDD_SQRT

// TD fma
// ret = a * b + c
static inline void rtd_fma(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE], double c[TDSIZE])
{
#ifdef BNC_USE_NEW_FMA
	// Proposed branch-free TW-FMA (66 flops, arXiv:2607.11391 Alg.2)
	bnc_twfma(ret, a, b, c);

	return;
#elif defined(USE_TD_BF)
	// branch-free baseline (Zhang-Aiken): mul_bf + add_bf
	{
		double tmp_bf[TDSIZE];
		c_td_mul_bf(a, b, tmp_bf);
		c_td_add_bf(tmp_bf, c, ret);
	}

	return;
#else // BNC_USE_NEW_FMA
	double tmp[TDSIZE];

#ifdef USE_ACCURATE_TD_MUL
	c_td_mul_accurate(a, b, tmp);
#else // USE_ACCURATE_TD_MUL
	c_td_mul_sloppy(a, b, tmp);
#endif // USE_ACCURATE_TD_MUL
	c_td_add(tmp, c, ret);
	//rtd_mul(tmp, a, b);
	//rtd_add(ret, c, tmp);

	return;
#endif // BNC_USE_NEW_FMA
}

// TD pow
// ret = base^power = exp(power * log(base))
static inline void rtd_pow(double ret[TDSIZE], double base[TDSIZE], double power[TDSIZE])
{
	double tmp[TDSIZE], tmp1[TDSIZE];

#if 0
	c_td_log(base, tmp);
	c_td_mul(power, tmp, tmp1);
	c_td_exp(tmp1, ret);
#endif // 0

	return;
}

#define RTD_CMP(a, b) rtd_cmp(a, b)
#define RTD_CMP_D(a, b) rtd_cmp_d(a, b)
#define RTD_CMP_UI(a, b) rtd_cmp_d(a, (double)(b))
#ifndef USE_RTD_SQRT
#define RTD_SQRT_D(ret, a) rtd_sqrt_d_mpfr(ret, a)
#define RTD_SQRT_UI(ret, a) rtd_sqrt_d_mpfr(ret, (double)(a))
#else // USE_RTD_SQRT
#define RTD_SQRT_D(ret, a) rtd_sqrt_d(ret, a)
#define RTD_SQRT_UI(ret, a) rtd_sqrt_d(ret, (double)(a))
#endif // USE_RTD_SQRT

#ifndef USE_RDD_FUNCTIONS
	#define set0_dd(val) SET0_DD(val)
	#define rdd_set0(val) SET0_DD(val)
	#define rdd_add(ret, a, b) RDD_ADD(ret, a, b)
	#define rdd_sub(ret, a, b) RDD_SUB(ret, a, b)
	#define rdd_mul(ret, a, b) RDD_MUL(ret, a, b)
	#define rdd_div(ret, a, b) RDD_DIV(ret, a, b)
#ifndef USE_RDD_SQRT
	#define rdd_sqrt(ret, a) rdd_sqrt_mpfr(ret, a)
	#define rdd_sqrt_d(ret, a) rdd_sqrt_d_mpfr(ret, a)
	#define rdd_sqrt_ui(ret, a) rdd_sqrt_d_mpfr(ret, (double)a)
#else // USR_RDD_SQRT
	#define rdd_sqrt(ret, a) RDD_SQRT(ret, a)
	#define rdd_sqrt_d(ret, a) RDD_SQRT_D(ret, a)
	#define rdd_sqrt_ui(ret, a) RDD_SQRT_UI(ret, a)
#endif // USE_RDD_SQRT
//	#define rdd_out_str(a) RDD_OUT_STR(a)
//	#define rdd_set_str(str, a) RDD_SET_STR(str, a)
//	#define rdd_get_str(a, str) RDD_GET_STR(a, str)
	#define rdd_get_d(a) RDD_GET_D(a)
	#define rdd_set_d(ret, d) RDD_SET_D(ret, d)
	#define rdd_set_ui(ret, d) RDD_SET_UI(ret, d)
	#define rdd_set(ret, org) RDD_SET(ret, org)
	#define rdd_neg(ret, a) RDD_NEG(ret, a)
	#define rdd_abs(ret, a) RDD_ABS(ret, a)
	#define rdd_cmp_ui(a, b) RDD_CMP_UI(a, b)
	#define rdd_ui_div(ret, a, b) RDD_UI_DIV(ret, a, b)
	#define rdd_ui_sub(ret, a, b) RDD_UI_SUB(ret, a, b)
	#define rdd_div_d(ret, a, b) RDD_DIV_D(ret, a, b)
	#define rdd_add_d(ret, a, b) RDD_ADD_D(ret, a, b)
	#define rdd_sub_d(ret, a, b) RDD_SUB_D(ret, a, b)
	#define rdd_mul_d(ret, a, b) RDD_MUL_D(ret, a, b)
	#define rdd_div_ui(ret, a, b) RDD_DIV_UI(ret, a, b)
	#define rdd_add_ui(ret, a, b) RDD_ADD_UI(ret, a, b)
	#define rdd_sub_ui(ret, a, b) RDD_SUB_UI(ret, a, b)
	#define rdd_mul_ui(ret, a, b) RDD_MUL_UI(ret, a, b)

	#define rdd_pi(ret) RDD_PI(ret)
	#define rdd_exp(ret, x) RDD_EXP(ret, x)
	#define rdd_sin(ret, x) RDD_SIN(ret, x)
	#define rdd_cos(ret, x) RDD_COS(ret, x)
	#define rdd_log(ret, x) RDD_LOG(ret, x)
	#define rdd_asin(ret, x) RDD_ASIN(ret, x)
	#define rdd_acos(ret, x) RDD_ACOS(ret, x)
	#define rdd_log10(ret, x) RDD_LOG10(ret, x)
	#define rdd_log2(ret, x) RDD_LOG2(ret, x)
	#define rdd_floor(ret, x) RDD_FLOOR(ret, x)
	#define rdd_ceil(ret, x) RDD_CEIL(ret, x)
	//#define rdd_pow(ret, x) RDD_POW(ret, x)
#endif // USE_RDD_FUNCTIONS

#ifndef USE_RTD_FUNCTIONS
	#define set0_td(val) SET0_TD(val)
	#define rtd_set0(val) SET0_TD(val)
	#define rtd_add(ret, a, b) RTD_ADD(ret, a, b) // default -> RTD_ADDQ
	#define rtd_addt(ret, a, b) RTD_ADDT(ret, a, b)
	#define rtd_addq(ret, a, b) RTD_ADDQ(ret, a, b)
	#define rtd_sub(ret, a, b) RTD_SUB(ret, a, b) // default -> RTD_SUBQ
	#define rtd_subt(ret, a, b) RTD_SUBT(ret, a, b)
	#define rtd_subq(ret, a, b) RTD_SUBQ(ret, a, b)
	#define rtd_mul(ret, a, b) RTD_MUL(ret, a, b)
	#define rtd_divt(ret, a, b) RTD_DIVT(ret, a, b)
	#define rtd_divtq(ret, a, b) RTD_DIVTQ(ret, a, b)
	#define rtd_divq(ret, a, b) RTD_DIVQ(ret, a, b)
	#define rtd_div(ret, a, b) RTD_DIV(ret, a, b) // default -> RTD_DIVQ
#ifndef USE_RTD_SQRT
	#define rtd_sqrt(ret, a)    rtd_sqrt_mpfr(ret, a)
	#define rtd_sqrt_d(ret, a)  rtd_sqrt_d_mpfr(ret, a)
	#define rtd_sqrt_ui(ret, a) rtd_sqrt_d_mpfr(ret, (double)a)
#else // USE_RTD_SQRT
	#define rtd_sqrt(ret, a) RTD_SQRT(ret, a)
	#define rtd_sqrt_d(ret, a) RTD_SQRT_D(ret, a)
	#define rtd_sqrt_ui(ret, a) RTD_SQRT_UI(ret, a)
#endif // USE_RTD_SQRT
//	#define rtd_out_str(a) RTD_OUT_STR(a)
//	#define rtd_set_str(str, a) RTD_SET_STR(str, a)
//	#define rtd_get_str(a, str) RTD_GET_STR(a, str)
	#define rtd_get_d(a) RTD_GET_D(a)
	#define rtd_set_d(ret, d) RTD_SET_D(ret, d)
	#define rtd_set_ui(ret, d) RTD_SET_UI(ret, d)
	#define rtd_set_dd(ret, org) RTD_SET_DD(ret, org)
	#define rtd_set_qd(ret, org) RTD_SET_QD(ret, org)
	#define rtd_set(ret, org) RTD_SET(ret, org)
	#define rtd_neg(ret, a) RTD_NEG(ret, a)
	#define rtd_abs(ret, a) RTD_ABS(ret, a)
	#define rtd_cmp_ui(a, b) RTD_CMP_UI(a, b)
	#define rtd_ui_div(ret, a, b) RTD_UI_DIV(ret, a, b)
	#define rtd_ui_sub(ret, a, b) RTD_UI_SUB(ret, a, b)
	#define rtd_div_d(ret, a, b) RTD_DIV_D(ret, a, b)
	#define rtd_add_d(ret, a, b) RTD_ADD_D(ret, a, b)
	#define rtd_sub_d(ret, a, b) RTD_SUB_D(ret, a, b)
	#define rtd_mul_d(ret, a, b) RTD_MUL_D(ret, a, b)
	#define rtd_mulq_d(ret, a, b) RTD_MULQ_D(ret, a, b) // 2024-10-21 T.Kouya
	#define rtd_div_ui(ret, a, b) RTD_DIV_UI(ret, a, b)
	#define rtd_add_ui(ret, a, b) RTD_ADD_UI(ret, a, b)
	#define rtd_sub_ui(ret, a, b) RTD_SUB_UI(ret, a, b)
	#define rtd_mul_ui(ret, a, b) RTD_MUL_UI(ret, a, b)

	#define rtd_pi(ret) RTD_PI(ret)
	#define rtd_exp(ret, x) RTD_EXP(ret, x)
	#define rtd_sin(ret, x) RTD_SIN(ret, x)
	#define rtd_cos(ret, x) RTD_COS(ret, x)
	#define rtd_log(ret, x) RTD_LOG(ret, x)
	#define rtd_asin(ret, x) RTD_ASIN(ret, x)
	#define rtd_acos(ret, x) RTD_ACOS(ret, x)
#endif // USE_RTD_FUNCTIONS


#ifndef USE_RQD_FUNCTIONS
	#define set0_qd(val) SET0_QD(val)
	#define rqd_set0(val) SET0_QD(val)
	#define rqd_add(ret, a, b) RQD_ADD(ret, a, b)
	#define rqd_sub(ret, a, b) RQD_SUB(ret, a, b)
	#define rqd_mul(ret, a, b) RQD_MUL(ret, a, b)
	#define rqd_div(ret, a, b) RQD_DIV(ret, a, b)
#ifndef USE_RQD_SQRT
	#define rqd_sqrt(ret, a)    rqd_sqrt_mpfr(ret, a)
	#define rqd_sqrt_d(ret, a)  rqd_sqrt_d_mpfr(ret, a)
	#define rqd_sqrt_ui(ret, a) rqd_sqrt_d_mpfr(ret, (double)a)
#else // USE_RQD_SQRT
	#define rqd_sqrt(ret, a) RQD_SQRT(ret, a)
	#define rqd_sqrt_d(ret, a) RQD_SQRT_D(ret, a)
	#define rqd_sqrt_ui(ret, a) RQD_SQRT_UI(ret, a)
#endif // USE_RQD_SQRT
//	#define rqd_out_str(a) RQD_OUT_STR(a)
//	#define rqd_set_str(str, a) RQD_SET_STR(str, a)
//	#define rqd_get_str(a, str) RQD_GET_STR(a, str)
	#define rqd_get_d(a) RQD_GET_D(a)
	#define rqd_set_d(ret, d) RQD_SET_D(ret, d)
	#define rqd_set_ui(ret, d) RQD_SET_UI(ret, d)
	#define rqd_set_dd(ret, org) RQD_SET_DD(ret, org)
	#define rqd_set_td(ret, org) RQD_SET_TD(ret, org)
	#define rqd_set(ret, org) RQD_SET(ret, org)
	#define rqd_neg(ret, a) RQD_NEG(ret, a)
	#define rqd_abs(ret, a) RQD_ABS(ret, a)
	#define rqd_cmp_ui(a, b) RQD_CMP_UI(a, b)
	#define rqd_ui_div(ret, a, b) RQD_UI_DIV(ret, a, b)
	#define rqd_ui_sub(ret, a, b) RQD_UI_SUB(ret, a, b)
	#define rqd_div_d(ret, a, b) RQD_DIV_D(ret, a, b)
	#define rqd_add_d(ret, a, b) RQD_ADD_D(ret, a, b)
	#define rqd_sub_d(ret, a, b) RQD_SUB_D(ret, a, b)
	#define rqd_mul_d(ret, a, b) RQD_MUL_D(ret, a, b)
	#define rqd_div_ui(ret, a, b) RQD_DIV_UI(ret, a, b)
	#define rqd_add_ui(ret, a, b) RQD_ADD_UI(ret, a, b)
	#define rqd_sub_ui(ret, a, b) RQD_SUB_UI(ret, a, b)
	#define rqd_mul_ui(ret, a, b) RQD_MUL_UI(ret, a, b)

	#define rqd_pi(ret) RQD_PI(ret)
	#define rqd_exp(ret, x) RQD_EXP(ret, x)
	#define rqd_sin(ret, x) RQD_SIN(ret, x)
	#define rqd_cos(ret, x) RQD_COS(ret, x)
	#define rqd_log(ret, x) RQD_LOG(ret, x)
	#define rqd_asin(ret, x) RQD_ASIN(ret, x)
	#define rqd_acos(ret, x) RQD_ACOS(ret, x)
#endif // USE_RQD_FUNCTIONS

//---------------------------------------------------------
// Complex number and arithmetic
//---------------------------------------------------------

//---------------------------
// ddcmplx, DDCmplx
//---------------------------
typedef struct{
	ddfloat re;
	ddfloat im;
} ddcmplx;

typedef ddcmplx *DDCmplx;

//---------------------------
// tdcmplx, TDCmplx
//---------------------------
typedef struct{
	tdfloat re;
	tdfloat im;
} tdcmplx;

typedef tdcmplx *TDCmplx;

//---------------------------
// qdcmplx, QDCmplx
//---------------------------
typedef struct{
	qdfloat re;
	qdfloat im;
} qdcmplx;

typedef qdcmplx *QDCmplx;

#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus

#endif // __BNC_RDD_H_

