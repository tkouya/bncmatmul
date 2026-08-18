/********************************************************************************/
/* rdd.h: Reverse definition for double-double and quadruple-double arithmetic  */
/* Copyright (C) 2016 Tomonori Kouya                                            */
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

#ifdef __cplusplus
	#include <cstdio>
	#include <cstdlib>
//	#include <cstring>
	#include <cmath>
	#include <iostream>
//	#include <iomanip>
//	#include "qd/dd_real.h"
	#include "qd/qd_real.h"
	#include "qd/fpu.h"
	#include "c_dd_qd.h"
#else // __cplusplus
	#include <stdio.h>
	#include <stdlib.h>
	#include <math.h>
	#ifdef USE_QDORG
		#include "qd/c_dd.h"
		#include "qd/c_qd.h"
		#include "qd/fpu.h"
	#else
		#include "c_dd_qd.h"
	#endif
#endif // __cplusplus


// DD & QD size
#define DDSIZE 2 // double * 2
#define TDSIZE 3 // double * 3
#define QDSIZE 4 // double * 4

// dsfloat, tsfloat, qsfloat
typedef struct { float val[DDSIZE]; } dsfloat; // 24 * 2 = 48
typedef struct { float val[TDSIZE]; } tsfloat; // 24 * 3 = 72
typedef struct { float val[QDSIZE]; } qsfloat; // 24 * 4 = 96

// ddfloat, tdfloat, qdfloat
typedef struct { double val[DDSIZE]; } ddfloat; // 53 * 2 = 106
typedef struct { double val[TDSIZE]; } tdfloat; // 53 * 3 = 159
typedef struct { double val[QDSIZE]; } qdfloat; // 53 * 4 = 212

// DD QD Macros
#ifdef __cplusplus
	#define SET0_DD(val) { val = (dd_real)0; }
	#define SET0_QD(val) { val = (qd_real)0; }
#else // __cplusplus
	#define SET0_DD(val) { val[0] = (double)0.0; val[1] = (double)0.0; }
	#define SET0_TD(val) { val[0] = (double)0.0; val[1] = (double)0.0; val[2] = (double)0.0; } 
	#define SET0_QD(val) { val[0] = (double)0.0; val[1] = (double)0.0; val[2] = (double)0.0; val[3] = (double)0.0; }
#endif // __cplusplus

#ifdef __cplusplus // (0)

// C++ Macros

#define RDD_ADD(ret, a, b) { ret = a + b; }
#define RDD_SUB(ret, a, b) { ret = a - b; }
#define RDD_MUL(ret, a, b) { ret = a * b; }
#define RDD_DIV(ret, a, b) { ret = a / b; }
#define RDD_SQRT(ret, a) { ret = sqrt(a); }
//#define RDD_OUT_STR(a) c_dd_write(a)
//#define RDD_OUT_STR(a) { std::cout << a; }

//#define RDD_SET_STR(str, a) { str = (a).to_string(); } // c_dd_swrite(a, 33, str, 48)
//#define RDD_GET_STR(a, str) { a = dd_real(a); } // c_dd_read(str, a)
//#define RDD_GET_STR(str, a) { str = (a).to_string(); } // c_dd_swrite(a, 33, str, 48)
//#define RDD_SET_STR(a, str) { a = dd_real(a); } // c_dd_read(str, a)

#define RDD_GET_D(a) ((double)a)
#define RDD_SET_D(ret, d) { ret = (double)d; }
#define RDD_SET_UI(ret, org) { ret = (double)(org); }
#define RDD_SET(ret, org) { ret = org; }
#define RDD_NEG(ret, a) { ret = -a; }
#define RDD_ABS(ret, a) { ret = abs(a); }
#define RDD_UI_DIV(ret, a, b) { ret = (double)a / b; }
#define RDD_UI_SUB(ret, a, b) { ret = (double)a - b; }
#define RDD_DIV_D(ret, a, b) { ret = a / (dd_real)b; }
#define RDD_ADD_D(ret, a, b) { ret = a + (dd_real)b; }
#define RDD_SUB_D(ret, a, b) { ret = a - (dd_real)b; }
#define RDD_MUL_D(ret, a, b) { ret = a * (dd_real)b; }
#define RDD_DIV_UI(ret, a, b) { ret = a /(double)(b); }
#define RDD_ADD_UI(ret, a, b) { ret = a +(double)(b); }
#define RDD_SUB_UI(ret, a, b) { ret = a -(double)(b); }
#define RDD_MUL_UI(ret, a, b) { ret = a *(double)(b); }
#define RDD_FMA(ret, a, b, c) { ret = a * b + c; }
#define RDD_PI(ret) { ret = dd_real::_pi; }
#define RDD_EXP(ret, x) { ret = exp(x); }
#define RDD_SIN(ret, x) { ret = sin(x); }
#define RDD_COS(ret, x) { ret = cos(x); }
#define RDD_LOG(ret, x) { ret = log(x); }
#define RDD_ASIN(ret, x) { ret = asin(x); }
#define RDD_ACOS(ret, x) { ret = acos(x); }

// TD only in C++ (common with C)
#define RTD_ADD(ret, a, b) c_td_add(a, b, ret)
#define RTD_SUB(ret, a, b) c_td_sub(a, b, ret)
#ifdef USE_ACCURATE_TD_MUL
	#define RTD_MUL(ret, a, b) c_td_mul_accurate(a, b, ret)
#else // USE_ACCURATE_TD_MUL
	#define RTD_MUL(ret, a, b) c_td_mul_sloppy(a, b, ret)
#endif // USE_ACCURATE_TD_MUL
//#define RTD_MUL(ret, a, b) c_td_mul(a, b, ret)
#define RTD_DIV(ret, a, b) c_td_div(a, b, ret)
#define RTD_SQRT(ret, a) c_td_sqrt(a, ret)
#define RTD_SQRT_UI(ret, a) c_td_sqrt_d((double)a, ret)
//#define RTD_OUT_STR(a) c_td_write(a)
//#define RTD_OUT_STR(a) rtd_out_str_base(stdout, 10, 33, a)

//#define RTD_SET_STR(str, a) c_td_swrite(a, 33, str, 48)
//#define RTD_GET_STR(a, str) c_td_read(str, a)
#define RTD_SET_STR(str, a) 
#define RTD_GET_STR(a, str) 

// rtd_get_str("str", a) -> "str" := (char *)a
// rtd_set_str(a, "str") -> a := (td_real)"str"
//#define RTD_GET_STR(str, a) c_td_swrite(a, 33, str, 48)
//#define RTD_SET_STR(a, str) c_td_read(str, a)

#define RTD_GET_D(a) ((a)[0])
#define RTD_SET_D(ret, d) c_td_copy_d((double)(d), ret)
#define RTD_SET_UI(ret, org) c_td_copy_d((double)(org), ret)
#define RTD_SET(ret, org) c_td_copy(org, ret)
#define RTD_NEG(ret, a) c_td_neg(a, ret)
#define RTD_ABS(ret, a) c_td_abs(a, ret)
#define RTD_UI_DIV(ret, a, b) c_td_div_d_td((double)(a), b, ret)
#define RTD_UI_SUB(ret, a, b) c_td_sub_d_td((double)(a), b, ret)
#define RTD_DIV_D(ret, a, b) c_td_div_td_d(a, b, ret)
#define RTD_ADD_D(ret, a, b) c_td_add_td_d(a, b, ret)
#define RTD_SUB_D(ret, a, b) c_td_sub_td_d(a, b, ret)
#define RTD_MUL_D(ret, a, b) c_td_mul_td_d(a, b, ret)
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

#define RQD_ADD(ret, a, b) { ret = a + b; }
#define RQD_SUB(ret, a, b) { ret = a - b; }
#define RQD_MUL(ret, a, b) { ret = a * b; }
#define RQD_DIV(ret, a, b) { ret = a / b; }
#define RQD_SQRT(ret, a) { ret = sqrt(a); }
//#define RQD_OUT_STR(a) c_qd_write(a)
//#define RQD_OUT_STR(a) { std::cout << a; }

//#define RQD_SET_STR(str, a) { str = (a).to_string(); } // c_qd_swrite(a, 66, str, 84)
//#define RQD_GET_STR(a, str) { a = qd_real(str); } //c_qd_read(str, a)

//#define RQD_GET_STR(str, a) { str = (a).to_string(); } // c_qd_swrite(a, 66, str, 84)
//#define RQD_SET_STR(a, str) { a = qd_real(str); } //c_qd_read(str, a)

#define RQD_GET_D(a) ((double)a)
#define RQD_SET_D(ret, d) { ret = (double)d; }
#define RQD_SET_UI(ret, org) { ret = (double)(org); }
#define RQD_SET(ret, org) { ret = org; }
#define RQD_NEG(ret, a) { ret = -a; }
#define RQD_ABS(ret, a) { ret = abs(a); }
#define RQD_UI_DIV(ret, a, b) { ret = (double)a / b; }
#define RQD_UI_SUB(ret, a, b) { ret = (double)a - b; }
#define RQD_DIV_D(ret, a, b) { ret = a / (qd_real)b; }
#define RQD_ADD_D(ret, a, b) { ret = a + (qd_real)b; }
#define RQD_SUB_D(ret, a, b) { ret = a - (qd_real)b; }
#define RQD_MUL_D(ret, a, b) { ret = a * (qd_real)b; }
#define RQD_DIV_UI(ret, a, b) { ret = a /(double)(b); }
#define RQD_ADD_UI(ret, a, b) { ret = a +(double)(b); }
#define RQD_SUB_UI(ret, a, b) { ret = a -(double)(b); }
#define RQD_MUL_UI(ret, a, b) { ret = a *(double)(b); }
#define RQD_FMA(ret, a, b, c) { ret = a * b + c; }
#define RQD_PI(ret) { ret = qd_real::_pi; }
#define RQD_EXP(ret, x) { ret = exp(x); }
#define RQD_SIN(ret, x) { ret = sin(x); }
#define RQD_COS(ret, x) { ret = cos(x); }
#define RQD_LOG(ret, x) { ret = log(x); }
#define RQD_ASIN(ret, x) { ret = asin(x); }
#define RQD_ACOS(ret, x) { ret = acos(x); }

// DD print(no appending CR)
//void rdd_out_str_base(FILE *fp, int base, int length, double val[DDSIZE]);

// DD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
inline int rdd_cmp(dd_real a, dd_real b) { if(a > b) return 1; else if(a == b) return 0; else return -1; }

// DD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
inline int rdd_cmp_d(dd_real a, double b) { if(a > b) return 1; else if(a == b) return 0; else return -1; }

// DD sqrt_d
//#define rdd_sqrt_d(ret, a) { ret = sqrt((dd_real)a); }
inline void rdd_sqrt_d(dd_real &ret, double a) { ret = sqrt((dd_real)a); }

#define RDD_CMP(a, b) rdd_cmp(a, b)
#define RDD_CMP_D(a, b) rdd_cmp_d(a, b)
#define RDD_CMP_UI(a, b) rdd_cmp_d(a, (double)(b))
#define RDD_SQRT_D(ret, a) rdd_sqrt_d(ret, a)
#define RDD_SQRT_UI(ret, a) rdd_sqrt_d(ret, (double)(a))

// QD print(no appending CR)
void rqd_out_str_base(FILE *fp, int base, int length, double val[QDSIZE]);

// QD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
inline int rqd_cmp(qd_real a, qd_real b) { if(a > b) return 1; else if(a == b) return 0; else return -1; }

// QD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
inline int rqd_cmp_d(qd_real a, double b) { if(a > b) return 1; else if(a == b) return 0; else return -1; }

// DD sqrt_d
//#define rqd_sqrt_d(ret, a) { ret = sqrt((qd_real)a); }
inline void rqd_sqrt_d(qd_real &ret, double a) { ret = sqrt((qd_real)a); }

#define RQD_CMP(a, b) rqd_cmp(a, b)
#define RQD_CMP_D(a, b) rqd_cmp_d(a, b)
#define RQD_CMP_UI(a, b) rqd_cmp_d(a, (double)(b))
#define RQD_SQRT_D(ret, a) rqd_sqrt_d(ret, a)
#define RQD_SQRT_UI(ret, a) rqd_sqrt_d(ret, (double)(a))

#define set0_dd(val) SET0_DD(val)
#define rdd_set0(val) SET0_DD(val)
#define rdd_add(ret, a, b) RDD_ADD(ret, a, b)
#define rdd_sub(ret, a, b) RDD_SUB(ret, a, b)
#define rdd_mul(ret, a, b) RDD_MUL(ret, a, b)
#define rdd_div(ret, a, b) RDD_DIV(ret, a, b)
#define rdd_sqrt(ret, a) RDD_SQRT(ret, a)
#define rdd_sqrt_d(ret, a) RDD_SQRT_D(ret, a)
#define rdd_sqrt_ui(ret, a) RDD_SQRT_UI(ret, a)
//#define rdd_out_str(a) RDD_OUT_STR(a)

//#define rdd_set_str(str, a) RDD_SET_STR(str, a)
//#define rdd_get_str(a, str) RDD_GET_STR(a, str)

// rdd_get_str("str", a) -> "str" := (char *)a
// rdd_set_str(a, "str") -> a := (dd_real)"str"
//#define rdd_get_str(str, a) RDD_GET_STR(str, a)
//#define rdd_set_str(a, str) RDD_SET_STR(a, str)

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
#define rdd_fma(ret, a, b) RDD_FMA(ret, a, b, c)
#define rdd_pi(ret) RDD_PI(ret)
#define rdd_exp(ret, x) RDD_EXP(ret, x) 
#define rdd_sin(ret, x) RDD_SIN(ret, x) 
#define rdd_cos(ret, x) RDD_COS(ret, x) 
#define rdd_log(ret, x) RDD_LOG(ret, x) 
#define rdd_asin(ret, x) RDD_ASIN(ret, x)
#define rdd_acos(ret, x) RDD_ACOS(ret, x)

#define set0_td(val) SET0_TD(val)
#define rtd_set0(val) SET0_TD(val)
#define rtd_add(ret, a, b) RTD_ADD(ret, a, b)
#define rtd_sub(ret, a, b) RTD_SUB(ret, a, b)
#define rtd_mul(ret, a, b) RTD_MUL(ret, a, b)
#define rtd_div(ret, a, b) RTD_DIV(ret, a, b)
#define rtd_sqrt(ret, a) RTD_SQRT(ret, a)
#define rtd_sqrt_d(ret, a) RTD_SQRT_D(ret, a)
#define rtd_sqrt_ui(ret, a) RTD_SQRT_UI(ret, a)
#define rtd_out_str(a) RTD_OUT_STR(a)
#define rtd_set_str(str, a) RTD_SET_STR(str, a)
#define rtd_get_str(a, str) RTD_GET_STR(a, str)
#define rtd_get_d(a) RTD_GET_D(a)
#define rtd_set_d(ret, d) RTD_SET_D(ret, d)
#define rtd_set_ui(ret, d) RTD_SET_UI(ret, d)
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

#define set0_qd(val) SET0_QD(val)
#define rqd_set0(val) SET0_QD(val)
#define rqd_add(ret, a, b) RQD_ADD(ret, a, b)
#define rqd_sub(ret, a, b) RQD_SUB(ret, a, b)
#define rqd_mul(ret, a, b) RQD_MUL(ret, a, b)
#define rqd_div(ret, a, b) RQD_DIV(ret, a, b)
#define rqd_sqrt(ret, a) RQD_SQRT(ret, a)
#define rqd_sqrt_d(ret, a) RQD_SQRT_D(ret, a)
#define rqd_sqrt_ui(ret, a) RQD_SQRT_UI(ret, a)
//#define rqd_out_str(a) RQD_OUT_STR(a)

//#define rqd_set_str(str, a) RQD_SET_STR(str, a)
//#define rqd_get_str(a, str) RQD_GET_STR(a, str)

// rqd_get_str("str", a) -> "str" := (char *)a
// rqd_set_str(a, "str") -> a := (qd_real)"str"
//#define rqd_get_str(str, a) RQD_GET_STR(str, a)
//#define rqd_set_str(a, str) RQD_SET_STR(a, str)

#define rqd_set_d(ret, d) RQD_SET_D(ret, d)
#define rqd_set_ui(ret, d) RQD_SET_UI(ret, d)
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
#define rqd_fma(ret, a, b) RQD_FMA(ret, a, b, c)
#define rqd_pi(ret) RQD_PI(ret)
#define rqd_exp(ret, x) RQD_EXP(ret, x) 
#define rqd_sin(ret, x) RQD_SIN(ret, x) 
#define rqd_cos(ret, x) RQD_COS(ret, x) 
#define rqd_log(ret, x) RQD_LOG(ret, x) 
#define rqd_asin(ret, x) RQD_ASIN(ret, x)
#define rqd_acos(ret, x) RQD_ACOS(ret, x)

#else // __cplusplus (0)

// DD in C
//#define RDD_ADD(ret, a, b) c_dd_add(a, b, ret)
#ifdef USE_ACCURATE_DD_ADD
	#define RDD_ADD(ret, a, b) c_dd_add(a, b, ret)
#else // USE_ACCURATE_DD_ADD
	#define RDD_ADD(ret, a, b) c_dd_add_sloppy(a, b, ret)
#endif // USE_ACCURATE_DD_ADD
#define RDD_SUB(ret, a, b) c_dd_sub(a, b, ret)
#define RDD_MUL(ret, a, b) c_dd_mul(a, b, ret)
#define RDD_DIV(ret, a, b) c_dd_div(a, b, ret)
#define RDD_SQRT(ret, a) c_dd_sqrt(a, ret)
//#define RDD_OUT_STR(a) c_dd_write(a)
#define RDD_OUT_STR(a) rdd_out_str_base(stdout, 10, 33, a)

//#define RDD_SET_STR(str, a) c_dd_swrite(a, 33, str, 48)
//#define RDD_GET_STR(a, str) c_dd_read(str, a)

// rdd_get_str("str", a) -> "str" := (char *)a
// rdd_set_str(a, "str") -> a := (dd_real)"str"
//#define RDD_GET_STR(str, a) c_dd_swrite(a, 33, str, 48)
//#define RDD_SET_STR(a, str) c_dd_read(str, a)
#define RDD_GET_STR(str, a) 
#define RDD_SET_STR(a, str) 

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

// TD only in C
#define RTD_ADD(ret, a, b) c_td_add(a, b, ret)
#define RTD_SUB(ret, a, b) c_td_sub(a, b, ret)
#ifdef USE_ACCURATE_TD_MUL
	#define RTD_MUL(ret, a, b) c_td_mul_accurate(a, b, ret)
#else // USE_ACCURATE_TD_MUL
	#define RTD_MUL(ret, a, b) c_td_mul_sloppy(a, b, ret)
#endif // USE_ACCURATE_TD_MUL
//#define RTD_MUL(ret, a, b) c_td_mul(a, b, ret)
#define RTD_DIV(ret, a, b) c_td_div(a, b, ret)
#define RTD_SQRT(ret, a) c_td_sqrt(a, ret)
//#define RTD_OUT_STR(a) c_td_write(a)
//#define RTD_OUT_STR(a) rtd_out_str_base(stdout, 10, 33, a)

//#define RTD_SET_STR(str, a) c_td_swrite(a, 33, str, 48)
//#define RTD_GET_STR(a, str) c_td_read(str, a)
#define RTD_SET_STR(str, a) 
#define RTD_GET_STR(a, str) 

// rtd_get_str("str", a) -> "str" := (char *)a
// rtd_set_str(a, "str") -> a := (td_real)"str"
//#define RTD_GET_STR(str, a) c_td_swrite(a, 33, str, 48)
//#define RTD_SET_STR(a, str) c_td_read(str, a)

#define RTD_GET_D(a) ((a)[0])
#define RTD_SET_D(ret, d) c_td_copy_d((double)(d), ret)
#define RTD_SET_UI(ret, org) c_td_copy_d((double)(org), ret)
#define RTD_SET(ret, org) c_td_copy(org, ret)
#define RTD_NEG(ret, a) c_td_neg(a, ret)
#define RTD_ABS(ret, a) c_td_abs(a, ret)
#define RTD_UI_DIV(ret, a, b) c_td_div_d_td((double)(a), b, ret)
#define RTD_UI_SUB(ret, a, b) c_td_sub_d_td((double)(a), b, ret)
#define RTD_DIV_D(ret, a, b) c_td_div_td_d(a, b, ret)
#define RTD_ADD_D(ret, a, b) c_td_add_td_d(a, b, ret)
#define RTD_SUB_D(ret, a, b) c_td_sub_td_d(a, b, ret)
#define RTD_MUL_D(ret, a, b) c_td_mul_td_d(a, b, ret)
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
#define RQD_ADD(ret, a, b) c_qd_add(a, b, ret)
#define RQD_SUB(ret, a, b) c_qd_sub(a, b, ret)
#define RQD_MUL(ret, a, b) c_qd_mul(a, b, ret)
#define RQD_DIV(ret, a, b) c_qd_div(a, b, ret)
#define RQD_SQRT(ret, a) c_qd_sqrt(a, ret)
//#define RQD_OUT_STR(a) c_qd_write(a)
//#define RQD_OUT_STR(a) rqd_out_str_base(stdout, 10, 64, a)

//#define RQD_SET_STR(str, a) c_qd_swrite(a, 66, str, 84)
//#define RQD_GET_STR(a, str) c_qd_read(str, a)
#define RQD_SET_STR(str, a)
#define RQD_GET_STR(a, str)

// rqd_get_str("str", a) -> "str" := (char *)a
// rqd_set_str(a, "str") -> a := (qd_real)"str"
//#define RQD_GET_STR(str, a) c_qd_swrite(a, 33, str, 48)
//#define RQD_SET_STR(a, str) c_qd_read(str, a)

#define RQD_GET_D(a) ((a)[0])
#define RQD_SET_D(ret, d) c_qd_copy_d((double)(d), ret)
#define RQD_SET_UI(ret, org) c_qd_copy_d((double)(org), ret)
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

// DD sqrt_d
static inline void rdd_sqrt_d(double ret[DDSIZE], double a)
{
	static double tmp[DDSIZE];

	c_dd_copy_d(a, tmp);
	c_dd_sqrt(tmp, ret);

	return;
}

// DD fma
// ret = a * b + c
static inline void rdd_fma(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE], double c[DDSIZE])
{
	static double tmp[DDSIZE];

	c_dd_mul(a, b, tmp);
	c_dd_add(tmp, c, ret);

	return;
}

// DD pow
// ret = base^power = exp(power * log(base))
static inline void rdd_pow(double ret[DDSIZE], double base[DDSIZE], double power[DDSIZE])
{
	static double tmp[DDSIZE], tmp1[DDSIZE];

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
	static char str[128];
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

// QD sqrt_d
static inline void rqd_sqrt_d(double ret[QDSIZE], double a)
{
	static double tmp[QDSIZE];

	c_qd_copy_d(a, tmp);
	c_qd_sqrt(tmp, ret);

	return;
}

// QD fma
// ret = a * b + c
static inline void rqd_fma(double ret[QDSIZE], double a[QDSIZE], double b[QDSIZE], double c[QDSIZE])
{
	static double tmp[QDSIZE];

	c_qd_mul(a, b, tmp);
	c_qd_add(tmp, c, ret);

	return;
}

// QD pow
// ret = base^power = exp(power * log(base))
static inline void rqd_pow(double ret[QDSIZE], double base[QDSIZE], double power[QDSIZE])
{
	static double tmp[QDSIZE], tmp1[QDSIZE];

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
	static char str[128];
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

// TD sqrt_d
static inline void rtd_sqrt_d(double ret[TDSIZE], double a)
{
	static double tmp[TDSIZE];

	c_td_copy_d(a, tmp);
	c_td_sqrt(tmp, ret);

	return;
}

// TD fma
// ret = a * b + c
static inline void rtd_fma(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE], double c[TDSIZE])
{
	static double tmp[TDSIZE];

#ifdef USE_ACCURATE_TD_MUL
	c_td_mul_accurate(a, b, tmp);
#else // USE_ACCURATE_TD_MUL
	c_td_mul_sloppy(a, b, tmp);
#endif // USE_ACCURATE_TD_MUL
	c_td_add(tmp, c, ret);
	//rtd_mul(tmp, a, b);
	//rtd_add(ret, c, tmp);

	return;
}

// TD pow
// ret = base^power = exp(power * log(base))
static inline void rtd_pow(double ret[TDSIZE], double base[TDSIZE], double power[TDSIZE])
{
	static double tmp[TDSIZE], tmp1[TDSIZE];

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
#define RTD_SQRT_D(ret, a) rtd_sqrt_d(ret, a)
#define RTD_SQRT_UI(ret, a) rtd_sqrt_d(ret, (double)(a))


#ifndef USE_RDD_FUNCTIONS
	#define set0_dd(val) SET0_DD(val)
	#define rdd_set0(val) SET0_DD(val)
	#define rdd_add(ret, a, b) RDD_ADD(ret, a, b)
	#define rdd_sub(ret, a, b) RDD_SUB(ret, a, b)
	#define rdd_mul(ret, a, b) RDD_MUL(ret, a, b)
	#define rdd_div(ret, a, b) RDD_DIV(ret, a, b)
	#define rdd_sqrt(ret, a) RDD_SQRT(ret, a)
	#define rdd_sqrt_d(ret, a) RDD_SQRT_D(ret, a)
	#define rdd_sqrt_ui(ret, a) RDD_SQRT_UI(ret, a)
//	#define rdd_out_str(a) RDD_OUT_STR(a)
	#define rdd_set_str(str, a) RDD_SET_STR(str, a)
	#define rdd_get_str(a, str) RDD_GET_STR(a, str)
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
#endif // USE_RDD_FUNCTIONS

#ifndef USE_RTD_FUNCTIONS
	#define set0_td(val) SET0_TD(val)
	#define rtd_set0(val) SET0_TD(val)
	#define rtd_add(ret, a, b) RTD_ADD(ret, a, b)
	#define rtd_sub(ret, a, b) RTD_SUB(ret, a, b)
	#define rtd_mul(ret, a, b) RTD_MUL(ret, a, b)
	#define rtd_div(ret, a, b) RTD_DIV(ret, a, b)
	#define rtd_sqrt(ret, a) RTD_SQRT(ret, a)
	#define rtd_sqrt_d(ret, a) RTD_SQRT_D(ret, a)
	#define rtd_sqrt_ui(ret, a) RTD_SQRT_UI(ret, a)
//	#define rtd_out_str(a) RTD_OUT_STR(a)
	#define rtd_set_str(str, a) RTD_SET_STR(str, a)
	#define rtd_get_str(a, str) RTD_GET_STR(a, str)
	#define rtd_get_d(a) RTD_GET_D(a)
	#define rtd_set_d(ret, d) RTD_SET_D(ret, d)
	#define rtd_set_ui(ret, d) RTD_SET_UI(ret, d)
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
	#define rqd_sqrt(ret, a) RQD_SQRT(ret, a)
	#define rqd_sqrt_d(ret, a) RQD_SQRT_D(ret, a)
	#define rqd_sqrt_ui(ret, a) RQD_SQRT_UI(ret, a)
//	#define rqd_out_str(a) RQD_OUT_STR(a)
	#define rqd_set_str(str, a) RQD_SET_STR(str, a)
	#define rqd_get_str(a, str) RQD_GET_STR(a, str)
	#define rqd_get_d(a) RQD_GET_D(a)
	#define rqd_set_d(ret, d) RQD_SET_D(ret, d)
	#define rqd_set_ui(ret, d) RQD_SET_UI(ret, d)
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

#endif // __cplusplus (0)

#endif // __BNC_RDD_H_

