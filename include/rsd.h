/********************************************************************************/
/* rsd.h: Reverse definition                                                    */
/*            for double-sngle, triple-single, and quadruple-single arithmetic  */
/* Copyright (C) 2021 Tomonori Kouya                                            */
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
#ifndef __BNC_RDS_H_
#define __BNC_RDS_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// DD & QD size
#define DSSIZE 2 // float * 2
#define TSSIZE 3 // float * 3
#define QSSIZE 4 // float * 4

// dsfloat, tsfloat, qsfloat
typedef struct { float val[DSSIZE]; } dsfloat; // 24 * 2 = 48
typedef struct { float val[TSSIZE]; } tsfloat; // 24 * 3 = 72
typedef struct { float val[QSSIZE]; } qsfloat; // 24 * 4 = 96

// DD QD Macros
#define SET0_DS(val) { val[0] = (float)0.0f; val[1] = (float)0.0f; }
#define SET0_TS(val) { val[0] = (float)0.0f; val[1] = (float)0.0f; val[2] = (float)0.0f; } 
#define SET0_QS(val) { val[0] = (float)0.0f; val[1] = (float)0.0f; val[2] = (float)0.0f; val[3] = (float)0.0f; }

// DD in C
//#define RDS_ADD(ret, a, b) c_ds_add(a, b, ret)
#ifdef USE_ACCURATE_DD_ADD
	#define RDS_ADD(ret, a, b) c_ds_add(a, b, ret)
#else // USE_ACCURATE_DD_ADD
	#define RDS_ADD(ret, a, b) c_ds_add_sloppy(a, b, ret)
#endif // USE_ACCURATE_DD_ADD
#define RDS_SUB(ret, a, b) c_ds_sub(a, b, ret)
#define RDS_MUL(ret, a, b) c_ds_mul(a, b, ret)
#ifdef USE_ACCURATE_DD_DIV
	#define RDS_DIV(ret, a, b) c_ds_div(a, b, ret)
#else // USE_ACCURATE_DD_DIV
	#define RDS_DIV(ret, a, b) c_ds_sloppy_div(a, b, ret)
#endif // USE_ACCURATE_DD_DIV
#define RDS_SQRT(ret, a) c_ds_sqrt(a, ret)
//#define RDS_OUT_STR(a) c_ds_write(a)
#define RDS_OUT_STR(a) rds_out_str_base(stdout, 10, 33, a)

//#define RDS_SET_STR(str, a) c_ds_swrite(a, 33, str, 48)
//#define RDS_GET_STR(a, str) c_ds_read(str, a)

// rds_get_str("str", a) -> "str" := (char *)a
// rds_set_str(a, "str") -> a := (dd_real)"str"
//#define RDS_GET_STR(str, a) c_ds_swrite(a, 33, str, 48)
//#define RDS_SET_STR(a, str) c_ds_read(str, a)
#define RDS_GET_STR(str, a) 
#define RDS_SET_STR(a, str) 

#define RDS_GET_D(a) ((a)[0])
#define RDS_SET_D(ret, d) c_ds_copy_d((double)(d), ret)
#define RDS_SET_UI(ret, org) c_ds_copy_d((double)(org), ret)
#define RDS_SET(ret, org) c_ds_copy(org, ret)
#define RDS_NEG(ret, a) c_ds_neg(a, ret)
#define RDS_ABS(ret, a) c_ds_abs(a, ret)
#define RDS_UI_DIV(ret, a, b) c_ds_div_d_dd((double)(a), b, ret)
#define RDS_UI_SUB(ret, a, b) c_ds_sub_d_dd((double)(a), b, ret)
#define RDS_DIV_D(ret, a, b) c_ds_div_dd_d(a, b, ret)
#define RDS_ADD_D(ret, a, b) c_ds_add_dd_d(a, b, ret)
#define RDS_SUB_D(ret, a, b) c_ds_sub_dd_d(a, b, ret)
#define RDS_MUL_D(ret, a, b) c_ds_mul_dd_d(a, b, ret)
#define RDS_DIV_UI(ret, a, b) c_ds_div_dd_d(a, (double)(b), ret)
#define RDS_ADD_UI(ret, a, b) c_ds_add_dd_d(a, (double)(b), ret)
#define RDS_SUB_UI(ret, a, b) c_ds_sub_dd_d(a, (double)(b), ret)
#define RDS_MUL_UI(ret, a, b) c_ds_mul_dd_d(a, (double)(b), ret)

#define RDS_PI(ret) c_ds_pi(ret)
#define RDS_EXP(ret, x) c_ds_exp(x, ret)
#define RDS_SIN(ret, x) c_ds_sin(x, ret)
#define RDS_COS(ret, x) c_ds_cos(x, ret)
#define RDS_LOG(ret, x) c_ds_log(x, ret)
#define RDS_ASIN(ret, x) c_ds_asin(x, ret)
#define RDS_ACOS(ret, x) c_ds_acos(x, ret)

// TD only in C
//#define RTS_ADD(ret, a, b) c_ts_add(a, b, ret)
#define RTS_ADDT(ret, a, b) c_ts_add(a, b, ret) // original
#define RTS_ADD(ret, a, b) c_ts_addq(a, b, ret) // default
#define RTS_ADDQ(ret, a, b) c_ts_addq(a, b, ret)
#define RTS_SUBT(ret, a, b) c_ts_sub(a, b, ret) // original
#define RTS_SUB(ret, a, b) c_ts_subq(a, b, ret) // default
#define RTS_SUBQ(ret, a, b) c_ts_subq(a, b, ret)
#ifdef USE_ACCURATE_TD_MUL
	#define RTS_MUL(ret, a, b) c_ts_mul_accurate(a, b, ret)
#else // USE_ACCURATE_TD_MUL
	#define RTS_MUL(ret, a, b) c_ts_mul_sloppy(a, b, ret)
#endif // USE_ACCURATE_TD_MUL
//#define RTS_MUL(ret, a, b) c_ts_mul(a, b, ret)
#define RTS_DIVT(ret, a, b) c_ts_divt(a, b, ret)
#define RTS_DIVTQ(ret, a, b) c_ts_divtq(a, b, ret)
#define RTS_DIV(ret, a, b) c_ts_divtq(a, b, ret)
#define RTS_DIVQ(ret, a, b) c_ts_divq(a, b, ret)
#define RTS_SQRT(ret, a) c_ts_sqrt(a, ret)
//#define RTS_OUT_STR(a) c_ts_write(a)
//#define RTS_OUT_STR(a) rts_out_str_base(stdout, 10, 33, a)

//#define RTS_SET_STR(str, a) c_ts_swrite(a, 33, str, 48)
//#define RTS_GET_STR(a, str) c_ts_read(str, a)
#define RTS_SET_STR(str, a) 
#define RTS_GET_STR(a, str) 

// rts_get_str("str", a) -> "str" := (char *)a
// rts_set_str(a, "str") -> a := (td_real)"str"
//#define RTS_GET_STR(str, a) c_ts_swrite(a, 33, str, 48)
//#define RTS_SET_STR(a, str) c_ts_read(str, a)

#define RTS_GET_D(a) ((a)[0])
#define RTS_SET_D(ret, d) c_ts_copy_d((double)(d), ret)
#define RTS_SET_UI(ret, org) c_ts_copy_d((double)(org), ret)
#define RTS_SET(ret, org) c_ts_copy(org, ret)
#define RTS_NEG(ret, a) c_ts_neg(a, ret)
#define RTS_ABS(ret, a) c_ts_abs(a, ret)
#define RTS_UI_DIV(ret, a, b) c_ts_div_d_td((double)(a), b, ret)
#define RTS_UI_SUB(ret, a, b) c_ts_sub_d_td((double)(a), b, ret)
#define RTS_DIV_D(ret, a, b) c_ts_div_td_d(a, b, ret)
#define RTS_ADD_D(ret, a, b) c_ts_add_td_d(a, b, ret)
#define RTS_SUB_D(ret, a, b) c_ts_sub_td_d(a, b, ret)
#define RTS_MUL_D(ret, a, b) c_ts_mul_td_d(a, b, ret)
#define RTS_DIV_UI(ret, a, b) c_ts_div_td_d(a, (double)(b), ret)
#define RTS_ADD_UI(ret, a, b) c_ts_add_td_d(a, (double)(b), ret)
#define RTS_SUB_UI(ret, a, b) c_ts_sub_td_d(a, (double)(b), ret)
#define RTS_MUL_UI(ret, a, b) c_ts_mul_td_d(a, (double)(b), ret)

#define RTS_PI(ret) c_ts_pi(ret)
#define RTS_EXP(ret, x) c_ts_exp(x, ret)
#define RTS_SIN(ret, x) c_ts_sin(x, ret)
#define RTS_COS(ret, x) c_ts_cos(x, ret)
#define RTS_LOG(ret, x) c_ts_log(x, ret)
#define RTS_ASIN(ret, x) c_ts_asin(x, ret)
#define RTS_ACOS(ret, x) c_ts_acos(x, ret)

// QD in C
#define RQS_ADD(ret, a, b) c_qs_add(a, b, ret)
#define RQS_SUB(ret, a, b) c_qs_sub(a, b, ret)
#define RQS_MUL(ret, a, b) c_qs_mul(a, b, ret)
#define RQS_DIV(ret, a, b) c_qs_div(a, b, ret)
#define RQS_SQRT(ret, a) c_qs_sqrt(a, ret)
//#define RQS_OUT_STR(a) c_qs_write(a)
//#define RQS_OUT_STR(a) rqs_out_str_base(stdout, 10, 64, a)

//#define RQS_SET_STR(str, a) c_qs_swrite(a, 66, str, 84)
//#define RQS_GET_STR(a, str) c_qs_read(str, a)
#define RQS_SET_STR(str, a)
#define RQS_GET_STR(a, str)

// rqs_get_str("str", a) -> "str" := (char *)a
// rqs_set_str(a, "str") -> a := (qd_real)"str"
//#define RQS_GET_STR(str, a) c_qs_swrite(a, 33, str, 48)
//#define RQS_SET_STR(a, str) c_qs_read(str, a)

#define RQS_GET_F(a) ((a)[0])
#define RQS_SET_F(ret, d) c_qs_copy_f((double)(d), ret)
#define RQS_SET_UI(ret, org) c_qs_copy_f((double)(org), ret)
#define RQS_SET(ret, org) c_qs_copy(org, ret)
#define RQS_NEG(ret, a) c_qs_neg(a, ret)
#define RQS_ABS(ret, a) c_qs_abs(a, ret)
#define RQS_UI_DIV(ret, a, b) c_qs_div_f_qd((double)(a), b, ret)
#define RQS_UI_SUB(ret, a, b) c_qs_sub_f_qd((double)(a), b, ret)
#define RQS_DIV_F(ret, a, b) c_qs_div_qd_f(a, b, ret)
#define RQS_ADD_F(ret, a, b) c_qs_add_qd_f(a, b, ret)
#define RQS_SUB_F(ret, a, b) c_qs_sub_qd_f(a, b, ret)
#define RQS_MUL_F(ret, a, b) c_qs_mul_qd_f(a, b, ret)
#define RQS_DIV_UI(ret, a, b) c_qs_div_qd_f(a, (double)(b), ret)
#define RQS_ADD_UI(ret, a, b) c_qs_add_qd_f(a, (double)(b), ret)
#define RQS_SUB_UI(ret, a, b) c_qs_sub_qd_f(a, (double)(b), ret)
#define RQS_MUL_UI(ret, a, b) c_qs_mul_qd_f(a, (double)(b), ret)

#define RQS_PI(ret) c_qs_pi(ret)
#define RQS_EXP(ret, x) c_qs_exp(x, ret)
#define RQS_SIN(ret, x) c_qs_sin(x, ret)
#define RQS_COS(ret, x) c_qs_cos(x, ret)
#define RQS_LOG(ret, x) c_qs_log(x, ret)
#define RQS_ASIN(ret, x) c_qs_asin(x, ret)
#define RQS_ACOS(ret, x) c_qs_acos(x, ret)

// DD print(no appending CR)
static inline void rds_out_str_base(FILE *fp, int base, int length, double val[DSSIZE])
{
	static char str[64];
	c_ds_swrite(val, (length > 40) ? 40 : length, str, 46);
	fprintf(fp, "%s", str);
}

// DD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
static inline int rds_cmp(double a[DSSIZE], double b[DSSIZE])
{
	int ret;

	c_ds_comp(a, b, &ret);

	return ret;
}

// DD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
static inline int rds_cmp_d(double a[DSSIZE], double b)
{
	int ret;

	c_ds_comp_dd_d(a, b, &ret);

	return ret;
}

// DD sqrt_d
static inline void rds_sqrt_d(double ret[DSSIZE], double a)
{
	static double tmp[DSSIZE];

	c_ds_copy_d(a, tmp);
	c_ds_sqrt(tmp, ret);

	return;
}

// DD fma
// ret = a * b + c
static inline void rds_fma(double ret[DSSIZE], double a[DSSIZE], double b[DSSIZE], double c[DSSIZE])
{
	static double tmp[DSSIZE];

	c_ds_mul(a, b, tmp);
	c_ds_add(tmp, c, ret);

	return;
}

// DD pow
// ret = base^power = exp(power * log(base))
static inline void rds_pow(double ret[DSSIZE], double base[DSSIZE], double power[DSSIZE])
{
	static double tmp[DSSIZE], tmp1[DSSIZE];

	c_ds_log(base, tmp);
	c_ds_mul(power, tmp, tmp1);
	c_ds_exp(tmp1, ret);

	return;
}

#define RDS_CMP(a, b) rds_cmp(a, b)
#define RDS_CMP_D(a, b) rds_cmp_d(a, b)
#define RDS_CMP_UI(a, b) rds_cmp_d(a, (double)(b))
#define RDS_SQRT_D(ret, a) rds_sqrt_d(ret, a)
#define RDS_SQRT_UI(ret, a) rds_sqrt_d(ret, (double)(a))

// QD print(no appending CR)
static inline void rqs_out_str_base(FILE *fp, int base, int length, double val[QSSIZE])
{
	static char str[128];
	// void c_qs_swrite(const double *a, int precision, char *s, int len);
	c_qs_swrite(val, (length > 70) ? 70 : length, str, 80);
	fprintf(fp, "%s", str);
}

// QD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
static inline int rqs_cmp(double a[QSSIZE], double b[QSSIZE])
{
	int ret;

	c_qs_comp(a, b, &ret);

	return ret;
}

// QD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
static inline int rqs_cmp_d(double a[QSSIZE], double b)
{
	int ret;

	c_qs_comp_qd_d(a, b, &ret);

	return ret;
}

// QD sqrt_d
static inline void rqs_sqrt_d(double ret[QSSIZE], double a)
{
	static double tmp[QSSIZE];

	c_qs_copy_d(a, tmp);
	c_qs_sqrt(tmp, ret);

	return;
}

// QD fma
// ret = a * b + c
static inline void rqs_fma(double ret[QSSIZE], double a[QSSIZE], double b[QSSIZE], double c[QSSIZE])
{
	static double tmp[QSSIZE];

	c_qs_mul(a, b, tmp);
	c_qs_add(tmp, c, ret);

	return;
}

// QD pow
// ret = base^power = exp(power * log(base))
static inline void rqs_pow(double ret[QSSIZE], double base[QSSIZE], double power[QSSIZE])
{
	static double tmp[QSSIZE], tmp1[QSSIZE];

	c_qs_log(base, tmp);
	c_qs_mul(power, tmp, tmp1);
	c_qs_exp(tmp1, ret);

	return;
}

#define RQS_CMP(a, b) rqs_cmp(a, b)
#define RQS_CMP_D(a, b) rqs_cmp_d(a, b)
#define RQS_CMP_UI(a, b) rqs_cmp_d(a, (double)(b))
#define RQS_SQRT_D(ret, a) rqs_sqrt_d(ret, a)
#define RQS_SQRT_UI(ret, a) rqs_sqrt_d(ret, (double)(a))

// TD print(no appending CR)
static inline void rts_out_str_base(FILE *fp, int base, int length, double val[TSSIZE])
{
	static char str[128];
	// void c_qs_swrite(const double *a, int precision, char *s, int len);
	c_ts_swrite(val, (length > 70) ? 70 : length, str, 80);
	fprintf(fp, "%s", str);
}

// TD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
static inline int rts_cmp(double a[TSSIZE], double b[TSSIZE])
{
	int ret;

	c_ts_comp(a, b, &ret);

	return ret;
}

// TD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
static inline int rts_cmp_d(double a[TSSIZE], double b)
{
	int ret;

	c_ts_comp_td_d(a, b, &ret);

	return ret;
}

// TD sqrt_d
static inline void rts_sqrt_d(double ret[TSSIZE], double a)
{
	static double tmp[TSSIZE];

	c_ts_copy_d(a, tmp);
	c_ts_sqrt(tmp, ret);

	return;
}

// TD fma
// ret = a * b + c
static inline void rts_fma(double ret[TSSIZE], double a[TSSIZE], double b[TSSIZE], double c[TSSIZE])
{
	static double tmp[TSSIZE];

#ifdef USE_ACCURATE_TD_MUL
	c_ts_mul_accurate(a, b, tmp);
#else // USE_ACCURATE_TD_MUL
	c_ts_mul_sloppy(a, b, tmp);
#endif // USE_ACCURATE_TD_MUL
	c_ts_add(tmp, c, ret);
	//rts_mul(tmp, a, b);
	//rts_add(ret, c, tmp);

	return;
}

// TD pow
// ret = base^power = exp(power * log(base))
static inline void rts_pow(double ret[TSSIZE], double base[TSSIZE], double power[TSSIZE])
{
	static double tmp[TSSIZE], tmp1[TSSIZE];

#if 0
	c_ts_log(base, tmp);
	c_ts_mul(power, tmp, tmp1);
	c_ts_exp(tmp1, ret);
#endif // 0

	return;
}

#define RTS_CMP(a, b) rts_cmp(a, b)
#define RTS_CMP_D(a, b) rts_cmp_d(a, b)
#define RTS_CMP_UI(a, b) rts_cmp_d(a, (double)(b))
#define RTS_SQRT_D(ret, a) rts_sqrt_d(ret, a)
#define RTS_SQRT_UI(ret, a) rts_sqrt_d(ret, (double)(a))


#ifndef USE_RDS_FUNCTIONS
	#define set0_dd(val) SET0_DD(val)
	#define rds_set0(val) SET0_DD(val)
	#define rds_add(ret, a, b) RDS_ADD(ret, a, b)
	#define rds_sub(ret, a, b) RDS_SUB(ret, a, b)
	#define rds_mul(ret, a, b) RDS_MUL(ret, a, b)
	#define rds_div(ret, a, b) RDS_DIV(ret, a, b)
	#define rds_sqrt(ret, a) RDS_SQRT(ret, a)
	#define rds_sqrt_d(ret, a) RDS_SQRT_D(ret, a)
	#define rds_sqrt_ui(ret, a) RDS_SQRT_UI(ret, a)
//	#define rds_out_str(a) RDS_OUT_STR(a)
	#define rds_set_str(str, a) RDS_SET_STR(str, a)
	#define rds_get_str(a, str) RDS_GET_STR(a, str)
	#define rds_get_d(a) RDS_GET_D(a)
	#define rds_set_d(ret, d) RDS_SET_D(ret, d)
	#define rds_set_ui(ret, d) RDS_SET_UI(ret, d)
	#define rds_set(ret, org) RDS_SET(ret, org)
	#define rds_neg(ret, a) RDS_NEG(ret, a)
	#define rds_abs(ret, a) RDS_ABS(ret, a)
	#define rds_cmp_ui(a, b) RDS_CMP_UI(a, b)
	#define rds_ui_div(ret, a, b) RDS_UI_DIV(ret, a, b)
	#define rds_ui_sub(ret, a, b) RDS_UI_SUB(ret, a, b)
	#define rds_div_d(ret, a, b) RDS_DIV_D(ret, a, b)
	#define rds_add_d(ret, a, b) RDS_ADD_D(ret, a, b)
	#define rds_sub_d(ret, a, b) RDS_SUB_D(ret, a, b)
	#define rds_mul_d(ret, a, b) RDS_MUL_D(ret, a, b)
	#define rds_div_ui(ret, a, b) RDS_DIV_UI(ret, a, b)
	#define rds_add_ui(ret, a, b) RDS_ADD_UI(ret, a, b)
	#define rds_sub_ui(ret, a, b) RDS_SUB_UI(ret, a, b)
	#define rds_mul_ui(ret, a, b) RDS_MUL_UI(ret, a, b)

	#define rds_pi(ret) RDS_PI(ret)
	#define rds_exp(ret, x) RDS_EXP(ret, x)
	#define rds_sin(ret, x) RDS_SIN(ret, x)
	#define rds_cos(ret, x) RDS_COS(ret, x)
	#define rds_log(ret, x) RDS_LOG(ret, x)
	#define rds_asin(ret, x) RDS_ASIN(ret, x)
	#define rds_acos(ret, x) RDS_ACOS(ret, x)
#endif // USE_RDS_FUNCTIONS

#ifndef USE_RTS_FUNCTIONS
	#define set0_td(val) SET0_TD(val)
	#define rts_set0(val) SET0_TD(val)
	#define rts_add(ret, a, b) RTS_ADD(ret, a, b) // default -> RTS_ADDQ
	#define rts_addt(ret, a, b) RTS_ADDT(ret, a, b)
	#define rts_addq(ret, a, b) RTS_ADDQ(ret, a, b)
	#define rts_sub(ret, a, b) RTS_SUB(ret, a, b) // default -> RTS_SUBQ
	#define rts_subt(ret, a, b) RTS_SUBT(ret, a, b)
	#define rts_subq(ret, a, b) RTS_SUBQ(ret, a, b)
	#define rts_mul(ret, a, b) RTS_MUL(ret, a, b)
	#define rts_divt(ret, a, b) RTS_DIVT(ret, a, b)
	#define rts_divtq(ret, a, b) RTS_DIVTQ(ret, a, b)
	#define rts_divq(ret, a, b) RTS_DIVQ(ret, a, b)
	#define rts_div(ret, a, b) RTS_DIV(ret, a, b) // default -> RTS_DIVQ
	#define rts_sqrt(ret, a) RTS_SQRT(ret, a)
	#define rts_sqrt_d(ret, a) RTS_SQRT_D(ret, a)
	#define rts_sqrt_ui(ret, a) RTS_SQRT_UI(ret, a)
//	#define rts_out_str(a) RTS_OUT_STR(a)
	#define rts_set_str(str, a) RTS_SET_STR(str, a)
	#define rts_get_str(a, str) RTS_GET_STR(a, str)
	#define rts_get_d(a) RTS_GET_D(a)
	#define rts_set_d(ret, d) RTS_SET_D(ret, d)
	#define rts_set_ui(ret, d) RTS_SET_UI(ret, d)
	#define rts_set(ret, org) RTS_SET(ret, org)
	#define rts_neg(ret, a) RTS_NEG(ret, a)
	#define rts_abs(ret, a) RTS_ABS(ret, a)
	#define rts_cmp_ui(a, b) RTS_CMP_UI(a, b)
	#define rts_ui_div(ret, a, b) RTS_UI_DIV(ret, a, b)
	#define rts_ui_sub(ret, a, b) RTS_UI_SUB(ret, a, b)
	#define rts_div_d(ret, a, b) RTS_DIV_D(ret, a, b)
	#define rts_add_d(ret, a, b) RTS_ADD_D(ret, a, b)
	#define rts_sub_d(ret, a, b) RTS_SUB_D(ret, a, b)
	#define rts_mul_d(ret, a, b) RTS_MUL_D(ret, a, b)
	#define rts_div_ui(ret, a, b) RTS_DIV_UI(ret, a, b)
	#define rts_add_ui(ret, a, b) RTS_ADD_UI(ret, a, b)
	#define rts_sub_ui(ret, a, b) RTS_SUB_UI(ret, a, b)
	#define rts_mul_ui(ret, a, b) RTS_MUL_UI(ret, a, b)

	#define rts_pi(ret) RTS_PI(ret)
	#define rts_exp(ret, x) RTS_EXP(ret, x)
	#define rts_sin(ret, x) RTS_SIN(ret, x)
	#define rts_cos(ret, x) RTS_COS(ret, x)
	#define rts_log(ret, x) RTS_LOG(ret, x)
	#define rts_asin(ret, x) RTS_ASIN(ret, x)
	#define rts_acos(ret, x) RTS_ACOS(ret, x)
#endif // USE_RTS_FUNCTIONS


#ifndef USE_RQS_FUNCTIONS
	#define set0_qd(val) SET0_QD(val)
	#define rqs_set0(val) SET0_QD(val)
	#define rqs_add(ret, a, b) RQS_ADD(ret, a, b)
	#define rqs_sub(ret, a, b) RQS_SUB(ret, a, b)
	#define rqs_mul(ret, a, b) RQS_MUL(ret, a, b)
	#define rqs_div(ret, a, b) RQS_DIV(ret, a, b)
	#define rqs_sqrt(ret, a) RQS_SQRT(ret, a)
	#define rqs_sqrt_d(ret, a) RQS_SQRT_D(ret, a)
	#define rqs_sqrt_ui(ret, a) RQS_SQRT_UI(ret, a)
//	#define rqs_out_str(a) RQS_OUT_STR(a)
	#define rqs_set_str(str, a) RQS_SET_STR(str, a)
	#define rqs_get_str(a, str) RQS_GET_STR(a, str)
	#define rqs_get_d(a) RQS_GET_D(a)
	#define rqs_set_d(ret, d) RQS_SET_D(ret, d)
	#define rqs_set_ui(ret, d) RQS_SET_UI(ret, d)
	#define rqs_set(ret, org) RQS_SET(ret, org)
	#define rqs_neg(ret, a) RQS_NEG(ret, a)
	#define rqs_abs(ret, a) RQS_ABS(ret, a)
	#define rqs_cmp_ui(a, b) RQS_CMP_UI(a, b)
	#define rqs_ui_div(ret, a, b) RQS_UI_DIV(ret, a, b)
	#define rqs_ui_sub(ret, a, b) RQS_UI_SUB(ret, a, b)
	#define rqs_div_d(ret, a, b) RQS_DIV_D(ret, a, b)
	#define rqs_add_d(ret, a, b) RQS_ADD_D(ret, a, b)
	#define rqs_sub_d(ret, a, b) RQS_SUB_D(ret, a, b)
	#define rqs_mul_d(ret, a, b) RQS_MUL_D(ret, a, b)
	#define rqs_div_ui(ret, a, b) RQS_DIV_UI(ret, a, b)
	#define rqs_add_ui(ret, a, b) RQS_ADD_UI(ret, a, b)
	#define rqs_sub_ui(ret, a, b) RQS_SUB_UI(ret, a, b)
	#define rqs_mul_ui(ret, a, b) RQS_MUL_UI(ret, a, b)

	#define rqs_pi(ret) RQS_PI(ret)
	#define rqs_exp(ret, x) RQS_EXP(ret, x)
	#define rqs_sin(ret, x) RQS_SIN(ret, x)
	#define rqs_cos(ret, x) RQS_COS(ret, x)
	#define rqs_log(ret, x) RQS_LOG(ret, x)
	#define rqs_asin(ret, x) RQS_ASIN(ret, x)
	#define rqs_acos(ret, x) RQS_ACOS(ret, x)
#endif // USE_RQS_FUNCTIONS

#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus

#endif // __BNC_RDS_H_

