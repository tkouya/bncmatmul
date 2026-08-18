// ---------------------------------------------------------------------------------
// mpfr_dtq_sd.h
// These following codes are originally written by members of MPFR development team.
// Modified by Tomonori Kouya
// All codes can be used under MPFR's license.
// ---------------------------------------------------------------------------------
#ifndef __MPFR_DTQ_SD_H__
#define __MPFR_DTQ_SD_H__

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include <stdio.h>
#include <math.h>
#include <float.h>

// Common defs
#include "bnc_common.h"

#ifdef USE_GMP

#ifndef FLT_MAX
	#define FLT_MAX 3.402823e+38
#endif // FLT_MAX

#ifndef DBL_MAX
	#define DBL_MAX 1.797693e+308 //3.402823e+308
#endif // DBL_MAX

// dsfloat, tsfloat, qsfloat
#ifndef DSSIZE
	#define DSSIZE 2
#endif // DSSIZE
#ifndef TSSIZE
	#define TSSIZE 3
#endif // TSSIZE
#ifndef QSSIZE
	#define QSSIZE 4
#endif // QSSIZE

// ddfloat, tdfloat, qdfloat
#ifndef DDSIZE
	#define DDSIZE 2
#endif // DDSIZE
#ifndef TDSIZE
	#define TDSIZE 3
#endif // TDSIZE
#ifndef QDSIZE
	#define QDSIZE 4
#endif // QDSIZE

#include "mpfr.h"
#include "mpc.h" // 2025-06-26(Thu)

// based on get_ld.c code in MPFR 4.0.0 
#define _IN_H_MPFR_EXP_ZERO (mpfr_get_emin() + 1)
#define _IN_H_MPFR_EXP_NAN  (mpfr_get_emin() + 2)
#define _IN_H_MPFR_EXP_INF  (mpfr_get_emin() + 3)
#define _IN_H_MPFR_EXP_UBF  (mpfr_get_emin() + 4)

#define _IN_H_MPFR_SIGN(x) ((x)->_mpfr_sign)
#define _IN_H_MPFR_SIGN_POS (1)
#define _IN_H_MPFR_SIGN_NEG (-1)

#define _IN_H_MPFR_IS_NEG(x) (MPFR_SIGN(x) < 0)
#define _IN_H_MPFR_IS_POS(x) (MPFR_SIGN(x) > 0)

#define _IN_H_MPFR_SET_POS(x) (MPFR_SIGN(x) = MPFR_SIGN_POS)
#define _IN_H_MPFR_SET_NEG(x) (MPFR_SIGN(x) = MPFR_SIGN_NEG)

// from mpf2mpfr.h
#ifndef MPFR_DEFAULT_RND
	#define MPFR_DEFAULT_RND mpfr_get_default_rounding_mode()
#endif // MPFR_DEFAULT_RND

// MPC_DEFAULT_RND
#ifndef MPC_DEFAULT_RND
	#define MPC_DEFAULT_RND get_bnc_default_rounding_mode_c()
#endif // MPC_DEFAULT_RND

// FALSE(=0), TRUE(=1)
//#define MPFR_IS_SINGULAR(x) (MPFR_EXP(x) <= MPFR_EXP_INF)
int mpfr_is_singular(mpfr_srcptr x);

/* generic code */
// ret_dd[2] = ret_dd[high == 0], ret_dd[low == 1]
//#define mpf_get_dd(ret_dd, x) mpfr_get_dd(ret_dd, x, MPFR_RNDN)
void mpfr_get_dd(double ret_dd[2], mpfr_srcptr x, mpfr_rnd_t rnd_mode);
//void mpf_get_dd(double ret_dd[2], mpf_srcptr x);

// ret_ds[2] = ret_ds[high == 0], ret_ds[low == 1]
void mpfr_get_ds(float ret_ds[2], mpfr_srcptr x, mpfr_rnd_t rnd_mode);

/* double-double code */
//#define mpf_set_dd(r, d) mpfr_set_dd(r, d, MPFR_RNDN)
int mpfr_set_dd(mpfr_ptr r, const double d[2], mpfr_rnd_t rnd_mode);
//int mpf_set_dd(mpf_ptr r, double d[2]);

/* double-sigle code */
int mpfr_set_ds(mpfr_ptr r, const float d[2], mpfr_rnd_t rnd_mode);

/* generic code */
// Triple-double precision
// ret_qd[3] = ret_qd[high == 0], ret_qd[1], ret_qd[2]
//#define mpf_get_td(ret_td, x) mpfr_get_td(ret_td, x, MPFR_RNDN)
void mpfr_get_td(double ret_td[3], mpfr_srcptr x, mpfr_rnd_t rnd_mode);
//void mpf_get_td(double ret_td[3], mpf_srcptr x);

/* generic code */
// Triple-single precision
// ret_ts[3] = ret_ts[high == 0], ret_ts[1], ret_ts[2]
//#define mpf_get_ts(ret_ts, x) mpfr_get_ts(ret_ts, x, MPFR_RNDN)
void mpfr_get_ts(float ret_ts[3], mpfr_srcptr x, mpfr_rnd_t rnd_mode);

/* Triple-double code */
//#define mpf_set_td(r, d) mpfr_set_td(r, d, MPFR_RNDN)
int mpfr_set_td(mpfr_ptr r, const double d[3], mpfr_rnd_t rnd_mode);
//int mpf_set_t(mpf_ptr r, double d[3]);

//* Triple-single code */
//#define mpf_set_ts(r, d) mpfr_set_ts(r, d, MPFR_RNDN)
int mpfr_set_ts(mpfr_ptr r, const float s[3], mpfr_rnd_t rnd_mode);

/* generic code */
// ret_qd[4] = ret_qd[high == 0], ret_qd[1], ret_qd[2], ret_qd[3]
//#define mpf_get_qd(ret_qd, x) mpfr_get_qd(ret_qd, x, MPFR_RNDN)
void mpfr_get_qd(double ret_qd[4], mpfr_srcptr x, mpfr_rnd_t rnd_mode);
//void mpf_get_qd(double ret_qd[4], mpf_srcptr x);

/* generic code */
// ret_qs[4] = ret_qs[high == 0], ret_qs[1], ret_qs[2], ret_qs[3]
//#define mpf_get_qs(ret_qs, x) mpfr_get_qs(ret_qs, x, MPFR_RNDN)
void mpfr_get_qs(float ret_qs[4], mpfr_srcptr x, mpfr_rnd_t rnd_mode);

/* double-double code */
//#define mpf_set_qd(r, d) mpfr_set_qd(r, d, MPFR_RNDN)
int mpfr_set_qd(mpfr_ptr r, const double d[4], mpfr_rnd_t rnd_mode);
//int mpf_set_qd(mpf_ptr r, double d[4]);

/* quad-single code */
//#define mpf_set_qs(r, d) mpfr_set_qs(r, d, MPFR_RNDN)
int mpfr_set_qs(mpfr_ptr r, const float d[4], mpfr_rnd_t rnd_mode);

// mpf to dd, ds
//void mpf_get_dd(double ret[DDSIZE], mpf_t val);
#define mpf_get_dd(ret, val) mpfr_get_dd(ret, val, MPFR_RNDN)
#define mpf_get_ds(ret, val) mpfr_get_ds(ret, val, MPFR_RNDN)

// dd, ds to mpf
//void mpf_set_dd(mpf_t ret, double val[DDSIZE]);
#define mpf_set_dd(ret, val) mpfr_set_dd(ret, val, MPFR_RNDN)
#define mpf_set_ds(ret, val) mpfr_set_ds(ret, val, MPFR_RNDN)

// mpf to td, ts
//void mpf_get_td(double ret[TDSIZE], mpf_t val);
#define mpf_get_td(ret, val) mpfr_get_td(ret, val, MPFR_RNDN)
#define mpf_get_ts(ret, val) mpfr_get_ts(ret, val, MPFR_RNDN)

// td, ts to mpf
//void mpf_set_td(mpf_t ret, double val[TDSIZE]);
#define mpf_set_td(ret, val) mpfr_set_td(ret, val, MPFR_RNDN)
#define mpf_set_ts(ret, val) mpfr_set_ts(ret, val, MPFR_RNDN)

// mpf to qd
//void mpf_get_qd(double ret[QDSIZE], mpf_t val);
#define mpf_get_qd(ret, val) mpfr_get_qd(ret, val, MPFR_RNDN)
#define mpf_get_qs(ret, val) mpfr_get_qs(ret, val, MPFR_RNDN)

// qd to mpf
//void mpf_set_qd(mpf_t ret, double val[QDSIZE]);
#define mpf_set_qd(ret, val) mpfr_set_qd(ret, val, MPFR_RNDN)
#define mpf_set_qs(ret, val) mpfr_set_qs(ret, val, MPFR_RNDN)

// 2025-06-26(Thu)
// -------------------------
// mpc to cdd, ctd, cqdfloat
// -------------------------
// MPC -> cddfloat
void mpc_get_cdd(cddfloat *ret, mpc_t org);
// cddfloat -> MPC
void mpc_set_cdd(mpc_t ret, cddfloat *org);
// MPC -> ctdfloat
void mpc_get_ctd(ctdfloat *ret, mpc_t org);
// ctdfloat -> MPC
void mpc_set_ctd(mpc_t ret, ctdfloat *org);
// MPC -> cqdfloat
void mpc_get_cqd(cqdfloat *ret, mpc_t org);
// cqdfloat -> MPC
void mpc_set_cqd(mpc_t ret, cqdfloat *org);

// rdd_out_str(dd_a)
void rdd_out_str(double dd[DDSIZE]);
void rds_out_str(float ds[DSSIZE]);

// rtd_out_str(td_a)
void rtd_out_str(double td[TDSIZE]);
void rts_out_str(float ts[TSSIZE]);

// rqd_out_str(td_a)
void rqd_out_str(double qd[QDSIZE]);
void rqs_out_str(float qs[QSSIZE]);

// 2021-07-15(Thu) Tomonori Kouya
void rdd_set_str(double ret[DDSIZE], const char *str);
void rdd_get_str(char *str, const double val[DDSIZE]);
void rtd_set_str(double ret[TDSIZE], const char *str);
void rtd_get_str(char *str, const double val[TDSIZE]);
void rqd_set_str(double ret[QDSIZE], const char *str);
void rqd_get_str(char *str, const double val[QDSIZE]);

// 2022-11-17
// ret := a^b
void rdd_pow_mpfr(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]);
void rtd_pow_mpfr(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]);
void rqd_pow_mpfr(double ret[QDSIZE], double a[QDSIZE], double b[QDSIZE]);

// 2023-06-02 (Fri)  T.Kouya
// ret := exp(x)
void rdd_exp_mpfr(double ret[DDSIZE], double x[DDSIZE]);
void rtd_exp_mpfr(double ret[TDSIZE], double x[TDSIZE]);
void rqd_exp_mpfr(double ret[QDSIZE], double x[QDSIZE]);

// 2024-04-23 (Tue)
// ret := sqrt(x)
void rdd_sqrt_mpfr(double ret[DDSIZE], double x[DDSIZE]);
void rtd_sqrt_mpfr(double ret[TDSIZE], double x[TDSIZE]);
void rqd_sqrt_mpfr(double ret[QDSIZE], double x[QDSIZE]);
// ret := sqrt_d((double)x)
void rdd_sqrt_d_mpfr(double ret[DDSIZE], double x);
void rtd_sqrt_d_mpfr(double ret[TDSIZE], double x);
void rqd_sqrt_d_mpfr(double ret[QDSIZE], double x);


// 2025-02-03 (Mon)
// ret := mpfr_func(x)
void rdd_func_mpfr(double ret[DDSIZE], int (* mpfr_func)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t), double x[DDSIZE]);
void rtd_func_mpfr(double ret[TDSIZE], int (* mpfr_func)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t), double x[TDSIZE]);
void rqd_func_mpfr(double ret[QDSIZE], int (* mpfr_func)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t), double x[QDSIZE]);

// 2025-06-26 (Thu)
// ret := mpc_func(x)
void rcdd_func_mpc(cddfloat *ret, int (* mpc_func)(mpc_ptr, mpc_srcptr, mpc_rnd_t), cddfloat *x);
void rctd_func_mpc(ctdfloat *ret, int (* mpc_func)(mpc_ptr, mpc_srcptr, mpc_rnd_t), ctdfloat *x);
void rcqd_func_mpc(cqdfloat *ret, int (* mpc_func)(mpc_ptr, mpc_srcptr, mpc_rnd_t), cqdfloat *x);

// mpc_arg(x)
void rcdd_arg(double ret[DDSIZE], cddfloat *x);
void rctd_arg(double ret[TDSIZE], ctdfloat *x);
void rcqd_arg(double ret[QDSIZE], cqdfloat *x);

// 2025-06-26(Thu)
// r[dtq]d_const_pi
void rdd_const_pi(double ret[DDSIZE]);
void rtd_const_pi(double ret[TDSIZE]);
void rqd_const_pi(double ret[QDSIZE]);

#if 0
// bug fix for r[dtq]_sqrt and _out_str
// 2025-02-06(Thu) T.Kouya
#ifndef USE_RDD_SQRT
#define rdd_sqrt rdd_sqrt_mpfr
#define rdd_sqrt_d rdd_sqrt_d_mpfr
#endif // USE_RDD_SQRT

#ifndef USE_RTD_SQRT
#define rtd_sqrt rtd_sqrt_mpfr
#define rtd_sqrt_d rtd_sqrt_d_mpfr
#endif // USE_RDD_SQRT

#ifndef USE_RQD_SQRT
#define rqd_sqrt rqd_sqrt_mpfr
#define rqd_sqrt_d rqd_sqrt_d_mpfr
#endif // USE_RDD_SQRT
#endif // 0

#endif // USE_GMP

#ifdef __cplusplus
} //end of extern "C" {
#endif // __cplusplus

#endif // __MPFR_DTQ_SD_H__
