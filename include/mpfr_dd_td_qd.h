// ---------------------------------------------------------------------------------
// mpfr_dd_td_qd.h
// These following codes are originally written by members of MPFR development team.
// Modified by Tomonori Kouya
// All codes can be used under MPFR's license.
// ---------------------------------------------------------------------------------
#ifndef __MPFR_DD_QD_H__
#define __MPFR_DD_QD_H__

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include <stdio.h>
#include <math.h>
#include <float.h>

#ifndef DBL_MAX
	#define DBL_MAX 3.402823e+308
#endif // DBL_MAX

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

// FALSE(=0), TRUE(=1)
//#define MPFR_IS_SINGULAR(x) (MPFR_EXP(x) <= MPFR_EXP_INF)
int mpfr_is_singular(mpfr_srcptr x);

/* generic code */
// ret_dd[2] = ret_dd[high == 0], ret_dd[low == 1]
//#define mpf_get_dd(ret_dd, x) mpfr_get_dd(ret_dd, x, MPFR_RNDN)
void mpfr_get_dd(double ret_dd[2], mpfr_srcptr x, mpfr_rnd_t rnd_mode);
//void mpf_get_dd(double ret_dd[2], mpf_srcptr x);

/* double-double code */
//#define mpf_set_dd(r, d) mpfr_set_dd(r, d, MPFR_RNDN)
int mpfr_set_dd(mpfr_ptr r, double d[2], mpfr_rnd_t rnd_mode);
//int mpf_set_dd(mpf_ptr r, double d[2]);

/* generic code */
// Triple-double precision
// ret_qd[3] = ret_qd[high == 0], ret_qd[1], ret_qd[2]
//#define mpf_get_td(ret_td, x) mpfr_get_td(ret_td, x, MPFR_RNDN)
void mpfr_get_td(double ret_td[3], mpfr_srcptr x, mpfr_rnd_t rnd_mode);
//void mpf_get_td(double ret_td[3], mpf_srcptr x);

/* Triple-double code */
//#define mpf_set_td(r, d) mpfr_set_td(r, d, MPFR_RNDN)
int mpfr_set_td(mpfr_ptr r, double d[3], mpfr_rnd_t rnd_mode);
//int mpf_set_t(mpf_ptr r, double d[3]);

/* generic code */
// ret_qd[4] = ret_qd[high == 0], ret_qd[1], ret_qd[2], ret_qd[3]
//#define mpf_get_qd(ret_qd, x) mpfr_get_qd(ret_qd, x, MPFR_RNDN)
void mpfr_get_qd(double ret_qd[4], mpfr_srcptr x, mpfr_rnd_t rnd_mode);
//void mpf_get_qd(double ret_qd[4], mpf_srcptr x);

/* double-double code */
//#define mpf_set_qd(r, d) mpfr_set_qd(r, d, MPFR_RNDN)
int mpfr_set_qd(mpfr_ptr r, double d[4], mpfr_rnd_t rnd_mode);
//int mpf_set_qd(mpf_ptr r, double d[4]);

// mpf to dd
//void mpf_get_dd(double ret[DDSIZE], mpf_t val);
#define mpf_get_dd(ret, val) mpfr_get_dd(ret, val, MPFR_RNDN)

// dd to mpf
//void mpf_set_dd(mpf_t ret, double val[DDSIZE]);
#define mpf_set_dd(ret, val) mpfr_set_dd(ret, val, MPFR_RNDN)

// mpf to qd
void mpf_get_qd(double ret[QDSIZE], mpf_t val);
#define mpf_get_qd(ret, val) mpfr_get_qd(ret, val, MPFR_RNDN)

// qd to mpf
//void mpf_set_qd(mpf_t ret, double val[QDSIZE]);
#define mpf_set_qd(ret, val) mpfr_set_qd(ret, val, MPFR_RNDN)

// rdd_out_str(dd_a)
void rdd_out_str(double dd[DDSIZE]);

// rtd_out_str(td_a)
void rtd_out_str(double td[TDSIZE]);

// rqd_out_str(td_a)
void rqd_out_str(double qd[QDSIZE]);

#ifdef __cplusplus
} //end of extern "C" {
#endif // __cplusplus

#endif // __MPFR_DD_QD_H__
