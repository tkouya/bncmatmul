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

/* ==== begin: former c_dd_qd.h (absorbed into rdd.h, 2026-08) ==== */
/********************************************************************************/
/*                                                                              */
/* c_dd_qd.h : Primary double-double, triple-double and quadruple-double        */
/*             floating-point arithmetic                                        */
/* Copyright (c) 2015-2026 Tomonori Kouya, All rights reserved.                 */
/*                                                                              */
/* 2019-06-14 appended static inline to main functions                          */
/* 2015-03-01 Translated to pure C code by T.Kouya                              */
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
/*
 * src/c_dd.cc
 *
 * This work was supported by the Director, Office of Science, Division
 * of Mathematical, Information, and Computational Sciences of the
 * U. Department of Energy under contract number DE-AC03-76SF00098.
 *
 * Copyright (c) 2000-2001
 *
 * Contains the C wrapper functions for double-double precision arithmetic.
 * This can be used from Fortran code.
 */

#ifndef __C_DD_QD_H
	#define __C_DD_QD_H

#define _QD_SPLITTER 134217729.0               // = 2^27 + 1
#define _QD_SPLIT_THRESH 6.69692879491417e+299 // = 2^996

#ifdef QD_VACPP_BUILTINS_H
/* For VisualAge C++ __fmadd */
#include <builtins.h>
#endif // QD_VACPP_BUILDINS_H

#ifndef __cpluplus
#include <math.h>
#else // __cplusplus
#include <cmath>
#endif // __cplusplus

// return a * b + c
#define DFMA(a, b, c) fma((a), (b), (c)) // double precision
#define SFMA(a, b, c) fmaf((a), (b), (c)) // sigle precision

// return a * b + c
#ifndef QD_FMA
#define QD_FMA(a, b, c) DFMA((a), (b), (c))
#endif // QD_FMA

// return a * b - c
#ifndef QD_FMS
#define QD_FMS(a, b, c) DFMA((a), (b), (-(c)))
#endif // QD_FMS

#include <limits.h>

#include <float.h>
#ifndef DBL_EPSILON
	#define DBL_EPSILON (1.1102230246251565404236316680908e-16) // 1/2 * 2^(-52) = 2^(-53)
#endif // DBL_EPSILON
#ifndef FLT_EPSILON
	#define FLT_EPSILON (5.9604644775390625e-8) // 1/2 * 2^(-23) = 2^(-24)
#endif // FLT_EPSILON

// DDSIZE, TDSIZE, QDSIZE
#ifndef DDSIZE
	#define DDSIZE 2
#endif // DDSIZE

#ifndef TDSIZE
	#define TDSIZE 3
#endif // TDSIZE

#ifndef QDSIZE
	#define QDSIZE 4
#endif // QDSIZE

#define TO_DOUBLE_PTR(a, ptr) ptr[0] = a.x[0]; ptr[1] = a.x[1];

#ifdef __cplusplus
extern "C" {
#endif // __cplusplos

#ifdef X86
#ifdef  _WIN32
#include <float.h>
#else

#ifdef HAVE_FPU_CONTROL_H
#include <fpu_control.h>
#endif

#ifndef _FPU_GETCW
#define _FPU_GETCW(x) asm volatile ("fnstcw %0":"=m" (x));
#endif

#ifndef _FPU_SETCW
#define _FPU_SETCW(x) asm volatile ("fldcw %0": :"m" (x));
#endif

#ifndef _FPU_EXTENDED
#define _FPU_EXTENDED 0x0300
#endif

#ifndef _FPU_DOUBLE
#define _FPU_DOUBLE 0x0200
#endif

#endif
#endif /* X86 */

#ifndef __cplusplus
#ifndef FPU_FIX_START
#define FPU_FIX_START
static inline void fpu_fix_start(unsigned int *old_cw) {
#ifdef X86
#ifdef _WIN32
#ifdef __BORLANDC__
  /* Win 32 Borland C */
  unsigned short cw = _control87(0, 0);
  _control87(0x0200, 0x0300);
  if (old_cw) {
    *old_cw = cw;
  }
#else
  /* Win 32 MSVC */
  unsigned int cw = _control87(0, 0);
  _control87(0x00010000, 0x00030000);
  if (old_cw) {
    *old_cw = cw;
  }
#endif
#else
  /* Linux */
  volatile unsigned short cw, new_cw;
  _FPU_GETCW(cw);

  new_cw = (cw & ~_FPU_EXTENDED) | _FPU_DOUBLE;
  _FPU_SETCW(new_cw);
  
  if (old_cw) {
    *old_cw = cw;
  }
#endif
#endif
}

static inline void fpu_fix_end(unsigned int *old_cw) {
#ifdef X86
#ifdef _WIN32

#ifdef __BORLANDC__
  /* Win 32 Borland C */
  if (old_cw) {
    unsigned short cw = (unsigned short) *old_cw;
    _control87(cw, 0xFFFF);
  }
#else // __BORLANDC__
  /* Win 32 MSVC */
  if (old_cw) {
    _control87(*old_cw, 0xFFFFFFFF);
  }
#endif // __BORLANDC__

#else // _WIN32
  /* Linux */
  if (old_cw) {
    int cw;
    cw = *old_cw;
    _FPU_SETCW(cw);
  }
#endif // _WIN32
#endif// X86
}

#ifdef HAVE_FORTRAN

#define f_fpu_fix_start FC_FUNC_(f_fpu_fix_start, F_FPU_FIX_START)
#define f_fpu_fix_end   FC_FUNC_(f_fpu_fix_end,   F_FPU_FIX_END)

static inline void f_fpu_fix_start(unsigned int *old_cw) {
  fpu_fix_start(old_cw);
}

static inline void f_fpu_fix_end(unsigned int *old_cw) {
  fpu_fix_end(old_cw);
}

#endif // HAVE_FORTRANB
#endif // FPU_FIX_START
#endif // __cplusplus


/**************************************/
/* DD : Double-double precision       */
/**************************************/
//  double _hi() const { return x[0]; }
//  double _lo() const { return x[1]; }
#define DD_HI(a)	((a)[0])
#define DD_LOW(a)	((a)[1])

// Definitions for Comparing Functions
typedef unsigned int	dd_bool;
#define DD_TRUE		(1UL)
#define DD_FALSE	(0UL)

#define DD_ISNAN(a) (isnan((a)[0]))
#define DD_ISINF(a) (isinf((a)[0]))
#define DD_ISZERO(a) ((a)[0] == 0.0)
#define DD_ISONE(a) (((a)[0] == 1.0) && ((a)[1] == 0.0))
#define DD_ISNEGATIVE(a) ((a)[0] < 0.0)
#define DD_NAN (FP_NAN)

// PI = 3.1415...
// (fixed 2026-08-14: the previous value here was pi/16, not pi)
static const double const_dd_pi[2] = {3.141592653589793116e+00, 1.224646799147353207e-16};
static const double const_dd_zero[2] = {0.0, 0.0};
static const double const_dd_one[2] = {1.0, 0.0};

static inline double to_double(double a) { return a; }
static inline int to_int(double a) { return (int)(a); }

/**************************************/
/* TD : Triple-double precision       */
/**************************************/
#define TD_HI(a)	((a)[0])
#define TD_LOW(a)	((a)[1])

// Definitions for Comparing Functions
typedef unsigned int	td_bool;
#define TD_TRUE		(1UL)
#define TD_FALSE	(0UL)

#define TD_ISNAN(a) (isnan((a)[0]))
#define TD_ISINF(a) (isinf((a)[0]))
#define TD_ISZERO(a) ((a)[0] == 0.0)
#define TD_ISNEGATIVE(a) ((a)[0] < 0.0)
#define TD_NAN (FP_NAN)

// PI = 3.1415...
// (fixed 2026-08-14: the previous value here was pi/1024, not pi)
static const double const_td_pi[3] = {3.141592653589793116e+00, 1.224646799147353207e-16, -2.994769809718339666e-33};

/**************************************/
/* QD : Quadruple-double precision    */
/**************************************/
#define QD_HI(a)	((a)[0])
#define QD_LOW(a)	((a)[3])

// Definitions for Comparing Functions
typedef unsigned int	qd_bool;
#define QD_TRUE		(1UL)
#define QD_FALSE	(0UL)

#ifndef __cplusplus
#define QD_ISINF(a) (isinf((a)[0]))
#define QD_ISNAN(a) (isnan((a)[0]))
#endif // ifndef __cplusplus
#define QD_ISZERO(a) ((a)[0] == 0.0)
#define QD_ISNEGATIVE(a) ((a)[0] < 0.0)
#define QD_NAN (FP_NAN)

// PI = 3.1415...
// (fixed 2026-08-14: the previous value here was pi/1024, not pi)
static const double const_qd_pi[4] = {3.141592653589793116e+00, 1.224646799147353207e-16, -2.994769809718339666e-33, 1.112454220863365282e-49};

/*********** Basic Functions ************ start **/
/* Computes fl(a+b) and err(a+b).  Assumes |a| >= |b|. */
static inline double quick_two_sum(double a, double b, double *err)
{
	double s;
	s = a + b;
	*err = b - (s - a);
	return s;
}

/* Computes fl(a-b) and err(a-b).  Assumes |a| >= |b| */
static inline double quick_two_diff(double a, double b, double *err)
{
	double s;
	s = a - b;
	*err = (a - s) - b;
	return s;
}

/* Computes fl(a+b) and err(a+b).  */
static inline double two_sum(double a, double b, double *err)
{
	double s;
	s = a + b;
	double bb;
	bb = s - a;
	*err = (a - (s - bb)) + (b - bb);
	return s;
}

/* Computes fl(a-b) and err(a-b).  */
static inline double two_diff(double a, double b, double *err)
{
	double s;
	s = a - b;
	double bb;
	bb = s - a;
	*err = (a - (s - bb)) - (b + bb);
	return s;
}

#ifndef QD_FMS
/* Computes high word and lo word of a */
static inline void split(double a, double *hi, double *lo)
{
	double temp;

	if (a > _QD_SPLIT_THRESH || a < -_QD_SPLIT_THRESH)
	{
		a *= 3.7252902984619140625e-09;  // 2^-28
		temp = _QD_SPLITTER * a;
		*hi = temp - (temp - a);
		*lo = a - (*hi);
		*hi *= 268435456.0;          // 2^28
		*lo *= 268435456.0;          // 2^28
	}
	else
	{
		temp = _QD_SPLITTER * a;
		*hi = temp - (temp - a);
		*lo = a - (*hi);
	}
}
#endif // QD_FMS

/* Computes fl(a*b) and err(a*b). */
static inline double two_prod(double a, double b, double *err)
{
#ifdef QD_FMS
	double p;
	p = a * b;

	*err = QD_FMS(a, b, p);

	return p;
#else  // QD_FMS
	double a_hi, a_lo, b_hi, b_lo;
	double p;
	p = a * b;

	split(a, &a_hi, &a_lo);
	split(b, &b_hi, &b_lo);
	*err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;

	return p;
#endif // QD_FMS
}

/* Computes fl(a*a) and err(a*a).  Faster than the above method. */
static inline double two_sqr(double a, double *err)
{
#ifdef QD_FMS
	double p;
	p = a * a;

	*err = QD_FMS(a, a, p);

	return p;
#else // QD_FMS
	double hi, lo;
	double q;
	q = a * a;

	split(a, &hi, &lo);
	*err = ((hi * hi - q) + 2.0 * hi * lo) + lo * lo;

	return q;
#endif // QD_FMS
}

/* Computes the nearest integer to d. */
//static inline double nint(double d)
static inline double _bnc_nint(double d)
{
//  if (d == std::floor(d))
	if (d == floor(d))
		return d;
	return floor(d + 0.5);
}

/* Computes the truncated integer. */
static inline double aint(double d)
{
//  return (d >= 0.0) ? std::floor(d) : std::ceil(d);
	return (d >= 0.0) ? floor(d) : ceil(d);
}

/* These are provided to give consistent 
   interface for double with double-double and quad-double. */
static inline void sincosh(double t, double *sinh_t, double *cosh_t)
{
	*sinh_t = sinh(t);
	*cosh_t = cosh(t);
}

static inline double sqr(double t)
{
	return t * t;
}

// 2026-02-24(Tue) (d, e) := FastTwoFMA(a, b, c)
// a * b + c = d + e
// |c| > 2|ab|
static inline double fast_two_fma(const double a, const double b, const double c, double *e)
{
	double d, t;

	d = DFMA(a, b, c);
	t = c - d;
	*e = DFMA(a, b, t);

	return d;
}

// 2026-03-21(Mon) (z, e) := FMAerror(a, b, c)
// a * b + c = z + e
static inline double fma_error(const double a, const double b, const double c, double *e)
{
	double z, t, p[DDSIZE], u[DDSIZE];

	z = DFMA(a, b, c);
	p[0] = two_prod(a, b, &(p[1]));
	u[0] = two_sum(c, p[0], &(u[1]));
	t = u[0] - z;
	*e = t + (p[1] + u[1]);

	return z;
}
/*********** Basic Functions ************ ended **/

// -------------------------------------------------
// -------------------- DD -------------------------
// -------------------------------------------------
/*********** Micellaneous ************/
/*  this == 0 */
static inline dd_bool dd_is_zero(const double *x)
{
//  return (x[0] == 0.0);
	if(x[0] == 0.0)
		return DD_TRUE;
	else
		return DD_FALSE;
}

/*  this == 1 */
static inline dd_bool dd_is_one(const double *x)
{
	//return (x[0] == 1.0 && x[1] == 0.0);
	if(x[0] == 1.0 && x[1] == 0.0)
		return DD_TRUE;
	else
		return DD_FALSE;
}

// ret := x
static inline void c_dd_set(const double *x, double *ret)
{
	ret[0] = x[0];
	ret[1] = x[1];
}

// ret := (double)x
static inline void c_dd_set_dd_d(const double x, double *ret)
{
	ret[0] = x;
	ret[1] = 0.0;
}

// ret := 0
static inline void c_dd_set0(double *ret)
{
	ret[0] = 0.0;
	ret[1] = 0.0;
}

// ret := 0
static inline void c_dd_setnan(double *ret)
{
	ret[0] = FP_NAN;
	ret[1] = FP_NAN;
}

// b := -a
static inline void c_dd_neg(const double *a, double *b)
{
	b[0] = -a[0];
	b[1] = -a[1];
}

/*  this > 0 */
static inline dd_bool dd_is_positive(const double *x)
{
//  return (x[0] > 0.0);
	if(x[0] > 0.0)
		return DD_TRUE;
	else
		return DD_FALSE;
}

/* this < 0 */
static inline dd_bool dd_is_negative(const double *x)
{
//  return (x[0] < 0.0);
	if(x[0] < 0.0)
		return DD_TRUE;
	else
		return DD_FALSE;
}
// c_dd_comp(a, b)
// a  > b -> return +1
// a  < b -> return -1
// a == b -> return 0
static inline void c_dd_comp(const double *a, const double *b, int *result)
{
//	dd_real aa(a), bb(b);

	// double-double < double-doube
	//if (aa < bb)
  	if(a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]))
		*result = -1;
	// double-double > double-doube
	//else if (aa > bb)
	else if(a[0] > b[0] || (a[0] == b[0] && a[1] > b[1]))
		*result = 1;
	else // aa == bb
		*result = 0;
#if 0 // #else // NATIVE_C
  dd_real aa(a), bb(b);
  if (aa < bb)
    *result = -1;
  else if (aa > bb)
    *result = 1;
  else 
    *result = 0;
#endif // NATIVE_C
}

// dd_a <=> b
static inline void c_dd_comp_dd_d(const double *a, double b, int *result)
{
	//	dd_real aa(a), bb(b);
	double bb[DDSIZE];

	c_dd_set_dd_d(b, bb);

	c_dd_comp(a, bb, result);
#if 0 //
  dd_real aa(a), bb(b);
  if (aa < bb)
    *result = -1;
  else if (aa > bb)
    *result = 1;
  else 
    *result = 0;
#endif // NATIVE_C
}

static inline void c_dd_comp_d_dd(double a, const double *b, int *result)
{
	double aa[DDSIZE];

	c_dd_set_dd_d(a, aa);

	c_dd_comp(aa, b, result);
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real aa(a), bb(b);
  if (aa < bb)
    *result = -1;
  else if (aa > bb)
    *result = 1;
  else 
    *result = 0;
#endif // NATIVE_C
}

static inline void c_dd_pi(double *a)
{
	c_dd_set(const_dd_pi, a);

//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  TO_DOUBLE_PTR(dd_real::_pi, a);
#endif // NATIVE_C
}

/* double-double = double + double */
static inline void c_d_add(double a, double b, double *c)
{
	double s, e;

	s = two_sum(a, b, &e);
//	return dd_real(s, e);
	c[0] = s;
	c[1] = e;
}
//#endif // NATIVE_C

/* add */
static inline void c_dd_add(const double *a, const double *b, double *c)
{
	/* This one satisfies IEEE style error bound, 
		due to K. Briggs and W. Kahan.                   */
	double s1, s2, t1, t2;

	s1 = two_sum(a[0], b[0], &s2);
	t1 = two_sum(a[1], b[1], &t2);
	s2 += t1;
	s1 = quick_two_sum(s1, s2, &s2);
	s2 += t2;
	s1 = quick_two_sum(s1, s2, &s2);
//  return dd_real(s1, s2);
	c[0] = s1;
	c[1] = s2;
#if 0 // NATIVE_C
  dd_real cc;
  cc = dd_real(a) + dd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif
}

// sloppy_add
//static inline void c_dd_add_sloppy(const double *a, const double *b, double *c, double *err)
static inline void c_dd_add_sloppy(const double *a, const double *b, double *c)
{
//#ifdef NATIVE_C
  /* This is the less accurate version ... obeys Cray-style
     error bound. */
	double s, e;

	s = two_sum(a[0], b[0], &e);
	e += (a[1] + b[1]);
//	s = quick_two_sum(s, e, &e);
	c[0] = quick_two_sum(s, e, &e);
	c[1] = e;
//  return dd_real(s, e);
//	*err = e;
#if 0 // NATIVE_C
  /* This is the less accurate version ... obeys Cray-style
     error bound. */
  double s, e;

  s = qd::two_sum(a.x[0], b.x[0], e);
  e += (a.x[1] + b.x[1]);
  s = qd::quick_two_sum(s, e, e);
  return dd_real(s, e);
#endif // NATIVE_C
}

// 2025-12-24(Wed) T.Kouya
// Branch free algorithm 
//void Add2(const double x[2], const double y[2], double z[2]) {
static inline void c_dd_add_bf(const double *a, const double *b, double *c)
{
    double g1, g2, g3, g4, g5, g6;
	double g1e, g2e, g3e, g6e;

	g1 = two_sum(a[0], b[0], &g1e); // ゲート1 [cite: 220, 241]
    g2 = two_sum(a[1], b[1], &g2e); // ゲート2 [cite: 220, 241]
	g3 = quick_two_sum(g1, g2, &g3e);
	g4 = g1e + g2e;
	g5 = g4 + g3e;
	g6 = quick_two_sum(g3, g5, &g6e);
	c[0] = g6;
	c[1] = g6e;
}

// bug fix2: 2022-10-12(Wed) by T.Kouya
// bug fix: 2020-11-27(Fri) by T.Kouya
// (dd_real)c := (dd_real)a + b
static inline void c_dd_add_dd_d(const double *a, double b, double *c)
{
//#ifdef NATIVE_C
	//double cc[DDSIZE];
	double s1, s2;
	//c[0] = a[0] + b;
	//c[1] = a[1];
	s1 = two_sum(a[0], b, &s2);
	//t1 = two_sum(a[1], b[1], &t2);
	//s2 += t1;
	s2 += a[1];
	s1 = quick_two_sum(s1, s2, &s2);
	//s2 += t2;
	//s1 = quick_two_sum(s1, s2, &s2);
//  return dd_real(s1, s2);
	c[0] = s1;
	c[1] = s2;
	
#if 0 //#else // NATIVE_C
  dd_real cc;
  cc = dd_real(a) + b;
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

// bug fix2: 2022-10-12(Wed) by T.Kouya
// bug fix: 2020-11-27(Fri) by T.Kouya
// (dd_real)c := a + (dd_real)b
static inline void c_dd_add_d_dd(double a, const double *b, double *c)
{
//#ifdef NATIVE_C
	//double cc[DDSIZE];
	//c[0] = a + b[0];
	//c[1] = b[1];
	//double cc[DDSIZE];
	double s1, s2;
	//c[0] = a[0] + b;
	//c[1] = a[1];
	s1 = two_sum(a, b[0], &s2);
	//t1 = two_sum(a[1], b[1], &t2);
	//s2 += t1;
	s2 += b[1];
	s1 = quick_two_sum(s1, s2, &s2);
	//s2 += t2;
	//s1 = quick_two_sum(s1, s2, &s2);
//  return dd_real(s1, s2);
	c[0] = s1;
	c[1] = s2;

#if 0 // #else // NATIVE_C
  dd_real cc;
  cc = a + dd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

/*********** Subtractions ************/
/* double-double = double - double */
static inline void c_d_sub(double a, double b, double *c)
{
	double s, e;
	s = two_diff(a, b, &e);
	c[0] = s;
	c[1] = e;
#if 0
  double s, e;
  s = qd::two_diff(a, b, e);
  return dd_real(s, e);
#endif
}

/* sub */
static inline void c_dd_sub(const double *a, const double *b, double *c)
{
	double s, e;
	s = two_diff(a[0], b[0], &e);
	e += a[1];
	e -= b[1];
	s = quick_two_sum(s, e, &e);
//	return dd_real(s, e);
	c[0] = s;
	c[1] = e;
#if 0 // QD_IEEE_ADD
  dd_real cc;
  cc = dd_real(a) - dd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

static inline void c_dd_sub_sloppy(const double *a, const double *b, double *c)
{
	double s1, s2, t1, t2;
	s1 = two_diff(a[0], b[0], &s2);
	t1 = two_diff(a[1], b[1], &t2);
	s2 += t1;
	s1 = quick_two_sum(s1, s2, &s2);
	s2 += t2;
	s1 = quick_two_sum(s1, s2, &s2);
	//return dd_real(s1, s2);
	c[0] = s1;
	c[1] = s2;
#if 0
  double s1, s2, t1, t2;
  s1 = qd::two_diff(a.x[0], b.x[0], s2);
  t1 = qd::two_diff(a.x[1], b.x[1], t2);
  s2 += t1;
  s1 = qd::quick_two_sum(s1, s2, s2);
  s2 += t2;
  s1 = qd::quick_two_sum(s1, s2, s2);
  return dd_real(s1, s2);
#endif
}

/* double-double - double */
static inline void c_dd_sub_dd_d(const double *a, double b, double *c)
{
	double s1, s2;
	s1 = two_diff(a[0], b, &s2);
	s2 += a[1];
	s1 = quick_two_sum(s1, s2, &s2);
	//return dd_real(s1, s2);
	c[0] = s1;
	c[1] = s2;
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real cc;
  cc = dd_real(a) - b;
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

/* double - double-double */
static inline void c_dd_sub_d_dd(double a, const double *b, double *c)
{
	double s1, s2;
	s1 = two_diff(a, b[0], &s2);
	s2 -= b[1];
	s1 = quick_two_sum(s1, s2, &s2);
//	return dd_real(s1, s2);
	c[0] = s1;
	c[1] = s2;
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real cc;
  cc = a - dd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}


/*********** Multiplications ************/
/* double-double = double * double */
static inline void c_d_mul(double a, double b, double *c)
{
	double p, e;
	p = two_prod(a, b, &e);
//	return dd_real(p, e);
	c[0] = p;
	c[1] = e;
}

/* mul */
static inline void c_dd_mul(const double *a, const double *b, double *c)
{
	double p1, p2;

	p1 = two_prod(a[0], b[0], &p2);
	p2 += (a[0] * b[1] + a[1] * b[0]);
	p1 = quick_two_sum(p1, p2, &p2);
//	return dd_real(p1, p2);
	c[0] = p1;
	c[1] = p2;
#if 0 // #else // NATIVE_C
  dd_real cc;
  cc = dd_real(a) * dd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

// 2025-12-24(Wed) T.Kouya
// Branch free algorithm
// void Mul2(const double x[2], const double y[2], double z[2]) {
static inline void c_dd_mul_bf(const double *a, const double *b, double *c)
{
	double p00, p01, p10;
	double pe00;
	double g1, g2, g3, ge3;

    p00 = two_prod(a[0], b[0], &pe00); // [cite: 250, 307]
    p01 = a[0] * b[1]; // [cite: 250, 307]
    p10 = a[1] * b[0]; // [cite: 250, 307]
	g1 = p01 + p10;
	g2 = pe00 + g1;
	g3 = quick_two_sum(p00, g2, &ge3);
	
    c[0] = g3;                       // [cite: 307]
    c[1] = ge3; // 全誤差の統合 [cite: 307]
}

static inline void c_dd_mul_dd_d(const double *a, double b, double *c)
{
	double p1, p2;

	p1 = two_prod(a[0], b, &p2);
	p2 += (a[1] * b);
	p1 = quick_two_sum(p1, p2, &p2);
//  return dd_real(p1, p2);
	c[0] = p1;
	c[1] = p2;
#if 0 // #else // NATIVE_C
  dd_real cc;
  cc = dd_real(a) * b;
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}
static inline void c_dd_mul_d_dd(double a, const double *b, double *c)
{
	c_dd_mul_dd_d(b, a, c);

//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real cc;
  cc = a * dd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

/*********** Divisions ************/
static inline void c_d_div(double a, double b, double *c)
{
	double q1, q2;
	double p1, p2;
	double s, e;

	q1 = a / b;

	/* Compute  a - q1 * b */
	p1 = two_prod(q1, b, &p2);
	s = two_diff(a, p1, &e);
	e -= p2;

	/* get next approximation */
	q2 = (s + e) / b;

	s = quick_two_sum(q1, q2, &e);

	//return dd_real(s, e);
	c[0] = s;
	c[1] = e;
}

/* FMA-driven divisions (dtq-0.0.3 fma_div ports; defined in bncelem_*.h,
   included at the end of this header).  Selected by BNC_USE_FMA_DIV.     */
static inline void bnc_dd_div_fma(const double *a, const double *b, double *ret);
static inline void bnc_td_div_fma(const double *a, const double *b, double *ret);
static inline void bnc_qd_div_fma(const double *a, const double *b, double *ret);

/* div */ // fix! 2021-01-25 by T.Kouya
static inline void c_dd_div(const double *a, const double *b, double *c)
{
#ifdef BNC_USE_FMA_DIV
	bnc_dd_div_fma(a, b, c);
}
static inline void c_dd_div_orig(const double *a, const double *b, double *c)
{
#endif /* BNC_USE_FMA_DIV */
	double q1, q2, q3;
	double r[DDSIZE], tmp[DDSIZE];

	q1 = a[0] / b[0];  /* approximate quotient */

//  r = a - q1 * b;
	c_dd_mul_dd_d(b, q1, r);
	c_dd_sub(a, r, r);

	q2 = r[0] / b[0];

//  r -= (q2 * b);
	c_dd_mul_dd_d(b, q2, tmp);
	c_dd_sub(r, tmp, r); // fix! 2021-01-25 by T.Kouya

//  q3 = r[0] / b[0];
	q3 = r[0] / b[0]; // fix! 2021-01-25 by T.Kouya

	q1 = quick_two_sum(q1, q2, &q2);
//  r = dd_real(q1, q2) + q3;
	r[0] = q1;
	r[1] = q2;
	c_dd_add_dd_d(r, q3, r);

	c[0] = r[0];
	c[1] = r[1];
//  return r;
#if 0 // #else // NATIVE_C
  dd_real cc;
  cc = dd_real(a) / dd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

/* double-double / double-double */
static inline void c_dd_sloppy_div_old(double *a, double *b, double *c)
{
	double s1, s2;
	double q1, q2;
//	dd_real r;
	double r[DDSIZE];

	q1 = a[0] / b[0];  /* approximate quotient */

	/* compute  this - q1 * dd */
	//r = b * q1;
	c_dd_mul_dd_d(b, q1, r);

	s1 = two_diff(a[0], r[0], &s2);
	s2 -= r[1];
	s2 += a[1];

	/* get next approximation */
	q2 = (s1 + s2) / b[0];

	/* renormalize */
	r[0] = quick_two_sum(q1, q2, &r[1]);
//	return r;

	c[0] = r[0];
	c[1] = r[1];
}
static inline void c_dd_sloppy_div(double *a, double *b, double *c)
{
	double s1, s2;
	double q1, q2;
	double temp[DDSIZE];
	
	// エイリアシング対策: 入力を先読み
	double a0 = a[0];
	double a1 = a[1];
	double b0 = b[0];

	// 第1近似: q1 = a / b
	q1 = a0 / b0;

	// temp = b * q1
	c_dd_mul_dd_d(b, q1, temp);

	// 残差: s = a - b*q1
	s1 = two_diff(a0, temp[0], &s2);
	s2 -= temp[1];
	s2 += a1;

	// 第2近似: q2 = s / b
	q2 = (s1 + s2) / b0;

	// 正規化: c = q1 + q2
	c[0] = quick_two_sum(q1, q2, &c[1]);
}

/* double-double / double */
static inline void c_dd_div_dd_d(const double *a, double b, double *c)
{
	double q1, q2;
	double p1, p2;
	double s, e;
//	dd_real r;
	double r[DDSIZE];

	q1 = a[0] / b;   /* approximate quotient. */

	/* Compute  this - q1 * d */
	p1 = two_prod(q1, b, &p2);
	s = two_diff(a[0], p1, &e);
	e += a[1];
	e -= p2;

	/* get next approximation. */
	q2 = (s + e) / b;

	/* renormalize */
	r[0] = quick_two_sum(q1, q2, &r[1]);

	// return r;
	c[0] = r[0];
	c[1] = r[1];
#if 0 // #else // NATIVE_C
  dd_real cc;
  cc = dd_real(a) / b;
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

/* double / double-double */
static inline void c_dd_div_d_dd(double a, const double *b, double *c)
{
	double tmp[DDSIZE];
	//return dd_real(a) / b;
	tmp[0] = a;
	tmp[1] = 0.0;

	c_dd_div(tmp, b, c);

//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real cc;
  cc = a / dd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

/* copy */
static inline void c_dd_copy(const double *a, double *b)
{
  b[0] = a[0];
  b[1] = a[1];
}

static inline void c_dd_copy_d(double a, double *b)
{
  b[0] = a;
  b[1] = 0.0;
}

/* b := a^2 */
static inline void c_d_sqr(double a, double *b)
{
	double p1, p2;
	p1 = two_sqr(a, &p2);
	// return dd_real(p1, p2);
	b[0] = p1;
	b[1] = p2;
}

// b := a^2
static inline void c_dd_sqr_d(double a, double *ret)
{
	double p1, p2;
	p1 = two_sqr(a, &p2);
//	return dd_real(p1, p2);
	ret[0] = p1;
	ret[1] = p2;
}

// b := a^2
static inline void c_dd_sqr(const double *a, double *b)
{
	double p1, p2;
	double s1, s2;

	p1 = two_sqr(a[0], &p2);
	p2 += 2.0 * a[0] * a[1];
	p2 += a[1] * a[1];
	s1 = quick_two_sum(p1, p2, &s2);
//	return dd_real(s1, s2);
	b[0] = s1;
	b[1] = s2;
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = sqr(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

/* b := a^2 */
/* bug fix: 2020-11-27(Fri) by T.Kouya
No use of AVX2
||C|| = ret =   2.68101840384000000e+12
1) tmp[0] =  -2.32302942206838977e-05
2) x2 =   3.05365596719460574e-07
3) tmp[0] =  -7.09373265666777470e-12
4) b[0] =   6.20329344000000000e+09
tmp =   6.20329344000000000e+09
ret =   6.20329344000000000e+09
6.20329e+09

Use of AVX2
||C|| = ret =   2.68101840384000000e+12
1) tmp[0] =  -2.32302942206838977e-05
2) x2 =   3.05365596719460574e-07
3) tmp[0] =  -7.09373265666777470e-12
4) b[0] =   0.00000000000000000e+00
tmp =   0.00000000000000000e+00
ret =   0.00000000000000000e+00
0
*/
static inline void c_dd_sqrt_old(const double *a, double *b)
{
  /* Strategy:  Use Karp's trick:  if x is an approximation
     to sqrt(a), then

        sqrt(a) = a*x + [a - (a*x)^2] * x / 2   (approx)

     The approximation is accurate to twice the accuracy of x.
     Also, the multiplication (a*x) and [-]*x can be done with
     only half the precision.
  */

 // if (a.is_zero())
	if(dd_is_zero(a) == DD_TRUE)
	{
		//dd_set0(b);
		b[0] = 0.0; b[1] = 0.0;
		return;
	}

//  if (a.is_negative()) {
	if(dd_is_negative(a) == DD_TRUE)
	{
		//dd_real::error("(dd_real::sqrt): Negative argument.");
		fprintf(stderr, "(c_dd_sqrt): Negative argument.");
		//return dd_real::_nan;
		//dd_set_d(NAN, b);
		b[0] = FP_NAN; b[1] = FP_NAN;
		return;
	}

	//  double x = 1.0 / std::sqrt(a.x[0]);
	double x;
#ifdef __cplusplus
	x = 1.0 / (double)std::sqrt((double)a[0]);
#else // __cplusplus
	x = 1.0 / (double)sqrt((double)a[0]);
#endif // __cplusplus

  	//double ax = a.x[0] * x;
	double ax, x2;

	ax = a[0] * x;

//  return dd_real::add(ax, (a - dd_real::sqr(ax)).x[0] * (x * 0.5));
	double tmp[DDSIZE], ax2[DDSIZE];

// 1) DD: tmp := a - ax^2
	c_dd_sqr_d(ax, ax2);
	c_dd_sub(a, ax2, tmp);
	//printf("1) tmp[0] = %25.17e\n", tmp[0]);

// 2) double: x2 := x * 0.5
	x2 = x * 0.5;
	//printf("2) x2 = %25.17e\n", x2);

// 3) DD: tmp[0] * x2
	//tmp[0] = two_prod(tmp[0], x2, &tmp[1]);
	c_dd_mul_dd_d(tmp, x2, tmp);
	//printf("3) tmp[0] = %25.17e\n", tmp[0]);

// 4) DD: ax + tmp
	c_dd_add_dd_d(tmp, ax, b);
	//printf("4) b[0] = %25.17e\n", b[0]);

//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = sqrt(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
static inline void c_dd_sqrt(const double *a, double *b)
{
	if(dd_is_zero(a) == DD_TRUE) {
		b[0] = 0.0; b[1] = 0.0;
		return;
	}
	if(dd_is_negative(a) == DD_TRUE) {
		fprintf(stderr, "(c_dd_sqrt): Negative argument.\n");
		b[0] = FP_NAN; b[1] = FP_NAN;
		return;
	}

	// fix! 2025-02-25(Wed) T.Kouya
#ifdef __cplusplus
	double x = 1.0 / (double)std::sqrt((double)a[0]);
#else // __cplusplus
	double x = 1.0 / (double)sqrt((double)a[0]);
#endif // __cplusplus
	//double x = 1.0 / sqrt(a[0]);
	double ax = a[0] * x;

	// すべての一時変数を明確に分離
	double ax_squared[DDSIZE];
	double error[DDSIZE];
	double correction[DDSIZE];
	double ax_dd[DDSIZE];

	// ax² を計算
	c_dd_sqr_d(ax, ax_squared);

	// error = a - ax²
	c_dd_sub(a, ax_squared, error);

	// correction = error * (x/2)
	double half_x = 0.5 * x;
	c_dd_mul_dd_d(error, half_x, correction);

	// ax を DD 形式に変換
	ax_dd[0] = ax;
	ax_dd[1] = 0.0;

	// b = ax + correction
	c_dd_add(ax_dd, correction, b);
}


static inline void c_dd_abs(const double *a, double *b)
{
	if(a[0] < 0.0)
	{
		b[0] = -a[0];
		b[1] = -a[1];
	}
	else
	{
		b[0] = a[0];
		b[1] = a[1];
	}
//	return (a.x[0] < 0.0) ? -a : a;
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = abs(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

static inline void c_dd_npwr(const double *a, int n, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = npwr(dd_real(a), n);
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

static inline void c_dd_nroot(const double *a, int n, double *b) {
// #ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = nroot(dd_real(a), n);
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

/* forward declaration; defined in bncelem_dd.h (included at the end of
   this header via bncelem.h) */
static inline void bnc_dd_nint(const double *a, double *ret);

// (dtq-0.0.3 port; the previous version of this function was an empty stub)
static inline void c_dd_nint(const double *a, double *b) {
	bnc_dd_nint(a, b);
}
static inline void c_dd_aint(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = aint(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

// b := floor(a)
static inline void c_dd_floor(const double *a, double *b)
{
	double hi = floor(a[0]);
	double lo = 0.0;
	if (hi == a[0]) {
	  /* High word is integer already.  Round the low word. */
	  lo = floor(a[1]);
	  hi = quick_two_sum(hi, lo, &lo);
	}
	//return dd_real(hi, lo);
	b[0] = hi;
	b[1] = lo;

//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = floor(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

// b := ceil(a)
static inline void c_dd_ceil(const double *a, double *b)
{
	double hi = ceil(a[0]);
	double lo = 0.0;	

	if (hi == a[0]) {
	  /* High word is integer already.  Round the low word. */
	  lo = ceil(a[1]);
	  hi = quick_two_sum(hi, lo, &lo);
	}	
	//return dd_real(hi, lo);
	b[0] = hi;
	b[1] = lo;

//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = ceil(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

/* Forward declarations of the dtq-0.0.3 FMA-based elementary functions
   (argument reduction + Taylor kernels whose terms are accumulated with
   one certified branch-free FMA each).  The definitions come from
   bncelem.h, which is included at the end of this header.               */
static inline void bnc_dd_exp(const double *a, double *ret);
static inline void bnc_dd_log(const double *a, double *ret);
static inline void bnc_dd_log10(const double *a, double *ret);
static inline void bnc_dd_sin(const double *a, double *ret);
static inline void bnc_dd_cos(const double *a, double *ret);
static inline void bnc_dd_sincos(const double *a, double *sin_a, double *cos_a);
static inline void bnc_qd_exp(const double *a, double *ret);
static inline void bnc_qd_log(const double *a, double *ret);
static inline void bnc_qd_log10(const double *a, double *ret);
static inline void bnc_qd_sin(const double *a, double *ret);
static inline void bnc_qd_cos(const double *a, double *ret);
static inline void bnc_qd_sincos(const double *a, double *sin_a, double *cos_a);

// ret := exp(x)
// (dtq-0.0.3 port: ln2 argument reduction + FMA-fused Taylor kernel)
static inline void c_dd_exp(const double *x, double *ret)
{
	bnc_dd_exp(x, ret);
}

// ret := exp(x)
// original implementation (plain Taylor series without argument reduction);
// kept for accuracy/performance comparison only
static inline void c_dd_exp_orig(const double *x, double *ret)
{
	double old_ans[2], xn[2], new_x[2];
	long int sign;
    unsigned long times;
	int result;
	
	/* initialize temporary variables */
	c_dd_set0(old_ans);
	c_dd_set0(xn);
	c_dd_set0(new_x);

	/* x < 0 */
	c_dd_comp_dd_d(x, 0.0, &result);
	if(result < 0)
	{
		sign = 0; /* minus */
		c_dd_neg(x, new_x);
	}
	else
	{
		sign = 1; /* plus */
		c_dd_set(x, new_x);
	}
	
	/* ans := 1 */
	/* old_ans := ans */
	/* xn  := x   */
	/* times := 0 */
	c_dd_set(const_dd_one, ret);
	c_dd_set(ret, old_ans);
	c_dd_set(const_dd_one, xn);

	times = 0;
	do
	{
 		/* x^n */
		c_dd_mul(xn, new_x, xn);
		/* /n! */
		c_dd_div_dd_d(xn, (double)(++times), xn);
		/* ans += ans + x^n/n! */
		c_dd_add(ret, xn, ret);
		/* ond_ans == ans */
		c_dd_comp(old_ans, ret, &result);
		if(result == 0)
			break;
		/* old_ans := ans */
        c_dd_set(ret, old_ans);
        //printf(" %ld ", times/2);
	}while(1);

	if(sign == 0) /* minus */
		c_dd_div_d_dd((double)1UL, ret, ret);

//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = exp(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

// b := log(a)
// (dtq-0.0.3 port: one Newton step on exp with the FMA-fused exp kernel)
static inline void c_dd_log(const double *a, double *b)
{
	bnc_dd_log(a, b);
}

// b := log(a)
// original implementation, kept for accuracy/performance comparison only
static inline void c_dd_log_orig(const double *a, double *b)
{
	double tmp[2], x[2] = {log(a[0]), 0.0},  one[2] = {1.0, 0.0};

 /* Strategy.  The Taylor series for log converges much more
     slowly than that of exp, due to the lack of the factorial
     term in the denominator.  Hence this routine instead tries
     to determine the root of the function

         f(x) = exp(x) - a

     using Newton iteration.  The iteration is given by

         x' = x - f(x)/f'(x) 
            = x - (1 - a * exp(-x))
            = x + a * exp(-x) - 1.
           
     Only one iteration is needed, since Newton's iteration
     approximately doubles the number of digits per iteration. */

	// a == 1
	if(DD_ISONE(a))
	{
		c_dd_set0(b);
		return;
	}

	// a <= 0
	if(a[0] <= 0.0)
	{
	    fprintf(stderr, "c_dd_log: Non-positive argument.");
		c_dd_setnan(b);
		return;
	}

	// x = x + a * exp(-x) - 1.0;
	// return x
	c_dd_neg(x, tmp);
	c_dd_exp_orig(tmp, b);
	c_dd_mul(a, b, tmp);
	c_dd_add(x, tmp, tmp);
	c_dd_sub(tmp, one, b);

//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = log(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

// port qd library to here
// (dtq-0.0.3 port: log(x) / log(10) with the tabulated log(10) constant)
static inline void c_dd_log10(const double *x, double *ret)
{
	bnc_dd_log10(x, ret);
}

// original implementation (recomputes log(10) on every call),
// kept for accuracy/performance comparison only
static inline void c_dd_log10_orig(const double *x, double *ret)
{
	double ln10[2];

	//if(mpf_sgn(x) <= 0)
	if(x[0] < 0.0)
	{
		fprintf(stderr, "c_dd_log10(x): illegal argument\n");
		return;
	}

	/* init */
	//mpf_init2(ln10, prec);
	c_dd_set0(ln10);

	/* log10 x := ln x / ln 10 */
	//mpf_ln(ans, x);
	c_dd_log(x, ret);
	//mpf_set_ui(ln10, 10UL);
	c_dd_set_dd_d((double)10UL, ln10);
	//mpf_ln(ln10, ln10);
	c_dd_log(ln10, ln10);
	//mpf_div(ans, ans, ln10);
	c_dd_div(ret, ln10, ret);

	/* clear */
	//mpf_clear(ln10);

	return; 

//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = log10(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

// log2(x)
static inline void c_dd_log2(const double *x, double *ret)
{
	double ln2[2];

	//if(mpf_sgn(x) <= 0)
	if(x[0] < 0.0)
	{
		fprintf(stderr, "c_dd_log2(x): illegal argument\n");
		return;
	}

	/* init */
	//mpf_init2(ln2, prec);
	c_dd_set0(ln2);

	/* log2 x := ln x / ln 2 */
	//mpf_ln(ans, x);
	c_dd_log(x, ret);
	//mpf_set_ui(ln2, 2UL);
	c_dd_set_dd_d((double)2UL, ln2);
	//mpf_ln(ln2, ln2);
	c_dd_log(ln2, ln2);
	//mpf_div(ans, ans, ln10);
	c_dd_div(ret, ln2, ret);

	/* clear */
	//mpf_clear(ln10);

	return; 
}

// pow(a, b) = a^b
static inline void c_dd_pow(const double *a, const double *b, double *ret)
{
	double tmp[2][2];

	// ret := exp(b * log(a))
	c_dd_log(a, tmp[0]);
	c_dd_mul(tmp[0], b, tmp[1]);
	c_dd_exp(tmp[1], ret);
}


// (dtq-0.0.3 port; the previous version of this function was an empty stub)
static inline void c_dd_sin(const double *a, double *b) {
	bnc_dd_sin(a, b);
}

// (dtq-0.0.3 port; the previous version of this function was an empty stub)
static inline void c_dd_cos(const double *a, double *b) {
	bnc_dd_cos(a, b);
}

// tan(a) = sin(a) / cos(a)
// (dtq-0.0.3 port; the previous version of this function was an empty stub)
static inline void c_dd_tan(const double *a, double *b) {
	double s[2], c[2];

	bnc_dd_sincos(a, s, c);
	c_dd_div(s, c, b);
}

static inline void c_dd_asin(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = asin(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

static inline void c_dd_acos(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = acos(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

static inline void c_dd_atan(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = atan(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

static inline void c_dd_atan2(const double *a, const double *b, double *c) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real cc;
  cc = atan2(dd_real(a), dd_real(b));
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

static inline void c_dd_sinh(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = sinh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
static inline void c_dd_cosh(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = cosh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
static inline void c_dd_tanh(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = tanh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

static inline void c_dd_asinh(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = asinh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
static inline void c_dd_acosh(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = acosh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
static inline void c_dd_atanh(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = atanh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

// (dtq-0.0.3 port; the previous version of this function was an empty stub)
static inline void c_dd_sincos(const double *a, double *s, double *c) {
	bnc_dd_sincos(a, s, c);
}

static inline void c_dd_sincosh(const double *a, double *s, double *c) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real ss, cc;
  sincosh(dd_real(a), ss, cc);
  TO_DOUBLE_PTR(ss, s);
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

static inline void c_dd_read(const char *s, double *a) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real aa(s);
  TO_DOUBLE_PTR(aa, a);
#endif // NATIVE_C
}

static inline void c_dd_swrite(const double *a, int precision, char *s, int len) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real(a).write(s, len, precision);
#endif // NATIVE_C
}

static inline void c_dd_write(const double *a) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  std::cout << dd_real(a).to_string(dd_real::_ndigits) << std::endl;
#endif // NATIVE_C
}

static inline void c_dd_rand(double *a)
{
	static const double m_const = 4.6566128730773926e-10;  /* = 2^{-31} */
	int i;
	double m = 4.6566128730773926e-10;
	//dd_real r = 0.0;
	double r[2] = {0.0, 0.0};
	double d;

  /* Strategy:  Generate 31 bits at a time, using lrand48 
     random number generator.  Shift the bits, and reapeat
     4 times. */

	for (i = 0; i < 4; i++, m *= m_const)
	{
//    d = std::rand() * m;
		d = rand() * m;
		//r += d;
		c_dd_add_dd_d(r, d, r);
	}
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real aa;
  aa = ddrand();
  TO_DOUBLE_PTR(aa, a);
#endif // NATIVE_C
}



/**************************************/
/* QD                                 */
/**************************************/
/*  this == 0 */
static inline qd_bool qd_is_zero(const double *x)
{
//  return (x[0] == 0.0);
	if(x[0] == 0.0)
		return QD_TRUE;
	else
		return QD_FALSE;
}

/*  this == 1 */
static inline qd_bool qd_is_one(const double *x)
{
	//return (x[0] == 1.0 && x[1] == 0.0);
	if(x[0] == 1.0 && x[1] == 0.0)
		return QD_TRUE;
	else
		return QD_FALSE;
}

/* this < 0 */
static inline qd_bool qd_is_negative(const double *x)
{
//  return (x[0] < 0.0);
	if(x[0] < 0.0)
		return QD_TRUE;
	else
		return QD_FALSE;
}

// ret := x
static inline void c_qd_copy(const double *x, double *ret)
{
	ret[0] = x[0];
	ret[1] = x[1];
	ret[2] = x[2];
	ret[3] = x[3];
}

/* (qd)b := (dd)a */
static inline void c_qd_copy_dd(const double *a, double *b)
{
	b[0] = a[0];
	b[1] = a[1];
	b[2] = 0.0;
	b[3] = 0.0;
}

/* (qd)b := (double)a */
static inline void c_qd_copy_d(double a, double *b)
{
	b[0] = a;
	b[1] = 0.0;
	b[2] = 0.0;
	b[3] = 0.0;
}

/********** Renormalization **********/
static inline void quick_renorm(double *c0, double *c1, double *c2, double *c3, double *c4)
{
	double t0, t1, t2, t3;
	double s;

	s   = quick_two_sum(*c3, *c4, &t3);
	s   = quick_two_sum(*c2, s  , &t2);
	s   = quick_two_sum(*c1, s  , &t1);
	*c0 = quick_two_sum(*c0, s  , &t0);

	s   = quick_two_sum(t2, t3, &t2);
	s   = quick_two_sum(t1, s , &t1);
	*c1 = quick_two_sum(t0, s , &t0);

	s   = quick_two_sum(t1, t2, &t1);
	*c2 = quick_two_sum(t0, s , &t0);

	*c3 = t0 + t1;
}

static inline void renorm(double *c0, double *c1, double *c2, double *c3)
{
	double s0, s1, s2 = 0.0, s3 = 0.0;

//	if (QD_ISINF(c0)) return;
	if (isinf(*c0)) return;

	s0  = quick_two_sum(*c2, *c3, c3);
	s0  = quick_two_sum(*c1,  s0, c2);
	*c0 = quick_two_sum(*c0,  s0, c1);

	s0 = *c0;
	s1 = *c1;
	if (s1 != 0.0)
	{
		s1 = quick_two_sum(s1, *c2, &s2);
		if (s2 != 0.0)
			s2 = quick_two_sum(s2, *c3, &s3);
		else
			s1 = quick_two_sum(s1, *c3, &s2);
	}
	else
	{
		s0 = quick_two_sum(s0, *c2, &s1);
		if (s1 != 0.0)
			s1 = quick_two_sum(s1, *c3, &s2);
		else
			s0 = quick_two_sum(s0, *c3, &s1);
	}

	*c0 = s0;
	*c1 = s1;
	*c2 = s2;
	*c3 = s3;
}

static inline void renorm4(double *c0, double *c1, double *c2, double *c3, double *c4)
{
	double s0, s1, s2 = 0.0, s3 = 0.0;

//	if (QD_ISINF(c0)) return;
	if (isinf(*c0)) return;

	s0  = quick_two_sum(*c3, *c4, c4);
	s0  = quick_two_sum(*c2, s0 , c3);
	s0  = quick_two_sum(*c1, s0 , c2);
	*c0 = quick_two_sum(*c0, s0 , c1);

	s0 = *c0;
	s1 = *c1;

//	s0 = quick_two_sum(*c0, *c1, &s1);
	if (s1 != 0.0)
	{
		s1 = quick_two_sum(s1, *c2, &s2);
		if (s2 != 0.0)
		{
			s2 = quick_two_sum(s2, *c3, &s3);
			if (s3 != 0.0)
				s3 += *c4;
			else
				s2 = quick_two_sum(s2, *c4, &s3); // fix!: 2020-11-06 by T.Kouya // s2 += *c4;
		}
		else
		{
			s1 = quick_two_sum(s1, *c3, &s2);
			if (s2 != 0.0)
				s2 = quick_two_sum(s2, *c4, &s3);
			else
				s1 = quick_two_sum(s1, *c4, &s2);
		}
	}
	else
	{
		s0 = quick_two_sum(s0, *c2, &s1);
		if (s1 != 0.0)
		{
			s1 = quick_two_sum(s1, *c3, &s2);
			if (s2 != 0.0)
				s2 = quick_two_sum(s2, *c4, &s3);
			else
				s1 = quick_two_sum(s1, *c4, &s2);
		}
		else
		{
			s0 = quick_two_sum(s0, *c3, &s1);
			if (s1 != 0.0)
				s1 = quick_two_sum(s1, *c4, &s2);
			else
				s0 = quick_two_sum(s0, *c4, &s1);
		}
	}

	*c0 = s0;
	*c1 = s1;
	*c2 = s2;
	*c3 = s3;
}


/********** Additions ************/
static inline void three_sum(double *a, double *b, double *c)
{
	double t1, t2, t3;

	t1 = two_sum(*a, *b, &t2);
	*a = two_sum(*c, t1, &t3);
	*b = two_sum(t2, t3, c);
}

static inline void three_sum2(double *a, double *b, double *c)
{
	double t1, t2, t3;

	t1 = two_sum(*a, *b, &t2);
	*a = two_sum(*c, t1, &t3);
	*b = t2 + t3;
}

/* quad-double + double-double */
static inline void c_qd_add_qd_dd(const double *a, const double *b, double *c)
{
	double s0, s1, s2, s3;
	double t0, t1;

//	s0 = two_sum(a[0], b._hi(), t0);
//	s1 = two_sum(a[1], b._lo(), t1);
	s0 = two_sum(a[0], DD_HI(b), &t0);
	s1 = two_sum(a[1], DD_LOW(b), &t1);

	s1 = two_sum(s1, t0, &t0);

	s2 = a[2];
	three_sum(&s2, &t0, &t1);

	s3 = two_sum(t0, a[3], &t0);
	t0 += t1;

	renorm4(&s0, &s1, &s2, &s3, &t0);

	c[0] = s0;
	c[1] = s1;
	c[2] = s2;
	c[3] = s3;

//	return qd_real(s0, s1, s2, s3);
	return;

#if 0 //
  qd_real cc;
  cc = qd_real(a) + dd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/* double-double + quad-double */
static inline void c_qd_add_dd_qd(const double *a, const double *b, double *c)
{
	c_qd_add_qd_dd(b, a, c);

#if 0 //
  qd_real cc;
  cc = dd_real(a) + qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/* quad-double + double */
static inline void c_qd_add_qd_d(const double *a, double b, double *c)
{
//	double c0, c1, c2, c3;
	double e;

	c[0] = two_sum(a[0], b, &e);
	c[1] = two_sum(a[1], e, &e);
	c[2] = two_sum(a[2], e, &e);
	c[3] = two_sum(a[3], e, &e);

	//qd::renorm(c0, c1, c2, c3, e);
	renorm4(&(c[0]), &(c[1]), &(c[2]), &(c[3]), &e);

//	return qd_real(c0, c1, c2, c3);
	return;
#if 0 //
  qd_real cc;
  cc = qd_real(a) + b;
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/* double + quad-double */
static inline void c_qd_add_d_qd(double a, const double *b, double *c)
{
	c_qd_add_qd_d(b, a, c);

#if 0
  qd_real cc;
  cc = a + qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/* s = quick_three_accum(a, b, c) adds c to the dd-pair (a, b).
 * If the result does not fit in two doubles, then the sum is 
 * output into s and (a,b) contains the remainder.  Otherwise
 * s is zero and (a,b) contains the sum. */
static inline double quick_three_accum(double *a, double *b, double c)
{
	double s;
//	bool za, zb;

	s = two_sum(*b, c, b);
	s = two_sum(*a, s, a);

//	za = (*a != 0.0);
//	zb = (*b != 0.0);

//  if (za && zb)
	if((*a != 0.0) && (*b != 0.0))
		return s;

//	if (!zb) {
	if(*b == 0.0)
	{
		*b = *a;
		*a = s;
	}
	else
	{
		*a = s;
	}

	return 0.0;
}

/* add */
//inline qd_real qd_real::ieee_add(const qd_real &a, const qd_real &b) {
static inline void c_qd_add(const double *a, const double *b, double *c)
{
	// IEEE add
	int i, j, k;
	double s, t;
	double u, v;   /* double-length accumulator */
	double x[4] = {0.0, 0.0, 0.0, 0.0};

	i = j = k = 0;
	if (fabs(a[i]) > fabs(b[j]))
		u = a[i++];
	else
		u = b[j++];
	if (fabs(a[i]) > fabs(b[j]))
		v = a[i++];
	else
		v = b[j++];

	u = quick_two_sum(u, v, &v);

	while (k < 4)
	{
		if (i >= 4 && j >= 4)
		{
			x[k] = u;
			if (k < 3)
				x[++k] = v;
			break;
		}

		if (i >= 4)
			t = b[j++];
		else if (j >= 4)
			t = a[i++];
		else if (fabs(a[i]) > fabs(b[j]))
			t = a[i++];
		else
			t = b[j++];

		s = quick_three_accum(&u, &v, t);

		if (s != 0.0) {
			x[k++] = s;
		}
	}

	/* add the rest. */
	for (k = i; k < 4; k++)
		x[3] += a[k];
	for (k = j; k < 4; k++)
		x[3] += b[k];

	renorm(&(x[0]), &(x[1]), &(x[2]), &(x[3]));
	//return qd_real(x[0], x[1], x[2], x[3]);
	c[0] = x[0];
	c[1] = x[1];
	c[2] = x[2];
	c[3] = x[3];
#if 0 //
  qd_real cc;
  cc = qd_real(a) + qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// Sloppy add
static inline void c_qd_add_sloppy(const double *a, const double *b, double *c)
{
	/*
	double s0, s1, s2, s3;
	double t0, t1, t2, t3;

	s0 = qd::two_sum(a[0], b[0], t0);
	s1 = qd::two_sum(a[1], b[1], t1);
	s2 = qd::two_sum(a[2], b[2], t2);
	s3 = qd::two_sum(a[3], b[3], t3);

	s1 = qd::two_sum(s1, t0, t0);
	qd::three_sum(s2, t0, t1);
	qd::three_sum2(s3, t0, t2);
	t0 = t0 + t1 + t3;

	qd::renorm(s0, s1, s2, s3, t0);
	return qd_real(s0, s1, s2, s3, t0);
	*/

	/* Same as above, but addition re-organized to minimize
	   data dependency ... unfortunately some compilers are
	   not very smart to do this automatically */
	double s0, s1, s2, s3;
	double t0, t1, t2, t3;

	double v0, v1, v2, v3;
	double u0, u1, u2, u3;
	double w0, w1, w2, w3;

	s0 = a[0] + b[0];
	s1 = a[1] + b[1];
	s2 = a[2] + b[2];
	s3 = a[3] + b[3];

	v0 = s0 - a[0];
	v1 = s1 - a[1];
	v2 = s2 - a[2];
	v3 = s3 - a[3];

	u0 = s0 - v0;
	u1 = s1 - v1;
	u2 = s2 - v2;
	u3 = s3 - v3;

	w0 = a[0] - u0;
	w1 = a[1] - u1;
	w2 = a[2] - u2;
	w3 = a[3] - u3;

	u0 = b[0] - v0;
	u1 = b[1] - v1;
	u2 = b[2] - v2;
	u3 = b[3] - v3;

	t0 = w0 + u0;
	t1 = w1 + u1;
	t2 = w2 + u2;
	t3 = w3 + u3;

	s1 = two_sum(s1, t0, &t0);
	three_sum(&s2, &t0, &t1);
	three_sum2(&s3, &t0, &t2);
	t0 = t0 + t1 + t3;

	/* renormalize */
	renorm4(&s0, &s1, &s2, &s3, &t0);
//	return qd_real(s0, s1, s2, s3);
	c[0] = s0;
	c[1] = s1;
	c[2] = s2;
	c[3] = s3;
}

// 2025-12-25(Wed) T.Kouya
// Branch free algorithm
// void Add4(const double x[4], const double y[4], double z[4]) {
static inline void c_qd_add_bf(const double *a, const double *b, double *c)
{
	double a0 , b0 , c0 , d0 , e0, f0, g0, h0;
	double a1 , b1 , c1 , d1 , e1, f1, g1, h1;
	double a2 , b2 , c2 , d2 , e2, f2, g2;
	double a3 , b3 , c3 , d3 , e3, f3, g3;
	double a4 , b4 , c4 , d4 , e4;
	double a5 , b5 , c5 , d5 , e5;
	double a6 , b6 , c6 , d6 , e6;
	double a7 , b7 , c7 , d7 , e7;
	double a8 , b8 , c8 , d8 , e8;
	double a9 , b9 , c9 , d9 ;
	double a10, b10, c10, d10;
	double a11, b11, c11, d11;
	double a12, b12, c12, d12;

	a0 = a[0];
    b0 = b[0];
    c0 = a[1];
    d0 = b[1];
    e0 = a[2];
    f0 = b[2];
    g0 = a[3];
    h0 = b[3];
    a1 = two_sum(a0, b0, &b1);
    c1 = two_sum(c0, d0, &d1);
    e1 = two_sum(e0, f0, &f1);
    g1 = two_sum(g0, h0, &h1);
    a2 = quick_two_sum(a1, c1, &c2);
    b2 = b1 + h1;
    d2 = two_sum(d1, e1, &e2);
    f2 = two_sum(f1, g1, &g2);
    b3 = two_sum(b2, g2, &g3);
    c3 = quick_two_sum(c2, d2, &d3);
    e3 = two_sum(e2, f2, &f3);
    a4 = quick_two_sum(a2, c3, &c4);
    d4 = quick_two_sum(d3, e3, &e4);
    b5 = two_sum(b3, d4, &d5);
    e5 = e4 + f3;
    b6 = two_sum(b5, c4, &c6);
    d6 = two_sum(d5, e5, &e6);
    a7 = quick_two_sum(a4, b6, &b7);
    c7 = quick_two_sum(c6, d6, &d7);
    e8 = e6 + g3;
    b8 = quick_two_sum(b7, c7, &c8);
    d9 = d7 + e8;
    a10 = quick_two_sum(a7, b8, &b10);
    c10 = quick_two_sum(c8, d9, &d10);
    b11 = quick_two_sum(b10, c10, &c11);
    c12 = quick_two_sum(c11, d10, &d12);

	c[0] = a10;
	c[1] = b11;
	c[2] = c12;
	c[3] = d12;
}

// Sloppy add for triple double prec
static inline void c_td_addq(const double *a, const double *b, double *c)
{

	/* Same as above, but addition re-organized to minimize
	   data dependency ... unfortunately some compilers are
	   not very smart to do this automatically */
	double s0, s1, s2;
	double t0, t1, t2;

	double v0, v1, v2;
	double u0, u1, u2;
	double w0, w1, w2;

	s0 = a[0] + b[0];
	s1 = a[1] + b[1];
	s2 = a[2] + b[2];
	//s3 = a[3] + b[3];

	v0 = s0 - a[0];
	v1 = s1 - a[1];
	v2 = s2 - a[2];
	//v3 = s3 - a[3];

	u0 = s0 - v0;
	u1 = s1 - v1;
	u2 = s2 - v2;
	//u3 = s3 - v3;

	w0 = a[0] - u0;
	w1 = a[1] - u1;
	w2 = a[2] - u2;
	//w3 = a[3] - u3;

	u0 = b[0] - v0;
	u1 = b[1] - v1;
	u2 = b[2] - v2;
	//u3 = b[3] - v3;

	t0 = w0 + u0;
	t1 = w1 + u1;
	t2 = w2 + u2;
	//t3 = w3 + u3;

	s1 = two_sum(s1, t0, &t0);
	three_sum(&s2, &t0, &t1);
	//three_sum2(&s3, &t0, &t2);
	t0 = t0 + t1 + t2;

	/* renormalize */
	renorm(&s0, &s1, &s2, &t0);
//	return qd_real(s0, s1, s2, s3);
	c[0] = s0;
	c[1] = s1;
	c[2] = s2;
	//c[3] = s3;
}



/********** Self-Additions ************/
// b := b + a
static inline void c_qd_selfadd(const double *a, double *b)
{
	double bb[QDSIZE];

	c_qd_copy(b, bb);
	c_qd_add(a, bb, b);

#if 0 //
	qd_real bb(b);
	bb += qd_real(a);
	TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b + (dd)a
static inline void c_qd_selfadd_dd(const double *a, double *b)
{
	double bb[QDSIZE];

	c_qd_copy(b, bb);
	c_qd_add_dd_qd(a, bb, b);

#if 0 //
  qd_real bb(b);
  bb += dd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b + (double)a
static inline void c_qd_selfadd_d(double a, double *b)
{
	double bb[QDSIZE];

	c_qd_copy(b, bb);
	c_qd_add_d_qd(a, bb, b);

#if 0 //
  qd_real bb(b);
  bb += a;
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}


/********** Unary Minus **********/
static inline void c_qd_neg(const double *a, double *b) {
	b[0] = -a[0];
	b[1] = -a[1];
	b[2] = -a[2];
	b[3] = -a[3];
}

// b := (dd)(-a)
static inline void c_qd_neg_dd(const double *a, double *b) {
	b[0] = -a[0];
	b[1] = -a[1];
	b[2] = 0.0;
	b[3] = 0.0;
}

// b := (double)(-a)
static inline void c_qd_neg_d(const double a, double *b) 
{
	b[0] = -a;
	b[1] = 0.0;
	b[2] = 0.0;
	b[3] = 0.0;
}

/********** Subtractions **********/
/* sub */
// c := a - b
static inline void c_qd_sub(const double *a, const double *b, double *c)
{
	double mb[QDSIZE];

	// a + (-b)
	c_qd_neg(b, mb);
	c_qd_add(a, mb, c);

#if 0
  qd_real cc;
  cc = qd_real(a) - qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// c := a - (dd)b
static inline void c_qd_sub_qd_dd(const double *a, const double *b, double *c)
{
	double mb[QDSIZE];

	// a + (-b)
	c_qd_neg_dd(b, mb);
	c_qd_add(a, mb, c);

#if 0
  qd_real cc;
  cc = qd_real(a) - dd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// c := (dd)a - b
static inline void c_qd_sub_dd_qd(const double *a, const double *b, double *c)
{
	double ma[QDSIZE];

	// (-a) + b
	c_qd_neg_dd(a, ma);
	c_qd_add(ma, b, c);

#if 0
  qd_real cc;
  cc = dd_real(a) - qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// c := a - (double)b
static inline void c_qd_sub_qd_d(const double *a, double b, double *c)
{
	double mb[QDSIZE];

	// a + (-b)
	c_qd_neg_d(b, mb);
	c_qd_add(a, mb, c);

#if 0
  qd_real cc;
  cc = qd_real(a) - b;
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// c := (double)a - b   (scalar minus QD)
// Bug fix: previous implementation computed (-a) + b = b - a (sign reversed).
// This corrupted c_qd_sqrt's Newton iteration since it expected (0.5 - h*r^2)
// but got (h*r^2 - 0.5), which converges to the wrong fixed point.
static inline void c_qd_sub_d_qd(double a, const double *b, double *c)
{
	double mb[QDSIZE];

	// a - b = a + (-b)
	c_qd_neg(b, mb);
	c_qd_add_d_qd(a, mb, c);

#if 0
  qd_real cc;
  cc = a - qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/********** Self-Subtractions **********/
/* selfsub */
// b := b + (-a) <-> b -= a
static inline void c_qd_selfsub(const double *a, double *b)
{
	double ma[QDSIZE];

	// (-a) + b
	c_qd_neg(a, ma);
	c_qd_add(ma, b, b);

#if 0
  qd_real bb(b);
  bb -= qd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b - (dd)a
static inline void c_qd_selfsub_dd(const double *a, double *b)
{
	double ma[QDSIZE];

	// (-a) + b
	c_qd_neg_dd(a, ma);
	c_qd_add(ma, b, b);

#if 0
  qd_real bb(b);
  bb -= dd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b - (double)a
static inline void c_qd_selfsub_d(double a, double *b)
{
	double ma[QDSIZE];

	// (-a) + b
	c_qd_neg_d(a, ma);
	c_qd_add(ma, b, b);

#if 0
  qd_real bb(b);
  bb -= a;
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

/********** Multiplications **********/
static inline void c_qd_mul(const double *a, const double *b, double *c)
{
	// acculate mul
	double p0, p1, p2, p3, p4, p5;
	double q0, q1, q2, q3, q4, q5;
	double p6, p7, p8, p9;
	double q6, q7, q8, q9;
	double r0, r1;
	double t0, t1;
	double s0, s1, s2;

	p0 = two_prod(a[0], b[0], &q0);

	p1 = two_prod(a[0], b[1], &q1);
	p2 = two_prod(a[1], b[0], &q2);

	p3 = two_prod(a[0], b[2], &q3);
	p4 = two_prod(a[1], b[1], &q4);
	p5 = two_prod(a[2], b[0], &q5);

	/* Start Accumulation */
	three_sum(&p1, &p2, &q0);

	/* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
	three_sum(&p2, &q1, &q2);
	three_sum(&p3, &p4, &p5);

	/* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
	s0 = two_sum(p2, p3, &t0);
	s1 = two_sum(q1, p4, &t1);
	s2 = q2 + p5;
	s1 = two_sum(s1, t0, &t0);
	s2 += (t0 + t1);

	/* O(eps^3) order terms */
	p6 = two_prod(a[0], b[3], &q6);
	p7 = two_prod(a[1], b[2], &q7);
	p8 = two_prod(a[2], b[1], &q8);
	p9 = two_prod(a[3], b[0], &q9);

	/* Nine-Two-Sum of q0, s1, q3, q4, q5, p6, p7, p8, p9. */
	q0 = two_sum(q0, q3, &q3);
	q4 = two_sum(q4, q5, &q5);
	p6 = two_sum(p6, p7, &p7);
	p8 = two_sum(p8, p9, &p9);

	/* Compute (t0, t1) = (q0, q3) + (q4, q5). */
	t0 = two_sum(q0, q4, &t1);
	t1 += (q3 + q5);

	/* Compute (r0, r1) = (p6, p7) + (p8, p9). */
	r0 = two_sum(p6, p8, &r1);
	r1 += (p7 + p9);

	/* Compute (q3, q4) = (t0, t1) + (r0, r1). */
	q3 = two_sum(t0, r0, &q4);
	q4 += (t1 + r1);

	/* Compute (t0, t1) = (q3, q4) + s1. */
	t0 = two_sum(q3, s1, &t1);
	t1 += q4;

	/* O(eps^4) terms -- Nine-One-Sum */
	t1 += a[1] * b[3] + a[2] * b[2] + a[3] * b[1] + q6 + q7 + q8 + q9 + s2;

	renorm4(&p0, &p1, &s0, &t0, &t1);
//	return qd_real(p0, p1, s0, t0);
	c[0] = p0;
	c[1] = p1;
	c[2] = s0;
	c[3] = t0;

#if 0
  qd_real cc;
  cc = qd_real(a) * qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// 2025-12-24(Wed) T.Kouya
// Branch free algorithm
// void Mul4(const double x[4], const double y[4], double z[4]) {
static inline void c_qd_mul_bf(const double *a, const double *b, double *c)
{
	double a0, b0, c0, d0, e0, f0, g0, h0, i0, j0, k0, l0, m0, n0, o0, p0;
	double a1, b1, c1, d1, e1, f1, g1, h1, i1, j1, k1, l1, m1, n1;
	double a2, b2, c2, d2, e2, f2, g2, h2, i2, j2, k2, l2, m2;
	double a3, b3, c3, d3, e3, f3, g3, h3;
	double a4, b4, c4, d4, e4, f4;
	double a5, b5, c5, d5;
	double a6, b6, c6, d6;
	double a7, b7, c7, d7;
	double a8, b8, c8, d8;
	double a9, b9, c9, d9;
	double a10, b10, c10, d10;

    a0 = two_prod(a[0], b[0], &b0);
    c0 = two_prod(a[0], b[1], &e0);
    d0 = two_prod(a[1], b[0], &f0);
    g0 = two_prod(a[0], b[2], &j0);
    h0 = two_prod(a[1], b[1], &k0);
    i0 = two_prod(a[2], b[0], &l0);
    m0 = a[0] * b[3];
    n0 = a[1] * b[2];
    o0 = a[2] * b[1];
    p0 = a[3] * b[0];
    c1 = two_sum(c0, d0, &d1);
    e1 = two_sum(e0, f0, &f1);
    g1 = two_sum(g0, i0, &i1);
    j1 = j0 + l0;
    m1 = m0 + p0;
    n1 = n0 + o0;
    b2 = two_sum(b0, c1, &c2);
    e2 = two_sum(e1, h0, &h2);
    f2 = f1 + j1;
    i2 = i1 + k0;
    m2 = m1 + n1;
    a3 = quick_two_sum(a0, b2, &b3);
    c3 = quick_two_sum(c2, d1, &d3);
    e3 = two_sum(e2, g1, &g3);
    f3 = f2 + m2;
    h3 = h2 + i2;
    c4 = two_sum(c3, e3, &e4);
    d4 = d3 + h3;
    f4 = f3 + g3;
    d5 = d4 + e4;
    c6 = two_sum(c4, d5, &d6);
    b7 = two_sum(b3, c6, &c7);
    d7 = d6 + f4;
    a8 = quick_two_sum(a3, b7, &b8);
    c8 = two_sum(c7, d7, &d8);
    b9 = two_sum(b8, c8, &c9);
    c10 = quick_two_sum(c9, d8, &d10);

	c[0] = a8;
	c[1] = b9;
	c[2] = c10;
	c[3] = d10;
}

// c_td_mulq_accurate
static inline void c_td_mulq(const double *a, const double *b, double *c)
{
	// acculate mul
	double p0, p1, p2, p3, p4, p5;
	double q0, q1, q2, q3, q4, q5;
	double p6, p7, p8, p9;
	double q6, q7, q8, q9;
	double r0, r1;
	double t0, t1;
	double s0, s1, s2;

	p0 = two_prod(a[0], b[0], &q0);

	p1 = two_prod(a[0], b[1], &q1);
	p2 = two_prod(a[1], b[0], &q2);

	p3 = two_prod(a[0], b[2], &q3);
	p4 = two_prod(a[1], b[1], &q4);
	p5 = two_prod(a[2], b[0], &q5);

	/* Start Accumulation */
	three_sum(&p1, &p2, &q0);

	/* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
	three_sum(&p2, &q1, &q2);
	three_sum(&p3, &p4, &p5);

	/* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
	s0 = two_sum(p2, p3, &t0);
	s1 = two_sum(q1, p4, &t1);
	s2 = q2 + p5;
	s1 = two_sum(s1, t0, &t0);
	s2 += (t0 + t1);

	/* O(eps^3) order terms */
	//p6 = two_prod(a[0], b[3], &q6);
	//p7 = two_prod(a[1], b[2], &q7);
	//p8 = two_prod(a[2], b[1], &q8);
	//p9 = two_prod(a[3], b[0], &q9);

	/* Nine-Two-Sum of q0, s1, q3, q4, q5, p6, p7, p8, p9. */
	q0 = two_sum(q0, q3, &q3);
	q4 = two_sum(q4, q5, &q5);
	//p6 = two_sum(p6, p7, &p7);
	//p8 = two_sum(p8, p9, &p9);

	/* Compute (t0, t1) = (q0, q3) + (q4, q5). */
	t0 = two_sum(q0, q4, &t1);
	t1 += (q3 + q5);

	/* Compute (r0, r1) = (p6, p7) + (p8, p9). */
	//r0 = two_sum(p6, p8, &r1);
	//r1 += (p7 + p9);

	/* Compute (q3, q4) = (t0, t1) + (r0, r1). */
	//q3 = two_sum(t0, r0, &q4);
	//q4 += (t1 + r1);

	/* Compute (t0, t1) = (q3, q4) + s1. */
	t0 = two_sum(q3, s1, &t1);
	//t1 += q4;

	/* O(eps^4) terms -- Nine-One-Sum */
	//t1 += a[1] * b[3] + a[2] * b[2] + a[3] * b[1] + q6 + q7 + q8 + q9 + s2;

	//renorm4(&p0, &p1, &s0, &t0, &t1);
	renorm(&p0, &p1, &s0, &t0);
//	return qd_real(p0, p1, s0, t0);
	c[0] = p0;
	c[1] = p1;
	c[2] = s0;
	//c[3] = t0;

#if 0
  qd_real cc;
  cc = qd_real(a) * qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/* quad-double * double-double */
/* a0 * b0                        0
        a0 * b1                   1
        a1 * b0                   2
             a1 * b1              3
             a2 * b0              4
                  a2 * b1         5
                  a3 * b0         6
                       a3 * b1    7 */
// c := a * (dd)b
static inline void c_qd_mul_qd_dd(const double *a, const double *b, double *c)
{
	double p0, p1, p2, p3, p4;
	double q0, q1, q2, q3, q4;
	double s0, s1, s2;
	double t0, t1;

	p0 = two_prod(a[0], DD_HI(b) , &q0);
	p1 = two_prod(a[0], DD_LOW(b), &q1);
	p2 = two_prod(a[1], DD_HI(b) , &q2);
	p3 = two_prod(a[1], DD_LOW(b), &q3);
	p4 = two_prod(a[2], DD_HI(b) , &q4);

	three_sum(&p1, &p2, &q0);

	/* Five-Three-Sum */
	three_sum(&p2, &p3, &p4);
	q1 = two_sum(q1, q2, &q2);
	s0 = two_sum(p2, q1, &t0);
	s1 = two_sum(p3, q2, &t1);
	s1 = two_sum(s1, t0, &t0);
	s2 = t0 + t1 + p4;
	p2 = s0;

	p3 = a[2] * DD_HI(b) + a[3] * DD_LOW(b) + q3 + q4;
	three_sum2(&p3, &q0, &s1);
	p4 = q0 + s2;

	renorm4(&p0, &p1, &p2, &p3, &p4);
	//return qd_real(p0, p1, p2, p3);
	c[0] = p0;
	c[1] = p1;
	c[2] = p2;
	c[3] = p3;

#if 0
  qd_real cc;
  cc = qd_real(a) * dd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// c := (dd)a * b
static inline void c_qd_mul_dd_qd(const double *a, const double *b, double *c)
{
	c_qd_mul_qd_dd(b, a, c);

#if 0
  qd_real cc;
  cc = dd_real(a) * qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// c := a * (double)b
static inline void c_qd_mul_qd_d(const double *a, double b, double *c)
{
	double p0, p1, p2, p3;
	double q0, q1, q2;
	double s0, s1, s2, s3, s4;

	p0 = two_prod(a[0], b, &q0);
	p1 = two_prod(a[1], b, &q1);
	p2 = two_prod(a[2], b, &q2);
	p3 = a[3] * b;

	s0 = p0;

	s1 = two_sum(q0, p1, &s2);

	three_sum(&s2, &q1, &p2);

	three_sum2(&q1, &q2, &p3);
	s3 = q1;

	s4 = q2 + p2;

	renorm4(&s0, &s1, &s2, &s3, &s4);
//	return qd_real(s0, s1, s2, s3);
	c[0] = s0;
	c[1] = s1;
	c[2] = s2;
	c[3] = s3;

#if 0
  qd_real cc;
  cc = qd_real(a) * b;
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// c := (double)a * b
static inline void c_qd_mul_d_qd(double a, const double *b, double *c)
{
	c_qd_mul_qd_d(b, a, c);

#if 0
  qd_real cc;
  cc = a * qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// c := a * (double)b
static inline void c_td_mulq_td_d(const double *a, double b, double *c)
{
	double p0, p1, p2; // p3;
	double q0, q1, q2;
	double s0, s1, s2, s3; //, s4;

	p0 = two_prod(a[0], b, &q0);
	p1 = two_prod(a[1], b, &q1);
	//p2 = two_prod(a[2], b, &q2);
	//p3 = a[3] * b;
	p2 = a[2] * b;

	s0 = p0;

	s1 = two_sum(q0, p1, &s2);

	three_sum(&s2, &q1, &p2);

	//three_sum2(&q1, &q2, &p3);
	s3 = q1 + p2;

	//s4 = q2 + p2;

	//renorm4(&s0, &s1, &s2, &s3, &s4);	
	renorm(&s0, &s1, &s2, &s3); // , &s4);
//	return qd_real(s0, s1, s2, s3);
	c[0] = s0;
	c[1] = s1;
	c[2] = s2;
	//c[3] = s3;

#if 0
  qd_real cc;
  cc = qd_real(a) * b;
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/* quad-double * quad-double */
/* a0 * b0                    0
        a0 * b1               1
        a1 * b0               2
             a0 * b2          3
             a1 * b1          4
             a2 * b0          5
                  a0 * b3     6
                  a1 * b2     7
                  a2 * b1     8
                  a3 * b0     9  */
static inline void c_qd_mul_sloppy(const double *a, const double *b, double *c)
{
	double p0, p1, p2, p3, p4, p5;
	double q0, q1, q2, q3, q4, q5;
	double t0, t1;
	double s0, s1, s2;

	p0 = two_prod(a[0], b[0], &q0);

	p1 = two_prod(a[0], b[1], &q1);
	p2 = two_prod(a[1], b[0], &q2);

	p3 = two_prod(a[0], b[2], &q3);
	p4 = two_prod(a[1], b[1], &q4);
	p5 = two_prod(a[2], b[0], &q5);

	/* Start Accumulation */
	three_sum(&p1, &p2, &q0);

	/* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
	three_sum(&p2, &q1, &q2);
	three_sum(&p3, &p4, &p5);

	/* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
	s0 = two_sum(p2, p3, &t0);
	s1 = two_sum(q1, p4, &t1);
	s2 = q2 + p5;
	s1 = two_sum(s1, t0, &t0);
	s2 += (t0 + t1);

	/* O(eps^3) order terms */
	s1 += a[0]*b[3] + a[1]*b[2] + a[2]*b[1] + a[3]*b[0] + q0 + q3 + q4 + q5;
	//renorm(p0, p1, s0, s1, s2);
	renorm4(&p0, &p1, &s0, &s1, &s2);
//	return qd_real(p0, p1, s0, s1);
	c[0] = p0;
	c[1] = p1;
	c[2] = s0;
	c[3] = s1;
}

// td multiplication based on qd mul
static inline void c_td_mulq_sloppy(const double *a, const double *b, double *c)
{
	double p0, p1, p2, p3, p4, p5;
	double q0, q1, q2, q3, q4, q5;
	double t0, t1;
	double s0, s1, s2;

	p0 = two_prod(a[0], b[0], &q0);

	p1 = two_prod(a[0], b[1], &q1);
	p2 = two_prod(a[1], b[0], &q2);

	p3 = two_prod(a[0], b[2], &q3);
	p4 = two_prod(a[1], b[1], &q4);
	p5 = two_prod(a[2], b[0], &q5);

	/* Start Accumulation */
	three_sum(&p1, &p2, &q0);

	/* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
	three_sum(&p2, &q1, &q2);
	three_sum(&p3, &p4, &p5);

	/* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
	s0 = two_sum(p2, p3, &t0);
	s1 = two_sum(q1, p4, &t1);
	s2 = q2 + p5;
	s1 = two_sum(s1, t0, &t0);
	s2 += (t0 + t1);

	/* O(eps^3) order terms */
	//s1 += a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0] + q0 + q3 + q4 + q5;
	s1 += a[1] * b[2] + a[2] * b[1] + q0 + q3 + q4 + q5;
	renorm(&p0, &p1, &s0, &s1);
	//renorm4(&p0, &p1, &s0, &s1, &s2);
//	return qd_real(p0, p1, s0, s1);
	c[0] = p0;
	c[1] = p1;
	c[2] = s0;
	//c[3] = s1;
}


/* quad-double ^ 2  = (x0 + x1 + x2 + x3) ^ 2
                    = x0 ^ 2 + 2 x0 * x1 + (2 x0 * x2 + x1 ^ 2)
                               + (2 x0 * x3 + 2 x1 * x2)           */
static inline void c_qd_sqr(const double *a, double *c)
{
	double p0, p1, p2, p3, p4, p5;
	double q0, q1, q2, q3;
	double s0, s1;
	double t0, t1;

	p0 = two_sqr(a[0], &q0);
	p1 = two_prod(2.0 * a[0], a[1], &q1);
	p2 = two_prod(2.0 * a[0], a[2], &q2);
	p3 = two_sqr(a[1], &q3);

	p1 = two_sum(q0, p1, &q0);

	q0 = two_sum(q0, q1, &q1);
	p2 = two_sum(p2, p3, &p3);

	s0 = two_sum(q0, p2, &t0);
	s1 = two_sum(q1, p3, &t1);

	s1 = two_sum(s1, t0, &t0);
	t0 += t1;

	s1 = quick_two_sum(s1, t0, &t0);
	p2 = quick_two_sum(s0, s1, &t1);
	p3 = quick_two_sum(t1, t0, &q0);

	p4 = 2.0 * a[0] * a[3];
	p5 = 2.0 * a[1] * a[2];

	p4 = two_sum(p4, p5, &p5);
	q2 = two_sum(q2, q3, &q3);

	t0 = two_sum(p4, q2, &t1);
	t1 = t1 + p5 + q3;

	p3 = two_sum(p3, t0, &p4);
	p4 = p4 + q0 + t1;

	renorm4(&p0, &p1, &p2, &p3, &p4);
//  return qd_real(p0, p1, p2, p3);
	c[0] = p0;
	c[1] = p1;
	c[2] = p2;
	c[3] = p3;

}


/********** Self-Multiplication **********/
/* selfmul */
// b := b * a
static inline void c_qd_selfmul(const double *a, double *b)
{
	double bb[QDSIZE];

	c_qd_copy(b, bb);
	c_qd_mul(a, bb, b);

#if 0
  qd_real bb(b);
  bb *= qd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b * (dd)a
static inline void c_qd_selfmul_dd(const double *a, double *b)
{
	double bb[QDSIZE];

	c_qd_copy(b, bb);
	c_qd_mul_qd_dd(bb, a, b);

#if 0
  qd_real bb(b);
  bb *= dd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b * (doule)a
static inline void c_qd_selfmul_d(double a, double *b)
{
	double bb[QDSIZE];

	c_qd_copy(b, bb);
	c_qd_mul_qd_d(bb, a, b);

#if 0
  qd_real bb(b);
  bb *= a;
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

/********** Divisions **********/

/* div */
// c := a / b
static inline void c_qd_div_accurate(const double *a, const double *b, double *c)
{
	//qd_real::accurate_div
	double q0, q1, q2, q3, q4;
	double r[QDSIZE], tmp[QDSIZE];
	//qd_real r;

	q0 = a[0] / b[0];
	//r = a - (b * q0);
	c_qd_mul_qd_d(b, q0, tmp);
	c_qd_sub(a, tmp, r);

	q1 = r[0] / b[0];
	//r -= (b * q1);
	c_qd_mul_qd_d(b, q1, tmp);
	c_qd_selfsub(tmp, r);
	//c_qd_sub(r, tmp, r);

	q2 = r[0] / b[0];
	//r -= (b * q2);
	c_qd_mul_qd_d(b, q2, tmp);
	c_qd_selfsub(tmp, r);
	//c_qd_sub(r, tmp, r);

	q3 = r[0] / b[0];

	//r -= (b * q3);
	c_qd_mul_qd_d(b, q3, tmp);
	c_qd_selfsub(tmp, r);
	//c_qd_sub(r, tmp, r);

	q4 = r[0] / b[0];

	renorm4(&q0, &q1, &q2, &q3, &q4);

//	return qd_real(q0, q1, q2, q3);
	c[0] = q0;
	c[1] = q1;
	c[2] = q2;
	c[3] = q3;

#if 0 //
  qd_real cc;
  cc = qd_real(a) / qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/* quad-double / quad-double */
// c := a / b
static inline void c_qd_div_sloppy(const double *a, const double *b, double *c)
{
	//qd_real qd_real::sloppy_div(const qd_real &a, const qd_real &b) {
	double q0, q1, q2, q3;
	double r[QDSIZE], tmp[QDSIZE];
	//qd_real r;

	q0 = a[0] / b[0];
	//r = a - (b * q0);
	c_qd_mul_qd_d(b, q0, tmp);
	c_qd_sub(a, tmp, r);

	q1 = r[0] / b[0];
	//r -= (b * q1);
	c_qd_mul_qd_d(b, q1, tmp);
	c_qd_selfsub(tmp, r);
	//c_qd_sub(r, tmp, r);

	q2 = r[0] / b[0];
	//r -= (b * q2);
	c_qd_mul_qd_d(b, q2, tmp);
	c_qd_selfsub(tmp, r);
	//c_qd_sub(r, tmp, r);

	q3 = r[0] / b[0];

	renorm(&q0, &q1, &q2, &q3);

	//return qd_real(q0, q1, q2, q3);
	c[0] = q0;
	c[1] = q1;
	c[2] = q2;
	c[3] = q3;
}


#ifndef USE_QD_DIV_ACCURATE
	#ifdef BNC_USE_FMA_DIV
		#define c_qd_div bnc_qd_div_fma
	#else
		#define c_qd_div c_qd_div_sloppy
	#endif
//	#define c_qd_div c_qd_div_accurate
#else // USE_QD_DIV_ACCURATE
	#define c_qd_div c_qd_div_accurate
#endif //  USE_QD_DIV_ACCURATE

/* quad-double / double-double */
// c := a / (dd)b
static inline void c_qd_div_qd_dd(const double *a, const double *b, double *c)
{
	//qd_real qd_real::sloppy_div(const qd_real &a, const dd_real &b) {
	double q0, q1, q2, q3;
	double r[QDSIZE], tmp[QDSIZE], qd_b[QDSIZE];
	//qd_real r;
	//qd_real qd_b(b);

	c_qd_copy_dd(b, qd_b);

	q0 = a[0] / DD_HI(b);
	//r = a - q0 * qd_b;
	c_qd_mul_d_qd(q0, qd_b, tmp);
	c_qd_sub(a, tmp, r);

	q1 = r[0] / DD_HI(b);
	//r -= (q1 * qd_b);
	c_qd_mul_d_qd(q1, qd_b, tmp);
	c_qd_selfsub(tmp, r);

	q2 = r[0] / DD_HI(b);
	//r -= (q2 * qd_b);
	c_qd_mul_d_qd(q2, qd_b, tmp);
	c_qd_selfsub(tmp, r);

	q3 = r[0] / DD_HI(b);

	//::renorm(q0, q1, q2, q3);
	renorm(&q0, &q1, &q2, &q3);
	//return qd_real(q0, q1, q2, q3);

	c[0] = q0;
	c[1] = q1;
	c[2] = q2;
	c[3] = q3;

#if 0 
  qd_real cc;
  cc = qd_real(a) / dd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/* double-double / quad-double */
// c := (dd)a / b
static inline void c_qd_div_dd_qd(const double *a, const double *b, double *c)
{
	double aa[QDSIZE];

	c_qd_copy_dd(a, aa);
	c_qd_div(aa, b, c);

#if 0 
  qd_real cc;
  cc = dd_real(a) / qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/* quad-double / double */
// c := a / (double)b
static inline void c_qd_div_qd_d(const double *a, double b, double *c)
{
	double bb[QDSIZE];

	c_qd_copy_d(b, bb);
	c_qd_div(a, bb, c);

#if 0 
  qd_real cc;
  cc = qd_real(a) / b;
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/* double / quad-double */
// c := (double)a / b
static inline void c_qd_div_d_qd(double a, const double *b, double *c)
{
	double aa[QDSIZE];

	c_qd_copy_d(a, aa);
	c_qd_div(aa, b, c);

#if 0 
  qd_real cc;
  cc = a / qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/********** Self-Divisions **********/
/* selfdiv */
// b := b / a
static inline void c_qd_selfdiv(const double *a, double *b) 
{
	double bb[QDSIZE];

	c_qd_copy(b, bb);
	c_qd_div(bb, a, b);

#if 0
  qd_real bb(b);
  bb /= qd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b / (dd)a
static inline void c_qd_selfdiv_dd(const double *a, double *b)
{
	double bb[QDSIZE];

	c_qd_copy(b, bb);
	c_qd_div_qd_dd(bb, a, b);

#if 0
  qd_real bb(b);
  bb /= dd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
static inline void c_qd_selfdiv_d(double a, double *b)
{
	double bb[QDSIZE];

	c_qd_copy(b, bb);
	c_qd_div_qd_d(bb, a, b);

#if 0
  qd_real bb(b);
  bb /= a;
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// c := (a[0] * b, a[1] * b, a[2] * b, a[3] * b)
static inline void c_qd_mul_pwr2(const double *a, double b, double *c)
{
	//return qd_real(a[0] * b, a[1] * b, a[2] * b, a[3] * b);
	c[0] = a[0] * b;
	c[1] = a[1] * b;
	c[2] = a[2] * b;
	c[3] = a[3] * b;
}

// qdva := x
static inline void c_qd_set(const double *x, double *qdval)
{
	qdval[0] = x[0];
	qdval[1] = x[1];
	qdval[2] = x[2];
	qdval[3] = x[3];
}

// qdval := 0
static inline void c_qd_set0(double *qdval)
{
	qdval[0] = 0.0;
	qdval[1] = 0.0;
	qdval[2] = 0.0;
	qdval[3] = 0.0;
}


// 2024-04-24(Tue) Incorrect! -> use rqd_sqrt_mpfr
static inline void c_qd_sqrt(const double *a, double *b)
{
//QD_API qd_real sqrt(const qd_real &a) {
  /* Strategy:  

     Perform the following Newton iteration:

       x' = x + (1 - a * x^2) * x / 2;
       
     which converges to 1/sqrt(a), starting with the
     double precision approximation to 1/sqrt(a).
     Since Newton's iteration more or less doubles the
     number of correct digits, we only need to perform it 
     twice.
  */
  double r[QDSIZE], h[QDSIZE], tmp[QDSIZE];

	//if (QD_ISZERO(a))
	if (qd_is_zero(a))
	{
		c_qd_set0(b);
		return;
	}

	//if (QD_ISNEGATIVE(a))
	if (qd_is_negative(a))
	{
		//fprintf(stderr, "(qd_real::sqrt): Negative argument.");
		fprintf(stderr, "(c_qd_sqrt): Negative argument.");
		b[0] = QD_NAN;
		return;
	}

//  qd_real r = (1.0 / std::sqrt(a[0]));
//  qd_real h = mul_pwr2(a, 0.5);
//	r[0] = 1.0 / sqrt(a[0]); r[1] = 0.0; r[2] = 0.0; r[3] = 0.0;

//inline qd_real mul_pwr2(const qd_real &a, double b) {
//  return qd_real(a[0] * b, a[1] * b, a[2] * b, a[3] * b);
//}
	h[0] = 1.0; h[1] = 0.0; h[2] = 0.0; h[3] = 0.0;
	//h[0] = a[0] * 0.5; h[1] = a[1] * 0.5; h[2] = a[2] * 0.5; h[3] = a[3] * 0.5;
#ifndef __cplusplus
	r[0] = (double)sqrt((double)a[0]);
#else // __cplusplus
	r[0] = (double)std::sqrt((double)a[0]);
#endif // __cplusplus
	r[1] = 0.0; r[2] = 0.0; r[3] = 0.0;
	c_qd_div(h, r, r);

	c_qd_mul_pwr2(a, 0.5, h);

//  r += ((0.5 - h * sqr(r)) * r);
//  r += ((0.5 - h * sqr(r)) * r);
//  r += ((0.5 - h * sqr(r)) * r);

	c_qd_sqr(r, tmp);
	c_qd_mul(h, tmp, tmp);
	c_qd_sub_d_qd(0.5, tmp, tmp);
	c_qd_mul(tmp, r, tmp);
	c_qd_add(tmp, r, r);

	c_qd_sqr(r, tmp);
	c_qd_mul(h, tmp, tmp);
	c_qd_sub_d_qd(0.5, tmp, tmp);
	c_qd_mul(tmp, r, tmp);
	c_qd_add(tmp, r, r);

	c_qd_sqr(r, tmp);
	c_qd_mul(h, tmp, tmp);
	c_qd_sub_d_qd(0.5, tmp, tmp);
	c_qd_mul(tmp, r, tmp);
	c_qd_add(tmp, r, r);

//  r *= a;
//	c_qd_selfmul(a, r);
	c_qd_mul(r, a, r);
	c_qd_copy(r, b);
//  return r;
#if 0
  qd_real bb;
  bb = sqrt(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := |a| 
static inline void c_qd_abs(const double *a, double *b)
{
	// return (a[0] < 0.0) ? -a : a;
	if(a[0] < 0.0)
		c_qd_neg(a, b);
	else
		c_qd_copy(a, b);
#if 0
  qd_real bb;
  bb = abs(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

static inline void c_qd_npwr(const double *a, int n, double *b) {
#if 0
  qd_real bb;
  bb = npwr(qd_real(a), n);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

static inline void c_qd_nroot(const double *a, int n, double *b) {
#if 0
  qd_real bb;
  bb = nroot(qd_real(a), n);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// rount to nearest integer
#define nint(a) (round(a))

static inline void c_qd_nint(const double *a, double *b)
{
	double x0, x1, x2, x3;

	x0 = nint(a[0]);
	x1 = x2 = x3 = 0.0;

	if (x0 == a[0])
	{
	  /* First double is already an integer. */
	  x1 = nint(a[1]);

	  if (x1 == a[1]) {
	    /* Second double is already an integer. */
	    x2 = nint(a[2]);
	    
	    if (x2 == a[2]) {
	      /* Third double is already an integer. */
	      x3 = nint(a[3]);
	    } else {
	      if(fabs(x2 - a[2]) == 0.5 && a[3] < 0.0) {
	        x2 -= 1.0;
	      }
	    }

	  } else {
	    if (fabs(x1 - a[1]) == 0.5 && a[2] < 0.0) {
	        x1 -= 1.0;
	    }
	  }

	} else {
	  /* First double is not an integer. */
	    if (fabs(x0 - a[0]) == 0.5 && a[1] < 0.0) {
	        x0 -= 1.0;
	    }
	}

	//renorm(&x0, &x1, &x2, &x3);
	//return qd_real(x0, x1, x2, x3);
	b[0] = x0;
	b[1] = x1;
	b[2] = x2;
	b[3] = x3;

#if 0
  qd_real bb;
  bb = nint(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

static inline void c_qd_floor(const double *a, double *b)
{
	double x0, x1, x2, x3;
	x1 = x2 = x3 = 0.0;
	x0 = floor(a[0]);

	if (x0 == a[0])
	{
		x1 = floor(a[1]);
		
		if (x1 == a[1])
		{
			x2 = floor(a[2]);

			if (x2 == a[2])
			{
				x3 = floor(a[3]);
			}
		}

		renorm(&x0, &x1, &x2, &x3);
		//return qd_real(x0, x1, x2, x3);
		b[0] = x0;
		b[1] = x1;
		b[2] = x2;
		b[3] = x3;
	}

 // return qd_real(x0, x1, x2, x3);
	b[0] = x0;
	b[1] = x1;
	b[2] = x2;
	b[3] = x3;

#if 0
  qd_real bb;
  bb = floor(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
static inline void c_qd_ceil(const double *a, double *b)
{
	double x0, x1, x2, x3;
	x1 = x2 = x3 = 0.0;
	x0 = ceil(a[0]);

	if (x0 == a[0])
	{
		x1 = ceil(a[1]);
		
		if (x1 == a[1])
		{
			x2 = ceil(a[2]);

			if (x2 == a[2])
			{
				x3 = ceil(a[3]);
			}
		}

		renorm(&x0, &x1, &x2, &x3);
		//return qd_real(x0, x1, x2, x3);
		b[0] = x0;
		b[1] = x1;
		b[2] = x2;
		b[3] = x3;
	}

	//return qd_real(x0, x1, x2, x3);
	b[0] = x0;
	b[1] = x1;
	b[2] = x2;
	b[3] = x3;
#if 0
  qd_real bb;
  bb = ceil(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

static inline void c_qd_aint(const double *a, double *b)
{
//  return (a[0] >= 0) ? floor(a) : ceil(a);
	if(a[0] >= 0.0)
		c_qd_floor(a, b);
	else
		c_qd_ceil(a, b);

#if 0
  qd_real bb;
  bb = aint(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}


// (dtq-0.0.3 ports; the previous versions of these functions were empty stubs)
static inline void c_qd_log(const double *a, double *b) {
	bnc_qd_log(a, b);
}
static inline void c_qd_log10(const double *a, double *b) {
	bnc_qd_log10(a, b);
}
static inline void c_qd_exp(const double *a, double *b) {
	bnc_qd_exp(a, b);
}

static inline void c_qd_sin(const double *a, double *b) {
	bnc_qd_sin(a, b);
}
static inline void c_qd_cos(const double *a, double *b) {
	bnc_qd_cos(a, b);
}
// tan(a) = sin(a) / cos(a)
static inline void c_qd_tan(const double *a, double *b) {
	double s[4], c[4];

	bnc_qd_sincos(a, s, c);
	c_qd_div(s, c, b);
}

static inline void c_qd_asin(const double *a, double *b) {
#if 0
 qd_real bb;
  bb = asin(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
static inline void c_qd_acos(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = acos(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
static inline void c_qd_atan(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = atan(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

static inline void c_qd_atan2(const double *a, const double *b, double *c) {
#if 0
  qd_real cc;
  cc = atan2(qd_real(a), qd_real(b));
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

static inline void c_qd_sinh(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = sinh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
static inline void c_qd_cosh(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = cosh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
static inline void c_qd_tanh(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = tanh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

static inline void c_qd_asinh(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = asinh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

static inline void c_qd_acosh(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = acosh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
static inline void c_qd_atanh(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = atanh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

static inline void c_qd_sincos(const double *a, double *s, double *c) {
#if 0
  qd_real ss, cc;
  sincos(qd_real(a), ss, cc);
  TO_DOUBLE_PTR(cc, c);
  TO_DOUBLE_PTR(ss, s);
#endif // 0
}

static inline void c_qd_sincosh(const double *a, double *s, double *c) {
#if 0
  qd_real ss, cc;
  sincosh(qd_real(a), ss, cc);
  TO_DOUBLE_PTR(cc, c);
  TO_DOUBLE_PTR(ss, s);
#endif // 0
}

static inline void c_qd_read(const char *s, double *a) {
#if 0
  qd_real aa(s);
  TO_DOUBLE_PTR(aa, a);
#endif // 0
}

static inline void c_qd_swrite(const double *a, int precision, char *s, int len) {
#if 0
  qd_real(a).write(s, len, precision);
#endif // 0
}

static inline void c_qd_write(const double *a) {
#if 0
  std::cout << qd_real(a).to_string(qd_real::_ndigits) << std::endl;
#endif // 0
}

static inline void c_qd_rand(double *a) {
#if 0
  qd_real aa;
  aa = qdrand();
  TO_DOUBLE_PTR(aa, a);
#endif // 0
}

static inline void c_qd_comp(const double *a, const double *b, int *result) 
{
	//  qd_real aa(a), bb(b);
	//if (aa < bb)
	if (a[0] < b[0] || (a[0] == b[0] && (a[1] < b[1] || (a[1] == b[1] && (a[2] < b[2] || (a[2] == b[2] && a[3] < b[3]))))))
	    *result = -1;
	//else if (aa > bb)
	else if (a[0] > b[0] || (a[0] == b[0] && (a[1] > b[1] || (a[1] == b[1] && (a[2] > b[2] || (a[2] == b[2] && a[3] > b[3]))))))
		*result = 1;
	else 
		*result = 0;
}

static inline void c_qd_comp_qd_d(const double *a, double b, int *result)
{
	// qd_real aa(a);
	//if (aa < b)
	if(a[0] < b || (a[0] == b && a[1] < 0.0))
		*result = -1;
	//else if (aa > b)
	else if (a[0] > b || (a[0] == b && a[1] > 0.0))
		*result = 1;
	else 
		*result = 0;
}

static inline void c_qd_comp_d_qd(double a, const double *b, int *result)
{
/* qd_real bb(b);
  if (a < bb)
    *result = -1;
  else if (a > bb)
    *result = 1;
  else 
    *result = 0;
*/
	c_qd_comp_qd_d(b, a, result);
	*result = -(*result);
}

static inline void c_qd_pi(double *a)
{
	c_qd_copy(const_qd_pi, a);

#if 0
  TO_DOUBLE_PTR(qd_real::_pi, a);
#endif // 0
}

/*******************************************/
/* Triple precision arithmetic             */
/*******************************************/
// N.Fabiano et.al.

/* (td)b := (td)a */
static inline void c_td_copy(const double *a, double *b)
{
	b[0] = a[0];
	b[1] = a[1];
	b[2] = a[2];
}

// (qd)c := (td)a
static inline void c_qd_copy_td(const double *a, double *c)
{
	c[0] = a[0];
	c[1] = a[1];
	c[2] = a[2];
	c[3] = 0.0;
}

/* (td)b := (qd)a */
static inline void c_td_copy_qd(const double *a, double *b)
{
	b[0] = a[0];
	b[1] = a[1];
	b[2] = a[2];
}

/* (td)b := (dd)a */
static inline void c_td_copy_dd(const double *a, double *b)
{
	b[0] = a[0];
	b[1] = a[1];
	b[2] = 0.0;
}

/* (td)b := (double)a */
static inline void c_td_copy_d(double a, double *b)
{
	b[0] = a;
	b[1] = 0.0;
	b[2] = 0.0;
}

// e[n] := vec_sum(x[n])
static inline void vec_sum(double *e, const double *x, int n)
{
    double s;

	s = x[--n];
	while(--n >= 0)
		s = two_sum(x[n], s, &e[n + 1]);
	e[0] = s;
}

// y[n] := vec_sum_err_branch(vseb)(k)(e[n])
static inline void vseb(double *y, int ny, const double *e, int ne)
{
	int i, j;
	double r, eps, temp, in_y[16];

//	printf("ny = %d\n", ny);
//	if(ny > ne)
//		ny = ne;

	j = 0;
	eps = e[0];
	for(i = 0; i < (ne - 2); i++)
	{
		r = two_sum(eps, e[i + 1], &temp);
		if(temp != 0.0)
		{
			in_y[j] = r;
			eps = temp;
			j++;
		}
		else
			eps = r;
//		printf("i, j = %d, %d\n", i, j);
	}
//	printf("j, j+1 = %d, %d\n", j, j + 1);
	in_y[j] = two_sum(eps, e[ne - 1], &in_y[j + 1]);

	for(i = j + 2; i < ne; i++)
		in_y[i] = 0.0;

	for(i = 0; i < ny; i++)
		y[i] = in_y[i];

}

// r[3] := to_td(a, b, c)
static inline void c_to_td(double *r, double a, double b, double c)
{
	double d[3], e[3];

	d[0] = two_sum(a, b, &d[1]);
	d[2] = c;
	vec_sum(e, d, 3);
	vseb(r, 3, e, 3);
}

// [TODO] double := c_td2d(a[3])

// Merge a[na] & b[nb] into c[na + nb]
// H.Okumura, "Elementary algorithms in C", 1991.
static inline void merge(double *c, double *a, int na, double *b, int nb)
{
	int i, j, k;

	i = j = k = 0;
	while((i < na) && (j < nb))
	{
		if(fabs(a[i]) >= fabs(b[j]))
			c[k++] = a[i++];
		else
			c[k++] = b[j++];
	}
	while(i < na)
		c[k++] = a[i++];
	while(j < nb)
		c[k++] = b[j++];
}

// c[3] := a[3] + b[3]
//static inline void c_td_add(double *c, double *a, double *b)
static inline void c_td_add(double *a, double *b, double *c)
{
	double z[6], e[6];

//	printf("c_td_add "); fflush(stdout);
	merge(z, a, 3, b, 3);
//	printf("merge"); fflush(stdout);
	vec_sum(e, z, 6);
//	printf(" vec_sum"); fflush(stdout);
	vseb(c, 3, e, 6);
//	printf(" vseb\n"); fflush(stdout);
}

// 2025-12-24(Wed) T.Kouya
// Branch free algorithm
//void Add3(const double x[3], const double y[3], double z[3]) {
static inline void c_td_add_bf(double *a, double *b, double *c)
{
	double a0, b0, c0, d0, e0, f0;
	double a1, b1, c1, d1, e1, f1;
	double a2, b2, c2, d2, e2;
	double a3, b3, c3, d3;
	double a4, b4, c4, d4;
	double a5, b5, c5, d5;
	double a6, b6, c6;
	double a7, b7, c7;
	double a8, b8, c8;

	a0 = a[0];
    b0 = b[0];
    c0 = a[1];
    d0 = b[1];
    e0 = a[2];
    f0 = b[2];
    a1 = two_sum(a0, b0, &b1);
    c1 = two_sum(c0, d0, &d1);
    e1 = two_sum(e0, f0, &f1);
    a2 = quick_two_sum(a1, c1, &c2);
    b2 = b1 + f1;
    d2 = two_sum(d1, e1, &e2);
    a3 = quick_two_sum(a2, d2, &d3);
    b3 = two_sum(b2, c2, &c3);
    c4 = c3 + e2;
    c5 = two_sum(c4, d3, &d5);
    b6 = two_sum(b3, c5, &c6);
    a7 = quick_two_sum(a3, b6, &b7);
    c7 = c6 + d5;
    b8 = quick_two_sum(b7, c7, &c8);

	c[0] = a7;
	c[1] = b8;
	c[2] = c8;
}

// 2026-05-02(Sat) T.Kouya
// Branch free algorithm: c[3] := a[3] + b  (b is a double)
// Derived from c_td_add_bf by setting b[1]=0, b[2]=0.
// (two_sum(0,x) returns (x,0) exactly, so the b[1]/b[2]-related
//  nodes collapse and the remaining graph stays branch free.)
static inline void c_td_add_d_bf(double *a, double b, double *c)
{
	double a1, b1;
	double a2, c2;
	double a3, d3;
	double b3, c3;
	double c5, d5;
	double b6, c6;
	double a7, b7, c7;
	double b8, c8;

	a1 = two_sum(a[0], b, &b1);
	a2 = quick_two_sum(a1, a[1], &c2);
	a3 = quick_two_sum(a2, a[2], &d3);
	b3 = two_sum(b1, c2, &c3);
	c5 = two_sum(c3, d3, &d5);
	b6 = two_sum(b3, c5, &c6);
	a7 = quick_two_sum(a3, b6, &b7);
	c7 = c6 + d5;
	b8 = quick_two_sum(b7, c7, &c8);
	c[0] = a7;
	c[1] = b8;
	c[2] = c8;
}

// 2026-05-02(Sat) T.Kouya
// Branch free algorithm: c[3] := a[3] + b[2]  (b is a double-double)
// Derived from c_td_add_bf by setting b[2]=0.
static inline void c_td_add_dd_bf(double *a, double *b, double *c)
{
	double a1, b1;
	double c1, d1;
	double a2, c2;
	double d2, e2;
	double a3, d3;
	double b3, c3;
	double c4, c5, d5;
	double b6, c6;
	double a7, b7, c7;
	double b8, c8;

	a1 = two_sum(a[0], b[0], &b1);
	c1 = two_sum(a[1], b[1], &d1);
	a2 = quick_two_sum(a1, c1, &c2);
	d2 = two_sum(d1, a[2], &e2);
	a3 = quick_two_sum(a2, d2, &d3);
	b3 = two_sum(b1, c2, &c3);
	c4 = c3 + e2;
	c5 = two_sum(c4, d3, &d5);
	b6 = two_sum(b3, c5, &c6);
	a7 = quick_two_sum(a3, b6, &b7);
	c7 = c6 + d5;
	b8 = quick_two_sum(b7, c7, &c8);
	c[0] = a7;
	c[1] = b8;
	c[2] = c8;
}

// c[3] := a[3] + b
static inline void c_td_add_td_d(double *a, double b, double *c)
{
// 2024-02-21(Wed) T.Kouya
#if 0
	double z[6], e[6]; //, in_b[] = {b};

//	printf("c_td_add "); fflush(stdout);
	merge(z, a, 3, &b, 1);
//	printf("merge"); fflush(stdout);
	vec_sum(e, z, 4);
//	printf(" vec_sum"); fflush(stdout);
	vseb(c, 3, e, 4);
//	printf(" vseb\n"); fflush(stdout);
#endif // 0
//	double c0, c1, c2, c3;
	double e;

	c[0] = two_sum(a[0], b, &e);
	c[1] = two_sum(a[1], e, &e);
	c[2] = two_sum(a[2], e, &e);
	//c[3] = two_sum(a[3], e, &e);

	//qd::renorm(c0, c1, c2, c3, e);
	//renorm4(&(c[0]), &(c[1]), &(c[2]), &(c[3]), &e);
	renorm(&(c[0]), &(c[1]), &(c[2]),  &e);

//	return qd_real(c0, c1, c2, c3);
	return;
}


// c[3] := -a[3]
//static inline void c_td_neg(double *c, double *a)
static inline void c_td_neg(const double *a, double *c)
{
	c[0] = -a[0];
	c[1] = -a[1];
	c[2] = -a[2];
}

// c[3] := a[3] - b[3]
//static inline void c_td_sub(double *c, double *a, double *b)
static inline void c_td_sub(double *a, double *b, double *c)
{
	double mb[3];

	c_td_neg(b, mb);
	c_td_add(a, mb, c);
}

static inline void c_td_subq(const double *a, const double *b, double *c)
{
	double mb[3];

	c_td_neg(b, mb);
	c_td_addq(a, mb, c);
}

// c[3] := a - b[3]
//static inline void c_td_sub_d_td(double *c, double a, double *b)
static inline void c_td_sub_d_td(double a, double *b, double *c)
{
	double tmp_a[3] = {(double)0.0, (double)0.0, (double)0.0};
	double mb[3];

	tmp_a[0] = a;

	c_td_neg(b, mb);
	c_td_add(tmp_a, mb, c);
}

// c[3] := a[3] - b
//static inline void c_td_sub_td_d(double *c, double *a, double b)
static inline void c_td_sub_td_d(double *a, double b, double *c)
{
	double mb[3] = {(double)0.0, (double)0.0, (double)0.0};

	mb[0] = -b;

//	c_td_neg(b, mb);
	c_td_add(a, mb, c);
}

// Accurate c[3] := a[3] * b[3]
//static inline void c_td_mul_accurate(double *c, double *a, double *b)
static inline void c_td_mul_accurate(double *a, double *b, double *c)
{
	double z00[2], z01[2], z10[2];
	double in_b[3], in_c, z[3], e[5], temp[5];

	//printf("c_td_mul_accurate "); fflush(stdout);

	z00[0] = two_prod(a[0], b[0], &z00[1]);
	z01[0] = two_prod(a[0], b[1], &z01[1]);
	z10[0] = two_prod(a[1], b[0], &z10[1]);
	z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
	vec_sum(in_b, z, 3);
	in_c = fma(a[1], b[1], in_b[2]);

	z[0] = fma(a[0], b[2], z10[1]);
	z[1] = fma(a[2], b[0], z01[1]);
	z[2] = z[0] + z[1];
	temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; temp[3] = in_c; temp[4] = z[2];
	vec_sum(e, temp, 5);
	c[0] = e[0];
	vseb(&c[1], 2, &e[1], 4);
}

// Sloppy c[3] := a[3] * b[3]
//static inline void c_td_mul_sloppy(double *c, double *a, double *b)
static inline void c_td_mul_sloppy(double *a, double *b, double *c)
{
	double z00[2], z01[2], z10[2];
	double in_b[3], in_c, z[3], e[4], temp[4];

	//printf("c_td_mul_sloppy "); fflush(stdout);

	z00[0] = two_prod(a[0], b[0], &z00[1]);
	z01[0] = two_prod(a[0], b[1], &z01[1]);
	z10[0] = two_prod(a[1], b[0], &z10[1]);
	z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
	vec_sum(in_b, z, 3);
	in_c = fma(a[1], b[1], in_b[2]);

	z[0] = fma(a[0], b[2], z10[1]);
	z[1] = fma(a[2], b[0], z01[1]);
	z[2] = z[0] + z[1];
	temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; temp[3] = in_c + z[2];
	vec_sum(e, temp, 4);
	c[0] = e[0];
	vseb(&c[1], 2, &e[1], 3);
}

// 2025-12-24(Wed) T.Kouya
// Branch free algorithm
// void Mul3(const double x[3], const double y[3], double z[3]) {
static inline void c_td_mul_bf(double *a, double *b, double *c)
{
	double a0, b0, c0, d0, e0, f0, g0, h0, i0;
	double a1, b1, c1, d1, e1, f1, g1, h1, i1;
	double a2, b2, c2, d2, e2, f2, g2, g3;
	double a3, b3, c3, d3, e3, e4;
	double a4, b4, c4;
	double a5, b5, c5;
	double a6, b6, c6;
	double a7, b7, c7;
	
    a0 = two_prod(a[0], b[0], &b0);
    c0 = two_prod(a[0], b[1], &e0);
    d0 = two_prod(a[1], b[0], &f0);
    g0 = a[0] * b[2];
    h0 = a[1] * b[1];
    i0 = a[2] * b[0];
    c1 = two_sum(c0, d0, &d1);
    e1 = two_sum(e0, f0, &f1);
    g1 = two_sum(g0, i0, &i1);
    b2 = two_sum(b0, c1, &c2);
    g2 = two_sum(g1, h0, &h1);
    a3 = quick_two_sum(a0, b2, &b3);
    c3 = two_sum(c2, d1, &d3);
    e3 = two_sum(e1, g2, &g3);
    c4 = two_sum(c3, e3, &e4);
    b5 = quick_two_sum(b3, c4, &c5);
    a6 = quick_two_sum(a3, b5, &b6);
    b7 = quick_two_sum(b6, c5, &c7);

	c[0] = a6;
	c[1] = b7;
	c[2] = c7;
}

// 2026-05-02(Sat) T.Kouya
// Branch free algorithm: c[3] := a[3] * b  (b is a double)
// Derived from c_td_mul_bf by setting b[1]=0, b[2]=0.
// Vanishing partial products: c0, e0, g0, h0 all become 0.
// Surviving: a0 = two_prod(a[0],b)+b0, d0 = two_prod(a[1],b)+f0, i0 = a[2]*b.
static inline void c_td_mul_d_bf(double *a, double b, double *c)
{
	double a0, b0, d0, f0, i0;
	double c2;
	double a3, b2, b3;
	double e3, g3;
	double c3, c4, e4;
	double b5, c5;
	double a6, b6;
	double b7, c7;

	a0 = two_prod(a[0], b, &b0);
	d0 = two_prod(a[1], b, &f0);
	i0 = a[2] * b;
	// c1 = d0,    d1 = 0   (from two_sum(0, d0))
	// e1 = f0,    f1 = 0   (from two_sum(0, f0))
	// g1 = i0,    i1 = 0   (from two_sum(0, i0))
	b2 = two_sum(b0, d0, &c2);
	// g2 = i0,    h1 = 0   (from two_sum(i0, 0))
	a3 = quick_two_sum(a0, b2, &b3);
	c3 = c2;                             // two_sum(c2, 0) = (c2, 0)
	e3 = two_sum(f0, i0, &g3);
	c4 = two_sum(c3, e3, &e4);
	b5 = quick_two_sum(b3, c4, &c5);
	a6 = quick_two_sum(a3, b5, &b6);
	b7 = quick_two_sum(b6, c5, &c7);

	c[0] = a6;
	c[1] = b7;
	c[2] = c7;
}

// 2026-05-02(Sat) T.Kouya
// Branch free algorithm: c[3] := a[3] * b[2]  (b is a double-double)
// Derived from c_td_mul_bf by setting b[2]=0.
// Vanishing: g0 = a[0]*b[2] = 0 only. All other partial products survive.
static inline void c_td_mul_dd_bf(double *a, double *b, double *c)
{
	double a0, b0, c0, d0, e0, f0, h0, i0;
	double c1, d1, e1, f1, i1;
	double b2, c2, g2, h1;
	double a3, b3, c3, d3, e3, g3;
	double c4, e4;
	double b5, c5;
	double a6, b6;
	double b7, c7;

	a0 = two_prod(a[0], b[0], &b0);
	c0 = two_prod(a[0], b[1], &e0);
	d0 = two_prod(a[1], b[0], &f0);
	// g0 = a[0] * b[2] = 0
	h0 = a[1] * b[1];
	i0 = a[2] * b[0];
	c1 = two_sum(c0, d0, &d1);
	e1 = two_sum(e0, f0, &f1);
	// g1 = i0, i1 = 0   (from two_sum(0, i0))
	b2 = two_sum(b0, c1, &c2);
	g2 = two_sum(i0, h0, &h1);           // g1 = i0, so g2 = two_sum(i0, h0)
	a3 = quick_two_sum(a0, b2, &b3);
	c3 = two_sum(c2, d1, &d3);
	e3 = two_sum(e1, g2, &g3);
	c4 = two_sum(c3, e3, &e4);
	b5 = quick_two_sum(b3, c4, &c5);
	a6 = quick_two_sum(a3, b5, &b6);
	b7 = quick_two_sum(b6, c5, &c7);

	c[0] = a6;
	c[1] = b7;
	c[2] = c7;
}

// define c_td_mul
#ifndef USE_TD_MUL_ACCURATE
	//#define c_td_mul c_td_mul_sloppy
	#define c_td_mul c_td_mulq_sloppy
#else // USE_TD_MUL_ACCURATE
	//#define c_td_mul c_td_mul_accurate
	#define c_td_mul c_td_mulq_sloppy
#endif // USE_TD_MUL_ACCURATE


// c[3] := a[2] * b[3]
static inline void c_td_mul_dd_td_sloppy(double *a, double *b, double *c)
{
	double z00[2], z01[2], z10[2];
	double in_b[3], in_c, z[3], e[4], temp[4];

	z00[0] = two_prod(a[0], b[0], &z00[1]);
	z01[0] = two_prod(a[0], b[1], &z01[1]);
	z10[0] = two_prod(a[1], b[0], &z10[1]);
	z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
	vec_sum(in_b, z, 3);
	in_c = fma(a[1], b[1], in_b[2]);

	z[0] = fma(a[0], b[2], z10[1]);
	z[1] = z[0] + z01[1];
	temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; temp[3] = in_c + z[1];
	vec_sum(e, temp, 4);
	c[0] = e[0];
	vseb(&c[1], 2, &e[1], 3);
}

// c[3] := a[2] * b[3]
static inline void c_td_mul_dd_td_accurate(double *a, double *b, double *c)
{
	double z00[2], z01[2], z10[2];
	double in_b[3], in_c, z[3], e[5], temp[5];

	z00[0] = two_prod(a[0], b[0], &z00[1]);
	z01[0] = two_prod(a[0], b[1], &z01[1]);
	z10[0] = two_prod(a[1], b[0], &z10[1]);
	z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
	vec_sum(in_b, z, 3);
	in_c = fma(a[1], b[1], in_b[2]);

	z[0] = fma(a[0], b[2], z10[1]);
	z[1] = z[0] + z01[1];
	temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; temp[3] = in_c; temp[4] = z[1];
	vec_sum(e, temp, 5);
	c[0] = e[0];
	vseb(&c[1], 2, &e[1], 4);
}

// define c_td_mul_dd_td
#ifndef USE_TD_MUL_DD_TD_ACCURATE
	#define c_td_mul_dd_td c_td_mul_dd_td_sloppy
//	#define c_td_mul_dd_td c_td_mul_dd_td_accurate
#else // USE_TD_MUL_DD_TD_ACCURATE
	#define c_td_mul_dd_td c_td_mul_dd_td_accurate
#endif // USE_TD_MUL_ACCURATE


// c[3] := a * b[3]
static inline void c_td_mul_d_td(const double a, const double *b, double *c)
{
#if 0 // 2024-04-23 T.Kouya
	double z00[2], z01[2], z10[2];
	double in_b[3], in_c, z[3], e[4], temp[4];

	z00[0] = two_prod(a, b[0], &z00[1]);
	z01[0] = two_prod(a, b[1], &z01[1]);
	//z10[0] = two_prod(a[1], b[0], &z10[1]);
	z[0] = z00[1]; z[1] = z01[0]; //z[2] = z10[0];
	vec_sum(in_b, z, 2);
	//in_c = fma(a[1], b[1], in_b[2]);

	z[0] = fma(a, b[2], z10[1]);
	z[1] = z[0] + z01[1];
	temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; temp[3] = z[1];
	vec_sum(e, temp, 4);
	c[0] = e[0];
	vseb(&c[1], 2, &e[1], 3);
#endif // 0
	double in_a[3] = {a, 0.0, 0.0};
	c_td_mulq_sloppy(in_a, b, c);
}

// c[3] := a[3] * b
static inline void c_td_mul_td_d(const double *a, const double b, double *c)
{
	c_td_mul_d_td(b, a, c);
}

// b := |a| 
static inline void c_td_abs(const double *a, double *b)
{
	// return (a[0] < 0.0) ? -a : a;
	if(a[0] < 0.0)
		c_td_neg(a, b);
	else
		c_td_copy(a, b);
}

static inline void c_td_comp(const double *a, const double *b, int *result) 
{
	//  qd_real aa(a), bb(b);
	//if (aa < bb)
	if (a[0] < b[0] || (a[0] == b[0] && (a[1] < b[1] || (a[1] == b[1] && (a[2] < b[2])))))
	    *result = -1;
	//else if (aa > bb)
	else if (a[0] > b[0] || (a[0] == b[0] && (a[1] > b[1] || (a[1] == b[1] && (a[2] > b[2] || (a[2] == b[2]))))))
		*result = 1;
	else 
		*result = 0;
}

static inline void c_td_comp_td_d(const double *a, double b, int *result)
{
	// qd_real aa(a);
	//if (aa < b)
	if(a[0] < b || (a[0] == b && a[1] < 0.0))
		*result = -1;
	//else if (aa > b)
	else if (a[0] > b || (a[0] == b && a[1] > 0.0))
		*result = 1;
	else 
		*result = 0;
}

static inline void c_td_comp_d_td(double a, const double *b, int *result)
{
/* qd_real bb(b);
  if (a < bb)
    *result = -1;
  else if (a > bb)
    *result = 1;
  else 
    *result = 0;
*/
	c_td_comp_td_d(b, a, result);
	*result = -(*result);
}

// c:= 2 - c_td_mul_dd_td(a[2], b[3])
static inline void c_td_2mtw_dd_td(double a[DDSIZE], double b[TDSIZE], double c[TDSIZE])
{
	double z00[2], z01[2], z10[2];
	double in_b[3], in_c, z[3], e[4], temp[4];

	z00[0] = two_prod(a[0], b[0], &z00[1]);
	z01[0] = two_prod(a[0], b[1], &z01[1]);
	z10[0] = two_prod(a[1], b[0], &z10[1]);
	z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
	vec_sum(in_b, z, 3);
	in_c = fma(a[1], b[1], in_b[2]);

	z[0] = fma(a[0], b[2], z10[1]);
	z[1] = z[0] + z01[1];
	temp[0] = -in_b[0]; temp[1] = -in_b[1]; temp[2] = -in_c; temp[3] = -z[1];
	vec_sum(e, temp, 4);
	c[0] = 1;
	e[0] = quick_two_sum(-z00[0], e[0], &temp[0]);
	vseb(&c[1], 2, e, 4);
 }

#define ONE_P_2DBL_EPS (1.00000000000000044e+00) // 1 + 2 * DBL_EPSILON
#define ONE_M_2DBL_EPS (9.99999999999999556e-01) // 1 - 2 * DBL_EPSILON

// c[3] := 1 / a[3]
//static inline void c_td_reci(double *c, double *a)
static inline void c_td_reci(double *a, double *c)
{
	double alpha, h1, in_b[2], in_b12, temp[3], d2[3];// = {2.0, 0.0, 0.0}; // d2 = 2

	c_to_td(d2, 2.0, 0.0, 0.0);

	alpha = ONE_P_2DBL_EPS / a[0];
	h1 =  fma(alpha, a[0], -ONE_P_2DBL_EPS); // alpha * a[0] - (1 + 2u)
	h1 = -fma(alpha, a[1], h1);
	in_b[0] = two_prod(alpha, ONE_M_2DBL_EPS, &in_b[1]);
	in_b12 = fma(alpha, h1, in_b[1]);
	in_b[0] = quick_two_sum(in_b[0], in_b12, &in_b[1]);
//	c_td_2mtw_dd_td(in_b, a, temp); // temp := 2 - c_td_mul_dd_td(b, a, temp)
	c_td_mul_dd_td(in_b, a, temp); c_td_sub(d2, temp, temp);
	c_td_mul_dd_td(in_b, temp, c);
}

// c[3] := a[3] / b[3]
//static inline void c_td_div(double *a, double *b, double *c)
static inline void c_td_divt(double *a, double *b, double *c)
{
	double alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];// = {(double)2.0, (double)0.0, (double)0.0}; // d2 = 2;

	c_to_td(d2, 2.0, 0.0, 0.0);

	alpha = ONE_P_2DBL_EPS / b[0];
	h1 =  fma(alpha, b[0], -ONE_P_2DBL_EPS);
	h1 = -fma(alpha, b[1], h1);
	in_b[0] = two_prod(alpha, ONE_M_2DBL_EPS, &in_b[1]);
	in_b12 = fma(alpha, h1, in_b[1]);
	in_b[0] = quick_two_sum(in_b[0], in_b12, &in_b[1]);
//	c_td_2mtw_dd_td(in_b, b, temp); // temp := 2 - c_td_mul_dd_td(in_b, b, temp)
	c_td_mul_dd_td(in_b, b, temp); c_td_sub(d2, temp, temp);
	c_td_mul_dd_td(in_b, a, in_c);
	c_td_mul(in_c, temp, c);
}

//#define c_td_div c_td_divtq
#ifdef BNC_USE_FMA_DIV
	#define c_td_div bnc_td_div_fma
#else
	#define c_td_div c_td_divq // 2024-01-31
#endif // BNC_USE_FMA_DIV

// c[3] := a[3] / b[3]
//static inline void c_td_div(double *a, double *b, double *c)
static inline void c_td_divtq(double *a, double *b, double *c)
{
	double alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];// = {(double)2.0, (double)0.0, (double)0.0}; // d2 = 2;

	c_to_td(d2, 2.0, 0.0, 0.0);

	alpha = ONE_P_2DBL_EPS / b[0];
	h1 =  fma(alpha, b[0], -ONE_P_2DBL_EPS);
	h1 = -fma(alpha, b[1], h1);
	in_b[0] = two_prod(alpha, ONE_M_2DBL_EPS, &in_b[1]);
	in_b12 = fma(alpha, h1, in_b[1]);
	in_b[0] = quick_two_sum(in_b[0], in_b12, &in_b[1]);
//	c_td_2mtw_dd_td(in_b, b, temp); // temp := 2 - c_td_mul_dd_td(in_b, b, temp)
	c_td_mul_dd_td(in_b, b, temp);
	//c_td_sub(d2, temp, temp);
	c_td_subq(d2, temp, temp);
	c_td_mul_dd_td(in_b, a, in_c);
	c_td_mul(in_c, temp, c);
}

/* triple-double / triple-double */
// c := a / b
static inline void c_td_divq(const double *a, const double *b, double *c)
{
	//qd_real qd_real::sloppy_div(const qd_real &a, const qd_real &b) {
	double q0, q1, q2, q3;
	double r[TDSIZE], tmp[TDSIZE];
	//qd_real r;

	q0 = a[0] / b[0];
	//r = a - (b * q0);
	c_td_mul_td_d(b, q0, tmp);
	c_td_subq(a, tmp, r);

	q1 = r[0] / b[0];
	//r -= (b * q1);
	c_td_mul_td_d(b, q1, tmp);
	c_td_subq(r, tmp, r);

	q2 = r[0] / b[0];
	//r -= (b * q2);
	c_td_mul_td_d(b, q2, tmp);
	c_td_subq(r, tmp, r);

	q3 = r[0] / b[0];

	renorm(&q0, &q1, &q2, &q3);

	//return qd_real(q0, q1, q2, q3);
	c[0] = q0;
	c[1] = q1;
	c[2] = q2;
	c[3] = q3;
}

/* double / triple-double */
// c := (double)a / b
static inline void c_td_div_d_td(double a, double *b, double *c)
{
	double aa[TDSIZE];

	c_td_copy_d(a, aa);
	c_td_div(aa, b, c);
}
/* triple-double / double */
// c := a / (double)b
static inline void c_td_div_td_d(double *a, double b, double *c)
{
	double bb[TDSIZE];

	c_td_copy_d(b, bb);
	c_td_div(a, bb, c);
}

// c[3] := sqrt(a[3])
static inline void c_td_sqrt(double *a, double *c)
{
	double tmp_a[4], tmp_c[4];

	c_qd_copy_td(a, tmp_a);

	c_qd_sqrt(tmp_a, tmp_c);
	c_td_copy_qd(tmp_c, c);	
}

// c[3] := sqrt(a[3])
static inline void c_td_sqrt_d(double a, double *c)
{
	double tmp_a[4], tmp_c[4];

	c_qd_copy_d(a, tmp_a);

	c_qd_sqrt(tmp_a, tmp_c);
	c_td_copy_qd(tmp_c, c);	
}



// c[3] := a[3]^2
static inline void c_td_sqr(double *a, double *c)
{
#ifdef USE_ACCURATE_TD_MUL
	c_td_mul_acculate(a, a, c);
#else // USE_ACCURATE_TD_MUL
	c_td_mul_sloppy(a, a, c);
#endif // USE_ACCURATE_TD_MUL
}

static inline void c_td_read(const char *s, double *a) {
#if 0
  qd_real aa(s);
  TO_DOUBLE_PTR(aa, a);
#endif // 0
}

static inline void c_td_swrite(const double *a, int precision, char *s, int len) {
#if 0
  qd_real(a).write(s, len, precision);
#endif // 0
}

static inline void c_td_write(const double *a) {
#if 0
  std::cout << qd_real(a).to_string(qd_real::_ndigits) << std::endl;
#endif // 0
}

// PI = 3.1415...
static inline void c_td_pi(double *a)
{
	c_td_copy(const_td_pi, a);
}


#ifdef __cplusplus
}
#endif

/* Clean up the local `nint` macro (defined around line 3495 for the
 * c_qd_nint helper).  Without this, the macro leaks to anyone who
 * transitively includes c_dd_qd.h and then qd/td_inline.h, where
 * `qd::nint(...)` would be expanded to the broken `qd::(round(...))`. */
#ifdef nint
#  undef nint
#endif

/* ------------------------------------------------------------------ */
/* dtq-0.0.3 FMA-based elementary functions (exp/expm1/log/log10/     */
/* sin/cos/sincos for DD, TD and QD).  This also provides the         */
/* definitions for the bnc_dd_* / bnc_qd_* forward declarations used  */
/* by c_dd_exp, c_dd_sin, c_qd_exp, ... above.                        */
/* ------------------------------------------------------------------ */
#include "bncelem.h"

/* The forward declarations cover the include order in which this file is
   processed inside bncelem_dd.h, before bncelem_td.h is seen.  They must
   stay OUTSIDE the extern "C" block below: when the include above did
   reach bncelem_td.h, these names are already declared there with C++
   linkage, and repeating them with "C" linkage is an error in C++.     */
static inline void bnc_td_exp(const double *a, double *ret);
static inline void bnc_td_log(const double *a, double *ret);
static inline void bnc_td_log10(const double *a, double *ret);
static inline void bnc_td_sin(const double *a, double *ret);
static inline void bnc_td_cos(const double *a, double *ret);
static inline void bnc_td_sincos(const double *a, double *sin_a, double *cos_a);

#ifdef __cplusplus
extern "C" {
#endif

/* TD elementary functions in the c_td_* namespace.  These had no C
   implementation before (rdd.h references them, so any C-only build
   using RTD_EXP etc. failed to link).                                 */

static inline void c_td_exp(const double *a, double *ret)
{
	bnc_td_exp(a, ret);
}

static inline void c_td_log(const double *a, double *ret)
{
	bnc_td_log(a, ret);
}

static inline void c_td_log10(const double *a, double *ret)
{
	bnc_td_log10(a, ret);
}

static inline void c_td_sin(const double *a, double *ret)
{
	bnc_td_sin(a, ret);
}

static inline void c_td_cos(const double *a, double *ret)
{
	bnc_td_cos(a, ret);
}

static inline void c_td_sincos(const double *a, double *s, double *c)
{
	bnc_td_sincos(a, s, c);
}

// tan(a) = sin(a) / cos(a)
static inline void c_td_tan(const double *a, double *ret)
{
	double s[3], c[3];

	bnc_td_sincos(a, s, c);
	c_td_div(s, c, ret);
}

#ifdef __cplusplus
}
#endif

#endif // __C_DD_QD_H

/* ==== end: former c_dd_qd.h ==== */
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
#define RTD_OUT_STR(a) rtd_out_str_base(stdout, 10, 33, a)

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
#define RQD_OUT_STR(a) rqd_out_str_base(stdout, 10, 64, a)

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

