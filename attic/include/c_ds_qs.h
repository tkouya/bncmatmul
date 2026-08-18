/* pure C code by T.Kouya */
/* 2019-06-14 appended static inline to main functions */

#ifndef __C_DS_QS_H
#define __C_DS_QS_H

//#define _QD_SPLITTER 134217729.0               // = 2^27(=ceil(53/2)) + 1
//#define _QD_SPLIT_THRESH 6.69692879491417e+299 // = 2^996(=1024-(27+1))
#define _QS_SPLITTER 4097.0              // = 2^12(=ceil(24/2)) + 1
#define _QS_SPLIT_THRESH 1.01412048018258E+31 // = 2^103(=128-(24+1))

#define _QS_D2F24 16777217.0  // 2^24 + 1
#define _QS_D2F29 536870913.0 // 2^29 + 1, 53 - 24 = 29!

#ifdef QS_VACPP_BUILTINS_H
/* For VisualAge C++ __fmadds */
#include <builtins.h>
#endif // QS_VACPP_BUILDINS_H

#include <math.h>

// return a * b + c
#define DFMA(a, b, c) fma((a), (b), (c)) // float precision
#define SFMA(a, b, c) fmaf((a), (b), (c)) // sigle precision

// return a * b + c
#define QS_FMA(a, b, c) SFMA((a), (b), (c))

// return a * b - c
#define QS_FMS(a, b, c) SFMA((a), (b), (-(c)))

#include <limits.h>

#include <float.h>
#ifndef DBL_EPSILON
	#define DBL_EPSILON (1.1102230246251565404236316680908e-16) // 1/2 * 2^(-52) = 2^(-53)
#endif // DBL_EPSILON
#ifndef FLT_EPSILON
	#define FLT_EPSILON (5.9604644775390625e-8) // 1/2 * 2^(-23) = 2^(-24)
#endif // FLT_EPSILON

// DSSIZE, TSSIZE, QSSIZE
#ifndef DSSIZE
	#define DSSIZE 2
#endif // DSSIZE

#ifndef TSSIZE
	#define TSSIZE 3
#endif // TSSIZE

#ifndef QSSIZE
	#define QSSIZE 4
#endif // QSSIZE

#define TO_FLOAT_PTR(a, ptr) ptr[0] = a.x[0]; ptr[1] = a.x[1];

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
/* DS : Double-float precision       */
/**************************************/
//  float _hi() const { return x[0]; }
//  float _lo() const { return x[1]; }
#define DS_HI(a)	((a)[0])
#define DS_LOW(a)	((a)[1])

// Definitions for Comparing Functions
typedef unsigned int	dd_bool;
#define DS_TRUE		(1UL)
#define DS_FALSE	(0UL)

#define DS_ISNAN(a) (isnanf((a)[0]))
#define DS_ISINF(a) (isinff((a)[0]))
#define DS_ISZERO(a) ((a)[0] == 0.0f)
#define DS_ISNEGATIVE(a) ((a)[0] < 0.0f)
#define DS_NAN (FP_NAN)

// PI = 3.1415...
// (fixed 2026-08-14: the previous value here was {0, 0})
static const float const_ds_pi[2] = {3.14159274e+00f, -8.74227766e-08f};

static inline float to_float(float a) { return a; }
static inline int to_intf(float a) { return (int)(a); }

/**************************************/
/* TS : Triple-float precision       */
/**************************************/
#define TS_HI(a)	((a)[0])
#define TS_LOW(a)	((a)[2])

// Definitions for Comparing Functions
typedef unsigned int	dd_bool;
#define TS_TRUE		(1UL)
#define TS_FALSE	(0UL)

#define TS_ISNAN(a) (isnanf((a)[0]))
#define TS_ISINF(a) (isinff((a)[0]))
#define TS_ISZERO(a) ((a)[0] == 0.0f)
#define TS_ISNEGATIVE(a) ((a)[0] < 0.0f)
#define TS_NAN (FP_NAN)

// PI = 3.1415...
// (fixed 2026-08-14: the previous value here was {0, 0, 0})
static const float const_ts_pi[3] = {3.14159274e+00f, -8.74227766e-08f, -3.43024902e-15f};

/**************************************/
/* QS : Quadruple-float precision    */
/**************************************/
#define QS_HI(a)	((a)[0])
#define QS_LOW(a)	((a)[3])

// Definitions for Comparing Functions
typedef unsigned int	ds_bool;
#define QS_TRUE		(1UL)
#define QS_FALSE	(0UL)

#ifndef __cplusplus
#define QS_ISINF(a) (isinff((a)[0]))
#define QS_ISNAN(a) (isnanf((a)[0]))
#endif // ifndef __cplusplus
#define QS_ISZERO(a) ((a)[0] == 0.0f)
#define QS_ISNEGATIVE(a) ((a)[0] < 0.0f)
#define QS_NAN (FP_NAN)

// PI = 3.1415...
// (fixed 2026-08-14: the previous value here was {0, 0, 0, 0})
static const float const_qs_pi[4] = {3.14159274e+00f, -8.74227766e-08f, -3.43024902e-15f, 2.1125998e-23f};

/*********** Basic Functions ************ start **/
/* Computes fl(a+b) and err(a+b).  Assumes |a| >= |b|. */
static inline float fquick_two_sum(float a, float b, float *err)
{
	float s;
	s = a + b;
	*err = b - (s - a);
	return s;
}

/* Computes fl(a-b) and err(a-b).  Assumes |a| >= |b| */
static inline float fquick_two_diff(float a, float b, float *err)
{
	float s;
	s = a - b;
	*err = (a - s) - b;
	return s;
}

/* Computes fl(a+b) and err(a+b).  */
static inline float ftwo_sum(float a, float b, float *err)
{
	float s;
	s = a + b;
	float bb;
	bb = s - a;
	*err = (a - (s - bb)) + (b - bb);
	return s;
}

/* Computes fl(a-b) and err(a-b).  */
static inline float ftwo_diff(float a, float b, float *err)
{
	float s;
	s = a - b;
	float bb;
	bb = s - a;
	*err = (a - (s - bb)) - (b + bb);
	return s;
}

#ifndef QS_FMS
/* Computes high word and lo word of a */
static inline void fsplit(float a, float *hi, float *lo)
{
	float temp;

	if (a > _QS_SPLIT_THRESH || a < -_QS_SPLIT_THRESH)
	{
		//a *= 3.7252902984619140625e-09;  // 2^-28
		a *=  1.220703125e-04; // 2^(-12-1)
		temp = _QS_SPLITTER * a;
		*hi = temp - (temp - a);
		*lo = a - (*hi);
		//*hi *= 268435456.0;          // 2^28
		//*lo *= 268435456.0;          // 2^28
		*hi *= 8192.0;          // 2^13
		*lo *= 8192.0;          // 2^13
	}
	else
	{
		temp = _QS_SPLITTER * a;
		*hi = temp - (temp - a);
		*lo = a - (*hi);
	}
}
#endif // QS_FMS

/* Computes fl(a*b) and err(a*b). */
static inline float ftwo_prod(float a, float b, float *err)
{
#ifdef QS_FMS
	float p;
	p = a * b;

	*err = QS_FMS(a, b, p);

	return p;
#else  // QS_FMS
	float a_hi, a_lo, b_hi, b_lo;
	float p;
	p = a * b;

	fsplit(a, &a_hi, &a_lo);
	fsplit(b, &b_hi, &b_lo);
	*err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;

	return p;
#endif // QS_FMS
}

/* Computes fl(a*a) and err(a*a).  Faster than the above method. */
static inline float ftwo_sqr(float a, float *err)
{
#ifdef QS_FMS
	float p;
	p = a * a;

	*err = QS_FMS(a, a, p);

	return p;
#else // QS_FMS
	float hi, lo;
	float q;
	q = a * a;

	fsplit(a, &hi, &lo);
	*err = ((hi * hi - q) + 2.0 * hi * lo) + lo * lo;

	return q;
#endif // QS_FMS
}

/* Computes the nearest integer to d. */
static inline float fnint(float d)
{
	if (d == floorf(d))
		return d;
	return floorf(d + 0.5);
}

/* Computes the truncated integer. */
static inline float faint(float d)
{
	return (d >= 0.0) ? floorf(d) : ceilf(d);
}

/* These are provided to give consistent 
   interface for float with float-float and quad-float. */
static inline void fsincosh(float t, float *sinh_t, float *cosh_t)
{
	*sinh_t = sinhf(t);
	*cosh_t = coshf(t);
}

static inline float fsqr(float t)
{
	return t * t;
}

// 2026-02-24(Tue) (d, e) := FastTwoFMA(a, b, c)
// a * b + c = d + e
// |c| > 2|ab|
static inline float ffast_two_fma(const float a, const float b, const float c, float *e)
{
	float d, t;

	d = SFMA(a, b, c);
	t = c - d;
	*e = SFMA(a, b, t);

	return d;
}

// 2026-03-21(Mon) (z, e) := FMAerror(a, b, c)
// a * b + c = z + e
static inline float ffma_error(const float a, const float b, const float c, float *e)
{
	float z, t, p[DSSIZE], u[DSSIZE];

	z = SFMA(a, b, c);
	p[0] = ftwo_prod(a, b, &(p[1]));
	u[0] = ftwo_sum(c, p[0], &(u[1]));
	t = u[0] - z;
	*e = t + (p[1] + u[1]);

	return z;
}
/*********** Basic Functions ************ ended **/

// -------------------------------------------------
// -------------------- DS -------------------------
// -------------------------------------------------
/*********** Micellaneous ************/
/*  this == 0 */
static inline ds_bool ds_is_zero(const float *x)
{
//  return (x[0] == 0.0);
	if(x[0] == 0.0)
		return DS_TRUE;
	else
		return DS_FALSE;
}

/*  this == 1 */
static inline ds_bool ds_is_one(const float *x)
{
	//return (x[0] == 1.0 && x[1] == 0.0);
	if(x[0] == 1.0f && x[1] == 0.0f)
		return DS_TRUE;
	else
		return DS_FALSE;
}

/*  this > 0 */
static inline ds_bool ds_is_positive(const float *x)
{
//  return (x[0] > 0.0f);
	if(x[0] > 0.0f)
		return DS_TRUE;
	else
		return DS_FALSE;
}

/* this < 0 */
static inline ds_bool ds_is_negative(const float *x)
{
//  return (x[0] < 0.0);
	if(x[0] < 0.0)
		return DS_TRUE;
	else
		return DS_FALSE;
}

/* float-float = float + float */
static inline void c_s_add(float a, float b, float *c)
{
	float s, e;

	s = ftwo_sum(a, b, &e);
//	return dd_real(s, e);
	c[0] = s;
	c[1] = e;
}
//#endif // NATIVE_C

/* add */
static inline void c_ds_add(const float *a, const float *b, float *c)
{
	/* This one satisfies IEEE style error bound, 
		due to K. Briggs and W. Kahan.                   */
	float s1, s2, t1, t2;

	s1 = ftwo_sum(a[0], b[0], &s2);
	t1 = ftwo_sum(a[1], b[1], &t2);
	s2 += t1;
	s1 = fquick_two_sum(s1, s2, &s2);
	s2 += t2;
	s1 = fquick_two_sum(s1, s2, &s2);
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
//static inline void c_ds_add_sloppy(const float *a, const float *b, float *c, float *err)
static inline void c_ds_add_sloppy(const float *a, const float *b, float *c)
{
//#ifdef NATIVE_C
  /* This is the less accurate version ... obeys Cray-style
     error bound. */
	float s, e;

	s = ftwo_sum(a[0], b[0], &e);
	e += (a[1] + b[1]);
//	s = fquick_two_sum(s, e, &e);
	c[0] = fquick_two_sum(s, e, &e);
	c[1] = e;
//  return dd_real(s, e);
//	*err = e;
}

// bug fix: 2025-... by T.Kouya — mirror of c_dd_add_dd_d (proper two_sum)
// The earlier sloppy form `c[0]=a[0]+b; c[1]=a[1]` dropped the rounding
// error of (a[0] + b) and reduced c_ds_sqrt to single-precision accuracy.
// (ds_real)c := (ds_real)a + b
static inline void c_ds_add_ds_f(const float *a, float b, float *c)
{
	float s1, s2;
	s1 = ftwo_sum(a[0], b, &s2);
	s2 += a[1];
	s1 = fquick_two_sum(s1, s2, &s2);
	c[0] = s1;
	c[1] = s2;
}


// (ds_real)c := a + (ds_real)b — mirror of DD version
static inline void c_ds_add_d_ds(float a, const float *b, float *c)
{
	float s1, s2;
	s1 = ftwo_sum(a, b[0], &s2);
	s2 += b[1];
	s1 = fquick_two_sum(s1, s2, &s2);
	c[0] = s1;
	c[1] = s2;
}

/*********** Subtractions ************/
/* float-float = float - float */
static inline void c_f_sub(float a, float b, float *c)
{
	float s, e;
	s = ftwo_diff(a, b, &e);
	c[0] = s;
	c[1] = e;
}

/* sub */
static inline void c_ds_sub(const float *a, const float *b, float *c)
{
	float s, e;
	s = ftwo_diff(a[0], b[0], &e);
	e += a[1];
	e -= b[1];
	s = fquick_two_sum(s, e, &e);
//	return dd_real(s, e);
	c[0] = s;
	c[1] = e;
}

static inline void c_ds_sub_sloppy(const float *a, const float *b, float *c)
{
	float s1, s2, t1, t2;
	s1 = ftwo_diff(a[0], b[0], &s2);
	t1 = ftwo_diff(a[1], b[1], &t2);
	s2 += t1;
	s1 = fquick_two_sum(s1, s2, &s2);
	s2 += t2;
	s1 = fquick_two_sum(s1, s2, &s2);
	//return dd_real(s1, s2);
	c[0] = s1;
	c[1] = s2;
}

/* float-float - float */
static inline void c_ds_sub_ds_f(const float *a, float b, float *c)
{
	float s1, s2;
	s1 = ftwo_diff(a[0], b, &s2);
	s2 += a[1];
	s1 = fquick_two_sum(s1, s2, &s2);
	//return dd_real(s1, s2);
	c[0] = s1;
	c[1] = s2;
}

/* float - float-float */
static inline void c_ds_sub_f_ds(float a, const float *b, float *c)
{
	float s1, s2;
	s1 = ftwo_diff(a, b[0], &s2);
	s2 -= b[1];
	s1 = fquick_two_sum(s1, s2, &s2);

	c[0] = s1;
	c[1] = s2;
}


/*********** Multiplications ************/
/* float-float = float * float */
static inline void c_f_mul(float a, float b, float *c)
{
	float p, e;
	p = ftwo_prod(a, b, &e);
//	return dd_real(p, e);
	c[0] = p;
	c[1] = e;
}

/* mul */
static inline void c_ds_mul(const float *a, const float *b, float *c)
{
	float p1, p2;

	p1 = ftwo_prod(a[0], b[0], &p2);
	p2 += (a[0] * b[1] + a[1] * b[0]);
	p1 = fquick_two_sum(p1, p2, &p2);
//	return dd_real(p1, p2);
	c[0] = p1;
	c[1] = p2;
}

static inline void c_ds_mul_ds_f(const float *a, float b, float *c)
{
	float p1, p2;

	p1 = ftwo_prod(a[0], b, &p2);
	p2 += (a[1] * b);
	p1 = fquick_two_sum(p1, p2, &p2);
//  return dd_real(p1, p2);
	c[0] = p1;
	c[1] = p2;
}
static inline void c_ds_mul_f_ds(float a, const float *b, float *c)
{
	c_ds_mul_ds_f(b, a, c);

//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real cc;
  cc = a * dd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

/*********** Divisions ************/
static inline void c_f_div(float a, float b, float *c)
{
	float q1, q2;
	float p1, p2;
	float s, e;

	q1 = a / b;

	/* Compute  a - q1 * b */
	p1 = ftwo_prod(q1, b, &p2);
	s = ftwo_diff(a, p1, &e);
	e -= p2;

	/* get next approximation */
	q2 = (s + e) / b;

	s = fquick_two_sum(q1, q2, &e);

	//return dd_real(s, e);
	c[0] = s;
	c[1] = e;
}

/* FMA-driven divisions (dtq-0.0.3 fma_div ports; defined in bncelem_f.h,
   included at the end of this header).  Selected by BNC_USE_FMA_DIV.     */
static inline void bnc_ds_div_fma(const float *a, const float *b, float *ret);
static inline void bnc_ts_div_fma(const float *a, const float *b, float *ret);
static inline void bnc_qs_div_fma(const float *a, const float *b, float *ret);

/* div */ // fix! 2021-01-25 by T.Kouya
static inline void c_ds_div(const float *a, const float *b, float *c)
{
#ifdef BNC_USE_FMA_DIV
	bnc_ds_div_fma(a, b, c);
}
static inline void c_ds_div_orig(const float *a, const float *b, float *c)
{
#endif /* BNC_USE_FMA_DIV */
	float q1, q2, q3;
	float r[DSSIZE], tmp[DSSIZE];

	q1 = a[0] / b[0];  /* approximate quotient */

//  r = a - q1 * b;
	c_ds_mul_ds_f(b, q1, r);
	c_ds_sub(a, r, r);

	q2 = r[0] / b[0];

//  r -= (q2 * b);
	c_ds_mul_ds_f(b, q2, tmp);
	c_ds_sub(r, tmp, r); // fix! 2021-01-25 by T.Kouya

//  q3 = r[0] / b[0];
	q3 = r[0] / b[0]; // fix! 2021-01-25 by T.Kouya

	q1 = fquick_two_sum(q1, q2, &q2);
//  r = dd_real(q1, q2) + q3;
	r[0] = q1;
	r[1] = q2;
	c_ds_add_ds_f(r, q3, r);

	c[0] = r[0];
	c[1] = r[1];
}

/* float-float / float-float */
static inline void c_ds_sloppy_div(float *a, float *b, float *c)
{
	float s1, s2;
	float q1, q2;
	float r[DSSIZE];

	q1 = a[0] / b[0];  /* approximate quotient */

	/* compute  this - q1 * dd */
	//r = b * q1;
	c_ds_mul_ds_f(b, q1, r);

	s1 = ftwo_diff(a[0], r[0], &s2);
	s2 -= r[1];
	s2 += a[1];

	/* get next approximation */
	q2 = (s1 + s2) / b[0];

	/* renormalize */
	r[0] = fquick_two_sum(q1, q2, &r[1]);
//	return r;

	c[0] = r[0];
	c[1] = r[1];
}

/* float-float / float */
static inline void c_ds_div_ds_f(const float *a, float b, float *c)
{
	static 	float q1, q2;
	float p1, p2;
	float s, e;
//	dd_real r;
	float r[DSSIZE];

	q1 = a[0] / b;   /* approximate quotient. */

	/* Compute  this - q1 * d */
	p1 = ftwo_prod(q1, b, &p2);
	s = ftwo_diff(a[0], p1, &e);
	e += a[1];
	e -= p2;

	/* get next approximation. */
	q2 = (s + e) / b;

	/* renormalize */
	r[0] = fquick_two_sum(q1, q2, &r[1]);

	// return r;
	c[0] = r[0];
	c[1] = r[1];
}

/* float / float-float */
static inline void c_ds_div_f_ds(float a, const float *b, float *c)
{
	float tmp[DSSIZE];
	//return dd_real(a) / b;
	tmp[0] = a;
	tmp[1] = 0.0;

	c_ds_div(tmp, b, c);

//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real cc;
  cc = a / dd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

/* copy */
static inline void c_ds_copy(const float *a, float *b)
{
	b[0] = a[0];
	b[1] = a[1];
}

static inline void c_ds_copy_f(const float a, float *b)
{
	b[0] = a;
	b[1] = 0.0f;
}

// [x0, x1] := split(da, 24)
static inline void c_ds_copy_d(const double da, float *b)
{
	double x0, x1, gamma, delta;

	gamma = _QS_D2F29 * da; // (2^29 + 1) * da
	delta = da - gamma;

	x0 = gamma + delta;
	b[0] = (float)x0;

	//x1 = da - (double)b[0];
	x1 = da - x0;
	b[1] = (float)x1;
}


/* b := a^2 */
static inline void c_f_sqr(float a, float *b)
{
	float p1, p2;
	p1 = ftwo_sqr(a, &p2);
	// return dd_real(p1, p2);
	b[0] = p1;
	b[1] = p2;
}

// b := a^2
static inline void c_ds_sqr_f(float a, float *ret)
{
	float p1, p2;
	p1 = ftwo_sqr(a, &p2);
//	return dd_real(p1, p2);
	ret[0] = p1;
	ret[1] = p2;
}

// b := a^2
static inline void c_ds_sqr(const float *a, float *b)
{
	float p1, p2;
	float s1, s2;

	p1 = ftwo_sqr(a[0], &p2);
	p2 += 2.0 * a[0] * a[1];
	p2 += a[1] * a[1];
	s1 = fquick_two_sum(p1, p2, &s2);
//	return dd_real(s1, s2);
	b[0] = s1;
	b[1] = s2;
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
static inline void c_ds_sqrt(const float *a, float *b)
{
  /* Strategy:  Use Karp's trick:  if x is an approximation
     to sqrt(a), then

        sqrt(a) = a*x + [a - (a*x)^2] * x / 2   (approx)

     The approximation is accurate to twice the accuracy of x.
     Also, the multiplication (a*x) and [-]*x can be done with
     only half the precision.
  */

 // if (a.is_zero())
	if(ds_is_zero(a) == DS_TRUE)
	{
		//dd_set0(b);
		b[0] = 0.0f; b[1] = 0.0f;
		return;
	}

//  if (a.is_negative()) {
	if(ds_is_negative(a) == DS_TRUE)
	{
		//dd_real::error("(dd_real::sqrt): Negative argument.");
		fprintf(stderr, "(c_ds_sqrt): Negative argument.");
		//return dd_real::_nan;
		//dd_set_f(NAN, b);
		b[0] = FP_NAN; b[1] = FP_NAN;
		return;
	}

	//  float x = 1.0 / std::sqrt(a.x[0]);
	float x;
	x = 1.0 / sqrtf(a[0]);

  	//float ax = a.x[0] * x;
	float ax, x2;

	ax = a[0] * x;

//  return dd_real::add(ax, (a - dd_real::sqr(ax)).x[0] * (x * 0.5));
	float tmp[DSSIZE], ax2[DSSIZE];

// 1) DS: tmp := a - ax^2
	c_ds_sqr_f(ax, ax2);
	c_ds_sub(a, ax2, tmp);
	//printf("1) tmp[0] = %25.17e\n", tmp[0]);

// 2) float: x2 := x * 0.5
	x2 = x * 0.5f;
	//printf("2) x2 = %25.17e\n", x2);

// 3) DS: tmp[0] * x2
	//tmp[0] = ftwo_prod(tmp[0], x2, &tmp[1]);
	c_ds_mul_ds_f(tmp, x2, tmp);
	//printf("3) tmp[0] = %25.17e\n", tmp[0]);

// 4) DS: ax + tmp
	c_ds_add_ds_f(tmp, ax, b);
	//printf("4) b[0] = %25.17e\n", b[0]);

//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = sqrt(dd_real(a));
  TO_DOUBLEstatic _PTR(bb, b);
#endif // NATIVE_C
}


static inline void c_ds_abs(const float *a, float *b)
{
	if(a[0] < 0.0f)
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

static inline void c_ds_npwr(const float *a, int n, float *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = npwr(dd_real(a), n);
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

static inline void c_ds_nroot(const float *a, int n, float *b) {
// #ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = nroot(dd_real(a), n);
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

static inline void c_ds_nint(const float *a, float *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = nint(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
static inline void c_ds_aint(const float *a, float *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = aint(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
static inline void c_ds_floor(const float *a, float *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = floor(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
static inline void c_ds_ceil(const float *a, float *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = ceil(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

/* Forward declarations of the dtq-0.0.3 float-based elementary functions
   (DS delegates to DD).  Definitions come from bncelem_f.h, included at
   the end of this header.  The previous versions of these c_ds_* /
   c_qs_* functions were empty stubs.                                    */
static inline void bnc_ds_exp(const float *a, float *ret);
static inline void bnc_ds_log(const float *a, float *ret);
static inline void bnc_ds_log10(const float *a, float *ret);
static inline void bnc_ds_sin(const float *a, float *ret);
static inline void bnc_ds_cos(const float *a, float *ret);
static inline void bnc_ds_sincos(const float *a, float *sin_a, float *cos_a);
static inline void bnc_qs_exp(const float *a, float *ret);
static inline void bnc_qs_log(const float *a, float *ret);
static inline void bnc_qs_log10(const float *a, float *ret);
static inline void bnc_qs_sin(const float *a, float *ret);
static inline void bnc_qs_cos(const float *a, float *ret);
static inline void bnc_qs_sincos(const float *a, float *sin_a, float *cos_a);

static inline void c_ds_log(const float *a, float *b) {
	bnc_ds_log(a, b);
}

static inline void c_ds_log10(const float *a, float *b) {
	bnc_ds_log10(a, b);
}

static inline void c_ds_exp(const float *a, float *b) {
	bnc_ds_exp(a, b);
}

static inline void c_ds_sin(const float *a, float *b) {
	bnc_ds_sin(a, b);
}

static inline void c_ds_cos(const float *a, float *b) {
	bnc_ds_cos(a, b);
}

// tan(a) = sin(a) / cos(a)
static inline void c_ds_tan(const float *a, float *b) {
	float s[2], c[2];

	bnc_ds_sincos(a, s, c);
	c_ds_div(s, c, b);
}

static inline void c_ds_asin(const float *a, float *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = asin(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

static inline void c_ds_acos(const float *a, float *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = acos(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

static inline void c_ds_atan(const float *a, float *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = atan(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

static inline void c_ds_atan2(const float *a, const float *b, float *c) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real cc;
  cc = atan2(dd_real(a), dd_real(b));
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

static inline void c_ds_sinh(const float *a, float *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = sinh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
static inline void c_ds_cosh(const float *a, float *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = cosh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
static inline void c_ds_tanh(const float *a, float *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = tanh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

static inline void c_ds_asinh(const float *a, float *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = asinh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
static inline void c_ds_acosh(const float *a, float *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = acosh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
static inline void c_ds_atanh(const float *a, float *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = atanh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

// (dtq-0.0.3 port; the previous version of this function was an empty stub)
static inline void c_ds_sincos(const float *a, float *s, float *c) {
	bnc_ds_sincos(a, s, c);
}

static inline void c_ds_sincosh(const float *a, float *s, float *c) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real ss, cc;
  sincosh(dd_real(a), ss, cc);
  TO_DOUBLE_PTR(ss, s);
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

static inline void c_ds_read(const char *s, float *a) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real aa(s);
  TO_DOUBLE_PTR(aa, a);
#endif // NATIVE_C
}

static inline void c_ds_swrite(const float *a, int precision, char *s, int len) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real(a).write(s, len, precision);
#endif // NATIVE_C
}

static inline void c_ds_write(const float *a) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  std::cout << dd_real(a).to_string(dd_real::_ndigits) << std::endl;
#endif // NATIVE_C
}

static inline void c_ds_neg(const float *a, float *b)
{
	b[0] = -a[0];
	b[1] = -a[1];
}

static inline void c_ds_rand(float *a)
{
	static const float m_const = 4.6566128730773926e-10;  /* = 2^{-31} */
	int i;
	float m = 4.6566128730773926e-10;
	//dd_real r = 0.0;
	float r[2] = {0.0f, 0.0f};
	float d;

  /* Strategy:  Generate 31 bits at a time, using lrand48 
     random number generator.  Shift the bits, and reapeat
     4 times. */

	for (i = 0; i < 4; i++, m *= m_const)
	{
//    d = std::rand() * m;
		d = rand() * m;
		//r += d;
		c_ds_add_ds_f(r, d, r);
	}
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real aa;
  aa = ddrand();
  TO_DOUBLE_PTR(aa, a);
#endif // NATIVE_C
}

static inline void c_ds_comp(const float *a, const float *b, int *result)
{
//	dd_real aa(a), bb(b);

	// float-float < float-doube
	//if (aa < bb)
  	if(a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]))
		*result = -1;
	// float-float > float-doube
	//else if (aa > bb)
	else if(a[0] > b[0] || (a[0] == b[0] && a[1] > b[1]))
		*result = 1;
	else 
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

static inline void c_ds_comp_ds_f(const float *a, float b, int *result)
{
	//	dd_real aa(a), bb(b);
	float bb[DSSIZE];

	c_ds_copy_f(b, bb);

	c_ds_comp(a, bb, result);
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

static inline void c_ds_comp_f_ds(float a, const float *b, int *result)
{
	float aa[DSSIZE];

	c_ds_copy_f(a, aa);

	c_ds_comp(aa, b, result);
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

static inline void c_ds_pi(float *a)
{
	c_ds_copy(const_ds_pi, a);

//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  TO_DOUBLE_PTR(dd_real::_pi, a);
#endif // NATIVE_C
}

/**************************************/
/* QS                                 */
/**************************************/

/* (qd)b := (qd)a */
static inline void c_qs_copy(const float *a, float *b)
{
	b[0] = a[0];
	b[1] = a[1];
	b[2] = a[2];
	b[3] = a[3];
}

/* (qd)b := (dd)a */
static inline void c_qs_copy_ds(const float *a, float *b)
{
	b[0] = a[0];
	b[1] = a[1];
	b[2] = 0.0f;
	b[3] = 0.0f;
}

/* (qd)b := (float)a */
static inline void c_qs_copy_f(const float a, float *b)
{
	b[0] = a;
	b[1] = 0.0f;
	b[2] = 0.0f;
	b[3] = 0.0f;
}

// [x0, x1, x2, 0] := split(da, 24)
static inline void c_qs_copy_d(const double da, float *b)
{
	double x0, x1, x2, tmp, gamma, delta;

	gamma = _QS_D2F29 * da; // (2^29 + 1) * da
	delta = da - gamma;

	x0 = gamma + delta;
	b[0] = (float)x0;

	//tmp = da - (double)b[0];
	tmp = da - x0;

	gamma = _QS_D2F29 * tmp; // (2^29 + 1) * x1
	delta = tmp - gamma;
	x1 = gamma + delta;
	b[1] = (float)x1;

	//x2 = tmp - (double)b[1];
	x2 = tmp - x1;
	b[2] = (float)x2;

	b[3] = 0.0f;
}


/********** Renormalization **********/
static inline void fquick_renorm(float *c0, float *c1, float *c2, float *c3, float *c4)
{
	float t0, t1, t2, t3;
	float s;

	s   = fquick_two_sum(*c3, *c4, &t3);
	s   = fquick_two_sum(*c2, s  , &t2);
	s   = fquick_two_sum(*c1, s  , &t1);
	*c0 = fquick_two_sum(*c0, s  , &t0);

	s   = fquick_two_sum(t2, t3, &t2);
	s   = fquick_two_sum(t1, s , &t1);
	*c1 = fquick_two_sum(t0, s , &t0);

	s   = fquick_two_sum(t1, t2, &t1);
	*c2 = fquick_two_sum(t0, s , &t0);

	*c3 = t0 + t1;
}

static inline void frenorm(float *c0, float *c1, float *c2, float *c3)
{
	float s0, s1, s2 = 0.0, s3 = 0.0;

//	if (QS_ISINF(c0)) return;
	if (isinff(*c0)) return;

	s0  = fquick_two_sum(*c2, *c3, c3);
	s0  = fquick_two_sum(*c1,  s0, c2);
	*c0 = fquick_two_sum(*c0,  s0, c1);

	s0 = *c0;
	s1 = *c1;
	if (s1 != 0.0)
	{
		s1 = fquick_two_sum(s1, *c2, &s2);
		if (s2 != 0.0)
			s2 = fquick_two_sum(s2, *c3, &s3);
		else
			s1 = fquick_two_sum(s1, *c3, &s2);
	}
	else
	{
		s0 = fquick_two_sum(s0, *c2, &s1);
		if (s1 != 0.0)
			s1 = fquick_two_sum(s1, *c3, &s2);
		else
			s0 = fquick_two_sum(s0, *c3, &s1);
	}

	*c0 = s0;
	*c1 = s1;
	*c2 = s2;
	*c3 = s3;
}

static inline void frenorm4(float *c0, float *c1, float *c2, float *c3, float *c4)
{
	float s0, s1, s2 = 0.0, s3 = 0.0;

//	if (QS_ISINF(c0)) return;
	if (isinff(*c0)) return;

	s0  = fquick_two_sum(*c3, *c4, c4);
	s0  = fquick_two_sum(*c2, s0 , c3);
	s0  = fquick_two_sum(*c1, s0 , c2);
	*c0 = fquick_two_sum(*c0, s0 , c1);

	s0 = *c0;
	s1 = *c1;

//	s0 = fquick_two_sum(*c0, *c1, &s1);
	if (s1 != 0.0)
	{
		s1 = fquick_two_sum(s1, *c2, &s2);
		if (s2 != 0.0)
		{
			s2 = fquick_two_sum(s2, *c3, &s3);
			if (s3 != 0.0)
				s3 += *c4;
			else
				s2 = fquick_two_sum(s2, *c4, &s3); // fix!: 2020-11-06 by T.Kouya // s2 += *c4;
		}
		else
		{
			s1 = fquick_two_sum(s1, *c3, &s2);
			if (s2 != 0.0)
				s2 = fquick_two_sum(s2, *c4, &s3);
			else
				s1 = fquick_two_sum(s1, *c4, &s2);
		}
	}
	else
	{
		s0 = fquick_two_sum(s0, *c2, &s1);
		if (s1 != 0.0)
		{
			s1 = fquick_two_sum(s1, *c3, &s2);
			if (s2 != 0.0)
				s2 = fquick_two_sum(s2, *c4, &s3);
			else
				s1 = fquick_two_sum(s1, *c4, &s2);
		}
		else
		{
			s0 = fquick_two_sum(s0, *c3, &s1);
			if (s1 != 0.0)
				s1 = fquick_two_sum(s1, *c4, &s2);
			else
				s0 = fquick_two_sum(s0, *c4, &s1);
		}
	}

	*c0 = s0;
	*c1 = s1;
	*c2 = s2;
	*c3 = s3;
}


/********** Additions ************/
static inline void fthree_sum(float *a, float *b, float *c)
{
	float t1, t2, t3;

	t1 = ftwo_sum(*a, *b, &t2);
	*a = ftwo_sum(*c, t1, &t3);
	*b = ftwo_sum(t2, t3, c);
}

static inline void fthree_sum2(float *a, float *b, float *c)
{
	float t1, t2, t3;

	t1 = ftwo_sum(*a, *b, &t2);
	*a = ftwo_sum(*c, t1, &t3);
	*b = t2 + t3;
}

/* quad-float + float-float */
static inline void c_qs_add_qs_ds(const float *a, const float *b, float *c)
{
	float s0, s1, s2, s3;
	float t0, t1;

//	s0 = two_sum(a[0], b._hi(), t0);
//	s1 = two_sum(a[1], b._lo(), t1);
	s0 = ftwo_sum(a[0], DS_HI(b), &t0);
	s1 = ftwo_sum(a[1], DS_LOW(b), &t1);

	s1 = ftwo_sum(s1, t0, &t0);

	s2 = a[2];
	fthree_sum(&s2, &t0, &t1);

	s3 = ftwo_sum(t0, a[3], &t0);
	t0 += t1;

	frenorm4(&s0, &s1, &s2, &s3, &t0);

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

/* float-float + quad-float */
static inline void c_qs_add_ds_qs(const float *a, const float *b, float *c)
{
	c_qs_add_qs_ds(b, a, c);

#if 0 //
  qd_real cc;
  cc = dd_real(a) + qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/* quad-float + float */
static inline void c_qs_add_qs_f(const float *a, float b, float *c)
{
//	float c0, c1, c2, c3;
	float e;

	c[0] = ftwo_sum(a[0], b, &e);
	c[1] = ftwo_sum(a[1], e, &e);
	c[2] = ftwo_sum(a[2], e, &e);
	c[3] = ftwo_sum(a[3], e, &e);

	//qd::renorm(c0, c1, c2, c3, e);
	frenorm4(&(c[0]), &(c[1]), &(c[2]), &(c[3]), &e);

//	return qd_real(c0, c1, c2, c3);
	return;
#if 0 //
  qd_real cc;
  cc = qd_real(a) + b;
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/* float + quad-float */
static inline void c_qs_add_d_qs(float a, const float *b, float *c)
{
	c_qs_add_qs_f(b, a, c);

#if 0
  qd_real cc;
  cc = a + qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/* s = quick_three_accum(a, b, c) adds c to the dd-pair (a, b).
 * If the result does not fit in two floats, then the sum is 
 * output into s and (a,b) contains the remainder.  Otherwise
 * s is zero and (a,b) contains the sum. */
static inline float fquick_three_accum(float *a, float *b, float c)
{
	float s;
//	bool za, zb;

	s = ftwo_sum(*b, c, b);
	s = ftwo_sum(*a, s, a);

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
static inline void c_qs_add(const float *a, const float *b, float *c)
{
	// IEEE add
	int i, j, k;
	float s, t;
	float u, v;   /* float-length accumulator */
	float x[4] = {0.0, 0.0, 0.0, 0.0};

	i = j = k = 0;
	if (fabsf(a[i]) > fabsf(b[j]))
		u = a[i++];
	else
		u = b[j++];
	if (fabsf(a[i]) > fabsf(b[j]))
		v = a[i++];
	else
		v = b[j++];

	u = fquick_two_sum(u, v, &v);

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
		else if (fabsf(a[i]) > fabsf(b[j]))
			t = a[i++];
		else
			t = b[j++];

		s = fquick_three_accum(&u, &v, t);

		if (s != 0.0) {
			x[k++] = s;
		}
	}

	/* add the rest. */
	for (k = i; k < 4; k++)
		x[3] += a[k];
	for (k = j; k < 4; k++)
		x[3] += b[k];

	frenorm(&(x[0]), &(x[1]), &(x[2]), &(x[3]));
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
static inline void c_qs_add_sloppy(const float *a, const float *b, float *c)
{
	/*
	float s0, s1, s2, s3;
	float t0, t1, t2, t3;

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
	float s0, s1, s2, s3;
	float t0, t1, t2, t3;

	float v0, v1, v2, v3;
	float u0, u1, u2, u3;
	float w0, w1, w2, w3;

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

	s1 = ftwo_sum(s1, t0, &t0);
	fthree_sum(&s2, &t0, &t1);
	fthree_sum2(&s3, &t0, &t2);
	t0 = t0 + t1 + t3;

	/* renormalize */
	frenorm4(&s0, &s1, &s2, &s3, &t0);
//	return qd_real(s0, s1, s2, s3);
	c[0] = s0;
	c[1] = s1;
	c[2] = s2;
	c[3] = s3;
}

// Sloppy add for triple float prec
static inline void c_ts_addq(const float *a, const float *b, float *c)
{

	/* Same as above, but addition re-organized to minimize
	   data dependency ... unfortunately some compilers are
	   not very smart to do this automatically */
	float s0, s1, s2;
	float t0, t1, t2;

	float v0, v1, v2;
	float u0, u1, u2;
	float w0, w1, w2;

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

	s1 = ftwo_sum(s1, t0, &t0);
	fthree_sum(&s2, &t0, &t1);
	//three_sum2(&s3, &t0, &t2);
	t0 = t0 + t1 + t2;

	/* renormalize */
	frenorm(&s0, &s1, &s2, &t0);
//	return qd_real(s0, s1, s2, s3);
	c[0] = s0;
	c[1] = s1;
	c[2] = s2;
	//c[3] = s3;
}



/********** Self-Additions ************/
// b := b + a
static inline void c_qs_selfadd(const float *a, float *b)
{
	float bb[QSSIZE];

	c_qs_copy(b, bb);
	c_qs_add(a, bb, b);

#if 0 //
	qd_real bb(b);
	bb += qd_real(a);
	TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b + (dd)a
static inline void c_qs_selfadd_ds(const float *a, float *b)
{
	float bb[QSSIZE];

	c_qs_copy(b, bb);
	c_qs_add_ds_qs(a, bb, b);

#if 0 //
  qd_real bb(b);
  bb += dd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b + (float)a
static inline void c_qs_selfadd_f(float a, float *b)
{
	float bb[QSSIZE];

	c_qs_copy(b, bb);
	c_qs_add_d_qs(a, bb, b);

#if 0 //
  qd_real bb(b);
  bb += a;
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}


/********** Unary Minus **********/
static inline void c_qs_neg(const float *a, float *b) {
	b[0] = -a[0];
	b[1] = -a[1];
	b[2] = -a[2];
	b[3] = -a[3];
}

// b := (dd)(-a)
static inline void c_qs_neg_ds(const float *a, float *b) {
	b[0] = -a[0];
	b[1] = -a[1];
	b[2] = 0.0;
	b[3] = 0.0;
}

// b := (float)(-a)
static inline void c_qs_neg_f(const float a, float *b) 
{
	b[0] = -a;
	b[1] = 0.0;
	b[2] = 0.0;
	b[3] = 0.0;
}

/********** Subtractions **********/
/* sub */
// c := a - b
static inline void c_qs_sub(const float *a, const float *b, float *c)
{
	float mb[QSSIZE];

	// a + (-b)
	c_qs_neg(b, mb);
	c_qs_add(a, mb, c);

#if 0
  qd_real cc;
  cc = qd_real(a) - qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// c := a - (dd)b
static inline void c_qs_sub_qs_ds(const float *a, const float *b, float *c)
{
	float mb[QSSIZE];

	// a + (-b)
	c_qs_neg_ds(b, mb);
	c_qs_add(a, mb, c);

#if 0
  qd_real cc;
  cc = qd_real(a) - dd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// c := (dd)a - b
static inline void c_qs_sub_ds_qs(const float *a, const float *b, float *c)
{
	float ma[QSSIZE];

	// (-a) + b
	c_qs_neg_ds(a, ma);
	c_qs_add(ma, b, c);

#if 0
  qd_real cc;
  cc = dd_real(a) - qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// c := a - (float)b
static inline void c_qs_sub_qs_f(const float *a, float b, float *c)
{
	float mb[QSSIZE];

	// a + (-b)
	c_qs_neg_f(b, mb);
	c_qs_add(a, mb, c);

#if 0
  qd_real cc;
  cc = qd_real(a) - b;
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// c := (float)a - b   (scalar minus QS)
// Bug fix: previous implementation computed (-a) + b = b - a (sign reversed).
// This corrupted c_qs_sqrt's Newton iteration since it expected (0.5 - h*r^2)
// but got (h*r^2 - 0.5), which converges to the wrong fixed point.
static inline void c_qs_sub_s_qs(float a, const float *b, float *c)
{
	float mb[QSSIZE];

	// a - b = a + (-b)
	c_qs_neg(b, mb);
	c_qs_add_d_qs(a, mb, c);

#if 0
  qd_real cc;
  cc = a - qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/********** Self-Subtractions **********/
/* selfsub */
// b := b + (-a) <-> b -= a
static inline void c_qs_selfsub(const float *a, float *b)
{
	float ma[QSSIZE];

	// (-a) + b
	c_qs_neg(a, ma);
	c_qs_add(ma, b, b);

#if 0
  qd_real bb(b);
  bb -= qd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b - (dd)a
static inline void c_qs_selfsub_ds(const float *a, float *b)
{
	float ma[QSSIZE];

	// (-a) + b
	c_qs_neg_ds(a, ma);
	c_qs_add(ma, b, b);

#if 0
  qd_real bb(b);
  bb -= dd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b - (float)a
static inline void c_qs_selfsub_f(float a, float *b)
{
	float ma[QSSIZE];

	// (-a) + b
	c_qs_neg_f(a, ma);
	c_qs_add(ma, b, b);

#if 0
  qd_real bb(b);
  bb -= a;
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

/********** Multiplications **********/
static inline void c_qs_mul(const float *a, const float *b, float *c)
{
	// acculate mul
	float p0, p1, p2, p3, p4, p5;
	float q0, q1, q2, q3, q4, q5;
	float p6, p7, p8, p9;
	float q6, q7, q8, q9;
	float r0, r1;
	float t0, t1;
	float s0, s1, s2;

	p0 = ftwo_prod(a[0], b[0], &q0);

	p1 = ftwo_prod(a[0], b[1], &q1);
	p2 = ftwo_prod(a[1], b[0], &q2);

	p3 = ftwo_prod(a[0], b[2], &q3);
	p4 = ftwo_prod(a[1], b[1], &q4);
	p5 = ftwo_prod(a[2], b[0], &q5);

	/* Start Accumulation */
	fthree_sum(&p1, &p2, &q0);

	/* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
	fthree_sum(&p2, &q1, &q2);
	fthree_sum(&p3, &p4, &p5);

	/* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
	s0 = ftwo_sum(p2, p3, &t0);
	s1 = ftwo_sum(q1, p4, &t1);
	s2 = q2 + p5;
	s1 = ftwo_sum(s1, t0, &t0);
	s2 += (t0 + t1);

	/* O(eps^3) order terms */
	p6 = ftwo_prod(a[0], b[3], &q6);
	p7 = ftwo_prod(a[1], b[2], &q7);
	p8 = ftwo_prod(a[2], b[1], &q8);
	p9 = ftwo_prod(a[3], b[0], &q9);

	/* Nine-Two-Sum of q0, s1, q3, q4, q5, p6, p7, p8, p9. */
	q0 = ftwo_sum(q0, q3, &q3);
	q4 = ftwo_sum(q4, q5, &q5);
	p6 = ftwo_sum(p6, p7, &p7);
	p8 = ftwo_sum(p8, p9, &p9);

	/* Compute (t0, t1) = (q0, q3) + (q4, q5). */
	t0 = ftwo_sum(q0, q4, &t1);
	t1 += (q3 + q5);

	/* Compute (r0, r1) = (p6, p7) + (p8, p9). */
	r0 = ftwo_sum(p6, p8, &r1);
	r1 += (p7 + p9);

	/* Compute (q3, q4) = (t0, t1) + (r0, r1). */
	q3 = ftwo_sum(t0, r0, &q4);
	q4 += (t1 + r1);

	/* Compute (t0, t1) = (q3, q4) + s1. */
	t0 = ftwo_sum(q3, s1, &t1);
	t1 += q4;

	/* O(eps^4) terms -- Nine-One-Sum */
	t1 += a[1] * b[3] + a[2] * b[2] + a[3] * b[1] + q6 + q7 + q8 + q9 + s2;

	frenorm4(&p0, &p1, &s0, &t0, &t1);
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

/* quad-float * float-float */
/* a0 * b0                        0
        a0 * b1                   1
        a1 * b0                   2
             a1 * b1              3
             a2 * b0              4
                  a2 * b1         5
                  a3 * b0         6
                       a3 * b1    7 */
// c := a * (dd)b
static inline void c_qs_mul_qs_ds(const float *a, const float *b, float *c)
{
	float p0, p1, p2, p3, p4;
	float q0, q1, q2, q3, q4;
	float s0, s1, s2;
	float t0, t1;

	p0 = ftwo_prod(a[0], DS_HI(b) , &q0);
	p1 = ftwo_prod(a[0], DS_LOW(b), &q1);
	p2 = ftwo_prod(a[1], DS_HI(b) , &q2);
	p3 = ftwo_prod(a[1], DS_LOW(b), &q3);
	p4 = ftwo_prod(a[2], DS_HI(b) , &q4);

	fthree_sum(&p1, &p2, &q0);

	/* Five-Three-Sum */
	fthree_sum(&p2, &p3, &p4);
	q1 = ftwo_sum(q1, q2, &q2);
	s0 = ftwo_sum(p2, q1, &t0);
	s1 = ftwo_sum(p3, q2, &t1);
	s1 = ftwo_sum(s1, t0, &t0);
	s2 = t0 + t1 + p4;
	p2 = s0;

	p3 = a[2] * DS_HI(b) + a[3] * DS_LOW(b) + q3 + q4;
	fthree_sum2(&p3, &q0, &s1);
	p4 = q0 + s2;

	frenorm4(&p0, &p1, &p2, &p3, &p4);
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
static inline void c_qs_mul_ds_qs(const float *a, const float *b, float *c)
{
	c_qs_mul_qs_ds(b, a, c);

#if 0
  qd_real cc;
  cc = dd_real(a) * qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// c := a * (float)b
static inline void c_qs_mul_qs_f(const float *a, float b, float *c)
{
	float p0, p1, p2, p3;
	float q0, q1, q2;
	float s0, s1, s2, s3, s4;

	p0 = ftwo_prod(a[0], b, &q0);
	p1 = ftwo_prod(a[1], b, &q1);
	p2 = ftwo_prod(a[2], b, &q2);
	p3 = a[3] * b;

	s0 = p0;

	s1 = ftwo_sum(q0, p1, &s2);

	fthree_sum(&s2, &q1, &p2);

	fthree_sum2(&q1, &q2, &p3);
	s3 = q1;

	s4 = q2 + p2;

	frenorm4(&s0, &s1, &s2, &s3, &s4);
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

// c := (float)a * b
static inline void c_qs_mul_f_qs(float a, const float *b, float *c)
{
	c_qs_mul_qs_f(b, a, c);

#if 0
  qd_real cc;
  cc = a * qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/* quad-float * quad-float */
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
static inline void c_qs_mul_sloppy(const float *a, const float *b, float *c)
{
	float p0, p1, p2, p3, p4, p5;
	float q0, q1, q2, q3, q4, q5;
	float t0, t1;
	float s0, s1, s2;

	p0 = ftwo_prod(a[0], b[0], &q0);

	p1 = ftwo_prod(a[0], b[1], &q1);
	p2 = ftwo_prod(a[1], b[0], &q2);

	p3 = ftwo_prod(a[0], b[2], &q3);
	p4 = ftwo_prod(a[1], b[1], &q4);
	p5 = ftwo_prod(a[2], b[0], &q5);

	/* Start Accumulation */
	fthree_sum(&p1, &p2, &q0);

	/* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
	fthree_sum(&p2, &q1, &q2);
	fthree_sum(&p3, &p4, &p5);

	/* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
	s0 = ftwo_sum(p2, p3, &t0);
	s1 = ftwo_sum(q1, p4, &t1);
	s2 = q2 + p5;
	s1 = ftwo_sum(s1, t0, &t0);
	s2 += (t0 + t1);

	/* O(eps^3) order terms */
	s1 += a[0]*b[3] + a[1]*b[2] + a[2]*b[1] + a[3]*b[0] + q0 + q3 + q4 + q5;
	//renorm(p0, p1, s0, s1, s2);
	frenorm4(&p0, &p1, &s0, &s1, &s2);
//	return qd_real(p0, p1, s0, s1);
	c[0] = p0;
	c[1] = p1;
	c[2] = s0;
	c[3] = s1;
}

/* quad-float ^ 2  = (x0 + x1 + x2 + x3) ^ 2
                    = x0 ^ 2 + 2 x0 * x1 + (2 x0 * x2 + x1 ^ 2)
                               + (2 x0 * x3 + 2 x1 * x2)           */
static inline void c_qs_sqr(const float *a, float *c)
{
	float p0, p1, p2, p3, p4, p5;
	float q0, q1, q2, q3;
	float s0, s1;
	float t0, t1;

	p0 = ftwo_sqr(a[0], &q0);
	p1 = ftwo_prod(2.0 * a[0], a[1], &q1);
	p2 = ftwo_prod(2.0 * a[0], a[2], &q2);
	p3 = ftwo_sqr(a[1], &q3);

	p1 = ftwo_sum(q0, p1, &q0);

	q0 = ftwo_sum(q0, q1, &q1);
	p2 = ftwo_sum(p2, p3, &p3);

	s0 = ftwo_sum(q0, p2, &t0);
	s1 = ftwo_sum(q1, p3, &t1);

	s1 = ftwo_sum(s1, t0, &t0);
	t0 += t1;

	s1 = fquick_two_sum(s1, t0, &t0);
	p2 = fquick_two_sum(s0, s1, &t1);
	p3 = fquick_two_sum(t1, t0, &q0);

	p4 = 2.0 * a[0] * a[3];
	p5 = 2.0 * a[1] * a[2];

	p4 = ftwo_sum(p4, p5, &p5);
	q2 = ftwo_sum(q2, q3, &q3);

	t0 = ftwo_sum(p4, q2, &t1);
	t1 = t1 + p5 + q3;

	p3 = ftwo_sum(p3, t0, &p4);
	p4 = p4 + q0 + t1;

	frenorm4(&p0, &p1, &p2, &p3, &p4);
//  return qd_real(p0, p1, p2, p3);
	c[0] = p0;
	c[1] = p1;
	c[2] = p2;
	c[3] = p3;

}


/********** Self-Multiplication **********/
/* selfmul */
// b := b * a
static inline void c_qs_selfmul(const float *a, float *b)
{
	float bb[QSSIZE];

	c_qs_copy(b, bb);
	c_qs_mul(a, bb, b);

#if 0
  qd_real bb(b);
  bb *= qd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b * (dd)a
static inline void c_qs_selfmul_ds(const float *a, float *b)
{
	float bb[QSSIZE];

	c_qs_copy(b, bb);
	c_qs_mul_qs_ds(bb, a, b);

#if 0
  qd_real bb(b);
  bb *= dd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b * (doule)a
static inline void c_qs_selfmul_f(float a, float *b)
{
	float bb[QSSIZE];

	c_qs_copy(b, bb);
	c_qs_mul_qs_f(bb, a, b);

#if 0
  qd_real bb(b);
  bb *= a;
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

/********** Divisions **********/

/* div */
// c := a / b
static inline void c_qs_div_accurate(const float *a, const float *b, float *c)
{
	//qd_real::accurate_div
	float q0, q1, q2, q3, q4;
	float r[QSSIZE], tmp[QSSIZE];
	//qd_real r;

	q0 = a[0] / b[0];
	//r = a - (b * q0);
	c_qs_mul_qs_f(b, q0, tmp);
	c_qs_sub(a, tmp, r);

	q1 = r[0] / b[0];
	//r -= (b * q1);
	c_qs_mul_qs_f(b, q1, tmp);
	c_qs_selfsub(tmp, r);
	//c_qs_sub(r, tmp, r);

	q2 = r[0] / b[0];
	//r -= (b * q2);
	c_qs_mul_qs_f(b, q2, tmp);
	c_qs_selfsub(tmp, r);
	//c_qs_sub(r, tmp, r);

	q3 = r[0] / b[0];

	//r -= (b * q3);
	c_qs_mul_qs_f(b, q3, tmp);
	c_qs_selfsub(tmp, r);
	//c_qs_sub(r, tmp, r);

	q4 = r[0] / b[0];

	frenorm4(&q0, &q1, &q2, &q3, &q4);

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

/* quad-float / quad-float */
// c := a / b
static inline void c_qs_div_sloppy(const float *a, const float *b, float *c)
{
	//qd_real qd_real::sloppy_div(const qd_real &a, const qd_real &b) {
	float q0, q1, q2, q3;
	float r[QSSIZE], tmp[QSSIZE];
	//qd_real r;

	q0 = a[0] / b[0];
	//r = a - (b * q0);
	c_qs_mul_qs_f(b, q0, tmp);
	c_qs_sub(a, tmp, r);

	q1 = r[0] / b[0];
	//r -= (b * q1);
	c_qs_mul_qs_f(b, q1, tmp);
	c_qs_selfsub(tmp, r);
	//c_qs_sub(r, tmp, r);

	q2 = r[0] / b[0];
	//r -= (b * q2);
	c_qs_mul_qs_f(b, q2, tmp);
	c_qs_selfsub(tmp, r);
	//c_qs_sub(r, tmp, r);

	q3 = r[0] / b[0];

	frenorm(&q0, &q1, &q2, &q3);

	//return qd_real(q0, q1, q2, q3);
	c[0] = q0;
	c[1] = q1;
	c[2] = q2;
	c[3] = q3;
}


#ifndef USE_QS_DIV_ACCURATE
	#ifdef BNC_USE_FMA_DIV
		#define c_qs_div bnc_qs_div_fma
	#else
		#define c_qs_div c_qs_div_sloppy
	#endif
//	#define c_qs_div c_qs_div_accurate
#else // USE_QS_DIV_ACCURATE
	#define c_qs_div c_qs_div_accurate
#endif //  USE_QS_DIV_ACCURATE

/* quad-float / float-float */
// c := a / (dd)b
static inline void c_qs_div_qs_ds(const float *a, const float *b, float *c)
{
	//qd_real qd_real::sloppy_div(const qd_real &a, const dd_real &b) {
	float q0, q1, q2, q3;
	float r[QSSIZE], tmp[QSSIZE], qd_b[QSSIZE];
	//qd_real r;
	//qd_real qd_b(b);

	c_qs_copy_ds(b, qd_b);

	q0 = a[0] / DS_HI(b);
	//r = a - q0 * qd_b;
	c_qs_mul_f_qs(q0, qd_b, tmp);
	c_qs_sub(a, tmp, r);

	q1 = r[0] / DS_HI(b);
	//r -= (q1 * qd_b);
	c_qs_mul_f_qs(q1, qd_b, tmp);
	c_qs_selfsub(tmp, r);

	q2 = r[0] / DS_HI(b);
	//r -= (q2 * qd_b);
	c_qs_mul_f_qs(q2, qd_b, tmp);
	c_qs_selfsub(tmp, r);

	q3 = r[0] / DS_HI(b);

	//::renorm(q0, q1, q2, q3);
	frenorm(&q0, &q1, &q2, &q3);
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

/* float-float / quad-float */
// c := (dd)a / b
static inline void c_qs_div_ds_qs(const float *a, const float *b, float *c)
{
	float aa[QSSIZE];

	c_qs_copy_ds(a, aa);
	c_qs_div(aa, b, c);

#if 0 
  qd_real cc;
  cc = dd_real(a) / qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/* quad-float / float */
// c := a / (float)b
static inline void c_qs_div_qs_f(const float *a, float b, float *c)
{
	float bb[QSSIZE];

	c_qs_copy_f(b, bb);
	c_qs_div(a, bb, c);

#if 0 
  qd_real cc;
  cc = qd_real(a) / b;
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/* float / quad-float */
// c := (float)a / b
static inline void c_qs_div_f_qs(float a, const float *b, float *c)
{
	float aa[QSSIZE];

	c_qs_copy_f(a, aa);
	c_qs_div(aa, b, c);

#if 0 
  qd_real cc;
  cc = a / qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/********** Self-Divisions **********/
/* selfdiv */
// b := b / a
static inline void c_qs_selfdiv(const float *a, float *b) 
{
	float bb[QSSIZE];

	c_qs_copy(b, bb);
	c_qs_div(bb, a, b);

#if 0
  qd_real bb(b);
  bb /= qd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b / (dd)a
static inline void c_qs_selfdiv_ds(const float *a, float *b)
{
	float bb[QSSIZE];

	c_qs_copy(b, bb);
	c_qs_div_qs_ds(bb, a, b);

#if 0
  qd_real bb(b);
  bb /= dd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
static inline void c_qs_selfdiv_f(float a, float *b)
{
	float bb[QSSIZE];

	c_qs_copy(b, bb);
	c_qs_div_qs_f(bb, a, b);

#if 0
  qd_real bb(b);
  bb /= a;
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// c := (a[0] * b, a[1] * b, a[2] * b, a[3] * b)
static inline void c_qs_mul_pwr2(const float *a, float b, float *c)
{
	//return qd_real(a[0] * b, a[1] * b, a[2] * b, a[3] * b);
	c[0] = a[0] * b;
	c[1] = a[1] * b;
	c[2] = a[2] * b;
	c[3] = a[3] * b;
}

static inline void c_qs_set0(float *qdval)
{
	qdval[0] = 0.0f;
	qdval[1] = 0.0f;
	qdval[2] = 0.0f;
	qdval[3] = 0.0f;
}

static inline void c_qs_sqrt(const float *a, float *b)
{
//QS_API qd_real sqrt(const qd_real &a) {
  /* Strategy:  

     Perform the following Newton iteration:

       x' = x + (1 - a * x^2) * x / 2;
       
     which converges to 1/sqrt(a), starting with the
     float precision approximation to 1/sqrt(a).
     Since Newton's iteration more or less floats the
     number of correct digits, we only need to perform it 
     twice.
  */
	float r[QSSIZE], h[QSSIZE], tmp[QSSIZE];

	if (QS_ISZERO(a))
	{
		c_qs_set0(b);
		return;
	}

	if (QS_ISNEGATIVE(a))
	{
		fprintf(stderr, "(qs_real::sqrt): Negative argument.");
		b[0] = QS_NAN;
		return;
	}

//  qd_real r = (1.0 / std::sqrt(a[0]));
//  qd_real h = mul_pwr2(a, 0.5);
//	r[0] = 1.0 / sqrt(a[0]); r[1] = 0.0; r[2] = 0.0; r[3] = 0.0;
	h[0] = 1.0f; h[1] = 0.0f; h[2] = 0.0f; h[3] = 0.0f;
	r[0] = sqrtf(a[0]); r[1] = 0.0f; r[2] = 0.0f; r[3] = 0.0f;
	c_qs_div(h, r, r);

	c_qs_mul_pwr2(a, 0.5, h);

//  r += ((0.5 - h * sqr(r)) * r);
//  r += ((0.5 - h * sqr(r)) * r);
//  r += ((0.5 - h * sqr(r)) * r);

	c_qs_sqr(r, tmp);
	c_qs_mul(h, tmp, tmp);
	c_qs_sub_s_qs(0.5, tmp, tmp);
	c_qs_mul(tmp, r, tmp);
	c_qs_add(tmp, r, r);

	c_qs_sqr(r, tmp);
	c_qs_mul(h, tmp, tmp);
	c_qs_sub_s_qs(0.5, tmp, tmp);
	c_qs_mul(tmp, r, tmp);
	c_qs_add(tmp, r, r);

	c_qs_sqr(r, tmp);
	c_qs_mul(h, tmp, tmp);
	c_qs_sub_s_qs(0.5, tmp, tmp);
	c_qs_mul(tmp, r, tmp);
	c_qs_add(tmp, r, r);

//  r *= a;
//	c_qs_selfmul(a, r);
	c_qs_mul(r, a, r);
	c_qs_copy(r, b);
//  return r;
#if 0
  qd_real bb;
  bb = sqrt(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := |a| 
static inline void c_qs_abs(const float *a, float *b)
{
	// return (a[0] < 0.0) ? -a : a;
	if(a[0] < 0.0)
		c_qs_neg(a, b);
	else
		c_qs_copy(a, b);
#if 0
  qd_real bb;
  bb = abs(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

static inline void c_qs_npwr(const float *a, int n, float *b) {
#if 0
  qd_real bb;
  bb = npwr(qd_real(a), n);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

static inline void c_qs_nroot(const float *a, int n, float *b) {
#if 0
  qd_real bb;
  bb = nroot(qd_real(a), n);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// round to nearest integer
#define nintf(a) (roundf(a))

static inline void c_qs_nint(const float *a, float *b)
{
	float x0, x1, x2, x3;

	x0 = nintf(a[0]);
	x1 = x2 = x3 = 0.0;

	if (x0 == a[0])
	{
	  /* First float is already an integer. */
	  x1 = nintf(a[1]);

	  if (x1 == a[1]) {
	    /* Second float is already an integer. */
	    x2 = nintf(a[2]);
	    
	    if (x2 == a[2]) {
	      /* Third float is already an integer. */
	      x3 = nintf(a[3]);
	    } else {
	      if(fabsf(x2 - a[2]) == 0.5 && a[3] < 0.0) {
	        x2 -= 1.0;
	      }
	    }

	  } else {
	    if (fabsf(x1 - a[1]) == 0.5 && a[2] < 0.0) {
	        x1 -= 1.0;
	    }
	  }

	} else {
	  /* First float is not an integer. */
	    if (fabsf(x0 - a[0]) == 0.5 && a[1] < 0.0) {
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

static inline void c_qs_floor(const float *a, float *b)
{
	float x0, x1, x2, x3;
	x1 = x2 = x3 = 0.0f;
	x0 = floorf(a[0]);

	if (x0 == a[0])
	{
		x1 = floorf(a[1]);
		
		if (x1 == a[1])
		{
			x2 = floorf(a[2]);

			if (x2 == a[2])
			{
				x3 = floorf(a[3]);
			}
		}

		frenorm(&x0, &x1, &x2, &x3);
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
static inline void c_qs_ceil(const float *a, float *b)
{
	float x0, x1, x2, x3;
	x1 = x2 = x3 = 0.0;
	x0 = ceilf(a[0]);

	if (x0 == a[0])
	{
		x1 = ceilf(a[1]);
		
		if (x1 == a[1])
		{
			x2 = ceilf(a[2]);

			if (x2 == a[2])
			{
				x3 = ceilf(a[3]);
			}
		}

		frenorm(&x0, &x1, &x2, &x3);
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

static inline void c_qs_aint(const float *a, float *b)
{
//  return (a[0] >= 0) ? floor(a) : ceil(a);
	if(a[0] >= 0.0)
		c_qs_floor(a, b);
	else
		c_qs_ceil(a, b);

#if 0
  qd_real bb;
  bb = aint(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}


// (dtq-0.0.3 ports; the previous versions of these functions were empty stubs)
static inline void c_qs_log(const float *a, float *b) {
	bnc_qs_log(a, b);
}
static inline void c_qs_log10(const float *a, float *b) {
	bnc_qs_log10(a, b);
}
static inline void c_qs_exp(const float *a, float *b) {
	bnc_qs_exp(a, b);
}

static inline void c_qs_sin(const float *a, float *b) {
	bnc_qs_sin(a, b);
}
static inline void c_qs_cos(const float *a, float *b) {
	bnc_qs_cos(a, b);
}
// tan(a) = sin(a) / cos(a)
static inline void c_qs_tan(const float *a, float *b) {
	float s[4], c[4];

	bnc_qs_sincos(a, s, c);
	c_qs_div(s, c, b);
}

static inline void c_qs_asin(const float *a, float *b) {
#if 0
 qd_real bb;
  bb = asin(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
static inline void c_qs_acos(const float *a, float *b) {
#if 0
  qd_real bb;
  bb = acos(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
static inline void c_qs_atan(const float *a, float *b) {
#if 0
  qd_real bb;
  bb = atan(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

static inline void c_qs_atan2(const float *a, const float *b, float *c) {
#if 0
  qd_real cc;
  cc = atan2(qd_real(a), qd_real(b));
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

static inline void c_qs_sinh(const float *a, float *b) {
#if 0
  qd_real bb;
  bb = sinh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
static inline void c_qs_cosh(const float *a, float *b) {
#if 0
  qd_real bb;
  bb = cosh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
static inline void c_qs_tanh(const float *a, float *b) {
#if 0
  qd_real bb;
  bb = tanh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

static inline void c_qs_asinh(const float *a, float *b) {
#if 0
  qd_real bb;
  bb = asinh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

static inline void c_qs_acosh(const float *a, float *b) {
#if 0
  qd_real bb;
  bb = acosh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
static inline void c_qs_atanh(const float *a, float *b) {
#if 0
  qd_real bb;
  bb = atanh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// (dtq-0.0.3 port; the previous version of this function was an empty stub)
static inline void c_qs_sincos(const float *a, float *s, float *c) {
	bnc_qs_sincos(a, s, c);
}

static inline void c_qs_sincosh(const float *a, float *s, float *c) {
#if 0
  qd_real ss, cc;
  sincosh(qd_real(a), ss, cc);
  TO_DOUBLE_PTR(cc, c);
  TO_DOUBLE_PTR(ss, s);
#endif // 0
}

static inline void c_qs_read(const char *s, float *a) {
#if 0
  qd_real aa(s);
  TO_DOUBLE_PTR(aa, a);
#endif // 0
}

static inline void c_qs_swrite(const float *a, int precision, char *s, int len) {
#if 0
  qd_real(a).write(s, len, precision);
#endif // 0
}

static inline void c_qs_write(const float *a) {
#if 0
  std::cout << qd_real(a).to_string(qd_real::_ndigits) << std::endl;
#endif // 0
}

static inline void c_qs_rand(float *a) {
#if 0
  qd_real aa;
  aa = qdrand();
  TO_DOUBLE_PTR(aa, a);
#endif // 0
}

static inline void c_qs_comp(const float *a, const float *b, int *result) 
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

static inline void c_qs_comp_qs_f(const float *a, float b, int *result)
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

static inline void c_qs_comp_d_qs(float a, const float *b, int *result)
{
/* qd_real bb(b);
  if (a < bb)
    *result = -1;
  else if (a > bb)
    *result = 1;
  else 
    *result = 0;
*/
	c_qs_comp_qs_f(b, a, result);
	*result = -(*result);
}

static inline void c_qs_pi(float *a)
{
	c_qs_copy(const_qs_pi, a);

#if 0
  TO_DOUBLE_PTR(qd_real::_pi, a);
#endif // 0
}

/*******************************************/
/* Triple precision arithmetic             */
/*******************************************/
// N.Fabiano et.al.

/* (td)b := (td)a */
static inline void c_ts_copy(const float *a, float *b)
{
	b[0] = a[0];
	b[1] = a[1];
	b[2] = a[2];
}

// (qd)c := (td)a
static inline void c_qs_copy_ts(const float *a, float *c)
{
	c[0] = a[0];
	c[1] = a[1];
	c[2] = a[2];
	c[3] = 0.0;
}

/* (td)b := (qd)a */
static inline void c_ts_copy_qs(const float *a, float *b)
{
	b[0] = a[0];
	b[1] = a[1];
	b[2] = a[2];
}

/* (td)b := (dd)a */
static inline void c_ts_copy_ds(const float *a, float *b)
{
	b[0] = a[0];
	b[1] = a[1];
	b[2] = 0.0f;
}

/* (td)b := (float)a */
static inline void c_ts_copy_f(const float a, float *b)
{
	b[0] = a;
	b[1] = 0.0f;
	b[2] = 0.0f;
}

// [x0, x1, x2] := split(da, 24)
static inline void c_ts_copy_d(const double da, float *b)
{
	double x0, x1, x2, tmp, gamma, delta;

	gamma = _QS_D2F29 * da; // (2^29 + 1) * da
	delta = da - gamma;

	x0 = gamma + delta;
	b[0] = (float)x0;

	//tmp = da - (double)b[0];
	tmp = da - x0;

	gamma = _QS_D2F29 * tmp; // (2^29 + 1) * x1
	delta = tmp - gamma;
	x1 = gamma + delta;
	b[1] = (float)x1;

	//x2 = tmp - (double)b[1];
	x2 = tmp - x1;
	b[2] = (float)x2;
}

// e[n] := vec_sum(x[n])
static inline void fvec_sum(float *e, const float *x, int n)
{
    float s;

	s = x[--n];
	while(--n >= 0)
		s = ftwo_sum(x[n], s, &e[n + 1]);
	e[0] = s;
}

// y[n] := vec_sum_err_branch(vseb)(k)(e[n])
static inline void fvseb(float *y, int ny, const float *e, int ne)
{
	int i, j;
	float r, eps, temp, in_y[16];

//	printf("ny = %d\n", ny);
//	if(ny > ne)
//		ny = ne;

	j = 0;
	eps = e[0];
	for(i = 0; i < (ne - 2); i++)
	{
		r = ftwo_sum(eps, e[i + 1], &temp);
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
	in_y[j] = ftwo_sum(eps, e[ne - 1], &in_y[j + 1]);

	for(i = j + 2; i < ne; i++)
		in_y[i] = 0.0;

	for(i = 0; i < ny; i++)
		y[i] = in_y[i];

}

// r[3] := to_ts(a, b, c)
static inline void c_to_ts(float *r, float a, float b, float c)
{
	float d[3], e[3];

	d[0] = ftwo_sum(a, b, &d[1]);
	d[2] = c;
	fvec_sum(e, d, 3);
	fvseb(r, 3, e, 3);
}

// [TODO] float := c_ts2d(a[3])

// Merge a[na] & b[nb] into c[na + nb]
// H.Okumura, "Elementary algorithms in C", 1991.
static inline void fmerge(float *c, float *a, int na, float *b, int nb)
{
	int i, j, k;

	i = j = k = 0;
	while((i < na) && (j < nb))
	{
		if(fabsf(a[i]) >= fabsf(b[j]))
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
//static inline void c_ts_add(float *c, float *a, float *b)
static inline void c_ts_add(float *a, float *b, float *c)
{
	float z[6], e[6];

//	printf("c_ts_add "); fflush(stdout);
	fmerge(z, a, 3, b, 3);
//	printf("merge"); fflush(stdout);
	fvec_sum(e, z, 6);
//	printf(" vec_sum"); fflush(stdout);
	fvseb(c, 3, e, 6);
//	printf(" vseb\n"); fflush(stdout);
}

// c[3] := a[3] + b
static inline void c_ts_add_ts_f(float *a, float b, float *c)
{
	float z[6], e[6]; //, in_b[] = {b};

//	printf("c_ts_add "); fflush(stdout);
	fmerge(z, a, 3, &b, 1);
//	printf("merge"); fflush(stdout);
	fvec_sum(e, z, 4);
//	printf(" vec_sum"); fflush(stdout);
	fvseb(c, 3, e, 4);
//	printf(" vseb\n"); fflush(stdout);
}


// c[3] := -a[3]
//static inline void c_ts_neg(float *c, float *a)
static inline void c_ts_neg(const float *a, float *c)
{
	c[0] = -a[0];
	c[1] = -a[1];
	c[2] = -a[2];
}

// c[3] := a[3] - b[3]
//static inline void c_ts_sub(float *c, float *a, float *b)
static inline void c_ts_sub(float *a, float *b, float *c)
{
	float mb[3];

	c_ts_neg(b, mb);
	c_ts_add(a, mb, c);
}

static inline void c_ts_subq(const float *a, const float *b, float *c)
{
	float mb[3];

	c_ts_neg(b, mb);
	c_ts_addq(a, mb, c);
}

// c[3] := a - b[3]
//static inline void c_ts_sub_d_ts(float *c, float a, float *b)
static inline void c_ts_sub_f_ts(const float a, float *b, float *c)
{
	float tmp_a[3] = {0.0f, 0.0f, 0.0f};
	float mb[3];

	tmp_a[0] = a;

	c_ts_neg(b, mb);
	c_ts_add(tmp_a, mb, c);
}

// c[3] := a[3] - b
//static inline void c_ts_sub_ts_f(float *c, float *a, float b)
static inline void c_ts_sub_ts_f(float *a, float b, float *c)
{
	float mb[3] = {-b, (float)0.0, (float)0.0};

//	c_ts_neg(b, mb);
	c_ts_add(a, mb, c);
}

// Accurate c[3] := a[3] * b[3]
//static inline void c_ts_mul_accurate(float *c, float *a, float *b)
static inline void c_ts_mul_accurate(float *a, float *b, float *c)
{
	float z00[2], z01[2], z10[2];
	float in_b[3], in_c, z[3], e[5], temp[5];

	//printf("c_ts_mul_accurate "); fflush(stdout);

	z00[0] = ftwo_prod(a[0], b[0], &z00[1]);
	z01[0] = ftwo_prod(a[0], b[1], &z01[1]);
	z10[0] = ftwo_prod(a[1], b[0], &z10[1]);
	z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
	fvec_sum(in_b, z, 3);
	in_c = fmaf(a[1], b[1], in_b[2]);

	z[0] = fmaf(a[0], b[2], z10[1]);
	z[1] = fmaf(a[2], b[0], z01[1]);
	z[2] = z[0] + z[1];
	temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; temp[3] = in_c; temp[4] = z[2];
	fvec_sum(e, temp, 5);
	c[0] = e[0];
	fvseb(&c[1], 2, &e[1], 4);
}

// Sloppy c[3] := a[3] * b[3]
//static inline void c_ts_mul_sloppy(float *c, float *a, float *b)
static inline void c_ts_mul_sloppy(float *a, float *b, float *c)
{
	float z00[2], z01[2], z10[2];
	float in_b[3], in_c, z[3], e[4], temp[4];

	//printf("c_ts_mul_sloppy "); fflush(stdout);

	z00[0] = ftwo_prod(a[0], b[0], &z00[1]);
	z01[0] = ftwo_prod(a[0], b[1], &z01[1]);
	z10[0] = ftwo_prod(a[1], b[0], &z10[1]);
	z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
	fvec_sum(in_b, z, 3);
	in_c = fmaf(a[1], b[1], in_b[2]);

	z[0] = fmaf(a[0], b[2], z10[1]);
	z[1] = fmaf(a[2], b[0], z01[1]);
	z[2] = z[0] + z[1];
	temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; temp[3] = in_c + z[2];
	fvec_sum(e, temp, 4);
	c[0] = e[0];
	fvseb(&c[1], 2, &e[1], 3);
}

// define c_ts_mul
#ifndef USE_TS_MUL_ACCURATE
	#define c_ts_mul c_ts_mul_sloppy
#else // USE_TS_MUL_ACCURATE
	#define c_ts_mul c_ts_mul_accurate
#endif // USE_TS_MUL_ACCURATE


// c[3] := a[2] * b[3]
static inline void c_ts_mul_ds_ts_sloppy(float *a, float *b, float *c)
{
	float z00[2], z01[2], z10[2];
	float in_b[3], in_c, z[3], e[4], temp[4];

	z00[0] = ftwo_prod(a[0], b[0], &z00[1]);
	z01[0] = ftwo_prod(a[0], b[1], &z01[1]);
	z10[0] = ftwo_prod(a[1], b[0], &z10[1]);
	z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
	fvec_sum(in_b, z, 3);
	in_c = fmaf(a[1], b[1], in_b[2]);

	z[0] = fmaf(a[0], b[2], z10[1]);
	z[1] = z[0] + z01[1];
	temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; temp[3] = in_c + z[1];
	fvec_sum(e, temp, 4);
	c[0] = e[0];
	fvseb(&c[1], 2, &e[1], 3);
}

// c[3] := a[2] * b[3]
static inline void c_ts_mul_ds_ts_accurate(float *a, float *b, float *c)
{
	float z00[2], z01[2], z10[2];
	float in_b[3], in_c, z[3], e[5], temp[5];

	z00[0] = ftwo_prod(a[0], b[0], &z00[1]);
	z01[0] = ftwo_prod(a[0], b[1], &z01[1]);
	z10[0] = ftwo_prod(a[1], b[0], &z10[1]);
	z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
	fvec_sum(in_b, z, 3);
	in_c = fmaf(a[1], b[1], in_b[2]);

	z[0] = fmaf(a[0], b[2], z10[1]);
	z[1] = z[0] + z01[1];
	temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; temp[3] = in_c; temp[4] = z[1];
	fvec_sum(e, temp, 5);
	c[0] = e[0];
	fvseb(&c[1], 2, &e[1], 4);
}

// define c_ts_mul_ds_ts
#ifndef USE_TS_MUL_DS_TS_ACCURATE
	#define c_ts_mul_ds_ts c_ts_mul_ds_ts_sloppy
//	#define c_ts_mul_ds_ts c_ts_mul_ds_ts_accurate
#else // USE_TS_MUL_DS_TS_ACCURATE
	#define c_ts_mul_ds_ts c_ts_mul_ds_ts_accurate
#endif // USE_TS_MUL_ACCURATE


// c[3] := a * b[3]
static inline void c_ts_mul_s_ts(const float a, const float *b, float *c)
{
	float z00[2], z01[2], z10[2];
	float in_b[3], in_c, z[3], e[4], temp[4];

	z00[0] = ftwo_prod(a, b[0], &z00[1]);
	z01[0] = ftwo_prod(a, b[1], &z01[1]);
	//z10[0] = ftwo_prod(a[1], b[0], &z10[1]);
	z[0] = z00[1]; z[1] = z01[0]; //z[2] = z10[0];
	fvec_sum(in_b, z, 2);
	//in_c = fma(a[1], b[1], in_b[2]);

	z[0] = fmaf(a, b[2], z10[1]);
	z[1] = z[0] + z01[1];
	temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; temp[3] = z[1];
	fvec_sum(e, temp, 4);
	c[0] = e[0];
	fvseb(&c[1], 2, &e[1], 3);
}

// c[3] := a[3] * b
static inline void c_ts_mul_ts_f(const float *a, const float b, float *c)
{
	c_ts_mul_s_ts(b, a, c);
}

// b := |a| 
static inline void c_ts_abs(const float *a, float *b)
{
	// return (a[0] < 0.0) ? -a : a;
	if(a[0] < 0.0)
		c_ts_neg(a, b);
	else
		c_ts_copy(a, b);
}

static inline void c_ts_comp(const float *a, const float *b, int *result) 
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

static inline void c_ts_comp_ts_f(const float *a, float b, int *result)
{
	// qd_real aa(a);
	//if (aa < b)
	if(a[0] < b || (a[0] == b && a[1] < 0.0f))
		*result = -1;
	//else if (aa > b)
	else if (a[0] > b || (a[0] == b && a[1] > 0.0f))
		*result = 1;
	else 
		*result = 0;
}

static inline void c_ts_comp_f_ts(float a, const float *b, int *result)
{
/* qd_real bb(b);
  if (a < bb)
    *result = -1;
  else if (a > bb)
    *result = 1;
  else 
    *result = 0;
*/
	c_ts_comp_ts_f(b, a, result);
	*result = -(*result);
}

// c:= 2 - c_ts_mul_ds_ts(a[2], b[3])
static inline void c_ts_2mtw_ds_ts(float a[DSSIZE], float b[TSSIZE], float c[TSSIZE])
{
	float z00[2], z01[2], z10[2];
	float in_b[3], in_c, z[3], e[4], temp[4];

	z00[0] = ftwo_prod(a[0], b[0], &z00[1]);
	z01[0] = ftwo_prod(a[0], b[1], &z01[1]);
	z10[0] = ftwo_prod(a[1], b[0], &z10[1]);
	z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
	fvec_sum(in_b, z, 3);
	in_c = fmaf(a[1], b[1], in_b[2]);

	z[0] = fmaf(a[0], b[2], z10[1]);
	z[1] = z[0] + z01[1];
	temp[0] = -in_b[0]; temp[1] = -in_b[1]; temp[2] = -in_c; temp[3] = -z[1];
	fvec_sum(e, temp, 4);
	c[0] = 1;
	e[0] = fquick_two_sum(-z00[0], e[0], &temp[0]);
	fvseb(&c[1], 2, e, 4);
 }

//#define ONE_P_2DBL_EPS (1.00000000000000044e+00) // 1 + 2 * DBL_EPSILON
//#define ONE_M_2DBL_EPS (9.99999999999999556e-01) //‬ 1 - 2 * DBL_EPSILON
#define ONE_P_2FLT_EPS (1.00000012e+00) // 1 + 2 * FLT_EPSILON
#define ONE_M_2FLT_EPS (9.99999881e-01) // 1 - 2 * FLT_EPSILON

// c[3] := 1 / a[3]
//static inline void c_ts_reci(float *c, float *a)
static inline void c_ts_reci(float *a, float *c)
{
	float alpha, h1, in_b[2], in_b12, temp[3], d2[3];// = {2.0, 0.0, 0.0}; // d2 = 2

	c_to_ts(d2, 2.0f, 0.0f, 0.0f);

	alpha = ONE_P_2FLT_EPS / a[0];
	h1 =  fmaf(alpha, a[0], -ONE_P_2FLT_EPS); // alpha * a[0] - (1 + 2u)
	h1 = -fmaf(alpha, a[1], h1);
	in_b[0] = ftwo_prod(alpha, ONE_M_2FLT_EPS, &in_b[1]);
	in_b12 = fmaf(alpha, h1, in_b[1]);
	in_b[0] = fquick_two_sum(in_b[0], in_b12, &in_b[1]);
//	c_ts_2mtw_ds_ts(in_b, a, temp); // temp := 2 - c_ts_mul_ds_ts(b, a, temp)
	c_ts_mul_ds_ts(in_b, a, temp); c_ts_sub(d2, temp, temp);
	c_ts_mul_ds_ts(in_b, temp, c);
}

// c[3] := a[3] / b[3]
//static inline void c_ts_div(float *a, float *b, float *c)
static inline void c_ts_divt(float *a, float *b, float *c)
{
	float alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];// = {(float)2.0, (float)0.0, (float)0.0}; // d2 = 2;

	c_to_ts(d2, 2.0f, 0.0f, 0.0f);

	alpha = ONE_P_2FLT_EPS / b[0];
	h1 =  fmaf(alpha, b[0], -ONE_P_2FLT_EPS);
	h1 = -fmaf(alpha, b[1], h1);
	in_b[0] = ftwo_prod(alpha, ONE_M_2FLT_EPS, &in_b[1]);
	in_b12 = fmaf(alpha, h1, in_b[1]);
	in_b[0] = fquick_two_sum(in_b[0], in_b12, &in_b[1]);
//	c_ts_2mtw_ds_ts(in_b, b, temp); // temp := 2 - c_ts_mul_ds_ts(in_b, b, temp)
	c_ts_mul_ds_ts(in_b, b, temp); c_ts_sub(d2, temp, temp);
	c_ts_mul_ds_ts(in_b, a, in_c);
	c_ts_mul(in_c, temp, c);
}

#ifdef BNC_USE_FMA_DIV
	#define c_ts_div bnc_ts_div_fma
#else
	#define c_ts_div c_ts_divtq
#endif // BNC_USE_FMA_DIV

// c[3] := a[3] / b[3]
//static inline void c_ts_div(float *a, float *b, float *c)
static inline void c_ts_divtq(float *a, float *b, float *c)
{
	float alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];// = {(float)2.0, (float)0.0, (float)0.0}; // d2 = 2;

	c_to_ts(d2, 2.0f, 0.0f, 0.0f);

	alpha = ONE_P_2FLT_EPS / b[0];
	h1 =  fmaf(alpha, b[0], -ONE_P_2FLT_EPS);
	h1 = -fmaf(alpha, b[1], h1);
	in_b[0] = ftwo_prod(alpha, ONE_M_2FLT_EPS, &in_b[1]);
	in_b12 = fmaf(alpha, h1, in_b[1]);
	in_b[0] = fquick_two_sum(in_b[0], in_b12, &in_b[1]);
//	c_ts_2mtw_ds_ts(in_b, b, temp); // temp := 2 - c_ts_mul_ds_ts(in_b, b, temp)
	c_ts_mul_ds_ts(in_b, b, temp);
	//c_ts_sub(d2, temp, temp);
	c_ts_subq(d2, temp, temp);
	c_ts_mul_ds_ts(in_b, a, in_c);
	c_ts_mul(in_c, temp, c);
}

/* triple-float / triple-float */
// c := a / b
static inline void c_ts_divq(const float *a, const float *b, float *c)
{
	//qd_real qd_real::sloppy_div(const qd_real &a, const qd_real &b) {
	float q0, q1, q2, q3;
	float r[TSSIZE], tmp[TSSIZE];
	//qd_real r;

	q0 = a[0] / b[0];
	//r = a - (b * q0);
	c_ts_mul_ts_f(b, q0, tmp);
	c_ts_subq(a, tmp, r);

	q1 = r[0] / b[0];
	//r -= (b * q1);
	c_ts_mul_ts_f(b, q1, tmp);
	c_ts_subq(r, tmp, r);

	q2 = r[0] / b[0];
	//r -= (b * q2);
	c_ts_mul_ts_f(b, q2, tmp);
	c_ts_subq(r, tmp, r);

	q3 = r[0] / b[0];

	frenorm(&q0, &q1, &q2, &q3);

	//return qd_real(q0, q1, q2, q3);
	c[0] = q0;
	c[1] = q1;
	c[2] = q2;
	c[3] = q3;
}

/* float / triple-float */
// c := (float)a / b
static inline void c_ts_div_f_ts(float a, float *b, float *c)
{
	float aa[TSSIZE];

	c_ts_copy_f(a, aa);
	c_ts_div(aa, b, c);
}
/* triple-float / float */
// c := a / (float)b
static inline void c_ts_div_ts_f(float *a, float b, float *c)
{
	float bb[TSSIZE];

	c_ts_copy_f(b, bb);
	c_ts_div(a, bb, c);
}

// c[3] := sqrt(a[3])
static inline void c_ts_sqrt(float *a, float *c)
{
	float tmp_a[4], tmp_c[4];

	c_qs_copy_ts(a, tmp_a);

	c_qs_sqrt(tmp_a, tmp_c);
	c_ts_copy_qs(tmp_c, c);
}

// c[3] := sqrt(a[3])
static inline void c_ts_sqrt_f(float a, float *c)
{
	float tmp_a[4], tmp_c[4];

	c_qs_copy_f(a, tmp_a);

	c_qs_sqrt(tmp_a, tmp_c);
	c_ts_copy_qs(tmp_c, c);	
}



// c[3] := a[3]^2
static inline void c_ts_sqr(float *a, float *c)
{
#ifdef USE_ACCURATE_TS_MUL
	c_ts_mul_acculate(a, a, c);
#else // USE_ACCURATE_TS_MUL
	c_ts_mul_sloppy(a, a, c);
#endif // USE_ACCURATE_TS_MUL
}

static inline void c_ts_read(const char *s, float *a) {
#if 0
  qd_real aa(s);
  TO_DOUBLE_PTR(aa, a);
#endif // 0
}

static inline void c_ts_swrite(const float *a, int precision, char *s, int len) {
#if 0
  qd_real(a).write(s, len, precision);
#endif // 0
}

static inline void c_ts_write(const float *a) {
#if 0
  std::cout << qd_real(a).to_string(qd_real::_ndigits) << std::endl;
#endif // 0
}

// PI = 3.1415...
static inline void c_ts_pi(float *a)
{
	c_ts_copy(const_ts_pi, a);
}


#ifdef __cplusplus
}
#endif

/* ------------------------------------------------------------------ */
/* dtq-0.0.3 float-based elementary functions (DS/TS/QS delegate to   */
/* DD/TD/QD).  Provides the definitions for the bnc_ds_* / bnc_qs_*   */
/* forward declarations used by c_ds_exp, c_qs_exp, ... above.        */
/* ------------------------------------------------------------------ */
#include "bncelem_f.h"

#ifdef __cplusplus
extern "C" {
#endif

/* TS elementary functions in the c_ts_* namespace (previously missing). */
static inline void c_ts_exp(const float *a, float *ret)
{
	bnc_ts_exp(a, ret);
}

static inline void c_ts_log(const float *a, float *ret)
{
	bnc_ts_log(a, ret);
}

static inline void c_ts_log10(const float *a, float *ret)
{
	bnc_ts_log10(a, ret);
}

static inline void c_ts_sin(const float *a, float *ret)
{
	bnc_ts_sin(a, ret);
}

static inline void c_ts_cos(const float *a, float *ret)
{
	bnc_ts_cos(a, ret);
}

static inline void c_ts_sincos(const float *a, float *s, float *c)
{
	bnc_ts_sincos(a, s, c);
}

#ifdef __cplusplus
}
#endif

#endif // __C_DS_QS_H
