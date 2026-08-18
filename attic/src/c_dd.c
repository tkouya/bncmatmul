/*
 * src/c_dd.cc
 *
 * This work was supported by the Director, Office of Science, Division
 * of Mathematical, Information, and Computational Sciences of the
 * U.S. Department of Energy under contract number DE-AC03-76SF00098.
 *
 * Copyright (c) 2000-2001
 *
 * Contains the C wrapper functions for double-double precision arithmetic.
 * This can be used from Fortran code.
 */

/* Translated to pure C code by T.Kouya */
/* 2015-03-01 */
#include "c_dd_qd.h"
#include "ddlinear.h" // Basic Linear Computations with DD and QD

//static const double _d_nan = std::numeric_limits<double>::quiet_NaN();
//static const double _d_inf = std::numeric_limits<double>::infinity();

#ifdef __cplusplus
extern "C" {
#endif

// copied from inline.h
//#ifdef NATIVE_C

/*********** Basic Functions ************/
/* Computes fl(a+b) and err(a+b).  Assumes |a| >= |b|. */
double quick_two_sum(double a, double b, double *err)
{
	double s = a + b;
	*err = b - (s - a);
	return s;
}

/* Computes fl(a-b) and err(a-b).  Assumes |a| >= |b| */
double quick_two_diff(double a, double b, double *err)
{
	double s = a - b;
	*err = (a - s) - b;
	return s;
}

/* Computes fl(a+b) and err(a+b).  */
double two_sum(double a, double b, double *err)
{
	double s = a + b;
	double bb = s - a;
	*err = (a - (s - bb)) + (b - bb);
	return s;
}

/* Computes fl(a-b) and err(a-b).  */
double two_diff(double a, double b, double *err)
{
	double s = a - b;
	double bb = s - a;
	*err = (a - (s - bb)) - (b + bb);
	return s;
}

#ifndef QD_FMS
/* Computes high word and lo word of a */
void split(double a, double *hi, double *lo)
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
#endif

/* Computes fl(a*b) and err(a*b). */
double two_prod(double a, double b, double *err)
{
#ifdef QD_FMS
	double p = a * b;

	*err = QD_FMS(a, b, p);

	return p;
#else
	double a_hi, a_lo, b_hi, b_lo;
	double p = a * b;

	split(a, &a_hi, &a_lo);
	split(b, &b_hi, &b_lo);
	*err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;

	return p;
#endif
}

/* Computes fl(a*a) and err(a*a).  Faster than the above method. */
double two_sqr(double a, double *err)
{
#ifdef QD_FMS
	double p = a * a;

	*err = QD_FMS(a, a, p);

	return p;
#else
	double hi, lo;
	double q = a * a;

	split(a, &hi, &lo);
	*err = ((hi * hi - q) + 2.0 * hi * lo) + lo * lo;

	return q;
#endif
}

/* Computes the nearest integer to d. */
double nint(double d)
{
//  if (d == std::floor(d))
	if (d == floor(d))
		return d;
	return floor(d + 0.5);
}

/* Computes the truncated integer. */
double aint(double d)
{
//  return (d >= 0.0) ? std::floor(d) : std::ceil(d);
	return (d >= 0.0) ? floor(d) : ceil(d);
}

/* These are provided to give consistent 
   interface for double with double-double and quad-double. */
void sincosh(double t, double *sinh_t, double *cosh_t)
{
	*sinh_t = sinh(t);
	*cosh_t = cosh(t);
}

double sqr(double t)
{
	return t * t;
}

double to_double(double a) { return a; }
//int    to_int(double a) { return static_cast<int>(a); }
int    to_int(double a) { return (int)(a); }

//#ifdef NATIVE_C
/* double-double = double + double */
void c_d_add(double a, double b, double *c)
{
	double s, e;

	s = two_sum(a, b, &e);
//	return dd_real(s, e);
	c[0] = s;
	c[1] = e;
}
//#endif // NATIVE_C

/* add */
void c_dd_add(const double *a, const double *b, double *c)
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
void c_dd_add_sloppy(const double *a, const double *b, double *c, double *err)
{
//#ifdef NATIVE_C
  /* This is the less accurate version ... obeys Cray-style
     error bound. */
	double s, e;

	s = two_sum(a[0], b[0], &e);
	e += (a[1] + b[1]);
	s = quick_two_sum(s, e, &e);
//  return dd_real(s, e);
	*err = e;
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

void c_dd_add_dd_d(const double *a, double b, double *c)
{
//#ifdef NATIVE_C
	double cc[DDSIZE];
	cc[0] = a[0] + b;
	cc[1] = a[1];
#if 0 //#else // NATIVE_C
  dd_real cc;
  cc = dd_real(a) + b;
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}
void c_dd_add_d_dd(double a, const double *b, double *c)
{
//#ifdef NATIVE_C
	double cc[DDSIZE];
	cc[0] = a + b[0];
	cc[1] = b[1];
#if 0 // #else // NATIVE_C
  dd_real cc;
  cc = a + dd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

/*********** Subtractions ************/
/* double-double = double - double */
void c_d_sub(double a, double b, double *c)
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
void c_dd_sub(const double *a, const double *b, double *c)
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

void c_dd_sub_sloppy(const double *a, const double *b, double *c)
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
void c_dd_sub_dd_d(const double *a, double b, double *c)
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
void c_dd_sub_d_dd(double a, const double *b, double *c)
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
void  c_d_mul(double a, double b, double *c)
{
	double p, e;
	p = two_prod(a, b, &e);
//	return dd_real(p, e);
	c[0] = p;
	c[1] = e;
}

/* mul */
void c_dd_mul(const double *a, const double *b, double *c)
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

void c_dd_mul_dd_d(const double *a, double b, double *c)
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
void c_dd_mul_d_dd(double a, const double *b, double *c)
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
void c_d_div(double a, double b, double *c)
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

/* div */
void c_dd_div(const double *a, const double *b, double *c)
{
	double q1, q2, q3;
	double r[DDSIZE];

	q1 = a[0] / b[0];  /* approximate quotient */

//  r = a - q1 * b;
	c_dd_mul_dd_d(b, q1, r);
	c_dd_sub(a, r, r);

	q2 = r[0] / b[0];

//  r -= (q2 * b);
	c_dd_mul_dd_d(b, q2, c);
	c_dd_sub(r, c, r);

//  q3 = r[0] / b[0];
	q3 = c[0] / b[0];

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
void c_dd_sloppy_div(double *a, double *b, double *c)
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

/* double-double / double */
void c_dd_div_dd_d(const double *a, double b, double *c)
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
void c_dd_div_d_dd(double a, const double *b, double *c)
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
void c_dd_copy(const double *a, double *b) {
  b[0] = a[0];
  b[1] = a[1];
}
void c_dd_copy_d(double a, double *b) {
  b[0] = a;
  b[1] = 0.0;
}


/* b := a^2 */
void c_d_sqr(double a, double *b)
{
	double p1, p2;
	p1 = two_sqr(a, &p2);
	// return dd_real(p1, p2);
	b[0] = p1;
	b[1] = p2;
}

/* b := a^2 */
void c_dd_sqrt(const double *a, double *b)
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

	double x = 1.0 / sqrt(a[0]);
	double ax = a[0] * x;
	double tmp[DDSIZE];

//  return dd_real::add(ax, (a - dd_real::sqr(ax)).x[0] * (x * 0.5));
	c_dd_sqr_d(ax, tmp);
	c_dd_sub(a, tmp, tmp);

//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = sqrt(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

// b := a^2
void c_dd_sqr_d(double a, double *ret)
{
	double p1, p2;
	p1 = two_sqr(a, &p2);
//	return dd_real(p1, p2);
	ret[0] = p1;
	ret[1] = p2;
}

// b := a^2
void c_dd_sqr(const double *a, double *b)
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

void c_dd_abs(const double *a, double *b)
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

void c_dd_npwr(const double *a, int n, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = npwr(dd_real(a), n);
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

void c_dd_nroot(const double *a, int n, double *b) {
// #ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = nroot(dd_real(a), n);
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

void c_dd_nint(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = nint(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
void c_dd_aint(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = aint(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
void c_dd_floor(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = floor(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
void c_dd_ceil(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = ceil(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

void c_dd_log(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = log(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

void c_dd_log10(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = log10(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

void c_dd_exp(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = exp(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

void c_dd_sin(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = sin(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

void c_dd_cos(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = cos(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

void c_dd_tan(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = tan(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

void c_dd_asin(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = asin(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

void c_dd_acos(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = acos(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

void c_dd_atan(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = atan(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

void c_dd_atan2(const double *a, const double *b, double *c) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real cc;
  cc = atan2(dd_real(a), dd_real(b));
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

void c_dd_sinh(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = sinh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
void c_dd_cosh(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = cosh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
void c_dd_tanh(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = tanh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

void c_dd_asinh(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = asinh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
void c_dd_acosh(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = acosh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}
void c_dd_atanh(const double *a, double *b) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real bb;
  bb = atanh(dd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // NATIVE_C
}

void c_dd_sincos(const double *a, double *s, double *c) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real ss, cc;
  sincos(dd_real(a), ss, cc);
  TO_DOUBLE_PTR(ss, s);
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

void c_dd_sincosh(const double *a, double *s, double *c) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real ss, cc;
  sincosh(dd_real(a), ss, cc);
  TO_DOUBLE_PTR(ss, s);
  TO_DOUBLE_PTR(cc, c);
#endif // NATIVE_C
}

void c_dd_read(const char *s, double *a) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real aa(s);
  TO_DOUBLE_PTR(aa, a);
#endif // NATIVE_C
}

void c_dd_swrite(const double *a, int precision, char *s, int len) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  dd_real(a).write(s, len, precision);
#endif // NATIVE_C
}

void c_dd_write(const double *a) {
//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  std::cout << dd_real(a).to_string(dd_real::_ndigits) << std::endl;
#endif // NATIVE_C
}

void c_dd_neg(const double *a, double *b)
{
	b[0] = -a[0];
	b[1] = -a[1];
}

void c_dd_rand(double *a)
{
	static const double m_const = 4.6566128730773926e-10;  /* = 2^{-31} */
	int i;
	double m = m_const;
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

void c_dd_comp(const double *a, const double *b, int *result)
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

void c_dd_comp_dd_d(const double *a, double b, int *result)
{
	//	dd_real aa(a), bb(b);
	double bb[DDSIZE];

	c_dd_copy_d(b, bb);

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

void c_dd_comp_d_dd(double a, const double *b, int *result)
{
	double aa[DDSIZE];

	c_dd_copy_d(a, aa);

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

void c_dd_pi(double *a)
{
	c_dd_copy(const_dd_pi, a);

//#ifdef NATIVE_C
#if 0 // #else // NATIVE_C
  TO_DOUBLE_PTR(dd_real::_pi, a);
#endif // NATIVE_C
}

/*********** Micellaneous ************/
/*  this == 0 */
dd_bool dd_is_zero(const double *x)
{
//  return (x[0] == 0.0);
	if(x[0] == 0.0)
		return DD_TRUE;
	else
		return DD_FALSE;
}

/*  this == 1 */
dd_bool dd_is_one(const double *x)
{
	//return (x[0] == 1.0 && x[1] == 0.0);
	if(x[0] == 1.0 && x[1] == 0.0)
		return DD_TRUE;
	else
		return DD_FALSE;
}

/*  this > 0 */
dd_bool dd_is_positive(const double *x)
{
//  return (x[0] > 0.0);
	if(x[0] > 0.0)
		return DD_TRUE;
	else
		return DD_FALSE;
}

/* this < 0 */
dd_bool dd_is_negative(const double *x)
{
//  return (x[0] < 0.0);
	if(x[0] < 0.0)
		return DD_TRUE;
	else
		return DD_FALSE;
}


#ifdef __cplusplus
}
#endif
