/*
 * include/qd_inline.h
 *
 * This work was supported by the Director, Office of Science, Division
 * of Mathematical, Information, and Computational Sciences of the
 * U.S. Department of Energy under contract number DE-AC03-76SF00098.
 *
 * Copyright (c) 2000-2001
 *
 * Contains small functions (suitable for inlining) in the quad-double
 * arithmetic package.
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

/* copy */

/* (qd)b := (qd)a */
void c_qd_copy(const double *a, double *b)
{
	b[0] = a[0];
	b[1] = a[1];
	b[2] = a[2];
	b[3] = a[3];
}

/* (qd)b := (dd)a */
void c_qd_copy_dd(const double *a, double *b)
{
	b[0] = a[0];
	b[1] = a[1];
	b[2] = 0.0;
	b[3] = 0.0;
}

/* (qd)b := (double)a */
void c_qd_copy_d(double a, double *b)
{
	b[0] = a;
	b[1] = 0.0;
	b[2] = 0.0;
	b[3] = 0.0;
}

/********** Renormalization **********/
void quick_renorm(double *c0, double *c1, double *c2, double *c3, double *c4)
{
	double t0, t1, t2, t3;
	double s;

	s  = quick_two_sum(*c3, *c4, &t3);
	s  = quick_two_sum(*c2, s , &t2);
	s  = quick_two_sum(*c1, s , &t1);
	*c0 = quick_two_sum(*c0, s , &t0);

	s  = quick_two_sum(t2, t3, &t2);
	s  = quick_two_sum(t1, s , &t1);
	*c1 = quick_two_sum(t0, s , &t0);

	s  = quick_two_sum(t1, t2, &t1);
	*c2 = quick_two_sum(t0, s , &t0);

	*c3 = t0 + t1;
}

void renorm(double *c0, double *c1, double *c2, double *c3)
{
	double s0, s1, s2 = 0.0, s3 = 0.0;

	if (QD_ISINF(c0)) return;

	s0 = quick_two_sum(*c2, *c3, c3);
	s0 = quick_two_sum(*c1, s0, c2);
	*c0 = quick_two_sum(*c0, s0, c1);

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

void renorm4(double *c0, double *c1, double *c2, double *c3, double *c4)
{
	double s0, s1, s2 = 0.0, s3 = 0.0;

	if (QD_ISINF(c0)) return;

	s0 = quick_two_sum(*c3, *c4, c4);
	s0 = quick_two_sum(*c2, s0, c3);
	s0 = quick_two_sum(*c1, s0, c2);
	*c0 = quick_two_sum(*c0, s0, c1);

	s0 = *c0;
	s1 = *c1;

	s0 = quick_two_sum(*c0, *c1, &s1);
	if (s1 != 0.0)
	{
		s1 = quick_two_sum(s1, *c2, &s2);
		if (s2 != 0.0)
		{
			s2 = quick_two_sum(s2, *c3, &s3);
			if (s3 != 0.0)
				s3 += *c4;
			else
				s2 += *c4;
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
void three_sum(double *a, double *b, double *c)
{
	double t1, t2, t3;

	t1 = two_sum(*a, *b, &t2);
	*a  = two_sum(*c, t1, &t3);
	*b  = two_sum(t2, t3, c);
}

void three_sum2(double *a, double *b, double *c)
{
	double t1, t2, t3;

	t1 = two_sum(*a, *b, &t2);
	*a  = two_sum(*c, t1, &t3);
	*b = t2 + t3;
}

/* quad-double + double-double */
void c_qd_add_qd_dd(const double *a, const double *b, double *c)
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
void c_qd_add_dd_qd(const double *a, const double *b, double *c)
{
	c_qd_add_qd_dd(b, a, c);

#if 0 //
  qd_real cc;
  cc = dd_real(a) + qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/* quad-double + double */
void c_qd_add_qd_d(const double *a, double b, double *c)
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
void c_qd_add_d_qd(double a, const double *b, double *c)
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
double quick_three_accum(double *a, double *b, double c)
{
	double s;
//	bool za, zb;

	s = two_sum(*b, c, b);
	s = two_sum(*a, s, a);

//	za = (*a != 0.0);
//	zb = (*b != 0.0);

//  if (za && zb)
	if((*a != 0.0) && (*b != 00))
		return s;

//	if (!zb) {
	if(*b != 0.0)
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
void c_qd_add(const double *a, const double *b, double *c)
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
		if (i >= 4 && j >= 4) {
			x[k] = u;
			if (k < 3)
				x[++k] = v;
			break;
		}

		if (i >= 4)
			t = b[j++];
		else if (j >= 4)
			t = a[i++];
		else if (fabs(a[i]) > fabs(b[j])) {
			t = a[i++];
		} else
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
void c_qd_add_sloppy(const double *a, const double *b, double *c)
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


/********** Self-Additions ************/
// b := b + a
void c_qd_selfadd(const double *a, double *b)
{
	static double bb[QDSIZE];

	c_qd_copy(b, bb);
	c_qd_add(a, bb, b);

#if 0 //
	qd_real bb(b);
	bb += qd_real(a);
	TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b + (dd)a
void c_qd_selfadd_dd(const double *a, double *b)
{
	static double bb[QDSIZE];

	c_qd_copy(b, bb);
	c_qd_add_dd_qd(a, bb, b);

#if 0 //
  qd_real bb(b);
  bb += dd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b + (double)a
void c_qd_selfadd_d(double a, double *b)
{
	static double bb[QDSIZE];

	c_qd_copy(b, bb);
	c_qd_add_d_qd(a, bb, b);

#if 0 //
  qd_real bb(b);
  bb += a;
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}


/********** Unary Minus **********/
void c_qd_neg(const double *a, double *b) {
	b[0] = -a[0];
	b[1] = -a[1];
	b[2] = -a[2];
	b[3] = -a[3];
}

// b := (dd)(-a)
void c_qd_neg_dd(const double *a, double *b) {
	b[0] = -a[0];
	b[1] = -a[1];
	b[2] = 0.0;
	b[3] = 0.0;
}

// b := (double)(-a)
void c_qd_neg_d(const double a, double *b) 
{
	b[0] = -a;
	b[1] = 0.0;
	b[2] = 0.0;
	b[3] = 0.0;
}

/********** Subtractions **********/
/* sub */
// c := a - b
void c_qd_sub(const double *a, const double *b, double *c)
{
	static double mb[QDSIZE];

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
void c_qd_sub_qd_dd(const double *a, const double *b, double *c)
{
	static double mb[QDSIZE];

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
void c_qd_sub_dd_qd(const double *a, const double *b, double *c)
{
	static double ma[QDSIZE];

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
void c_qd_sub_qd_d(const double *a, double b, double *c)
{
	static double mb[QDSIZE];

	// a + (-b)
	c_qd_neg_d(b, mb);
	c_qd_add(a, mb, c);

#if 0
  qd_real cc;
  cc = qd_real(a) - b;
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// c := (double)a - b
void c_qd_sub_d_qd(double a, const double *b, double *c)
{
	static double ma[QDSIZE];

	// (-a) + b
	c_qd_neg_d(a, ma);
	c_qd_add(ma, b, c);

#if 0
  qd_real cc;
  cc = a - qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

/********** Self-Subtractions **********/
/* selfsub */
void c_qd_selfsub(const double *a, double *b)
{
	static double ma[QDSIZE];

	// (-a) + b
	c_qd_neg(a, ma);
	c_qd_sub(ma, b, b);

#if 0
  qd_real bb(b);
  bb -= qd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b - (dd)a
void c_qd_selfsub_dd(const double *a, double *b)
{
	static double ma[QDSIZE];

	// (-a) + b
	c_qd_neg_dd(a, ma);
	c_qd_sub(ma, b, b);

#if 0
  qd_real bb(b);
  bb -= dd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b - (double)a
void c_qd_selfsub_d(double a, double *b)
{
	static double ma[QDSIZE];

	// (-a) + b
	c_qd_neg_d(a, ma);
	c_qd_sub(ma, b, b);

#if 0
  qd_real bb(b);
  bb -= a;
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

/********** Multiplications **********/
void c_qd_mul(const double *a, const double *b, double *c)
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
void c_qd_mul_qd_dd(const double *a, const double *b, double *c)
{
	double p0, p1, p2, p3, p4;
	double q0, q1, q2, q3, q4;
	double s0, s1, s2;
	double t0, t1;

	p0 = two_prod(a[0], DD_HI(b), &q0);
	p1 = two_prod(a[0], DD_LOW(b), &q1);
	p2 = two_prod(a[1], DD_HI(b), &q2);
	p3 = two_prod(a[1], DD_LOW(b), &q3);
	p4 = two_prod(a[2], DD_HI(b), &q4);

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
void c_qd_mul_dd_qd(const double *a, const double *b, double *c)
{
	c_qd_mul_qd_dd(b, a, c);

#if 0
  qd_real cc;
  cc = dd_real(a) * qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// c := a * (double)b
void c_qd_mul_qd_d(const double *a, double b, double *c)
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
void c_qd_mul_d_qd(double a, const double *b, double *c)
{
	c_qd_mul_qd_d(b, a, c);

#if 0
  qd_real cc;
  cc = a * qd_real(b);
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
void c_qd_mul_sloppy(const double *a, const double *b, double *c)
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

/* quad-double ^ 2  = (x0 + x1 + x2 + x3) ^ 2
                    = x0 ^ 2 + 2 x0 * x1 + (2 x0 * x2 + x1 ^ 2)
                               + (2 x0 * x3 + 2 x1 * x2)           */
void c_qd_sqr(const double *a, double *c)
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
void c_qd_selfmul(const double *a, double *b)
{
	static double bb[QDSIZE];

	c_qd_copy(b, bb);
	c_qd_mul(a, bb, b);

#if 0
  qd_real bb(b);
  bb *= qd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b * (dd)a
void c_qd_selfmul_dd(const double *a, double *b)
{
	static double bb[QDSIZE];

	c_qd_copy(b, bb);
	c_qd_mul_qd_dd(bb, a, b);

#if 0
  qd_real bb(b);
  bb *= dd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b * (doule)a
void c_qd_selfmul_d(double a, double *b)
{
	static double bb[QDSIZE];

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
void c_qd_div(const double *a, const double *b, double *c)
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

	q2 = r[0] / b[0];
	//r -= (b * q2);
	c_qd_mul_qd_d(b, q2, tmp);
	c_qd_selfsub(tmp, r);

	q3 = r[0] / b[0];

	//r -= (b * q3);
	c_qd_mul_qd_d(b, q3, tmp);
	c_qd_selfsub(tmp, r);

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
void c_qd_div_sloppy(const double *a, const double *b, double *c)
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

	q2 = r[0] / b[0];
	//r -= (b * q2);
	c_qd_mul_qd_d(b, q2, tmp);
	c_qd_selfsub(tmp, r);

	q3 = r[0] / b[0];

	renorm(&q0, &q1, &q2, &q3);

	//return qd_real(q0, q1, q2, q3);
	c[0] = q0;
	c[1] = q1;
	c[2] = q2;
	c[3] = q3;
}

/* quad-double / double-double */
// c := a / (dd)b
void c_qd_div_qd_dd(const double *a, const double *b, double *c)
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
void c_qd_div_dd_qd(const double *a, const double *b, double *c)
{
	static double aa[QDSIZE];

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
void c_qd_div_qd_d(const double *a, double b, double *c)
{
	static double bb[QDSIZE];

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
void c_qd_div_d_qd(double a, const double *b, double *c)
{
	static double aa[QDSIZE];

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
void c_qd_selfdiv(const double *a, double *b) 
{
	static double bb[QDSIZE];

	c_qd_copy(b, bb);
	c_qd_div(bb, a, b);

#if 0
  qd_real bb(b);
  bb /= qd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := b / (dd)a
void c_qd_selfdiv_dd(const double *a, double *b)
{
	static double bb[QDSIZE];

	c_qd_copy(b, bb);
	c_qd_div_qd_dd(bb, a, b);

#if 0
  qd_real bb(b);
  bb /= dd_real(a);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
void c_qd_selfdiv_d(double a, double *b)
{
	static double bb[QDSIZE];

	c_qd_copy(b, bb);
	c_qd_div_qd_d(bb, a, b);

#if 0
  qd_real bb(b);
  bb /= a;
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// c := (a[0] * b, a[1] * b, a[2] * b, a[3] * b)
void c_qd_mul_pwr2(const double *a, double b, double *c)
{
	//return qd_real(a[0] * b, a[1] * b, a[2] * b, a[3] * b);
	c[0] = a[0] * b;
	c[1] = a[1] * b;
	c[2] = a[2] * b;
	c[3] = a[3] * b;
}

void c_qd_set0(double *qdval)
{
	qdval[0] = 0.0;
	qdval[1] = 0.0;
	qdval[2] = 0.0;
	qdval[3] = 0.0;
}

void c_qd_sqrt(const double *a, double *b)
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
  static double r[QDSIZE], h[QDSIZE], tmp[QDSIZE];

	if (QD_ISZERO(a))
	{
		c_qd_set0(b);
		return;
	}

	if (QD_ISNEGATIVE(a))
	{
		fprintf(stderr, "(qd_real::sqrt): Negative argument.");
		b[0] = QD_NAN;
		return;
	}

//  qd_real r = (1.0 / std::sqrt(a[0]));
//  qd_real h = mul_pwr2(a, 0.5);
	r[0] = 1.0 / sqrt(a[0]); r[1] = 0.0; r[2] = 0.0; r[3] = 0.0;
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
	c_qd_selfmul(a, r);
	c_qd_copy(r, b);
//  return r;
#if 0
  qd_real bb;
  bb = sqrt(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// b := |a| 
void c_qd_abs(const double *a, double *b)
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

void c_qd_npwr(const double *a, int n, double *b) {
#if 0
  qd_real bb;
  bb = npwr(qd_real(a), n);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

void c_qd_nroot(const double *a, int n, double *b) {
#if 0
  qd_real bb;
  bb = nroot(qd_real(a), n);
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

// rount to nearest integer
#define nint(a) (round(a))

void c_qd_nint(const double *a, double *b)
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
void c_qd_aint(const double *a, double *b)
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

void c_qd_floor(const double *a, double *b)
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
void c_qd_ceil(const double *a, double *b)
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

void c_qd_log(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = log(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
void c_qd_log10(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = log10(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
void c_qd_exp(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = exp(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

void c_qd_sin(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = sin(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
void c_qd_cos(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = cos(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
void c_qd_tan(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = tan(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

void c_qd_asin(const double *a, double *b) {
#if 0
 qd_real bb;
  bb = asin(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
void c_qd_acos(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = acos(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
void c_qd_atan(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = atan(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

void c_qd_atan2(const double *a, const double *b, double *c) {
#if 0
  qd_real cc;
  cc = atan2(qd_real(a), qd_real(b));
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

void c_qd_sinh(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = sinh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
void c_qd_cosh(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = cosh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
void c_qd_tanh(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = tanh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

void c_qd_asinh(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = asinh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
void c_qd_acosh(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = acosh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}
void c_qd_atanh(const double *a, double *b) {
#if 0
  qd_real bb;
  bb = atanh(qd_real(a));
  TO_DOUBLE_PTR(bb, b);
#endif // 0
}

void c_qd_sincos(const double *a, double *s, double *c) {
#if 0
  qd_real ss, cc;
  sincos(qd_real(a), ss, cc);
  TO_DOUBLE_PTR(cc, c);
  TO_DOUBLE_PTR(ss, s);
#endif // 0
}

void c_qd_sincosh(const double *a, double *s, double *c) {
#if 0
  qd_real ss, cc;
  sincosh(qd_real(a), ss, cc);
  TO_DOUBLE_PTR(cc, c);
  TO_DOUBLE_PTR(ss, s);
#endif // 0
}

void c_qd_read(const char *s, double *a) {
#if 0
  qd_real aa(s);
  TO_DOUBLE_PTR(aa, a);
#endif // 0
}

void c_qd_swrite(const double *a, int precision, char *s, int len) {
#if 0
  qd_real(a).write(s, len, precision);
#endif // 0
}

void c_qd_write(const double *a) {
#if 0
  std::cout << qd_real(a).to_string(qd_real::_ndigits) << std::endl;
#endif // 0
}

void c_qd_rand(double *a) {
#if 0
  qd_real aa;
  aa = qdrand();
  TO_DOUBLE_PTR(aa, a);
#endif // 0
}

void c_qd_comp(const double *a, const double *b, int *result) 
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

void c_qd_comp_qd_d(const double *a, double b, int *result)
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

void c_qd_comp_d_qd(double a, const double *b, int *result)
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

void c_qd_pi(double *a)
{
	c_qd_copy(const_qd_pi, a);

#if 0
  TO_DOUBLE_PTR(qd_real::_pi, a);
#endif // 0
}

#ifdef __cplusplus
}
#endif
