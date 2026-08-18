/********************************************************************************/
/* bncfma.h : Branch-free fused multiply-add (FMA) for DW/TW/QW arithmetic       */
/*            z := x * y + c                                                    */
/*                                                                              */
/* Implementation of Algorithms 1-3 of                                          */
/*   T. Kouya, "Performance evaluation of branch-free fused multiply-add        */
/*   algorithms for multi-component-type multiple-precision floating-point      */
/*   arithmetic", arXiv:2607.11391v1 (2026).                                    */
/* Operation-by-operation port of the reference implementation fma_ref.c        */
/* (Appendix of the above), so every routine matches the FPANVerifier-certified */
/* netlists dwfma_f2s.fpan / twfma_fix2.fpan / qwfma_fix3.fpan.                 */
/*                                                                              */
/* Copyright (C) 2026 Tomonori Kouya                                            */
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
 * IMPORTANT: compile with -ffp-contract=off.  If the compiler is allowed to
 * contract a*b+c into an fma on its own, two_sum / two_prod stop being
 * error-free transformations and the verified netlist no longer holds.
 * (BNCmatmul builds already pass -ffp-contract=off everywhere.)
 *
 * Certified error bounds (u = 2^-p, FPANVerifier, all precisions p at once):
 *   DW: |z - (xy+c)| <=  34 u^2 (|xy| + |c|)   17 flops
 *   TW: |z - (xy+c)| <= 184 u^3 (|xy| + |c|)   66 flops
 *   QW: |z - (xy+c)| <= 812 u^4 (|xy| + |c|)  146 flops
 * (a fused hardware FMA counts as 1 flop; two_sum = 6, fast_two_sum = 3,
 *  two_prod = 2)
 *
 * Naming: the paper's fast_two_sum() is BNCmatmul's quick_two_sum().
 *   double-based DW/TW/QW == BNCmatmul DD/TD/QD
 *   single-based DW/TW/QW == BNCmatmul DS/TS/QS  (bnc_*fmaf)
 */
#ifndef __BNC_FMA_D_H
#define __BNC_FMA_D_H

#include <math.h>

/* two_sum / quick_two_sum / two_prod (double) come from c_dd_qd.h
   (header-only static inline). */
#include "rdd.h"       /* c_dd_qd.h was absorbed into rdd.h */

/* ==============================================================================
 *  DW-FMA : z = x*y + c   (K = 2, 17 flops, Algorithm 1)
 *  matches dwfma_f2s.fpan (trailing FastTwoSum, 20 -> 17 flops)
 * ============================================================================== */
static inline void bnc_dwfma(double z[2], const double x[2], const double y[2], const double c[2])
{
	double P00, E00, P01, P10, l, v, w, s, t, tp;

	P00 = two_prod(x[0], y[0], &E00);   /* 2 */
	P01 = x[0] * y[1];                  /* 1  (E01 discarded) */
	P10 = x[1] * y[0];                  /* 1  (E10 discarded, x[1]*y[1] truncated at O(u^2)) */
	l   = P01 + P10;                    /* 1 */
	v   = E00 + c[1];                   /* 1 */
	w   = v + l;                        /* 1 */
	s   = two_sum(P00, c[0], &t);       /* 6 */
	tp  = t + w;                        /* 1 */
	z[0] = quick_two_sum(s, tp, &z[1]); /* 3  exp(s) >= exp(tp) machine-proved */
}

/* ==============================================================================
 *  TW-FMA : z = x*y + c   (K = 3, 66 flops, Algorithm 2)
 *  matches twfma_fix2.fpan
 * ============================================================================== */
static inline void bnc_twfma(double z[3], const double x[3], const double y[3], const double c[3])
{
	double P00, E00, P01, E01, P10, E10, P02, P11, P20, sg, G;
	double A, q1, q2, q3, B, r, m1, m2, w0, w1, w2;

	P00 = two_prod(x[0], y[0], &E00);          /* 2 */
	P01 = two_prod(x[0], y[1], &E01);          /* 2 */
	P10 = two_prod(x[1], y[0], &E10);          /* 2 */
	P02 = x[0] * y[2];                         /* 1 */
	P11 = x[1] * y[1];                         /* 1 */
	P20 = x[2] * y[0];                         /* 1 */
	sg  = (P02 + P20) + P11;                   /* 2  diagonal i+j=2, transposed pair first */
	G   = ((E01 + E10) + sg) + c[2];           /* 3 */
	A   = two_sum(P01, P10, &q1);              /* 6 */
	A   = two_sum(A,   E00, &q2);              /* 6 */
	A   = two_sum(A,   c[1], &q3);             /* 6 */
	G   = G + ((q1 + q2) + q3);                /* 3 */
	B   = two_sum(P00, c[0], &r);              /* 6 */
	m1  = two_sum(r, A, &m2);                  /* 6 */
	m2  = m2 + G;                              /* 1 */
	/* renorm pass 1/3 */
	w0 = quick_two_sum(B, m1, &w1);
	w1 = two_sum(w1, m2, &w2);
	/* renorm pass 2/3 */
	w0 = two_sum(w0, w1, &w1);
	w1 = quick_two_sum(w1, w2, &w2);
	/* renorm pass 3/3 */
	z[0] = quick_two_sum(w0, w1, &w1);
	z[1] = quick_two_sum(w1, w2, &z[2]);
}

/* ==============================================================================
 *  QW-FMA : z = x*y + c   (K = 4, 146 flops, Algorithm 3)
 *  matches qwfma_fix3.fpan (two unproved FastTwoSum demoted to TwoSum)
 * ============================================================================== */
static inline void bnc_qwfma(double z[4], const double x[4], const double y[4], const double c[4])
{
	double P00, E00, P01, E01, P10, E10, P02, E02, P11, E11, P20, E20;
	double P03, P12, P21, P30, D, B, r, Et;
	double A1, f1, f2, f3, f4;
	double A2, g1, g2, g3, g4, g5, g6, g7, g8, g9;
	double A3, t1, t2, t3, t4;
	double w0, w1, w2, w3;

	/* product expansion: TwoProd for i+j <= 2, plain products for i+j = 3,
	   truncation for i+j >= 4 (O(u^4)) */
	P00 = two_prod(x[0], y[0], &E00);          /* 2 */
	P01 = two_prod(x[0], y[1], &E01);          /* 2 */
	P10 = two_prod(x[1], y[0], &E10);          /* 2 */
	P02 = two_prod(x[0], y[2], &E02);          /* 2 */
	P11 = two_prod(x[1], y[1], &E11);          /* 2 */
	P20 = two_prod(x[2], y[0], &E20);          /* 2 */
	P03 = x[0] * y[3];                         /* 1 */
	P12 = x[1] * y[2];                         /* 1 */
	P21 = x[2] * y[1];                         /* 1 */
	P30 = x[3] * y[0];                         /* 1 */
	D   = (P03 + P30) + (P12 + P21);           /* 3  diagonal i+j=3, transposed pairs first */
	B   = two_sum(P00, c[0], &r);              /* 6  L0 */

	/* L1: accumulation at weight u */
	A1  = two_sum(P01, P10,  &f1);             /* 6 */
	A1  = two_sum(A1,  E00,  &f2);             /* 6 */
	A1  = two_sum(A1,  c[1], &f3);             /* 6 */
	A1  = two_sum(A1,  r,    &f4);             /* 6 */

	/* L2: 9-gate TwoSum accumulation at weight u^2 */
	A2  = two_sum(P02, P20,  &g1);             /* 6 */
	A2  = two_sum(A2,  P11,  &g2);             /* 6 */
	Et  = two_sum(E01, E10,  &g4);             /* 6  SYM: pair E01,E10 first (commutativity) */
	A2  = two_sum(A2,  Et,   &g3);             /* 6 */
	A2  = two_sum(A2,  c[2], &g5);             /* 6 */
	A2  = two_sum(A2,  f1,   &g6);             /* 6 */
	A2  = two_sum(A2,  f2,   &g7);             /* 6 */
	A2  = two_sum(A2,  f3,   &g8);             /* 6 */
	A2  = two_sum(A2,  f4,   &g9);             /* 6 */

	/* L3: rounded sums at weight u^3 (same order as the discarded wires of qwfma.fpan) */
	t1  = E02 + E20;                           /* 1 */
	t2  = E11 + D;                             /* 1 */
	t3  = (t1 + t2) + c[3];                    /* 2 */
	t1  = g1 + g2;                             /* 1 */
	t2  = g3 + g4;                             /* 1 */
	t1  = t1 + t2;                             /* 1 */
	t2  = g6 + g7;                             /* 1 */
	t4  = g8 + g9;                             /* 1 */
	t2  = t2 + t4;                             /* 1 */
	t1  = t1 + t2;                             /* 1 */
	t1  = t1 + g5;                             /* 1 */
	A3  = t3 + t1;                             /* 1  13 adds in total */

	/* renormalization (only (B,A1) and (sig,w3) admit provable FastTwoSum) */
	/* renorm pass 1/5 */
	w0 = quick_two_sum(B, A1, &w1);
	w1 = two_sum(w1, A2, &w2);
	w2 = two_sum(w2, A3, &w3);
	/* renorm pass 2/5 */
	w0 = two_sum(w0, w1, &w1);
	w1 = two_sum(w1, w2, &w2);
	w2 = quick_two_sum(w2, w3, &w3);
	/* renorm pass 3/5 */
	w0 = two_sum(w0, w1, &w1);
	w1 = quick_two_sum(w1, w2, &w2);
	w2 = quick_two_sum(w2, w3, &w3);
	/* renorm pass 4/5 */
	w0 = quick_two_sum(w0, w1, &w1);
	w1 = quick_two_sum(w1, w2, &w2);
	w2 = quick_two_sum(w2, w3, &w3);
	/* renorm pass 5/5 */
	z[0] = quick_two_sum(w0, w1, &w1);
	z[1] = quick_two_sum(w1, w2, &w2);
	z[2] = quick_two_sum(w2, w3, &z[3]);
}

/* ==============================================================================
 *  div/sqrt-safe variants (scalar double multiplier):  z = x*y + c
 *
 *  Port of dw_fma_safe / tw_fma_safe / qw_fma_safe of dtq-0.0.3
 *  (qd/dd_inline.h, qd/td_inline.h, qd/qd_inline.h).
 *
 *  The certified bounds of bnc_dwfma/twfma/qwfma above are machine-proved for
 *  operands that are normalized (non-overlapping) expansions.  The Newton
 *  iterations of division and square root feed residuals that are NOT
 *  non-overlapping, so no FastTwoSum (quick_two_sum) precondition can be
 *  asserted for any gate: a FastTwoSum whose precondition fails does not even
 *  satisfy s + e = a + b.  These variants therefore use TwoSum everywhere
 *  (DW: 20 flops, and all renormalization passes of TW/QW are demoted); a
 *  safe variant is never cheaper, and never less accurate, than the standard
 *  one.  Use them whenever an operand may be unnormalized (bnc_*_div_fma in
 *  bncelem_*.h are the in-library consumers).
 * ============================================================================== */

/* DW-FMA (safe) : z = x*y + c, y is a plain double */
static inline void bnc_dwfma_safe(double z[2], const double x[2], double y, const double c[2])
{
	double P00, E00, P01, P10, l, v, w, s, t, tp;

	P00 = two_prod(x[0], y, &E00);
	P01 = 0.0;                          /* scalar multiplier: x[0]*y[1] term absent */
	P10 = x[1] * y;
	l   = P01 + P10;
	v   = E00 + c[1];
	w   = v + l;
	s   = two_sum(P00, c[0], &t);
	tp  = t + w;
	z[0] = two_sum(s, tp, &z[1]);       /* demoted FastTwoSum */
}

/* TW-FMA (safe) : z = x*y + c, y is a plain double */
static inline void bnc_twfma_safe(double z[3], const double x[3], double y, const double c[3])
{
	double P00, E00, P01, E01, P10, E10, P02, P11, P20, sg, G;
	double A, q1, q2, q3, B, r, m1, m2, w0, w1, w2;

	P00 = two_prod(x[0], y, &E00);
	P01 = 0.0; E01 = 0.0;
	P10 = two_prod(x[1], y, &E10);
	P02 = 0.0;
	P11 = 0.0;
	P20 = x[2] * y;
	sg  = (P02 + P20) + P11;
	G   = ((E01 + E10) + sg) + c[2];
	A   = two_sum(P01, P10, &q1);
	A   = two_sum(A,   E00, &q2);
	A   = two_sum(A,   c[1], &q3);
	G   = G + ((q1 + q2) + q3);
	B   = two_sum(P00, c[0], &r);
	m1  = two_sum(r, A, &m2);
	m2  = m2 + G;
	/* renorm pass 1/3 (all TwoSum) */
	w0 = two_sum(B, m1, &w1);
	w1 = two_sum(w1, m2, &w2);
	/* renorm pass 2/3 */
	w0 = two_sum(w0, w1, &w1);
	w1 = two_sum(w1, w2, &w2);
	/* renorm pass 3/3 */
	z[0] = two_sum(w0, w1, &w1);
	z[1] = two_sum(w1, w2, &z[2]);
}

/* QW-FMA (safe) : z = x*y + c, y is a plain double */
static inline void bnc_qwfma_safe(double z[4], const double x[4], double y, const double c[4])
{
	double P00, E00, P01, E01, P10, E10, P02, E02, P11, E11, P20, E20;
	double P03, P12, P21, P30, D, B, r, Et;
	double A1, f1, f2, f3, f4;
	double A2, g1, g2, g3, g4, g5, g6, g7, g8, g9;
	double A3, t1, t2, t3, t4;
	double w0, w1, w2, w3;

	P00 = two_prod(x[0], y, &E00);
	P01 = 0.0; E01 = 0.0;
	P10 = two_prod(x[1], y, &E10);
	P02 = 0.0; E02 = 0.0;
	P11 = 0.0; E11 = 0.0;
	P20 = two_prod(x[2], y, &E20);
	P03 = 0.0;
	P12 = 0.0;
	P21 = 0.0;
	P30 = x[3] * y;
	D   = (P03 + P30) + (P12 + P21);
	B   = two_sum(P00, c[0], &r);
	A1  = two_sum(P01, P10,  &f1);
	A1  = two_sum(A1,  E00,  &f2);
	A1  = two_sum(A1,  c[1], &f3);
	A1  = two_sum(A1,  r,    &f4);
	A2  = two_sum(P02, P20,  &g1);
	A2  = two_sum(A2,  P11,  &g2);
	Et  = two_sum(E01, E10,  &g4);
	A2  = two_sum(A2,  Et,   &g3);
	A2  = two_sum(A2,  c[2], &g5);
	A2  = two_sum(A2,  f1,   &g6);
	A2  = two_sum(A2,  f2,   &g7);
	A2  = two_sum(A2,  f3,   &g8);
	A2  = two_sum(A2,  f4,   &g9);
	t1  = E02 + E20;
	t2  = E11 + D;
	t3  = (t1 + t2) + c[3];
	t1  = g1 + g2;
	t2  = g3 + g4;
	t1  = t1 + t2;
	t2  = g6 + g7;
	t4  = g8 + g9;
	t2  = t2 + t4;
	t1  = t1 + t2;
	t1  = t1 + g5;
	A3  = t3 + t1;
	/* renorm pass 1/5 (all TwoSum) */
	w0 = two_sum(B, A1, &w1);
	w1 = two_sum(w1, A2, &w2);
	w2 = two_sum(w2, A3, &w3);
	/* renorm pass 2/5 */
	w0 = two_sum(w0, w1, &w1);
	w1 = two_sum(w1, w2, &w2);
	w2 = two_sum(w2, w3, &w3);
	/* renorm pass 3/5 */
	w0 = two_sum(w0, w1, &w1);
	w1 = two_sum(w1, w2, &w2);
	w2 = two_sum(w2, w3, &w3);
	/* renorm pass 4/5 */
	w0 = two_sum(w0, w1, &w1);
	w1 = two_sum(w1, w2, &w2);
	w2 = two_sum(w2, w3, &w3);
	/* renorm pass 5/5 */
	z[0] = two_sum(w0, w1, &w1);
	z[1] = two_sum(w1, w2, &w2);
	z[2] = two_sum(w2, w3, &z[3]);
}

#endif /* __BNC_FMA_D_H */
