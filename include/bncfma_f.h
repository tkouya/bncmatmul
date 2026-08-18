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
#ifndef __BNC_FMA_F_H
#define __BNC_FMA_F_H

#include <math.h>

/* ftwo_sum / fquick_two_sum / ftwo_prod (float) come from c_ds_qs.h
   (header-only static inline). */
#include "c_ds_qs.h"

/* ==============================================================================
 *  Single-precision base (DS / TS / QS): identical netlists on binary32.
 *  The certified bounds hold for every precision p, hence also for binary32.
 * ============================================================================== */

/* DS-FMA : z = x*y + c   (K = 2, 17 flops) */
static inline void bnc_dwfmaf(float z[2], const float x[2], const float y[2], const float c[2])
{
	float P00, E00, P01, P10, l, v, w, s, t, tp;

	P00 = ftwo_prod(x[0], y[0], &E00);
	P01 = x[0] * y[1];
	P10 = x[1] * y[0];
	l   = P01 + P10;
	v   = E00 + c[1];
	w   = v + l;
	s   = ftwo_sum(P00, c[0], &t);
	tp  = t + w;
	z[0] = fquick_two_sum(s, tp, &z[1]);
}

/* TS-FMA : z = x*y + c   (K = 3, 66 flops) */
static inline void bnc_twfmaf(float z[3], const float x[3], const float y[3], const float c[3])
{
	float P00, E00, P01, E01, P10, E10, P02, P11, P20, sg, G;
	float A, q1, q2, q3, B, r, m1, m2, w0, w1, w2;

	P00 = ftwo_prod(x[0], y[0], &E00);
	P01 = ftwo_prod(x[0], y[1], &E01);
	P10 = ftwo_prod(x[1], y[0], &E10);
	P02 = x[0] * y[2];
	P11 = x[1] * y[1];
	P20 = x[2] * y[0];
	sg  = (P02 + P20) + P11;
	G   = ((E01 + E10) + sg) + c[2];
	A   = ftwo_sum(P01, P10, &q1);
	A   = ftwo_sum(A,   E00, &q2);
	A   = ftwo_sum(A,   c[1], &q3);
	G   = G + ((q1 + q2) + q3);
	B   = ftwo_sum(P00, c[0], &r);
	m1  = ftwo_sum(r, A, &m2);
	m2  = m2 + G;
	/* renorm pass 1/3 */
	w0 = fquick_two_sum(B, m1, &w1);
	w1 = ftwo_sum(w1, m2, &w2);
	/* renorm pass 2/3 */
	w0 = ftwo_sum(w0, w1, &w1);
	w1 = fquick_two_sum(w1, w2, &w2);
	/* renorm pass 3/3 */
	z[0] = fquick_two_sum(w0, w1, &w1);
	z[1] = fquick_two_sum(w1, w2, &z[2]);
}

/* QS-FMA : z = x*y + c   (K = 4, 146 flops) */
static inline void bnc_qwfmaf(float z[4], const float x[4], const float y[4], const float c[4])
{
	float P00, E00, P01, E01, P10, E10, P02, E02, P11, E11, P20, E20;
	float P03, P12, P21, P30, D, B, r, Et;
	float A1, f1, f2, f3, f4;
	float A2, g1, g2, g3, g4, g5, g6, g7, g8, g9;
	float A3, t1, t2, t3, t4;
	float w0, w1, w2, w3;

	P00 = ftwo_prod(x[0], y[0], &E00);
	P01 = ftwo_prod(x[0], y[1], &E01);
	P10 = ftwo_prod(x[1], y[0], &E10);
	P02 = ftwo_prod(x[0], y[2], &E02);
	P11 = ftwo_prod(x[1], y[1], &E11);
	P20 = ftwo_prod(x[2], y[0], &E20);
	P03 = x[0] * y[3];
	P12 = x[1] * y[2];
	P21 = x[2] * y[1];
	P30 = x[3] * y[0];
	D   = (P03 + P30) + (P12 + P21);
	B   = ftwo_sum(P00, c[0], &r);

	A1  = ftwo_sum(P01, P10,  &f1);
	A1  = ftwo_sum(A1,  E00,  &f2);
	A1  = ftwo_sum(A1,  c[1], &f3);
	A1  = ftwo_sum(A1,  r,    &f4);

	A2  = ftwo_sum(P02, P20,  &g1);
	A2  = ftwo_sum(A2,  P11,  &g2);
	Et  = ftwo_sum(E01, E10,  &g4);
	A2  = ftwo_sum(A2,  Et,   &g3);
	A2  = ftwo_sum(A2,  c[2], &g5);
	A2  = ftwo_sum(A2,  f1,   &g6);
	A2  = ftwo_sum(A2,  f2,   &g7);
	A2  = ftwo_sum(A2,  f3,   &g8);
	A2  = ftwo_sum(A2,  f4,   &g9);

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

	/* renorm pass 1/5 */
	w0 = fquick_two_sum(B, A1, &w1);
	w1 = ftwo_sum(w1, A2, &w2);
	w2 = ftwo_sum(w2, A3, &w3);
	/* renorm pass 2/5 */
	w0 = ftwo_sum(w0, w1, &w1);
	w1 = ftwo_sum(w1, w2, &w2);
	w2 = fquick_two_sum(w2, w3, &w3);
	/* renorm pass 3/5 */
	w0 = ftwo_sum(w0, w1, &w1);
	w1 = fquick_two_sum(w1, w2, &w2);
	w2 = fquick_two_sum(w2, w3, &w3);
	/* renorm pass 4/5 */
	w0 = fquick_two_sum(w0, w1, &w1);
	w1 = fquick_two_sum(w1, w2, &w2);
	w2 = fquick_two_sum(w2, w3, &w3);
	/* renorm pass 5/5 */
	z[0] = fquick_two_sum(w0, w1, &w1);
	z[1] = fquick_two_sum(w1, w2, &w2);
	z[2] = fquick_two_sum(w2, w3, &z[3]);
}

/* ==============================================================================
 *  div/sqrt-safe variants (scalar float multiplier):  z = x*y + c
 *  Single-precision counterparts of bnc_dwfma_safe / bnc_twfma_safe /
 *  bnc_qwfma_safe (see bncfma_d.h for the rationale): every FastTwoSum is
 *  demoted to TwoSum because the Newton residuals of division/square root
 *  are not non-overlapping expansions.  Ports of dw_fma_safe (ds_real.h),
 *  tw_fma_safe (ts_real.h) and qw_fma_safe (qs_real.h) of dtq-0.0.3.
 * ============================================================================== */

/* DS-FMA (safe) : z = x*y + c, y is a plain float */
static inline void bnc_dwfmaf_safe(float z[2], const float x[2], float y, const float c[2])
{
	float P00, E00, P01, P10, l, v, w, s, t, tp;

	P00 = ftwo_prod(x[0], y, &E00);
	P01 = 0.0f;                         /* scalar multiplier: x[0]*y[1] term absent */
	P10 = x[1] * y;
	l   = P01 + P10;
	v   = E00 + c[1];
	w   = v + l;
	s   = ftwo_sum(P00, c[0], &t);
	tp  = t + w;
	z[0] = ftwo_sum(s, tp, &z[1]);      /* demoted FastTwoSum */
}

/* TS-FMA (safe) : z = x*y + c, y is a plain float */
static inline void bnc_twfmaf_safe(float z[3], const float x[3], float y, const float c[3])
{
	float P00, E00, P01, E01, P10, E10, P02, P11, P20, sg, G;
	float A, q1, q2, q3, B, r, m1, m2, w0, w1, w2;

	P00 = ftwo_prod(x[0], y, &E00);
	P01 = 0.0f; E01 = 0.0f;
	P10 = ftwo_prod(x[1], y, &E10);
	P02 = 0.0f;
	P11 = 0.0f;
	P20 = x[2] * y;
	sg  = (P02 + P20) + P11;
	G   = ((E01 + E10) + sg) + c[2];
	A   = ftwo_sum(P01, P10, &q1);
	A   = ftwo_sum(A,   E00, &q2);
	A   = ftwo_sum(A,   c[1], &q3);
	G   = G + ((q1 + q2) + q3);
	B   = ftwo_sum(P00, c[0], &r);
	m1  = ftwo_sum(r, A, &m2);
	m2  = m2 + G;
	/* renorm pass 1/3 (all TwoSum) */
	w0 = ftwo_sum(B, m1, &w1);
	w1 = ftwo_sum(w1, m2, &w2);
	/* renorm pass 2/3 */
	w0 = ftwo_sum(w0, w1, &w1);
	w1 = ftwo_sum(w1, w2, &w2);
	/* renorm pass 3/3 */
	z[0] = ftwo_sum(w0, w1, &w1);
	z[1] = ftwo_sum(w1, w2, &z[2]);
}

/* QS-FMA (safe) : z = x*y + c, y is a plain float */
static inline void bnc_qwfmaf_safe(float z[4], const float x[4], float y, const float c[4])
{
	float P00, E00, P01, E01, P10, E10, P02, E02, P11, E11, P20, E20;
	float P03, P12, P21, P30, D, B, r, Et;
	float A1, f1, f2, f3, f4;
	float A2, g1, g2, g3, g4, g5, g6, g7, g8, g9;
	float A3, t1, t2, t3, t4;
	float w0, w1, w2, w3;

	P00 = ftwo_prod(x[0], y, &E00);
	P01 = 0.0f; E01 = 0.0f;
	P10 = ftwo_prod(x[1], y, &E10);
	P02 = 0.0f; E02 = 0.0f;
	P11 = 0.0f; E11 = 0.0f;
	P20 = ftwo_prod(x[2], y, &E20);
	P03 = 0.0f;
	P12 = 0.0f;
	P21 = 0.0f;
	P30 = x[3] * y;
	D   = (P03 + P30) + (P12 + P21);
	B   = ftwo_sum(P00, c[0], &r);
	A1  = ftwo_sum(P01, P10,  &f1);
	A1  = ftwo_sum(A1,  E00,  &f2);
	A1  = ftwo_sum(A1,  c[1], &f3);
	A1  = ftwo_sum(A1,  r,    &f4);
	A2  = ftwo_sum(P02, P20,  &g1);
	A2  = ftwo_sum(A2,  P11,  &g2);
	Et  = ftwo_sum(E01, E10,  &g4);
	A2  = ftwo_sum(A2,  Et,   &g3);
	A2  = ftwo_sum(A2,  c[2], &g5);
	A2  = ftwo_sum(A2,  f1,   &g6);
	A2  = ftwo_sum(A2,  f2,   &g7);
	A2  = ftwo_sum(A2,  f3,   &g8);
	A2  = ftwo_sum(A2,  f4,   &g9);
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
	w0 = ftwo_sum(B, A1, &w1);
	w1 = ftwo_sum(w1, A2, &w2);
	w2 = ftwo_sum(w2, A3, &w3);
	/* renorm pass 2/5 */
	w0 = ftwo_sum(w0, w1, &w1);
	w1 = ftwo_sum(w1, w2, &w2);
	w2 = ftwo_sum(w2, w3, &w3);
	/* renorm pass 3/5 */
	w0 = ftwo_sum(w0, w1, &w1);
	w1 = ftwo_sum(w1, w2, &w2);
	w2 = ftwo_sum(w2, w3, &w3);
	/* renorm pass 4/5 */
	w0 = ftwo_sum(w0, w1, &w1);
	w1 = ftwo_sum(w1, w2, &w2);
	w2 = ftwo_sum(w2, w3, &w3);
	/* renorm pass 5/5 */
	z[0] = ftwo_sum(w0, w1, &w1);
	z[1] = ftwo_sum(w1, w2, &w2);
	z[2] = ftwo_sum(w2, w3, &z[3]);
}

#endif /* __BNC_FMA_F_H */
