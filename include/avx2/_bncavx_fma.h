/********************************************************************************/
/* _bncavx_fma.h : AVX2/AVX-512 branch-free fused multiply-add for DW/TW/QW     */
/*                 z := x * y + c                                              */
/*                                                                              */
/* AVX2/AVX-512 vectorization of Algorithms 1-3 of                              */
/*   T. Kouya, "Performance evaluation of branch-free fused multiply-add        */
/*   algorithms for multi-component-type multiple-precision floating-point      */
/*   arithmetic", arXiv:2607.11391v1 (2026).                                    */
/* Operation-by-operation identical to the scalar reference in include/bncfma_d */
/* .h / bncfma_f.h, hence bitwise identical results (vectorization does not     */
/* change the rounding).                                                        */
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
 * Requires -ffp-contract=off (as everywhere in BNCmatmul): the EFTs
 * two_sum / two_prod stop being error-free if the compiler contracts a*b+c.
 *
 * Certified error bounds (u = 2^-p, FPANVerifier, all precisions p at once):
 *   DW: |z - (xy+c)| <=  34 u^2 (|xy| + |c|)   17 flops
 *   TW: |z - (xy+c)| <= 184 u^3 (|xy| + |c|)   66 flops
 *   QW: |z - (xy+c)| <= 812 u^4 (|xy| + |c|)  146 flops
 *
 * Primitive mapping (see _bncavx_deft.h / _bncavx_feft.h and their _avx512
 * counterparts):
 *   two_sum(a,b,&s,&e)       -> s = _bncavx2_dtwo_sum(a, b, &e)
 *   fast_two_sum(a,b,&s,&e)  -> s = _bncavx2_dquick_two_sum(a, b, &e)
 *   two_prod(a,b,&p,&e)      -> p = _bncavx2_dtwo_prod(a, b, &e)  (hardware FMA:
 *                               e = _mm256_fmsub_pd(a, b, p))
 *   plain a+b / a*b          -> _mm256_add_pd / _mm256_mul_pd
 * (same with _ps for float base, _bncavx512_* / _mm512_* for AVX-512.)
 *
 * AVX2   : 4 double lanes (__m256d) or  8 float lanes (__m256) per call.
 * AVX-512: 8 double lanes (__m512d) or 16 float lanes (__m512) per call.
 */
#ifndef __BNCAVX_FMA_H
#define __BNCAVX_FMA_H

#if defined(__AVX2__) || defined(__AVX512F__)

#include <immintrin.h>

/* error-free transformations used below (self-contained) */
#include "_bncavx_deft.h"
#include "_bncavx_feft.h"
#include "_bncavx_deft_avx512.h"
#include "_bncavx_feft_avx512.h"

#endif /* defined(__AVX2__) || defined(__AVX512F__) */

#if defined(__AVX2__)

/* ==============================================================================
 *  AVX2, double base (DD / TD / QD) -- 4 lanes per call
 * ============================================================================== */

/* ------------------------------------------------------------------
 * _bncavx2_dwfma : z[2] := x[2] * y[2] + c[2]   (K = 2, 17 flops)
 * ------------------------------------------------------------------ */
static inline void _bncavx2_dwfma(
        __m256d       z[2],
        const __m256d x[2],
        const __m256d y[2],
        const __m256d c[2])
{
    __m256d P00, E00, P01, P10, l, v, w, s, t, tp;

    P00 = _bncavx2_dtwo_prod(x[0], y[0], &E00);
    P01 = _mm256_mul_pd(x[0], y[1]);
    P10 = _mm256_mul_pd(x[1], y[0]);
    l   = _mm256_add_pd(P01, P10);
    v   = _mm256_add_pd(E00, c[1]);
    w   = _mm256_add_pd(v, l);
    s   = _bncavx2_dtwo_sum(P00, c[0], &t);
    tp  = _mm256_add_pd(t, w);
    z[0] = _bncavx2_dquick_two_sum(s, tp, &(z[1]));
}

/* ------------------------------------------------------------------
 * _bncavx2_twfma : z[3] := x[3] * y[3] + c[3]   (K = 3, 66 flops)
 * ------------------------------------------------------------------ */
static inline void _bncavx2_twfma(
        __m256d       z[3],
        const __m256d x[3],
        const __m256d y[3],
        const __m256d c[3])
{
    __m256d P00, E00, P01, E01, P10, E10, P02, P11, P20, sg, G;
    __m256d A, q1, q2, q3, B, r, m1, m2, w0, w1, w2;

    P00 = _bncavx2_dtwo_prod(x[0], y[0], &E00);
    P01 = _bncavx2_dtwo_prod(x[0], y[1], &E01);
    P10 = _bncavx2_dtwo_prod(x[1], y[0], &E10);
    P02 = _mm256_mul_pd(x[0], y[2]);
    P11 = _mm256_mul_pd(x[1], y[1]);
    P20 = _mm256_mul_pd(x[2], y[0]);
    sg  = _mm256_add_pd(_mm256_add_pd(P02, P20), P11);
    G   = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(E01, E10), sg), c[2]);
    A   = _bncavx2_dtwo_sum(P01, P10,  &q1);
    A   = _bncavx2_dtwo_sum(A,   E00,  &q2);
    A   = _bncavx2_dtwo_sum(A,   c[1], &q3);
    G   = _mm256_add_pd(G, _mm256_add_pd(_mm256_add_pd(q1, q2), q3));
    B   = _bncavx2_dtwo_sum(P00, c[0], &r);
    m1  = _bncavx2_dtwo_sum(r, A, &m2);
    m2  = _mm256_add_pd(m2, G);
    /* renorm pass 1/3 */
    w0 = _bncavx2_dquick_two_sum(B, m1, &w1);
    w1 = _bncavx2_dtwo_sum(w1, m2, &w2);
    /* renorm pass 2/3 */
    w0 = _bncavx2_dtwo_sum(w0, w1, &w1);
    w1 = _bncavx2_dquick_two_sum(w1, w2, &w2);
    /* renorm pass 3/3 */
    z[0] = _bncavx2_dquick_two_sum(w0, w1, &w1);
    z[1] = _bncavx2_dquick_two_sum(w1, w2, &(z[2]));
}

/* ------------------------------------------------------------------
 * _bncavx2_qwfma : z[4] := x[4] * y[4] + c[4]   (K = 4, 146 flops)
 * ------------------------------------------------------------------ */
static inline void _bncavx2_qwfma(
        __m256d       z[4],
        const __m256d x[4],
        const __m256d y[4],
        const __m256d c[4])
{
    __m256d P00, E00, P01, E01, P10, E10, P02, E02, P11, E11, P20, E20;
    __m256d P03, P12, P21, P30, D, B, r, Et;
    __m256d A1, f1, f2, f3, f4;
    __m256d A2, g1, g2, g3, g4, g5, g6, g7, g8, g9;
    __m256d A3, t1, t2, t3, t4;
    __m256d w0, w1, w2, w3;

    P00 = _bncavx2_dtwo_prod(x[0], y[0], &E00);
    P01 = _bncavx2_dtwo_prod(x[0], y[1], &E01);
    P10 = _bncavx2_dtwo_prod(x[1], y[0], &E10);
    P02 = _bncavx2_dtwo_prod(x[0], y[2], &E02);
    P11 = _bncavx2_dtwo_prod(x[1], y[1], &E11);
    P20 = _bncavx2_dtwo_prod(x[2], y[0], &E20);
    P03 = _mm256_mul_pd(x[0], y[3]);
    P12 = _mm256_mul_pd(x[1], y[2]);
    P21 = _mm256_mul_pd(x[2], y[1]);
    P30 = _mm256_mul_pd(x[3], y[0]);
    D   = _mm256_add_pd(_mm256_add_pd(P03, P30), _mm256_add_pd(P12, P21));
    B   = _bncavx2_dtwo_sum(P00, c[0], &r);

    /* L1 */
    A1  = _bncavx2_dtwo_sum(P01, P10,  &f1);
    A1  = _bncavx2_dtwo_sum(A1,  E00,  &f2);
    A1  = _bncavx2_dtwo_sum(A1,  c[1], &f3);
    A1  = _bncavx2_dtwo_sum(A1,  r,    &f4);

    /* L2 */
    A2  = _bncavx2_dtwo_sum(P02, P20,  &g1);
    A2  = _bncavx2_dtwo_sum(A2,  P11,  &g2);
    Et  = _bncavx2_dtwo_sum(E01, E10,  &g4);
    A2  = _bncavx2_dtwo_sum(A2,  Et,   &g3);
    A2  = _bncavx2_dtwo_sum(A2,  c[2], &g5);
    A2  = _bncavx2_dtwo_sum(A2,  f1,   &g6);
    A2  = _bncavx2_dtwo_sum(A2,  f2,   &g7);
    A2  = _bncavx2_dtwo_sum(A2,  f3,   &g8);
    A2  = _bncavx2_dtwo_sum(A2,  f4,   &g9);

    /* L3 */
    t1  = _mm256_add_pd(E02, E20);
    t2  = _mm256_add_pd(E11, D);
    t3  = _mm256_add_pd(_mm256_add_pd(t1, t2), c[3]);
    t1  = _mm256_add_pd(g1, g2);
    t2  = _mm256_add_pd(g3, g4);
    t1  = _mm256_add_pd(t1, t2);
    t2  = _mm256_add_pd(g6, g7);
    t4  = _mm256_add_pd(g8, g9);
    t2  = _mm256_add_pd(t2, t4);
    t1  = _mm256_add_pd(t1, t2);
    t1  = _mm256_add_pd(t1, g5);
    A3  = _mm256_add_pd(t3, t1);

    /* renormalization */
    /* renorm pass 1/5 */
    w0 = _bncavx2_dquick_two_sum(B, A1, &w1);
    w1 = _bncavx2_dtwo_sum(w1, A2, &w2);
    w2 = _bncavx2_dtwo_sum(w2, A3, &w3);
    /* renorm pass 2/5 */
    w0 = _bncavx2_dtwo_sum(w0, w1, &w1);
    w1 = _bncavx2_dtwo_sum(w1, w2, &w2);
    w2 = _bncavx2_dquick_two_sum(w2, w3, &w3);
    /* renorm pass 3/5 */
    w0 = _bncavx2_dtwo_sum(w0, w1, &w1);
    w1 = _bncavx2_dquick_two_sum(w1, w2, &w2);
    w2 = _bncavx2_dquick_two_sum(w2, w3, &w3);
    /* renorm pass 4/5 */
    w0 = _bncavx2_dquick_two_sum(w0, w1, &w1);
    w1 = _bncavx2_dquick_two_sum(w1, w2, &w2);
    w2 = _bncavx2_dquick_two_sum(w2, w3, &w3);
    /* renorm pass 5/5 */
    z[0] = _bncavx2_dquick_two_sum(w0, w1, &w1);
    z[1] = _bncavx2_dquick_two_sum(w1, w2, &w2);
    z[2] = _bncavx2_dquick_two_sum(w2, w3, &(z[3]));
}

/* ==============================================================================
 *  AVX2, float base (DS / TS / QS) -- 8 lanes per call
 * ============================================================================== */

static inline void _bncavx2_dwfmaf(
        __m256       z[2],
        const __m256 x[2],
        const __m256 y[2],
        const __m256 c[2])
{
    __m256 P00, E00, P01, P10, l, v, w, s, t, tp;

    P00 = _bncavx2_ftwo_prod(x[0], y[0], &E00);
    P01 = _mm256_mul_ps(x[0], y[1]);
    P10 = _mm256_mul_ps(x[1], y[0]);
    l   = _mm256_add_ps(P01, P10);
    v   = _mm256_add_ps(E00, c[1]);
    w   = _mm256_add_ps(v, l);
    s   = _bncavx2_ftwo_sum(P00, c[0], &t);
    tp  = _mm256_add_ps(t, w);
    z[0] = _bncavx2_fquick_two_sum(s, tp, &(z[1]));
}

static inline void _bncavx2_twfmaf(
        __m256       z[3],
        const __m256 x[3],
        const __m256 y[3],
        const __m256 c[3])
{
    __m256 P00, E00, P01, E01, P10, E10, P02, P11, P20, sg, G;
    __m256 A, q1, q2, q3, B, r, m1, m2, w0, w1, w2;

    P00 = _bncavx2_ftwo_prod(x[0], y[0], &E00);
    P01 = _bncavx2_ftwo_prod(x[0], y[1], &E01);
    P10 = _bncavx2_ftwo_prod(x[1], y[0], &E10);
    P02 = _mm256_mul_ps(x[0], y[2]);
    P11 = _mm256_mul_ps(x[1], y[1]);
    P20 = _mm256_mul_ps(x[2], y[0]);
    sg  = _mm256_add_ps(_mm256_add_ps(P02, P20), P11);
    G   = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(E01, E10), sg), c[2]);
    A   = _bncavx2_ftwo_sum(P01, P10,  &q1);
    A   = _bncavx2_ftwo_sum(A,   E00,  &q2);
    A   = _bncavx2_ftwo_sum(A,   c[1], &q3);
    G   = _mm256_add_ps(G, _mm256_add_ps(_mm256_add_ps(q1, q2), q3));
    B   = _bncavx2_ftwo_sum(P00, c[0], &r);
    m1  = _bncavx2_ftwo_sum(r, A, &m2);
    m2  = _mm256_add_ps(m2, G);
    /* renorm pass 1/3 */
    w0 = _bncavx2_fquick_two_sum(B, m1, &w1);
    w1 = _bncavx2_ftwo_sum(w1, m2, &w2);
    /* renorm pass 2/3 */
    w0 = _bncavx2_ftwo_sum(w0, w1, &w1);
    w1 = _bncavx2_fquick_two_sum(w1, w2, &w2);
    /* renorm pass 3/3 */
    z[0] = _bncavx2_fquick_two_sum(w0, w1, &w1);
    z[1] = _bncavx2_fquick_two_sum(w1, w2, &(z[2]));
}

static inline void _bncavx2_qwfmaf(
        __m256       z[4],
        const __m256 x[4],
        const __m256 y[4],
        const __m256 c[4])
{
    __m256 P00, E00, P01, E01, P10, E10, P02, E02, P11, E11, P20, E20;
    __m256 P03, P12, P21, P30, D, B, r, Et;
    __m256 A1, f1, f2, f3, f4;
    __m256 A2, g1, g2, g3, g4, g5, g6, g7, g8, g9;
    __m256 A3, t1, t2, t3, t4;
    __m256 w0, w1, w2, w3;

    P00 = _bncavx2_ftwo_prod(x[0], y[0], &E00);
    P01 = _bncavx2_ftwo_prod(x[0], y[1], &E01);
    P10 = _bncavx2_ftwo_prod(x[1], y[0], &E10);
    P02 = _bncavx2_ftwo_prod(x[0], y[2], &E02);
    P11 = _bncavx2_ftwo_prod(x[1], y[1], &E11);
    P20 = _bncavx2_ftwo_prod(x[2], y[0], &E20);
    P03 = _mm256_mul_ps(x[0], y[3]);
    P12 = _mm256_mul_ps(x[1], y[2]);
    P21 = _mm256_mul_ps(x[2], y[1]);
    P30 = _mm256_mul_ps(x[3], y[0]);
    D   = _mm256_add_ps(_mm256_add_ps(P03, P30), _mm256_add_ps(P12, P21));
    B   = _bncavx2_ftwo_sum(P00, c[0], &r);

    /* L1 */
    A1  = _bncavx2_ftwo_sum(P01, P10,  &f1);
    A1  = _bncavx2_ftwo_sum(A1,  E00,  &f2);
    A1  = _bncavx2_ftwo_sum(A1,  c[1], &f3);
    A1  = _bncavx2_ftwo_sum(A1,  r,    &f4);

    /* L2 */
    A2  = _bncavx2_ftwo_sum(P02, P20,  &g1);
    A2  = _bncavx2_ftwo_sum(A2,  P11,  &g2);
    Et  = _bncavx2_ftwo_sum(E01, E10,  &g4);
    A2  = _bncavx2_ftwo_sum(A2,  Et,   &g3);
    A2  = _bncavx2_ftwo_sum(A2,  c[2], &g5);
    A2  = _bncavx2_ftwo_sum(A2,  f1,   &g6);
    A2  = _bncavx2_ftwo_sum(A2,  f2,   &g7);
    A2  = _bncavx2_ftwo_sum(A2,  f3,   &g8);
    A2  = _bncavx2_ftwo_sum(A2,  f4,   &g9);

    /* L3 */
    t1  = _mm256_add_ps(E02, E20);
    t2  = _mm256_add_ps(E11, D);
    t3  = _mm256_add_ps(_mm256_add_ps(t1, t2), c[3]);
    t1  = _mm256_add_ps(g1, g2);
    t2  = _mm256_add_ps(g3, g4);
    t1  = _mm256_add_ps(t1, t2);
    t2  = _mm256_add_ps(g6, g7);
    t4  = _mm256_add_ps(g8, g9);
    t2  = _mm256_add_ps(t2, t4);
    t1  = _mm256_add_ps(t1, t2);
    t1  = _mm256_add_ps(t1, g5);
    A3  = _mm256_add_ps(t3, t1);

    /* renorm pass 1/5 */
    w0 = _bncavx2_fquick_two_sum(B, A1, &w1);
    w1 = _bncavx2_ftwo_sum(w1, A2, &w2);
    w2 = _bncavx2_ftwo_sum(w2, A3, &w3);
    /* renorm pass 2/5 */
    w0 = _bncavx2_ftwo_sum(w0, w1, &w1);
    w1 = _bncavx2_ftwo_sum(w1, w2, &w2);
    w2 = _bncavx2_fquick_two_sum(w2, w3, &w3);
    /* renorm pass 3/5 */
    w0 = _bncavx2_ftwo_sum(w0, w1, &w1);
    w1 = _bncavx2_fquick_two_sum(w1, w2, &w2);
    w2 = _bncavx2_fquick_two_sum(w2, w3, &w3);
    /* renorm pass 4/5 */
    w0 = _bncavx2_fquick_two_sum(w0, w1, &w1);
    w1 = _bncavx2_fquick_two_sum(w1, w2, &w2);
    w2 = _bncavx2_fquick_two_sum(w2, w3, &w3);
    /* renorm pass 5/5 */
    z[0] = _bncavx2_fquick_two_sum(w0, w1, &w1);
    z[1] = _bncavx2_fquick_two_sum(w1, w2, &w2);
    z[2] = _bncavx2_fquick_two_sum(w2, w3, &(z[3]));
}

#endif /* defined(__AVX2__) */

#if defined(__AVX512F__)

/* ==============================================================================
 *  AVX-512, double base (DD / TD / QD) -- 8 lanes per call
 * ============================================================================== */

/* ------------------------------------------------------------------
 * _bncavx512_dwfma : z[2] := x[2] * y[2] + c[2]   (K = 2, 17 flops)
 * ------------------------------------------------------------------ */
static inline void _bncavx512_dwfma(
        __m512d       z[2],
        const __m512d x[2],
        const __m512d y[2],
        const __m512d c[2])
{
    __m512d P00, E00, P01, P10, l, v, w, s, t, tp;

    P00 = _bncavx512_dtwo_prod(x[0], y[0], &E00);
    P01 = _mm512_mul_pd(x[0], y[1]);
    P10 = _mm512_mul_pd(x[1], y[0]);
    l   = _mm512_add_pd(P01, P10);
    v   = _mm512_add_pd(E00, c[1]);
    w   = _mm512_add_pd(v, l);
    s   = _bncavx512_dtwo_sum(P00, c[0], &t);
    tp  = _mm512_add_pd(t, w);
    z[0] = _bncavx512_dquick_two_sum(s, tp, &(z[1]));
}

/* ------------------------------------------------------------------
 * _bncavx512_twfma : z[3] := x[3] * y[3] + c[3]   (K = 3, 66 flops)
 * ------------------------------------------------------------------ */
static inline void _bncavx512_twfma(
        __m512d       z[3],
        const __m512d x[3],
        const __m512d y[3],
        const __m512d c[3])
{
    __m512d P00, E00, P01, E01, P10, E10, P02, P11, P20, sg, G;
    __m512d A, q1, q2, q3, B, r, m1, m2, w0, w1, w2;

    P00 = _bncavx512_dtwo_prod(x[0], y[0], &E00);
    P01 = _bncavx512_dtwo_prod(x[0], y[1], &E01);
    P10 = _bncavx512_dtwo_prod(x[1], y[0], &E10);
    P02 = _mm512_mul_pd(x[0], y[2]);
    P11 = _mm512_mul_pd(x[1], y[1]);
    P20 = _mm512_mul_pd(x[2], y[0]);
    sg  = _mm512_add_pd(_mm512_add_pd(P02, P20), P11);
    G   = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(E01, E10), sg), c[2]);
    A   = _bncavx512_dtwo_sum(P01, P10,  &q1);
    A   = _bncavx512_dtwo_sum(A,   E00,  &q2);
    A   = _bncavx512_dtwo_sum(A,   c[1], &q3);
    G   = _mm512_add_pd(G, _mm512_add_pd(_mm512_add_pd(q1, q2), q3));
    B   = _bncavx512_dtwo_sum(P00, c[0], &r);
    m1  = _bncavx512_dtwo_sum(r, A, &m2);
    m2  = _mm512_add_pd(m2, G);
    /* renorm pass 1/3 */
    w0 = _bncavx512_dquick_two_sum(B, m1, &w1);
    w1 = _bncavx512_dtwo_sum(w1, m2, &w2);
    /* renorm pass 2/3 */
    w0 = _bncavx512_dtwo_sum(w0, w1, &w1);
    w1 = _bncavx512_dquick_two_sum(w1, w2, &w2);
    /* renorm pass 3/3 */
    z[0] = _bncavx512_dquick_two_sum(w0, w1, &w1);
    z[1] = _bncavx512_dquick_two_sum(w1, w2, &(z[2]));
}

/* ------------------------------------------------------------------
 * _bncavx512_qwfma : z[4] := x[4] * y[4] + c[4]   (K = 4, 146 flops)
 * ------------------------------------------------------------------ */
static inline void _bncavx512_qwfma(
        __m512d       z[4],
        const __m512d x[4],
        const __m512d y[4],
        const __m512d c[4])
{
    __m512d P00, E00, P01, E01, P10, E10, P02, E02, P11, E11, P20, E20;
    __m512d P03, P12, P21, P30, D, B, r, Et;
    __m512d A1, f1, f2, f3, f4;
    __m512d A2, g1, g2, g3, g4, g5, g6, g7, g8, g9;
    __m512d A3, t1, t2, t3, t4;
    __m512d w0, w1, w2, w3;

    P00 = _bncavx512_dtwo_prod(x[0], y[0], &E00);
    P01 = _bncavx512_dtwo_prod(x[0], y[1], &E01);
    P10 = _bncavx512_dtwo_prod(x[1], y[0], &E10);
    P02 = _bncavx512_dtwo_prod(x[0], y[2], &E02);
    P11 = _bncavx512_dtwo_prod(x[1], y[1], &E11);
    P20 = _bncavx512_dtwo_prod(x[2], y[0], &E20);
    P03 = _mm512_mul_pd(x[0], y[3]);
    P12 = _mm512_mul_pd(x[1], y[2]);
    P21 = _mm512_mul_pd(x[2], y[1]);
    P30 = _mm512_mul_pd(x[3], y[0]);
    D   = _mm512_add_pd(_mm512_add_pd(P03, P30), _mm512_add_pd(P12, P21));
    B   = _bncavx512_dtwo_sum(P00, c[0], &r);

    /* L1 */
    A1  = _bncavx512_dtwo_sum(P01, P10,  &f1);
    A1  = _bncavx512_dtwo_sum(A1,  E00,  &f2);
    A1  = _bncavx512_dtwo_sum(A1,  c[1], &f3);
    A1  = _bncavx512_dtwo_sum(A1,  r,    &f4);

    /* L2 */
    A2  = _bncavx512_dtwo_sum(P02, P20,  &g1);
    A2  = _bncavx512_dtwo_sum(A2,  P11,  &g2);
    Et  = _bncavx512_dtwo_sum(E01, E10,  &g4);
    A2  = _bncavx512_dtwo_sum(A2,  Et,   &g3);
    A2  = _bncavx512_dtwo_sum(A2,  c[2], &g5);
    A2  = _bncavx512_dtwo_sum(A2,  f1,   &g6);
    A2  = _bncavx512_dtwo_sum(A2,  f2,   &g7);
    A2  = _bncavx512_dtwo_sum(A2,  f3,   &g8);
    A2  = _bncavx512_dtwo_sum(A2,  f4,   &g9);

    /* L3 */
    t1  = _mm512_add_pd(E02, E20);
    t2  = _mm512_add_pd(E11, D);
    t3  = _mm512_add_pd(_mm512_add_pd(t1, t2), c[3]);
    t1  = _mm512_add_pd(g1, g2);
    t2  = _mm512_add_pd(g3, g4);
    t1  = _mm512_add_pd(t1, t2);
    t2  = _mm512_add_pd(g6, g7);
    t4  = _mm512_add_pd(g8, g9);
    t2  = _mm512_add_pd(t2, t4);
    t1  = _mm512_add_pd(t1, t2);
    t1  = _mm512_add_pd(t1, g5);
    A3  = _mm512_add_pd(t3, t1);

    /* renormalization */
    /* renorm pass 1/5 */
    w0 = _bncavx512_dquick_two_sum(B, A1, &w1);
    w1 = _bncavx512_dtwo_sum(w1, A2, &w2);
    w2 = _bncavx512_dtwo_sum(w2, A3, &w3);
    /* renorm pass 2/5 */
    w0 = _bncavx512_dtwo_sum(w0, w1, &w1);
    w1 = _bncavx512_dtwo_sum(w1, w2, &w2);
    w2 = _bncavx512_dquick_two_sum(w2, w3, &w3);
    /* renorm pass 3/5 */
    w0 = _bncavx512_dtwo_sum(w0, w1, &w1);
    w1 = _bncavx512_dquick_two_sum(w1, w2, &w2);
    w2 = _bncavx512_dquick_two_sum(w2, w3, &w3);
    /* renorm pass 4/5 */
    w0 = _bncavx512_dquick_two_sum(w0, w1, &w1);
    w1 = _bncavx512_dquick_two_sum(w1, w2, &w2);
    w2 = _bncavx512_dquick_two_sum(w2, w3, &w3);
    /* renorm pass 5/5 */
    z[0] = _bncavx512_dquick_two_sum(w0, w1, &w1);
    z[1] = _bncavx512_dquick_two_sum(w1, w2, &w2);
    z[2] = _bncavx512_dquick_two_sum(w2, w3, &(z[3]));
}

/* ==============================================================================
 *  AVX-512, float base (DS / TS / QS) -- 16 lanes per call
 * ============================================================================== */

static inline void _bncavx512_dwfmaf(
        __m512       z[2],
        const __m512 x[2],
        const __m512 y[2],
        const __m512 c[2])
{
    __m512 P00, E00, P01, P10, l, v, w, s, t, tp;

    P00 = _bncavx512_ftwo_prod(x[0], y[0], &E00);
    P01 = _mm512_mul_ps(x[0], y[1]);
    P10 = _mm512_mul_ps(x[1], y[0]);
    l   = _mm512_add_ps(P01, P10);
    v   = _mm512_add_ps(E00, c[1]);
    w   = _mm512_add_ps(v, l);
    s   = _bncavx512_ftwo_sum(P00, c[0], &t);
    tp  = _mm512_add_ps(t, w);
    z[0] = _bncavx512_fquick_two_sum(s, tp, &(z[1]));
}

static inline void _bncavx512_twfmaf(
        __m512       z[3],
        const __m512 x[3],
        const __m512 y[3],
        const __m512 c[3])
{
    __m512 P00, E00, P01, E01, P10, E10, P02, P11, P20, sg, G;
    __m512 A, q1, q2, q3, B, r, m1, m2, w0, w1, w2;

    P00 = _bncavx512_ftwo_prod(x[0], y[0], &E00);
    P01 = _bncavx512_ftwo_prod(x[0], y[1], &E01);
    P10 = _bncavx512_ftwo_prod(x[1], y[0], &E10);
    P02 = _mm512_mul_ps(x[0], y[2]);
    P11 = _mm512_mul_ps(x[1], y[1]);
    P20 = _mm512_mul_ps(x[2], y[0]);
    sg  = _mm512_add_ps(_mm512_add_ps(P02, P20), P11);
    G   = _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(E01, E10), sg), c[2]);
    A   = _bncavx512_ftwo_sum(P01, P10,  &q1);
    A   = _bncavx512_ftwo_sum(A,   E00,  &q2);
    A   = _bncavx512_ftwo_sum(A,   c[1], &q3);
    G   = _mm512_add_ps(G, _mm512_add_ps(_mm512_add_ps(q1, q2), q3));
    B   = _bncavx512_ftwo_sum(P00, c[0], &r);
    m1  = _bncavx512_ftwo_sum(r, A, &m2);
    m2  = _mm512_add_ps(m2, G);
    /* renorm pass 1/3 */
    w0 = _bncavx512_fquick_two_sum(B, m1, &w1);
    w1 = _bncavx512_ftwo_sum(w1, m2, &w2);
    /* renorm pass 2/3 */
    w0 = _bncavx512_ftwo_sum(w0, w1, &w1);
    w1 = _bncavx512_fquick_two_sum(w1, w2, &w2);
    /* renorm pass 3/3 */
    z[0] = _bncavx512_fquick_two_sum(w0, w1, &w1);
    z[1] = _bncavx512_fquick_two_sum(w1, w2, &(z[2]));
}

static inline void _bncavx512_qwfmaf(
        __m512       z[4],
        const __m512 x[4],
        const __m512 y[4],
        const __m512 c[4])
{
    __m512 P00, E00, P01, E01, P10, E10, P02, E02, P11, E11, P20, E20;
    __m512 P03, P12, P21, P30, D, B, r, Et;
    __m512 A1, f1, f2, f3, f4;
    __m512 A2, g1, g2, g3, g4, g5, g6, g7, g8, g9;
    __m512 A3, t1, t2, t3, t4;
    __m512 w0, w1, w2, w3;

    P00 = _bncavx512_ftwo_prod(x[0], y[0], &E00);
    P01 = _bncavx512_ftwo_prod(x[0], y[1], &E01);
    P10 = _bncavx512_ftwo_prod(x[1], y[0], &E10);
    P02 = _bncavx512_ftwo_prod(x[0], y[2], &E02);
    P11 = _bncavx512_ftwo_prod(x[1], y[1], &E11);
    P20 = _bncavx512_ftwo_prod(x[2], y[0], &E20);
    P03 = _mm512_mul_ps(x[0], y[3]);
    P12 = _mm512_mul_ps(x[1], y[2]);
    P21 = _mm512_mul_ps(x[2], y[1]);
    P30 = _mm512_mul_ps(x[3], y[0]);
    D   = _mm512_add_ps(_mm512_add_ps(P03, P30), _mm512_add_ps(P12, P21));
    B   = _bncavx512_ftwo_sum(P00, c[0], &r);

    /* L1 */
    A1  = _bncavx512_ftwo_sum(P01, P10,  &f1);
    A1  = _bncavx512_ftwo_sum(A1,  E00,  &f2);
    A1  = _bncavx512_ftwo_sum(A1,  c[1], &f3);
    A1  = _bncavx512_ftwo_sum(A1,  r,    &f4);

    /* L2 */
    A2  = _bncavx512_ftwo_sum(P02, P20,  &g1);
    A2  = _bncavx512_ftwo_sum(A2,  P11,  &g2);
    Et  = _bncavx512_ftwo_sum(E01, E10,  &g4);
    A2  = _bncavx512_ftwo_sum(A2,  Et,   &g3);
    A2  = _bncavx512_ftwo_sum(A2,  c[2], &g5);
    A2  = _bncavx512_ftwo_sum(A2,  f1,   &g6);
    A2  = _bncavx512_ftwo_sum(A2,  f2,   &g7);
    A2  = _bncavx512_ftwo_sum(A2,  f3,   &g8);
    A2  = _bncavx512_ftwo_sum(A2,  f4,   &g9);

    /* L3 */
    t1  = _mm512_add_ps(E02, E20);
    t2  = _mm512_add_ps(E11, D);
    t3  = _mm512_add_ps(_mm512_add_ps(t1, t2), c[3]);
    t1  = _mm512_add_ps(g1, g2);
    t2  = _mm512_add_ps(g3, g4);
    t1  = _mm512_add_ps(t1, t2);
    t2  = _mm512_add_ps(g6, g7);
    t4  = _mm512_add_ps(g8, g9);
    t2  = _mm512_add_ps(t2, t4);
    t1  = _mm512_add_ps(t1, t2);
    t1  = _mm512_add_ps(t1, g5);
    A3  = _mm512_add_ps(t3, t1);

    /* renorm pass 1/5 */
    w0 = _bncavx512_fquick_two_sum(B, A1, &w1);
    w1 = _bncavx512_ftwo_sum(w1, A2, &w2);
    w2 = _bncavx512_ftwo_sum(w2, A3, &w3);
    /* renorm pass 2/5 */
    w0 = _bncavx512_ftwo_sum(w0, w1, &w1);
    w1 = _bncavx512_ftwo_sum(w1, w2, &w2);
    w2 = _bncavx512_fquick_two_sum(w2, w3, &w3);
    /* renorm pass 3/5 */
    w0 = _bncavx512_ftwo_sum(w0, w1, &w1);
    w1 = _bncavx512_fquick_two_sum(w1, w2, &w2);
    w2 = _bncavx512_fquick_two_sum(w2, w3, &w3);
    /* renorm pass 4/5 */
    w0 = _bncavx512_fquick_two_sum(w0, w1, &w1);
    w1 = _bncavx512_fquick_two_sum(w1, w2, &w2);
    w2 = _bncavx512_fquick_two_sum(w2, w3, &w3);
    /* renorm pass 5/5 */
    z[0] = _bncavx512_fquick_two_sum(w0, w1, &w1);
    z[1] = _bncavx512_fquick_two_sum(w1, w2, &w2);
    z[2] = _bncavx512_fquick_two_sum(w2, w3, &(z[3]));
}

#endif /* defined(__AVX512F__) */

#endif /* __BNCAVX_FMA_H */
