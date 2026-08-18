/********************************************************************************/
/* _bncneon_fma.h : Arm NEON branch-free fused multiply-add for DW/TW/QW        */
/*                  z := x * y + c                                             */
/*                                                                              */
/* NEON vectorization of Algorithms 1-3 of                                      */
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
 * Primitive mapping (see _bncneon_deft.h / _bncneon_feft.h):
 *   two_sum(a,b,&s,&e)       -> s = _bncneon_dtwo_sum(a, b, &e)
 *   fast_two_sum(a,b,&s,&e)  -> s = _bncneon_dquick_two_sum(a, b, &e)
 *   two_prod(a,b,&p,&e)      -> p = _bncneon_dtwo_prod(a, b, &e)   (hardware FMA)
 *   plain a+b / a*b          -> vaddq_f64 / vmulq_f64
 *
 * 2 double lanes (float64x2_t) or 4 float lanes (float32x4_t) per call.
 */
#ifndef __BNCNEON_FMA_H
#define __BNCNEON_FMA_H

#if defined(__ARM_NEON)
#include <arm_neon.h>

/* error-free transformations used below (self-contained) */
#include "_bncneon_deft.h"
#include "_bncneon_feft.h"

/* ==============================================================================
 *  double base (DD / TD / QD)
 * ============================================================================== */

/* ------------------------------------------------------------------
 * _bncneon_dwfma : z[2] := x[2] * y[2] + c[2]   (K = 2, 17 flops)
 * ------------------------------------------------------------------ */
static inline void _bncneon_dwfma(
        float64x2_t       z[2],
        const float64x2_t x[2],
        const float64x2_t y[2],
        const float64x2_t c[2])
{
    float64x2_t P00, E00, P01, P10, l, v, w, s, t, tp;

    P00 = _bncneon_dtwo_prod(x[0], y[0], &E00);
    P01 = vmulq_f64(x[0], y[1]);
    P10 = vmulq_f64(x[1], y[0]);
    l   = vaddq_f64(P01, P10);
    v   = vaddq_f64(E00, c[1]);
    w   = vaddq_f64(v, l);
    s   = _bncneon_dtwo_sum(P00, c[0], &t);
    tp  = vaddq_f64(t, w);
    z[0] = _bncneon_dquick_two_sum(s, tp, &(z[1]));
}

/* ------------------------------------------------------------------
 * _bncneon_twfma : z[3] := x[3] * y[3] + c[3]   (K = 3, 66 flops)
 * ------------------------------------------------------------------ */
static inline void _bncneon_twfma(
        float64x2_t       z[3],
        const float64x2_t x[3],
        const float64x2_t y[3],
        const float64x2_t c[3])
{
    float64x2_t P00, E00, P01, E01, P10, E10, P02, P11, P20, sg, G;
    float64x2_t A, q1, q2, q3, B, r, m1, m2, w0, w1, w2;

    P00 = _bncneon_dtwo_prod(x[0], y[0], &E00);
    P01 = _bncneon_dtwo_prod(x[0], y[1], &E01);
    P10 = _bncneon_dtwo_prod(x[1], y[0], &E10);
    P02 = vmulq_f64(x[0], y[2]);
    P11 = vmulq_f64(x[1], y[1]);
    P20 = vmulq_f64(x[2], y[0]);
    sg  = vaddq_f64(vaddq_f64(P02, P20), P11);
    G   = vaddq_f64(vaddq_f64(vaddq_f64(E01, E10), sg), c[2]);
    A   = _bncneon_dtwo_sum(P01, P10,  &q1);
    A   = _bncneon_dtwo_sum(A,   E00,  &q2);
    A   = _bncneon_dtwo_sum(A,   c[1], &q3);
    G   = vaddq_f64(G, vaddq_f64(vaddq_f64(q1, q2), q3));
    B   = _bncneon_dtwo_sum(P00, c[0], &r);
    m1  = _bncneon_dtwo_sum(r, A, &m2);
    m2  = vaddq_f64(m2, G);
    /* renorm pass 1/3 */
    w0 = _bncneon_dquick_two_sum(B, m1, &w1);
    w1 = _bncneon_dtwo_sum(w1, m2, &w2);
    /* renorm pass 2/3 */
    w0 = _bncneon_dtwo_sum(w0, w1, &w1);
    w1 = _bncneon_dquick_two_sum(w1, w2, &w2);
    /* renorm pass 3/3 */
    z[0] = _bncneon_dquick_two_sum(w0, w1, &w1);
    z[1] = _bncneon_dquick_two_sum(w1, w2, &(z[2]));
}

/* ------------------------------------------------------------------
 * _bncneon_qwfma : z[4] := x[4] * y[4] + c[4]   (K = 4, 146 flops)
 * ------------------------------------------------------------------ */
static inline void _bncneon_qwfma(
        float64x2_t       z[4],
        const float64x2_t x[4],
        const float64x2_t y[4],
        const float64x2_t c[4])
{
    float64x2_t P00, E00, P01, E01, P10, E10, P02, E02, P11, E11, P20, E20;
    float64x2_t P03, P12, P21, P30, D, B, r, Et;
    float64x2_t A1, f1, f2, f3, f4;
    float64x2_t A2, g1, g2, g3, g4, g5, g6, g7, g8, g9;
    float64x2_t A3, t1, t2, t3, t4;
    float64x2_t w0, w1, w2, w3;

    P00 = _bncneon_dtwo_prod(x[0], y[0], &E00);
    P01 = _bncneon_dtwo_prod(x[0], y[1], &E01);
    P10 = _bncneon_dtwo_prod(x[1], y[0], &E10);
    P02 = _bncneon_dtwo_prod(x[0], y[2], &E02);
    P11 = _bncneon_dtwo_prod(x[1], y[1], &E11);
    P20 = _bncneon_dtwo_prod(x[2], y[0], &E20);
    P03 = vmulq_f64(x[0], y[3]);
    P12 = vmulq_f64(x[1], y[2]);
    P21 = vmulq_f64(x[2], y[1]);
    P30 = vmulq_f64(x[3], y[0]);
    D   = vaddq_f64(vaddq_f64(P03, P30), vaddq_f64(P12, P21));
    B   = _bncneon_dtwo_sum(P00, c[0], &r);

    /* L1 */
    A1  = _bncneon_dtwo_sum(P01, P10,  &f1);
    A1  = _bncneon_dtwo_sum(A1,  E00,  &f2);
    A1  = _bncneon_dtwo_sum(A1,  c[1], &f3);
    A1  = _bncneon_dtwo_sum(A1,  r,    &f4);

    /* L2 */
    A2  = _bncneon_dtwo_sum(P02, P20,  &g1);
    A2  = _bncneon_dtwo_sum(A2,  P11,  &g2);
    Et  = _bncneon_dtwo_sum(E01, E10,  &g4);
    A2  = _bncneon_dtwo_sum(A2,  Et,   &g3);
    A2  = _bncneon_dtwo_sum(A2,  c[2], &g5);
    A2  = _bncneon_dtwo_sum(A2,  f1,   &g6);
    A2  = _bncneon_dtwo_sum(A2,  f2,   &g7);
    A2  = _bncneon_dtwo_sum(A2,  f3,   &g8);
    A2  = _bncneon_dtwo_sum(A2,  f4,   &g9);

    /* L3 */
    t1  = vaddq_f64(E02, E20);
    t2  = vaddq_f64(E11, D);
    t3  = vaddq_f64(vaddq_f64(t1, t2), c[3]);
    t1  = vaddq_f64(g1, g2);
    t2  = vaddq_f64(g3, g4);
    t1  = vaddq_f64(t1, t2);
    t2  = vaddq_f64(g6, g7);
    t4  = vaddq_f64(g8, g9);
    t2  = vaddq_f64(t2, t4);
    t1  = vaddq_f64(t1, t2);
    t1  = vaddq_f64(t1, g5);
    A3  = vaddq_f64(t3, t1);

    /* renormalization */
    /* renorm pass 1/5 */
    w0 = _bncneon_dquick_two_sum(B, A1, &w1);
    w1 = _bncneon_dtwo_sum(w1, A2, &w2);
    w2 = _bncneon_dtwo_sum(w2, A3, &w3);
    /* renorm pass 2/5 */
    w0 = _bncneon_dtwo_sum(w0, w1, &w1);
    w1 = _bncneon_dtwo_sum(w1, w2, &w2);
    w2 = _bncneon_dquick_two_sum(w2, w3, &w3);
    /* renorm pass 3/5 */
    w0 = _bncneon_dtwo_sum(w0, w1, &w1);
    w1 = _bncneon_dquick_two_sum(w1, w2, &w2);
    w2 = _bncneon_dquick_two_sum(w2, w3, &w3);
    /* renorm pass 4/5 */
    w0 = _bncneon_dquick_two_sum(w0, w1, &w1);
    w1 = _bncneon_dquick_two_sum(w1, w2, &w2);
    w2 = _bncneon_dquick_two_sum(w2, w3, &w3);
    /* renorm pass 5/5 */
    z[0] = _bncneon_dquick_two_sum(w0, w1, &w1);
    z[1] = _bncneon_dquick_two_sum(w1, w2, &w2);
    z[2] = _bncneon_dquick_two_sum(w2, w3, &(z[3]));
}

/* ==============================================================================
 *  float base (DS / TS / QS) -- 4 lanes per call
 * ============================================================================== */

static inline void _bncneon_dwfmaf(
        float32x4_t       z[2],
        const float32x4_t x[2],
        const float32x4_t y[2],
        const float32x4_t c[2])
{
    float32x4_t P00, E00, P01, P10, l, v, w, s, t, tp;

    P00 = _bncneon_ftwo_prod(x[0], y[0], &E00);
    P01 = vmulq_f32(x[0], y[1]);
    P10 = vmulq_f32(x[1], y[0]);
    l   = vaddq_f32(P01, P10);
    v   = vaddq_f32(E00, c[1]);
    w   = vaddq_f32(v, l);
    s   = _bncneon_ftwo_sum(P00, c[0], &t);
    tp  = vaddq_f32(t, w);
    z[0] = _bncneon_fquick_two_sum(s, tp, &(z[1]));
}

static inline void _bncneon_twfmaf(
        float32x4_t       z[3],
        const float32x4_t x[3],
        const float32x4_t y[3],
        const float32x4_t c[3])
{
    float32x4_t P00, E00, P01, E01, P10, E10, P02, P11, P20, sg, G;
    float32x4_t A, q1, q2, q3, B, r, m1, m2, w0, w1, w2;

    P00 = _bncneon_ftwo_prod(x[0], y[0], &E00);
    P01 = _bncneon_ftwo_prod(x[0], y[1], &E01);
    P10 = _bncneon_ftwo_prod(x[1], y[0], &E10);
    P02 = vmulq_f32(x[0], y[2]);
    P11 = vmulq_f32(x[1], y[1]);
    P20 = vmulq_f32(x[2], y[0]);
    sg  = vaddq_f32(vaddq_f32(P02, P20), P11);
    G   = vaddq_f32(vaddq_f32(vaddq_f32(E01, E10), sg), c[2]);
    A   = _bncneon_ftwo_sum(P01, P10,  &q1);
    A   = _bncneon_ftwo_sum(A,   E00,  &q2);
    A   = _bncneon_ftwo_sum(A,   c[1], &q3);
    G   = vaddq_f32(G, vaddq_f32(vaddq_f32(q1, q2), q3));
    B   = _bncneon_ftwo_sum(P00, c[0], &r);
    m1  = _bncneon_ftwo_sum(r, A, &m2);
    m2  = vaddq_f32(m2, G);
    /* renorm pass 1/3 */
    w0 = _bncneon_fquick_two_sum(B, m1, &w1);
    w1 = _bncneon_ftwo_sum(w1, m2, &w2);
    /* renorm pass 2/3 */
    w0 = _bncneon_ftwo_sum(w0, w1, &w1);
    w1 = _bncneon_fquick_two_sum(w1, w2, &w2);
    /* renorm pass 3/3 */
    z[0] = _bncneon_fquick_two_sum(w0, w1, &w1);
    z[1] = _bncneon_fquick_two_sum(w1, w2, &(z[2]));
}

static inline void _bncneon_qwfmaf(
        float32x4_t       z[4],
        const float32x4_t x[4],
        const float32x4_t y[4],
        const float32x4_t c[4])
{
    float32x4_t P00, E00, P01, E01, P10, E10, P02, E02, P11, E11, P20, E20;
    float32x4_t P03, P12, P21, P30, D, B, r, Et;
    float32x4_t A1, f1, f2, f3, f4;
    float32x4_t A2, g1, g2, g3, g4, g5, g6, g7, g8, g9;
    float32x4_t A3, t1, t2, t3, t4;
    float32x4_t w0, w1, w2, w3;

    P00 = _bncneon_ftwo_prod(x[0], y[0], &E00);
    P01 = _bncneon_ftwo_prod(x[0], y[1], &E01);
    P10 = _bncneon_ftwo_prod(x[1], y[0], &E10);
    P02 = _bncneon_ftwo_prod(x[0], y[2], &E02);
    P11 = _bncneon_ftwo_prod(x[1], y[1], &E11);
    P20 = _bncneon_ftwo_prod(x[2], y[0], &E20);
    P03 = vmulq_f32(x[0], y[3]);
    P12 = vmulq_f32(x[1], y[2]);
    P21 = vmulq_f32(x[2], y[1]);
    P30 = vmulq_f32(x[3], y[0]);
    D   = vaddq_f32(vaddq_f32(P03, P30), vaddq_f32(P12, P21));
    B   = _bncneon_ftwo_sum(P00, c[0], &r);

    A1  = _bncneon_ftwo_sum(P01, P10,  &f1);
    A1  = _bncneon_ftwo_sum(A1,  E00,  &f2);
    A1  = _bncneon_ftwo_sum(A1,  c[1], &f3);
    A1  = _bncneon_ftwo_sum(A1,  r,    &f4);

    A2  = _bncneon_ftwo_sum(P02, P20,  &g1);
    A2  = _bncneon_ftwo_sum(A2,  P11,  &g2);
    Et  = _bncneon_ftwo_sum(E01, E10,  &g4);
    A2  = _bncneon_ftwo_sum(A2,  Et,   &g3);
    A2  = _bncneon_ftwo_sum(A2,  c[2], &g5);
    A2  = _bncneon_ftwo_sum(A2,  f1,   &g6);
    A2  = _bncneon_ftwo_sum(A2,  f2,   &g7);
    A2  = _bncneon_ftwo_sum(A2,  f3,   &g8);
    A2  = _bncneon_ftwo_sum(A2,  f4,   &g9);

    t1  = vaddq_f32(E02, E20);
    t2  = vaddq_f32(E11, D);
    t3  = vaddq_f32(vaddq_f32(t1, t2), c[3]);
    t1  = vaddq_f32(g1, g2);
    t2  = vaddq_f32(g3, g4);
    t1  = vaddq_f32(t1, t2);
    t2  = vaddq_f32(g6, g7);
    t4  = vaddq_f32(g8, g9);
    t2  = vaddq_f32(t2, t4);
    t1  = vaddq_f32(t1, t2);
    t1  = vaddq_f32(t1, g5);
    A3  = vaddq_f32(t3, t1);

    /* renorm pass 1/5 */
    w0 = _bncneon_fquick_two_sum(B, A1, &w1);
    w1 = _bncneon_ftwo_sum(w1, A2, &w2);
    w2 = _bncneon_ftwo_sum(w2, A3, &w3);
    /* renorm pass 2/5 */
    w0 = _bncneon_ftwo_sum(w0, w1, &w1);
    w1 = _bncneon_ftwo_sum(w1, w2, &w2);
    w2 = _bncneon_fquick_two_sum(w2, w3, &w3);
    /* renorm pass 3/5 */
    w0 = _bncneon_ftwo_sum(w0, w1, &w1);
    w1 = _bncneon_fquick_two_sum(w1, w2, &w2);
    w2 = _bncneon_fquick_two_sum(w2, w3, &w3);
    /* renorm pass 4/5 */
    w0 = _bncneon_fquick_two_sum(w0, w1, &w1);
    w1 = _bncneon_fquick_two_sum(w1, w2, &w2);
    w2 = _bncneon_fquick_two_sum(w2, w3, &w3);
    /* renorm pass 5/5 */
    z[0] = _bncneon_fquick_two_sum(w0, w1, &w1);
    z[1] = _bncneon_fquick_two_sum(w1, w2, &w2);
    z[2] = _bncneon_fquick_two_sum(w2, w3, &(z[3]));
}

#endif /* defined(__ARM_NEON) */
#endif /* __BNCNEON_FMA_H */
