/********************************************************************************/
/* _bncsve2_fma.h : Arm SVE2 branch-free fused multiply-add for DW/TW/QW        */
/*                  z := x * y + c                                             */
/*                                                                              */
/* SVE2 vectorization of Algorithms 1-3 of                                      */
/*   T. Kouya, "Performance evaluation of branch-free fused multiply-add        */
/*   algorithms for multi-component-type multiple-precision floating-point      */
/*   arithmetic", arXiv:2607.11391v1 (2026).                                    */
/* Operation-by-operation identical to the scalar reference (include/bncfma_d.h */
/* / bncfma_f.h) and to the NEON version, hence bitwise identical results.      */
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
 * Sizeless SVE types must NOT be array elements, so every component is a
 * separate variable / pointer argument (same convention as _bncsve2_dd.h).
 *
 * Primitive mapping (see _bncsve2_deft.h / _bncsve2_feft.h):
 *   two_sum      -> _bncsve2_dtwo_sum(pg, a, b, &e)
 *   fast_two_sum -> _bncsve2_dquick_two_sum(pg, a, b, &e)
 *   two_prod     -> _bncsve2_dtwo_prod(pg, a, b, &e)   (hardware FMA)
 *   plain +, *   -> svadd_f64_x, svmul_f64_x
 *
 * Requires -ffp-contract=off.
 */
#ifndef __BNCSVE2_FMA_H
#define __BNCSVE2_FMA_H

#if defined(__ARM_SVE2)
#include <arm_sve.h>

/* error-free transformations used below (self-contained) */
#include "_bncsve2_deft.h"
#include "_bncsve2_feft.h"

/* ==============================================================================
 *  double base (DD / TD / QD)
 * ============================================================================== */

/* (*z0,*z1) := (x0,x1) * (y0,y1) + (c0,c1)   -- DW-FMA, 17 flops */
static inline void _bncsve2_dwfma(
        svbool_t pg,
        svfloat64_t *z0, svfloat64_t *z1,
        svfloat64_t x0, svfloat64_t x1,
        svfloat64_t y0, svfloat64_t y1,
        svfloat64_t c0, svfloat64_t c1)
{
    svfloat64_t P00, E00, P01, P10, l, v, w, s, t, tp;

    P00 = _bncsve2_dtwo_prod(pg, x0, y0, &E00);
    P01 = svmul_f64_x(pg, x0, y1);
    P10 = svmul_f64_x(pg, x1, y0);
    l   = svadd_f64_x(pg, P01, P10);
    v   = svadd_f64_x(pg, E00, c1);
    w   = svadd_f64_x(pg, v, l);
    s   = _bncsve2_dtwo_sum(pg, P00, c0, &t);
    tp  = svadd_f64_x(pg, t, w);
    *z0 = _bncsve2_dquick_two_sum(pg, s, tp, z1);
}

/* (*z0,*z1,*z2) := x * y + c   -- TW-FMA, 66 flops */
static inline void _bncsve2_twfma(
        svbool_t pg,
        svfloat64_t *z0, svfloat64_t *z1, svfloat64_t *z2,
        svfloat64_t x0, svfloat64_t x1, svfloat64_t x2,
        svfloat64_t y0, svfloat64_t y1, svfloat64_t y2,
        svfloat64_t c0, svfloat64_t c1, svfloat64_t c2)
{
    svfloat64_t P00, E00, P01, E01, P10, E10, P02, P11, P20, sg, G;
    svfloat64_t A, q1, q2, q3, B, r, m1, m2, w0, w1, w2;

    P00 = _bncsve2_dtwo_prod(pg, x0, y0, &E00);
    P01 = _bncsve2_dtwo_prod(pg, x0, y1, &E01);
    P10 = _bncsve2_dtwo_prod(pg, x1, y0, &E10);
    P02 = svmul_f64_x(pg, x0, y2);
    P11 = svmul_f64_x(pg, x1, y1);
    P20 = svmul_f64_x(pg, x2, y0);
    sg  = svadd_f64_x(pg, svadd_f64_x(pg, P02, P20), P11);
    G   = svadd_f64_x(pg, svadd_f64_x(pg, svadd_f64_x(pg, E01, E10), sg), c2);
    A   = _bncsve2_dtwo_sum(pg, P01, P10, &q1);
    A   = _bncsve2_dtwo_sum(pg, A,   E00, &q2);
    A   = _bncsve2_dtwo_sum(pg, A,   c1,  &q3);
    G   = svadd_f64_x(pg, G, svadd_f64_x(pg, svadd_f64_x(pg, q1, q2), q3));
    B   = _bncsve2_dtwo_sum(pg, P00, c0, &r);
    m1  = _bncsve2_dtwo_sum(pg, r, A, &m2);
    m2  = svadd_f64_x(pg, m2, G);
    /* renorm pass 1/3 */
    w0 = _bncsve2_dquick_two_sum(pg, B, m1, &w1);
    w1 = _bncsve2_dtwo_sum(pg, w1, m2, &w2);
    /* renorm pass 2/3 */
    w0 = _bncsve2_dtwo_sum(pg, w0, w1, &w1);
    w1 = _bncsve2_dquick_two_sum(pg, w1, w2, &w2);
    /* renorm pass 3/3 */
    *z0 = _bncsve2_dquick_two_sum(pg, w0, w1, &w1);
    *z1 = _bncsve2_dquick_two_sum(pg, w1, w2, z2);
}

/* (*z0,..,*z3) := x * y + c   -- QW-FMA, 146 flops */
static inline void _bncsve2_qwfma(
        svbool_t pg,
        svfloat64_t *z0, svfloat64_t *z1, svfloat64_t *z2, svfloat64_t *z3,
        svfloat64_t x0, svfloat64_t x1, svfloat64_t x2, svfloat64_t x3,
        svfloat64_t y0, svfloat64_t y1, svfloat64_t y2, svfloat64_t y3,
        svfloat64_t c0, svfloat64_t c1, svfloat64_t c2, svfloat64_t c3)
{
    svfloat64_t P00, E00, P01, E01, P10, E10, P02, E02, P11, E11, P20, E20;
    svfloat64_t P03, P12, P21, P30, D, B, r, Et;
    svfloat64_t A1, f1, f2, f3, f4;
    svfloat64_t A2, g1, g2, g3, g4, g5, g6, g7, g8, g9;
    svfloat64_t A3, t1, t2, t3, t4;
    svfloat64_t w0, w1, w2, w3;

    P00 = _bncsve2_dtwo_prod(pg, x0, y0, &E00);
    P01 = _bncsve2_dtwo_prod(pg, x0, y1, &E01);
    P10 = _bncsve2_dtwo_prod(pg, x1, y0, &E10);
    P02 = _bncsve2_dtwo_prod(pg, x0, y2, &E02);
    P11 = _bncsve2_dtwo_prod(pg, x1, y1, &E11);
    P20 = _bncsve2_dtwo_prod(pg, x2, y0, &E20);
    P03 = svmul_f64_x(pg, x0, y3);
    P12 = svmul_f64_x(pg, x1, y2);
    P21 = svmul_f64_x(pg, x2, y1);
    P30 = svmul_f64_x(pg, x3, y0);
    D   = svadd_f64_x(pg, svadd_f64_x(pg, P03, P30), svadd_f64_x(pg, P12, P21));
    B   = _bncsve2_dtwo_sum(pg, P00, c0, &r);

    /* L1 */
    A1  = _bncsve2_dtwo_sum(pg, P01, P10, &f1);
    A1  = _bncsve2_dtwo_sum(pg, A1,  E00, &f2);
    A1  = _bncsve2_dtwo_sum(pg, A1,  c1,  &f3);
    A1  = _bncsve2_dtwo_sum(pg, A1,  r,   &f4);

    /* L2 */
    A2  = _bncsve2_dtwo_sum(pg, P02, P20, &g1);
    A2  = _bncsve2_dtwo_sum(pg, A2,  P11, &g2);
    Et  = _bncsve2_dtwo_sum(pg, E01, E10, &g4);
    A2  = _bncsve2_dtwo_sum(pg, A2,  Et,  &g3);
    A2  = _bncsve2_dtwo_sum(pg, A2,  c2,  &g5);
    A2  = _bncsve2_dtwo_sum(pg, A2,  f1,  &g6);
    A2  = _bncsve2_dtwo_sum(pg, A2,  f2,  &g7);
    A2  = _bncsve2_dtwo_sum(pg, A2,  f3,  &g8);
    A2  = _bncsve2_dtwo_sum(pg, A2,  f4,  &g9);

    /* L3 */
    t1  = svadd_f64_x(pg, E02, E20);
    t2  = svadd_f64_x(pg, E11, D);
    t3  = svadd_f64_x(pg, svadd_f64_x(pg, t1, t2), c3);
    t1  = svadd_f64_x(pg, g1, g2);
    t2  = svadd_f64_x(pg, g3, g4);
    t1  = svadd_f64_x(pg, t1, t2);
    t2  = svadd_f64_x(pg, g6, g7);
    t4  = svadd_f64_x(pg, g8, g9);
    t2  = svadd_f64_x(pg, t2, t4);
    t1  = svadd_f64_x(pg, t1, t2);
    t1  = svadd_f64_x(pg, t1, g5);
    A3  = svadd_f64_x(pg, t3, t1);

    /* renormalization */
    /* renorm pass 1/5 */
    w0 = _bncsve2_dquick_two_sum(pg, B, A1, &w1);
    w1 = _bncsve2_dtwo_sum(pg, w1, A2, &w2);
    w2 = _bncsve2_dtwo_sum(pg, w2, A3, &w3);
    /* renorm pass 2/5 */
    w0 = _bncsve2_dtwo_sum(pg, w0, w1, &w1);
    w1 = _bncsve2_dtwo_sum(pg, w1, w2, &w2);
    w2 = _bncsve2_dquick_two_sum(pg, w2, w3, &w3);
    /* renorm pass 3/5 */
    w0 = _bncsve2_dtwo_sum(pg, w0, w1, &w1);
    w1 = _bncsve2_dquick_two_sum(pg, w1, w2, &w2);
    w2 = _bncsve2_dquick_two_sum(pg, w2, w3, &w3);
    /* renorm pass 4/5 */
    w0 = _bncsve2_dquick_two_sum(pg, w0, w1, &w1);
    w1 = _bncsve2_dquick_two_sum(pg, w1, w2, &w2);
    w2 = _bncsve2_dquick_two_sum(pg, w2, w3, &w3);
    /* renorm pass 5/5 */
    *z0 = _bncsve2_dquick_two_sum(pg, w0, w1, &w1);
    *z1 = _bncsve2_dquick_two_sum(pg, w1, w2, &w2);
    *z2 = _bncsve2_dquick_two_sum(pg, w2, w3, z3);
}

/* ==============================================================================
 *  float base (DS / TS / QS)
 * ============================================================================== */

static inline void _bncsve2_dwfmaf(
        svbool_t pg,
        svfloat32_t *z0, svfloat32_t *z1,
        svfloat32_t x0, svfloat32_t x1,
        svfloat32_t y0, svfloat32_t y1,
        svfloat32_t c0, svfloat32_t c1)
{
    svfloat32_t P00, E00, P01, P10, l, v, w, s, t, tp;

    P00 = _bncsve2_ftwo_prod(pg, x0, y0, &E00);
    P01 = svmul_f32_x(pg, x0, y1);
    P10 = svmul_f32_x(pg, x1, y0);
    l   = svadd_f32_x(pg, P01, P10);
    v   = svadd_f32_x(pg, E00, c1);
    w   = svadd_f32_x(pg, v, l);
    s   = _bncsve2_ftwo_sum(pg, P00, c0, &t);
    tp  = svadd_f32_x(pg, t, w);
    *z0 = _bncsve2_fquick_two_sum(pg, s, tp, z1);
}

static inline void _bncsve2_twfmaf(
        svbool_t pg,
        svfloat32_t *z0, svfloat32_t *z1, svfloat32_t *z2,
        svfloat32_t x0, svfloat32_t x1, svfloat32_t x2,
        svfloat32_t y0, svfloat32_t y1, svfloat32_t y2,
        svfloat32_t c0, svfloat32_t c1, svfloat32_t c2)
{
    svfloat32_t P00, E00, P01, E01, P10, E10, P02, P11, P20, sg, G;
    svfloat32_t A, q1, q2, q3, B, r, m1, m2, w0, w1, w2;

    P00 = _bncsve2_ftwo_prod(pg, x0, y0, &E00);
    P01 = _bncsve2_ftwo_prod(pg, x0, y1, &E01);
    P10 = _bncsve2_ftwo_prod(pg, x1, y0, &E10);
    P02 = svmul_f32_x(pg, x0, y2);
    P11 = svmul_f32_x(pg, x1, y1);
    P20 = svmul_f32_x(pg, x2, y0);
    sg  = svadd_f32_x(pg, svadd_f32_x(pg, P02, P20), P11);
    G   = svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, E01, E10), sg), c2);
    A   = _bncsve2_ftwo_sum(pg, P01, P10, &q1);
    A   = _bncsve2_ftwo_sum(pg, A,   E00, &q2);
    A   = _bncsve2_ftwo_sum(pg, A,   c1,  &q3);
    G   = svadd_f32_x(pg, G, svadd_f32_x(pg, svadd_f32_x(pg, q1, q2), q3));
    B   = _bncsve2_ftwo_sum(pg, P00, c0, &r);
    m1  = _bncsve2_ftwo_sum(pg, r, A, &m2);
    m2  = svadd_f32_x(pg, m2, G);
    /* renorm pass 1/3 */
    w0 = _bncsve2_fquick_two_sum(pg, B, m1, &w1);
    w1 = _bncsve2_ftwo_sum(pg, w1, m2, &w2);
    /* renorm pass 2/3 */
    w0 = _bncsve2_ftwo_sum(pg, w0, w1, &w1);
    w1 = _bncsve2_fquick_two_sum(pg, w1, w2, &w2);
    /* renorm pass 3/3 */
    *z0 = _bncsve2_fquick_two_sum(pg, w0, w1, &w1);
    *z1 = _bncsve2_fquick_two_sum(pg, w1, w2, z2);
}

static inline void _bncsve2_qwfmaf(
        svbool_t pg,
        svfloat32_t *z0, svfloat32_t *z1, svfloat32_t *z2, svfloat32_t *z3,
        svfloat32_t x0, svfloat32_t x1, svfloat32_t x2, svfloat32_t x3,
        svfloat32_t y0, svfloat32_t y1, svfloat32_t y2, svfloat32_t y3,
        svfloat32_t c0, svfloat32_t c1, svfloat32_t c2, svfloat32_t c3)
{
    svfloat32_t P00, E00, P01, E01, P10, E10, P02, E02, P11, E11, P20, E20;
    svfloat32_t P03, P12, P21, P30, D, B, r, Et;
    svfloat32_t A1, f1, f2, f3, f4;
    svfloat32_t A2, g1, g2, g3, g4, g5, g6, g7, g8, g9;
    svfloat32_t A3, t1, t2, t3, t4;
    svfloat32_t w0, w1, w2, w3;

    P00 = _bncsve2_ftwo_prod(pg, x0, y0, &E00);
    P01 = _bncsve2_ftwo_prod(pg, x0, y1, &E01);
    P10 = _bncsve2_ftwo_prod(pg, x1, y0, &E10);
    P02 = _bncsve2_ftwo_prod(pg, x0, y2, &E02);
    P11 = _bncsve2_ftwo_prod(pg, x1, y1, &E11);
    P20 = _bncsve2_ftwo_prod(pg, x2, y0, &E20);
    P03 = svmul_f32_x(pg, x0, y3);
    P12 = svmul_f32_x(pg, x1, y2);
    P21 = svmul_f32_x(pg, x2, y1);
    P30 = svmul_f32_x(pg, x3, y0);
    D   = svadd_f32_x(pg, svadd_f32_x(pg, P03, P30), svadd_f32_x(pg, P12, P21));
    B   = _bncsve2_ftwo_sum(pg, P00, c0, &r);

    A1  = _bncsve2_ftwo_sum(pg, P01, P10, &f1);
    A1  = _bncsve2_ftwo_sum(pg, A1,  E00, &f2);
    A1  = _bncsve2_ftwo_sum(pg, A1,  c1,  &f3);
    A1  = _bncsve2_ftwo_sum(pg, A1,  r,   &f4);

    A2  = _bncsve2_ftwo_sum(pg, P02, P20, &g1);
    A2  = _bncsve2_ftwo_sum(pg, A2,  P11, &g2);
    Et  = _bncsve2_ftwo_sum(pg, E01, E10, &g4);
    A2  = _bncsve2_ftwo_sum(pg, A2,  Et,  &g3);
    A2  = _bncsve2_ftwo_sum(pg, A2,  c2,  &g5);
    A2  = _bncsve2_ftwo_sum(pg, A2,  f1,  &g6);
    A2  = _bncsve2_ftwo_sum(pg, A2,  f2,  &g7);
    A2  = _bncsve2_ftwo_sum(pg, A2,  f3,  &g8);
    A2  = _bncsve2_ftwo_sum(pg, A2,  f4,  &g9);

    t1  = svadd_f32_x(pg, E02, E20);
    t2  = svadd_f32_x(pg, E11, D);
    t3  = svadd_f32_x(pg, svadd_f32_x(pg, t1, t2), c3);
    t1  = svadd_f32_x(pg, g1, g2);
    t2  = svadd_f32_x(pg, g3, g4);
    t1  = svadd_f32_x(pg, t1, t2);
    t2  = svadd_f32_x(pg, g6, g7);
    t4  = svadd_f32_x(pg, g8, g9);
    t2  = svadd_f32_x(pg, t2, t4);
    t1  = svadd_f32_x(pg, t1, t2);
    t1  = svadd_f32_x(pg, t1, g5);
    A3  = svadd_f32_x(pg, t3, t1);

    /* renorm pass 1/5 */
    w0 = _bncsve2_fquick_two_sum(pg, B, A1, &w1);
    w1 = _bncsve2_ftwo_sum(pg, w1, A2, &w2);
    w2 = _bncsve2_ftwo_sum(pg, w2, A3, &w3);
    /* renorm pass 2/5 */
    w0 = _bncsve2_ftwo_sum(pg, w0, w1, &w1);
    w1 = _bncsve2_ftwo_sum(pg, w1, w2, &w2);
    w2 = _bncsve2_fquick_two_sum(pg, w2, w3, &w3);
    /* renorm pass 3/5 */
    w0 = _bncsve2_ftwo_sum(pg, w0, w1, &w1);
    w1 = _bncsve2_fquick_two_sum(pg, w1, w2, &w2);
    w2 = _bncsve2_fquick_two_sum(pg, w2, w3, &w3);
    /* renorm pass 4/5 */
    w0 = _bncsve2_fquick_two_sum(pg, w0, w1, &w1);
    w1 = _bncsve2_fquick_two_sum(pg, w1, w2, &w2);
    w2 = _bncsve2_fquick_two_sum(pg, w2, w3, &w3);
    /* renorm pass 5/5 */
    *z0 = _bncsve2_fquick_two_sum(pg, w0, w1, &w1);
    *z1 = _bncsve2_fquick_two_sum(pg, w1, w2, &w2);
    *z2 = _bncsve2_fquick_two_sum(pg, w2, w3, z3);
}

#endif /* defined(__ARM_SVE2) */
#endif /* __BNCSVE2_FMA_H */
