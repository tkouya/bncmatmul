// -------------------------------------------------------
// _bncsve2_qs.h  --  SVE2 Quad-Single (QS) per-limb-variable helpers
// -------------------------------------------------------
// Rewritten 2026.  QS = four svfloat32_t.  TS = three svfloat32_t.  Mirrors
// include/neon/_bncneon_qs.h.  This file also defines _bncsve2_rts_addq /
// _bncsve2_rts_mulq because those use _bncsve2_frenorm which lives here
// (mirrors the NEON pattern where qs.h hosts ts's renorm-based helpers).
// -------------------------------------------------------
#ifndef __BNCSVE2_QS_H
#define __BNCSVE2_QS_H

#if defined(__ARM_SVE2)
#include <arm_sve.h>

#ifndef QSSIZE
    #define QSSIZE 4
#endif
#ifndef TSSIZE
    #define TSSIZE 3
#endif

/* ============================================================
 *  set0 / set_f / neg for QS
 * ============================================================ */
static inline void _bncsve2_rqs_set0(svfloat32_t *r0, svfloat32_t *r1,
                                      svfloat32_t *r2, svfloat32_t *r3)
{
    *r0 = svdup_n_f32(0.0f);
    *r1 = svdup_n_f32(0.0f);
    *r2 = svdup_n_f32(0.0f);
    *r3 = svdup_n_f32(0.0f);
}

static inline void _bncsve2_rqs_set_f(svfloat32_t *r0, svfloat32_t *r1,
                                       svfloat32_t *r2, svfloat32_t *r3,
                                       svfloat32_t v)
{
    *r0 = v;
    *r1 = svdup_n_f32(0.0f);
    *r2 = svdup_n_f32(0.0f);
    *r3 = svdup_n_f32(0.0f);
}

static inline void _bncsve2_rqs_neg(svbool_t pg,
                                     svfloat32_t *r0, svfloat32_t *r1,
                                     svfloat32_t *r2, svfloat32_t *r3,
                                     svfloat32_t a0, svfloat32_t a1,
                                     svfloat32_t a2, svfloat32_t a3)
{
    *r0 = svneg_f32_x(pg, a0);
    *r1 = svneg_f32_x(pg, a1);
    *r2 = svneg_f32_x(pg, a2);
    *r3 = svneg_f32_x(pg, a3);
}

/* TS set0 / set_f / neg */
static inline void _bncsve2_rts_set0(svfloat32_t *r0, svfloat32_t *r1,
                                      svfloat32_t *r2)
{
    *r0 = svdup_n_f32(0.0f);
    *r1 = svdup_n_f32(0.0f);
    *r2 = svdup_n_f32(0.0f);
}

static inline void _bncsve2_rts_set_f(svfloat32_t *r0, svfloat32_t *r1,
                                       svfloat32_t *r2,
                                       svfloat32_t v)
{
    *r0 = v;
    *r1 = svdup_n_f32(0.0f);
    *r2 = svdup_n_f32(0.0f);
}

static inline void _bncsve2_rts_neg(svbool_t pg,
                                     svfloat32_t *r0, svfloat32_t *r1,
                                     svfloat32_t *r2,
                                     svfloat32_t a0, svfloat32_t a1,
                                     svfloat32_t a2)
{
    *r0 = svneg_f32_x(pg, a0);
    *r1 = svneg_f32_x(pg, a1);
    *r2 = svneg_f32_x(pg, a2);
}

/* ============================================================
 *  fthree_sum / fthree_sum2  (float versions of three_sum)
 * ============================================================ */
static inline void _bncsve2_fthree_sum(svbool_t pg,
                                        svfloat32_t *a, svfloat32_t *b,
                                        svfloat32_t *c)
{
    svfloat32_t t1, t2, t3;
    t1 = _bncsve2_ftwo_sum(pg, *a, *b, &t2);
    *a = _bncsve2_ftwo_sum(pg, *c, t1, &t3);
    *b = _bncsve2_ftwo_sum(pg, t2, t3, c);
}

static inline void _bncsve2_fthree_sum2(svbool_t pg,
                                         svfloat32_t *a, svfloat32_t *b,
                                         svfloat32_t *c)
{
    svfloat32_t t1, t2, t3;
    t1 = _bncsve2_ftwo_sum(pg, *a, *b, &t2);
    *a = _bncsve2_ftwo_sum(pg, *c, t1, &t3);
    *b = svadd_f32_x(pg, t2, t3);
}

/* ============================================================
 *  frenorm / frenorm4  (float, branch-free vectorized)
 *  Semantics match c_ds_qs.h's frenorm() / frenorm4() per lane.
 *  See _bncsve2_qd.h for the detailed comment-explanation.
 *  (NOTE: per-lane isinf early-return is NOT replicated.)
 * ============================================================ */
static inline void _bncsve2_frenorm(svbool_t pg,
                                     svfloat32_t *c0, svfloat32_t *c1,
                                     svfloat32_t *c2, svfloat32_t *c3)
{
    svfloat32_t zero = svdup_n_f32(0.0f);
    svfloat32_t s0, s1;
    svfloat32_t nc1, nc2, nc3;

    s0  = _bncsve2_fquick_two_sum(pg, *c2, *c3, &nc3);
    s0  = _bncsve2_fquick_two_sum(pg, *c1,  s0, &nc2);
    *c0 = _bncsve2_fquick_two_sum(pg, *c0,  s0, &nc1);

    s0 = *c0;
    s1 = nc1;

    /* Path A: s1 != 0 */
    svfloat32_t s2_a;
    svfloat32_t s1_a = _bncsve2_fquick_two_sum(pg, s1, nc2, &s2_a);
    svfloat32_t s3_a;
    svfloat32_t s2_A = _bncsve2_fquick_two_sum(pg, s2_a, nc3, &s3_a);  /* s2_a != 0 sub */
    svfloat32_t s2_b;
    svfloat32_t s1_B = _bncsve2_fquick_two_sum(pg, s1_a, nc3, &s2_b);  /* s2_a == 0 sub */

    /* Path B/D start */
    svfloat32_t s1_c;
    svfloat32_t s0_c = _bncsve2_fquick_two_sum(pg, s0, nc2, &s1_c);
    svfloat32_t s2_c;
    svfloat32_t s1_C = _bncsve2_fquick_two_sum(pg, s1_c, nc3, &s2_c);  /* s1_c != 0 sub */
    svfloat32_t s1_d;
    svfloat32_t s0_D = _bncsve2_fquick_two_sum(pg, s0_c, nc3, &s1_d);  /* s1_c == 0 sub */

    svbool_t cmp_s1  = svcmpne_f32(pg, s1, zero);
    svbool_t cmp_s2a = svcmpne_f32(pg, s2_a, zero);
    svbool_t cmp_s1c = svcmpne_f32(pg, s1_c, zero);

    svfloat32_t fin_s0 = svsel_f32(cmp_s1, s0, svsel_f32(cmp_s1c, s0_c, s0_D));
    svfloat32_t s1_when_s1nz = svsel_f32(cmp_s2a, s1_a, s1_B);
    svfloat32_t s1_when_s1z  = svsel_f32(cmp_s1c, s1_C, s1_d);
    svfloat32_t fin_s1 = svsel_f32(cmp_s1, s1_when_s1nz, s1_when_s1z);
    svfloat32_t s2_when_s1nz = svsel_f32(cmp_s2a, s2_A, s2_b);
    svfloat32_t s2_when_s1z  = svsel_f32(cmp_s1c, s2_c, zero);
    svfloat32_t fin_s2 = svsel_f32(cmp_s1, s2_when_s1nz, s2_when_s1z);
    svbool_t   path_A  = svand_b_z(pg, cmp_s1, cmp_s2a);
    svfloat32_t fin_s3 = svsel_f32(path_A, s3_a, zero);

    *c0 = fin_s0;
    *c1 = fin_s1;
    *c2 = fin_s2;
    *c3 = fin_s3;
}

static inline void _bncsve2_frenorm4(svbool_t pg,
                                      svfloat32_t *c0, svfloat32_t *c1,
                                      svfloat32_t *c2, svfloat32_t *c3,
                                      svfloat32_t *c4)
{
    svfloat32_t zero = svdup_n_f32(0.0f);
    svfloat32_t s0, s1;
    svfloat32_t nc1, nc2, nc3, nc4;

    s0  = _bncsve2_fquick_two_sum(pg, *c3, *c4, &nc4);
    s0  = _bncsve2_fquick_two_sum(pg, *c2,  s0, &nc3);
    s0  = _bncsve2_fquick_two_sum(pg, *c1,  s0, &nc2);
    *c0 = _bncsve2_fquick_two_sum(pg, *c0,  s0, &nc1);

    s0 = *c0;
    s1 = nc1;

    /* Path A: s1 != 0 */
    svfloat32_t s2_A1;
    svfloat32_t s1_A1 = _bncsve2_fquick_two_sum(pg, s1, nc2, &s2_A1);

    svfloat32_t s3_A11;
    svfloat32_t s2_A11 = _bncsve2_fquick_two_sum(pg, s2_A1, nc3, &s3_A11);
    svfloat32_t s3_A111 = svadd_f32_x(pg, s3_A11, nc4);
    svfloat32_t s3_A112;
    svfloat32_t s2_A112 = _bncsve2_fquick_two_sum(pg, s2_A11, nc4, &s3_A112);
    svbool_t cmp_s3A11 = svcmpne_f32(pg, s3_A11, zero);
    svfloat32_t s2_A11_final = svsel_f32(cmp_s3A11, s2_A11, s2_A112);
    svfloat32_t s3_A11_final = svsel_f32(cmp_s3A11, s3_A111, s3_A112);

    svfloat32_t s2_A2;
    svfloat32_t s1_A2  = _bncsve2_fquick_two_sum(pg, s1_A1, nc3, &s2_A2);
    svfloat32_t s3_A21;
    svfloat32_t s2_A21 = _bncsve2_fquick_two_sum(pg, s2_A2, nc4, &s3_A21);
    svfloat32_t s2_A22;
    svfloat32_t s1_A22 = _bncsve2_fquick_two_sum(pg, s1_A2, nc4, &s2_A22);
    svbool_t cmp_s2A2  = svcmpne_f32(pg, s2_A2, zero);
    svfloat32_t s1_A12_final = svsel_f32(cmp_s2A2, s1_A2,  s1_A22);
    svfloat32_t s2_A12_final = svsel_f32(cmp_s2A2, s2_A21, s2_A22);
    svfloat32_t s3_A12_final = svsel_f32(cmp_s2A2, s3_A21, zero);

    svbool_t cmp_s2A1 = svcmpne_f32(pg, s2_A1, zero);
    svfloat32_t s1_A_final = svsel_f32(cmp_s2A1, s1_A1,        s1_A12_final);
    svfloat32_t s2_A_final = svsel_f32(cmp_s2A1, s2_A11_final, s2_A12_final);
    svfloat32_t s3_A_final = svsel_f32(cmp_s2A1, s3_A11_final, s3_A12_final);
    svfloat32_t s0_A_final = s0;

    /* Path B: s1 == 0 */
    svfloat32_t s1_B1;
    svfloat32_t s0_B  = _bncsve2_fquick_two_sum(pg, s0, nc2, &s1_B1);

    svfloat32_t s2_B11;
    svfloat32_t s1_B11 = _bncsve2_fquick_two_sum(pg, s1_B1, nc3, &s2_B11);
    svfloat32_t s3_B111;
    svfloat32_t s2_B111 = _bncsve2_fquick_two_sum(pg, s2_B11, nc4, &s3_B111);
    svfloat32_t s2_B112;
    svfloat32_t s1_B112 = _bncsve2_fquick_two_sum(pg, s1_B11, nc4, &s2_B112);
    svbool_t cmp_s2B11 = svcmpne_f32(pg, s2_B11, zero);
    svfloat32_t s1_B11_final = svsel_f32(cmp_s2B11, s1_B11,  s1_B112);
    svfloat32_t s2_B11_final = svsel_f32(cmp_s2B11, s2_B111, s2_B112);
    svfloat32_t s3_B11_final = svsel_f32(cmp_s2B11, s3_B111, zero);

    svfloat32_t s1_B2;
    svfloat32_t s0_B2  = _bncsve2_fquick_two_sum(pg, s0_B, nc3, &s1_B2);
    svfloat32_t s2_B21;
    svfloat32_t s1_B21 = _bncsve2_fquick_two_sum(pg, s1_B2, nc4, &s2_B21);
    svfloat32_t s1_B22;
    svfloat32_t s0_B22 = _bncsve2_fquick_two_sum(pg, s0_B2, nc4, &s1_B22);
    svbool_t cmp_s1B2  = svcmpne_f32(pg, s1_B2, zero);
    svfloat32_t s0_B12_final = svsel_f32(cmp_s1B2, s0_B2,  s0_B22);
    svfloat32_t s1_B12_final = svsel_f32(cmp_s1B2, s1_B21, s1_B22);
    svfloat32_t s2_B12_final = svsel_f32(cmp_s1B2, s2_B21, zero);

    svbool_t cmp_s1B1 = svcmpne_f32(pg, s1_B1, zero);
    svfloat32_t s0_B_final = svsel_f32(cmp_s1B1, s0_B,         s0_B12_final);
    svfloat32_t s1_B_final = svsel_f32(cmp_s1B1, s1_B11_final, s1_B12_final);
    svfloat32_t s2_B_final = svsel_f32(cmp_s1B1, s2_B11_final, s2_B12_final);
    svfloat32_t s3_B_final = svsel_f32(cmp_s1B1, s3_B11_final, zero);

    svbool_t cmp_s1nz = svcmpne_f32(pg, s1, zero);
    *c0 = svsel_f32(cmp_s1nz, s0_A_final, s0_B_final);
    *c1 = svsel_f32(cmp_s1nz, s1_A_final, s1_B_final);
    *c2 = svsel_f32(cmp_s1nz, s2_A_final, s2_B_final);
    *c3 = svsel_f32(cmp_s1nz, s3_A_final, s3_B_final);
}

/* ============================================================
 *  QS addition / multiplication (sloppy default = frenorm4-based)
 * ============================================================ */

static inline void _bncsve2_rqs_add_sloppy(
        svbool_t pg,
        svfloat32_t *r0, svfloat32_t *r1, svfloat32_t *r2, svfloat32_t *r3,
        svfloat32_t a0, svfloat32_t a1, svfloat32_t a2, svfloat32_t a3,
        svfloat32_t b0, svfloat32_t b1, svfloat32_t b2, svfloat32_t b3)
{
    svfloat32_t s0 = svadd_f32_x(pg, a0, b0);
    svfloat32_t s1 = svadd_f32_x(pg, a1, b1);
    svfloat32_t s2 = svadd_f32_x(pg, a2, b2);
    svfloat32_t s3 = svadd_f32_x(pg, a3, b3);

    svfloat32_t v0 = svsub_f32_x(pg, s0, a0);
    svfloat32_t v1 = svsub_f32_x(pg, s1, a1);
    svfloat32_t v2 = svsub_f32_x(pg, s2, a2);
    svfloat32_t v3 = svsub_f32_x(pg, s3, a3);

    svfloat32_t u0 = svsub_f32_x(pg, s0, v0);
    svfloat32_t u1 = svsub_f32_x(pg, s1, v1);
    svfloat32_t u2 = svsub_f32_x(pg, s2, v2);
    svfloat32_t u3 = svsub_f32_x(pg, s3, v3);

    svfloat32_t w0 = svsub_f32_x(pg, a0, u0);
    svfloat32_t w1 = svsub_f32_x(pg, a1, u1);
    svfloat32_t w2 = svsub_f32_x(pg, a2, u2);
    svfloat32_t w3 = svsub_f32_x(pg, a3, u3);

    u0 = svsub_f32_x(pg, b0, v0);
    u1 = svsub_f32_x(pg, b1, v1);
    u2 = svsub_f32_x(pg, b2, v2);
    u3 = svsub_f32_x(pg, b3, v3);

    svfloat32_t t0 = svadd_f32_x(pg, w0, u0);
    svfloat32_t t1 = svadd_f32_x(pg, w1, u1);
    svfloat32_t t2 = svadd_f32_x(pg, w2, u2);
    svfloat32_t t3 = svadd_f32_x(pg, w3, u3);

    s1 = _bncsve2_ftwo_sum(pg, s1, t0, &t0);
    _bncsve2_fthree_sum (pg, &s2, &t0, &t1);
    _bncsve2_fthree_sum2(pg, &s3, &t0, &t2);   /* t2 is an INPUT here and is consumed by this gate */
    /* Bailey's sloppy QD/QS addition:  t0 = t0 + t1 + t3.
       2026-07-28 fix: t2 must NOT be added again here - it has already been
       absorbed by the fthree_sum2 gate above.  The stray "+ t2" double-counted
       an O(u^3) term and cost QS additions ~10 digits (max rel.err 1.3e-18
       instead of 1.6e-30).  Matches _bncsve2_rqd_add_sloppy (double),
       _bncneon_rqs_add_sloppy (NEON) and c_qs_add (scalar). */
    t0 = svadd_f32_x(pg, svadd_f32_x(pg, t0, t1), t3);

    _bncsve2_frenorm4(pg, &s0, &s1, &s2, &s3, &t0);
    *r0 = s0; *r1 = s1; *r2 = s2; *r3 = s3;
}

static inline void _bncsve2_rqs_mul_sloppy(
        svbool_t pg,
        svfloat32_t *r0, svfloat32_t *r1, svfloat32_t *r2, svfloat32_t *r3,
        svfloat32_t a0, svfloat32_t a1, svfloat32_t a2, svfloat32_t a3,
        svfloat32_t b0, svfloat32_t b1, svfloat32_t b2, svfloat32_t b3)
{
    svfloat32_t p0, p1, p2, p3, p4, p5;
    svfloat32_t q0, q1, q2, q3, q4, q5;
    svfloat32_t t0, t1;
    svfloat32_t s0, s1, s2;

    p0 = _bncsve2_ftwo_prod(pg, a0, b0, &q0);
    p1 = _bncsve2_ftwo_prod(pg, a0, b1, &q1);
    p2 = _bncsve2_ftwo_prod(pg, a1, b0, &q2);

    p3 = _bncsve2_ftwo_prod(pg, a0, b2, &q3);
    p4 = _bncsve2_ftwo_prod(pg, a1, b1, &q4);
    p5 = _bncsve2_ftwo_prod(pg, a2, b0, &q5);

    _bncsve2_fthree_sum(pg, &p1, &p2, &q0);

    _bncsve2_fthree_sum(pg, &p2, &q1, &q2);
    _bncsve2_fthree_sum(pg, &p3, &p4, &p5);

    s0 = _bncsve2_ftwo_sum(pg, p2, p3, &t0);
    s1 = _bncsve2_ftwo_sum(pg, q1, p4, &t1);
    s2 = svadd_f32_x(pg, q2, p5);
    s1 = _bncsve2_ftwo_sum(pg, s1, t0, &t0);
    s2 = svadd_f32_x(pg, s2, svadd_f32_x(pg, t0, t1));

    s1 = svmla_f32_x(pg, s1, a0, b3);
    s1 = svmla_f32_x(pg, s1, a1, b2);
    s1 = svmla_f32_x(pg, s1, a2, b1);
    s1 = svmla_f32_x(pg, s1, a3, b0);
    s1 = svadd_f32_x(pg, s1, q0);
    s1 = svadd_f32_x(pg, s1, q3);
    s1 = svadd_f32_x(pg, s1, q4);
    s1 = svadd_f32_x(pg, s1, q5);

    _bncsve2_frenorm4(pg, &p0, &p1, &s0, &s1, &s2);

    *r0 = p0; *r1 = p1; *r2 = s0; *r3 = s1;
}

/*
 * Branch-free QS add (mirrors NEON _bncneon_rqs_add_bf).
 * Float twin of _bncsve2_rqd_add_bf.
 */
static inline void _bncsve2_rqs_add_bf(
        svbool_t pg,
        svfloat32_t *r0, svfloat32_t *r1, svfloat32_t *r2, svfloat32_t *r3,
        svfloat32_t a0, svfloat32_t a1, svfloat32_t a2, svfloat32_t a3,
        svfloat32_t b0, svfloat32_t b1, svfloat32_t b2, svfloat32_t b3)
{
    svfloat32_t e_b1, e_d1, e_f1, e_h1;
    svfloat32_t s_a1 = _bncsve2_ftwo_sum(pg, a0, b0, &e_b1);
    svfloat32_t s_c1 = _bncsve2_ftwo_sum(pg, a1, b1, &e_d1);
    svfloat32_t s_e1 = _bncsve2_ftwo_sum(pg, a2, b2, &e_f1);
    svfloat32_t s_g1 = _bncsve2_ftwo_sum(pg, a3, b3, &e_h1);

    svfloat32_t e_c2;
    svfloat32_t s_a2 = _bncsve2_fquick_two_sum(pg, s_a1, s_c1, &e_c2);
    svfloat32_t s_b2 = svadd_f32_x(pg, e_b1, e_h1);
    svfloat32_t e_e2;
    svfloat32_t s_d2 = _bncsve2_ftwo_sum(pg, e_d1, s_e1, &e_e2);
    svfloat32_t e_g2;
    svfloat32_t s_f2 = _bncsve2_ftwo_sum(pg, e_f1, s_g1, &e_g2);

    svfloat32_t e_g3;
    svfloat32_t s_b3 = _bncsve2_ftwo_sum(pg, s_b2, e_g2, &e_g3);
    svfloat32_t e_d3;
    svfloat32_t s_c3 = _bncsve2_fquick_two_sum(pg, e_c2, s_d2, &e_d3);
    svfloat32_t e_f3;
    svfloat32_t s_e3 = _bncsve2_ftwo_sum(pg, e_e2, s_f2, &e_f3);

    svfloat32_t e_c4;
    svfloat32_t s_a4 = _bncsve2_fquick_two_sum(pg, s_a2, s_c3, &e_c4);
    svfloat32_t e_e4;
    svfloat32_t s_d4 = _bncsve2_fquick_two_sum(pg, e_d3, s_e3, &e_e4);

    svfloat32_t e_d5;
    svfloat32_t s_b5 = _bncsve2_ftwo_sum(pg, s_b3, s_d4, &e_d5);
    svfloat32_t s_e5 = svadd_f32_x(pg, e_e4, e_f3);

    svfloat32_t e_c6;
    svfloat32_t s_b6 = _bncsve2_ftwo_sum(pg, s_b5, e_c4, &e_c6);
    svfloat32_t e_e6;
    svfloat32_t s_d6 = _bncsve2_ftwo_sum(pg, e_d5, s_e5, &e_e6);

    svfloat32_t e_b7;
    svfloat32_t s_a7 = _bncsve2_fquick_two_sum(pg, s_a4, s_b6, &e_b7);
    svfloat32_t e_d7;
    svfloat32_t s_c7 = _bncsve2_fquick_two_sum(pg, e_c6, s_d6, &e_d7);

    svfloat32_t s_e8 = svadd_f32_x(pg, e_e6, e_g3);
    svfloat32_t e_c8;
    svfloat32_t s_b8 = _bncsve2_fquick_two_sum(pg, e_b7, s_c7, &e_c8);

    svfloat32_t s_d9 = svadd_f32_x(pg, e_d7, s_e8);
    svfloat32_t e_b10;
    svfloat32_t s_a10 = _bncsve2_fquick_two_sum(pg, s_a7, s_b8, &e_b10);
    svfloat32_t e_d10;
    svfloat32_t s_c10 = _bncsve2_fquick_two_sum(pg, e_c8, s_d9, &e_d10);

    svfloat32_t e_c11;
    svfloat32_t s_b11 = _bncsve2_fquick_two_sum(pg, e_b10, s_c10, &e_c11);
    svfloat32_t e_d12;
    svfloat32_t s_c12 = _bncsve2_fquick_two_sum(pg, e_c11, e_d10, &e_d12);

    *r0 = s_a10; *r1 = s_b11; *r2 = s_c12; *r3 = e_d12;
}

/*
 * Branch-free QS mul (mirrors NEON _bncneon_rqs_mul_bf).
 */
static inline void _bncsve2_rqs_mul_bf(
        svbool_t pg,
        svfloat32_t *r0, svfloat32_t *r1, svfloat32_t *r2, svfloat32_t *r3,
        svfloat32_t a0, svfloat32_t a1, svfloat32_t a2, svfloat32_t a3,
        svfloat32_t b0, svfloat32_t b1, svfloat32_t b2, svfloat32_t b3)
{
    svfloat32_t e_b0, e_e0, e_f0, e_j0, e_k0, e_l0;
    svfloat32_t s_a0 = _bncsve2_ftwo_prod(pg, a0, b0, &e_b0);
    svfloat32_t s_c0 = _bncsve2_ftwo_prod(pg, a0, b1, &e_e0);
    svfloat32_t s_d0 = _bncsve2_ftwo_prod(pg, a1, b0, &e_f0);
    svfloat32_t s_g0 = _bncsve2_ftwo_prod(pg, a0, b2, &e_j0);
    svfloat32_t s_h0 = _bncsve2_ftwo_prod(pg, a1, b1, &e_k0);
    svfloat32_t s_i0 = _bncsve2_ftwo_prod(pg, a2, b0, &e_l0);

    svfloat32_t s_m0 = svmul_f32_x(pg, a0, b3);
    svfloat32_t s_n0 = svmul_f32_x(pg, a1, b2);
    svfloat32_t s_o0 = svmul_f32_x(pg, a2, b1);
    svfloat32_t s_p0 = svmul_f32_x(pg, a3, b0);

    svfloat32_t e_d1;
    svfloat32_t s_c1 = _bncsve2_ftwo_sum(pg, s_c0, s_d0, &e_d1);
    svfloat32_t e_f1;
    svfloat32_t s_e1 = _bncsve2_ftwo_sum(pg, e_e0, e_f0, &e_f1);
    svfloat32_t e_i1;
    svfloat32_t s_g1 = _bncsve2_ftwo_sum(pg, s_g0, s_i0, &e_i1);

    svfloat32_t s_j1 = svadd_f32_x(pg, e_j0, e_l0);
    svfloat32_t s_m1 = svadd_f32_x(pg, s_m0, s_p0);
    svfloat32_t s_n1 = svadd_f32_x(pg, s_n0, s_o0);

    svfloat32_t e_c2;
    svfloat32_t s_b2 = _bncsve2_ftwo_sum(pg, e_b0, s_c1, &e_c2);
    svfloat32_t e_h2;
    svfloat32_t s_e2 = _bncsve2_ftwo_sum(pg, s_e1, s_h0, &e_h2);

    svfloat32_t s_f2 = svadd_f32_x(pg, e_f1, s_j1);
    svfloat32_t s_i2 = svadd_f32_x(pg, e_i1, e_k0);
    svfloat32_t s_m2 = svadd_f32_x(pg, s_m1, s_n1);

    svfloat32_t e_b3;
    svfloat32_t s_a3 = _bncsve2_fquick_two_sum(pg, s_a0, s_b2, &e_b3);
    svfloat32_t e_d3;
    svfloat32_t s_c3 = _bncsve2_fquick_two_sum(pg, e_c2, e_d1, &e_d3);
    svfloat32_t e_g3;
    svfloat32_t s_e3 = _bncsve2_ftwo_sum(pg, s_e2, s_g1, &e_g3);

    svfloat32_t s_f3 = svadd_f32_x(pg, s_f2, s_m2);
    svfloat32_t s_h3 = svadd_f32_x(pg, e_h2, s_i2);

    svfloat32_t e_e4;
    svfloat32_t s_c4 = _bncsve2_ftwo_sum(pg, s_c3, s_e3, &e_e4);
    svfloat32_t s_d4 = svadd_f32_x(pg, e_d3, s_h3);
    svfloat32_t s_f4 = svadd_f32_x(pg, s_f3, e_g3);

    svfloat32_t s_d5 = svadd_f32_x(pg, s_d4, e_e4);
    svfloat32_t e_d6;
    svfloat32_t s_c6 = _bncsve2_ftwo_sum(pg, s_c4, s_d5, &e_d6);

    svfloat32_t e_c7;
    svfloat32_t s_b7 = _bncsve2_ftwo_sum(pg, e_b3, s_c6, &e_c7);
    svfloat32_t s_d7 = svadd_f32_x(pg, e_d6, s_f4);

    svfloat32_t e_b8;
    svfloat32_t s_a8 = _bncsve2_fquick_two_sum(pg, s_a3, s_b7, &e_b8);
    svfloat32_t e_d8;
    svfloat32_t s_c8 = _bncsve2_ftwo_sum(pg, e_c7, s_d7, &e_d8);

    svfloat32_t e_c9;
    svfloat32_t s_b9 = _bncsve2_ftwo_sum(pg, e_b8, s_c8, &e_c9);
    svfloat32_t e_d10;
    svfloat32_t s_c10 = _bncsve2_fquick_two_sum(pg, e_c9, e_d8, &e_d10);

    *r0 = s_a8; *r1 = s_b9; *r2 = s_c10; *r3 = e_d10;
}

/* TS addition / multiplication (renorm-based "q" variant — the default) */
static inline void _bncsve2_rts_addq(
        svbool_t pg,
        svfloat32_t *r0, svfloat32_t *r1, svfloat32_t *r2,
        svfloat32_t a0, svfloat32_t a1, svfloat32_t a2,
        svfloat32_t b0, svfloat32_t b1, svfloat32_t b2)
{
    svfloat32_t s0 = svadd_f32_x(pg, a0, b0);
    svfloat32_t s1 = svadd_f32_x(pg, a1, b1);
    svfloat32_t s2 = svadd_f32_x(pg, a2, b2);

    svfloat32_t v0 = svsub_f32_x(pg, s0, a0);
    svfloat32_t v1 = svsub_f32_x(pg, s1, a1);
    svfloat32_t v2 = svsub_f32_x(pg, s2, a2);

    svfloat32_t u0 = svsub_f32_x(pg, s0, v0);
    svfloat32_t u1 = svsub_f32_x(pg, s1, v1);
    svfloat32_t u2 = svsub_f32_x(pg, s2, v2);

    svfloat32_t w0 = svsub_f32_x(pg, a0, u0);
    svfloat32_t w1 = svsub_f32_x(pg, a1, u1);
    svfloat32_t w2 = svsub_f32_x(pg, a2, u2);

    u0 = svsub_f32_x(pg, b0, v0);
    u1 = svsub_f32_x(pg, b1, v1);
    u2 = svsub_f32_x(pg, b2, v2);

    svfloat32_t t0 = svadd_f32_x(pg, w0, u0);
    svfloat32_t t1 = svadd_f32_x(pg, w1, u1);
    svfloat32_t t2 = svadd_f32_x(pg, w2, u2);

    s1 = _bncsve2_ftwo_sum(pg, s1, t0, &t0);
    _bncsve2_fthree_sum(pg, &s2, &t0, &t1);
    t0 = svadd_f32_x(pg, svadd_f32_x(pg, t0, t1), t2);

    _bncsve2_frenorm(pg, &s0, &s1, &s2, &t0);
    *r0 = s0; *r1 = s1; *r2 = s2;
}

static inline void _bncsve2_rts_mulq(
        svbool_t pg,
        svfloat32_t *r0, svfloat32_t *r1, svfloat32_t *r2,
        svfloat32_t a0, svfloat32_t a1, svfloat32_t a2,
        svfloat32_t b0, svfloat32_t b1, svfloat32_t b2)
{
    svfloat32_t p0, p1, p2, p3, p4, p5;
    svfloat32_t q0, q1, q2, q3, q4;
    svfloat32_t q5_unused;
    svfloat32_t t0, t1;
    svfloat32_t s0, s1;

    p0 = _bncsve2_ftwo_prod(pg, a0, b0, &q0);
    p1 = _bncsve2_ftwo_prod(pg, a0, b1, &q1);
    p2 = _bncsve2_ftwo_prod(pg, a1, b0, &q2);

    p3 = _bncsve2_ftwo_prod(pg, a0, b2, &q3);
    p4 = _bncsve2_ftwo_prod(pg, a1, b1, &q4);
    p5 = _bncsve2_ftwo_prod(pg, a2, b0, &q5_unused);
    (void)q5_unused;

    _bncsve2_fthree_sum(pg, &p1, &p2, &q0);

    _bncsve2_fthree_sum2(pg, &p2, &q1, &q2);
    _bncsve2_fthree_sum2(pg, &p3, &p4, &p5);

    s0 = _bncsve2_ftwo_sum(pg, p2, p3, &t0);
    s1 = _bncsve2_ftwo_sum(pg, q1, p4, &t1);
    s1 = _bncsve2_ftwo_sum(pg, s1, t0, &t0);

    s1 = svmla_f32_x(pg, s1, a1, b2);
    s1 = svmla_f32_x(pg, s1, a2, b1);
    s1 = svadd_f32_x(pg, s1, q0);
    s1 = svadd_f32_x(pg, s1, q3);
    s1 = svadd_f32_x(pg, s1, q4);

    _bncsve2_frenorm(pg, &p0, &p1, &s0, &s1);
    *r0 = p0; *r1 = p1; *r2 = s0;
}

/*
 * Branch-free TS add (mirrors NEON _bncneon_rts_add_bf).
 * Float-version twin of _bncsve2_rtd_add_bf — uses ftwo_sum / fquick_two_sum.
 */
static inline void _bncsve2_rts_add_bf(
        svbool_t pg,
        svfloat32_t *r0, svfloat32_t *r1, svfloat32_t *r2,
        svfloat32_t a0, svfloat32_t a1, svfloat32_t a2,
        svfloat32_t b0, svfloat32_t b1, svfloat32_t b2)
{
    svfloat32_t b1_e, d1, f1;
    svfloat32_t a_s0 = _bncsve2_ftwo_sum(pg, a0, b0, &b1_e);
    svfloat32_t c_s1 = _bncsve2_ftwo_sum(pg, a1, b1, &d1);
    svfloat32_t e_s1 = _bncsve2_ftwo_sum(pg, a2, b2, &f1);

    svfloat32_t c2;
    svfloat32_t a_s2 = _bncsve2_fquick_two_sum(pg, a_s0, c_s1, &c2);
    svfloat32_t b_s2 = svadd_f32_x(pg, b1_e, f1);
    svfloat32_t e2;
    svfloat32_t d_s2 = _bncsve2_ftwo_sum(pg, d1, e_s1, &e2);

    svfloat32_t d3;
    svfloat32_t a_s3 = _bncsve2_fquick_two_sum(pg, a_s2, d_s2, &d3);
    svfloat32_t c3;
    svfloat32_t b_s3 = _bncsve2_ftwo_sum(pg, b_s2, c2, &c3);

    svfloat32_t c4 = svadd_f32_x(pg, c3, e2);
    svfloat32_t d5;
    svfloat32_t c_s5 = _bncsve2_ftwo_sum(pg, c4, d3, &d5);

    svfloat32_t c6;
    svfloat32_t b_s6 = _bncsve2_ftwo_sum(pg, b_s3, c_s5, &c6);
    svfloat32_t b7;
    svfloat32_t a_s7 = _bncsve2_fquick_two_sum(pg, a_s3, b_s6, &b7);

    svfloat32_t c7 = svadd_f32_x(pg, c6, d5);
    svfloat32_t c_s8;
    svfloat32_t b_s8 = _bncsve2_fquick_two_sum(pg, b7, c7, &c_s8);

    *r0 = a_s7;
    *r1 = b_s8;
    *r2 = c_s8;
}

/*
 * Branch-free TS mul (mirrors NEON _bncneon_rts_mul_bf).
 */
static inline void _bncsve2_rts_mul_bf(
        svbool_t pg,
        svfloat32_t *r0, svfloat32_t *r1, svfloat32_t *r2,
        svfloat32_t a0, svfloat32_t a1, svfloat32_t a2,
        svfloat32_t b0, svfloat32_t b1, svfloat32_t b2)
{
    svfloat32_t b0_e, e0, f0;
    svfloat32_t a_p0 = _bncsve2_ftwo_prod(pg, a0, b0, &b0_e);
    svfloat32_t c_p0 = _bncsve2_ftwo_prod(pg, a0, b1, &e0);
    svfloat32_t d_p0 = _bncsve2_ftwo_prod(pg, a1, b0, &f0);

    svfloat32_t g0 = svmul_f32_x(pg, a0, b2);
    svfloat32_t h0 = svmul_f32_x(pg, a1, b1);
    svfloat32_t i0 = svmul_f32_x(pg, a2, b0);

    svfloat32_t d1;
    svfloat32_t c_s1 = _bncsve2_ftwo_sum(pg, c_p0, d_p0, &d1);
    svfloat32_t e1 = svadd_f32_x(pg, e0, f0);
    svfloat32_t g1 = svadd_f32_x(pg, g0, i0);

    svfloat32_t c2;
    svfloat32_t b_s2 = _bncsve2_ftwo_sum(pg, b0_e, c_s1, &c2);
    svfloat32_t g2 = svadd_f32_x(pg, g1, h0);

    svfloat32_t b3;
    svfloat32_t a_s3 = _bncsve2_fquick_two_sum(pg, a_p0, b_s2, &b3);
    svfloat32_t c3 = svadd_f32_x(pg, c2, d1);
    svfloat32_t e3 = svadd_f32_x(pg, e1, g2);

    svfloat32_t c4 = svadd_f32_x(pg, c3, e3);
    svfloat32_t c5;
    svfloat32_t b_s5 = _bncsve2_fquick_two_sum(pg, b3, c4, &c5);

    svfloat32_t b6;
    svfloat32_t a_s6 = _bncsve2_fquick_two_sum(pg, a_s3, b_s5, &b6);
    svfloat32_t c7;
    svfloat32_t b_s7 = _bncsve2_fquick_two_sum(pg, b6, c5, &c7);

    *r0 = a_s6;
    *r1 = b_s7;
    *r2 = c7;
}

/* Default-variant macros (TS/QS).
 *  - QS: USE_QS_BF -> _bncsve2_rqs_{add,mul}_bf (branch-free, mirrors
 *        NEON BF variant; shorter dependency chain than _sloppy + renorm5).
 *        Otherwise -> _bncsve2_rqs_{add,mul}_sloppy.
 *  - TS: USE_TS_BF -> _bncsve2_rts_{add,mul}_bf (branch-free, mirrors
 *        NEON BF variant).  Otherwise -> _bncsve2_rts_{addq,mulq}
 *        (renorm-based, slower but tighter error bound).
 *  BF variants are recommended for matvec/matmul where the long renorm
 *  dependency chain dominates. */
#ifndef _bncsve2_rqs_add
    #ifdef USE_QS_BF
        #define _bncsve2_rqs_add  _bncsve2_rqs_add_bf
    #else
        #define _bncsve2_rqs_add  _bncsve2_rqs_add_sloppy
    #endif
#endif
#ifndef _bncsve2_rqs_mul
    #ifdef USE_QS_BF
        #define _bncsve2_rqs_mul  _bncsve2_rqs_mul_bf
    #else
        #define _bncsve2_rqs_mul  _bncsve2_rqs_mul_sloppy
    #endif
#endif
#ifndef _bncsve2_rts_add
    #ifdef USE_TS_BF
        #define _bncsve2_rts_add  _bncsve2_rts_add_bf
    #else
        #define _bncsve2_rts_add  _bncsve2_rts_addq
    #endif
#endif
#ifndef _bncsve2_rts_mul
    #ifdef USE_TS_BF
        #define _bncsve2_rts_mul  _bncsve2_rts_mul_bf
    #else
        #define _bncsve2_rts_mul  _bncsve2_rts_mulq
    #endif
#endif

/* -------- horizontal EFT reductions (mirror _bncneon_r{ts,qs}_*128) -------- */
#ifndef _BNC_SVE_MAXLEN_S
#define _BNC_SVE_MAXLEN_S 64   /* SVE max vector = 2048 bit => 64 floats */
#endif

/* ---- TS (3 limbs) ---- */
static inline void _bncsve2_rts_abs(svbool_t pg,
                                    svfloat32_t *r0, svfloat32_t *r1, svfloat32_t *r2,
                                    svfloat32_t a0, svfloat32_t a1, svfloat32_t a2)
{
    svbool_t neg = svcmplt_f32(pg, a0, svdup_n_f32(0.0f));
    *r0 = svsel_f32(neg, svneg_f32_x(pg, a0), a0);
    *r1 = svsel_f32(neg, svneg_f32_x(pg, a1), a1);
    *r2 = svsel_f32(neg, svneg_f32_x(pg, a2), a2);
}

static inline void _bncsve2_rts_sum128(svbool_t pg, float ret[TSSIZE],
                                       svfloat32_t a0, svfloat32_t a1, svfloat32_t a2)
{
    long int n = (long int)svcntw(), k;
    float b0[_BNC_SVE_MAXLEN_S], b1[_BNC_SVE_MAXLEN_S], b2[_BNC_SVE_MAXLEN_S], v[TSSIZE];
    (void)pg;
    svst1_f32(svptrue_b32(), b0, a0);
    svst1_f32(svptrue_b32(), b1, a1);
    svst1_f32(svptrue_b32(), b2, a2);
    v[0]=b0[0]; v[1]=b1[0]; v[2]=b2[0];
    rts_set(ret, v);
    for(k = 1; k < n; k++){ v[0]=b0[k]; v[1]=b1[k]; v[2]=b2[k]; rts_add(ret, ret, v); }
}

static inline void _bncsve2_rts_abssum128(svbool_t pg, float ret[TSSIZE],
                                          svfloat32_t a0, svfloat32_t a1, svfloat32_t a2)
{
    _bncsve2_rts_sum128(pg, ret, a0, a1, a2);
}

static inline void _bncsve2_rts_norm128(svbool_t pg, float ret[TSSIZE],
                                        svfloat32_t a0, svfloat32_t a1, svfloat32_t a2)
{
    long int n = (long int)svcntw(), k;
    float b0[_BNC_SVE_MAXLEN_S], b1[_BNC_SVE_MAXLEN_S], b2[_BNC_SVE_MAXLEN_S], v[TSSIZE], t[TSSIZE];
    (void)pg;
    svst1_f32(svptrue_b32(), b0, a0);
    svst1_f32(svptrue_b32(), b1, a1);
    svst1_f32(svptrue_b32(), b2, a2);
    v[0]=b0[0]; v[1]=b1[0]; v[2]=b2[0]; rts_mul(t, v, v); rts_set(ret, t);
    for(k = 1; k < n; k++){ v[0]=b0[k]; v[1]=b1[k]; v[2]=b2[k]; rts_mul(t, v, v); rts_add(ret, ret, t); }
    rts_sqrt(t, ret); rts_set(ret, t);
}

/* ---- QS (4 limbs) ---- */
static inline void _bncsve2_rqs_abs(svbool_t pg,
                                    svfloat32_t *r0, svfloat32_t *r1,
                                    svfloat32_t *r2, svfloat32_t *r3,
                                    svfloat32_t a0, svfloat32_t a1,
                                    svfloat32_t a2, svfloat32_t a3)
{
    svbool_t neg = svcmplt_f32(pg, a0, svdup_n_f32(0.0f));
    *r0 = svsel_f32(neg, svneg_f32_x(pg, a0), a0);
    *r1 = svsel_f32(neg, svneg_f32_x(pg, a1), a1);
    *r2 = svsel_f32(neg, svneg_f32_x(pg, a2), a2);
    *r3 = svsel_f32(neg, svneg_f32_x(pg, a3), a3);
}

static inline void _bncsve2_rqs_sum128(svbool_t pg, float ret[QSSIZE],
                                       svfloat32_t a0, svfloat32_t a1,
                                       svfloat32_t a2, svfloat32_t a3)
{
    long int n = (long int)svcntw(), k;
    float b0[_BNC_SVE_MAXLEN_S], b1[_BNC_SVE_MAXLEN_S],
          b2[_BNC_SVE_MAXLEN_S], b3[_BNC_SVE_MAXLEN_S], v[QSSIZE];
    (void)pg;
    svst1_f32(svptrue_b32(), b0, a0);
    svst1_f32(svptrue_b32(), b1, a1);
    svst1_f32(svptrue_b32(), b2, a2);
    svst1_f32(svptrue_b32(), b3, a3);
    v[0]=b0[0]; v[1]=b1[0]; v[2]=b2[0]; v[3]=b3[0];
    rqs_set(ret, v);
    for(k = 1; k < n; k++){ v[0]=b0[k]; v[1]=b1[k]; v[2]=b2[k]; v[3]=b3[k]; rqs_add(ret, ret, v); }
}

static inline void _bncsve2_rqs_abssum128(svbool_t pg, float ret[QSSIZE],
                                          svfloat32_t a0, svfloat32_t a1,
                                          svfloat32_t a2, svfloat32_t a3)
{
    _bncsve2_rqs_sum128(pg, ret, a0, a1, a2, a3);
}

static inline void _bncsve2_rqs_norm128(svbool_t pg, float ret[QSSIZE],
                                        svfloat32_t a0, svfloat32_t a1,
                                        svfloat32_t a2, svfloat32_t a3)
{
    long int n = (long int)svcntw(), k;
    float b0[_BNC_SVE_MAXLEN_S], b1[_BNC_SVE_MAXLEN_S],
          b2[_BNC_SVE_MAXLEN_S], b3[_BNC_SVE_MAXLEN_S], v[QSSIZE], t[QSSIZE];
    (void)pg;
    svst1_f32(svptrue_b32(), b0, a0);
    svst1_f32(svptrue_b32(), b1, a1);
    svst1_f32(svptrue_b32(), b2, a2);
    svst1_f32(svptrue_b32(), b3, a3);
    v[0]=b0[0]; v[1]=b1[0]; v[2]=b2[0]; v[3]=b3[0]; rqs_mul(t, v, v); rqs_set(ret, t);
    for(k = 1; k < n; k++){ v[0]=b0[k]; v[1]=b1[k]; v[2]=b2[k]; v[3]=b3[k]; rqs_mul(t, v, v); rqs_add(ret, ret, t); }
    rqs_sqrt(t, ret); rqs_set(ret, t);
}

#endif // defined(__ARM_SVE2)
#endif // __BNCSVE2_QS_H
