// -------------------------------------------------------
// _bncsve2_qd.h  --  SVE2 Quad-Double (QD) per-limb-variable helpers
// -------------------------------------------------------
// Rewritten 2026.  A QD value is FOUR separate svfloat64_t variables.
// A TD value is THREE.  All operands are passed by value (inputs) or by
// pointer (outputs).  Mirrors include/neon/_bncneon_qd.h.
//
// renorm / renorm4: the NEON header bounces through vgetq_lane_f64 +
// scalar renorm(); SVE2 cannot lane-extract VL-agnostically, so we
// implement branch-free vectorized versions using svsel/svcmpne.  The
// semantics match include/c_dd_qd.h's renorm()/renorm4() per lane (the
// data-dependent zero-skip cascade is realized via predicate selection
// after computing all candidates).
//
// FORWARD declarations of _bncsve2_rtd_addq / _bncsve2_rtd_mulq mirror
// the NEON pattern (those are the default TD add/mul; their bodies use
// _bncsve2_renorm which is defined here, so they live here too).
// -------------------------------------------------------
#ifndef __BNCSVE2_QD_H
#define __BNCSVE2_QD_H

#if defined(__ARM_SVE2)
#include <arm_sve.h>

#ifndef QDSIZE
    #define QDSIZE 4
#endif
#ifndef TDSIZE
    #define TDSIZE 3
#endif

/* ============================================================
 *  set0 / set_d / neg for QD
 * ============================================================ */
static inline void _bncsve2_rqd_set0(svfloat64_t *r0, svfloat64_t *r1,
                                      svfloat64_t *r2, svfloat64_t *r3)
{
    *r0 = svdup_n_f64(0.0);
    *r1 = svdup_n_f64(0.0);
    *r2 = svdup_n_f64(0.0);
    *r3 = svdup_n_f64(0.0);
}

static inline void _bncsve2_rqd_set_d(svfloat64_t *r0, svfloat64_t *r1,
                                       svfloat64_t *r2, svfloat64_t *r3,
                                       svfloat64_t v)
{
    *r0 = v;
    *r1 = svdup_n_f64(0.0);
    *r2 = svdup_n_f64(0.0);
    *r3 = svdup_n_f64(0.0);
}

static inline void _bncsve2_rqd_neg(svbool_t pg,
                                     svfloat64_t *r0, svfloat64_t *r1,
                                     svfloat64_t *r2, svfloat64_t *r3,
                                     svfloat64_t a0, svfloat64_t a1,
                                     svfloat64_t a2, svfloat64_t a3)
{
    *r0 = svneg_f64_x(pg, a0);
    *r1 = svneg_f64_x(pg, a1);
    *r2 = svneg_f64_x(pg, a2);
    *r3 = svneg_f64_x(pg, a3);
}

/* ============================================================
 *  three_sum / three_sum2  (mirrors NEON _bncneon_three_sum*)
 * ============================================================ */
static inline void _bncsve2_three_sum(svbool_t pg,
                                       svfloat64_t *a, svfloat64_t *b,
                                       svfloat64_t *c)
{
    svfloat64_t t1, t2, t3;
    t1 = _bncsve2_dtwo_sum(pg, *a, *b, &t2);
    *a = _bncsve2_dtwo_sum(pg, *c, t1, &t3);
    *b = _bncsve2_dtwo_sum(pg, t2, t3, c);
}

static inline void _bncsve2_three_sum2(svbool_t pg,
                                        svfloat64_t *a, svfloat64_t *b,
                                        svfloat64_t *c)
{
    svfloat64_t t1, t2, t3;
    t1 = _bncsve2_dtwo_sum(pg, *a, *b, &t2);
    *a = _bncsve2_dtwo_sum(pg, *c, t1, &t3);
    *b = svadd_f64_x(pg, t2, t3);
}

/* ============================================================
 *  renorm / renorm4 -- branch-free vectorized versions.
 *
 *  Scalar renorm(c0, c1, c2, c3):
 *      s0 = quick_two_sum(c2, c3, &c3);
 *      s0 = quick_two_sum(c1,  s0, &c2);
 *      *c0 = quick_two_sum(*c0, s0, &c1);
 *      s0 = *c0; s1 = *c1;
 *      if (s1 != 0) {
 *          s1 = qts(s1, c2, &s2);
 *          if (s2 != 0) s2 = qts(s2, c3, &s3);
 *          else         s1 = qts(s1, c3, &s2);
 *      } else {
 *          s0 = qts(s0, c2, &s1);
 *          if (s1 != 0) s1 = qts(s1, c3, &s2);
 *          else         s0 = qts(s0, c3, &s1);
 *      }
 *      *c0 = s0; *c1 = s1; *c2 = s2; *c3 = s3;
 *
 *  Vectorized: compute all 4 candidate outcomes per lane and select.
 *  NOTE: the scalar code also has an "if (isinf(c0)) return;" guard that
 *  we DO NOT replicate (matmul never feeds inf to renorm, and a per-lane
 *  early-return is impossible VL-agnostically; the worst case is that
 *  an inf input produces NaN intermediates rather than being preserved
 *  -- acceptable for this library's use cases).
 * ============================================================ */
static inline void _bncsve2_renorm(svbool_t pg,
                                    svfloat64_t *c0, svfloat64_t *c1,
                                    svfloat64_t *c2, svfloat64_t *c3)
{
    svfloat64_t zero = svdup_n_f64(0.0);
    svfloat64_t s0, s1, s2, s3;
    svfloat64_t nc1, nc2, nc3;  /* "new c1, c2, c3" out of the first cascade */

    /* First branch-free cascade (no zero-skip). */
    s0  = _bncsve2_dquick_two_sum(pg, *c2, *c3, &nc3);
    s0  = _bncsve2_dquick_two_sum(pg, *c1,  s0, &nc2);
    *c0 = _bncsve2_dquick_two_sum(pg, *c0,  s0, &nc1);

    s0 = *c0;
    s1 = nc1;

    /* path A: s1!=0, s2_a!=0  -> s2 = qts(s2_a, nc3, &s3)         (s0 stays) */
    /* path B: s1!=0, s2_a==0  -> s1 = qts(s1_a, nc3, &s2_b); s3=0           */
    /* path C: s1==0, s1_c!=0  -> s1 = qts(s1_c, nc3, &s2_c); s3=0           */
    /* path D: s1==0, s1_c==0  -> s0 = qts(s0_c, nc3, &s1_d); s2=s3=0        */
    svfloat64_t s2_a, s3_a, s1_a, s2_b;
    svfloat64_t s1_c, s2_c, s2_c2, s0_c, s1_d;

    s1_a = _bncsve2_dquick_two_sum(pg, s1,  nc2, &s2_a);
    s3_a = svdup_n_f64(0.0);
    s3_a = _bncsve2_dquick_two_sum(pg, s2_a, nc3, &s3_a);   /* path A: refine s2 -> s2', s3 */
    /* In path A we want s2 = qts(s2_a, nc3, &s3), so: */
    svfloat64_t s2_A = _bncsve2_dquick_two_sum(pg, s2_a, nc3, &s3_a); /* re-bind: s2_A is "qts hi", s3_a is "qts lo" */
    /* path B: s1 = qts(s1_a, nc3, &s2_b) */
    svfloat64_t s1_B = _bncsve2_dquick_two_sum(pg, s1_a, nc3, &s2_b);

    /* path C/D start from s0 = qts(s0, nc2, &s1_c) */
    s0_c = _bncsve2_dquick_two_sum(pg, s0, nc2, &s1_c);
    /* path C: s1 = qts(s1_c, nc3, &s2_c) */
    svfloat64_t s1_C = _bncsve2_dquick_two_sum(pg, s1_c, nc3, &s2_c2);
    s2_c = s2_c2;
    /* path D: s0 = qts(s0_c, nc3, &s1_d) */
    svfloat64_t s0_D = _bncsve2_dquick_two_sum(pg, s0_c, nc3, &s1_d);

    /* Selectors */
    svbool_t cmp_s1   = svcmpne_f64(pg, s1, zero);            /* s1 != 0 */
    svbool_t cmp_s2a  = svcmpne_f64(pg, s2_a, zero);          /* s2_a != 0 */
    svbool_t cmp_s1c  = svcmpne_f64(pg, s1_c, zero);          /* s1_c != 0 */

    /* Compose per-lane results */
    /* Final s0:
     *   if s1!=0      -> s0 (unchanged)
     *   else if s1_c!=0 -> s0_c (path C)
     *   else          -> s0_D
     */
    svfloat64_t s0_else = svsel_f64(cmp_s1c, s0_c, s0_D);
    svfloat64_t fin_s0  = svsel_f64(cmp_s1, s0, s0_else);

    /* Final s1:
     *   if s1!=0 && s2_a!=0 -> s1_a
     *   if s1!=0 && s2_a==0 -> s1_B    (qts result)
     *   if s1==0 && s1_c!=0 -> s1_C
     *   if s1==0 && s1_c==0 -> s1_d
     */
    svfloat64_t s1_when_s1nz   = svsel_f64(cmp_s2a, s1_a, s1_B);
    svfloat64_t s1_when_s1z    = svsel_f64(cmp_s1c, s1_C, s1_d);
    svfloat64_t fin_s1         = svsel_f64(cmp_s1, s1_when_s1nz, s1_when_s1z);

    /* Final s2:
     *   if s1!=0 && s2_a!=0 -> s2_A
     *   if s1!=0 && s2_a==0 -> s2_b
     *   if s1==0 && s1_c!=0 -> s2_c
     *   if s1==0 && s1_c==0 -> 0
     */
    svfloat64_t s2_when_s1nz  = svsel_f64(cmp_s2a, s2_A, s2_b);
    svfloat64_t s2_when_s1z   = svsel_f64(cmp_s1c, s2_c, zero);
    svfloat64_t fin_s2        = svsel_f64(cmp_s1, s2_when_s1nz, s2_when_s1z);

    /* Final s3:
     *   if s1!=0 && s2_a!=0 -> s3_a
     *   else                -> 0
     */
    svbool_t   path_A    = svand_b_z(pg, cmp_s1, cmp_s2a);
    svfloat64_t fin_s3   = svsel_f64(path_A, s3_a, zero);

    *c0 = fin_s0;
    *c1 = fin_s1;
    *c2 = fin_s2;
    *c3 = fin_s3;
}

/* renorm4 -- 5-element-in, 4-element-out branch-free. */
static inline void _bncsve2_renorm4(svbool_t pg,
                                     svfloat64_t *c0, svfloat64_t *c1,
                                     svfloat64_t *c2, svfloat64_t *c3,
                                     svfloat64_t *c4)
{
    svfloat64_t zero = svdup_n_f64(0.0);
    svfloat64_t s0, s1;
    svfloat64_t nc1, nc2, nc3, nc4;

    /* First cascade (no zero-skip). */
    s0  = _bncsve2_dquick_two_sum(pg, *c3, *c4, &nc4);
    s0  = _bncsve2_dquick_two_sum(pg, *c2,  s0, &nc3);
    s0  = _bncsve2_dquick_two_sum(pg, *c1,  s0, &nc2);
    *c0 = _bncsve2_dquick_two_sum(pg, *c0,  s0, &nc1);

    s0 = *c0;
    s1 = nc1;

    /* Decision tree (per lane), matching c_dd_qd.h renorm4():
     *
     *  if (s1 != 0) {
     *      s1 = qts(s1, c2, &s2);            // s2_A1
     *      if (s2_A1 != 0) {
     *          s2 = qts(s2_A1, c3, &s3);     // s2_A11, s3_A11
     *          if (s3_A11 != 0) s3 += c4;    // s3_A111
     *          else             s2 = qts(s2_A11, c4, &s3); // s2_A112, s3_A112
     *      } else {
     *          s1 = qts(s1_A1, c3, &s2);     // s1_A2, s2_A2
     *          if (s2_A2 != 0) s2 = qts(s2_A2, c4, &s3);   // s2_A21, s3_A21
     *          else            s1 = qts(s1_A2, c4, &s2);   // s1_A22, s2_A22
     *      }
     *  } else {
     *      s0 = qts(s0, c2, &s1);            // s1_B1
     *      if (s1_B1 != 0) {
     *          s1 = qts(s1_B1, c3, &s2);     // s1_B11, s2_B11
     *          if (s2_B11 != 0) s2 = qts(s2_B11, c4, &s3); // s2_B111, s3_B111
     *          else             s1 = qts(s1_B11, c4, &s2); // s1_B112, s2_B112
     *      } else {
     *          s0 = qts(s0_B, c3, &s1);      // s0_B2, s1_B2
     *          if (s1_B2 != 0) s1 = qts(s1_B2, c4, &s2);   // s1_B21, s2_B21
     *          else            s0 = qts(s0_B2, c4, &s1);   // s0_B22, s1_B22
     *      }
     *  }
     *
     *  We compute every candidate and select with svsel.  s2 is 0 by default
     *  in the scalar paths where it's not explicitly assigned.  s3 likewise.
     */

    /* ===== Path A: s1 != 0 ===== */
    svfloat64_t s2_A1;
    svfloat64_t s1_A1 = _bncsve2_dquick_two_sum(pg, s1, nc2, &s2_A1);

    /* A11 sub: s2_A1 != 0 */
    svfloat64_t s3_A11;
    svfloat64_t s2_A11 = _bncsve2_dquick_two_sum(pg, s2_A1, nc3, &s3_A11);
    svfloat64_t s3_A111 = svadd_f64_x(pg, s3_A11, nc4);                            /* s3_A11 != 0 */
    svfloat64_t s3_A112;
    svfloat64_t s2_A112 = _bncsve2_dquick_two_sum(pg, s2_A11, nc4, &s3_A112);       /* s3_A11 == 0 */
    svbool_t cmp_s3A11 = svcmpne_f64(pg, s3_A11, zero);
    svfloat64_t s2_A11_final = svsel_f64(cmp_s3A11, s2_A11, s2_A112);
    svfloat64_t s3_A11_final = svsel_f64(cmp_s3A11, s3_A111, s3_A112);

    /* A12 sub: s2_A1 == 0 */
    svfloat64_t s2_A2;
    svfloat64_t s1_A2  = _bncsve2_dquick_two_sum(pg, s1_A1, nc3, &s2_A2);
    svfloat64_t s3_A21;
    svfloat64_t s2_A21 = _bncsve2_dquick_two_sum(pg, s2_A2, nc4, &s3_A21);          /* s2_A2 != 0 */
    svfloat64_t s2_A22;
    svfloat64_t s1_A22 = _bncsve2_dquick_two_sum(pg, s1_A2, nc4, &s2_A22);          /* s2_A2 == 0 */
    svbool_t cmp_s2A2  = svcmpne_f64(pg, s2_A2, zero);
    svfloat64_t s1_A12_final = svsel_f64(cmp_s2A2, s1_A2,  s1_A22);
    svfloat64_t s2_A12_final = svsel_f64(cmp_s2A2, s2_A21, s2_A22);
    svfloat64_t s3_A12_final = svsel_f64(cmp_s2A2, s3_A21, zero);

    /* Merge A11 vs A12 by s2_A1 != 0 */
    svbool_t cmp_s2A1 = svcmpne_f64(pg, s2_A1, zero);
    svfloat64_t s1_A_final = svsel_f64(cmp_s2A1, s1_A1,        s1_A12_final);
    svfloat64_t s2_A_final = svsel_f64(cmp_s2A1, s2_A11_final, s2_A12_final);
    svfloat64_t s3_A_final = svsel_f64(cmp_s2A1, s3_A11_final, s3_A12_final);
    /* s0 unchanged in path A */
    svfloat64_t s0_A_final = s0;

    /* ===== Path B: s1 == 0 ===== */
    svfloat64_t s1_B1;
    svfloat64_t s0_B  = _bncsve2_dquick_two_sum(pg, s0, nc2, &s1_B1);

    /* B11 sub: s1_B1 != 0 */
    svfloat64_t s2_B11;
    svfloat64_t s1_B11 = _bncsve2_dquick_two_sum(pg, s1_B1, nc3, &s2_B11);
    svfloat64_t s3_B111;
    svfloat64_t s2_B111 = _bncsve2_dquick_two_sum(pg, s2_B11, nc4, &s3_B111);       /* s2_B11 != 0 */
    svfloat64_t s2_B112;
    svfloat64_t s1_B112 = _bncsve2_dquick_two_sum(pg, s1_B11, nc4, &s2_B112);       /* s2_B11 == 0 */
    svbool_t cmp_s2B11 = svcmpne_f64(pg, s2_B11, zero);
    svfloat64_t s1_B11_final = svsel_f64(cmp_s2B11, s1_B11,  s1_B112);
    svfloat64_t s2_B11_final = svsel_f64(cmp_s2B11, s2_B111, s2_B112);
    svfloat64_t s3_B11_final = svsel_f64(cmp_s2B11, s3_B111, zero);

    /* B12 sub: s1_B1 == 0 */
    svfloat64_t s1_B2;
    svfloat64_t s0_B2  = _bncsve2_dquick_two_sum(pg, s0_B, nc3, &s1_B2);
    svfloat64_t s2_B21;
    svfloat64_t s1_B21 = _bncsve2_dquick_two_sum(pg, s1_B2, nc4, &s2_B21);          /* s1_B2 != 0 */
    svfloat64_t s1_B22;
    svfloat64_t s0_B22 = _bncsve2_dquick_two_sum(pg, s0_B2, nc4, &s1_B22);          /* s1_B2 == 0 */
    svbool_t cmp_s1B2  = svcmpne_f64(pg, s1_B2, zero);
    svfloat64_t s0_B12_final = svsel_f64(cmp_s1B2, s0_B2,  s0_B22);
    svfloat64_t s1_B12_final = svsel_f64(cmp_s1B2, s1_B21, s1_B22);
    svfloat64_t s2_B12_final = svsel_f64(cmp_s1B2, s2_B21, zero);
    /* s3 in B12 is always 0 */

    /* Merge B11 vs B12 by s1_B1 != 0 */
    svbool_t cmp_s1B1 = svcmpne_f64(pg, s1_B1, zero);
    svfloat64_t s0_B_final = svsel_f64(cmp_s1B1, s0_B,        s0_B12_final);
    svfloat64_t s1_B_final = svsel_f64(cmp_s1B1, s1_B11_final, s1_B12_final);
    svfloat64_t s2_B_final = svsel_f64(cmp_s1B1, s2_B11_final, s2_B12_final);
    svfloat64_t s3_B_final = svsel_f64(cmp_s1B1, s3_B11_final, zero);

    /* Final merge: path A vs path B by (s1 != 0) */
    svbool_t cmp_s1nz = svcmpne_f64(pg, s1, zero);
    *c0 = svsel_f64(cmp_s1nz, s0_A_final, s0_B_final);
    *c1 = svsel_f64(cmp_s1nz, s1_A_final, s1_B_final);
    *c2 = svsel_f64(cmp_s1nz, s2_A_final, s2_B_final);
    *c3 = svsel_f64(cmp_s1nz, s3_A_final, s3_B_final);
}

/* ============================================================
 *  QD addition / multiplication (sloppy default = renorm4-based)
 * ============================================================ */

/* (*r0,*r1,*r2,*r3) := (a0..a3) + (b0..b3)  --  sloppy */
static inline void _bncsve2_rqd_add_sloppy(
        svbool_t pg,
        svfloat64_t *r0, svfloat64_t *r1, svfloat64_t *r2, svfloat64_t *r3,
        svfloat64_t a0, svfloat64_t a1, svfloat64_t a2, svfloat64_t a3,
        svfloat64_t b0, svfloat64_t b1, svfloat64_t b2, svfloat64_t b3)
{
    svfloat64_t s0 = svadd_f64_x(pg, a0, b0);
    svfloat64_t s1 = svadd_f64_x(pg, a1, b1);
    svfloat64_t s2 = svadd_f64_x(pg, a2, b2);
    svfloat64_t s3 = svadd_f64_x(pg, a3, b3);

    svfloat64_t v0 = svsub_f64_x(pg, s0, a0);
    svfloat64_t v1 = svsub_f64_x(pg, s1, a1);
    svfloat64_t v2 = svsub_f64_x(pg, s2, a2);
    svfloat64_t v3 = svsub_f64_x(pg, s3, a3);

    svfloat64_t u0 = svsub_f64_x(pg, s0, v0);
    svfloat64_t u1 = svsub_f64_x(pg, s1, v1);
    svfloat64_t u2 = svsub_f64_x(pg, s2, v2);
    svfloat64_t u3 = svsub_f64_x(pg, s3, v3);

    svfloat64_t w0 = svsub_f64_x(pg, a0, u0);
    svfloat64_t w1 = svsub_f64_x(pg, a1, u1);
    svfloat64_t w2 = svsub_f64_x(pg, a2, u2);
    svfloat64_t w3 = svsub_f64_x(pg, a3, u3);

    u0 = svsub_f64_x(pg, b0, v0);
    u1 = svsub_f64_x(pg, b1, v1);
    u2 = svsub_f64_x(pg, b2, v2);
    u3 = svsub_f64_x(pg, b3, v3);

    svfloat64_t t0 = svadd_f64_x(pg, w0, u0);
    svfloat64_t t1 = svadd_f64_x(pg, w1, u1);
    svfloat64_t t2 = svadd_f64_x(pg, w2, u2);
    svfloat64_t t3 = svadd_f64_x(pg, w3, u3);

    s1 = _bncsve2_dtwo_sum(pg, s1, t0, &t0);
    _bncsve2_three_sum (pg, &s2, &t0, &t1);
    _bncsve2_three_sum2(pg, &s3, &t0, &t2);
    t0 = svadd_f64_x(pg, svadd_f64_x(pg, t0, t1), t3);

    _bncsve2_renorm4(pg, &s0, &s1, &s2, &s3, &t0);
    *r0 = s0; *r1 = s1; *r2 = s2; *r3 = s3;
}

/* (*r0..*r3) := (a0..a3) * (b0..b3)  --  sloppy */
static inline void _bncsve2_rqd_mul_sloppy(
        svbool_t pg,
        svfloat64_t *r0, svfloat64_t *r1, svfloat64_t *r2, svfloat64_t *r3,
        svfloat64_t a0, svfloat64_t a1, svfloat64_t a2, svfloat64_t a3,
        svfloat64_t b0, svfloat64_t b1, svfloat64_t b2, svfloat64_t b3)
{
    svfloat64_t p0, p1, p2, p3, p4, p5;
    svfloat64_t q0, q1, q2, q3, q4, q5;
    svfloat64_t t0, t1;
    svfloat64_t s0, s1, s2;

    p0 = _bncsve2_dtwo_prod(pg, a0, b0, &q0);
    p1 = _bncsve2_dtwo_prod(pg, a0, b1, &q1);
    p2 = _bncsve2_dtwo_prod(pg, a1, b0, &q2);

    p3 = _bncsve2_dtwo_prod(pg, a0, b2, &q3);
    p4 = _bncsve2_dtwo_prod(pg, a1, b1, &q4);
    p5 = _bncsve2_dtwo_prod(pg, a2, b0, &q5);

    _bncsve2_three_sum(pg, &p1, &p2, &q0);

    _bncsve2_three_sum(pg, &p2, &q1, &q2);
    _bncsve2_three_sum(pg, &p3, &p4, &p5);

    s0 = _bncsve2_dtwo_sum(pg, p2, p3, &t0);
    s1 = _bncsve2_dtwo_sum(pg, q1, p4, &t1);
    s2 = svadd_f64_x(pg, q2, p5);
    s1 = _bncsve2_dtwo_sum(pg, s1, t0, &t0);
    s2 = svadd_f64_x(pg, s2, svadd_f64_x(pg, t0, t1));

    /* O(eps^3) order terms */
    s1 = svmla_f64_x(pg, s1, a0, b3);
    s1 = svmla_f64_x(pg, s1, a1, b2);
    s1 = svmla_f64_x(pg, s1, a2, b1);
    s1 = svmla_f64_x(pg, s1, a3, b0);
    s1 = svadd_f64_x(pg, s1, q0);
    s1 = svadd_f64_x(pg, s1, q3);
    s1 = svadd_f64_x(pg, s1, q4);
    s1 = svadd_f64_x(pg, s1, q5);

    _bncsve2_renorm4(pg, &p0, &p1, &s0, &s1, &s2);

    *r0 = p0; *r1 = p1; *r2 = s0; *r3 = s1;
}

/*
 * Branch-free QD add (mirrors NEON _bncneon_rqd_add_bf).
 * Uses only EFT primitives (no renorm / no three_sum / no two_sum chains
 * that span all 4 limbs at once) — much shorter dependency chain than
 * _bncsve2_rqd_add_sloppy + renorm5.  Output: ~3 ULPs in lowest QD limb.
 */
static inline void _bncsve2_rqd_add_bf(
        svbool_t pg,
        svfloat64_t *r0, svfloat64_t *r1, svfloat64_t *r2, svfloat64_t *r3,
        svfloat64_t a0, svfloat64_t a1, svfloat64_t a2, svfloat64_t a3,
        svfloat64_t b0, svfloat64_t b1, svfloat64_t b2, svfloat64_t b3)
{
    svfloat64_t e_b1, e_d1, e_f1, e_h1;
    svfloat64_t s_a1 = _bncsve2_dtwo_sum(pg, a0, b0, &e_b1);
    svfloat64_t s_c1 = _bncsve2_dtwo_sum(pg, a1, b1, &e_d1);
    svfloat64_t s_e1 = _bncsve2_dtwo_sum(pg, a2, b2, &e_f1);
    svfloat64_t s_g1 = _bncsve2_dtwo_sum(pg, a3, b3, &e_h1);

    svfloat64_t e_c2;
    svfloat64_t s_a2 = _bncsve2_dquick_two_sum(pg, s_a1, s_c1, &e_c2);
    svfloat64_t s_b2 = svadd_f64_x(pg, e_b1, e_h1);
    svfloat64_t e_e2;
    svfloat64_t s_d2 = _bncsve2_dtwo_sum(pg, e_d1, s_e1, &e_e2);
    svfloat64_t e_g2;
    svfloat64_t s_f2 = _bncsve2_dtwo_sum(pg, e_f1, s_g1, &e_g2);

    svfloat64_t e_g3;
    svfloat64_t s_b3 = _bncsve2_dtwo_sum(pg, s_b2, e_g2, &e_g3);
    svfloat64_t e_d3;
    svfloat64_t s_c3 = _bncsve2_dquick_two_sum(pg, e_c2, s_d2, &e_d3);
    svfloat64_t e_f3;
    svfloat64_t s_e3 = _bncsve2_dtwo_sum(pg, e_e2, s_f2, &e_f3);

    svfloat64_t e_c4;
    svfloat64_t s_a4 = _bncsve2_dquick_two_sum(pg, s_a2, s_c3, &e_c4);
    svfloat64_t e_e4;
    svfloat64_t s_d4 = _bncsve2_dquick_two_sum(pg, e_d3, s_e3, &e_e4);

    svfloat64_t e_d5;
    svfloat64_t s_b5 = _bncsve2_dtwo_sum(pg, s_b3, s_d4, &e_d5);
    svfloat64_t s_e5 = svadd_f64_x(pg, e_e4, e_f3);

    svfloat64_t e_c6;
    svfloat64_t s_b6 = _bncsve2_dtwo_sum(pg, s_b5, e_c4, &e_c6);
    svfloat64_t e_e6;
    svfloat64_t s_d6 = _bncsve2_dtwo_sum(pg, e_d5, s_e5, &e_e6);

    svfloat64_t e_b7;
    svfloat64_t s_a7 = _bncsve2_dquick_two_sum(pg, s_a4, s_b6, &e_b7);
    svfloat64_t e_d7;
    svfloat64_t s_c7 = _bncsve2_dquick_two_sum(pg, e_c6, s_d6, &e_d7);

    svfloat64_t s_e8 = svadd_f64_x(pg, e_e6, e_g3);
    svfloat64_t e_c8;
    svfloat64_t s_b8 = _bncsve2_dquick_two_sum(pg, e_b7, s_c7, &e_c8);

    svfloat64_t s_d9 = svadd_f64_x(pg, e_d7, s_e8);
    svfloat64_t e_b10;
    svfloat64_t s_a10 = _bncsve2_dquick_two_sum(pg, s_a7, s_b8, &e_b10);
    svfloat64_t e_d10;
    svfloat64_t s_c10 = _bncsve2_dquick_two_sum(pg, e_c8, s_d9, &e_d10);

    svfloat64_t e_c11;
    svfloat64_t s_b11 = _bncsve2_dquick_two_sum(pg, e_b10, s_c10, &e_c11);
    svfloat64_t e_d12;
    svfloat64_t s_c12 = _bncsve2_dquick_two_sum(pg, e_c11, e_d10, &e_d12);

    *r0 = s_a10; *r1 = s_b11; *r2 = s_c12; *r3 = e_d12;
}

/*
 * Branch-free QD mul (mirrors NEON _bncneon_rqd_mul_bf).
 */
static inline void _bncsve2_rqd_mul_bf(
        svbool_t pg,
        svfloat64_t *r0, svfloat64_t *r1, svfloat64_t *r2, svfloat64_t *r3,
        svfloat64_t a0, svfloat64_t a1, svfloat64_t a2, svfloat64_t a3,
        svfloat64_t b0, svfloat64_t b1, svfloat64_t b2, svfloat64_t b3)
{
    svfloat64_t e_b0, e_e0, e_f0, e_j0, e_k0, e_l0;
    svfloat64_t s_a0 = _bncsve2_dtwo_prod(pg, a0, b0, &e_b0);
    svfloat64_t s_c0 = _bncsve2_dtwo_prod(pg, a0, b1, &e_e0);
    svfloat64_t s_d0 = _bncsve2_dtwo_prod(pg, a1, b0, &e_f0);
    svfloat64_t s_g0 = _bncsve2_dtwo_prod(pg, a0, b2, &e_j0);
    svfloat64_t s_h0 = _bncsve2_dtwo_prod(pg, a1, b1, &e_k0);
    svfloat64_t s_i0 = _bncsve2_dtwo_prod(pg, a2, b0, &e_l0);

    svfloat64_t s_m0 = svmul_f64_x(pg, a0, b3);
    svfloat64_t s_n0 = svmul_f64_x(pg, a1, b2);
    svfloat64_t s_o0 = svmul_f64_x(pg, a2, b1);
    svfloat64_t s_p0 = svmul_f64_x(pg, a3, b0);

    svfloat64_t e_d1;
    svfloat64_t s_c1 = _bncsve2_dtwo_sum(pg, s_c0, s_d0, &e_d1);
    svfloat64_t e_f1;
    svfloat64_t s_e1 = _bncsve2_dtwo_sum(pg, e_e0, e_f0, &e_f1);
    svfloat64_t e_i1;
    svfloat64_t s_g1 = _bncsve2_dtwo_sum(pg, s_g0, s_i0, &e_i1);

    svfloat64_t s_j1 = svadd_f64_x(pg, e_j0, e_l0);
    svfloat64_t s_m1 = svadd_f64_x(pg, s_m0, s_p0);
    svfloat64_t s_n1 = svadd_f64_x(pg, s_n0, s_o0);

    svfloat64_t e_c2;
    svfloat64_t s_b2 = _bncsve2_dtwo_sum(pg, e_b0, s_c1, &e_c2);
    svfloat64_t e_h2;
    svfloat64_t s_e2 = _bncsve2_dtwo_sum(pg, s_e1, s_h0, &e_h2);

    svfloat64_t s_f2 = svadd_f64_x(pg, e_f1, s_j1);
    svfloat64_t s_i2 = svadd_f64_x(pg, e_i1, e_k0);
    svfloat64_t s_m2 = svadd_f64_x(pg, s_m1, s_n1);

    svfloat64_t e_b3;
    svfloat64_t s_a3 = _bncsve2_dquick_two_sum(pg, s_a0, s_b2, &e_b3);
    svfloat64_t e_d3;
    svfloat64_t s_c3 = _bncsve2_dquick_two_sum(pg, e_c2, e_d1, &e_d3);
    svfloat64_t e_g3;
    svfloat64_t s_e3 = _bncsve2_dtwo_sum(pg, s_e2, s_g1, &e_g3);

    svfloat64_t s_f3 = svadd_f64_x(pg, s_f2, s_m2);
    svfloat64_t s_h3 = svadd_f64_x(pg, e_h2, s_i2);

    svfloat64_t e_e4;
    svfloat64_t s_c4 = _bncsve2_dtwo_sum(pg, s_c3, s_e3, &e_e4);
    svfloat64_t s_d4 = svadd_f64_x(pg, e_d3, s_h3);
    svfloat64_t s_f4 = svadd_f64_x(pg, s_f3, e_g3);

    svfloat64_t s_d5 = svadd_f64_x(pg, s_d4, e_e4);
    svfloat64_t e_d6;
    svfloat64_t s_c6 = _bncsve2_dtwo_sum(pg, s_c4, s_d5, &e_d6);

    svfloat64_t e_c7;
    svfloat64_t s_b7 = _bncsve2_dtwo_sum(pg, e_b3, s_c6, &e_c7);
    svfloat64_t s_d7 = svadd_f64_x(pg, e_d6, s_f4);

    svfloat64_t e_b8;
    svfloat64_t s_a8 = _bncsve2_dquick_two_sum(pg, s_a3, s_b7, &e_b8);
    svfloat64_t e_d8;
    svfloat64_t s_c8 = _bncsve2_dtwo_sum(pg, e_c7, s_d7, &e_d8);

    svfloat64_t e_c9;
    svfloat64_t s_b9 = _bncsve2_dtwo_sum(pg, e_b8, s_c8, &e_c9);
    svfloat64_t e_d10;
    svfloat64_t s_c10 = _bncsve2_dquick_two_sum(pg, e_c9, e_d8, &e_d10);

    *r0 = s_a8; *r1 = s_b9; *r2 = s_c10; *r3 = e_d10;
}

/* TD addition (renorm-based, the default in NEON path). */
static inline void _bncsve2_rtd_addq(
        svbool_t pg,
        svfloat64_t *r0, svfloat64_t *r1, svfloat64_t *r2,
        svfloat64_t a0, svfloat64_t a1, svfloat64_t a2,
        svfloat64_t b0, svfloat64_t b1, svfloat64_t b2)
{
    svfloat64_t s0 = svadd_f64_x(pg, a0, b0);
    svfloat64_t s1 = svadd_f64_x(pg, a1, b1);
    svfloat64_t s2 = svadd_f64_x(pg, a2, b2);

    svfloat64_t v0 = svsub_f64_x(pg, s0, a0);
    svfloat64_t v1 = svsub_f64_x(pg, s1, a1);
    svfloat64_t v2 = svsub_f64_x(pg, s2, a2);

    svfloat64_t u0 = svsub_f64_x(pg, s0, v0);
    svfloat64_t u1 = svsub_f64_x(pg, s1, v1);
    svfloat64_t u2 = svsub_f64_x(pg, s2, v2);

    svfloat64_t w0 = svsub_f64_x(pg, a0, u0);
    svfloat64_t w1 = svsub_f64_x(pg, a1, u1);
    svfloat64_t w2 = svsub_f64_x(pg, a2, u2);

    u0 = svsub_f64_x(pg, b0, v0);
    u1 = svsub_f64_x(pg, b1, v1);
    u2 = svsub_f64_x(pg, b2, v2);

    svfloat64_t t0 = svadd_f64_x(pg, w0, u0);
    svfloat64_t t1 = svadd_f64_x(pg, w1, u1);
    svfloat64_t t2 = svadd_f64_x(pg, w2, u2);

    s1 = _bncsve2_dtwo_sum(pg, s1, t0, &t0);
    _bncsve2_three_sum(pg, &s2, &t0, &t1);
    t0 = svadd_f64_x(pg, svadd_f64_x(pg, t0, t1), t2);

    _bncsve2_renorm(pg, &s0, &s1, &s2, &t0);
    *r0 = s0; *r1 = s1; *r2 = s2;
}

/* TD multiplication (renorm-based, the default). */
static inline void _bncsve2_rtd_mulq(
        svbool_t pg,
        svfloat64_t *r0, svfloat64_t *r1, svfloat64_t *r2,
        svfloat64_t a0, svfloat64_t a1, svfloat64_t a2,
        svfloat64_t b0, svfloat64_t b1, svfloat64_t b2)
{
    svfloat64_t p0, p1, p2, p3, p4, p5;
    svfloat64_t q0, q1, q2, q3, q4;
    svfloat64_t t0, t1;
    svfloat64_t s0, s1;
    svfloat64_t q5_unused;

    p0 = _bncsve2_dtwo_prod(pg, a0, b0, &q0);
    p1 = _bncsve2_dtwo_prod(pg, a0, b1, &q1);
    p2 = _bncsve2_dtwo_prod(pg, a1, b0, &q2);

    p3 = _bncsve2_dtwo_prod(pg, a0, b2, &q3);
    p4 = _bncsve2_dtwo_prod(pg, a1, b1, &q4);
    p5 = _bncsve2_dtwo_prod(pg, a2, b0, &q5_unused);
    (void)q5_unused;

    _bncsve2_three_sum(pg, &p1, &p2, &q0);

    _bncsve2_three_sum2(pg, &p2, &q1, &q2);
    _bncsve2_three_sum2(pg, &p3, &p4, &p5);

    s0 = _bncsve2_dtwo_sum(pg, p2, p3, &t0);
    s1 = _bncsve2_dtwo_sum(pg, q1, p4, &t1);
    s1 = _bncsve2_dtwo_sum(pg, s1, t0, &t0);

    s1 = svmla_f64_x(pg, s1, a1, b2);
    s1 = svmla_f64_x(pg, s1, a2, b1);
    s1 = svadd_f64_x(pg, s1, q0);
    s1 = svadd_f64_x(pg, s1, q3);
    s1 = svadd_f64_x(pg, s1, q4);

    _bncsve2_renorm(pg, &p0, &p1, &s0, &s1);
    *r0 = p0; *r1 = p1; *r2 = s0;
}

/*
 * Branch-free TD add (mirrors NEON _bncneon_rtd_add_bf).
 * Uses only EFT primitives + svadd (no renorm / no three_sum).
 * Per-limb-variable signature: takes a0,a1,a2 / b0,b1,b2 by value,
 * writes *r0,*r1,*r2.  Output bound: ~3 ULPs in the lowest limb.
 */
static inline void _bncsve2_rtd_add_bf(
        svbool_t pg,
        svfloat64_t *r0, svfloat64_t *r1, svfloat64_t *r2,
        svfloat64_t a0, svfloat64_t a1, svfloat64_t a2,
        svfloat64_t b0, svfloat64_t b1, svfloat64_t b2)
{
    svfloat64_t b1_e, d1, f1;
    svfloat64_t a_s0 = _bncsve2_dtwo_sum(pg, a0, b0, &b1_e);
    svfloat64_t c_s1 = _bncsve2_dtwo_sum(pg, a1, b1, &d1);
    svfloat64_t e_s1 = _bncsve2_dtwo_sum(pg, a2, b2, &f1);

    svfloat64_t c2;
    svfloat64_t a_s2 = _bncsve2_dquick_two_sum(pg, a_s0, c_s1, &c2);
    svfloat64_t b_s2 = svadd_f64_x(pg, b1_e, f1);
    svfloat64_t e2;
    svfloat64_t d_s2 = _bncsve2_dtwo_sum(pg, d1, e_s1, &e2);

    svfloat64_t d3;
    svfloat64_t a_s3 = _bncsve2_dquick_two_sum(pg, a_s2, d_s2, &d3);
    svfloat64_t c3;
    svfloat64_t b_s3 = _bncsve2_dtwo_sum(pg, b_s2, c2, &c3);

    svfloat64_t c4 = svadd_f64_x(pg, c3, e2);
    svfloat64_t d5;
    svfloat64_t c_s5 = _bncsve2_dtwo_sum(pg, c4, d3, &d5);

    svfloat64_t c6;
    svfloat64_t b_s6 = _bncsve2_dtwo_sum(pg, b_s3, c_s5, &c6);
    svfloat64_t b7;
    svfloat64_t a_s7 = _bncsve2_dquick_two_sum(pg, a_s3, b_s6, &b7);

    svfloat64_t c7 = svadd_f64_x(pg, c6, d5);
    svfloat64_t c_s8;
    svfloat64_t b_s8 = _bncsve2_dquick_two_sum(pg, b7, c7, &c_s8);

    *r0 = a_s7;
    *r1 = b_s8;
    *r2 = c_s8;
}

/*
 * Branch-free TD mul (mirrors NEON _bncneon_rtd_mul_bf).
 */
static inline void _bncsve2_rtd_mul_bf(
        svbool_t pg,
        svfloat64_t *r0, svfloat64_t *r1, svfloat64_t *r2,
        svfloat64_t a0, svfloat64_t a1, svfloat64_t a2,
        svfloat64_t b0, svfloat64_t b1, svfloat64_t b2)
{
    svfloat64_t b0_e, e0, f0;
    svfloat64_t a_p0 = _bncsve2_dtwo_prod(pg, a0, b0, &b0_e);
    svfloat64_t c_p0 = _bncsve2_dtwo_prod(pg, a0, b1, &e0);
    svfloat64_t d_p0 = _bncsve2_dtwo_prod(pg, a1, b0, &f0);

    svfloat64_t g0 = svmul_f64_x(pg, a0, b2);
    svfloat64_t h0 = svmul_f64_x(pg, a1, b1);
    svfloat64_t i0 = svmul_f64_x(pg, a2, b0);

    svfloat64_t d1;
    svfloat64_t c_s1 = _bncsve2_dtwo_sum(pg, c_p0, d_p0, &d1);
    svfloat64_t e1 = svadd_f64_x(pg, e0, f0);
    svfloat64_t g1 = svadd_f64_x(pg, g0, i0);

    svfloat64_t c2;
    svfloat64_t b_s2 = _bncsve2_dtwo_sum(pg, b0_e, c_s1, &c2);
    svfloat64_t g2 = svadd_f64_x(pg, g1, h0);

    svfloat64_t b3;
    svfloat64_t a_s3 = _bncsve2_dquick_two_sum(pg, a_p0, b_s2, &b3);
    svfloat64_t c3 = svadd_f64_x(pg, c2, d1);
    svfloat64_t e3 = svadd_f64_x(pg, e1, g2);

    svfloat64_t c4 = svadd_f64_x(pg, c3, e3);
    svfloat64_t c5;
    svfloat64_t b_s5 = _bncsve2_dquick_two_sum(pg, b3, c4, &c5);

    svfloat64_t b6;
    svfloat64_t a_s6 = _bncsve2_dquick_two_sum(pg, a_s3, b_s5, &b6);
    svfloat64_t c7;
    svfloat64_t b_s7 = _bncsve2_dquick_two_sum(pg, b6, c5, &c7);

    *r0 = a_s6;
    *r1 = b_s7;
    *r2 = c7;
}

/* TD set0 / set_d / neg / set_dd */
static inline void _bncsve2_rtd_set0(svfloat64_t *r0, svfloat64_t *r1,
                                      svfloat64_t *r2)
{
    *r0 = svdup_n_f64(0.0);
    *r1 = svdup_n_f64(0.0);
    *r2 = svdup_n_f64(0.0);
}

static inline void _bncsve2_rtd_set_d(svfloat64_t *r0, svfloat64_t *r1,
                                       svfloat64_t *r2,
                                       svfloat64_t v)
{
    *r0 = v;
    *r1 = svdup_n_f64(0.0);
    *r2 = svdup_n_f64(0.0);
}

static inline void _bncsve2_rtd_neg(svbool_t pg,
                                     svfloat64_t *r0, svfloat64_t *r1,
                                     svfloat64_t *r2,
                                     svfloat64_t a0, svfloat64_t a1,
                                     svfloat64_t a2)
{
    *r0 = svneg_f64_x(pg, a0);
    *r1 = svneg_f64_x(pg, a1);
    *r2 = svneg_f64_x(pg, a2);
}

/* Default-variant macros (TD/QD).
 *  - QD: USE_QD_BF -> _bncsve2_rqd_{add,mul}_bf (branch-free, mirrors
 *        NEON BF variant; shorter dependency chain than _sloppy + renorm5).
 *        Otherwise -> _bncsve2_rqd_{add,mul}_sloppy.
 *  - TD: USE_TD_BF -> _bncsve2_rtd_{add,mul}_bf (branch-free, mirrors
 *        NEON BF variant).  Otherwise -> _bncsve2_rtd_{addq,mulq}
 *        (renorm-based, slower but tighter error bound).
 *  BF variants are recommended for matvec/matmul where the long renorm
 *  dependency chain dominates; the user can #define to override. */
#ifndef _bncsve2_rqd_add
    #ifdef USE_QD_BF
        #define _bncsve2_rqd_add  _bncsve2_rqd_add_bf
    #else
        #define _bncsve2_rqd_add  _bncsve2_rqd_add_sloppy
    #endif
#endif
#ifndef _bncsve2_rqd_mul
    #ifdef USE_QD_BF
        #define _bncsve2_rqd_mul  _bncsve2_rqd_mul_bf
    #else
        #define _bncsve2_rqd_mul  _bncsve2_rqd_mul_sloppy
    #endif
#endif
#ifndef _bncsve2_rtd_add
    #ifdef USE_TD_BF
        #define _bncsve2_rtd_add  _bncsve2_rtd_add_bf
    #else
        #define _bncsve2_rtd_add  _bncsve2_rtd_addq
    #endif
#endif
#ifndef _bncsve2_rtd_mul
    #ifdef USE_TD_BF
        #define _bncsve2_rtd_mul  _bncsve2_rtd_mul_bf
    #else
        #define _bncsve2_rtd_mul  _bncsve2_rtd_mulq
    #endif
#endif

/* (*c0..*c3) := (a0..a3) + b  -- QD + D  (mirror _bncavx2/neon_rqd_add_d) */
static inline void _bncsve2_rqd_add_d(svbool_t pg,
                                      svfloat64_t *c0, svfloat64_t *c1,
                                      svfloat64_t *c2, svfloat64_t *c3,
                                      svfloat64_t a0, svfloat64_t a1,
                                      svfloat64_t a2, svfloat64_t a3,
                                      svfloat64_t b)
{
    svfloat64_t e0, e1, s0, s1, s2, s3;
    s0 = _bncsve2_dtwo_sum(pg, a0, b,  &e0);
    s1 = _bncsve2_dtwo_sum(pg, a1, e0, &e1);
    s2 = _bncsve2_dtwo_sum(pg, a2, e1, &e0);
    s3 = _bncsve2_dtwo_sum(pg, a3, e0, &e1);
    _bncsve2_renorm4(pg, &s0, &s1, &s2, &s3, &e1);
    *c0 = s0; *c1 = s1; *c2 = s2; *c3 = s3;
}

/* -------- horizontal EFT reductions (mirror _bncneon_r{td,qd}_*128d) -------- */
#ifndef _BNC_SVE_MAXLEN_D
#define _BNC_SVE_MAXLEN_D 32   /* SVE max vector = 2048 bit => 32 doubles */
#endif

/* ---- TD (3 limbs) ---- */
static inline void _bncsve2_rtd_abs(svbool_t pg,
                                    svfloat64_t *r0, svfloat64_t *r1, svfloat64_t *r2,
                                    svfloat64_t a0, svfloat64_t a1, svfloat64_t a2)
{
    svbool_t neg = svcmplt_f64(pg, a0, svdup_n_f64(0.0));
    *r0 = svsel_f64(neg, svneg_f64_x(pg, a0), a0);
    *r1 = svsel_f64(neg, svneg_f64_x(pg, a1), a1);
    *r2 = svsel_f64(neg, svneg_f64_x(pg, a2), a2);
}

static inline void _bncsve2_rtd_sum128d(svbool_t pg, double ret[TDSIZE],
                                        svfloat64_t a0, svfloat64_t a1, svfloat64_t a2)
{
    long int n = (long int)svcntd(), k;
    double b0[_BNC_SVE_MAXLEN_D], b1[_BNC_SVE_MAXLEN_D], b2[_BNC_SVE_MAXLEN_D], v[TDSIZE];
    (void)pg;
    svst1_f64(svptrue_b64(), b0, a0);
    svst1_f64(svptrue_b64(), b1, a1);
    svst1_f64(svptrue_b64(), b2, a2);
    v[0] = b0[0]; v[1] = b1[0]; v[2] = b2[0];
    rtd_set(ret, v);
    for(k = 1; k < n; k++){ v[0]=b0[k]; v[1]=b1[k]; v[2]=b2[k]; rtd_add(ret, ret, v); }
}

static inline void _bncsve2_rtd_abssum128d(svbool_t pg, double ret[TDSIZE],
                                           svfloat64_t a0, svfloat64_t a1, svfloat64_t a2)
{
    _bncsve2_rtd_sum128d(pg, ret, a0, a1, a2);
}

/* ---- QD (4 limbs) ---- */
static inline void _bncsve2_rqd_abs(svbool_t pg,
                                    svfloat64_t *r0, svfloat64_t *r1,
                                    svfloat64_t *r2, svfloat64_t *r3,
                                    svfloat64_t a0, svfloat64_t a1,
                                    svfloat64_t a2, svfloat64_t a3)
{
    svbool_t neg = svcmplt_f64(pg, a0, svdup_n_f64(0.0));
    *r0 = svsel_f64(neg, svneg_f64_x(pg, a0), a0);
    *r1 = svsel_f64(neg, svneg_f64_x(pg, a1), a1);
    *r2 = svsel_f64(neg, svneg_f64_x(pg, a2), a2);
    *r3 = svsel_f64(neg, svneg_f64_x(pg, a3), a3);
}

static inline void _bncsve2_rqd_sum128d(svbool_t pg, double ret[QDSIZE],
                                        svfloat64_t a0, svfloat64_t a1,
                                        svfloat64_t a2, svfloat64_t a3)
{
    long int n = (long int)svcntd(), k;
    double b0[_BNC_SVE_MAXLEN_D], b1[_BNC_SVE_MAXLEN_D],
           b2[_BNC_SVE_MAXLEN_D], b3[_BNC_SVE_MAXLEN_D], v[QDSIZE];
    (void)pg;
    svst1_f64(svptrue_b64(), b0, a0);
    svst1_f64(svptrue_b64(), b1, a1);
    svst1_f64(svptrue_b64(), b2, a2);
    svst1_f64(svptrue_b64(), b3, a3);
    v[0] = b0[0]; v[1] = b1[0]; v[2] = b2[0]; v[3] = b3[0];
    rqd_set(ret, v);
    for(k = 1; k < n; k++){ v[0]=b0[k]; v[1]=b1[k]; v[2]=b2[k]; v[3]=b3[k]; rqd_add(ret, ret, v); }
}

static inline void _bncsve2_rqd_abssum128d(svbool_t pg, double ret[QDSIZE],
                                           svfloat64_t a0, svfloat64_t a1,
                                           svfloat64_t a2, svfloat64_t a3)
{
    _bncsve2_rqd_sum128d(pg, ret, a0, a1, a2, a3);
}

#endif // defined(__ARM_SVE2)
#endif // __BNCSVE2_QD_H
