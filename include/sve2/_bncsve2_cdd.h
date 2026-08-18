#ifndef __BNCSVE2_cdd_H__
#define __BNCSVE2_cdd_H__
/* Native SVE2 complex double-double (cdd) kernels (2-lane, derived from the
 * NEON kernels in neon/_bncneon_cdd.h).  A complex DD value is carried as four
 * svfloat64_t vectors: real = (re0=hi, re1=lo), imag = (im0=hi, im1=lo).
 * These mirror the call convention used in cdd_poly.c / cddlu.c.
 * Real DD primitives (_bncsve2_rdd_{add,mul,neg}) come from _bncsve2_dd.h,
 * which is included before this header in bncsve2.h. */

/* (*ret) := a + b  (component-wise DD add) */
static inline void _bncsve2_rcdd_add(svbool_t pg,
        svfloat64_t *ret_re0, svfloat64_t *ret_re1,
        svfloat64_t *ret_im0, svfloat64_t *ret_im1,
        svfloat64_t a_re0, svfloat64_t a_re1, svfloat64_t a_im0, svfloat64_t a_im1,
        svfloat64_t b_re0, svfloat64_t b_re1, svfloat64_t b_im0, svfloat64_t b_im1)
{
    _bncsve2_rdd_add(pg, ret_re0, ret_re1, a_re0, a_re1, b_re0, b_re1);
    _bncsve2_rdd_add(pg, ret_im0, ret_im1, a_im0, a_im1, b_im0, b_im1);
}

/* (*ret) := a - b  (component-wise DD sub via add of negation) */
static inline void _bncsve2_rcdd_sub(svbool_t pg,
        svfloat64_t *ret_re0, svfloat64_t *ret_re1,
        svfloat64_t *ret_im0, svfloat64_t *ret_im1,
        svfloat64_t a_re0, svfloat64_t a_re1, svfloat64_t a_im0, svfloat64_t a_im1,
        svfloat64_t b_re0, svfloat64_t b_re1, svfloat64_t b_im0, svfloat64_t b_im1)
{
    svfloat64_t nb_re0, nb_re1, nb_im0, nb_im1;
    _bncsve2_rdd_neg(pg, &nb_re0, &nb_re1, b_re0, b_re1);
    _bncsve2_rdd_neg(pg, &nb_im0, &nb_im1, b_im0, b_im1);
    _bncsve2_rdd_add(pg, ret_re0, ret_re1, a_re0, a_re1, nb_re0, nb_re1);
    _bncsve2_rdd_add(pg, ret_im0, ret_im1, a_im0, a_im1, nb_im0, nb_im1);
}

/* (*ret) := a * b  (complex DD multiply, 4M)
 *   ret_re = a_re*b_re - a_im*b_im
 *   ret_im = a_re*b_im + a_im*b_re                                          */
static inline void _bncsve2_rcdd_mul(svbool_t pg,
        svfloat64_t *ret_re0, svfloat64_t *ret_re1,
        svfloat64_t *ret_im0, svfloat64_t *ret_im1,
        svfloat64_t a_re0, svfloat64_t a_re1, svfloat64_t a_im0, svfloat64_t a_im1,
        svfloat64_t b_re0, svfloat64_t b_re1, svfloat64_t b_im0, svfloat64_t b_im1)
{
    svfloat64_t t1_0, t1_1, t2_0, t2_1, nt2_0, nt2_1;

    /* ret_re = a_re*b_re - a_im*b_im */
    _bncsve2_rdd_mul(pg, &t1_0, &t1_1, a_re0, a_re1, b_re0, b_re1);
    _bncsve2_rdd_mul(pg, &t2_0, &t2_1, a_im0, a_im1, b_im0, b_im1);
    _bncsve2_rdd_neg(pg, &nt2_0, &nt2_1, t2_0, t2_1);
    _bncsve2_rdd_add(pg, ret_re0, ret_re1, t1_0, t1_1, nt2_0, nt2_1);

    /* ret_im = a_re*b_im + a_im*b_re */
    _bncsve2_rdd_mul(pg, &t1_0, &t1_1, a_re0, a_re1, b_im0, b_im1);
    _bncsve2_rdd_mul(pg, &t2_0, &t2_1, a_im0, a_im1, b_re0, b_re1);
    _bncsve2_rdd_add(pg, ret_im0, ret_im1, t1_0, t1_1, t2_0, t2_1);
}

#endif
