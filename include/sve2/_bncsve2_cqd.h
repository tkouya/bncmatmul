#ifndef __BNCSVE2_cqd_H__
#define __BNCSVE2_cqd_H__
/* Native SVE2 complex quad-double (cqd) kernels (2-lane, derived from the
 * NEON kernels in neon/_bncneon_cqd.h).  A complex QD value is carried as eight
 * svfloat64_t vectors: real = (re0..re3), imag = (im0..im3).
 * These mirror the call convention used in cqd_poly.c / cqdlu.c.
 * Real QD primitives (_bncsve2_rqd_{add,mul,neg}) come from _bncsve2_qd.h,
 * which is included before this header in bncsve2.h. */

/* (*ret) := a + b  (component-wise QD add) */
static inline void _bncsve2_rcqd_add(svbool_t pg,
        svfloat64_t *ret_re0, svfloat64_t *ret_re1, svfloat64_t *ret_re2, svfloat64_t *ret_re3,
        svfloat64_t *ret_im0, svfloat64_t *ret_im1, svfloat64_t *ret_im2, svfloat64_t *ret_im3,
        svfloat64_t a_re0, svfloat64_t a_re1, svfloat64_t a_re2, svfloat64_t a_re3,
        svfloat64_t a_im0, svfloat64_t a_im1, svfloat64_t a_im2, svfloat64_t a_im3,
        svfloat64_t b_re0, svfloat64_t b_re1, svfloat64_t b_re2, svfloat64_t b_re3,
        svfloat64_t b_im0, svfloat64_t b_im1, svfloat64_t b_im2, svfloat64_t b_im3)
{
    _bncsve2_rqd_add(pg, ret_re0, ret_re1, ret_re2, ret_re3, a_re0, a_re1, a_re2, a_re3, b_re0, b_re1, b_re2, b_re3);
    _bncsve2_rqd_add(pg, ret_im0, ret_im1, ret_im2, ret_im3, a_im0, a_im1, a_im2, a_im3, b_im0, b_im1, b_im2, b_im3);
}

/* (*ret) := a - b  (component-wise QD sub via add of negation) */
static inline void _bncsve2_rcqd_sub(svbool_t pg,
        svfloat64_t *ret_re0, svfloat64_t *ret_re1, svfloat64_t *ret_re2, svfloat64_t *ret_re3,
        svfloat64_t *ret_im0, svfloat64_t *ret_im1, svfloat64_t *ret_im2, svfloat64_t *ret_im3,
        svfloat64_t a_re0, svfloat64_t a_re1, svfloat64_t a_re2, svfloat64_t a_re3,
        svfloat64_t a_im0, svfloat64_t a_im1, svfloat64_t a_im2, svfloat64_t a_im3,
        svfloat64_t b_re0, svfloat64_t b_re1, svfloat64_t b_re2, svfloat64_t b_re3,
        svfloat64_t b_im0, svfloat64_t b_im1, svfloat64_t b_im2, svfloat64_t b_im3)
{
    svfloat64_t nb_re0, nb_re1, nb_re2, nb_re3, nb_im0, nb_im1, nb_im2, nb_im3;
    _bncsve2_rqd_neg(pg, &nb_re0, &nb_re1, &nb_re2, &nb_re3, b_re0, b_re1, b_re2, b_re3);
    _bncsve2_rqd_neg(pg, &nb_im0, &nb_im1, &nb_im2, &nb_im3, b_im0, b_im1, b_im2, b_im3);
    _bncsve2_rqd_add(pg, ret_re0, ret_re1, ret_re2, ret_re3, a_re0, a_re1, a_re2, a_re3, nb_re0, nb_re1, nb_re2, nb_re3);
    _bncsve2_rqd_add(pg, ret_im0, ret_im1, ret_im2, ret_im3, a_im0, a_im1, a_im2, a_im3, nb_im0, nb_im1, nb_im2, nb_im3);
}

/* (*ret) := a * b  (complex QD multiply, 4M)
 *   ret_re = a_re*b_re - a_im*b_im
 *   ret_im = a_re*b_im + a_im*b_re                                          */
static inline void _bncsve2_rcqd_mul(svbool_t pg,
        svfloat64_t *ret_re0, svfloat64_t *ret_re1, svfloat64_t *ret_re2, svfloat64_t *ret_re3,
        svfloat64_t *ret_im0, svfloat64_t *ret_im1, svfloat64_t *ret_im2, svfloat64_t *ret_im3,
        svfloat64_t a_re0, svfloat64_t a_re1, svfloat64_t a_re2, svfloat64_t a_re3,
        svfloat64_t a_im0, svfloat64_t a_im1, svfloat64_t a_im2, svfloat64_t a_im3,
        svfloat64_t b_re0, svfloat64_t b_re1, svfloat64_t b_re2, svfloat64_t b_re3,
        svfloat64_t b_im0, svfloat64_t b_im1, svfloat64_t b_im2, svfloat64_t b_im3)
{
    svfloat64_t t1_0, t1_1, t1_2, t1_3, t2_0, t2_1, t2_2, t2_3, nt2_0, nt2_1, nt2_2, nt2_3;

    /* ret_re = a_re*b_re - a_im*b_im */
    _bncsve2_rqd_mul(pg, &t1_0, &t1_1, &t1_2, &t1_3, a_re0, a_re1, a_re2, a_re3, b_re0, b_re1, b_re2, b_re3);
    _bncsve2_rqd_mul(pg, &t2_0, &t2_1, &t2_2, &t2_3, a_im0, a_im1, a_im2, a_im3, b_im0, b_im1, b_im2, b_im3);
    _bncsve2_rqd_neg(pg, &nt2_0, &nt2_1, &nt2_2, &nt2_3, t2_0, t2_1, t2_2, t2_3);
    _bncsve2_rqd_add(pg, ret_re0, ret_re1, ret_re2, ret_re3, t1_0, t1_1, t1_2, t1_3, nt2_0, nt2_1, nt2_2, nt2_3);

    /* ret_im = a_re*b_im + a_im*b_re */
    _bncsve2_rqd_mul(pg, &t1_0, &t1_1, &t1_2, &t1_3, a_re0, a_re1, a_re2, a_re3, b_im0, b_im1, b_im2, b_im3);
    _bncsve2_rqd_mul(pg, &t2_0, &t2_1, &t2_2, &t2_3, a_im0, a_im1, a_im2, a_im3, b_re0, b_re1, b_re2, b_re3);
    _bncsve2_rqd_add(pg, ret_im0, ret_im1, ret_im2, ret_im3, t1_0, t1_1, t1_2, t1_3, t2_0, t2_1, t2_2, t2_3);
}

#endif
