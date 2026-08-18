#ifndef __BNCSVE2_ctd_H__
#define __BNCSVE2_ctd_H__
/* Native SVE2 complex triple-double (ctd) kernels (2-lane, derived from the
 * NEON kernels in neon/_bncneon_ctd.h).  A complex TD value is carried as six
 * svfloat64_t vectors: real = (re0,re1,re2), imag = (im0,im1,im2).
 * These mirror the call convention used in ctd_poly.c / ctdlu.c.
 * Real TD primitives (_bncsve2_rtd_{add,mul,neg}) come from _bncsve2_qd.h,
 * which is included before this header in bncsve2.h. */

/* (*ret) := a + b  (component-wise TD add) */
static inline void _bncsve2_rctd_add(svbool_t pg,
        svfloat64_t *ret_re0, svfloat64_t *ret_re1, svfloat64_t *ret_re2,
        svfloat64_t *ret_im0, svfloat64_t *ret_im1, svfloat64_t *ret_im2,
        svfloat64_t a_re0, svfloat64_t a_re1, svfloat64_t a_re2,
        svfloat64_t a_im0, svfloat64_t a_im1, svfloat64_t a_im2,
        svfloat64_t b_re0, svfloat64_t b_re1, svfloat64_t b_re2,
        svfloat64_t b_im0, svfloat64_t b_im1, svfloat64_t b_im2)
{
    _bncsve2_rtd_add(pg, ret_re0, ret_re1, ret_re2, a_re0, a_re1, a_re2, b_re0, b_re1, b_re2);
    _bncsve2_rtd_add(pg, ret_im0, ret_im1, ret_im2, a_im0, a_im1, a_im2, b_im0, b_im1, b_im2);
}

/* (*ret) := a - b  (component-wise TD sub via add of negation) */
static inline void _bncsve2_rctd_sub(svbool_t pg,
        svfloat64_t *ret_re0, svfloat64_t *ret_re1, svfloat64_t *ret_re2,
        svfloat64_t *ret_im0, svfloat64_t *ret_im1, svfloat64_t *ret_im2,
        svfloat64_t a_re0, svfloat64_t a_re1, svfloat64_t a_re2,
        svfloat64_t a_im0, svfloat64_t a_im1, svfloat64_t a_im2,
        svfloat64_t b_re0, svfloat64_t b_re1, svfloat64_t b_re2,
        svfloat64_t b_im0, svfloat64_t b_im1, svfloat64_t b_im2)
{
    svfloat64_t nb_re0, nb_re1, nb_re2, nb_im0, nb_im1, nb_im2;
    _bncsve2_rtd_neg(pg, &nb_re0, &nb_re1, &nb_re2, b_re0, b_re1, b_re2);
    _bncsve2_rtd_neg(pg, &nb_im0, &nb_im1, &nb_im2, b_im0, b_im1, b_im2);
    _bncsve2_rtd_add(pg, ret_re0, ret_re1, ret_re2, a_re0, a_re1, a_re2, nb_re0, nb_re1, nb_re2);
    _bncsve2_rtd_add(pg, ret_im0, ret_im1, ret_im2, a_im0, a_im1, a_im2, nb_im0, nb_im1, nb_im2);
}

/* (*ret) := a * b  (complex TD multiply, 4M)
 *   ret_re = a_re*b_re - a_im*b_im
 *   ret_im = a_re*b_im + a_im*b_re                                          */
static inline void _bncsve2_rctd_mul(svbool_t pg,
        svfloat64_t *ret_re0, svfloat64_t *ret_re1, svfloat64_t *ret_re2,
        svfloat64_t *ret_im0, svfloat64_t *ret_im1, svfloat64_t *ret_im2,
        svfloat64_t a_re0, svfloat64_t a_re1, svfloat64_t a_re2,
        svfloat64_t a_im0, svfloat64_t a_im1, svfloat64_t a_im2,
        svfloat64_t b_re0, svfloat64_t b_re1, svfloat64_t b_re2,
        svfloat64_t b_im0, svfloat64_t b_im1, svfloat64_t b_im2)
{
    svfloat64_t t1_0, t1_1, t1_2, t2_0, t2_1, t2_2, nt2_0, nt2_1, nt2_2;

    /* ret_re = a_re*b_re - a_im*b_im */
    _bncsve2_rtd_mul(pg, &t1_0, &t1_1, &t1_2, a_re0, a_re1, a_re2, b_re0, b_re1, b_re2);
    _bncsve2_rtd_mul(pg, &t2_0, &t2_1, &t2_2, a_im0, a_im1, a_im2, b_im0, b_im1, b_im2);
    _bncsve2_rtd_neg(pg, &nt2_0, &nt2_1, &nt2_2, t2_0, t2_1, t2_2);
    _bncsve2_rtd_add(pg, ret_re0, ret_re1, ret_re2, t1_0, t1_1, t1_2, nt2_0, nt2_1, nt2_2);

    /* ret_im = a_re*b_im + a_im*b_re */
    _bncsve2_rtd_mul(pg, &t1_0, &t1_1, &t1_2, a_re0, a_re1, a_re2, b_im0, b_im1, b_im2);
    _bncsve2_rtd_mul(pg, &t2_0, &t2_1, &t2_2, a_im0, a_im1, a_im2, b_re0, b_re1, b_re2);
    _bncsve2_rtd_add(pg, ret_im0, ret_im1, ret_im2, t1_0, t1_1, t1_2, t2_0, t2_1, t2_2);
}

#endif
