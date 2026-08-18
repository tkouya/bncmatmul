// Test ceval_[d,dd,td,qd,mpf]poly
#include <stdio.h>
#include <stdlib.h>
#include "get_secv.h"
#include "bncmatmul.h"

// mpf_relerr
// ret := |approx - true_val| / |true_val| (if true_val != 0)
//        |approx - true_val|              (if true_val == 0)
void mpf_relerr(mpf_t ret, mpf_t approx, mpf_t true_val)
{
    // ret := |approx - true_val|
    mpf_sub(ret, approx, true_val);
    mpf_abs(ret, ret);
    if(mpf_cmp_ui(true_val, 0UL) != 0)
    {
        mpf_div(ret, ret, true_val);
        mpf_abs(ret, ret);
    }
}


// d_relerr_mpf
// ret := |approx - true_val| / |true_val| (if true_val != 0)
//        |approx - true_val|              (if true_val == 0)
double drelerr_mpf(double approx, mpf_t true_val)
{
    mpf_t in_ret, in_approx;
    double ret;

    mpf_init2(in_ret, 53);
    mpf_init2(in_approx, 53);

    mpf_set_d(in_approx, approx);

    // ret := |approx - true_val|
    mpf_sub(in_ret, in_approx, true_val);
    mpf_abs(in_ret, in_ret);
    if(mpf_cmp_ui(true_val, 0UL) != 0)
    {
        mpf_div(in_ret, in_ret, true_val);
        mpf_abs(in_ret, in_ret);
    }

    ret = mpf_get_d(in_ret);

    mpf_clear(in_ret);
    mpf_clear(in_approx);

    return ret;
}

// dd_relerr_mpf
// ret := |approx - true_val| / |true_val| (if true_val != 0)
//        |approx - true_val|              (if true_val == 0)
void dd_relerr_mpf(double ret[DDSIZE], double approx[DDSIZE], mpf_t true_val)
{
    mpf_t in_ret, in_approx;

    mpf_init2(in_ret, 53 * DDSIZE);
    mpf_init2(in_approx, 53 * DDSIZE);

    mpf_set_dd(in_approx, approx);

    // ret := |approx - true_val|
    mpf_sub(in_ret, in_approx, true_val);
    mpf_abs(in_ret, in_ret);
    if(mpf_cmp_ui(true_val, 0UL) != 0)
    {
        mpf_div(in_ret, in_ret, true_val);
        mpf_abs(in_ret, in_ret);
    }

    mpf_get_dd(ret, in_ret);

    mpf_clear(in_ret);
    mpf_clear(in_approx);
}

// td_relerr_mpf
// ret := |approx - true_val| / |true_val| (if true_val != 0)
//        |approx - true_val|              (if true_val == 0)
void td_relerr_mpf(double ret[TDSIZE], double approx[TDSIZE], mpf_t true_val)
{
    mpf_t in_ret, in_approx;

    mpf_init2(in_ret, 53 * TDSIZE);
    mpf_init2(in_approx, 53 * TDSIZE);

    mpf_set_td(in_approx, approx);

    // ret := |approx - true_val|
    mpf_sub(in_ret, in_approx, true_val);
    mpf_abs(in_ret, in_ret);
    if(mpf_cmp_ui(true_val, 0UL) != 0)
    {
        mpf_div(in_ret, in_ret, true_val);
        mpf_abs(in_ret, in_ret);
    }

    mpf_get_td(ret, in_ret);

    mpf_clear(in_ret);
    mpf_clear(in_approx);
}

// qd_relerr_mpf
// ret := |approx - true_val| / |true_val| (if true_val != 0)
//        |approx - true_val|              (if true_val == 0)
void qd_relerr_mpf(double ret[QDSIZE], double approx[QDSIZE], mpf_t true_val)
{
    mpf_t in_ret, in_approx;

    mpf_init2(in_ret, 53 * QDSIZE);
    mpf_init2(in_approx, 53 * QDSIZE);

    mpf_set_qd(in_approx, approx);

    // ret := |approx - true_val|
    mpf_sub(in_ret, in_approx, true_val);
    mpf_abs(in_ret, in_ret);
    if(mpf_cmp_ui(true_val, 0UL) != 0)
    {
        mpf_div(in_ret, in_ret, true_val);
        mpf_abs(in_ret, in_ret);
    }

    mpf_get_qd(ret, in_ret);

    mpf_clear(in_ret);
    mpf_clear(in_approx);
}

// mpc_relerr
// ret := |approx - true_val| / |true_val| (if true_val != 0)
//        |approx - true_val|              (if true_val == 0)
// ret_c := relerr(real(approx), real(true_val)) + relerr(imag(approx), imag(true_val))
void mpc_relerr(mpf_t ret, mpc_t ret_c, mpc_t approx, mpc_t true_val)
{
    mpf_t tmp;
    mpc_t ctmp;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();
    mpfr_rnd_t rnd = get_bnc_default_rounding_mode();

    // ret_c := relerr(real(approx), real(true_val)) + relerr(imag(approx), imag(true_val))
    mpf_relerr(mpc_realref(ret_c), mpc_realref(approx), mpc_realref(true_val));
    mpf_relerr(mpc_imagref(ret_c), mpc_imagref(approx), mpc_imagref(true_val));

    // ret := |approx - true_val| / |true_val|
    mpc_init2(ctmp, mpf_get_prec(ret));
    mpf_init2(tmp, mpf_get_prec(ret));

    // ret := |approx - true_val|
    mpc_sub(ctmp, approx, true_val, rndc);
    mpc_abs(ret, ctmp, rnd);
    mpc_abs(tmp, true_val, rnd);
    if(mpf_cmp_ui(tmp, 0UL) != 0)
    {
        mpf_div(ret, ret, tmp);
    }

    mpc_clear(ctmp);
    mpf_clear(tmp);
}

// ret := (double)val
double _Complex mpc_get_d(mpc_t val)
{
    double _Complex ret;

    ret = mpf_get_d(mpc_realref(val)) \
        + mpf_get_d(mpc_imagref(val)) * I;

    return ret;
}
// ret := (mpc_t)val
void mpc_set_double(mpc_t ret, double _Complex val)
{
    mpf_set_d(mpc_realref(ret), creal(val));
    mpf_set_d(mpc_imagref(ret), cimag(val));
}

// ret := (DD)val
void mpc_get_dd(cddfloat *ret, mpc_t val)
{
    mpf_get_dd(ret->val_re, mpc_realref(val));
    mpf_get_dd(ret->val_im, mpc_imagref(val));
}
// ret := (mpc_t)val
void mpc_set_dd(mpc_t ret, cddfloat *val)
{
    mpf_set_dd(mpc_realref(ret), val->val_re);
    mpf_set_dd(mpc_imagref(ret), val->val_im);
}

// ret := (TD)val
void mpc_get_td(ctdfloat *ret, mpc_t val)
{
    mpf_get_td(ret->val_re, mpc_realref(val));
    mpf_get_td(ret->val_im, mpc_imagref(val));
}
// ret := (mpc_t)val
void mpc_set_td(mpc_t ret, ctdfloat *val)
{
    mpf_set_td(mpc_realref(ret), val->val_re);
    mpf_set_td(mpc_imagref(ret), val->val_im);
}
// ret := (QD)val
void mpc_get_qd(cqdfloat *ret, mpc_t val)
{
    mpf_get_qd(ret->val_re, mpc_realref(val));
    mpf_get_qd(ret->val_im, mpc_imagref(val));
}
// ret := (mpc_t)val
void mpc_set_qd(mpc_t ret, cqdfloat *val)
{
    mpf_set_qd(mpc_realref(ret), val->val_re);
    mpf_set_qd(mpc_imagref(ret), val->val_im);
}

// cd_relerr_mpc
void cd_relerr_mpc(double *ret, double _Complex *ret_c, double _Complex approx, mpc_t true_val)
{
    mpf_t tmp, in_ret;
    mpc_t ctmp, in_ret_c, in_approx;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();
    mpfr_rnd_t rnd = get_bnc_default_rounding_mode();

    mpf_init2(in_ret, 53);
    mpc_init2(in_ret_c, 53);
    mpc_init2(in_approx, 53);

    mpc_set_double(in_approx, approx);

    // ret_c := relerr(real(approx), real(true_val)) + relerr(imag(approx), imag(true_val))
    mpf_relerr(mpc_realref(in_ret_c), mpc_realref(in_approx), mpc_realref(true_val));
    mpf_relerr(mpc_imagref(in_ret_c), mpc_imagref(in_approx), mpc_imagref(true_val));
    *ret_c = mpc_get_d(in_ret_c);

    // ret := |approx - true_val| / |true_val|
    mpc_init2(ctmp, mpf_get_prec(in_ret));
    mpf_init2(tmp, mpf_get_prec(in_ret));

    // ret := |approx - true_val|
    mpc_sub(ctmp, in_approx, true_val, rndc);
    mpc_abs(in_ret, ctmp, rnd);
    mpc_abs(tmp, true_val, rnd);
    if(mpf_cmp_ui(tmp, 0UL) != 0)
    {
        mpf_div(in_ret, in_ret, tmp);
    }
    *ret = mpf_get_d(in_ret);

    mpc_clear(ctmp);
    mpf_clear(tmp);
    mpf_clear(in_ret);
    mpc_clear(in_ret_c);
    mpc_clear(in_approx);
}

// cdd_relerr_mpc
void cdd_relerr_mpc(double ret[DDSIZE], cddfloat *ret_c, cddfloat *approx, mpc_t true_val)
{
    mpf_t tmp, in_ret;
    mpc_t ctmp, in_ret_c, in_approx;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();
    mpfr_rnd_t rnd = get_bnc_default_rounding_mode();

    mpf_init2(in_ret, 53 * DDSIZE);
    mpc_init2(in_ret_c, 53 * DDSIZE);
    mpc_init2(in_approx, 53 * DDSIZE);

    mpc_set_dd(in_approx, approx);

    // ret_c := relerr(real(approx), real(true_val)) + relerr(imag(approx), imag(true_val))
    mpf_relerr(mpc_realref(in_ret_c), mpc_realref(in_approx), mpc_realref(true_val));
    mpf_relerr(mpc_imagref(in_ret_c), mpc_imagref(in_approx), mpc_imagref(true_val));
    mpc_get_dd(ret_c, in_ret_c);

    // ret := |approx - true_val| / |true_val|
    mpc_init2(ctmp, mpf_get_prec(in_ret));
    mpf_init2(tmp, mpf_get_prec(in_ret));

    // ret := |approx - true_val|
    mpc_sub(ctmp, in_approx, true_val, rndc);
    mpc_abs(in_ret, ctmp, rnd);
    mpc_abs(tmp, true_val, rnd);
    if(mpf_cmp_ui(tmp, 0UL) != 0)
    {
        mpf_div(in_ret, in_ret, tmp);
    }
    mpf_get_dd(ret, in_ret);

    mpc_clear(ctmp);
    mpf_clear(tmp);
    mpf_clear(in_ret);
    mpc_clear(in_ret_c);
    mpc_clear(in_approx);
}

// ctd_relerr_mpc
void ctd_relerr_mpc(double ret[TDSIZE], ctdfloat *ret_c, ctdfloat *approx, mpc_t true_val)
{
    mpf_t tmp, in_ret;
    mpc_t ctmp, in_ret_c, in_approx;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();
    mpfr_rnd_t rnd = get_bnc_default_rounding_mode();

    mpf_init2(in_ret, 53 * TDSIZE);
    mpc_init2(in_ret_c, 53 * TDSIZE);
    mpc_init2(in_approx, 53 * TDSIZE);

    mpc_set_td(in_approx, approx);

    // ret_c := relerr(real(approx), real(true_val)) + relerr(imag(approx), imag(true_val))
    mpf_relerr(mpc_realref(in_ret_c), mpc_realref(in_approx), mpc_realref(true_val));
    mpf_relerr(mpc_imagref(in_ret_c), mpc_imagref(in_approx), mpc_imagref(true_val));
    mpc_get_td(ret_c, in_ret_c);

    // ret := |approx - true_val| / |true_val|
    mpc_init2(ctmp, mpf_get_prec(in_ret));
    mpf_init2(tmp, mpf_get_prec(in_ret));

    // ret := |approx - true_val|
    mpc_sub(ctmp, in_approx, true_val, rndc);
    mpc_abs(in_ret, ctmp, rnd);
    mpc_abs(tmp, true_val, rnd);
    if(mpf_cmp_ui(tmp, 0UL) != 0)
    {
        mpf_div(in_ret, in_ret, tmp);
    }
    mpf_get_td(ret, in_ret);

    mpc_clear(ctmp);
    mpf_clear(tmp);
    mpf_clear(in_ret);
    mpc_clear(in_ret_c);
    mpc_clear(in_approx);
}

// cdq_relerr_mpc
void cqd_relerr_mpc(double ret[QDSIZE], cqdfloat *ret_c, cqdfloat *approx, mpc_t true_val)
{
    mpf_t tmp, in_ret;
    mpc_t ctmp, in_ret_c, in_approx;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();
    mpfr_rnd_t rnd = get_bnc_default_rounding_mode();

    mpf_init2(in_ret, 53 * QDSIZE);
    mpc_init2(in_ret_c, 53 * QDSIZE);
    mpc_init2(in_approx, 53 * QDSIZE);

    mpc_set_qd(in_approx, approx);

    // ret_c := relerr(real(approx), real(true_val)) + relerr(imag(approx), imag(true_val))
    mpf_relerr(mpc_realref(in_ret_c), mpc_realref(in_approx), mpc_realref(true_val));
    mpf_relerr(mpc_imagref(in_ret_c), mpc_imagref(in_approx), mpc_imagref(true_val));
    mpc_get_qd(ret_c, in_ret_c);

    // ret := |approx - true_val| / |true_val|
    mpc_init2(ctmp, mpf_get_prec(in_ret));
    mpf_init2(tmp, mpf_get_prec(in_ret));

    // ret := |approx - true_val|
    mpc_sub(ctmp, in_approx, true_val, rndc);
    mpc_abs(in_ret, ctmp, rnd);
    mpc_abs(tmp, true_val, rnd);
    if(mpf_cmp_ui(tmp, 0UL) != 0)
    {
        mpf_div(in_ret, in_ret, tmp);
    }
    mpf_get_qd(ret, in_ret);

    mpc_clear(ctmp);
    mpf_clear(tmp);
    mpf_clear(in_ret);
    mpc_clear(in_ret_c);
    mpc_clear(in_approx);
}

int main(int argc, char *argv[])
{
    unsigned long prec = 212, true_prec = 512, itimes, max_itimes, min_max_itimes = 1000, citimes, max_citimes = 5;//10;
    long int degree = 128, start_deg, end_deg, step_deg, deg;
    long int i, complex_flag = 1;
    double deval, deval_avx2, dx;
    double dd_tmp[DDSIZE], *dd_ptr;
    double ddx[DDSIZE], ddeval[DDSIZE], ddeval_avx2[DDSIZE];
    double tdx[TDSIZE], tdeval[TDSIZE], tdeval_avx2[TDSIZE];
    double qdx[QDSIZE], qdeval[QDSIZE], qdeval_avx2[QDSIZE];
    double drelerr[20], ddrelerr[20][DDSIZE], tdrelerr[20][TDSIZE], qdrelerr[20][QDSIZE];
    double _Complex cdrelerr[20];
    double _Complex dceval, dceval_avx2, dcx;
    DCmplx dcx_d, dceval_d_avx2;
    cddfloat ddceval, ddcx, ddceval_avx2, cddrelerr[20];
    ctdfloat tdceval, tdcx, tdceval_avx2, ctdrelerr[20];
    cqdfloat qdceval, qdcx, qdceval_avx2, cqdrelerr[20];
    double stime, etime, comptime[20], ccomptime[20], min_time = 1.0;// 0.1; //1.0; // 0.1;

    DPoly dpol;
    DDPoly ddpol;
    TDPoly tdpol;
    QDPoly qdpol;
    MPFPoly mpfpol, mpfpol_true;
    mpf_t mpf_tmp, mpf_eval, mpf_x, mpf_tmp_true, mpf_x_true, mpf_eval_true;
    mpc_t mpc_tmp, mpc_eval, mpc_x, mpc_tmp_true, mpc_x_true, mpc_eval_true;
    MPFCmplx mpfcmplx_tmp, mpfcmplx_eval, mpfcmplx_x, mpfcmplx_tmp_true, mpfcmplx_x_true, mpfcmplx_eval_true;

    if(argc <= 1)
    {
        printf("Usage: %s [start_deg] [end_deg] [step_deg] [r(eal) or c(omplex)]\n", argv[0]);
        return 0;
    }

    end_deg = 1024;
    step_deg = 16;

    if(argc >= 2)
    {
        start_deg = (long int)atoi(argv[1]);
        if(argc >= 3)
        {
            end_deg = (long int)atoi(argv[2]);
            if(argc >= 4)
            {
                step_deg = (long int)atoi(argv[3]);
                if(argc >= 5)
                {
                    if(argv[4][0] == 'r')
                        complex_flag = 0; // real
                    else
                        complex_flag = 1;
                }
            }
        }
    }
    if(complex_flag == 0)
        printf("real polynomial eval!\n");
    else
        printf("complex polynomial eval!\n");

    printf("start_deg, end_deg, step_deg = %ld, %ld, %ld\n", start_deg, end_deg, step_deg);

    set_bnc_default_prec(prec);

    mpf_init2(mpf_tmp, prec);
    mpf_init2(mpf_x, prec);
    mpf_init2(mpf_eval, prec);

    mpc_init2(mpc_tmp, prec);
    mpc_init2(mpc_x, prec);
    mpc_init2(mpc_eval, prec);

    mpfcmplx_tmp = init2_mpfcmplx(prec);
    mpfcmplx_x   = init2_mpfcmplx(prec);
    mpfcmplx_eval= init2_mpfcmplx(prec);

    mpf_init2(mpf_tmp_true, true_prec);
    mpf_init2(mpf_x_true, true_prec);
    mpf_init2(mpf_eval_true, true_prec);

    mpc_init2(mpc_tmp_true, true_prec);
    mpc_init2(mpc_x_true, true_prec);
    mpc_init2(mpc_eval_true, true_prec);

    mpfcmplx_tmp_true  = init2_mpfcmplx(true_prec);
    mpfcmplx_x_true    = init2_mpfcmplx(true_prec);
    mpfcmplx_eval_true = init2_mpfcmplx(true_prec);

    // main loop

// 1.907e-07  1.998e-12 2.003e-07  1.998e-12 2.289e-07  1.703e-16 8.392e-07  1.418e-32 8.392e-07  1.418e-32 2.289e-06  6.617e-33 6.714e-06  6.696e-49 6.714e-06  6.696e-49 1.099e-05  6.696e-49 1.648e-05  8.929e-66 1.648e-05  8.929e-66 1.648e-05  8.929e-66 2.441e-05  2.474e-64 6.714e-06  2.474e-64
    printf("degree, " \
        "  D Est AVX2,   D Est,   D Horner," \
        " DD Est AVX2,  DD Est,  DD Horner," \
        " TD Est AVX2,  TD Est,  TD Horner," \
        " QD Est AVX2,  QD Est,  QD Horner," \
        " MPF Est, MPF Horner," \
        "  D Est AVX2,   D Est,   D Horner," \
        " DD Est AVX2,  DD Est,  DD Horner," \
        " TD Est AVX2,  TD Est,  TD Horner," \
        " QD Est AVX2,  QD Est,  QD Horner," \
        " MPFR(%ld-bit) Est, MPFR(%ld-bit) Horner," \
        "  D Normal/AVX2, Horner/Est, Horner/Est AVX2," \
        " DD Normal/AVX2, Horner/Est, Horner/Est AVX2," \
        " TD Normal/AVX2, Horner/Est, Horner/Est AVX2," \
        " QD Normal/AVX2, Horner/Est, Horner/Est AVX2," \
        "MPFR(%ld-bit) Horner/Est, PFR(%ld-bit) Est/QD, PFR(%ld-bit) Est/QD AVX2\n", prec, prec, prec, prec, prec
    );
    for(degree = start_deg; degree <= end_deg; degree += step_deg)
    {

        // init
        dpol = init_dpoly(degree + 1);
        ddpol = init_ddpoly(degree + 1);
        tdpol = init_tdpoly(degree + 1);
        qdpol = init_qdpoly(degree + 1);
        mpfpol = init2_mpfpoly(degree + 1, prec);
        mpfpol_true = init2_mpfpoly(degree + 1, true_prec);

        // set coef from random
        //mpf_srand(20250123);
        mpf_srand(20250207);
        for(i = 0; i <= degree; i++)
        {
            mpf_urand(mpf_tmp_true);
            mpf_set(mpf_tmp, mpf_tmp_true);
            set_mpfpoly_i(mpfpol, i, mpf_tmp_true);
            set_mpfpoly_i(mpfpol_true, i, mpf_tmp_true);
            set_dpoly_i(dpol, i, mpf_get_d(mpf_tmp_true));
            //mpf_get_dd(dd_tmp, mpf_tmp);
            //set_ddpoly_i(ddpol, i, dd_tmp);
            set_ddpoly_i_mpf(ddpol, i, mpf_tmp_true);
            set_tdpoly_i_mpf(tdpol, i, mpf_tmp_true);
            set_qdpoly_i_mpf(qdpol, i, mpf_tmp_true);
        }

        // print
    /*    for(i = 0; i <= degree; i++)
        {
            dd_ptr = get_ddpoly_i(ddpol, i);
            mpfr_printf("%5d: %25.17e %25.17e %25.17RNe\n", i, get_dpoly_i(dpol, i), dd_ptr[0], get_mpfpoly_i(mpfpol, i));
        }
    */

        // -------
        // Real
        // -------
        if(complex_flag == 0) // real eval
        {


            // print
            mpf_urand(mpf_x_true);
            mpf_set(mpf_x, mpf_x_true);
            dx = mpf_get_d(mpf_x);
            mpf_get_dd(ddx, mpf_x);
            mpf_get_td(tdx, mpf_x);
            mpf_get_qd(qdx, mpf_x);

            eval_mpfpoly(mpf_eval_true, mpfpol_true, mpf_x_true);
            //mpfr_printf("mpf_eval   = %25.17RNe\n", mpf_eval);

            // benchmark: double_avx2
            max_itimes = min_max_itimes; // 4;
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    deval_avx2 = _bncavx2_eval_dpoly_estrin(dpol, dx); // eval_dpoly(dpol, dx);
                etime = get_secv() - stime;

                drelerr[0] = drelerr_mpf(deval_avx2, mpf_eval_true);
                
                if(etime >= min_time) {
                    comptime[0] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            // benchmark: double
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    deval = eval_dpoly_estrin(dpol, dx);
                etime = get_secv() - stime;

                drelerr[1] = drelerr_mpf(deval, mpf_eval_true);

                if(etime >= min_time) {
                    comptime[1] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            // benchmark: double
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    deval = eval_dpoly_horner(dpol, dx);
                etime = get_secv() - stime;

                drelerr[2] = drelerr_mpf(deval, mpf_eval_true);

                if(etime >= min_time) {
                    comptime[2] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            //eval_ddpoly(ddeval, ddpol, ddx);
            // benchmark: DD avx2
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    _bncavx2_eval_ddpoly_estrin(ddeval_avx2, ddpol, ddx);
                etime = get_secv() - stime;

                dd_relerr_mpf(ddrelerr[0], ddeval_avx2, mpf_eval_true);
                drelerr[3] = ddrelerr[0][0];

                if(etime >= min_time) {
                    comptime[3] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            //eval_ddpoly(ddeval, ddpol, ddx);
            // benchmark: DD
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    eval_ddpoly_estrin(ddeval, ddpol, ddx);
                etime = get_secv() - stime;
        
                dd_relerr_mpf(ddrelerr[0], ddeval, mpf_eval_true);
                drelerr[4] = ddrelerr[0][0];

                if(etime >= min_time) {
                    comptime[4] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);
        
                //eval_ddpoly(ddeval, ddpol, ddx);
            // benchmark: DD
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    eval_ddpoly_horner(ddeval, ddpol, ddx);
                etime = get_secv() - stime;
        
                dd_relerr_mpf(ddrelerr[0], ddeval, mpf_eval_true);
                drelerr[5] = ddrelerr[0][0];

                if(etime >= min_time) {
                    comptime[5] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            // benchmark: TD avx2
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    _bncavx2_eval_tdpoly_estrin(tdeval_avx2, tdpol, tdx);
                etime = get_secv() - stime;
        
                td_relerr_mpf(tdrelerr[0], tdeval_avx2, mpf_eval_true);
                drelerr[6] = tdrelerr[0][0];

                if(etime >= min_time) {
                    comptime[6] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            // benchmark: TD
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    eval_tdpoly_estrin(tdeval, tdpol, tdx);
                etime = get_secv() - stime;
        
                td_relerr_mpf(tdrelerr[0], tdeval, mpf_eval_true);
                drelerr[7] = tdrelerr[0][0];

                if(etime >= min_time) {
                    comptime[7] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            // benchmark: TD
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    eval_tdpoly_horner(tdeval, tdpol, tdx);
                etime = get_secv() - stime;
        
                td_relerr_mpf(tdrelerr[0], tdeval, mpf_eval_true);
                drelerr[8] = tdrelerr[0][0];
        
                if(etime >= min_time) {
                    comptime[8] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            // benchmark: QD avx2
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    _bncavx2_eval_qdpoly_estrin(qdeval_avx2, qdpol, qdx);
                etime = get_secv() - stime;
        
                qd_relerr_mpf(qdrelerr[0], qdeval_avx2, mpf_eval_true);
                drelerr[9] = qdrelerr[0][0];

                if(etime >= min_time) {
                    comptime[9] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            // benchmark: QD
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    eval_qdpoly_estrin(qdeval, qdpol, qdx);
                etime = get_secv() - stime;
        
                qd_relerr_mpf(qdrelerr[0], qdeval, mpf_eval_true);
                drelerr[10] = qdrelerr[0][0];
        
                if(etime >= min_time) {
                    comptime[10] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            // benchmark: QD
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    eval_qdpoly_horner(qdeval, qdpol, qdx);
                etime = get_secv() - stime;
        
                qd_relerr_mpf(qdrelerr[0], qdeval, mpf_eval_true);
                drelerr[11] = qdrelerr[0][0];
        
                if(etime >= min_time) {
                    comptime[11] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);


            //eval_mpfpoly(mpf_eval, mpfpol, mpf_x);
            // benchmark: MPFR
            max_itimes = min_max_itimes; // 4
            do { 
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    eval_mpfpoly_estrin(mpf_eval, mpfpol, mpf_x);
                etime = get_secv() - stime;
        
                mpf_relerr(mpf_tmp, mpf_eval, mpf_eval_true);
                drelerr[12] = mpf_get_d(mpf_tmp);

                if(etime >= min_time) {
                    comptime[12] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            //eval_mpfpoly(mpf_eval, mpfpol, mpf_x);
            // benchmark: MPFR
            max_itimes = min_max_itimes; // 4
            do { 
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    eval_mpfpoly_horner(mpf_eval, mpfpol, mpf_x);
                etime = get_secv() - stime;
        
                mpf_relerr(mpf_tmp, mpf_eval, mpf_eval_true);
                drelerr[13] = mpf_get_d(mpf_tmp);
        
                if(etime >= min_time) {
                    comptime[13] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);
    /*
            printf("x = %25.17e\n", dx);
            printf("%10.3e us double avx2: %25.17e\n", comptime[0] * 1000000.0, deval_avx2);
            printf("%10.3e us double     : %25.17e\n", comptime[1] * 1000000.0, deval);
            //printf("DD    :  %25.17e\n", ddeval[0]);
            printf("%10.3e us DD avx2    :   ", comptime[2] * 1000000.0); rdd_out_str(ddeval_avx2); printf("\n");
            printf("%10.3e us DD         :   ", comptime[3] * 1000000.0); rdd_out_str(ddeval); printf("\n");
            printf("%10.3e us TD avx2    :   ", comptime[4] * 1000000.0); rtd_out_str(tdeval_avx2); printf("\n");
            printf("%10.3e us TD         :   ", comptime[5] * 1000000.0); rtd_out_str(tdeval); printf("\n");
            printf("%10.3e us QD avx2    :   ", comptime[6] * 1000000.0); rqd_out_str(qdeval_avx2); printf("\n");
            printf("%10.3e us QD         :   ", comptime[7] * 1000000.0); rqd_out_str(qdeval); printf("\n");
            mpfr_printf("%10.3e us MPFR(%5ld):   %RNe\n", comptime[8] * 1000000.0, prec, mpf_eval);
    */
            printf("%5ld, ", degree);
            // comptime
            for(i = 0; i < 14; i++)
            printf("%10.3e, ", comptime[i] * 1000000.0);
            // relerr
            for(i = 0; i < 14; i++)
            printf("%10.3e, ", drelerr[i]);
            // Speedup ratio
            printf("%5.3g, ", comptime[1] / comptime[0]); // D Estrin Normal/AVX2
            printf("%5.3g, ", comptime[2] / comptime[1]); // D Horner/Estrin
            printf("%5.3g, ", comptime[2] / comptime[0]); // D Horner/Estrin AVX2
            printf("%5.3g, ", comptime[4] / comptime[3]); // DD Estrin Normal/AVX2
            printf("%5.3g, ", comptime[5] / comptime[4]); // DD Horner/Estrin
            printf("%5.3g, ", comptime[5] / comptime[3]); // DD Horner/Estrin AVX2
            printf("%5.3g, ", comptime[7] / comptime[6]); // TD Estrin Normal/AVX2
            printf("%5.3g, ", comptime[8] / comptime[7]); // TD Horner/Estrin
            printf("%5.3g, ", comptime[8] / comptime[6]); // TD Horner/Estrin AVX2
            printf("%5.3g, ", comptime[10] / comptime[9]); // QD Estrin Normal/AVX2
            printf("%5.3g, ", comptime[11] / comptime[10]); // QD Horner/Estrin
            printf("%5.3g, ", comptime[11] / comptime[9]); // QD Horner/Estrin AVX2
            printf("%5.3g, ", comptime[13] / comptime[12]); // MPF Horner/Estrin /AVX2
            printf("%5.3g, ", comptime[12] / comptime[9]); //  MPF Estrin/QD Estrin AVX2
            printf("%5.3g, ", comptime[12] / comptime[10]); // MPF Estrin/QD Estrin
            printf("\n");

        } // complex_flag
        else // if(complex_flag != 0)
        {
            // -------
            // Complex
            // -------


            // print
            mpf_urand(mpc_realref(mpc_x_true));
            mpf_urand(mpc_imagref(mpc_x_true));
            mpc_abs(mpf_tmp_true, mpc_x_true, MPFR_RNDN);
            mpc_div_fr(mpc_x_true, mpc_x_true, mpf_tmp_true, MPC_RNDNN); // normalize
            //mpfr_printf("x = %25.17RNE, %25.17RNe\n", mpc_realref(mpc_x_true), mpc_imagref(mpc_x_true));

            ceval_mpfpoly(mpc_eval_true, mpfpol_true, mpc_x_true);
            //mpfr_printf("mpf_ceval   = %25.17RNe, %25.17RNe\n", mpc_realref(mpc_eval_true), mpc_imagref(mpc_eval_true));

            //dcx = mpf_get_d(mpf_x) + mpf_get_d(mpf_tmp) * I;
            dcx = mpf_get_d(mpc_realref(mpc_x_true)) + mpf_get_d(mpc_imagref(mpc_x_true)) * I;

            mpf_get_dd(ddcx.val_re, mpc_realref(mpc_x_true)); //mpf_x);
            mpf_get_dd(ddcx.val_im, mpc_imagref(mpc_x_true)); // mpf_tmp);
            mpf_get_td(tdcx.val_re, mpc_realref(mpc_x_true)); //mpf_x);
            mpf_get_td(tdcx.val_im, mpc_imagref(mpc_x_true)); // mpf_tmp);  
            mpf_get_qd(qdcx.val_re, mpc_realref(mpc_x_true)); //mpf_x);
            mpf_get_qd(qdcx.val_im, mpc_imagref(mpc_x_true)); // mpf_tmp);
            mpc_set(mpc_x, mpc_x_true, get_bnc_default_rounding_mode_c());

            // benchmark: double avx2
            for(citimes = 0; citimes < max_citimes; citimes++)
            {
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    _bncavx2_ceval_dpoly_estrin((DCmplx)&dceval_avx2, dpol, (DCmplx)&dcx);
                etime = get_secv() - stime;
        
                cd_relerr_mpc(&drelerr[0], &cdrelerr[0], dceval_avx2, mpc_eval_true);

                if(etime >= 0.01) {
                    comptime[0] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            // benchmark: double
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    ceval_dpoly_estrin((DCmplx)&dceval, dpol, (DCmplx)&dcx);
                etime = get_secv() - stime;

                cd_relerr_mpc(&drelerr[1], &cdrelerr[1], dceval, mpc_eval_true);

                if(etime >= 0.01) {
                    comptime[1] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            // benchmark: double
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    ceval_dpoly_horner((DCmplx)&dceval, dpol, (DCmplx)&dcx);
                etime = get_secv() - stime;

                cd_relerr_mpc(&drelerr[2], &cdrelerr[2], dceval, mpc_eval_true);

                if(etime >= 0.01) {
                    comptime[2] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);


            //eval_ddpoly(ddeval, ddpol, ddx);
            // benchmark: DD avx2
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    _bncavx2_ceval_ddpoly_estrin(&ddceval_avx2, ddpol, &ddcx);
                etime = get_secv() - stime;
        
                cdd_relerr_mpc(ddrelerr[3], &cddrelerr[3], &ddceval_avx2, mpc_eval_true);
                drelerr[3] = ddrelerr[3][0];

                if(etime >= 0.01) {
                    comptime[3] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            //eval_ddpoly(ddeval, ddpol, ddx);
            // benchmark: DD
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    ceval_ddpoly_estrin(&ddceval, ddpol, &ddcx);
                etime = get_secv() - stime;
        
                cdd_relerr_mpc(ddrelerr[4], &cddrelerr[4], &ddceval, mpc_eval_true);
                drelerr[4] = ddrelerr[4][0];

                if(etime >= 0.01) {
                    comptime[4] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            //eval_ddpoly(ddeval, ddpol, ddx);
            // benchmark: DD
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    ceval_ddpoly_horner(&ddceval, ddpol, &ddcx);
                etime = get_secv() - stime;
        
                cdd_relerr_mpc(ddrelerr[5], &cddrelerr[5], &ddceval, mpc_eval_true);
                drelerr[5] = ddrelerr[5][0];

                if(etime >= 0.01) {
                    comptime[5] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            // benchmark: TD avx2
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    _bncavx2_ceval_tdpoly_estrin(&tdceval_avx2, tdpol, &tdcx);
                etime = get_secv() - stime;

                ctd_relerr_mpc(tdrelerr[6], &ctdrelerr[6], &tdceval_avx2, mpc_eval_true);
                drelerr[6] = tdrelerr[6][0];

                if(etime >= 0.01) {
                    comptime[6] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            // benchmark: TD avx2
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    ceval_tdpoly_estrin(&tdceval_avx2, tdpol, &tdcx);
                etime = get_secv() - stime;

                ctd_relerr_mpc(tdrelerr[7], &ctdrelerr[7], &tdceval_avx2, mpc_eval_true);
                drelerr[7] = tdrelerr[7][0];

                if(etime >= 0.01) {
                    comptime[7] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            // benchmark: TD
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    ceval_tdpoly_horner(&tdceval, tdpol, &tdcx);
                etime = get_secv() - stime;
        
                ctd_relerr_mpc(tdrelerr[8], &ctdrelerr[8], &tdceval, mpc_eval_true);
                drelerr[8] = tdrelerr[8][0];

                if(etime >= 0.01) {
                    comptime[8] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            // benchmark: QD avx2
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    _bncavx2_ceval_qdpoly_estrin(&qdceval_avx2, qdpol, &qdcx);
                etime = get_secv() - stime;

                cqd_relerr_mpc(qdrelerr[9], &cqdrelerr[9], &qdceval_avx2, mpc_eval_true);
                drelerr[9] = qdrelerr[9][0];

                if(etime >= 0.01) {
                    comptime[9] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            // benchmark: QD
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    ceval_qdpoly_estrin(&qdceval_avx2, qdpol, &qdcx);
                etime = get_secv() - stime;

                cqd_relerr_mpc(qdrelerr[10], &cqdrelerr[10], &qdceval_avx2, mpc_eval_true);
                drelerr[10] = qdrelerr[10][0];

                if(etime >= 0.01) {
                    comptime[10] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            // benchmark: QD
            max_itimes = min_max_itimes; // 4
            do {
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    ceval_qdpoly_horner(&qdceval, qdpol, &qdcx);
                etime = get_secv() - stime;

                cqd_relerr_mpc(qdrelerr[11], &cqdrelerr[11], &qdceval, mpc_eval_true);
                drelerr[11] = qdrelerr[11][0];

                if(etime >= 0.01) {
                    comptime[11] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            //eval_mpfpoly(mpf_eval, mpfpol, mpf_x);
            // benchmark: MPFR
            max_itimes = min_max_itimes; // 4
            do { 
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    ceval_mpfpoly_estrin(mpc_eval, mpfpol, mpc_x);
                etime = get_secv() - stime;

                mpc_relerr(mpf_tmp, mpc_tmp, mpc_eval, mpc_eval_true);
                drelerr[12] = mpf_get_d(mpf_tmp);

                if(etime >= 0.01) {
                    comptime[12] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            //eval_mpfpoly(mpf_eval, mpfpol, mpf_x);
            // benchmark: MPFR
            max_itimes = min_max_itimes; // 4
            do { 
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    ceval_mpfpoly_horner(mpc_eval, mpfpol, mpc_x);
                etime = get_secv() - stime;

                mpc_relerr(mpf_tmp, mpc_tmp, mpc_eval, mpc_eval_true);
                drelerr[13] = mpf_get_d(mpf_tmp);

                if(etime >= 0.01) {
                    comptime[13] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);

            //_bncold_eval_mpfpoly(mpf_eval, mpfpol, mpf_x);
            // benchmark: MPFR
            max_itimes = min_max_itimes; // 4
            do { 
                stime = get_secv();
                for(itimes = 0; itimes < max_itimes; itimes++)
                    _bncold_ceval_mpfpoly(mpfcmplx_eval, mpfpol, mpfcmplx_x);
                etime = get_secv() - stime;

                mpf_set(mpc_realref(mpc_eval), mpfcmplx_eval->re);
                mpf_set(mpc_imagref(mpc_eval), mpfcmplx_eval->im);

                mpc_relerr(mpf_tmp, mpc_tmp, mpc_eval, mpc_eval_true);
                drelerr[14] = mpf_get_d(mpf_tmp);

                if(etime >= 0.01) {
                    comptime[14] = etime / (double)max_itimes;
                    break;
                }
                else
                    max_itimes *= 4;
            } while(1);


            if(citimes == 0)
                for(int i = 0; i < 20; i++) ccomptime[i] = comptime[i];
            else
                for(int i = 0; i < 20; i++) ccomptime[i] += comptime[i];
        } // ctimes
        for(i = 0; i < 20; i++) comptime[i] = ccomptime[i] / (double)max_citimes; //5.0;
    /*
            printf("x = %25.17e\n", dx);
            printf("real part\n");
            printf("%10.3e us cdouble avx2: %25.17e\n", comptime[0] * 1000000.0, creal(dceval_avx2));
            printf("%10.3e us cdouble     : %25.17e\n", comptime[1] * 1000000.0, creal(dceval));
            //printf("DD    :  %25.17e\n", ddeval[0]);
            printf("%10.3e us CDD avx2    :   ", comptime[2] * 1000000.0); rdd_out_str(ddceval_avx2.val_re); printf("\n");
            printf("%10.3e us CDD         :   ", comptime[3] * 1000000.0); rdd_out_str(ddceval.val_re); printf("\n");
            printf("%10.3e us CTD avx2    :   ", comptime[4] * 1000000.0); rtd_out_str(tdceval_avx2.val_re); printf("\n");
            printf("%10.3e us CTD         :   ", comptime[5] * 1000000.0); rtd_out_str(tdceval.val_re); printf("\n");
            printf("%10.3e us CQD avx2    :   ", comptime[6] * 1000000.0); rqd_out_str(qdceval_avx2.val_re); printf("\n");
            printf("%10.3e us CQD         :   ", comptime[7] * 1000000.0); rqd_out_str(qdceval.val_re); printf("\n");
            mpfr_printf("%10.3e us CMPFR(%5ld):   %RNe\n", comptime[8] * 1000000.0, prec, mpc_realref(mpc_eval));
            printf("imag part\n");
            printf("%10.3e us cdouble avx2: %25.17e\n", comptime[0] * 1000000.0, cimag(dceval_avx2));
            printf("%10.3e us cdouble     : %25.17e\n", comptime[1] * 1000000.0, cimag(dceval));
            //printf("DD    :  %25.17e\n", ddeval[0]);
            printf("%10.3e us CDD avx2    :   ", comptime[2] * 1000000.0); rdd_out_str(ddceval_avx2.val_im); printf("\n");
            printf("%10.3e us CDD         :   ", comptime[3] * 1000000.0); rdd_out_str(ddceval.val_im); printf("\n");
            printf("%10.3e us CTD avx2    :   ", comptime[4] * 1000000.0); rtd_out_str(tdceval_avx2.val_im); printf("\n");
            printf("%10.3e us CTD         :   ", comptime[5] * 1000000.0); rtd_out_str(tdceval.val_im); printf("\n");
            printf("%10.3e us CQD avx2    :   ", comptime[6] * 1000000.0); rqd_out_str(qdceval_avx2.val_im); printf("\n");
            printf("%10.3e us CQD         :   ", comptime[7] * 1000000.0); rqd_out_str(qdceval.val_im); printf("\n");
            mpfr_printf("%10.3e us CMPFR(%5ld):   %RNe\n", comptime[8] * 1000000.0, prec, mpc_imagref(mpc_eval));
    */

            printf("%5ld, ", degree);
            // comptime
            for(i = 0; i < 14; i++)
            printf("%10.3e, ", comptime[i] * 1000000.0);
            // relerr
            for(i = 0; i < 14; i++)
            printf("%10.3e, ", drelerr[i]);
            // Speedup ratio
            printf("%5.3g, ", comptime[1] / comptime[0]); // D Estrin Normal/AVX2
            printf("%5.3g, ", comptime[2] / comptime[1]); // D Horner/Estrin
            printf("%5.3g, ", comptime[2] / comptime[0]); // D Horner/Estrin AVX2
            printf("%5.3g, ", comptime[4] / comptime[3]); // DD Estrin Normal/AVX2
            printf("%5.3g, ", comptime[5] / comptime[4]); // DD Horner/Estrin
            printf("%5.3g, ", comptime[5] / comptime[3]); // DD Horner/Estrin AVX2
            printf("%5.3g, ", comptime[7] / comptime[6]); // TD Estrin Normal/AVX2
            printf("%5.3g, ", comptime[8] / comptime[7]); // TD Horner/Estrin
            printf("%5.3g, ", comptime[8] / comptime[6]); // TD Horner/Estrin AVX2
            printf("%5.3g, ", comptime[10] / comptime[9]); // QD Estrin Normal/AVX2
            printf("%5.3g, ", comptime[11] / comptime[10]); // QD Horner/Estrin
            printf("%5.3g, ", comptime[11] / comptime[9]); // QD Horner/Estrin AVX2
            printf("%5.3g, ", comptime[13] / comptime[12]); // MPF Horner/Estrin /AVX2
            printf("%5.3g, ", comptime[12] / comptime[9]); //  MPF Estrin/QD Estrin AVX2
            printf("%5.3g, ", comptime[12] / comptime[10]); // MPF Estrin/QD Estrin
            printf("\n");


        } // complex_flag;

        free_dpoly(dpol);
        free_ddpoly(ddpol);
        free_tdpoly(tdpol);
        free_qdpoly(qdpol);
        free_mpfpoly(mpfpol);
        free_mpfpoly(mpfpol_true);
    } // main_loop

    // clear
    mpf_clear(mpf_tmp);
    mpf_clear(mpf_x);
    mpf_clear(mpf_eval);
    mpc_clear(mpc_tmp);
    mpc_clear(mpc_x);
    mpc_clear(mpc_eval);

    mpf_clear(mpf_tmp_true);
    mpf_clear(mpf_x_true);
    mpf_clear(mpf_eval_true);
    mpc_clear(mpc_tmp_true);
    mpc_clear(mpc_x_true);
    mpc_clear(mpc_eval_true);

    return 0;
}