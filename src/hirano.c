/********************************************************************************/
/* hirano.c: Robust solver for lgebraic Equations using Hirano method           */
/* copyright (c) 2025 Tomonori Kouya                                            */
/*                                                                              */
/* Ver. 0.1 2025-03-10: First implementation                                    */
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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

// MPF & MPC
//#define USE_GMP
//#define USE_MPFR
//#include "bncmatmul.h"
#include "poly.h"

// --------------------------
// Double precision
// --------------------------

// Horner method
// return p(x)
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
double _Complex dhorner(double _Complex x, DPoly pol) // double coef[], long int deg)
{
    long int i;
    double _Complex ret;

    ret = pol->coef[pol->deg] + 0.0 * I;
    for(i = pol->deg - 1; i >= 0; i--)
        ret = ret * x + (pol->coef[i] + 0.0 * I);

    return ret;
}

// Horner method
// return p(x)
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
double _Complex cdhorner(double _Complex x, CDPoly pol) // double coef[], long int deg)
{
    long int i;
    double _Complex ret;

    ret = pol->coef[pol->deg]; // + 0.0 * I;
    for(i = pol->deg - 1; i >= 0; i--)
        ret = ret * x + pol->coef[i]; //  + 0.0 * I);

    return ret;
}

// Coef of p(x + d)
// 
// based on Horner method
void dcoef_horner(double _Complex ret_coef[], double _Complex x, DPoly pol) // double coef[], long int deg)
{
    long int l, i;

    // Initial setting
    for(i = 0; i <= pol->deg; i++)
        ret_coef[i] = pol->coef[i] + 0.0 * I;

    // ret_coef[i] := p^(l)(x) / l!
    for(l = 0; l <= pol->deg; l++)
    {
        for(i = pol->deg - 1; i >= l; i--)
            ret_coef[i] = x * ret_coef[i + 1] + ret_coef[i];
    }
}

// Coef of p(x + d)
// 
// based on Horner method
void cdcoef_horner(double _Complex ret_coef[], double _Complex x, CDPoly pol) // double coef[], long int deg)
{
    long int l, i;

    // Initial setting
    for(i = 0; i <= pol->deg; i++)
        ret_coef[i] = pol->coef[i]; // + 0.0 * I;

    // ret_coef[i] := p^(l)(x) / l!
    for(l = 0; l <= pol->deg; l++)
    {
        for(i = pol->deg - 1; i >= l; i--)
            ret_coef[i] = x * ret_coef[i + 1] + ret_coef[i];
    }
}

// return j and max|a_j x^j| from p(x)
long int absmax_dpoly(double *absmax_anxn, double _Complex x, DPoly pol) //double coef[], long int deg)
{
    long int i, ret_i;
    double abs_anxn;
    double _Complex xn = x, anxn;

    ret_i = 0;
    *absmax_anxn = pol->coef[0]; //fabs(get_dpoly_i(pol, 0)); // coef[0]);
    for(i = 1; i <= pol->deg; i++)
    {
        anxn = pol->coef[i] * xn;
        abs_anxn = cabs(anxn); // fabs(anxn);
        if(*absmax_anxn < abs_anxn)
        {
            ret_i = i;
            *absmax_anxn = abs_anxn;
        }
        xn *= x;
    }

    return ret_i;
}

// return j and max|a_j x^j| from p(x)
long int absmax_cdpoly(double *absmax_anxn, double _Complex x, CDPoly pol) //double coef[], long int deg)
{
    long int i, ret_i;
    double abs_anxn;
    double _Complex xn = x, anxn;

    ret_i = 0;
    *absmax_anxn = cabs(get_cdpoly_i(pol, 0)); // coef[0]);
    for(i = 1; i <= pol->deg; i++)
    {
        anxn = pol->coef[i] * xn;
        abs_anxn = cabs(anxn); // fabs(anxn);
        if(*absmax_anxn < abs_anxn)
        {
            ret_i = i;
            *absmax_anxn = abs_anxn;
        }
        xn *= x;
    }

    return ret_i;
}

// get_plus_arg
// return arg(x) in [0, 2 PI]
double dget_plus_arg(double _Complex x)
{
    double ret = carg(x);

    if(ret < 0)
        ret = 2.0 * M_PI + ret;

    return ret;
}

// get_nearest_int
long int dget_nearest_int(double real_x)
{
    long int int_x_floor, int_x_ceil;
    double dist_x_floor, dist_x_ceil;

    int_x_floor = (long int)floor(real_x);
    int_x_ceil  = (long int)ceil (real_x);

    dist_x_floor = fabs(real_x - (double)int_x_floor);
    dist_x_ceil  = fabs(real_x - (double)int_x_ceil);

    if(dist_x_floor < dist_x_ceil) return int_x_floor;
    else return int_x_ceil;
}

// get_min_branch
double _Complex dget_min_branch(double _Complex x, double mu, double _Complex coef[], long int i_num, long int i_den)
{
    long int j = i_den;
    double phi, psi, pi2 = 2.0 * M_PI, real_j;
    double _Complex ret = FP_NAN + FP_NAN * I;

    if(cabs(coef[j]) == 0.0) return ret;

    phi = dget_plus_arg(x) / pi2;
    psi = dget_plus_arg(-coef[i_num] / coef[j]) / pi2;

    real_j = (double)dget_nearest_int((double)j * (0.5 - phi) - psi);
    //printf("real_j = %25.17e\n", real_j);

    ret = pow(cabs(mu * coef[i_num] / coef[j]), 1.0 / (double)j) * cexp(pi2 * I * ((psi + real_j) / (double)j));

    //printf("pow(%25.17e, %25.17e) = %25.17e\n", cabs(mu * coef[i_num] / coef[j]), 1.0 / (double)j, pow(cabs(mu * coef[i_num] / coef[j]), 1.0 / (double)j));
    //printf("cexp(%25.17e + %25.17e * I) = %25.17e + %25.17e * I\n", creal(pi2 * I * ((psi + real_j) / (double)j)), cimag(pi2 * I * ((psi + real_j) / (double)j)), creal(cexp(pi2 * I * ((psi + real_j) / (double)j))), cimag(cexp(pi2 * I * ((psi + real_j) / (double)j))));

    return ret;
}

// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
//long int dhirano(double _Complex *ret, double _Complex init_x, double coef[], long int deg, double reps, double aeps, long int maxtimes)
long int dhirano(double _Complex *ret, double _Complex init_x, DPoly pol, double reps, double aeps, long int maxtimes)
{
    int stop_flag = 0;
    long int i, j, max_j, times, absmin_j, deg = pol->deg;
    double _Complex old_x, new_x, pn_x_d, ctmp;
    double _Complex *in_coef, *d;
    //dcmplx ctmp, pn_x_d;
    double abs_pn_new_x, absmax_anxn, absmin_d, abs_d;
    double mu, beta = 3.0 / 4.0, lambda = 2.0; // from Sugiura & Murota

    // coef of p(x + d)
    in_coef = (double _Complex *)calloc(deg + 1, sizeof(double _Complex));
    d = (double _Complex *)calloc(deg + 1, sizeof(double _Complex));

    old_x = init_x;

    // Main loop
    for(times = 0; times < maxtimes; times++)
    {
        dcoef_horner(in_coef, old_x, pol); // coef, deg);
        mu = 1.0;

        printf("%5ld %25.17e + %25.17e * I -> %15.7e + %15.7e * I\n", times, creal(old_x), cimag(old_x), creal(in_coef[0]), cimag(in_coef[0]));

        // (1)
        d[0] = 0.0 + 0.0 * I;
        if(cabs(in_coef[1]) != 0.0)
            d[0] = -in_coef[0] / in_coef[1];

        //mpfr_printf("d[0] = %25.17e + %25.17e * I\n", creal(d[0]), cimag(d[0]));

        new_x = old_x + d[0];
        pn_x_d = dhorner(new_x, pol); // coef, deg);
        //ceval_dpoly(&pn_x_d, pol, &new_x); // coef, deg);

        if(cabs(pn_x_d) <= beta * cabs(in_coef[0]))
            old_x = new_x;
        else
        {
            for(max_j = 2; max_j <= deg; max_j++)
            {
                //printf("max_j = %ld\n", max_j);
                mu /= lambda;
                d[max_j] = dget_min_branch(old_x, mu, in_coef, 0, max_j);
                absmin_d = cabs(d[max_j]);
                absmin_j = max_j;
                for(j = 0; j < max_j; j++)
                {
                    d[j] = dget_min_branch(old_x, mu, in_coef, 0, j + 1) / pow(lambda, 1.0 / (double)(j + 1));
                    //printf("d[%ld] = %25.17e + %25.17e * I, lambda^%ld = %25.17e\n", j, creal(d[j]), cimag(d[j]), j + 1, pow(lambda, 1.0 / (double)(j + 1)));
                    abs_d = cabs(d[j]);
                    if(abs_d < absmin_d)
                    {
                        absmin_d = abs_d;
                        absmin_j = j;
                    }
                }
                //printf("absmin_j = %ld\n", absmin_j);
                new_x = old_x + d[absmin_j];
                pn_x_d = dhorner(new_x, pol); // coef, deg);
                //ceval_dpoly(&pn_x_d, new_x, pol); // coef, deg);
                // printf("d[%ld] = %25.17e + %25.17e * I\n", absmin_j, creal(d[absmin_j]), cimag(d[absmin_j]));
                // printf("old_x = %25.17e + %25.17e * I\n", creal(old_x), cimag(old_x));
                // printf("new_x = %25.17e + %25.17e * I\n", creal(new_x), cimag(new_x));
                // printf("pn_x_d = %25.17e + %25.17e * I\n", creal(new_x), cimag(new_x));
                // printf("tmp = %25.17e <= tmp2 = %25.17e ? \n", cabs(pn_x_d), (1.0 - (1.0 - beta) * mu) * cabs(in_coef[0]));
                if(cabs(pn_x_d) <= (1.0 - (1.0 - beta) * mu) * cabs(in_coef[0]))
                {
                    old_x = new_x;
                    break;
                }        
            }
        }

        // check stopping rule
        absmax_dpoly(&absmax_anxn, new_x, pol); // coef, deg);
        abs_pn_new_x = cabs(dhorner(new_x, pol)); // coef, deg));
        //ceval_dpoly(&ctmp, new_x, pol); // coef, deg));
        //abs_pn_new_x = cabs(ctmp);
        if(abs_pn_new_x <= absmax_anxn * reps + aeps)
            break;
    }

    free(in_coef);
    free(d);

    *ret = new_x;

    return times;
}

// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int cdhirano(double _Complex *ret, double _Complex init_x, CDPoly pol, double reps, double aeps, long int maxtimes)
{
    int stop_flag = 0;
    long int i, j, max_j, times, absmin_j, deg = pol->deg;
    double _Complex old_x, new_x, pn_x_d, ctmp;
    double _Complex *in_coef, *d;
    //dcmplx ctmp, pn_x_d;
    double abs_pn_new_x, absmax_anxn, absmin_d, abs_d;
    double mu, beta = 3.0 / 4.0, lambda = 2.0; // from Sugiura & Murota

    // coef of p(x + d)
    in_coef = (double _Complex *)calloc(deg + 1, sizeof(double _Complex));
    d = (double _Complex *)calloc(deg + 1, sizeof(double _Complex));

    old_x = init_x;

    // Main loop
    for(times = 0; times < maxtimes; times++)
    {
        cdcoef_horner(in_coef, old_x, pol); // coef, deg);
        mu = 1.0;

        printf("%5ld %25.17e + %25.17e * I -> %15.7e + %15.7e * I\n", times, creal(old_x), cimag(old_x), creal(in_coef[0]), cimag(in_coef[0]));

        // (1)
        d[0] = 0.0 + 0.0 * I;
        if(cabs(in_coef[1]) != 0.0)
            d[0] = -in_coef[0] / in_coef[1];

        //mpfr_printf("d[0] = %25.17e + %25.17e * I\n", creal(d[0]), cimag(d[0]));

        new_x = old_x + d[0];
        pn_x_d = cdhorner(new_x, pol); // coef, deg);
        //ceval_dpoly(&pn_x_d, pol, &new_x); // coef, deg);

        if(cabs(pn_x_d) <= beta * cabs(in_coef[0]))
            old_x = new_x;
        else
        {
            for(max_j = 2; max_j <= deg; max_j++)
            {
                //printf("max_j = %ld\n", max_j);
                mu /= lambda;
                d[max_j] = dget_min_branch(old_x, mu, in_coef, 0, max_j);
                absmin_d = cabs(d[max_j]);
                absmin_j = max_j;
                for(j = 0; j < max_j; j++)
                {
                    d[j] = dget_min_branch(old_x, mu, in_coef, 0, j + 1) / pow(lambda, 1.0 / (double)(j + 1));
                    //printf("d[%ld] = %25.17e + %25.17e * I, lambda^%ld = %25.17e\n", j, creal(d[j]), cimag(d[j]), j + 1, pow(lambda, 1.0 / (double)(j + 1)));
                    abs_d = cabs(d[j]);
                    if(abs_d < absmin_d)
                    {
                        absmin_d = abs_d;
                        absmin_j = j;
                    }
                }
                //printf("absmin_j = %ld\n", absmin_j);
                new_x = old_x + d[absmin_j];
                pn_x_d = cdhorner(new_x, pol); // coef, deg);
                //ceval_dpoly(&pn_x_d, new_x, pol); // coef, deg);
                // printf("d[%ld] = %25.17e + %25.17e * I\n", absmin_j, creal(d[absmin_j]), cimag(d[absmin_j]));
                // printf("old_x = %25.17e + %25.17e * I\n", creal(old_x), cimag(old_x));
                // printf("new_x = %25.17e + %25.17e * I\n", creal(new_x), cimag(new_x));
                // printf("pn_x_d = %25.17e + %25.17e * I\n", creal(new_x), cimag(new_x));
                // printf("tmp = %25.17e <= tmp2 = %25.17e ? \n", cabs(pn_x_d), (1.0 - (1.0 - beta) * mu) * cabs(in_coef[0]));
                if(cabs(pn_x_d) <= (1.0 - (1.0 - beta) * mu) * cabs(in_coef[0]))
                {
                    old_x = new_x;
                    break;
                }        
            }
        }

        // check stopping rule
        absmax_cdpoly(&absmax_anxn, new_x, pol); // coef, deg);
        abs_pn_new_x = cabs(cdhorner(new_x, pol)); // coef, deg));
        //ceval_dpoly(&ctmp, new_x, pol); // coef, deg));
        //abs_pn_new_x = cabs(ctmp);
        if(abs_pn_new_x <= absmax_anxn * reps + aeps)
            break;
    }

    free(in_coef);
    free(d);

    *ret = new_x;

    return times;
}

// --------------------------
// MPF & MPC
// --------------------------
#ifdef USE_GMP

// Horner method
// return p(x)
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
void mpf_horner(mpc_t ret, mpc_t x, MPFPoly poly) // mpf_t coef[], long int deg)
{
    long int i;
    //double _Complex ret;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

    //ret = coef[deg] + 0.0 * I;
    mpc_set_ui(ret, 0UL, rndc);
    mpc_set_fr(ret, get_mpfpoly_i(poly, poly->deg), rndc); // , 0UL);
    for(i = poly->deg - 1; i >= 0; i--)
    {
        //ret = ret * x + (coef[i] + 0.0 * I);
        mpc_mul(ret, ret, x, rndc);
        mpc_add_fr(ret, ret, get_mpfpoly_i(poly, i), rndc);
    }

    //return ret;
    return;
}

// Coef of p(x + d)
// 
// based on Horner method
void mpf_coef_horner(CMPFArray ret_coef, mpc_t x, MPFPoly poly) // mpf_t coef[], long int deg)
{
    long int l, i;
    mpc_t ctmp;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

    mpc_init2(ctmp, ret_coef->prec); // mpc_get_prec(ret_coef[0]));

    // Initial setting
    for(i = 0; i <= poly->deg; i++)
    {
        //ret_coef[i] = coef[i] + 0.0 * I;
        mpc_set_fr(get_cmpfarray_i(ret_coef, i), get_mpfpoly_i(poly, i), rndc);
    }

    // ret_coef[i] := p^(l)(x) / l!
    for(l = 0; l <= poly->deg; l++)
    {
        for(i = poly->deg - 1; i >= l; i--)
        {
            //ret_coef[i] = x * ret_coef[i + 1] + ret_coef[i];
            mpc_mul(ctmp, get_cmpfarray_i(ret_coef, i + 1), x, rndc);
            mpc_add(get_cmpfarray_i(ret_coef, i), ctmp, get_cmpfarray_i(ret_coef, i), rndc);
        }
    }

    mpc_clear(ctmp);
}

// Horner method
// return p(x)
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
void mpc_horner(mpc_t ret, mpc_t x, CMPFPoly poly) // mpf_t coef[], long int deg)
{
    long int i;
    //double _Complex ret;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

    //ret = coef[deg] + 0.0 * I;
    mpc_set_ui(ret, 0UL, rndc);
    mpc_set(ret, get_cmpfpoly_i(poly, poly->deg), rndc); // , 0UL);
    for(i = poly->deg - 1; i >= 0; i--)
    {
        //ret = ret * x + (coef[i] + 0.0 * I);
        mpc_mul(ret, ret, x, rndc);
        mpc_add(ret, ret, get_cmpfpoly_i(poly, i), rndc);
    }

    //return ret;
    return;
}

void mpc_coef_horner(CMPFArray ret_coef, mpc_t x, CMPFPoly poly) // mpf_t coef[], long int deg)
{
    long int l, i;
    mpc_t ctmp;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

    mpc_init2(ctmp, ret_coef->prec); // mpc_get_prec(ret_coef[0]));

    // Initial setting
    for(i = 0; i <= poly->deg; i++)
    {
        //ret_coef[i] = coef[i] + 0.0 * I;
        mpc_set(get_cmpfarray_i(ret_coef, i), get_cmpfpoly_i(poly, i), rndc);
    }

    // ret_coef[i] := p^(l)(x) / l!
    for(l = 0; l <= poly->deg; l++)
    {
        for(i = poly->deg - 1; i >= l; i--)
        {
            //ret_coef[i] = x * ret_coef[i + 1] + ret_coef[i];
            mpc_mul(ctmp, get_cmpfarray_i(ret_coef, i + 1), x, rndc);
            mpc_add(get_cmpfarray_i(ret_coef, i), ctmp, get_cmpfarray_i(ret_coef, i), rndc);
        }
    }

    mpc_clear(ctmp);
}

// return j and max|a_j x^j| from p(x)
long int absmax_mpfpoly(mpf_t absmax_anxn, mpc_t x, MPFPoly poly) // mpf_t coef[], long int deg)
{
    unsigned long prec = mpf_get_prec(absmax_anxn);
    long int i, ret_i;
    mpf_t abs_anxn;
    mpc_t xn, anxn;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();
    mpfr_rnd_t rnd = get_bnc_default_rounding_mode();

    mpf_init2(abs_anxn, prec);
    mpc_init2(xn, prec);
    mpc_init2(anxn, prec);

    // xn := x
    mpc_set(xn, x, rndc);

    ret_i = 0;
    //*absmax_anxn = fabs(coef[0]);
    mpf_abs(absmax_anxn, get_mpfpoly_i(poly, 0));
 
    for(i = 1; i <= poly->deg; i++)
    {
        //anxn = coef[i] * xn;
        mpc_mul_fr(anxn, xn, get_mpfpoly_i(poly, i), rndc);
        //abs_anxn = fabs(anxn);
        mpc_abs(abs_anxn, anxn, rnd);
        //if(*absmax_anxn < abs_anxn)
        if(mpf_cmp(absmax_anxn, abs_anxn) < 0)
        {
            ret_i = i;
            //*absmax_anxn = abs_anxn;
            mpf_set(absmax_anxn, abs_anxn);
        }
        //xn *= x;
        mpc_mul(xn, xn, x, rndc);
    }

    // fix! 2026-02-01 by T.Kouya
    mpf_clear(abs_anxn);
    mpc_clear(xn);
    mpc_clear(anxn);

    return ret_i;
}

long int absmax_cmpfpoly(mpf_t absmax_anxn, mpc_t x, CMPFPoly poly) // mpf_t coef[], long int deg)
{
    unsigned long prec = mpf_get_prec(absmax_anxn);
    long int i, ret_i;
    mpf_t abs_anxn;
    mpc_t xn, anxn;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();
    mpfr_rnd_t rnd = get_bnc_default_rounding_mode();

    mpf_init2(abs_anxn, prec);
    mpc_init2(xn, prec);
    mpc_init2(anxn, prec);

    // xn := x
    mpc_set(xn, x, rndc);

    ret_i = 0;
    //*absmax_anxn = fabs(coef[0]);
    mpc_abs(absmax_anxn, get_cmpfpoly_i(poly, 0), rnd);
 
    for(i = 1; i <= poly->deg; i++)
    {
        //anxn = coef[i] * xn;
        mpc_mul(anxn, xn, get_cmpfpoly_i(poly, i), rndc);
        //abs_anxn = fabs(anxn);
        mpc_abs(abs_anxn, anxn, rnd);
        //if(*absmax_anxn < abs_anxn)
        if(mpf_cmp(absmax_anxn, abs_anxn) < 0)
        {
            ret_i = i;
            //*absmax_anxn = abs_anxn;
            mpf_set(absmax_anxn, abs_anxn);
        }
        //xn *= x;
        mpc_mul(xn, xn, x, rndc);
    }

    // fix! 2026-02-01 by T.Kouya
    mpf_clear(abs_anxn);
    mpc_clear(xn);
    mpc_clear(anxn);

    return ret_i;
}


// get_plus_arg
// return arg(x) in [0, 2 PI]
void mpf_get_plus_arg(mpf_t ret, mpc_t x)
{
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();
    mpf_t pi2;

    // pi2 := 2 * PI
    mpf_init2(pi2, mpf_get_prec(ret));
    mpfr_const_pi(pi2, get_bnc_default_rounding_mode());
    mpf_mul_ui(pi2, pi2, 2UL);

    //double ret = carg(x);
    mpc_arg(ret, x, rndc);

    //if(ret < 0)
    if(mpf_cmp_ui(ret, 0UL) < 0)
    {
        //ret = 2.0 * M_PI + ret;
        mpf_add(ret, ret, pi2);
    }

    mpf_clear(pi2);

    //return ret;
    return;
}

// get_nearest_int
void mpf_get_nearest_int(mpf_t ret, mpf_t real_x)
{
    unsigned long prec = mpf_get_prec(ret);
    //long int int_x_floor, int_x_ceil;
    //double dist_x_floor, dist_x_ceil;
    mpf_t int_x_floor, int_x_ceil;
    mpf_t dist_x_floor, dist_x_ceil;
    mpfr_rnd_t rnd = get_bnc_default_rounding_mode();

    mpf_init2(int_x_floor, prec);
    mpf_init2(int_x_ceil, prec);
    mpf_init2(dist_x_floor, prec);
    mpf_init2(dist_x_ceil, prec);

    //int_x_floor = (long int)floor(real_x);
    mpfr_floor(int_x_floor, real_x); // , rnd);
    //int_x_ceil  = (long int)ceil (real_x);
    mpfr_ceil(int_x_ceil, real_x); // , rnd);

    //dist_x_floor = fabs(real_x - (double)int_x_floor);
    mpf_sub(dist_x_floor, real_x, int_x_floor); mpf_abs(dist_x_floor, dist_x_floor);
    //dist_x_ceil  = fabs(real_x - (double)int_x_ceil);
    mpf_sub(dist_x_ceil, real_x, int_x_ceil); mpf_abs(dist_x_ceil, dist_x_ceil);

    //if(dist_x_floor < dist_x_ceil) return int_x_floor;
    if(mpf_cmp(dist_x_floor, dist_x_ceil)) mpf_set(ret, int_x_floor);
    //else return int_x_ceil;
    else mpf_set(ret, int_x_ceil);

    mpf_clear(int_x_floor);
    mpf_clear(int_x_ceil);
    mpf_clear(dist_x_floor);
    mpf_clear(dist_x_ceil);
}

// get_min_branch
void mpf_get_min_branch(mpc_t ret, mpc_t x, mpf_t mu, CMPFArray coef, long int i_num, long int i_den)
{
    unsigned long prec = mpc_get_prec(ret);
    long int j = i_den;
    mpf_t phi, psi, pi2, real_j, tmp, tmp2, pow_tmp;
    mpc_t ctmp, ctmp2;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();
    mpfr_rnd_t rnd = get_bnc_default_rounding_mode();

    //mpfr_printf("j = i_den = %ld, coef[j] = %25.17RNe + %25.17RNe * I \n", j, mpc_realref(get_cmpfarray_i(coef, j)), mpc_imagref(get_cmpfarray_i(coef, j)));
    //printf("prec = %ld\n", prec);
    mpf_init2(pi2, prec);

    mpf_init2(phi, prec);
    mpf_init2(psi, prec);
    mpf_init2(real_j, prec);
    mpf_init2(tmp, prec);
    mpf_init2(tmp2, prec);
    mpf_init2(pow_tmp, prec);
    mpc_set_nan(ret); // = FP_NAN + FP_NAN * I;
    mpc_init2(ctmp, prec);


    mpc_abs(tmp, get_cmpfarray_i(coef, j), rnd);
    //mpfr_printf("|coef[j = %ld]| = %25.17RNe\n", j, tmp);
    if(mpf_cmp_ui(tmp, 0UL) == 0) return;

    // pi2 := 2 * PI
    mpfr_const_pi(pi2, rnd); //get_bnc_default_rounding_mode());
    mpf_mul_ui(pi2, pi2, 2UL);
    //mpfr_printf("pi2 = %RNe\n", pi2);

    //phi = dget_plus_arg(x) / pi2;
    mpf_get_plus_arg(phi, x); mpf_div(phi, phi, pi2);
    //psi = dget_plus_arg(-coef[i_num] / coef[j]) / pi2;
    mpc_div(ctmp,  get_cmpfarray_i(coef, i_num),  get_cmpfarray_i(coef, j), rndc);
    mpc_neg(ctmp, ctmp, rndc);
    mpf_get_plus_arg(psi, ctmp);
    mpf_div(psi, psi, pi2);

    //real_j = (double)dget_nearest_int((double)j * (0.5 - phi) - psi);
    mpf_set_ui(real_j, 1UL); mpf_div_ui(real_j, real_j, 2UL);
    mpf_sub(real_j, real_j, phi);
    mpf_mul_ui(real_j, real_j, (unsigned long)j);
    mpf_sub(real_j, real_j, psi);
    mpf_get_nearest_int(real_j, real_j);
    //mpfr_printf("real_j = %25.17RNe\n", real_j);

    // ret = pow(
    //     cabs(mu * coef[i_num] / coef[j]),
    //     1.0 / (double)j
    // ) * cexp(
    //     pi2 * I * ((psi + real_j) / (double)j)
    // );
    mpc_mul_fr(ctmp,  get_cmpfarray_i(coef, i_num), mu, rndc);
    mpc_div(ctmp, ctmp, get_cmpfarray_i(coef, j), rndc); // Fix! 2025-03-03 T.Kouya
    mpc_abs(tmp, ctmp, rnd);
    mpf_set_ui(tmp2, 1UL); mpf_div_ui(tmp2, tmp2, (unsigned long)j);
    mpfr_pow(pow_tmp, tmp, tmp2, rnd);
    //mpfr_printf("pow(%25.17RNe, %25.17RNe) = %25.17RNe\n", tmp, tmp2, pow_tmp);

    mpf_add(tmp, psi, real_j);
    mpf_div_ui(tmp, tmp, (unsigned long)j);
    mpf_mul(tmp, tmp, pi2);
    mpf_set_ui(tmp2, 0UL);
    mpc_set_fr_fr(ctmp, tmp2, tmp, rndc);
    mpc_exp(ret, ctmp, rndc);
    //mpfr_printf("cexp(%25.17RNe + %25.17RNe * I) = %25.17RNe + %25.17RNe * I\n", mpc_realref(ctmp), mpc_imagref(ctmp), mpc_realref(ret), mpc_imagref(ret));
    mpc_mul_fr(ret, ret, pow_tmp, rndc);

    mpf_clear(pi2);
    mpf_clear(psi);
    mpf_clear(phi);
    mpf_clear(real_j);
    mpf_clear(tmp);
    mpf_clear(tmp2);
    mpf_clear(pow_tmp);
    mpc_clear(ctmp);

    return;
}

// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int mpf_hirano(mpc_t ret, mpc_t init_x, MPFPoly poly, mpf_t reps, mpf_t aeps, long int maxtimes)
{
    unsigned long prec = mpc_get_prec(ret);
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();
    mpfr_rnd_t rnd = get_bnc_default_rounding_mode();
    int stop_flag = 0;
    long int i, j, max_j, times, absmin_j, deg = poly->deg;
    mpc_t old_x, new_x, pn_x_d, ctmp;
    //mpc_t *in_coef, *d;
    CMPFArray in_coef, d;
    mpf_t abs_pn_new_x, absmax_anxn, absmin_d, abs_d;
    mpf_t mu, beta, lambda;
    mpf_t tmp, tmp1, tmp2;

    // prec == 0
    if(prec <= 0)
    {
        prec = get_bnc_default_prec();\
    }

    mpc_init2(old_x, prec);
    mpc_init2(new_x, prec);
    mpc_init2(pn_x_d, prec);
    mpc_init2(ctmp, prec);

    mpf_init2(abs_pn_new_x, prec);
    mpf_init2(absmax_anxn, prec);
    mpf_init2(absmin_d, prec);
    mpf_init2(abs_d, prec);
    mpf_init2(mu, prec);
     // from Sugihara & Murota
    mpf_init2(beta, prec); mpf_set_ui(beta, 3UL); mpf_div_ui(beta, beta, 4UL);
    mpf_init2(lambda, prec); mpf_set_ui(lambda, 2UL);
    mpf_init2(tmp, prec);
    mpf_init2(tmp1, prec);
    mpf_init2(tmp2, prec);

    // coef of p(x + d)
    //in_coef = (double _Complex *)calloc(deg + 1, sizeof(double _Complex));
    //d = (double _Complex *)calloc(deg, sizeof(double _Complex));
    in_coef = init2_cmpfarray(deg + 1, prec);
    //printf("init d, prec=%ld, deg = %ld\n", prec, deg);
    d = init2_cmpfarray(deg + 1, prec);

    //old_x = init_x;
    mpc_set(old_x, init_x, rndc);

    // Main loop
    for(times = 0; times < maxtimes; times++)
    {
        mpf_coef_horner(in_coef, old_x, poly); // coef, deg);
        //mu = 1.0;
        mpf_set_ui(mu, 1UL);

        //mpfr_printf("%5ld %25.17RNe + %25.17RNe * I -> %15.7RNe + %15.7RNe * I\n", times, mpc_realref(old_x), mpc_imagref(old_x), mpc_realref(get_cmpfarray_i(in_coef, 0)), mpc_imagref(get_cmpfarray_i(in_coef, 0)));

        // (1)
        //d[0] = 0.0 + 0.0 * I;
        mpc_set_ui_ui(ctmp, 0UL, 0UL, rndc);
        set_cmpfarray_i(d, 0, ctmp);
        //if(fabs(in_coef[1]) != 0.0)
        mpc_abs(tmp, get_cmpfarray_i(in_coef, 1), rnd);
        if(mpf_cmp_ui(tmp, 0UL) != 0)
        {
            //d[0] = -in_coef[0] / in_coef[1];
            mpc_div(ctmp, get_cmpfarray_i(in_coef, 0), get_cmpfarray_i(in_coef, 1), rndc);
            mpc_neg(ctmp, ctmp, rndc);
            set_cmpfarray_i(d, 0, ctmp);
        }
        //mpfr_printf("d[0] = %25.17RNe + %25.17RNe * I\n", mpc_realref(get_cmpfarray_i(d, 0)), mpc_imagref(get_cmpfarray_i(d, 0)));

        //new_x = old_x + d[0];
        mpc_add(new_x, old_x, get_cmpfarray_i(d, 0), rndc);
        //pn_x_d = dhorner(new_x, coef, deg);
        //mpf_horner(pn_x_d, new_x, poly); // coef, deg);
        ceval_mpfpoly(pn_x_d, poly, new_x); // coef, deg);

        mpc_abs(tmp, pn_x_d, rnd);
        mpc_abs(tmp2, get_cmpfarray_i(in_coef, 0), rnd);
        mpf_mul(tmp2, tmp2, beta);
        //if(cabs(pn_x_d) <= beta * cabs(in_coef[0]))
        if(mpf_cmp(tmp, tmp2) <= 0)
        {
            // old_x = new_x;
            //printf("%ld: new_x := c0 + c1 * old_x\n", times);
            mpc_set(old_x, new_x, rndc);
        }
        else
        {
            //printf("%ld: new_x := c0 + c? * old_x^?\n", times);

            for(max_j = 2; max_j <= deg; max_j++)
            {
                //printf("max_j = %ld\n", max_j);
                //mpfr_printf("d[max_j] = %25.17RNe + %25.17RNe * I\n", mpc_realref(get_cmpfarray_i(d, max_j)), mpc_imagref(get_cmpfarray_i(d, max_j)));
                //mu /= lambda;
                mpf_div(mu, mu, lambda);
                //d[max_j] = dget_min_branch(old_x, mu, in_coef, 0, max_j);
                mpf_get_min_branch(get_cmpfarray_i(d, max_j), old_x, mu, in_coef, 0, max_j);
                //mpfr_printf("d[max_j = %ld] = %25.17RNe + %25.17RNe\n", max_j, mpc_realref(get_cmpfarray_i(d, max_j)), mpc_imagref(get_cmpfarray_i(d, max_j)));
                //absmin_d = cabs(d[max_j]);
                mpc_abs(absmin_d, get_cmpfarray_i(d, max_j), rnd);
                absmin_j = max_j;
                for(j = 0; j < max_j; j++)
                {
                    //printf("%ld ", j);
                    //d[j] = dget_min_branch(old_x, mu, in_coef, 0, j + 1) / pow(lambda, 1.0 / (double)(j + 1));
                    mpf_get_min_branch(get_cmpfarray_i(d, j), old_x, mu, in_coef, 0, j + 1);
                    mpf_set_ui(tmp, 1UL);
                    mpf_div_ui(tmp, tmp, (unsigned long)(j + 1));
                    mpfr_pow(tmp1, lambda, tmp, rnd);
                    //mpfr_printf("d[%ld] = %25.17RNe + %25.17RNe * I, lambda^%ld = %25.17RNe\n", j, mpc_realref(get_cmpfarray_i(d, j)), mpc_imagref(get_cmpfarray_i(d,j)), j + 1, tmp1);
                    mpc_div_fr(get_cmpfarray_i(d, j), get_cmpfarray_i(d, j), tmp1, rndc);
                    //abs_d = cabs(d[j]);
                    mpc_abs(abs_d, get_cmpfarray_i(d, j), rnd);
                    if(mpf_cmp(abs_d, absmin_d) < 0) //abs_d < absmin_d)
                    {
                        // absmin_d = abs_d;
                        mpf_set(absmin_d, abs_d);
                        absmin_j = j;
                    }
                }
                //printf("absmin_j = %ld\n", absmin_j);
                //new_x = old_x + d[absmin_j];
                mpc_add(new_x, old_x, get_cmpfarray_i(d, absmin_j), rndc);
                //mpfr_printf("d[%ld] = %25.17RNe + %25.17RNe * I\n", absmin_j, mpc_realref(get_cmpfarray_i(d, absmin_j)), mpc_imagref(get_cmpfarray_i(d, absmin_j)));
                //mpfr_printf("old_x =  %25.17RNe + %25.17RNe * I\n", mpc_realref(old_x), mpc_imagref(old_x));
                //mpfr_printf("new_x =  %25.17RNe + %25.17RNe * I\n", mpc_realref(new_x), mpc_imagref(new_x));
                //mpfr_printf("pn_x_d = %25.17RNe + %25.17RNe * I\n", mpc_realref(new_x), mpc_imagref(new_x));
                //pn_x_d = dhorner(new_x, coef, deg);
                //mpf_horner(pn_x_d, new_x, poly); // coef, deg);
                ceval_mpfpoly(pn_x_d, poly, new_x); // coef, deg);

                mpc_abs(tmp, pn_x_d, rnd);// pn_x_d := abs(p(new_x))
                mpf_set_ui(tmp2, 1UL); // tmp2 := 1
                mpf_sub(tmp2, tmp2, beta); // tmp2 := 1 - beta
                mpf_mul(tmp2, tmp2, mu); // tmp2 := (1 - beta) * mu
                mpf_sub_ui(tmp2, tmp2, 1UL); // tmp2 := tmp2 - 1 = (1 - beta) * mu - 1
                mpf_neg(tmp2, tmp2); // tmp2 := -tmp2 = 1 - (1 - beta) * mu
                mpc_abs(tmp1, get_cmpfarray_i(in_coef, 0), rnd); // tmp1 := abs(in_coef[0])
                mpf_mul(tmp2, tmp2, tmp1); // tmp2 := tmp2 * tmp1
                //mpfr_printf("tmp = %25.17RNe <= tmp2 = %25.17RNe ? \n", tmp, tmp2);
                //if(cabs(pn_x_d) <= (1.0 - (1.0 - beta) * mu) * cabs(in_coef[0]))
                if(mpf_cmp(tmp, tmp2) <= 0)
                {
                    //old_x = new_x;
                    //printf("%ld - %ld: new_x := old_x + c%ld * x^%ld\n", times, max_j, absmin_j, absmin_j);
                    mpc_set(old_x, new_x, rndc);
                    break;
                }        
            }
        }

        // check stopping rule
        //dabsmax_poly(&absmax_anxn, new_x, coef, deg);
        absmax_mpfpoly(absmax_anxn, new_x, poly); // coef, deg);
        //abs_pn_new_x = cabs(dhorner(new_x, coef, deg));
        //mpf_horner(ctmp, new_x, poly); // coef, deg);
        ceval_mpfpoly(ctmp, poly, new_x); // coef, deg);
        mpc_abs(abs_pn_new_x, ctmp, rnd);
        mpf_mul(tmp, absmax_anxn, reps);
        mpf_add(tmp, tmp, aeps);
        //if(abs_pn_new_x <= absmax_anxn * reps + aeps)
        if(mpf_cmp(abs_pn_new_x, tmp) <= 0)
            break;
    }

    free_cmpfarray(in_coef);
    free_cmpfarray(d);

    //*ret = new_x;
    mpc_set(ret, new_x, rndc);

    mpc_clear(old_x);
    mpc_clear(new_x);
    mpc_clear(pn_x_d);
    mpc_clear(ctmp);

    mpf_clear(abs_pn_new_x);
    mpf_clear(absmax_anxn);
    mpf_clear(absmin_d);
    mpf_clear(abs_d);
    mpf_clear(mu);
    mpf_clear(beta);
    mpf_clear(lambda);
    mpf_clear(tmp);
    mpf_clear(tmp1);
    mpf_clear(tmp2);

    return times;
}

// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int mpc_hirano(mpc_t ret, mpc_t init_x, CMPFPoly poly, mpf_t reps, mpf_t aeps, long int maxtimes)
{
    unsigned long prec = mpc_get_prec(ret);
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();
    mpfr_rnd_t rnd = get_bnc_default_rounding_mode();
    int stop_flag = 0;
    long int i, j, max_j, times, absmin_j, deg = poly->deg;
    mpc_t old_x, new_x, pn_x_d, ctmp;
    //mpc_t *in_coef, *d;
    CMPFArray in_coef, d;
    mpf_t abs_pn_new_x, absmax_anxn, absmin_d, abs_d;
    mpf_t mu, beta, lambda;
    mpf_t tmp, tmp1, tmp2;

    // prec == 0
    if(prec <= 0)
    {
        prec = get_bnc_default_prec();\
    }

    mpc_init2(old_x, prec);
    mpc_init2(new_x, prec);
    mpc_init2(pn_x_d, prec);
    mpc_init2(ctmp, prec);

    mpf_init2(abs_pn_new_x, prec);
    mpf_init2(absmax_anxn, prec);
    mpf_init2(absmin_d, prec);
    mpf_init2(abs_d, prec);
    mpf_init2(mu, prec);
     // from Sugihara & Murota
    mpf_init2(beta, prec); mpf_set_ui(beta, 3UL); mpf_div_ui(beta, beta, 4UL);
    mpf_init2(lambda, prec); mpf_set_ui(lambda, 2UL);
    mpf_init2(tmp, prec);
    mpf_init2(tmp1, prec);
    mpf_init2(tmp2, prec);

    // coef of p(x + d)
    //in_coef = (double _Complex *)calloc(deg + 1, sizeof(double _Complex));
    //d = (double _Complex *)calloc(deg, sizeof(double _Complex));
    in_coef = init2_cmpfarray(deg + 1, prec);
    //printf("init d, prec=%ld, deg = %ld\n", prec, deg);
    d = init2_cmpfarray(deg + 1, prec);

    //old_x = init_x;
    mpc_set(old_x, init_x, rndc);

    // Main loop
    for(times = 0; times < maxtimes; times++)
    {
        mpc_coef_horner(in_coef, old_x, poly); // coef, deg);
        //mu = 1.0;
        mpf_set_ui(mu, 1UL);

        //mpfr_printf("%5ld %25.17RNe + %25.17RNe * I -> %15.7RNe + %15.7RNe * I\n", times, mpc_realref(old_x), mpc_imagref(old_x), mpc_realref(get_cmpfarray_i(in_coef, 0)), mpc_imagref(get_cmpfarray_i(in_coef, 0)));

        // (1)
        //d[0] = 0.0 + 0.0 * I;
        mpc_set_ui_ui(ctmp, 0UL, 0UL, rndc);
        set_cmpfarray_i(d, 0, ctmp);
        //if(fabs(in_coef[1]) != 0.0)
        mpc_abs(tmp, get_cmpfarray_i(in_coef, 1), rnd);
        if(mpf_cmp_ui(tmp, 0UL) != 0)
        {
            //d[0] = -in_coef[0] / in_coef[1];
            mpc_div(ctmp, get_cmpfarray_i(in_coef, 0), get_cmpfarray_i(in_coef, 1), rndc);
            mpc_neg(ctmp, ctmp, rndc);
            set_cmpfarray_i(d, 0, ctmp);
        }
        //mpfr_printf("d[0] = %25.17RNe + %25.17RNe * I\n", mpc_realref(get_cmpfarray_i(d, 0)), mpc_imagref(get_cmpfarray_i(d, 0)));

        //new_x = old_x + d[0];
        mpc_add(new_x, old_x, get_cmpfarray_i(d, 0), rndc);
        //pn_x_d = dhorner(new_x, coef, deg);
        //mpc_horner(pn_x_d, new_x, poly); // coef, deg);
        eval_cmpfpoly(pn_x_d, poly, new_x);

        mpc_abs(tmp, pn_x_d, rnd);
        mpc_abs(tmp2, get_cmpfarray_i(in_coef, 0), rnd);
        mpf_mul(tmp2, tmp2, beta);
        //if(cabs(pn_x_d) <= beta * cabs(in_coef[0]))
        if(mpf_cmp(tmp, tmp2) <= 0)
        {
            // old_x = new_x;
            //printf("%ld: new_x := c0 + c1 * old_x\n", times);
            mpc_set(old_x, new_x, rndc);
        }
        else
        {
            //printf("%ld: new_x := c0 + c? * old_x^?\n", times);

            for(max_j = 2; max_j <= deg; max_j++)
            {
                //printf("max_j = %ld\n", max_j);
                //mpfr_printf("d[max_j] = %25.17RNe + %25.17RNe * I\n", mpc_realref(get_cmpfarray_i(d, max_j)), mpc_imagref(get_cmpfarray_i(d, max_j)));
                //mu /= lambda;
                mpf_div(mu, mu, lambda);
                //d[max_j] = dget_min_branch(old_x, mu, in_coef, 0, max_j);
                mpf_get_min_branch(get_cmpfarray_i(d, max_j), old_x, mu, in_coef, 0, max_j);
                //mpfr_printf("d[max_j = %ld] = %25.17RNe + %25.17RNe\n", max_j, mpc_realref(get_cmpfarray_i(d, max_j)), mpc_imagref(get_cmpfarray_i(d, max_j)));
                //absmin_d = cabs(d[max_j]);
                mpc_abs(absmin_d, get_cmpfarray_i(d, max_j), rnd);
                absmin_j = max_j;
                for(j = 0; j < max_j; j++)
                {
                    //printf("%ld ", j);
                    //d[j] = dget_min_branch(old_x, mu, in_coef, 0, j + 1) / pow(lambda, 1.0 / (double)(j + 1));
                    mpf_get_min_branch(get_cmpfarray_i(d, j), old_x, mu, in_coef, 0, j + 1);
                    mpf_set_ui(tmp, 1UL);
                    mpf_div_ui(tmp, tmp, (unsigned long)(j + 1));
                    mpfr_pow(tmp1, lambda, tmp, rnd);
                    //mpfr_printf("d[%ld] = %25.17RNe + %25.17RNe * I, lambda^%ld = %25.17RNe\n", j, mpc_realref(get_cmpfarray_i(d, j)), mpc_imagref(get_cmpfarray_i(d,j)), j + 1, tmp1);
                    mpc_div_fr(get_cmpfarray_i(d, j), get_cmpfarray_i(d, j), tmp1, rndc);
                    //abs_d = cabs(d[j]);
                    mpc_abs(abs_d, get_cmpfarray_i(d, j), rnd);
                    if(mpf_cmp(abs_d, absmin_d) < 0) //abs_d < absmin_d)
                    {
                        // absmin_d = abs_d;
                        mpf_set(absmin_d, abs_d);
                        absmin_j = j;
                    }
                }
                //printf("absmin_j = %ld\n", absmin_j);
                //new_x = old_x + d[absmin_j];
                mpc_add(new_x, old_x, get_cmpfarray_i(d, absmin_j), rndc);
                //mpfr_printf("d[%ld] = %25.17RNe + %25.17RNe * I\n", absmin_j, mpc_realref(get_cmpfarray_i(d, absmin_j)), mpc_imagref(get_cmpfarray_i(d, absmin_j)));
                //mpfr_printf("old_x =  %25.17RNe + %25.17RNe * I\n", mpc_realref(old_x), mpc_imagref(old_x));
                //mpfr_printf("new_x =  %25.17RNe + %25.17RNe * I\n", mpc_realref(new_x), mpc_imagref(new_x));
                //mpfr_printf("pn_x_d = %25.17RNe + %25.17RNe * I\n", mpc_realref(new_x), mpc_imagref(new_x));
                //pn_x_d = dhorner(new_x, coef, deg);
                //mpc_horner(pn_x_d, new_x, poly); // coef, deg);
                eval_cmpfpoly(pn_x_d, poly, new_x);

                mpc_abs(tmp, pn_x_d, rnd);// pn_x_d := abs(p(new_x))
                mpf_set_ui(tmp2, 1UL); // tmp2 := 1
                mpf_sub(tmp2, tmp2, beta); // tmp2 := 1 - beta
                mpf_mul(tmp2, tmp2, mu); // tmp2 := (1 - beta) * mu
                mpf_sub_ui(tmp2, tmp2, 1UL); // tmp2 := tmp2 - 1 = (1 - beta) * mu - 1
                mpf_neg(tmp2, tmp2); // tmp2 := -tmp2 = 1 - (1 - beta) * mu
                mpc_abs(tmp1, get_cmpfarray_i(in_coef, 0), rnd); // tmp1 := abs(in_coef[0])
                mpf_mul(tmp2, tmp2, tmp1); // tmp2 := tmp2 * tmp1
                //mpfr_printf("tmp = %25.17RNe <= tmp2 = %25.17RNe ? \n", tmp, tmp2);
                //if(cabs(pn_x_d) <= (1.0 - (1.0 - beta) * mu) * cabs(in_coef[0]))
                if(mpf_cmp(tmp, tmp2) <= 0)
                {
                    //old_x = new_x;
                    //printf("%ld - %ld: new_x := old_x + c%ld * x^%ld\n", times, max_j, absmin_j, absmin_j);
                    mpc_set(old_x, new_x, rndc);
                    break;
                }        
            }
        }

        // check stopping rule
        //dabsmax_poly(&absmax_anxn, new_x, coef, deg);
        absmax_cmpfpoly(absmax_anxn, new_x, poly); // coef, deg);
        //abs_pn_new_x = cabs(dhorner(new_x, coef, deg));
        //mpc_horner(ctmp, new_x, poly); // coef, deg);
        eval_cmpfpoly(ctmp, poly, new_x); // coef, deg);
        mpc_abs(abs_pn_new_x, ctmp, rnd);
        mpf_mul(tmp, absmax_anxn, reps);
        mpf_add(tmp, tmp, aeps);
        //if(abs_pn_new_x <= absmax_anxn * reps + aeps)
        if(mpf_cmp(abs_pn_new_x, tmp) <= 0)
            break;
    }

    free_cmpfarray(in_coef);
    free_cmpfarray(d);

    //*ret = new_x;
    mpc_set(ret, new_x, rndc);

    mpc_clear(old_x);
    mpc_clear(new_x);
    mpc_clear(pn_x_d);
    mpc_clear(ctmp);

    mpf_clear(abs_pn_new_x);
    mpf_clear(absmax_anxn);
    mpf_clear(absmin_d);
    mpf_clear(abs_d);
    mpf_clear(mu);
    mpf_clear(beta);
    mpf_clear(lambda);
    mpf_clear(tmp);
    mpf_clear(tmp1);
    mpf_clear(tmp2);

    return times;
}

// Ono's Problem
void ono_poly(MPFPoly poly, long int deg)
{
    unsigned long prec = poly->prec;
	long int i, j;
	mpf_t sum, tmp;

	//mpf_init2(sum, mpf_get_prec(coef[0]));
	//mpf_init2(tmp, mpf_get_prec(coef[0]));

    mpf_init2(sum, prec);
    mpf_init2(tmp, prec);

	mpf_set_ui(tmp, 1UL);
    set_mpfpoly_i(poly, deg, tmp);
    //poly->coef[deg] = TKMAC_CONST_MP_ONE;
	for(i = 1; i <= deg / 2; i++)
	{
		//poly->coef[deg - (i * 2 - 1)] = TKMAC_CONST_MP_ZERO;
        mpf_set_ui(tmp, 0UL);
        set_mpfpoly_i(poly, deg - (i * 2 - 1), tmp);
		//sum = TKMAC_CONST_MP_ZERO;
        mpf_set_ui(sum, 0UL);
		for(j = 1; j <= i; j++)
        {
            mpf_set_ui(tmp, (unsigned long)(2 * j + 1));
            mpf_ui_div(tmp, 1UL, tmp);
            //tmp *= poly->coef[deg - 2 * (i - j)];
            mpf_mul(tmp, tmp, get_mpfpoly_i(poly, deg - 2 * (i - j)));
			//sum += (1.0 / (2 * j + 1)) * coef[deg - 2 * (i - j)];
            //sum += tmp;
            mpf_add(sum, sum, tmp);
        }
		//sum *= -deg / (2 * i);
        mpf_set_ui(tmp, (unsigned long)(2 * i));
        mpf_ui_div(tmp, deg, tmp);
        //tmp = -tmp;
        mpf_neg(tmp, tmp);
        //sum *= tmp;
        mpf_mul(sum, sum, tmp);
		//poly->coef[deg - i * 2] = sum;
        set_mpfpoly_i(poly, deg - i * 2, sum);

        setdegree_mpfpoly(poly);
    }

    //setdegree_mpfpoly(poly);

    mpf_clear(sum);
    mpf_clear(tmp);
}

// Wilkinson's example: (x - 1)(x - 2) ... (x - n) = 0
void wilkinson_poly(MPFPoly ret, long int n)
{
    unsigned long prec = ret->prec;
    long int i;
    MPFPoly tmp, xmn; // x - n
    mpf_t mp_n, one, minus_one;

    mpf_init2(mp_n, prec);
    mpf_init2(one, prec);
    mpf_init2(minus_one, prec);

    xmn = init2_mpfpoly(2, prec);
    tmp = init2_mpfpoly(ret->max_len, prec);

    set0_mpfpoly(xmn);
    set0_mpfpoly(ret);
    set0_mpfpoly(tmp);

    mpf_set_ui(one, 1UL);
    mpf_neg(minus_one, one);

    // ret := x - 1
    set_mpfpoly_i(ret, 0, minus_one); // -one);
    set_mpfpoly_i(ret, 1, one);

    for(i = 2; i <= n; i++)
    {
        // xmn := x - n
        mpf_set_ui(mp_n, (unsigned long)i);
        mpf_neg(mp_n, mp_n);
        set_mpfpoly_i(xmn, 0, mp_n);
        set_mpfpoly_i(xmn, 1, one);

        // ret *= (x - n)
        mul_mpfpoly(tmp, ret, xmn);
        subst_mpfpoly(ret, tmp);
    }

    free_mpfpoly(xmn);
    free_mpfpoly(tmp);

    mpf_clear(mp_n);
    mpf_clear(one);
    mpf_clear(minus_one);
}

// Deflation of polynomial
// p(x) / (x - r)
void deflation_cmpfpoly(CMPFPoly ret, CMPFPoly pol, mpc_t root)
{
    long int i;
    //double _Complex ret;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();
    mpc_t ctmp;
    CMPFPoly in_ret;
    
    if(pol->deg < 1)
        return;

    mpc_init2(ctmp, ret->prec);
    in_ret = init_set_cmpfpoly(ret);

    // clear
    set0_cmpfpoly(in_ret);

    //ret[deg - 1] := coef[deg];
    set_cmpfpoly_i(in_ret, pol->deg - 1, get_cmpfpoly_i(pol, pol->deg));
    for(i = pol->deg - 2; i >= 0; i--) // i >= 0; i--)
    {
        mpc_mul(ctmp, get_cmpfpoly_i(in_ret, i + 1), root, rndc);
        mpc_add(ctmp, ctmp, get_cmpfpoly_i(pol, i + 1), rndc);
        //ret[i] := ret[i + 1] * root + coef[i + 1];
        set_cmpfpoly_i(in_ret, i, ctmp);
    }

    setdegree_cmpfpoly(in_ret);
    subst_cmpfpoly(ret, in_ret);

    mpc_clear(ctmp);
    free_cmpfpoly(in_ret);
    //return ret;
    return;
}
#endif // USE_GMP