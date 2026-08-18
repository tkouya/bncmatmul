/********************************************************************************/
/* td_hirano.c: Robust solver for lgebraic Equations using Hirano method        */
/* copyright (c) 2025 Tomonori Kouya                                            */
/*                                                                              */
/* Ver. 0.1 2025-06-26: First implementation                                    */
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
#include "tdlinear.h"
#include "poly.h"

// --------------------------
// TD
// --------------------------
//#ifdef USE_GMP
#ifdef USE_TDLINEAR

// Horner method
// return p(x)
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
void td_horner(ctdfloat *ret, ctdfloat *x, TDPoly poly) // mpf_t coef[], long int deg)
{
    long int i;
    //ctdfloat ret;

    //ret = coef[deg] + 0.0 * I;
    rctd_set_ui(ret, 0UL);
    rctd_set_td(ret, get_tdpoly_i(poly, poly->deg)); // , 0UL);
    for(i = poly->deg - 1; i >= 0; i--)
    {
        //ret = ret * x + (coef[i] + 0.0 * I);
        rctd_mul(ret, ret, x);
        rctd_add_td(ret, ret, get_tdpoly_i(poly, i));
    }

    //return ret;
    return;
}

// Coef of p(x + d)
// 
// based on Horner method
void td_coef_horner(CTDVector ret_coef, ctdfloat *x, TDPoly poly) // mpf_t coef[], long int deg)
{
    long int l, i;
    ctdfloat ctmp, ctmp2;

    // Initial setting
    for(i = 0; i <= poly->deg; i++)
    {
        //ret_coef[i] = coef[i] + 0.0 * I;
        rctd_set_td(&ctmp2, get_tdpoly_i(poly, i));
        set_ctdvector_i(ret_coef, i, &ctmp2);
    }

    // ret_coef[i] := p^(l)(x) / l!
    for(l = 0; l <= poly->deg; l++)
    {
        for(i = poly->deg - 1; i >= l; i--)
        {
            //ret_coef[i] = x * ret_coef[i + 1] + ret_coef[i];
            rctd_mul(&ctmp, get_ctdvector_i(ret_coef, i + 1), x);
            rctd_add(&ctmp2, &ctmp, get_ctdvector_i(ret_coef, i));
            set_ctdvector_i(ret_coef, i, &ctmp2);
        }
    }
}

// Horner method
// return p(x)
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
void ctd_horner(ctdfloat *ret, ctdfloat *x, CTDPoly poly) // mpf_t coef[], long int deg)
{
    long int i;
    //double _Complex ret;

    //ret = coef[deg] + 0.0 * I;
    rctd_set_ui(ret, 0UL);
    rctd_set(ret, get_ctdpoly_i(poly, poly->deg)); // , 0UL);
    for(i = poly->deg - 1; i >= 0; i--)
    {
        //ret = ret * x + (coef[i] + 0.0 * I);
        rctd_mul(ret, ret, x);
        rctd_add(ret, ret, get_ctdpoly_i(poly, i));
    }

    //return ret;
    return;
}

void ctd_coef_horner(CTDVector ret_coef, ctdfloat *x, CTDPoly poly) // mpf_t coef[], long int deg)
{
    long int l, i;
    ctdfloat ctmp, ctmp2;

    // Initial setting
    for(i = 0; i <= poly->deg; i++)
    {
        //ret_coef[i] = coef[i] + 0.0 * I;
        rctd_set(&ctmp2, get_ctdpoly_i(poly, i));
        set_ctdvector_i(ret_coef, i, &ctmp2);
    }

    // ret_coef[i] := p^(l)(x) / l!
    for(l = 0; l <= poly->deg; l++)
    {
        for(i = poly->deg - 1; i >= l; i--)
        {
            //ret_coef[i] = x * ret_coef[i + 1] + ret_coef[i];
            rctd_mul(&ctmp, get_ctdvector_i(ret_coef, i + 1), x);
            rctd_add(&ctmp2, &ctmp, get_ctdvector_i(ret_coef, i));
            set_ctdvector_i(ret_coef, i, &ctmp2);
        }
    }
}

// return j and max|a_j x^j| from p(x)
long int absmax_tdpoly(double absmax_anxn[TDSIZE], ctdfloat *x, TDPoly poly) // mpf_t coef[], long int deg)
{
    long int i, ret_i;
    double abs_anxn[TDSIZE];
    ctdfloat xn, anxn;

    // xn := x
    rctd_set(&xn, x);

    ret_i = 0;
    //*absmax_anxn = fabs(coef[0]);
    rtd_abs(absmax_anxn, get_tdpoly_i(poly, 0));
 
    for(i = 1; i <= poly->deg; i++)
    {
        //anxn = coef[i] * xn;
        rctd_mul_td(&anxn, &xn, get_tdpoly_i(poly, i));
        //abs_anxn = fabs(anxn);
        rctd_abs_td(abs_anxn, &anxn);
        //if(*absmax_anxn < abs_anxn)
        if(rtd_cmp(absmax_anxn, abs_anxn) < 0)
        {
            ret_i = i;
            //*absmax_anxn = abs_anxn;
            rtd_set(absmax_anxn, abs_anxn);
        }
        //xn *= x;
        rctd_mul(&xn, &xn, x);
    }

    return ret_i;
}

long int absmax_ctdpoly(double absmax_anxn[TDSIZE], ctdfloat *x, CTDPoly poly) // mpf_t coef[], long int deg)
{
    long int i, ret_i;
    double abs_anxn[TDSIZE];
    ctdfloat xn, anxn;

    // xn := x
    rctd_set(&xn, x);

    ret_i = 0;
    //*absmax_anxn = fabs(coef[0]);
    rctd_abs_td(absmax_anxn, get_ctdpoly_i(poly, 0));
 
    for(i = 1; i <= poly->deg; i++)
    {
        //anxn = coef[i] * xn;
        rctd_mul(&anxn, &xn, get_ctdpoly_i(poly, i));
        //abs_anxn = fabs(anxn);
        rctd_abs_td(abs_anxn, &anxn);
        //if(*absmax_anxn < abs_anxn)
        if(rtd_cmp(absmax_anxn, abs_anxn) < 0)
        {
            ret_i = i;
            //*absmax_anxn = abs_anxn;
            rtd_set(absmax_anxn, abs_anxn);
        }
        //xn *= x;
        rctd_mul(&xn, &xn, x);
    }

    return ret_i;
}


// get_plus_arg
// return arg(x) in [0, 2 PI]
void td_get_plus_arg(double ret[TDSIZE], ctdfloat *x)
{
    double pi2[TDSIZE];

    // pi2 := 2 * PI
    rtd_const_pi(pi2);
    rtd_mul_ui(pi2, pi2, 2UL);

    //double ret = carg(x);
    rctd_arg(ret, x);

    //if(ret < 0)
    if(rtd_cmp_ui(ret, 0UL) < 0)
    {
        //ret = 2.0 * M_PI + ret;
        rtd_add(ret, pi2, ret);
    }

    //return ret;
    return;
}

// get_nearest_int
void td_get_nearest_int(double ret[TDSIZE], double real_x[TDSIZE])
{
     //long int int_x_floor, int_x_ceil;
    //double dist_x_floor, dist_x_ceil;
    double int_x_floor[TDSIZE], int_x_ceil[TDSIZE];
    double dist_x_floor[TDSIZE], dist_x_ceil[TDSIZE];

    //int_x_floor = (long int)floor(real_x);
    rtd_func_mpfr(int_x_floor, mpfr_rint_floor, real_x); // _dd);
    //int_x_ceil  = (long int)ceil (real_x);
    rtd_func_mpfr(int_x_ceil, mpfr_rint_ceil, real_x); // _dd);

    //dist_x_floor = fabs(real_x - (double)int_x_floor);
    rtd_sub(dist_x_floor, real_x, int_x_floor); rtd_abs(dist_x_floor, dist_x_floor);
    //dist_x_ceil  = fabs(real_x - (double)int_x_ceil);
    rtd_sub(dist_x_ceil, real_x, int_x_ceil); rtd_abs(dist_x_ceil, dist_x_ceil);

    //if(dist_x_floor < dist_x_ceil) return int_x_floor;
    if(rtd_cmp(dist_x_floor, dist_x_ceil) < 0) rtd_set(ret, int_x_floor);
    //else return int_x_ceil;
    else rtd_set(ret, int_x_ceil);
}

// get_min_branch
void td_get_min_branch(ctdfloat *ret, ctdfloat *x, double mu[TDSIZE], CTDVector coef, long int i_num, long int i_den)
{
    long int j = i_den;
    double phi[TDSIZE], psi[TDSIZE], pi2[TDSIZE], real_j[TDSIZE];
    ctdfloat ctmp, ctmp2;
    double tmp[TDSIZE], tmp1[TDSIZE], tmp2[TDSIZE];

    //if(cabs(coef[j]) == 0.0) return ret;
    rctd_abs_td(tmp, get_ctdvector_i(coef, j));
    if(rtd_cmp_ui(tmp, 0UL) == 0)
    {
        rctd_set_ui_ui(ret, 0UL, 0UL);
        return;
    }

    rtd_const_pi(pi2);
    rtd_mul_ui(pi2, pi2, 2UL);

    //phi = dget_plus_arg(x) / pi2;
    td_get_plus_arg(phi, x);
    rtd_div(phi, phi, pi2);

    //psi = dget_plus_arg(-coef[i_num] / coef[j]) / pi2;
    rctd_div(&ctmp, get_ctdvector_i(coef, i_num), get_ctdvector_i(coef, j));
    rctd_neg(&ctmp, &ctmp);
    td_get_plus_arg(psi, &ctmp);
    rtd_div(psi, psi, pi2);

    //real_j = (double)dget_nearest_int((double)j * (0.5 - phi) - psi);
    rtd_set_ui(tmp, 1UL);
    rtd_div_ui(tmp, tmp, 2UL); // tmp := 0.5
    rtd_sub(tmp, tmp, phi); // tmp := 0.5 - phi
    rtd_mul_ui(tmp, tmp, (unsigned long int)j); // tmp := j * (0.5 - phi)
    rtd_sub(tmp, tmp, psi); // tmp := tmp - psi
    td_get_nearest_int(real_j, tmp);

    //ret = pow(cabs(mu * coef[i_num] / coef[j]), 1.0 / (double)j) * cexp(pi2 * I * ((psi + real_j) / (double)j));
    
    // Part 1: pow(cabs(mu * coef[i_num] / coef[j]), 1.0 / (double)j)
    rctd_div(&ctmp, get_ctdvector_i(coef, i_num), get_ctdvector_i(coef, j));
    rctd_mul_td(&ctmp, &ctmp, mu);
    rctd_abs_td(tmp, &ctmp);
    rtd_set_ui(tmp1, 1UL);
    rtd_div_ui(tmp1, tmp1, (unsigned long int)j);
    rtd_pow_mpfr(tmp, tmp, tmp1);

    // Part 2: cexp(pi2 * I * ((psi + real_j) / (double)j))
    rtd_add(tmp1, psi, real_j);
    rtd_div_ui(tmp1, tmp1, (unsigned long int)j);
    rtd_mul(tmp1, pi2, tmp1);
    rtd_set_ui(tmp2, 0UL);
    rctd_set_td_td(&ctmp, tmp2, tmp1);
    rctd_func_mpc(&ctmp2, mpc_exp, &ctmp);
    //rctd_exp(&ctmp2, &ctmp2);

    // Multiply: ret = tmp * ctmp2
    rctd_mul_td(ret, &ctmp2, tmp);

    return;
}

// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int td_hirano(ctdfloat *ret, ctdfloat *init_x, TDPoly poly, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
{
    int stop_flag = 0;
    long int i, j, max_j, times, absmin_j, deg = poly->deg;
    ctdfloat old_x, new_x, pn_x_d, ctmp, ctmp2;
    //ctdfloat *in_coef, *d;
    CTDVector in_coef, d;
    double abs_pn_new_x[TDSIZE], absmax_anxn[TDSIZE], absmin_d[TDSIZE], abs_d[TDSIZE];
    double mu[TDSIZE], beta[TDSIZE], lambda[TDSIZE];
    double tmp[TDSIZE], tmp1[TDSIZE], tmp2[TDSIZE];

    // from Sugihara & Murota
    rtd_set_ui(beta, 3UL); rtd_div_ui(beta, beta, 4UL);
    rtd_set_ui(lambda, 2UL);

    // coef of p(x + d)
    //in_coef = (double _Complex *)calloc(deg + 1, sizeof(double _Complex));
    //d = (double _Complex *)calloc(deg, sizeof(double _Complex));
    in_coef = init_ctdvector(deg + 1);
    //printf("init d=%ld, deg = %ld\n", deg);
    d = init_ctdvector(deg + 1);

    //old_x = init_x;
    rctd_set(&old_x, init_x);

    // Main loop
    for(times = 0; times < maxtimes; times++)
    {
        td_coef_horner(in_coef, &old_x, poly); // coef, deg);
        //mu = 1.0;
        rtd_set_ui(mu, 1UL);

        //mpfr_printf("%5ld %25.17RNe + %25.17RNe * I -> %15.7RNe + %15.7RNe * I\n", times, rctd_realref(old_x), rctd_imagref(old_x), rctd_realref(get_ctdvector_i(in_coef, 0)), rctd_imagref(get_ctdvector_i(in_coef, 0)));

        // (1)
        //d[0] = 0.0 + 0.0 * I;
        rctd_set_ui_ui(&ctmp, 0UL, 0UL);
        set_ctdvector_i(d, 0, &ctmp);
        //if(fabs(in_coef[1]) != 0.0)
        rctd_abs_td(tmp, get_ctdvector_i(in_coef, 1));
        if(rtd_cmp_ui(tmp, 0UL) != 0)
        {
            //d[0] = -in_coef[0] / in_coef[1];
            rctd_div(&ctmp, get_ctdvector_i(in_coef, 0), get_ctdvector_i(in_coef, 1));
            rctd_neg(&ctmp, &ctmp);
            set_ctdvector_i(d, 0, &ctmp);
        }
        //mpfr_printf("d[0] = %25.17RNe + %25.17RNe * I\n", rctd_realref(get_ctdvector_i(d, 0)), rctd_imagref(get_ctdvector_i(d, 0)));

        //new_x = old_x + d[0];
        rctd_add(&new_x, &old_x, get_ctdvector_i(d, 0));
        //pn_x_d = dhorner(new_x, coef, deg);
        //rctd_horner(pn_x_d, new_x, poly); // coef, deg);
        ceval_tdpoly(&pn_x_d, poly, &new_x);

        rctd_abs_td(tmp, &pn_x_d);
        rctd_abs_td(tmp2, get_ctdvector_i(in_coef, 0));
        rtd_mul(tmp2, tmp2, beta);
        //if(cabs(pn_x_d) <= beta * cabs(in_coef[0]))
        if(rtd_cmp(tmp, tmp2) <= 0)
        {
            // old_x = new_x;
            //printf("%ld: new_x := c0 + c1 * old_x\n", times);
            rctd_set(&old_x, &new_x);
        }
        else
        {
            //printf("%ld: new_x := c0 + c? * old_x^?\n", times);

            for(max_j = 2; max_j <= deg; max_j++)
            {
                //printf("max_j = %ld\n", max_j);
                //mpfr_printf("d[max_j] = %25.17RNe + %25.17RNe * I\n", rctd_realref(get_ctdvector_i(d, max_j)), rctd_imagref(get_ctdvector_i(d, max_j)));
                //mu /= lambda;
                rtd_div(mu, mu, lambda);
                //d[max_j] = dget_min_branch(old_x, mu, in_coef, 0, max_j);
                td_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, max_j);
                set_ctdvector_i(d, max_j, &ctmp2);
                //mpfr_printf("d[max_j = %ld] = %25.17RNe + %25.17RNe\n", max_j, rctd_realref(get_ctdvector_i(d, max_j)), rctd_imagref(get_ctdvector_i(d, max_j)));
                //absmin_d = cabs(d[max_j]);
                rctd_abs_td(absmin_d, get_ctdvector_i(d, max_j));
                absmin_j = max_j;
                for(j = 0; j < max_j; j++)
                {
                    //printf("%ld ", j);
                    //d[j] = dget_min_branch(old_x, mu, in_coef, 0, j + 1) / pow(lambda, 1.0 / (double)(j + 1));
                    td_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, j + 1);
                    set_ctdvector_i(d, j, &ctmp2);
                    rtd_set_ui(tmp, 1UL);
                    rtd_div_ui(tmp, tmp, (unsigned long)(j + 1));
                    rtd_pow_mpfr(tmp1, lambda, tmp);
                    //mpfr_printf("d[%ld] = %25.17RNe + %25.17RNe * I, lambda^%ld = %25.17RNe\n", j, rctd_realref(get_ctdvector_i(d, j)), rctd_imagref(get_ctdvector_i(d,j)), j + 1, tmp1);
                    rctd_div_td(&ctmp2, get_ctdvector_i(d, j), tmp1);
                    set_ctdvector_i(d, j, &ctmp2);
                    //abs_d = cabs(d[j]);
                    rctd_abs_td(abs_d, get_ctdvector_i(d, j));
                    if(rtd_cmp(abs_d, absmin_d) < 0) //abs_d < absmin_d)
                    {
                        // absmin_d = abs_d;
                        rtd_set(absmin_d, abs_d);
                        absmin_j = j;
                    }
                }
                //printf("absmin_j = %ld\n", absmin_j);
                //new_x = old_x + d[absmin_j];
                rctd_add(&new_x, &old_x, get_ctdvector_i(d, absmin_j));
                //mpfr_printf("d[%ld] = %25.17RNe + %25.17RNe * I\n", absmin_j, rctd_realref(get_ctdvector_i(d, absmin_j)), rctd_imagref(get_ctdvector_i(d, absmin_j)));
                //mpfr_printf("old_x =  %25.17RNe + %25.17RNe * I\n", rctd_realref(old_x), rctd_imagref(old_x));
                //mpfr_printf("new_x =  %25.17RNe + %25.17RNe * I\n", rctd_realref(new_x), rctd_imagref(new_x));
                //mpfr_printf("pn_x_d = %25.17RNe + %25.17RNe * I\n", rctd_realref(new_x), rctd_imagref(new_x));
                //pn_x_d = dhorner(new_x, coef, deg);
                //rctd_horner(pn_x_d, new_x, poly); // coef, deg);
                ceval_tdpoly(&pn_x_d, poly, &new_x);

                rctd_abs_td(tmp, &pn_x_d);// pn_x_d := abs(p(new_x))
                rtd_set_ui(tmp2, 1UL); // tmp2 := 1
                rtd_sub(tmp2, tmp2, beta); // tmp2 := 1 - beta
                rtd_mul(tmp2, tmp2, mu); // tmp2 := (1 - beta) * mu
                rtd_sub_ui(tmp2, tmp2, 1UL); // tmp2 := tmp2 - 1 = (1 - beta) * mu - 1
                rtd_neg(tmp2, tmp2); // tmp2 := -tmp2 = 1 - (1 - beta) * mu
                rctd_abs_td(tmp1, get_ctdvector_i(in_coef, 0)); // tmp1 := abs(in_coef[0])
                rtd_mul(tmp2, tmp2, tmp1); // tmp2 := tmp2 * tmp1
                //mpfr_printf("tmp = %25.17RNe <= tmp2 = %25.17RNe ? \n", tmp, tmp2);
                //if(cabs(pn_x_d) <= (1.0 - (1.0 - beta) * mu) * cabs(in_coef[0]))
                if(rtd_cmp(tmp, tmp2) <= 0)
                {
                    //old_x = new_x;
                    //printf("%ld - %ld: new_x := old_x + c%ld * x^%ld\n", times, max_j, absmin_j, absmin_j);
                    rctd_set(&old_x, &new_x);
                    break;
                }        
            }
        }

        // check stopping rule
        //dabsmax_poly(&absmax_anxn, new_x, coef, deg);
        absmax_tdpoly(absmax_anxn, &new_x, poly); // coef, deg);
        //abs_pn_new_x = cabs(dhorner(new_x, coef, deg));
        //rctd_horner(ctmp, new_x, poly); // coef, deg);
        ceval_tdpoly(&ctmp, poly, &new_x); // coef, deg);
        rctd_abs_td(abs_pn_new_x, &ctmp);
        rtd_mul(tmp, absmax_anxn, reps);
        rtd_add(tmp, tmp, aeps);
        //if(abs_pn_new_x <= absmax_anxn * reps + aeps)
        if(rtd_cmp(abs_pn_new_x, tmp) <= 0)
            break;
    }

    free_ctdvector(in_coef);
    free_ctdvector(d);

    //*ret = new_x;
    rctd_set(ret, &new_x);

    return times;
}

// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int ctd_hirano(ctdfloat *ret, ctdfloat *init_x, CTDPoly poly, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
{
    int stop_flag = 0;
    long int i, j, max_j, times, absmin_j, deg = poly->deg;
    ctdfloat old_x, new_x, pn_x_d, ctmp, ctmp2;
    //ctdfloat *in_coef, *d;
    CTDVector in_coef, d;
    double abs_pn_new_x[TDSIZE], absmax_anxn[TDSIZE], absmin_d[TDSIZE], abs_d[TDSIZE];
    double mu[TDSIZE], beta[TDSIZE], lambda[TDSIZE];
    double tmp[TDSIZE], tmp1[TDSIZE], tmp2[TDSIZE];

    // from Sugihara & Murota
    rtd_set_ui(beta, 3UL); rtd_div_ui(beta, beta, 4UL);
    rtd_set_ui(lambda, 2UL);

    // coef of p(x + d)
    //in_coef = (double _Complex *)calloc(deg + 1, sizeof(double _Complex));
    //d = (double _Complex *)calloc(deg, sizeof(double _Complex));
    in_coef = init_ctdvector(deg + 1);
    //printf("init d=%ld, deg = %ld\n", deg);
    d = init_ctdvector(deg + 1);

    //old_x = init_x;
    rctd_set(&old_x, init_x);

    // Main loop
    for(times = 0; times < maxtimes; times++)
    {
        ctd_coef_horner(in_coef, &old_x, poly); // coef, deg);
        //mu = 1.0;
        rtd_set_ui(mu, 1UL);

        //mpfr_printf("%5ld %25.17RNe + %25.17RNe * I -> %15.7RNe + %15.7RNe * I\n", times, rctd_realref(old_x), rctd_imagref(old_x), rctd_realref(get_ctdvector_i(in_coef, 0)), rctd_imagref(get_ctdvector_i(in_coef, 0)));

        // (1)
        //d[0] = 0.0 + 0.0 * I;
        rctd_set_ui_ui(&ctmp, 0UL, 0UL);
        set_ctdvector_i(d, 0, &ctmp);
        //if(fabs(in_coef[1]) != 0.0)
        rctd_abs_td(tmp, get_ctdvector_i(in_coef, 1));
        if(rtd_cmp_ui(tmp, 0UL) != 0)
        {
            //d[0] = -in_coef[0] / in_coef[1];
            rctd_div(&ctmp, get_ctdvector_i(in_coef, 0), get_ctdvector_i(in_coef, 1));
            rctd_neg(&ctmp, &ctmp);
            set_ctdvector_i(d, 0, &ctmp);
        }
        //mpfr_printf("d[0] = %25.17RNe + %25.17RNe * I\n", rctd_realref(get_ctdvector_i(d, 0)), rctd_imagref(get_ctdvector_i(d, 0)));

        //new_x = old_x + d[0];
        rctd_add(&new_x, &old_x, get_ctdvector_i(d, 0));
        //pn_x_d = dhorner(new_x, coef, deg);
        //rctd_horner(pn_x_d, new_x, poly); // coef, deg);
        eval_ctdpoly(&pn_x_d, poly, &new_x);

        rctd_abs_td(tmp, &pn_x_d);
        rctd_abs_td(tmp2, get_ctdvector_i(in_coef, 0));
        rtd_mul(tmp2, tmp2, beta);
        //if(cabs(pn_x_d) <= beta * cabs(in_coef[0]))
        if(rtd_cmp(tmp, tmp2) <= 0)
        {
            // old_x = new_x;
            //printf("%ld: new_x := c0 + c1 * old_x\n", times);
            rctd_set(&old_x, &new_x);
        }
        else
        {
            //printf("%ld: new_x := c0 + c? * old_x^?\n", times);

            for(max_j = 2; max_j <= deg; max_j++)
            {
                //printf("max_j = %ld\n", max_j);
                //mpfr_printf("d[max_j] = %25.17RNe + %25.17RNe * I\n", rctd_realref(get_ctdvector_i(d, max_j)), rctd_imagref(get_ctdvector_i(d, max_j)));
                //mu /= lambda;
                rtd_div(mu, mu, lambda);
                //d[max_j] = dget_min_branch(old_x, mu, in_coef, 0, max_j);
                td_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, max_j);
                set_ctdvector_i(d, max_j, &ctmp2);
                //mpfr_printf("d[max_j = %ld] = %25.17RNe + %25.17RNe\n", max_j, rctd_realref(get_ctdvector_i(d, max_j)), rctd_imagref(get_ctdvector_i(d, max_j)));
                //absmin_d = cabs(d[max_j]);
                rctd_abs_td(absmin_d, get_ctdvector_i(d, max_j));
                absmin_j = max_j;
                for(j = 0; j < max_j; j++)
                {
                    //printf("%ld ", j);
                    //d[j] = dget_min_branch(old_x, mu, in_coef, 0, j + 1) / pow(lambda, 1.0 / (double)(j + 1));
                    td_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, j + 1);
                    set_ctdvector_i(d, j, &ctmp2);
                    rtd_set_ui(tmp, 1UL);
                    rtd_div_ui(tmp, tmp, (unsigned long)(j + 1));
                    rtd_pow_mpfr(tmp1, lambda, tmp);
                    //mpfr_printf("d[%ld] = %25.17RNe + %25.17RNe * I, lambda^%ld = %25.17RNe\n", j, rctd_realref(get_ctdvector_i(d, j)), rctd_imagref(get_ctdvector_i(d,j)), j + 1, tmp1);
                    rctd_div_td(&ctmp2, get_ctdvector_i(d, j), tmp1);
                    set_ctdvector_i(d, j, &ctmp2);
                    //abs_d = cabs(d[j]);
                    rctd_abs_td(abs_d, get_ctdvector_i(d, j));
                    if(rtd_cmp(abs_d, absmin_d) < 0) //abs_d < absmin_d)
                    {
                        // absmin_d = abs_d;
                        rtd_set(absmin_d, abs_d);
                        absmin_j = j;
                    }
                }
                //printf("absmin_j = %ld\n", absmin_j);
                //new_x = old_x + d[absmin_j];
                rctd_add(&new_x, &old_x, get_ctdvector_i(d, absmin_j));
                //mpfr_printf("d[%ld] = %25.17RNe + %25.17RNe * I\n", absmin_j, rctd_realref(get_ctdvector_i(d, absmin_j)), rctd_imagref(get_ctdvector_i(d, absmin_j)));
                //mpfr_printf("old_x =  %25.17RNe + %25.17RNe * I\n", rctd_realref(old_x), rctd_imagref(old_x));
                //mpfr_printf("new_x =  %25.17RNe + %25.17RNe * I\n", rctd_realref(new_x), rctd_imagref(new_x));
                //mpfr_printf("pn_x_d = %25.17RNe + %25.17RNe * I\n", rctd_realref(new_x), rctd_imagref(new_x));
                //pn_x_d = dhorner(new_x, coef, deg);
                //rctd_horner(pn_x_d, new_x, poly); // coef, deg);
                eval_ctdpoly(&pn_x_d, poly, &new_x);

                rctd_abs_td(tmp, &pn_x_d);// pn_x_d := abs(p(new_x))
                rtd_set_ui(tmp2, 1UL); // tmp2 := 1
                rtd_sub(tmp2, tmp2, beta); // tmp2 := 1 - beta
                rtd_mul(tmp2, tmp2, mu); // tmp2 := (1 - beta) * mu
                rtd_sub_ui(tmp2, tmp2, 1UL); // tmp2 := tmp2 - 1 = (1 - beta) * mu - 1
                rtd_neg(tmp2, tmp2); // tmp2 := -tmp2 = 1 - (1 - beta) * mu
                rctd_abs_td(tmp1, get_ctdvector_i(in_coef, 0)); // tmp1 := abs(in_coef[0])
                rtd_mul(tmp2, tmp2, tmp1); // tmp2 := tmp2 * tmp1
                //mpfr_printf("tmp = %25.17RNe <= tmp2 = %25.17RNe ? \n", tmp, tmp2);
                //if(cabs(pn_x_d) <= (1.0 - (1.0 - beta) * mu) * cabs(in_coef[0]))
                if(rtd_cmp(tmp, tmp2) <= 0)
                {
                    //old_x = new_x;
                    //printf("%ld - %ld: new_x := old_x + c%ld * x^%ld\n", times, max_j, absmin_j, absmin_j);
                    rctd_set(&old_x, &new_x);
                    break;
                }        
            }
        }

        // check stopping rule
        //dabsmax_poly(&absmax_anxn, new_x, coef, deg);
        absmax_ctdpoly(absmax_anxn, &new_x, poly); // coef, deg);
        //abs_pn_new_x = cabs(dhorner(new_x, coef, deg));
        //rctd_horner(ctmp, new_x, poly); // coef, deg);
        eval_ctdpoly(&ctmp, poly, &new_x); // coef, deg);
        rctd_abs_td(abs_pn_new_x, &ctmp);
        rtd_mul(tmp, absmax_anxn, reps);
        rtd_add(tmp, tmp, aeps);
        //if(abs_pn_new_x <= absmax_anxn * reps + aeps)
        if(rtd_cmp(abs_pn_new_x, tmp) <= 0)
            break;
    }

    free_ctdvector(in_coef);
    free_ctdvector(d);

    //*ret = new_x;
    rctd_set(ret, &new_x);

    return times;
}

// Deflation of polynomial
// p(x) / (x - r)
void deflation_ctdpoly(CTDPoly ret, CTDPoly pol, ctdfloat *root)
{
    long int i;
    //double _Complex ret;
    ctdfloat ctmp;
    CTDPoly in_ret;
    
    if(pol->deg < 1)
        return;

    in_ret = init_set_ctdpoly(ret);

    // clear
    set0_ctdpoly(in_ret);

    //ret[deg - 1] := coef[deg];
    set_ctdpoly_i(in_ret, pol->deg - 1, get_ctdpoly_i(pol, pol->deg));
    for(i = pol->deg - 2; i >= 0; i--) // i >= 0; i--)
    {
        rctd_mul(&ctmp, get_ctdpoly_i(in_ret, i + 1), root);
        rctd_add(&ctmp, &ctmp, get_ctdpoly_i(pol, i + 1));
        //ret[i] := ret[i + 1] * root + coef[i + 1];
        set_ctdpoly_i(in_ret, i, &ctmp);
    }

    setdegree_ctdpoly(in_ret);
    subst_ctdpoly(ret, in_ret);

    free_ctdpoly(in_ret);
    //return ret;
    return;
}
#endif // USE_TDLINEAR
