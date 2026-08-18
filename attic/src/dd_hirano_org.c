/********************************************************************************/
/* dd_hirano.c: Robust solver for lgebraic Equations using Hirano method        */
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
#include "ddlinear.h"
#include "poly.h"

// --------------------------
// DD
// --------------------------
//#ifdef USE_GMP
#ifdef USE_DDLINEAR

// Horner method
// return p(x)
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
void dd_horner(cddfloat *ret, cddfloat *x, DDPoly poly) // mpf_t coef[], long int deg)
{
    long int i;
    //cddfloat ret;

    //ret = coef[deg] + 0.0 * I;
    rcdd_set_ui_ui(ret, 0UL, 0UL);
    rcdd_set_dd(ret, get_ddpoly_i(poly, poly->deg)); // , 0UL);
    for(i = poly->deg - 1; i >= 0; i--)
    {
        //ret = ret * x + (coef[i] + 0.0 * I);
        rcdd_mul(ret, ret, x);
        rcdd_add_dd(ret, ret, get_ddpoly_i(poly, i));
    }

    //return ret;
    return;
}

// Coef of p(x + d)
// 
// based on Horner method
void dd_coef_horner(CDDVector ret_coef, cddfloat *x, DDPoly poly) // mpf_t coef[], long int deg)
{
    long int l, i;
    cddfloat ctmp, ctmp2;

    // Initial setting
    rcdd_set0(&ctmp2);
    rcdd_set0(&ctmp);
    for(i = 0; i <= poly->deg; i++)
    {
        //ret_coef[i] = coef[i] + 0.0 * I;
        rcdd_set_dd(&ctmp2, get_ddpoly_i(poly, i));
        set_cddvector_i(ret_coef, i, &ctmp2);
    }

    // ret_coef[i] := p^(l)(x) / l!
    for(l = 0; l <= poly->deg; l++)
    {
        for(i = poly->deg - 1; i >= l; i--)
        {
            //ret_coef[i] = x * ret_coef[i + 1] + ret_coef[i];
            rcdd_mul(&ctmp, get_cddvector_i(ret_coef, i + 1), x);
            rcdd_add(&ctmp2, &ctmp, get_cddvector_i(ret_coef, i));
            set_cddvector_i(ret_coef, i, &ctmp2);
        }
    }
}

// Horner method
// return p(x)
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
void cdd_horner(cddfloat *ret, cddfloat *x, CDDPoly poly) // mpf_t coef[], long int deg)
{
    long int i;
    //double _Complex ret;

    //ret = coef[deg] + 0.0 * I;
    rcdd_set_ui(ret, 0UL);
    rcdd_set(ret, get_cddpoly_i(poly, poly->deg)); // , 0UL);
    for(i = poly->deg - 1; i >= 0; i--)
    {
        //ret = ret * x + (coef[i] + 0.0 * I);
        rcdd_mul(ret, ret, x);
        rcdd_add(ret, ret, get_cddpoly_i(poly, i));
    }

    //return ret;
    return;
}

void cdd_coef_horner(CDDVector ret_coef, cddfloat *x, CDDPoly poly) // mpf_t coef[], long int deg)
{
    long int l, i;
    cddfloat ctmp, ctmp2;

    // Initial setting
    for(i = 0; i <= poly->deg; i++)
    {
        //ret_coef[i] = coef[i] + 0.0 * I;
        rcdd_set(&ctmp2, get_cddpoly_i(poly, i));
        set_cddvector_i(ret_coef, i, &ctmp2);
    }

    // ret_coef[i] := p^(l)(x) / l!
    for(l = 0; l <= poly->deg; l++)
    {
        for(i = poly->deg - 1; i >= l; i--)
        {
            //ret_coef[i] = x * ret_coef[i + 1] + ret_coef[i];
            rcdd_mul(&ctmp, get_cddvector_i(ret_coef, i + 1), x);
            rcdd_add(&ctmp2, &ctmp, get_cddvector_i(ret_coef, i));
            set_cddvector_i(ret_coef, i, &ctmp2);
        }
    }
}

// return j and max|a_j x^j| from p(x)
long int absmax_ddpoly(double absmax_anxn[DDSIZE], cddfloat *x, DDPoly poly) // mpf_t coef[], long int deg)
{
    long int i, ret_i;
    double abs_anxn[DDSIZE];
    cddfloat xn, anxn;

    // xn := x
    rcdd_set(&xn, x);

    ret_i = 0;
    //*absmax_anxn = fabs(coef[0]);
    rdd_abs(absmax_anxn, get_ddpoly_i(poly, 0));
 
    for(i = 1; i <= poly->deg; i++)
    {
        //anxn = coef[i] * xn;
        rcdd_mul_dd(&anxn, &xn, get_ddpoly_i(poly, i));
        //abs_anxn = fabs(anxn);
        rcdd_abs_dd(abs_anxn, &anxn);
        //if(*absmax_anxn < abs_anxn)
        if(rdd_cmp(absmax_anxn, abs_anxn) < 0)
        {
            ret_i = i;
            //*absmax_anxn = abs_anxn;
            rdd_set(absmax_anxn, abs_anxn);
        }
        //xn *= x;
        rcdd_mul(&xn, &xn, x);
    }

    return ret_i;
}

long int absmax_cddpoly(double absmax_anxn[DDSIZE], cddfloat *x, CDDPoly poly) // mpf_t coef[], long int deg)
{
    long int i, ret_i;
    double abs_anxn[DDSIZE];
    cddfloat xn, anxn;

    // xn := x
    rcdd_set(&xn, x);

    ret_i = 0;
    //*absmax_anxn = fabs(coef[0]);
    rcdd_abs_dd(absmax_anxn, get_cddpoly_i(poly, 0));
 
    for(i = 1; i <= poly->deg; i++)
    {
        //anxn = coef[i] * xn;
        rcdd_mul(&anxn, &xn, get_cddpoly_i(poly, i));
        //abs_anxn = fabs(anxn);
        rcdd_abs_dd(abs_anxn, &anxn);
        //if(*absmax_anxn < abs_anxn)
        if(rdd_cmp(absmax_anxn, abs_anxn) < 0)
        {
            ret_i = i;
            //*absmax_anxn = abs_anxn;
            rdd_set(absmax_anxn, abs_anxn);
        }
        //xn *= x;
        rcdd_mul(&xn, &xn, x);
    }

    return ret_i;
}


// get_plus_arg
// return arg(x) in [0, 2 PI]
void dd_get_plus_arg(double ret[DDSIZE], cddfloat *x)
{
    double pi2[DDSIZE];

    // pi2 := 2 * PI
    rdd_const_pi(pi2);
    rdd_mul_ui(pi2, pi2, 2UL);

    //double ret = carg(x);
    rcdd_arg(ret, x);

    //if(ret < 0)
    if(rdd_cmp_ui(ret, 0UL) < 0)
    {
        //ret = 2.0 * M_PI + ret;
        rdd_add(ret, ret, pi2);
    }

    //return ret;
    return;
}

// get_nearest_int
void dd_get_nearest_int(double ret[DDSIZE], double real_x[DDSIZE])
{
    //long int int_x_floor, int_x_ceil;
    //double dist_x_floor, dist_x_ceil;
    double int_x_floor[DDSIZE], int_x_ceil[DDSIZE];
    double dist_x_floor[DDSIZE], dist_x_ceil[DDSIZE];

    //int_x_floor = (long int)floor(real_x);
    rdd_func_mpfr(int_x_floor, mpfr_rint_floor, real_x); // _dd);
    //int_x_ceil  = (long int)ceil (real_x);
    rdd_func_mpfr(int_x_ceil, mpfr_rint_ceil, real_x); // _dd);

    //dist_x_floor = fabs(real_x - (double)int_x_floor);
    rdd_sub(dist_x_floor, real_x, int_x_floor); rdd_abs(dist_x_floor, dist_x_floor);
    //dist_x_ceil  = fabs(real_x - (double)int_x_ceil);
    rdd_sub(dist_x_ceil, real_x, int_x_ceil); rdd_abs(dist_x_ceil, dist_x_ceil);

    //if(dist_x_floor < dist_x_ceil) return int_x_floor;
    if(rdd_cmp(dist_x_floor, dist_x_ceil) < 0) rdd_set(ret, int_x_floor);
    //else return int_x_ceil;
    else rdd_set(ret, int_x_ceil);

}

// get_min_branch
void dd_get_min_branch(cddfloat *ret, cddfloat *x, double mu[DDSIZE], CDDVector coef, long int i_num, long int i_den)
{
    long int j = i_den;
    double phi[DDSIZE], psi[DDSIZE], pi2[DDSIZE], real_j[DDSIZE], tmp[DDSIZE], tmp2[DDSIZE], pow_tmp[DDSIZE];
    cddfloat ctmp, ctmp2;

    //mpfr_printf("j = i_den = %ld, coef[j] = %25.17RNe + %25.17RNe * I \n", j, rcdd_realref(get_cddvector_i(coef, j)), rcdd_imagref(get_cddvector_i(coef, j)));
    //printf("prec = %ld\n");

    rcdd_abs_dd(tmp, get_cddvector_i(coef, j));
    //mpfr_printf("|coef[j = %ld]| = %25.17RNe\n", j, tmp);
    if(rdd_cmp_ui(tmp, 0UL) == 0) return;

    // pi2 := 2 * PI
    rdd_const_pi(pi2); //get_bnc_default_rounding_mode());
    rdd_mul_ui(pi2, pi2, 2UL);
    //mpfr_printf("pi2 = %RNe\n", pi2);

    //phi = dget_plus_arg(x) / pi2;
    dd_get_plus_arg(phi, x); rdd_div(phi, phi, pi2);
    //psi = dget_plus_arg(-coef[i_num] / coef[j]) / pi2;
    rcdd_div(&ctmp, get_cddvector_i(coef, i_num),  get_cddvector_i(coef, j));
    rcdd_neg(&ctmp, &ctmp);
    dd_get_plus_arg(psi, &ctmp);
    rdd_div(psi, psi, pi2);

    //real_j = (double)dget_nearest_int((double)j * (0.5 - phi) - psi);
    rdd_set_ui(real_j, 1UL); rdd_div_ui(real_j, real_j, 2UL);
    rdd_sub(real_j, real_j, phi);
    rdd_mul_ui(real_j, real_j, (unsigned long)j);
    rdd_sub(real_j, real_j, psi);
    dd_get_nearest_int(real_j, real_j);
    //mpfr_printf("real_j = %25.17RNe\n", real_j);

    // ret = pow(
    //     cabs(mu * coef[i_num] / coef[j]),
    //     1.0 / (double)j
    // ) * cexp(
    //     pi2 * I * ((psi + real_j) / (double)j)
    // );

    // ctmp := mu * coef[i_num]
    rcdd_mul_dd(&ctmp, get_cddvector_i(coef, i_num), mu);
    // ctmp := ctmp / coef[j] 
    rcdd_div(&ctmp, &ctmp, get_cddvector_i(coef, j)); // Fix! 2025-03-03 T.Kouya
    // tmp := |ctmp|
    rcdd_abs_dd(tmp, &ctmp);
    // tmp2 := 1 / j
    rdd_set_ui(tmp2, 1UL); rdd_div_ui(tmp2, tmp2, (unsigned long)j);
    // pow_tmp := pow(|ctmp|, 1/j)
    rdd_pow_mpfr(pow_tmp, tmp, tmp2);
    //mpfr_printf("pow(%25.17RNe, %25.17RNe) = %25.17RNe\n", tmp, tmp2, pow_tmp);

    // tmp := psi + real_j
    rdd_add(tmp, psi, real_j);
    // tmp := (psi + real_j) / j
    rdd_div_ui(tmp, tmp, (unsigned long)j);
    // tmp := tmp * pi2
    rdd_mul(tmp, tmp, pi2);
    // tmp2 := 0
    rdd_set_ui(tmp2, 0UL);
    // ctmp := 0 + tmp * I
    rcdd_set_dd_dd(&ctmp, tmp2, tmp);
    // ret := cexp(ctmp);
    rcdd_func_mpc(ret, mpc_exp, &ctmp);
    //mpfr_printf("cexp(%25.17RNe + %25.17RNe * I) = %25.17RNe + %25.17RNe * I\n", rcdd_realref(ctmp), rcdd_imagref(ctmp), rcdd_realref(ret), rcdd_imagref(ret));
    // ret := cexp(ctmp) * pow_tmp
    rcdd_mul_dd(ret, ret, pow_tmp);

    return;
}

// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int dd_hirano(cddfloat *ret, cddfloat *init_x, DDPoly poly, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
{
    int stop_flag = 0;
    long int i, j, max_j, times, absmin_j, deg = poly->deg;
    cddfloat old_x, new_x, pn_x_d, ctmp, ctmp2;
    //cddfloat *in_coef, *d;
    CDDVector in_coef, d;
    double abs_pn_new_x[DDSIZE], absmax_anxn[DDSIZE], absmin_d[DDSIZE], abs_d[DDSIZE];
    double mu[DDSIZE], beta[DDSIZE], lambda[DDSIZE];
    double tmp[DDSIZE], tmp1[DDSIZE], tmp2[DDSIZE];

    // from Sugihara & Murota
    rdd_set_ui(beta, 3UL); rdd_div_ui(beta, beta, 4UL);
    rdd_set_ui(lambda, 2UL);
    
    // coef of p(x + d)
    //in_coef = (double _Complex *)calloc(deg + 1, sizeof(double _Complex));
    //d = (double _Complex *)calloc(deg, sizeof(double _Complex));
    in_coef = init_cddvector(deg + 1);
    //printf("init d=%ld, deg = %ld\n", deg);
    d = init_cddvector(deg + 1);

    //old_x = init_x;
    rcdd_set(&old_x, init_x);

    // Main loop
    for(times = 0; times < maxtimes; times++)
    {
        dd_coef_horner(in_coef, &old_x, poly); // coef, deg);
        //mu = 1.0;
        rdd_set_ui(mu, 1UL);

        //mpfr_printf("%5ld %25.17RNe + %25.17RNe * I -> %15.7RNe + %15.7RNe * I\n", times, rcdd_realref(old_x), rcdd_imagref(old_x), rcdd_realref(get_cddvector_i(in_coef, 0)), rcdd_imagref(get_cddvector_i(in_coef, 0)));

        // (1)
        //d[0] = 0.0 + 0.0 * I;
        rcdd_set_ui_ui(&ctmp, 0UL, 0UL);
        set_cddvector_i(d, 0, &ctmp);
        //if(fabs(in_coef[1]) != 0.0)
        rcdd_abs_dd(tmp, get_cddvector_i(in_coef, 1));
        if(rdd_cmp_ui(tmp, 0UL) != 0)
        {
            //d[0] = -in_coef[0] / in_coef[1];
            rcdd_div(&ctmp, get_cddvector_i(in_coef, 0), get_cddvector_i(in_coef, 1));
            rcdd_neg(&ctmp, &ctmp);
            set_cddvector_i(d, 0, &ctmp);
        }
        //mpfr_printf("d[0] = %25.17RNe + %25.17RNe * I\n", rcdd_realref(get_cddvector_i(d, 0)), rcdd_imagref(get_cddvector_i(d, 0)));

        //new_x = old_x + d[0];
        rcdd_add(&new_x, &old_x, get_cddvector_i(d, 0));
        //pn_x_d = dhorner(new_x, coef, deg);
        //mpf_horner(pn_x_d, new_x, poly); // coef, deg);
        ceval_ddpoly(&pn_x_d, poly, &new_x); // coef, deg);

        rcdd_abs_dd(tmp, &pn_x_d);
        rcdd_abs_dd(tmp2, get_cddvector_i(in_coef, 0));
        rdd_mul(tmp2, tmp2, beta);
        //if(cabs(pn_x_d) <= beta * cabs(in_coef[0]))
        if(rdd_cmp(tmp, tmp2) <= 0)
        {
            // old_x = new_x;
            //printf("%ld: new_x := c0 + c1 * old_x\n", times);
            rcdd_set(&old_x, &new_x);
        }
        else
        {
            //printf("%ld: new_x := c0 + c? * old_x^?\n", times);

            for(max_j = 2; max_j <= deg; max_j++)
            {
                //printf("max_j = %ld\n", max_j);
                //mpfr_printf("d[max_j] = %25.17RNe + %25.17RNe * I\n", rcdd_realref(get_cddvector_i(d, max_j)), rcdd_imagref(get_cddvector_i(d, max_j)));
                //mu /= lambda;
                rdd_div(mu, mu, lambda);
                //d[max_j] = dget_min_branch(old_x, mu, in_coef, 0, max_j);
                dd_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, max_j);
                set_cddvector_i(d, max_j, &ctmp2);
                //mpfr_printf("d[max_j = %ld] = %25.17RNe + %25.17RNe\n", max_j, rcdd_realref(get_cddvector_i(d, max_j)), rcdd_imagref(get_cddvector_i(d, max_j)));
                //absmin_d = cabs(d[max_j]);
                rcdd_abs_dd(absmin_d, get_cddvector_i(d, max_j));
                absmin_j = max_j;
                for(j = 0; j < max_j; j++)
                {
                    //printf("%ld ", j);
                    //d[j] = dget_min_branch(old_x, mu, in_coef, 0, j + 1) / pow(lambda, 1.0 / (double)(j + 1));
                    dd_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, j + 1);
                    set_cddvector_i(d, j, &ctmp2);
                    rdd_set_ui(tmp, 1UL);
                    rdd_div_ui(tmp, tmp, (unsigned long)(j + 1));
                    rdd_pow_mpfr(tmp1, lambda, tmp);
                    //mpfr_printf("d[%ld] = %25.17RNe + %25.17RNe * I, lambda^%ld = %25.17RNe\n", j, rcdd_realref(get_cddvector_i(d, j)), rcdd_imagref(get_cddvector_i(d,j)), j + 1, tmp1);
                    rcdd_div_dd(&ctmp2, get_cddvector_i(d, j), tmp1);
                    set_cddvector_i(d, j, &ctmp2);
                    //abs_d = cabs(d[j]);
                    rcdd_abs_dd(abs_d, get_cddvector_i(d, j));
                    if(rdd_cmp(abs_d, absmin_d) < 0) //abs_d < absmin_d)
                    {
                        // absmin_d = abs_d;
                        rdd_set(absmin_d, abs_d);
                        absmin_j = j;
                    }
                }
                //printf("absmin_j = %ld\n", absmin_j);
                //new_x = old_x + d[absmin_j];
                rcdd_add(&new_x, &old_x, get_cddvector_i(d, absmin_j));
                //mpfr_printf("d[%ld] = %25.17RNe + %25.17RNe * I\n", absmin_j, rcdd_realref(get_cddvector_i(d, absmin_j)), rcdd_imagref(get_cddvector_i(d, absmin_j)));
                //mpfr_printf("old_x =  %25.17RNe + %25.17RNe * I\n", rcdd_realref(old_x), rcdd_imagref(old_x));
                //mpfr_printf("new_x =  %25.17RNe + %25.17RNe * I\n", rcdd_realref(new_x), rcdd_imagref(new_x));
                //mpfr_printf("pn_x_d = %25.17RNe + %25.17RNe * I\n", rcdd_realref(new_x), rcdd_imagref(new_x));
                //pn_x_d = dhorner(new_x, coef, deg);
                //mpf_horner(pn_x_d, new_x, poly); // coef, deg);
                ceval_ddpoly(&pn_x_d, poly, &new_x); // coef, deg);

                rcdd_abs_dd(tmp, &pn_x_d);// pn_x_d := abs(p(new_x))
                rdd_set_ui(tmp2, 1UL); // tmp2 := 1
                rdd_sub(tmp2, tmp2, beta); // tmp2 := 1 - beta
                rdd_mul(tmp2, tmp2, mu); // tmp2 := (1 - beta) * mu
                rdd_sub_ui(tmp2, tmp2, 1UL); // tmp2 := tmp2 - 1 = (1 - beta) * mu - 1
                rdd_neg(tmp2, tmp2); // tmp2 := -tmp2 = 1 - (1 - beta) * mu
                rcdd_abs_dd(tmp1, get_cddvector_i(in_coef, 0)); // tmp1 := abs(in_coef[0])
                rdd_mul(tmp2, tmp2, tmp1); // tmp2 := tmp2 * tmp1
                //mpfr_printf("tmp = %25.17RNe <= tmp2 = %25.17RNe ? \n", tmp, tmp2);
                //if(cabs(pn_x_d) <= (1.0 - (1.0 - beta) * mu) * cabs(in_coef[0]))
                if(rdd_cmp(tmp, tmp2) <= 0)
                {
                    //old_x = new_x;
                    //printf("%ld - %ld: new_x := old_x + c%ld * x^%ld\n", times, max_j, absmin_j, absmin_j);
                    rcdd_set(&old_x, &new_x);
                    break;
                }        
            }
        }

        // check stopping rule
        //dabsmax_poly(&absmax_anxn, new_x, coef, deg);
        absmax_ddpoly(absmax_anxn, &new_x, poly); // coef, deg);
        //abs_pn_new_x = cabs(dhorner(new_x, coef, deg));
        //mpf_horner(ctmp, new_x, poly); // coef, deg);
        ceval_ddpoly(&ctmp, poly, &new_x); // coef, deg);
        rcdd_abs_dd(abs_pn_new_x, &ctmp);
        rdd_mul(tmp, absmax_anxn, reps);
        rdd_add(tmp, tmp, aeps);
        //if(abs_pn_new_x <= absmax_anxn * reps + aeps)
        if(rdd_cmp(abs_pn_new_x, tmp) <= 0)
            break;
    }

    free_cddvector(in_coef);
    free_cddvector(d);

    //*ret = new_x;
    rcdd_set(ret, &new_x);

    return times;
}

// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int cdd_hirano(cddfloat *ret, cddfloat *init_x, CDDPoly poly, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
{
    int stop_flag = 0;
    long int i, j, max_j, times, absmin_j, deg = poly->deg;
    cddfloat old_x, new_x, pn_x_d, ctmp, ctmp2;
    //cddfloat *in_coef, *d;
    CDDVector in_coef, d;
    double abs_pn_new_x[DDSIZE], absmax_anxn[DDSIZE], absmin_d[DDSIZE], abs_d[DDSIZE];
    double mu[DDSIZE], beta[DDSIZE], lambda[DDSIZE];
    double tmp[DDSIZE], tmp1[DDSIZE], tmp2[DDSIZE];

    // from Sugihara & Murota
    rdd_set_ui(beta, 3UL); rdd_div_ui(beta, beta, 4UL);
    rdd_set_ui(lambda, 2UL);

    // coef of p(x + d)
    //in_coef = (double _Complex *)calloc(deg + 1, sizeof(double _Complex));
    //d = (double _Complex *)calloc(deg, sizeof(double _Complex));
    in_coef = init_cddvector(deg + 1);
    //printf("init d=%ld, deg = %ld\n", deg);
    d = init_cddvector(deg + 1);

    //old_x = init_x;
    rcdd_set(&old_x, init_x);

    // Main loop
    for(times = 0; times < maxtimes; times++)
    {
        cdd_coef_horner(in_coef, &old_x, poly); // coef, deg);
        //mu = 1.0;
        rdd_set_ui(mu, 1UL);

        //mpfr_printf("%5ld %25.17RNe + %25.17RNe * I -> %15.7RNe + %15.7RNe * I\n", times, rcdd_realref(old_x), rcdd_imagref(old_x), rcdd_realref(get_cddvector_i(in_coef, 0)), rcdd_imagref(get_cddvector_i(in_coef, 0)));

        // (1)
        //d[0] = 0.0 + 0.0 * I;
        rcdd_set_ui_ui(&ctmp, 0UL, 0UL);
        set_cddvector_i(d, 0, &ctmp);
        //if(fabs(in_coef[1]) != 0.0)
        rcdd_abs_dd(tmp, get_cddvector_i(in_coef, 1));
        if(rdd_cmp_ui(tmp, 0UL) != 0)
        {
            //d[0] = -in_coef[0] / in_coef[1];
            rcdd_div(&ctmp, get_cddvector_i(in_coef, 0), get_cddvector_i(in_coef, 1));
            rcdd_neg(&ctmp, &ctmp);
            set_cddvector_i(d, 0, &ctmp);
        }
        //mpfr_printf("d[0] = %25.17RNe + %25.17RNe * I\n", rcdd_realref(get_cddvector_i(d, 0)), rcdd_imagref(get_cddvector_i(d, 0)));

        //new_x = old_x + d[0];
        rcdd_add(&new_x, &old_x, get_cddvector_i(d, 0));
        //pn_x_d = dhorner(new_x, coef, deg);
        //rcdd_horner(pn_x_d, new_x, poly); // coef, deg);
        eval_cddpoly(&pn_x_d, poly, &new_x);

        rcdd_abs_dd(tmp, &pn_x_d);
        rcdd_abs_dd(tmp2, get_cddvector_i(in_coef, 0));
        rdd_mul(tmp2, tmp2, beta);
        //if(cabs(pn_x_d) <= beta * cabs(in_coef[0]))
        if(rdd_cmp(tmp, tmp2) <= 0)
        {
            // old_x = new_x;
            //printf("%ld: new_x := c0 + c1 * old_x\n", times);
            rcdd_set(&old_x, &new_x);
        }
        else
        {
            //printf("%ld: new_x := c0 + c? * old_x^?\n", times);

            for(max_j = 2; max_j <= deg; max_j++)
            {
                //printf("max_j = %ld\n", max_j);
                //mpfr_printf("d[max_j] = %25.17RNe + %25.17RNe * I\n", rcdd_realref(get_cddvector_i(d, max_j)), rcdd_imagref(get_cddvector_i(d, max_j)));
                //mu /= lambda;
                rdd_div(mu, mu, lambda);
                //d[max_j] = dget_min_branch(old_x, mu, in_coef, 0, max_j);
                dd_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, max_j);
                set_cddvector_i(d, max_j, &ctmp2);
                //mpfr_printf("d[max_j = %ld] = %25.17RNe + %25.17RNe\n", max_j, rcdd_realref(get_cddvector_i(d, max_j)), rcdd_imagref(get_cddvector_i(d, max_j)));
                //absmin_d = cabs(d[max_j]);
                rcdd_abs_dd(absmin_d, get_cddvector_i(d, max_j));
                absmin_j = max_j;
                for(j = 0; j < max_j; j++)
                {
                    //printf("%ld ", j);
                    //d[j] = dget_min_branch(old_x, mu, in_coef, 0, j + 1) / pow(lambda, 1.0 / (double)(j + 1));
                    dd_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, j + 1);
                    set_cddvector_i(d, j, &ctmp2);
                    rdd_set_ui(tmp, 1UL);
                    rdd_div_ui(tmp, tmp, (unsigned long)(j + 1));
                    rdd_pow_mpfr(tmp1, lambda, tmp);
                    //mpfr_printf("d[%ld] = %25.17RNe + %25.17RNe * I, lambda^%ld = %25.17RNe\n", j, rcdd_realref(get_cddvector_i(d, j)), rcdd_imagref(get_cddvector_i(d,j)), j + 1, tmp1);
                    rcdd_div_dd(&ctmp2, get_cddvector_i(d, j), tmp1);
                    set_cddvector_i(d, j, &ctmp2);
                    //abs_d = cabs(d[j]);
                    rcdd_abs_dd(abs_d, get_cddvector_i(d, j));
                    if(rdd_cmp(abs_d, absmin_d) < 0) //abs_d < absmin_d)
                    {
                        // absmin_d = abs_d;
                        rdd_set(absmin_d, abs_d);
                        absmin_j = j;
                    }
                }
                //printf("absmin_j = %ld\n", absmin_j);
                //new_x = old_x + d[absmin_j];
                rcdd_add(&new_x, &old_x, get_cddvector_i(d, absmin_j));
                //mpfr_printf("d[%ld] = %25.17RNe + %25.17RNe * I\n", absmin_j, rcdd_realref(get_cddvector_i(d, absmin_j)), rcdd_imagref(get_cddvector_i(d, absmin_j)));
                //mpfr_printf("old_x =  %25.17RNe + %25.17RNe * I\n", rcdd_realref(old_x), rcdd_imagref(old_x));
                //mpfr_printf("new_x =  %25.17RNe + %25.17RNe * I\n", rcdd_realref(new_x), rcdd_imagref(new_x));
                //mpfr_printf("pn_x_d = %25.17RNe + %25.17RNe * I\n", rcdd_realref(new_x), rcdd_imagref(new_x));
                //pn_x_d = dhorner(new_x, coef, deg);
                //rcdd_horner(pn_x_d, new_x, poly); // coef, deg);
                eval_cddpoly(&pn_x_d, poly, &new_x);

                rcdd_abs_dd(tmp, &pn_x_d);// pn_x_d := abs(p(new_x))
                rdd_set_ui(tmp2, 1UL); // tmp2 := 1
                rdd_sub(tmp2, tmp2, beta); // tmp2 := 1 - beta
                rdd_mul(tmp2, tmp2, mu); // tmp2 := (1 - beta) * mu
                rdd_sub_ui(tmp2, tmp2, 1UL); // tmp2 := tmp2 - 1 = (1 - beta) * mu - 1
                rdd_neg(tmp2, tmp2); // tmp2 := -tmp2 = 1 - (1 - beta) * mu
                rcdd_abs_dd(tmp1, get_cddvector_i(in_coef, 0)); // tmp1 := abs(in_coef[0])
                rdd_mul(tmp2, tmp2, tmp1); // tmp2 := tmp2 * tmp1
                //mpfr_printf("tmp = %25.17RNe <= tmp2 = %25.17RNe ? \n", tmp, tmp2);
                //if(cabs(pn_x_d) <= (1.0 - (1.0 - beta) * mu) * cabs(in_coef[0]))
                if(rdd_cmp(tmp, tmp2) <= 0)
                {
                    //old_x = new_x;
                    //printf("%ld - %ld: new_x := old_x + c%ld * x^%ld\n", times, max_j, absmin_j, absmin_j);
                    rcdd_set(&old_x, &new_x);
                    break;
                }        
            }
        }

        // check stopping rule
        //dabsmax_poly(&absmax_anxn, new_x, coef, deg);
        absmax_cddpoly(absmax_anxn, &new_x, poly); // coef, deg);
        //abs_pn_new_x = cabs(dhorner(new_x, coef, deg));
        //rcdd_horner(ctmp, new_x, poly); // coef, deg);
        eval_cddpoly(&ctmp, poly, &new_x); // coef, deg);
        rcdd_abs_dd(abs_pn_new_x, &ctmp);
        rdd_mul(tmp, absmax_anxn, reps);
        rdd_add(tmp, tmp, aeps);
        //if(abs_pn_new_x <= absmax_anxn * reps + aeps)
        if(rdd_cmp(abs_pn_new_x, tmp) <= 0)
            break;
    }

    free_cddvector(in_coef);
    free_cddvector(d);

    //*ret = new_x;
    rcdd_set(ret, &new_x);

    return times;
}

// Deflation of polynomial
// p(x) / (x - r)
void deflation_cddpoly(CDDPoly ret, CDDPoly pol, cddfloat *root)
{
    long int i;
    //double _Complex ret;
    cddfloat ctmp;
    CDDPoly in_ret;
    
    if(pol->deg < 1)
        return;

    in_ret = init_set_cddpoly(ret);

    // clear
    set0_cddpoly(in_ret);

    //ret[deg - 1] := coef[deg];
    set_cddpoly_i(in_ret, pol->deg - 1, get_cddpoly_i(pol, pol->deg));
    for(i = pol->deg - 2; i >= 0; i--) // i >= 0; i--)
    {
        rcdd_mul(&ctmp, get_cddpoly_i(in_ret, i + 1), root);
        rcdd_add(&ctmp, &ctmp, get_cddpoly_i(pol, i + 1));
        //ret[i] := ret[i + 1] * root + coef[i + 1];
        set_cddpoly_i(in_ret, i, &ctmp);
    }

    setdegree_cddpoly(in_ret);
    subst_cddpoly(ret, in_ret);

    free_cddpoly(in_ret);
    //return ret;
    return;
}
#endif // USE_DDLINEAR
