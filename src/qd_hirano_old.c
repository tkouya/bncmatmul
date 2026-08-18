/********************************************************************************/
/* qd_hirano.c: Robust solver for lgebraic Equations using Hirano method        */
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
#include "qdlinear.h"
#include "poly.h"

// --------------------------
// QD
// --------------------------
//#ifdef USE_GMP
#ifdef USE_QDLINEAR

// Horner method
// return p(x)
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
void qd_horner(cqdfloat *ret, cqdfloat *x, QDPoly poly) // mpf_t coef[], long int deg)
{
    long int i;
    //cqdfloat ret;

    //ret = coef[deg] + 0.0 * I;
    rcqd_set_ui(ret, 0UL);
    rcqd_set_qd(ret, get_qdpoly_i(poly, poly->deg)); // , 0UL);
    for(i = poly->deg - 1; i >= 0; i--)
    {
        //ret = ret * x + (coef[i] + 0.0 * I);
        rcqd_mul(ret, ret, x);
        rcqd_add_qd(ret, ret, get_qdpoly_i(poly, i));
    }

    //return ret;
    return;
}

// Coef of p(x + d)
// 
// based on Horner method
void qd_coef_horner(CQDVector ret_coef, cqdfloat *x, QDPoly poly) // mpf_t coef[], long int deg)
{
    long int l, i;
    cqdfloat ctmp, ctmp2;

    // Initial setting
    for(i = 0; i <= poly->deg; i++)
    {
        //ret_coef[i] = coef[i] + 0.0 * I;
        rcqd_set_qd(&ctmp2, get_qdpoly_i(poly, i));
        set_cqdvector_i(ret_coef, i, &ctmp2);
    }

    // ret_coef[i] := p^(l)(x) / l!
    for(l = 0; l <= poly->deg; l++)
    {
        for(i = poly->deg - 1; i >= l; i--)
        {
            //ret_coef[i] = x * ret_coef[i + 1] + ret_coef[i];
            rcqd_mul(&ctmp, get_cqdvector_i(ret_coef, i + 1), x);
            rcqd_add(&ctmp2, &ctmp, get_cqdvector_i(ret_coef, i));
            set_cqdvector_i(ret_coef, i, &ctmp2);
        }
    }
}

// Horner method
// return p(x)
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
void cqd_horner(cqdfloat *ret, cqdfloat *x, CQDPoly poly) // mpf_t coef[], long int deg)
{
    long int i;
    //double _Complex ret;

    //ret = coef[deg] + 0.0 * I;
    rcqd_set_ui(ret, 0UL);
    rcqd_set(ret, get_cqdpoly_i(poly, poly->deg)); // , 0UL);
    for(i = poly->deg - 1; i >= 0; i--)
    {
        //ret = ret * x + (coef[i] + 0.0 * I);
        rcqd_mul(ret, ret, x);
        rcqd_add(ret, ret, get_cqdpoly_i(poly, i));
    }

    //return ret;
    return;
}

void cqd_coef_horner(CQDVector ret_coef, cqdfloat *x, CQDPoly poly) // mpf_t coef[], long int deg)
{
    long int l, i;
    cqdfloat ctmp, ctmp2;

    // Initial setting
    for(i = 0; i <= poly->deg; i++)
    {
        //ret_coef[i] = coef[i] + 0.0 * I;
        rcqd_set(&ctmp2, get_cqdpoly_i(poly, i));
        set_cqdvector_i(ret_coef, i, &ctmp2);
    }

    // ret_coef[i] := p^(l)(x) / l!
    for(l = 0; l <= poly->deg; l++)
    {
        for(i = poly->deg - 1; i >= l; i--)
        {
            //ret_coef[i] = x * ret_coef[i + 1] + ret_coef[i];
            rcqd_mul(&ctmp, get_cqdvector_i(ret_coef, i + 1), x);
            rcqd_add(&ctmp2, &ctmp, get_cqdvector_i(ret_coef, i));
            set_cqdvector_i(ret_coef, i, &ctmp2);
        }
    }
}

// return j and max|a_j x^j| from p(x)
long int absmax_qdpoly(double absmax_anxn[QDSIZE], cqdfloat *x, QDPoly poly) // mpf_t coef[], long int deg)
{
    long int i, ret_i;
    double abs_anxn[QDSIZE];
    cqdfloat xn, anxn;

    // xn := x
    rcqd_set(&xn, x);

    ret_i = 0;
    //*absmax_anxn = fabs(coef[0]);
    rqd_abs(absmax_anxn, get_qdpoly_i(poly, 0));
 
    for(i = 1; i <= poly->deg; i++)
    {
        //anxn = coef[i] * xn;
        rcqd_mul_qd(&anxn, &xn, get_qdpoly_i(poly, i));
        //abs_anxn = fabs(anxn);
        rcqd_abs_qd(abs_anxn, &anxn);
        //if(*absmax_anxn < abs_anxn)
        if(rqd_cmp(absmax_anxn, abs_anxn) < 0)
        {
            ret_i = i;
            //*absmax_anxn = abs_anxn;
            rqd_set(absmax_anxn, abs_anxn);
        }
        //xn *= x;
        rcqd_mul(&xn, &xn, x);
    }

    return ret_i;
}

long int absmax_cqdpoly(double absmax_anxn[QDSIZE], cqdfloat *x, CQDPoly poly) // mpf_t coef[], long int deg)
{
    long int i, ret_i;
    double abs_anxn[QDSIZE];
    cqdfloat xn, anxn;

    // xn := x
    rcqd_set(&xn, x);

    ret_i = 0;
    //*absmax_anxn = fabs(coef[0]);
    rcqd_abs_qd(absmax_anxn, get_cqdpoly_i(poly, 0));
 
    for(i = 1; i <= poly->deg; i++)
    {
        //anxn = coef[i] * xn;
        rcqd_mul(&anxn, &xn, get_cqdpoly_i(poly, i));
        //abs_anxn = fabs(anxn);
        rcqd_abs_qd(abs_anxn, &anxn);
        //if(*absmax_anxn < abs_anxn)
        if(rqd_cmp(absmax_anxn, abs_anxn) < 0)
        {
            ret_i = i;
            //*absmax_anxn = abs_anxn;
            rqd_set(absmax_anxn, abs_anxn);
        }
        //xn *= x;
        rcqd_mul(&xn, &xn, x);
    }

    return ret_i;
}


// get_plus_arg
// return arg(x) in [0, 2 PI]
void qd_get_plus_arg(double ret[QDSIZE], cqdfloat *x)
{
    double pi2[QDSIZE];

    // pi2 := 2 * PI
    rqd_const_pi(pi2);
    rqd_mul_ui(pi2, pi2, 2UL);

    //double ret = carg(x);
    rcqd_arg(ret, x);

    //if(ret < 0)
    if(rqd_cmp_ui(ret, 0UL) < 0)
    {
        //ret = 2.0 * M_PI + ret;
        rqd_add(ret, pi2, ret);
    }

    //return ret;
    return;
}

// get_nearest_int
void qd_get_nearest_int(double ret[QDSIZE], double real_x[QDSIZE])
{
    //long int int_x_floor, int_x_ceil;
    //double dist_x_floor, dist_x_ceil;
    double int_x_floor[QDSIZE], int_x_ceil[QDSIZE];
    double dist_x_floor[QDSIZE], dist_x_ceil[QDSIZE];

    //int_x_floor = (long int)floor(real_x);
    rqd_func_mpfr(int_x_floor, mpfr_rint_floor, real_x); // _dd);
    //int_x_ceil  = (long int)ceil (real_x);
    rqd_func_mpfr(int_x_ceil, mpfr_rint_ceil, real_x); // _dd);

    //dist_x_floor = fabs(real_x - (double)int_x_floor);
    rqd_sub(dist_x_floor, real_x, int_x_floor); rqd_abs(dist_x_floor, dist_x_floor);
    //dist_x_ceil  = fabs(real_x - (double)int_x_ceil);
    rqd_sub(dist_x_ceil, real_x, int_x_ceil); rqd_abs(dist_x_ceil, dist_x_ceil);

    //if(dist_x_floor < dist_x_ceil) return int_x_floor;
    if(rqd_cmp(dist_x_floor, dist_x_ceil) < 0) rqd_set(ret, int_x_floor);
    //else return int_x_ceil;
    else rqd_set(ret, int_x_ceil);
}

// get_min_branch
void qd_get_min_branch(cqdfloat *ret, cqdfloat *x, double mu[QDSIZE], CQDVector coef, long int i_num, long int i_den)
{
    long int j = i_den;
    double phi[QDSIZE], psi[QDSIZE], pi2[QDSIZE], real_j[QDSIZE];
    cqdfloat ctmp, ctmp2;
    double tmp[QDSIZE], tmp1[QDSIZE], tmp2[QDSIZE];

    //if(cabs(coef[j]) == 0.0) return ret;
    rcqd_abs_qd(tmp, get_cqdvector_i(coef, j));
    if(rqd_cmp_ui(tmp, 0UL) == 0)
    {
        rcqd_set_ui_ui(ret, 0UL, 0UL);
        return;
    }

    rqd_const_pi(pi2);
    rqd_mul_ui(pi2, pi2, 2UL);

    //phi = dget_plus_arg(x) / pi2;
    qd_get_plus_arg(phi, x);
    rqd_div(phi, phi, pi2);

    //psi = dget_plus_arg(-coef[i_num] / coef[j]) / pi2;
    rcqd_div(&ctmp, get_cqdvector_i(coef, i_num), get_cqdvector_i(coef, j));
    rcqd_neg(&ctmp, &ctmp);
    qd_get_plus_arg(psi, &ctmp);
    rqd_div(psi, psi, pi2);

    //real_j = (double)dget_nearest_int((double)j * (0.5 - phi) - psi);
    rqd_set_ui(tmp, 1UL);
    rqd_div_ui(tmp, tmp, 2UL); // tmp := 0.5
    rqd_sub(tmp, tmp, phi); // tmp := 0.5 - phi
    rqd_mul_ui(tmp, tmp, (unsigned long int)j); // tmp := j * (0.5 - phi)
    rqd_sub(tmp, tmp, psi); // tmp := tmp - psi
    qd_get_nearest_int(real_j, tmp);

    //ret = pow(cabs(mu * coef[i_num] / coef[j]), 1.0 / (double)j) * cexp(pi2 * I * ((psi + real_j) / (double)j));
    
    // Part 1: pow(cabs(mu * coef[i_num] / coef[j]), 1.0 / (double)j)
    rcqd_div(&ctmp, get_cqdvector_i(coef, i_num), get_cqdvector_i(coef, j));
    rcqd_mul_qd(&ctmp, &ctmp, mu);
    rcqd_abs_qd(tmp, &ctmp);
    rqd_set_ui(tmp1, 1UL);
    rqd_div_ui(tmp1, tmp1, (unsigned long int)j);
    rqd_pow_mpfr(tmp, tmp, tmp1);

    // Part 2: cexp(pi2 * I * ((psi + real_j) / (double)j))
    rqd_add(tmp1, psi, real_j);
    rqd_div_ui(tmp1, tmp1, (unsigned long int)j);
    rqd_mul(tmp1, pi2, tmp1);
    rqd_set_ui(tmp2, 0UL);
    rcqd_set_qd_qd(&ctmp, tmp2, tmp1);
    //rcqd_exp(&ctmp2, &ctmp2);
    rcqd_func_mpc(&ctmp2, mpc_exp, &ctmp);
    // Multiply: ret = tmp * ctmp2
    rcqd_mul_qd(ret, &ctmp2, tmp);

    return;
}

// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int qd_hirano(cqdfloat *ret, cqdfloat *init_x, QDPoly poly, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
{
    int stop_flag = 0;
    long int i, j, max_j, times, absmin_j, deg = poly->deg;
    cqdfloat old_x, new_x, pn_x_d, ctmp, ctmp2;
    //cqdfloat *in_coef, *d;
    CQDVector in_coef, d;
    double abs_pn_new_x[QDSIZE], absmax_anxn[QDSIZE], absmin_d[QDSIZE], abs_d[QDSIZE];
    double mu[QDSIZE], beta[QDSIZE], lambda[QDSIZE];
    double tmp[QDSIZE], tmp1[QDSIZE], tmp2[QDSIZE];

    // from Sugihara & Murota
    rqd_set_ui(beta, 3UL); rqd_div_ui(beta, beta, 4UL);
    rqd_set_ui(lambda, 2UL);

    // coef of p(x + d)
    //in_coef = (double _Complex *)calloc(deg + 1, sizeof(double _Complex));
    //d = (double _Complex *)calloc(deg, sizeof(double _Complex));
    in_coef = init_cqdvector(deg + 1);
    //printf("init d=%ld, deg = %ld\n", deg);
    d = init_cqdvector(deg + 1);

    //old_x = init_x;
    rcqd_set(&old_x, init_x);

    // Main loop
    for(times = 0; times < maxtimes; times++)
    {
        qd_coef_horner(in_coef, &old_x, poly); // coef, deg);
        //mu = 1.0;
        rqd_set_ui(mu, 1UL);

        //mpfr_printf("%5ld %25.17RNe + %25.17RNe * I -> %15.7RNe + %15.7RNe * I\n", times, rcqd_realref(old_x), rcqd_imagref(old_x), rcqd_realref(get_cqdvector_i(in_coef, 0)), rcqd_imagref(get_cqdvector_i(in_coef, 0)));

        // (1)
        //d[0] = 0.0 + 0.0 * I;
        rcqd_set_ui_ui(&ctmp, 0UL, 0UL);
        set_cqdvector_i(d, 0, &ctmp);
        //if(fabs(in_coef[1]) != 0.0)
        rcqd_abs_qd(tmp, get_cqdvector_i(in_coef, 1));
        if(rqd_cmp_ui(tmp, 0UL) != 0)
        {
            //d[0] = -in_coef[0] / in_coef[1];
            rcqd_div(&ctmp, get_cqdvector_i(in_coef, 0), get_cqdvector_i(in_coef, 1));
            rcqd_neg(&ctmp, &ctmp);
            set_cqdvector_i(d, 0, &ctmp);
        }
        //mpfr_printf("d[0] = %25.17RNe + %25.17RNe * I\n", rcqd_realref(get_cqdvector_i(d, 0)), rcqd_imagref(get_cqdvector_i(d, 0)));

        //new_x = old_x + d[0];
        rcqd_add(&new_x, &old_x, get_cqdvector_i(d, 0));
        //pn_x_d = dhorner(new_x, coef, deg);
        //rcqd_horner(pn_x_d, new_x, poly); // coef, deg);
        ceval_qdpoly(&pn_x_d, poly, &new_x);

        rcqd_abs_qd(tmp, &pn_x_d);
        rcqd_abs_qd(tmp2, get_cqdvector_i(in_coef, 0));
        rqd_mul(tmp2, tmp2, beta);
        //if(cabs(pn_x_d) <= beta * cabs(in_coef[0]))
        if(rqd_cmp(tmp, tmp2) <= 0)
        {
            // old_x = new_x;
            //printf("%ld: new_x := c0 + c1 * old_x\n", times);
            rcqd_set(&old_x, &new_x);
        }
        else
        {
            //printf("%ld: new_x := c0 + c? * old_x^?\n", times);

            for(max_j = 2; max_j <= deg; max_j++)
            {
                //printf("max_j = %ld\n", max_j);
                //mpfr_printf("d[max_j] = %25.17RNe + %25.17RNe * I\n", rcqd_realref(get_cqdvector_i(d, max_j)), rcqd_imagref(get_cqdvector_i(d, max_j)));
                //mu /= lambda;
                rqd_div(mu, mu, lambda);
                //d[max_j] = dget_min_branch(old_x, mu, in_coef, 0, max_j);
                qd_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, max_j);
                set_cqdvector_i(d, max_j, &ctmp2);
                //mpfr_printf("d[max_j = %ld] = %25.17RNe + %25.17RNe\n", max_j, rcqd_realref(get_cqdvector_i(d, max_j)), rcqd_imagref(get_cqdvector_i(d, max_j)));
                //absmin_d = cabs(d[max_j]);
                rcqd_abs_qd(absmin_d, get_cqdvector_i(d, max_j));
                absmin_j = max_j;
                for(j = 0; j < max_j; j++)
                {
                    //printf("%ld ", j);
                    //d[j] = dget_min_branch(old_x, mu, in_coef, 0, j + 1) / pow(lambda, 1.0 / (double)(j + 1));
                    qd_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, j + 1);
                    set_cqdvector_i(d, j, &ctmp2);
                    rqd_set_ui(tmp, 1UL);
                    rqd_div_ui(tmp, tmp, (unsigned long)(j + 1));
                    rqd_pow_mpfr(tmp1, lambda, tmp);
                    //mpfr_printf("d[%ld] = %25.17RNe + %25.17RNe * I, lambda^%ld = %25.17RNe\n", j, rcqd_realref(get_cqdvector_i(d, j)), rcqd_imagref(get_cqdvector_i(d,j)), j + 1, tmp1);
                    rcqd_div_qd(&ctmp2, get_cqdvector_i(d, j), tmp1);
                    set_cqdvector_i(d, j, &ctmp2);
                    //abs_d = cabs(d[j]);
                    rcqd_abs_qd(abs_d, get_cqdvector_i(d, j));
                    if(rqd_cmp(abs_d, absmin_d) < 0) //abs_d < absmin_d)
                    {
                        // absmin_d = abs_d;
                        rqd_set(absmin_d, abs_d);
                        absmin_j = j;
                    }
                }
                //printf("absmin_j = %ld\n", absmin_j);
                //new_x = old_x + d[absmin_j];
                rcqd_add(&new_x, &old_x, get_cqdvector_i(d, absmin_j));
                //mpfr_printf("d[%ld] = %25.17RNe + %25.17RNe * I\n", absmin_j, rcqd_realref(get_cqdvector_i(d, absmin_j)), rcqd_imagref(get_cqdvector_i(d, absmin_j)));
                //mpfr_printf("old_x =  %25.17RNe + %25.17RNe * I\n", rcqd_realref(old_x), rcqd_imagref(old_x));
                //mpfr_printf("new_x =  %25.17RNe + %25.17RNe * I\n", rcqd_realref(new_x), rcqd_imagref(new_x));
                //mpfr_printf("pn_x_d = %25.17RNe + %25.17RNe * I\n", rcqd_realref(new_x), rcqd_imagref(new_x));
                //pn_x_d = dhorner(new_x, coef, deg);
                //rcqd_horner(pn_x_d, new_x, poly); // coef, deg);
                ceval_qdpoly(&pn_x_d, poly, &new_x);

                rcqd_abs_qd(tmp, &pn_x_d);// pn_x_d := abs(p(new_x))
                rqd_set_ui(tmp2, 1UL); // tmp2 := 1
                rqd_sub(tmp2, tmp2, beta); // tmp2 := 1 - beta
                rqd_mul(tmp2, tmp2, mu); // tmp2 := (1 - beta) * mu
                rqd_sub_ui(tmp2, tmp2, 1UL); // tmp2 := tmp2 - 1 = (1 - beta) * mu - 1
                rqd_neg(tmp2, tmp2); // tmp2 := -tmp2 = 1 - (1 - beta) * mu
                rcqd_abs_qd(tmp1, get_cqdvector_i(in_coef, 0)); // tmp1 := abs(in_coef[0])
                rqd_mul(tmp2, tmp2, tmp1); // tmp2 := tmp2 * tmp1
                //mpfr_printf("tmp = %25.17RNe <= tmp2 = %25.17RNe ? \n", tmp, tmp2);
                //if(cabs(pn_x_d) <= (1.0 - (1.0 - beta) * mu) * cabs(in_coef[0]))
                if(rqd_cmp(tmp, tmp2) <= 0)
                {
                    //old_x = new_x;
                    //printf("%ld - %ld: new_x := old_x + c%ld * x^%ld\n", times, max_j, absmin_j, absmin_j);
                    rcqd_set(&old_x, &new_x);
                    break;
                }        
            }
        }

        // check stopping rule
        //dabsmax_poly(&absmax_anxn, new_x, coef, deg);
        absmax_qdpoly(absmax_anxn, &new_x, poly); // coef, deg);
        //abs_pn_new_x = cabs(dhorner(new_x, coef, deg));
        //rcqd_horner(ctmp, new_x, poly); // coef, deg);
        ceval_qdpoly(&ctmp, poly, &new_x); // coef, deg);
        rcqd_abs_qd(abs_pn_new_x, &ctmp);
        rqd_mul(tmp, absmax_anxn, reps);
        rqd_add(tmp, tmp, aeps);
        //if(abs_pn_new_x <= absmax_anxn * reps + aeps)
        if(rqd_cmp(abs_pn_new_x, tmp) <= 0)
            break;
    }

    free_cqdvector(in_coef);
    free_cqdvector(d);

    //*ret = new_x;
    rcqd_set(ret, &new_x);

    return times;
}

// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int cqd_hirano(cqdfloat *ret, cqdfloat *init_x, CQDPoly poly, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
{
    int stop_flag = 0;
    long int i, j, max_j, times, absmin_j, deg = poly->deg;
    cqdfloat old_x, new_x, pn_x_d, ctmp, ctmp2;
    //cqdfloat *in_coef, *d;
    CQDVector in_coef, d;
    double abs_pn_new_x[QDSIZE], absmax_anxn[QDSIZE], absmin_d[QDSIZE], abs_d[QDSIZE];
    double mu[QDSIZE], beta[QDSIZE], lambda[QDSIZE];
    double tmp[QDSIZE], tmp1[QDSIZE], tmp2[QDSIZE];

    // from Sugihara & Murota
    rqd_set_ui(beta, 3UL); rqd_div_ui(beta, beta, 4UL);
    rqd_set_ui(lambda, 2UL);

    // coef of p(x + d)
    //in_coef = (double _Complex *)calloc(deg + 1, sizeof(double _Complex));
    //d = (double _Complex *)calloc(deg, sizeof(double _Complex));
    in_coef = init_cqdvector(deg + 1);
    //printf("init d=%ld, deg = %ld\n", deg);
    d = init_cqdvector(deg + 1);

    //old_x = init_x;
    rcqd_set(&old_x, init_x);

    // Main loop
    for(times = 0; times < maxtimes; times++)
    {
        cqd_coef_horner(in_coef, &old_x, poly); // coef, deg);
        //mu = 1.0;
        rqd_set_ui(mu, 1UL);

        //mpfr_printf("%5ld %25.17RNe + %25.17RNe * I -> %15.7RNe + %15.7RNe * I\n", times, rcqd_realref(old_x), rcqd_imagref(old_x), rcqd_realref(get_cqdvector_i(in_coef, 0)), rcqd_imagref(get_cqdvector_i(in_coef, 0)));

        // (1)
        //d[0] = 0.0 + 0.0 * I;
        rcqd_set_ui_ui(&ctmp, 0UL, 0UL);
        set_cqdvector_i(d, 0, &ctmp);
        //if(fabs(in_coef[1]) != 0.0)
        rcqd_abs_qd(tmp, get_cqdvector_i(in_coef, 1));
        if(rqd_cmp_ui(tmp, 0UL) != 0)
        {
            //d[0] = -in_coef[0] / in_coef[1];
            rcqd_div(&ctmp, get_cqdvector_i(in_coef, 0), get_cqdvector_i(in_coef, 1));
            rcqd_neg(&ctmp, &ctmp);
            set_cqdvector_i(d, 0, &ctmp);
        }
        //mpfr_printf("d[0] = %25.17RNe + %25.17RNe * I\n", rcqd_realref(get_cqdvector_i(d, 0)), rcqd_imagref(get_cqdvector_i(d, 0)));

        //new_x = old_x + d[0];
        rcqd_add(&new_x, &old_x, get_cqdvector_i(d, 0));
        //pn_x_d = dhorner(new_x, coef, deg);
        //rcqd_horner(pn_x_d, new_x, poly); // coef, deg);
        eval_cqdpoly(&pn_x_d, poly, &new_x);

        rcqd_abs_qd(tmp, &pn_x_d);
        rcqd_abs_qd(tmp2, get_cqdvector_i(in_coef, 0));
        rqd_mul(tmp2, tmp2, beta);
        //if(cabs(pn_x_d) <= beta * cabs(in_coef[0]))
        if(rqd_cmp(tmp, tmp2) <= 0)
        {
            // old_x = new_x;
            //printf("%ld: new_x := c0 + c1 * old_x\n", times);
            rcqd_set(&old_x, &new_x);
        }
        else
        {
            //printf("%ld: new_x := c0 + c? * old_x^?\n", times);

            for(max_j = 2; max_j <= deg; max_j++)
            {
                //printf("max_j = %ld\n", max_j);
                //mpfr_printf("d[max_j] = %25.17RNe + %25.17RNe * I\n", rcqd_realref(get_cqdvector_i(d, max_j)), rcqd_imagref(get_cqdvector_i(d, max_j)));
                //mu /= lambda;
                rqd_div(mu, mu, lambda);
                //d[max_j] = dget_min_branch(old_x, mu, in_coef, 0, max_j);
                qd_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, max_j);
                set_cqdvector_i(d, max_j, &ctmp2);
                //mpfr_printf("d[max_j = %ld] = %25.17RNe + %25.17RNe\n", max_j, rcqd_realref(get_cqdvector_i(d, max_j)), rcqd_imagref(get_cqdvector_i(d, max_j)));
                //absmin_d = cabs(d[max_j]);
                rcqd_abs_qd(absmin_d, get_cqdvector_i(d, max_j));
                absmin_j = max_j;
                for(j = 0; j < max_j; j++)
                {
                    //printf("%ld ", j);
                    //d[j] = dget_min_branch(old_x, mu, in_coef, 0, j + 1) / pow(lambda, 1.0 / (double)(j + 1));
                    qd_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, j + 1);
                    set_cqdvector_i(d, j, &ctmp2);
                    rqd_set_ui(tmp, 1UL);
                    rqd_div_ui(tmp, tmp, (unsigned long)(j + 1));
                    rqd_pow_mpfr(tmp1, lambda, tmp);
                    //mpfr_printf("d[%ld] = %25.17RNe + %25.17RNe * I, lambda^%ld = %25.17RNe\n", j, rcqd_realref(get_cqdvector_i(d, j)), rcqd_imagref(get_cqdvector_i(d,j)), j + 1, tmp1);
                    rcqd_div_qd(&ctmp2, get_cqdvector_i(d, j), tmp1);
                    set_cqdvector_i(d, j, &ctmp2);
                    //abs_d = cabs(d[j]);
                    rcqd_abs_qd(abs_d, get_cqdvector_i(d, j));
                    if(rqd_cmp(abs_d, absmin_d) < 0) //abs_d < absmin_d)
                    {
                        // absmin_d = abs_d;
                        rqd_set(absmin_d, abs_d);
                        absmin_j = j;
                    }
                }
                //printf("absmin_j = %ld\n", absmin_j);
                //new_x = old_x + d[absmin_j];
                rcqd_add(&new_x, &old_x, get_cqdvector_i(d, absmin_j));
                //mpfr_printf("d[%ld] = %25.17RNe + %25.17RNe * I\n", absmin_j, rcqd_realref(get_cqdvector_i(d, absmin_j)), rcqd_imagref(get_cqdvector_i(d, absmin_j)));
                //mpfr_printf("old_x =  %25.17RNe + %25.17RNe * I\n", rcqd_realref(old_x), rcqd_imagref(old_x));
                //mpfr_printf("new_x =  %25.17RNe + %25.17RNe * I\n", rcqd_realref(new_x), rcqd_imagref(new_x));
                //mpfr_printf("pn_x_d = %25.17RNe + %25.17RNe * I\n", rcqd_realref(new_x), rcqd_imagref(new_x));
                //pn_x_d = dhorner(new_x, coef, deg);
                //rcqd_horner(pn_x_d, new_x, poly); // coef, deg);
                eval_cqdpoly(&pn_x_d, poly, &new_x);

                rcqd_abs_qd(tmp, &pn_x_d);// pn_x_d := abs(p(new_x))
                rqd_set_ui(tmp2, 1UL); // tmp2 := 1
                rqd_sub(tmp2, tmp2, beta); // tmp2 := 1 - beta
                rqd_mul(tmp2, tmp2, mu); // tmp2 := (1 - beta) * mu
                rqd_sub_ui(tmp2, tmp2, 1UL); // tmp2 := tmp2 - 1 = (1 - beta) * mu - 1
                rqd_neg(tmp2, tmp2); // tmp2 := -tmp2 = 1 - (1 - beta) * mu
                rcqd_abs_qd(tmp1, get_cqdvector_i(in_coef, 0)); // tmp1 := abs(in_coef[0])
                rqd_mul(tmp2, tmp2, tmp1); // tmp2 := tmp2 * tmp1
                //mpfr_printf("tmp = %25.17RNe <= tmp2 = %25.17RNe ? \n", tmp, tmp2);
                //if(cabs(pn_x_d) <= (1.0 - (1.0 - beta) * mu) * cabs(in_coef[0]))
                if(rqd_cmp(tmp, tmp2) <= 0)
                {
                    //old_x = new_x;
                    //printf("%ld - %ld: new_x := old_x + c%ld * x^%ld\n", times, max_j, absmin_j, absmin_j);
                    rcqd_set(&old_x, &new_x);
                    break;
                }        
            }
        }

        // check stopping rule
        //dabsmax_poly(&absmax_anxn, new_x, coef, deg);
        absmax_cqdpoly(absmax_anxn, &new_x, poly); // coef, deg);
        //abs_pn_new_x = cabs(dhorner(new_x, coef, deg));
        //rcqd_horner(ctmp, new_x, poly); // coef, deg);
        eval_cqdpoly(&ctmp, poly, &new_x); // coef, deg);
        rcqd_abs_qd(abs_pn_new_x, &ctmp);
        rqd_mul(tmp, absmax_anxn, reps);
        rqd_add(tmp, tmp, aeps);
        //if(abs_pn_new_x <= absmax_anxn * reps + aeps)
        if(rqd_cmp(abs_pn_new_x, tmp) <= 0)
            break;
    }

    free_cqdvector(in_coef);
    free_cqdvector(d);

    //*ret = new_x;
    rcqd_set(ret, &new_x);

    return times;
}

// Deflation of polynomial
// p(x) / (x - r)
void deflation_cqdpoly(CQDPoly ret, CQDPoly pol, cqdfloat *root)
{
    long int i;
    //double _Complex ret;
    cqdfloat ctmp;
    CQDPoly in_ret;
    
    if(pol->deg < 1)
        return;

    in_ret = init_set_cqdpoly(ret);

    // clear
    set0_cqdpoly(in_ret);

    //ret[deg - 1] := coef[deg];
    set_cqdpoly_i(in_ret, pol->deg - 1, get_cqdpoly_i(pol, pol->deg));
    for(i = pol->deg - 2; i >= 0; i--) // i >= 0; i--)
    {
        rcqd_mul(&ctmp, get_cqdpoly_i(in_ret, i + 1), root);
        rcqd_add(&ctmp, &ctmp, get_cqdpoly_i(pol, i + 1));
        //ret[i] := ret[i + 1] * root + coef[i + 1];
        set_cqdpoly_i(in_ret, i, &ctmp);
    }

    setdegree_cqdpoly(in_ret);
    subst_cqdpoly(ret, in_ret);

    free_cqdpoly(in_ret);
    //return ret;
    return;
}
#endif // USE_QDLINEAR
