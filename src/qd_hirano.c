/********************************************************************************/
/* qd_hirano.c: Robust solver for Algebraic Equations using Hirano method       */
/* copyright (c) 2025 Tomonori Kouya                                            */
/*                                                                              */
/* Ver. 0.2 2025-02-07: QD precision implementation                             */
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

#include "qdlinear.h"
#include "poly.h"

// --------------------------
// QD precision
// --------------------------

// Missing function in rcqd.h
// ret := a_real + a_imag * I (where a_imag is usually 0)
static inline void rcqd_set_qd_ui(cqdfloat *ret, double a_real[QDSIZE], unsigned long a_imag)
{
    rqd_set(ret->val_re, a_real);
    rqd_set_ui(ret->val_im, a_imag);
}

// Horner method for real polynomial with complex evaluation point
// return p(x) where p has real coefficients
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
void qd_horner(cqdfloat *ret, cqdfloat *x, QDPoly poly)
{
    long int i;
    cqdfloat tmp;

    // ret = coef[deg] + 0i
    rcqd_set_qd_ui(ret, get_qdpoly_i(poly, poly->deg), 0UL);
    
    for(i = poly->deg - 1; i >= 0; i--)
    {
        // ret = ret * x + coef[i]
        rcqd_mul(ret, ret, x);
        rcqd_set_qd_ui(&tmp, get_qdpoly_i(poly, i), 0UL);
        rcqd_add(ret, ret, &tmp);
    }
    
    return;
}

// Horner method for complex polynomial
// return p(x)
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
void cqd_horner(cqdfloat *ret, cqdfloat *x, CQDPoly poly)
{
    long int i;

    // ret = coef[deg]
    rcqd_set(ret, get_cqdpoly_i(poly, poly->deg));
    
    for(i = poly->deg - 1; i >= 0; i--)
    {
        // ret = ret * x + coef[i]
        rcqd_mul(ret, ret, x);
        rcqd_add(ret, ret, get_cqdpoly_i(poly, i));
    }
    
    return;
}

// Coefficients of p(x + d) for real polynomial
// based on Horner method
//void qd_coef_horner(cqdfloat ret_coef, cqdfloat *x, QDPoly poly)
void qd_coef_horner(cqdfloat ret_coef[], cqdfloat *x, QDPoly poly)
{
    long int l, i;
    cqdfloat ctmp, ctmp2;

    // Initial setting: convert real coefficients to complex
    for(i = 0; i <= poly->deg; i++)
    {
        rcqd_set_qd_ui(&ctmp2, get_qdpoly_i(poly, i), 0UL);
        rcqd_set(&ret_coef[i], &ctmp2);
    }

    // ret_coef[i] := p^(l)(x) / l!
    for(l = 0; l <= poly->deg; l++)
    {
        for(i = poly->deg - 1; i >= l; i--)
        {
            // ret_coef[i] = x * ret_coef[i + 1] + ret_coef[i]
            rcqd_mul(&ctmp, &ret_coef[i + 1], x);
            rcqd_add(&ctmp2, &ctmp, &ret_coef[i]);
            rcqd_set(&ret_coef[i], &ctmp2);
        }
    }
}

// Coefficients of p(x + d) for complex polynomial
// based on Horner method
void cqd_coef_horner(cqdfloat ret_coef[], cqdfloat *x, CQDPoly poly)
{
    long int l, i;
    cqdfloat ctmp, ctmp2;

    // Initial setting
    for(i = 0; i <= poly->deg; i++)
    {
        rcqd_set(&ctmp2, get_cqdpoly_i(poly, i));
        rcqd_set(&ret_coef[i], &ctmp2);
    }

    // ret_coef[i] := p^(l)(x) / l!
    for(l = 0; l <= poly->deg; l++)
    {
        for(i = poly->deg - 1; i >= l; i--)
        {
            // ret_coef[i] = x * ret_coef[i + 1] + ret_coef[i]
            rcqd_mul(&ctmp, &ret_coef[i + 1], x);
            rcqd_add(&ctmp2, &ctmp, &ret_coef[i]);
            rcqd_set(&ret_coef[i], &ctmp2);
        }
    }
}

// Return j and max|a_j x^j| from p(x) for real polynomial
long int absmax_qdpoly(double absmax_anxn[QDSIZE], cqdfloat *x, QDPoly poly)
{
    long int i, ret_i;
    double abs_anxn[QDSIZE];
    cqdfloat xn, anxn, tmp_c;

    // xn := x
    rcqd_set(&xn, x);

    ret_i = 0;
    // Convert real coefficient to complex and get absolute value
    rcqd_set_qd_ui(&tmp_c, get_qdpoly_i(poly, 0), 0UL);
    rcqd_abs_qd(absmax_anxn, &tmp_c);
 
    for(i = 1; i <= poly->deg; i++)
    {
        // anxn = coef[i] * xn
        rcqd_set_qd_ui(&tmp_c, get_qdpoly_i(poly, i), 0UL);
        rcqd_mul(&anxn, &xn, &tmp_c);
        
        // abs_anxn = |anxn|
        rcqd_abs_qd(abs_anxn, &anxn);
        
        if(rqd_cmp(absmax_anxn, abs_anxn) < 0)
        {
            ret_i = i;
            rqd_set(absmax_anxn, abs_anxn);
        }
        
        // xn *= x
        rcqd_mul(&xn, &xn, x);
    }

    return ret_i;
}

// Return j and max|a_j x^j| from p(x) for complex polynomial
long int absmax_cqdpoly(double absmax_anxn[QDSIZE], cqdfloat *x, CQDPoly poly)
{
    long int i, ret_i;
    double abs_anxn[QDSIZE];
    cqdfloat xn, anxn;

    // xn := x
    rcqd_set(&xn, x);

    ret_i = 0;
    rcqd_abs_qd(absmax_anxn, get_cqdpoly_i(poly, 0));
 
    for(i = 1; i <= poly->deg; i++)
    {
        // anxn = coef[i] * xn
        rcqd_mul(&anxn, &xn, get_cqdpoly_i(poly, i));
        
        // abs_anxn = |anxn|
        rcqd_abs_qd(abs_anxn, &anxn);
        
        if(rqd_cmp(absmax_anxn, abs_anxn) < 0)
        {
            ret_i = i;
            rqd_set(absmax_anxn, abs_anxn);
        }
        
        // xn *= x
        rcqd_mul(&xn, &xn, x);
    }

    return ret_i;
}

// Get argument in [0, 2*PI]
void qd_get_plus_arg(double ret[QDSIZE], cqdfloat *x)
{
    double pi2[QDSIZE];

    // pi2 := 2 * PI
    rqd_const_pi(pi2);
    rqd_mul_ui(pi2, pi2, 2UL);

    // ret = arg(x)
    rcqd_arg(ret, x);

    // if(ret < 0) ret = 2*PI + ret
    if(rqd_cmp_ui(ret, 0UL) < 0)
    {
        rqd_add(ret, ret, pi2);
    }
    
    return;
}

// Get nearest integer
void qd_get_nearest_int(double ret[QDSIZE], double real_x[QDSIZE])
{
    double int_x_floor[QDSIZE], int_x_ceil[QDSIZE];
    double dist_x_floor[QDSIZE], dist_x_ceil[QDSIZE];

    // int_x_floor = floor(real_x)
    rqd_func_mpfr(int_x_floor, mpfr_rint_floor, real_x);
    
    // int_x_ceil = ceil(real_x)
    rqd_func_mpfr(int_x_ceil, mpfr_rint_ceil, real_x);

    // dist_x_floor = |real_x - int_x_floor|
    rqd_sub(dist_x_floor, real_x, int_x_floor);
    rqd_abs(dist_x_floor, dist_x_floor);
    
    // dist_x_ceil = |real_x - int_x_ceil|
    rqd_sub(dist_x_ceil, real_x, int_x_ceil);
    rqd_abs(dist_x_ceil, dist_x_ceil);

    // Return the nearest
    if(rqd_cmp(dist_x_floor, dist_x_ceil) < 0)
        rqd_set(ret, int_x_floor);
    else
        rqd_set(ret, int_x_ceil);
}

// Get minimum branch
void qd_get_min_branch(cqdfloat *ret, cqdfloat *x, double mu[QDSIZE], 
                       cqdfloat coef[], long int i_num, long int i_den)
{
    long int j = i_den;
    double phi[QDSIZE], psi[QDSIZE], pi2[QDSIZE], real_j[QDSIZE];
    double tmp[QDSIZE], tmp2[QDSIZE], pow_tmp[QDSIZE];
    cqdfloat ctmp, ctmp2;

    // Check if coef[j] == 0
    rcqd_abs_qd(tmp, &coef[j]);
    if(rqd_cmp_ui(tmp, 0UL) == 0)
    {
        //rcqd_set_ui_ui(ret, 0UL, 0UL);
        rcqd_set_d_d(ret, FP_NAN, FP_NAN);
        return;
    }

    // pi2 := 2 * PI
    rqd_const_pi(pi2);
    rqd_mul_ui(pi2, pi2, 2UL);

    // phi = arg(x) / (2*PI)
    qd_get_plus_arg(phi, x);
    rqd_div(phi, phi, pi2);
    
    // psi = arg(-coef[i_num] / coef[j]) / (2*PI)
    rcqd_div(&ctmp, &coef[i_num], &coef[j]); // &coef, i_num), &coef, j));
    rcqd_neg(&ctmp, &ctmp);
    qd_get_plus_arg(psi, &ctmp);
    rqd_div(psi, psi, pi2);

    // real_j = nearest_int(j * (0.5 - phi) - psi)
    rqd_set_ui(real_j, 1UL);
    rqd_div_ui(real_j, real_j, 2UL);      // real_j = 0.5
    rqd_sub(real_j, real_j, phi);          // real_j = 0.5 - phi
    rqd_mul_ui(real_j, real_j, (unsigned long)j);  // real_j = j * (0.5 - phi)
    rqd_sub(real_j, real_j, psi);          // real_j = j * (0.5 - phi) - psi
    qd_get_nearest_int(real_j, real_j);

    // Compute the branch:
    // ret = pow(|mu * coef[i_num] / coef[j]|, 1/j) 
    //       * exp(2*PI*I * ((psi + real_j) / j))

    // Part 1: pow_tmp = pow(|mu * coef[i_num] / coef[j]|, 1/j)
    
    // ctmp = mu * coef[i_num]
    rcqd_mul_qd(&ctmp, &coef[i_num], mu);
    
    // ctmp = ctmp / coef[j]
    rcqd_div(&ctmp, &ctmp, &coef[j]);
    
    // tmp = |ctmp|
    rcqd_abs_qd(tmp, &ctmp);
    
    // tmp2 = 1/j
    rqd_set_ui(tmp2, 1UL);
    rqd_div_ui(tmp2, tmp2, (unsigned long)j);
    
    // pow_tmp = pow(tmp, tmp2)
    rqd_pow_mpfr(pow_tmp, tmp, tmp2);

    // Part 2: exp(2*PI*I * ((psi + real_j) / j))
    
    // tmp = psi + real_j
    rqd_add(tmp, psi, real_j);
    
    // tmp = (psi + real_j) / j
    rqd_div_ui(tmp, tmp, (unsigned long)j);
    
    // tmp = tmp * 2*PI
    rqd_mul(tmp, tmp, pi2);
    
    // ctmp = 0 + tmp*I
    rqd_set_ui(tmp2, 0UL);
    rcqd_set_qd_qd(&ctmp, tmp2, tmp);
    
    // ret = exp(ctmp)
    rcqd_func_mpc(ret, mpc_exp, &ctmp);
    
    // ret = ret * pow_tmp
    rcqd_mul_qd(ret, ret, pow_tmp);

    return;
}

// Hirano method for real polynomial
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int qd_hirano(cqdfloat *ret, cqdfloat *init_x, QDPoly poly, 
                   double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
{
    long int i, j, max_j, times, absmin_j, deg = poly->deg;
    cqdfloat old_x, new_x, pn_x_d, ctmp, ctmp2;
    cqdfloat *in_coef, *d;
    double abs_pn_new_x[QDSIZE], absmax_anxn[QDSIZE], absmin_d[QDSIZE], abs_d[QDSIZE];
    double mu[QDSIZE], beta[QDSIZE], lambda[QDSIZE];
    double tmp[QDSIZE], tmp1[QDSIZE], tmp2[QDSIZE];

    // Parameters from Sugiura & Murota
    rqd_set_ui(beta, 3UL);
    rqd_div_ui(beta, beta, 4UL);        // beta = 3/4
    rqd_set_ui(lambda, 2UL);             // lambda = 2

    // Allocate coefficient arrays
    in_coef = (cqdfloat *)calloc(deg + 1, sizeof(cqdfloat));
    d = (cqdfloat *)calloc(deg + 1, sizeof(cqdfloat));

    // old_x = init_x
    rcqd_set(&old_x, init_x);

    // Main iteration loop
    for(times = 0; times < maxtimes; times++)
    {
        // Compute coefficients of p(old_x + d)
        qd_coef_horner(in_coef, &old_x, poly);
        
        // mu = 1
        rqd_set_ui(mu, 1UL);

        // Step (1): Try Newton's method
        // d[0] = 0
        rcqd_set_ui_ui(&ctmp, 0UL, 0UL);
        rcqd_set(&d[0], &ctmp);
        
        // if |in_coef[1]| != 0
        rcqd_abs_qd(tmp, &in_coef[1]);
        if(rqd_cmp_ui(tmp, 0UL) != 0)
        {
            // d[0] = -in_coef[0] / in_coef[1]
            rcqd_div(&ctmp, &in_coef[0], &in_coef[1]);
            rcqd_neg(&ctmp, &ctmp);
            rcqd_set(&d[0], &ctmp);
        }

        // new_x = old_x + d[0]
        rcqd_add(&new_x, &old_x, &d[0]);
        
        // pn_x_d = p(new_x)
        qd_horner(&pn_x_d, &new_x, poly);

        // Check if |p(new_x)| <= beta * |in_coef[0]|
        rcqd_abs_qd(tmp, &pn_x_d);
        rcqd_abs_qd(tmp2, &in_coef[0]);
        rqd_mul(tmp2, tmp2, beta);
        
        if(rqd_cmp(tmp, tmp2) <= 0)
        {
            // Accept Newton step
            rcqd_set(&old_x, &new_x);
        }
        else
        {
            // Try higher order corrections
            for(max_j = 2; max_j <= deg; max_j++)
            {
                // mu /= lambda
                rqd_div(mu, mu, lambda);
                
                // d[max_j] = get_min_branch(old_x, mu, in_coef, 0, max_j)
                qd_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, max_j);
                rcqd_set(&d[max_j], &ctmp2);
                
                // absmin_d = |d[max_j]|
                rcqd_abs_qd(absmin_d, &d[max_j]);
                absmin_j = max_j;
                
                // Find j with minimum |d[j] / lambda^(1/(j+1))|
                for(j = 0; j < max_j; j++)
                {
                    // d[j] = get_min_branch(old_x, mu, in_coef, 0, j+1)
                    qd_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, j + 1);
                    rcqd_set(&d[j], &ctmp2);
                    
                    // tmp1 = lambda^(1/(j+1))
                    rqd_set_ui(tmp, 1UL);
                    rqd_div_ui(tmp, tmp, (unsigned long)(j + 1));
                    rqd_pow_mpfr(tmp1, lambda, tmp);
                    
                    // d[j] = d[j] / tmp1
                    rcqd_div_qd(&ctmp2, &d[j], tmp1);
                    rcqd_set(&d[j], &ctmp2);
                    
                    // abs_d = |d[j]|
                    rcqd_abs_qd(abs_d, &d[j]);
                    
                    if(rqd_cmp(abs_d, absmin_d) < 0)
                    {
                        rqd_set(absmin_d, abs_d);
                        absmin_j = j;
                    }
                }
                
                // new_x = old_x + d[absmin_j]
                rcqd_add(&new_x, &old_x, &d[absmin_j]);
                
                // pn_x_d = p(new_x)
                qd_horner(&pn_x_d, &new_x, poly);
                
                // Check if |p(new_x)| <= (1 - (1-beta)*mu) * |in_coef[0]|
                rcqd_abs_qd(tmp, &pn_x_d);
                rqd_set_ui(tmp2, 1UL);
                rqd_sub(tmp2, tmp2, beta);        // tmp2 = 1 - beta
                rqd_mul(tmp2, tmp2, mu);          // tmp2 = (1-beta) * mu
                rqd_sub_ui(tmp2, tmp2, 1UL);      // tmp2 = (1-beta)*mu - 1
                rqd_neg(tmp2, tmp2);              // tmp2 = 1 - (1-beta)*mu
                rcqd_abs_qd(tmp1, &in_coef[0]);
                rqd_mul(tmp2, tmp2, tmp1);        // tmp2 = (1 - (1-beta)*mu) * |in_coef[0]|
                
                if(rqd_cmp(tmp, tmp2) <= 0)
                {
                    rcqd_set(&old_x, &new_x);
                    break;
                }
            }
        }

        // Check stopping criterion
        absmax_qdpoly(absmax_anxn, &new_x, poly);
        qd_horner(&ctmp, &new_x, poly);
        rcqd_abs_qd(abs_pn_new_x, &ctmp);
        
        // Check if |p(new_x)| <= max|a_j*x^j| * reps + aeps
        rqd_mul(tmp, absmax_anxn, reps);
        rqd_add(tmp, tmp, aeps);
        
        if(rqd_cmp(abs_pn_new_x, tmp) <= 0)
            break;
    }

    free(in_coef);
    free(d);

    rcqd_set(ret, &new_x);

    return times;
}

// Hirano method for complex polynomial
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int cqd_hirano(cqdfloat *ret, cqdfloat *init_x, CQDPoly poly, 
                    double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
{
    long int i, j, max_j, times, absmin_j, deg = poly->deg;
    cqdfloat old_x, new_x, pn_x_d, ctmp, ctmp2;
    cqdfloat *in_coef, *d;
    double abs_pn_new_x[QDSIZE], absmax_anxn[QDSIZE], absmin_d[QDSIZE], abs_d[QDSIZE];
    double mu[QDSIZE], beta[QDSIZE], lambda[QDSIZE];
    double tmp[QDSIZE], tmp1[QDSIZE], tmp2[QDSIZE];

    // Parameters from Sugiura & Murota
    rqd_set_ui(beta, 3UL);
    rqd_div_ui(beta, beta, 4UL);        // beta = 3/4
    rqd_set_ui(lambda, 2UL);             // lambda = 2

    // Allocate coefficient arrays
    in_coef = (cqdfloat *)calloc(deg + 1, sizeof(cqdfloat));
    d = (cqdfloat *)calloc(deg + 1, sizeof(cqdfloat)); // (cqdfloat *)calloc((deg + 1);

    // old_x = init_x
    rcqd_set(&old_x, init_x);

    // Main iteration loop
    for(times = 0; times < maxtimes; times++)
    {
        // Compute coefficients of p(old_x + d)
        cqd_coef_horner(in_coef, &old_x, poly);
        
        // mu = 1
        rqd_set_ui(mu, 1UL);

        // Step (1): Try Newton's method
        // d[0] = 0
        rcqd_set_ui_ui(&ctmp, 0UL, 0UL);
        rcqd_set(&d[0], &ctmp);
        
        // if |in_coef[1]| != 0
        rcqd_abs_qd(tmp, &in_coef[1]);
        if(rqd_cmp_ui(tmp, 0UL) != 0)
        {
            // d[0] = -in_coef[0] / in_coef[1]
            rcqd_div(&ctmp, &in_coef[0], &in_coef[1]);
            rcqd_neg(&ctmp, &ctmp);
            rcqd_set(&d[0], &ctmp);
        }

        // new_x = old_x + d[0]
        rcqd_add(&new_x, &old_x, &d[0]);
        
        // pn_x_d = p(new_x)
        cqd_horner(&pn_x_d, &new_x, poly);

        // Check if |p(new_x)| <= beta * |in_coef[0]|
        rcqd_abs_qd(tmp, &pn_x_d);
        rcqd_abs_qd(tmp2, &in_coef[0]);
        rqd_mul(tmp2, tmp2, beta);
        
        if(rqd_cmp(tmp, tmp2) <= 0)
        {
            // Accept Newton step
            rcqd_set(&old_x, &new_x);
        }
        else
        {
            // Try higher order corrections
            for(max_j = 2; max_j <= deg; max_j++)
            {
                // mu /= lambda
                rqd_div(mu, mu, lambda);
                
                // d[max_j] = get_min_branch(old_x, mu, in_coef, 0, max_j)
                qd_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, max_j);
                rcqd_set(&d[max_j], &ctmp2);
                
                // absmin_d = |d[max_j]|
                rcqd_abs_qd(absmin_d, &d[max_j]);
                absmin_j = max_j;
                
                // Find j with minimum |d[j] / lambda^(1/(j+1))|
                for(j = 0; j < max_j; j++)
                {
                    // d[j] = get_min_branch(old_x, mu, in_coef, 0, j+1)
                    qd_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, j + 1);
                    rcqd_set(&d[j], &ctmp2);
                    
                    // tmp1 = lambda^(1/(j+1))
                    rqd_set_ui(tmp, 1UL);
                    rqd_div_ui(tmp, tmp, (unsigned long)(j + 1));
                    rqd_pow_mpfr(tmp1, lambda, tmp);
                    
                    // d[j] = d[j] / tmp1
                    rcqd_div_qd(&ctmp2, &d[j], tmp1);
                    rcqd_set(&d[j], &ctmp2);
                    
                    // abs_d = |d[j]|
                    rcqd_abs_qd(abs_d, &d[j]);
                    
                    if(rqd_cmp(abs_d, absmin_d) < 0)
                    {
                        rqd_set(absmin_d, abs_d);
                        absmin_j = j;
                    }
                }
                
                // new_x = old_x + d[absmin_j]
                rcqd_add(&new_x, &old_x, &d[absmin_j]);
                
                // pn_x_d = p(new_x)
                cqd_horner(&pn_x_d, &new_x, poly);
                
                // Check if |p(new_x)| <= (1 - (1-beta)*mu) * |in_coef[0]|
                rcqd_abs_qd(tmp, &pn_x_d);
                rqd_set_ui(tmp2, 1UL);
                rqd_sub(tmp2, tmp2, beta);        // tmp2 = 1 - beta
                rqd_mul(tmp2, tmp2, mu);          // tmp2 = (1-beta) * mu
                rqd_sub_ui(tmp2, tmp2, 1UL);      // tmp2 = (1-beta)*mu - 1
                rqd_neg(tmp2, tmp2);              // tmp2 = 1 - (1-beta)*mu
                rcqd_abs_qd(tmp1, &in_coef[0]);
                rqd_mul(tmp2, tmp2, tmp1);        // tmp2 = (1 - (1-beta)*mu) * |in_coef[0]|
                
                if(rqd_cmp(tmp, tmp2) <= 0)
                {
                    rcqd_set(&old_x, &new_x);
                    break;
                }
            }
        }

        // Check stopping criterion
        absmax_cqdpoly(absmax_anxn, &new_x, poly);
        cqd_horner(&ctmp, &new_x, poly);
        rcqd_abs_qd(abs_pn_new_x, &ctmp);
        
        // Check if |p(new_x)| <= max|a_j*x^j| * reps + aeps
        rqd_mul(tmp, absmax_anxn, reps);
        rqd_add(tmp, tmp, aeps);
        
        if(rqd_cmp(abs_pn_new_x, tmp) <= 0)
            break;
    }

    free(in_coef);
    free(d);

    rcqd_set(ret, &new_x);

    return times;
}

// Deflation of polynomial: p(x) / (x - r)
void deflation_cqdpoly(CQDPoly ret, CQDPoly pol, cqdfloat *root)
{
    long int i;
    cqdfloat ctmp;
    CQDPoly in_ret;
    
    if(pol->deg < 1)
        return;

    in_ret = init_set_cqdpoly(ret);

    // Clear coefficients
    set0_cqdpoly(in_ret);

    // ret[deg-1] = coef[deg]
    set_cqdpoly_i(in_ret, pol->deg - 1, get_cqdpoly_i(pol, pol->deg));
    
    // Synthetic division
    for(i = pol->deg - 2; i >= 0; i--)
    {
        // ret[i] = ret[i+1] * root + coef[i+1]
        rcqd_mul(&ctmp, get_cqdpoly_i(in_ret, i + 1), root);
        rcqd_add(&ctmp, &ctmp, get_cqdpoly_i(pol, i + 1));
        set_cqdpoly_i(in_ret, i, &ctmp);
    }

    setdegree_cqdpoly(in_ret);
    subst_cqdpoly(ret, in_ret);

    free_cqdpoly(in_ret);
    
    return;
}

