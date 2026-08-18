/********************************************************************************/
/* dd_hirano.c: Robust solver for Algebraic Equations using Hirano method       */
/* copyright (c) 2025 Tomonori Kouya                                            */
/*                                                                              */
/* Ver. 0.2 2025-02-07: DD precision implementation                             */
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

#include "ddlinear.h"
#include "poly.h"

// --------------------------
// DD precision
// --------------------------

// Missing function in rcdd.h
// ret := a_real + a_imag * I (where a_imag is usually 0)
static inline void rcdd_set_dd_ui(cddfloat *ret, double a_real[DDSIZE], unsigned long a_imag)
{
    rdd_set(ret->val_re, a_real);
    rdd_set_ui(ret->val_im, a_imag);
}

// Horner method for real polynomial with complex evaluation point
// return p(x) where p has real coefficients
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
void dd_horner(cddfloat *ret, cddfloat *x, DDPoly poly)
{
    long int i;
    cddfloat tmp;

    // ret = coef[deg] + 0i
    rcdd_set_dd_ui(ret, get_ddpoly_i(poly, poly->deg), 0UL);
    
    for(i = poly->deg - 1; i >= 0; i--)
    {
        // ret = ret * x + coef[i]
        rcdd_mul(ret, ret, x);
        rcdd_set_dd_ui(&tmp, get_ddpoly_i(poly, i), 0UL);
        rcdd_add(ret, ret, &tmp);
    }
    
    return;
}

// Horner method for complex polynomial
// return p(x)
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
void cdd_horner(cddfloat *ret, cddfloat *x, CDDPoly poly)
{
    long int i;

    // ret = coef[deg]
    rcdd_set(ret, get_cddpoly_i(poly, poly->deg));
    
    for(i = poly->deg - 1; i >= 0; i--)
    {
        // ret = ret * x + coef[i]
        rcdd_mul(ret, ret, x);
        rcdd_add(ret, ret, get_cddpoly_i(poly, i));
    }
    
    return;
}

// Coefficients of p(x + d) for real polynomial
// based on Horner method
//void dd_coef_horner(cddfloat ret_coef, cddfloat *x, DDPoly poly)
void dd_coef_horner(cddfloat ret_coef[], cddfloat *x, DDPoly poly)
{
    long int l, i;
    cddfloat ctmp, ctmp2;

    // Initial setting: convert real coefficients to complex
    for(i = 0; i <= poly->deg; i++)
    {
        rcdd_set_dd_ui(&ctmp2, get_ddpoly_i(poly, i), 0UL);
        rcdd_set(&ret_coef[i], &ctmp2);
    }

    // ret_coef[i] := p^(l)(x) / l!
    for(l = 0; l <= poly->deg; l++)
    {
        for(i = poly->deg - 1; i >= l; i--)
        {
            // ret_coef[i] = x * ret_coef[i + 1] + ret_coef[i]
            rcdd_mul(&ctmp, &ret_coef[i + 1], x);
            rcdd_add(&ctmp2, &ctmp, &ret_coef[i]);
            rcdd_set(&ret_coef[i], &ctmp2);
        }
    }
}

// Coefficients of p(x + d) for complex polynomial
// based on Horner method
void cdd_coef_horner(cddfloat ret_coef[], cddfloat *x, CDDPoly poly)
{
    long int l, i;
    cddfloat ctmp, ctmp2;

    // Initial setting
    for(i = 0; i <= poly->deg; i++)
    {
        rcdd_set(&ctmp2, get_cddpoly_i(poly, i));
        rcdd_set(&ret_coef[i], &ctmp2);
    }

    // ret_coef[i] := p^(l)(x) / l!
    for(l = 0; l <= poly->deg; l++)
    {
        for(i = poly->deg - 1; i >= l; i--)
        {
            // ret_coef[i] = x * ret_coef[i + 1] + ret_coef[i]
            rcdd_mul(&ctmp, &ret_coef[i + 1], x);
            rcdd_add(&ctmp2, &ctmp, &ret_coef[i]);
            rcdd_set(&ret_coef[i], &ctmp2);
        }
    }
}

// Return j and max|a_j x^j| from p(x) for real polynomial
long int absmax_ddpoly(double absmax_anxn[DDSIZE], cddfloat *x, DDPoly poly)
{
    long int i, ret_i;
    double abs_anxn[DDSIZE];
    cddfloat xn, anxn, tmp_c;

    // xn := x
    rcdd_set(&xn, x);

    ret_i = 0;
    // Convert real coefficient to complex and get absolute value
    rcdd_set_dd_ui(&tmp_c, get_ddpoly_i(poly, 0), 0UL);
    rcdd_abs_dd(absmax_anxn, &tmp_c);
 
    for(i = 1; i <= poly->deg; i++)
    {
        // anxn = coef[i] * xn
        rcdd_set_dd_ui(&tmp_c, get_ddpoly_i(poly, i), 0UL);
        rcdd_mul(&anxn, &xn, &tmp_c);
        
        // abs_anxn = |anxn|
        rcdd_abs_dd(abs_anxn, &anxn);
        
        if(rdd_cmp(absmax_anxn, abs_anxn) < 0)
        {
            ret_i = i;
            rdd_set(absmax_anxn, abs_anxn);
        }
        
        // xn *= x
        rcdd_mul(&xn, &xn, x);
    }

    return ret_i;
}

// Return j and max|a_j x^j| from p(x) for complex polynomial
long int absmax_cddpoly(double absmax_anxn[DDSIZE], cddfloat *x, CDDPoly poly)
{
    long int i, ret_i;
    double abs_anxn[DDSIZE];
    cddfloat xn, anxn;

    // xn := x
    rcdd_set(&xn, x);

    ret_i = 0;
    rcdd_abs_dd(absmax_anxn, get_cddpoly_i(poly, 0));
 
    for(i = 1; i <= poly->deg; i++)
    {
        // anxn = coef[i] * xn
        rcdd_mul(&anxn, &xn, get_cddpoly_i(poly, i));
        
        // abs_anxn = |anxn|
        rcdd_abs_dd(abs_anxn, &anxn);
        
        if(rdd_cmp(absmax_anxn, abs_anxn) < 0)
        {
            ret_i = i;
            rdd_set(absmax_anxn, abs_anxn);
        }
        
        // xn *= x
        rcdd_mul(&xn, &xn, x);
    }

    return ret_i;
}

// Get argument in [0, 2*PI]
void dd_get_plus_arg(double ret[DDSIZE], cddfloat *x)
{
    double pi2[DDSIZE];

    // pi2 := 2 * PI
    rdd_const_pi(pi2);
    rdd_mul_ui(pi2, pi2, 2UL);

    // ret = arg(x)
    rcdd_arg(ret, x);

    // if(ret < 0) ret = 2*PI + ret
    if(rdd_cmp_ui(ret, 0UL) < 0)
    {
        rdd_add(ret, ret, pi2);
    }
    
    return;
}

// Get nearest integer
void dd_get_nearest_int(double ret[DDSIZE], double real_x[DDSIZE])
{
    double int_x_floor[DDSIZE], int_x_ceil[DDSIZE];
    double dist_x_floor[DDSIZE], dist_x_ceil[DDSIZE];

    // int_x_floor = floor(real_x)
    rdd_func_mpfr(int_x_floor, mpfr_rint_floor, real_x);
    
    // int_x_ceil = ceil(real_x)
    rdd_func_mpfr(int_x_ceil, mpfr_rint_ceil, real_x);

    // dist_x_floor = |real_x - int_x_floor|
    rdd_sub(dist_x_floor, real_x, int_x_floor);
    rdd_abs(dist_x_floor, dist_x_floor);
    
    // dist_x_ceil = |real_x - int_x_ceil|
    rdd_sub(dist_x_ceil, real_x, int_x_ceil);
    rdd_abs(dist_x_ceil, dist_x_ceil);

    // Return the nearest
    if(rdd_cmp(dist_x_floor, dist_x_ceil) < 0)
        rdd_set(ret, int_x_floor);
    else
        rdd_set(ret, int_x_ceil);
}

// Get minimum branch
void dd_get_min_branch(cddfloat *ret, cddfloat *x, double mu[DDSIZE], 
                       cddfloat coef[], long int i_num, long int i_den)
{
    long int j = i_den;
    double phi[DDSIZE], psi[DDSIZE], pi2[DDSIZE], real_j[DDSIZE];
    double tmp[DDSIZE], tmp2[DDSIZE], pow_tmp[DDSIZE];
    cddfloat ctmp, ctmp2;

    // Check if coef[j] == 0
    rcdd_abs_dd(tmp, &coef[j]);
    if(rdd_cmp_ui(tmp, 0UL) == 0)
    {
        //rcdd_set_ui_ui(ret, 0UL, 0UL);
        rcdd_set_d_d(ret, FP_NAN, FP_NAN);
        return;
    }

    // pi2 := 2 * PI
    rdd_const_pi(pi2);
    rdd_mul_ui(pi2, pi2, 2UL);

    // phi = arg(x) / (2*PI)
    dd_get_plus_arg(phi, x);
    rdd_div(phi, phi, pi2);
    
    // psi = arg(-coef[i_num] / coef[j]) / (2*PI)
    rcdd_div(&ctmp, &coef[i_num], &coef[j]); // &coef, i_num), &coef, j));
    rcdd_neg(&ctmp, &ctmp);
    dd_get_plus_arg(psi, &ctmp);
    rdd_div(psi, psi, pi2);

    // real_j = nearest_int(j * (0.5 - phi) - psi)
    rdd_set_ui(real_j, 1UL);
    rdd_div_ui(real_j, real_j, 2UL);      // real_j = 0.5
    rdd_sub(real_j, real_j, phi);          // real_j = 0.5 - phi
    rdd_mul_ui(real_j, real_j, (unsigned long)j);  // real_j = j * (0.5 - phi)
    rdd_sub(real_j, real_j, psi);          // real_j = j * (0.5 - phi) - psi
    dd_get_nearest_int(real_j, real_j);

    // Compute the branch:
    // ret = pow(|mu * coef[i_num] / coef[j]|, 1/j) 
    //       * exp(2*PI*I * ((psi + real_j) / j))

    // Part 1: pow_tmp = pow(|mu * coef[i_num] / coef[j]|, 1/j)
    
    // ctmp = mu * coef[i_num]
    rcdd_mul_dd(&ctmp, &coef[i_num], mu);
    
    // ctmp = ctmp / coef[j]
    rcdd_div(&ctmp, &ctmp, &coef[j]);
    
    // tmp = |ctmp|
    rcdd_abs_dd(tmp, &ctmp);
    
    // tmp2 = 1/j
    rdd_set_ui(tmp2, 1UL);
    rdd_div_ui(tmp2, tmp2, (unsigned long)j);
    
    // pow_tmp = pow(tmp, tmp2)
    rdd_pow_mpfr(pow_tmp, tmp, tmp2);

    // Part 2: exp(2*PI*I * ((psi + real_j) / j))
    
    // tmp = psi + real_j
    rdd_add(tmp, psi, real_j);
    
    // tmp = (psi + real_j) / j
    rdd_div_ui(tmp, tmp, (unsigned long)j);
    
    // tmp = tmp * 2*PI
    rdd_mul(tmp, tmp, pi2);
    
    // ctmp = 0 + tmp*I
    rdd_set_ui(tmp2, 0UL);
    rcdd_set_dd_dd(&ctmp, tmp2, tmp);
    
    // ret = exp(ctmp)
    rcdd_func_mpc(ret, mpc_exp, &ctmp);
    
    // ret = ret * pow_tmp
    rcdd_mul_dd(ret, ret, pow_tmp);

    return;
}

// Hirano method for real polynomial
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int dd_hirano(cddfloat *ret, cddfloat *init_x, DDPoly poly, 
                   double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
{
    long int i, j, max_j, times, absmin_j, deg = poly->deg;
    cddfloat old_x, new_x, pn_x_d, ctmp, ctmp2;
    cddfloat *in_coef, *d;
    double abs_pn_new_x[DDSIZE], absmax_anxn[DDSIZE], absmin_d[DDSIZE], abs_d[DDSIZE];
    double mu[DDSIZE], beta[DDSIZE], lambda[DDSIZE];
    double tmp[DDSIZE], tmp1[DDSIZE], tmp2[DDSIZE];

    // Parameters from Sugiura & Murota
    rdd_set_ui(beta, 3UL);
    rdd_div_ui(beta, beta, 4UL);        // beta = 3/4
    rdd_set_ui(lambda, 2UL);             // lambda = 2

    // Allocate coefficient arrays
    in_coef = (cddfloat *)calloc(deg + 1, sizeof(cddfloat));
    d = (cddfloat *)calloc(deg + 1, sizeof(cddfloat));

    // old_x = init_x
    rcdd_set(&old_x, init_x);

    // Main iteration loop
    for(times = 0; times < maxtimes; times++)
    {
        // Compute coefficients of p(old_x + d)
        dd_coef_horner(in_coef, &old_x, poly);
        
        // mu = 1
        rdd_set_ui(mu, 1UL);

        // Step (1): Try Newton's method
        // d[0] = 0
        rcdd_set_ui_ui(&ctmp, 0UL, 0UL);
        rcdd_set(&d[0], &ctmp);
        
        // if |in_coef[1]| != 0
        rcdd_abs_dd(tmp, &in_coef[1]);
        if(rdd_cmp_ui(tmp, 0UL) != 0)
        {
            // d[0] = -in_coef[0] / in_coef[1]
            rcdd_div(&ctmp, &in_coef[0], &in_coef[1]);
            rcdd_neg(&ctmp, &ctmp);
            rcdd_set(&d[0], &ctmp);
        }

        // new_x = old_x + d[0]
        rcdd_add(&new_x, &old_x, &d[0]);
        
        // pn_x_d = p(new_x)
        dd_horner(&pn_x_d, &new_x, poly);

        // Check if |p(new_x)| <= beta * |in_coef[0]|
        rcdd_abs_dd(tmp, &pn_x_d);
        rcdd_abs_dd(tmp2, &in_coef[0]);
        rdd_mul(tmp2, tmp2, beta);
        
        if(rdd_cmp(tmp, tmp2) <= 0)
        {
            // Accept Newton step
            rcdd_set(&old_x, &new_x);
        }
        else
        {
            // Try higher order corrections
            for(max_j = 2; max_j <= deg; max_j++)
            {
                // mu /= lambda
                rdd_div(mu, mu, lambda);
                
                // d[max_j] = get_min_branch(old_x, mu, in_coef, 0, max_j)
                dd_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, max_j);
                rcdd_set(&d[max_j], &ctmp2);
                
                // absmin_d = |d[max_j]|
                rcdd_abs_dd(absmin_d, &d[max_j]);
                absmin_j = max_j;
                
                // Find j with minimum |d[j] / lambda^(1/(j+1))|
                for(j = 0; j < max_j; j++)
                {
                    // d[j] = get_min_branch(old_x, mu, in_coef, 0, j+1)
                    dd_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, j + 1);
                    rcdd_set(&d[j], &ctmp2);
                    
                    // tmp1 = lambda^(1/(j+1))
                    rdd_set_ui(tmp, 1UL);
                    rdd_div_ui(tmp, tmp, (unsigned long)(j + 1));
                    rdd_pow_mpfr(tmp1, lambda, tmp);
                    
                    // d[j] = d[j] / tmp1
                    rcdd_div_dd(&ctmp2, &d[j], tmp1);
                    rcdd_set(&d[j], &ctmp2);
                    
                    // abs_d = |d[j]|
                    rcdd_abs_dd(abs_d, &d[j]);
                    
                    if(rdd_cmp(abs_d, absmin_d) < 0)
                    {
                        rdd_set(absmin_d, abs_d);
                        absmin_j = j;
                    }
                }
                
                // new_x = old_x + d[absmin_j]
                rcdd_add(&new_x, &old_x, &d[absmin_j]);
                
                // pn_x_d = p(new_x)
                dd_horner(&pn_x_d, &new_x, poly);
                
                // Check if |p(new_x)| <= (1 - (1-beta)*mu) * |in_coef[0]|
                rcdd_abs_dd(tmp, &pn_x_d);
                rdd_set_ui(tmp2, 1UL);
                rdd_sub(tmp2, tmp2, beta);        // tmp2 = 1 - beta
                rdd_mul(tmp2, tmp2, mu);          // tmp2 = (1-beta) * mu
                rdd_sub_ui(tmp2, tmp2, 1UL);      // tmp2 = (1-beta)*mu - 1
                rdd_neg(tmp2, tmp2);              // tmp2 = 1 - (1-beta)*mu
                rcdd_abs_dd(tmp1, &in_coef[0]);
                rdd_mul(tmp2, tmp2, tmp1);        // tmp2 = (1 - (1-beta)*mu) * |in_coef[0]|
                
                if(rdd_cmp(tmp, tmp2) <= 0)
                {
                    rcdd_set(&old_x, &new_x);
                    break;
                }
            }
        }

        // Check stopping criterion
        absmax_ddpoly(absmax_anxn, &new_x, poly);
        dd_horner(&ctmp, &new_x, poly);
        rcdd_abs_dd(abs_pn_new_x, &ctmp);
        
        // Check if |p(new_x)| <= max|a_j*x^j| * reps + aeps
        rdd_mul(tmp, absmax_anxn, reps);
        rdd_add(tmp, tmp, aeps);
        
        if(rdd_cmp(abs_pn_new_x, tmp) <= 0)
            break;
    }

    free(in_coef);
    free(d);

    rcdd_set(ret, &new_x);

    return times;
}

// Hirano method for complex polynomial
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int cdd_hirano(cddfloat *ret, cddfloat *init_x, CDDPoly poly, 
                    double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
{
    long int i, j, max_j, times, absmin_j, deg = poly->deg;
    cddfloat old_x, new_x, pn_x_d, ctmp, ctmp2;
    cddfloat *in_coef, *d;
    double abs_pn_new_x[DDSIZE], absmax_anxn[DDSIZE], absmin_d[DDSIZE], abs_d[DDSIZE];
    double mu[DDSIZE], beta[DDSIZE], lambda[DDSIZE];
    double tmp[DDSIZE], tmp1[DDSIZE], tmp2[DDSIZE];

    // Parameters from Sugiura & Murota
    rdd_set_ui(beta, 3UL);
    rdd_div_ui(beta, beta, 4UL);        // beta = 3/4
    rdd_set_ui(lambda, 2UL);             // lambda = 2

    // Allocate coefficient arrays
    in_coef = (cddfloat *)calloc(deg + 1, sizeof(cddfloat));
    d = (cddfloat *)calloc(deg + 1, sizeof(cddfloat)); // (cddfloat *)calloc((deg + 1);

    // old_x = init_x
    rcdd_set(&old_x, init_x);

    // Main iteration loop
    for(times = 0; times < maxtimes; times++)
    {
        // Compute coefficients of p(old_x + d)
        cdd_coef_horner(in_coef, &old_x, poly);
        
        // mu = 1
        rdd_set_ui(mu, 1UL);

        // Step (1): Try Newton's method
        // d[0] = 0
        rcdd_set_ui_ui(&ctmp, 0UL, 0UL);
        rcdd_set(&d[0], &ctmp);
        
        // if |in_coef[1]| != 0
        rcdd_abs_dd(tmp, &in_coef[1]);
        if(rdd_cmp_ui(tmp, 0UL) != 0)
        {
            // d[0] = -in_coef[0] / in_coef[1]
            rcdd_div(&ctmp, &in_coef[0], &in_coef[1]);
            rcdd_neg(&ctmp, &ctmp);
            rcdd_set(&d[0], &ctmp);
        }

        // new_x = old_x + d[0]
        rcdd_add(&new_x, &old_x, &d[0]);
        
        // pn_x_d = p(new_x)
        cdd_horner(&pn_x_d, &new_x, poly);

        // Check if |p(new_x)| <= beta * |in_coef[0]|
        rcdd_abs_dd(tmp, &pn_x_d);
        rcdd_abs_dd(tmp2, &in_coef[0]);
        rdd_mul(tmp2, tmp2, beta);
        
        if(rdd_cmp(tmp, tmp2) <= 0)
        {
            // Accept Newton step
            rcdd_set(&old_x, &new_x);
        }
        else
        {
            // Try higher order corrections
            for(max_j = 2; max_j <= deg; max_j++)
            {
                // mu /= lambda
                rdd_div(mu, mu, lambda);
                
                // d[max_j] = get_min_branch(old_x, mu, in_coef, 0, max_j)
                dd_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, max_j);
                rcdd_set(&d[max_j], &ctmp2);
                
                // absmin_d = |d[max_j]|
                rcdd_abs_dd(absmin_d, &d[max_j]);
                absmin_j = max_j;
                
                // Find j with minimum |d[j] / lambda^(1/(j+1))|
                for(j = 0; j < max_j; j++)
                {
                    // d[j] = get_min_branch(old_x, mu, in_coef, 0, j+1)
                    dd_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, j + 1);
                    rcdd_set(&d[j], &ctmp2);
                    
                    // tmp1 = lambda^(1/(j+1))
                    rdd_set_ui(tmp, 1UL);
                    rdd_div_ui(tmp, tmp, (unsigned long)(j + 1));
                    rdd_pow_mpfr(tmp1, lambda, tmp);
                    
                    // d[j] = d[j] / tmp1
                    rcdd_div_dd(&ctmp2, &d[j], tmp1);
                    rcdd_set(&d[j], &ctmp2);
                    
                    // abs_d = |d[j]|
                    rcdd_abs_dd(abs_d, &d[j]);
                    
                    if(rdd_cmp(abs_d, absmin_d) < 0)
                    {
                        rdd_set(absmin_d, abs_d);
                        absmin_j = j;
                    }
                }
                
                // new_x = old_x + d[absmin_j]
                rcdd_add(&new_x, &old_x, &d[absmin_j]);
                
                // pn_x_d = p(new_x)
                cdd_horner(&pn_x_d, &new_x, poly);
                
                // Check if |p(new_x)| <= (1 - (1-beta)*mu) * |in_coef[0]|
                rcdd_abs_dd(tmp, &pn_x_d);
                rdd_set_ui(tmp2, 1UL);
                rdd_sub(tmp2, tmp2, beta);        // tmp2 = 1 - beta
                rdd_mul(tmp2, tmp2, mu);          // tmp2 = (1-beta) * mu
                rdd_sub_ui(tmp2, tmp2, 1UL);      // tmp2 = (1-beta)*mu - 1
                rdd_neg(tmp2, tmp2);              // tmp2 = 1 - (1-beta)*mu
                rcdd_abs_dd(tmp1, &in_coef[0]);
                rdd_mul(tmp2, tmp2, tmp1);        // tmp2 = (1 - (1-beta)*mu) * |in_coef[0]|
                
                if(rdd_cmp(tmp, tmp2) <= 0)
                {
                    rcdd_set(&old_x, &new_x);
                    break;
                }
            }
        }

        // Check stopping criterion
        absmax_cddpoly(absmax_anxn, &new_x, poly);
        cdd_horner(&ctmp, &new_x, poly);
        rcdd_abs_dd(abs_pn_new_x, &ctmp);
        
        // Check if |p(new_x)| <= max|a_j*x^j| * reps + aeps
        rdd_mul(tmp, absmax_anxn, reps);
        rdd_add(tmp, tmp, aeps);
        
        if(rdd_cmp(abs_pn_new_x, tmp) <= 0)
            break;
    }

    free(in_coef);
    free(d);

    rcdd_set(ret, &new_x);

    return times;
}

// Deflation of polynomial: p(x) / (x - r)
void deflation_cddpoly(CDDPoly ret, CDDPoly pol, cddfloat *root)
{
    long int i;
    cddfloat ctmp;
    CDDPoly in_ret;
    
    if(pol->deg < 1)
        return;

    in_ret = init_set_cddpoly(ret);

    // Clear coefficients
    set0_cddpoly(in_ret);

    // ret[deg-1] = coef[deg]
    set_cddpoly_i(in_ret, pol->deg - 1, get_cddpoly_i(pol, pol->deg));
    
    // Synthetic division
    for(i = pol->deg - 2; i >= 0; i--)
    {
        // ret[i] = ret[i+1] * root + coef[i+1]
        rcdd_mul(&ctmp, get_cddpoly_i(in_ret, i + 1), root);
        rcdd_add(&ctmp, &ctmp, get_cddpoly_i(pol, i + 1));
        set_cddpoly_i(in_ret, i, &ctmp);
    }

    setdegree_cddpoly(in_ret);
    subst_cddpoly(ret, in_ret);

    free_cddpoly(in_ret);
    
    return;
}

