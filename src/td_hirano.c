/********************************************************************************/
/* td_hirano.c: Robust solver for Algebraic Equations using Hirano method       */
/* copyright (c) 2025 Tomonori Kouya                                            */
/*                                                                              */
/* Ver. 0.2 2025-02-08: TD precision implementation                             */
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

#include "tdlinear.h"
#include "poly.h"

// --------------------------
// TD precision
// --------------------------

// Missing function in rctd.h
// ret := a_real + a_imag * I (where a_imag is usually 0)
static inline void rctd_set_td_ui(ctdfloat *ret, double a_real[TDSIZE], unsigned long a_imag)
{
    rtd_set(ret->val_re, a_real);
    rtd_set_ui(ret->val_im, a_imag);
}

// Horner method for real polynomial with complex evaluation point
// return p(x) where p has real coefficients
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
void td_horner(ctdfloat *ret, ctdfloat *x, TDPoly poly)
{
    long int i;
    ctdfloat tmp;

    // ret = coef[deg] + 0i
    rctd_set_td_ui(ret, get_tdpoly_i(poly, poly->deg), 0UL);
    
    for(i = poly->deg - 1; i >= 0; i--)
    {
        // ret = ret * x + coef[i]
        rctd_mul(ret, ret, x);
        rctd_set_td_ui(&tmp, get_tdpoly_i(poly, i), 0UL);
        rctd_add(ret, ret, &tmp);
    }
    
    return;
}

// Horner method for complex polynomial
// return p(x)
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
void ctd_horner(ctdfloat *ret, ctdfloat *x, CTDPoly poly)
{
    long int i;
    ctdfloat in_ret;

    // ret = coef[deg]
    rctd_set(ret, get_ctdpoly_i(poly, poly->deg));
    
    for(i = poly->deg - 1; i >= 0; i--)
    {
        // ret = ret * x + coef[i]
        rctd_mul(&in_ret, ret, x);
        rctd_add(ret, &in_ret, get_ctdpoly_i(poly, i));
    }
    
    return;
}

// Coefficients of p(x + d) for real polynomial
// based on Horner method
//void td_coef_horner(ctdfloat ret_coef, ctdfloat *x, TDPoly poly)
void td_coef_horner(ctdfloat ret_coef[], ctdfloat *x, TDPoly poly)
{
    long int l, i;
    ctdfloat ctmp, ctmp2;

    // Initial setting: convert real coefficients to complex
    for(i = 0; i <= poly->deg; i++)
    {
        rctd_set_td_ui(&ctmp2, get_tdpoly_i(poly, i), 0UL);
        rctd_set(&ret_coef[i], &ctmp2);
    }

    // ret_coef[i] := p^(l)(x) / l!
    for(l = 0; l <= poly->deg; l++)
    {
        for(i = poly->deg - 1; i >= l; i--)
        {
            // ret_coef[i] = x * ret_coef[i + 1] + ret_coef[i]
            rctd_mul(&ctmp, &ret_coef[i + 1], x);
            rctd_add(&ctmp2, &ctmp, &ret_coef[i]);
            rctd_set(&ret_coef[i], &ctmp2);
        }
    }
}

// Coefficients of p(x + d) for complex polynomial
// based on Horner method
void ctd_coef_horner(ctdfloat ret_coef[], ctdfloat *x, CTDPoly poly)
{
    long int l, i;
    ctdfloat ctmp, ctmp2;

    // Initial setting
    for(i = 0; i <= poly->deg; i++)
    {
        rctd_set(&ctmp2, get_ctdpoly_i(poly, i));
        rctd_set(&ret_coef[i], &ctmp2);
    }

    // ret_coef[i] := p^(l)(x) / l!
    for(l = 0; l <= poly->deg; l++)
    {
        for(i = poly->deg - 1; i >= l; i--)
        {
            // ret_coef[i] = x * ret_coef[i + 1] + ret_coef[i]
            rctd_mul(&ctmp, &ret_coef[i + 1], x);
            rctd_add(&ctmp2, &ctmp, &ret_coef[i]);
            rctd_set(&ret_coef[i], &ctmp2);
        }
    }
}

// Return j and max|a_j x^j| from p(x) for real polynomial
long int absmax_tdpoly(double absmax_anxn[TDSIZE], ctdfloat *x, TDPoly poly)
{
    long int i, ret_i;
    double abs_anxn[TDSIZE];
    ctdfloat xn, anxn, tmp_c;

    // xn := x
    rctd_set(&xn, x);

    ret_i = 0;
    // Convert real coefficient to complex and get absolute value
    rctd_set_td_ui(&tmp_c, get_tdpoly_i(poly, 0), 0UL);
    rctd_abs_td(absmax_anxn, &tmp_c);
 
    for(i = 1; i <= poly->deg; i++)
    {
        // anxn = coef[i] * xn
        rctd_set_td_ui(&tmp_c, get_tdpoly_i(poly, i), 0UL);
        rctd_mul(&anxn, &xn, &tmp_c);
        
        // abs_anxn = |anxn|
        rctd_abs_td(abs_anxn, &anxn);
        
        if(rtd_cmp(absmax_anxn, abs_anxn) < 0)
        {
            ret_i = i;
            rtd_set(absmax_anxn, abs_anxn);
        }
        
        // xn *= x
        rctd_mul(&xn, &xn, x);
    }

    return ret_i;
}

// Return j and max|a_j x^j| from p(x) for complex polynomial
long int absmax_ctdpoly(double absmax_anxn[TDSIZE], ctdfloat *x, CTDPoly poly)
{
    long int i, ret_i;
    double abs_anxn[TDSIZE];
    ctdfloat xn, anxn;

    // xn := x
    rctd_set(&xn, x);

    ret_i = 0;
    rctd_abs_td(absmax_anxn, get_ctdpoly_i(poly, 0));
 
    for(i = 1; i <= poly->deg; i++)
    {
        // anxn = coef[i] * xn
        rctd_mul(&anxn, &xn, get_ctdpoly_i(poly, i));
        
        // abs_anxn = |anxn|
        rctd_abs_td(abs_anxn, &anxn);
        
        if(rtd_cmp(absmax_anxn, abs_anxn) < 0)
        {
            ret_i = i;
            rtd_set(absmax_anxn, abs_anxn);
        }
        
        // xn *= x
        rctd_mul(&xn, &xn, x);
    }

    return ret_i;
}

// Get argument in [0, 2*PI]
void td_get_plus_arg(double ret[TDSIZE], ctdfloat *x)
{
    double pi2[TDSIZE];

    // pi2 := 2 * PI
    rtd_const_pi(pi2);
    rtd_mul_ui(pi2, pi2, 2UL);

    // ret = arg(x)
    rctd_arg(ret, x);

    // if(ret < 0) ret = 2*PI + ret
    if(rtd_cmp_ui(ret, 0UL) < 0)
    {
        rtd_add(ret, ret, pi2);
    }
    
    return;
}

// Get nearest integer
void td_get_nearest_int(double ret[TDSIZE], double real_x[TDSIZE])
{
    double int_x_floor[TDSIZE], int_x_ceil[TDSIZE];
    double dist_x_floor[TDSIZE], dist_x_ceil[TDSIZE];

    // int_x_floor = floor(real_x)
    rtd_func_mpfr(int_x_floor, mpfr_rint_floor, real_x);
    
    // int_x_ceil = ceil(real_x)
    rtd_func_mpfr(int_x_ceil, mpfr_rint_ceil, real_x);

    // dist_x_floor = |real_x - int_x_floor|
    rtd_sub(dist_x_floor, real_x, int_x_floor);
    rtd_abs(dist_x_floor, dist_x_floor);
    
    // dist_x_ceil = |real_x - int_x_ceil|
    rtd_sub(dist_x_ceil, real_x, int_x_ceil);
    rtd_abs(dist_x_ceil, dist_x_ceil);

    // Return the nearest
    if(rtd_cmp(dist_x_floor, dist_x_ceil) < 0)
        rtd_set(ret, int_x_floor);
    else
        rtd_set(ret, int_x_ceil);
}

// Get minimum branch
void td_get_min_branch(ctdfloat *ret, ctdfloat *x, double mu[TDSIZE], 
                       ctdfloat coef[], long int i_num, long int i_den)
{
    long int j = i_den;
    double phi[TDSIZE], psi[TDSIZE], pi2[TDSIZE], real_j[TDSIZE];
    double tmp[TDSIZE], tmp2[TDSIZE], pow_tmp[TDSIZE];
    ctdfloat ctmp, ctmp2, in_ret;

    // Check if coef[j] == 0
    rctd_abs_td(tmp, &coef[j]);
    if(rtd_cmp_ui(tmp, 0UL) == 0)
    {
        //rctd_set_ui_ui(ret, 0UL, 0UL);
        rctd_set_d_d(ret, FP_NAN, FP_NAN);
        return;
    }

    // pi2 := 2 * PI
    rtd_const_pi(pi2);
    rtd_mul_ui(pi2, pi2, 2UL);

    // phi = arg(x) / (2*PI)
    td_get_plus_arg(phi, x);
    rtd_div(phi, phi, pi2);
    
    // psi = arg(-coef[i_num] / coef[j]) / (2*PI)
    rctd_div(&ctmp, &coef[i_num], &coef[j]); // &coef, i_num), &coef, j));
    rctd_neg(&ctmp, &ctmp);
    td_get_plus_arg(psi, &ctmp);
    rtd_div(psi, psi, pi2);

    // real_j = nearest_int(j * (0.5 - phi) - psi)
    rtd_set_ui(real_j, 1UL);
    rtd_div_ui(real_j, real_j, 2UL);      // real_j = 0.5
    rtd_sub(real_j, real_j, phi);          // real_j = 0.5 - phi
    rtd_mul_ui(real_j, real_j, (unsigned long)j);  // real_j = j * (0.5 - phi)
    rtd_sub(real_j, real_j, psi);          // real_j = j * (0.5 - phi) - psi
    td_get_nearest_int(real_j, real_j);

    // Compute the branch:
    // ret = pow(|mu * coef[i_num] / coef[j]|, 1/j) 
    //       * exp(2*PI*I * ((psi + real_j) / j))

    // Part 1: pow_tmp = pow(|mu * coef[i_num] / coef[j]|, 1/j)
    
    // ctmp = mu * coef[i_num]
    rctd_mul_td(&ctmp2, &coef[i_num], mu);
    
    // ctmp = ctmp / coef[j]
    rctd_div(&ctmp, &ctmp2, &coef[j]);
    
    // tmp = |ctmp|
    rctd_abs_td(tmp, &ctmp);
    
    // tmp2 = 1/j
    rtd_set_ui(tmp2, 1UL);
    rtd_div_ui(tmp2, tmp2, (unsigned long)j);
    
    // pow_tmp = pow(tmp, tmp2)
    rtd_pow_mpfr(pow_tmp, tmp, tmp2);

    // Part 2: exp(2*PI*I * ((psi + real_j) / j))
    
    // tmp = psi + real_j
    rtd_add(tmp, psi, real_j);
    
    // tmp = (psi + real_j) / j
    rtd_div_ui(tmp, tmp, (unsigned long)j);
    
    // tmp = tmp * 2*PI
    rtd_mul(tmp, tmp, pi2);
    
    // ctmp = 0 + tmp*I
    rtd_set_ui(tmp2, 0UL);
    rctd_set_td_td(&ctmp, tmp2, tmp);
    
    // ret = exp(ctmp)
    rctd_func_mpc(&in_ret, mpc_exp, &ctmp);
    
    // ret = ret * pow_tmp
    rctd_mul_td(ret, &in_ret, pow_tmp);

    return;
}

// Hirano method for real polynomial
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int td_hirano(ctdfloat *ret, ctdfloat *init_x, TDPoly poly, 
                   double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
{
    long int i, j, max_j, times, absmin_j, deg = poly->deg;
    ctdfloat old_x, new_x, pn_x_d, ctmp, ctmp2;
    ctdfloat *in_coef, *d;
    double abs_pn_new_x[TDSIZE], absmax_anxn[TDSIZE], absmin_d[TDSIZE], abs_d[TDSIZE];
    double mu[TDSIZE], beta[TDSIZE], lambda[TDSIZE];
    double tmp[TDSIZE], tmp1[TDSIZE], tmp2[TDSIZE];

    // Parameters from Sugiura & Murota
    rtd_set_ui(beta, 3UL);
    rtd_div_ui(beta, beta, 4UL);        // beta = 3/4
    rtd_set_ui(lambda, 2UL);             // lambda = 2

    // Allocate coefficient arrays
    in_coef = (ctdfloat *)calloc(deg + 1, sizeof(ctdfloat));
    d = (ctdfloat *)calloc(deg + 1, sizeof(ctdfloat));

    // old_x = init_x
    rctd_set(&old_x, init_x);

    // Main iteration loop
    for(times = 0; times < maxtimes; times++)
    {
        // Compute coefficients of p(old_x + d)
        td_coef_horner(in_coef, &old_x, poly);
        
        // mu = 1
        rtd_set_ui(mu, 1UL);

        // Step (1): Try Newton's method
        // d[0] = 0
        rctd_set_ui_ui(&ctmp, 0UL, 0UL);
        rctd_set(&d[0], &ctmp);
        
        // if |in_coef[1]| != 0
        rctd_abs_td(tmp, &in_coef[1]);
        if(rtd_cmp_ui(tmp, 0UL) != 0)
        {
            // d[0] = -in_coef[0] / in_coef[1]
            rctd_div(&ctmp, &in_coef[0], &in_coef[1]);
            rctd_neg(&ctmp, &ctmp);
            rctd_set(&d[0], &ctmp);
        }

        // new_x = old_x + d[0]
        rctd_add(&new_x, &old_x, &d[0]);
        
        // pn_x_d = p(new_x)
        td_horner(&pn_x_d, &new_x, poly);

        // Check if |p(new_x)| <= beta * |in_coef[0]|
        rctd_abs_td(tmp, &pn_x_d);
        rctd_abs_td(tmp2, &in_coef[0]);
        rtd_mul(tmp2, tmp2, beta);
        
        if(rtd_cmp(tmp, tmp2) <= 0)
        {
            // Accept Newton step
            rctd_set(&old_x, &new_x);
        }
        else
        {
            // Try higher order corrections
            for(max_j = 2; max_j <= deg; max_j++)
            {
                // mu /= lambda
                rtd_div(mu, mu, lambda);
                
                // d[max_j] = get_min_branch(old_x, mu, in_coef, 0, max_j)
                td_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, max_j);
                rctd_set(&d[max_j], &ctmp2);
                
                // absmin_d = |d[max_j]|
                rctd_abs_td(absmin_d, &d[max_j]);
                absmin_j = max_j;
                
                // Find j with minimum |d[j] / lambda^(1/(j+1))|
                for(j = 0; j < max_j; j++)
                {
                    // d[j] = get_min_branch(old_x, mu, in_coef, 0, j+1)
                    td_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, j + 1);
                    rctd_set(&d[j], &ctmp2);
                    
                    // tmp1 = lambda^(1/(j+1))
                    rtd_set_ui(tmp, 1UL);
                    rtd_div_ui(tmp, tmp, (unsigned long)(j + 1));
                    rtd_pow_mpfr(tmp1, lambda, tmp);
                    
                    // d[j] = d[j] / tmp1
                    rctd_div_td(&ctmp2, &d[j], tmp1);
                    rctd_set(&d[j], &ctmp2);
                    
                    // abs_d = |d[j]|
                    rctd_abs_td(abs_d, &d[j]);
                    
                    if(rtd_cmp(abs_d, absmin_d) < 0)
                    {
                        rtd_set(absmin_d, abs_d);
                        absmin_j = j;
                    }
                }
                
                // new_x = old_x + d[absmin_j]
                rctd_add(&new_x, &old_x, &d[absmin_j]);
                
                // pn_x_d = p(new_x)
                td_horner(&pn_x_d, &new_x, poly);
                
                // Check if |p(new_x)| <= (1 - (1-beta)*mu) * |in_coef[0]|
                rctd_abs_td(tmp, &pn_x_d);
                rtd_set_ui(tmp2, 1UL);
                rtd_sub(tmp2, tmp2, beta);        // tmp2 = 1 - beta
                rtd_mul(tmp2, tmp2, mu);          // tmp2 = (1-beta) * mu
                rtd_sub_ui(tmp2, tmp2, 1UL);      // tmp2 = (1-beta)*mu - 1
                rtd_neg(tmp2, tmp2);              // tmp2 = 1 - (1-beta)*mu
                rctd_abs_td(tmp1, &in_coef[0]);
                rtd_mul(tmp2, tmp2, tmp1);        // tmp2 = (1 - (1-beta)*mu) * |in_coef[0]|
                
                if(rtd_cmp(tmp, tmp2) <= 0)
                {
                    rctd_set(&old_x, &new_x);
                    break;
                }
            }
        }

        // Check stopping criterion
        absmax_tdpoly(absmax_anxn, &new_x, poly);
        td_horner(&ctmp, &new_x, poly);
        rctd_abs_td(abs_pn_new_x, &ctmp);
        
        // Check if |p(new_x)| <= max|a_j*x^j| * reps + aeps
        rtd_mul(tmp, absmax_anxn, reps);
        rtd_add(tmp, tmp, aeps);
        
        if(rtd_cmp(abs_pn_new_x, tmp) <= 0)
            break;
    }

    free(in_coef);
    free(d);

    rctd_set(ret, &new_x);

    return times;
}

// Hirano method for complex polynomial
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int ctd_hirano(ctdfloat *ret, ctdfloat *init_x, CTDPoly poly, 
                    double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
{
    long int i, j, max_j, times, absmin_j, deg = poly->deg;
    ctdfloat old_x, new_x, pn_x_d, ctmp, ctmp2;
    ctdfloat *in_coef, *d;
    double abs_pn_new_x[TDSIZE], absmax_anxn[TDSIZE], absmin_d[TDSIZE], abs_d[TDSIZE];
    double mu[TDSIZE], beta[TDSIZE], lambda[TDSIZE];
    double tmp[TDSIZE], tmp1[TDSIZE], tmp2[TDSIZE];

    // Parameters from Sugiura & Murota
    rtd_set_ui(beta, 3UL);
    rtd_div_ui(beta, beta, 4UL);        // beta = 3/4
    rtd_set_ui(lambda, 2UL);             // lambda = 2

    // Allocate coefficient arrays
    in_coef = (ctdfloat *)calloc(deg + 1, sizeof(ctdfloat));
    d = (ctdfloat *)calloc(deg + 1, sizeof(ctdfloat)); // (ctdfloat *)calloc((deg + 1);

    // old_x = init_x
    rctd_set(&old_x, init_x);

    // Main iteration loop
    for(times = 0; times < maxtimes; times++)
    {
        // Compute coefficients of p(old_x + d)
        ctd_coef_horner(in_coef, &old_x, poly);
        
        // mu = 1
        rtd_set_ui(mu, 1UL);

        // Step (1): Try Newton's method
        // d[0] = 0
        rctd_set_ui_ui(&ctmp, 0UL, 0UL);
        rctd_set(&d[0], &ctmp);
        
        // if |in_coef[1]| != 0
        rctd_abs_td(tmp, &in_coef[1]);
        if(rtd_cmp_ui(tmp, 0UL) != 0)
        {
            // d[0] = -in_coef[0] / in_coef[1]
            rctd_div(&ctmp, &in_coef[0], &in_coef[1]);
            rctd_neg(&ctmp, &ctmp);
            rctd_set(&d[0], &ctmp);
        }

        // new_x = old_x + d[0]
        rctd_add(&new_x, &old_x, &d[0]);
        
        // pn_x_d = p(new_x)
        ctd_horner(&pn_x_d, &new_x, poly);

        // Check if |p(new_x)| <= beta * |in_coef[0]|
        rctd_abs_td(tmp, &pn_x_d);
        rctd_abs_td(tmp2, &in_coef[0]);
        rtd_mul(tmp2, tmp2, beta);
        
        if(rtd_cmp(tmp, tmp2) <= 0)
        {
            // Accept Newton step
            rctd_set(&old_x, &new_x);
        }
        else
        {
            // Try higher order corrections
            for(max_j = 2; max_j <= deg; max_j++)
            {
                // mu /= lambda
                rtd_div(mu, mu, lambda);
                
                // d[max_j] = get_min_branch(old_x, mu, in_coef, 0, max_j)
                td_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, max_j);
                rctd_set(&d[max_j], &ctmp2);
                
                // absmin_d = |d[max_j]|
                rctd_abs_td(absmin_d, &d[max_j]);
                absmin_j = max_j;
                
                // Find j with minimum |d[j] / lambda^(1/(j+1))|
                for(j = 0; j < max_j; j++)
                {
                    // d[j] = get_min_branch(old_x, mu, in_coef, 0, j+1)
                    td_get_min_branch(&ctmp2, &old_x, mu, in_coef, 0, j + 1);
                    rctd_set(&d[j], &ctmp2);
                    
                    // tmp1 = lambda^(1/(j+1))
                    rtd_set_ui(tmp, 1UL);
                    rtd_div_ui(tmp, tmp, (unsigned long)(j + 1));
                    rtd_pow_mpfr(tmp1, lambda, tmp);
                    
                    // d[j] = d[j] / tmp1
                    rctd_div_td(&ctmp2, &d[j], tmp1);
                    rctd_set(&d[j], &ctmp2);
                    
                    // abs_d = |d[j]|
                    rctd_abs_td(abs_d, &d[j]);
                    
                    if(rtd_cmp(abs_d, absmin_d) < 0)
                    {
                        rtd_set(absmin_d, abs_d);
                        absmin_j = j;
                    }
                }
                
                // new_x = old_x + d[absmin_j]
                rctd_add(&new_x, &old_x, &d[absmin_j]);
                
                // pn_x_d = p(new_x)
                ctd_horner(&pn_x_d, &new_x, poly);
                
                // Check if |p(new_x)| <= (1 - (1-beta)*mu) * |in_coef[0]|
                rctd_abs_td(tmp, &pn_x_d);
                rtd_set_ui(tmp2, 1UL);
                rtd_sub(tmp2, tmp2, beta);        // tmp2 = 1 - beta
                rtd_mul(tmp2, tmp2, mu);          // tmp2 = (1-beta) * mu
                rtd_sub_ui(tmp2, tmp2, 1UL);      // tmp2 = (1-beta)*mu - 1
                rtd_neg(tmp2, tmp2);              // tmp2 = 1 - (1-beta)*mu
                rctd_abs_td(tmp1, &in_coef[0]);
                rtd_mul(tmp2, tmp2, tmp1);        // tmp2 = (1 - (1-beta)*mu) * |in_coef[0]|
                
                if(rtd_cmp(tmp, tmp2) <= 0)
                {
                    rctd_set(&old_x, &new_x);
                    break;
                }
            }
        }

        // Check stopping criterion
        absmax_ctdpoly(absmax_anxn, &new_x, poly);
        ctd_horner(&ctmp, &new_x, poly);
        rctd_abs_td(abs_pn_new_x, &ctmp);
        
        // Check if |p(new_x)| <= max|a_j*x^j| * reps + aeps
        rtd_mul(tmp, absmax_anxn, reps);
        rtd_add(tmp, tmp, aeps);
        
        if(rtd_cmp(abs_pn_new_x, tmp) <= 0)
            break;
    }

    free(in_coef);
    free(d);

    rctd_set(ret, &new_x);

    return times;
}

// Deflation of polynomial: p(x) / (x - r)
void deflation_ctdpoly(CTDPoly ret, CTDPoly pol, ctdfloat *root)
{
    long int i;
    ctdfloat ctmp;
    CTDPoly in_ret;
    
    if(pol->deg < 1)
        return;

    in_ret = init_set_ctdpoly(ret);

    // Clear coefficients
    set0_ctdpoly(in_ret);

    // ret[deg-1] = coef[deg]
    set_ctdpoly_i(in_ret, pol->deg - 1, get_ctdpoly_i(pol, pol->deg));
    
    // Synthetic division
    for(i = pol->deg - 2; i >= 0; i--)
    {
        // ret[i] = ret[i+1] * root + coef[i+1]
        rctd_mul(&ctmp, get_ctdpoly_i(in_ret, i + 1), root);
        rctd_add(&ctmp, &ctmp, get_ctdpoly_i(pol, i + 1));
        set_ctdpoly_i(in_ret, i, &ctmp);
    }

    setdegree_ctdpoly(in_ret);
    subst_ctdpoly(ret, in_ret);

    free_ctdpoly(in_ret);
    
    return;
}

