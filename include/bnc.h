/********************************************************************************/
/* BNC header file                                                              */
/* Copyright (c) 2000-2012 Tomonori Kouya, All rights reserved.                 */
/*                                                                              */
/* History:                                                                     */
/* Version 0.0.1  2000.5/23: modifying mpf_sin, mpf_cos                         */
/*         0.0.2  2000.5/26: append mpf_ln, mpf_log10                           */
/*         0.0.3  2000.6/13: append FStack, DStack, MPFStack                    */
/*         0.0.4  2000.6/14: append and test euler.c ex_nim.c                   */
/*         0.0.5  2000.6/19: append and test newton.c                           */
/*         0.0.6  2000.6/19: append and test rk.c                               */
/*         0.0.7  2000.6/20: append set0_[f,d,mpf]vector                        */
/*         0.0.7a 2000.6/20: append set0_[f,d,mpf]matrix                        */
/*                           append setI_[f,d,mpf]matrix                        */
/*         0.0.7b 2000.7/04: revise lu.c                                        */
/*         0.0.7c 2000.7/05: revise euler.c, rk.c, rkcoef.c                     */
/*         0.0.7d 2000.12/4: append power.c                                     */
/*         0.0.7e 2001.4/23: append integral.c                                  */
/*         0.0.7f 2001.4/24: append bnc_mpf_[init]_set_d                        */
/*         0.0.7g 2001.6/05: append ivpower.c                                   */
/*         0.1    2001.8/02: setting precision for                              */
/*                           MPF* functions                                     */
/*         0.1a   2001.8/08: Bug Fix : init_*vector[2],                         */
/*                           rkcoef.c, ex_nim.c                                 */
/*         0.2    2002.5/04: append poly.c, complex.c,                          */
/*                           array, dka.c                                       */
/*         0.2a   2002.5/28: append iterative.c                                 */
/*         0.3    2002.9/05: include mpfr                                       */
/*         0.3a   2003.2/27: append get_sec.c & bug fix                         */
/*         0.3b   2003.2/27: separate modules of ODEs                           */
/*         0.3c   2003.6/11: append qr.c & gtestmat.c                           */
/*         0.3d   2003.12/6: append get_real_sec()                              */
/*         0.3e   2004.1/07: include new functions                              */
/*                                       from mpfr-2.0.2                        */
/*         0.3f   2004.3/06: append xmtrapezoidal_fs funcs                      */
/*                                       into integral.c                        */
/*         0.4    2004.3/08: append rational.c                                  */
/*         0.4a   2004.9/18: modifiy linear.c with mpfr_fma                     */
/*         0.4b   2004.11/3: append krylov.c                                    */
/*         0.5    2005.6/07: append diff.c                                      */
/*         0.5a   2005.6/07: modify gtestmat.c                                  */
/*         0.6    2005.7/12: append fread_write.c                               */
/*         0.6a   2005.7/12: modify linear.c(copy_*vector_ij)                   */
/*         0.6b   2006.1/13: modify lu.c(True LU decomposion)                   */
/*         0.6b1  2006.1/26: append mul_*poly in poly.c                         */
/*                2006.1/26: append mpf_fma in efunc.c                          */
/*         0.6b2  2006.2/13: append norms of matrix                             */
/*         0.6b2  2006.3/14: Bug fix : mpf_max, min                             */
/*         0.6b3  2006.7/26: modify diff.c                                      */
/*         0.6b4  2006.10/13: Bug fix : macro in bnc.h                          */
/*         0.6b5  2006.10/13: append mharmonic.c                                */
/*         0.6b6  2007.01/11: Bug fix : poly.c                                  */
/*         0.6b7  2007.06/26: Append algebraic_eq.c                             */
/*         0.6b8  2008.06/04: Bug fix : complex.c, array.c                      */
/*         0.6c   2008.06/04: Bug fix : Makefile (Thanks, Elena)                */
/*         0.6d   2011.01/31: Fix MPI functions for mpfr 3.x                    */
/*         0.7    2011.08/25: append bncsparse                                  */
/*         0.79   2012-03-18: append clinear.c clu.c                            */
/*                            modify cg.c, krylov.c                             */
/*         0.791  2012-06-03: Bug fix: linear.c linear_append.c poly.c          */
/*         0.792  2012-07-05: append bmatrix.c                                  */
/*         0.7921 2012-07-13: modify readmatrix.c & bug fix                     */
/*         0.7922 2012-07-18: Buf fix: lu.c, bmatrix.c, poly.c                  */
/*         0.8    2013-02-28: Append print_date                                 */
/*         0.8a   2013-06-07: Bug fix efunc.c                                   */
/*         0.8b   2015-04-20: Rename DCG, djacobi -> bnc_DCG, bnc_djacobi       */
/*         0.8c   2018-02-26: Fix "double complex" errors in C++                */
/*         0.9    2019-07-22: append *element_block to MPFMatrix                */
/*         0.9a   2024-11-21: bncpack has been included in bncmatmul            */
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
#ifndef __BNC_H
#define __BNC_H

// Accelerated BLAS library 
//#include "bncmatmul.h"
#ifdef USE_GMP
#include "gmp.h"
#ifdef USE_MPFR
#include "mpfr.h"
#endif // USE_MPFR
#endif // USE_GMP

// get_secv
#include "get_secv.h"

// MPQVector, MPQMatrix
#ifdef USE_GMP
typedef struct{
	mpq_t *element;
	long int dim;
} mpqvector;

typedef mpqvector *MPQVector;

#endif // USE_GMP

typedef struct{
//	unsigned int type;
	mpq_t *element;
	long int row_dim, col_dim;
} mpqmatrix;

typedef mpqmatrix *MPQMatrix;

/*************************************************/
/* Stack                                         */
/*************************************************/
/* fstack */
typedef struct {
	float *array; /* stack */
	long int size; /* Height of stack */
	long index; /* pointer to top of stack */
} fstack;

typedef fstack *FStack;

/* dstack */
typedef struct {
	double *array; /* stack */
	long int size; /* Height of stack */
	long index; /* pointer to top of stack */
} dstack;

typedef dstack *DStack;

#ifdef USE_GMP
/* mpfstack */
typedef struct {
	unsigned long prec; /* precision */
	mpf_t *array; /* stack */
	long int size; /* Height of stack */
	long index; /* pointer to top of stack */
} mpfstack;

typedef mpfstack *MPFStack;
#endif // USE_GMP

/* errors */
// #define STACK_OVERFLOW	10
// #define STACK_UNDERFLOW	20


/*************************************************/
/* stack.c: Stack                                */
/*************************************************/
FStack init_fstack(long int stack_size);
void free_fstack(FStack st);
void push_fstack(FStack st, float val);
float pop_fstack(FStack st);
DStack init_dstack(long int stack_size);
void free_dstack(DStack st);
void push_dstack(DStack st, double val);
double pop_dstack(DStack st);
#ifdef USE_GMP
MPFStack init_mpfstack(long int stack_size);
MPFStack init2_mpfstack(long int stack_size, unsigned long prec);
void free_mpfstack(MPFStack st);
void push_mpfstack(MPFStack st, mpf_t val);
void pop_mpfstack(mpf_t rval, MPFStack st);
#endif // USE_GMP

#if 0 //

/*************************************************/
/* power.c, ivpower.c: Power and Inverse Power Methods */
/*************************************************/
//long int fpower_eig(float *, FVector, FMatrix, FVector, float, float, long);
//long int fivpower_eig(float *, FVector, FMatrix, FVector, float, float, long);
long int dpower_eig(double *, DVector, DMatrix, DVector, double, double, long);
long int divpower_eig(double *, DVector, DMatrix, DVector, double, double, long);
#ifdef USE_GMP
long int mpf_power_eig(mpf_t, MPFVector, MPFMatrix, MPFVector, mpf_t, mpf_t, long);
long int mpf_ivpower_eig(mpf_t, MPFVector, MPFMatrix, MPFVector, mpf_t, mpf_t, long);
#endif // USE_GMP

/*************************************************/
/* newton.c: Newton/Simplified Newton Methods    */
/*************************************************/
//long int fnewton_1(float *, float, float (* func)(float), float (* dfunc)(float), long int, float, float);
//long int fsnewton_1(float *, float, float (* func)(float), float (* dfunc)(float), long int, float, float);
//long int fnewton(FVector, FVector, void (* func)(FVector, FVector), void (* jfunc)(FMatrix, FVector), long int, float, float);
//long int fsnewton(FVector, FVector, void (* func)(FVector, FVector), void (* jfunc)(FMatrix, FVector), long int, float, float);
long int dnewton_1(double *, double, double (* func)(double), double (* dfunc)(double), long int, double, double);
long int dsnewton_1(double *, double, double (* func)(double), double (* dfunc)(double), long int, double, double);
long int dnewton(DVector, DVector, void (* func)(DVector, DVector), void (* jfunc)(DMatrix, DVector), long int, double, double);
long int dsnewton(DVector, DVector, void (* func)(DVector, DVector), void (* jfunc)(DMatrix, DVector), long int, double, double);
#ifdef USE_GMP
long int mpf_newton_1(mpf_t, mpf_t, void (* func)(mpf_t, mpf_t), void (* dfunc)(mpf_t, mpf_t), long int, mpf_t, mpf_t);
long int mpf_snewton_1(mpf_t, mpf_t, void (* func)(mpf_t, mpf_t), void (* dfunc)(mpf_t, mpf_t), long int, mpf_t, mpf_t);
long int mpf_newton(MPFVector, MPFVector, void (* func)(MPFVector, MPFVector), void (* jfunc)(MPFMatrix, MPFVector), long int, mpf_t, mpf_t);
long int mpf_snewton(MPFVector, MPFVector, void (* func)(MPFVector, MPFVector), void (* jfunc)(MPFMatrix, MPFVector), long int, mpf_t, mpf_t);
#endif // USE_GMP


/*************************************************/
/* integral.c: Numerical Integral                */
/*************************************************/
//float ftrapezoidal_fs(float, float, float(* func)(float), long int);
double dtrapezoidal_fs(double, double, double(* func)(double), long int);
double dmtrapezoidal_fs(double, double, double(* func)(double), long int);
#ifdef USE_GMP
void mpf_trapezoidal_fs(mpf_t, mpf_t, mpf_t, void(* func)(mpf_t, mpf_t), long int);
void mpf_mtrapezoidal_fs(mpf_t, mpf_t, mpf_t, void(* func)(mpf_t, mpf_t), long int);
#endif // USE_GMP


/*****************************************************/
/* iterative.c: Iterative Methods for Linear Systems */
/*****************************************************/
//void get_residual_fvector(FVector, FVector, FMatrix, FVector);
//void fjacobi(FVector, FMatrix, FVector, float, float, long int);
//void fgs(FVector, FMatrix, FVector, float, float, long int);
//void fsor(FVector, FMatrix, FVector, float, float, float, long int);

void get_residual_dvector(DVector, DVector, DMatrix, DVector);
//void djacobi(DVector, DMatrix, DVector, double, double, long int);
void bnc_djacobi(DVector, DMatrix, DVector, double, double, long int);
void dgs(DVector, DMatrix, DVector, double, double, long int);
void dsor(DVector, DMatrix, DVector, double, double, double, long int);

#ifdef USE_GMP
void mpf_jacobi(MPFVector, MPFMatrix, MPFVector, mpf_t, mpf_t, long int);
void mpf_gs(MPFVector, MPFMatrix, MPFVector, mpf_t, mpf_t, long int);
void mpf_sor(MPFVector, MPFMatrix, MPFVector, mpf_t, mpf_t, mpf_t, long int);
#endif // USE_GMP

/************************************************/
/* qr.c: Gram-Schmidt, Modified Gram-Schmidt, QR*/
/************************************************/
void dgram_schmidt(DMatrix, DMatrix, DMatrix);
void dmgram_schmidt(DMatrix, DMatrix, DMatrix);
void dqr(DMatrix, long int);

#ifdef USE_GMP
void mpf_gram_schmidt(MPFMatrix, MPFMatrix, MPFMatrix);
void mpf_mgram_schmidt(MPFMatrix, MPFMatrix, MPFMatrix);
void mpf_qr(MPFMatrix, long int);
#endif // USE_GMP

/*************************************************/
/* rational.c: Functions for Vector Types        */
/*************************************************/

#ifdef USE_GMP
void mpq_get_f(mpf_t, mpq_t);

MPQVector init_mpqvector(long int dimension);
void free_mpqvector(MPQVector vec);

mpq_ptr get_mpqvector_i(MPQVector vec, long int index);
void set_mpqvector_i(MPQVector vec, long int index, mpq_t val);
void set_mpqvector_i_z(MPQVector vec, long int index, mpz_t val);
void set_mpqvector_i_str_b(MPQVector vec, long int index, const char *str, int base);
void set_mpqvector_i_str(MPQVector vec, long int index, const char *str);
void set_mpqvector_i_ui(MPQVector vec, long int index, unsigned long val_num, unsigned long val_den);
void set_mpqvector_i_si(MPQVector vec, long int index, long val_num, unsigned long val_den);

void print_mpqvector(MPQVector vec);

void ip_mpqvector(mpq_t ret, MPQVector a, MPQVector b);
#endif // USE_GMP

/*************************************************/
/* diff.c: Numerical Differentiations            */
/*************************************************/
double drelerr(double, double);
double dcentral_diff(double, double (*func)(double), double);
double dcentral_diff15(double, double (*func)(double), double);
double dcentral_diff17(double, double (*func)(double), double);
double dfnmdiff(double, double (*func)(double), double, double, double, long int, long int *);
void djfnmdiff(DMatrix, DVector, void (*func)(DVector, double, DVector), double, double, double, long int);
#ifdef USE_GMP
void mpfrelerr(mpf_t, mpf_t, mpf_t);
void mpf_central_diff(mpf_t, mpf_t, void (*func)(mpf_t, mpf_t), mpf_t);
void mpf_central_diff15(mpf_t, mpf_t, void (*func)(mpf_t, mpf_t), mpf_t);
void mpf_central_diff17(mpf_t, mpf_t, void (*func)(mpf_t, mpf_t), mpf_t);
void mpffnmdiff(mpf_t, mpf_t, void (*func)(mpf_t, mpf_t), mpf_t, mpf_t, mpf_t, long int, long int *);
void mpf_jfnmdiff(MPFMatrix, MPFVector, void (*func)(MPFVector, mpf_t, MPFVector), mpf_t, mpf_t, mpf_t, long int);
#endif // USE_GMP

/*************************************************/
/* mharmonic.c: Extrapolation for Integration    */
/*************************************************/
unsigned long ex_nim_den(long int);
double ex_harmonic_den(long int istage, long int jstage);
double dmharmonic_integral_1step(double x_start, double x_end, double (* func)(double), long int num_stage, double reps, double aeps, long int *conv_flag);
double dmharmonic_integral(double x_start, double x_end, double (* func)(double), long int max_num_stage);
double dmharmonic_backward_integral(double x_start, double x_end, double (* func)(double), long int max_num_stage);
double dmromberg_integral_1step(double x_start, double x_end, double (* func)(double), long int num_stage, double reps, double aeps, long int *conv_flag);
double dmromberg_integral(double x_start, double x_end, double (* func)(double), long int max_num_stage);
#ifdef USE_GMP
void mpf_ex_harmonic_den(mpf_t ret, long int istage, long int jstage);
void mpf_ex_harmonic1_den(mpf_t ret, long int istage, long int jstage);
void mpf_ex_harmonic2_den(mpf_t ret, long int istage, long int jstage);
void mpf_ex_harmonic3_den(mpf_t ret, long int istage, long int jstage);
void mpf_mharmonic_integral_1step(mpf_t ret, mpf_t x_start, mpf_t x_end, void (* func)(mpf_t, mpf_t), long int num_stage, mpf_t reps, mpf_t aeps, long int *conv_flag);
void mpf_mharmonic_integral(mpf_t ret, mpf_t x_start, mpf_t x_end, void (* func)(mpf_t, mpf_t), long int max_num_stage);
void mpf_mromberg_integral_1step(mpf_t ret, mpf_t x_start, mpf_t x_end, void (* func)(mpf_t, mpf_t), long int num_stage, mpf_t reps, mpf_t aeps, long int *conv_flag);
void mpf_mromberg_integral(mpf_t ret, mpf_t x_start, mpf_t x_end, void (* func)(mpf_t, mpf_t), long int max_num_stage);
#endif // USE_GMP

#endif // 0

#endif /* End of __BNC_H */
