/********************************************************************************/
/* mpi_bnc.h:                                                                   */
/* Copyright (C) 2003-2011 Tomonori Kouya, All rights reserved.                 */
/*                                                                              */
/* Version 0.0 : 2003.08/21 merge mpi_linear.c & mpi_dka.c                      */
/* Version 0.1 : 2003.10/04 append mpi_aux.c                                    */
/* Version 0.2 : 2003.10/04 append mpi_matrix_mul.c                             */
/* Version 0.3 : 2004.02/21 append mpi_ex_*.c                                   */
/* Version 0.4 : 2004.02/23 append mpi_ex_*_lo.c                                */
/* Version 0.4a: 2004.03/06 append functions                                    */
/* Version 0.4b: 2005.05/06 append mpi_krylov.c                                 */
/* Version 0.5 : 2005.06/24 change ODE routines                                 */
/* Version 0.5a: 2005.07/12 append allgather                                    */
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
#ifndef __STDIO_H__
#include <stdio.h>
#endif

#ifndef __MATH_H__
#include <math.h>
#endif

#ifndef __MPI_INCLUDE
#include "mpi.h"
#endif

#ifdef USE_GMP
#include "gmp.h"
#ifdef USE_MPFR
#include "mpfr.h"
#include "mpf2mpfr.h"
#endif
#endif

#ifndef __BNC_H
#include "bnc.h"
#endif

#ifndef __MPI_GMP_H
#include "mpi_gmp.h"
#endif

#ifndef __MPI_BNC_H
#define __MPI_BNC_H
/***************************************/
/* Auxillary Functions                 */
/***************************************/
#ifdef USE_GMP
void _mpi_set_bnc_default_prec(unsigned long, MPI_Comm);
void _mpi_set_bnc_default_prec_decimal(unsigned long, MPI_Comm);
#ifdef USE_MPFR
void _mpi_set_bnc_rounding_mode(mp_rnd_t, MPI_Comm);
#endif
#endif

/***************************************/
/* Basic Linear Computation            */
/***************************************/

/* double */

/* init for vector */
DVector _mpi_init_dvector(long d_dim[], long int dimension, MPI_Comm comm);

/* free dvector */
void _mpi_free_dvector(DVector vec);

/* Divide original vector on P0 */
void _mpi_divide_dvector(DVector d_vec, long int d_dim[], DVector src_vec, MPI_Comm comm);
/* Collect vectors on P0 */
void _mpi_collect_dvector(DVector src_vec, long int d_dim[], DVector d_vec, MPI_Comm comm);

/* Matrix */

/* init for matrix */
void _mpi_init_dmatrix(DMatrix ret[], long d_dim[], long int dimension, MPI_Comm comm);

/* free dmatrix */
void _mpi_free_dmatrix(DMatrix mat[], MPI_Comm comm);

/* Divide original matrix on P0 */
void _mpi_divide_dmatrix(DMatrix d_mat[], long int d_dim[], DMatrix src_mat, MPI_Comm comm);

/* Collect matrices on P0 */
void _mpi_collect_dmatrix(DMatrix src_mat, long int d_dim[], DMatrix d_mat[], MPI_Comm comm);

/* Inner Product: (a, b) */
/* (1) eval (a, b) locally */
/* (2) Allreduce (a, b) */
double _mpi_ip_dvector(DVector in_a, DVector in_b, MPI_Comm comm);

double _mpi_norm2_dvector(DVector in_a, MPI_Comm comm);
double _mpi_norm1_dvector(DVector in_a, MPI_Comm comm);
double _mpi_normi_dvector(DVector in_a, MPI_Comm comm);

/* Matrix * Vector: Ax = y*/
/* (1) Allgather x */
/* (2) localy = A * x */
void _mpi_mul_dmatrix_dvec(DVector ret, DMatrix a[], DVector x, DVector x_all, MPI_Comm comm);

/* mpf_t */
#ifdef USE_GMP

/* init for vector */
MPFVector _mpi_init_mpfvector(long d_dim[], long int dimension, MPI_Comm comm);

/* free dvector */
void _mpi_free_mpfvector(MPFVector vec);

/* Divide original vector on P0 */
void _mpi_divide_mpfvector(MPFVector d_vec, long int d_dim[], MPFVector src_vec, MPI_Comm comm);

/* Collect vectors on P0 */
void _mpi_collect_mpfvector(MPFVector src_vec, long int d_dim[], MPFVector d_vec, MPI_Comm comm);

/* Matrix */

/* init for matrix */
void _mpi_init_mpfmatrix(MPFMatrix ret[], long d_dim[], long int dimension, MPI_Comm comm);

/* free MPFMatrix */
void _mpi_free_mpfmatrix(MPFMatrix mat[], MPI_Comm comm);

/* Divide original matrix on P0 */
void _mpi_divide_mpfmatrix(MPFMatrix d_mat[], long int d_dim[], MPFMatrix src_mat, MPI_Comm comm);

/* Collect matrices on P0 */
void _mpi_collect_mpfmatrix(MPFMatrix src_mat, long int d_dim[], MPFMatrix d_mat[], MPI_Comm comm);

/* Inner Product: (a, b) */
/* (1) eval (a, b) locally */
/* (2) Allreduce (a, b) */
void _mpi_ip_mpfvector(mpf_t ret, MPFVector in_a, MPFVector in_b, MPI_Comm comm);

void _mpi_norm2_mpfvector(mpf_t ret, MPFVector in_a, MPI_Comm comm);
void _mpi_norm1_mpfvector(mpf_t ret, MPFVector in_a, MPI_Comm comm);
void _mpi_normi_mpfvector(mpf_t ret, MPFVector in_a, MPI_Comm comm);

/* Matrix * Vector: Ax = y*/
/* (1) Allgather x */
/* (2) locally = A * x */
void _mpi_mul_mpfmatrix_mpfvec(MPFVector ret, MPFMatrix a[], MPFVector x, MPFVector x_all, MPI_Comm comm);
#endif

/***************************************/
/* Complex Number                      */
/***************************************/

/* double complex */

void *allocbuf_dcmplx(int incount);

int pack_cdarray(CDArray array, void *buf);

void unpack_cdarray(void *buf, CDArray array, long int size);

#ifdef USE_GMP

/* mpf_t */

MPI_Datatype _bnc_mpfcmplx;
MPI_Datatype _tmp_bnc_mpfcmplx;

#define MPI_BNC_MPFCMPLX _bnc_mpfcmplx

/* typedef and commit to mpich */
void commit_mpi_mpfcmplx(MPI_Datatype *mpfcmplx_t, unsigned long prec, MPI_Comm comm);

/* clear type */
void free_mpi_mpfcmplx(MPI_Datatype *mpfcmplx_t);

size_t get_bufsize_mpfcmplx(MPFCmplx a, int incount);

void *allocbuf_mpfcmplx(unsigned long prec, int incount);

int pack_cmpfarray(CMPFArray array, void *buf);

void unpack_cmpfarray(void *buf, CMPFArray array, long int size);

#endif

/*************************************************/
/* mpi_integral.c                                */
/*************************************************/

/* double: Trapezoidal rule */
void _mpi_dtrapezoidal_fs(double *ptr_ret, double x_start, double x_end, double (*func)(double x), long int num_div, MPI_Comm comm);
void _mpi_dtrapezoidal_fs_all(double *ptr_ret, double x_start, double x_end, double (*func)(double x), long int num_div, MPI_Comm comm);
void _mpi_dmtrapezoidal_fs_all(double *ptr_ret, double x_start, double x_end, double (*func)(double x), long int num_div, MPI_Comm comm);


#ifdef USE_GMP
/* Trapezoidal rule */
void _mpi_mpf_trapezoidal_fs(mpf_t ret, mpf_t x_start, mpf_t x_end, void (*func)(mpf_t, mpf_t), long int num_div, MPI_Comm comm, MPI_Datatype mpi_mpf_type);
void _mpi_mpf_trapezoidal_fs_all(mpf_t ret, mpf_t x_start, mpf_t x_end, void (*func)(mpf_t, mpf_t), long int num_div, MPI_Comm comm);
void _mpi_mpf_mtrapezoidal_fs_all(mpf_t ret, mpf_t x_start, mpf_t x_end, void (*func)(mpf_t, mpf_t), long int num_div, MPI_Comm comm);
#endif

/*************************************************/
/* DCG, MPFCG                                    */
/*************************************************/

long int _mpi_DCG(DVector local_answer, DMatrix local_a[], DVector local_b, double reps, double aeps, long int maxtimes, long int dim, MPI_Comm comm);

#ifdef USE_GMP
long int _mpi_MPFCG(MPFVector local_answer, MPFMatrix local_a[], MPFVector local_b, mpf_t reps, mpf_t aeps, long int maxtimes, long int dim, MPI_Comm comm);
#endif

/*************************************************/
/* Krylov Subspace Methods                       */
/*************************************************/

long int _mpi_DBiCG(DVector local_answer, DMatrix local_a[], DMatrix local_at[], DVector local_b, double reps, double aeps, long int maxtimes, long int dim, MPI_Comm comm);
long int _mpi_DCGS(DVector local_answer, DMatrix local_a[], DVector local_b, double reps, double aeps, long int maxtimes, long int dim, MPI_Comm comm);
long int _mpi_DBiCGSTAB(DVector local_answer, DMatrix local_a[], DVector local_b, double reps, double aeps, long int maxtimes, long int dim, MPI_Comm comm);
long int _mpi_DGPBiCG(DVector local_answer, DMatrix local_a[], DVector local_b, double reps, double aeps, long int maxtimes, long int dim, MPI_Comm comm);

#ifdef USE_GMP
long int _mpi_MPFBiCG(MPFVector local_answer, MPFMatrix local_a[], MPFMatrix local_at[], MPFVector local_b, mpf_t reps, mpf_t aeps, long int maxtimes, long int dim, MPI_Comm comm);
long int _mpi_MPFCGS(MPFVector local_answer, MPFMatrix local_a[], MPFVector local_b, mpf_t reps, mpf_t aeps, long int maxtimes, long int dim, MPI_Comm comm);
long int _mpi_MPFBiCGSTAB(MPFVector local_answer, MPFMatrix local_a[], MPFVector local_b, mpf_t reps, mpf_t aeps, long int maxtimes, long int dim, MPI_Comm comm);
long int _mpi_MPFGPBiCG(MPFVector local_answer, MPFMatrix local_a[], MPFVector local_b, mpf_t reps, mpf_t aeps, long int maxtimes, long int dim, MPI_Comm comm);
#endif

/*************************************************/
/* mpi_dka.c: Durand-Kerner-Aberth Methods       */
/*************************************************/

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void _mpi_ddka_init(CDArray local_x_init, DPoly func, MPI_Comm comm);

/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
void _mpi_ddka(long int *lasttimes, CDArray ans, CDArray local_ans, CDArray x_init, CDArray local_x_init, DPoly func, long int maxtimes, double abs_eps, double rel_eps, MPI_Comm comm);

#ifdef USE_GMP

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void _mpi_mpf_dka_init(CMPFArray local_x_init, MPFPoly func, MPI_Comm comm);

/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
void _mpi_mpf_dka(long int *lasttimes, CMPFArray ans, CMPFArray local_ans, CMPFArray x_init, CMPFArray local_x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps, MPI_Comm comm);

#endif

/*******************************************/
/* mpi_bcast.c: Broadcast BNC defined data */
/*******************************************/

/* Bcast DPoly from P0 to other processes */
void _mpi_bcast_dpoly(DPoly poly, MPI_Comm comm);

/* Bcast DVector from P0 to other processes */
void _mpi_bcast_dvector(DVector vec, MPI_Comm comm);

/* Allgather DVector especially for ODEs */
void _mpi_allgather_dvector(DVector full_vec, DVector local_vec, MPI_Comm comm);

#ifdef USE_GMP

/* Bcast MPFPoly from P0 to other processes */
void _mpi_bcast_mpfpoly(MPFPoly poly, MPI_Comm comm);

/* Bcast MPFVector from P0 to other processes */
void _mpi_bcast_mpfvector(MPFVector vec, MPI_Comm comm);

/* Allgather DVector especially for ODEs */
void _mpi_allgather_mpfvector(MPFVector full_vec, MPFVector local_vec, MPI_Comm comm);

#endif

/********************************************/
/* mpi_matrix_mul.c: matrix-matrix multiply */
/********************************************/

/* initialize a index of d_mat on PE_i */
long int *init_mm_index(long int mat_dim);

/* shift index */
void shift_mm_index(long int *index, long int mat_dim);

/* check index */
int check_mm_index(long int *index, long int mat_dim);

/* Shift index of d_mat on PE_i */
/* <- col_index[0] <- col_index[1] <- ... <- col_index[mat_dim - 1] <- col_index[0] */
void _mpi_send_west(long int *col_index, long int mat_dim);

/* double */

/* send dmatrix to "dest_proc"th PE */
void _mpi_send_dmatrix(DMatrix mat, int dest_proc, MPI_Comm comm);

/* recv dmatrix from "src_proc"th PE */
void _mpi_recv_dmatrix(DMatrix mat, int src_proc, MPI_Comm comm);


/* Send submatrix to uppper PE */
/* mat on PE_{i-1} <- mat on PE_i */
void _mpi_dmatrix_send_north(DMatrix mat, DMatrix tmp_mat, long int *row_index, long int mat_dim, MPI_Comm comm);

/* Matrix * Matrix: ret := AB */
void _mpi_mul_dmatrix(DMatrix ret[], DMatrix a[], DMatrix b[], MPI_Comm comm);

#ifdef USE_GMP

/* mpf_t */

/* send mpfmatrix to "dest_proc"th PE */
void _mpi_send_mpfmatrix(MPFMatrix mat, int dest_proc, MPI_Comm comm);

/* recv mpfmatrix from "src_proc"th PE */
void _mpi_recv_mpfmatrix(MPFMatrix mat, int src_proc, MPI_Comm comm);

/* Send submatrix to uppper PE */
/* mat on PE_{i-1} <- mat on PE_i */
void _mpi_mpfmatrix_send_north(MPFMatrix mat, MPFMatrix tmp_mat, long int *row_index, long int mat_dim, MPI_Comm comm);

/* Matrix * Matrix: ret := AB */
void _mpi_mul_mpfmatrix(MPFMatrix ret[], MPFMatrix a[], MPFMatrix b[], MPI_Comm comm);

#endif

/********************************************/
/* mpi_ex_nim.c: Extrapolation with Romberg sequence */
/********************************************/

#define MPI_MAX_DEX_NIM_STAGE 32

DVector dex_nim_local_y[MPI_MAX_DEX_NIM_STAGE][MPI_MAX_DEX_NIM_STAGE];
long int dex_nim_stage;
DVector dex_nim_ytmp[4], dex_nim_local_ytmp[4];
double dex_nim_in_h;
long int dex_nim_local_dim, dex_nim_ddim[MPI_GMP_MAXPROCS];

void _mpi_init_dex_nim(long int stage, long int dimension, MPI_Comm comm);
void _mpi_clear_dex_nim(void);

int _mpi_dex_nim_1step(DVector, double, DVector, DVector, double, void (* func)(DVector, double, DVector, MPI_Comm), double, double, MPI_Comm);
void _mpi_dex_nim_fs(FILE *, double, DVector, double, DVector, long int, void (* func)(DVector, double, DVector, MPI_Comm), double, double, long int, MPI_Comm);
void _mpi_dex_nim(FILE *, double, DVector, double, DVector, double, void (* func)(DVector, double, DVector, MPI_Comm), double, double, long int, void (* ansfunc)(DVector, double), MPI_Comm comm);

int _mpi_dex_nim_1step_lo(DVector, double, DVector, DVector, double, DMatrix[], void (* gfunc)(DVector, double, MPI_Comm), double, double, MPI_Comm);
void _mpi_dex_nim_fs_lo(FILE *, double, DVector, double, DVector, long int, DMatrix, void (* gfunc)(DVector, double, MPI_Comm), double, double, long int, MPI_Comm);
void _mpi_dex_nim_lo(FILE *, double, DVector, double, DVector, double, DMatrix, void (* gfunc)(DVector, double, MPI_Comm), double, double, long int, MPI_Comm);

#ifdef USE_GMP

#define MPI_MAX_EX_NIM_STAGE 32

/* inner variables for mpf_ex_nim */
MPFVector ex_nim_local_y[MPI_MAX_EX_NIM_STAGE][MPI_MAX_EX_NIM_STAGE];
long int ex_nim_stage;
MPFVector ex_nim_local_ytmp[4], ex_nim_ytmp[4];
mpf_t ex_nim_in_h, ex_nim_mpf_tmp;
long int ex_nim_local_dim, ex_nim_ddim[MPI_GMP_MAXPROCS];

void _mpi_init_mpf_ex_nim(long int stage, long int dimension, unsigned long prec, MPI_Comm comm);
void _mpi_clear_mpf_ex_nim(void);

int _mpi_mpf_ex_nim_1step(MPFVector, mpf_t, MPFVector, MPFVector, mpf_t, void (* func)(MPFVector, mpf_t, MPFVector, MPI_Comm), mpf_t, mpf_t, MPI_Comm);
void _mpi_mpf_ex_nim_fs(FILE *, mpf_t, MPFVector, mpf_t, MPFVector, long int, void (* func)(MPFVector, mpf_t, MPFVector, MPI_Comm), mpf_t, mpf_t, long int, MPI_Comm);
void _mpi_mpf_ex_nim(FILE *, mpf_t, MPFVector, mpf_t, MPFVector, mpf_t, void (* func)(MPFVector, mpf_t, MPFVector, MPI_Comm), mpf_t, mpf_t, long int, void (* ansfunc)(MPFVector, mpf_t), long int , MPI_Comm);

int _mpi_mpf_ex_nim_1step_lo(MPFVector, mpf_t, MPFVector, MPFVector, mpf_t, MPFMatrix[], void (* gfunc)(MPFVector, mpf_t, MPI_Comm), mpf_t, mpf_t, MPI_Comm);
void _mpi_mpf_ex_nim_fs_lo(FILE *, mpf_t, MPFVector, mpf_t, MPFVector, long int, MPFMatrix, void (* gfunc)(MPFVector, mpf_t, MPI_Comm), mpf_t, mpf_t, long int, MPI_Comm);
void _mpi_mpf_ex_nim_lo(FILE *, mpf_t, MPFVector, mpf_t, MPFVector, mpf_t, MPFMatrix, void (* gfunc)(MPFVector, mpf_t, MPI_Comm), mpf_t, mpf_t, long int, MPI_Comm);

#endif

/********************************************/
/* mpi_ex_harmonic.c: Extrapolation with harmonic sequence */
/********************************************/

#define MPI_MAX_DEX_HARMONIC_STAGE 32

DVector dex_harmonic_local_y[MPI_MAX_DEX_HARMONIC_STAGE][MPI_MAX_DEX_HARMONIC_STAGE];
long int dex_harmonic_stage;
DVector dex_harmonic_ytmp[4], dex_harmonic_local_ytmp[4];
double dex_harmonic_in_h;
long int dex_harmonic_local_dim, dex_harmonic_ddim[MPI_GMP_MAXPROCS];

double ex_harmonic_den(long int istage, long int jstage);
void _mpi_init_dex_harmonic(long int stage, long int dimension, MPI_Comm comm);
void _mpi_clear_dex_harmonic(void);

int _mpi_dex_harmonic_1step(DVector, double, DVector, DVector, double, void (* func)(DVector, double, DVector, MPI_Comm), double, double, MPI_Comm);
void _mpi_dex_harmonic_fs(FILE *, double, DVector, double, DVector, long int, void (* func)(DVector, double, DVector, MPI_Comm), double, double, long int, MPI_Comm);
void _mpi_dex_harmonic(FILE *, double, DVector, double, DVector, double, void (* func)(DVector, double, DVector, MPI_Comm), double, double, long int, void (* ansfunc)(DVector, double), MPI_Comm comm);
int _mpi_dex_harmonic_1step_lo(DVector, double, DVector, DVector, double, DMatrix[], void (* gfunc)(DVector, double, MPI_Comm), double, double, MPI_Comm);
void _mpi_dex_harmonic_fs_lo(FILE *, double, DVector, double, DVector, long int, DMatrix, void (* gfunc)(DVector, double, MPI_Comm), double, double, long int, MPI_Comm);
void _mpi_dex_harmonic_lo(FILE *, double, DVector, double, DVector, double, DMatrix, void (* gfunc)(DVector, double, MPI_Comm), double, double, long int, MPI_Comm);

#ifdef USE_GMP

#define MPI_MAX_EX_HARMONIC_STAGE 32

/* inner variables for mpf_ex_harmonic */
MPFVector ex_harmonic_local_y[MPI_MAX_EX_HARMONIC_STAGE][MPI_MAX_EX_HARMONIC_STAGE];
long int ex_harmonic_stage;
MPFVector ex_harmonic_local_ytmp[4], ex_harmonic_ytmp[4];
mpf_t ex_harmonic_in_h, ex_harmonic_mpf_tmp;
long int ex_harmonic_local_dim, ex_harmonic_ddim[MPI_GMP_MAXPROCS];

void _mpi_init_mpf_ex_harmonic(long int stage, long int dimension, unsigned long prec, MPI_Comm comm);
void _mpi_clear_mpf_ex_harmonic(void);
//void mpf_ex_harmonic_den(mpf_t ret, long int istage, long int jstage);

int _mpi_mpf_ex_harmonic_1step(MPFVector, mpf_t, MPFVector, MPFVector, mpf_t, void (* func)(MPFVector, mpf_t, MPFVector, MPI_Comm), mpf_t, mpf_t, MPI_Comm);
void _mpi_mpf_ex_harmonic_fs(FILE *, mpf_t, MPFVector, mpf_t, MPFVector, long int, void (* func)(MPFVector, mpf_t, MPFVector, MPI_Comm), mpf_t, mpf_t, long int, MPI_Comm);
void _mpi_mpf_ex_harmonic(FILE *, mpf_t, MPFVector, mpf_t, MPFVector, mpf_t, void (* func)(MPFVector, mpf_t, MPFVector, MPI_Comm), mpf_t, mpf_t, long int, void (* ansfunc)(MPFVector, mpf_t), long int , MPI_Comm);

int _mpi_mpf_ex_harmonic_1step_lo(MPFVector, mpf_t, MPFVector, MPFVector, mpf_t, MPFMatrix[], void (* gfunc)(MPFVector, mpf_t, MPI_Comm), mpf_t, mpf_t, MPI_Comm);
void _mpi_mpf_ex_harmonic_fs_lo(FILE *, mpf_t, MPFVector, mpf_t, MPFVector, long int, MPFMatrix, void (* gfunc)(MPFVector, mpf_t, MPI_Comm), mpf_t, mpf_t, long int, MPI_Comm);
void _mpi_mpf_ex_harmonic_lo(FILE *, mpf_t, MPFVector, mpf_t, MPFVector, mpf_t, MPFMatrix, void (* gfunc)(MPFVector, mpf_t, MPI_Comm), mpf_t, mpf_t, long int, MPI_Comm);

#endif

/********************************************/
/* mpi_mharmonic.c: Extrapolation with Romberg and harmonic sequences */
/********************************************/

/* harmonic integral with modified trapezoidal rule*/
double _mpi_dmharmonic_integral_1step(double x_start, double x_end, double (* func)(double), long int num_stage, double reps, double aeps, long int *conv_flag, MPI_Comm comm);

/* harmonic integral with modified trapezoidal rule*/
void _mpi_dmharmonic_integral(double *ret, double x_start, double x_end, double (* func)(double), long int max_num_stage, MPI_Comm comm);

/* Romberg integral with modified trapezoidal rule */
double _mpi_dmromberg_integral_1step(double x_start, double x_end, double (* func)(double), long int num_stage, double reps, double aeps, long int *conv_flag, MPI_Comm comm);

/* Romberg integral with modified trapezoidal rule*/
void _mpi_dmromberg_integral(double *ret, double x_start, double x_end, double (* func)(double), long int max_num_stage, MPI_Comm comm);

#ifdef USE_GMP
/* (stage/step)^2 - 1 */
//void mpf_ex_harmonic1_den(mpf_t ret, long int istage, long int jstage);

/* (stage/step)^2 - 1 */
//void mpf_ex_harmonic2_den(mpf_t ret, long int istage, long int jstage);

/* (stage/step)^2 - 1 */
//void mpf_ex_harmonic3_den(mpf_t ret, long int istage, long int jstage);

/* harmonic integral with modified trapezoidal rule */
void _mpi_mpf_mharmonic_integral_1step(mpf_t ret, mpf_t x_start, mpf_t x_end, void (* func)(mpf_t, mpf_t), long int num_stage, mpf_t reps, mpf_t aeps, long int *conv_flag, MPI_Comm);

/* harmonic integral with modified trapezoidal rule*/
void _mpi_mpf_mharmonic_integral(mpf_t ret, mpf_t x_start, mpf_t x_end, void (* func)(mpf_t, mpf_t), long int max_num_stage, MPI_Comm comm);

/* Romberg integral with modified trapezoidal rule */
void _mpi_mpf_mromberg_integral_1step(mpf_t ret, mpf_t x_start, mpf_t x_end, void (* func)(mpf_t, mpf_t), long int num_stage, mpf_t reps, mpf_t aeps, long int *conv_flag, MPI_Comm comm);

/* Romberg integral with modified trapezoidal rule*/
void _mpi_mpf_mromberg_integral(mpf_t ret, mpf_t x_start, mpf_t x_end, void (* func)(mpf_t, mpf_t), long int max_num_stage, MPI_Comm comm);
#endif

#endif /* End of __MPI_BNC_H */
