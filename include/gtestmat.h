/********************************************************************************/
/* gtestmat.h:                                                                  */
/* Copyright (C) 2003-2011 Tomonori Kouya                                       */
/*                                                                              */
/* Version 0.1: 2005.06/23 append arg "dim"                                     */
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
/************************************************/
/* Get Test Matrices                            */
/*                                              */
/* 1. Hilbert Matrix                            */
/* 2. Lotkin Matrix                             */
/* 3. Frank Matrix                              */
/* 4. Tridiagonal Matrix                        */
/* 5. Integer Symmetrix Random Matrix           */
/* 6. Integer Unsymmetrix Random Matrix         */
/* 7. Diagonal Matrix                           */
/* 8. Toeplitz Matrix                           */
/************************************************/
#ifndef __BNC_GTESTMAT_H_
#define __BNC_GTESTMAT_H_

//#include "bnc.h"
#include "dlinear.h"
#include "mpflinear.h"

/* Double */

/* 1. Hilbert Matrix */
void hilbert_dmatrix(DMatrix a, long int dim);

/* 2. Lotkin Matrix */
void lotkin_dmatrix(DMatrix a, long int dim);

/* 3. Frank Matrix */
void frank_dmatrix(DMatrix a, long int dim);

/* 4. Tridiagonal Matrix */
void tridiag_dmatrix(DMatrix a, DVector low_subdiag, DVector diag, DVector up_subdiag, long int dim);


/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_dmatrix(DMatrix mat, long int max, long int seed, long int dim);

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_dmatrix(DMatrix mat, long int max, long int seed, long int dim);

/* 7. Real Diagonal Matrix */
void diag_dmatrix(DMatrix mat, DVector diag, long int dim);

/* 8. Toeplitz Matrix */
void toeplitz_dmatrix(DMatrix mat, double gamma_param, long int dim);

/* MPF */
#ifdef USE_GMP

/* 1. Hilbert Matrix */
void hilbert_mpfmatrix(MPFMatrix a, long int dim);


/* 2. Lotkin Matrix */
void lotkin_mpfmatrix(MPFMatrix a, long int dim);

/* 3. Frank Matrix */
void frank_mpfmatrix(MPFMatrix a, long int dim);

/* 4. Tridiagonal Matrix */
void tridiag_mpfmatrix(MPFMatrix a, MPFVector low_subdiag, MPFVector diag, MPFVector up_subdiag, long int dim);

/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_mpfmatrix(MPFMatrix mat, long int max, long int seed, long int dim);

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_mpfmatrix(MPFMatrix mat, long int max, long int seed, long int dim);

/* 7. Real Diagonal Matrix */
void diag_mpfmatrix(MPFMatrix mat, MPFVector diag, long int dim);

/* 8. Toeplitz Matrix */
void toeplitz_mpfmatrix(MPFMatrix mat, mpf_t gamma_param, long int dim);

#endif // USE_GMP

#endif // __BNC_GTESTMAT_H_
