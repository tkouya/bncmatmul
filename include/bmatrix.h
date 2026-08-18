/********************************************************************************/
/* bmatrix.h: Band matrix definitions and funtions                              */
/* Copyright (C) 2024 Tomonori Kouya                                            */
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
/********************************************************************************//**************************************************/
/* Band Matrix Type:                             */
/*              FBMatrix, DBMatrix, MPFBMatrix   */
/*************************************************/
// define _BNC_MATMUL_STRASSEN_H
#ifndef _BNC_BMATRIX_H
#define _BNC_BMATRIX_H

#include <stdio.h>
#include <math.h>

//#include "bnc.h"
//#include "flinear.h"
#include "dlinear.h"
#include "ddlinear.h"
#include "mpflinear.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// temporary
//typedef fmatrix *DBMatrix;
//#include "bmatrix.h"
typedef struct{
//	unsigned int type;
	float *element;
	long int dim, upper_dim, lower_dim;
} fbmatrix;

typedef fbmatrix *FBMatrix;

// temporary
//typedef dmatrix *DBMatrix;
//#include "bmatrix.h"
typedef struct{
//	unsigned int type;
	double *element;
	long int dim, upper_dim, lower_dim;
} dbmatrix;

typedef dbmatrix *DBMatrix;

/* float */

/* initialize band matrix */
FBMatrix init_fbmatrix(long int dim, long int upper_dim, long int lower_dim);

/* free dbmatrix */
void free_fbmatrix(FBMatrix mat);

/* return mat[i][j] */
float get_fbmatrix_ij(FBMatrix mat, long int row_index, long int col_index);

/* set mat[i][j] = val */
void set_fbmatrix_ij(FBMatrix mat, long int row_index, long int col_index, float val);

/* Multiply DBMatrix * DVector */
int mul_fbmatrix_fvec(FVector ret, FBMatrix mat, FVector vec);

/* double */

/* initialize band matrix */
DBMatrix init_dbmatrix(long int dim, long int upper_dim, long int lower_dim);

/* free dbmatrix */
void free_dbmatrix(DBMatrix mat);

/* return mat[i][j] */
double get_dbmatrix_ij(DBMatrix mat, long int row_index, long int col_index);

/* set mat[i][j] = val */
void set_dbmatrix_ij(DBMatrix mat, long int row_index, long int col_index, double val);

/* Multiply DBMatrix * DVector */
int mul_dbmatrix_dvec(DVector ret, DBMatrix mat, DVector vec);

/* Multiply DBMatrix^T * DVector */
int mul_dbmatrixt_dvec(DVector ret, DBMatrix mat, DVector vec);

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Band Matrix        */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 2012-07-01 (Sun) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DBLUdecomp(DBMatrix a);

/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                       (LU Decomposed Square Band Matrix) */
/*                                 (Double Precision)       */
/*                                                          */
/*                       ver. 0.0 2012-07-01 Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveDBLS(DVector answer, DBMatrix lu, DVector b);

/* print band matrix */
void print_dbmatrix(DBMatrix mat);

/* set zero matrix */
void set0_dbmatrix(DBMatrix mat);

/* set identity matrix */
void setI_dbmatrix(DBMatrix mat);

/* ret := mat_a + mat_b */
void add_dbmatrix(DBMatrix ret, DBMatrix a, DBMatrix b);

/* ret := mat_a - mat_b */
void sub_dbmatrix(DBMatrix ret, DBMatrix a, DBMatrix b);

/* ret := mat_a - mat_b */
void sub_dmatrix_dbmat_dmat(DMatrix ret, DBMatrix a, DMatrix b);

/* ret := val * a */
void cmul_dbmatrix(DBMatrix ret, double val, DBMatrix a);

/* ret := a * b */

/* ret := a * b */
void mul_dmatrix_dbmat_dbmat(DMatrix ret, DBMatrix a, DBMatrix b);

/* ret := a * b */
void mul_dmatrix_dmat_dbmat(DMatrix ret, DMatrix a, DBMatrix b);

/* ret := a * b */
void mul_dmatrix_dbmat_dmat(DMatrix ret, DBMatrix a, DMatrix b);

/* ret := a */
void subst_dbmatrix(DBMatrix ret, DBMatrix a);

/* ret := (DBMatrix)a */
void subst_fbmatrix_dbmat(FBMatrix ret, DBMatrix a);

/* ret := (FBMatrix)a */
void subst_dbmatrix_fbmat(DBMatrix ret, FBMatrix a);

/* ret := (DMatrix)a */
void subst_dmatrix_dbmat(DMatrix ret, DBMatrix a);


/***************************************/
/* ddbmatrix.c                        */
/**************************************/

// DD

typedef struct{
	double *element; // element[2]
	double zero[DDSIZE]; // = {0.0, 0.0}; // = 0
	long int dim, upper_dim, lower_dim;
} ddbmatrix;

typedef ddbmatrix *DDBMatrix;

/* initialize band matrix */
DDBMatrix init_ddbmatrix(long int dim, long int upper_dim, long int lower_dim);

/* free mpfbmatrix */
void free_ddbmatrix(DDBMatrix mat);

/* return mat[i][j] */
double *get_ddbmatrix_ij(DDBMatrix mat, long int row_index, long int col_index);

/* set mat[i][j] = val */
void set_ddbmatrix_ij(DDBMatrix mat, long int row_index, long int col_index, double val[DDSIZE]);

/* set mat[i][j] = val */
void set_ddbmatrix_ij_d(DDBMatrix mat, long int row_index, long int col_index, double val);

/* set mat[i][j] = val */
void set_ddbmatrix_ij_ui(DDBMatrix mat, long int row_index, long int col_index, unsigned long val);

/* Multiply DDBMatrix * MPFVector */
int mul_ddbmatrix_ddvec(DDVector ret, DDBMatrix mat, DDVector vec);

/* Multiply DDBMatrix^T * MPFVector */
int mul_ddbmatrixt_ddvec(DDVector ret, DDBMatrix mat, DDVector vec);

int DDBLUdecomp(DDBMatrix a);

int SolveDDBLS(DDVector answer, DDBMatrix lu, DDVector b);

/* print band matrix */
void print_ddbmatrix(DDBMatrix mat);

/* set zero matrix */
void set0_ddbmatrix(DDBMatrix mat);

/* set identity matrix */
void setI_ddbmatrix(DDBMatrix mat);

/* Frobenius norm */
void normf_ddbmatrix(double ret[DDSIZE], DDBMatrix mat);

/* ret := mat_a + mat_b */
void add_ddbmatrix(DDBMatrix ret, DDBMatrix a, DDBMatrix b);

/* ret := mat_a - mat_b */
void sub_ddbmatrix(DDBMatrix ret, DDBMatrix a, DDBMatrix b);

/* ret := mat_a - mat_b */
void sub_ddmatrix_ddbmat_ddmat(DDMatrix ret, DDBMatrix a, DDMatrix b);

/* ret := val * a */
void cmul_ddbmatrix(DDBMatrix ret, double val[DDSIZE], DDBMatrix a);

/* ret := a * b */
void mul_ddmatrix_ddbmat_ddbmat(DDMatrix ret, DDBMatrix a, DDBMatrix b);

/* ret := a * b */
void mul_ddmatrix_ddmat_ddbmat(DDMatrix ret, DDMatrix a, DDBMatrix b);

/* ret := a * b */
void mul_ddmatrix_ddbmat_ddmat(DDMatrix ret, DDBMatrix a, DDMatrix b);

/* ret := a */
void subst_ddbmatrix(DDBMatrix ret, DDBMatrix a);

/* ret := (DBMatrix)a */
void subst_dbmatrix_ddbmat(DBMatrix ret, DDBMatrix a);

/* ret := (DDBMatrix)a */
void subst_ddbmatrix_dbmat(DDBMatrix ret, DBMatrix a);

/* ret := (DDMatrix)a */
void subst_ddmatrix_ddbmat(DDMatrix ret, DDBMatrix a);

/* MPF */

#ifdef USE_GMP

/* initialize band matrix */
MPFBMatrix init_mpfbmatrix(long int dim, long int upper_dim, long int lower_dim);

/* initialize band matrix */
MPFBMatrix init2_mpfbmatrix(long int dim, long int upper_dim, long int lower_dim, unsigned long prec);

/* free mpfbmatrix */
void free_mpfbmatrix(MPFBMatrix mat);

/* return mat[i][j] */
mpf_ptr get_mpfbmatrix_ij(MPFBMatrix mat, long int row_index, long int col_index);

/* set mat[i][j] = val */
void set_mpfbmatrix_ij(MPFBMatrix mat, long int row_index, long int col_index, mpf_t val);

/* set mat[i][j] = val */
void set_mpfbmatrix_ij_d(MPFBMatrix mat, long int row_index, long int col_index, double val);

/* set mat[i][j] = val */
void set_mpfbmatrix_ij_ui(MPFBMatrix mat, long int row_index, long int col_index, unsigned long val);

/* Multiply MPFBMatrix * MPFVector */
int mul_mpfbmatrix_mpfvec(MPFVector ret, MPFBMatrix mat, MPFVector vec);

/* Multiply MPFBMatrix^T * MPFVector */
int mul_mpfbmatrixt_mpfvec(MPFVector ret, MPFBMatrix mat, MPFVector vec);

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Band Matrix        */
/*                                 (Multiple Precision)     */
/*                                                          */
/*                 ver. 0.0 2012-07-03 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int MPFBLUdecomp(MPFBMatrix a);

/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                       (LU Decomposed Square Band Matrix) */
/*                                 (Multiple Precision)     */
/*                                                          */
/*                       ver. 0.0 2012-07-01 Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveMPFBLS(MPFVector answer, MPFBMatrix lu, MPFVector b);

/* print band matrix */
void print_mpfbmatrix(MPFBMatrix mat);

/* set zero matrix */
void set0_mpfbmatrix(MPFBMatrix mat);

/* set identity matrix */
void setI_mpfbmatrix(MPFBMatrix mat);

/* ret := mat_a + mat_b */
void add_mpfbmatrix(MPFBMatrix ret, MPFBMatrix a, MPFBMatrix b);

/* ret := mat_a - mat_b */
void sub_mpfbmatrix(MPFBMatrix ret, MPFBMatrix a, MPFBMatrix b);

/* ret := mat_a - mat_b */
void sub_mpfmatrix_mpfbmat_mpfmat(MPFMatrix ret, MPFBMatrix a, MPFMatrix b);

/* ret := val * a */
void cmul_mpfbmatrix(MPFBMatrix ret, mpf_t val, MPFBMatrix a);

/* ret := a * b */
void mul_mpfmatrix_mpfbmat_mpfbmat(MPFMatrix ret, MPFBMatrix a, MPFBMatrix b);

/* ret := a * b */
void mul_mpfmatrix_mpfmat_mpfbmat(MPFMatrix ret, MPFMatrix a, MPFBMatrix b);

/* ret := a * b */
void mul_mpfmatrix_mpfbmat_mpfmat(MPFMatrix ret, MPFBMatrix a, MPFMatrix b);

/* ret := a */
void subst_mpfbmatrix(MPFBMatrix ret, MPFBMatrix a);

/* ret := (DBMatrix)a */
void subst_dbmatrix_mpfbmat(DBMatrix ret, MPFBMatrix a);

/* ret := (MPFBMatrix)a */
void subst_mpfbmatrix_dbmat(MPFBMatrix ret, DBMatrix a);

/* ret := (MPFMatrix)a */
void subst_mpfmatrix_mpfbmat(MPFMatrix ret, MPFBMatrix a);

#endif // USE_GMP

#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus

#endif // ifndef _BNC_BMATRIX_H
