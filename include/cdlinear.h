/********************************************************************************/
/* cdlinear.h: Double precision complex Vector, Matrix                          */
/* Copyright (c) 2024 Tomonori Kouya                                            */
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
// define __BNC_CLINEAR_H__
#ifndef __BNC_CDLINEAR_H__
  #define __BNC_CDLINEAR_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h> // float _Complex & double _Complex

// Common defs
#include "bnc_common.h" // [F,D,MPF]Complx

//#include "bnc.h"
#include "dlinear.h"
#include "flinear.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**********************************************/
/* cdlinear.c: Complex linear computation     */
/**********************************************/
// CDVector: double precision complex vector
// based on double _Complex defined in C99
typedef struct{
// 2018-02-26 by T.Kouya
	//double _Complex *element;
	double _Complex *element;
    //dcmplx *element;
	long int dim;
} cdvector;

typedef cdvector *CDVector;

// CDMatrix: double precision complex matrix
// based on double _Complex defined in C99
typedef struct{
	//double _Complex *element;
    double _Complex *element;
	//dcmplx *element;
	long int row_dim, col_dim;
} cdmatrix;

typedef cdmatrix *CDMatrix;

CDVector init_cdvector(long int);
void free_cdvector(CDVector);
// 2018-02-26 by T.Kouya
double _Complex get_cdvector_i(CDVector, long int);
double _Complex ip_cdvector(CDVector, CDVector);
double _Complex dotp_cdvector(CDVector, CDVector); // 2024-11-06 (Wed) appended
double _Complex get_cdmatrix_ij(CDMatrix, long int, long int);
void set_cdmatrix_ij(CDMatrix, long int, long int, double _Complex);
void set_cdvector_i(CDVector vec, long int, double _Complex);
void print_cdvector(CDVector);
void print_cdmatrix(CDMatrix);
void add_cdvector(CDVector, CDVector, CDVector);
void add2_cdvector(CDVector, CDVector);
void sub_cdvector(CDVector, CDVector, CDVector);
void sub2_cdvector(CDVector, CDVector);
void cmul_cdvector(CDVector, double _Complex, CDVector);
void cmul2_cdvector(CDVector, double _Complex);
void add_cmul_cdvector(CDVector, CDVector, double _Complex, CDVector);
void sub_cmul_cdvector(CDVector, CDVector, double _Complex, CDVector); // 2024-11-06 (Wed) appended
double norm1_cdvector(CDVector);
double norm2_cdvector(CDVector);
double normi_cdvector(CDVector);
void subst_cdvector(CDVector, CDVector);
void conj_cdvector(CDVector, CDVector);
void neg_cdvector(CDVector, CDVector);
void subst_cdvector_dvec(CDVector, DVector);
void subst_dvector_real_cdvec(DVector, CDVector);
void subst_dvector_image_cdvec(DVector, CDVector);
void set0_cdvector(CDVector);
void copy_cdvector_ij(CDVector, long int, long int, CDVector, long int, long int);
// ret_real + ret_image * I := src
void separate_cdvector(DVector, DVector, CDVector);
// ret := src_real + src_image * I
void merge_cdvector(CDVector, DVector, DVector);

CDMatrix init_cdmatrix(long int, long int);
void free_cdmatrix(CDMatrix);
void print_cdmatrix(CDMatrix);
double normf_cdmatrix(CDMatrix);
double normi_cdmatrix(CDMatrix);
double norm1_cdmatrix(CDMatrix);
void add_cdmatrix(CDMatrix, CDMatrix, CDMatrix);
void sub_cdmatrix(CDMatrix, CDMatrix, CDMatrix);
void cmul_cdmatrix(CDMatrix, double _Complex, CDMatrix);
void mul_cdmatrix(CDMatrix, CDMatrix, CDMatrix);
void transpose_cdmatrix(CDMatrix, CDMatrix);
void star_cdmatrix(CDMatrix, CDMatrix);
void subst_cdmatrix(CDMatrix, CDMatrix);
void conj_cdmatrix(CDMatrix, CDMatrix);
void neg_cdmatrix(CDMatrix, CDMatrix);
void subst_cdmatrix_dmat(CDMatrix, DMatrix);
void set0_cdmatrix(CDMatrix);
void setI_cdmatrix(CDMatrix);
void mul_cdmatrix_cdvec(CDVector, CDMatrix, CDVector);
void mul_cdmatrixt_cdvec(CDVector, CDMatrix, CDVector);

/* v = conj(a)^T * vb */
void mul_cdmatrixs_cdvec(CDVector v, CDMatrix a, CDVector vb);

void inv_cdmatrix(CDMatrix);
// ret_real + ret_image * I := src
void separate_cdmatrix(DMatrix, DMatrix, CDMatrix);
// ret := src_real + src_image * I
void merge_cdmatrix(CDMatrix, DMatrix, DMatrix);

/**********************************************/
/* cdlu.c : LU decomposition of complex matrix */
/**********************************************/
int CDLUdecomp(CDMatrix);
int SolveCDLS(CDVector, CDMatrix, CDVector);
int CDLUdecompP(CDMatrix, long int[]);
int SolveCDLSP(CDVector, CDMatrix, CDVector, long int[]);
int CDLUdecompC(CDMatrix, long int[], long int[]);
int SolveCDLSC(CDVector, CDMatrix, CDVector, long int[], long int[]);

#ifdef __cplusplus
} //extern "C" {
#endif // __cplusplus

// define __BNC_CDLINEAR_H__
#endif // ifndef __BNC_CDLINEAR_H__
