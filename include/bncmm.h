/********************************************************************************/
/*                                                                              */
/* bncmm.h : Header file for Reading & Writing MatrixMarket Format Files        */
/* Copyright (c) 2006-2011 Tomonori Kouya, All rights reserved.                 */
/*                                                                              */
/* 2011-08-26 Version 0.0: make bncmm.h and readmatrix.c                        */
/* 2011-08-29 Version 0.1: append writematrix.c                                 */
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
#include <string.h>
#include <malloc.h>
#include "mmio.h" // From MatrixMarket

//#include "bnc.h"
#include "dlinear.h"
#include "mpflinear.h"
//#include "clinear.h"
#include "cdlinear.h"
#include "cmpflinear.h"

#include "bncsparse.h"

#ifndef __BNCMM_H
#define __BNCMM_H

/* double */

// read MatrixMarket format (coodinate type only) as vector
DVector init_dvector_readMMcoordinate(const char *);

// read MatrixMarket format (coodinate type only) as dense matrix
DMatrix init_dmatrix_readMMcoordinate(const char *);

// MatrixMarket(Coordinate) struct
typedef struct {
	int row_index;
	int col_index;
	double val;
} mmcoordinate;

// compare mmcoordinate
int compare_mmcoordinate(const void *, const void *);

// read MatrixMarket format (coodinate type only) as sparse matrix
DRSMatrix init_drsmatrix_readMMcoordinate(const char *);

// writer matrix as MatrixMarket format (coodinate type) 
int writeMMcoordinate_dmatrix(const char *, DMatrix);

// writer vector as MatrixMarket format (coodinate type) 
int writeMMcoordinate_dvector(const char *, DVector);

// writer vector as MatrixMarket format (array type) 
int writeMMarray_dvector(const char *, DVector);

// compare mmcoordinate to sort as columnwise
int compare_mmcoordinate_col(const void *, const void *);

// writer matrix as MatrixMarket format (coodinate type) 
int writeMMcoordinate_drsmatrix(const char *, DRSMatrix);

/* MPFR */

#ifdef USE_GMP

//#define MAX_VAL_STR 512
#define MAX_VAL_STR 4096
//#define MAX_VAL_STR 8192
//#define MAX_VAL_STR 16384

/* real body for init* functions */
MPFVector _init2_mpfvector_readMMcoordinate(const char *, unsigned long);
MPFMatrix _init2_mpfmatrix_readMMcoordinate(const char *, unsigned long);
MPFRSMatrix _init2_mpfrsmatrix_readMMcoordinate(const char *, unsigned long);

// read MatrixMarket format (coodinate type only) as vector if possible
MPFVector init2_mpfvector_readMMcoordinate(const char *, unsigned long);

// read MatrixMarket format (coodinate type only) as vector if possible
MPFVector init_mpfvector_readMMcoordinate(const char *);

// read MatrixMarket format (coodinate type only) as dense matrix
MPFMatrix init2_mpfmatrix_readMMcoordinate(const char *, unsigned long);

// read MatrixMarket format (coodinate type only) as dense matrix
MPFMatrix init_mpfmatrix_readMMcoordinate(const char *);

// MatrixMarket(Coordinate) struct
typedef struct {
	int row_index;
	int col_index;
	char val_str[MAX_VAL_STR];
} mmcoordinate_str;

// MatrixMarket(Coordinate) struct
typedef struct {
	int row_index;
	int col_index;
	mpf_ptr val;
} mmcoordinate_mpf;

// compare mmcoordinate
int compare_mmcoordinate_str(const void *, const void *);

// read MatrixMarket format (coodinate type only) as sparse matrix
MPFRSMatrix init2_mpfrsmatrix_readMMcoordinate(const char *, unsigned long);

// read MatrixMarket format (coodinate type only) as sparse matrix
MPFRSMatrix init_mpfrsmatrix_readMMcoordinate(const char *);

// writer vector as MatrixMarket format (coodinate type) 
int writeMMcoordinate_mpfvector(const char *, MPFVector);

// writer vector as MatrixMarket format (array type) 
int writeMMarray_mpfvector(const char *, MPFVector);

// writer matrix as MatrixMarket format (coodinate type) 
int writeMMcoordinate_mpfmatrix(const char *, MPFMatrix);

// compare mmcoordinate to sort as columnwise
int compare_mmcoordinate_mpf_col(const void *, const void *);

// writer matrix as MatrixMarket format (coodinate type) 
int writeMMcoordinate_mpfrsmatrix(const char *, MPFRSMatrix);

#endif // USE_GMP

#endif // __BNCMM_H
