/*************************************************/
/* Band Matrix Type:                             */
/*              FBMatrix, DBMatrix, MPFBMatrix   */
/*************************************************/
// define _BNC_MATMUL_STRASSEN_H
#ifndef _BNC_BMATRIX_H
#define _BNC_BMATRIX_H

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//#ifdef USE_GMP
	//#include "gmp.h"
	//#ifdef USE_MPFR
	//	#include "mpfr.h"
	//#endif // USE_MPFR
	#include "mpflinear.h"
//#endif // USE_GMP

#define BNC_SUCCESS (0)
#define BNC_ERROR (-1)

typedef struct{
//	unsigned int type;
	float *element;
	long int dim, upper_dim, lower_dim;
} fbmatrix;

typedef fbmatrix *FBMatrix;

typedef struct{
//	unsigned int type;
	double *element;
	long int dim, upper_dim, lower_dim;
} dbmatrix;

typedef dbmatrix *DBMatrix;

#ifdef USE_GMP
typedef struct{
	unsigned long int prec;
//	unsigned int type;
	mpf_t *element;
//	mpf_ptr *element;
	long int dim, upper_dim, lower_dim;
	mpf_t zero; // = 0
} mpfbmatrix;

typedef mpfbmatrix *MPFBMatrix;
#endif // USE_GMP

#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus

#endif // ifndef _BNC_BMATRIX_H
