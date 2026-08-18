/********************************************************************************/
/* block_linear_band.c: Linear computations with block band matrix              */
/* Copyright (c) 2016 Tomonori Kouya                                            */
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
/**************************************************/
/* Implicit Runge-Kutta Method based on Jay's way */
/**************************************************/
#include <stdio.h>
#include <math.h>
#include <omp.h>

#include "bnc.h"

#ifdef USE_ATLAS
  #include "clapack.h"
#elif USE_IMKL
  #include "mkl_types.h"
  #include "mkl_cblas.h"
  #include "mkl_lapacke.h"
#endif

//#include "birk.h"

#ifndef IRKG_MAX_STAGE
	#define IRKG_MAX_STAGE 12
#endif

#ifdef USE_PTHREAD
\\#include "birk_pthread.h"
#endif

/* ret := sum^n_{i,j=1} aij^2 */
double sumsqr_dbmatrix(DBMatrix mat)
{
	long int i, j, min_j, max_j;
	double ret, tmp;

	ret = 0.0;

	for(i = 0; i < mat->dim; i++)
	{
		// Lower triangular element
		min_j = i - mat->lower_dim;
		if(min_j < 0)
			min_j = 0;

		// Upper triangular element
		max_j = i + mat->upper_dim + 1;
		if(max_j > mat->dim)
			max_j = mat->dim;

		for(j = min_j; j < max_j; j++)
		{
			tmp = get_dbmatrix_ij(mat, i, j);
			ret += tmp * tmp;
		}

	}

	return ret;
}

/* Frobenius Norm for tridiagonal block matrix */
double normf_dbmatrix_tridiag_blocks(DBMatrix aim1[], DBMatrix aii[], DBMatrix aip1[], long int num_blocks)
{
	long int index;
	double ret;

	ret = 0.0;

	if(num_blocks < 1)
		return ret;

	ret += sumsqr_dbmatrix(aii [0]);
	if(num_blocks >= 2)
	{
		ret += sumsqr_dbmatrix(aip1[0]);
		#pragma omp parallel for reduction(+:ret)
		for(index = 1; index <= num_blocks - 2; index++)
		{
			ret += sumsqr_dbmatrix(aim1[index - 1]);
			ret += sumsqr_dbmatrix(aii [index    ]);
			ret += sumsqr_dbmatrix(aip1[index    ]);
		}
		ret += sumsqr_dbmatrix(aim1[num_blocks - 2]);
		ret += sumsqr_dbmatrix(aii [num_blocks - 1]);
	}

	return sqrt(ret);
}

/* [ ret[0]              ]    [ b[0]              ]   [ aii [0] aip1[0]                                                   ]   [ x[0]              ] */
/* [ ret[1]              ]    [ b[1]              ]   [ aim1[0] aii [1] aip1[1]                                           ]   [ x[1]              ] */
/* [ ................... ] := [ ................. ] - [ ................................................................. ] * [ ................. ] */
/* [ ret[num_blocks - 2] ]    [ b[num_blocks - 2] ]   [    aim1[num_blocks - 3] aii [num_blocks - 2] aip1[num_blocsk - 2] ]   [ x[num_blocks - 2] ] */
/* [ ret[num_blocks - 1] ]    [ b[num_blocks - 1] ]   [                         aim1[num_blocks - 2] aii [num_blocks - 1] ]   [ x[num_blocks - 1] ] */
void residual_dbmat_dvec_tridiag_blocks(DVector ret[], DVector b[], DBMatrix aim1[], DBMatrix aii[], DBMatrix aip1[], DVector x[], long int num_blocks)
{
	long int index;
	DVector tmpvec[4];

	// initialize
	tmpvec[0] = init_dvector(ret[0]->dim);
	tmpvec[1] = init_dvector(ret[0]->dim);
	tmpvec[2] = init_dvector(ret[0]->dim);
	tmpvec[3] = init_dvector(ret[0]->dim);

//	mul_dmatrix_dvec(r, mat, x);
//	sub_dvector(r, b, r);

	// ret[0] := b[0] - (aii[0] * x[0] + aip1[0] * x[1])
	mul_dbmatrix_dvec(tmpvec[0], aii [0], x[0]);
	mul_dbmatrix_dvec(tmpvec[1], aip1[0], x[1]);
	add_dvector(tmpvec[2], tmpvec[0], tmpvec[1]);
	sub_dvector(ret[0], b[0], tmpvec[2]);

	// ret[index] ;= b[index] - ( aim1[index - 1] * x[index - 1] + aii[index] * x[index] + aip1[index] * x[index + 1] )

	#pragma omp parallel for
	for(index = 1; index <= num_blocks - 2; index++)
	{
		mul_dbmatrix_dvec(tmpvec[0], aim1[index - 1], x[index - 1]);
		mul_dbmatrix_dvec(tmpvec[1], aii [index]    , x[index]    );
		mul_dbmatrix_dvec(tmpvec[2], aip1[index]    , x[index + 1]);
		add_dvector(tmpvec[3], tmpvec[0], tmpvec[1]);
		add_dvector(tmpvec[3], tmpvec[3], tmpvec[2]);
		sub_dvector(ret[index], b[index], tmpvec[3]);
	}

	if(num_blocks >= 2)
	{
		// ret[num_blocks - 1] := b[num_blocks - 1] - ( aim1[num_blocks - 2] * x[num_blocks - 2] + aii[num_blocks - 1] * x[num_blocks - 1] )
		mul_dbmatrix_dvec(tmpvec[0], aim1[num_blocks - 2], x[num_blocks - 2]);
		mul_dbmatrix_dvec(tmpvec[1], aii [num_blocks - 1], x[num_blocks - 1]);
		add_dvector(tmpvec[2], tmpvec[0], tmpvec[1]);
		sub_dvector(ret[num_blocks - 1], b[num_blocks - 1], tmpvec[2]);
	}

	// free
	free_dvector(tmpvec[0]);
	free_dvector(tmpvec[1]);
	free_dvector(tmpvec[2]);
	free_dvector(tmpvec[3]);
}

/* mul vec */
/* ret[] := a[] * x[] */
void mul_dbmat_dvec_tridiag_blocks(DVector ret[], DBMatrix aim1[], DBMatrix aii[], DBMatrix aip1[], DVector x[], long int num_blocks)
{
	long int index;
	DVector tmpvec[4];

#pragma omp parallel shared(ret, aim1, aii, aip1, x, num_blocks)
{ // openmp block

	// initialize
	tmpvec[0] = init_dvector(ret[0]->dim);
	tmpvec[1] = init_dvector(ret[0]->dim);
	tmpvec[2] = init_dvector(ret[0]->dim);

//	mul_dmatrix_dvec(r, mat, x);
//	sub_dvector(r, b, r);

	// ret[0] := aii[0] * x[0] + aip1[0] * x[1]
	mul_dbmatrix_dvec(tmpvec[0], aii [0], x[0]);
	mul_dbmatrix_dvec(tmpvec[1], aip1[0], x[1]);
	add_dvector(ret[0], tmpvec[0], tmpvec[1]);

	// ret[index] ;= b[index] - ( aim1[index - 1] * x[index - 1] + aii[index] * x[index] + aip1[index] * x[index + 1] )
	#pragma omp parallel for
	for(index = 1; index <= num_blocks - 2; index++)
	{
		mul_dbmatrix_dvec(tmpvec[0], aim1[index - 1], x[index - 1]);
		mul_dbmatrix_dvec(tmpvec[1], aii [index]    , x[index]    );
		mul_dbmatrix_dvec(tmpvec[2], aip1[index]    , x[index + 1]);
		add_dvector(ret[index], tmpvec[0], tmpvec[1]);
		add_dvector(ret[index], ret[index], tmpvec[2]);
	}


	if(num_blocks >= 2)
	{
		// ret[num_blocks - 1] := b[num_blocks - 1] - ( aim1[num_blocks - 2] * x[num_blocks - 2] + aii[num_blocks - 1] * x[num_blocks - 1] )
		mul_dbmatrix_dvec(tmpvec[0], aim1[num_blocks - 2], x[num_blocks - 2]);
		mul_dbmatrix_dvec(tmpvec[1], aii [num_blocks - 1], x[num_blocks - 1]);
		add_dvector(ret[num_blocks - 1], tmpvec[0], tmpvec[1]);
	}

	// free
	free_dvector(tmpvec[0]);
	free_dvector(tmpvec[1]);
	free_dvector(tmpvec[2]);
} // end openmp block
}

/* mul vec */
/* ret[] := a[] * vb[] */
void mul_dbtridiag_dvec_blocks(DVector ret[], DBMatrix aim1[], DBMatrix aii[], DBMatrix aip1[], DVector vb[], DVector tmpv[], long int num_blocks)
{
	long int block_i;

	mul_dbmatrix_dvec( ret[0], aii [0], vb[0]);

	if(num_blocks >= 2)
	{
		mul_dbmatrix_dvec(tmpv[0], aip1[0], vb[1]);
		add_dvector(ret[0], ret[0], tmpv[0]);

		for(block_i = 1; block_i < num_blocks - 1; block_i++)
		{
			mul_dbmatrix_dvec(ret[block_i], aim1 [block_i - 1], vb[block_i - 1]);
			mul_dbmatrix_dvec(  tmpv[block_i], aii  [block_i    ], vb[block_i    ]);
			add_dvector(ret[block_i], ret[block_i], tmpv[block_i]);
			mul_dbmatrix_dvec(  tmpv[block_i], aip1 [block_i    ], vb[block_i + 1]);
			add_dvector(ret[block_i], ret[block_i], tmpv[block_i]);
		}

		mul_dbmatrix_dvec( ret[num_blocks - 1], aim1 [num_blocks - 2], vb[num_blocks - 2]);
		mul_dbmatrix_dvec(tmpv[num_blocks - 1], aii  [num_blocks - 1], vb[num_blocks - 1]);
		add_dvector(ret[num_blocks - 1], ret[num_blocks - 1], tmpv[num_blocks - 1]);
	}
}

void residual_dbmat_dvec_tridiag_blocks2(DVector ret[], DVector b[], DBMatrix aim1[], DBMatrix aii[], DBMatrix aip1[], DVector x[], long int num_blocks)
{
	mul_dbmat_dvec_tridiag_blocks(ret, aim1, aii, aip1, x, num_blocks);
	sub_dvector_blocks(ret, b, ret, num_blocks);
}

// substitution of tridiagonal block matrix
void subst_fbmatrix_dbmat_tridiag_blocks(FBMatrix aim1_f[], FBMatrix aii_f[], FBMatrix aip1_f[], DBMatrix aim1[], DBMatrix aii[], DBMatrix aip1[], long int num_blocks)
{
	long int index;

	if(num_blocks < 1)
		return;

 	subst_fbmatrix_dbmat(aii_f [0], aii [0]);
	if(num_blocks >= 2)
	{
		subst_fbmatrix_dbmat(aip1_f[0], aip1[0]);
		for(index = 1; index < num_blocks - 1; index++)
		{
			subst_fbmatrix_dbmat(aim1_f[index - 1], aim1[index - 1]);
			subst_fbmatrix_dbmat(aii_f [index]    , aii [index]);
			subst_fbmatrix_dbmat(aip1_f[index]    , aip1[index]);
		}
		subst_fbmatrix_dbmat(aim1_f[num_blocks - 2], aim1[num_blocks - 2]);
		subst_fbmatrix_dbmat(aii_f [num_blocks - 1], aii [num_blocks - 1]);
	}
}

/************************************************************/
/*                                                          */
/*             BiCGSTAB Method for Real Matrix              */
/*                         for Real Block TridiagonalMatrix */
/*                                                          */
/*                 ver. 0.0 2012-03-01 (Thu) Tomonori Kouya */
/*                                                          */
/************************************************************/
long int DBBiCGSTAB_triblock(DVector answer[], DBMatrix aim1[], DBMatrix aii[], DBMatrix aip1[], DVector b[], double reps, double aeps, long int maxtimes, long int num_blocks)
/******************************************************************************/
/*                                                                            */
/* ENTRIES                                                                    */
/*       DVector answer: Solution for Ax = b such as                          */
/*                                                                            */
/*       [aii [0] aip1[0]                             ] [x[0]   ]   [b[0]   ] */
/*       [aim1[0] aii [1] aip1[1]                     ] [x[1]   ]   [b[1]   ] */
/*       [        .......................             ] [ ...   ] = [....   ] */
/*       [           aim1[nb-3] aii [nb-2] aip1[nb-2] ] [x[nb-2]]   [b[nb-2]] */
/*       [                      aim1[nb-2] aii [nb-1] ] [x[nb-1]]   [b[nb-1]] */
/*                                                                            */
/*       DVector b: Constant vector b   (given by user)                       */
/*       double reps: Relative tolerance (given by user)                      */
/*       double aeps: Absolute tolerance (given by user)                      */
/*       long int maxtimes: Maximum iterative times (given by user)           */
/*       long int num_blocks: Number of Blocks(= nb) (given by user)          */
/*                                                                            */
/* RETURNS                                                                    */
/*       DVector answer: Solution for Ax = b                                  */
/*                                                                            */
/* ERRORS                                                                     */
/* Positive value ... Normal : Iterative Times                                */
/*      -1 ... Rho is zero.                                                   */
/*      -2 ... Denominator of Alpha is zero.                                  */
/*      -3 ... Denominator of Omega is zero.                                  */
/*      -4 ... Numerator of Omega is zero.                                    */
/*      -5 ... Not Converge.                                                  */
/*                                                                            */
/******************************************************************************/
{
	long int i, j, times, dim, block_i, retcode;
	double alpha, alpha_num, alpha_den;
	double beta, beta_num;
	double rho, old_rho;
	double omega, omega_den;
	double dtmp, init_resnorm;
	DVector *vec[9], *tmpv; /* Temporary Vectors */

/* Set initial value */
	tmpv = calloc(sizeof(dvector), num_blocks);
	for(i = 0; i < 9; i++)
		vec[i] = calloc(sizeof(dvector), num_blocks);

	for(block_i = 0; block_i < num_blocks; block_i++)
	{
		tmpv[block_i] = init_dvector(answer[block_i]->dim);

		for(i = 0; i < 9; i++)
			vec[i][block_i] = init_dvector(answer[block_i]->dim);
	}

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... v */
	/* vec[6] ... s */
	/* vec[7] ... s^T */
	/* vec[8] ... t */

	subst_dvector_blocks(vec[1], b, num_blocks); 
	subst_dvector_blocks(vec[2], b, num_blocks);

	beta_num = ip_dvector_blocks(vec[1], vec[1], num_blocks);
	init_resnorm = sqrt(beta_num);

	old_rho = 0.0;
	rho = 0.0;

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		rho = ip_dvector_blocks(vec[2], vec[1], num_blocks);

		if(rho == 0.0)
		{
			fprintf(stderr, "Rho is zero!(DBiCGSTAB, %ld)\n", times);
			retcode = -1;
			break;
			//return -1;
		}

		if(times == 0)
		{
			/* p := r */
			subst_dvector_blocks(vec[3], vec[1], num_blocks);
		}
		else
		{
			beta = (rho / old_rho) * (alpha / omega);

			/* p := r + beta (p - omega v) */
			add_cmul_dvector_blocks(vec[4], vec[3], -omega, vec[5], num_blocks);
			add_cmul_dvector_blocks(vec[3], vec[1], beta, vec[4], num_blocks);
		}
		/* precondition */
		
		/* v := Apt */
		//mul_dmatrix_dvec(vec[5], a, vec[3]);
	//#ifdef USE_PTHREAD
	//	_pthread_mul_dbtridiag_dvec_blocks(vec[5], aim1, aii, aip1, vec[3], tmpv, num_blocks, MAX_PTHREAD_NUM);
	//#else
#ifdef _OPENMP
		_bncomp_mul_dbtridiag_dvec_blocks(vec[5], aim1, aii, aip1, vec[3], tmpv, num_blocks);
#else
		mul_dbtridiag_dvec_blocks(vec[5], aim1, aii, aip1, vec[3], tmpv, num_blocks);
#endif
	//#endif

		alpha_den = ip_dvector_blocks(vec[2], vec[5], num_blocks);
		if(alpha_den == 0.0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(DBiCGSTAB, %ld)\n", times);
			//return -2;
			retcode = -2;
			break;
		}
		alpha = rho / alpha_den;

		/* s = r - alpha v */
		add_cmul_dvector_blocks(vec[6], vec[1], -alpha, vec[5], num_blocks);

		/* Stopping Criteria */
		dtmp = norm2_dvector_blocks(vec[6], num_blocks);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			/* x = x + alpha pt */
			add_cmul_dvector_blocks(vec[0], vec[0], alpha, vec[3], num_blocks);

			subst_dvector_blocks(answer, vec[0], num_blocks);

			/* free vec[0]..[3]; */
		/*	for(block_i = 0; block_i < num_blocks; block_i++)
			{
				free_dvector(tmpv[block_i]);

				for(i = 0; i < 9; i++)
					free_dvector(vec[i][block_i]);
			}
		*/
		//	free(tmpv);
		//	for(i = 0; i < 9; i++) free(vec[i]);

			//return times;
			retcode = times;
			break;
		}

		/* precondition */

		//mul_dmatrix_dvec(vec[8], a, vec[6]);
	//#ifdef USE_PTHREAD
	//	_pthread_mul_dbtridiag_dvec_blocks(vec[8], aim1, aii, aip1, vec[6], tmpv, num_blocks, MAX_PTHREAD_NUM);
	//#else
#ifdef _OPENMP
		_bncomp_mul_dbtridiag_dvec_blocks(vec[8], aim1, aii, aip1, vec[6], tmpv, num_blocks);
#else
		mul_dbtridiag_dvec_blocks(vec[8], aim1, aii, aip1, vec[6], tmpv, num_blocks);
#endif
	//#endif

		/* omega = (t, s) / (t, t) */
		omega_den = ip_dvector_blocks(vec[8], vec[8], num_blocks);
		if(omega_den == 0)
		{
			fprintf(stderr, "Denominator of Omega is zero!(DBiCGSTAB, %ld)\n", times);
			//return -3;
			retcode = -3;
			break;
		}
		omega = ip_dvector_blocks(vec[8], vec[6], num_blocks);
		if(omega == 0)
		{
			fprintf(stderr, "Numerator of Omega is zero!(DBiCGSTAB, %ld)\n", times);
			//return -4;
			retcode = -4;
			break;
		}
		omega = omega / omega_den;

		/* x = x + alpha pt + omega st */
		add_cmul_dvector_blocks(vec[4], vec[0], alpha, vec[3], num_blocks);
		add_cmul_dvector_blocks(vec[0], vec[4], omega, vec[6], num_blocks);

		/* residual */
		add_cmul_dvector_blocks(vec[1], vec[6], -omega, vec[8], num_blocks);

		beta_num = ip_dvector_blocks(vec[1], vec[1], num_blocks);

		/* Stopping Criteria */
		dtmp = sqrt(beta_num);
		if(dtmp <= aeps + reps * init_resnorm)
		{
		//	subst_dvector_blocks(answer, vec[0], num_blocks);

			/* free vec[0]..[3]; */
		/*	for(block_i = 0; block_i < num_blocks; block_i++)
			{
				free_dvector(tmpv[block_i]);

				for(i = 0; i < 9; i++)
					free_dvector(vec[i][block_i]);
			}
		*/	//free(tmpv);
			//for(i = 0; i < 9; i++) free(vec[i]);

			//return times;
			retcode = times;
			break;
		}
		//printf("||r_k||_2 = %25.17e\n", dtmp);

		old_rho = rho;
	}

	/* Not converge */
	if(times >= maxtimes)
	{
		retcode = -5;
		fprintf(stderr, "Not converge!(DBiCGSTAB, %ld)\n", times);
	}

//END_DBiCGSTAB_TRIBLOCK:

	subst_dvector_blocks(answer, vec[0], num_blocks);

	/* free vec[0]..[3]; */
	for(block_i = 0; block_i < num_blocks; block_i++)
	{
		free_dvector(tmpv[block_i]);

		for(i = 0; i < 9; i++)
				free_dvector(vec[i][block_i]);
	}
	free(tmpv);
	for(i = 0; i < 9; i++)
		free(vec[i]);

	return retcode;

}



// Block LU decomposition for tridiagonal block matrix
int FBLUdecomp_triblock(FMatrix hmat_lu[], FBMatrix aim1[], FBMatrix aii[], FBMatrix aip1[], long int num_blocks)
{
	long int index;
	int retcode = 0;
	FMatrix tmpmat[2];

	if(num_blocks < 1)
	{
		fprintf(stderr, "ERROR: FBLUdecomp_triblock's numblock is less than 1!\n");
		return -1;
	}

	// initialize
	tmpmat[0] = init_fmatrix(aii[0]->dim, aii[0]->dim);
	tmpmat[1] = init_fmatrix(aii[0]->dim, aii[0]->dim);

	// h[0] = aii[0]
	subst_fmatrix_fbmat(hmat_lu[0], aii[0]);
	if(FLUdecomp(hmat_lu[0]) < -1)
	{
		fprintf(stderr, "ERROR: FBLUdecomp_triblock(0, FLUdecomp) is fail!\n");
		retcode = -1;
		goto END_FBLUDECOMP_TRIBLOCK;
		//return -1;
	}

	// h[i] := aii[i] - aim1[i-1] * aii[i-1]^(-1) * aip1[i-1]
	for(index = 1; index < num_blocks; index++)
	{
		// hmat[index] := aii[index] - aim1[index - 1] * hmat[index - 1]^(-1) * aip1[index - 1]
		if(SolveFLSs_fbmat(tmpmat[0], hmat_lu[index - 1], aip1[index - 1]) < 0)
		{
			fprintf(stderr, "ERROR: FBLUdecomp_triblock(%ld, SolveFLSs) is fail!\n", index);
		//	return -1;
			retcode = -1;
			goto END_FBLUDECOMP_TRIBLOCK;
		}
		mul_fmatrix_fbmat_fmat(tmpmat[1], aim1[index - 1], tmpmat[0]);
		sub_fmatrix_fbmat_fmat(hmat_lu[index], aii[index], tmpmat[1]);

		if(FLUdecomp(hmat_lu[index]) < 0)
		{
			fprintf(stderr, "ERROR: FLUdecomp_triblock(%ld, FLUdecomp) is fail!\n", index);
		//	return -1;
			retcode = -1;
			goto END_FBLUDECOMP_TRIBLOCK;
		}
	}

END_FBLUDECOMP_TRIBLOCK:

	// free
	free_fmatrix(tmpmat[0]);
	free_fmatrix(tmpmat[1]);

	return retcode;
}

// Solve linear systems of equation with block LU decomposed matrix
int SolveFBLS_triblock(FVector answer[], FBMatrix aim1[], FMatrix hmat_lu[], FBMatrix aip1[], FVector b[], long int num_blocks)
{
	int retcode = 0;
	long int index, dim;
	FVector tmpvec[2];
	FVector y[IRKG_MAX_STAGE];

	if(num_blocks < 1)
		return -1;

	// initialize
	dim = answer[0]->dim;
	tmpvec[0] = init_fvector(dim);
	tmpvec[1] = init_fvector(dim);
	for(index = 0; index < num_blocks; index++)
		y[index] = init_fvector(dim);

	// Forward substitution
	subst_fvector(y[0], b[0]);

	for(index = 1; index < num_blocks; index++)
	{
		// y[index] := y[index] - aip1[index - 1] * hmat[index - 1]^(-1) * y[index - 1]
		if(SolveFLS(tmpvec[0], hmat_lu[index - 1], y[index - 1]) < 0)
		{
			fprintf(stderr, "ERROR: SolveFBLS_triblock(%ld, Forward) is fail!\n", index);
			//return -1;
			retcode = -1;
			goto END_SOLVEFBLS_TRIBLOCK;
		}
		mul_fbmatrix_fvec(tmpvec[1], aim1[index - 1], tmpvec[0]);
		sub_fvector(y[index], b[index], tmpvec[1]);
	}

	// Backward substitution
	if(SolveFLS(answer[num_blocks - 1], hmat_lu[num_blocks - 1], y[num_blocks - 1]) < 0)
	{
		fprintf(stderr, "ERROR: SolveFBLS_triblock(%ld, Backword) is fail!\n", num_blocks);
		//return -1;
		retcode = -1;
		goto END_SOLVEFBLS_TRIBLOCK;

	}

	if(num_blocks >= 2)
	{
		for(index = num_blocks - 2; index >= 0; index--)
		{
			// x[index] := hmat[index]^(-1) * (y[index] - aip1[index] * x[index + 1])
			mul_fbmatrix_fvec(tmpvec[0], aip1[index], answer[index + 1]);
			sub_fvector(tmpvec[1], y[index], tmpvec[0]);
			if(SolveFLS(answer[index], hmat_lu[index], tmpvec[1]) < 0)
			{
				fprintf(stderr, "ERROR: SolveFBLS_triblock(%ld, backword) is fail!\n", index);
				//return -1;
				retcode = -1;
				goto END_SOLVEFBLS_TRIBLOCK;
			}
		}
	}

END_SOLVEFBLS_TRIBLOCK:

	// free
	free_fvector(tmpvec[0]);
	free_fvector(tmpvec[1]);
	for(index = 0; index < num_blocks; index++)
		free_fvector(y[index]);

	return retcode;
}

// Block LU decomposition for tridiagonal block matrix
int DBLUdecomp_triblock(DMatrix hmat_lu[], DBMatrix aim1[], DBMatrix aii[], DBMatrix aip1[], long int num_blocks)
{
	int retcode = 0;
	long int index;
	DMatrix tmpmat[2];

	if(num_blocks < 1)
		return -1;

	// initialize
	tmpmat[0] = init_dmatrix(aii[0]->dim, aii[0]->dim);
	tmpmat[1] = init_dmatrix(aii[0]->dim, aii[0]->dim);

	// h[0] = aii[0]
	subst_dmatrix_dbmat(hmat_lu[0], aii[0]);
	if(DLUdecomp(hmat_lu[0]) < 0)
	{
		fprintf(stderr, "ERROR: DBLUdecomp_triblock(0, DLUdecomp) is fail!\n");
		//return -1;
		retcode = -1;
		goto END_DBLUDECOMP_TRIBLOCK;
	}

	// h[i] := aii[i] - aim1[i-1] * aii[i-1]^(-1) * aip1[i-1]
	for(index = 1; index < num_blocks; index++)
	{
		// hmat[index] := aii[index] - aim1[index - 1] * hmat[index - 1]^(-1) * aip1[index - 1]
		if(SolveDLSs_dbmat(tmpmat[0], hmat_lu[index - 1], aip1[index - 1]) < 0)
		{
			fprintf(stderr, "ERROR: DBLUdecomp_triblock(%ld, SolveDLSs) is fail!\n", index);
			retcode = -1;
			goto END_DBLUDECOMP_TRIBLOCK;
		}
		mul_dmatrix_dbmat_dmat(tmpmat[1], aim1[index - 1], tmpmat[0]);
		sub_dmatrix_dbmat_dmat(hmat_lu[index], aii[index], tmpmat[1]);

		if(DLUdecomp(hmat_lu[index]) < 0)
		{
			fprintf(stderr, "ERROR: DBLUdecomp_triblock(%ld, DLUdecomp) is fail!\n", index);
			retcode = -1;
			goto END_DBLUDECOMP_TRIBLOCK;
		}
	}

END_DBLUDECOMP_TRIBLOCK:

	// free
	free_dmatrix(tmpmat[0]);
	free_dmatrix(tmpmat[1]);

	return retcode;
}

// Solve linear systems of equation with block LU decomposed matrix
int SolveDBLS_triblock(DVector answer[], DBMatrix aim1[], DMatrix hmat_lu[], DBMatrix aip1[], DVector b[], long int num_blocks)
{
	int retcode = 0;
	long int index, dim;
	DVector tmpvec[2];
	DVector y[IRKG_MAX_STAGE];

	if(num_blocks < 1)
		return -1;

	// initialize
	dim = answer[0]->dim;
	tmpvec[0] = init_dvector(dim);
	tmpvec[1] = init_dvector(dim);
	for(index = 0; index < num_blocks; index++)
		y[index] = init_dvector(dim);

	// Forward substitution
	subst_dvector(y[0], b[0]);

	for(index = 1; index < num_blocks; index++)
	{
		// y[index] := y[index] - aip1[index - 1] * hmat[index - 1]^(-1) * y[index - 1]
		if(SolveDLS(tmpvec[0], hmat_lu[index - 1], y[index - 1]) < 0)
		{
			fprintf(stderr, "ERROR: SolveDBLS_triblock(%ld, Forward) is fail!\n", index);
			retcode = -1;
			goto END_SOLVEDLS_TRIBLOCK;
		}
		mul_dbmatrix_dvec(tmpvec[1], aim1[index - 1], tmpvec[0]);
		sub_dvector(y[index], b[index], tmpvec[1]);
	}

	// Backward substitution
	if(SolveDLS(answer[num_blocks - 1], hmat_lu[num_blocks - 1], y[num_blocks - 1]) < 0)
	{
		fprintf(stderr, "ERROR: SolveDBLSs_triblock(%ld, Backward) is fail!\n", num_blocks - 1);
		retcode = -1;
		goto END_SOLVEDLS_TRIBLOCK;
	}

	if(num_blocks >= 2)
	{
		for(index = num_blocks - 2; index >= 0; index--)
		{
			// x[index] := hmat[index]^(-1) * (y[index] - aip1[index] * x[index + 1])
			mul_dbmatrix_dvec(tmpvec[0], aip1[index], answer[index + 1]);
			sub_dvector(tmpvec[1], y[index], tmpvec[0]);
			if(SolveDLS(answer[index], hmat_lu[index], tmpvec[1]) < 0)
			{
				fprintf(stderr, "ERROR: SolveDLSs_triblock(%ld, Backward) is fail!\n", index);
				retcode = -1;
				goto END_SOLVEDLS_TRIBLOCK;
			}
		}
	}

END_SOLVEDLS_TRIBLOCK:

	// free
	free_dvector(tmpvec[0]);
	free_dvector(tmpvec[1]);
	for(index = 0; index < num_blocks; index++)
		free_dvector(y[index]);

	return retcode;

}

/************************************************************/
/*                                                          */
/*             BiCGSTAB Method for Real Matrix              */
/*                         for Real Block TridiagonalMatrix */
/*                                                          */
/*                 ver. 0.0 2012-03-01 (Thu) Tomonori Kouya */
/*                                                          */
/************************************************************/
long int DBBiCGSTAB_triblock_irk(DVector answer[], DBMatrix aim1[], DBMatrix aii[], DBMatrix aip1[], DBMatrix hmat[], DVector b[], double reps, double aeps, long int maxtimes, long int num_blocks)
/******************************************************************************/
/*                                                                            */
/* ENTRIES                                                                    */
/*       DVector answer: Solution for Ax = b such as                          */
/*                                                                            */
/*       [aii [0] aip1[0]                             ] [x[0]   ]   [b[0]   ] */
/*       [aim1[0] aii [1] aip1[1]                     ] [x[1]   ]   [b[1]   ] */
/*       [        .......................             ] [ ...   ] = [....   ] */
/*       [           aim1[nb-3] aii [nb-2] aip1[nb-2] ] [x[nb-2]]   [b[nb-2]] */
/*       [                      aim1[nb-2] aii [nb-1] ] [x[nb-1]]   [b[nb-1]] */
/*                                                                            */
/*       DVector b: Constant vector b   (given by user)                       */
/*       double reps: Relative tolerance (given by user)                      */
/*       double aeps: Absolute tolerance (given by user)                      */
/*       long int maxtimes: Maximum iterative times (given by user)           */
/*       long int num_blocks: Number of Blocks(= nb) (given by user)          */
/*                                                                            */
/* RETURNS                                                                    */
/*       DVector answer: Solution for Ax = b                                  */
/*                                                                            */
/* ERRORS                                                                     */
/* Positive value ... Normal : Iterative Times                                */
/*      -1 ... Rho is zero.                                                   */
/*      -2 ... Denominator of Alpha is zero.                                  */
/*      -3 ... Denominator of Omega is zero.                                  */
/*      -4 ... Numerator of Omega is zero.                                    */
/*      -5 ... Not Converge.                                                  */
/*                                                                            */
/******************************************************************************/
{
	long int i, j, times, dim, block_i;
	double alpha, alpha_num, alpha_den;
	double beta, beta_num;
	double rho, old_rho;
	double omega, omega_den;
	double dtmp, init_resnorm;
	DVector *vec[9], *tmpv, *tmpu, *tmpw; /* Temporary Vectors */

//	printf("--BICGSTAB--0\n");

/* Set initial value */
	tmpv = calloc(sizeof(dvector), num_blocks);
	tmpu = calloc(sizeof(dvector), num_blocks);
	tmpw = calloc(sizeof(dvector), num_blocks);
	for(i = 0; i < 9; i++)
		vec[i] = calloc(sizeof(dvector), num_blocks);

	for(block_i = 0; block_i < num_blocks; block_i++)
	{
		tmpv[block_i] = init_dvector(answer[block_i]->dim);
		tmpu[block_i] = init_dvector(answer[block_i]->dim);
		tmpw[block_i] = init_dvector(answer[block_i]->dim);

		for(i = 0; i < 9; i++)
			vec[i][block_i] = init_dvector(answer[block_i]->dim);
	}

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... v */
	/* vec[6] ... s */
	/* vec[7] ... s^T */
	/* vec[8] ... t */

	subst_dvector_blocks(vec[1], b, num_blocks); 
	subst_dvector_blocks(vec[2], b, num_blocks);

	beta_num = ip_dvector_blocks(vec[1], vec[1], num_blocks);
	init_resnorm = sqrt(beta_num);

	old_rho = 0.0;
	rho = 0.0;

/* preparing precondition */
//	for(block_i = 0; block_i < num_blocks; block_i++)
//		DBLUdecomp(hmat[block_i]);

/* r := P^(-1) * (b - A * x_0) = P^(-1) * r_0 */

/***** Start left preconditioning *****/
	// Forward substitute
	subst_dvector(tmpv[0], vec[1][0]);
	for(block_i = 0; block_i < num_blocks - 1; block_i++)
	{
		SolveDBLS(tmpu[block_i], hmat[block_i], tmpv[block_i]);
		mul_dbmatrix_dvec(tmpw[block_i + 1], aim1[block_i], tmpu[block_i]);
		sub_dvector(tmpv[block_i + 1], vec[1][block_i + 1], tmpw[block_i + 1]);
	}

	// Backward substitute
	SolveDBLS(vec[1][num_blocks - 1], hmat[num_blocks - 1], tmpv[num_blocks - 1]);
	for(block_i = num_blocks - 2; block_i >= 0; block_i--)
	{
		mul_dbmatrix_dvec(tmpw[block_i], aip1[block_i], vec[1][block_i + 1]);
		sub_dvector(tmpu[block_i], tmpv[block_i], tmpw[block_i]);
		SolveDBLS(vec[1][block_i], hmat[block_i], tmpu[block_i]);
	}
/***** End left preconditioning *****/
	//	printf("||vec[1]||_2 = %25.17e\n", norm2_dvector_blocks(vec[1], num_blocks));
	//	printf("||vec[2]||_2 = %25.17e\n", norm2_dvector_blocks(vec[2], num_blocks));

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		rho = ip_dvector_blocks(vec[2], vec[1], num_blocks);

		if(rho == 0.0)
		{
			fprintf(stderr, "Rho is zero!(DBiCGSTAB, %ld)\n", times);
			return -1;
		}

		if(times == 0)
		{
			/* p := r */
			subst_dvector_blocks(vec[3], vec[1], num_blocks);
		}
		else
		{
			beta = (rho / old_rho) * (alpha / omega);

			/* p := r + beta (p - omega v) */
			add_cmul_dvector_blocks(vec[4], vec[3], -omega, vec[5], num_blocks);
			add_cmul_dvector_blocks(vec[3], vec[1], beta, vec[4], num_blocks);
		}
		
	/***** Start v := Apt *****/
		//mul_dmatrix_dvec(vec[5], a, vec[3]);
	//#ifdef USE_PTHREAD
	//	_pthread_mul_dbtridiag_dvec_blocks(vec[5], aim1, aii, aip1, vec[3], tmpv, num_blocks, MAX_PTHREAD_NUM);
	//#else
#ifdef _OPENMP
		_bncomp_mul_dbtridiag_dvec_blocks(vec[5], aim1, aii, aip1, vec[3], tmpv, num_blocks);
#else
		mul_dbtridiag_dvec_blocks(vec[5], aim1, aii, aip1, vec[3], tmpv, num_blocks);
#endif
	//#endif
	/***** End v:= Apt *****/

	/* v := P^(-1) * v */

	/***** Start left preconditioning *****/
		// Forward substitute
		subst_dvector(tmpv[0], vec[5][0]);
		for(block_i = 0; block_i < num_blocks - 1; block_i++)
		{
			SolveDBLS(tmpu[block_i], hmat[block_i], tmpv[block_i]);
			mul_dbmatrix_dvec(tmpw[block_i + 1], aim1[block_i], tmpu[block_i]);
			sub_dvector(tmpv[block_i + 1], vec[5][block_i + 1], tmpw[block_i + 1]);
		}

		// Backward substitute
		SolveDBLS(vec[5][num_blocks - 1], hmat[num_blocks - 1], tmpv[num_blocks - 1]);
		for(block_i = num_blocks - 2; block_i >= 0; block_i--)
		{
			mul_dbmatrix_dvec(tmpw[block_i], aip1[block_i], vec[5][block_i + 1]);
			sub_dvector(tmpu[block_i], tmpv[block_i], tmpw[block_i]);
			SolveDBLS(vec[5][block_i], hmat[block_i], tmpu[block_i]);
		}
	/***** End left preconditioning *****/

		alpha_den = ip_dvector_blocks(vec[2], vec[5], num_blocks);
		if(alpha_den == 0.0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(DBiCGSTAB, %ld)\n", times);
			return -2;
		}
		alpha = rho / alpha_den;

		/* s = r - alpha v */
		add_cmul_dvector_blocks(vec[6], vec[1], -alpha, vec[5], num_blocks);

		/* Stopping Criteria */
		dtmp = norm2_dvector_blocks(vec[6], num_blocks);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			/* x = x + alpha pt */
			add_cmul_dvector_blocks(vec[0], vec[0], alpha, vec[3], num_blocks);

			subst_dvector_blocks(answer, vec[0], num_blocks);

			/* free vec[0]..[3]; */
			for(block_i = 0; block_i < num_blocks; block_i++)
			{
				free_dvector(tmpv[block_i]);
				free_dvector(tmpu[block_i]);
				free_dvector(tmpw[block_i]);

				for(i = 0; i < 9; i++)
						free_dvector(vec[i][block_i]);
			}
			free(tmpv);	free(tmpu);	free(tmpw);
			for(i = 0; i < 9; i++) free(vec[i]);

			return times;
		}

	/***** Start t := As *****/
		//mul_dmatrix_dvec(vec[8], a, vec[6]);
	//#ifdef USE_PTHREAD
	//	_pthread_mul_dbtridiag_dvec_blocks(vec[8], aim1, aii, aip1, vec[6], tmpv, num_blocks, MAX_PTHREAD_NUM);
	//#else
#ifdef _OPENMP
		_bncomp_mul_dbtridiag_dvec_blocks(vec[8], aim1, aii, aip1, vec[6], tmpv, num_blocks);
#else
		mul_dbtridiag_dvec_blocks(vec[8], aim1, aii, aip1, vec[6], tmpv, num_blocks);
#endif
	//#endif
	/***** End t := As *****/

	/* t := P^(-1) * t */
	/***** Start left preconditioning *****/
		// Forward substitute
		subst_dvector(tmpv[0], vec[8][0]);
		for(block_i = 0; block_i < num_blocks - 1; block_i++)
		{
			SolveDBLS(tmpu[block_i], hmat[block_i], tmpv[block_i]);
			mul_dbmatrix_dvec(tmpw[block_i + 1], aim1[block_i], tmpu[block_i]);
			sub_dvector(tmpv[block_i + 1], vec[8][block_i + 1], tmpw[block_i + 1]);
		}

		// Backward substitute
		SolveDBLS(vec[8][num_blocks - 1], hmat[num_blocks - 1], tmpv[num_blocks - 1]);
		for(block_i = num_blocks - 2; block_i >= 0; block_i--)
		{
			mul_dbmatrix_dvec(tmpw[block_i], aip1[block_i], vec[8][block_i + 1]);
			sub_dvector(tmpu[block_i], tmpv[block_i], tmpw[block_i]);
			SolveDBLS(vec[8][block_i], hmat[block_i], tmpu[block_i]);
		}
	/***** End left preconditioning *****/

		/* omega = (t, s) / (t, t) */
		omega_den = ip_dvector_blocks(vec[8], vec[8], num_blocks);
		if(omega_den == 0)
		{
			fprintf(stderr, "Denominator of Omega is zero!(DBiCGSTAB, %ld)\n", times);
			return -3;
		}
		omega = ip_dvector_blocks(vec[8], vec[6], num_blocks);
		if(omega == 0)
		{
			fprintf(stderr, "Numerator of Omega is zero!(DBiCGSTAB, %ld)\n", times);
			return -4;
		}
		omega = omega / omega_den;

		/* x = x + alpha pt + omega st */
		add_cmul_dvector_blocks(vec[4], vec[0], alpha, vec[3], num_blocks);
		add_cmul_dvector_blocks(vec[0], vec[4], omega, vec[6], num_blocks);

		/* residual */
		add_cmul_dvector_blocks(vec[1], vec[6], -omega, vec[8], num_blocks);

		beta_num = ip_dvector_blocks(vec[1], vec[1], num_blocks);

		/* Stopping Criteria */
		dtmp = sqrt(beta_num);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			subst_dvector_blocks(answer, vec[0], num_blocks);

			/* free vec[0]..[3]; */
			for(block_i = 0; block_i < num_blocks; block_i++)
			{
				free_dvector(tmpv[block_i]);
				free_dvector(tmpu[block_i]);
				free_dvector(tmpw[block_i]);

				for(i = 0; i < 9; i++)
						free_dvector(vec[i][block_i]);
			}
			free(tmpv);	free(tmpu);	free(tmpw);
			for(i = 0; i < 9; i++) free(vec[i]);

			return times;
		}
		//printf("%ld -> ||r_k||_2 = %25.17e\n", times, dtmp);

		old_rho = rho;
	}

	/* Not converge */
	subst_dvector_blocks(answer, vec[0], num_blocks);

	/* free vec[0]..[3]; */
	for(block_i = 0; block_i < num_blocks; block_i++)
	{
		free_dvector(tmpv[block_i]);
		free_dvector(tmpu[block_i]);
		free_dvector(tmpw[block_i]);

		for(i = 0; i < 9; i++)
				free_dvector(vec[i][block_i]);
	}
	free(tmpv);	free(tmpu);	free(tmpw);
	for(i = 0; i < 9; i++) free(vec[i]);

	fprintf(stderr, "Not converge!(DBBiCGSTAB, %ld)\n", times);
	return -5;

}

// MPF
#ifdef USE_GMP

/* ret := sum^n_{i,j=1} aij^2 */
void sumsqr_mpfbmatrix(mpf_t ret, MPFBMatrix mat)
{
	long int i, j, jmin, jmax;
	mpf_t tmp;

	mpf_init2(tmp, mpf_get_prec(ret));

	mpf_set_ui(ret, 0UL);

	for(i = 0; i < mat->dim; i++)
	{
		for(j = 0; j < mat->dim; j++)
		{
			mpf_mul(tmp, get_mpfbmatrix_ij(mat, i, j), get_mpfbmatrix_ij(mat, i, j));
			mpf_add(ret, ret, tmp);
		}
	}

	mpf_clear(tmp);
}

/* Frobenius Norm for tridiagonal block matrix */
void normf_mpfbmatrix_tridiag_blocks(mpf_t ret, MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], long int num_blocks)
{
	long int index;
	mpf_t tmp;

	if(num_blocks < 1)
		return;

	mpf_init2(tmp, mpf_get_prec(ret));

	mpf_set_ui(tmp, 0UL);
	mpf_set_ui(ret, 0UL);

	sumsqr_mpfbmatrix(tmp, aii [0]); mpf_add(ret, ret, tmp);

	if(num_blocks >= 2)
	{
		sumsqr_mpfbmatrix(tmp, aip1[0]); mpf_add(ret, ret, tmp);

		////#pragma ompparallel for
		for(index = 1; index <= num_blocks - 2; index++)
		{
			sumsqr_mpfbmatrix(tmp, aim1[index - 1]); mpf_add(ret, ret, tmp);
			sumsqr_mpfbmatrix(tmp, aii [index    ]); mpf_add(ret, ret, tmp);
			sumsqr_mpfbmatrix(tmp, aip1[index    ]); mpf_add(ret, ret, tmp);
		}

		sumsqr_mpfbmatrix(tmp, aim1[num_blocks - 2]); mpf_add(ret, ret, tmp);
		sumsqr_mpfbmatrix(tmp, aii [num_blocks - 1]); mpf_add(ret, ret, tmp);
	}

	mpf_sqrt(ret, ret);

	mpf_clear(tmp);
}

/* [ ret[0]              ]    [ b[0]              ]   [ aii [0] aip1[0]                                                   ]   [ x[0]              ] */
/* [ ret[1]              ]    [ b[1]              ]   [ aim1[0] aii [1] aip1[1]                                           ]   [ x[1]              ] */
/* [ ................... ] := [ ................. ] - [ ................................................................. ] * [ ................. ] */
/* [ ret[num_blocks - 2] ]    [ b[num_blocks - 2] ]   [    aim1[num_blocks - 3] aii [num_blocks - 2] aip1[num_blocsk - 2] ]   [ x[num_blocks - 2] ] */
/* [ ret[num_blocks - 1] ]    [ b[num_blocks - 1] ]   [                         aim1[num_blocks - 2] aii [num_blocks - 1] ]   [ x[num_blocks - 1] ] */
void residual_mpfbmat_mpfvec_tridiag_blocks(MPFVector ret[], MPFVector b[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], MPFVector x[], long int num_blocks)
{
	long int index;
	MPFVector tmpvec[4];

	if(num_blocks < 1)
		return;

//#pragma omp parallel shared(aim1, aii, aip1, x, ret, b, num_blocks)
{ // openmp start

	// initialize
	tmpvec[0] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
	tmpvec[1] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
	tmpvec[2] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
	tmpvec[3] = init2_mpfvector(ret[0]->dim, ret[0]->prec);

//	mul_mpfmatrix_mpfvec(r, mat, x);
//	sub_mpfvector(r, b, r);

	// ret[0] := b[0] - (aii[0] * x[0] + aip1[0] * x[1])
	mul_mpfbmatrix_mpfvec(tmpvec[0], aii [0], x[0]);
	sub_mpfvector(ret[0], b[0], tmpvec[0]);

	if(num_blocks >= 2)
	{
		mul_mpfbmatrix_mpfvec(tmpvec[1], aip1[0], x[1]);
		add_mpfvector(tmpvec[2], tmpvec[0], tmpvec[1]);
		sub_mpfvector(ret[0], b[0], tmpvec[2]);

		// ret[index] ;= b[index] - ( aim1[index - 1] * x[index - 1] + aii[index] * x[index] + aip1[index] * x[index + 1] )

		#pragma omp for private(index)
		for(index = 1; index <= num_blocks - 2; index++)
		{
			mul_mpfbmatrix_mpfvec(tmpvec[0], aim1[index - 1], x[index - 1]);
			mul_mpfbmatrix_mpfvec(tmpvec[1], aii [index]    , x[index]    );
			mul_mpfbmatrix_mpfvec(tmpvec[2], aip1[index]    , x[index + 1]);
			add_mpfvector(tmpvec[3], tmpvec[0], tmpvec[1]);
			add_mpfvector(tmpvec[3], tmpvec[3], tmpvec[2]);
			sub_mpfvector(ret[index], b[index], tmpvec[3]);
		}

		// ret[num_blocks - 1] := b[num_blocks - 1] - ( aim1[num_blocks - 2] * x[num_blocks - 2] + aii[num_blocks - 1] * x[num_blocks - 1] )
		mul_mpfbmatrix_mpfvec(tmpvec[0], aim1[num_blocks - 2], x[num_blocks - 2]);
		mul_mpfbmatrix_mpfvec(tmpvec[1], aii [num_blocks - 1], x[num_blocks - 1]);
		add_mpfvector(tmpvec[2], tmpvec[0], tmpvec[1]);
		sub_mpfvector(ret[num_blocks - 1], b[num_blocks - 1], tmpvec[2]);
	}

	// free
	free_mpfvector(tmpvec[0]);
	free_mpfvector(tmpvec[1]);
	free_mpfvector(tmpvec[2]);
	free_mpfvector(tmpvec[3]);
} // openmp end
}

/* mul vec */
/* ret[] := a[] * x[] */
void mul_mpfbmat_mpfvec_tridiag_blocks(MPFVector ret[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], MPFVector x[], long int num_blocks)
{
	long int index;
	MPFVector tmpvec[3];

	if(num_blocks < 1)
		return;

	// initialize
	tmpvec[0] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
	tmpvec[1] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
	tmpvec[2] = init2_mpfvector(ret[0]->dim, ret[0]->prec);

//	mul_mpfmatrix_mpfvec(r, mat, x);
//	sub_mpfvector(r, b, r);

	// ret[0] := (aii[0] * x[0] + aip1[0] * x[1])
	mul_mpfbmatrix_mpfvec(tmpvec[0], aii [0], x[0]);
	subst_mpfvector(ret[0], tmpvec[0]);

	if(num_blocks >= 2)
	{
		mul_mpfbmatrix_mpfvec(tmpvec[1], aip1[0], x[1]);
		add_mpfvector(ret[0], tmpvec[0], tmpvec[1]);

		// ret[index] ;= ( aim1[index - 1] * x[index - 1] + aii[index] * x[index] + aip1[index] * x[index + 1] )
////#pragma ompparallel
//{
		#pragma omp parallel for private(index)
		for(index = 1; index <= num_blocks - 2; index++)
		{
			mul_mpfbmatrix_mpfvec(tmpvec[0], aim1[index - 1], x[index - 1]);
			mul_mpfbmatrix_mpfvec(tmpvec[1], aii [index]    , x[index]    );
			mul_mpfbmatrix_mpfvec(tmpvec[2], aip1[index]    , x[index + 1]);
			add_mpfvector(ret[index], tmpvec[0], tmpvec[1]);
			add_mpfvector(ret[index], ret[index], tmpvec[2]);
		}
//}
		// ret[num_blocks - 1] := b[num_blocks - 1] - ( aim1[num_blocks - 2] * x[num_blocks - 2] + aii[num_blocks - 1] * x[num_blocks - 1] )
		mul_mpfbmatrix_mpfvec(tmpvec[0], aim1[num_blocks - 2], x[num_blocks - 2]);
		mul_mpfbmatrix_mpfvec(tmpvec[1], aii [num_blocks - 1], x[num_blocks - 1]);
		add_mpfvector(ret[num_blocks - 1], tmpvec[0], tmpvec[1]);
	}

	// free
	free_mpfvector(tmpvec[0]);
	free_mpfvector(tmpvec[1]);
	free_mpfvector(tmpvec[2]);
}

/* mul vec */
/* ret[] := a[] * x[] with working area */
void mul_mpfbtridiag_mpfvec_blocks(MPFVector ret[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], MPFVector x[], MPFVector tmpvec[], long int num_blocks)
{
	long int index;

	if(num_blocks < 1)
		return;

//	mul_mpfmatrix_mpfvec(r, mat, x);
//	sub_mpfvector(r, b, r);

	// ret[0] := (aii[0] * x[0] + aip1[0] * x[1])
	mul_mpfbmatrix_mpfvec(tmpvec[0], aii [0], x[0]);
	subst_mpfvector(ret[0], tmpvec[0]);

	if(num_blocks >= 2)
	{
		mul_mpfbmatrix_mpfvec(tmpvec[1], aip1[0], x[1]);
		add_mpfvector(ret[0], tmpvec[0], tmpvec[1]);

		// ret[index] ;= ( aim1[index - 1] * x[index - 1] + aii[index] * x[index] + aip1[index] * x[index + 1] )
////#pragma ompparallel
//{
		#pragma omp parallel for private(index)
		for(index = 1; index <= num_blocks - 2; index++)
		{
			mul_mpfbmatrix_mpfvec(tmpvec[0], aim1[index - 1], x[index - 1]);
			mul_mpfbmatrix_mpfvec(tmpvec[1], aii [index]    , x[index]    );
			mul_mpfbmatrix_mpfvec(tmpvec[2], aip1[index]    , x[index + 1]);
			add_mpfvector(ret[index], tmpvec[0], tmpvec[1]);
			add_mpfvector(ret[index], ret[index], tmpvec[2]);
		}
//}
		// ret[num_blocks - 1] := b[num_blocks - 1] - ( aim1[num_blocks - 2] * x[num_blocks - 2] + aii[num_blocks - 1] * x[num_blocks - 1] )
		mul_mpfbmatrix_mpfvec(tmpvec[0], aim1[num_blocks - 2], x[num_blocks - 2]);
		mul_mpfbmatrix_mpfvec(tmpvec[1], aii [num_blocks - 1], x[num_blocks - 1]);
		add_mpfvector(ret[num_blocks - 1], tmpvec[0], tmpvec[1]);
	}
}

void residual_mpfbmat_mpfvec_tridiag_blocks2(MPFVector ret[], MPFVector b[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], MPFVector x[], long int num_blocks)
{
	mul_mpfbmat_mpfvec_tridiag_blocks(ret, aim1, aii, aip1, x, num_blocks);
	sub_mpfvector_blocks(ret, b, ret, num_blocks);
}

// substitution of tridiagonal block matrix
void subst_mpfbmatrix_tridiag_blocks(MPFBMatrix aim1_f[], MPFBMatrix aii_f[], MPFBMatrix aip1_f[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], long int num_blocks)
{
	long int index;

	if(num_blocks < 1)
		return;

 	subst_mpfbmatrix(aii_f [0], aii [0]);
	if(num_blocks >= 2)
	{
		subst_mpfbmatrix(aip1_f[0], aip1[0]);
		//#pragma ompparallel for 
		for(index = 1; index < num_blocks - 1; index++)
		{
			subst_mpfbmatrix(aim1_f[index - 1], aim1[index - 1]);
			subst_mpfbmatrix(aii_f [index]    , aii [index]);
			subst_mpfbmatrix(aip1_f[index]    , aip1[index]);
		}
		subst_mpfbmatrix(aim1_f[num_blocks - 2], aim1[num_blocks - 2]);
		subst_mpfbmatrix(aii_f [num_blocks - 1], aii [num_blocks - 1]);
	}
}

// substitution of tridiagonal block matrix
void subst_dbmatrix_mpfbmat_tridiag_blocks(DBMatrix aim1_f[], DBMatrix aii_f[], DBMatrix aip1_f[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], long int num_blocks)
{
	long int index;

	if(num_blocks < 1)
		return;

 	subst_dbmatrix_mpfbmat(aii_f [0], aii [0]);
	if(num_blocks >= 2)
	{
		subst_dbmatrix_mpfbmat(aip1_f[0], aip1[0]);
		//#pragma ompparallel for
		for(index = 1; index < num_blocks - 1; index++)
		{
			subst_dbmatrix_mpfbmat(aim1_f[index - 1], aim1[index - 1]);
			subst_dbmatrix_mpfbmat(aii_f [index]    , aii [index]);
			subst_dbmatrix_mpfbmat(aip1_f[index]    , aip1[index]);
		}
		subst_dbmatrix_mpfbmat(aim1_f[num_blocks - 2], aim1[num_blocks - 2]);
		subst_dbmatrix_mpfbmat(aii_f [num_blocks - 1], aii [num_blocks - 1]);
	}
}

/************************************************************/
/*                                                          */
/*             BiCGSTAB Method                              */
/*                         for Real Block TridiagonalMatrix */
/*                                                          */
/*                 ver. 0.0 2012-03-07 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
long int MPFBBiCGSTAB_triblock(MPFVector answer[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], MPFVector b[], mpf_t reps, mpf_t aeps, long int maxtimes, long int num_blocks)
/******************************************************************************/
/*                                                                            */
/* ENTRIES                                                                    */
/*       MPFVector answer: Solution for Ax = b such as                        */
/*                                                                            */
/*       [aii [0] aip1[0]                             ] [x[0]   ]   [b[0]   ] */
/*       [aim1[0] aii [1] aip1[1]                     ] [x[1]   ]   [b[1]   ] */
/*       [        .......................             ] [ ...   ] = [....   ] */
/*       [           aim1[nb-3] aii [nb-2] aip1[nb-2] ] [x[nb-2]]   [b[nb-2]] */
/*       [                      aim1[nb-2] aii [nb-1] ] [x[nb-1]]   [b[nb-1]] */
/*                                                                            */
/*       MPFVector b: Constant vector b   (given by user)                     */
/*       mpf_t reps: Relative tolerance (given by user)                       */
/*       mpf_t aeps: Absolute tolerance (given by user)                       */
/*       long int maxtimes: Maximum iterative times (given by user)           */
/*       long int num_blocks: Number of Blocks(= nb) (given by user)          */
/*                                                                            */
/* RETURNS                                                                    */
/*       MPFVector answer: Solution for Ax = b                                */
/*                                                                            */
/* ERRORS                                                                     */
/* Positive value ... Normal : Iterative Times                                */
/*      -1 ... Rho is zero.                                                   */
/*      -2 ... Denominator of Alpha is zero.                                  */
/*      -3 ... Denominator of Omega is zero.                                  */
/*      -4 ... Numerator of Omega is zero.                                    */
/*      -5 ... Not Converge.                                                  */
/*                                                                            */
/******************************************************************************/
{
	long int i, j, times, dim, block_i, return_val;
	unsigned long prec;
	mpf_t dtmp, dtmp1;
	mpf_t alpha, alpha_num, alpha_den;
	mpf_t beta, beta_num;
	mpf_t rho, old_rho;
	mpf_t omega, omega_den;
	mpf_t init_resnorm;
	MPFVector *vec[9], *tmpv; /* Temporary Vectors */

	dim = answer[0]->dim;
	prec = answer[0]->prec;
	return_val = 0;

/* Set initial value */
	mpf_init2(alpha, prec);
	mpf_init2(alpha_num, prec);
	mpf_init2(alpha_den, prec);
	mpf_init2(rho, prec);
	mpf_init2(old_rho, prec);
	mpf_init2(beta, prec);
	mpf_init2(beta_num, prec);
	mpf_init2(omega, prec);
	mpf_init2(omega_den, prec);
	mpf_init2(dtmp, prec);
	mpf_init2(dtmp1, prec);
	mpf_init2(init_resnorm, prec);

/* Set initial value */
	tmpv = calloc(sizeof(mpfvector), num_blocks);
	for(i = 0; i < 9; i++)
		vec[i] = calloc(sizeof(mpfvector), num_blocks);

	for(block_i = 0; block_i < num_blocks; block_i++)
	{
		tmpv[block_i] = init2_mpfvector(answer[block_i]->dim, prec);

		for(i = 0; i < 9; i++)
			vec[i][block_i] = init2_mpfvector(answer[block_i]->dim, prec);
	}

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... v */
	/* vec[6] ... s */
	/* vec[7] ... s^T */
	/* vec[8] ... t */

	subst_mpfvector_blocks(vec[1], b, num_blocks); 
	subst_mpfvector_blocks(vec[2], b, num_blocks);

	ip_mpfvector_blocks(beta_num, vec[1], vec[1], num_blocks);
	mpf_sqrt(init_resnorm, beta_num);

	mpf_set_ui(old_rho, 0UL);
	mpf_set_ui(rho, 0UL);

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_mpfvector_blocks(rho, vec[2], vec[1], num_blocks);

		if(mpf_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(MPFBiCGSTAB_triblock, %ld)\n", times);
			return_val = -1;
			break;
		}

		if(times == 0)
		{
			/* p := r */
			subst_mpfvector_blocks(vec[3], vec[1], num_blocks);
		}
		else
		{
			mpf_div(beta, rho, old_rho);
			mpf_div(dtmp, alpha, omega);
			mpf_mul(beta, beta, dtmp);

			/* p := r + beta (p - omega v) */
			mpf_neg(dtmp, omega);
			add_cmul_mpfvector_blocks(vec[4], vec[3], dtmp, vec[5], num_blocks);
			add_cmul_mpfvector_blocks(vec[3], vec[1], beta, vec[4], num_blocks);
		}
		/* precondition */
		
		/* v := Apt */
		//mul_mpfmatrix_mpfvec(vec[5], a, vec[3]);
		mul_mpfbmatrix_mpfvec(vec[5][0], aii [0], vec[3][0]);
		mul_mpfbmatrix_mpfvec(  tmpv[0], aip1[0], vec[3][1]);
		add_mpfvector(vec[5][0], vec[5][0], tmpv[0]);

//#pragma ompparallel
{
		#pragma omp parallel for private(block_i)
		for(block_i = 1; block_i < num_blocks - 1; block_i++)
		{
			mul_mpfbmatrix_mpfvec(vec[5][block_i], aim1 [block_i - 1], vec[3][block_i - 1]);

			mul_mpfbmatrix_mpfvec(  tmpv[block_i], aii  [block_i    ], vec[3][block_i    ]);
			add_mpfvector(vec[5][block_i], vec[5][block_i], tmpv[block_i]);

			mul_mpfbmatrix_mpfvec(  tmpv[block_i], aip1 [block_i    ], vec[3][block_i + 1]);
			add_mpfvector(vec[5][block_i], vec[5][block_i], tmpv[block_i]);
		}
}
		mul_mpfbmatrix_mpfvec(vec[5][num_blocks - 1], aim1 [num_blocks - 2], vec[3][num_blocks - 2]);
		mul_mpfbmatrix_mpfvec(  tmpv[num_blocks - 1], aii  [num_blocks - 1], vec[3][num_blocks - 1]);
		add_mpfvector(vec[5][num_blocks - 1], vec[5][num_blocks - 1], tmpv[num_blocks - 1]);

		ip_mpfvector_blocks(alpha_den, vec[2], vec[5], num_blocks);
		if(mpf_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(MPFBiCGSTAB_triblock, %ld)\n", times);
			return_val = -2;
			break;
		}
		mpf_div(alpha, rho, alpha_den);

		/* s = r - alpha v */
		mpf_neg(dtmp, alpha);
		add_cmul_mpfvector_blocks(vec[6], vec[1], dtmp, vec[5], num_blocks);

		/* Stopping Criteria */
		norm2_mpfvector_blocks(dtmp, vec[6], num_blocks);
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			/* x = x + alpha pt */
			add_cmul_mpfvector_blocks(vec[0], vec[0], alpha, vec[3], num_blocks);

			//subst_mpfvector_blocks(answer, vec[0], num_blocks);

			return_val = times;
			break;
		}

		/* precondition */

		//mul_mpfmatrix_mpfvec(vec[8], a, vec[6]);
		mul_mpfbmatrix_mpfvec(vec[8][0], aii [0], vec[6][0]);
		mul_mpfbmatrix_mpfvec(  tmpv[0], aip1[0], vec[6][1]);
		add_mpfvector(vec[8][0], vec[8][0], tmpv[0]);

//#pragma ompparallel
{
		#pragma omp parallel for private(block_i)
		for(block_i = 1; block_i < num_blocks - 1; block_i++)
		{
			mul_mpfbmatrix_mpfvec(vec[8][block_i], aim1 [block_i - 1], vec[6][block_i - 1]);

			mul_mpfbmatrix_mpfvec(  tmpv[block_i], aii  [block_i    ], vec[6][block_i    ]);
			add_mpfvector(vec[8][block_i], vec[8][block_i], tmpv[block_i]);

			mul_mpfbmatrix_mpfvec(  tmpv[block_i], aip1 [block_i    ], vec[6][block_i + 1]);
			add_mpfvector(vec[8][block_i], vec[8][block_i], tmpv[block_i]);
		}
}

		mul_mpfbmatrix_mpfvec(vec[8][num_blocks - 1], aim1 [num_blocks - 2], vec[6][num_blocks - 2]);
		mul_mpfbmatrix_mpfvec(  tmpv[num_blocks - 1], aii  [num_blocks - 1], vec[6][num_blocks - 1]);
		add_mpfvector(vec[8][num_blocks - 1], vec[8][num_blocks - 1], tmpv[num_blocks - 1]);

		/* omega = (t, s) / (t, t) */
		ip_mpfvector_blocks(omega_den, vec[8], vec[8], num_blocks);
		if(mpf_cmp_ui(omega_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Omega is zero!(MPFBiCGSTAB_triblock, %ld)\n", times);
			return_val = -3;
			break;
		}
		ip_mpfvector_blocks(omega, vec[8], vec[6], num_blocks);
		if(mpf_cmp_ui(omega, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Omega is zero!(MPFBiCGSTAB_triblock, %ld)\n", times);
			return_val = -4;
			break;
		}
		mpf_div(omega, omega, omega_den);

		/* x = x + alpha pt + omega st */
		add_cmul_mpfvector_blocks(vec[4], vec[0], alpha, vec[3], num_blocks);
		add_cmul_mpfvector_blocks(vec[0], vec[4], omega, vec[6], num_blocks);

		/* residual */
		mpf_neg(dtmp, omega);
		add_cmul_mpfvector_blocks(vec[1], vec[6], dtmp, vec[8], num_blocks);

		ip_mpfvector_blocks(beta_num, vec[1], vec[1], num_blocks);

		/* Stopping Criteria */
		mpf_sqrt(dtmp, beta_num);
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_mpfvector_blocks(answer, vec[0], num_blocks);
			return_val = times;
			break;
		}

		mpf_set(old_rho, rho);
	}

	/* Not converge */
	subst_mpfvector_blocks(answer, vec[0], num_blocks);

	/* free vec[0]..[3]; */
	for(block_i = 0; block_i < num_blocks; block_i++)
	{
		free_mpfvector(tmpv[block_i]);

		for(i = 0; i < 9; i++)
				free_mpfvector(vec[i][block_i]);
	}
	free(tmpv);
	for(i = 0; i < 9; i++)
		free(vec[i]);

	mpf_clear(alpha); mpf_clear(alpha_num); mpf_clear(alpha_den);
	mpf_clear(rho); mpf_clear(old_rho);
	mpf_clear(beta); mpf_clear(beta_num);
	mpf_clear(omega); mpf_clear(omega_den);
	mpf_clear(dtmp); mpf_clear(dtmp1); mpf_clear(init_resnorm);

	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(MPFBiCGSTAB_triblock, %ld)\n", times);
		return_val = -5;
	}

	return return_val;
}


// Block LU decomposition for tridiagonal block matrix
int MPFBLUdecomp_triblock(MPFMatrix hmat_lu[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], long int num_blocks)
{
	int retcode = 0;
	long int index;
	MPFMatrix tmpmat[2];

	if(num_blocks < 1)
		return -1;

	// initialize
	tmpmat[0] = init2_mpfmatrix(aii[0]->dim, aii[0]->dim, aii[0]->prec);
	tmpmat[1] = init2_mpfmatrix(aii[0]->dim, aii[0]->dim, aii[0]->prec);

	// h[0] = aii[0]
	subst_mpfmatrix_mpfbmat(hmat_lu[0], aii[0]);
	if(MPFLUdecomp(hmat_lu[0]) < 0)
	{
		fprintf(stderr, "ERROR: MPFBLUdecomp_triblock(0, MPFLUdecomp) is fail!\n");
		retcode = -1;
		goto END_MPFBLUDECOMP_TRIBLOCK;
	}

	// h[i] := aii[i] - aim1[i-1] * aii[i-1]^(-1) * aip1[i-1]
	for(index = 1; index < num_blocks; index++)
	{
		// hmat[index] := aii[index] - aim1[index - 1] * hmat[index - 1]^(-1) * aip1[index - 1]
		if(SolveMPFLSs_mpfbmat(tmpmat[0], hmat_lu[index - 1], aip1[index - 1]) < 0)
		{
			fprintf(stderr, "ERROR: MPFDLUdecomp_triblock(%ld, SolveMPFLSs) is fail!\n", index);
			retcode = -1;
			goto END_MPFBLUDECOMP_TRIBLOCK;
		}
		mul_mpfmatrix_mpfbmat_mpfmat(tmpmat[1], aim1[index - 1], tmpmat[0]);
		sub_mpfmatrix_mpfbmat_mpfmat(hmat_lu[index], aii[index], tmpmat[1]);

		if(MPFLUdecomp(hmat_lu[index]) < 0)
		{
			fprintf(stderr, "ERROR: MPFLUdecomp_triblock(%ld, MPFLUdecomp) is fail!\n", index);
			retcode = -1;
			goto END_MPFBLUDECOMP_TRIBLOCK;
		}
	}

END_MPFBLUDECOMP_TRIBLOCK:

	// free
	free_mpfmatrix(tmpmat[0]);
	free_mpfmatrix(tmpmat[1]);

	return retcode;
}

// Solve linear systems of equation with block LU decomposed matrix
int SolveMPFBLS_triblock(MPFVector answer[], MPFBMatrix aim1[], MPFMatrix hmat_lu[], MPFBMatrix aip1[], MPFVector b[], long int num_blocks)
{
	int retcode = 0;
	unsigned long prec;
	long int index, dim;
	MPFVector tmpvec[2];
	MPFVector y[IRKG_MAX_STAGE];

	if(num_blocks < 1)
		return -1;

	// initialize
	prec = answer[0]->prec;
	dim = answer[0]->dim;
	tmpvec[0] = init2_mpfvector(dim, prec);
	tmpvec[1] = init2_mpfvector(dim, prec);
	for(index = 0; index < num_blocks; index++)
		y[index] = init2_mpfvector(dim, prec);

	// Forward substitution
	subst_mpfvector(y[0], b[0]);

	for(index = 1; index < num_blocks; index++)
	{
		// y[index] := y[index] - aip1[index - 1] * hmat[index - 1]^(-1) * y[index - 1]
		if(SolveMPFLS(tmpvec[0], hmat_lu[index - 1], y[index - 1]) < 0)
		{
			fprintf(stderr, "ERROR: SolveMPFBLS_triblock(%ld, Forward) is fail!\n", index);
			retcode = -1;
			goto END_SOLVEMPFBLS_TRIBLOCK;
		}
		mul_mpfbmatrix_mpfvec(tmpvec[1], aim1[index - 1], tmpvec[0]);
		sub_mpfvector(y[index], b[index], tmpvec[1]);
	}

	// Backward substitution
	if(SolveMPFLS(answer[num_blocks - 1], hmat_lu[num_blocks - 1], y[num_blocks - 1]) < 0)
	{
		fprintf(stderr, "ERROR: SolveMPFLS_triblock(%ld, Backward) is fail!\n", num_blocks - 1);
		retcode = -1;
		goto END_SOLVEMPFBLS_TRIBLOCK;
	}

	if(num_blocks >= 2)
	{
		for(index = num_blocks - 2; index >= 0; index--)
		{
			// x[index] := hmat[index]^(-1) * (y[index] - aip1[index] * x[index + 1])
			mul_mpfbmatrix_mpfvec(tmpvec[0], aip1[index], answer[index + 1]);
			sub_mpfvector(tmpvec[1], y[index], tmpvec[0]);
			if(SolveMPFLS(answer[index], hmat_lu[index], tmpvec[1]) < 0)
			{
				fprintf(stderr, "ERROR: SolveMPFLS_triblock(%ld, Backward) is fail!\n", index);
				retcode = -1;
				goto END_SOLVEMPFBLS_TRIBLOCK;
			}
		}
	}

END_SOLVEMPFBLS_TRIBLOCK:

	// free
	free_mpfvector(tmpvec[0]);
	free_mpfvector(tmpvec[1]);
	for(index = 0; index < num_blocks; index++)
		free_mpfvector(y[index]);

	return retcode;
}

/************************************************************/
/*                                                          */
/*             BiCGSTAB Method                              */
/*                         for Real Block TridiagonalMatrix */
/*                                                          */
/*                 ver. 0.0 2012-03-07 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
long int MPFBBiCGSTAB_triblock_irk(MPFVector answer[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], MPFBMatrix hmat[], MPFVector b[], mpf_t reps, mpf_t aeps, long int maxtimes, long int num_blocks)
/******************************************************************************/
/*                                                                            */
/* ENTRIES                                                                    */
/*       MPFVector answer: Solution for Ax = b such as                        */
/*                                                                            */
/*       [aii [0] aip1[0]                             ] [x[0]   ]   [b[0]   ] */
/*       [aim1[0] aii [1] aip1[1]                     ] [x[1]   ]   [b[1]   ] */
/*       [        .......................             ] [ ...   ] = [....   ] */
/*       [           aim1[nb-3] aii [nb-2] aip1[nb-2] ] [x[nb-2]]   [b[nb-2]] */
/*       [                      aim1[nb-2] aii [nb-1] ] [x[nb-1]]   [b[nb-1]] */
/*                                                                            */
/*       MPFVector b: Constant vector b   (given by user)                     */
/*       mpf_t reps: Relative tolerance (given by user)                       */
/*       mpf_t aeps: Absolute tolerance (given by user)                       */
/*       long int maxtimes: Maximum iterative times (given by user)           */
/*       long int num_blocks: Number of Blocks(= nb) (given by user)          */
/*                                                                            */
/* RETURNS                                                                    */
/*       MPFVector answer: Solution for Ax = b                                */
/*                                                                            */
/* ERRORS                                                                     */
/* Positive value ... Normal : Iterative Times                                */
/*      -1 ... Rho is zero.                                                   */
/*      -2 ... Denominator of Alpha is zero.                                  */
/*      -3 ... Denominator of Omega is zero.                                  */
/*      -4 ... Numerator of Omega is zero.                                    */
/*      -5 ... Not Converge.                                                  */
/*                                                                            */
/******************************************************************************/
{
	long int i, j, times, dim, block_i, return_val;
	unsigned long prec;
	mpf_t dtmp, dtmp1;
	mpf_t alpha, alpha_num, alpha_den;
	mpf_t beta, beta_num;
	mpf_t rho, old_rho;
	mpf_t omega, omega_den;
	mpf_t init_resnorm;
	MPFVector *vec[9], *tmpv, *tmpu, *tmpw; /* Temporary Vectors */

	dim = answer[0]->dim;
	prec = answer[0]->prec;
	return_val = 0;

/* Set initial value */
	mpf_init2(alpha, prec);
	mpf_init2(alpha_num, prec);
	mpf_init2(alpha_den, prec);
	mpf_init2(rho, prec);
	mpf_init2(old_rho, prec);
	mpf_init2(beta, prec);
	mpf_init2(beta_num, prec);
	mpf_init2(omega, prec);
	mpf_init2(omega_den, prec);
	mpf_init2(dtmp, prec);
	mpf_init2(dtmp1, prec);
	mpf_init2(init_resnorm, prec);

/* Set initial value */
	tmpv = calloc(sizeof(mpfvector), num_blocks);
	tmpu = calloc(sizeof(mpfvector), num_blocks);
	tmpw = calloc(sizeof(mpfvector), num_blocks);
	for(i = 0; i < 9; i++)
		vec[i] = calloc(sizeof(mpfvector), num_blocks);

	for(block_i = 0; block_i < num_blocks; block_i++)
	{
		tmpv[block_i] = init2_mpfvector(answer[block_i]->dim, prec);
		tmpu[block_i] = init2_mpfvector(answer[block_i]->dim, prec);
		tmpw[block_i] = init2_mpfvector(answer[block_i]->dim, prec);

		for(i = 0; i < 9; i++)
			vec[i][block_i] = init2_mpfvector(answer[block_i]->dim, prec);
	}

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... v */
	/* vec[6] ... s */
	/* vec[7] ... s^T */
	/* vec[8] ... t */

	subst_mpfvector_blocks(vec[1], b, num_blocks); 
	subst_mpfvector_blocks(vec[2], b, num_blocks);

	ip_mpfvector_blocks(beta_num, vec[1], vec[1], num_blocks);
	mpf_sqrt(init_resnorm, beta_num);

	mpf_set_ui(old_rho, 0UL);
	mpf_set_ui(rho, 0UL);

/* preparing precondition */
	//#pragma ompparallel for
	for(block_i = 0; block_i < num_blocks; block_i++)
		MPFBLUdecomp(hmat[block_i]);

/* r := P^(-1) * (b - A * x_0) = P^(-1) * r_0 */

/***** Start left preconditioning *****/
	// Forward substitute
	subst_mpfvector(tmpv[0], vec[1][0]);
	for(block_i = 0; block_i < num_blocks - 1; block_i++)
	{
		SolveMPFBLS(tmpu[block_i], hmat[block_i], tmpv[block_i]);
		mul_mpfbmatrix_mpfvec(tmpw[block_i + 1], aim1[block_i], tmpu[block_i]);
		sub_mpfvector(tmpv[block_i + 1], vec[1][block_i + 1], tmpw[block_i + 1]);
	}

	// Backward substitute
	SolveMPFBLS(vec[1][num_blocks - 1], hmat[num_blocks - 1], tmpv[num_blocks - 1]);
	for(block_i = num_blocks - 2; block_i >= 0; block_i--)
	{
		mul_mpfbmatrix_mpfvec(tmpw[block_i], aip1[block_i], vec[1][block_i + 1]);
		sub_mpfvector(tmpu[block_i], tmpv[block_i], tmpw[block_i]);
		SolveMPFBLS(vec[1][block_i], hmat[block_i], tmpu[block_i]);
	}
/***** End left preconditioning *****/
	//	printf("||vec[1]||_2 = %25.17e\n", norm2_mpfvector_blocks(vec[1], num_blocks));
	//	printf("||vec[2]||_2 = %25.17e\n", norm2_mpfvector_blocks(vec[2], num_blocks));

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_mpfvector_blocks(rho, vec[2], vec[1], num_blocks);

		if(mpf_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(MPFBiCGSTAB_triblock, %ld)\n", times);
			return_val = -1;
			break;
		}

		if(times == 0)
		{
			/* p := r */
			subst_mpfvector_blocks(vec[3], vec[1], num_blocks);
		}
		else
		{
			mpf_div(beta, rho, old_rho);
			mpf_div(dtmp, alpha, omega);
			mpf_mul(beta, beta, dtmp);

			/* p := r + beta (p - omega v) */
			mpf_neg(dtmp, omega);
			add_cmul_mpfvector_blocks(vec[4], vec[3], dtmp, vec[5], num_blocks);
			add_cmul_mpfvector_blocks(vec[3], vec[1], beta, vec[4], num_blocks);
		}

		/* v := Apt */
		//mul_mpfmatrix_mpfvec(vec[5], a, vec[3]);
		mul_mpfbmatrix_mpfvec(vec[5][0], aii [0], vec[3][0]);
		mul_mpfbmatrix_mpfvec(  tmpv[0], aip1[0], vec[3][1]);
		add_mpfvector(vec[5][0], vec[5][0], tmpv[0]);

		//#pragma ompparallel for
		for(block_i = 1; block_i < num_blocks - 1; block_i++)
		{
			mul_mpfbmatrix_mpfvec(vec[5][block_i], aim1 [block_i - 1], vec[3][block_i - 1]);

			mul_mpfbmatrix_mpfvec(  tmpv[block_i], aii  [block_i    ], vec[3][block_i    ]);
			add_mpfvector(vec[5][block_i], vec[5][block_i], tmpv[block_i]);

			mul_mpfbmatrix_mpfvec(  tmpv[block_i], aip1 [block_i    ], vec[3][block_i + 1]);
			add_mpfvector(vec[5][block_i], vec[5][block_i], tmpv[block_i]);
		}

		mul_mpfbmatrix_mpfvec(vec[5][num_blocks - 1], aim1 [num_blocks - 2], vec[3][num_blocks - 2]);
		mul_mpfbmatrix_mpfvec(  tmpv[num_blocks - 1], aii  [num_blocks - 1], vec[3][num_blocks - 1]);
		add_mpfvector(vec[5][num_blocks - 1], vec[5][num_blocks - 1], tmpv[num_blocks - 1]);

		/* v := P^(-1) * v */

	/***** Start left preconditioning *****/
		// Forward substitute
		subst_mpfvector(tmpv[0], vec[5][0]);
		for(block_i = 0; block_i < num_blocks - 1; block_i++)
		{
			SolveMPFBLS(tmpu[block_i], hmat[block_i], tmpv[block_i]);
			mul_mpfbmatrix_mpfvec(tmpw[block_i + 1], aim1[block_i], tmpu[block_i]);
			sub_mpfvector(tmpv[block_i + 1], vec[5][block_i + 1], tmpw[block_i + 1]);
		}

		// Backward substitute
		SolveMPFBLS(vec[5][num_blocks - 1], hmat[num_blocks - 1], tmpv[num_blocks - 1]);
		for(block_i = num_blocks - 2; block_i >= 0; block_i--)
		{
			mul_mpfbmatrix_mpfvec(tmpw[block_i], aip1[block_i], vec[5][block_i + 1]);
			sub_mpfvector(tmpu[block_i], tmpv[block_i], tmpw[block_i]);
			SolveMPFBLS(vec[5][block_i], hmat[block_i], tmpu[block_i]);
		}
	/***** End left preconditioning *****/

		ip_mpfvector_blocks(alpha_den, vec[2], vec[5], num_blocks);
		if(mpf_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(MPFBiCGSTAB_triblock, %ld)\n", times);
			return_val = -2;
			break;
		}
		mpf_div(alpha, rho, alpha_den);

		/* s = r - alpha v */
		mpf_neg(dtmp, alpha);
		add_cmul_mpfvector_blocks(vec[6], vec[1], dtmp, vec[5], num_blocks);

		/* Stopping Criteria */
		norm2_mpfvector_blocks(dtmp, vec[6], num_blocks);
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			/* x = x + alpha pt */
			add_cmul_mpfvector_blocks(vec[0], vec[0], alpha, vec[3], num_blocks);

			//subst_mpfvector_blocks(answer, vec[0], num_blocks);

			return_val = times;
			break;
		}

		/* precondition */

		//mul_mpfmatrix_mpfvec(vec[8], a, vec[6]);
		mul_mpfbmatrix_mpfvec(vec[8][0], aii [0], vec[6][0]);
		mul_mpfbmatrix_mpfvec(  tmpv[0], aip1[0], vec[6][1]);
		add_mpfvector(vec[8][0], vec[8][0], tmpv[0]);

		//#pragma ompparallel for
		for(block_i = 1; block_i < num_blocks - 1; block_i++)
		{
			mul_mpfbmatrix_mpfvec(vec[8][block_i], aim1 [block_i - 1], vec[6][block_i - 1]);

			mul_mpfbmatrix_mpfvec(  tmpv[block_i], aii  [block_i    ], vec[6][block_i    ]);
			add_mpfvector(vec[8][block_i], vec[8][block_i], tmpv[block_i]);

			mul_mpfbmatrix_mpfvec(  tmpv[block_i], aip1 [block_i    ], vec[6][block_i + 1]);
			add_mpfvector(vec[8][block_i], vec[8][block_i], tmpv[block_i]);
		}

		mul_mpfbmatrix_mpfvec(vec[8][num_blocks - 1], aim1 [num_blocks - 2], vec[6][num_blocks - 2]);
		mul_mpfbmatrix_mpfvec(  tmpv[num_blocks - 1], aii  [num_blocks - 1], vec[6][num_blocks - 1]);
		add_mpfvector(vec[8][num_blocks - 1], vec[8][num_blocks - 1], tmpv[num_blocks - 1]);

		/* t := P^(-1) * t */

	/***** Start left preconditioning *****/
		// Forward substitute
		subst_mpfvector(tmpv[0], vec[8][0]);
		for(block_i = 0; block_i < num_blocks - 1; block_i++)
		{
			SolveMPFBLS(tmpu[block_i], hmat[block_i], tmpv[block_i]);
			mul_mpfbmatrix_mpfvec(tmpw[block_i + 1], aim1[block_i], tmpu[block_i]);
			sub_mpfvector(tmpv[block_i + 1], vec[8][block_i + 1], tmpw[block_i + 1]);
		}

		// Backward substitute
		SolveMPFBLS(vec[8][num_blocks - 1], hmat[num_blocks - 1], tmpv[num_blocks - 1]);
		for(block_i = num_blocks - 2; block_i >= 0; block_i--)
		{
			mul_mpfbmatrix_mpfvec(tmpw[block_i], aip1[block_i], vec[8][block_i + 1]);
			sub_mpfvector(tmpu[block_i], tmpv[block_i], tmpw[block_i]);
			SolveMPFBLS(vec[8][block_i], hmat[block_i], tmpu[block_i]);
		}
	/***** End left preconditioning *****/

		/* omega = (t, s) / (t, t) */
		ip_mpfvector_blocks(omega_den, vec[8], vec[8], num_blocks);
		if(mpf_cmp_ui(omega_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Omega is zero!(MPFBiCGSTAB_triblock, %ld)\n", times);
			return_val = -3;
			break;
		}
		ip_mpfvector_blocks(omega, vec[8], vec[6], num_blocks);
		if(mpf_cmp_ui(omega, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Omega is zero!(MPFBiCGSTAB_triblock, %ld)\n", times);
			return_val = -4;
			break;
		}
		mpf_div(omega, omega, omega_den);

		/* x = x + alpha pt + omega st */
		add_cmul_mpfvector_blocks(vec[4], vec[0], alpha, vec[3], num_blocks);
		add_cmul_mpfvector_blocks(vec[0], vec[4], omega, vec[6], num_blocks);

		/* residual */
		mpf_neg(dtmp, omega);
		add_cmul_mpfvector_blocks(vec[1], vec[6], dtmp, vec[8], num_blocks);

		ip_mpfvector_blocks(beta_num, vec[1], vec[1], num_blocks);

		/* Stopping Criteria */
		mpf_sqrt(dtmp, beta_num);
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			// subst_mpfvector_blocks(answer, vec[0], num_blocks);

			return_val = times;
			break;
		}

		mpf_set(old_rho, rho);
	}

	/* Not converge */
	subst_mpfvector_blocks(answer, vec[0], num_blocks);

	/* free vec[0]..[3]; */
	for(block_i = 0; block_i < num_blocks; block_i++)
	{
		free_mpfvector(tmpv[block_i]);
		free_mpfvector(tmpu[block_i]);
		free_mpfvector(tmpw[block_i]);

		for(i = 0; i < 9; i++)
				free_mpfvector(vec[i][block_i]);
	}
	free(tmpv); free(tmpu); free(tmpw);
	for(i = 0; i < 9; i++) free(vec[i]);
	
	mpf_clear(alpha); mpf_clear(alpha_num); mpf_clear(alpha_den);
	mpf_clear(rho); mpf_clear(old_rho);
	mpf_clear(beta); mpf_clear(beta_num);
	mpf_clear(omega); mpf_clear(omega_den);
	mpf_clear(dtmp); mpf_clear(dtmp1); mpf_clear(init_resnorm);

	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(MPFBiCGSTAB_triblock, %ld)\n", times);
		return_val = -5;
	}

	return return_val;
}
#endif 
